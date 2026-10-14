"""Multi-identity Telegram outbound: pool, send, typing, topics, reactions.

This module is the OUTBOUND half of the Telegram edge. The inbound adapter
(:mod:`rtech_chatd.telegram_adapter`) delegates its :class:`ChannelAdapter`
outbound methods here, so the protocol stays the only seam the edge programs
against while this module owns every send mechanic:

* **Per-identity client pool.** Each agent posts under its OWN bot — the whole
  point of the Telegram pivot (real distinct identities with avatars, no
  webhook ventriloquism). The pool maps ``agent_id -> token provider``; a send
  stamped with an UNKNOWN agent fails LOUD (:class:`UnknownAgentIdentityError`)
  and never falls back to another identity. Bots are built lazily at first use
  so a token lives only inside its client.
* **HTML + 4096 chunking** via :mod:`rtech_chatd.telegram_html` (chunk the
  markdown, convert per chunk). A Bot API entity-parse 400 degrades that chunk
  to a plain-text resend — content lands, loudly, never dropped.
* **Rate limiting owned by the edge** via
  :class:`rtech_chatd.telegram_rate_limiter.TelegramRateLimiter`, acquired
  before every send. A served FloodWait (429) records its penalty in the
  limiter and RE-RAISES to the caller: the edge's piece-mark render semantics
  already resume only the missing piece, so retry/backoff policy
  stays with the render loop, never a blind retry here.
* **Forum topics with a reply-chain degrade.** ``create_thread`` on a forum
  supergroup creates a real topic; on a plain group it returns the synthetic
  opaque id ``reply:<anchor_message_id>``, which :meth:`TelegramSender.send`
  resolves to ``reply_parameters`` under the anchor (the TLDR) — same landing
  guarantee, degraded shape. Thread ids are documented opaque on the protocol,
  so the marker never leaks transport knowledge upward.
* **Typing sustain.** Telegram's ``sendChatAction`` expires after ~5s and bots
  have no presence dot, so ack + typing IS the liveness signal: ``typing(on)``
  runs a keyed (chat, agent) keepalive loop re-sending the action until
  ``typing(off)`` cancels it. Best-effort and contained — a typing failure
  never touches a send.

``SendTarget`` semantics on this surface: ``channel_id`` is ALWAYS the chat id;
``thread_id`` is a topic (or degrade marker) WITHIN that chat, never a
standalone destination — Telegram threads are per-chat sub-ids, unlike Discord
where a thread id is itself addressable. ``SendTarget.resolve()`` (thread wins)
is therefore not used here.
"""

from __future__ import annotations

import re
from asyncio import CancelledError, Lock, Task, create_task, current_task, get_running_loop, sleep
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import (
    FSInputFile,
    InlineKeyboardMarkup,
    Message,
    ReactionTypeEmoji,
    ReplyParameters,
)
from loguru import logger

from .channel import ChunkProgress, SendTarget, attach_chunk_progress
from .telegram_html import telegram_chunks
from .telegram_rate_limiter import TelegramRateLimiter

# Match a Bot API URL segment carrying the token (``/bot<TOKEN>/...``). aiogram
# transport errors can embed the full request URL — which contains the bot
# token — in their str(), so EVERY logged exception in this module (and the
# inbound adapter, which imports this) is routed through :func:`redact_token`
# first (security lane).
_BOT_TOKEN_URL_RE = re.compile(r"/bot[^/\s]+")


def redact_token(text: str) -> str:
    """Return ``text`` with any ``/bot<TOKEN>`` URL segment redacted.

    The single owner of Telegram log-redaction for this package: the inbound
    adapter and this sender both route exception strings through here so a
    leaked request URL can never carry a credential into journald.
    """
    return _BOT_TOKEN_URL_RE.sub("/bot<redacted>", text)


# Zero-arg token provider (config-ref resolved elsewhere; fetched at first use,
# never stored beyond the built client, never logged) — the same contract as
# the adapters' TokenProvider.
TokenProvider = Callable[[], str]
BotFactory = Callable[[str], Bot]
# Provider of the edge's own (polling) bot, for sends with no agent identity
# stamped — bound by the adapter at construction.
DefaultBotProvider = Callable[[], Bot]

# Interval between typing keepalive refreshes. Telegram expires a chat action
# after ~5 seconds; refreshing at 4 keeps the indicator visibly continuous.
TYPING_REFRESH_INTERVAL_S = 4.0
# Backoff after a FAILED chat-action send (dead transport / flood): typing is
# best-effort, so a failure pauses the loop rather than killing or spamming it.
TYPING_FAILURE_BACKOFF_S = 10.0
# A missing terminal frame must not leave a chat action refreshing forever. This
# is deliberately a sender-owned liveness lease, not a connector-owned completion
# signal: the typed terminal TurnFrame still stops typing early when it arrives.
TYPING_MAX_LIFETIME_S = 120.0

# The receipt reaction set on the asker's message the moment an intake hit
# fires (three-signal contract, signal 1).
RECEIPT_REACTION = "👀"

# Synthetic thread-id namespace for the non-forum degrade (module docstring).
_REPLY_MARKER_PREFIX = "reply:"

# Substring identifying a Bot API entity-parse failure inside a 400. The Bot
# API has no error code taxonomy beyond HTTP status + description text; this
# phrase is the documented, stable prefix ("Bad Request: can't parse entities:
# ...").
_PARSE_ERROR_MARKER = "can't parse entities"


def _chunk_plan_fingerprint(payloads: list[str]) -> str:
    """Return a content-safe identity for one deterministic chunk plan.

    Length-prefixing distinguishes payload boundaries without retaining or
    logging any rendered/private text. A changed input or chunking policy
    produces a different plan, which makes a supplied resume cursor unsafe.
    """

    digest = sha256()
    for payload in payloads:
        encoded = payload.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


class UnknownAgentIdentityError(LookupError):
    """A send was stamped with an agent id the identity pool does not know.

    Deliberately FATAL for the send (never contained into a fallback): posting
    one agent's words under another agent's bot is the exact ventriloquism
    anti-pattern per-identity sending replaces.
    """


class InteractiveMarkupIdentityError(ValueError):
    """Interactive markup was attached to an agent-stamped (non-polled) send.

    Only the edge's default bot runs the getUpdates poll loop that receives
    callback taps; a button rendered under a pool identity's bot could never
    deliver its tap. Refusing loud keeps a dead, un-actionable button from ever
    reaching a human — interactive markup must ride the polled default identity.
    """


class TelegramSender:
    """Own every Telegram outbound mechanic behind the ChannelAdapter seam."""

    def __init__(
        self,
        *,
        identity_token_providers: dict[str, TokenProvider],
        rate_limiter: TelegramRateLimiter | None = None,
        bot_factory: BotFactory | None = None,
        default_bot_provider: DefaultBotProvider | None = None,
    ) -> None:
        """Args:
        identity_token_providers: ``agent_id -> token provider`` for the
            per-identity pool. May be empty (all sends then use the default
            bot — a single-identity deploy).
        rate_limiter: Send pacing; a fresh :class:`TelegramRateLimiter` is
            built when omitted. ONE limiter paces ALL identities: the
            per-chat/global budgets are what the humans in the chat
            experience, regardless of which bot posts.
        bot_factory: Test seam building a ``Bot`` from a token.
        default_bot_provider: Returns the edge's own bot for unstamped sends;
            bound later via :meth:`bind_default_bot` when constructed before
            the adapter.
        """
        self._identity_token_providers = dict(identity_token_providers)
        self._limiter = rate_limiter or TelegramRateLimiter()
        self._bot_factory = bot_factory or Bot
        self._default_bot_provider = default_bot_provider
        self._identity_bots: dict[str, Bot] = {}
        # (chat_id, thread-or-"", agent_id-or-"") -> live typing keepalive task.
        # The thread dimension keeps two topics of one chat from coalescing
        # into (and cancelling) each other's indicator.
        self._typing_tasks: dict[tuple[str, str, str], Task[None]] = {}
        # (chat_id, agent_id-or-"") -> send serialization lock: a multi-chunk
        # logical message must not interleave with a concurrent send from the
        # same identity to the same chat (A1, B1, A2 splits A's continuation).
        # Locks are tiny and per-(chat, identity); no eviction needed at chat
        # edge scale.
        self._send_locks: dict[tuple[str, str], Lock] = {}

    def bind_default_bot(self, provider: DefaultBotProvider) -> None:
        """Late-bind the edge-bot provider (the adapter owns that bot's lifecycle)."""
        self._default_bot_provider = provider

    @property
    def known_identities(self) -> frozenset[str]:
        """Return the agent ids this pool can post as."""
        return frozenset(self._identity_token_providers)

    def _bot_for(self, agent_id: str | None) -> Bot:
        """Return the client for ``agent_id`` (lazily built), or the default bot.

        Raises:
            UnknownAgentIdentityError: When ``agent_id`` is stamped but not in
                the pool — loud, never a fallback identity.
            RuntimeError: When an unstamped send has no default bot bound, or
                a pool token resolves empty (unprovisioned).
        """
        if agent_id is None:
            if self._default_bot_provider is None:
                raise RuntimeError("chatd.tg.sender: no default bot bound for unstamped send")
            return self._default_bot_provider()
        bot = self._identity_bots.get(agent_id)
        if bot is not None:
            return bot
        provider = self._identity_token_providers.get(agent_id)
        if provider is None:
            raise UnknownAgentIdentityError(
                f"chatd.tg.sender: no bot identity configured for agent {agent_id!r} "
                f"(known: {sorted(self._identity_token_providers)}); refusing to "
                "post under another identity"
            )
        token = provider()
        if not token:
            raise RuntimeError(f"chatd.tg.sender: bot token for agent {agent_id!r} not provisioned")
        bot = self._bot_factory(token)
        self._identity_bots[agent_id] = bot
        logger.info("chatd.tg.sender | identity client built | agent={}", agent_id)
        return bot

    @staticmethod
    def _resolve_destination(target: SendTarget) -> tuple[int, int | None, int | None]:
        """Return ``(chat_id, message_thread_id, reply_to_message_id)`` for a send.

        ``thread_id`` semantics (module docstring): a numeric id is a forum
        topic within the chat; a ``reply:<id>`` marker is the non-forum degrade
        and becomes a reply anchor. An explicit ``target.reply_to`` wins over a
        degrade marker's anchor (the caller was specific).
        """
        # This surface's ids are always numeric strings (the inbound adapter
        # canonicalizes chat ids; reply anchors are prior message ids). Guard
        # the casts so a malformed target fails as a NAMED contract violation,
        # not a bare int() traceback mid-send.
        try:
            chat_id = int(target.channel_id)
        except ValueError:
            raise ValueError(
                f"chatd.tg.sender: channel_id must be a numeric Telegram chat id, "
                f"got {target.channel_id!r}"
            ) from None
        thread_id: int | None = None
        reply_to: int | None = None
        if target.reply_to:
            try:
                reply_to = int(target.reply_to)
            except ValueError:
                raise ValueError(
                    f"chatd.tg.sender: reply_to must be a numeric message id, "
                    f"got {target.reply_to!r}"
                ) from None
        raw_thread = target.thread_id
        if raw_thread:
            if raw_thread.startswith(_REPLY_MARKER_PREFIX):
                anchor = raw_thread.removeprefix(_REPLY_MARKER_PREFIX)
                if reply_to is None and anchor.isdigit():
                    reply_to = int(anchor)
            elif raw_thread.isdigit():
                thread_id = int(raw_thread)
            else:
                logger.warning(
                    "chatd.tg.sender | unrecognized thread id ignored | thread={}", raw_thread
                )
        return chat_id, thread_id, reply_to

    async def send(
        self,
        target: SendTarget,
        text: str,
        files: list[Path],
        *,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> str | None:
        """Send ``text`` (HTML, chunked) and ``files`` to ``target``.

        Returns the FIRST sent message's id (the thread-anchor contract), or
        ``None`` when nothing was sent. The reply anchor attaches to the first
        chunk only; every chunk individually clears the rate limiter.

        ``reply_markup`` (additive-optional) attaches an inline
        keyboard to the FIRST text chunk only — the buttons ride the head of the
        logical message, never repeat on every chunk. ``None`` (default) is the
        exact earlier send: every existing call site is byte-identical. The
        keyboard is DISPLAY; its buttons carry only the bounded typed
        callback-data contract (:mod:`rtech_chatd.telegram_callback`), and a tap
        becomes a typed steer on the bus — the sender never puts a control signal
        in a button beyond that contract.

        Interactive markup is REFUSED on an agent-stamped send
        (:class:`InteractiveMarkupIdentityError`): only the edge's default bot
        runs a getUpdates poll loop, so a button rendered under a pool identity's
        bot could never deliver its tap — a dead button, silently un-actionable.
        Interactive markup therefore rides the polled default identity only.

        Raises:
            UnknownAgentIdentityError: Stamped agent not in the pool.
            InteractiveMarkupIdentityError: ``reply_markup`` set on an
                agent-stamped send (a non-polled identity cannot receive taps).
            TelegramRetryAfter: A served FloodWait, AFTER its penalty is
                recorded in the limiter — backoff policy belongs to the caller
                (the edge resumes only the missing render piece on redelivery).
            Exception: Any other send failure, after logging; a half-sent
                multi-chunk message resumes via the caller's piece semantics.
        """
        if reply_markup is not None and target.as_agent is not None:
            raise InteractiveMarkupIdentityError(
                f"chatd.tg.sender: interactive markup refused for agent "
                f"{target.as_agent!r}: only the polled default bot receives taps, so a "
                "button under a pool identity would be a dead, un-actionable button; "
                "route interactive markup through the default (unstamped) identity"
            )
        bot = self._bot_for(target.as_agent)
        chat_id, thread_id, reply_to = self._resolve_destination(target)
        is_group = chat_id < 0
        payloads = telegram_chunks(text) if text.strip() else []
        plan_fingerprint = _chunk_plan_fingerprint(payloads)
        start_index = 0
        resume = target.chunk_resume
        if resume is not None:
            if resume.chunk_count == len(payloads) and resume.plan_fingerprint == plan_fingerprint:
                start_index = resume.chunks_landed
                logger.info(
                    "chatd.tg.send | resuming chunk plan | chat={} landed={} total={}",
                    target.channel_id,
                    start_index,
                    len(payloads),
                )
            else:
                logger.warning(
                    "chatd.tg.send | chunk plan changed; restarting | chat={} prior_total={} "
                    "current_total={}",
                    target.channel_id,
                    resume.chunk_count,
                    len(payloads),
                )

        # One logical message at a time per (chat, identity): without this,
        # two concurrent multi-chunk sends interleave (A1, B1, A2), splitting
        # A's continuation in the surface.
        lock_key = (target.channel_id, target.as_agent or "")
        lock = self._send_locks.get(lock_key)
        if lock is None:
            lock = Lock()
            self._send_locks[lock_key] = lock

        first_id: str | None = None
        async with lock:
            for index in range(start_index, len(payloads)):
                payload = payloads[index]
                reply_params = (
                    ReplyParameters(message_id=reply_to, allow_sending_without_reply=True)
                    if index == 0 and reply_to is not None
                    else None
                )
                # The inline keyboard rides the FIRST chunk only (like the reply
                # anchor): the buttons belong to the head of the logical message,
                # never repeated on every continuation chunk.
                chunk_markup = reply_markup if index == 0 else None
                await self._limiter.acquire(target.channel_id, is_group=is_group)
                try:
                    try:
                        sent = await self._send_payload(
                            bot,
                            chat_id,
                            payload,
                            parse_mode="HTML",
                            thread_id=thread_id,
                            reply_params=reply_params,
                            channel_key=target.channel_id,
                            reply_markup=chunk_markup,
                        )
                    except TelegramBadRequest as exc:
                        if _PARSE_ERROR_MARKER not in str(exc).lower():
                            logger.error(
                                "chatd.tg.send | send failed | chat={} reason={}",
                                target.channel_id,
                                redact_token(str(exc)),
                            )
                            raise
                        # Entity-parse degrade: the HTML tripped the Bot API
                        # parser. Resend the payload PLAIN (no parse_mode — tags
                        # render literally) so the content lands, loudly, never
                        # dropped. The resend goes through the SAME FloodWait-
                        # recording path as the original send. The keyboard rides
                        # the plain resend too so the buttons are never dropped by
                        # a parse degrade.
                        logger.warning(
                            "chatd.tg.send | HTML parse rejected; resending plain | "
                            "chat={} reason={}",
                            target.channel_id,
                            redact_token(str(exc)),
                        )
                        await self._limiter.acquire(target.channel_id, is_group=is_group)
                        sent = await self._send_payload(
                            bot,
                            chat_id,
                            payload,
                            parse_mode=None,
                            thread_id=thread_id,
                            reply_params=reply_params,
                            channel_key=target.channel_id,
                            reply_markup=chunk_markup,
                        )
                except Exception as exc:
                    attach_chunk_progress(
                        exc,
                        ChunkProgress(
                            chunks_landed=index,
                            chunk_count=len(payloads),
                            plan_fingerprint=plan_fingerprint,
                        ),
                    )
                    raise
                if first_id is None:
                    first_id = str(sent.message_id)

            for path in files:
                await self._limiter.acquire(target.channel_id, is_group=is_group)
                try:
                    sent_doc = await bot.send_document(
                        chat_id, FSInputFile(str(path)), message_thread_id=thread_id
                    )
                except TelegramRetryAfter as exc:
                    self._limiter.apply_flood_wait(target.channel_id, float(exc.retry_after))
                    raise
                except Exception as exc:
                    raise self._sanitized_transport_error(target.channel_id, exc) from None
                if first_id is None:
                    first_id = str(sent_doc.message_id)

        logger.info(
            "chatd.tg.send | ok | chat={} thread={} chunks={} files={} agent={}",
            target.channel_id,
            target.thread_id,
            len(payloads),
            len(files),
            target.as_agent,
        )
        return first_id

    async def _send_payload(
        self,
        bot: Bot,
        chat_id: int,
        payload: str,
        *,
        parse_mode: str | None,
        thread_id: int | None,
        reply_params: ReplyParameters | None,
        channel_key: str,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> Message:
        """One ``send_message`` with the shared FloodWait + transport handling.

        A 429 records its penalty in the limiter and re-raises (the caller
        owns backoff) — including on the parse-degrade PLAIN resend, which the
        gauntlet caught bypassing the penalty ledger. A ``TelegramBadRequest``
        propagates untouched for the caller's parse-degrade decision. Any
        OTHER exception (aiohttp transport errors can embed the ``/bot<TOKEN>``
        request URL) is logged redacted and re-raised SANITIZED so a generic
        upstream ``reason={exc}`` log can never leak the credential.

        ``reply_markup`` attaches an inline keyboard; the caller
        passes it only for the first chunk (see :meth:`send`).
        """
        try:
            if parse_mode is not None:
                return await bot.send_message(
                    chat_id,
                    payload,
                    parse_mode=parse_mode,
                    message_thread_id=thread_id,
                    reply_parameters=reply_params,
                    reply_markup=reply_markup,
                )
            return await bot.send_message(
                chat_id,
                payload,
                message_thread_id=thread_id,
                reply_parameters=reply_params,
                reply_markup=reply_markup,
            )
        except TelegramRetryAfter as exc:
            self._limiter.apply_flood_wait(channel_key, float(exc.retry_after))
            logger.warning(
                "chatd.tg.send | FloodWait surfaced to caller | chat={} retry_after={}s",
                channel_key,
                exc.retry_after,
            )
            raise
        except TelegramBadRequest:
            raise
        except CancelledError:
            raise
        except Exception as exc:
            raise self._sanitized_transport_error(channel_key, exc) from None

    @staticmethod
    def _sanitized_transport_error(channel_key: str, exc: Exception) -> RuntimeError:
        """Log a transport failure REDACTED and return a token-free error.

        aiogram/aiohttp errors can stringify to the full request URL (which
        embeds the bot token). Upstream layers (the edge) log raised errors
        generically, so the original exception must not travel — it is logged
        here once, redacted, and replaced by a RuntimeError carrying only the
        redacted text.
        """
        detail = redact_token(f"{type(exc).__name__}: {exc}")
        logger.error("chatd.tg.send | transport failure | chat={} reason={}", channel_key, detail)
        return RuntimeError(f"chatd.tg.send failed | chat={channel_key} | {detail}")

    @staticmethod
    def _typing_key(target: SendTarget) -> tuple[str, str, str]:
        """Return the typing-task key: (chat, thread-or-'', agent-or-'').

        The thread dimension keeps concurrent debates in two topics of one
        chat from coalescing into a single indicator task; the start (intake
        sink) and stop (edge terminal render) sides both derive the thread
        from the SAME channel origin, so the keys pair up.
        """
        return (target.channel_id, target.thread_id or "", target.as_agent or "")

    async def typing(self, target: SendTarget, on: bool) -> None:
        """Start/stop the (chat, thread, agent)-keyed typing keepalive.

        Starting is idempotent per key. Stopping a key that never started is a
        no-op. An unknown stamped identity is LOUD here too (same pool rule) —
        but raised before any task spawns, so no orphan loop.
        """
        key = self._typing_key(target)
        if not on:
            task = self._typing_tasks.pop(key, None)
            if task is not None:
                task.cancel()
            return
        if key in self._typing_tasks:
            return
        bot = self._bot_for(target.as_agent)
        chat_id, thread_id, _ = self._resolve_destination(target)
        deadline = get_running_loop().time() + TYPING_MAX_LIFETIME_S
        self._typing_tasks[key] = create_task(
            self._typing_loop(key, bot, chat_id, thread_id, deadline),
            name=f"tg-typing-{target.channel_id}-{target.as_agent or 'edge'}",
        )

    async def _typing_loop(
        self,
        key: tuple[str, str, str],
        bot: Bot,
        chat_id: int,
        thread_id: int | None,
        deadline: float,
    ) -> None:
        """Re-send typing until cancelled or its bounded liveness lease expires."""
        try:
            while True:
                remaining = deadline - get_running_loop().time()
                if remaining <= 0:
                    logger.info(
                        "chatd.tg.typing | liveness lease expired | chat={} agent={}",
                        key[0],
                        key[2] or "edge",
                    )
                    return
                try:
                    await bot.send_chat_action(chat_id, "typing", message_thread_id=thread_id)
                except CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 — typing is best-effort liveness.
                    logger.warning(
                        "chatd.tg.typing | chat action failed; backing off | chat={} "
                        "agent={} reason={}",
                        key[0],
                        key[2] or "edge",
                        redact_token(str(exc)),
                    )
                    await sleep(min(TYPING_FAILURE_BACKOFF_S, remaining))
                    continue
                await sleep(min(TYPING_REFRESH_INTERVAL_S, remaining))
        except CancelledError:
            pass
        finally:
            # Self-remove ONLY when the registered task is still THIS task: a
            # typing(off) -> typing(on) restart installs a NEW task under the
            # same key before this cancelled loop unwinds, and an unconditional
            # pop would orphan that fresh task.
            try:
                me = current_task()
            except RuntimeError:  # pragma: no cover - loop already torn down
                me = None
            if me is not None and self._typing_tasks.get(key) is me:
                self._typing_tasks.pop(key, None)

    async def create_thread(self, channel_id: str, message_id: str, name: str) -> str | None:
        """Create a forum topic for a debate turn, or return the reply degrade.

        On a Topics-enabled supergroup: ``createForumTopic`` (needs the edge
        bot to be admin with ``can_manage_topics``) and the numeric topic id is
        returned. On a NON-forum chat the synthetic ``reply:<message_id>``
        marker is returned so the full text degrades to a reply chain under the
        TLDR anchor. A create/API failure returns ``None`` (protocol fail-soft;
        the edge degrades to the parent channel) — never raises.
        """
        try:
            bot = self._bot_for(None)
            chat = await bot.get_chat(int(channel_id))
            if not getattr(chat, "is_forum", False):
                logger.info(
                    "chatd.tg.thread | non-forum chat; reply-chain degrade | chat={}", channel_id
                )
                return f"{_REPLY_MARKER_PREFIX}{message_id}"
            topic = await bot.create_forum_topic(int(channel_id), name)
        except Exception as exc:  # noqa: BLE001 — protocol contract: fail soft, never raise.
            logger.warning(
                "chatd.tg.thread | create failed | chat={} reason={}",
                channel_id,
                redact_token(str(exc)),
            )
            return None
        logger.info(
            "chatd.tg.thread | topic created | chat={} topic={}",
            channel_id,
            topic.message_thread_id,
        )
        return str(topic.message_thread_id)

    async def react_receipt(self, channel_id: str, message_id: str) -> None:
        """Set the 👀 receipt reaction on the asker's message (signal 1 of 3).

        Best-effort and CONTAINED: a reaction failure logs and returns — it
        must never block or fail the intake mint it accompanies.
        """
        try:
            bot = self._bot_for(None)
            await bot.set_message_reaction(
                int(channel_id),
                int(message_id),
                reaction=[ReactionTypeEmoji(emoji=RECEIPT_REACTION)],
            )
        except Exception as exc:  # noqa: BLE001 — receipt is best-effort, never blocks the mint.
            logger.warning(
                "chatd.tg.react | receipt reaction failed | chat={} message_id={} reason={}",
                channel_id,
                message_id,
                redact_token(str(exc)),
            )
            return
        logger.debug("chatd.tg.react | receipt set | chat={} message_id={}", channel_id, message_id)

    async def answer_callback(
        self,
        callback_query_id: str,
        *,
        text: str | None = None,
        show_alert: bool = False,
        as_agent: str | None = None,
        channel_id: str | None = None,
    ) -> bool:
        """Acknowledge a tap, containing ordinary errors but surfacing FloodWait."""
        limiter_key = channel_id or "callback"
        try:
            bot = self._bot_for(as_agent)
            await self._limiter.acquire(limiter_key, is_group=limiter_key.startswith("-"))
            await bot.answer_callback_query(callback_query_id, text=text, show_alert=show_alert)
        except TelegramRetryAfter as exc:
            self._limiter.apply_flood_wait(limiter_key, float(exc.retry_after))
            raise
        except CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — ack is best-effort; never blocks the steer.
            logger.warning(
                "chatd.tg.callback | answer failed | agent={} reason={}",
                as_agent or "edge",
                redact_token(str(exc)),
            )
            return False
        logger.debug("chatd.tg.callback | answered | agent={}", as_agent or "edge")
        return True

    async def edit_reply_markup(
        self,
        target: SendTarget,
        message_id: str,
        *,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> bool:
        """Replace or clear a message keyboard, surfacing only FloodWait."""
        try:
            bot = self._bot_for(target.as_agent)
            chat_id, _thread_id, _reply_to = self._resolve_destination(target)
            await self._limiter.acquire(target.channel_id, is_group=chat_id < 0)
            await bot.edit_message_reply_markup(
                chat_id=chat_id,
                message_id=int(message_id),
                reply_markup=reply_markup,
            )
        except TelegramRetryAfter as exc:
            self._limiter.apply_flood_wait(target.channel_id, float(exc.retry_after))
            raise
        except CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — flow edit is best-effort; never blocks the steer.
            logger.warning(
                "chatd.tg.callback | edit reply_markup failed | chat={} reason={}",
                target.channel_id,
                redact_token(str(exc)),
            )
            return False
        logger.debug(
            "chatd.tg.callback | reply_markup edited | chat={} message_id={}",
            target.channel_id,
            message_id,
        )
        return True

    async def edit_text(
        self,
        target: SendTarget,
        message_id: str,
        text: str,
        *,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> bool:
        """Edit text and optional markup, with the sender's HTML fallback."""
        try:
            bot = self._bot_for(target.as_agent)
            chat_id, _thread_id, _reply_to = self._resolve_destination(target)
            await self._limiter.acquire(target.channel_id, is_group=chat_id < 0)
        except TelegramRetryAfter as exc:
            self._limiter.apply_flood_wait(target.channel_id, float(exc.retry_after))
            raise
        except CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — resolve failure is contained like the edit itself.
            logger.warning(
                "chatd.tg.callback | edit text setup failed | chat={} reason={}",
                target.channel_id,
                redact_token(str(exc)),
            )
            return False
        try:
            try:
                await bot.edit_message_text(
                    text,
                    chat_id=chat_id,
                    message_id=int(message_id),
                    parse_mode="HTML",
                    reply_markup=reply_markup,
                )
            except TelegramBadRequest as exc:
                if _PARSE_ERROR_MARKER not in str(exc).lower():
                    raise
                # Entity-parse degrade: the HTML tripped the Bot API parser.
                # Re-edit PLAIN (tags render literally) so the content lands.
                logger.warning(
                    "chatd.tg.callback | edit HTML parse rejected; retrying plain | "
                    "chat={} reason={}",
                    target.channel_id,
                    redact_token(str(exc)),
                )
                await bot.edit_message_text(
                    text,
                    chat_id=chat_id,
                    message_id=int(message_id),
                    reply_markup=reply_markup,
                )
        except TelegramRetryAfter as exc:
            self._limiter.apply_flood_wait(target.channel_id, float(exc.retry_after))
            raise
        except CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — flow edit is best-effort; never blocks the steer.
            logger.warning(
                "chatd.tg.callback | edit text failed | chat={} reason={}",
                target.channel_id,
                redact_token(str(exc)),
            )
            return False
        logger.debug(
            "chatd.tg.callback | text edited | chat={} message_id={}",
            target.channel_id,
            message_id,
        )
        return True

    async def close(self) -> None:
        """Cancel typing loops and close every pool client's HTTP session."""
        for task in list(self._typing_tasks.values()):
            task.cancel()
        self._typing_tasks.clear()
        for agent_id, bot in list(self._identity_bots.items()):
            try:
                await bot.session.close()
            except Exception as exc:  # noqa: BLE001 — teardown must never raise.
                logger.warning(
                    "chatd.tg.sender | session close failed | agent={} reason={}",
                    agent_id,
                    redact_token(str(exc)),
                )
        self._identity_bots.clear()
        self._send_locks.clear()
        logger.info("chatd.tg.sender | closed")


__all__ = [
    "RECEIPT_REACTION",
    "TYPING_FAILURE_BACKOFF_S",
    "TYPING_MAX_LIFETIME_S",
    "TYPING_REFRESH_INTERVAL_S",
    "InteractiveMarkupIdentityError",
    "TelegramSender",
    "TokenProvider",
    "UnknownAgentIdentityError",
    "redact_token",
]
