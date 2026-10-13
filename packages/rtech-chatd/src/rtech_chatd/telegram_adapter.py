"""aiogram wiring of the :class:`ChannelAdapter` protocol (Telegram, inbound).

This is the Telegram peer of :mod:`rtech_chatd.discord_adapter`. It is the
INBOUND half of the Telegram edge (issue #108); the outbound half (issue #109)
— multi-identity sends, HTML chunking, forum topics, rate limiting, typing,
the 👀 receipt reaction — lives in :mod:`rtech_chatd.telegram_sender` and is
DELEGATED to via an injected :class:`TelegramSender`, so this adapter remains
the single :class:`ChannelAdapter` seam the edge programs against. With no
sender injected the outbound methods keep their #108 fail-soft stub behavior
(log + ``None``), so an inbound-only deploy is unchanged.

The whole point of this slice is that the EXISTING pipeline runs unchanged on
Telegram: this adapter maps an aiogram ``Update`` to the transport-neutral
:class:`InboundMessage`, applies the SAME pure gates
(:func:`rtech_chatd.gates.should_respond`) and the SAME injected intake sink
(:meth:`_maybe_intake`) the Discord adapter does, and yields gated messages from
:meth:`events`. Nothing above :class:`ChannelAdapter` learns which surface a
message came from.

Structural choices this slice locks in:

* **chatd owns the loop.** :meth:`events` drives a raw ``bot.get_updates``
  long-poll (offset-tracked) and feeds normalized messages onto an
  ``asyncio.Queue`` that the async generator drains — the wrap-handlers-into-a-
  queue shape the ticket asks for. We do NOT hand structure to aiogram's
  ``Dispatcher``; the poll loop, the ``allowed_updates`` subscription, and the
  offset are ours so the edge stays legible.
* **Normalize the chat id ONCE, at the edge.** The Bot API already delivers the
  canonical ``-100…`` supergroup id (``chat.id``); we stringify it and NEVER
  re-derive it downstream. The whole 3-tier send-fallback mess the old
  coordinator repo carried came from NOT normalizing here — this edge stores the
  canonical string and everything keyed on ``channel_id`` (config, dedup, gate
  scoping) sees one id space.
* **Adapter-local dedup keyed on (chat_id, message_id).** Telegram message ids
  are PER-CHAT, not globally unique like Discord snowflakes, so the dedup key is
  the COMPOSITE — see :class:`_TelegramMessageDedup` for why we do not reuse
  ``intake_sink._MessageIdDedup``.

Crash-boundary policy (at-least-once, then dedup — a deliberate trade):

  The getUpdates offset and the ``(chat_id, message_id)`` dedup store are BOTH
  process-local: neither survives a restart. A single live polling session owns
  the offset, and the offset is advanced BEFORE an update is handled, not after,
  so a crash between "handled a batch" and "confirmed that batch's offset to
  Telegram" makes the next process re-fetch and re-handle exactly that one
  unconfirmed getUpdates batch. The composite dedup then suppresses any message
  the pre-crash process already admitted WITHIN THE SAME LIVE SESSION, but a
  fresh process starts with an empty store, so a cross-restart redelivery of the
  last unconfirmed batch CAN re-admit. This is at-least-once across a restart,
  bounded to one getUpdates batch, followed by in-session composite dedup. The
  trade is deliberate: confirming the offset BEFORE handling would swap duplicate
  delivery for message LOSS on a crash, which is strictly worse for a chat edge
  (a missed human question is invisible; a duplicated one is a redundant, gated,
  deduped wake). Durable offset/dedup is explicitly out of scope for this slice.

The bot token is NEVER a literal here. It is fetched at connect time from an
injected ``token_provider`` (a config ref, resolved elsewhere) and never logged;
only sanitized ids (chat id, message id, user id) ever reach a log line.
"""

from __future__ import annotations

import asyncio
import inspect
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from aiogram import Bot
from aiogram.enums import ChatType, MessageEntityType
from aiogram.types import CallbackQuery, Message, MessageEntity, Update
from loguru import logger

from .channel import InboundMessage, SendTarget
from .config import GateConfig, IntakeConfig, TelegramConfig
from .gates import intake_hit, should_respond
from .telegram_sender import TelegramSender, redact_token

# A zero-arg callable returning the bot token (from a config ref). Kept as a
# provider so the token is fetched at connect time and never stored/logged —
# identical contract to the Discord adapter's TokenProvider.
TokenProvider = Callable[[], str]

# Invoked with the inbound message on a deterministic intake hit (issue #93). The
# sink owns minting + publishing the triage TaskFrame; the adapter never touches
# the bus. May be sync OR async: the in-repo bus publisher is a coroutine, so the
# call site awaits an awaitable result inside the intake error boundary (a
# sync-only contract would leave an ``async def`` sink's coroutine un-awaited —
# publishing nothing and escaping the boundary). Mirrors the Discord adapter.
IntakeSink = Callable[[InboundMessage], Awaitable[None] | None]

# Invoked with a normalized inbound callback (issue #117). The sink owns
# authentication, acknowledgement, and mapping an authorized action into a
# bus-published SteerFrame; the adapter never touches the bus. May be sync or
# async — awaited inside the callback error boundary exactly like IntakeSink.
CallbackSink = Callable[["InboundCallback"], Awaitable[None] | None]

# Bot factory (test seam). The real path builds ``Bot(token)`` at connect time so
# the token lives only for the connection; tests inject a fake with the same
# ``get_updates`` / ``session.close`` surface and no network.
BotFactory = Callable[[str], Bot]

# Dedup store bounds. Same philosophy as intake_sink._MessageIdDedup: a large LRU
# aged out by a monotonic horizon, sized far above any chat's per-horizon volume
# so capacity eviction never trims an id still inside the horizon under normal
# load, with a loud warning if an id-flood ever forces it.
DEFAULT_TG_DEDUP_MAX_ENTRIES = 65536
DEFAULT_TG_DEDUP_HORIZON_SECONDS = 3600.0

# Monotonic clock injected for testability (dedup time-eviction is monotonic — it
# never needs to survive a restart, exactly like the intake-sink store).
Clock = Callable[[], float]

# Pause after a FAILED getUpdates poll. A healthy empty poll already spent the
# server-side long-poll hold, but a dead network/API makes get_updates raise
# instantly — without this pause the poll loop hot-spins, flooding the error log
# and burning CPU until the transport recovers.
POLL_FAILURE_BACKOFF_SECONDS = 5.0

# Bound the inbound queue so an overloaded/stalled consumer cannot grow it without
# limit. The backpressure chain is intentional and lossless: a FULL queue makes
# ``await self._inbound.put(msg)`` stall the poll loop -> the loop stops calling
# getUpdates -> the offset is never advanced past the stalled batch -> Telegram
# holds redelivery of those updates server-side (its own retention). Nothing is
# dropped and overload is never silent — it manifests as the poll loop parking on
# a full queue, exactly the flow control a chat edge wants.
DEFAULT_INBOUND_QUEUE_MAX = 1024

# Telegram's fixed pseudo-user id for the ``GroupAnonymousBot`` sender that fronts
# a message posted by an anonymous supergroup admin. The update carries this fake
# ``from_user`` (``is_bot=True``) with the REAL author in ``sender_chat``; it is a
# human admin, not a bot, so the adapter must not bot-gate it. This id is a Bot
# API constant.
GROUP_ANONYMOUS_BOT_ID = 1087968824

# Log-redaction is owned by :mod:`rtech_chatd.telegram_sender` (one owner for
# the whole Telegram surface). NEVER log a raw exception str in this module —
# route it through the shared redactor first so an aiogram request URL
# (``/bot<TOKEN>/method``) cannot carry the credential into journald.
_redact_token = redact_token


class _TelegramMessageDedup:
    """A count+time-bounded set of already-admitted ``(chat_id, message_id)`` keys.

    This is the adapter-local twin of
    :class:`rtech_chatd.intake_sink._MessageIdDedup`, deliberately NOT imported
    from it. The reason is the KEY: Discord message ids are globally-unique
    snowflakes, so that store keys on ``message_id`` alone. Telegram message ids
    are PER-CHAT — every chat restarts its own id sequence, so id ``42`` in chat
    A and id ``42`` in chat B are DIFFERENT messages. Keying on ``message_id``
    alone would let a message in one chat suppress an unrelated message that
    happens to share the id in another chat. The key here is the COMPOSITE
    ``(chat_id, message_id)`` so cross-chat id collisions both admit, while a true
    redelivery of the same message in the same chat is suppressed.

    Everything else mirrors the intake-sink store exactly:

    * COUNT bound: an :class:`~collections.OrderedDict` LRU — inserting past
      ``max_entries`` evicts oldest-inserted (expired-first, then oldest).
    * TIME bound: a key older than ``horizon_s`` (monotonic) is treated as absent,
      so the same message can re-admit only after the window fully elapses.

    At-most-once is EXACT within normal volume (the cap sits far above any chat's
    per-horizon message count, so capacity eviction only ever removes already-
    expired keys) and BOUNDED-BEST-EFFORT under a key-flood: a capacity eviction
    forced to remove a not-yet-expired key logs a loud warning (the dedup window
    has degraded), so overload is never silent and a redelivered key could then
    re-admit inside the window. The cap is the memory backstop, the warning is the
    loud signal, the horizon is the normal-case correctness bound.

    Monotonic time is deliberate: a long-poll reconnect starts a fresh offset
    from Telegram's server-side retention, so this store has no cross-restart
    durability need — unlike :class:`rtech_agentd.completed_store`, whose horizon
    must survive one.
    """

    def __init__(self, *, max_entries: int, horizon_s: float, clock: Clock) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if horizon_s <= 0:
            raise ValueError("horizon_s must be > 0")
        self._max_entries = max_entries
        self._horizon_s = horizon_s
        self._clock = clock
        # (chat_id, message_id) -> monotonic insert time. OrderedDict order =
        # insertion order (LRU by age of first admit).
        self._seen: OrderedDict[tuple[str, str], float] = OrderedDict()

    def check_and_remember(self, chat_id: str, message_id: str) -> bool:
        """Return True when ``(chat_id, message_id)`` is FRESH, False on a repeat.

        A fresh key is recorded and returns True. A repeat still inside the
        horizon returns False (suppress). A repeat whose prior record has aged
        past the horizon is treated as fresh: re-recorded and returns True.
        """
        key = (chat_id, message_id)
        now = self._clock()
        self._evict_expired(now)
        prior = self._seen.get(key)
        if prior is not None and (now - prior) <= self._horizon_s:
            # Known and still within the window — suppress. Refresh recency so a
            # steadily-repeated key does not get LRU-evicted out from under the
            # suppression (its window keeps sliding while repeats keep arriving).
            self._seen.move_to_end(key)
            return False
        self._seen[key] = now
        self._seen.move_to_end(key)
        self._evict_over_capacity(now)
        return True

    def _evict_expired(self, now: float) -> None:
        """Drop every key whose record has aged past the horizon (oldest first)."""
        while self._seen:
            _oldest_key, oldest_at = next(iter(self._seen.items()))
            if (now - oldest_at) <= self._horizon_s:
                break
            self._seen.popitem(last=False)

    def _evict_over_capacity(self, now: float) -> None:
        """Evict oldest-inserted keys until within ``max_entries``.

        :meth:`_evict_expired` already ran, so any remaining over-capacity
        eviction removes the OLDEST surviving key. If that survivor is still
        INSIDE the horizon, the dedup window has degraded (a key-flood outran the
        cap) — log a loud warning naming the cap + horizon so overload is never
        silent. Without it a redelivered key whose record was capacity-evicted
        would re-admit inside the window; at-most-once becomes bounded-best-effort,
        so we make it loud.
        """
        while len(self._seen) > self._max_entries:
            _oldest_key, oldest_at = next(iter(self._seen.items()))
            if (now - oldest_at) <= self._horizon_s:
                logger.warning(
                    "chatd.tg.dedup | window degraded | capacity eviction removed a "
                    "not-yet-expired key (cap={} horizon_s={}); a redelivered "
                    "(chat_id, message_id) may re-admit inside the window",
                    self._max_entries,
                    self._horizon_s,
                )
            self._seen.popitem(last=False)

    def __len__(self) -> int:
        return len(self._seen)


def canonical_chat_id(chat_id: int) -> str:
    """Return the canonical Telegram chat id STRING (normalize once, at the edge).

    The Bot API already hands supergroup/channel ids in the canonical ``-100…``
    form (and private/basic-group ids as their own signed ints), so canonicalizing
    is a stringify — but we route EVERY chat id through this one function so the
    normalization has a single, testable home and downstream code never re-derives
    an id from an access-hash or a bare MTProto id (the latent bug the old
    coordinator repo papered over with a 3-tier send fallback).
    """
    return str(chat_id)


def _mention_ids(text: str | None, entities: list[MessageEntity] | None) -> frozenset[str]:
    """Collect the intake hit-gate's ``mentioned_ids`` from message entities.

    Telegram carries two DISTINCT mention flavors, and their id spaces differ —
    the duality the hit-gate must match against:

    * ``mention`` entities are ``@username`` spans with NO user object attached
      (Telegram does not resolve a username to a user id in the update). We pull
      the username out of ``text`` via aiogram's UTF-16-correct
      :meth:`MessageEntity.extract_from` and normalize it to a lowercase handle
      WITHOUT the leading ``@`` — so a config that lists a member by ``@handle``
      matches. The entity ``offset``/``length`` are UTF-16 code-unit indices per
      the Bot API, NOT Python codepoint indices: any non-BMP char (an emoji)
      appearing BEFORE the mention shifts the two apart, so a naive
      ``text[offset:offset+length]`` slice would extract the wrong span (the
      username itself is ASCII-only, but the text preceding it is not). The
      extract handles the surrogate arithmetic; the raw slice was the bug.
    * ``text_mention`` entities DO carry a resolved ``user`` object (used to
      mention users who have no public username), so we take the numeric
      ``user.id`` as a string — the same id space as ``author``.

    The hit-gate (:func:`rtech_chatd.gates._intake_content_hit`) matches configured
    member ids against this set as opaque strings, so a deploy lists usernames
    (lowercased, no ``@``) for public members and numeric ids for username-less
    ones. Both live in the one ``mentioned_ids`` set; the config decides which
    form it uses per member.
    """
    if not entities:
        return frozenset()
    ids: set[str] = set()
    body = text or ""
    for entity in entities:
        if entity.type == MessageEntityType.MENTION:
            # ``@username`` span — no user object. Use aiogram's UTF-16-correct
            # extract: entity offsets are UTF-16 code units, so a raw Python slice
            # drifts once a non-BMP char precedes the mention.
            raw = entity.extract_from(body)
            handle = raw.lstrip("@").strip().lower()
            if handle:
                ids.add(handle)
        elif entity.type == MessageEntityType.TEXT_MENTION and entity.user is not None:
            # Resolved user (username-less mention) — numeric id, author id space.
            ids.add(str(entity.user.id))
    return frozenset(ids)


def _mentions_me(
    message: Message,
    mentioned_ids: frozenset[str],
    *,
    is_dm: bool,
    self_username: str | None,
    self_user_id: str | None,
) -> bool:
    """Return whether THIS bot is addressed by ``message``.

    A private chat implicitly addresses the bot (mirrors the Discord DM rule). In
    a group the bot is mentioned when its ``@username`` (a ``mention`` entity,
    normalized to a lowercase handle) or its numeric id (a ``text_mention``) is in
    the message's ``mentioned_ids`` set — the same set the hit-gate consumes, so
    the two never disagree about what "mentions me" means.
    """
    if is_dm:
        return True
    if self_username is not None and self_username.lstrip("@").lower() in mentioned_ids:
        return True
    if self_user_id is not None and self_user_id in mentioned_ids:
        return True
    return False


def map_message(
    message: Message,
    *,
    self_username: str | None,
    self_user_id: str | None,
) -> InboundMessage:
    """Map an aiogram ``Message`` to a normalized :class:`InboundMessage`.

    Normalizes the chat id ONCE (:func:`canonical_chat_id`), resolves the forum
    ``message_thread_id`` to ``thread_id``, and computes the mention flags/ids the
    pure gates consume from message entities.

    ``guild_id`` is always ``None`` on Telegram — there is no guild/server tier;
    the supergroup IS the channel, so the canonical ``-100…`` chat id already
    carries the full addressing the Discord adapter split across guild+channel.
    ``mentions_other_bot`` / ``mentions_human`` stay at their defaults: the
    multi-agent "talking to someone else" gate has nothing to fire on here.

    Anonymous supergroup admins are HUMANS, not bots: Telegram fronts their
    messages with a fake ``from_user`` (:data:`GROUP_ANONYMOUS_BOT_ID`,
    ``is_bot=True``) and carries the real author identity in ``sender_chat``. This
    map treats that case as a human (``is_bot=False``) and derives ``author`` from
    ``sender_chat.id`` — bot-gating it would silently drop a real admin's
    @mention. The same ``sender_chat`` fallback also names the author of a channel
    post whose ``from_user`` is absent.
    """
    chat = message.chat
    is_dm = chat.type == ChatType.PRIVATE

    # thread_id is set ONLY for forum topics. aiogram itself gates topic routing
    # on ``is_topic_message``; ``message_thread_id`` is ALSO populated for an
    # ordinary (non-forum) reply thread, so keying thread_id off the raw id alone
    # would conflate a plain reply chain with a real forum topic and mis-target
    # #109's topic-scoped sends. Non-forum reply threads DELIBERATELY do not set
    # thread_id (they surface as ``is_reply`` instead).
    thread_id: str | None = None
    if message.is_topic_message and message.message_thread_id is not None:
        thread_id = str(message.message_thread_id)

    # A message body may arrive as ``text`` OR (for media) as ``caption``; the
    # entities travel on the matching field. Prefer text, fall back to caption, so
    # a captioned media message with a trigger still normalizes correctly.
    body = message.text if message.text is not None else message.caption
    entities = message.entities if message.entities is not None else message.caption_entities
    text = (body or "").strip()

    mentioned_ids = _mention_ids(body, entities)

    # Author + bot flag. The anonymous-admin case (from_user is the fake
    # GroupAnonymousBot, or absent entirely for a channel post) is a HUMAN whose
    # real identity lives in sender_chat: derive author from sender_chat.id and
    # force is_bot=False so the message is not bot-gated. Otherwise the real
    # from_user drives both.
    from_user = message.from_user
    if from_user is None or from_user.id == GROUP_ANONYMOUS_BOT_ID:
        author = str(message.sender_chat.id) if message.sender_chat is not None else ""
        is_bot = False
    else:
        author = str(from_user.id)
        is_bot = bool(from_user.is_bot)

    mentions_me = _mentions_me(
        message,
        mentioned_ids,
        is_dm=is_dm,
        self_username=self_username,
        self_user_id=self_user_id,
    )

    return InboundMessage(
        channel_id=canonical_chat_id(chat.id),
        thread_id=thread_id,
        author=author,
        text=text,
        is_bot=is_bot,
        is_dm=is_dm,
        mentions_me=mentions_me,
        is_reply=message.reply_to_message is not None,
        # The multi-agent mention gates have no signal on this surface, and there
        # is no guild/server tier on Telegram — guild_id is always None.
        guild_id=None,
        message_id=str(message.message_id),
        mentioned_ids=mentioned_ids,
    )


@dataclass(frozen=True)
class InboundCallback:
    """Represent a normalized inline-keyboard tap (issue #117 interactive layer).

    ``data`` remains opaque: the callback sink authenticates it against its
    issued-capability registry before it can become a typed
    :class:`~rtech_frames.frames.SteerFrame` on the bus. Missing ``message_id`` or
    ``channel_id`` deliberately fails closed at that sink — an inaccessible or
    absent callback message cannot prove a capability's bound coordinates.
    """

    callback_query_id: str
    channel_id: str | None
    message_id: str | None
    author: str
    data: str | None


def map_callback(callback: CallbackQuery) -> InboundCallback:
    """Map an aiogram ``CallbackQuery`` to a normalized :class:`InboundCallback`.

    An inaccessible or absent callback message cannot prove a capability's bound
    coordinates, so it leaves them ``None`` for the callback sink to reject. The
    chat id is routed through the same :func:`canonical_chat_id` normalization
    point every other Telegram edge path uses, so downstream code never
    re-derives it.
    """
    message = callback.message
    channel_id: str | None = None
    message_id: str | None = None
    if message is not None:
        chat = getattr(message, "chat", None)
        if chat is not None:
            channel_id = canonical_chat_id(chat.id)
        raw_message_id = getattr(message, "message_id", None)
        if raw_message_id is not None:
            message_id = str(raw_message_id)
    author = str(callback.from_user.id) if callback.from_user is not None else ""
    return InboundCallback(
        callback_query_id=callback.id,
        channel_id=channel_id,
        message_id=message_id,
        author=author,
        data=callback.data,
    )


class TelegramChannelAdapter:
    """A :class:`ChannelAdapter` backed by aiogram long-polling (inbound, #108).

    Gating decisions come from the pure :mod:`rtech_chatd.gates`; this class owns
    only the surface mechanics — the getUpdates loop, chat-id normalization, and
    the adapter-local ``(chat_id, message_id)`` dedup. Inbound gated messages are
    pushed onto an ``asyncio.Queue`` that :meth:`events` drains, exactly like the
    Discord adapter.

    Outbound (:meth:`send` / :meth:`typing` / :meth:`create_thread`) is out of
    scope for this slice: the methods exist to satisfy the protocol but are
    fail-soft stubs that log "lands in #109" and return ``None``. Chunking, forum
    topics, and rate limiting are #109's work; nothing here silently half-sends.
    """

    def __init__(
        self,
        *,
        token_provider: TokenProvider,
        gate_config: GateConfig,
        telegram_config: TelegramConfig | None = None,
        intake_config: IntakeConfig | None = None,
        intake_sink: IntakeSink | None = None,
        callback_sink: CallbackSink | None = None,
        bot: Bot | None = None,
        bot_factory: BotFactory | None = None,
        sender: TelegramSender | None = None,
        clock: Clock | None = None,
        dedup_max_entries: int = DEFAULT_TG_DEDUP_MAX_ENTRIES,
        dedup_horizon_s: float = DEFAULT_TG_DEDUP_HORIZON_SECONDS,
    ) -> None:
        """Args:
        token_provider: Zero-arg callable returning the bot token at connect
            time (from a config ref — never a literal, never logged).
        gate_config: Gate configuration for :func:`should_respond` /
            :func:`intake_hit`.
        telegram_config: Telegram edge knobs (group allowlist, allowed_updates,
            poll timeout). Defaults to a plain :class:`TelegramConfig`.
        intake_config: Optional intake hit-gate config (issue #93). When None or
            disabled, no intake behavior runs.
        intake_sink: Callback invoked with the inbound message on an intake hit.
            Required for intake to fire; the sink mints/publishes the TaskFrame.
        callback_sink: Callback invoked with a normalized inline-keyboard tap
            (issue #117). Required for the interactive layer to act; the sink
            authenticates the tap, answers the query, and maps an authorized
            typed command into a bus-published SteerFrame. When None, callback
            queries are still answered (so the client spinner stops) but produce
            no steer — an inbound-only / non-interactive deploy is unchanged.
        bot: Optional pre-built aiogram ``Bot`` (test seam). When set it is used
            directly and the token provider is not consulted at connect.
        bot_factory: Optional factory building a ``Bot`` from the token at connect
            time (test seam). When both ``bot`` and ``bot_factory`` are None, the
            real ``Bot`` is constructed from the provider's token.
        sender: Optional outbound half (issue #109). When set, ``send`` /
            ``typing`` / ``create_thread`` delegate to it, the 👀 receipt
            reaction fires at intake-hit time, and its default-bot lane is
            bound to this adapter's polling bot. When None, outbound keeps the
            #108 fail-soft stub behavior (inbound-only deploy, unchanged).
        clock: Monotonic clock for dedup time-eviction (injectable for tests).
        dedup_max_entries / dedup_horizon_s: Dedup store bounds.
        """
        self._token_provider = token_provider
        self._gate_config = gate_config
        self._telegram_config = telegram_config or TelegramConfig()
        self._intake_config = intake_config or IntakeConfig()
        self._intake_sink = intake_sink
        self._callback_sink = callback_sink
        self._bot = bot
        self._bot_factory = bot_factory
        # Bounded (see DEFAULT_INBOUND_QUEUE_MAX): a full queue stalls the poll
        # loop's ``await put`` -> the offset stops advancing -> Telegram holds
        # redelivery server-side. Backpressure, never silent drop.
        self._inbound: asyncio.Queue[InboundMessage] = asyncio.Queue(
            maxsize=DEFAULT_INBOUND_QUEUE_MAX
        )
        self._dedup = _TelegramMessageDedup(
            max_entries=dedup_max_entries,
            horizon_s=dedup_horizon_s,
            clock=clock or _monotonic,
        )
        self._offset: int | None = None
        self._self_username: str | None = None
        self._self_user_id: str | None = None
        self._closed = asyncio.Event()
        self._sender = sender
        if sender is not None:
            # The sender's unstamped (edge-identity) sends use THIS adapter's
            # polling bot; per-agent identities live in the sender's own pool.
            sender.bind_default_bot(self._resolve_bot)
        # Live fire-and-forget receipt-reaction tasks (issue #109 signal 1).
        # Held so the event loop cannot garbage-collect an in-flight task;
        # each removes itself on completion.
        self._reaction_tasks: set[asyncio.Task[None]] = set()

    def _resolve_bot(self) -> Bot:
        """Return the aiogram ``Bot`` to poll, building it at connect time.

        The token is fetched HERE (connect time) from the provider and handed to
        the freshly-built ``Bot`` so it lives only for the connection; it is never
        stored on the adapter and never logged. A pre-built/factory bot (test
        seam) bypasses the provider.
        """
        if self._bot is not None:
            return self._bot
        token = self._token_provider()
        if not token:
            raise RuntimeError("chatd.tg: bot token not provisioned (empty token from provider)")
        factory = self._bot_factory or Bot
        self._bot = factory(token)
        return self._bot

    async def _load_self(self, bot: Bot) -> None:
        """Cache this bot's own username + id (for the ``mentions_me`` gate).

        Best-effort: a getMe failure is logged and leaves the identity unset, so
        the group ``@username`` mention path simply does not fire (private-chat
        implicit addressing still works). Never crashes the loop.
        """
        try:
            me = await bot.get_me()
        except Exception as exc:  # noqa: BLE001 — identity is best-effort; never crash the loop.
            logger.warning(
                "chatd.tg.start | get_me failed; mention-by-username disabled | {} | {}",
                type(exc).__name__,
                _redact_token(str(exc)),
            )
            return
        self._self_username = me.username
        self._self_user_id = str(me.id)
        logger.info("chatd.tg.start | identity resolved | user_id={}", self._self_user_id)

    async def _poll_once(self, bot: Bot) -> list[Update]:
        """Fetch one batch of updates via long-poll getUpdates.

        aiogram retries transient transport failures internally; this wraps the
        call so an unexpected error is logged (token-redacted) and degraded to an
        empty batch rather than tearing down :meth:`_run_poll_loop`. A FAILED poll
        (unlike a healthy empty one, which already spent the long-poll hold)
        returns immediately, so it pauses :data:`POLL_FAILURE_BACKOFF_SECONDS`
        before handing control back — otherwise a dead transport hot-spins the
        loop.

        The in-flight long-poll is raced against :attr:`_closed` so :meth:`close`
        interrupts a parked ``get_updates`` promptly instead of eating the full
        server-side hold; on close the poll future is cancelled and ``[]`` is
        returned. The failure backoff is likewise interruptible (``wait_for`` on
        the close event) so shutdown never waits out a bogus backoff either.
        """
        # ``request_timeout`` is the aiogram session HTTP timeout (default 60s).
        # The server-side long-poll hold (``timeout``) MUST NOT exceed it, or every
        # poll with poll_timeout_s >= 60 would raise before the server responds —
        # a permanent backoff loop delivering zero updates. Give the HTTP layer a
        # buffer past the hold.
        poll = asyncio.ensure_future(
            bot.get_updates(
                offset=self._offset,
                timeout=self._telegram_config.poll_timeout_s,
                allowed_updates=list(self._telegram_config.allowed_updates),
                request_timeout=self._telegram_config.poll_timeout_s + 10,
            )
        )
        closed = asyncio.ensure_future(self._closed.wait())
        try:
            done, _pending = await asyncio.wait({poll, closed}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            poll.cancel()
            closed.cancel()
            raise
        # Closed first (or both): cancel the poll and return empty — shutdown.
        if poll not in done:
            poll.cancel()
            closed.cancel()
            return []
        closed.cancel()
        try:
            updates = poll.result()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — degrade a poll failure, do not kill the loop.
            logger.error(
                "chatd.tg.poll | get_updates failed | {} | reason={}",
                type(exc).__name__,
                _redact_token(str(exc)),
            )
            try:
                # Interruptible backoff: a pending close ends the wait early so a
                # clean shutdown never waits out the backoff.
                await asyncio.wait_for(self._closed.wait(), timeout=POLL_FAILURE_BACKOFF_SECONDS)
            except TimeoutError:
                pass
            return []
        return list(updates)

    async def _run_poll_loop(self, bot: Bot) -> None:
        """Drive the offset-tracked long-poll until the adapter closes.

        Owns the getUpdates offset (``last update_id + 1``) so an update is
        acknowledged exactly once at the transport, and dispatches each update to
        :meth:`_handle_update`. The composite ``(chat_id, message_id)`` dedup below
        the transport still guards against a genuine redelivery (e.g. a batch we
        processed but failed to advance the offset for).
        """
        while not self._closed.is_set():
            updates = await self._poll_once(bot)
            for update in updates:
                self._offset = update.update_id + 1
                try:
                    await self._handle_update(update)
                except Exception as exc:  # noqa: BLE001 — one bad update never kills the loop.
                    logger.error(
                        "chatd.tg.update | handler failed | update_id={} | {} | reason={}",
                        update.update_id,
                        type(exc).__name__,
                        _redact_token(str(exc)),
                    )

    async def _handle_update(self, update: Update) -> None:
        """Dispatch one aiogram ``Update`` to its inbound handler.

        ``message`` and ``edited_message`` carry the text inbound surface;
        ``callback_query`` carries an inline-keyboard tap (issue #117), routed to
        the callback handler. ``message_reaction`` is subscribed (so #109
        inherits the feed) but not yet acted on — logged at debug and dropped.
        Anything else is ignored.
        """
        if update.callback_query is not None:
            await self._on_callback(update.callback_query)
            return
        if update.edited_message is not None:
            # v1 policy (#108): edits are LOGGED and DROPPED, never crash. A v1
            # agent does not re-answer an edited question; surfacing the edit is
            # enough for an operator to see it happened.
            edited = update.edited_message
            logger.info(
                "chatd.tg.edited | ignored (v1 policy) | chat={} message_id={}",
                canonical_chat_id(edited.chat.id),
                edited.message_id,
            )
            return
        if update.message is not None:
            await self._on_message(update.message)
            return
        if update.message_reaction is not None:
            logger.debug("chatd.tg.reaction | received (no-op until #109)")
            return

    async def _on_message(self, message: Message) -> None:
        """Gate an inbound message and enqueue it when the agent should reply.

        Handles the group→supergroup migration service message first (log loud,
        emit nothing), then applies the transport group allowlist, the composite
        dedup, the intake sink, and finally :func:`should_respond` — mirroring the
        Discord adapter's ``_on_message`` order so the SAME pipeline runs.
        """
        # Group -> supergroup migration: Telegram sends a service message carrying
        # ``migrate_to_chat_id`` on the OLD basic-group chat; the group's id changes
        # to a new ``-100…`` supergroup id. Anything keyed on the old chat id
        # (config, dedup) silently breaks if this is ignored, so log the old->new
        # mapping LOUDLY. The service message itself is not real content — emit
        # nothing for it.
        if message.migrate_to_chat_id is not None:
            logger.warning(
                "chatd.tg.migrate | group upgraded to supergroup | old={} new={}",
                canonical_chat_id(message.chat.id),
                canonical_chat_id(message.migrate_to_chat_id),
            )
            return
        # The mirror side of a migration also arrives (``migrate_from_chat_id`` on
        # the new supergroup's service message); it too is a service message with
        # no content, so drop it quietly.
        if message.migrate_from_chat_id is not None:
            logger.info(
                "chatd.tg.migrate | supergroup migrated-from | new={} old={}",
                canonical_chat_id(message.chat.id),
                canonical_chat_id(message.migrate_from_chat_id),
            )
            return

        chat_id = canonical_chat_id(message.chat.id)

        # Transport group allowlist (edge-level scope cut): when configured, a chat
        # outside it is dropped before normalization/gating. Empty = no cut; the
        # pure gates still decide every message.
        allowed_groups = self._telegram_config.allowed_groups
        is_group = message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}
        if allowed_groups and is_group and chat_id not in allowed_groups:
            logger.debug("chatd.tg.inbound | drop=group-not-allowed | chat={}", chat_id)
            return

        # Composite dedup: Telegram message ids are per-chat, so key on
        # (chat_id, message_id). A genuine redelivery in the same chat is
        # suppressed; the same id in a DIFFERENT chat admits (different message).
        message_id = str(message.message_id)
        if not self._dedup.check_and_remember(chat_id, message_id):
            logger.info(
                "chatd.tg.inbound | dedup=repeat | chat={} message_id={}", chat_id, message_id
            )
            return

        msg = map_message(
            message,
            self_username=self._self_username,
            self_user_id=self._self_user_id,
        )

        # A bot can only ever be itself here (Bot API never delivers another bot's
        # message), so "is this me?" is: the author is our own resolved id. When we
        # could not resolve identity (get_me failed), fall back to False — we would
        # rather double-handle than silently swallow, and the gates still filter.
        is_self = self._self_user_id is not None and msg.author == self._self_user_id

        # Intake hit-gate (issue #93): independent of the responder, evaluated on
        # the same message. Double-activation (a member-config mention that also
        # mentions us) is a deliberate, documented property — dedup of that is the
        # agentd-side sink's job, not the adapter's.
        await self._maybe_intake(msg, is_self=is_self)

        if not should_respond(msg, self._gate_config, is_self=is_self):
            return

        logger.info("chatd.tg.inbound | enqueue | chat={} thread={}", msg.channel_id, msg.thread_id)
        # Race the put against close, exactly like the poll/backoff/get awaits: a
        # FULL queue parks this await (the intended lossless backpressure), and
        # without the race close() could never free a poll loop parked here —
        # shutdown would deadlock under sustained overload. On close-wins the
        # message is deliberately released un-enqueued: the process is exiting and
        # the crash-boundary policy already covers it (the batch's offset was never
        # confirmed while parked, so Telegram redelivers it to the next process).
        put = asyncio.ensure_future(self._inbound.put(msg))
        closed = asyncio.ensure_future(self._closed.wait())
        try:
            await asyncio.wait({put, closed}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            put.cancel()
            closed.cancel()
            raise
        if put.done():
            closed.cancel()
            put.result()
            return
        put.cancel()
        logger.warning(
            "chatd.tg.inbound | released on close (queue full) | chat={} message_id={} | "
            "unconfirmed offset means Telegram redelivers this batch on next start",
            msg.channel_id,
            msg.message_id,
        )

    async def _maybe_intake(self, msg: InboundMessage, *, is_self: bool) -> None:
        """Fire the intake sink when ``msg`` is a deterministic intake hit.

        The sink may be sync or async. An async sink's coroutine is awaited HERE,
        inside the error boundary, so a real (async) bus publish actually runs and
        any failure it raises is contained — never a silently un-awaited coroutine
        that publishes nothing and escapes the boundary. Identical shape to the
        Discord adapter's ``_maybe_intake``.
        """
        if self._intake_sink is None or not self._intake_config.enabled:
            return
        if not intake_hit(msg, self._gate_config, self._intake_config, is_self=is_self):
            return
        # Receipt reaction (issue #109, signal 1 of 3): the MOMENT the hit-gate
        # fires, 👀 lands on the asker's message — fire-and-forget so it can
        # never block or delay the mint, independent of the sink's dedup /
        # responder-yield outcome (the human's "it landed" signal is about the
        # message being SEEN, not about which path answers it). Contained
        # inside the sender; the task set keeps a strong ref until done.
        if self._sender is not None and msg.message_id is not None:
            task = asyncio.create_task(
                self._sender.react_receipt(msg.channel_id, msg.message_id),
                name=f"tg-receipt-{msg.channel_id}-{msg.message_id}",
            )
            self._reaction_tasks.add(task)
            task.add_done_callback(self._reaction_tasks.discard)
        try:
            result = self._intake_sink(msg)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 — sink is caller code; never crash the gateway.
            logger.error(
                "chatd.tg.intake | sink failed | chat={} | {} | reason={}",
                msg.channel_id,
                type(exc).__name__,
                _redact_token(str(exc)),
            )

    async def _on_callback(self, callback: CallbackQuery) -> None:
        """Normalize an inline-keyboard tap and route it to the callback sink (#117).

        The adapter only normalizes surface coordinates and delegates all
        capability validation to the injected sink. No malformed, unknown,
        stale, mismatched, or replayed payload can create a steer at this layer.
        When no sink is wired the callback is still acknowledged safely.
        """
        mapped = map_callback(callback)
        if self._callback_sink is None:
            # A tap arrived but this deploy wired no callback sink (non-interactive
            # / inbound-only). Answer so the spinner stops; produce no steer.
            logger.info("chatd.tg.callback | no callback sink configured; answering only")
            await self._answer_callback_only(callback.id, channel_id=mapped.channel_id)
            return
        logger.debug("chatd.tg.callback | normalized tap")
        try:
            result = self._callback_sink(mapped)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 — sink is caller code; never crash the gateway.
            logger.error(
                "chatd.tg.callback | sink failed | {} | reason={}",
                type(exc).__name__,
                _redact_token(str(exc)),
            )
            # The sink owns the answer on its happy path; on a sink failure ack
            # here so the human's spinner still stops despite the failed steer.
            await self._answer_callback_only(callback.id, channel_id=mapped.channel_id)

    async def _answer_callback_only(
        self, callback_query_id: str, *, channel_id: str | None = None
    ) -> None:
        """Answer a callback query with no steer (spinner-stop), contained.

        Used for a no-sink deploy or a sink failure: the human's button must stop
        spinning even when nothing steers. Delegated to the sender's contained
        ``answer_callback``; a missing sender (inbound-only deploy) leaves the ack
        a no-op (nothing to answer with), logged.
        """
        if self._sender is None:
            logger.debug("chatd.tg.callback | no sender; cannot answer query (inbound-only)")
            return
        await self._sender.answer_callback(callback_query_id, channel_id=channel_id)

    async def events(self) -> AsyncIterator[InboundMessage]:
        """Yield gated, normalized inbound messages, then STOP when the adapter closes.

        Each iteration races the next queued message against :attr:`_closed`. While
        the adapter is live it yields inbound messages as they arrive. When
        :meth:`close` fires, the generator drains whatever is already queued (so a
        message enqueued just before close is not silently lost) and then RETURNS,
        ending the async iteration — it does NOT block forever on an empty queue
        the poll loop will never feed again. Both futures are cancelled on every
        path so no pending ``get``/``wait`` future leaks.
        """
        while True:
            if self._closed.is_set():
                # Closing: drain the backlog, then end the iteration.
                while not self._inbound.empty():
                    yield self._inbound.get_nowait()
                return
            getter = asyncio.ensure_future(self._inbound.get())
            closed = asyncio.ensure_future(self._closed.wait())
            try:
                done, _pending = await asyncio.wait(
                    {getter, closed}, return_when=asyncio.FIRST_COMPLETED
                )
            except asyncio.CancelledError:
                getter.cancel()
                closed.cancel()
                raise
            if getter in done:
                closed.cancel()
                yield getter.result()
                continue
            # Closed won the race: cancel the pending get, drain, and return.
            getter.cancel()
            closed.cancel()
            while not self._inbound.empty():
                yield self._inbound.get_nowait()
            return

    async def send(self, target: SendTarget, text: str, files: list[Path]) -> str | None:
        """Send via the injected :class:`TelegramSender` (issue #109).

        Delegation only — chunking, HTML rendering, identity routing, rate
        limiting and the FloodWait-to-caller contract all live in the sender.
        With no sender configured (inbound-only deploy) this keeps the #108
        fail-soft stub behavior: log, send nothing, return ``None``.
        """
        if self._sender is None:
            logger.info(
                "chatd.tg.send | no sender configured (inbound-only deploy) | target={}",
                target.resolve(),
            )
            return None
        return await self._sender.send(target, text, files)

    async def typing(self, target: SendTarget, on: bool) -> None:
        """Start/stop the sender's (chat, agent)-keyed typing keepalive (#109)."""
        if self._sender is None:
            logger.debug(
                "chatd.tg.typing | no sender configured | target={} on={}",
                target.resolve(),
                on,
            )
            return None
        await self._sender.typing(target, on)
        return None

    async def create_thread(self, channel_id: str, message_id: str, name: str) -> str | None:
        """Create a forum topic (or the reply-chain degrade marker) via the sender.

        With no sender configured returns ``None`` (the protocol's fail-soft
        degrade path) so a caller that would open a debate topic falls back to
        the parent chat rather than dropping content.
        """
        if self._sender is None:
            logger.info(
                "chatd.tg.thread | no sender configured (inbound-only deploy) | chat={}",
                channel_id,
            )
            return None
        return await self._sender.create_thread(channel_id, message_id, name)

    async def start(self) -> None:
        """Connect the bot, resolve identity, and run the poll loop forever.

        Fetches the token at connect time (never logged), caches this bot's own
        identity for the mention gate, then drives :meth:`_run_poll_loop` until
        :meth:`close` is called.

        Raises:
            RuntimeError: If the adapter has already been closed. A closed adapter
                has a set ``_closed`` event, so the poll loop would exit on its
                first iteration and the queue would never be fed again — starting
                one is a caller error, so fail loud rather than run a dead loop.
        """
        if self._closed.is_set():
            raise RuntimeError("chatd.tg: start() called on an already-closed adapter")
        bot = self._resolve_bot()
        logger.info("chatd.tg.start | connecting")
        await self._load_self(bot)
        try:
            await self._run_poll_loop(bot)
        finally:
            logger.info("chatd.tg.start | poll loop exited")

    async def close(self) -> None:
        """Signal the poll loop to stop; close the sender pool and bot session."""
        self._closed.set()
        for task in list(self._reaction_tasks):
            task.cancel()
        self._reaction_tasks.clear()
        if self._sender is not None:
            try:
                await self._sender.close()
            except Exception as exc:  # noqa: BLE001 — teardown must never raise.
                logger.warning(
                    "chatd.tg.close | sender close failed | {} | {}",
                    type(exc).__name__,
                    _redact_token(str(exc)),
                )
        bot = self._bot
        if bot is not None:
            try:
                await bot.session.close()
            except Exception as exc:  # noqa: BLE001 — teardown must never raise.
                logger.warning(
                    "chatd.tg.close | session close failed | {} | {}",
                    type(exc).__name__,
                    _redact_token(str(exc)),
                )
        logger.info("chatd.tg.close | done")


def _monotonic() -> float:
    """Return a monotonic clock reading (module default for dedup eviction)."""
    import time

    return time.monotonic()


__all__ = [
    "DEFAULT_INBOUND_QUEUE_MAX",
    "DEFAULT_TG_DEDUP_HORIZON_SECONDS",
    "DEFAULT_TG_DEDUP_MAX_ENTRIES",
    "GROUP_ANONYMOUS_BOT_ID",
    "POLL_FAILURE_BACKOFF_SECONDS",
    "BotFactory",
    "CallbackSink",
    "InboundCallback",
    "IntakeSink",
    "TelegramChannelAdapter",
    "TokenProvider",
    "canonical_chat_id",
    "map_callback",
    "map_message",
]
