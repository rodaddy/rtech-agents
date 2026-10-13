"""discord.py wiring of the :class:`ChannelAdapter` protocol.

This is the only module that imports discord.py. It:

* declares intents (message_content, guild_messages; dm_messages only when the
  gate policy explicitly allows DMs);
* maps a discord ``Message`` to a normalised :class:`InboundMessage`;
* applies the pure gates (:func:`should_respond` / :func:`thread_decision`);
* creates an auto-thread when the decision says so and retargets that trigger;
* sends via :func:`chunk_message`, resolving ``thread_id > channel_id``;
* runs a 429-tolerant 12s typing keepalive loop.

The bot token is NEVER a literal here. It is resolved at runtime from a config
ref via an injected ``token_provider`` callable, and never logged.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import discord
from loguru import logger

from .channel import InboundMessage, SendTarget
from .chunking import chunk_message
from .config import GateConfig, IntakeConfig, ThreadConfig
from .gates import intake_hit, should_respond, thread_decision

# Typing indicator lasts ~10s; refresh every 12s.
_TYPING_INTERVAL_SECONDS = 12.0
# Default backoff when a 429 carries no retry-after.
_TYPING_DEFAULT_BACKOFF_SECONDS = 5.0

# A zero-arg callable returning the bot token (from a config ref). Kept as a
# provider so the token is fetched at connect time and never stored/logged.
TokenProvider = Callable[[], str]

# Invoked with the ORIGINAL (non-thread-retargeted) inbound message when a
# deterministic intake hit fires (issue #93). The sink owns minting + publishing
# the triage TaskFrame; the adapter never touches the bus. Kept separate from the
# responder queue so intake channel identity is the message's own channel.
#
# The sink may be sync OR async: the in-repo bus publisher is a coroutine, so a
# sync-only contract would let an ``async def`` sink return an un-awaited
# coroutine (publishing nothing, and bypassing the error boundary). The call site
# awaits the result when it is awaitable, inside the existing error boundary.
IntakeSink = Callable[[InboundMessage], Awaitable[None] | None]


def _map_message(message: discord.Message, self_user: discord.abc.User | None) -> InboundMessage:
    """Map a discord ``Message`` to a normalised :class:`InboundMessage`.

    Strips our own mention markup from the text, resolves DM/thread/parent, and
    computes the mention flags the gates consume.
    """
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_thread = isinstance(message.channel, discord.Thread)

    thread_id: str | None = None
    channel_id = str(message.channel.id)
    if is_thread:
        thread_id = str(message.channel.id)
        parent_id = getattr(message.channel, "parent_id", None)
        if parent_id:
            channel_id = str(parent_id)

    mentions_me = self_user is not None and self_user in message.mentions
    # In a DM the bot is implicitly addressed.
    if is_dm:
        mentions_me = True

    mentions_other_bot = any(m.bot and m != self_user for m in message.mentions)
    mentions_human = any((not m.bot) and m != self_user for m in message.mentions)
    mentioned_ids = frozenset(str(m.id) for m in message.mentions)

    guild = getattr(message, "guild", None)
    guild_id = str(guild.id) if guild is not None else None

    text = (message.content or "").strip()
    if self_user is not None:
        text = text.replace(f"<@{self_user.id}>", "").replace(f"<@!{self_user.id}>", "").strip()

    return InboundMessage(
        channel_id=channel_id,
        thread_id=thread_id,
        author=str(message.author.id),
        text=text,
        is_bot=bool(getattr(message.author, "bot", False)),
        is_dm=is_dm,
        mentions_me=mentions_me,
        is_reply=message.reference is not None,
        mentions_other_bot=mentions_other_bot,
        mentions_human=mentions_human,
        guild_id=guild_id,
        message_id=str(message.id),
        mentioned_ids=mentioned_ids,
    )


class DiscordChannelAdapter:
    """A :class:`ChannelAdapter` backed by discord.py.

    Gating decisions come from the pure :mod:`rtech_chatd.gates`; this class
    owns only the surface mechanics. Inbound gated messages are pushed onto an
    async queue that :meth:`events` drains.
    """

    def __init__(
        self,
        *,
        token_provider: TokenProvider,
        gate_config: GateConfig,
        thread_config: ThreadConfig,
        intake_config: IntakeConfig | None = None,
        intake_sink: IntakeSink | None = None,
        client: discord.Client | None = None,
    ) -> None:
        """Args:
        token_provider: Zero-arg callable returning the bot token at connect
            time (from a config ref — never a literal, never logged).
        gate_config: Gate configuration for :func:`should_respond`.
        thread_config: Thread configuration for :func:`thread_decision`.
        intake_config: Optional intake hit-gate config (issue #93). When None or
            disabled, no intake behavior runs (existing deploys unchanged).
        intake_sink: Callback invoked with the original inbound message on an
            intake hit. Required for intake to fire; the sink mints/publishes the
            triage TaskFrame.
        client: Optional pre-built client (test seam). When None, a client is
            built with the required intents.
        """
        self._token_provider = token_provider
        self._gate_config = gate_config
        self._thread_config = thread_config
        self._intake_config = intake_config or IntakeConfig()
        self._intake_sink = intake_sink
        self._inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self._typing_tasks: dict[str, asyncio.Task[None]] = {}

        if client is not None:
            self._client = client
        else:
            intents = discord.Intents.default()
            intents.message_content = True
            intents.guild_messages = True
            # DM events follow the gate policy (security floor: a DM is a
            # free-response surface outside the designated inbox, so it stays
            # OFF unless explicitly allowed).
            intents.dm_messages = gate_config.allow_dms
            self._client = discord.Client(intents=intents)

        self._register_handlers()

    def _register_handlers(self) -> None:
        adapter = self

        @self._client.event
        async def on_message(message: discord.Message) -> None:  # noqa: D401
            await adapter._on_message(message)

    async def _on_message(self, message: discord.Message) -> None:
        """Gate an incoming message and enqueue it when the agent should reply."""
        self_user = self._client.user
        is_self = self_user is not None and message.author == self_user

        # Skip Discord system messages (thread renames/pins/joins); allow
        # default + reply types only.
        if message.type not in {discord.MessageType.default, discord.MessageType.reply}:
            return

        msg = _map_message(message, self_user)

        # Intake hit-gate (issue #93): independent of the responder. Evaluated on
        # the original (pre-thread-retarget) message so the minted frame carries
        # the channel identity the question actually arrived in.
        #
        # Double-activation is a known, deliberate property of this slice: when
        # the triage agent is directly @mentioned AND is a configured intake
        # member_id, BOTH this intake path and should_respond below fire for the
        # same message. Dedup (keyed by message_id) is the agentd-side sink's
        # responsibility, NOT the adapter's — this slice does not suppress it.
        await self._maybe_intake(msg, is_self=is_self)

        if not should_respond(
            msg,
            self._gate_config,
            is_self=is_self,
        ):
            return

        # Auto-thread decision — create the thread and re-target the message.
        decision = thread_decision(msg, self._thread_config)
        if decision.create:
            try:
                thread = await message.create_thread(name=decision.name, auto_archive_duration=1440)
            except discord.DiscordException as exc:
                logger.warning("chatd.autothread | create failed | reason={}", exc)
                thread = None
            if thread is not None:
                thread_id = str(thread.id)
                msg = InboundMessage(
                    channel_id=str(message.channel.id),
                    thread_id=thread_id,
                    author=msg.author,
                    text=msg.text,
                    is_bot=msg.is_bot,
                    is_dm=msg.is_dm,
                    mentions_me=msg.mentions_me,
                    is_reply=msg.is_reply,
                    mentions_other_bot=msg.mentions_other_bot,
                    mentions_human=msg.mentions_human,
                    guild_id=msg.guild_id,
                    message_id=msg.message_id,
                    mentioned_ids=msg.mentioned_ids,
                )
        logger.info(
            "chatd.inbound | enqueue | channel={} thread={}",
            msg.channel_id,
            msg.thread_id,
        )
        await self._inbound.put(msg)

    async def _maybe_intake(self, msg: InboundMessage, *, is_self: bool) -> None:
        """Fire the intake sink when ``msg`` is a deterministic intake hit.

        The sink may be sync or async. An async sink's coroutine is awaited HERE,
        inside the error boundary, so a real (async) bus publish actually runs and
        any failure it raises is contained — never a silently un-awaited coroutine
        that publishes nothing and escapes the boundary.
        """
        if self._intake_sink is None or not self._intake_config.enabled:
            return
        if not intake_hit(msg, self._gate_config, self._intake_config, is_self=is_self):
            return
        try:
            result = self._intake_sink(msg)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 — sink is caller code; never crash the gateway.
            logger.error("chatd.intake | sink failed | channel={} reason={}", msg.channel_id, exc)

    async def events(self) -> AsyncIterator[InboundMessage]:
        """Yield gated, normalised inbound messages until cancelled."""
        while True:
            msg = await self._inbound.get()
            yield msg

    async def _resolve_channel(self, target: SendTarget) -> discord.abc.Messageable | None:
        """Resolve the destination channel (thread id wins over channel id)."""
        target_id = int(target.resolve())
        channel = self._client.get_channel(target_id)
        if channel is None:
            try:
                channel = await self._client.fetch_channel(target_id)
            except discord.DiscordException as exc:
                logger.error(
                    "chatd.send | channel resolve failed | id={} reason={}", target_id, exc
                )
                return None
        if isinstance(channel, discord.abc.Messageable):
            return channel
        logger.error("chatd.send | resolved channel is not messageable | id={}", target_id)
        return None

    async def send(self, target: SendTarget, text: str, files: list[Path]) -> str | None:
        """Send ``text`` (chunked) and ``files`` to ``target``.

        The text is split with :func:`chunk_message`; a reply reference (when
        ``target.reply_to`` is set) attaches to the first chunk only. Returns the
        FIRST sent message's id (the thread-anchor contract on
        :class:`ChannelAdapter`), or ``None`` when nothing was sent.

        Raises:
            RuntimeError: If the destination cannot be resolved to a messageable
                channel — failed sends fail LOUD; returning normally would let
                callers record a delivery that never happened.
        """
        channel = await self._resolve_channel(target)
        if channel is None:
            raise RuntimeError(
                f"chatd: send destination unresolvable/not messageable (target={target.resolve()})"
            )

        reference: discord.MessageReference | discord.Message | None = None
        if target.reply_to:
            try:
                ref_msg = await channel.fetch_message(int(target.reply_to))
                reference = ref_msg.to_reference(fail_if_not_exists=False)
            except discord.DiscordException as exc:
                logger.debug("chatd.send | reply-ref fetch failed | reason={}", exc)

        chunks = chunk_message(text) if text.strip() else []
        discord_files = [discord.File(str(path)) for path in files]

        first_sent: discord.Message | None = None
        for index, chunk in enumerate(chunks):
            # Reply reference (if any) attaches to the first chunk only. A None
            # reference is omitted entirely — discord.py's send() reference param
            # does not accept None.
            chunk_ref = reference if index == 0 else None
            send_files = discord_files if index == 0 else []
            try:
                if chunk_ref is not None:
                    sent = await channel.send(content=chunk, files=send_files, reference=chunk_ref)
                else:
                    sent = await channel.send(content=chunk, files=send_files)
            except discord.DiscordException as exc:
                if chunk_ref is not None:
                    logger.warning(
                        "chatd.send | reply ref rejected, retrying plain | reason={}", exc
                    )
                    sent = await channel.send(content=chunk, files=send_files)
                else:
                    logger.error("chatd.send | send failed | reason={}", exc)
                    raise
            if first_sent is None:
                first_sent = sent

        # Files with no text still need delivery.
        if not chunks and discord_files:
            first_sent = await channel.send(files=discord_files)

        logger.info("chatd.send | ok | target={} chunks={}", target.resolve(), len(chunks))
        return str(first_sent.id) if first_sent is not None else None

    async def create_thread(self, channel_id: str, message_id: str, name: str) -> str | None:
        """Create ONE thread anchored to ``message_id`` in ``channel_id``.

        Mirrors the inbound auto-thread creation (:meth:`_on_message`): it resolves
        the parent channel, fetches the anchor message, and calls
        ``message.create_thread`` with the SAME ``auto_archive_duration`` (1440) so
        the outbound debate thread uses the identical Discord API objects. Fails
        soft — a resolve/fetch/create failure is logged and returns ``None`` so the
        edge degrades the debate full text to the parent channel rather than
        dropping it. Never raises for an ordinary create failure.
        """
        channel = await self._resolve_channel(SendTarget(channel_id=channel_id))
        if channel is None:
            logger.warning("chatd.thread | create failed | unresolvable | id={}", channel_id)
            return None
        try:
            anchor = await channel.fetch_message(int(message_id))
            thread = await anchor.create_thread(name=name, auto_archive_duration=1440)
        except discord.DiscordException as exc:
            logger.warning("chatd.thread | create failed | reason={}", exc)
            return None
        logger.info("chatd.thread | created | channel={} thread={}", channel_id, thread.id)
        return str(thread.id)

    async def typing(self, target: SendTarget, on: bool) -> None:
        """Start or stop the 429-tolerant typing keepalive for ``target``."""
        key = target.resolve()
        if not on:
            task = self._typing_tasks.pop(key, None)
            if task is not None:
                task.cancel()
            return
        if key in self._typing_tasks:
            return
        self._typing_tasks[key] = asyncio.create_task(self._typing_loop(key))

    async def _typing_loop(self, channel_id: str) -> None:
        """Refresh the typing indicator every 12s; tolerate 429 backoff."""
        try:
            while True:
                try:
                    channel = self._client.get_channel(int(channel_id))
                    if isinstance(channel, discord.abc.Messageable):
                        await channel.typing()
                except asyncio.CancelledError:
                    return
                except discord.HTTPException as exc:
                    retry_after = getattr(exc, "retry_after", None)
                    backoff = float(retry_after) if retry_after else _TYPING_DEFAULT_BACKOFF_SECONDS
                    logger.warning(
                        "chatd.typing | rate-limited/failed | channel={} backoff={:.1f}s",
                        channel_id,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue
                await asyncio.sleep(_TYPING_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            pass
        finally:
            self._typing_tasks.pop(channel_id, None)

    async def start(self) -> None:
        """Connect and run the client. Fetches the token at connect time."""
        token = self._token_provider()
        if not token:
            raise RuntimeError("chatd: bot token not provisioned (empty token from provider)")
        logger.info("chatd.start | connecting")
        await self._client.start(token)

    async def close(self) -> None:
        """Cancel typing loops and close the client."""
        for task in list(self._typing_tasks.values()):
            task.cancel()
        self._typing_tasks.clear()
        await self._client.close()
        logger.info("chatd.close | done")


__all__ = [
    "DiscordChannelAdapter",
    "IntakeSink",
    "TokenProvider",
]
