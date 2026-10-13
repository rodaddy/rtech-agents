"""The chat-surface abstraction: ``ChannelAdapter`` + its wire dataclasses.

A ``ChannelAdapter`` is the seam between a concrete chat surface (Discord v1,
WhatsApp later) and the rest of chatd. It yields normalised
:class:`InboundMessage` values, sends text/files to a :class:`SendTarget`, and
toggles a typing indicator. Nothing above this layer touches discord.py.

The dataclasses are transport-neutral on purpose: the pure gate/thread
decision functions in :mod:`rtech_chatd.gates` consume :class:`InboundMessage`
without importing discord.py, so the whole behavioral contract is unit-testable
without a live gateway.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class InboundMessage:
    """A normalised inbound chat message.

    Attributes:
        channel_id: Parent channel id (the guild/text channel). For a message
            already inside a thread this is the thread's PARENT channel.
        thread_id: Thread id when the message arrived inside a thread, else
            None (a top-level channel message).
        author: Opaque author id (stringified; never a display name in logs).
        text: Message body with our own mention markup already stripped.
        is_bot: True when the author is a bot account.
        is_dm: True when the message arrived in a direct message.
        mentions_me: True when this agent's user is @mentioned (or, in DMs,
            implicitly addressed — the adapter sets this per surface rules).
        is_reply: True when the message is a reply to another message.
        mentions_other_bot: True when the message @mentions a bot that is NOT
            us (multi-agent channels: "talking to someone else").
        mentions_human: True when the message @mentions a human that is not us.
        guild_id: Guild (server) id, or None in a DM. Rides intake frames so
            the downstream OB channel lane is known.
        message_id: Opaque id of this message, or None when unavailable. Used as
            the reply reference so an intake answer lands on the asker's message.
        mentioned_ids: Opaque ids of every account @mentioned in this message
            (including us). The deterministic intake hit-gate matches configured
            trio member ids against this set.
    """

    channel_id: str
    thread_id: str | None
    author: str
    text: str
    is_bot: bool = False
    is_dm: bool = False
    mentions_me: bool = False
    is_reply: bool = False
    mentions_other_bot: bool = False
    mentions_human: bool = False
    guild_id: str | None = None
    message_id: str | None = None
    mentioned_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ChunkProgress:
    """A validated checkpoint for one chunked logical text send.

    The surface that chunks the text creates this value when its send raises
    after landing zero or more chunks. The edge stores it with the render-piece
    mark and returns it on redelivery. A surface MUST validate the plan fields
    before resuming; a changed plan restarts from its first chunk rather than
    silently skipping content.
    """

    chunks_landed: int
    chunk_count: int
    plan_fingerprint: str

    def __post_init__(self) -> None:
        """Reject malformed checkpoints before an adapter can honor one."""
        if self.chunk_count < 0 or not 0 <= self.chunks_landed <= self.chunk_count:
            raise ValueError("chunk progress must be within its chunk count")
        if not self.plan_fingerprint:
            raise ValueError("chunk progress requires a plan fingerprint")


_CHUNK_PROGRESS_ATTR = "_rtech_chatd_chunk_progress"


def attach_chunk_progress(exc: Exception, progress: ChunkProgress) -> None:
    """Attach a checkpoint without replacing the caller-visible exception type."""

    setattr(exc, _CHUNK_PROGRESS_ATTR, progress)


def chunk_progress_from(exc: Exception) -> ChunkProgress | None:
    """Return a sender-attached checkpoint, if this failure exposes one."""

    progress = getattr(exc, _CHUNK_PROGRESS_ATTR, None)
    return progress if isinstance(progress, ChunkProgress) else None


@dataclass(frozen=True)
class SendTarget:
    """Where an outbound message should land.

    ``thread_id`` takes precedence over ``channel_id`` at send time (a thread
    is addressed by its own id), mirroring the Hermes send contract.

    Attributes:
        channel_id: Parent channel id.
        thread_id: Optional thread id; when set, the message goes to the
            thread rather than the parent channel. OPAQUE to callers: a
            surface may return a namespaced synthetic id from
            ``create_thread`` (e.g. Telegram's non-forum reply-chain degrade)
            that only its own send path interprets.
        reply_to: Optional message id to attach a reply reference to.
        as_agent: Optional typed agent identity the send should be voiced
            under (multi-identity render). ``None`` (default) is
            the surface's own single/default identity — every earlier call
            site is byte-identical. A surface with per-identity clients
            (Telegram) resolves it against its pool and fails LOUD on an
            unknown id, never a fallback identity; a single-identity surface
            (Discord v1) ignores it.
        chunk_resume: Opaque checkpoint from a prior partial chunked send.
            Surfaces that support chunk resume validate it against their
            regenerated plan before skipping any chunk; other surfaces ignore
            it. ``None`` preserves the pre-cursor send contract.
    """

    channel_id: str
    thread_id: str | None = None
    reply_to: str | None = None
    as_agent: str | None = None
    chunk_resume: ChunkProgress | None = None

    def resolve(self) -> str:
        """Return the effective destination id (thread wins over channel)."""
        return self.thread_id if self.thread_id else self.channel_id


@runtime_checkable
class ChannelAdapter(Protocol):
    """The chat-surface protocol chatd programs against.

    Implementations own all surface-specific mechanics (intents, gateway
    events, rate limits). They MUST apply the gate decisions from
    :mod:`rtech_chatd.gates` before emitting an :class:`InboundMessage`, or
    document why a raw event is surfaced unfiltered.
    """

    def events(self) -> AsyncIterator[InboundMessage]:
        """Yield gated, normalised inbound messages until the adapter closes."""
        ...

    async def send(self, target: SendTarget, text: str, files: list[Path]) -> str | None:
        """Send ``text`` (chunked as needed) and any ``files`` to ``target``.

        Returns the surface message id of the FIRST chunk sent, or ``None`` when
        the surface does not expose one (or nothing was sent). Callers use it as
        a thread anchor (the debate TLDR message opens its own thread); it is
        advisory — a ``None`` degrades, never fails.
        """
        ...

    async def typing(self, target: SendTarget, on: bool) -> None:
        """Start (``on=True``) or stop the typing indicator for ``target``."""
        ...

    async def create_thread(self, channel_id: str, message_id: str, name: str) -> str | None:
        """Create ONE thread anchored to ``message_id`` in ``channel_id``.

        The outbound counterpart of the inbound auto-thread creation: an intake
        triage edge uses it to open a single debate thread for a plain-channel
        ask so full turn text lands in-thread instead of walling the channel.

        Returns the new thread id, or ``None`` when the surface cannot create the
        thread (the caller degrades: full text falls back to the parent channel
        so debate content is never dropped). It MUST NOT raise for an ordinary
        create failure — it fails soft by returning ``None``.
        """
        ...


__all__ = [
    "ChannelAdapter",
    "ChunkProgress",
    "InboundMessage",
    "SendTarget",
    "attach_chunk_progress",
    "chunk_progress_from",
]
