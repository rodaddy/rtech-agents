"""Map chat edge messages to and from typed bus frames."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

from loguru import logger
from rtech_frames import ChannelOrigin, StatusFrame, TaskFrame, TurnFrame

from .channel import InboundMessage, SendTarget


class _HasOrigin(Protocol):
    """A frame carrying an optional channel origin (TaskFrame or TurnFrame).

    Both frames expose ``origin``, so the reply-target resolver reads
    the field structurally rather than binding to one concrete frame type — the
    intake path resolves from a :class:`TaskFrame`, the external reply-render path
    from a :class:`TurnFrame`.
    """

    @property
    def origin(self) -> ChannelOrigin | None: ...


DEFAULT_ORIGIN_EDGE = "chatd"
DEFAULT_INTAKE_ORIGIN_EDGE = "chatd-intake"
DEFAULT_SCOPE: Literal["single"] = "single"

# A pure text->text filter applied to rendered text before send. It MUST be
# total: return a possibly empty string, never raise for ordinary content.
OutboundFilter = Callable[[str], str]

_STATUS_TEXT: dict[str, str] = {
    "working": "Working...",
    "input_required": "Input required.",
    "blocked": "Blocked.",
}
_VERIFYING_TEXT = "Got it — I’m verifying a few things before I answer."


@dataclass(frozen=True)
class OutboundSend:
    """Carry a resolved outbound send target and filtered display text."""

    target: SendTarget
    text: str

    @property
    def suppressed(self) -> bool:
        """Return True when filtering suppressed the whole message."""
        return not self.text.strip()


def inbound_to_task_frame(
    msg: InboundMessage,
    *,
    task_id: str,
    stack: str,
    origin_edge: str = DEFAULT_ORIGIN_EDGE,
    scope: Literal["single", "duo", "trio"] = DEFAULT_SCOPE,
    participants: list[str] | None = None,
) -> TaskFrame:
    """Mint a typed task frame from a gated raw human input message.

    The inbound edge does not infer stance, done state, participants, or control
    signals. It only copies the requester/text fields and supplies the typed
    frame identity fields provided by the caller. ``stack`` is the typed
    stack-scope stamp — REQUIRED addressing config supplied by the
    caller, never a defaulted shared literal and never a control signal
    inferred by the edge. ``participants`` is likewise caller-supplied typed
    identity data (duo/trio frames require it).

    Raises:
        ValueError: If ``stack`` is blank — an unscoped task would be dropped
            by every stack-guarded orchestrator, so the edge fails closed.
    """

    if not stack.strip():
        logger.error("bridge.inbound_to_task_frame | blank stack rejected task_id={}", task_id)
        raise ValueError("stack must be non-empty; there is no default stack id")
    frame = TaskFrame(
        id=task_id,
        origin_edge=origin_edge,
        requester=msg.author,
        text=msg.text,
        scope=scope,
        participants=participants or [],
        stack=stack,
        created_at=datetime.now(UTC),
    )
    logger.debug(
        "bridge.inbound_to_task_frame | origin_edge={} channel={} thread={} task_id={}",
        origin_edge,
        msg.channel_id,
        msg.thread_id,
        task_id,
    )
    return frame


def inbound_to_intake_task_frame(
    msg: InboundMessage,
    *,
    task_id: str,
    stack: str,
    triage_agent: str,
    origin_edge: str = DEFAULT_INTAKE_ORIGIN_EDGE,
    scope: Literal["single", "duo", "trio"] = DEFAULT_SCOPE,
    participants: list[str] | None = None,
    debate: bool = False,
    arbiter: str | None = None,
) -> TaskFrame:
    """Mint a triage TaskFrame from an intake-gated channel message.

    Like :func:`inbound_to_task_frame`, ``stack`` is REQUIRED addressing config:
    a blank stack would mint a task every stack-guarded orchestrator
    drops, so the edge fails closed. ``created_at`` is always stamped (the
    agentd-side dedup/retry horizon of fails closed on frames it cannot age).

    Unlike the plain mint, this packs the originating chat-surface identity into
    ``origin`` (:class:`rtech_frames.ChannelOrigin`) so the triage agent knows the
    OB channel lane and the reply destination. The mint stays a pure field copy —
    it infers no stance, participants, or control signals.

    Args:
        msg: The intake-gated inbound message (already a confirmed hit).
        task_id: Caller-supplied unique task id.
        stack: Required stack-scope stamp; blank fails closed.
        triage_agent: The configured triage agent id (the deployed
            ``agentd-<name>``, e.g. ``agentd-nagatha``); folded into
            ``origin_edge`` so downstream logs/subjects name the destination.
        origin_edge: Edge label prefix (default ``chatd-intake``).
        scope: Task scope (default ``single`` — one triage turn).
        participants: Caller-supplied duo/trio identities, if any.
        debate: When True, marks the carried ``origin`` as a MULTI-participant
            debate (duo/trio) so a triage edge renders the answers as per-agent
            TLDRs in-channel plus full text in-thread. Default False keeps the
            single-scope in-channel full-text render. It is a render DECISION
            carried typed on the frame, never inferred from surface geometry.
        arbiter: The debate ROLE authority (the moderating participant that writes
            the final TLDR ruling instead of arguing itself). Stamped only on a
            debate mint (the triage agent id); ``None`` on a single-scope mint. It
            MUST be one of ``participants`` (:class:`TaskFrame` validates this).

    Returns:
        A :class:`TaskFrame` carrying the question, channel identity, and stamp.

    Raises:
        ValueError: If ``stack`` is blank.
    """
    if not stack.strip():
        logger.error(
            "bridge.inbound_to_intake_task_frame | blank stack rejected task_id={}", task_id
        )
        raise ValueError("stack must be non-empty; there is no default stack id")

    origin = ChannelOrigin(
        channel_id=msg.channel_id,
        guild_id=msg.guild_id,
        thread_id=msg.thread_id,
        message_id=msg.message_id,
        asker=msg.author,
        debate=debate,
    )
    frame = TaskFrame(
        id=task_id,
        origin_edge=f"{origin_edge}:{triage_agent}",
        requester=msg.author,
        text=msg.text,
        scope=scope,
        participants=participants or [],
        stack=stack,
        created_at=datetime.now(UTC),
        origin=origin,
        arbiter=arbiter,
    )
    logger.info(
        "bridge.inbound_to_intake_task_frame | triage={} guild={} channel={} thread={} task_id={}",
        triage_agent,
        msg.guild_id,
        msg.channel_id,
        msg.thread_id,
        task_id,
    )
    return frame


def intake_reply_target(frame: _HasOrigin) -> SendTarget | None:
    """Resolve where an intake answer should land, from the frame's origin.

    The TurnFrame reply path already exists (:func:`turn_frame_to_send`); this
    only supplies the :class:`SendTarget` so the answer returns to the SAME
    channel/thread the question came from, as a reply to the asker's message.
    Accepts any origin-carrying frame (a :class:`TaskFrame` on the intake mint
    path, a :class:`TurnFrame` on the external reply-render path).
    Returns None when the frame carries no channel origin (a non-intake frame).
    """
    origin = frame.origin
    if origin is None:
        return None
    return SendTarget(
        channel_id=origin.channel_id,
        thread_id=origin.thread_id,
        reply_to=origin.message_id,
    )


def turn_frame_to_send(
    frame: TurnFrame,
    *,
    target: SendTarget,
    outbound_filter: OutboundFilter | None = None,
    attributed: bool = False,
) -> OutboundSend:
    """Render a turn frame's visible text to a chat send.

    ``TurnFrame.visible_text`` is the only turn content rendered by chatd. Stance
    and refs remain typed bus fields for other consumers. With ``attributed``
    the send is prefixed with the typed ``agent_id`` so every participant's turn
    renders as its OWN visibly-attributed reply (multi-agent render-back) — pure
    presentation from a typed field, never a control signal.
    """

    text = frame.visible_text + _verification_suffix(frame)
    if attributed:
        text = f"**{frame.agent_id}**: {text}"
    return _text_to_send(text, target=target, outbound_filter=outbound_filter)


def _verification_suffix(frame: TurnFrame) -> str:
    """Render typed evidence metadata for a full turn response."""

    verification = frame.verification
    if verification is None:
        return ""

    sections: list[str] = []
    source_lines = list(
        dict.fromkeys(
            f"- {source.label}" + (f" — {source.locator}" if source.locator else "")
            for source in verification.sources
        )
    )
    if source_lines:
        sections.append("Sources checked\n" + "\n".join(source_lines))

    # Every non-empty limitations list is surfaced regardless of status. A caveat
    # rides the report only because the runtime chose to keep it (blank entries are
    # rejected at frame construction), so a grounded/not_applicable turn that still
    # carries a limitation must show it — dropping it silently would hide a legible
    # verification caveat the security floor requires be surfaced, never dropped.
    if verification.limitations:
        limitation_lines = list(dict.fromkeys(f"- {item}" for item in verification.limitations))
        sections.append("Limitations\n" + "\n".join(limitation_lines))

    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


# Char budget for the ``because`` clause in a debate TLDR line. A TLDR is a
# scannable one-liner in the parent channel; the full reasoning lives in-thread,
# so the stance reason is capped (with an ellipsis on truncation) rather than
# spilling the whole turn into the channel.
DEBATE_TLDR_BECAUSE_MAX = 200


def turn_frame_to_tldr(frame: TurnFrame, *, named: bool = True) -> str:
    """Render a debate turn as a short, STANCE-DERIVED channel TLDR line.

    The line is built ONLY from typed fields — the author ``agent_id`` and the
    typed ``stance`` (``position`` + a length-capped ``because``) — never scraped
    from the turn's prose ``visible_text`` (the full text goes to the thread).

    ``named`` controls the attribution prefix. True (default, Discord v1): the
    agent name is prefixed because all posts go out under the single edge bot
    identity. False: the surface posts under the agent's OWN bot,
    so the prefix would be redundant ventriloquism — the identity IS the
    attribution. Shape: ``[**<agent>** — ]<position>: <because (≤ cap)>``.
    """
    because = frame.stance.because
    if len(because) > DEBATE_TLDR_BECAUSE_MAX:
        because = because[: DEBATE_TLDR_BECAUSE_MAX - 1].rstrip() + "…"
    line = f"{frame.stance.position}: {because}"
    if named:
        line = f"**{frame.agent_id}** — {line}"
    return line


def status_frame_to_send(
    frame: StatusFrame,
    *,
    target: SendTarget,
    outbound_filter: OutboundFilter | None = None,
    attributed: bool = False,
) -> OutboundSend:
    """Render a minimal typed status indicator to a chat send."""

    text = _VERIFYING_TEXT if frame.phase == "verifying" else _STATUS_TEXT[frame.state]
    if attributed:
        text = f"**{frame.agent_id}**: {text}"
    return _text_to_send(text, target=target, outbound_filter=outbound_filter)


def _text_to_send(
    text: str,
    *,
    target: SendTarget,
    outbound_filter: OutboundFilter | None,
) -> OutboundSend:
    """Apply the outbound filter to already-rendered text."""

    rendered = text
    if outbound_filter is not None:
        rendered = outbound_filter(text)
        logger.debug(
            "bridge.text_to_send | filtered in_len={} out_len={}",
            len(text),
            len(rendered),
        )

    send = OutboundSend(target=target, text=rendered)
    if send.suppressed:
        logger.info("bridge.text_to_send | suppressed by outbound filter")
    else:
        logger.debug(
            "bridge.text_to_send | ok | channel={} thread={} len={}",
            target.channel_id,
            target.thread_id,
            len(rendered),
        )
    return send


__all__ = [
    "DEBATE_TLDR_BECAUSE_MAX",
    "DEFAULT_INTAKE_ORIGIN_EDGE",
    "DEFAULT_ORIGIN_EDGE",
    "DEFAULT_SCOPE",
    "OutboundFilter",
    "OutboundSend",
    "inbound_to_intake_task_frame",
    "inbound_to_task_frame",
    "intake_reply_target",
    "status_frame_to_send",
    "turn_frame_to_send",
    "turn_frame_to_tldr",
]
