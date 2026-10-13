"""NATS ingress/egress — the typed-frame seam for the managed session.

The bus carries typed :mod:`rtech_frames` frames; agentd is a **dumb edge** that
translates at the boundary and holds NO coordination state. ``TaskFrame`` comes
in, ``TurnFrame`` goes out. Done is orchestrator-derived from the frame graph —
agentd never emits it.

Inbound (subscriptions):

* ``direct(env, agent)`` and the per-agent worker lanes from
  ``fleet_subscriber.role.lanes_for_role`` (dispatch / relay / wake / inbox).
  Dispatch is subscribed in BOTH the bare and the per-channel-wildcard form —
  the known fleet gotcha where a bare subscriber never sees a per-channel
  dispatch.
* Envelope kinds drive behavior:
  - ``chat`` / ``dispatch`` / ``relay`` / ``wake`` / ``inbox`` → a ``TaskFrame``
    is composed at the edge and its ``text`` drives a supervisor turn (text
    mapped via ``fleet_subscriber.mapping.envelope_to_message``).
  - ``steer`` → ``session.send`` into the live turn (inline steering).
  - ``interrupt`` → ``session.interrupt``.

Outbound (via a supervisor event sink): each completed turn becomes a typed
``TurnFrame`` — ``visible_text`` is the guard-sanitized buffered turn text and
``stance`` is the typed stance the engine filled (never scraped from text). The
frame is serialized into the envelope payload. Chat turns reply with
``kind="chat_reply"`` (correlation_id = inbound id) on ``direct(env, "chatd")``.
Dispatched work replies with ``kind="turn"`` on ``agent_status(env, task_id)``.

Turn↔inbound correlation: the session is single and serial, so inbound
turn-producing envelopes are queued FIFO and each ``TurnDone`` is paired with the
oldest pending context.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from fleet_nats import Envelope, subjects
from fleet_subscriber.mapping import envelope_to_message
from fleet_subscriber.role import SubscriberRole, lanes_for_role
from loguru import logger
from rtech_frames import TaskFrame, TurnFrame

from .engine import EngineEvent, TurnDone, failure_visible_text
from .outbound_guard import sanitize_outbound_text
from .supervisor import Publisher, Supervisor

# Inbound kinds that start a supervisor turn (vs. steer/interrupt control kinds).
_TURN_KINDS = frozenset({"chat", "dispatch", "relay", "wake", "inbox"})
_STEER_KIND = "steer"
_INTERRUPT_KIND = "interrupt"

# The chat egress target: replies to chat turns go to the chat daemon's direct lane.
_CHATD_AGENT = "chatd"

# The origin edge label stamped onto an inbound ``TaskFrame`` — this agent's NATS
# seam is the edge that lifted the envelope onto the typed-frame contract.
_ORIGIN_EDGE = "agentd-nats"
# agentd manages ONE session, so an inbound task is always single-agent scope.
_TASK_SCOPE: Literal["single"] = "single"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_id() -> str:
    return uuid4().hex


@dataclass(frozen=True)
class PendingTurn:
    """The inbound context needed to route a turn's reply back correctly."""

    origin_kind: str
    inbound_id: str
    task_id: str | None
    channel: str | None


class NatsIngress:
    """Subscribes fleet lanes, drives the session, and publishes replies."""

    def __init__(
        self,
        *,
        agent: str,
        env: str,
        supervisor: Supervisor,
        publisher: Publisher,
        default_channel: str | None = None,
        role: SubscriberRole = SubscriberRole.WORKER,
    ) -> None:
        self._agent = agent
        self._env = env
        self._supervisor = supervisor
        self._publish = publisher
        self._default_channel = default_channel
        self._role = role
        self._pending: deque[PendingTurn] = deque()
        self._lock = asyncio.Lock()
        # Register the reply sink so every turn's output is routed out.
        supervisor.add_event_sink(self._on_engine_event)

    # --- subject planning -----------------------------------------------------

    def subscription_subjects(self) -> list[str]:
        """The concrete subjects this agent subscribes to (deduped, ordered).

        Includes ``direct``, every role lane, and — critically — BOTH the bare
        and per-channel-wildcard dispatch forms so per-channel dispatches are
        never missed.
        """
        subs: list[str] = [subjects.direct(self._env, self._agent)]
        for lane in lanes_for_role(self._role):
            if lane == "dispatch":
                subs.append(subjects.agent_dispatch(self._env, self._agent))
                subs.append(subjects.agent_dispatch_wildcard(self._env, self._agent))
            elif lane == "relay":
                subs.append(subjects.agent_relay(self._env, self._agent))
            elif lane == "wake":
                subs.append(subjects.agent_wake(self._env, self._agent))
            elif lane == "inbox":
                subs.append(subjects.agent_inbox(self._env, self._agent))
        # Dedupe, preserve order.
        seen: set[str] = set()
        ordered: list[str] = []
        for s in subs:
            if s not in seen:
                seen.add(s)
                ordered.append(s)
        logger.info("nats_ingress.subjects | agent={} subjects={}", self._agent, ordered)
        return ordered

    # --- inbound --------------------------------------------------------------

    async def handle_envelope(self, envelope: Envelope) -> None:
        """Dispatch one inbound envelope by kind. The FleetBus handler entry."""
        logger.info(
            "nats_ingress.inbound | kind={} id={} task_id={} channel={}",
            envelope.kind,
            envelope.id,
            envelope.task_id,
            envelope.channel,
        )
        try:
            if envelope.kind == _STEER_KIND:
                await self._handle_steer(envelope)
            elif envelope.kind == _INTERRUPT_KIND:
                await self._handle_interrupt(envelope)
            elif envelope.kind in _TURN_KINDS:
                await self._handle_turn(envelope)
            else:
                logger.warning(
                    "nats_ingress.inbound | unhandled kind={} id={}", envelope.kind, envelope.id
                )
        except Exception as exc:
            logger.error(
                "nats_ingress.inbound | handler failed kind={} id={} reason={}",
                envelope.kind,
                envelope.id,
                exc,
            )

    async def _handle_turn(self, envelope: Envelope) -> None:
        """Compose a ``TaskFrame`` at the edge and drive its text as a turn."""
        synthetic = envelope_to_message(
            envelope,
            home_channel=self._default_channel,
            synthetic_user_id=envelope.sender or "fleet-nats",
        )
        if synthetic is None:
            logger.warning(
                "nats_ingress.turn | skipped (no text/channel) | kind={} id={}",
                envelope.kind,
                envelope.id,
            )
            return
        # Dumb-edge translation: lift the envelope onto the typed contract. The
        # session consumes ``TaskFrame.text``; no control signal rides the prose.
        task = TaskFrame(
            id=envelope.task_id or envelope.id,
            origin_edge=_ORIGIN_EDGE,
            requester=envelope.sender or "fleet-nats",
            text=synthetic.text,
            scope=_TASK_SCOPE,
            created_at=datetime.now(UTC),
        )
        pending = PendingTurn(
            origin_kind=envelope.kind,
            inbound_id=envelope.id,
            task_id=envelope.task_id,
            channel=envelope.channel or self._default_channel,
        )
        async with self._lock:
            self._pending.append(pending)
        logger.debug(
            "nats_ingress.turn | sending turn | id={} task_frame_id={} len={}",
            envelope.id,
            task.id,
            len(task.text),
        )
        await self._supervisor.session.send(task.text)

    async def _handle_steer(self, envelope: Envelope) -> None:
        """Inject a steer envelope's text into the session.

        If a turn is already in flight (a pending context is queued), the steer
        folds into it — the ORIGINAL pending context answers the eventual
        ``TurnDone``, so we add none. If nothing is in flight, the steer starts a
        fresh turn whose reply must route back to chat, so we register a pending
        context (as a chat reply) for it.
        """
        text = envelope.payload.get("text")
        if not isinstance(text, str) or not text.strip():
            logger.warning("nats_ingress.steer | empty steer text | id={}", envelope.id)
            return
        async with self._lock:
            in_flight = bool(self._pending)
            if not in_flight:
                self._pending.append(
                    PendingTurn(
                        origin_kind="chat",
                        inbound_id=envelope.id,
                        task_id=envelope.task_id,
                        channel=envelope.channel or self._default_channel,
                    )
                )
        logger.info(
            "nats_ingress.steer | id={} len={} folded_into_active_turn={}",
            envelope.id,
            len(text),
            in_flight,
        )
        await self._supervisor.session.send(text)

    async def _handle_interrupt(self, envelope: Envelope) -> None:
        """Interrupt the active turn."""
        logger.info("nats_ingress.interrupt | id={}", envelope.id)
        await self._supervisor.session.interrupt()

    # --- outbound (event sink) ------------------------------------------------

    async def _on_engine_event(self, event: EngineEvent) -> None:
        """Route a turn's completion to a reply envelope. Deltas are ignored here."""
        if not isinstance(event, TurnDone):
            return
        async with self._lock:
            pending = self._pending.popleft() if self._pending else None
        if pending is None:
            # A turn with no tracked inbound context (e.g. an engine-initiated
            # message). Nothing to reply-correlate; log and drop.
            logger.warning("nats_ingress.reply | turn_done with no pending context; dropping")
            return
        await self._publish_reply(pending, event)

    async def _publish_reply(self, pending: PendingTurn, turn: TurnDone) -> None:
        """Build a typed ``TurnFrame`` for a completed turn and publish it.

        The frame is the truth: ``visible_text`` is the guard-sanitized whole
        turn, ``stance`` is the typed stance the engine filled. agentd emits NO
        done — that is orchestrator-derived from the frame graph.
        """
        if turn.is_error:
            # Failed turns render a human one-liner; the raw payload (often the
            # vendor's error body) is logged ONCE here, never shipped.
            logger.error(
                "nats_ingress.reply | failed turn; payload withheld from chat | "
                "inbound_id={} payload={}",
                pending.inbound_id,
                turn.text,
            )
            outbound_text = failure_visible_text(turn.text)
        else:
            outbound_text = turn.text
        # Buffer-complete-then-sanitize: the guard sees the WHOLE turn text.
        sanitized = sanitize_outbound_text(outbound_text, platform="nats")
        assert isinstance(sanitized, str)
        frame = TurnFrame(
            agent_id=self._agent,
            task_id=pending.task_id or pending.inbound_id,
            visible_text=sanitized or "(no output)",
            stance=turn.stance,
            refs=[pending.inbound_id],
        )
        await self._publish_turn_frame(frame, pending)

    async def _publish_turn_frame(self, frame: TurnFrame, pending: PendingTurn) -> None:
        """Route a completed turn's ``TurnFrame`` to its lane by origin kind.

        Chat replies go to chatd's direct lane; dispatched work goes to the task
        status lane. A work turn with no ``task_id`` has no lane and is dropped.
        """
        if pending.origin_kind == "chat":
            await self._emit_frame(
                frame, pending, subjects.direct(self._env, _CHATD_AGENT), "chat_reply", _CHATD_AGENT
            )
            return
        if not pending.task_id:
            logger.warning(
                "nats_ingress.reply | work turn without task_id; kind={} — no status lane",
                pending.origin_kind,
            )
            return
        await self._emit_frame(
            frame, pending, subjects.agent_status(self._env, pending.task_id), "turn", None
        )

    async def _emit_frame(
        self, frame: TurnFrame, pending: PendingTurn, subject: str, kind: str, to: str | None
    ) -> None:
        """Serialize a ``TurnFrame`` into the envelope payload and publish it."""
        env = Envelope.new(
            msg_id=_new_id(),
            ts=_now_iso(),
            sender=self._agent,
            kind=kind,
            to=to,
            task_id=pending.task_id,
            channel=pending.channel,
            correlation_id=pending.inbound_id,
            payload=frame.model_dump(mode="json"),
        )
        try:
            await self._publish(subject, env)
            logger.info(
                "nats_ingress.reply | {} published | subject={} corr={} len={}",
                kind,
                subject,
                pending.inbound_id,
                len(frame.visible_text),
            )
        except Exception as exc:
            logger.error("nats_ingress.reply | {} publish failed: {}", kind, exc)
