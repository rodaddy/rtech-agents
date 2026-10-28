"""Cover bridged-participant done derivation.

The orchestrator — never the bridge edge — owns terminal semantics for a
bridged participant: it consumes the typed fleet ``Done`` through the
``FleetSignalSource`` seam, records it as in-graph ``TerminalEvidence``, and
derives done in one place. Every fleet envelope a test delivers round-trips
through the REAL wire codec (``Envelope.to_bytes`` -> ``from_bytes``), so the
typed signal must genuinely survive serialization.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fleet_nats import Envelope
from fleet_nats.sentinel import done_envelope
from loguru import logger
from rtech_frames import Stance, StatusFrame, TaskFrame, TurnFrame
from rtech_frames.codec import Frame
from rtech_orchestrator import Orchestrator, OrchestratorConfig, subjects
from rtech_orchestrator.fleet_signals import evidence_from_envelope
from rtech_orchestrator.transport import FrameHandler, FrameTransportError

BRIDGED_ID = "birk"
NATIVE_ID = "agentd"
STACK_ID = "test-stack"


class _FakeSubscription:
    """Retirable handle a fake tracked-subscribe returns to ``subscribe_all``."""

    def __init__(
        self,
        handlers: dict[str, list[FrameHandler]],
        subject: str,
        handler: FrameHandler,
    ) -> None:
        self._handlers = handlers
        self._subject = subject
        self._handler = handler

    async def unsubscribe(self) -> None:
        live = self._handlers.get(self._subject, [])
        if self._handler in live:
            live.remove(self._handler)
        if not live:
            self._handlers.pop(self._subject, None)


class FakeFrameBus:
    """Record publishes and route deliveries (inline per tests-namespace rule)."""

    def __init__(self) -> None:
        """Initialize empty publish log and subscription table."""
        self.published: list[tuple[str, Frame]] = []
        self.handlers: dict[str, list[FrameHandler]] = {}

    async def publish_frame(self, subject: str, frame: Frame) -> None:
        """Record a published (subject, frame) pair."""
        self.published.append((subject, frame))

    async def confirm_publishes(self) -> None:
        """Model a successful server confirmation round-trip."""
        return None

    async def subscribe_frame(self, subject: str, handler: FrameHandler) -> None:
        """Register a handler for a subject."""
        self.handlers.setdefault(subject, []).append(handler)

    async def subscribe_frame_tracked(
        self, subject: str, handler: FrameHandler
    ) -> _FakeSubscription:
        """Register a handler and return its retirable subscription handle."""
        self.handlers.setdefault(subject, []).append(handler)
        return _FakeSubscription(self.handlers, subject, handler)

    async def deliver(self, subject: str, frame: Frame) -> None:
        """Deliver a frame to every handler registered for a subject.

        Raises:
            KeyError: If no handler is registered for the subject.
        """
        handlers = self.handlers.get(subject)
        if not handlers:
            logger.error("no subscription for subject", subject=subject)
            raise KeyError(f"no subscription registered for {subject}")
        for handler in list(handlers):
            await handler(frame)


class FakeFleetSignalSource:
    """Record-and-drive ``FleetSignalSource`` with wire-real envelope delivery."""

    def __init__(self) -> None:
        self.handlers: dict[str, Callable[[Envelope], Awaitable[None]]] = {}
        self.unsubscribed: list[str] = []
        self.fail_subscribe: bool = False

    async def subscribe_done(
        self, task_id: str, handler: Callable[[Envelope], Awaitable[None]]
    ) -> Callable[[], Awaitable[None]]:
        if self.fail_subscribe:
            raise ConnectionError("fleet subscribe failed (test)")
        self.handlers[task_id] = handler

        async def _unsubscribe() -> None:
            self.unsubscribed.append(task_id)
            self.handlers.pop(task_id, None)

        return _unsubscribe

    async def deliver(self, task_id: str, envelope: Envelope) -> None:
        """Deliver an envelope through the REAL wire codec to the task handler."""
        handler = self.handlers[task_id]
        await handler(Envelope.from_bytes(envelope.to_bytes()))


def _task(task_id: str = "t1", participants: list[str] | None = None) -> TaskFrame:
    return TaskFrame(
        id=task_id,
        origin_edge="web",
        requester="rico",
        text="ask the bridged agent",
        scope="single",
        participants=participants if participants is not None else [BRIDGED_ID],
        stack=STACK_ID,
    )


def _content_turn(task_id: str = "t1", agent_id: str = BRIDGED_ID) -> TurnFrame:
    # What the bridge publishes: content with passthrough stance — note the
    # EMPTY needs_from, which must never complete a bridged participant.
    return TurnFrame(
        agent_id=agent_id,
        task_id=task_id,
        visible_text="bridged answer",
        stance=Stance(position="responded", because="hermes-bridge", needs_from=[]),
    )


def _fleet_done(task_id: str = "t1", sender: str = BRIDGED_ID) -> Envelope:
    return done_envelope(
        msg_id="d1",
        ts="2026-07-12T00:00:02+00:00",
        sender=sender,
        task_id=task_id,
        outcome="completed",
    )


def _orchestrator(bus: FakeFrameBus, fleet: FakeFleetSignalSource | None) -> Orchestrator:
    return Orchestrator(
        bus=bus,
        config=OrchestratorConfig(
            single_agent_id=NATIVE_ID,
            stack_id=STACK_ID,
            bridged_agents=frozenset({BRIDGED_ID}),
        ),
        fleet_signals=fleet,
    )


async def _started(
    fleet: FakeFleetSignalSource | None = None,
) -> tuple[Orchestrator, FakeFrameBus, FakeFleetSignalSource | None]:
    bus = FakeFrameBus()
    orch = _orchestrator(bus, fleet)
    await orch.start()
    return orch, bus, fleet


def _results(bus: FakeFrameBus, task_id: str = "t1") -> list[Frame]:
    return [f for s, f in bus.published if s == subjects.task_result(task_id)]


async def test_bridged_content_turn_alone_never_derives_done() -> None:
    """The bridge's terminal-LOOKING content turn (empty needs_from) is inert."""
    fleet = FakeFleetSignalSource()
    _orch, bus, _ = await _started(fleet)
    await bus.deliver(subjects.TASK_INGRESS, _task())
    await bus.deliver(subjects.task_turn("t1"), _content_turn())
    assert _results(bus) == []


async def test_typed_fleet_done_derives_done_and_publishes_content_turn() -> None:
    """Orchestrator (not the bridge) derives done from the typed fleet Done."""
    fleet = FakeFleetSignalSource()
    _orch, bus, _ = await _started(fleet)
    await bus.deliver(subjects.TASK_INGRESS, _task())
    # Dispatch subscribed the task's fleet done lane.
    assert "t1" in fleet.handlers
    turn = _content_turn()
    await bus.deliver(subjects.task_turn("t1"), turn)
    await fleet.deliver("t1", _fleet_done())
    results = _results(bus)
    assert len(results) == 1
    # The result is the bridge's CONTENT turn re-published by the orchestrator,
    # never a newly authored terminal frame.
    assert results[0] == turn
    # Lifecycle: the fleet done lane is dropped once the task finished.
    assert fleet.unsubscribed == ["t1"]


async def test_done_before_content_turn_finishes_on_turn_arrival() -> None:
    """Subject-ordering race: evidence first, content later — still completes."""
    fleet = FakeFleetSignalSource()
    _orch, bus, _ = await _started(fleet)
    await bus.deliver(subjects.TASK_INGRESS, _task())
    await fleet.deliver("t1", _fleet_done())
    assert _results(bus) == []  # done derived but no result turn yet
    turn = _content_turn()
    await bus.deliver(subjects.task_turn("t1"), turn)
    assert _results(bus) == [turn]


async def test_spoofed_fleet_sender_never_terminates() -> None:
    fleet = FakeFleetSignalSource()
    _orch, bus, _ = await _started(fleet)
    await bus.deliver(subjects.TASK_INGRESS, _task())
    await bus.deliver(subjects.task_turn("t1"), _content_turn())
    await fleet.deliver("t1", _fleet_done(sender="mallory"))
    assert _results(bus) == []
    # The genuine signal still completes afterwards.
    await fleet.deliver("t1", _fleet_done())
    assert len(_results(bus)) == 1


async def test_done_without_typed_signal_fails_closed() -> None:
    fleet = FakeFleetSignalSource()
    _orch, bus, _ = await _started(fleet)
    await bus.deliver(subjects.TASK_INGRESS, _task())
    await bus.deliver(subjects.task_turn("t1"), _content_turn())
    forged = Envelope.new(
        msg_id="d1",
        ts="2026-07-12T00:00:02+00:00",
        sender=BRIDGED_ID,
        kind="done",
        task_id="t1",
        payload={"outcome": "completed"},
    )
    await fleet.deliver("t1", forged)
    assert _results(bus) == []


async def test_bridged_dispatch_without_signal_source_fails_closed() -> None:
    """No seam to observe terminal evidence -> loud dispatch failure, never done."""
    _orch, bus, _ = await _started(fleet=None)
    await bus.deliver(subjects.TASK_INGRESS, _task())
    await bus.deliver(subjects.task_turn("t1"), _content_turn())
    assert _results(bus) == []


async def test_fleet_subscribe_failure_fails_closed() -> None:
    fleet = FakeFleetSignalSource()
    fleet.fail_subscribe = True
    _orch, bus, _ = await _started(fleet)
    await bus.deliver(subjects.TASK_INGRESS, _task())
    await bus.deliver(subjects.task_turn("t1"), _content_turn())
    assert _results(bus) == []


async def test_native_participant_unaffected_by_bridged_config() -> None:
    """A task dispatched to the native agent completes on turn terminality alone."""
    fleet = FakeFleetSignalSource()
    _orch, bus, _ = await _started(fleet)
    task = _task(participants=[])  # routes to the configured native default
    await bus.deliver(subjects.TASK_INGRESS, task)
    assert fleet.handlers == {}  # no fleet watch for a native-only task
    turn = _content_turn(agent_id=NATIVE_ID)
    await bus.deliver(subjects.task_turn("t1"), turn)
    assert _results(bus) == [turn]


# --- evidence_from_envelope (pure, wire-real) ---------------------------------


def _wire(envelope: Envelope) -> Envelope:
    return Envelope.from_bytes(envelope.to_bytes())


def test_evidence_from_valid_done() -> None:
    evidence = evidence_from_envelope(
        envelope=_wire(_fleet_done()), task_id="t1", bridged=frozenset({BRIDGED_ID})
    )
    assert evidence is not None
    assert evidence.agent_id == BRIDGED_ID
    assert evidence.task_id == "t1"
    assert evidence.source == "fleet-done"
    assert evidence.outcome == "completed"


def test_evidence_rejects_non_done_kind() -> None:
    status = _wire(
        Envelope.new(
            msg_id="r1",
            ts="2026-07-12T00:00:01+00:00",
            sender=BRIDGED_ID,
            kind="status",
            task_id="t1",
            payload={"text": "working on it"},
        )
    )
    assert (
        evidence_from_envelope(envelope=status, task_id="t1", bridged=frozenset({BRIDGED_ID}))
        is None
    )


def test_evidence_rejects_unbridged_sender() -> None:
    assert (
        evidence_from_envelope(
            envelope=_wire(_fleet_done(sender="mallory")),
            task_id="t1",
            bridged=frozenset({BRIDGED_ID}),
        )
        is None
    )


def test_evidence_rejects_task_mismatch() -> None:
    assert (
        evidence_from_envelope(
            envelope=_wire(_fleet_done(task_id="t9")),
            task_id="t1",
            bridged=frozenset({BRIDGED_ID}),
        )
        is None
    )


def test_evidence_rejects_typed_task_forgery() -> None:
    # Envelope.task_id forged to match; the typed signal's own task_id disagrees.
    forged = _wire(_fleet_done(task_id="t9")).model_copy(update={"task_id": "t1"})
    assert (
        evidence_from_envelope(envelope=forged, task_id="t1", bridged=frozenset({BRIDGED_ID}))
        is None
    )


# --- round-3 races: subscribe-before-publish + stale paired status ------------


class FastDoneBus(FakeFrameBus):
    """Deliver the gateway's one-shot typed Done INSIDE the dispatch publish.

    Models the round-3 P1 race: a gateway so fast its typed ``Done`` hits the
    fleet lane before the orchestrator's routing coroutine resumes. The Done is
    lost forever unless the fleet subscription already exists at publish time.
    """

    def __init__(self, fleet: FakeFleetSignalSource, dispatch_subject: str) -> None:
        super().__init__()
        self._fleet = fleet
        self._dispatch_subject = dispatch_subject
        self.done_delivered: bool = False

    async def publish_frame(self, subject: str, frame: Frame) -> None:
        await super().publish_frame(subject, frame)
        if subject != self._dispatch_subject or self.done_delivered:
            return
        # The fleet watch MUST already be installed, or the one-shot signal
        # has no handler and the task can never accrue terminal evidence.
        assert "t1" in self._fleet.handlers, "fleet done subscription installed after publish"
        self.done_delivered = True
        await self._fleet.deliver("t1", _fleet_done())


async def test_fast_done_during_single_dispatch_is_not_missed() -> None:
    fleet = FakeFleetSignalSource()
    bus = FastDoneBus(fleet, subjects.agent_task(BRIDGED_ID))
    orch = _orchestrator(bus, fleet)
    await orch.start()
    await bus.deliver(subjects.TASK_INGRESS, _task())
    assert bus.done_delivered
    # The evidence landed before the dispatch record; the content turn arrival
    # re-derives done and publishes the result.
    turn = _content_turn()
    await bus.deliver(subjects.task_turn("t1"), turn)
    assert _results(bus) == [turn]


async def test_fast_done_during_multi_fanout_is_not_missed() -> None:
    # Multi routing must install the fleet watch before the FIRST publish:
    # the bridged participant is fanned out first here and completes inside
    # its own dispatch publish.
    fleet = FakeFleetSignalSource()
    bus = FastDoneBus(fleet, subjects.agent_task(BRIDGED_ID))
    orch = _orchestrator(bus, fleet)
    await orch.start()
    task = TaskFrame(
        id="t1",
        origin_edge="web",
        requester="rico",
        text="ask both",
        scope="duo",
        participants=[BRIDGED_ID, NATIVE_ID],
        stack=STACK_ID,
    )
    await bus.deliver(subjects.TASK_INGRESS, task)
    assert bus.done_delivered
    await bus.deliver(subjects.task_turn("t1"), _content_turn())
    native_turn = TurnFrame(
        agent_id=NATIVE_ID,
        task_id="t1",
        visible_text="native answer",
        stance=Stance(position="agree", because="done", needs_from=[]),
    )
    await bus.deliver(subjects.task_turn("t1"), native_turn)
    assert len(_results(bus)) == 1


class FailingDispatchBus(FakeFrameBus):
    """Raise on the agent-task dispatch publish to exercise watch rollback."""

    def __init__(self, dispatch_subject: str) -> None:
        super().__init__()
        self._dispatch_subject = dispatch_subject

    async def publish_frame(self, subject: str, frame: Frame) -> None:
        if subject == self._dispatch_subject:
            raise FrameTransportError("dispatch publish failed (test)")
        await super().publish_frame(subject, frame)


async def test_dispatch_publish_failure_rolls_back_fleet_watch() -> None:
    fleet = FakeFleetSignalSource()
    bus = FailingDispatchBus(subjects.agent_task(BRIDGED_ID))
    orch = _orchestrator(bus, fleet)
    await orch.start()
    await bus.deliver(subjects.TASK_INGRESS, _task())
    # The pre-publish watch was installed, then rolled back on the failure.
    assert fleet.unsubscribed == ["t1"]
    assert fleet.handlers == {}
    # The task failed closed: a later content turn never produces a result.
    await bus.deliver(subjects.task_turn("t1"), _content_turn())
    assert _results(bus) == []


async def test_late_paired_working_status_does_not_contradict_result() -> None:
    # Round-3 P1 #2: evidence first, content turn next (result published and
    # latched), the bridge's paired working status last. The stale echo must
    # neither retract the published result nor flip the derivation.
    fleet = FakeFleetSignalSource()
    _orch, bus, _ = await _started(fleet)
    await bus.deliver(subjects.TASK_INGRESS, _task())
    await fleet.deliver("t1", _fleet_done())
    turn = _content_turn()
    await bus.deliver(subjects.task_turn("t1"), turn)
    assert _results(bus) == [turn]
    await bus.deliver(
        subjects.task_status("t1"),
        StatusFrame(task_id="t1", agent_id=BRIDGED_ID, state="working"),
    )
    assert _results(bus) == [turn]


async def test_evidence_then_working_then_turn_still_completes() -> None:
    # Worst ordering: the typed Done outruns BOTH halves of the bridge's
    # turn/working pair. The stale working echo must not deadlock the task.
    fleet = FakeFleetSignalSource()
    _orch, bus, _ = await _started(fleet)
    await bus.deliver(subjects.TASK_INGRESS, _task())
    await fleet.deliver("t1", _fleet_done())
    await bus.deliver(
        subjects.task_status("t1"),
        StatusFrame(task_id="t1", agent_id=BRIDGED_ID, state="working"),
    )
    turn = _content_turn()
    await bus.deliver(subjects.task_turn("t1"), turn)
    assert _results(bus) == [turn]
