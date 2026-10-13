"""Cover routing, accumulation, derived-done publish, idempotency, and scoping."""

from __future__ import annotations

from loguru import logger
from rtech_frames import Stance, StatusFrame, TaskFrame, TurnFrame
from rtech_frames.codec import Frame, decode_frame, encode_frame
from rtech_orchestrator import Orchestrator, OrchestratorConfig, subjects
from rtech_orchestrator.graph import InMemoryGraphStore
from rtech_orchestrator.transport import (
    FrameConfirmError,
    FrameHandler,
    FramePublishError,
    FrameSubscribeError,
)

AGENT_ID = "agentd"
STACK_ID = "test-stack"


class _FakeSubscription:
    """Retirable handle a fake tracked-subscribe hands back to ``subscribe_all``.

    ``unsubscribe`` removes this exact handler from the subject's handler list,
    mirroring a live NATS unsubscribe — so a rolled-back or retired lane stops
    delivering.
    """

    def __init__(
        self,
        handlers: dict[str, list[FrameHandler]],
        subject: str,
        handler: FrameHandler,
    ) -> None:
        self._handlers = handlers
        self._subject = subject
        self._handler = handler
        self.unsubscribed = 0

    async def unsubscribe(self) -> None:
        self.unsubscribed += 1
        live = self._handlers.get(self._subject, [])
        if self._handler in live:
            live.remove(self._handler)
        if not live:
            self._handlers.pop(self._subject, None)


class FakeFrameBus:
    """Record publishes and route deliveries to registered handlers.

    Defined inline (not in a shared ``fakes`` module) because every package's
    ``tests`` package shares one module namespace under pytest, so a
    ``tests.fakes`` import resolves to whichever package loads first. Subjects
    fan out to EVERY registered handler (a list per subject), so two
    orchestrators subscribed to the same ingress model the shared live bus
    (issue #75 cross-stack shape).
    """

    def __init__(self) -> None:
        """Initialize empty publish log and subscription table."""
        self.published: list[tuple[str, Frame]] = []
        self.handlers: dict[str, list[FrameHandler]] = {}
        self.confirms: int = 0

    async def publish_frame(self, subject: str, frame: Frame) -> None:
        """Record a published (subject, frame) pair."""
        self.published.append((subject, frame))

    async def confirm_publishes(self) -> None:
        """Count confirmation round-trips; the fake's publishes always land."""
        self.confirms += 1

    async def subscribe_frame(self, subject: str, handler: FrameHandler) -> None:
        """Register a handler for a subject (many handlers per subject)."""
        self.handlers.setdefault(subject, []).append(handler)

    async def subscribe_frame_tracked(
        self, subject: str, handler: FrameHandler
    ) -> _FakeSubscription:
        """Register a handler and return its retirable subscription handle."""
        self.handlers.setdefault(subject, []).append(handler)
        return _FakeSubscription(self.handlers, subject, handler)

    async def deliver(self, subject: str, frame: Frame) -> None:
        """Deliver a frame to every handler registered for a subject.

        Args:
            subject: The subject whose handlers receive the frame.
            frame: The frame to deliver.

        Raises:
            KeyError: If no handler is registered for the subject.
        """
        handlers = self.handlers.get(subject)
        if not handlers:
            logger.error("no subscription for subject", subject=subject)
            raise KeyError(f"no subscription registered for {subject}")
        for handler in list(handlers):
            await handler(frame)


def _task(
    scope: str = "single",
    participants: list[str] | None = None,
    stack: str = STACK_ID,
) -> TaskFrame:
    """Build a task frame for a given scope, optional participants, and stack."""
    return TaskFrame(
        id="t1",
        origin_edge="web",
        requester="rico",
        text="do the thing",
        scope=scope,  # type: ignore[arg-type]
        participants=participants or [],
        stack=stack,
    )


def _turn(needs: list[str] | None = None, agent_id: str = AGENT_ID) -> TurnFrame:
    """Build a turn frame; ``needs`` populates the outstanding cross-agent need."""
    return TurnFrame(
        agent_id=agent_id,
        task_id="t1",
        visible_text="here is my take",
        stance=Stance(position="p", because="b", needs_from=needs or []),
    )


def _orchestrator(bus: FakeFrameBus, stack_id: str = STACK_ID) -> Orchestrator:
    """Build an orchestrator wired to the fake bus and default single agent."""
    return Orchestrator(
        bus=bus, config=OrchestratorConfig(single_agent_id=AGENT_ID, stack_id=stack_id)
    )


async def test_single_task_routes_to_agent() -> None:
    """(1) A single-scope task is published verbatim to the agent lane."""
    bus = FakeFrameBus()
    orch = _orchestrator(bus)
    await orch.start()
    task = _task()
    await bus.deliver(subjects.TASK_INGRESS, task)
    assert (subjects.agent_task(AGENT_ID), task) in bus.published


async def test_single_task_routes_to_configured_id_not_a_literal() -> None:
    """(1b) Routing follows the CONFIGURED id, never a hardcoded ``agentd``."""
    bus = FakeFrameBus()
    orch = Orchestrator(
        bus=bus, config=OrchestratorConfig(single_agent_id="agentd-bilby", stack_id=STACK_ID)
    )
    await orch.start()
    task = _task()
    await bus.deliver(subjects.TASK_INGRESS, task)
    assert (subjects.agent_task("agentd-bilby"), task) in bus.published
    assert (subjects.agent_task("agentd"), task) not in bus.published


async def test_terminal_turn_publishes_result() -> None:
    """(2) A terminal turn drives derived done and a result publish."""
    bus = FakeFrameBus()
    orch = _orchestrator(bus)
    await orch.start()
    task = _task()
    await bus.deliver(subjects.TASK_INGRESS, task)
    turn = _turn()
    await bus.deliver(subjects.task_turn(task.id), turn)
    assert (subjects.task_result(task.id), turn) in bus.published


async def test_status_after_finish_does_not_republish() -> None:
    """(3) A late status after finish must not publish a second result."""
    bus = FakeFrameBus()
    orch = _orchestrator(bus)
    await orch.start()
    task = _task()
    await bus.deliver(subjects.TASK_INGRESS, task)
    await bus.deliver(subjects.task_turn(task.id), _turn())
    await bus.deliver(
        subjects.task_status(task.id),
        StatusFrame(task_id="t1", agent_id=AGENT_ID, state="working"),
    )
    result_subject = subjects.task_result(task.id)
    result_publishes = [p for p in bus.published if p[0] == result_subject]
    assert len(result_publishes) == 1


async def test_non_terminal_turn_publishes_no_result() -> None:
    """(4) A turn with an outstanding need must not publish a result."""
    bus = FakeFrameBus()
    orch = _orchestrator(bus)
    await orch.start()
    task = _task()
    await bus.deliver(subjects.TASK_INGRESS, task)
    await bus.deliver(subjects.task_turn(task.id), _turn(needs=["peer"]))
    result_subject = subjects.task_result(task.id)
    assert not any(p[0] == result_subject for p in bus.published)


async def test_single_task_with_participant_routes_to_named_agent() -> None:
    """(5) A single-scope task with a participant routes to it, not the default."""
    bus = FakeFrameBus()
    orch = _orchestrator(bus)
    await orch.start()
    task = _task(participants=["opus-b"])
    await bus.deliver(subjects.TASK_INGRESS, task)
    assert (subjects.agent_task("opus-b"), task) in bus.published
    assert (subjects.agent_task(AGENT_ID), task) not in bus.published


async def test_duo_task_fans_out_to_both_participants() -> None:
    """(6) A duo task publishes the TaskFrame to each participant's agent lane."""
    bus = FakeFrameBus()
    orch = _orchestrator(bus)
    await orch.start()
    task = _task(scope="duo", participants=["codex-a", "opus-b"])
    await bus.deliver(subjects.TASK_INGRESS, task)
    assert (subjects.agent_task("codex-a"), task) in bus.published
    assert (subjects.agent_task("opus-b"), task) in bus.published


async def test_duo_fan_out_two_turns_derive_done_one_result() -> None:
    """(7) Two fake agents each emit one terminal turn -> derived done -> one result."""
    bus = FakeFrameBus()
    orch = _orchestrator(bus)
    await orch.start()
    task = _task(scope="duo", participants=["codex-a", "opus-b"])
    await bus.deliver(subjects.TASK_INGRESS, task)

    # First terminal turn: duo not yet complete (only one distinct agent).
    await bus.deliver(subjects.task_turn(task.id), _turn(agent_id="codex-a"))
    result_subject = subjects.task_result(task.id)
    assert not any(p[0] == result_subject for p in bus.published)

    # Second distinct agent's terminal turn completes the duo -> one result.
    second = _turn(agent_id="opus-b")
    await bus.deliver(subjects.task_turn(task.id), second)
    result_publishes = [p for p in bus.published if p[0] == result_subject]
    assert result_publishes == [(result_subject, second)]


# --- issue #75: ingress stack scoping + participant-scoped done ----------------


async def test_foreign_stack_task_is_not_consumed() -> None:
    """(8) A task stamped for another stack is ignored: no dispatch, no lanes."""
    bus = FakeFrameBus()
    orch = _orchestrator(bus)
    await orch.start()
    task = _task(stack="other-stack")
    await bus.deliver(subjects.TASK_INGRESS, task)
    assert bus.published == []
    assert subjects.task_turn(task.id) not in bus.handlers


async def test_unscoped_legacy_task_fails_closed() -> None:
    """(9) A legacy task with an empty stack field is not consumed."""
    bus = FakeFrameBus()
    orch = _orchestrator(bus)
    await orch.start()
    task = _task(stack="")
    await bus.deliver(subjects.TASK_INGRESS, task)
    assert bus.published == []


async def test_shared_ingress_only_matching_stack_dispatches() -> None:
    """(10) Two orchestrators on one bus: only the matching stack consumes.

    This is the defect B shared-ingress shape: dev + deployed orchestrators
    both subscribed to ``rtech.task.new`` and both dispatched every task.
    """
    bus = FakeFrameBus()
    dev = Orchestrator(
        bus=bus, config=OrchestratorConfig(single_agent_id="agentd-dev", stack_id="dev-stack")
    )
    deployed = Orchestrator(
        bus=bus, config=OrchestratorConfig(single_agent_id="agentd-bilby", stack_id="prod-stack")
    )
    await dev.start()
    await deployed.start()

    task = _task(stack="dev-stack")
    await bus.deliver(subjects.TASK_INGRESS, task)

    assert (subjects.agent_task("agentd-dev"), task) in bus.published
    assert (subjects.agent_task("agentd-bilby"), task) not in bus.published


async def test_foreign_terminal_turn_does_not_derive_done() -> None:
    """(11) Defect B repro: a foreign agent's terminal turn never finishes the task.

    The orchestrator dispatched to its configured agent; a terminal turn from a
    foreign stack's agent (``agentd-bilby``) lands on the task's turn lane
    first. Done must WAIT for the dispatched participant, and the result must
    carry the participant's turn, exactly once.
    """
    bus = FakeFrameBus()
    orch = _orchestrator(bus)
    await orch.start()
    task = _task()
    await bus.deliver(subjects.TASK_INGRESS, task)

    result_subject = subjects.task_result(task.id)

    # Foreign terminal turn arrives BEFORE the dispatched agent's turn.
    foreign = _turn(agent_id="agentd-bilby")
    await bus.deliver(subjects.task_turn(task.id), foreign)
    assert not any(p[0] == result_subject for p in bus.published)

    # The dispatched participant's terminal turn derives done.
    own = _turn(agent_id=AGENT_ID)
    await bus.deliver(subjects.task_turn(task.id), own)
    result_publishes = [p for p in bus.published if p[0] == result_subject]
    assert result_publishes == [(result_subject, own)]


async def test_duplicate_foreign_turns_never_publish_result() -> None:
    """(12) Repeated foreign terminal turns never satisfy the participant set."""
    bus = FakeFrameBus()
    orch = _orchestrator(bus)
    await orch.start()
    task = _task()
    await bus.deliver(subjects.TASK_INGRESS, task)
    for _ in range(3):
        await bus.deliver(subjects.task_turn(task.id), _turn(agent_id="agentd-bilby"))
    result_subject = subjects.task_result(task.id)
    assert not any(p[0] == result_subject for p in bus.published)


# --- round 2: byte-backed wire, dispatch confirmation, ingress idempotency -----


class ByteFakeFrameBus:
    """Fake bus that carries RAW BYTES through the real codec on every hop.

    ``publish_frame`` encodes via ``encode_frame`` and delivers the bytes to
    every handler registered for the subject, each of which decodes its OWN
    copy via ``decode_frame`` — so subject/payload binding, codec round-trips,
    and two-consumer fan-out all traverse the real wire path (issue #75
    round 2). ``deliver_bytes`` injects foreign raw bytes exactly as a live
    NATS delivery would.
    """

    def __init__(self) -> None:
        """Initialize empty byte publish log and subscription table."""
        self.published: list[tuple[str, bytes]] = []
        self.handlers: dict[str, list[FrameHandler]] = {}

    async def publish_frame(self, subject: str, frame: Frame) -> None:
        """Encode the frame and deliver its bytes to the subject's handlers."""
        data = encode_frame(frame)
        self.published.append((subject, data))
        await self.deliver_bytes(subject, data)

    async def confirm_publishes(self) -> None:
        """Model an in-fake server round-trip that always succeeds."""
        return None

    async def subscribe_frame(self, subject: str, handler: FrameHandler) -> None:
        """Register a typed handler for a subject (many handlers per subject)."""
        self.handlers.setdefault(subject, []).append(handler)

    async def subscribe_frame_tracked(
        self, subject: str, handler: FrameHandler
    ) -> _FakeSubscription:
        """Register a typed handler and return its retirable subscription handle."""
        self.handlers.setdefault(subject, []).append(handler)
        return _FakeSubscription(self.handlers, subject, handler)

    async def deliver_bytes(self, subject: str, data: bytes) -> None:
        """Decode bytes per registered handler and invoke it, like a live sub."""
        for handler in list(self.handlers.get(subject, [])):
            await handler(decode_frame(data))

    def published_frames(self, subject: str) -> list[Frame]:
        """Return decoded frames published on a subject, in publish order."""
        return [decode_frame(data) for subj, data in self.published if subj == subject]


async def test_defect_b_repro_over_encoded_wire() -> None:
    """(13) Defect B over the real codec path: two orchestrators, raw bytes.

    Dev + deployed orchestrators share one byte bus subscribed to the same
    ingress. Only the matching stack dispatches; a foreign agent's terminal
    turn (raw bytes, arriving first) never derives done; exactly one result
    publishes and it carries the dispatched participant's turn.
    """
    bus = ByteFakeFrameBus()
    dev = Orchestrator(
        bus=bus, config=OrchestratorConfig(single_agent_id="agentd-dev", stack_id="dev-stack")
    )
    deployed = Orchestrator(
        bus=bus, config=OrchestratorConfig(single_agent_id="agentd-bilby", stack_id="prod-stack")
    )
    await dev.start()
    await deployed.start()

    task = _task(stack="dev-stack")
    await bus.deliver_bytes(subjects.TASK_INGRESS, encode_frame(task))

    # Only the matching stack dispatched, over encoded bytes.
    assert bus.published_frames(subjects.agent_task("agentd-dev")) == [task]
    assert bus.published_frames(subjects.agent_task("agentd-bilby")) == []

    result_subject = subjects.task_result(task.id)

    # Foreign terminal turn bytes land on the turn lane FIRST.
    foreign = _turn(agent_id="agentd-bilby")
    await bus.deliver_bytes(subjects.task_turn(task.id), encode_frame(foreign))
    assert bus.published_frames(result_subject) == []

    # The dispatched participant's terminal turn derives done -> one result.
    own = _turn(agent_id="agentd-dev")
    await bus.deliver_bytes(subjects.task_turn(task.id), encode_frame(own))
    assert bus.published_frames(result_subject) == [own]

    # Duplicate foreign bytes after completion never re-publish.
    await bus.deliver_bytes(subjects.task_turn(task.id), encode_frame(foreign))
    assert bus.published_frames(result_subject) == [own]


async def test_lane_rejects_mismatched_task_id_over_encoded_wire() -> None:
    """(14) A frame on task A's subject carrying task B's id never accumulates."""
    bus = ByteFakeFrameBus()
    orch = _orchestrator(bus)
    await orch.start()
    task = _task()
    await bus.deliver_bytes(subjects.TASK_INGRESS, encode_frame(task))

    # Terminal turn for a DIFFERENT task id delivered on t1's turn lane.
    alien_turn = TurnFrame(
        agent_id=AGENT_ID,
        task_id="t2",
        visible_text="wrong lane",
        stance=Stance(position="p", because="b"),
    )
    await bus.deliver_bytes(subjects.task_turn(task.id), encode_frame(alien_turn))
    assert bus.published_frames(subjects.task_result(task.id)) == []

    # Status with a mismatched id is likewise rejected, then the real turn wins.
    alien_status = StatusFrame(task_id="t2", agent_id="alien", state="blocked")
    await bus.deliver_bytes(subjects.task_status(task.id), encode_frame(alien_status))
    own = _turn(agent_id=AGENT_ID)
    await bus.deliver_bytes(subjects.task_turn(task.id), encode_frame(own))
    assert bus.published_frames(subjects.task_result(task.id)) == [own]


async def test_redelivered_task_is_idempotent() -> None:
    """(15) Redelivery of the same task frame never re-subscribes or re-dispatches."""
    bus = FakeFrameBus()
    orch = _orchestrator(bus)
    await orch.start()
    task = _task()
    await bus.deliver(subjects.TASK_INGRESS, task)
    await bus.deliver(subjects.TASK_INGRESS, task)

    dispatches = [p for p in bus.published if p[0] == subjects.agent_task(AGENT_ID)]
    assert len(dispatches) == 1
    assert len(bus.handlers[subjects.task_turn(task.id)]) == 1
    assert len(bus.handlers[subjects.task_status(task.id)]) == 1


async def test_same_id_different_payload_is_rejected() -> None:
    """(16) A task id reused with a different payload is rejected, not merged."""
    bus = FakeFrameBus()
    orch = _orchestrator(bus)
    await orch.start()
    original = _task()
    await bus.deliver(subjects.TASK_INGRESS, original)
    mutated = TaskFrame(
        id=original.id,
        origin_edge="web",
        requester="mallory",
        text="do something else",
        scope="single",
        stack=STACK_ID,
    )
    await bus.deliver(subjects.TASK_INGRESS, mutated)

    dispatches = [p for p in bus.published if p[0] == subjects.agent_task(AGENT_ID)]
    assert dispatches == [(subjects.agent_task(AGENT_ID), original)]
    assert len(bus.handlers[subjects.task_turn(original.id)]) == 1


class FailingPublishBus(FakeFrameBus):
    """Fake bus whose publishes fail (as typed errors) on selected subjects."""

    def __init__(self, fail_subjects: set[str]) -> None:
        """Record the subjects whose publishes raise ``FramePublishError``."""
        super().__init__()
        self.fail_subjects = fail_subjects

    async def publish_frame(self, subject: str, frame: Frame) -> None:
        """Raise a typed publish error for failing subjects, else record."""
        if subject in self.fail_subjects:
            raise FramePublishError(f"publish to {subject} failed: injected")
        await super().publish_frame(subject, frame)


async def test_failed_single_dispatch_records_no_participant() -> None:
    """(17) A failed dispatch publish never marks the agent as dispatched.

    Without a recorded participant the later terminal turn from the intended
    agent must not derive done — the task failed closed, loudly, instead of
    completing on a dispatch that never happened.
    """
    bus = FailingPublishBus({subjects.agent_task(AGENT_ID)})
    orch = _orchestrator(bus)
    await orch.start()
    task = _task()
    await bus.deliver(subjects.TASK_INGRESS, task)

    await bus.deliver(subjects.task_turn(task.id), _turn(agent_id=AGENT_ID))
    assert not any(p[0] == subjects.task_result(task.id) for p in bus.published)


async def test_partial_duo_fanout_fails_closed() -> None:
    """(18) A partial multi-lane fan-out failure marks the task dispatch-failed.

    The first participant's publish succeeds, the second fails: even if both
    agents somehow emit terminal turns, done is never derived and no result
    publishes — the explicit partial-failure state wins.
    """
    bus = FailingPublishBus({subjects.agent_task("opus-b")})
    orch = _orchestrator(bus)
    await orch.start()
    task = _task(scope="duo", participants=["codex-a", "opus-b"])
    await bus.deliver(subjects.TASK_INGRESS, task)

    assert (subjects.agent_task("codex-a"), task) in bus.published
    assert not any(p[0] == subjects.agent_task("opus-b") for p in bus.published)

    await bus.deliver(subjects.task_turn(task.id), _turn(agent_id="codex-a"))
    await bus.deliver(subjects.task_turn(task.id), _turn(agent_id="opus-b"))
    assert not any(p[0] == subjects.task_result(task.id) for p in bus.published)


async def test_failed_lane_subscription_fails_closed() -> None:
    """(19) A failed lane subscription aborts dispatch entirely.

    The turn lane subscribes first; its failure means nothing was established,
    so there is nothing to roll back — the task simply fails closed.
    """

    class FailingSubscribeBus(FakeFrameBus):
        async def subscribe_frame_tracked(
            self, subject: str, handler: FrameHandler
        ) -> _FakeSubscription:
            if subject == subjects.task_turn("t1"):
                raise FrameSubscribeError(f"subscribe to {subject} failed: injected")
            return await super().subscribe_frame_tracked(subject, handler)

    bus = FailingSubscribeBus()
    orch = _orchestrator(bus)
    await orch.start()
    await bus.deliver(subjects.TASK_INGRESS, _task())

    assert not any(p[0] == subjects.agent_task(AGENT_ID) for p in bus.published)


async def test_partial_lane_subscription_rolls_back_atomically() -> None:
    """(#80) A status-lane failure rolls the already-live turn lane back.

    The turn lane subscribes first and succeeds; the status lane then fails.
    ``subscribe_all`` must retire the turn lane (no live-but-untracked handle),
    the task fails closed (no dispatch), and no lane survives to deliver.
    """

    class PartialSubscribeBus(FakeFrameBus):
        def __init__(self) -> None:
            super().__init__()
            self.turn_sub: _FakeSubscription | None = None

        async def subscribe_frame_tracked(
            self, subject: str, handler: FrameHandler
        ) -> _FakeSubscription:
            if subject == subjects.task_status("t1"):
                raise FrameSubscribeError(f"subscribe to {subject} failed: injected")
            sub = await super().subscribe_frame_tracked(subject, handler)
            if subject == subjects.task_turn("t1"):
                self.turn_sub = sub
            return sub

    bus = PartialSubscribeBus()
    orch = _orchestrator(bus)
    await orch.start()
    await bus.deliver(subjects.TASK_INGRESS, _task())

    # Fails closed: nothing dispatched to the agent lane.
    assert not any(p[0] == subjects.agent_task(AGENT_ID) for p in bus.published)
    # Atomic rollback: the earlier turn lane was retired, so no handler is left
    # live-but-untracked on the turn subject.
    assert bus.turn_sub is not None
    assert bus.turn_sub.unsubscribed == 1
    assert subjects.task_turn("t1") not in bus.handlers


async def test_rollback_leak_is_held_not_discarded() -> None:
    """(#80) A rollback whose unsubscribe fails holds the live handle, not drops it.

    Turn subscribes; status fails; but the turn lane's ``unsubscribe`` ALSO
    fails during rollback. The still-live handle must ride on
    ``FrameSubscribeError.leaked`` and be stashed for retry — never discarded.
    """

    class LeakingUnsubSubscription(_FakeSubscription):
        async def unsubscribe(self) -> None:
            raise RuntimeError("unsubscribe failed during rollback")

    class LeakingSubscribeBus(FakeFrameBus):
        async def subscribe_frame_tracked(
            self, subject: str, handler: FrameHandler
        ) -> _FakeSubscription:
            if subject == subjects.task_status("t1"):
                raise FrameSubscribeError(f"subscribe to {subject} failed: injected")
            self.handlers.setdefault(subject, []).append(handler)
            return LeakingUnsubSubscription(self.handlers, subject, handler)

    bus = LeakingSubscribeBus()
    orch = _orchestrator(bus)
    await orch.start()
    await bus.deliver(subjects.TASK_INGRESS, _task())

    # Fails closed, and the un-retirable turn lane is held for retry, never
    # silently forgotten.
    assert not any(p[0] == subjects.agent_task(AGENT_ID) for p in bus.published)
    leaked = orch._leaked_subscriptions.get("t1")
    assert leaked is not None
    assert subjects.task_turn("t1") in leaked.subjects


# --- round 3: dispatch-confirmation contract (publish -> confirm -> record) ----


class BufferingConfirmBus(FakeFrameBus):
    """Fake bus modeling core-NATS enqueue-only publish + explicit confirm.

    ``publish_frame`` always "succeeds" but only BUFFERS the frame (the
    reconnect-buffering shape from round-3 H1). ``confirm_publishes`` either
    raises ``FrameConfirmError`` (flush timeout) or flushes the buffer into
    ``published`` — so a frame is visible on the bus only after a confirmed
    hand-off, exactly like the live wire.
    """

    def __init__(self, *, confirm_ok: bool) -> None:
        """Initialize with the configured confirmation outcome."""
        super().__init__()
        self.confirm_ok = confirm_ok
        self.buffered: list[tuple[str, Frame]] = []

    async def publish_frame(self, subject: str, frame: Frame) -> None:
        """Buffer the frame client-side; enqueue is NOT a hand-off."""
        self.buffered.append((subject, frame))

    async def confirm_publishes(self) -> None:
        """Flush buffered frames on success; raise the typed error on timeout."""
        self.confirms += 1
        if not self.confirm_ok:
            raise FrameConfirmError("publish confirmation flush failed: injected timeout")
        self.published.extend(self.buffered)
        self.buffered.clear()


async def test_flush_timeout_records_no_participant_and_fails_closed() -> None:
    """(20) Round-3 H1: an enqueued-but-unconfirmed publish is not a dispatch.

    The single-lane publish succeeds (buffered client-side), the confirmation
    flush times out: no participant may be recorded, the task latches the
    typed partial-failure state, and the intended agent's later terminal turn
    (the buffer could still drain) never derives done.
    """
    bus = BufferingConfirmBus(confirm_ok=False)
    store = InMemoryGraphStore()
    orch = Orchestrator(
        bus=bus,
        config=OrchestratorConfig(single_agent_id=AGENT_ID, stack_id=STACK_ID),
        store=store,
    )
    await orch.start()
    task = _task()
    await bus.deliver(subjects.TASK_INGRESS, task)

    graph = store.get(task.id)
    assert graph is not None
    assert graph.participants == frozenset()
    assert graph.dispatch_failed is True

    await bus.deliver(subjects.task_turn(task.id), _turn(agent_id=AGENT_ID))
    assert not any(p[0] == subjects.task_result(task.id) for p in bus.published)


async def test_buffered_publish_not_counted_until_flushed() -> None:
    """(21) A reconnect-buffered publish counts only after the flush round-trip.

    At confirmation time the participant set must still be EMPTY (record
    happens strictly after confirm); after a successful confirm the dispatch
    record lands and the frame is on the wire.
    """
    bus = BufferingConfirmBus(confirm_ok=True)
    store = InMemoryGraphStore()
    orch = Orchestrator(
        bus=bus,
        config=OrchestratorConfig(single_agent_id=AGENT_ID, stack_id=STACK_ID),
        store=store,
    )
    await orch.start()
    task = _task()

    participants_at_confirm: list[frozenset[str]] = []
    original_confirm = bus.confirm_publishes

    async def spying_confirm() -> None:
        graph = store.get(task.id)
        assert graph is not None
        participants_at_confirm.append(graph.participants)
        await original_confirm()

    bus.confirm_publishes = spying_confirm  # type: ignore[method-assign]
    await bus.deliver(subjects.TASK_INGRESS, task)

    assert participants_at_confirm == [frozenset()]
    graph = store.get(task.id)
    assert graph is not None
    assert graph.participants == frozenset({AGENT_ID})
    assert (subjects.agent_task(AGENT_ID), task) in bus.published


async def test_duo_flush_timeout_rolls_back_and_fails_closed() -> None:
    """(22) A duo batch whose confirmation fails records nobody and fails closed.

    Both lane publishes enqueue, the batch flush times out: the participant
    set is rolled back to empty, the partial-failure state latches, and both
    agents' terminal turns (buffered frames could still drain to them) never
    derive done.
    """
    bus = BufferingConfirmBus(confirm_ok=False)
    store = InMemoryGraphStore()
    orch = Orchestrator(
        bus=bus,
        config=OrchestratorConfig(single_agent_id=AGENT_ID, stack_id=STACK_ID),
        store=store,
    )
    await orch.start()
    task = _task(scope="duo", participants=["codex-a", "opus-b"])
    await bus.deliver(subjects.TASK_INGRESS, task)

    graph = store.get(task.id)
    assert graph is not None
    assert graph.participants == frozenset()
    assert graph.dispatch_failed is True

    await bus.deliver(subjects.task_turn(task.id), _turn(agent_id="codex-a"))
    await bus.deliver(subjects.task_turn(task.id), _turn(agent_id="opus-b"))
    assert not any(p[0] == subjects.task_result(task.id) for p in bus.published)


async def test_confirmed_dispatch_records_exactly_once() -> None:
    """(23) The confirmed path records the participant set exactly once.

    One task delivery -> one publish batch, one confirmation round-trip, one
    recorded participant set; an identical redelivery re-confirms nothing and
    the terminal turn publishes exactly one result.
    """
    bus = FakeFrameBus()
    store = InMemoryGraphStore()
    orch = Orchestrator(
        bus=bus,
        config=OrchestratorConfig(single_agent_id=AGENT_ID, stack_id=STACK_ID),
        store=store,
    )
    await orch.start()
    task = _task()
    await bus.deliver(subjects.TASK_INGRESS, task)
    await bus.deliver(subjects.TASK_INGRESS, task)

    assert bus.confirms == 1
    graph = store.get(task.id)
    assert graph is not None
    assert graph.participants == frozenset({AGENT_ID})
    dispatches = [p for p in bus.published if p[0] == subjects.agent_task(AGENT_ID)]
    assert len(dispatches) == 1

    own = _turn(agent_id=AGENT_ID)
    await bus.deliver(subjects.task_turn(task.id), own)
    results = [p for p in bus.published if p[0] == subjects.task_result(task.id)]
    assert results == [(subjects.task_result(task.id), own)]
