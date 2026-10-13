"""Bounded restart (circuit breaker) + bounded context rotation (issue #84).

House style: injected ``FakeEngine`` / ``RecordingPublisher`` / fake sleep, a fake
clock for age/backoff, policies constructed with explicit tunables (never env) so
each behavior is pinned deterministically.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from rtech_agentd.engine import EngineError, TextDelta, ToolsPolicy, TurnDone
from rtech_agentd.lifecycle_policy import RestartPolicy, RotationPolicy
from rtech_agentd.registry import ActiveSessionRegistry, SessionRegistry
from rtech_agentd.supervisor import CircuitOpenError, Supervisor, SupervisorConfig
from rtech_frames import LifecycleFrame, StatusFrame

from .fakes import FakeEngine, RecordingPublisher


def _sup(
    engine: FakeEngine,
    pub: RecordingPublisher,
    home: Path,
    *,
    restart_policy: RestartPolicy,
    rotation_policy: RotationPolicy,
    sleep: object = asyncio.sleep,
    engine_factory: object = None,
    **overrides: object,
) -> Supervisor:
    cfg = SupervisorConfig(agent="tester", env="dev", **overrides)  # type: ignore[arg-type]
    return Supervisor(
        session=engine,
        config=cfg,
        home=home,
        publisher=pub,
        system_prompt="SYS",
        tools_policy=ToolsPolicy(),
        active_registry=ActiveSessionRegistry(),
        sleep=sleep,  # type: ignore[arg-type]
        restart_policy=restart_policy,
        rotation_policy=rotation_policy,
        engine_factory=engine_factory,  # type: ignore[arg-type]
    )


# --- RestartPolicy (pure) ----------------------------------------------------


def test_restart_backoff_is_exponential_and_capped() -> None:
    policy = RestartPolicy(max_attempts=6, backoff_base_s=1.0, backoff_cap_s=8.0)
    backoffs = [policy.on_dead_session().backoff_s for _ in range(6)]
    # 1, 2, 4, 8, then capped at 8, 8.
    assert backoffs == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


def test_restart_circuit_opens_after_max_attempts() -> None:
    policy = RestartPolicy(max_attempts=3, backoff_base_s=1.0, backoff_cap_s=60.0)
    decisions = [policy.on_dead_session() for _ in range(4)]
    assert [d.should_restart for d in decisions] == [True, True, True, False]
    assert policy.attempts == 3
    # Once open it stays open on further dead sessions.
    assert policy.on_dead_session().should_restart is False


def test_restart_counter_resets_on_healthy() -> None:
    policy = RestartPolicy(max_attempts=3, backoff_base_s=1.0, backoff_cap_s=60.0)
    policy.on_dead_session()
    policy.on_dead_session()
    assert policy.attempts == 2
    policy.on_healthy()
    assert policy.attempts == 0
    # Full budget available again.
    assert [policy.on_dead_session().should_restart for _ in range(3)] == [True, True, True]


def test_restart_healthy_window_elapses_by_clock() -> None:
    clock = {"t": 0.0}
    policy = RestartPolicy(
        max_attempts=3,
        backoff_base_s=1.0,
        backoff_cap_s=60.0,
        healthy_reset_s=300.0,
        now=lambda: clock["t"],
    )
    policy.on_connected()
    clock["t"] = 250.0
    assert policy.seconds_since_connect() == 250.0
    assert policy.seconds_since_connect() < policy.healthy_reset_s
    clock["t"] = 301.0
    assert policy.seconds_since_connect() >= policy.healthy_reset_s


# --- RestartPolicy wired into the supervisor ---------------------------------


@pytest.mark.asyncio
async def test_supervisor_backs_off_between_restart_attempts(tmp_path: Path) -> None:
    """Each restart sleeps the policy's backoff via the injected sleep."""
    SessionRegistry(tmp_path, "session-registry.json", "tester").save("ref-1")
    engine = FakeEngine()
    engine.stream_failures = 2  # crash twice, then serve
    pub = RecordingPublisher()
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        if delay >= 100.0:  # heartbeat/watchdog: block for real, never re-fires
            await asyncio.sleep(delay)
        # backoff sleeps (1s, 2s) resolve instantly so the test window is tight.

    sup = _sup(
        engine,
        pub,
        tmp_path,
        restart_policy=RestartPolicy(max_attempts=5, backoff_base_s=1.0, backoff_cap_s=60.0),
        rotation_policy=RotationPolicy(max_turns=1000, max_age_s=1e9),
        sleep=fake_sleep,
        heartbeat_interval_s=1000.0,
        stall_timeout_s=1000.0,
    )
    await sup.start()
    run = asyncio.create_task(sup.run())
    await asyncio.sleep(0.05)
    await sup.stop()
    run.cancel()

    # Two crashes -> two authorized restarts with 1s then 2s backoff (heartbeat
    # sleep is 1000s and never lands in this window).
    assert 1.0 in sleeps and 2.0 in sleeps
    # One startup resume + two recovery resumes.
    assert engine.resume_refs == ["ref-1", "ref-1", "ref-1"]


@pytest.mark.asyncio
async def test_circuit_opens_blocks_and_stops_resuming(tmp_path: Path) -> None:
    """After max attempts: blocked status + circuit-open frame, no more resumes."""
    SessionRegistry(tmp_path, "session-registry.json", "tester").save("ref-x")
    engine = FakeEngine()
    engine.stream_failures = 50  # never recovers
    pub = RecordingPublisher()

    async def fake_sleep(delay: float) -> None:
        if delay >= 100.0:  # heartbeat/watchdog: block for real
            await asyncio.sleep(delay)
        # tiny backoff resolves instantly so the circuit opens fast.

    sup = _sup(
        engine,
        pub,
        tmp_path,
        restart_policy=RestartPolicy(max_attempts=2, backoff_base_s=0.001, backoff_cap_s=0.001),
        rotation_policy=RotationPolicy(max_turns=1000, max_age_s=1e9),
        sleep=fake_sleep,
        heartbeat_interval_s=1000.0,
        stall_timeout_s=1000.0,
    )
    await sup.start()
    sup.track_in_flight_task("task-boom")
    run = asyncio.create_task(sup.run())
    # run() terminates with the fatal CircuitOpenError marker once the budget is
    # spent — the signal __main__ maps to a non-zero exit for systemd.
    with pytest.raises(CircuitOpenError):
        await asyncio.wait_for(run, timeout=1.0)

    assert sup.circuit_open is True
    # One startup resume + exactly max_attempts (2) recovery resumes, then it stops.
    assert engine.resume_refs == ["ref-x", "ref-x", "ref-x"]

    frames = [LifecycleFrame.model_validate(e.payload) for _s, e in pub.by_kind("lifecycle")]
    circuit = [f for f in frames if f.reason == "crash" and f.detail == "circuit_open"]
    assert len(circuit) == 1

    blocked = [
        StatusFrame.model_validate(e.payload)
        for _s, e in pub.by_kind("status")
        if e.payload.get("state") == "blocked"
    ]
    assert StatusFrame(task_id="task-boom", agent_id="tester", state="blocked") in blocked

    await sup.stop()


@pytest.mark.asyncio
async def test_healthy_turn_done_resets_restart_budget(tmp_path: Path) -> None:
    """A crash consumes an attempt; a later clean TurnDone resets the budget."""
    engine = FakeEngine()
    engine.stream_failures = 1
    pub = RecordingPublisher()
    policy = RestartPolicy(max_attempts=5, backoff_base_s=0.001, backoff_cap_s=0.001)

    async def fake_sleep(delay: float) -> None:
        if delay >= 100.0:  # heartbeat/watchdog: block for real
            await asyncio.sleep(delay)

    sup = _sup(
        engine,
        pub,
        tmp_path,
        restart_policy=policy,
        rotation_policy=RotationPolicy(max_turns=1000, max_age_s=1e9),
        sleep=fake_sleep,
        heartbeat_interval_s=1000.0,
        stall_timeout_s=1000.0,
    )
    await sup.start()
    run = asyncio.create_task(sup.run())
    await asyncio.sleep(0.05)  # one crash -> attempt consumed
    assert policy.attempts == 1
    await engine.push(TurnDone(text="ok", session_ref="ref-ok"))
    await asyncio.sleep(0.05)
    assert policy.attempts == 0
    await sup.stop()
    run.cancel()


@pytest.mark.asyncio
async def test_healthy_reset_s_resets_budget_by_clock_without_turn_done(tmp_path: Path) -> None:
    """The time-based healthy reset refunds the budget with NO TurnDone (fake clock).

    healthy_reset_s was documented but never wired: an idle-but-connected session
    that consumed attempts earlier would carry them into a much later transient
    crash and lock out prematurely. The watchdog now resets the budget once the
    healthy window elapses on the policy's injected monotonic clock.
    """
    SessionRegistry(tmp_path, "session-registry.json", "tester").save("ref-h")
    engine = FakeEngine()
    engine.stream_failures = 1  # crash once, then serve (consumes one attempt)
    pub = RecordingPublisher()
    clock = {"t": 0.0}
    policy = RestartPolicy(
        max_attempts=5,
        backoff_base_s=0.001,
        backoff_cap_s=0.001,
        healthy_reset_s=300.0,
        now=lambda: clock["t"],
    )
    sup = _sup(
        engine,
        pub,
        tmp_path,
        restart_policy=policy,
        rotation_policy=RotationPolicy(max_turns=1000, max_age_s=1e9),
        heartbeat_interval_s=1000.0,
        stall_timeout_s=0.03,  # fast watchdog tick so the reset check runs
    )
    await sup.start()
    run = asyncio.create_task(sup.run())
    await asyncio.sleep(0.05)  # one crash -> one attempt consumed, then recovered
    assert policy.attempts == 1

    # No TurnDone at all. Advance the clock past the healthy window; the watchdog's
    # periodic reset check then refunds the budget.
    clock["t"] = 301.0
    await asyncio.sleep(0.1)
    assert policy.attempts == 0

    await sup.stop()
    run.cancel()


# --- RotationPolicy (pure) ---------------------------------------------------


def test_rotation_triggers_at_turn_bound() -> None:
    policy = RotationPolicy(max_turns=3, max_age_s=1e9)
    reasons = [policy.on_turn_done() for _ in range(3)]
    assert [d.should_rotate for d in reasons] == [False, False, True]
    assert reasons[-1].reason == "turns"


def test_rotation_triggers_at_age_bound() -> None:
    clock = {"t": 0.0}
    policy = RotationPolicy(max_turns=1000, max_age_s=100.0, now=lambda: clock["t"])
    assert policy.on_turn_done().should_rotate is False
    clock["t"] = 150.0
    decision = policy.on_turn_done()
    assert decision.should_rotate is True
    assert decision.reason == "age"


def test_rotation_defer_backs_off_one_turn() -> None:
    policy = RotationPolicy(max_turns=3, max_age_s=1e9)
    for _ in range(3):
        policy.on_turn_done()
    policy.defer()
    # Next turn re-triggers rather than every subsequent turn firing.
    assert policy.on_turn_done().should_rotate is True


# --- RotationPolicy wired into the supervisor --------------------------------


@pytest.mark.asyncio
async def test_rotation_swaps_to_fresh_engine_and_persists_new_ref(tmp_path: Path) -> None:
    """Turn-bound rotation: summary turn, build+start a FRESH engine, persist its ref.

    The swap must use a brand-new engine instance from the factory (identity
    assert) — never in-place close()+start() on the same object — so the real
    ClaudeEngine cannot bleed a stale reader/queue across the rotation.
    """
    engine = FakeEngine()
    fresh = FakeEngine()
    fresh._session_ref = "new-ref"  # noqa: SLF001 - pin the fresh engine's ref
    pub = RecordingPublisher()
    sup = _sup(
        engine,
        pub,
        tmp_path,
        restart_policy=RestartPolicy(max_attempts=5),
        rotation_policy=RotationPolicy(max_turns=1, max_age_s=1e9, summary_timeout_s=1000.0),
        engine_factory=lambda: fresh,
        heartbeat_interval_s=1000.0,
        stall_timeout_s=1000.0,
    )
    await sup.start()
    run = asyncio.create_task(sup.run())

    # One completed turn crosses max_turns=1 -> rotation fires, summary sent.
    await engine.push(TurnDone(text="first", session_ref="old-ref"))
    await asyncio.sleep(0.05)
    assert any(s.text.startswith("Your context is being rotated") for s in engine.sends)

    # The summary turn completes -> swap to the fresh engine.
    await engine.push(TurnDone(text="SUMMARY", session_ref="ignored-old-ref"))
    await asyncio.sleep(0.05)

    # A FRESH engine instance was built, started, and swapped in (identity assert).
    assert sup.session is fresh
    assert sup.session is not engine
    assert fresh.started is True
    assert fresh.start_count == 1
    assert engine.closed is True  # the OLD engine was closed
    # The persisted ref comes from the FRESH engine's health, not the old one.
    assert SessionRegistry(tmp_path, "session-registry.json", "tester").load() == "new-ref"

    frames = [LifecycleFrame.model_validate(e.payload) for _s, e in pub.by_kind("lifecycle")]
    rot = [f for f in frames if f.reason == "requested_restart" and f.initiator == "rotation"]
    assert len(rot) == 1

    await sup.stop()
    run.cancel()


@pytest.mark.asyncio
async def test_rotation_disabled_without_engine_factory(tmp_path: Path) -> None:
    """No engine_factory -> rotation is disabled (never fires), logged once."""
    engine = FakeEngine()
    pub = RecordingPublisher()
    sup = _sup(
        engine,
        pub,
        tmp_path,
        restart_policy=RestartPolicy(max_attempts=5),
        rotation_policy=RotationPolicy(max_turns=1, max_age_s=1e9),
        engine_factory=None,  # rotation disabled
        heartbeat_interval_s=1000.0,
        stall_timeout_s=1000.0,
    )
    await sup.start()
    run = asyncio.create_task(sup.run())

    # Multiple completed turns would each cross max_turns=1, but with no factory
    # no summary directive is EVER sent and the session is never swapped.
    await engine.push(TurnDone(text="one", session_ref="ref-1"))
    await engine.push(TurnDone(text="two", session_ref="ref-2"))
    await asyncio.sleep(0.05)

    assert not any(s.text.startswith("Your context is being rotated") for s in engine.sends)
    assert sup.session is engine  # never swapped
    assert engine.start_count == 1  # only the startup start; no rotation restart

    await sup.stop()
    run.cancel()


@pytest.mark.asyncio
async def test_task_arriving_mid_summary_aborts_rotation_and_serves_task(tmp_path: Path) -> None:
    """Work landing during the summary turn aborts rotation; the task is served.

    The summary turn's own TurnDone is consumed by the rotation state machine and
    NEVER correlated to the arriving task; the session is not swapped; the task is
    served and its TurnFrame is published under its OWN task_id (issue #84).
    """
    from rtech_agentd.frame_turn import FrameTurnRunner
    from rtech_frames import TaskFrame, TurnFrame
    from rtech_orchestrator import subjects

    from .fakes import FakeFrameBus

    SessionRegistry(tmp_path, "session-registry.json", "tester").save("keep-ref")
    engine = FakeEngine()
    fresh_built = False

    def _factory() -> FakeEngine:
        nonlocal fresh_built
        fresh_built = True
        return FakeEngine()

    pub = RecordingPublisher()
    sup = _sup(
        engine,
        pub,
        tmp_path,
        restart_policy=RestartPolicy(max_attempts=5),
        rotation_policy=RotationPolicy(max_turns=1, max_age_s=1e9, summary_timeout_s=1000.0),
        engine_factory=_factory,
        heartbeat_interval_s=1000.0,
        stall_timeout_s=1000.0,
    )
    bus = FakeFrameBus()
    runner = FrameTurnRunner(agent_id="tester", supervisor=sup, bus=bus)
    await sup.start()
    run = asyncio.create_task(sup.run())
    await runner.start()

    # Task 1 runs and completes -> crosses max_turns=1 -> rotation fires (summary).
    t1 = TaskFrame(
        id="T1",
        origin_edge="web",
        requester="rico",
        text="do one",
        scope="single",
        created_at=datetime.now(UTC),
    )
    await bus.deliver(subjects.agent_task("tester"), t1)
    await engine.push(TurnDone(text="one done", session_ref="keep-ref"))
    await asyncio.sleep(0.05)
    assert any(s.text.startswith("Your context is being rotated") for s in engine.sends)
    assert sup._rotation_pending is not None  # noqa: SLF001 - summary in flight

    # Task 2 arrives DURING the summary turn -> rotation aborts, task is served.
    t2 = TaskFrame(
        id="T2",
        origin_edge="web",
        requester="rico",
        text="do two",
        scope="single",
        created_at=datetime.now(UTC),
    )
    await bus.deliver(subjects.agent_task("tester"), t2)
    await asyncio.sleep(0)
    assert sup._rotation_aborted is True  # noqa: SLF001

    # The summary turn's TurnDone lands first (sent first) -> discarded, no swap.
    await engine.push(TurnDone(text="SUMMARY (too late)", session_ref="summary-ref"))
    await asyncio.sleep(0.05)
    # Then task 2's own TurnDone lands -> served and correlated to T2.
    await engine.push(TurnDone(text="two done", session_ref="keep-ref"))
    await asyncio.sleep(0.05)

    # No swap: same engine, old ref preserved, no fresh engine built.
    assert sup.session is engine
    assert fresh_built is False
    assert SessionRegistry(tmp_path, "session-registry.json", "tester").load() == "keep-ref"

    # T2 was served with CORRECT correlation; the summary was never mapped to a task.
    turns = [f for _s, f in bus.by_type(TurnFrame)]
    turn_task_ids = [f.task_id for f in turns]
    assert "T2" in turn_task_ids
    assert not any(f.visible_text.startswith("SUMMARY") for f in turns)

    await sup.stop()
    run.cancel()


@pytest.mark.asyncio
async def test_task_arriving_during_summary_send_aborts_rotation_and_serves_task(
    tmp_path: Path,
) -> None:
    """A task landing INSIDE the summary send() await aborts rotation; no swap.

    The idle-only guard is only sound if the ``_rotation_pending`` mark is set
    BEFORE the summary send's await, in one synchronous section with the idle
    check. This pins that window: a ``GatedSendEngine`` parks inside ``send()``
    while a TaskFrame is delivered, so the task's :meth:`track_in_flight_task`
    fires DURING the send. It must see the pending rotation and abort it — the
    summary's own TurnDone is discarded, the session is NOT swapped, and the task
    is served under its OWN task_id. Marking after the send (the race) would let
    rotation proceed against a no-longer-idle session and swap it under live work.
    """
    from rtech_agentd.frame_turn import FrameTurnRunner
    from rtech_frames import TaskFrame, TurnFrame
    from rtech_orchestrator import subjects

    from .fakes import FakeFrameBus

    class GatedSendEngine(FakeEngine):
        """A FakeEngine whose summary send parks on a gate, yielding control.

        The FIRST send of the rotation directive waits on ``send_entered`` (set so
        the driver knows the await window is open) and blocks on ``send_release``
        until the driver has injected task work; every other send is immediate.
        """

        def __init__(self) -> None:
            super().__init__()
            self.send_entered = asyncio.Event()
            self.send_release = asyncio.Event()
            self._gated_once = False

        async def send(self, text: str) -> None:
            if text.startswith("Your context is being rotated") and not self._gated_once:
                self._gated_once = True
                self.send_entered.set()
                await self.send_release.wait()
            await super().send(text)

    SessionRegistry(tmp_path, "session-registry.json", "tester").save("keep-ref")
    engine = GatedSendEngine()
    fresh_built = False

    def _factory() -> FakeEngine:
        nonlocal fresh_built
        fresh_built = True
        return FakeEngine()

    pub = RecordingPublisher()
    sup = _sup(
        engine,
        pub,
        tmp_path,
        restart_policy=RestartPolicy(max_attempts=5),
        rotation_policy=RotationPolicy(max_turns=1, max_age_s=1e9, summary_timeout_s=1000.0),
        engine_factory=_factory,
        heartbeat_interval_s=1000.0,
        stall_timeout_s=1000.0,
    )
    bus = FakeFrameBus()
    runner = FrameTurnRunner(agent_id="tester", supervisor=sup, bus=bus)
    await sup.start()
    run = asyncio.create_task(sup.run())
    await runner.start()

    # Task 1 completes -> crosses max_turns=1 -> rotation fires. The summary send
    # parks inside its await (send_entered), with _rotation_pending ALREADY marked.
    t1 = TaskFrame(
        id="T1",
        origin_edge="web",
        requester="rico",
        text="do one",
        scope="single",
        created_at=datetime.now(UTC),
    )
    await bus.deliver(subjects.agent_task("tester"), t1)
    await engine.push(TurnDone(text="one done", session_ref="keep-ref"))
    await asyncio.wait_for(engine.send_entered.wait(), timeout=1.0)
    assert sup._rotation_pending is not None  # noqa: SLF001 - marked BEFORE the send await

    # Task 2 arrives WHILE the summary send is still in flight -> must abort.
    t2 = TaskFrame(
        id="T2",
        origin_edge="web",
        requester="rico",
        text="do two",
        scope="single",
        created_at=datetime.now(UTC),
    )
    await bus.deliver(subjects.agent_task("tester"), t2)
    await asyncio.sleep(0)
    assert sup._rotation_aborted is True  # noqa: SLF001 - abort seen during the send window

    # Release the parked send; the abort is honored on return (no swap).
    engine.send_release.set()
    await asyncio.sleep(0.05)

    # The summary turn's TurnDone lands -> discarded, no swap. Then T2's own TurnDone.
    await engine.push(TurnDone(text="SUMMARY (too late)", session_ref="summary-ref"))
    await asyncio.sleep(0.05)
    await engine.push(TurnDone(text="two done", session_ref="keep-ref"))
    await asyncio.sleep(0.05)

    # No session swap: same engine, old ref preserved, no fresh engine built.
    assert sup.session is engine
    assert fresh_built is False
    assert SessionRegistry(tmp_path, "session-registry.json", "tester").load() == "keep-ref"

    # T2 served with CORRECT correlation; the summary was never mapped to a task.
    turns = [f for _s, f in bus.by_type(TurnFrame)]
    assert "T2" in [f.task_id for f in turns]
    assert not any(f.visible_text.startswith("SUMMARY") for f in turns)

    # Rotation deferred with backoff (policy count dropped just under the bound),
    # so the NEXT idle TurnDone re-fires rather than every turn re-firing.
    assert sup._rotation_policy._deferred is True  # noqa: SLF001

    await sup.stop()
    run.cancel()


@pytest.mark.asyncio
async def test_rotation_summary_timeout_preserves_ref_and_does_not_retry(tmp_path: Path) -> None:
    """A summary timeout keeps the old ref, emits a structured failure, no retry."""
    SessionRegistry(tmp_path, "session-registry.json", "tester").save("keep-ref")
    engine = FakeEngine()
    pub = RecordingPublisher()

    # A controllable sleep: gate ONLY the 120s summary deadline; heartbeat/
    # watchdog (1000s) block for real so they never re-fire in the window.
    released = asyncio.Event()

    async def fake_sleep(delay: float) -> None:
        if delay == 120.0:  # the summary timeout deadline
            await released.wait()
            return
        if delay >= 100.0:  # heartbeat/watchdog
            await asyncio.sleep(delay)
            return
        await asyncio.sleep(0)

    sup = _sup(
        engine,
        pub,
        tmp_path,
        restart_policy=RestartPolicy(max_attempts=5),
        rotation_policy=RotationPolicy(max_turns=1, max_age_s=1e9, summary_timeout_s=120.0),
        engine_factory=lambda: FakeEngine(),
        sleep=fake_sleep,
        heartbeat_interval_s=1000.0,
        stall_timeout_s=1000.0,
    )
    errors: list[EngineError] = []

    async def _sink(event: object) -> None:
        if isinstance(event, EngineError):
            errors.append(event)

    sup.add_event_sink(_sink)

    await sup.start()
    run = asyncio.create_task(sup.run())

    await engine.push(TurnDone(text="first", session_ref="keep-ref"))
    await asyncio.sleep(0.05)
    sends_before = len(engine.sends)
    assert sends_before == 1  # the single summary directive, one attempt

    # Fire the deadline: summary times out.
    released.set()
    await asyncio.sleep(0.05)

    # Old ref preserved, session NOT swapped.
    assert SessionRegistry(tmp_path, "session-registry.json", "tester").load() == "keep-ref"
    assert engine.start_count == 0  # start() only from a confirmed swap; none happened

    # A structured EngineError-derived failure was emitted to sinks.
    assert any("rotation aborted" in e.message for e in errors)
    # The lifecycle failure frame carries the rotation_summary_failed detail.
    frames = [LifecycleFrame.model_validate(e.payload) for _s, e in pub.by_kind("lifecycle")]
    failed = [f for f in frames if f.detail == "rotation_summary_failed"]
    assert len(failed) == 1

    # The summary is NOT retried immediately (no second directive send).
    assert len(engine.sends) == sends_before

    await sup.stop()
    run.cancel()


@pytest.mark.asyncio
async def test_rotation_never_fires_mid_turn(tmp_path: Path) -> None:
    """A delta (mid-turn) never triggers rotation; only a completed TurnDone can."""
    engine = FakeEngine()
    pub = RecordingPublisher()
    sup = _sup(
        engine,
        pub,
        tmp_path,
        restart_policy=RestartPolicy(max_attempts=5),
        rotation_policy=RotationPolicy(max_turns=1, max_age_s=1e9),
        engine_factory=lambda: FakeEngine(),
        heartbeat_interval_s=1000.0,
        stall_timeout_s=1000.0,
    )
    await sup.start()
    run = asyncio.create_task(sup.run())

    # Several mid-turn deltas: rotation must NOT fire (no summary directive).
    await engine.push(TextDelta(text="thinking-1"))
    await engine.push(TextDelta(text="thinking-2"))
    await asyncio.sleep(0.05)
    assert engine.sends == []  # no rotation summary sent mid-turn

    await sup.stop()
    run.cancel()
