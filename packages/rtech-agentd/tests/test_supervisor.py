"""Supervisor: registry resume across restart, heartbeat, stall watchdog."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from rtech_agentd.engine import TextDelta, ToolsPolicy, TurnDone
from rtech_agentd.heartbeat import HeartbeatConfig, HeartbeatPublisher, heartbeat_subject
from rtech_agentd.lifecycle_policy import RestartPolicy, RotationPolicy
from rtech_agentd.registry import ActiveSessionRegistry, DuplicateSessionError
from rtech_agentd.supervisor import (
    SessionRegistry,
    Supervisor,
    SupervisorConfig,
)
from rtech_frames import LifecycleFrame, StatusFrame
from rtech_orchestrator import subjects as rtech_subjects

from .fakes import FakeEngine, RecordingPublisher

Sleep = Callable[[float], Awaitable[None]]


def _fast_restart_policy() -> RestartPolicy:
    """A restart policy whose backoff is negligible (existing crash-path tests)."""
    return RestartPolicy(max_attempts=5, backoff_base_s=0.001, backoff_cap_s=0.001)


def _sup(
    engine: FakeEngine,
    pub: RecordingPublisher,
    home: Path,
    *,
    active_registry: ActiveSessionRegistry | None = None,
    sleep: Sleep = asyncio.sleep,
    restart_policy: RestartPolicy | None = None,
    rotation_policy: RotationPolicy | None = None,
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
        active_registry=active_registry or ActiveSessionRegistry(),
        sleep=sleep,
        restart_policy=restart_policy,
        rotation_policy=rotation_policy,
    )


def test_registry_atomic_roundtrip(tmp_path: Path) -> None:
    reg = SessionRegistry(tmp_path, "reg.json")
    assert reg.load() is None
    reg.save("session-xyz")
    assert reg.load() == "session-xyz"
    # No leftover temp files.
    leftovers = list(tmp_path.glob(".session-registry-*.tmp"))
    assert leftovers == []


def test_registry_refs_are_keyed_by_agent_id(tmp_path: Path) -> None:
    first = SessionRegistry(tmp_path, "reg.json", "first")
    second = SessionRegistry(tmp_path, "reg.json", "second")

    first.save("session-first")
    second.save("session-second")

    assert first.load() == "session-first"
    assert second.load() == "session-second"


def test_registry_ignores_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "reg.json"
    path.write_text("{not json", encoding="utf-8")
    reg = SessionRegistry(tmp_path, "reg.json")
    assert reg.load() is None  # corrupt -> None, never raises


@pytest.mark.asyncio
async def test_fresh_start_when_no_registry(tmp_path: Path) -> None:
    engine = FakeEngine()
    sup = _sup(engine, RecordingPublisher(), tmp_path)
    await sup.start()
    assert engine.started is True
    assert engine.resumed_ref is None
    await sup.stop()


@pytest.mark.asyncio
async def test_duplicate_session_start_for_same_agent_is_rejected(tmp_path: Path) -> None:
    registry = ActiveSessionRegistry()
    first = _sup(FakeEngine(), RecordingPublisher(), tmp_path, active_registry=registry)
    second = _sup(FakeEngine(), RecordingPublisher(), tmp_path, active_registry=registry)

    await first.start()
    with pytest.raises(DuplicateSessionError):
        await second.start()

    await first.stop()


@pytest.mark.asyncio
async def test_resume_across_restart(tmp_path: Path) -> None:
    # First supervisor: run pump, complete a turn -> registry gets the ref.
    engine1 = FakeEngine()
    pub = RecordingPublisher()
    sup1 = _sup(engine1, pub, tmp_path, heartbeat_interval_s=100.0, stall_timeout_s=100.0)
    await sup1.start()
    run1 = asyncio.create_task(sup1.run())
    await engine1.push(TextDelta(text="hi"))
    await engine1.push(TurnDone(text="hi", session_ref="resume-ref-42"))
    await asyncio.sleep(0.05)  # let the pump persist the ref
    await sup1.stop()
    run1.cancel()

    assert SessionRegistry(tmp_path, "session-registry.json", "tester").load() == "resume-ref-42"

    # Second supervisor on the same home: must RESUME that ref, not start fresh.
    engine2 = FakeEngine()
    sup2 = _sup(engine2, RecordingPublisher(), tmp_path)
    await sup2.start()
    assert engine2.resumed_ref == "resume-ref-42"
    await sup2.stop()


@pytest.mark.asyncio
async def test_heartbeat_cadence_uses_configured_subject_with_fake_clock() -> None:
    pub = RecordingPublisher()
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        if len(sleeps) >= 3:
            raise asyncio.CancelledError

    heartbeat = HeartbeatPublisher(
        config=HeartbeatConfig(agent_id="tester", env="dev", interval_s=2.5),
        publisher=pub,
        active_turn=lambda: False,
        sleep=fake_sleep,
    )

    with pytest.raises(asyncio.CancelledError):
        await heartbeat.run()

    assert sleeps == [2.5, 2.5, 2.5]
    assert [subject for subject, _env in pub.by_kind("heartbeat")] == [
        heartbeat_subject("dev", "tester"),
        heartbeat_subject("dev", "tester"),
        heartbeat_subject("dev", "tester"),
    ]


@pytest.mark.asyncio
async def test_supervisor_heartbeat_published(tmp_path: Path) -> None:
    engine = FakeEngine()
    pub = RecordingPublisher()
    sup = _sup(engine, pub, tmp_path, heartbeat_interval_s=0.01, stall_timeout_s=100.0)
    await sup.start()
    run = asyncio.create_task(sup.run())
    await asyncio.sleep(0.05)
    await sup.stop()
    run.cancel()
    hbs = pub.by_kind("heartbeat")
    assert len(hbs) >= 1
    subject, env = hbs[0]
    assert subject == "dev.agent.heartbeat.tester"
    assert env.sender == "tester"


@pytest.mark.asyncio
async def test_crash_resume_republishes_working_status(tmp_path: Path) -> None:
    SessionRegistry(tmp_path, "session-registry.json", "tester").save("resume-ref-99")
    engine = FakeEngine()
    engine.stream_failures = 1
    pub = RecordingPublisher()
    sup = _sup(
        engine,
        pub,
        tmp_path,
        heartbeat_interval_s=100.0,
        stall_timeout_s=100.0,
        restart_policy=_fast_restart_policy(),
    )
    await sup.start()
    sup.track_in_flight_task("task-123")
    run = asyncio.create_task(sup.run())
    await asyncio.sleep(0.05)
    await sup.stop()
    run.cancel()

    assert engine.resume_refs == ["resume-ref-99", "resume-ref-99"]
    statuses = pub.by_kind("status")
    assert len(statuses) == 1
    subject, env = statuses[0]
    assert subject == "dev.agent.status.task-123"
    assert StatusFrame.model_validate(env.payload) == StatusFrame(
        task_id="task-123",
        agent_id="tester",
        state="working",
    )


@pytest.mark.asyncio
async def test_stall_watchdog_emits_typed_blocked_status(tmp_path: Path) -> None:
    engine = FakeEngine()
    pub = RecordingPublisher()
    sup = _sup(engine, pub, tmp_path, heartbeat_interval_s=100.0, stall_timeout_s=0.03)
    await sup.start()
    sup.track_in_flight_task("task-stalled")
    run = asyncio.create_task(sup.run())
    await engine.push(TextDelta(text="thinking"))
    await asyncio.sleep(0.2)
    await sup.stop()
    run.cancel()
    stalls = [e for _s, e in pub.by_kind("status") if e.payload.get("state") == "blocked"]
    assert len(stalls) >= 1
    assert StatusFrame.model_validate(stalls[0].payload) == StatusFrame(
        task_id="task-stalled",
        agent_id="tester",
        state="blocked",
    )


@pytest.mark.asyncio
async def test_watchdog_maps_stall_to_frame_turn_tracked_task(tmp_path: Path) -> None:
    """Through frame_turn: a wedged turn's stall maps to the task, never 'no task'.

    Regression for the #51 mapping bug: FrameTurnRunner registers the in-flight
    task at send, so the watchdog emits a blocked status for THAT task_id instead
    of logging 'stall status skipped, no task' during a live turn.
    """
    from datetime import UTC, datetime

    from rtech_agentd.frame_turn import FrameTurnRunner
    from rtech_frames import TaskFrame
    from rtech_orchestrator import subjects

    from .fakes import FakeFrameBus

    engine = FakeEngine()
    pub = RecordingPublisher()
    sup = _sup(engine, pub, tmp_path, heartbeat_interval_s=100.0, stall_timeout_s=0.03)
    bus = FakeFrameBus()
    runner = FrameTurnRunner(agent_id="tester", supervisor=sup, bus=bus)

    await sup.start()
    run = asyncio.create_task(sup.run())
    await runner.start()

    # Stamp created_at: agentd fails closed on untimestamped untracked frames (#83).
    task = TaskFrame(
        id="T-wedge",
        origin_edge="web",
        requester="rico",
        text="hang",
        scope="single",
        created_at=datetime.now(UTC),
    )
    await bus.deliver(subjects.agent_task("tester"), task)
    # Open the turn (a delta) but never complete it — the turn wedges.
    await engine.push(TextDelta(text="thinking"))
    await asyncio.sleep(0.2)
    await sup.stop()
    run.cancel()

    assert sup.in_flight_task_id == "T-wedge"
    stalls = [e for _s, e in pub.by_kind("status") if e.payload.get("state") == "blocked"]
    assert stalls, "watchdog should emit a blocked status for the tracked task"
    assert StatusFrame.model_validate(stalls[0].payload) == StatusFrame(
        task_id="T-wedge",
        agent_id="tester",
        state="blocked",
    )


@pytest.mark.asyncio
async def test_status_frames_never_emit_completion_state(tmp_path: Path) -> None:
    engine = FakeEngine()
    engine.stream_failures = 1
    pub = RecordingPublisher()
    SessionRegistry(tmp_path, "session-registry.json", "tester").save("resume-ref-1")
    sup = _sup(
        engine,
        pub,
        tmp_path,
        heartbeat_interval_s=100.0,
        stall_timeout_s=100.0,
        restart_policy=_fast_restart_policy(),
    )
    await sup.start()
    sup.track_in_flight_task("task-abc")
    run = asyncio.create_task(sup.run())
    await asyncio.sleep(0.05)
    await sup.stop()
    run.cancel()

    for _subject, env in pub.by_kind("status"):
        assert StatusFrame.model_validate(env.payload).state in {"working", "blocked"}


@pytest.mark.asyncio
async def test_stop_publishes_shutdown_lifecycle_frame(tmp_path: Path) -> None:
    """stop() publishes a shutdown lifecycle frame with reason + initiator (#84)."""
    engine = FakeEngine()
    pub = RecordingPublisher()
    sup = _sup(engine, pub, tmp_path)
    await sup.start()
    await sup.stop(reason="shutdown", initiator="signal:SIGTERM")

    events = pub.by_kind("lifecycle")
    assert len(events) == 1
    subject, env = events[0]
    assert subject == rtech_subjects.agent_lifecycle("tester")
    frame = LifecycleFrame.model_validate(env.payload)
    assert frame.reason == "shutdown"
    assert frame.initiator == "signal:SIGTERM"
    assert frame.agent_id == "tester"
    assert frame.correlation_id  # a per-event id is stamped


@pytest.mark.asyncio
async def test_crash_recovery_publishes_correlated_lifecycle_pair(tmp_path: Path) -> None:
    """Recovery emits crash + supervisor_recovery sharing one correlation id (#84)."""
    SessionRegistry(tmp_path, "session-registry.json", "tester").save("resume-ref-7")
    engine = FakeEngine()
    engine.stream_failures = 1
    pub = RecordingPublisher()
    sup = _sup(
        engine,
        pub,
        tmp_path,
        heartbeat_interval_s=100.0,
        stall_timeout_s=100.0,
        restart_policy=_fast_restart_policy(),
    )
    await sup.start()
    run = asyncio.create_task(sup.run())
    await asyncio.sleep(0.05)
    await sup.stop()
    run.cancel()

    frames = [LifecycleFrame.model_validate(e.payload) for _s, e in pub.by_kind("lifecycle")]
    crash = [f for f in frames if f.reason == "crash"]
    recovery = [f for f in frames if f.reason == "supervisor_recovery"]
    assert len(crash) == 1
    assert len(recovery) == 1
    # The death and its recovery correlate through the SAME id.
    assert crash[0].correlation_id == recovery[0].correlation_id
    # stream-error path stamps the distinguishing detail.
    assert crash[0].detail == "stream-error"
    assert recovery[0].detail == "stream-error"


@pytest.mark.asyncio
async def test_recovery_action_failure_emits_correlated_recovery_failed_crash(
    tmp_path: Path,
) -> None:
    """A failed recovery action pairs the crash frame and re-raises (#84 slice B).

    The pump crashes, ``_recover_dead_session`` publishes a ``crash`` frame, then
    the restart action itself raises. Instead of orphaning that crash frame, the
    supervisor publishes a matching ``crash``/``recovery-failed`` frame under the
    SAME correlation id and re-raises — retry is slice A's RestartPolicy up-stack.
    """

    class DeadEngine(FakeEngine):
        async def resume(self, session_ref: str) -> None:
            raise RuntimeError("resume failed")

        async def start(self, system_prompt: str, tools_policy: ToolsPolicy) -> None:
            raise RuntimeError("start failed")

    engine = DeadEngine()
    pub = RecordingPublisher()
    sup = _sup(engine, pub, tmp_path, heartbeat_interval_s=100.0, stall_timeout_s=100.0)
    # Bypass start() (which would itself hit the raising engine); drive recovery
    # directly so we assert on the pump's crash-frame pairing in isolation.
    with pytest.raises(RuntimeError):
        await sup._recover_dead_session("stream-error")

    crash = [
        LifecycleFrame.model_validate(e.payload)
        for _s, e in pub.by_kind("lifecycle")
        if e.payload.get("reason") == "crash"
    ]
    assert len(crash) == 2
    # Both crash frames correlate through the SAME id: the death and its
    # recovery-failed close carry one correlation id, never an orphan.
    assert crash[0].correlation_id == crash[1].correlation_id
    assert crash[0].detail == "stream-error"
    assert crash[1].detail == "recovery-failed"
    # No recovery frame is published when the action failed.
    assert not [
        f for f in pub.by_kind("lifecycle") if f[1].payload.get("reason") == "supervisor_recovery"
    ]


@pytest.mark.asyncio
async def test_double_stop_publishes_exactly_one_shutdown_frame(tmp_path: Path) -> None:
    """A second stop() is a no-op for the lifecycle publish (#84 slice B)."""
    engine = FakeEngine()
    pub = RecordingPublisher()
    sup = _sup(engine, pub, tmp_path)
    await sup.start()
    await sup.stop()
    await sup.stop()

    shutdowns = [e for _s, e in pub.by_kind("lifecycle") if e.payload.get("reason") == "shutdown"]
    assert len(shutdowns) == 1


@pytest.mark.asyncio
async def test_stop_proceeds_when_lifecycle_publish_fails(tmp_path: Path) -> None:
    """A failed lifecycle publish must never prevent a clean stop (#84)."""

    class FailingPublisher(RecordingPublisher):
        async def __call__(self, subject: str, envelope: object) -> None:  # type: ignore[override]
            raise RuntimeError("bus down")

    engine = FakeEngine()
    sup = _sup(engine, FailingPublisher(), tmp_path)
    await sup.start()
    # Must not raise despite the publish failing.
    await sup.stop()
    assert engine.closed is True
