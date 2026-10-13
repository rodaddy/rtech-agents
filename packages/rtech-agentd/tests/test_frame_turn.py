"""Frame-driven managed turn: TaskFrame in, working+TurnFrame out, no done."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rtech_agentd.completed_store import CompletedTaskStore
from rtech_agentd.engine import ToolsPolicy, TurnDone
from rtech_agentd.frame_turn import (
    COMPLETED_TASK_CACHE_SIZE,
    FUTURE_SKEW_TOLERANCE_S,
    MAX_SIBLING_ROUNDS,
    FrameTurnConfig,
    FrameTurnRunner,
    _render_debate_task,
    _short_name,
)
from rtech_agentd.scripted_engine import ScriptedEngine
from rtech_agentd.supervisor import Supervisor, SupervisorConfig
from rtech_frames import Stance, StatusFrame, SteerFrame, TaskFrame, TurnFrame
from rtech_frames.codec import Frame, encode_frame
from rtech_frames.subs import SubscriptionSet
from rtech_orchestrator import subjects

from .fakes import FakeEngine, FakeFrameBus, RecordingPublisher

_AGENT = "tester"


def _make(tmp_path: Path) -> tuple[FakeEngine, FrameTurnRunner, FakeFrameBus]:
    engine = FakeEngine()
    # The supervisor's fleet Publisher is separate from the FrameBus: a
    # RecordingPublisher satisfies the fleet-envelope publisher seam, while the
    # FakeFrameBus is the typed-frame seam the runner uses.
    sup = Supervisor(
        session=engine,
        config=SupervisorConfig(agent=_AGENT, env="dev"),
        home=tmp_path,
        publisher=RecordingPublisher(),
        system_prompt="SYS",
        tools_policy=ToolsPolicy(),
    )
    bus = FakeFrameBus()
    runner = FrameTurnRunner(agent_id=_AGENT, supervisor=sup, bus=bus)
    return engine, runner, bus


def _task(task_id: str = "T-1", text: str = "run it") -> TaskFrame:
    # Every live producer (nats_ingress / chatd / webchat) stamps ``created_at``,
    # and the orchestrator republishes the frame verbatim, so a freshly-stamped
    # frame is the real on-wire shape agentd sees. Tests that specifically exercise
    # the missing-timestamp fail-closed path build their own untimestamped frame.
    return TaskFrame(
        id=task_id,
        origin_edge="web",
        requester="rico",
        text=text,
        scope="single",
        created_at=datetime.now(UTC),
    )


async def test_task_frame_emits_working_then_turn_with_typed_stance(tmp_path: Path) -> None:
    engine, runner, bus = _make(tmp_path)
    await runner.start()
    task = _task()

    await bus.deliver(subjects.agent_task(_AGENT), task)
    assert engine.sends[-1].text == task.text

    status_subj = subjects.task_status(task.id)
    working = [f for f in bus.frames_on(status_subj) if isinstance(f, StatusFrame)]
    assert working, "expected a StatusFrame on the task status lane"
    assert working[0].state == "working"
    assert working[0].task_id == task.id

    await runner._on_engine_event(
        TurnDone(text="hi", stance=Stance(position="answered", because="did it"))
    )
    turn_subj = subjects.task_turn(task.id)
    turns = [f for f in bus.frames_on(turn_subj) if isinstance(f, TurnFrame)]
    assert len(turns) == 1
    turn = turns[0]
    assert turn.visible_text == "hi"
    assert turn.stance.position == "answered"
    assert turn.stance.because == "did it"
    assert turn.task_id == task.id
    assert turn.agent_id == _AGENT


async def test_status_transitions(tmp_path: Path) -> None:
    _engine, runner, bus = _make(tmp_path)
    task_id = "T-status"

    await runner.publish_status(task_id, "input_required")
    await runner.publish_status(task_id, "blocked")

    subject = subjects.task_status(task_id)
    frames = [f for f in bus.frames_on(subject) if isinstance(f, StatusFrame)]
    states = [f.state for f in frames]
    assert states == ["input_required", "blocked"]
    assert all(f.task_id == task_id for f in frames)


async def test_no_done_token_emitted(tmp_path: Path) -> None:
    _engine, runner, bus = _make(tmp_path)
    await runner.start()
    task = _task(task_id="T-2")

    await bus.deliver(subjects.agent_task(_AGENT), task)
    await runner._on_engine_event(TurnDone(text="wrapping up [[done:completed]]"))

    # Forbidden tokens assembled from fragments so the sentinel grep-gate over
    # this package stays clean (the literals must not appear in source).
    done_token = "[[" + "done"
    sentinel = "done" + "_sentinel"
    assert bus.published, "expected at least one published frame"
    for _subject, frame in bus.published:
        blob = encode_frame(frame)
        assert done_token.encode("utf-8") not in blob
        assert sentinel.encode("utf-8") not in blob

    # The completed turn is a plain TurnFrame — no done-shaped frame exists.
    turns = bus.by_type(TurnFrame)
    assert len(turns) == 1
    statuses = bus.by_type(StatusFrame)
    assert all(s.state in {"working", "input_required", "blocked"} for _subj, s in statuses)


async def test_turn_without_pending_task_is_dropped(tmp_path: Path) -> None:
    _engine, runner, bus = _make(tmp_path)

    await runner._on_engine_event(
        TurnDone(text="orphan", stance=Stance(position="answered", because="x"))
    )
    assert bus.by_type(TurnFrame) == []


# --- completion contract: in-flight tracking + no-steer completion (issue #51) --


def _make_supervisor_and_runner(
    engine: FakeEngine | ScriptedEngine, tmp_path: Path
) -> tuple[Supervisor, FrameTurnRunner, FakeFrameBus]:
    """Build a supervisor + runner over a fake bus for the given engine."""
    sup = Supervisor(
        session=engine,
        config=SupervisorConfig(agent=_AGENT, env="dev", stall_timeout_s=100.0),
        home=tmp_path,
        publisher=RecordingPublisher(),
        system_prompt="SYS",
        tools_policy=ToolsPolicy(),
    )
    bus = FakeFrameBus()
    runner = FrameTurnRunner(agent_id=_AGENT, supervisor=sup, bus=bus)
    return sup, runner, bus


async def test_task_frame_tracks_in_flight_task_until_done(tmp_path: Path) -> None:
    """The supervisor (thus the watchdog) sees the active task DURING the turn."""
    _engine, runner, bus = _make(tmp_path)
    sup = runner._supervisor
    await runner.start()
    task = _task(task_id="T-track")

    # Delivering the task registers it as in-flight — never "no task" mid-turn.
    await bus.deliver(subjects.agent_task(_AGENT), task)
    assert sup.in_flight_task_id == task.id

    # The terminal TurnDone clears it (via the supervisor's own event handling).
    await sup._handle_event(TurnDone(text="done", session_ref="ref"))
    assert sup.in_flight_task_id is None


async def test_failed_send_retires_in_flight_tracking(tmp_path: Path) -> None:
    """A send that never opened a turn rolls back tracking + pending state."""
    engine, runner, bus = _make(tmp_path)
    sup = runner._supervisor
    await runner.start()

    async def _boom(_text: str) -> None:
        raise RuntimeError("send blew up")

    engine.send = _boom  # type: ignore[method-assign]
    task = _task(task_id="T-fail")
    await bus.deliver(subjects.agent_task(_AGENT), task)

    assert sup.in_flight_task_id is None
    # A later orphan TurnDone must NOT be mis-correlated to the failed task.
    await runner._on_engine_event(TurnDone(text="orphan"))
    assert bus.by_type(TurnFrame) == []


async def test_scripted_task_completes_without_steer(tmp_path: Path) -> None:
    """Regression pin (#51): a scripted task send drives a TurnFrame, NO steer.

    The scripted engine completes its own turn — no ``complete_open_turn`` flush,
    no SteerFrame. Task in -> StatusFrame(working) -> TurnFrame out, end to end
    through the supervisor pump, on a fake bus.
    """
    engine = ScriptedEngine()
    sup, runner, bus = _make_supervisor_and_runner(engine, tmp_path)

    await sup.start()
    run_task = asyncio.create_task(sup.run(), name="contract-supervisor-run")
    await runner.start()

    task = _task(task_id="T-scripted")
    try:
        await bus.deliver(subjects.agent_task(_AGENT), task)

        turn_subj = subjects.task_turn(task.id)
        deadline = asyncio.get_running_loop().time() + 2.0
        while asyncio.get_running_loop().time() < deadline:
            if any(isinstance(f, TurnFrame) for f in bus.frames_on(turn_subj)):
                break
            await asyncio.sleep(0.01)
    finally:
        await sup.stop()
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass

    # Exactly one TurnFrame, and it was reached with zero SteerFrames on the wire.
    turns = [f for f in bus.frames_on(subjects.task_turn(task.id)) if isinstance(f, TurnFrame)]
    assert len(turns) == 1
    assert turns[0].task_id == task.id
    assert turns[0].visible_text
    # The turn completed with zero SteerFrames on the wire (no steer flush).
    assert bus.by_type(SteerFrame) == []


# --- steer target contract: broadcast default honored, ignores are loud (#74) ---


def _steer(task_id: str, target: str, text: str = "steer text") -> SteerFrame:
    return SteerFrame(task_id=task_id, text=text, target_agent=target)


async def test_steer_broadcast_target_all_reaches_active_agent(tmp_path: Path) -> None:
    """A steer with the edge-default target "all" is injected, not dropped (#74).

    The edge fills ``target_agent="all"`` when the human names no agent; agentd
    honors it as a first-class broadcast target for its active task, and the
    steer lands in the frame-graph refs of the eventual TurnFrame.
    """
    engine, runner, bus = _make(tmp_path)
    await runner.start()
    task = _task(task_id="T-broadcast")
    await bus.deliver(subjects.agent_task(_AGENT), task)

    steer = _steer(task.id, "all")
    await bus.deliver(subjects.task_steer(task.id), steer)
    assert engine.sends[-1].text == steer.text

    await runner._on_engine_event(
        TurnDone(text="done", stance=Stance(position="answered", because="x"))
    )
    turns = [f for f in bus.frames_on(subjects.task_turn(task.id)) if isinstance(f, TurnFrame)]
    assert len(turns) == 1
    assert turns[0].refs.count(task.id) == 2  # task ref + accepted steer ref


async def test_steer_explicit_self_target_still_accepted(tmp_path: Path) -> None:
    engine, runner, bus = _make(tmp_path)
    await runner.start()
    task = _task(task_id="T-explicit")
    await bus.deliver(subjects.agent_task(_AGENT), task)

    steer = _steer(task.id, _AGENT)
    await bus.deliver(subjects.task_steer(task.id), steer)
    assert engine.sends[-1].text == steer.text


async def test_steer_wrong_explicit_target_is_ignored_loudly(tmp_path: Path) -> None:
    """A steer naming a different agent is NOT injected and the ignore is loud.

    No silent drops (#74): the guard trigger emits a warning carrying the typed
    context (target, self, task_id), never a bare return.
    """
    from loguru import logger

    engine, runner, bus = _make(tmp_path)
    await runner.start()
    task = _task(task_id="T-wrong")
    await bus.deliver(subjects.agent_task(_AGENT), task)
    sends_before = len(engine.sends)

    records: list[str] = []
    sink_id = logger.add(records.append, level="WARNING")
    try:
        await bus.deliver(subjects.task_steer(task.id), _steer(task.id, "someone-else"))
    finally:
        logger.remove(sink_id)

    assert len(engine.sends) == sends_before  # never reached the session
    loud = [r for r in records if "steer IGNORED" in r]
    assert loud, "expected a loud warning for the ignored steer"
    assert "someone-else" in loud[0]
    assert task.id in loud[0]


async def test_steer_broadcast_for_non_active_task_is_ignored_loudly(tmp_path: Path) -> None:
    """The non-active ignore is loud WITH the full typed context (#74/#76).

    The lane handler is bound to its task id, so the non-active branch is
    exercised directly: a lane-matching payload whose task is no longer active
    logs target, self, AND task_id — operators can identify which daemon
    rejected a broadcast, not just which task.
    """
    from loguru import logger

    engine, runner, bus = _make(tmp_path)
    await runner.start()
    task = _task(task_id="T-active")
    await bus.deliver(subjects.agent_task(_AGENT), task)
    sends_before = len(engine.sends)

    records: list[str] = []
    sink_id = logger.add(records.append, level="WARNING")
    try:
        await runner._on_steer_frame("T-not-active", _steer("T-not-active", "all"))
    finally:
        logger.remove(sink_id)

    assert len(engine.sends) == sends_before
    loud = [r for r in records if "non-active task" in r]
    assert loud, "expected a loud warning for the non-active ignore"
    assert "target=all" in loud[0]
    assert f"self={_AGENT}" in loud[0]
    assert "task_id=T-not-active" in loud[0]


# --- steer lane live semantics: bound handlers, retired subscriptions (#76) -----


async def test_steer_lane_payload_task_mismatch_is_rejected_loudly(tmp_path: Path) -> None:
    """A payload cannot ride a foreign lane: lane/payload mismatch is rejected.

    Two tasks are active; a frame published on task A's steer lane carrying
    ``task_id=B`` (the stale-lane spoof from the #76 review) must NOT be
    injected, and the ignore logs the full typed context including the lane.
    """
    from loguru import logger

    engine, runner, bus = _make(tmp_path)
    await runner.start()
    task_a = _task(task_id="T-lane-a")
    task_b = _task(task_id="T-lane-b")
    await bus.deliver(subjects.agent_task(_AGENT), task_a)
    await bus.deliver(subjects.agent_task(_AGENT), task_b)
    sends_before = len(engine.sends)

    records: list[str] = []
    sink_id = logger.add(records.append, level="WARNING")
    try:
        await bus.deliver(subjects.task_steer(task_a.id), _steer(task_b.id, "all"))
    finally:
        logger.remove(sink_id)

    assert len(engine.sends) == sends_before  # never reached the session
    loud = [r for r in records if "lane/payload task mismatch" in r]
    assert loud, "expected a loud warning for the lane/payload mismatch"
    assert "target=all" in loud[0]
    assert f"self={_AGENT}" in loud[0]
    assert f"task_id={task_b.id}" in loud[0]
    assert f"lane_task_id={task_a.id}" in loud[0]


async def test_completed_task_steer_lane_is_unsubscribed(tmp_path: Path) -> None:
    """Completion retires the LIVE lane subscription, not just guard state (#76)."""
    _engine, runner, bus = _make(tmp_path)
    await runner.start()
    task = _task(task_id="T-retire")
    await bus.deliver(subjects.agent_task(_AGENT), task)
    assert len(bus.handlers_on(subjects.task_steer(task.id))) == 1

    await runner._on_engine_event(
        TurnDone(text="done", stance=Stance(position="answered", because="x"))
    )
    assert bus.handlers_on(subjects.task_steer(task.id)) == []


async def test_duplicate_task_while_active_is_a_no_op(tmp_path: Path) -> None:
    """An active task id is dispatched and marked working exactly once."""
    engine, runner, bus = _make(tmp_path)
    await runner.start()
    task = _task(task_id="T-active-duplicate")

    await bus.deliver(subjects.agent_task(_AGENT), task)
    await bus.deliver(subjects.agent_task(_AGENT), task)

    assert [send.text for send in engine.sends] == [task.text]
    assert len(bus.handlers_on(subjects.task_steer(task.id))) == 1
    working = [
        frame
        for frame in bus.frames_on(subjects.task_status(task.id))
        if isinstance(frame, StatusFrame) and frame.state == "working"
    ]
    assert len(working) == 1


async def test_duplicate_task_after_completion_is_a_no_op(tmp_path: Path) -> None:
    """A completed task id is not redispatched or marked working again."""
    engine, runner, bus = _make(tmp_path)
    await runner.start()
    task = _task(task_id="T-completed-duplicate")

    await bus.deliver(subjects.agent_task(_AGENT), task)
    await runner._on_engine_event(
        TurnDone(text="round one", stance=Stance(position="answered", because="x"))
    )
    await bus.deliver(subjects.agent_task(_AGENT), task)

    assert [send.text for send in engine.sends] == [task.text]
    assert bus.handlers_on(subjects.task_steer(task.id)) == []
    working = [
        frame
        for frame in bus.frames_on(subjects.task_status(task.id))
        if isinstance(frame, StatusFrame) and frame.state == "working"
    ]
    assert len(working) == 1


async def test_distinct_task_ids_are_dispatched_independently(tmp_path: Path) -> None:
    """Deduplication for one task id does not suppress another task id."""
    engine, runner, bus = _make(tmp_path)
    await runner.start()
    first = _task(task_id="T-distinct-1", text="first")
    second = _task(task_id="T-distinct-2", text="second")

    await bus.deliver(subjects.agent_task(_AGENT), first)
    await bus.deliver(subjects.agent_task(_AGENT), second)

    assert [send.text for send in engine.sends] == [first.text, second.text]


async def test_completed_task_survives_count_bound_within_replay_horizon(tmp_path: Path) -> None:
    """The verifier's 1,024-intervening-completion replay remains a no-op."""
    engine, runner, bus = _make(tmp_path)
    await runner.start()
    original = _task(task_id="T-retained", text="original")

    await bus.deliver(subjects.agent_task(_AGENT), original)
    await runner._on_engine_event(TurnDone(text="done"))
    for index in range(COMPLETED_TASK_CACHE_SIZE):
        task = _task(task_id=f"T-intervening-{index}", text=f"task {index}")
        await bus.deliver(subjects.agent_task(_AGENT), task)
        await runner._on_engine_event(TurnDone(text="done"))

    sends_before_replay = len(engine.sends)
    await bus.deliver(subjects.agent_task(_AGENT), original)

    assert len(engine.sends) == sends_before_replay
    assert len(runner._completed_tasks) == COMPLETED_TASK_CACHE_SIZE + 1


async def test_stale_untracked_task_fails_closed_without_side_effects(tmp_path: Path) -> None:
    """A task older than the configured retry horizon never reaches the engine."""
    engine = FakeEngine()
    sup = Supervisor(
        session=engine,
        config=SupervisorConfig(agent=_AGENT, env="dev"),
        home=tmp_path,
        publisher=RecordingPublisher(),
        system_prompt="SYS",
        tools_policy=ToolsPolicy(),
    )
    bus = FakeFrameBus()
    runner = FrameTurnRunner(
        agent_id=_AGENT,
        supervisor=sup,
        bus=bus,
        config=FrameTurnConfig(task_replay_horizon_s=60.0),
    )
    await runner.start()
    stale = _task(task_id="T-stale").model_copy(
        update={"created_at": datetime.now(UTC) - timedelta(seconds=61)}
    )

    await bus.deliver(subjects.agent_task(_AGENT), stale)

    assert engine.sends == []
    assert bus.frames_on(subjects.task_status(stale.id)) == []
    assert bus.handlers_on(subjects.task_steer(stale.id)) == []
    assert list(runner._pending) == []
    assert runner._supervisor.in_flight_task_id is None
    assert runner._steer_refs == {}
    assert runner._persisted_completions == {}


async def test_fresh_timestamped_distinct_task_dispatches(tmp_path: Path) -> None:
    """The stale guard leaves fresh, distinct timestamped tasks unaffected."""
    engine, runner, bus = _make(tmp_path)
    await runner.start()
    fresh = _task(task_id="T-fresh").model_copy(update={"created_at": datetime.now(UTC)})

    await bus.deliver(subjects.agent_task(_AGENT), fresh)

    assert [send.text for send in engine.sends] == [fresh.text]


async def test_untimestamped_untracked_task_fails_closed(tmp_path: Path) -> None:
    """An untracked frame with no ``created_at`` cannot be proven within-horizon.

    It is indistinguishable from an evicted, arbitrarily old completed replay,
    so it must FAIL CLOSED (issue #83) rather than fail open and risk a
    double-dispatch. This is the symmetric half of the dedup cache: within-horizon
    completed replays are deduped by the cache; anything the cache cannot vouch
    for and cannot prove within-horizon is refused before any side effect.
    """
    engine, runner, bus = _make(tmp_path)
    await runner.start()
    # A legacy/untimestamped frame (created_at=None) is the pre-#83 fail-open branch.
    untimestamped = _task(task_id="T-no-ts").model_copy(update={"created_at": None})
    assert untimestamped.created_at is None

    await bus.deliver(subjects.agent_task(_AGENT), untimestamped)

    assert engine.sends == []
    assert bus.frames_on(subjects.task_status(untimestamped.id)) == []
    assert bus.handlers_on(subjects.task_steer(untimestamped.id)) == []
    assert list(runner._pending) == []
    # A regression that mutated supervisor tracking, the active-task marker, or
    # the persisted dedup state on the refuse path would otherwise stay green
    # (issue #83 review). Pin them untouched.
    assert runner._supervisor.in_flight_task_id is None
    assert runner._steer_refs == {}
    assert runner._persisted_completions == {}


async def test_within_horizon_replay_survives_eviction_pressure(tmp_path: Path) -> None:
    """A within-horizon completed replay is NEVER re-dispatched, regardless of
    how many completions intervene — the pinned issue #83 invariant.

    A long horizon means the completed key is retained past the soft count
    bound (the cache grows loudly rather than under-retain), so replaying the
    exact completed frame after well over the count bound of intervening
    completions is a no-op. Timestamps stay fresh so the stale guard never fires.
    """
    engine = FakeEngine()
    sup = Supervisor(
        session=engine,
        config=SupervisorConfig(agent=_AGENT, env="dev"),
        home=tmp_path,
        publisher=RecordingPublisher(),
        system_prompt="SYS",
        tools_policy=ToolsPolicy(),
    )
    bus = FakeFrameBus()
    runner = FrameTurnRunner(
        agent_id=_AGENT,
        supervisor=sup,
        bus=bus,
        config=FrameTurnConfig(task_replay_horizon_s=3600.0),
    )
    await runner.start()
    now = datetime.now(UTC)
    original = _task(task_id="T-horizon", text="original").model_copy(update={"created_at": now})

    await bus.deliver(subjects.agent_task(_AGENT), original)
    await runner._on_engine_event(TurnDone(text="done"))
    for index in range(COMPLETED_TASK_CACHE_SIZE + 5):
        task = _task(task_id=f"T-fill-{index}", text=f"task {index}").model_copy(
            update={"created_at": now}
        )
        await bus.deliver(subjects.agent_task(_AGENT), task)
        await runner._on_engine_event(TurnDone(text="done"))

    sends_before_replay = len(engine.sends)
    await bus.deliver(subjects.agent_task(_AGENT), original)

    assert len(engine.sends) == sends_before_replay
    # The key survived: retention grew past the soft count bound to honor the horizon.
    assert (original.id, _AGENT) in runner._completed_tasks


async def test_failed_send_retires_steer_lane_subscription(tmp_path: Path) -> None:
    """The failed-send rollback also retires the live lane subscription (#76)."""
    engine, runner, bus = _make(tmp_path)
    await runner.start()

    async def _boom(_text: str) -> None:
        raise RuntimeError("send blew up")

    engine.send = _boom  # type: ignore[method-assign]
    task = _task(task_id="T-fail-lane")
    await bus.deliver(subjects.agent_task(_AGENT), task)
    assert bus.handlers_on(subjects.task_steer(task.id)) == []


def test_broadcast_target_is_the_frames_owned_constant() -> None:
    """agentd consumes the rtech-frames broadcast token, never its own literal (#76)."""
    from rtech_agentd.frame_turn import BROADCAST_TARGET as agentd_broadcast
    from rtech_frames import BROADCAST_TARGET as frames_broadcast

    assert agentd_broadcast is frames_broadcast


# --- steer lane over the subscribe_all primitive (issue #80) -------------------


async def test_steer_lane_installed_through_subscription_set(tmp_path: Path) -> None:
    """(#80) The steer lane is held as a ``SubscriptionSet``, not a raw callable.

    The atomic primitive replaces the hand-rolled per-task unsubscribe handle:
    an active task carries exactly one live-set handle, and completion retires
    it through the set (the lane stops delivering).
    """
    from rtech_frames.subs import SubscriptionSet

    _engine, runner, bus = _make(tmp_path)
    await runner.start()
    task = _task(task_id="T-set")
    await bus.deliver(subjects.agent_task(_AGENT), task)

    held = runner._steer_subscriptions[task.id]
    assert isinstance(held, SubscriptionSet)
    assert len(held) == 1
    assert subjects.task_steer(task.id) in held.subjects

    await runner._on_engine_event(
        TurnDone(text="done", stance=Stance(position="answered", because="x"))
    )
    assert bus.handlers_on(subjects.task_steer(task.id)) == []
    assert task.id not in runner._steer_subscriptions


async def test_steer_subscribe_failure_fails_lane_without_leaking(tmp_path: Path) -> None:
    """(#80) A steer-lane subscribe failure leaves NOTHING live-but-untracked.

    A single-subject set that fails to subscribe has nothing already made, so
    ``subscribe_all`` rolls back to an empty set: no handler is registered, the
    reserved guard slot is released, and no leaked handle is held. The task
    itself still runs (a missing steer lane is non-fatal, as before).
    """

    class SteerFailBus(FakeFrameBus):
        async def subscribe_frame(self, subject: str, handler: object) -> object:
            if subject == subjects.task_steer("T-steer-fail"):
                raise RuntimeError("steer subscribe blew up")
            return await super().subscribe_frame(subject, handler)  # type: ignore[arg-type]

    engine = FakeEngine()
    sup = Supervisor(
        session=engine,
        config=SupervisorConfig(agent=_AGENT, env="dev"),
        home=tmp_path,
        publisher=RecordingPublisher(),
        system_prompt="SYS",
        tools_policy=ToolsPolicy(),
    )
    bus = SteerFailBus()
    runner = FrameTurnRunner(agent_id=_AGENT, supervisor=sup, bus=bus)
    await runner.start()
    task = _task(task_id="T-steer-fail")

    await bus.deliver(subjects.agent_task(_AGENT), task)

    # Lane failed closed: no handler registered, no reserved slot, no leak held.
    assert bus.handlers_on(subjects.task_steer(task.id)) == []
    assert task.id not in runner._steer_subscriptions
    assert task.id not in runner._leaked_steer_subscriptions
    # The task still dispatched to the engine (steer lane is non-fatal).
    assert [send.text for send in engine.sends] == [task.text]


# --- sibling turn fanout: the real inter-agent debate (issue #81) --------------


def _runner_for(agent_id: str, tmp_path: Path) -> tuple[FakeEngine, FrameTurnRunner, FakeFrameBus]:
    """Build a runner whose participant identity is ``agent_id``."""
    engine = FakeEngine()
    sup = Supervisor(
        session=engine,
        config=SupervisorConfig(agent=agent_id, env="dev"),
        home=tmp_path,
        publisher=RecordingPublisher(),
        system_prompt="SYS",
        tools_policy=ToolsPolicy(),
    )
    bus = FakeFrameBus()
    runner = FrameTurnRunner(agent_id=agent_id, supervisor=sup, bus=bus)
    return engine, runner, bus


def _duo_task(participants: list[str], task_id: str = "T-duo") -> TaskFrame:
    return TaskFrame(
        id=task_id,
        origin_edge="web",
        requester="rico",
        text="debate it",
        scope="duo",
        participants=participants,
        created_at=datetime.now(UTC),
    )


def _sibling_turn(author: str, task_id: str, needs: list[str] | None = None) -> TurnFrame:
    return TurnFrame(
        agent_id=author,
        task_id=task_id,
        visible_text=f"{author} position",
        stance=Stance(position="proposed", because="reasoned", needs_from=needs or []),
    )


async def test_participant_receives_sibling_turn(tmp_path: Path) -> None:
    """(a) A co-participant's TurnFrame on the shared lane reaches the session.

    A duo task subscribes this agent to the shared per-task turn lane; a sibling
    turn is injected (visible text + typed stance), while this agent's OWN turn
    echo on the same lane is never re-injected.
    """
    engine, runner, bus = _runner_for("alice", tmp_path)
    await runner.start()
    task = _duo_task(["alice", "bob"])
    await bus.deliver(subjects.agent_task("alice"), task)
    # The fanout lane is live for the duo task.
    assert len(bus.handlers_on(subjects.task_turn(task.id))) == 1
    sends_before = len(engine.sends)

    await bus.deliver(subjects.task_turn(task.id), _sibling_turn("bob", task.id))
    assert len(engine.sends) == sends_before + 1
    injected = engine.sends[-1].text
    assert "bob" in injected
    assert "bob position" in injected

    # This agent's own turn echoing back on the shared lane is NOT re-injected.
    await bus.deliver(subjects.task_turn(task.id), _sibling_turn("alice", task.id))
    assert len(engine.sends) == sends_before + 1


async def test_single_scope_task_has_no_sibling_lane(tmp_path: Path) -> None:
    """A single-participant task never subscribes the sibling fanout lane."""
    _engine, runner, bus = _runner_for("alice", tmp_path)
    await runner.start()
    task = _task(task_id="T-solo")  # scope=single, no co-participants
    await bus.deliver(subjects.agent_task("alice"), task)
    assert bus.handlers_on(subjects.task_turn(task.id)) == []
    assert task.id not in runner._sibling_subscriptions


async def test_second_round_references_a_real_sibling(tmp_path: Path) -> None:
    """(b) A sibling turn re-opens a round whose stance references the real sibling.

    After alice finishes round one, bob's real position arrives on the shared
    lane and re-opens alice's turn; alice's round-two TurnFrame names bob in
    ``needs_from`` — a real sibling dependency, not a role-played one.
    """
    engine, runner, bus = _runner_for("alice", tmp_path)
    await runner.start()
    task = _duo_task(["alice", "bob"])
    await bus.deliver(subjects.agent_task("alice"), task)

    # Round one: alice opens the debate still needing bob (non-terminal), so the
    # debate has NOT settled and the fanout lane stays live for bob's turn.
    await runner._on_engine_event(
        TurnDone(
            text="alice round one",
            stance=Stance(position="opened", because="kickoff", needs_from=["bob"]),
        )
    )
    round_one = [f for f in bus.frames_on(subjects.task_turn(task.id)) if isinstance(f, TurnFrame)]
    assert len(round_one) == 1

    # Bob's real position arrives, still needing alice (the debate is live, not
    # settled) -> alice re-engages a follow-up round.
    await bus.deliver(subjects.task_turn(task.id), _sibling_turn("bob", task.id, needs=["alice"]))
    assert "bob" in engine.sends[-1].text

    # Round two: alice's stance now names the REAL sibling in needs_from.
    await runner._on_engine_event(
        TurnDone(
            text="alice round two",
            stance=Stance(position="revised", because="bob raised X", needs_from=["bob"]),
        )
    )
    turns = [f for f in bus.frames_on(subjects.task_turn(task.id)) if isinstance(f, TurnFrame)]
    assert len(turns) == 2
    assert turns[1].agent_id == "alice"
    assert turns[1].stance.needs_from == ["bob"]


async def test_fanout_done_derivation_terminates(tmp_path: Path) -> None:
    """(c) Done derives correctly over a multi-round fanned debate; no hang/double-count.

    The frame graph carries two rounds per participant. While either agent's
    latest turn is non-terminal (``needs_from`` populated) the task is NOT done;
    once every participant's LATEST turn is terminal it derives done exactly once
    — the fanout adds rounds without breaking the orchestrator's derivation.
    """
    from rtech_frames import is_done

    task = _duo_task(["alice", "bob"])
    participants = frozenset({"alice", "bob"})

    # Round one: each references the other -> both non-terminal -> not done.
    alice_r1 = _sibling_turn("alice", task.id, needs=["bob"])
    bob_r1 = _sibling_turn("bob", task.id, needs=["alice"])
    graph: list[TurnFrame] = [alice_r1, bob_r1]
    assert not is_done(task, graph, participants=participants)

    # Alice resolves (terminal), bob still outstanding -> still not done.
    alice_r2 = _sibling_turn("alice", task.id, needs=[])
    graph.append(alice_r2)
    assert not is_done(task, graph, participants=participants)

    # Bob resolves -> every latest turn terminal -> done derives (once, no hang).
    bob_r2 = _sibling_turn("bob", task.id, needs=[])
    graph.append(bob_r2)
    assert is_done(task, graph, participants=participants)


async def test_sibling_lane_outlives_a_single_turn_done(tmp_path: Path) -> None:
    """The fanout lane survives INDIVIDUAL TurnDones only until the budget exhausts.

    A co-participant can publish after this agent's round completes; the lane
    must remain live to deliver the NEXT sibling (whoever-finishes-first must not
    go deaf mid-debate) — but the debate is STRUCTURED BOUNDED ROUNDS (#87), so
    the lane is retired the moment this agent has opened its cap of
    sibling-triggered follow-ups, not left live forever.
    """
    engine, runner, bus = _runner_for("alice", tmp_path)
    await runner.start()
    task = _duo_task(["alice", "bob"])
    await bus.deliver(subjects.agent_task("alice"), task)

    # A TurnDone alone (no sibling triggered it) does not spend the budget and
    # does not retire the lane: a late sibling must still land. Alice still needs
    # bob (non-terminal), so the debate is unsettled and the lane stays live.
    await runner._on_engine_event(
        TurnDone(
            text="alice done", stance=Stance(position="answered", because="x", needs_from=["bob"])
        )
    )
    assert len(bus.handlers_on(subjects.task_turn(task.id))) == 1
    assert isinstance(runner._sibling_subscriptions.get(task.id), SubscriptionSet)

    # Each sibling turn opens one follow-up; the lane stays live until the last
    # of the budgeted follow-ups is opened, then it is retired. Both sides stay
    # non-terminal (still needing each other) so the debate runs to the budget
    # cap rather than settling below it.
    for _round in range(MAX_SIBLING_ROUNDS):
        assert len(bus.handlers_on(subjects.task_turn(task.id))) == 1
        await bus.deliver(
            subjects.task_turn(task.id), _sibling_turn("bob", task.id, needs=["alice"])
        )
        await runner._on_engine_event(
            TurnDone(
                text="alice follow-up",
                stance=Stance(position="revised", because="bob", needs_from=["bob"]),
            )
        )
    # Budget exhausted -> lane retired, no longer outliving the bounded debate.
    assert bus.handlers_on(subjects.task_turn(task.id)) == []
    assert task.id not in runner._sibling_subscriptions


async def test_failed_send_retires_sibling_lane(tmp_path: Path) -> None:
    """A send that never opened a turn rolls back the sibling lane too (no leak)."""
    engine, runner, bus = _runner_for("alice", tmp_path)
    await runner.start()

    async def _boom(_text: str) -> None:
        raise RuntimeError("send blew up")

    engine.send = _boom  # type: ignore[method-assign]
    task = _duo_task(["alice", "bob"], task_id="T-duo-fail")
    await bus.deliver(subjects.agent_task("alice"), task)

    assert bus.handlers_on(subjects.task_turn(task.id)) == []
    assert task.id not in runner._sibling_subscriptions


async def test_steer_lane_retire_failure_is_held_not_discarded(tmp_path: Path) -> None:
    """(#80) A lane whose unsubscribe fails on completion is HELD, never dropped.

    ``SubscriptionSet.retire()`` retains a handle whose unsubscribe raises and
    reports incomplete; the runner must stash the still-live set for retry in
    ``_leaked_steer_subscriptions`` rather than let the lane escape untracked.
    """

    class UnretirableBus(FakeFrameBus):
        async def subscribe_frame(self, subject: str, handler: object) -> object:
            self.subscribed.setdefault(subject, []).append(handler)  # type: ignore[arg-type]

            async def _boom_unsubscribe() -> None:
                raise RuntimeError("unsubscribe blew up")

            return _boom_unsubscribe

    engine = FakeEngine()
    sup = Supervisor(
        session=engine,
        config=SupervisorConfig(agent=_AGENT, env="dev"),
        home=tmp_path,
        publisher=RecordingPublisher(),
        system_prompt="SYS",
        tools_policy=ToolsPolicy(),
    )
    bus = UnretirableBus()
    runner = FrameTurnRunner(agent_id=_AGENT, supervisor=sup, bus=bus)
    await runner.start()
    task = _task(task_id="T-unretirable")
    await bus.deliver(subjects.agent_task(_AGENT), task)

    await runner._on_engine_event(
        TurnDone(text="done", stance=Stance(position="answered", because="x"))
    )

    # The un-retirable lane is held for retry, never silently discarded.
    leaked = runner._leaked_steer_subscriptions.get(task.id)
    assert leaked is not None
    assert subjects.task_steer(task.id) in leaked.subjects
    assert task.id not in runner._steer_subscriptions


# --- bounded sibling rounds over a DELIVERING bus (issue #87 defect fixes) ------
#
# The plain FakeFrameBus records publishes but never loops them back to
# subscribed handlers, so the self-reinforcing sibling loop and the lane leak
# stayed invisible (all green). ``DeliveringFrameBus`` closes that gap: a
# published TurnFrame is fanned to every still-subscribed handler on its
# subject, exactly like live NATS. Each subscribed runner's ``_on_sibling_turn``
# then feeds its live session and opens a follow-up round, so an unbounded
# design would spin forever — the tests below pin that it does NOT.


class DeliveringFrameBus(FakeFrameBus):
    """A FakeFrameBus that loops published frames back to subscribed handlers.

    Mirrors live NATS: publishing on a subject delivers to every handler still
    subscribed to it. Deliveries are drained via an internal queue (not nested
    recursion) so a handler that publishes mid-delivery does not blow the stack;
    a runaway loop instead trips the ``max_deliveries`` fuse and fails loudly,
    turning "unbounded fanout" from a hang into a red test.
    """

    def __init__(self, max_deliveries: int = 200) -> None:
        super().__init__()
        self._pending: list[tuple[str, Frame]] = []
        self._draining = False
        self.delivery_count = 0
        self._max_deliveries = max_deliveries

    async def publish_frame(self, subject: str, frame: Frame) -> None:
        super_published = self.published
        super_published.append((subject, frame))
        self._pending.append((subject, frame))
        if self._draining:
            return
        self._draining = True
        try:
            while self._pending:
                subj, fr = self._pending.pop(0)
                self.delivery_count += 1
                if self.delivery_count > self._max_deliveries:
                    raise AssertionError(
                        "DeliveringFrameBus fuse tripped: unbounded fanout "
                        f"(> {self._max_deliveries} deliveries)"
                    )
                for handler in list(self.subscribed.get(subj, [])):
                    await handler(fr)
        finally:
            self._draining = False


class AutoRespondEngine(FakeEngine):
    """A FakeEngine that publishes a real follow-up TurnFrame after each send.

    Wired to its owning runner, every ``send`` drives one ``_on_engine_event``
    TurnDone — so an injected sibling turn produces a published follow-up
    TurnFrame on the shared lane, closing the loop the DeliveringFrameBus fans.
    The FIRST send (the initial task dispatch) is skipped so the debate is opened
    explicitly by the test, matching how a real turn completes once per prompt.
    """

    def __init__(self) -> None:
        super().__init__()
        self.runner: FrameTurnRunner | None = None
        self._seen_first = False
        self.round_no = 0

    async def send(self, text: str) -> None:
        await super().send(text)
        if self.runner is None:
            return
        if not self._seen_first:
            # The initial task dispatch opens the first turn; the test emits its
            # TurnDone explicitly, so do not auto-fire for it.
            self._seen_first = True
            return
        self.round_no += 1
        await self.runner._on_engine_event(
            TurnDone(
                text=f"{self.runner._agent_id} round {self.round_no}",
                stance=Stance(position="revised", because="sibling raised a point"),
            )
        )


def _wire_auto_runner(
    agent_id: str, bus: FakeFrameBus, tmp_path: Path
) -> tuple[AutoRespondEngine, FrameTurnRunner]:
    """Build a runner whose engine auto-publishes a follow-up per injected turn."""
    engine = AutoRespondEngine()
    sup = Supervisor(
        session=engine,
        config=SupervisorConfig(agent=agent_id, env="dev"),
        home=tmp_path,
        publisher=RecordingPublisher(),
        system_prompt="SYS",
        tools_policy=ToolsPolicy(),
    )
    runner = FrameTurnRunner(agent_id=agent_id, supervisor=sup, bus=bus)
    engine.runner = runner
    return engine, runner


async def test_three_participant_debate_terminates_at_the_round_budget(tmp_path: Path) -> None:
    """(a) A 3-participant fanned debate TERMINATES: each agent opens exactly
    MAX_SIBLING_ROUNDS sibling-triggered follow-ups, then further sibling turns
    are dropped — the loop does not run forever.

    Over the delivering bus every follow-up turn is published and fanned to the
    other participants, whose follow-ups fan back — a self-reinforcing loop. An
    unbounded design would trip the bus fuse (a hang turned red); the bounded
    design settles once each agent has spent exactly its budget.
    """
    bus = DeliveringFrameBus()
    runners: dict[str, FrameTurnRunner] = {}
    for agent_id in ("alice", "bob", "carol"):
        _engine, runner = _wire_auto_runner(agent_id, bus, tmp_path)
        runners[agent_id] = runner
        await runner.start()

    task = TaskFrame(
        id="T-trio",
        origin_edge="web",
        requester="rico",
        text="debate it",
        scope="trio",
        participants=["alice", "bob", "carol"],
        created_at=datetime.now(UTC),
    )
    for agent_id in runners:
        await bus.deliver(subjects.agent_task(agent_id), task)

    # Alice opens the debate; her published turn fans to the others and the
    # self-reinforcing cascade drains to quiescence (no fuse trip == terminated).
    await runners["alice"]._on_engine_event(
        TurnDone(text="alice opens", stance=Stance(position="opened", because="kickoff"))
    )

    # Every agent opened EXACTLY its budget of sibling-triggered follow-ups —
    # no more (bounded), and every participant actually engaged the debate.
    for agent_id, runner in runners.items():
        # The count is cleared when the lane retires at the cap, so a cleared
        # (absent) count is the terminal state; presence would mean it stalled
        # below the cap, which for this fully-connected trio must not happen.
        assert task.id not in runner._sibling_round_counts, (
            f"{agent_id} did not reach the cap / clear at retirement"
        )
        # The lane is retired for every participant once its budget is spent.
        assert task.id not in runner._sibling_subscriptions, (
            f"{agent_id} lane outlived the bounded debate"
        )

    # Total published follow-up turns per agent is capped at MAX_SIBLING_ROUNDS:
    # alice's opener plus at most one follow-up per budgeted round each.
    turns = [f for _s, f in bus.by_type(TurnFrame)]
    per_author: dict[str, int] = {}
    for turn in turns:
        per_author[turn.agent_id] = per_author.get(turn.agent_id, 0) + 1
    # bob and carol publish only sibling-triggered follow-ups (<= cap); alice
    # publishes her opener plus follow-ups (<= cap + 1).
    assert 1 <= per_author.get("bob", 0) <= MAX_SIBLING_ROUNDS
    assert 1 <= per_author.get("carol", 0) <= MAX_SIBLING_ROUNDS
    assert 1 <= per_author.get("alice", 0) <= MAX_SIBLING_ROUNDS + 1


async def test_sibling_lane_retires_when_budget_exhausts(tmp_path: Path) -> None:
    """(b) The sibling lane is retired once the round budget is spent — the
    subscription count returns to baseline, so the lane never leaks past the
    bounded debate.
    """
    engine, runner, bus = _runner_for("alice", tmp_path)
    await runner.start()
    task = _duo_task(["alice", "bob"])
    baseline = len(bus.handlers_on(subjects.task_turn(task.id)))  # 0 before the task
    await bus.deliver(subjects.agent_task("alice"), task)
    assert len(bus.handlers_on(subjects.task_turn(task.id))) == baseline + 1

    # Spend all but the last of the budget: the lane stays live and the count
    # climbs one per sibling-triggered follow-up.
    for expected in range(1, MAX_SIBLING_ROUNDS):
        await bus.deliver(subjects.task_turn(task.id), _sibling_turn("bob", task.id))
        assert runner._sibling_round_counts.get(task.id, 0) == expected
        assert len(bus.handlers_on(subjects.task_turn(task.id))) == baseline + 1

    # The final budgeted follow-up exhausts the budget and retires the lane
    # in the same delivery.
    await bus.deliver(subjects.task_turn(task.id), _sibling_turn("bob", task.id))

    # Lane retired -> subscription count back to baseline, no leak, count cleared.
    assert len(bus.handlers_on(subjects.task_turn(task.id))) == baseline
    assert task.id not in runner._sibling_subscriptions
    assert task.id not in runner._sibling_round_counts


async def test_steer_delivered_during_a_sibling_triggered_round(tmp_path: Path) -> None:
    """(c) A human SteerFrame arriving during a sibling-triggered follow-up round
    IS delivered — the re-pend path restores the active-task marker so mid-debate
    steer works exactly as on the first round (not silently dropped).
    """
    engine, runner, bus = _runner_for("alice", tmp_path)
    await runner.start()
    task = _duo_task(["alice", "bob"])
    await bus.deliver(subjects.agent_task("alice"), task)

    # Round one completes still needing bob (non-terminal), so the debate is live,
    # not settled; the first-round active-task marker is torn down at TurnDone.
    await runner._on_engine_event(
        TurnDone(
            text="alice round one",
            stance=Stance(position="opened", because="kickoff", needs_from=["bob"]),
        )
    )
    assert task.id not in runner._steer_refs

    # A sibling turn (still needing alice -> the debate is live) re-opens a
    # follow-up round: the active-task marker is restored so it is steerable.
    await bus.deliver(subjects.task_turn(task.id), _sibling_turn("bob", task.id, needs=["alice"]))
    assert task.id in runner._steer_refs
    sends_before = len(engine.sends)

    # A human steer arriving mid-debate reaches the live session, not the floor.
    steer = _steer(task.id, "all", text="reconsider point Y")
    await bus.deliver(subjects.task_steer(task.id), steer)
    assert len(engine.sends) == sends_before + 1
    assert engine.sends[-1].text == steer.text


# --- adversarial re-review fixes: settle-aware retire, cap terminus, atomic reserve


async def test_duo_debate_settling_below_cap_retires_lane_and_clears_counter(
    tmp_path: Path,
) -> None:
    """(a) A duo debate that SETTLES below the cap retires the sibling lane and
    clears its counter — no lane/counter leak for the common consensus case (#81).

    Alice opens still needing bob (non-terminal); bob's first turn also needs
    alice, so the debate is live and the lane stays. When bob then publishes a
    TERMINAL turn (``needs_from`` empty) AND alice's latest is already terminal,
    every participant's latest observed turn is terminal — the debate has settled
    below the cap, so the lane is retired and the counter cleared without ever
    reaching ``MAX_SIBLING_ROUNDS``.
    """
    engine, runner, bus = _runner_for("alice", tmp_path)
    await runner.start()
    task = _duo_task(["alice", "bob"])
    await bus.deliver(subjects.agent_task("alice"), task)
    assert len(bus.handlers_on(subjects.task_turn(task.id))) == 1

    # Bob's first turn still needs alice (live), and alice re-engages a follow-up.
    await bus.deliver(subjects.task_turn(task.id), _sibling_turn("bob", task.id, needs=["alice"]))
    # Alice publishes a TERMINAL follow-up (consensus reached from her side).
    await runner._on_engine_event(
        TurnDone(text="alice agrees", stance=Stance(position="agreed", because="bob is right"))
    )
    # Lane still live: bob's latest observed turn is not yet terminal.
    assert len(bus.handlers_on(subjects.task_turn(task.id))) == 1
    assert task.id in runner._sibling_round_counts

    # Bob now publishes a TERMINAL turn -> every latest turn terminal -> settled.
    await bus.deliver(subjects.task_turn(task.id), _sibling_turn("bob", task.id, needs=[]))

    # The lane is retired and the counter cleared BELOW the cap — no leak.
    assert bus.handlers_on(subjects.task_turn(task.id)) == []
    assert task.id not in runner._sibling_subscriptions
    assert task.id not in runner._sibling_round_counts
    assert runner._sibling_round_counts.get(task.id, 0) < MAX_SIBLING_ROUNDS


async def test_cap_with_non_terminal_latest_emits_one_closing_terminal_turn(
    tmp_path: Path,
) -> None:
    """(b) Budget exhaustion with a NON-TERMINAL latest turn emits exactly one
    closing terminal TurnFrame, and rtech_frames is_done flips True over the graph.

    Every budgeted follow-up alice publishes still needs bob (non-terminal). Once
    the cap drops all further siblings, alice would never publish a terminal turn
    on her own and orchestrator-derived done would hang False forever. The
    frame-level terminus publishes ONE closing turn — same position/because,
    ``needs_from=[]`` — so done derives True.
    """
    from rtech_frames import is_done

    engine, runner, bus = _runner_for("alice", tmp_path)
    await runner.start()
    task = _duo_task(["alice", "bob"])
    await bus.deliver(subjects.agent_task("alice"), task)

    turn_subj = subjects.task_turn(task.id)
    # Drive every budgeted round: each sibling opens a follow-up, and alice's
    # follow-up stays NON-TERMINAL (still needs bob).
    for _round in range(MAX_SIBLING_ROUNDS):
        await bus.deliver(turn_subj, _sibling_turn("bob", task.id, needs=["alice"]))
        await runner._on_engine_event(
            TurnDone(
                text="alice still disagrees",
                stance=Stance(position="dissent", because="unconvinced", needs_from=["bob"]),
            )
        )

    # Exactly one closing terminal turn was published by alice (needs_from empty).
    alice_terminal = [
        f
        for f in bus.frames_on(turn_subj)
        if isinstance(f, TurnFrame) and f.agent_id == "alice" and not f.stance.needs_from
    ]
    assert len(alice_terminal) == 1
    closing = alice_terminal[0]
    assert closing.stance.needs_from == []
    assert closing.stance.position == "dissent"  # honest: the position that stood
    # Terminality lives in the typed stance; the visible_text REUSES alice's real
    # last substantive answer (issue #87 follow-up), never a synthetic placeholder.
    assert closing.visible_text == "alice still disagrees"
    assert "round budget" not in closing.visible_text.lower()
    assert "stands as final" not in closing.visible_text.lower()

    # A second dropped sibling does NOT emit another closing turn (at most once).
    # The lane is already retired, so drive the handler directly (a real late
    # sibling on the retired lane would simply not be delivered).
    await runner._on_sibling_turn(task.id, _sibling_turn("bob", task.id, needs=["alice"]))
    alice_terminal_after = [
        f
        for f in bus.frames_on(turn_subj)
        if isinstance(f, TurnFrame) and f.agent_id == "alice" and not f.stance.needs_from
    ]
    assert len(alice_terminal_after) == 1

    # Over the resulting graph (alice terminal via the closing turn; bob terminal)
    # rtech_frames.is_done flips True — the cap no longer hangs completion.
    alice_turns = [
        f for f in bus.frames_on(turn_subj) if isinstance(f, TurnFrame) and f.agent_id == "alice"
    ]
    graph: list[TurnFrame] = [*alice_turns, _sibling_turn("bob", task.id, needs=[])]
    assert is_done(task, graph, participants=frozenset({"alice", "bob"}))


async def test_closing_publish_failure_retries_on_next_event_exactly_once(
    tmp_path: Path,
) -> None:
    """D7 repro: a closing-turn publish that RAISES once is retried on the next
    observed turn, and exactly one closing turn is ever published.

    frame_turn used to clear the closing/sibling state (and this task's origin)
    BEFORE ``_publish_closing_turn``; a publish failure was caught and returned
    with no retry, so the capped debate never produced its closing turn and the
    participant stayed non-terminal in the orchestrator graph forever. The fix
    publishes via the CHECKED path and clears the closing-decision state ONLY on a
    confirmed publish; a failure re-arms the emit-if-hung path so the next
    ``_publish_turn`` retries — and the one-shot guard + success clear together
    prevent a double-publish.
    """
    from rtech_frames import is_done

    class ClosingFailsOnceBus(FakeFrameBus):
        """Raise on the FIRST cap-closing publish (a terminal alice turn)."""

        def __init__(self) -> None:
            super().__init__()
            self.closing_attempts = 0

        async def publish_frame_checked(self, subject: str, frame: Frame) -> None:
            # The cap-closing turn is the only terminal alice turn that rides the
            # CHECKED publish path (issue #87 D7); detect it by that, not by any
            # placeholder text — the closing visible_text now reuses alice's real
            # substantive answer (issue #87 follow-up).
            if (
                isinstance(frame, TurnFrame)
                and frame.agent_id == "alice"
                and not frame.stance.needs_from
            ):
                self.closing_attempts += 1
                if self.closing_attempts == 1:
                    raise RuntimeError("closing publish blew up (once)")
            await super().publish_frame_checked(subject, frame)

    engine = FakeEngine()
    sup = Supervisor(
        session=engine,
        config=SupervisorConfig(agent="alice", env="dev"),
        home=tmp_path,
        publisher=RecordingPublisher(),
        system_prompt="SYS",
        tools_policy=ToolsPolicy(),
    )
    bus = ClosingFailsOnceBus()
    runner = FrameTurnRunner(agent_id="alice", supervisor=sup, bus=bus)
    await runner.start()
    task = _duo_task(["alice", "bob"])
    await bus.deliver(subjects.agent_task("alice"), task)

    turn_subj = subjects.task_turn(task.id)
    # Drive every budgeted round with a NON-TERMINAL alice follow-up so the cap
    # leaves her hung and the frame-level closing terminus fires.
    for _round in range(MAX_SIBLING_ROUNDS):
        await bus.deliver(turn_subj, _sibling_turn("bob", task.id, needs=["alice"]))
        await runner._on_engine_event(
            TurnDone(
                text="alice still disagrees",
                stance=Stance(position="dissent", because="unconvinced", needs_from=["bob"]),
            )
        )

    # The first closing publish RAISED, so no closing turn is on the wire yet, and
    # the decision state stays armed for retry.
    def _closing_turns() -> list[TurnFrame]:
        return [
            f
            for f in bus.frames_on(turn_subj)
            if isinstance(f, TurnFrame) and f.agent_id == "alice" and not f.stance.needs_from
        ]

    assert bus.closing_attempts == 1
    assert _closing_turns() == []
    assert task.id in runner._sibling_budget_spent  # re-armed, not cleared
    assert task.id not in runner._sibling_closing_emitted  # guard rolled back

    # The re-armed emit-if-hung path re-fires on the next observed turn for this
    # task (its budget-spent marker + last stance survived the failure), and the
    # retry publish now succeeds: exactly one closing turn lands.
    await runner._emit_closing_turn_if_hung(task.id)
    closing = _closing_turns()
    assert len(closing) == 1
    assert closing[0].stance.needs_from == []
    assert closing[0].stance.position == "dissent"
    # Real substantive text preserved on the retried closing turn, not a placeholder.
    assert closing[0].visible_text == "alice still disagrees"
    assert "round budget" not in closing[0].visible_text.lower()
    # The decision state is now cleared (confirmed publish).
    assert task.id not in runner._sibling_budget_spent

    # A further re-fire does NOT publish a second closing turn (one-shot + the
    # decision state is cleared).
    await runner._emit_closing_turn_if_hung(task.id)
    assert len(_closing_turns()) == 1
    assert bus.closing_attempts == 2  # one failed + one succeeded; no third attempt

    # Completion no longer hangs: over the resulting graph is_done flips True.
    alice_turns = [
        f for f in bus.frames_on(turn_subj) if isinstance(f, TurnFrame) and f.agent_id == "alice"
    ]
    graph: list[TurnFrame] = [*alice_turns, _sibling_turn("bob", task.id, needs=[])]
    assert is_done(task, graph, participants=frozenset({"alice", "bob"}))


async def test_closing_turn_not_emitted_when_latest_turn_already_terminal(
    tmp_path: Path,
) -> None:
    """(c) The closing turn is NOT emitted when alice's latest budgeted turn was
    already terminal — there is no hang to close, so no synthetic terminus fires.
    """
    engine, runner, bus = _runner_for("alice", tmp_path)
    await runner.start()
    task = _duo_task(["alice", "bob"])
    await bus.deliver(subjects.agent_task("alice"), task)
    turn_subj = subjects.task_turn(task.id)

    for round_no in range(MAX_SIBLING_ROUNDS):
        await bus.deliver(turn_subj, _sibling_turn("bob", task.id, needs=["alice"]))
        # The FINAL budgeted follow-up is terminal (alice concedes on the last
        # round); earlier rounds stay non-terminal so the budget is fully spent.
        last = round_no == MAX_SIBLING_ROUNDS - 1
        await runner._on_engine_event(
            TurnDone(
                text="alice",
                stance=Stance(
                    position="revised",
                    because="round",
                    needs_from=[] if last else ["bob"],
                ),
            )
        )

    # Count alice's terminal turns: exactly the ONE she published herself — no
    # extra synthetic closing turn was appended on top.
    alice_terminal = [
        f
        for f in bus.frames_on(turn_subj)
        if isinstance(f, TurnFrame) and f.agent_id == "alice" and not f.stance.needs_from
    ]
    assert len(alice_terminal) == 1
    assert "final" not in alice_terminal[0].visible_text.lower()

    # A later dropped sibling still emits nothing (alice's latest is terminal).
    # The lane is retired, so drive the handler directly.
    await runner._on_sibling_turn(task.id, _sibling_turn("bob", task.id, needs=["alice"]))
    alice_terminal_after = [
        f
        for f in bus.frames_on(turn_subj)
        if isinstance(f, TurnFrame) and f.agent_id == "alice" and not f.stance.needs_from
    ]
    assert len(alice_terminal_after) == 1


async def test_cap_closing_turn_preserves_substantive_text_after_empty_follow_up(
    tmp_path: Path,
) -> None:
    """An empty post-cap follow-up cannot erase the last real answer."""
    engine, runner, bus = _runner_for("bilby", tmp_path)
    await runner.start()
    task = _duo_task(["bilby", "skippy"])
    await bus.deliver(subjects.agent_task("bilby"), task)
    turn_subj = subjects.task_turn(task.id)

    substantive = "Bilby's substantive position survives exactly."
    for round_no in range(MAX_SIBLING_ROUNDS):
        await bus.deliver(turn_subj, _sibling_turn("skippy", task.id, needs=["bilby"]))
        await runner._on_engine_event(
            TurnDone(
                # The final follow-up lands after the cap retires the subscription.
                text="" if round_no == MAX_SIBLING_ROUNDS - 1 else substantive,
                stance=Stance(
                    position="pro-bus", because="single source of truth", needs_from=["skippy"]
                ),
            )
        )

    bilby_turns = [
        f for f in bus.frames_on(turn_subj) if isinstance(f, TurnFrame) and f.agent_id == "bilby"
    ]
    assert bilby_turns[-2].visible_text == "(no output)"
    assert bilby_turns[-2].stance.needs_from == ["skippy"]

    closing = bilby_turns[-1]
    assert closing.visible_text == substantive
    assert closing.stance.needs_from == []
    assert closing.stance.position == "pro-bus"


async def test_cap_closing_turn_emits_exactly_one_frame_preserving_text(
    tmp_path: Path,
) -> None:
    """The bounded-round safety still emits AT MOST ONE closing frame with empty
    ``needs_from`` even as the visible_text is now the agent's real content.

    Guards the invariant that preserving text did not multiply closing frames or
    leak a second terminus on a later dropped sibling.
    """
    engine, runner, bus = _runner_for("bilby", tmp_path)
    await runner.start()
    task = _duo_task(["bilby", "skippy"])
    await bus.deliver(subjects.agent_task("bilby"), task)
    turn_subj = subjects.task_turn(task.id)

    final_text = "Bilby's closing case: bounded rounds keep the debate honest."
    for round_no in range(MAX_SIBLING_ROUNDS):
        await bus.deliver(turn_subj, _sibling_turn("skippy", task.id, needs=["bilby"]))
        await runner._on_engine_event(
            TurnDone(
                text=final_text if round_no == MAX_SIBLING_ROUNDS - 1 else "earlier round",
                stance=Stance(position="dissent", because="bounded", needs_from=["skippy"]),
            )
        )

    # Drive extra dropped siblings on the retired lane: still exactly one closing.
    for _ in range(3):
        await runner._on_sibling_turn(task.id, _sibling_turn("skippy", task.id, needs=["bilby"]))

    bilby_terminal = [
        f
        for f in bus.frames_on(turn_subj)
        if isinstance(f, TurnFrame) and f.agent_id == "bilby" and not f.stance.needs_from
    ]
    assert len(bilby_terminal) == 1
    assert bilby_terminal[0].visible_text == final_text


async def test_concurrent_sibling_deliveries_reserve_under_lock_do_not_overshoot(
    tmp_path: Path,
) -> None:
    """(d) Two concurrent sibling deliveries reserve the round slot under the SAME
    lock as the '< cap' check, so total follow-ups == cap EXACTLY (no overshoot).

    A send that yields lets a second delivery interleave BEFORE the first
    increments. Under the old check-then-increment split both could pass the
    '< cap' check and overshoot; reserving atomically caps total follow-ups at
    exactly ``MAX_SIBLING_ROUNDS``.
    """
    gate = asyncio.Event()

    class YieldingEngine(FakeEngine):
        def __init__(self) -> None:
            super().__init__()
            self.gated = False

        async def send(self, text: str) -> None:
            # Once armed, yield inside the send so a concurrently-dispatched
            # sibling delivery runs its check-and-reserve while this one is
            # mid-await. The initial task dispatch runs ungated.
            if self.gated:
                await gate.wait()
            await super().send(text)

    engine = YieldingEngine()
    sup = Supervisor(
        session=engine,
        config=SupervisorConfig(agent="alice", env="dev"),
        home=tmp_path,
        publisher=RecordingPublisher(),
        system_prompt="SYS",
        tools_policy=ToolsPolicy(),
    )
    bus = FakeFrameBus()
    runner = FrameTurnRunner(agent_id="alice", supervisor=sup, bus=bus)
    await runner.start()
    task = _duo_task(["alice", "bob"])
    await bus.deliver(subjects.agent_task("alice"), task)
    engine.gated = True
    sends_before = len(engine.sends)

    # Fire cap+2 concurrent sibling deliveries; all pass the initial guard region
    # before any send completes, then the gate releases them together.
    concurrency = MAX_SIBLING_ROUNDS + 2
    deliveries = [
        runner._on_sibling_turn(task.id, _sibling_turn("bob", task.id, needs=["alice"]))
        for _ in range(concurrency)
    ]
    gathered = asyncio.gather(*deliveries)
    await asyncio.sleep(0)  # let every coroutine reach the gate
    gate.set()
    await gathered

    # Exactly MAX_SIBLING_ROUNDS follow-up sends opened — the atomic reserve made
    # the 'exactly the cap' guarantee hard, with no overshoot. (The over-cap
    # deliveries hit the drop path and retired the lane, which clears the counter,
    # so the send count is the load-bearing invariant here.)
    follow_up_sends = len(engine.sends) - sends_before
    assert follow_up_sends == MAX_SIBLING_ROUNDS


# --- persisted dedup across restarts (issue #83) -----------------------------


def _persisting_runner(
    engine: FakeEngine, home: Path, *, horizon_s: float = 300.0
) -> tuple[FrameTurnRunner, FakeFrameBus]:
    """Build a runner whose completed-task dedup persists under ``home``."""
    sup = Supervisor(
        session=engine,
        config=SupervisorConfig(agent=_AGENT, env="dev"),
        home=home,
        publisher=RecordingPublisher(),
        system_prompt="SYS",
        tools_policy=ToolsPolicy(),
    )
    bus = FakeFrameBus()
    runner = FrameTurnRunner(
        agent_id=_AGENT,
        supervisor=sup,
        bus=bus,
        config=FrameTurnConfig(task_replay_horizon_s=horizon_s, state_home=home),
    )
    return runner, bus


async def test_completed_dedup_survives_a_restart(tmp_path: Path) -> None:
    """A completed frame redelivered to a FRESH runner over the same state home
    within the horizon is NOT re-dispatched — the persisted half of #83.

    The pre-fix cache was memory-only, so a restart within the horizon started
    empty and a redelivered completed frame (fresh timestamp) was re-dispatched.
    """
    state_home = tmp_path / "agent-home"
    engine1 = FakeEngine()
    runner1, bus1 = _persisting_runner(engine1, state_home)
    await runner1.start()
    task = _task(task_id="T-restart")

    await bus1.deliver(subjects.agent_task(_AGENT), task)
    await runner1._on_engine_event(TurnDone(text="done"))
    assert len(engine1.sends) == 1

    # A restart: a brand-new runner + engine over the SAME state home. It reloads
    # the persisted completion and refuses the redelivered frame.
    engine2 = FakeEngine()
    runner2, bus2 = _persisting_runner(engine2, state_home)
    await runner2.start()
    assert (task.id, _AGENT) in runner2._persisted_completions

    await bus2.deliver(subjects.agent_task(_AGENT), task)

    assert engine2.sends == []
    assert bus2.frames_on(subjects.task_status(task.id)) == []


async def test_persisted_dedup_evicts_past_horizon_on_load(tmp_path: Path) -> None:
    """A completion older than the horizon does NOT dedup a fresh redelivery.

    Wall-clock horizon: the persisted file is pruned on load, so a stale
    completion neither reloads nor blocks a legitimately fresh task.
    """
    state_home = tmp_path / "agent-home"
    store = CompletedTaskStore(state_home, _AGENT)
    # Persist a completion well past the horizon.
    old = datetime.now(UTC) - timedelta(seconds=1000)
    store.save({("T-old", _AGENT): old}, horizon_s=300.0)

    engine = FakeEngine()
    runner, bus = _persisting_runner(engine, state_home, horizon_s=300.0)
    await runner.start()
    # The stale entry was evicted on load — nothing reloaded.
    assert runner._persisted_completions == {}

    fresh = _task(task_id="T-old").model_copy(update={"created_at": datetime.now(UTC)})
    await bus.deliver(subjects.agent_task(_AGENT), fresh)
    assert [send.text for send in engine.sends] == [fresh.text]


async def test_corrupt_dedup_file_opens_distrust_window_and_refuses_within_horizon(
    tmp_path: Path,
) -> None:
    """A CORRUPT dedup file opens a distrust window: an untracked-but-within-
    horizon frame is REFUSED (fail closed, loud), while a frame delivered AFTER
    the window dispatches normally (issue #83).

    Corrupt = provable integrity loss, so the store cannot silently start empty
    and fail open. It starts empty AND distrusts for ``horizon_s`` seconds; the
    corrupt file is preserved for forensics rather than overwritten.
    """
    from loguru import logger

    state_home = tmp_path / "agent-home"
    state_home.mkdir(parents=True)
    store = CompletedTaskStore(state_home, _AGENT)
    store.path.write_text("{not json", encoding="utf-8")

    engine = FakeEngine()
    runner, bus = _persisting_runner(engine, state_home, horizon_s=300.0)

    records: list[str] = []
    sink_id = logger.add(records.append, level="WARNING")
    try:
        await runner.start()
        # Corrupt load: empty dedup set, distrust window OPEN, bad file preserved.
        assert runner._persisted_completions == {}
        assert runner._completed_store is not None
        assert runner._completed_store.in_distrust_window()
        preserved = list(state_home.glob("*.corrupt-*"))
        assert preserved, "corrupt dedup file must be preserved for forensics"

        # A fresh, within-horizon untracked frame is REFUSED while the window is
        # open — fail closed, no dispatch, no side effects, loud per refusal.
        within = _task(task_id="T-distrust").model_copy(update={"created_at": datetime.now(UTC)})
        await bus.deliver(subjects.agent_task(_AGENT), within)
        assert engine.sends == []
        assert bus.frames_on(subjects.task_status(within.id)) == []
        assert list(runner._pending) == []
    finally:
        logger.remove(sink_id)

    loud = [r for r in records if "distrust window OPEN" in r]
    assert loud, "expected a loud per-refusal warning naming the distrust window"
    assert "T-distrust" in loud[-1]

    # After the window closes, normal operation resumes: force the window shut
    # and a fresh frame dispatches (anything redeliverable has aged past horizon).
    runner._completed_store._distrust_until = datetime.now(UTC) - timedelta(seconds=1)
    assert not runner._completed_store.in_distrust_window()
    after = _task(task_id="T-after-window").model_copy(update={"created_at": datetime.now(UTC)})
    await bus.deliver(subjects.agent_task(_AGENT), after)
    assert [send.text for send in engine.sends] == [after.text]


async def test_value_level_damage_opens_distrust_and_keeps_parseable_entries(
    tmp_path: Path,
) -> None:
    """An unparseable timestamp inside structurally-valid JSON is still damage:
    the dropped record's redelivery would dispatch as fresh, so it opens the
    SAME distrust window structural corruption does, while parseable entries
    are kept (they refuse strictly more) (issue #83, cross-review find).
    """
    import json as _json

    state_home = tmp_path / "agent-home"
    state_home.mkdir(parents=True)
    store = CompletedTaskStore(state_home, _AGENT)
    good_ts = datetime.now(UTC).isoformat()
    store.path.write_text(
        _json.dumps({"T-good": {_AGENT: good_ts}, "T-bad": {_AGENT: "not-a-time"}}),
        encoding="utf-8",
    )

    engine = FakeEngine()
    runner, bus = _persisting_runner(engine, state_home, horizon_s=300.0)
    await runner.start()

    # Damage detected: distrust window OPEN; the parseable record is retained.
    assert runner._completed_store is not None
    assert runner._completed_store.in_distrust_window()
    assert ("T-good", _AGENT) in runner._persisted_completions
    assert ("T-bad", _AGENT) not in runner._persisted_completions

    # Both the retained completed task AND an untracked frame are refused while
    # the window is open — the damaged record's redelivery cannot slip through.
    redelivered_bad = _task(task_id="T-bad").model_copy(update={"created_at": datetime.now(UTC)})
    await bus.deliver(subjects.agent_task(_AGENT), redelivered_bad)
    assert engine.sends == []


async def test_missing_dedup_file_starts_empty_no_distrust(tmp_path: Path) -> None:
    """No dedup file yet: legitimate first boot — start empty, INFO log, NO
    distrust window, and dispatch proceeds normally (issue #83).
    """
    from loguru import logger

    state_home = tmp_path / "does-not-exist-yet"
    engine = FakeEngine()

    # The store loads at runner construction, so arm the sink BEFORE building it.
    records: list[str] = []
    sink_id = logger.add(records.append, level="INFO")
    try:
        runner, bus = _persisting_runner(engine, state_home)
        await runner.start()
    finally:
        logger.remove(sink_id)

    assert runner._persisted_completions == {}
    assert runner._completed_store is not None
    # Missing != corrupt: no distrust window, so within-horizon frames dispatch.
    assert not runner._completed_store.in_distrust_window()
    info = [r for r in records if "first boot" in r]
    assert info, "expected a loud INFO log for the missing-file first boot"

    task = _task(task_id="T-first-boot")
    await bus.deliver(subjects.agent_task(_AGENT), task)
    assert [send.text for send in engine.sends] == [task.text]


async def test_nul_bearing_task_id_round_trips_and_dedups_across_restart(tmp_path: Path) -> None:
    """A NUL-bearing task id round-trips the NESTED store: complete -> restart ->
    the redelivered frame is REFUSED (issue #83 hostile-key hardening).

    The nested ``{task_id: {agent_id: ts}}`` shape lets an arbitrary string
    (NUL bytes included) be a native JSON object key — no composite key, no
    separator to mangle. The completion persists and dedups a redelivery to a
    fresh runner over the same per-agent state home.
    """
    state_home = tmp_path / "agent-home"
    nul_id = "T-with\x00nul\x00bytes"
    engine1 = FakeEngine()
    runner1, bus1 = _persisting_runner(engine1, state_home)
    await runner1.start()
    task = _task(task_id=nul_id)

    await bus1.deliver(subjects.agent_task(_AGENT), task)
    await runner1._on_engine_event(TurnDone(text="done"))
    assert len(engine1.sends) == 1

    # Restart over the same state home: the NUL-bearing key reloaded intact.
    engine2 = FakeEngine()
    runner2, bus2 = _persisting_runner(engine2, state_home)
    await runner2.start()
    assert (nul_id, _AGENT) in runner2._persisted_completions

    await bus2.deliver(subjects.agent_task(_AGENT), task)
    assert engine2.sends == []
    assert bus2.frames_on(subjects.task_status(nul_id)) == []


def test_per_agent_filename_derivation_including_sanitize_hash_path() -> None:
    """The dedup filename is PER-AGENT and collision-safe (issue #83).

    Every filename carries a short stable hash of the RAW id — unconditionally,
    so a crafted clean id can never impersonate a dirty id's stem+hash (the
    conditional-hash scheme allowed exactly that collision). Distinct agents
    get distinct files; the same raw id is stable.
    """
    home = Path("/tmp/agent-home")
    plain = CompletedTaskStore(home, "alice")
    assert plain.path.name.startswith("completed-tasks-alice-")
    assert plain.path.name.endswith(".json")

    # The crafted-collision counterexample from review: a clean id shaped like
    # another id's sanitized stem + raw-hash must NOT share that id's file.
    import hashlib as _hashlib

    dirty_hash = _hashlib.sha256(b"foo/bar").hexdigest()[:12]
    crafted = CompletedTaskStore(home, f"foo-bar-{dirty_hash}")
    victim = CompletedTaskStore(home, "foo/bar")
    assert crafted.path != victim.path

    # Distinct agents -> distinct files (single-writer per agent by construction).
    bob = CompletedTaskStore(home, "bob")
    assert plain.path != bob.path

    # A separator-bearing id is sanitized AND hashed: no path traversal, and the
    # raw-id hash guarantees no collision with another id sharing the safe stem.
    dirty = CompletedTaskStore(home, "team/agent:one")
    assert "/" not in dirty.path.name
    assert ":" not in dirty.path.name
    assert dirty.path.name.startswith("completed-tasks-team-agent-one-")
    assert dirty.path.name.endswith(".json")

    # Two raw ids that sanitize to the SAME stem stay distinct via the raw hash.
    collide_a = CompletedTaskStore(home, "a/b")
    collide_b = CompletedTaskStore(home, "a:b")
    assert collide_a.path != collide_b.path

    # Same raw id is always the same filename (stable across constructions).
    again = CompletedTaskStore(home, "team/agent:one")
    assert again.path == dirty.path


# --- fail closed on future-dated frames (issue #83) --------------------------


async def test_future_dated_beyond_skew_tolerance_fails_closed(tmp_path: Path) -> None:
    """A created_at more than the skew tolerance in the FUTURE fails closed.

    A future-dated frame yields a NEGATIVE age that would trivially pass
    ``age > horizon`` and reach the engine; beyond the tolerance it is refused.
    """
    engine, runner, bus = _make(tmp_path)
    await runner.start()
    future = _task(task_id="T-future").model_copy(
        update={"created_at": datetime.now(UTC) + timedelta(seconds=FUTURE_SKEW_TOLERANCE_S + 1)}
    )

    await bus.deliver(subjects.agent_task(_AGENT), future)

    assert engine.sends == []
    assert bus.frames_on(subjects.task_status(future.id)) == []
    assert bus.handlers_on(subjects.task_steer(future.id)) == []
    assert list(runner._pending) == []


async def test_future_dated_within_skew_tolerance_dispatches(tmp_path: Path) -> None:
    """A created_at within the skew tolerance clamps its age to 0 and dispatches.

    Mild clock skew must not refuse a legitimate task; the age is clamped to 0
    (never negative) so it is treated as brand new, not stale.
    """
    engine, runner, bus = _make(tmp_path)
    await runner.start()
    slightly_ahead = _task(task_id="T-skew").model_copy(
        update={"created_at": datetime.now(UTC) + timedelta(seconds=FUTURE_SKEW_TOLERANCE_S - 1)}
    )

    await bus.deliver(subjects.agent_task(_AGENT), slightly_ahead)

    assert [send.text for send in engine.sends] == [slightly_ahead.text]


async def test_current_origin_reflects_active_task(tmp_path: Path) -> None:
    """The runner exposes the active task's ChannelOrigin for the wake tool (#93).

    A wake fired mid-turn must carry the SAME origin as the triggering intake
    task, so the runner tracks the active task's origin and surfaces it via
    ``current_origin``. A task with no origin yields ``None``.
    """
    from rtech_frames import ChannelOrigin

    engine, runner, bus = _make(tmp_path)
    await runner.start()
    assert runner.current_origin() is None

    origin = ChannelOrigin(channel_id="c-1", guild_id="g-1", message_id="m-1", asker="a-1")
    intake = _task(task_id="T-origin").model_copy(update={"origin": origin})
    await bus.deliver(subjects.agent_task(_AGENT), intake)

    assert runner.current_origin() == origin


async def test_active_origin_cleared_on_turn_done_no_bleed(tmp_path: Path) -> None:
    """The active origin does not bleed past its task's TurnDone (issue #93).

    A wake fired in the gap between one task completing and the next dispatching
    must NOT read the finished task's origin — that would route a woken agent's
    answer to a stale channel. After TurnDone the origin is cleared.
    """
    from rtech_frames import ChannelOrigin

    engine, runner, bus = _make(tmp_path)
    await runner.start()

    origin = ChannelOrigin(channel_id="c-1", asker="a-1")
    intake = _task(task_id="T-o1").model_copy(update={"origin": origin})
    await bus.deliver(subjects.agent_task(_AGENT), intake)
    assert runner.current_origin() == origin

    await runner._on_engine_event(TurnDone(text="done"))
    assert runner.current_origin() is None


async def test_current_origin_tracks_fifo_head_across_two_pending_tasks(tmp_path: Path) -> None:
    """current_origin resolves the EXECUTING turn = FIFO head, not the latest pend.

    Task A (origin X) dispatches and executes; task B (origin Y) arrives before A's
    TurnDone. The executing turn is still A (the pending head), so current_origin
    must report X — never B's origin (the single-slot bug read B's mid-A-turn).
    After A's TurnDone, B is the head/executing task, so current_origin flips to Y.
    """
    from rtech_frames import ChannelOrigin

    engine, runner, bus = _make(tmp_path)
    await runner.start()

    origin_x = ChannelOrigin(channel_id="c-x", asker="a-x")
    origin_y = ChannelOrigin(channel_id="c-y", asker="a-y")
    task_a = _task(task_id="T-A").model_copy(update={"origin": origin_x})
    task_b = _task(task_id="T-B").model_copy(update={"origin": origin_y})

    await bus.deliver(subjects.agent_task(_AGENT), task_a)
    # B arrives mid-A-turn (before A's TurnDone): A is still the executing head.
    await bus.deliver(subjects.agent_task(_AGENT), task_b)
    assert runner.current_origin() == origin_x

    # A completes -> B is now the head/executing task -> its origin is current.
    await runner._on_engine_event(TurnDone(text="a done"))
    assert runner.current_origin() == origin_y

    # B completes -> nothing pending -> no origin.
    await runner._on_engine_event(TurnDone(text="b done"))
    assert runner.current_origin() is None


async def test_failed_send_rollback_of_later_task_retains_active_origin(tmp_path: Path) -> None:
    """A failed-send rollback of B (arriving mid-A-turn) keeps A's origin (#93).

    B's send blows up, rolling back ONLY B's per-task state; A is still the
    executing head, so current_origin must still report A's origin X. A full
    rollback of a SOLE task then leaves no pending head -> current_origin is None.
    """
    from rtech_frames import ChannelOrigin

    engine, runner, bus = _make(tmp_path)
    await runner.start()

    origin_x = ChannelOrigin(channel_id="c-x", asker="a-x")
    task_a = _task(task_id="T-A2").model_copy(update={"origin": origin_x})
    await bus.deliver(subjects.agent_task(_AGENT), task_a)
    assert runner.current_origin() == origin_x

    # B's send fails: its pend path recorded an origin, then rollback removes ONLY
    # B's entry. A stays the pending head, so A's origin survives.
    async def _boom(_text: str) -> None:
        raise RuntimeError("send blew up")

    engine.send = _boom  # type: ignore[method-assign]
    origin_y = ChannelOrigin(channel_id="c-y", asker="a-y")
    task_b = _task(task_id="T-B2").model_copy(update={"origin": origin_y})
    await bus.deliver(subjects.agent_task(_AGENT), task_b)
    assert runner.current_origin() == origin_x

    # A full rollback of the SOLE remaining task clears the head entirely.
    engine2, runner2, bus2 = _make(tmp_path)
    await runner2.start()
    engine2.send = _boom  # type: ignore[method-assign]
    sole = _task(task_id="T-sole").model_copy(update={"origin": origin_x})
    await bus2.deliver(subjects.agent_task(_AGENT), sole)
    assert runner2.current_origin() is None


async def test_reply_turn_carries_triggering_task_origin(tmp_path: Path) -> None:
    """A reply TurnFrame copies the triggering intake task's origin (issue #93).

    An intake/wake task carries a ChannelOrigin; the reply TurnFrame agentd
    publishes must copy it so a triage edge can render the answer back into the
    asking channel. A non-intake task (no origin) yields a reply with origin None
    (byte-identical to the legacy wire).
    """
    from rtech_frames import ChannelOrigin

    engine, runner, bus = _make(tmp_path)
    await runner.start()

    origin = ChannelOrigin(channel_id="c-9", guild_id="g-9", message_id="m-9", asker="a-9")
    intake = _task(task_id="T-reply").model_copy(update={"origin": origin})
    await bus.deliver(subjects.agent_task(_AGENT), intake)
    await runner._on_engine_event(
        TurnDone(text="the answer", stance=Stance(position="answered", because="checked"))
    )

    turns = [f for f in bus.frames_on(subjects.task_turn(intake.id)) if isinstance(f, TurnFrame)]
    assert len(turns) == 1
    assert turns[0].origin == origin

    # A non-intake task yields a reply with no origin.
    plain = _task(task_id="T-plain")
    assert plain.origin is None
    await bus.deliver(subjects.agent_task(_AGENT), plain)
    await runner._on_engine_event(
        TurnDone(text="plain answer", stance=Stance(position="answered", because="done"))
    )
    plain_turns = [
        f for f in bus.frames_on(subjects.task_turn(plain.id)) if isinstance(f, TurnFrame)
    ]
    assert len(plain_turns) == 1
    assert plain_turns[0].origin is None


async def test_sibling_repend_keeps_local_origin_not_sibling_frame(tmp_path: Path) -> None:
    """A sibling-triggered re-pended round keeps the LOCAL origin authority (R1).

    The first round's TurnDone no longer pops ``_task_origins`` for a fanned task:
    the local entry (written once at TaskFrame pend, the sole authority) survives
    until the task is fully retired, so a sibling-triggered follow-up round still
    replies in-channel WITHOUT ever re-reading it from the sibling frame.
    """
    from rtech_frames import ChannelOrigin

    engine, runner, bus = _runner_for("alice", tmp_path)
    await runner.start()
    origin = ChannelOrigin(channel_id="c-sib", asker="a-sib")
    task = _duo_task(["alice", "bob"]).model_copy(update={"origin": origin})
    await bus.deliver(subjects.agent_task("alice"), task)

    # Round one: alice opens still needing bob. R1: the fanned task's origin entry
    # SURVIVES her TurnDone (previously it was popped and later lost).
    await runner._on_engine_event(
        TurnDone(
            text="alice round one",
            stance=Stance(position="opened", because="kickoff", needs_from=["bob"]),
        )
    )
    assert runner._task_origins.get(task.id) == origin  # survives TurnDone (R1)

    # Bob's real turn re-pends alice's follow-up round. The origin is unchanged —
    # it is NOT re-derived from bob's frame.
    bob_turn = _sibling_turn("bob", task.id, needs=["alice"])
    await bus.deliver(subjects.task_turn(task.id), bob_turn)
    assert runner._task_origins.get(task.id) == origin

    # Round two's reply carries the LOCAL origin.
    await runner._on_engine_event(
        TurnDone(
            text="alice round two",
            stance=Stance(position="revised", because="bob raised X", needs_from=["bob"]),
        )
    )
    turns = [f for f in bus.frames_on(subjects.task_turn(task.id)) if isinstance(f, TurnFrame)]
    assert turns[-1].agent_id == "alice"
    assert turns[-1].origin == origin


async def test_sibling_frame_origin_never_becomes_task_authority(tmp_path: Path) -> None:
    """R2: a sibling's TurnFrame origin NEVER overwrites this task's reply route.

    Reproduces the misroute: a hostile/foreign sibling frame carries a DIFFERENT
    ChannelOrigin. Under the old ``setdefault(frame.task_id, frame.origin)`` re-pend
    restore, if the local entry had been cleared the sibling's origin became task
    A's authority and the follow-up reply was routed to the SIBLING's channel. With
    R1+R2 the local origin (written only at TaskFrame pend) is the sole authority
    and the sibling's origin is ignored entirely.
    """
    from rtech_frames import ChannelOrigin

    engine, runner, bus = _runner_for("alice", tmp_path)
    await runner.start()
    real_origin = ChannelOrigin(channel_id="c-asker", asker="a-real")
    hostile_origin = ChannelOrigin(channel_id="c-hostile", asker="a-attacker")
    task = _duo_task(["alice", "bob"]).model_copy(update={"origin": real_origin})
    await bus.deliver(subjects.agent_task("alice"), task)

    # Round one opens still needing bob (non-terminal); the local origin persists.
    await runner._on_engine_event(
        TurnDone(
            text="alice round one",
            stance=Stance(position="opened", because="kickoff", needs_from=["bob"]),
        )
    )

    # Bob's sibling turn arrives carrying a DIFFERENT (hostile) origin and re-pends
    # alice's follow-up round. The task's origin authority must stay the REAL one.
    bob_turn = _sibling_turn("bob", task.id, needs=["alice"]).model_copy(
        update={"origin": hostile_origin}
    )
    await bus.deliver(subjects.task_turn(task.id), bob_turn)
    assert runner._task_origins.get(task.id) == real_origin

    # The follow-up reply routes to the REAL asker channel, never the sibling's.
    await runner._on_engine_event(
        TurnDone(
            text="alice round two",
            stance=Stance(position="revised", because="bob raised X", needs_from=["bob"]),
        )
    )
    turns = [f for f in bus.frames_on(subjects.task_turn(task.id)) if isinstance(f, TurnFrame)]
    assert turns[-1].origin == real_origin
    assert turns[-1].origin != hostile_origin


async def test_cap_closing_turn_carries_task_origin(tmp_path: Path) -> None:
    """R1: the cap-closing terminal turn carries the task's origin, not None.

    The round-cap closing turn is published LATER than the final budgeted
    follow-up's TurnDone. Under the old code that TurnDone popped ``_task_origins``,
    so ``_publish_closing_turn`` re-read an empty map and emitted origin=None — the
    triage edge then never rendered the round-capped answer. With R1 the origin
    survives until the closing terminus resolves, so the closing turn carries it.
    """
    from rtech_frames import ChannelOrigin

    engine, runner, bus = _runner_for("alice", tmp_path)
    await runner.start()
    origin = ChannelOrigin(channel_id="c-cap", asker="a-cap", message_id="m-cap")
    task = _duo_task(["alice", "bob"]).model_copy(update={"origin": origin})
    await bus.deliver(subjects.agent_task("alice"), task)

    turn_subj = subjects.task_turn(task.id)
    # Drive every budgeted round with a NON-TERMINAL alice follow-up so the cap
    # leaves her hung and the frame-level closing terminus fires.
    for _round in range(MAX_SIBLING_ROUNDS):
        await bus.deliver(turn_subj, _sibling_turn("bob", task.id, needs=["alice"]))
        await runner._on_engine_event(
            TurnDone(
                text="alice still disagrees",
                stance=Stance(position="dissent", because="unconvinced", needs_from=["bob"]),
            )
        )

    # Exactly one closing terminal turn, and it CARRIES the task origin (R1).
    closing = [
        f
        for f in bus.frames_on(turn_subj)
        if isinstance(f, TurnFrame) and f.agent_id == "alice" and not f.stance.needs_from
    ]
    assert len(closing) == 1
    assert closing[0].origin == origin  # was None before R1
    # And the origin entry is fully retired once the terminus resolves (no leak).
    assert task.id not in runner._task_origins


# ---------------------------------------------------- debate-role framing (#debate)

_TRIO = ["agentd-nagatha", "agentd-bilby", "agentd-iris"]


def _debate_task(
    *, arbiter: str | None, participants: list[str] | None = None, text: str = "settle this"
) -> TaskFrame:
    parts = participants if participants is not None else _TRIO
    scope = {1: "single", 2: "duo", 3: "trio"}[len(parts)]
    return TaskFrame(
        id="deb-1",
        origin_edge="chatd-intake:agentd-nagatha",
        requester="rico",
        text=text,
        scope=scope,  # type: ignore[arg-type]
        participants=parts,
        arbiter=arbiter,
        created_at=datetime.now(UTC),
    )


def test_short_name_strips_prefix_and_stack_suffix() -> None:
    assert _short_name("agentd-nagatha") == "nagatha"
    assert _short_name("agentd-nagatha-dev") == "nagatha"
    assert _short_name("agentd-bilby-prod") == "bilby"
    assert _short_name("plain") == "plain"


def test_render_debate_task_participant_directive() -> None:
    # A non-arbiter participant is told to argue only its own position and engage
    # sibling turns, NOT summarize; its short name and the roster are named.
    task = _debate_task(arbiter="agentd-nagatha")
    rendered = _render_debate_task(task, "agentd-bilby")
    assert rendered.startswith("[debate task] You are bilby, one of 3 debaters")
    assert "nagatha, bilby, iris" in rendered
    assert "Speak ONLY for yourself" in rendered
    assert "the arbiter does that" in rendered
    # The task text rides after the directive.
    assert rendered.endswith("\n\nsettle this")
    assert "ARBITER" not in rendered


def test_render_debate_task_arbiter_directive() -> None:
    # The arbiter is told to moderate and TLDR, not argue the substance.
    task = _debate_task(arbiter="agentd-nagatha")
    rendered = _render_debate_task(task, "agentd-nagatha")
    assert rendered.startswith("[debate task — you are the ARBITER]")
    assert "Moderate; do not argue the substance yourself" in rendered
    assert "short TLDR ruling/summary of THEIR positions" in rendered
    assert rendered.endswith("\n\nsettle this")


def test_render_debate_task_no_arbiter_treats_all_as_participants() -> None:
    # No arbiter set (duo with no moderator): every agent gets the participant
    # directive, never the arbiter one.
    task = _debate_task(arbiter=None, participants=["agentd-bilby", "agentd-iris"])
    rendered = _render_debate_task(task, "agentd-bilby")
    assert rendered.startswith("[debate task] You are bilby, one of 2 debaters")
    assert "ARBITER" not in rendered


def test_render_debate_task_single_scope_is_byte_identical_passthrough() -> None:
    # A single-scope task (<= 1 participant) is returned EXACTLY as frame.text —
    # no directive, byte-identical to today's send.
    single = _task()
    assert _render_debate_task(single, _AGENT) == single.text
    one = _debate_task(arbiter=None, participants=["agentd-bilby"])
    assert _render_debate_task(one, "agentd-bilby") == one.text


async def test_debate_task_send_carries_participant_framing(tmp_path: Path) -> None:
    # Integration: a multi-participant TaskFrame delivered to a NON-arbiter agent
    # sends the participant directive (not just frame.text) to the engine session.
    engine, runner, bus = _make_named(tmp_path, agent_id="agentd-bilby")
    await runner.start()
    task = _debate_task(arbiter="agentd-nagatha")
    await bus.deliver(subjects.agent_task("agentd-bilby"), task)
    sent = engine.sends[-1].text
    assert sent.startswith("[debate task] You are bilby")
    assert sent.endswith("\n\nsettle this")


async def test_debate_task_send_carries_arbiter_framing(tmp_path: Path) -> None:
    # Integration: the arbiter agent receives the arbiter directive.
    engine, runner, bus = _make_named(tmp_path, agent_id="agentd-nagatha")
    await runner.start()
    task = _debate_task(arbiter="agentd-nagatha")
    await bus.deliver(subjects.agent_task("agentd-nagatha"), task)
    sent = engine.sends[-1].text
    assert sent.startswith("[debate task — you are the ARBITER]")


def _make_named(
    tmp_path: Path, *, agent_id: str
) -> tuple[FakeEngine, FrameTurnRunner, FakeFrameBus]:
    engine = FakeEngine()
    sup = Supervisor(
        session=engine,
        config=SupervisorConfig(agent=agent_id, env="dev"),
        home=tmp_path,
        publisher=RecordingPublisher(),
        system_prompt="SYS",
        tools_policy=ToolsPolicy(),
    )
    bus = FakeFrameBus()
    runner = FrameTurnRunner(agent_id=agent_id, supervisor=sup, bus=bus)
    return engine, runner, bus
