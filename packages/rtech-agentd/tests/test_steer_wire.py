"""Typed-frame steer wire: SteerFrame injected into the LIVE managed turn."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from rtech_agentd.engine import TextDelta, ToolsPolicy, TurnDone
from rtech_agentd.frame_turn import FrameTurnRunner
from rtech_agentd.supervisor import Supervisor, SupervisorConfig
from rtech_frames import SteerFrame, TaskFrame, TurnFrame
from rtech_orchestrator import subjects

from .fakes import FakeEngine, FakeFrameBus, RecordingPublisher

_AGENT = "tester"


def _make(tmp_path: Path) -> tuple[FakeEngine, FrameTurnRunner, FakeFrameBus]:
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
    runner = FrameTurnRunner(agent_id=_AGENT, supervisor=sup, bus=bus)
    return engine, runner, bus


def _task(task_id: str = "T-1", text: str = "run it") -> TaskFrame:
    # Live producers stamp created_at; agentd fails closed on untimestamped
    # untracked frames (issue #83), so model the real on-wire shape here.
    return TaskFrame(
        id=task_id,
        origin_edge="web",
        requester="rico",
        text=text,
        scope="single",
        created_at=datetime.now(UTC),
    )


def _steer(task_id: str, text: str, *, target: str = _AGENT, interrupt: bool = False) -> SteerFrame:
    return SteerFrame(task_id=task_id, text=text, target_agent=target, interrupt=interrupt)


async def test_steer_lands_mid_turn(tmp_path: Path) -> None:
    engine, runner, bus = _make(tmp_path)
    await runner.start()
    task = _task()

    # Inbound task -> first send (turn not yet active).
    await bus.deliver(subjects.agent_task(_AGENT), task)
    assert engine.sends[-1].text == task.text
    assert engine.sends[-1].during_active_turn is False

    # The turn is now streaming (a delta arrived): mark it active.
    await engine.push(TextDelta(text="working..."))
    engine.mark_turn_active()

    # A steer arrives on the task's steer lane -> injected DURING the active turn.
    await bus.deliver(subjects.task_steer(task.id), _steer(task.id, "do it differently"))
    assert len(engine.sends) == 2
    assert engine.sends[1].text == "do it differently"
    assert engine.sends[1].during_active_turn is True


async def test_steer_for_non_active_task_is_ignored(tmp_path: Path) -> None:
    engine, runner, _bus = _make(tmp_path)
    await runner.start()
    # No task delivered, so no task is active. A steer injected directly is dropped.
    await runner._on_steer_frame("T-nope", _steer("T-nope", "steer into the void"))
    assert engine.sends == []


async def test_steer_for_other_agent_is_ignored(tmp_path: Path) -> None:
    engine, runner, bus = _make(tmp_path)
    await runner.start()
    task = _task()
    await bus.deliver(subjects.agent_task(_AGENT), task)
    before = len(engine.sends)

    # Task IS active, but the steer targets a different agent -> ignored.
    await bus.deliver(subjects.task_steer(task.id), _steer(task.id, "not for you", target="other"))
    assert len(engine.sends) == before


async def test_interrupt_steer_aborts_then_injects(tmp_path: Path) -> None:
    engine, runner, bus = _make(tmp_path)
    await runner.start()
    task = _task()
    await bus.deliver(subjects.agent_task(_AGENT), task)
    engine.mark_turn_active()

    # An interrupt-flagged steer: abort the active turn, THEN inject the text.
    steer = _steer(task.id, "stop and redo", interrupt=True)
    await bus.deliver(subjects.task_steer(task.id), steer)
    assert engine.interrupts == 1
    assert engine.sends[-1].text == "stop and redo"


async def test_interrupt_active_turn_method(tmp_path: Path) -> None:
    engine, runner, bus = _make(tmp_path)
    await runner.start()
    task = _task()
    await bus.deliver(subjects.agent_task(_AGENT), task)

    # The direct seam a steer(interrupt=True) drives: guarded on an active task.
    await runner.interrupt_active_turn(task.id)
    assert engine.interrupts == 1

    # An interrupt for an unknown task is a guarded no-op.
    await runner.interrupt_active_turn("T-unknown")
    assert engine.interrupts == 1


async def test_steer_ref_lands_in_turn_frame_refs(tmp_path: Path) -> None:
    engine, runner, bus = _make(tmp_path)
    await runner.start()
    task = _task()
    await bus.deliver(subjects.agent_task(_AGENT), task)
    engine.mark_turn_active()
    await bus.deliver(subjects.task_steer(task.id), _steer(task.id, "steer one"))

    # Complete the turn: the emitted TurnFrame.refs shows the steer causality.
    await runner._on_engine_event(TurnDone(text="done thinking"))
    turns = [f for f in bus.frames_on(subjects.task_turn(task.id)) if isinstance(f, TurnFrame)]
    assert len(turns) == 1
    assert turns[0].refs == [task.id, task.id]
