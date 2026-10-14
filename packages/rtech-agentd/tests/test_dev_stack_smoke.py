"""Stack-wiring smoke: scripted engine + fake bus drive task -> working + turn.

The local dev stack (``scripts/dev_stack.py``) wires Orchestrator + Supervisor +
FrameTurnRunner + web-chat edge. This test exercises the SAME agentd wiring the
dev stack builds -- ``build_engine`` (scripted) feeding a ``Supervisor`` whose
event pump is consumed by a ``FrameTurnRunner`` over a fake typed-frame bus -- and
asserts a delivered ``TaskFrame`` produces a ``StatusFrame(working)`` then a
``TurnFrame`` on the bus, without a live NATS server. Scripts are not importable
as package modules, so the wiring (not the script file) is what we prove here.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from rtech_agentd.engine import ToolsPolicy
from rtech_agentd.engine_select import build_engine
from rtech_agentd.frame_turn import FrameTurnRunner
from rtech_agentd.scripted_engine import ScriptedEngine
from rtech_agentd.supervisor import Supervisor, SupervisorConfig
from rtech_frames import StatusFrame, TaskFrame, TurnFrame
from rtech_orchestrator import subjects

from .fakes import FakeFrameBus, RecordingPublisher

_AGENT = "agentd"


async def _wait_for(predicate: object, timeout: float = 2.0) -> None:
    """Poll ``predicate`` (a zero-arg callable) until true or timeout."""
    assert callable(predicate)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out waiting for condition")


async def test_dev_stack_wiring_task_drives_working_and_turn(tmp_path: Path) -> None:
    engine, kind = build_engine({"E2E_ENGINE": "scripted"})
    assert kind == "scripted"
    assert isinstance(engine, ScriptedEngine)

    sup = Supervisor(
        session=engine,
        config=SupervisorConfig(agent=_AGENT, env="dev"),
        home=tmp_path,
        publisher=RecordingPublisher(),
        system_prompt="dev-stack smoke",
        tools_policy=ToolsPolicy(),
    )
    bus = FakeFrameBus()
    runner = FrameTurnRunner(agent_id=_AGENT, supervisor=sup, bus=bus)

    await sup.start()
    run_task = asyncio.create_task(sup.run(), name="smoke-supervisor-run")
    await runner.start()

    task = TaskFrame(
        id="T-smoke",
        origin_edge="web",
        requester="rico",
        text="do it",
        scope="single",
        created_at=datetime.now(UTC),
    )
    try:
        await bus.deliver(subjects.agent_task(_AGENT), task)

        status_subj = subjects.task_status(task.id)
        await _wait_for(
            lambda: any(
                isinstance(f, StatusFrame) and f.state == "working"
                for f in bus.frames_on(status_subj)
            )
        )

        # Completion contract: the scripted task send completes its
        # own turn — no steer flush. The supervisor pump fans the TurnDone to the
        # runner, which publishes the TurnFrame with no further prompting.
        turn_subj = subjects.task_turn(task.id)
        await _wait_for(lambda: any(isinstance(f, TurnFrame) for f in bus.frames_on(turn_subj)))
    finally:
        await sup.stop()
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass

    turns = [f for f in bus.frames_on(subjects.task_turn(task.id)) if isinstance(f, TurnFrame)]
    assert len(turns) == 1
    assert turns[0].task_id == task.id
    assert turns[0].agent_id == _AGENT
    assert turns[0].visible_text
