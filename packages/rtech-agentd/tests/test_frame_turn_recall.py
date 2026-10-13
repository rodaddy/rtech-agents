"""Per-turn Open Brain recall: bounded memory block prepend, best-effort (Phase 4)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from rtech_agentd.engine import ToolsPolicy
from rtech_agentd.frame_turn import (
    MEMORY_BLOCK_FOOTER,
    MEMORY_BLOCK_HEADER,
    FrameTurnRunner,
)
from rtech_agentd.memory import AgentMemoryLane
from rtech_agentd.supervisor import Supervisor, SupervisorConfig
from rtech_frames import TaskFrame
from rtech_orchestrator import subjects

from .fakes import FakeEngine, FakeFrameBus, RecordingPublisher

_AGENT = "tester"


class _StubMemory:
    """A tiny recall stub (NOT a real AgentMemoryLane) for the recall path."""

    def __init__(self, block: str = "", *, raise_on_recall: bool = False) -> None:
        self._block = block
        self._raise = raise_on_recall

    @property
    def max_items(self) -> int:
        return 8

    def recall(self, query: str, *, limit: int | None = None) -> str:
        if self._raise:
            raise RuntimeError("ob boom")
        return self._block


def _make(
    tmp_path: Path, memory: object | None = None
) -> tuple[FakeEngine, FrameTurnRunner, FakeFrameBus]:
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
        memory=cast("AgentMemoryLane | None", memory),
    )
    return engine, runner, bus


def _task(task_id: str = "T-1", text: str = "run it") -> TaskFrame:
    return TaskFrame(
        id=task_id,
        origin_edge="web",
        requester="rico",
        text=text,
        scope="single",
        created_at=datetime.now(UTC),
    )


async def test_recall_block_prepended(tmp_path: Path) -> None:
    """A non-empty recall prepends the bounded, delimited memory block."""
    engine, runner, bus = _make(tmp_path, _StubMemory(block="REMEMBERED FACT"))
    await runner.start()

    await bus.deliver(subjects.agent_task(_AGENT), _task(text="do the thing"))

    sent = engine.sends[-1].text
    assert sent.startswith(MEMORY_BLOCK_HEADER)
    assert "REMEMBERED FACT" in sent
    assert MEMORY_BLOCK_FOOTER in sent
    assert "do the thing" in sent


async def test_recall_failure_degrades_to_original_turn(tmp_path: Path) -> None:
    """A recall failure is contained: the original turn is sent unchanged."""
    engine, runner, bus = _make(tmp_path, _StubMemory(raise_on_recall=True))
    await runner.start()

    await bus.deliver(subjects.agent_task(_AGENT), _task(text="do the thing"))

    assert len(engine.sends) == 1
    assert engine.sends[-1].text == "do the thing"
    assert MEMORY_BLOCK_HEADER not in engine.sends[-1].text


async def test_empty_recall_leaves_turn_unchanged(tmp_path: Path) -> None:
    """An empty recall leaves the turn byte-identical to the render (no block)."""
    engine, runner, bus = _make(tmp_path, _StubMemory(block=""))
    await runner.start()

    await bus.deliver(subjects.agent_task(_AGENT), _task(text="do the thing"))

    assert engine.sends[-1].text == "do the thing"


async def test_no_memory_lane_default(tmp_path: Path) -> None:
    """No memory lane configured leaves the turn unchanged."""
    engine, runner, bus = _make(tmp_path, None)
    await runner.start()

    await bus.deliver(subjects.agent_task(_AGENT), _task(text="do the thing"))

    assert engine.sends[-1].text == "do the thing"
