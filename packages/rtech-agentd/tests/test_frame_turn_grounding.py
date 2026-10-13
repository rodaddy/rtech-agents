"""Focused evidence-gate proofs for the frame-turn runner."""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from rtech_agentd.engine import ToolsPolicy, ToolUse, TurnDone
from rtech_agentd.frame_turn import FrameTurnConfig, FrameTurnRunner
from rtech_agentd.grounding import GROUNDING_HEADER
from rtech_agentd.memory import AgentMemoryLane, MemoryRecallResult, MemoryRecallSource
from rtech_agentd.supervisor import Supervisor, SupervisorConfig
from rtech_frames import (
    ChannelOrigin,
    EvidenceSource,
    Stance,
    StatusFrame,
    TaskFrame,
    TurnFrame,
    VerificationReport,
)
from rtech_orchestrator import subjects

from .fakes import FakeEngine, FakeFrameBus, RecordingPublisher

_AGENT = "tester"


class _Memory:
    """Synchronous source-aware memory double with optional bounded blocking."""

    max_items = 8

    def __init__(
        self,
        *,
        block: str = "fact",
        block_until: threading.Event | None = None,
        raise_error: bool = False,
    ) -> None:
        self.block = block
        self.block_until = block_until
        self.raise_error = raise_error
        self.started = threading.Event()
        self.calls = 0
        self.timeout = 30.0

    def recall_with_sources(self, query: str, *, limit: int | None = None) -> MemoryRecallResult:
        self.calls += 1
        self.started.set()
        if self.raise_error:
            raise RuntimeError("OB unavailable")
        if self.block_until is not None:
            self.block_until.wait(timeout=1)
        return MemoryRecallResult(
            block=f"- [OB1] {self.block}",
            sources=(MemoryRecallSource(label="OB1", source="memory-note", kind="fact"),),
        )

    def bound_http_timeout(self, seconds: float) -> None:
        """Clamp the HTTP timeout DOWN, mirroring AgentMemoryLane."""
        if seconds > 0 and seconds < self.timeout:
            self.timeout = seconds

    def recall(self, query: str, *, limit: int | None = None) -> str:
        """Retain the legacy recall surface for the disabled-gate proof."""
        self.calls += 1
        return f"- [OB1] {self.block}"


def _make(
    tmp_path: Path,
    memory: object | None,
    *,
    config: FrameTurnConfig | None = None,
) -> tuple[FakeEngine, FrameTurnRunner, FakeFrameBus]:
    engine = FakeEngine()
    supervisor = Supervisor(
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
        supervisor=supervisor,
        bus=bus,
        config=config or FrameTurnConfig(evidence_gate_enabled=True),
        memory=cast("AgentMemoryLane | None", memory),
    )
    return engine, runner, bus


def _task(task_id: str, *, origin: ChannelOrigin | None = None) -> TaskFrame:
    return TaskFrame(
        id=task_id,
        origin_edge="web",
        requester="rico",
        text="What is the current fact?",
        scope="single",
        origin=origin,
        created_at=datetime.now(UTC),
    )


def _report(
    *,
    claim_kind: str = "factual",
    status: str = "partial",
    sources: list[EvidenceSource] | None = None,
) -> VerificationReport:
    return VerificationReport.model_validate(
        {
            "claim_kind": claim_kind,
            "status": status,
            "ob_checked": False,
            "sources": sources or [],
        }
    )


@pytest.mark.parametrize("field", ["conversation_idle_threshold_s", "ob_recall_deadline_s"])
def test_evidence_gate_timing_config_must_be_positive(field: str) -> None:
    with pytest.raises(ValueError):
        FrameTurnConfig(**{field: 0})


async def test_cold_ack_precedes_recall_and_dispatch(tmp_path: Path) -> None:
    release = threading.Event()
    memory = _Memory(block_until=release)
    engine, runner, bus = _make(tmp_path, memory)
    await runner.start()

    deliver = asyncio.create_task(bus.deliver(subjects.agent_task(_AGENT), _task("cold")))
    assert await asyncio.to_thread(memory.started.wait, 0.5)
    assert engine.sends == []
    assert isinstance(bus.published[-1][1], StatusFrame)
    assert cast("StatusFrame", bus.published[-1][1]).phase == "verifying"

    release.set()
    await deliver
    assert GROUNDING_HEADER in engine.sends[-1].text


async def test_every_turn_is_recalled_and_warm_turn_uses_responding_phase(tmp_path: Path) -> None:
    memory = _Memory()
    engine, runner, bus = _make(tmp_path, memory)
    await runner.start()
    origin = ChannelOrigin(channel_id="channel")

    await bus.deliver(subjects.agent_task(_AGENT), _task("first", origin=origin))
    await runner._on_engine_event(
        TurnDone(text="hello", verification=_report(claim_kind="social", status="not_applicable"))
    )
    await bus.deliver(subjects.agent_task(_AGENT), _task("warm", origin=origin))

    statuses = [frame for _, frame in bus.published if isinstance(frame, StatusFrame)]
    assert [status.phase for status in statuses] == ["verifying", "responding"]
    assert memory.calls == 2
    assert GROUNDING_HEADER in engine.sends[-1].text


async def test_timeout_is_recorded_as_a_typed_limitation(tmp_path: Path) -> None:
    release = threading.Event()
    memory = _Memory(block_until=release)
    config = FrameTurnConfig(evidence_gate_enabled=True, ob_recall_deadline_s=0.01)
    _, runner, bus = _make(tmp_path, memory, config=config)
    await runner.start()

    await bus.deliver(subjects.agent_task(_AGENT), _task("timeout"))
    await runner._on_engine_event(TurnDone(text="answer", verification=_report()))

    turn = cast("TurnFrame", bus.published[-1][1])
    assert turn.verification is not None
    assert "Open Brain recall timed out" in turn.verification.limitations
    assert not turn.verification.ob_checked
    assert "verification was insufficient" in turn.visible_text
    release.set()


async def test_recall_failure_is_recorded_as_a_typed_limitation(tmp_path: Path) -> None:
    memory = _Memory(raise_error=True)
    _, runner, bus = _make(tmp_path, memory)
    await runner.start()

    await bus.deliver(subjects.agent_task(_AGENT), _task("failure"))
    await runner._on_engine_event(TurnDone(text="answer", verification=_report()))

    turn = cast("TurnFrame", bus.published[-1][1])
    assert turn.verification is not None
    assert "Open Brain recall failed" in turn.verification.limitations
    assert not turn.verification.ob_checked


async def test_tool_names_merge_without_tool_inputs_and_fabricated_ob_is_dropped(
    tmp_path: Path,
) -> None:
    memory = _Memory()
    _, runner, bus = _make(tmp_path, memory)
    await runner.start()
    await bus.deliver(subjects.agent_task(_AGENT), _task("tools"))
    await runner._on_engine_event(
        ToolUse(tool_id="call", name="repo_search", tool_input={"secret": "nope"})
    )
    reported = _report(
        sources=[
            EvidenceSource(
                source_id="OB1", kind="open_brain", label="fabricated", authority="memory"
            ),
            EvidenceSource(source_id="OB999", kind="open_brain", label="fake", authority="memory"),
        ]
    )
    await runner._on_engine_event(TurnDone(text="answer", verification=reported))

    turn = cast("TurnFrame", bus.published[-1][1])
    assert turn.verification is not None
    assert [source.source_id for source in turn.verification.sources] == ["OB1"]
    assert turn.verification.sources[0].label == "memory-note"
    assert turn.verification.methods == ["repo_search"]
    assert "nope" not in str(turn.verification)


async def test_reported_non_ob_sources_require_observed_tool_and_safe_locator(
    tmp_path: Path,
) -> None:
    memory = _Memory()
    _, runner, bus = _make(tmp_path, memory)
    await runner.start()
    await bus.deliver(subjects.agent_task(_AGENT), _task("fabricated"))
    reported = _report(
        sources=[
            EvidenceSource(source_id="repo-unsafe", kind="repo", label="repo", locator="../secret"),
            EvidenceSource(
                source_id="url-http", kind="url", label="url", locator="http://example.test"
            ),
            EvidenceSource(source_id="live-empty", kind="live", label="live"),
            EvidenceSource(source_id="user", kind="user_context", label="Rico", authority="user"),
        ]
    )
    await runner._on_engine_event(TurnDone(text="answer", verification=reported))

    fabricated = cast("TurnFrame", bus.published[-1][1])
    assert fabricated.verification is not None
    assert fabricated.verification.sources == []
    assert fabricated.verification.status == "unverified"
    assert "verification was insufficient" in fabricated.visible_text

    await bus.deliver(subjects.agent_task(_AGENT), _task("observed"))
    await runner._on_engine_event(
        ToolUse(tool_id="call", name="repo_search", tool_input={"path": "packages/frames.py"})
    )
    await runner._on_engine_event(
        ToolUse(tool_id="c2", name="web_fetch", tool_input={"url": "https://example.test/docs"})
    )
    reported = _report(
        sources=[
            EvidenceSource(
                source_id="repo-ok", kind="repo", label="frames", locator="packages/frames.py"
            ),
            EvidenceSource(
                source_id="url-ok", kind="url", label="docs", locator="https://example.test/docs"
            ),
        ]
    )
    await runner._on_engine_event(TurnDone(text="answer", verification=reported))

    observed = cast("TurnFrame", bus.published[-1][1])
    assert observed.verification is not None
    assert [source.source_id for source in observed.verification.sources] == ["repo-ok", "url-ok"]
    assert observed.verification.methods == ["repo_search", "web_fetch"]


async def test_sibling_follow_up_gets_fresh_grounding_and_verification(tmp_path: Path) -> None:
    memory = _Memory()
    engine, runner, bus = _make(tmp_path, memory)
    await runner.start()
    task = _task("debate").model_copy(update={"scope": "duo", "participants": [_AGENT, "bob"]})
    await bus.deliver(subjects.agent_task(_AGENT), task)
    await runner._on_engine_event(
        TurnDone(
            text="round one",
            stance=Stance(position="opened", because="waiting", needs_from=["bob"]),
            verification=_report(claim_kind="social", status="not_applicable"),
        )
    )

    sibling = TurnFrame(
        agent_id="bob",
        task_id=task.id,
        visible_text="bob evidence",
        stance=Stance(position="reply", because="engaging", needs_from=[_AGENT]),
        refs=[task.id],
    )
    await bus.deliver(subjects.task_turn(task.id), sibling)
    assert memory.calls == 2
    assert runner._grounding[task.id].eligibility == "factual"
    assert GROUNDING_HEADER in engine.sends[-1].text
    await runner._on_engine_event(
        TurnDone(
            text="round two",
            stance=Stance(position="revised", because="bob", needs_from=["bob"]),
            verification=_report(
                sources=[EvidenceSource(source_id="OB1", kind="open_brain", label="fabricated")]
            ),
        )
    )

    turns = [frame for _, frame in bus.published if isinstance(frame, TurnFrame)]
    assert turns[-1].verification is not None
    assert turns[-1].verification.ob_checked
    assert [source.source_id for source in turns[-1].verification.sources] == ["OB1"]


async def test_sibling_fold_keeps_the_active_turn_grounding(tmp_path: Path) -> None:
    memory = _Memory()
    _, runner, bus = _make(tmp_path, memory)
    await runner.start()
    task = _task("fold").model_copy(update={"scope": "duo", "participants": [_AGENT, "bob"]})
    await bus.deliver(subjects.agent_task(_AGENT), task)
    active_grounding = runner._grounding[task.id]

    await bus.deliver(
        subjects.task_turn(task.id),
        TurnFrame(
            agent_id="bob",
            task_id=task.id,
            visible_text="bob evidence",
            stance=Stance(position="reply", because="engaging", needs_from=[_AGENT]),
            refs=[task.id],
        ),
    )

    assert memory.calls == 1
    assert runner._grounding[task.id] is active_grounding
    await runner._on_engine_event(
        TurnDone(
            text="round one",
            verification=_report(
                sources=[EvidenceSource(source_id="OB1", kind="open_brain", label="fabricated")]
            ),
        )
    )
    turn = cast("TurnFrame", bus.published[-1][1])
    assert turn.verification is not None
    assert turn.verification.ob_checked


async def test_cap_closing_turn_has_runtime_verification_when_gated(tmp_path: Path) -> None:
    _, runner, bus = _make(tmp_path, _Memory())
    await runner._publish_closing_turn(
        "cap",
        Stance(position="position", because="reason", needs_from=[]),
        None,
        None,
    )

    closing = cast("TurnFrame", bus.published[-1][1])
    assert closing.verification is not None
    assert closing.verification.status == "not_applicable"
    assert closing.verification.claim_kind == "action"
    assert closing.verification.methods == ["runtime_cap_closure"]


async def test_cold_factual_without_sources_falls_back_and_social_is_allowed(
    tmp_path: Path,
) -> None:
    memory = _Memory()
    _, runner, bus = _make(tmp_path, memory)
    await runner.start()
    await bus.deliver(subjects.agent_task(_AGENT), _task("factual"))
    await runner._on_engine_event(TurnDone(text="confident answer", verification=_report()))
    factual = cast("TurnFrame", bus.published[-1][1])
    assert "verification was insufficient" in factual.visible_text

    social_task = _task("social").model_copy(update={"text": "Hello!"})
    await bus.deliver(subjects.agent_task(_AGENT), social_task)
    await runner._on_engine_event(
        TurnDone(
            text="Hey there!",
            verification=_report(claim_kind="social", status="not_applicable"),
        )
    )
    social = cast("TurnFrame", bus.published[-1][1])
    assert social.visible_text == "Hey there!"


async def test_grounding_state_cleans_on_completion_and_send_failure(tmp_path: Path) -> None:
    memory = _Memory()
    _, runner, _ = _make(tmp_path, memory)
    await runner.start()
    await runner._on_task_frame(_task("complete"))
    await runner._on_engine_event(
        TurnDone(text="done", verification=_report(claim_kind="social", status="not_applicable"))
    )
    assert runner._grounding == {}

    engine, failing_runner, _ = _make(tmp_path, memory)

    async def _fail_send(text: str) -> None:
        raise RuntimeError("send failed")

    engine.send = _fail_send  # type: ignore[method-assign]
    await failing_runner.start()
    await failing_runner._on_task_frame(_task("failed"))
    assert failing_runner._grounding == {}


async def test_legacy_gate_disabled_remains_compatible(tmp_path: Path) -> None:
    memory = _Memory()
    engine, runner, bus = _make(tmp_path, memory, config=FrameTurnConfig())
    await runner.start()
    await bus.deliver(subjects.agent_task(_AGENT), _task("legacy"))

    status = cast("StatusFrame", bus.published[-1][1])
    assert status.phase is None
    assert GROUNDING_HEADER not in engine.sends[-1].text
    assert "[recalled memory" in engine.sends[-1].text


async def test_only_stance_tool_drops_reported_repo_source_end_to_end(tmp_path: Path) -> None:
    memory = _Memory()
    _, runner, bus = _make(tmp_path, memory)
    await runner.start()
    await bus.deliver(subjects.agent_task(_AGENT), _task("stance-only"))
    await runner._on_engine_event(
        ToolUse(tool_id="call", name="mcp__rtech_stance__stance", tool_input={})
    )
    reported = _report(
        sources=[
            EvidenceSource(
                source_id="repo-ok", kind="repo", label="frames", locator="packages/frames.py"
            )
        ]
    )
    await runner._on_engine_event(TurnDone(text="answer", verification=reported))

    dropped = cast("TurnFrame", bus.published[-1][1])
    assert dropped.verification is not None
    assert dropped.verification.sources == []

    await bus.deliver(subjects.agent_task(_AGENT), _task("stance-and-search"))
    await runner._on_engine_event(
        ToolUse(
            tool_id="c1",
            name="repo_search",
            tool_input={"path": "packages/frames.py"},
        )
    )
    await runner._on_engine_event(
        ToolUse(tool_id="c2", name="mcp__rtech_stance__stance", tool_input={})
    )
    reported = _report(
        sources=[
            EvidenceSource(
                source_id="repo-ok", kind="repo", label="frames", locator="packages/frames.py"
            )
        ]
    )
    await runner._on_engine_event(TurnDone(text="answer", verification=reported))

    survived = cast("TurnFrame", bus.published[-1][1])
    assert survived.verification is not None
    assert [source.source_id for source in survived.verification.sources] == ["repo-ok"]


async def test_recall_bounds_http_timeout_to_deadline_on_dispatch(tmp_path: Path) -> None:
    memory = _Memory()
    config = FrameTurnConfig(evidence_gate_enabled=True, ob_recall_deadline_s=2.0)
    _, runner, bus = _make(tmp_path, memory, config=config)
    await runner.start()

    await bus.deliver(subjects.agent_task(_AGENT), _task("bound"))

    assert memory.timeout <= config.ob_recall_deadline_s
