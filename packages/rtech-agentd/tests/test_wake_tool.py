"""Wake MCP tool: schema shape, outcome rendering, and engine exposure gating.

The ``wake_agent`` tool mirrors the ``stance`` tool: an in-process SDK MCP server
registered on the Claude engine. These tests pin the tool identity/schema, the
outcome-render text the model reads, and the exposure gate — the wake tool is
registered + allow-listed ONLY when a WakeService is attached (the opt-in), and
NEVER otherwise.
"""

from __future__ import annotations

from rtech_agentd.claude_engine import ClaudeEngine
from rtech_agentd.config import EngineConfig
from rtech_agentd.engine import ToolsPolicy
from rtech_agentd.stance_tool import STANCE_QUALIFIED_TOOL
from rtech_agentd.wake_service import WakeConfig, WakeError, WakeResult, WakeService
from rtech_agentd.wake_tool import (
    _WAKE_INPUT_SCHEMA,
    WAKE_QUALIFIED_TOOL,
    WAKE_SERVER_NAME,
    WAKE_TOOL_NAME,
    _render_outcome,
    build_wake_server,
)

from .fakes import FakeFrameBus

_TRIAGE = "agentd-nagatha"
_BILBY = "agentd-bilby"


def _service(bus: FakeFrameBus | None = None) -> WakeService:
    return WakeService(
        config=WakeConfig(triage_agent=_TRIAGE, allowed_agents=frozenset({_BILBY}), stack="dev"),
        bus=bus or FakeFrameBus(),
        origin_provider=lambda: None,
        id_factory=lambda: "tid-1",
    )


def _engine(*, wake: bool) -> ClaudeEngine:
    cfg = EngineConfig(kind="claude", model="claude-x", litellm_base=None, backup_model=None)
    engine = ClaudeEngine(cfg)
    if wake:
        engine.set_wake_service(_service())
    return engine


def test_tool_identity_and_qualified_name() -> None:
    assert WAKE_TOOL_NAME == "wake_agent"
    assert WAKE_QUALIFIED_TOOL == f"mcp__{WAKE_SERVER_NAME}__{WAKE_TOOL_NAME}"


def test_tool_input_schema_shape() -> None:
    # The typed tool input the model fills — three string fields.
    assert _WAKE_INPUT_SCHEMA == {"agent_id": str, "task_text": str, "reason": str}


def test_build_wake_server_names_the_server() -> None:
    server = build_wake_server(_service())
    assert server["type"] == "sdk"
    assert server["name"] == WAKE_SERVER_NAME


def test_render_success_carries_task_id() -> None:
    text = _render_outcome(WakeResult(agent_id=_BILBY, task_id="tid-9"))
    assert "tid-9" in text
    assert _BILBY in text


def test_render_error_is_a_clear_refusal() -> None:
    text = _render_outcome(WakeError(reason="not_allowed", detail="nope"))
    assert "REFUSED" in text
    assert "not_allowed" in text


def test_wake_tool_exposed_when_service_attached() -> None:
    engine = _engine(wake=True)
    options = engine._build_options("SYS", ToolsPolicy())
    assert WAKE_QUALIFIED_TOOL in options.allowed_tools
    assert WAKE_SERVER_NAME in options.mcp_servers
    # Stance is always present; wake is additive alongside it.
    assert STANCE_QUALIFIED_TOOL in options.allowed_tools


def test_wake_tool_NOT_exposed_without_service() -> None:
    engine = _engine(wake=False)
    options = engine._build_options("SYS", ToolsPolicy())
    assert WAKE_QUALIFIED_TOOL not in options.allowed_tools
    assert WAKE_SERVER_NAME not in options.mcp_servers
    # The stance tool is still there — only wake is gated on the opt-in.
    assert STANCE_QUALIFIED_TOOL in options.allowed_tools


def test_set_wake_service_rejected_after_start_guard() -> None:
    # A live client blocks late attach (the tool registers only at start()).
    engine = _engine(wake=False)
    engine._client = object()  # type: ignore[assignment]
    try:
        engine.set_wake_service(_service())
    except RuntimeError as exc:
        assert "before start()" in str(exc)
    else:  # pragma: no cover - the guard must raise
        raise AssertionError("expected set_wake_service to reject a live session")


def test_resume_re_registers_wake_and_stance_tools() -> None:
    # A resumed session must keep the in-process tools — otherwise a resumed
    # triage agent silently loses wake (and stance) on restart/recovery.
    engine = _engine(wake=True)
    mcp_servers, allowed_tools, _directive = engine._tool_registration()
    assert WAKE_SERVER_NAME in mcp_servers
    assert WAKE_QUALIFIED_TOOL in allowed_tools
    assert STANCE_QUALIFIED_TOOL in allowed_tools


async def test_tool_handler_end_to_end_publishes_and_renders() -> None:
    # Drive the actual SDK MCP tool handler: args -> WakeService.wake -> publish
    # -> rendered task id. Guards against arg-extraction / result-shaping drift.
    from mcp.types import CallToolRequest, CallToolRequestParams
    from rtech_frames import TaskFrame

    bus = FakeFrameBus()
    server = build_wake_server(_service(bus))
    handler = server["instance"].request_handlers[CallToolRequest]
    request = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(
            name=WAKE_TOOL_NAME,
            arguments={"agent_id": _BILBY, "task_text": "q", "reason": "r"},
        ),
    )

    result = await handler(request)

    assert "tid-1" in result.root.content[0].text
    assert len(bus.by_type(TaskFrame)) == 1
