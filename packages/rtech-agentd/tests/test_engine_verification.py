"""Prove typed verification stays structural at the engine boundary."""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import AssistantMessage, ResultMessage, ToolUseBlock
from rtech_agentd.claude_engine import ClaudeEngine
from rtech_agentd.config import EngineConfig
from rtech_agentd.engine import ToolUse, TurnDone, unverified_verification
from rtech_agentd.stance_tool import (
    _STANCE_INPUT_SCHEMA,
    STANCE_QUALIFIED_TOOL,
    stance_from_tool_input,
)


def _verification() -> dict[str, Any]:
    return {
        "status": "grounded",
        "claim_kind": "factual",
        "ob_checked": True,
        "sources": [
            {
                "source_id": "OB1",
                "kind": "open_brain",
                "label": "current task context",
                "locator": "entity-1",
                "authority": "memory",
            }
        ],
        "limitations": [],
        "methods": ["open-brain"],
    }


def _stance_input(**overrides: object) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "position": "answered",
        "because": "checked the supplied context",
        "needs_from": [],
        "verification": _verification(),
    }
    payload.update(overrides)
    return payload


def _engine() -> ClaudeEngine:
    return ClaudeEngine(
        EngineConfig(kind="claude", model="claude-x", litellm_base=None, backup_model=None)
    )


def _result(session_id: str) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id=session_id,
    )


def test_turn_done_default_is_deterministic_unverified() -> None:
    first = unverified_verification()
    second = unverified_verification()
    assert first == second
    assert first.status == "unverified"
    assert first.claim_kind == "mixed"
    assert first.ob_checked is False
    assert first.limitations == ["verification was not reported"]


def test_stance_tool_schema_and_capture_round_trip_verification() -> None:
    assert _STANCE_INPUT_SCHEMA["verification"] is dict
    captured = stance_from_tool_input(_stance_input())
    assert captured is not None
    stance, verification = captured
    assert stance.position == "answered"
    assert verification.status == "grounded"
    assert verification.sources[0].source_id == "OB1"


def test_stance_tool_malformed_verification_falls_back_without_coercion() -> None:
    malformed = {**_verification(), "methods": [1]}
    captured = stance_from_tool_input(_stance_input(verification=malformed))
    assert captured is not None
    _stance, verification = captured
    assert verification == unverified_verification()


async def test_claude_captures_and_resets_pending_verification_per_turn() -> None:
    engine = _engine()
    text_buf: list[str] = []
    await engine._handle_message(  # noqa: SLF001 - exercise the typed SDK reader seam
        AssistantMessage(
            content=[
                ToolUseBlock(
                    id="tool-1",
                    name=STANCE_QUALIFIED_TOOL,
                    input=_stance_input(),
                )
            ],
            model="claude-x",
        ),
        text_buf,
    )
    tool_use = await engine._queue.get()  # noqa: SLF001 - inspect observability event
    assert isinstance(tool_use, ToolUse)
    assert tool_use.tool_input == {}
    await engine._handle_message(_result("session-1"), text_buf)  # noqa: SLF001
    first = await engine._queue.get()  # noqa: SLF001
    assert isinstance(first, TurnDone)
    assert first.verification.status == "grounded"
    assert engine._pending_stance is None  # noqa: SLF001 - reset is the contract
    assert engine._pending_verification is None  # noqa: SLF001 - reset with stance

    await engine._handle_message(_result("session-1"), text_buf)  # noqa: SLF001
    second = await engine._queue.get()  # noqa: SLF001
    assert isinstance(second, TurnDone)
    assert second.verification == unverified_verification()
