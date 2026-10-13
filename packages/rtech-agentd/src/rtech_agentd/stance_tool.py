"""Structured stance elicitation for the managed Claude turn.

``TurnFrame.stance`` is a typed field the engine fills — NEVER a value scraped
from turn prose. To fill it honestly the engine gives the model a single
in-process SDK MCP tool, ``stance``, and asks it to call that tool once at the
end of every turn with its typed position. The tool's ``input`` (a real
structured object the SDK hands us) is captured verbatim and rides straight onto
``TurnDone.stance``. No regex, no trailer parsing, no prose scraping — the stance
is exactly the typed arguments the model passed to a tool.

This module owns:

* :data:`STANCE_TOOL_NAME` / :data:`STANCE_QUALIFIED_TOOL` — the tool identity
  the reader matches on and the allow-list entry the SDK needs.
* :func:`build_stance_server` — construct the SDK MCP server carrying the tool.
* :func:`stance_from_tool_input` — validate a captured tool ``input`` dict into a
  typed :class:`~rtech_frames.Stance` (fails safe to ``None`` on bad shape).
* :data:`STANCE_DIRECTIVE` — the minimal system-prompt instruction appended to
  the composed persona prompt so the model knows to call the tool.
"""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool
from claude_agent_sdk.types import McpSdkServerConfig
from loguru import logger
from pydantic import ValidationError
from rtech_frames import Stance, VerificationReport

from .engine import unverified_verification

STANCE_SERVER_NAME = "rtech_stance"
STANCE_TOOL_NAME = "stance"
STANCE_QUALIFIED_TOOL = f"mcp__{STANCE_SERVER_NAME}__{STANCE_TOOL_NAME}"

STANCE_DIRECTIVE = (
    "\n\n---\n"
    "At the END of every turn, after your visible reply, call the `stance` tool "
    "exactly once to record your typed stance on this task:\n"
    "- position: a short label for where you landed (e.g. 'answered', "
    "'need-input', 'blocked').\n"
    "- because: one clause explaining that position.\n"
    "- needs_from: names of other agents you still need something from; leave "
    "empty when your turn stands on its own.\n"
    "- verification: the evidence report for this answer: status, claim_kind, "
    "ob_checked, sources, limitations, and methods. Use empty lists when none.\n"
    "This is a structured signal for the coordination bus, not part of your "
    "visible reply. Never write the stance as prose in your message."
)

_STANCE_INPUT_SCHEMA: dict[str, Any] = {
    "position": str,
    "because": str,
    "needs_from": list,
    "verification": dict,
}


def stance_from_tool_input(raw: dict[str, Any]) -> tuple[Stance, VerificationReport] | None:
    """Validate structured stance and verification from one tool input.

    Args:
        raw: The ``input`` object the model passed to the ``stance`` tool.

    Returns:
        A validated ``(Stance, VerificationReport)`` pair, or ``None`` when the
        stance itself is malformed. Missing/malformed verification retains the
        valid stance and gets the deterministic ``unverified`` fallback.
    """
    try:
        position = raw["position"]
        because = raw["because"]
        needs = raw["needs_from"]
        if (
            not isinstance(position, str)
            or not position.strip()
            or not isinstance(because, str)
            or not because.strip()
            or not isinstance(needs, list)
            or any(not isinstance(need, str) or not need.strip() for need in needs)
        ):
            raise ValueError("malformed stance shape")
        stance = Stance(
            position=position.strip(),
            because=because.strip(),
            needs_from=needs,
        )
    except (KeyError, TypeError, ValueError, ValidationError):
        logger.warning("stance_tool.parse | malformed stance input, ignoring")
        return None
    verification_raw = raw.get("verification")
    try:
        verification = VerificationReport.model_validate(verification_raw, strict=True)
    except ValidationError:
        logger.warning("stance_tool.parse | malformed verification input; using default")
        verification = unverified_verification()
    return stance, verification


def build_stance_server() -> McpSdkServerConfig:
    """Build the in-process SDK MCP server exposing the ``stance`` tool.

    The tool handler is a no-op acknowledgement: the value is the *typed input*
    the model supplies, which the engine reader captures from the tool-use block.
    The handler just closes the tool call so the model's turn can complete.
    """

    @tool(
        STANCE_TOOL_NAME,
        "Record your typed stance and verification for this task. "
        "Call once at the end of every turn.",
        _STANCE_INPUT_SCHEMA,
    )
    async def _stance(args: dict[str, Any]) -> dict[str, Any]:
        logger.debug("stance_tool.invoked")
        return {"content": [{"type": "text", "text": "stance recorded"}]}

    return create_sdk_mcp_server(name=STANCE_SERVER_NAME, tools=[_stance])
