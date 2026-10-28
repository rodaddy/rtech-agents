"""In-process SDK MCP tool exposing the wake capability to the Claude engine.

Mirrors :mod:`rtech_agentd.stance_tool` exactly: a single in-process
:func:`~claude_agent_sdk.create_sdk_mcp_server` tool the managed Claude session
may call. Where the ``stance`` tool is passive (its typed input is captured by the
engine reader), ``wake_agent`` is ACTIVE — its handler calls the injected
:class:`~rtech_agentd.wake_service.WakeService`, which mints + publishes a fresh
:class:`~rtech_frames.TaskFrame` toward an allow-listed teammate and returns the
minted ``task_id``. That id is handed straight back as the tool result so the
triage model (Nimbus) can reference it in her visible reply.

This module owns:

* :data:`WAKE_SERVER_NAME` / :data:`WAKE_QUALIFIED_TOOL` — the tool identity and
  the allow-list entry the SDK needs to permit the call.
* :func:`build_wake_server` — construct the SDK MCP server, binding the tool
  handler to a :class:`WakeService` instance.
* :data:`WAKE_DIRECTIVE` — the minimal system-prompt instruction appended to the
  composed persona prompt so the triage model knows the wake tool exists and when
  to use it.

The wake server is registered on the Claude engine ONLY when a
:class:`WakeService` is present (the allow-list opt-in). A non-waker agent never
sees the tool. Codex has NO wake plumbing in this slice (conservative; additive
later) — see :mod:`rtech_agentd.codex_engine`.
"""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool
from claude_agent_sdk.types import McpSdkServerConfig
from loguru import logger

from .wake_service import WakeError, WakeOutcome, WakeService

WAKE_SERVER_NAME = "rtech_wake"
WAKE_TOOL_NAME = "wake_agent"
WAKE_QUALIFIED_TOOL = f"mcp__{WAKE_SERVER_NAME}__{WAKE_TOOL_NAME}"

WAKE_DIRECTIVE = (
    "\n\n---\n"
    "You are the triage agent. When a question is better answered by a teammate, "
    "call the `wake_agent` tool to dispatch it to them:\n"
    "- agent_id: the teammate's agent id (e.g. 'agentd-birk', 'agentd-sprocket'). "
    "Only ids you are configured to wake are permitted; others are refused.\n"
    "- task_text: the exact question/task the teammate should answer.\n"
    "- reason: one short clause on why this teammate is the right one.\n"
    "The tool replies with the minted task id (or a typed error). Reference that "
    "task id in your visible reply so the human knows who you handed it to. If a "
    "wake is refused, answer as yourself instead — never claim a teammate was "
    "woken when the tool returned an error."
)

_WAKE_INPUT_SCHEMA: dict[str, Any] = {
    "agent_id": str,
    "task_text": str,
    "reason": str,
}


def build_wake_server(wake_service: WakeService) -> McpSdkServerConfig:
    """Build the in-process SDK MCP server exposing the ``wake_agent`` tool.

    The handler calls :meth:`WakeService.wake`, which enforces the allow-list,
    mints + publishes the wake frame, and returns a typed outcome. The outcome is
    rendered into the tool result text so the model sees either the minted task id
    or a clear, typed refusal — the wake never raises into the turn.

    Args:
        wake_service: The service instance that mints + publishes wake frames for
            this triage agent (carries the allow-list, stack, and origin provider).

    Returns:
        A configured SDK MCP server carrying the single ``wake_agent`` tool.
    """

    @tool(
        WAKE_TOOL_NAME,
        "Wake an allow-listed teammate: dispatch a task to them and get back the "
        "minted task id. Refused for non-allow-listed or self ids.",
        _WAKE_INPUT_SCHEMA,
    )
    async def _wake_agent(args: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(args.get("agent_id") or "")
        task_text = str(args.get("task_text") or "")
        reason = str(args.get("reason") or "")
        logger.debug("wake_tool.invoked | target={} reason={}", agent_id, reason)
        outcome = await wake_service.wake(agent_id, task_text, reason)
        return {"content": [{"type": "text", "text": _render_outcome(outcome)}]}

    return create_sdk_mcp_server(name=WAKE_SERVER_NAME, tools=[_wake_agent])


def _render_outcome(outcome: WakeOutcome) -> str:
    """Render a wake outcome into the tool-result text the model reads.

    A success returns the minted task id (referenceable in the reply); a typed
    error returns a clear refusal string keyed on the closed ``reason`` so the
    model never fabricates a wake that did not happen. Branching on both concrete
    types keeps this exhaustive under ``mypy --strict``.
    """
    if isinstance(outcome, WakeError):
        return f"WAKE REFUSED ({outcome.reason}): {outcome.detail}"
    return f"WOKE {outcome.agent_id} — task_id={outcome.task_id}"


__all__ = [
    "WAKE_DIRECTIVE",
    "WAKE_QUALIFIED_TOOL",
    "WAKE_SERVER_NAME",
    "WAKE_TOOL_NAME",
    "build_wake_server",
]
