"""Live smoke test against a real ``codex app-server`` (opt-in, auto-skip).

Gated behind ``RTECH_CODEX_LIVE=1`` AND a resolvable ``codex`` CLI — mirroring
the fleet-bus live-bus pattern (auto-skip off the target). It drives a real
child through the full contract: start → a turn that must PONG → resume-recall,
proving the handshake, threadId capture, event stream, and cross-process resume
against the actual binary. Kept out of the default unit gate so CI stays offline.
"""

from __future__ import annotations

import os

import pytest
from rtech_agentd.codex_engine import CodexEngine
from rtech_agentd.config import EngineConfig
from rtech_agentd.engine import ToolsPolicy, TurnDone
from rtech_agentd.engine_select import find_codex_cli

_LIVE = os.getenv("RTECH_CODEX_LIVE") == "1" and find_codex_cli() is not None

pytestmark = pytest.mark.skipif(
    not _LIVE,
    reason="set RTECH_CODEX_LIVE=1 with an authenticated `codex` CLI to run the live smoke",
)


async def _run_turn_until_done(engine: CodexEngine) -> TurnDone:
    """Consume the event stream until the current turn reaches TurnDone."""
    async for event in engine.events():
        if isinstance(event, TurnDone):
            return event
    raise AssertionError("stream ended before TurnDone")


@pytest.mark.asyncio
async def test_live_codex_start_turn_and_resume() -> None:
    model = os.getenv("RTECH_CODEX_MODEL", "gpt-5.6-sol")
    cfg = EngineConfig(kind="codex", model=model, litellm_base=None, backup_model=None)
    engine = CodexEngine(cfg)
    session_ref: str | None = None
    try:
        await engine.start(
            "You are a terse test agent. When asked to say PONG, reply with exactly PONG.",
            ToolsPolicy(),
        )
        await engine.send("Say PONG.")
        done = await _run_turn_until_done(engine)
        assert "PONG" in done.text.upper()
        session_ref = done.session_ref
        assert session_ref
    finally:
        await engine.close()

    # Cross-process resume: a fresh child recalls the same thread.
    resumed = CodexEngine(cfg)
    try:
        assert session_ref is not None
        await resumed.resume(session_ref)
        health = await resumed.health()
        assert health.session_ref == session_ref
    finally:
        await resumed.close()
