"""rtech-agentd: per-agent supervisor daemon (managed engine session + NATS).

Phases 2-3 of the governing spec: a managed, inline-steerable engine session
behind an ``EngineSession`` protocol, a supervisor that owns its lifecycle
(resume / heartbeat / stall watchdog), a NATS ingress that maps fleet-bus
envelopes to turns / steers / interrupts and publishes guarded replies, and the
outbound guard on the reply path.

Public surface:
    engine          - EngineSession protocol + typed EngineEvent union.
    claude_engine   - ClaudeEngine: EngineSession over the Claude Agent SDK.
    engine_select   - build_engine: one engine-selection policy shared by callers.
    supervisor      - Supervisor + SessionRegistry.
    nats_ingress    - NatsIngress: fleet-bus consumer/publisher.
    frame_turn      - FrameTurnRunner: typed-frame managed-turn seam (TaskFrame
                      in, TurnFrame out).
    stance_tool     - structured stance elicitation.
    outbound_guard  - sanitize_outbound_text (reply-path guard).
    config          - load_agent_config + typed AgentConfig.
    persona_glue    - build_system_prompt (rtech_persona composition).
    memory          - AgentMemoryLane: Open Brain memory lane (Phase 4).
"""

from __future__ import annotations

from .engine import (
    EngineError,
    EngineEvent,
    EngineHealth,
    EngineSession,
    TextDelta,
    ToolsPolicy,
    ToolUse,
    TurnDone,
)
from .engine_select import (
    EngineSelectionError,
    build_engine,
    find_claude_cli,
    find_codex_cli,
)
from .frame_turn import FrameTurnRunner
from .memory import (
    CANONICAL_BASE_URL,
    REQUIRED_CONTRACT_VERSION,
    AgentMemoryLane,
    MemoryConfig,
    MemoryContractError,
    MemoryError,
    MemoryRedactionError,
    MemoryWriteResult,
)

__version__ = "0.0.1"

__all__ = [
    "CANONICAL_BASE_URL",
    "REQUIRED_CONTRACT_VERSION",
    "AgentMemoryLane",
    "EngineError",
    "EngineEvent",
    "EngineHealth",
    "EngineSelectionError",
    "EngineSession",
    "FrameTurnRunner",
    "MemoryConfig",
    "MemoryContractError",
    "MemoryError",
    "MemoryRedactionError",
    "MemoryWriteResult",
    "TextDelta",
    "ToolUse",
    "ToolsPolicy",
    "TurnDone",
    "__version__",
    "build_engine",
    "find_claude_cli",
    "find_codex_cli",
]
