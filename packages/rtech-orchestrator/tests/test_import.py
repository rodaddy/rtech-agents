"""Assert the orchestrator public surface is importable."""

from __future__ import annotations

import rtech_orchestrator as pkg


def test_public_symbols_exist() -> None:
    """Every advertised public symbol must be present."""
    for name in (
        "FrameBus",
        "FrameHandler",
        "GraphStore",
        "InMemoryGraphStore",
        "Orchestrator",
        "OrchestratorConfig",
        "TargetResolutionError",
        "TaskGraph",
        "resolve_target_agent_id",
        "subjects",
    ):
        assert hasattr(pkg, name), name
    assert pkg.__version__ == "0.0.1"
