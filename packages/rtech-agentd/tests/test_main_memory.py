"""_build_memory_lane: token-gated, offline-safe OB recall lane wiring (Phase 4)."""

from __future__ import annotations

import pytest
from rtech_agentd.__main__ import _build_memory_lane
from rtech_agentd.memory import AgentMemoryLane


def test_missing_token_disables_recall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENBRAIN_TOKEN", raising=False)
    assert _build_memory_lane("agentd-tester") is None


def test_blank_token_disables_recall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENBRAIN_TOKEN", "   ")
    assert _build_memory_lane("agentd-tester") is None


def test_token_enables_recall_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENBRAIN_TOKEN", "test-token")
    monkeypatch.setenv("OPENBRAIN_NAMESPACE", "ns.test")
    result = _build_memory_lane("agentd-tester")
    assert result is not None
    assert isinstance(result, AgentMemoryLane)


def test_per_agent_suffixed_token_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    # __main__ now routes through the shared factory, so a per-agent suffixed var
    # (OPENBRAIN_TOKEN_TESTER for agentd-tester) is picked up from the process env
    # even with no global OPENBRAIN_TOKEN set.
    monkeypatch.delenv("OPENBRAIN_TOKEN", raising=False)
    monkeypatch.setenv("OPENBRAIN_TOKEN_TESTER", "per-agent-token")
    result = _build_memory_lane("agentd-tester")
    assert result is not None
    assert isinstance(result, AgentMemoryLane)
