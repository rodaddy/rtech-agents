"""Cover orchestrator target/stack resolution: env, registry, loud unset failure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rtech_orchestrator.config import (
    StackResolutionError,
    TargetResolutionError,
    resolve_stack_id,
    resolve_target_agent_id,
)


def test_target_env_wins() -> None:
    """RTECH_TARGET_AGENT_ID is the primary source."""
    assert resolve_target_agent_id(env={"RTECH_TARGET_AGENT_ID": "agentd-bilby"}) == "agentd-bilby"


def test_single_agent_id_is_backcompat_alias() -> None:
    """RTECH_SINGLE_AGENT_ID still resolves when the primary var is absent."""
    resolved = resolve_target_agent_id(env={"RTECH_SINGLE_AGENT_ID": "agentd-nagatha"})
    assert resolved == "agentd-nagatha"


def test_target_env_beats_alias() -> None:
    """The primary var wins over the back-compat alias when both are set."""
    env = {"RTECH_TARGET_AGENT_ID": "agentd-bilby", "RTECH_SINGLE_AGENT_ID": "agentd-old"}
    assert resolve_target_agent_id(env=env) == "agentd-bilby"


def test_registry_resolves_single_entry(tmp_path: Path) -> None:
    """A JSON registry supplies the target when no env var is set."""
    reg = tmp_path / "agents.json"
    reg.write_text(json.dumps({"single": "agentd-bilby"}), encoding="utf-8")
    assert resolve_target_agent_id(env={"RTECH_AGENT_REGISTRY": str(reg)}) == "agentd-bilby"


def test_unset_fails_loudly() -> None:
    """No env var and no registry raises rather than defaulting to a shared id."""
    with pytest.raises(TargetResolutionError):
        resolve_target_agent_id(env={})


def test_blank_target_fails_loudly() -> None:
    """A whitespace-only target is treated as unset and fails loudly."""
    with pytest.raises(TargetResolutionError):
        resolve_target_agent_id(env={"RTECH_TARGET_AGENT_ID": "   "})


def test_missing_registry_file_fails_loudly(tmp_path: Path) -> None:
    """A registry path that does not exist is not a silent fallback."""
    missing = tmp_path / "nope.json"
    with pytest.raises(TargetResolutionError):
        resolve_target_agent_id(env={"RTECH_AGENT_REGISTRY": str(missing)})


def test_registry_without_single_key_fails_loudly(tmp_path: Path) -> None:
    """A registry lacking a non-empty 'single' entry fails loudly."""
    reg = tmp_path / "agents.json"
    reg.write_text(json.dumps({"duo": ["a", "b"]}), encoding="utf-8")
    with pytest.raises(TargetResolutionError):
        resolve_target_agent_id(env={"RTECH_AGENT_REGISTRY": str(reg)})


def test_stack_id_resolves_from_env() -> None:
    """RTECH_STACK_ID supplies the ingress-guard stack id (issue #75)."""
    assert resolve_stack_id(env={"RTECH_STACK_ID": "mini-m4-dev"}) == "mini-m4-dev"


def test_stack_id_unset_fails_loudly() -> None:
    """No stack id means no ingress guard: fail closed at startup, no default."""
    with pytest.raises(StackResolutionError):
        resolve_stack_id(env={})


def test_stack_id_blank_fails_loudly() -> None:
    """A whitespace-only stack id is treated as unset and fails loudly."""
    with pytest.raises(StackResolutionError):
        resolve_stack_id(env={"RTECH_STACK_ID": "   "})


def test_bridged_agents_default_empty() -> None:
    from rtech_orchestrator import resolve_bridged_agents

    assert resolve_bridged_agents({}) == frozenset()


def test_bridged_agents_parse_and_strip() -> None:
    from rtech_orchestrator import resolve_bridged_agents

    resolved = resolve_bridged_agents({"RTECH_BRIDGED_AGENTS": " bilby , skippy "})
    assert resolved == frozenset({"bilby", "skippy"})


def test_bridged_agents_blank_fragment_rejected() -> None:
    import pytest
    from rtech_orchestrator import resolve_bridged_agents

    with pytest.raises(ValueError, match="blank id fragments"):
        resolve_bridged_agents({"RTECH_BRIDGED_AGENTS": "bilby,,skippy"})
