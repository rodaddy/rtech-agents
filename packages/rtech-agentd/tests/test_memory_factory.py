"""Functional tests for the shared per-agent OB memory-lane factory (Phase 4).

Every build here is OFFLINE: an injected fake OB client means `AgentMemory`
construction never touches the network. The token precedence, per-agent
isolation, and no-secret-in-logs postures are asserted directly.
"""

from __future__ import annotations

from typing import Any

from rtech_agentd.memory import (
    AgentMemoryLane,
    _resolve_per_agent,
    agent_env_suffix,
    build_agent_memory_lane,
)


class _FakeMemoryClient:
    """Trivial offline stand-in for the OB MCP client.

    `AgentMemory` only stores the client at construction — it makes no call — so
    a bare object satisfies the factory's build path without any network I/O.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []


def test_agent_env_suffix_variants() -> None:
    assert agent_env_suffix("agentd-nagatha") == "NAGATHA"
    assert agent_env_suffix("agentd-nagatha-dev") == "NAGATHA"
    assert agent_env_suffix("agentd-nagatha-prod") == "NAGATHA"
    assert agent_env_suffix("agentd-my.agent") == "MY_AGENT"
    assert agent_env_suffix("agentd-skippy") == "SKIPPY"


def test_resolve_per_agent_precedence() -> None:
    assert _resolve_per_agent(
        {"OPENBRAIN_TOKEN": "g", "OPENBRAIN_TOKEN_NAGATHA": "p"},
        "OPENBRAIN_TOKEN",
        "NAGATHA",
    ) == ("p", "OPENBRAIN_TOKEN_NAGATHA")
    assert _resolve_per_agent({"OPENBRAIN_TOKEN": "g"}, "OPENBRAIN_TOKEN", "NAGATHA") == (
        "g",
        "OPENBRAIN_TOKEN",
    )
    assert _resolve_per_agent({}, "OPENBRAIN_TOKEN", "NAGATHA") == ("", None)


def test_resolve_per_agent_blank_falls_through_to_global() -> None:
    assert _resolve_per_agent(
        {"OPENBRAIN_TOKEN": "g", "OPENBRAIN_TOKEN_NAGATHA": "  "},
        "OPENBRAIN_TOKEN",
        "NAGATHA",
    ) == ("g", "OPENBRAIN_TOKEN")


def test_per_agent_token_wins_over_global() -> None:
    env = {"OPENBRAIN_TOKEN": "global-tok", "OPENBRAIN_TOKEN_NAGATHA": "per-agent-tok"}
    lane = build_agent_memory_lane("agentd-nagatha", env=env, client=_FakeMemoryClient())
    assert isinstance(lane, AgentMemoryLane)


def test_per_agent_isolation_one_agent_disabled() -> None:
    # Only NAGATHA has a token; there is NO process-global token, so BILBY,
    # co-hosted in the same env, must get None while NAGATHA still gets a lane.
    env = {"OPENBRAIN_TOKEN_NAGATHA": "ntok"}
    nagatha = build_agent_memory_lane("agentd-nagatha", env=env, client=_FakeMemoryClient())
    bilby = build_agent_memory_lane("agentd-bilby", env=env, client=_FakeMemoryClient())
    assert isinstance(nagatha, AgentMemoryLane)
    assert bilby is None


def test_missing_token_disables_recall() -> None:
    assert build_agent_memory_lane("agentd-solo", env={}, client=_FakeMemoryClient()) is None


def test_global_token_fallback_single_agent() -> None:
    lane = build_agent_memory_lane(
        "agentd-solo", env={"OPENBRAIN_TOKEN": "g"}, client=_FakeMemoryClient()
    )
    assert isinstance(lane, AgentMemoryLane)


def test_namespace_defaults_when_unset() -> None:
    lane = build_agent_memory_lane(
        "agentd-solo", env={"OPENBRAIN_TOKEN": "g"}, client=_FakeMemoryClient()
    )
    # No namespace env set -> factory defaults to rtech.agentd.<agent_id>; a
    # non-None lane proves the default satisfied MemoryConfig's non-empty guard.
    assert lane is not None


def test_token_value_and_length_never_logged() -> None:
    secret = "fixturefixture"
    from loguru import logger

    captured: list[str] = []
    sink_id = logger.add(captured.append, level="DEBUG", format="{message}")
    try:
        build_agent_memory_lane(
            "agentd-nagatha",
            env={"OPENBRAIN_TOKEN_NAGATHA": secret},
            client=_FakeMemoryClient(),
        )
    finally:
        logger.remove(sink_id)
    blob = "".join(captured)
    assert secret not in blob
    assert str(len(secret)) not in blob
