"""Cover agentd identity resolution: env override + collision-safe host default."""

from __future__ import annotations

import pytest
from rtech_agentd.identity import (
    AgentIdentity,
    default_agent_id,
    resolve_agent_id,
)


def test_env_override_wins() -> None:
    """An explicit RTECH_AGENT_ID is used verbatim."""
    identity = resolve_agent_id(env={"RTECH_AGENT_ID": "agentd-birk"})
    assert identity.agent_id == "agentd-birk"


def test_env_override_is_stripped() -> None:
    """Surrounding whitespace on the env value is trimmed."""
    identity = resolve_agent_id(env={"RTECH_AGENT_ID": "  agentd-nimbus  "})
    assert identity.agent_id == "agentd-nimbus"


def test_default_includes_hostname() -> None:
    """The default id is ``agentd-<hostname>``, never the bare shared literal."""
    identity = resolve_agent_id(env={})
    assert identity.agent_id.startswith("agentd-")
    assert identity.agent_id != "agentd"
    assert identity.agent_id == default_agent_id()


def test_blank_env_falls_back_to_host_default() -> None:
    """A whitespace-only env value is treated as unset (host default applies)."""
    identity = resolve_agent_id(env={"RTECH_AGENT_ID": "   "})
    assert identity.agent_id == default_agent_id()
    assert identity.agent_id != "agentd"


def test_default_is_never_bare_agentd() -> None:
    """``default_agent_id`` always carries a host suffix — never bare ``agentd``."""
    value = default_agent_id()
    assert value.startswith("agentd-")
    assert len(value) > len("agentd-")


def test_empty_identity_rejected() -> None:
    """The frozen identity rejects an empty id at construction."""
    with pytest.raises(ValueError):
        AgentIdentity(agent_id="")
    with pytest.raises(ValueError):
        AgentIdentity(agent_id="   ")
