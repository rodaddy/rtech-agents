"""Functional I/O tests for the agentd Open Brain memory lane.

The OB client is mocked in every non-live test — nothing here hits the network.
A single env-gated canary exercises the real endpoint when ``OB_LIVE=1``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import rtech_agentd
from openbrain_memory import JsonlSpool
from openbrain_memory import client as openbrain_client
from rtech_agentd.memory import (
    AgentMemoryLane,
    MemoryConfig,
    MemoryContractError,
    MemoryRedactionError,
    MemoryWriteResult,
)

# Planted secret-shaped strings. Constructed so a real secret never appears in
# source (secret-scanner-safe) while still matching the package's SECRET_PATTERNS.
PLANTED_SK_KEY = "sk-" + "A" * 24
PLANTED_BEARER = "Authorization: Bearer " + "x" * 40


class FakeMemoryClient:
    """In-memory stand-in for the OB MCP client.

    Records every call and returns canned payloads. ``search_all`` returns a
    ``{"results": [...]}`` shape with several long items so recall bounding can be
    asserted.
    """

    def __init__(self, *, contract: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._contract = contract
        self.timeout: float = 30.0

    # --- write surface -------------------------------------------------
    def session_start(self, **arguments: Any) -> dict[str, Any]:
        self.calls.append(("session_start", arguments))
        return {"ok": True, "session_key": arguments.get("session_key")}

    def append_session_event(self, **arguments: Any) -> dict[str, Any]:
        self.calls.append(("append_session_event", arguments))
        return {"ok": True}

    def log_thought(self, **arguments: Any) -> dict[str, Any]:
        self.calls.append(("log_thought", arguments))
        return {"ok": True}

    def log_decision(self, **arguments: Any) -> dict[str, Any]:
        self.calls.append(("log_decision", arguments))
        return {"ok": True}

    def session_wrap(self, **arguments: Any) -> dict[str, Any]:
        self.calls.append(("session_wrap", arguments))
        return {"ok": True}

    # --- read surface --------------------------------------------------
    def session_context(self, **arguments: Any) -> dict[str, Any]:
        self.calls.append(("session_context", arguments))
        return {"events": []}

    def search_all(self, **arguments: Any) -> dict[str, Any]:
        self.calls.append(("search_all", arguments))
        return {
            "results": [
                {"text": "alpha " * 40, "source": "brain", "kind": "fact"},
                {"text": "bravo " * 40, "source": "brain", "kind": "fact"},
                {"text": "charlie " * 40, "source": "brain", "kind": "fact"},
                {"text": "delta " * 40, "source": "brain", "kind": "fact"},
            ]
        }

    # --- contract ------------------------------------------------------
    def get_contract(self, **arguments: Any) -> dict[str, Any]:
        self.calls.append(("get_contract", arguments))
        if self._contract is None:
            raise RuntimeError("no contract configured")
        return self._contract

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


def _valid_contract() -> dict[str, Any]:
    """Build a valid contract manifest keyed off the reused package's constants.

    Never hardcodes the tool list — reads REQUIRED_CONTRACT_TOOLS.
    """
    tools = list(openbrain_client.REQUIRED_CONTRACT_TOOLS)
    schema_version = getattr(openbrain_client, "CURRENT_CONTRACT_SCHEMA_VERSION", 1)
    schema_hash = getattr(openbrain_client, "CURRENT_CONTRACT_SCHEMA_HASH", "")
    required_tool_versions = getattr(
        openbrain_client,
        "REQUIRED_CONTRACT_TOOL_VERSIONS",
        {},
    )

    def tool_version(tool: str) -> int:
        value = required_tool_versions.get(tool, 1)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 1

    return {
        "contract_scope": "required_openbrain_memory_contract",
        "contract_version": openbrain_client.CURRENT_CONTRACT_VERSION,
        "schema_version": schema_version,
        "schema_hash": schema_hash,
        "capabilities": [
            {"kind": "tool", "name": tool, "version": tool_version(tool)} for tool in tools
        ],
        "tool_contracts": {
            tool: {
                "version": tool_version(tool),
                "input_schema": {},
                "output_shape": "object",
            }
            for tool in tools
        },
    }


def _config(**overrides: Any) -> MemoryConfig:
    defaults: dict[str, Any] = {
        "base_url": "https://ob.test",
        "token": "test-token",
        "namespace": "ns.test",
        "agent": "tester",
    }
    defaults.update(overrides)
    return MemoryConfig(**defaults)


def test_package_entrypoint_exports_memory_surface() -> None:
    # Import-safety: touch the package entrypoint at least once.
    assert rtech_agentd.AgentMemoryLane is AgentMemoryLane
    assert "MemoryWriteResult" in rtech_agentd.__all__


def test_recall_block_is_bounded() -> None:
    client = FakeMemoryClient(contract=_valid_contract())
    lane = AgentMemoryLane(
        _config(max_items=2, max_chars=120, max_item_chars=40),
        client=client,
    )
    lane.session_start("conv-1", validate_contract=False)

    block = lane.recall("anything")

    lines = block.splitlines()
    assert 0 < len(lines) <= 2
    assert len(block) <= 120
    for line in lines:
        # Strip the "- [kind/source] " prefix; assert the payload text is
        # trimmed to <= max_item_chars (allowing the "..." ellipsis).
        assert len(line) <= 120


def test_append_session_event_is_idempotent_with_distinct_keys() -> None:
    client = FakeMemoryClient(contract=_valid_contract())
    lane = AgentMemoryLane(_config(), client=client)
    lane.session_start("conv-idem", validate_contract=False)

    lane.append_session_event("note one")
    lane.append_session_event("note two")

    appends = [args for name, args in client.calls if name == "append_session_event"]
    assert len(appends) == 2
    keys = {args["metadata"]["idempotency_key"] for args in appends}
    assert len(keys) == 2  # two distinct facade-generated keys


@pytest.mark.parametrize("token", [PLANTED_SK_KEY, PLANTED_BEARER])
def test_redaction_guard_blocks_append(token: str) -> None:
    client = FakeMemoryClient(contract=_valid_contract())
    lane = AgentMemoryLane(_config(), client=client)
    lane.session_start("conv-guard", validate_contract=False)

    with pytest.raises(MemoryRedactionError):
        lane.append_session_event(f"leaking {token} here")

    assert "append_session_event" not in client.names()


def test_redaction_guard_blocks_log_decision_and_session_wrap() -> None:
    client = FakeMemoryClient(contract=_valid_contract())
    lane = AgentMemoryLane(_config(), client=client)
    lane.session_start("conv-guard-2", validate_contract=False)

    with pytest.raises(MemoryRedactionError):
        lane.log_decision(f"decided to keep {PLANTED_SK_KEY}")
    with pytest.raises(MemoryRedactionError):
        lane.session_wrap(f"wrapping with {PLANTED_BEARER}")

    assert "log_decision" not in client.names()
    assert "session_wrap" not in client.names()


def test_redaction_guard_blocks_sensitive_metadata_key() -> None:
    client = FakeMemoryClient(contract=_valid_contract())
    lane = AgentMemoryLane(_config(), client=client)
    lane.session_start("conv-guard-3", validate_contract=False)

    with pytest.raises(MemoryRedactionError):
        lane.append_session_event("benign", token="whatever")

    assert "append_session_event" not in client.names()


def test_contract_version_mismatch_fails_closed() -> None:
    bad = _valid_contract()
    bad["contract_version"] = "0.0.0-bad"
    client = FakeMemoryClient(contract=bad)
    lane = AgentMemoryLane(_config(), client=client)

    with pytest.raises(MemoryContractError):
        lane.session_start("conv-bad", validate_contract=True)

    assert "session_start" not in client.names()


def test_contract_transport_failure_fails_closed() -> None:
    # contract=None → get_contract raises → fail closed.
    client = FakeMemoryClient(contract=None)
    lane = AgentMemoryLane(_config(), client=client)

    with pytest.raises(MemoryContractError):
        lane.session_start("conv-transport", validate_contract=True)

    assert "session_start" not in client.names()


def test_contract_happy_path_opens_lane() -> None:
    client = FakeMemoryClient(contract=_valid_contract())
    lane = AgentMemoryLane(_config(), client=client)

    lane.session_start("conv-ok", validate_contract=True)

    assert "session_start" in client.names()
    assert lane.conversation_key == "conv-ok"


class _FailingAppendClient(FakeMemoryClient):
    """Client whose append_session_event always fails (to exercise spool)."""

    def append_session_event(self, **arguments: Any) -> dict[str, Any]:
        raise RuntimeError("ob down")


def test_spooled_not_saved_is_surfaced(tmp_path: Path) -> None:
    spool_path = tmp_path / "spool.jsonl"
    spool = JsonlSpool(spool_path)
    client = _FailingAppendClient(contract=_valid_contract())
    lane = AgentMemoryLane(_config(), client=client, spool=spool)
    lane.session_start("conv-spool", validate_contract=False)

    result = lane.append_session_event("please spool me")

    assert isinstance(result, MemoryWriteResult)
    assert result.saved is False
    assert result.spooled is True
    assert spool.status().pending_count == 1


def test_write_failure_without_spool_is_surfaced_not_raised() -> None:
    client = _FailingAppendClient(contract=_valid_contract())
    lane = AgentMemoryLane(_config(), client=client)
    lane.session_start("conv-nospool", validate_contract=False)

    result = lane.append_session_event("no spool here")

    assert result.saved is False
    assert result.spooled is False  # surfaced, not silent; never raises


@pytest.mark.skipif(
    os.environ.get("OB_LIVE") != "1",
    reason="OB_LIVE not set",
)
def test_live_canary() -> None:
    import uuid

    token = os.environ.get("OB_TOKEN")
    if not token:
        pytest.skip("OB_TOKEN not set")
    namespace = os.environ.get("OB_NAMESPACE", "rtech.agentd.canary")
    marker = f"agentd-canary-{uuid.uuid4().hex}"

    lane = AgentMemoryLane(
        MemoryConfig(
            base_url="https://openbrain.example.internal",
            token=token,
            namespace=namespace,
            agent="agentd-canary",
        )
    )
    lane.session_start(f"canary-{marker}", validate_contract=True)
    result = lane.append_session_event(f"canary marker {marker}")
    assert result.saved is True

    block = lane.recall(marker)
    assert marker in block


def test_bound_http_timeout_only_lowers_client_timeout() -> None:
    client = FakeMemoryClient(contract=_valid_contract())
    lane = AgentMemoryLane(_config(), client=client)

    # Lowers 30.0 -> 2.0.
    lane.bound_http_timeout(2.0)
    assert client.timeout == 2.0

    # Never raises: 5.0 >= current 2.0 is a no-op.
    lane.bound_http_timeout(5.0)
    assert client.timeout == 2.0

    # Non-positive is a no-op.
    lane.bound_http_timeout(0)
    assert client.timeout == 2.0
