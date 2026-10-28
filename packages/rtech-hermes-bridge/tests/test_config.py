"""Test bridge configuration loading and validation."""

from __future__ import annotations

import pytest
from rtech_hermes_bridge.config import BridgeConfig


def test_defaults_from_single_fleet_id() -> None:
    cfg = BridgeConfig(fleet_ids=("birk",))
    assert cfg.fleet_ids == ("birk",)
    assert cfg.fleet_env == "dev"
    assert cfg.rtech_nats_url.startswith("nats://")
    assert cfg.fleet_nats_url.startswith("nats://")
    assert cfg.task_timeout_seconds == 900.0


def test_rejects_empty_fleet_ids() -> None:
    with pytest.raises(ValueError, match="at least one"):
        BridgeConfig(fleet_ids=())


def test_rejects_blank_fleet_id() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        BridgeConfig(fleet_ids=("birk", "  "))


def test_rejects_duplicate_fleet_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        BridgeConfig(fleet_ids=("birk", "birk"))


def test_rejects_bad_env() -> None:
    with pytest.raises(ValueError, match="fleet_env"):
        BridgeConfig(fleet_ids=("birk",), fleet_env="staging")


def test_rejects_non_nats_url() -> None:
    with pytest.raises(ValueError, match="nats://"):
        BridgeConfig(fleet_ids=("birk",), rtech_nats_url="http://x")


def test_rejects_nonpositive_interval() -> None:
    with pytest.raises(ValueError, match="positive"):
        BridgeConfig(fleet_ids=("birk",), heartbeat_interval_seconds=0)


def test_rejects_nonpositive_task_timeout() -> None:
    with pytest.raises(ValueError, match="task_timeout_seconds"):
        BridgeConfig(fleet_ids=("birk",), task_timeout_seconds=0)


def test_from_env_requires_fleet_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RTECH_BRIDGE_FLEET_IDS", raising=False)
    with pytest.raises(ValueError, match="RTECH_BRIDGE_FLEET_IDS"):
        BridgeConfig.from_env()


def test_from_env_parses_multiple_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RTECH_BRIDGE_FLEET_IDS", " birk , sprocket ")
    monkeypatch.setenv("RTECH_BRIDGE_FLEET_ENV", "dev")
    monkeypatch.setenv("RTECH_BRIDGE_TASK_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("RTECH_BRIDGE_DEFAULT_CHANNEL", "chan-1")
    cfg = BridgeConfig.from_env()
    assert cfg.fleet_ids == ("birk", "sprocket")
    assert cfg.default_channel == "chan-1"
    assert cfg.task_timeout_seconds == 120.0


def test_from_env_blank_channel_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RTECH_BRIDGE_FLEET_IDS", "birk")
    monkeypatch.setenv("RTECH_BRIDGE_DEFAULT_CHANNEL", "   ")
    assert BridgeConfig.from_env().default_channel is None


def test_max_queue_depth_env_and_validation() -> None:
    import os
    from unittest.mock import patch

    import pytest
    from rtech_hermes_bridge.config import BridgeConfig

    with patch.dict(
        os.environ,
        {"RTECH_BRIDGE_FLEET_IDS": "birk", "RTECH_BRIDGE_MAX_QUEUE_DEPTH": "3"},
        clear=False,
    ):
        assert BridgeConfig.from_env().max_queue_depth == 3
    with pytest.raises(ValueError, match="max_queue_depth"):
        BridgeConfig(fleet_ids=("birk",), max_queue_depth=0)
