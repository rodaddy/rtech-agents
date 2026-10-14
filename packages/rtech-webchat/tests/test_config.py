"""Cover the required stack id on the web-chat edge."""

from __future__ import annotations

import pytest
from rtech_webchat.config import WebchatConfig


def test_from_env_requires_stack_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # No RTECH_STACK_ID: the edge fails loudly at startup, mirroring the
    # orchestrator — never a defaulted shared literal like "dev".
    monkeypatch.delenv("RTECH_STACK_ID", raising=False)
    with pytest.raises(ValueError, match="RTECH_STACK_ID"):
        WebchatConfig.from_env()


def test_from_env_rejects_blank_stack_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RTECH_STACK_ID", "   ")
    with pytest.raises(ValueError, match="RTECH_STACK_ID"):
        WebchatConfig.from_env()


def test_from_env_resolves_stack_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RTECH_STACK_ID", "mini-m4-dev")
    config = WebchatConfig.from_env()
    assert config.stack == "mini-m4-dev"


def test_explicit_blank_stack_rejected() -> None:
    with pytest.raises(ValueError, match="stack must be non-empty"):
        WebchatConfig(stack=" ")
