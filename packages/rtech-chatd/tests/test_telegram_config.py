"""TelegramConfig env plumbing + fail-loud guards (issue #108, inbound).

Mirrors the discipline of ``test_intake_config.py``: default construction, env
parsing, and — the load-bearing part — every fail-loud guard the config enforces
so a misconfigured Telegram edge is caught at build, not mid-poll. Covers the
poll-timeout hot-loop/HTTP-timeout guards, the configured-but-meaningless group
allowlist guard, and the allowed_updates blindness guards.
"""

from __future__ import annotations

import pytest
from rtech_chatd.config import (
    DEFAULT_TELEGRAM_ALLOWED_UPDATES,
    DEFAULT_TELEGRAM_POLL_TIMEOUT_SECONDS,
    TelegramConfig,
)

_TG_ENV_VARS = ("RTECH_TG_ALLOWED_GROUPS", "RTECH_TG_POLL_TIMEOUT_S")


def _clear_tg_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _TG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------- defaults


def test_defaults() -> None:
    cfg = TelegramConfig()
    assert cfg.allowed_groups == frozenset()
    assert cfg.allowed_updates == DEFAULT_TELEGRAM_ALLOWED_UPDATES
    assert cfg.poll_timeout_s == DEFAULT_TELEGRAM_POLL_TIMEOUT_SECONDS
    # message + message_reaction + callback_query must be in the default
    # subscription (each is never auto-included → silent blindness if omitted).
    assert "message" in cfg.allowed_updates
    assert "message_reaction" in cfg.allowed_updates
    assert "callback_query" in cfg.allowed_updates


def test_from_env_all_unset_is_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_tg_env(monkeypatch)
    cfg = TelegramConfig.from_env()
    assert cfg.allowed_groups == frozenset()  # unset = no allowlist (Discord parity)
    assert cfg.poll_timeout_s == DEFAULT_TELEGRAM_POLL_TIMEOUT_SECONDS


# --------------------------------------------------------------- group parsing


def test_from_env_parses_allowed_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_tg_env(monkeypatch)
    monkeypatch.setenv("RTECH_TG_ALLOWED_GROUPS", "-1001234567890, -1002222222222 ,555")
    cfg = TelegramConfig.from_env()
    assert cfg.allowed_groups == frozenset({"-1001234567890", "-1002222222222", "555"})


def test_from_env_unset_groups_is_empty_no_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    # UNSET is the documented Discord-parity no-allowlist behavior (NOT an error);
    # the pure gates still decide every message.
    _clear_tg_env(monkeypatch)
    assert TelegramConfig.from_env().allowed_groups == frozenset()


# --------------------------------------------------------------- timeout parsing


def test_from_env_parses_poll_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_tg_env(monkeypatch)
    monkeypatch.setenv("RTECH_TG_POLL_TIMEOUT_S", "45")
    assert TelegramConfig.from_env().poll_timeout_s == 45


# --------------------------------------------------------------- fail-loud guards


def test_from_env_malformed_timeout_raises_naming_var(monkeypatch: pytest.MonkeyPatch) -> None:
    # A non-integer must RAISE (naming the var), never silently default.
    _clear_tg_env(monkeypatch)
    monkeypatch.setenv("RTECH_TG_POLL_TIMEOUT_S", "not-a-number")
    with pytest.raises(ValueError, match="RTECH_TG_POLL_TIMEOUT_S"):
        TelegramConfig.from_env()


def test_from_env_timeout_zero_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # timeout=0 returns immediately and hot-loops the poll → must be rejected.
    _clear_tg_env(monkeypatch)
    monkeypatch.setenv("RTECH_TG_POLL_TIMEOUT_S", "0")
    with pytest.raises(ValueError, match="RTECH_TG_POLL_TIMEOUT_S"):
        TelegramConfig.from_env()


def test_from_env_negative_timeout_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_tg_env(monkeypatch)
    monkeypatch.setenv("RTECH_TG_POLL_TIMEOUT_S", "-5")
    with pytest.raises(ValueError, match="RTECH_TG_POLL_TIMEOUT_S"):
        TelegramConfig.from_env()


def test_from_env_set_but_blank_groups_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # SET but every token blank (",,") collapses to nothing → a configured-but-
    # meaningless allowlist must fail closed, not accept every group.
    _clear_tg_env(monkeypatch)
    monkeypatch.setenv("RTECH_TG_ALLOWED_GROUPS", " , , ")
    with pytest.raises(ValueError, match="RTECH_TG_ALLOWED_GROUPS"):
        TelegramConfig.from_env()


def test_from_env_empty_string_groups_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Present-but-empty ("") is SET, so it fails closed (unset it for no-allowlist).
    _clear_tg_env(monkeypatch)
    monkeypatch.setenv("RTECH_TG_ALLOWED_GROUPS", "")
    with pytest.raises(ValueError, match="RTECH_TG_ALLOWED_GROUPS"):
        TelegramConfig.from_env()


def test_from_env_invalid_group_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # A token that is not a signed integer is a config error, not a chat we
    # silently never match.
    _clear_tg_env(monkeypatch)
    monkeypatch.setenv("RTECH_TG_ALLOWED_GROUPS", "-1001234567890, not-an-id")
    with pytest.raises(ValueError, match="RTECH_TG_ALLOWED_GROUPS"):
        TelegramConfig.from_env()


# --------------------------------------------------------------- __post_init__


def test_poll_timeout_zero_construction_raises() -> None:
    with pytest.raises(ValueError, match="poll_timeout_s must be >= 1"):
        TelegramConfig(poll_timeout_s=0)


def test_poll_timeout_negative_construction_raises() -> None:
    with pytest.raises(ValueError, match="poll_timeout_s must be >= 1"):
        TelegramConfig(poll_timeout_s=-1)


def test_allowed_updates_missing_message_reaction_raises() -> None:
    # Omitting message_reaction makes the bot silently blind to every reaction.
    with pytest.raises(ValueError, match="message_reaction"):
        TelegramConfig(allowed_updates=("message", "edited_message"))


def test_allowed_updates_missing_message_raises() -> None:
    with pytest.raises(ValueError, match="'message'"):
        TelegramConfig(allowed_updates=("message_reaction", "callback_query"))


def test_allowed_updates_missing_callback_query_raises() -> None:
    # Omitting callback_query silently disables every interactive steer button
    # (issue #117): the Bot API never auto-includes tap updates in getUpdates.
    with pytest.raises(ValueError, match="callback_query"):
        TelegramConfig(allowed_updates=("message", "message_reaction"))


def test_default_allowed_updates_include_callback_query() -> None:
    # The default subscription must carry callback_query so the interactive layer's
    # taps reach the poll loop out of the box.
    assert "callback_query" in TelegramConfig().allowed_updates
    assert "callback_query" in DEFAULT_TELEGRAM_ALLOWED_UPDATES
