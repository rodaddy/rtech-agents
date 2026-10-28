"""Tests for the runnable chatd service config + token resolution."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from loguru import logger
from rtech_chatd.channel import SendTarget
from rtech_chatd.config import AllowBots
from rtech_chatd.edge import DiscordEdge
from rtech_chatd.service import (
    ChatdServiceConfig,
    resolve_required_token,
    run_components,
    run_service,
)


def _set_min_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RTECH_STACK_ID", "dev")
    monkeypatch.setenv("RTECH_CHATD_INBOX_CHANNEL", "chan-inbox")
    monkeypatch.setenv("RTECH_CHATD_TOKEN_REF", "env:RTECH_CHATD_BOT_TOKEN")


def test_from_env_fails_closed_on_missing_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RTECH_STACK_ID", raising=False)
    monkeypatch.setenv("RTECH_CHATD_INBOX_CHANNEL", "chan-inbox")
    with pytest.raises(ValueError, match="RTECH_STACK_ID"):
        ChatdServiceConfig.from_env()


def test_from_env_fails_closed_on_blank_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RTECH_STACK_ID", "   ")
    monkeypatch.setenv("RTECH_CHATD_INBOX_CHANNEL", "chan-inbox")
    with pytest.raises(ValueError, match="RTECH_STACK_ID"):
        ChatdServiceConfig.from_env()


def test_from_env_fails_closed_on_missing_inbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RTECH_STACK_ID", "dev")
    monkeypatch.delenv("RTECH_CHATD_INBOX_CHANNEL", raising=False)
    with pytest.raises(ValueError, match="inbox_channel is required"):
        ChatdServiceConfig.from_env()


def test_from_env_fails_closed_on_allow_bots_all(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_min_env(monkeypatch)
    monkeypatch.setenv("RTECH_CHATD_ALLOW_BOTS", "all")
    with pytest.raises(ValueError, match="all is rejected"):
        ChatdServiceConfig.from_env()


def test_resolve_required_token_fails_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_min_env(monkeypatch)
    monkeypatch.delenv("RTECH_CHATD_BOT_TOKEN", raising=False)
    config = ChatdServiceConfig.from_env()
    with pytest.raises(RuntimeError, match="bot token not provisioned"):
        resolve_required_token(config)


def test_resolve_required_token_returns_value_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_min_env(monkeypatch)
    monkeypatch.setenv("RTECH_CHATD_BOT_TOKEN", "test-token-value")
    config = ChatdServiceConfig.from_env()
    assert resolve_required_token(config) == "test-token-value"


def test_participants_parsing_trims_and_drops_empties(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_min_env(monkeypatch)
    monkeypatch.setenv("RTECH_CHATD_PARTICIPANTS", "a, b ,c")
    config = ChatdServiceConfig.from_env()
    assert config.participants == ("a", "b", "c")


def test_gate_and_thread_config_carry_inbox_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_min_env(monkeypatch)
    config = ChatdServiceConfig.from_env()
    assert config.gate_config().free_response_channels == frozenset({"chan-inbox"})
    assert config.thread_config().free_response_channels == frozenset({"chan-inbox"})


def test_from_env_happy_path_builds_trio(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_min_env(monkeypatch)
    monkeypatch.setenv("RTECH_CHATD_PARTICIPANTS", "agent-a,agent-b,agent-c")
    monkeypatch.setenv("RTECH_CHATD_ALLOW_BOTS", "mentions")
    config = ChatdServiceConfig.from_env()
    assert config.stack == "dev"
    assert config.edge.inbox_channel == "chan-inbox"
    assert config.participants == ("agent-a", "agent-b", "agent-c")
    assert config.allow_bots is AllowBots.MENTIONS


def test_from_env_rejects_literal_token_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    # A literal token where an indirection ref belongs fails closed WITHOUT the
    # value ever reaching the error/log (no credential leak to journald).
    _set_min_env(monkeypatch)
    monkeypatch.setenv("RTECH_CHATD_TOKEN_REF", "literal-token-value")
    with pytest.raises(ValueError) as excinfo:
        ChatdServiceConfig.from_env()
    assert "literal-token-value" not in str(excinfo.value)


def test_from_env_intake_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_min_env(monkeypatch)
    monkeypatch.delenv("RTECH_CHATD_INTAKE_ENABLED", raising=False)
    config = ChatdServiceConfig.from_env()
    assert config.intake.enabled is False


def test_from_env_intake_enabled_carries_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_min_env(monkeypatch)
    monkeypatch.setenv("RTECH_CHATD_INTAKE_ENABLED", "1")
    monkeypatch.setenv("RTECH_CHATD_TRIAGE_AGENT", "agentd-nimbus")
    monkeypatch.setenv("RTECH_CHATD_INTAKE_MEMBER_IDS", "77,88")
    config = ChatdServiceConfig.from_env()
    assert config.intake.enabled is True
    assert config.intake.triage_agent == "agentd-nimbus"
    assert config.intake.member_ids == frozenset({"77", "88"})


async def test_run_service_wires_normalized_persistent_debate_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production composition root carries both persistent lane maps to the edge."""

    _set_min_env(monkeypatch)
    monkeypatch.setenv("RTECH_CHATD_DEBATE_LANES", "agentd-nimbus-dev=94")
    monkeypatch.setenv("RTECH_CHATD_DEBATE_CHAT_LANES", "+123:agentd-dash-alpha=93")
    config = ChatdServiceConfig.from_env()
    captured: list[DiscordEdge] = []

    class _Bus:
        def __init__(self, url: str) -> None:
            self.url = url

    class _Adapter:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    async def _run_components(adapter: object, bus: object, edge: DiscordEdge) -> None:
        captured.append(edge)

    monkeypatch.setattr("rtech_chatd.service.NatsFrameBus", _Bus)
    monkeypatch.setattr("rtech_chatd.service.DiscordChannelAdapter", _Adapter)
    monkeypatch.setattr("rtech_chatd.service.resolve_required_token", lambda _: "token")
    monkeypatch.setattr("rtech_chatd.service.run_components", _run_components)

    await run_service(config)
    assert len(captured) == 1
    assert captured[0]._debate_lanes == {"nimbus": "94"}  # noqa: SLF001
    assert captured[0]._debate_chat_lanes == {("123", "dash-alpha"): "93"}  # noqa: SLF001


async def test_run_service_wires_presence_notify_from_home_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Presence notify targets the home channel + carries the Dash URL."""

    _set_min_env(monkeypatch)
    monkeypatch.setenv("DISCORD_HOME_CHANNEL", "home-777")
    monkeypatch.setenv("RTECH_CHATD_DASH_BASE_URL", "https://dash.example.internal")
    config = ChatdServiceConfig.from_env()
    captured: list[DiscordEdge] = []

    class _Bus:
        def __init__(self, url: str) -> None:
            self.url = url

    class _Adapter:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    async def _run_components(adapter: object, bus: object, edge: DiscordEdge) -> None:
        captured.append(edge)

    monkeypatch.setattr("rtech_chatd.service.NatsFrameBus", _Bus)
    monkeypatch.setattr("rtech_chatd.service.DiscordChannelAdapter", _Adapter)
    monkeypatch.setattr("rtech_chatd.service.resolve_required_token", lambda _: "token")
    monkeypatch.setattr("rtech_chatd.service.run_components", _run_components)

    await run_service(config)
    assert len(captured) == 1
    edge = captured[0]
    assert edge._presence_consumer is not None  # noqa: SLF001
    assert edge._presence_notify_target is not None  # noqa: SLF001
    assert edge._presence_notify_target.channel_id == "home-777"  # noqa: SLF001
    assert edge._dash_base_url == "https://dash.example.internal"  # noqa: SLF001


async def test_run_service_no_home_channel_leaves_presence_unsubscribed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No home channel => no presence notify target => the edge holds no consumer."""

    _set_min_env(monkeypatch)
    monkeypatch.delenv("DISCORD_HOME_CHANNEL", raising=False)
    config = ChatdServiceConfig.from_env()
    captured: list[DiscordEdge] = []

    class _Bus:
        def __init__(self, url: str) -> None:
            self.url = url

    class _Adapter:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    async def _run_components(adapter: object, bus: object, edge: DiscordEdge) -> None:
        captured.append(edge)

    monkeypatch.setattr("rtech_chatd.service.NatsFrameBus", _Bus)
    monkeypatch.setattr("rtech_chatd.service.DiscordChannelAdapter", _Adapter)
    monkeypatch.setattr("rtech_chatd.service.resolve_required_token", lambda _: "token")
    monkeypatch.setattr("rtech_chatd.service.run_components", _run_components)

    await run_service(config)
    assert len(captured) == 1
    assert captured[0]._presence_consumer is None  # noqa: SLF001
    assert captured[0]._presence_notify_target is None  # noqa: SLF001


class _CleanExitAdapter:
    """An adapter whose start() returns immediately (a dead-but-clean client)."""

    def __init__(self) -> None:
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    async def events(self) -> AsyncIterator[object]:
        # Block forever so run_inbound never completes on its own; the loud
        # RuntimeError from the returning start() is what tears the group down.
        await asyncio.Event().wait()
        if False:  # pragma: no cover - only to make this an async generator
            yield object()

    async def send(self, target: SendTarget, text: str, files: list[Path]) -> None:
        return None

    async def typing(self, target: SendTarget, on: bool) -> None:
        return None


class _NoopSubscription:
    async def unsubscribe(self) -> None:
        return None


class _LifecycleBus:
    """A bus that records connect/close and no-ops publish/subscribe."""

    def __init__(self) -> None:
        self.connected = False
        self.closed = False

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.closed = True

    async def publish(self, subject: str, data: bytes) -> None:
        return None

    async def subscribe(self, subject: str, handler: object) -> _NoopSubscription:
        return _NoopSubscription()


async def test_client_clean_exit_fails_loud_and_tears_down() -> None:
    adapter = _CleanExitAdapter()
    bus = _LifecycleBus()
    edge = DiscordEdge(adapter, bus, stack="dev")  # type: ignore[arg-type]

    caught = False
    try:
        await run_components(adapter, bus, edge)  # type: ignore[arg-type]
    except* RuntimeError:
        caught = True

    assert caught is True
    assert bus.connected is True
    assert bus.closed is True
    assert adapter.closed is True


# A teardown exception whose message carries a credential-shaped secret: the
# teardown logs must render the safe stage + exception TYPE, never this text.
_TEARDOWN_SECRET = "nats://user:s3cr3t-t0ken-DEADBEEF@bus.internal:4222"  # noqa: S105


class _SecretRaisingEdge:
    """An edge whose close() raises with a credential-bearing message."""

    def __init__(self) -> None:
        self.close_calls = 0

    async def run_inbound(self) -> None:
        await asyncio.Event().wait()

    async def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError(f"edge close: {_TEARDOWN_SECRET}")


class _SecretRaisingAdapter(_CleanExitAdapter):
    """An adapter whose close() raises with a credential-bearing message."""

    async def close(self) -> None:
        self.closed = True
        raise RuntimeError(f"adapter close: {_TEARDOWN_SECRET}")


class _SecretRaisingBus(_LifecycleBus):
    """A bus whose close() raises with a credential-bearing message."""

    async def close(self) -> None:
        self.closed = True
        raise RuntimeError(f"bus close: {_TEARDOWN_SECRET}")


async def test_teardown_logs_omit_raw_exception_messages() -> None:
    # Regression (finding #1): every teardown failure log must render the safe
    # stage + exception TYPE, never the raw exception message, which can carry a
    # connection/credential secret. Cleanup still runs all three closes and the
    # loud RuntimeError from the clean client exit still propagates.
    captured: list[str] = []
    sink_id = logger.add(lambda m: captured.append(m.record["message"]), level="ERROR")

    adapter = _SecretRaisingAdapter()
    bus = _SecretRaisingBus()
    edge = _SecretRaisingEdge()

    caught = False
    try:
        await run_components(adapter, bus, edge)  # type: ignore[arg-type]
    except* RuntimeError:
        caught = True
    finally:
        logger.remove(sink_id)

    # Loud failure preserved and all three teardown stages ran despite raising.
    assert caught is True
    assert edge.close_calls == 1
    assert adapter.closed is True
    assert bus.closed is True

    joined = "\n".join(captured)
    # Every stage logged its safe stage + type.
    assert "stage=edge error_type=RuntimeError" in joined
    assert "stage=adapter error_type=RuntimeError" in joined
    assert "stage=bus error_type=RuntimeError" in joined
    # The secret appears in none of the rendered teardown logs.
    assert _TEARDOWN_SECRET not in joined
    assert "s3cr3t" not in joined
