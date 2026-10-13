"""Home channel config + deliver_to_home path (Hermes reuse, advisory divergence)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from rtech_chatd.channel import InboundMessage, SendTarget
from rtech_chatd.config import HOME_ADVISORY, ChatdConfig, HomeChannel
from rtech_chatd.edge import DiscordEdge


@dataclass
class FakeAdapter:
    """Record send() calls; never yields inbound here."""

    sent: list[tuple[SendTarget, str]] = field(default_factory=list)

    async def events(self) -> AsyncIterator[InboundMessage]:
        return
        yield  # pragma: no cover - makes this an async generator

    async def send(self, target: SendTarget, text: str, files: list[Path]) -> None:
        self.sent.append((target, text))

    async def typing(self, target: SendTarget, on: bool) -> None:
        return None


@dataclass
class FakeSubscription:
    """Stub subscription."""

    async def unsubscribe(self) -> None:
        return None


@dataclass
class FakeFrameBus:
    """Minimal bus stub; the home path never touches it."""

    published: list[tuple[str, bytes]] = field(default_factory=list)

    async def publish(self, subject: str, data: bytes) -> None:
        self.published.append((subject, data))

    async def subscribe(self, subject: str, handler: object) -> FakeSubscription:
        return FakeSubscription()


def test_from_env_populates_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCORD_HOME_CHANNEL", "chan-home")
    monkeypatch.setenv("DISCORD_HOME_CHANNEL_NAME", "Ops")
    monkeypatch.setenv("DISCORD_HOME_CHANNEL_THREAD_ID", "thread-home")
    cfg = ChatdConfig.from_env()
    assert cfg.home_channel == HomeChannel(chat_id="chan-home", name="Ops", thread_id="thread-home")


def test_from_env_no_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISCORD_HOME_CHANNEL", raising=False)
    cfg = ChatdConfig.from_env()
    assert cfg.home_channel is None


def test_blank_chat_id_raises() -> None:
    with pytest.raises(ValueError):
        HomeChannel(chat_id="")


def test_home_send_target_thread_wins() -> None:
    cfg = ChatdConfig(home_channel=HomeChannel(chat_id="chan-home", thread_id="t-1"))
    target = cfg.home_send_target()
    assert target == SendTarget(channel_id="chan-home", thread_id="t-1")
    assert target is not None
    assert target.resolve() == "t-1"


def test_home_send_target_none_when_unset() -> None:
    cfg = ChatdConfig()
    assert cfg.home_send_target() is None


async def test_deliver_to_home_guards_and_sends() -> None:
    adapter = FakeAdapter()
    edge = DiscordEdge(
        adapter,
        FakeFrameBus(),
        stack="dev",
        home_target=SendTarget(channel_id="chan-home", thread_id="t-1"),
    )
    ok = await edge.deliver_to_home("cron result leak sk-ABCDEFGHIJKLMNOP1234567890")

    assert ok is True
    assert len(adapter.sent) == 1
    target, text = adapter.sent[0]
    assert target == SendTarget(channel_id="chan-home", thread_id="t-1")
    assert "sk-ABCDEFGHIJKLMNOP1234567890" not in text
    assert "[redacted]" in text


async def test_deliver_to_home_without_target_does_not_send() -> None:
    adapter = FakeAdapter()
    edge = DiscordEdge(adapter, FakeFrameBus(), stack="dev", home_target=None)
    ok = await edge.deliver_to_home("cron result")

    assert ok is False
    assert adapter.sent == []  # no send anywhere, no advisory posted
    assert edge.home_advisory == HOME_ADVISORY  # advisory is caller-only
