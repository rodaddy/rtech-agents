"""Test doubles for the two bridge buses.

``FakeFleetBus`` records envelope publishes and lets a test drive delivery into
subscribed handlers — both the envelope-decoding ``subscribe`` (fleet side) and
the raw ``raw_subscribe`` (rtech frame side). Envelope delivery round-trips every
envelope through the REAL wire codec (``Envelope.to_bytes`` -> ``from_bytes``) so
tests prove the bridge's wire contract, not in-memory object passing: typed
signals must survive serialization for the bridge to see them. ``subscribe``
returns a :class:`FakeSubscription` so lifecycle (unsubscribe) is observable.
It also exposes the two internal attributes ``RtechFrameBus`` reaches for a raw
byte publish (``_guard_payload_size`` + ``_nc.publish``) so the rtech-side frame
path is exercised end to end without a live NATS server.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import pytest
from fleet_nats import Envelope


@dataclass
class _FakeMsg:
    """Minimal stand-in for ``nats.aio.msg.Msg`` (only ``data`` is used)."""

    data: bytes
    subject: str = ""
    reply: str = ""


class _FakeNatsClient:
    """The tiny slice of the nats-py client that ``RtechFrameBus`` publishes through."""

    def __init__(self) -> None:
        self.max_payload: int = 1024 * 1024
        self.raw_published: list[tuple[str, bytes]] = []
        self.fail_publish: bool = False

    async def publish(self, subject: str, data: bytes) -> None:
        if self.fail_publish:
            raise ConnectionError("raw publish failed (test)")
        self.raw_published.append((subject, data))


class FakeSubscription:
    """Record-only stand-in for ``nats.aio.subscription.Subscription``."""

    def __init__(self, bus: FakeFleetBus, subject: str) -> None:
        self._bus = bus
        self.subject = subject
        self.unsubscribed = False

    async def unsubscribe(self) -> None:
        self.unsubscribed = True
        self._bus._env_handlers.pop(self.subject, None)


class FakeFleetBus:
    """A record-and-drive stand-in for ``fleet_nats.FleetBus``."""

    def __init__(self) -> None:
        self.published: list[tuple[str, Envelope]] = []
        self.fail_publish: bool = False
        self.fail_subscribe: bool = False
        # subject -> envelope handler (from subscribe)
        self._env_handlers: dict[str, Callable[[Envelope], Awaitable[None]]] = {}
        # subject -> raw Msg handler (from raw_subscribe)
        self._raw_handlers: dict[str, Callable[[Any], Awaitable[None]]] = {}
        self.subscriptions: list[FakeSubscription] = []
        self._nc = _FakeNatsClient()

    # --- FleetBus surface the runtime/heartbeat use --------------------------

    async def publish(self, subject: str, envelope: Envelope) -> None:
        if self.fail_publish:
            raise ConnectionError("publish failed (test)")
        self.published.append((subject, envelope))

    async def subscribe(
        self, subject: str, handler: Callable[[Envelope], Awaitable[None]]
    ) -> FakeSubscription:
        if self.fail_subscribe:
            raise ConnectionError("subscribe failed (test)")
        self._env_handlers[subject] = handler
        subscription = FakeSubscription(self, subject)
        self.subscriptions.append(subscription)
        return subscription

    async def raw_subscribe(self, subject: str, handler: Callable[[Any], Awaitable[None]]) -> None:
        self._raw_handlers[subject] = handler

    def _guard_payload_size(self, subject: str, data: bytes) -> None:
        if len(data) > self._nc.max_payload:
            raise ValueError(f"payload too large on {subject}")

    # --- test drive helpers --------------------------------------------------

    async def deliver_raw(self, subject: str, data: bytes) -> None:
        """Deliver raw bytes to a ``raw_subscribe`` handler."""
        handler = self._raw_handlers[subject]
        await handler(_FakeMsg(data=data, subject=subject))

    async def deliver_envelope(self, subject: str, envelope: Envelope) -> None:
        """Deliver an envelope through the REAL wire codec to a ``subscribe`` handler.

        The round-trip (``to_bytes`` -> ``from_bytes``) mirrors what the live bus
        does, so typed signals reach the handler only if they genuinely survive
        serialization — enshrined object-passing cannot fake the wire contract.
        """
        handler = self._env_handlers[subject]
        await handler(Envelope.from_bytes(envelope.to_bytes()))

    def raw_published(self) -> list[tuple[str, bytes]]:
        """Return the raw byte publishes routed through the fake nats client."""
        return self._nc.raw_published

    def has_raw_sub(self, subject: str) -> bool:
        return subject in self._raw_handlers

    def has_env_sub(self, subject: str) -> bool:
        return subject in self._env_handlers


@pytest.fixture
def fake_bus_cls() -> type[FakeFleetBus]:
    """Return the ``FakeFleetBus`` class.

    Delivered as a fixture (not a cross-module import) so it never rides the
    shared ``tests`` package name — the multi-package workspace has several
    ``tests/`` dirs, and a ``from .helper import`` resolves ``tests`` to whichever
    package pytest imported first (a real collision this repo already masks by
    giving two packages an identically-named ``FakeFrameBus``). A conftest fixture
    is auto-discovered per-directory and sidesteps that ambiguity entirely.
    """
    return FakeFleetBus
