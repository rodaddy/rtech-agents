"""Test the fronted fleet-id heartbeat."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rtech_hermes_bridge.heartbeat import BridgeHeartbeat, rtech_heartbeat_subject

if TYPE_CHECKING:
    from .conftest import FakeFleetBus


def test_heartbeat_subject_matches_picker_wildcard() -> None:
    # The web-chat presence registry subscribes rtech.agent.*.heartbeat.
    assert rtech_heartbeat_subject("bilby") == "rtech.agent.bilby.heartbeat"


def test_heartbeat_subject_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        rtech_heartbeat_subject("")


def test_heartbeat_rejects_nonpositive_interval(fake_bus_cls: type[FakeFleetBus]) -> None:
    with pytest.raises(ValueError, match="positive"):
        BridgeHeartbeat(bus=fake_bus_cls(), fleet_id="bilby", interval_s=0)  # type: ignore[arg-type]


async def test_publish_once_emits_envelope_as_fleet_id(
    fake_bus_cls: type[FakeFleetBus],
) -> None:
    bus = fake_bus_cls()
    hb = BridgeHeartbeat(bus=bus, fleet_id="bilby", interval_s=30)  # type: ignore[arg-type]
    await hb.publish_once()
    assert len(bus.published) == 1
    subject, envelope = bus.published[0]
    assert subject == "rtech.agent.bilby.heartbeat"
    # The picker records envelope.sender as the agent id — must be the fleet id.
    assert envelope.sender == "bilby"
    assert envelope.kind == "heartbeat"
    assert envelope.payload["bridge"] is True
