"""Heartbeat: fleet subject + fleet-core payload, active_turn as a typed extra."""

from __future__ import annotations

import pytest
from fleet_nats import Envelope, parse_heartbeat, subjects
from rtech_agentd.heartbeat import HeartbeatConfig, HeartbeatPublisher, heartbeat_subject


class _RecordingPublisher:
    """Capture (subject, envelope) publishes for assertions."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, Envelope]] = []

    async def __call__(self, subject: str, envelope: Envelope) -> None:
        self.sent.append((subject, envelope))


async def _noop_sleep(_seconds: float) -> None:
    return None


def test_heartbeat_subject_uses_fleet_scheme() -> None:
    assert heartbeat_subject("dev", "nova") == subjects.agent_heartbeat("dev", "nova")
    assert heartbeat_subject("dev", "nova") == "dev.agent.heartbeat.nova"


def test_heartbeat_subject_rejects_empty() -> None:
    with pytest.raises(ValueError):
        heartbeat_subject("", "nova")
    with pytest.raises(ValueError):
        heartbeat_subject("dev", "")


def test_config_requires_env_and_agent() -> None:
    with pytest.raises(ValueError):
        HeartbeatConfig(agent_id="", env="dev")
    with pytest.raises(ValueError):
        HeartbeatConfig(agent_id="nova", env="")
    with pytest.raises(ValueError):
        HeartbeatConfig(agent_id="nova", env="dev", interval_s=0)


@pytest.mark.asyncio
async def test_publish_once_emits_fleet_heartbeat() -> None:
    publisher = _RecordingPublisher()
    pub = HeartbeatPublisher(
        config=HeartbeatConfig(agent_id="nova", env="dev", lanes=("dev.agent.dispatch.nova",)),
        publisher=publisher,
        active_turn=lambda: True,
        sleep=_noop_sleep,
    )

    await pub.publish_once()

    assert len(publisher.sent) == 1
    subject, envelope = publisher.sent[0]
    assert subject == "dev.agent.heartbeat.nova"

    # Round-trip the wire bytes: fleet clients parse the SAME typed Heartbeat.
    decoded = Envelope.from_bytes(envelope.to_bytes())
    parsed = parse_heartbeat(decoded)
    assert parsed.agent_id == "nova"
    assert parsed.role == "worker"
    assert parsed.lanes == ("dev.agent.dispatch.nova",)
    assert parsed.seq == 0
    # active_turn rides as a typed extra on the fleet Heartbeat typed_signal —
    # a typed field carried through the wire, never a prose trailer.
    signal = decoded.typed_signal
    assert signal is not None
    assert (signal.model_extra or {}).get("active_turn") is True


@pytest.mark.asyncio
async def test_publish_once_increments_seq() -> None:
    publisher = _RecordingPublisher()
    pub = HeartbeatPublisher(
        config=HeartbeatConfig(agent_id="nova", env="dev"),
        publisher=publisher,
        active_turn=lambda: False,
        sleep=_noop_sleep,
    )

    await pub.publish_once()
    await pub.publish_once()
    await pub.publish_once()

    seqs = [parse_heartbeat(env).seq for _subject, env in publisher.sent]
    assert seqs == [0, 1, 2]


@pytest.mark.asyncio
async def test_publish_once_carries_active_turn_false() -> None:
    publisher = _RecordingPublisher()
    pub = HeartbeatPublisher(
        config=HeartbeatConfig(agent_id="nova", env="dev"),
        publisher=publisher,
        active_turn=lambda: False,
        sleep=_noop_sleep,
    )

    await pub.publish_once()

    _subject, envelope = publisher.sent[0]
    signal = Envelope.from_bytes(envelope.to_bytes()).typed_signal
    assert signal is not None
    assert (signal.model_extra or {}).get("active_turn") is False
