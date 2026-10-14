"""Test NatsFrameBus lifecycle safe-logging (credential-leak fix)."""

from __future__ import annotations

import pytest
from loguru import logger
from rtech_webchat.bus import NatsFrameBus


@pytest.mark.asyncio
async def test_connect_failure_logs_type_only_never_credential_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed FleetBus connect logs safe context + exception type, never str(exc).

    A NATS connect failure can carry the credential-bearing
    ``nats://user:pass@host`` URL in its message; that text must never reach a log
    line. This mirrors the guarded presence subscribe path (app.py) and the repo
    safe-log convention. A dedicated loguru sink captures exactly what the edge
    emits (loguru binds the real stderr at import time, so ``capsys`` cannot see
    it).
    """

    secret_url = "nats://svc:sup3r-s3cret-t0ken@bus.internal:4222"
    bus = NatsFrameBus(secret_url, env="dev")

    async def _raise_with_credential() -> None:
        raise ConnectionError(f"connect refused for {secret_url}")

    monkeypatch.setattr(bus._bus, "connect", _raise_with_credential)  # noqa: SLF001

    records: list[str] = []
    sink_id = logger.add(records.append, level="TRACE", format="{message}")
    try:
        with pytest.raises(ConnectionError):
            await bus.connect()
    finally:
        logger.remove(sink_id)

    logged = "".join(records)
    # No fragment of the credential-bearing URL leaks into any log line.
    assert "sup3r-s3cret-t0ken" not in logged
    assert "svc:" not in logged
    assert secret_url not in logged
    assert "bus.internal" not in logged
    # The safe exception type is logged so operators can still triage.
    assert "ConnectionError" in logged
    assert "webchat.bus.connect | failed error_type=ConnectionError" in logged


@pytest.mark.asyncio
async def test_subscribe_failure_logs_type_only_never_credential_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed raw_subscribe logs safe context + exception type, never str(exc).

    ``NatsFrameBus.subscribe`` is the owning bus boundary in front of
    ``FleetBus.raw_subscribe``; a subscribe failure can carry the
    credential-bearing ``nats://user:pass@host`` URL in its message, and that
    text must never reach a log line before app.py's safe presence log runs. A
    dedicated loguru sink captures exactly what the edge emits (loguru binds the
    real stderr at import time, so ``capsys`` cannot see it).
    """

    secret_url = "nats://svc:sup3r-s3cret-t0ken@bus.internal:4222"
    bus = NatsFrameBus(secret_url, env="dev")

    async def _raise_with_credential(subject: str, handler: object) -> None:
        raise ConnectionError(f"subscribe refused for {secret_url}")

    monkeypatch.setattr(bus._bus, "raw_subscribe", _raise_with_credential)  # noqa: SLF001

    records: list[str] = []
    sink_id = logger.add(records.append, level="TRACE", format="{message}")
    try:
        with pytest.raises(ConnectionError):
            await bus.subscribe("rtech.dev.agent.*.heartbeat", _noop_handler)
    finally:
        logger.remove(sink_id)

    logged = "".join(records)
    # No fragment of the credential-bearing URL leaks into any log line.
    assert "sup3r-s3cret-t0ken" not in logged
    assert "svc:" not in logged
    assert secret_url not in logged
    assert "bus.internal" not in logged
    # The safe exception type is logged so operators can still triage.
    assert "ConnectionError" in logged
    assert "webchat.bus.subscribe | subject=rtech.dev.agent.*.heartbeat" in logged
    assert "failed error_type=ConnectionError" in logged


async def _noop_handler(data: bytes) -> None:
    """Discard delivered frame bytes; the subscribe fails before any delivery."""
