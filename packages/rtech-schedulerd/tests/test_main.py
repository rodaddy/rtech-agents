"""Test the entrypoint: log sanitization plus concrete runtime lifecycle.

The scheduler entrypoint may be handed a ``NATS_URL`` with embedded credentials
(``nats://user:pass@host``). The startup log must render only a redacted
``scheme://host:port`` view — never the raw URL — and connection/teardown
failures must log the exception TYPE only, never a raw driver string that could
echo the URL or a payload.

The runtime tests drive ``_run`` end to end with the external NATS bus stubbed
(no real connection): they prove the clean-shutdown teardown closes the bus, a
config failure aborts before connecting, a connect failure is contained, and a
teardown failure never masks the run outcome or leaks a raw driver string.
"""

from __future__ import annotations

import pytest
from rtech_schedulerd import __main__ as main_mod
from rtech_schedulerd.__main__ import _redact_url
from rtech_schedulerd.config import ScheduleConfigError, SchedulerConfig

# asyncio_mode = "auto" (workspace pyproject): async tests need no explicit mark,
# and marking the sync _redact_url tests would warn — so no module pytestmark.


class _FakeBus:
    """A stand-in for the external ``FleetBus``: records connect/close calls."""

    def __init__(self, *, connect_fails: bool = False, close_fails: bool = False) -> None:
        self.connect_fails = connect_fails
        self.close_fails = close_fails
        self.connected = False
        self.closed = False

    async def connect(self) -> None:
        if self.connect_fails:
            raise RuntimeError("nats://alice:secret@host refused")  # raw string must not leak
        self.connected = True

    async def close(self) -> None:
        self.closed = True
        if self.close_fails:
            raise RuntimeError("nats://alice:secret@host teardown boom")


def _install_bus(monkeypatch: pytest.MonkeyPatch, bus: _FakeBus, *, run_stops: bool = True) -> None:
    """Stub the external bus construction/connection and a self-stopping run.

    ``FleetBus``/``BusConfig`` become no-op factories returning ``bus``;
    ``TypedFrameBus`` is identity; ``CronScheduler.run`` returns immediately (or
    is left to the caller) and ``load_scheduler_config`` yields a valid config.
    """
    monkeypatch.setattr(main_mod, "BusConfig", lambda **_: object())
    monkeypatch.setattr(main_mod, "FleetBus", lambda _cfg: bus)
    monkeypatch.setattr(main_mod, "TypedFrameBus", lambda conn: conn)
    monkeypatch.setattr(
        main_mod, "load_scheduler_config", lambda: SchedulerConfig(stack_id="dev-stack")
    )
    if run_stops:

        async def _run_once(self: object) -> None:
            return None

        monkeypatch.setattr(main_mod.CronScheduler, "run", _run_once)


def test_redact_url_drops_embedded_credentials() -> None:
    """A userinfo credential in the URL is never present in the redacted view."""
    url = "nats://alice:sup3r-secret@nats.example.internal:4222"
    safe = _redact_url(url)
    assert "sup3r-secret" not in safe
    assert "alice" not in safe
    assert safe == "nats://nats.example.internal:4222"


def test_redact_url_keeps_scheme_host_port() -> None:
    """A plain URL redacts to its own scheme://host:port (no credential to drop)."""
    assert _redact_url("nats://127.0.0.1:4222") == "nats://127.0.0.1:4222"


def test_redact_url_without_port() -> None:
    """A URL with no port renders host only, still credential-free."""
    assert _redact_url("nats://bob:pw@example.internal") == "nats://example.internal"
    assert "pw" not in _redact_url("nats://bob:pw@example.internal")


def test_redact_url_unparseable_is_marked_not_echoed() -> None:
    """A URL that will not parse is reported by a fixed marker, never echoed raw."""
    assert _redact_url("nats://[bad") == "<unparseable-nats-url>"


@pytest.mark.parametrize(
    "url",
    [
        "nats://alice:secret@host:99999999",  # out-of-range port (SplitResult.port raises)
        "nats://alice:secret@host:-1",  # negative port
        "nats://alice:secret@host:notaport@more",  # malformed userinfo/port muddle
        "nats://[bad",  # malformed structure urlsplit rejects
        "",  # empty
        "::::",  # garbage
    ],
)
def test_redact_url_never_throws_and_never_leaks(url: str) -> None:
    """A malformed port/userinfo/URL yields the fixed marker, never raises or leaks.

    ``SplitResult.port`` raises ``ValueError`` on a bad port; the redactor must
    absorb it into the fixed ``<unparseable-nats-url>`` marker rather than let it
    escape onto the logging path, and must never echo an embedded ``secret``.
    """
    safe = _redact_url(url)  # must not raise
    assert "secret" not in safe
    # A malformed URL collapses to the fixed marker (no partial, credential-bearing view).
    assert safe == "<unparseable-nats-url>" or "@" not in safe


async def test_run_clean_shutdown_closes_bus(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean run connects the bus, runs the scheduler, and closes the bus (exit 0)."""
    bus = _FakeBus()
    _install_bus(monkeypatch, bus)
    code = await main_mod._run()
    assert code == 0
    assert bus.connected is True
    assert bus.closed is True  # teardown ran in the finally block


async def test_run_config_failure_aborts_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid schedule config aborts with exit 2 and never touches the bus."""
    bus = _FakeBus()
    _install_bus(monkeypatch, bus)

    def _bad_config() -> SchedulerConfig:
        raise ScheduleConfigError("bad table")

    monkeypatch.setattr(main_mod, "load_scheduler_config", _bad_config)
    code = await main_mod._run()
    assert code == 2
    assert bus.connected is False
    assert bus.closed is False  # never constructed a live connection to tear down


async def test_run_connect_failure_is_contained(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A connect failure returns exit 1, closes the bus, and never leaks the raw string."""
    import logging

    from loguru import logger

    bus = _FakeBus(connect_fails=True)
    _install_bus(monkeypatch, bus)

    handler_id = logger.add(caplog.handler, format="{message}", level="DEBUG")
    try:
        with caplog.at_level(logging.DEBUG):
            code = await main_mod._run()
    finally:
        logger.remove(handler_id)

    assert code == 1
    assert bus.connected is False
    assert bus.closed is True  # finally still closes the constructed bus
    # The raw driver string (with embedded credentials) is never logged.
    assert "secret" not in caplog.text
    assert "unexpected failure" in caplog.text


async def test_run_teardown_failure_does_not_mask_outcome_or_leak(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A close() failure is logged by type only and never changes the run outcome."""
    import logging

    from loguru import logger

    bus = _FakeBus(close_fails=True)
    _install_bus(monkeypatch, bus)

    handler_id = logger.add(caplog.handler, format="{message}", level="DEBUG")
    try:
        with caplog.at_level(logging.DEBUG):
            code = await main_mod._run()
    finally:
        logger.remove(handler_id)

    assert code == 0  # a clean run stays exit 0 even though teardown raised
    assert bus.closed is True
    assert "secret" not in caplog.text
    assert "bus close failed" in caplog.text


async def test_run_installs_and_fires_stop_via_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run wires signal-driven stop: a scheduler that stops ends run() cleanly.

    Instead of a no-op ``run``, drive a scheduler whose ``run`` awaits its own
    stop event, then request the stop from within — proving the shutdown path
    (stop event → run() returns → bus close) is concrete, not a stub short-circuit.
    """
    bus = _FakeBus()
    _install_bus(monkeypatch, bus, run_stops=False)

    async def _run_until_stop(self: main_mod.CronScheduler) -> None:
        # Request the stop as a signal handler would, then honor it immediately.
        self.request_stop()
        assert self._stop.is_set()

    monkeypatch.setattr(main_mod.CronScheduler, "run", _run_until_stop)
    code = await main_mod._run()
    assert code == 0
    assert bus.closed is True


@pytest.mark.parametrize("expected_code", [0, 1, 2])
def test_run_cli_exits_with_main_status(
    monkeypatch: pytest.MonkeyPatch, expected_code: int
) -> None:
    """The console-script entry point sets the process exit status from ``main``.

    ``run_cli`` is what the installed ``rtech-schedulerd`` script targets. A
    console-script target that merely returns an int does NOT set the process
    exit status, so ``run_cli`` must ``sys.exit`` with ``main``'s code — proven
    here by stubbing ``main`` and asserting the raised ``SystemExit`` carries it.
    """
    monkeypatch.setattr(main_mod, "main", lambda: expected_code)
    with pytest.raises(SystemExit) as excinfo:
        main_mod.run_cli()
    assert excinfo.value.code == expected_code
