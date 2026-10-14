"""Run the cron scheduler as a standalone long-running service.

On the k-board LXC the scheduler is its own lingering user unit beside the stack
(same pattern as the Telegram edge): it connects a fleet-nats bus, bridges it to the typed
``FrameBus`` seam (the shared :class:`~rtech_orchestrator.typedbus.TypedFrameBus`),
loads the frozen schedule table, and mints TaskFrames on cron occurrences until
SIGINT/SIGTERM.

Run it: ``python -m rtech_schedulerd``. Configuration is read from the
environment:

* ``NATS_URL`` — the fleet NATS server (default the live fleet bus).
* ``RTECH_SCHEDULE_FILE`` — the JSON schedule table (required; fails closed).
* ``RTECH_STACK_ID`` — this stack's id, stamped on every minted frame so a
  stack-guarded orchestrator consumes it. No default: a missing id
  fails loudly at startup.

Secret env VALUES are never logged. A ``NATS_URL`` may embed credentials
(``nats://user:pass@host``), so the raw URL is NEVER logged — only a redacted
``scheme://host:port`` form is. Task text, requester identity, and NATS payloads
are never logged, and raw connection/teardown exception text is never logged —
only the exception TYPE — so a driver error string can never leak a URL,
credential, or payload. Only sanitized schedule/task-id summaries are logged.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from urllib.parse import urlsplit

from fleet_nats import BusConfig, FleetBus
from loguru import logger
from rtech_orchestrator.typedbus import TypedFrameBus

from .config import ScheduleConfigError, load_scheduler_config
from .scheduler import CronScheduler

DEFAULT_NATS_URL = "nats://127.0.0.1:4222"
ENV_NATS_URL = "NATS_URL"
BUS_AGENT_ID = "schedulerd"
BUS_ENV = "dev"


def _resolve_url() -> str:
    """Return the NATS URL from ``NATS_URL`` (default the live fleet bus)."""
    return os.environ.get(ENV_NATS_URL, DEFAULT_NATS_URL)


_UNPARSEABLE_URL = "<unparseable-nats-url>"


def _redact_url(url: str) -> str:
    """Return a log-safe ``scheme://host:port`` view of a NATS URL.

    A ``NATS_URL`` may embed ``user:pass@`` credentials; those (and any query or
    path) are dropped so the value is never a credential in a log line. This
    function NEVER raises and NEVER echoes the raw URL: any parse failure — a
    malformed structure that :func:`urlsplit` rejects, or a malformed/out-of-range
    port or userinfo that raises only when the component is read — collapses to
    the fixed ``<unparseable-nats-url>`` marker. The port access is inside the
    guarded block precisely because ``SplitResult.port`` raises ``ValueError`` on
    a bad port (e.g. ``:99999999``), which would otherwise escape as an unhandled
    exception on the logging path.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname or "<no-host>"
        port = f":{parts.port}" if parts.port is not None else ""
        scheme = parts.scheme or "nats"
    except ValueError:
        return _UNPARSEABLE_URL
    return f"{scheme}://{host}{port}"


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, scheduler: CronScheduler) -> None:
    """Wire SIGINT/SIGTERM to request a clean scheduler shutdown."""

    def _request_stop() -> None:
        logger.info("schedulerd.signal | stop requested")
        scheduler.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except (NotImplementedError, RuntimeError) as exc:
            logger.warning("schedulerd.signal | handler unavailable | sig={} reason={}", sig, exc)


async def _run() -> int:
    """Load config, connect the bus, run the scheduler until a stop signal."""
    url = _resolve_url()
    safe_url = _redact_url(url)
    try:
        config = load_scheduler_config()
    except ScheduleConfigError:
        # The category is enough; the exception text can echo a file path or the
        # raw config, so log the fixed category + type only, never the message.
        logger.error(
            "schedulerd.run | schedule config invalid; aborting | error_type={}",
            ScheduleConfigError.__name__,
        )
        return 2
    logger.info(
        "schedulerd.run | starting | nats_url={} stack_id={} schedules={}",
        safe_url,
        config.stack_id,
        len(config.schedules),
    )

    loop = asyncio.get_running_loop()
    bus_conn: FleetBus | None = None
    exit_code = 0
    try:
        bus_conn = FleetBus(BusConfig(url=url, agent_id=BUS_AGENT_ID, env=BUS_ENV))
        await bus_conn.connect()
        typed_bus = TypedFrameBus(bus_conn)
        scheduler = CronScheduler(config=config, bus=typed_bus)
        _install_signal_handlers(loop, scheduler)
        logger.info("schedulerd.run | scheduler armed; awaiting occurrences")
        await scheduler.run()
        logger.info("schedulerd.run | scheduler run complete")
    except Exception as exc:
        # Exception TYPE only: a driver/connect error string can carry the URL
        # (with embedded credentials) or a payload, so it is never logged raw.
        logger.error("schedulerd.run | unexpected failure | error_type={}", type(exc).__name__)
        exit_code = 1
    finally:
        if bus_conn is not None:
            try:
                await bus_conn.close()
            except Exception as exc:
                logger.error(
                    "schedulerd.teardown | bus close failed | error_type={}",
                    type(exc).__name__,
                )
        logger.info("schedulerd.run | shutdown complete exit_code={}", exit_code)
    return exit_code


def main() -> int:
    """Run the scheduler event loop and return the process exit code."""
    return asyncio.run(_run())


def run_cli() -> None:
    """Synchronous console-script entry point that sets the process exit status.

    The ``rtech-schedulerd`` console script points here. A console-script target
    that merely *returns* an int does not set the process exit status, so this
    wrapper calls :func:`sys.exit` with :func:`main`'s code — a clean run exits 0,
    a config failure exits 2, and a runtime failure exits 1.
    """
    sys.exit(main())


if __name__ == "__main__":
    run_cli()
