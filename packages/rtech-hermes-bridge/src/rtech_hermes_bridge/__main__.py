"""Run the Hermes bridge as a standalone service (``python -m rtech_hermes_bridge``).

The bridge connects two fleet buses (rtech-side and Hermes-side — the same live
bus by default, distinct connections per subject namespace), starts the
translation loop and per-fleet-id heartbeats, and runs until SIGINT/SIGTERM.

Configuration is read from the environment (see :mod:`config`):

* ``RTECH_BRIDGE_FLEET_IDS`` -- comma-separated fleet agent ids to front
  (required, e.g. ``bilby`` or ``bilby,skippy``).
* ``RTECH_BRIDGE_FLEET_ENV`` -- fleet subject env prefix (default ``dev``; must
  match the gateway's ``FLEET_ENV``).
* ``RTECH_BRIDGE_RTECH_NATS_URL`` / ``RTECH_BRIDGE_FLEET_NATS_URL`` -- the two bus
  urls (both default to the live fleet bus).
* ``RTECH_BRIDGE_DEFAULT_CHANNEL`` -- fallback Discord channel for a task that
  carries none (the gateway skips a turn with no resolvable channel).
* ``RTECH_BRIDGE_HEARTBEAT_INTERVAL_SECONDS`` -- picker liveness cadence.
* ``RTECH_BRIDGE_TASK_TIMEOUT_SECONDS`` -- in-flight task expiry (default 900).

Secret env VALUES are never logged; bus urls are host:port only, not secrets.
"""

from __future__ import annotations

import asyncio
import signal
import sys

from fleet_nats import FleetBus
from loguru import logger

from .config import BridgeConfig
from .runtime import HermesBridge, build_buses


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event) -> None:
    """Wire SIGINT/SIGTERM to set ``stop_event`` for a clean shutdown."""

    def _request_stop() -> None:
        logger.info("hermes_bridge.signal | stop requested")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except (NotImplementedError, RuntimeError) as exc:
            logger.warning(
                "hermes_bridge.signal | handler unavailable | sig={} reason={}", sig, exc
            )


async def _run() -> int:
    """Load config, connect both buses, run the bridge until a stop signal."""
    try:
        config = BridgeConfig.from_env()
    except ValueError as exc:
        logger.error("hermes_bridge.run | invalid config | {}", exc)
        return 2
    logger.info(
        "hermes_bridge.run | starting | fleet_ids={} env={} rtech_url={} fleet_url={}",
        config.fleet_ids,
        config.fleet_env,
        config.rtech_nats_url,
        config.fleet_nats_url,
    )

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    _install_signal_handlers(loop, stop_event)

    rtech_bus, fleet_bus = build_buses(config)
    bridge: HermesBridge | None = None
    exit_code = 0
    connected: list[FleetBus] = []
    try:
        await rtech_bus.connect()
        connected.append(rtech_bus)
        await fleet_bus.connect()
        connected.append(fleet_bus)

        bridge = HermesBridge(config=config, rtech_bus=rtech_bus, fleet_bus=fleet_bus)
        await bridge.start()
        logger.info("hermes_bridge.run | started; awaiting tasks")

        await stop_event.wait()
        logger.info("hermes_bridge.run | shutdown starting")
    except Exception as exc:
        logger.error("hermes_bridge.run | unexpected failure reason={}", exc)
        exit_code = 1
    finally:
        if bridge is not None:
            try:
                await bridge.stop()
            except Exception as exc:
                logger.error("hermes_bridge.teardown | bridge stop failed reason={}", exc)
        for bus in connected:
            try:
                await bus.close()
            except Exception as exc:
                logger.error("hermes_bridge.teardown | bus close failed reason={}", exc)
        logger.info("hermes_bridge.run | shutdown complete exit_code={}", exit_code)
    return exit_code


def main() -> int:
    """Entry point: run the bridge event loop and return the exit code."""
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(main())
