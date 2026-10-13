"""Run the orchestrator as a standalone long-running service.

The local dev stack (``scripts/dev_stack.py``) boots the orchestrator, agentd,
and the web-chat edge together IN-PROCESS. On the k-board LXC each role is its
own systemd unit, so the orchestrator needs a standalone entrypoint that owns
exactly the orchestrator wiring dev_stack does: connect a fleet-nats bus, bridge
it to the typed ``FrameBus`` seam, start the :class:`Orchestrator`, and run until
SIGINT/SIGTERM.

The typed seam is bridged onto the raw transport by the ONE shared
:class:`rtech_orchestrator.typedbus.TypedFrameBus` (also used by dev_stack and
e2e_evidence). Run it: ``python -m rtech_orchestrator``.
Configuration is read from the environment:

* ``NATS_URL`` -- the fleet NATS server (default the live fleet bus).
* ``RTECH_TARGET_AGENT_ID`` -- the agentd id ``scope=single`` tasks route to.
  ``RTECH_SINGLE_AGENT_ID`` is accepted as a back-compat alias, and
  ``RTECH_AGENT_REGISTRY`` may point at a JSON registry instead. There is NO
  hardcoded default: an unresolved target fails loudly at startup (issue #49) so
  the orchestrator never silently routes to a shared ``agentd`` literal.
* ``RTECH_STACK_ID`` -- this stack's id for the ingress guard (issue #75). Only
  ``TaskFrame``s whose ``stack`` field matches are consumed. No default: a
  missing stack id fails loudly at startup so two stacks on the shared bus can
  never cross-consume through a defaulted guard.
* ``RTECH_BRIDGED_AGENTS`` -- optional comma-separated fleet ids fronted by the
  hermes bridge (PR #73 redesign). When set, the orchestrator opens a second
  fleet-namespace connection (``RTECH_FLEET_ENV``, default ``dev``) and consumes
  the typed fleet ``Done`` for those participants directly.

Secret env VALUES are never logged. The bus URL is host:port only, not a secret.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys

from fleet_nats import BusConfig, FleetBus
from loguru import logger

from .config import (
    OrchestratorConfig,
    StackResolutionError,
    TargetResolutionError,
    resolve_bridged_agents,
    resolve_stack_id,
    resolve_target_agent_id,
)
from .fleet_signals import FleetBusDoneSource
from .orchestrator import Orchestrator
from .typedbus import TypedFrameBus

# --- constants ---------------------------------------------------------------

DEFAULT_NATS_URL = "nats://127.0.0.1:4222"
ENV_NATS_URL = "NATS_URL"
BUS_AGENT_ID = "orchestrator"
BUS_ENV = "dev"
# Fleet-side settings, used only when RTECH_BRIDGED_AGENTS is configured. The
# fleet env prefix must match the gateway's FLEET_ENV.
ENV_FLEET_NATS_URL = "RTECH_FLEET_NATS_URL"
ENV_FLEET_ENV = "RTECH_FLEET_ENV"
DEFAULT_FLEET_ENV = "dev"


# --- environment -------------------------------------------------------------


def _resolve_url() -> str:
    """Return the NATS URL from ``NATS_URL`` (default the live fleet bus)."""
    return os.environ.get(ENV_NATS_URL, DEFAULT_NATS_URL)


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event) -> None:
    """Wire SIGINT/SIGTERM to set ``stop_event`` for a clean shutdown."""

    def _request_stop() -> None:
        logger.info("orchestratord.signal | stop requested")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except (NotImplementedError, RuntimeError) as exc:
            logger.warning(
                "orchestratord.signal | handler unavailable | sig={} reason={}", sig, exc
            )


async def _run() -> int:
    """Connect the bus, start the orchestrator, run until a stop signal."""
    url = _resolve_url()
    try:
        single_agent_id = resolve_target_agent_id()
    except TargetResolutionError as exc:
        logger.error("orchestratord.run | target agent id unresolved; aborting | {}", exc)
        return 2
    try:
        stack_id = resolve_stack_id()
    except StackResolutionError as exc:
        logger.error("orchestratord.run | stack id unresolved; aborting | {}", exc)
        return 2
    try:
        bridged_agents = resolve_bridged_agents()
    except ValueError as exc:
        logger.error("orchestratord.run | bridged agents malformed; aborting | {}", exc)
        return 2
    logger.info(
        "orchestratord.run | starting | nats_url={} single_agent_id={} stack_id={}",
        url,
        single_agent_id,
        stack_id,
    )

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    _install_signal_handlers(loop, stop_event)

    bus_conn: FleetBus | None = None
    fleet_conn: FleetBus | None = None
    exit_code = 0
    try:
        bus_conn = FleetBus(BusConfig(url=url, agent_id=BUS_AGENT_ID, env=BUS_ENV))
        await bus_conn.connect()
        typed_bus = TypedFrameBus(bus_conn)
        fleet_signals: FleetBusDoneSource | None = None
        if bridged_agents:
            fleet_url = os.environ.get(ENV_FLEET_NATS_URL, url)
            fleet_env = os.environ.get(ENV_FLEET_ENV, DEFAULT_FLEET_ENV)
            fleet_conn = FleetBus(BusConfig(url=fleet_url, agent_id=BUS_AGENT_ID, env=fleet_env))
            await fleet_conn.connect()
            fleet_signals = FleetBusDoneSource(bus=fleet_conn, env=fleet_env)
            logger.info(
                "orchestratord.run | fleet done consumption enabled | bridged={} env={}",
                sorted(bridged_agents),
                fleet_env,
            )
        orch = Orchestrator(
            bus=typed_bus,
            config=OrchestratorConfig(
                single_agent_id=single_agent_id,
                stack_id=stack_id,
                bridged_agents=bridged_agents,
            ),
            fleet_signals=fleet_signals,
        )
        await orch.start()
        logger.info("orchestratord.run | orchestrator started; awaiting tasks")
        await stop_event.wait()
        logger.info("orchestratord.run | shutdown starting")
    except Exception as exc:
        logger.error("orchestratord.run | unexpected failure reason={}", exc)
        exit_code = 1
    finally:
        for conn, label in ((fleet_conn, "fleet"), (bus_conn, "rtech")):
            if conn is None:
                continue
            try:
                await conn.close()
            except Exception as exc:
                logger.error("orchestratord.teardown | {} bus close failed reason={}", label, exc)
        logger.info("orchestratord.run | shutdown complete exit_code={}", exit_code)
    return exit_code


def main() -> int:
    """Entry point: run the orchestrator event loop and return the exit code."""
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(main())
