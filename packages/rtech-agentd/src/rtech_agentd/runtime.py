"""Runtime assembly — wire a supervisor + ingress onto a FleetBus.

Small orchestration seam used by the daemon entrypoint, the live NATS test, and
the skeleton runner. It keeps the wiring (subscribe every ingress subject, start
the supervisor, run the loops) in ONE place so tests and the real daemon share
the exact same binding logic.

The engine is injected (``ClaudeEngine`` in production, a fake under test) so
this module has no SDK dependency of its own.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from fleet_nats import BusConfig, FleetBus
from loguru import logger

from .engine import EngineSession, ToolsPolicy
from .nats_ingress import NatsIngress
from .supervisor import Supervisor, SupervisorConfig


@dataclass
class Runtime:
    """A bound, running agentd instance (bus + supervisor + ingress)."""

    bus: FleetBus
    supervisor: Supervisor
    ingress: NatsIngress
    _run_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the session, subscribe every ingress subject, run the loops."""
        await self.supervisor.start()
        for subject in self.ingress.subscription_subjects():
            await self.bus.subscribe(subject, self.ingress.handle_envelope)
        self._run_task = asyncio.create_task(self.supervisor.run(), name="agentd-run")
        logger.info("runtime.start | agentd running")

    async def stop(self) -> None:
        """Stop the supervisor and close the bus."""
        await self.supervisor.stop()
        if self._run_task is not None:
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass
        await self.bus.close()
        logger.info("runtime.stop | agentd stopped")


async def build_runtime(
    *,
    session: EngineSession,
    agent: str,
    env: str,
    nats_url: str,
    system_prompt: str,
    tools_policy: ToolsPolicy,
    home: Path,
    default_channel: str | None = None,
    heartbeat_interval_s: float = 30.0,
    stall_timeout_s: float = 90.0,
) -> Runtime:
    """Construct and connect a :class:`Runtime` (does not start the loops).

    Args:
        session: The engine session to manage (injected; SDK-free here).
        agent: Agent id (subject + envelope sender).
        env: Fleet env prefix (``dev``/``prod``).
        nats_url: NATS server URL.
        system_prompt: The composed persona system prompt.
        tools_policy: Allow/deny tool lists for the session.
        home: Agent home dir (session registry lives here).
        default_channel: Fallback channel for envelope→turn mapping.
        heartbeat_interval_s: Heartbeat publish interval.
        stall_timeout_s: Watchdog stall threshold.

    Returns:
        A connected :class:`Runtime`; call :meth:`Runtime.start` to run it.
    """
    bus_config = BusConfig(url=nats_url, agent_id=agent, env=env, name=f"agentd-{agent}")
    bus = FleetBus(bus_config)
    await bus.connect()

    sup_config = SupervisorConfig(
        agent=agent,
        env=env,
        heartbeat_interval_s=heartbeat_interval_s,
        stall_timeout_s=stall_timeout_s,
    )
    supervisor = Supervisor(
        session=session,
        config=sup_config,
        home=home,
        publisher=bus.publish,
        system_prompt=system_prompt,
        tools_policy=tools_policy,
    )
    ingress = NatsIngress(
        agent=agent,
        env=env,
        supervisor=supervisor,
        publisher=bus.publish,
        default_channel=default_channel,
    )
    logger.info("runtime.build | bound | agent={} env={} url={}", agent, env, nats_url)
    return Runtime(bus=bus, supervisor=supervisor, ingress=ingress)
