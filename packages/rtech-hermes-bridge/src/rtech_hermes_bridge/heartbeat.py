"""Publish a fronted fleet id's liveness heartbeat onto the rtech bus.

The rtech web-chat picker lists an agent when it observes a heartbeat on
``rtech.agent.*.heartbeat`` and records the envelope's ``sender`` as the agent id
(``rtech_webchat.presence.PresenceRegistry.handle_heartbeat``). A bridged Hermes
agent has no agentd emitting that beat, so the bridge emits one PER FRONTED FLEET
ID — stamped with ``sender=<fleet_id>`` — so ``bilby`` (and any other fronted id)
appears in the picker exactly like a native agentd agent.

The beat is a fleet ``Envelope`` (``kind="heartbeat"``) on the rtech heartbeat
subject — the SAME shape agentd emits (``rtech_agentd.heartbeat``), so the picker
cannot tell a bridged agent from a native one. Built via ``Envelope.new`` (the
sanctioned encoder), never hand-rolled.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from uuid import uuid4

from fleet_nats import Envelope, FleetBus
from loguru import logger

Sleep = Callable[[float], Awaitable[None]]

_HEARTBEAT_KIND = "heartbeat"


def rtech_heartbeat_subject(fleet_id: str) -> str:
    """Return the rtech heartbeat subject for a fronted fleet id.

    Matches the wildcard the web-chat presence registry subscribes
    (``rtech.agent.*.heartbeat``) and the subject agentd emits on, so a bridged
    agent's beat is indistinguishable from a native one.

    Raises:
        ValueError: If ``fleet_id`` is empty.
    """
    if not fleet_id:
        raise ValueError("fleet_id must be non-empty")
    return f"rtech.agent.{fleet_id}.heartbeat"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_id() -> str:
    return uuid4().hex


class BridgeHeartbeat:
    """Publish a fronted fleet id's heartbeat at a fixed cadence."""

    def __init__(
        self,
        *,
        bus: FleetBus,
        fleet_id: str,
        interval_s: float,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        """Configure the heartbeat.

        Args:
            bus: The connected rtech-side bus to publish the beat on.
            fleet_id: The fronted fleet agent id (stamped as the envelope sender).
            interval_s: Seconds between beats.
            sleep: The sleep primitive (injectable for tests).

        Raises:
            ValueError: If ``interval_s`` is not positive.
        """
        if interval_s <= 0:
            raise ValueError(f"interval_s must be positive, got {interval_s}")
        self._bus = bus
        self._fleet_id = fleet_id
        self._interval_s = interval_s
        self._sleep = sleep
        self._subject = rtech_heartbeat_subject(fleet_id)
        self._stopped = asyncio.Event()

    @property
    def subject(self) -> str:
        """Return the heartbeat subject this publisher emits to."""
        return self._subject

    def stop(self) -> None:
        """Signal the run loop to exit after the current beat/sleep."""
        self._stopped.set()

    async def run(self) -> None:
        """Publish beats until cancelled or stopped."""
        logger.debug("bridge_heartbeat.run | entry | subject={}", self._subject)
        try:
            while not self._stopped.is_set():
                await self.publish_once()
                await self._sleep(self._interval_s)
        except asyncio.CancelledError:
            logger.info("bridge_heartbeat.run | cancelled | subject={}", self._subject)
            raise
        except Exception as exc:
            logger.error("bridge_heartbeat.run | failed | subject={} reason={}", self._subject, exc)
            raise

    async def publish_once(self) -> None:
        """Publish one heartbeat envelope for the fronted fleet id.

        A publish failure is logged and swallowed (heartbeat is best-effort — a
        dropped beat must not kill the loop; the next beat recovers presence).
        """
        envelope = Envelope.new(
            msg_id=_new_id(),
            ts=_now_iso(),
            sender=self._fleet_id,
            kind=_HEARTBEAT_KIND,
            payload={"bridge": True},
        )
        try:
            await self._bus.publish(self._subject, envelope)
        except Exception as exc:
            logger.error(
                "bridge_heartbeat.publish | failed | subject={} reason={}", self._subject, exc
            )
            return
        logger.debug("bridge_heartbeat.publish | sent | subject={}", self._subject)
