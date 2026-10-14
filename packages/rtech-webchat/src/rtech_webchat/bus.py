"""Adapt FleetBus to raw rtech frame bytes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

from fleet_nats import BusConfig, FleetBus
from loguru import logger

if TYPE_CHECKING:
    from nats.aio.msg import Msg

FrameHandler = Callable[[bytes], Awaitable[None]]


class Subscription(Protocol):
    """Represent the unsubscribe surface used by websocket streams."""

    async def unsubscribe(self) -> None:
        """Cancel the subscription."""


class FrameBus(Protocol):
    """Publish and subscribe to encoded rtech frame bytes."""

    async def publish(self, subject: str, data: bytes) -> None:
        """Publish encoded frame bytes."""

    async def subscribe(self, subject: str, handler: FrameHandler) -> Subscription:
        """Subscribe to encoded frame bytes."""

    def is_connected(self) -> bool:
        """Report whether the underlying transport is currently connected.

        This is the typed liveness surface the presence pipeline consults at read
        time: a subscription established once can still be silently
        severed by a later connection drop, so the health verdict must be able to
        re-derive ``subscription_failed`` from live transport state rather than a
        one-shot success flag.
        """


class NatsFrameBus:
    """Use FleetBus for connection lifecycle while carrying raw frame bytes."""

    def __init__(self, nats_url: str, *, env: str = "dev") -> None:
        """Create the bus adapter without connecting."""

        self._bus = FleetBus(
            BusConfig(
                url=nats_url,
                agent_id="web-chat",
                env=env,
                name="rtech-webchat",
            )
        )

    async def __aenter__(self) -> NatsFrameBus:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def connect(self) -> None:
        """Open the underlying FleetBus connection."""

        logger.debug("webchat.bus.connect | entry")
        try:
            await self._bus.connect()
        except Exception as exc:
            # Log only the exception *type*, never str(exc): a FleetBus connect
            # failure can embed the credential-bearing nats://user:pass@host URL,
            # which must never reach a log line. Matches the guarded presence
            # subscribe path (app.py) and the repo safe-log convention.
            logger.error(
                "webchat.bus.connect | failed error_type={}",
                type(exc).__name__,
            )
            raise
        logger.info("webchat.bus.connect | connected")

    def is_connected(self) -> bool:
        """Report whether the underlying FleetBus transport is connected.

        ``FleetBus.is_connected`` is a property; this method wraps it so the edge
        exposes one uniform callable liveness probe.
        """

        return self._bus.is_connected

    async def close(self) -> None:
        """Close the underlying FleetBus connection."""

        logger.debug("webchat.bus.close | entry")
        try:
            await self._bus.close()
        except Exception as exc:
            logger.error("webchat.bus.close | failed reason={}", exc)
            raise
        logger.info("webchat.bus.close | closed")

    async def publish(self, subject: str, data: bytes) -> None:
        """Publish encoded frame bytes on the connected FleetBus client."""

        logger.debug("webchat.bus.publish | subject={} byte_count={}", subject, len(data))
        try:
            self._bus._guard_payload_size(subject, data)  # noqa: SLF001
            await self._bus._nc.publish(subject, data)  # noqa: SLF001
        except Exception as exc:
            logger.error("webchat.bus.publish | subject={} failed reason={}", subject, exc)
            raise
        logger.info("webchat.bus.publish | subject={}", subject)

    async def subscribe(self, subject: str, handler: FrameHandler) -> Subscription:
        """Subscribe to raw frame bytes."""

        logger.debug("webchat.bus.subscribe | subject={}", subject)

        async def _on_message(msg: Msg) -> None:
            try:
                await handler(msg.data)
            except Exception as exc:
                logger.error(
                    "webchat.bus.subscribe | handler failed subject={} reason={}",
                    subject,
                    exc,
                )

        try:
            subscription = await self._bus.raw_subscribe(subject, _on_message)
        except Exception as exc:
            # Log only the exception *type*, never str(exc): a FleetBus/NATS
            # subscribe failure can embed the credential-bearing
            # nats://user:pass@host URL, which must never reach a log line. This
            # is the owning bus boundary, so it must sanitize before app.py's
            # safe presence log ever sees the failure. Matches the guarded
            # connect path above and the repo safe-log convention.
            logger.error(
                "webchat.bus.subscribe | subject={} failed error_type={}",
                subject,
                type(exc).__name__,
            )
            raise
        logger.info("webchat.bus.subscribe | subject={}", subject)
        return subscription
