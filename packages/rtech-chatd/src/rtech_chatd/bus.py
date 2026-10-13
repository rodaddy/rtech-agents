"""Adapt FleetBus to raw rtech frame bytes for the chatd edge."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

from fleet_nats import BusConfig, FleetBus
from loguru import logger

if TYPE_CHECKING:
    from nats.aio.msg import Msg

FrameHandler = Callable[[bytes], Awaitable[None]]


class Subscription(Protocol):
    """Represent the unsubscribe surface used by outbound render streams."""

    async def unsubscribe(self) -> None:
        """Cancel the subscription."""


class FrameBus(Protocol):
    """Publish and subscribe to encoded rtech frame bytes."""

    async def publish(self, subject: str, data: bytes) -> None:
        """Publish encoded frame bytes."""

    async def subscribe(self, subject: str, handler: FrameHandler) -> Subscription:
        """Subscribe to encoded frame bytes."""


class NatsFrameBus:
    """Use FleetBus for connection lifecycle while carrying raw frame bytes."""

    def __init__(self, nats_url: str) -> None:
        """Create the bus adapter without connecting."""

        self._bus = FleetBus(
            BusConfig(
                url=nats_url,
                agent_id="chatd",
                env="dev",
                name="rtech-chatd",
            )
        )

    async def __aenter__(self) -> NatsFrameBus:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def connect(self) -> None:
        """Open the underlying FleetBus connection."""

        logger.debug("chatd.bus.connect | entry")
        try:
            await self._bus.connect()
        except Exception as exc:
            # Emit only the exception TYPE. A NATS connect failure can carry the
            # credentialed URL (nats://host?token=...) in its message text;
            # rendering exc here would leak it into the logs.
            logger.error("chatd.bus.connect | failed error_type={}", type(exc).__name__)
            raise
        logger.info("chatd.bus.connect | connected")

    async def close(self) -> None:
        """Close the underlying FleetBus connection."""

        logger.debug("chatd.bus.close | entry")
        try:
            await self._bus.close()
        except Exception as exc:
            # Emit only the exception TYPE. A close failure can surface the
            # credentialed URL (nats://host?token=...) in its message text;
            # rendering exc here would leak it into the logs.
            logger.error("chatd.bus.close | failed error_type={}", type(exc).__name__)
            raise
        logger.info("chatd.bus.close | closed")

    async def publish(self, subject: str, data: bytes) -> None:
        """Publish encoded frame bytes on the connected FleetBus client."""

        logger.debug("chatd.bus.publish | subject={} byte_count={}", subject, len(data))
        try:
            self._bus._guard_payload_size(subject, data)  # noqa: SLF001
            await self._bus._nc.publish(subject, data)  # noqa: SLF001
        except Exception as exc:
            # Emit only the exception TYPE and safe subject context. The message
            # text of a callback publish failure can carry the credentialed NATS
            # URL (nats://host?token=...); rendering exc here would leak it past
            # an already-sanitized caller.
            logger.error(
                "chatd.bus.publish | subject={} failed error_type={}",
                subject,
                type(exc).__name__,
            )
            raise
        logger.info("chatd.bus.publish | subject={}", subject)

    async def subscribe(self, subject: str, handler: FrameHandler) -> Subscription:
        """Subscribe to raw frame bytes."""

        logger.debug("chatd.bus.subscribe | subject={}", subject)

        async def _on_message(msg: Msg) -> None:
            try:
                await handler(msg.data)
            except Exception as exc:
                # Emit only the exception TYPE and safe subject context. A handler
                # failure can surface the credentialed NATS URL (nats://host?token=...)
                # in its message text; rendering exc here would leak it into the logs.
                logger.error(
                    "chatd.bus.subscribe | handler failed subject={} error_type={}",
                    subject,
                    type(exc).__name__,
                )

        try:
            subscription = await self._bus.raw_subscribe(subject, _on_message)
        except Exception as exc:
            # Emit only the exception TYPE and safe subject context. A subscribe
            # failure can surface the credentialed NATS URL (nats://host?token=...)
            # in its message text; rendering exc here would leak it into the logs.
            logger.error(
                "chatd.bus.subscribe | subject={} failed error_type={}",
                subject,
                type(exc).__name__,
            )
            raise
        logger.info("chatd.bus.subscribe | subject={}", subject)
        return subscription
