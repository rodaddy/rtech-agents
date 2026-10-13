"""Adapt ``FleetBus`` to raw rtech frame bytes on the rtech-side bus.

The rtech contract carries ``rtech_frames`` frames as raw JSON bytes (encoded via
``rtech_frames.codec``), while ``FleetBus.publish`` only takes a fleet
``Envelope``. This thin wrapper reuses ``FleetBus`` for its connection lifecycle,
auth, and payload-size guard while publishing/subscribing raw frame bytes — the
same pattern the web-chat edge (``rtech_webchat.bus.NatsFrameBus``) and agentd
(``rtech_agentd.__main__.TypedFrameBus``) use. It is kept local to the bridge so
the bridge does not depend on the heavier web-chat package for a two-method seam.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from fleet_nats import FleetBus
    from nats.aio.msg import Msg
    from nats.aio.subscription import Subscription


class RtechFrameBus:
    """Publish/subscribe raw rtech frame bytes over a connected ``FleetBus``."""

    def __init__(self, bus: FleetBus) -> None:
        """Wrap an already-constructed ``FleetBus`` (connected by the caller)."""
        self._bus = bus

    async def raw_publish(self, subject: str, data: bytes) -> None:
        """Publish raw frame bytes on the connected bus, size-guarded.

        Args:
            subject: The NATS subject to publish on.
            data: The already-encoded frame bytes.

        Raises:
            Exception: Re-raises any publish failure so the caller's guard can
                roll back idempotency state and let a retry re-publish.
        """
        try:
            self._bus._guard_payload_size(subject, data)  # noqa: SLF001
            await self._bus._nc.publish(subject, data)  # noqa: SLF001
        except Exception as exc:
            logger.error("rtech_frame_bus.publish | subject={} failed reason={}", subject, exc)
            raise
        logger.debug("rtech_frame_bus.publish | subject={} bytes={}", subject, len(data))

    async def raw_subscribe(
        self, subject: str, handler: Callable[[Msg], Awaitable[None]]
    ) -> Subscription:
        """Subscribe with the raw :class:`Msg` so the handler owns frame decoding.

        Args:
            subject: The NATS subject to subscribe to.
            handler: Async callback invoked with each raw message.

        Returns:
            The NATS subscription handle.
        """
        logger.debug("rtech_frame_bus.subscribe | subject={}", subject)
        return await self._bus.raw_subscribe(subject, handler)
