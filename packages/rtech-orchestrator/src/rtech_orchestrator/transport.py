"""Define the transport seam the orchestrator depends on.

``FrameBus`` is the Protocol the orchestrator uses to publish and subscribe
typed frames. A fleet-nats-backed adapter (``encode_frame`` / ``decode_frame``
over FleetBus raw bytes) is a later slice; this Protocol is the seam that keeps
the orchestrator transport-agnostic and testable.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from rtech_frames.codec import Frame
from rtech_frames.subs import Subscription

FrameHandler = Callable[[Frame], Awaitable[None]]
UnsubscribeFn = Callable[[], Awaitable[None]]


class FrameTransportError(RuntimeError):
    """Signal a typed transport failure on the frame bus seam.

    ``FrameBus`` implementations raise these instead of swallowing transport
    errors: the orchestrator's dispatch record must reflect CONFIRMED publishes
    only, so a failed publish has to surface as a typed
    error, never a logged no-op.
    """


class FramePublishError(FrameTransportError):
    """Signal that publishing a frame to a subject failed."""


class FrameSubscribeError(FrameTransportError):
    """Signal that subscribing a handler to a subject failed."""


class FrameConfirmError(FrameTransportError):
    """Signal that confirming enqueued publishes with the server failed.

    On core NATS ``publish`` only enqueues the PUB command client-side; the
    server round-trip (``flush``) is the confirmation boundary
    (``docs/dispatch-confirmation-contract.md``). A timeout or transport
    failure here means the batch's hand-off is UNCONFIRMED — the dispatch
    record must not be written.
    """


class FrameBus(Protocol):
    """Publish/subscribe seam for typed frames."""

    async def publish_frame(self, subject: str, frame: Frame) -> None:
        """Publish a typed frame to a subject.

        Args:
            subject: The NATS subject to publish on.
            frame: The typed frame to publish.

        Raises:
            FramePublishError: If the transport could not confirm the publish.
        """
        ...

    async def subscribe_frame(self, subject: str, handler: FrameHandler) -> UnsubscribeFn | None:
        """Subscribe a handler to typed frames on a subject.

        Args:
            subject: The NATS subject to subscribe to.
            handler: The async callable invoked with each decoded frame.

        Returns:
            An idempotent async callable that retires this subscription (stops
            handler delivery) — the handle the agentd ``FrameTurnRunner`` uses to
            retire a sibling/steer lane at task completion — or ``None`` when the
            transport cannot offer per-subscription retirement. Callers that only
            need the subscription may ignore the return.

        Raises:
            FrameSubscribeError: If the transport could not establish the
                subscription.
        """
        ...

    async def subscribe_frame_tracked(self, subject: str, handler: FrameHandler) -> Subscription:
        """Subscribe a handler and return a retirable subscription handle.

        Same delivery semantics as :meth:`subscribe_frame`, but the returned
        :class:`~rtech_frames.subs.Subscription` exposes ``unsubscribe`` so the
        caller can retire the subscription — the handle the typed-bus
        :func:`~rtech_frames.subs.subscribe_all` primitive needs to roll a
        partially-established set back on failure. A bus that cannot
        offer per-subscription retirement cannot support atomic rollback.

        Args:
            subject: The NATS subject to subscribe to.
            handler: The async callable invoked with each decoded frame.

        Returns:
            A retirable :class:`~rtech_frames.subs.Subscription` for this lane.

        Raises:
            FrameSubscribeError: If the transport could not establish the
                subscription.
        """
        ...

    async def confirm_publishes(self) -> None:
        """Confirm every previously enqueued publish with the server.

        The dispatch-confirmation boundary
        (``docs/dispatch-confirmation-contract.md``): on core NATS this is a
        ``flush`` round-trip that returns only after the server has processed
        every PUB enqueued on the connection — including publishes buffered
        across a reconnect. Acknowledged transports (JetStream PubAck) confirm
        inside ``publish_frame`` and implement this as a no-op.

        Raises:
            FrameConfirmError: If the server round-trip failed or timed out;
                the enqueued publishes are UNCONFIRMED and must not be
                recorded as dispatched.
        """
        ...
