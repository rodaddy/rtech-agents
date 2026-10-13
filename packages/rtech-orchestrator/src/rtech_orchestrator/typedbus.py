"""Bridge the typed ``FrameBus`` seam onto the live ``fleet_nats.FleetBus``.

This is the ONE shared implementation used by the orchestrator runner
(``rtech_orchestrator.__main__``) and the dev/e2e harnesses
(``scripts/dev_stack.py``, ``scripts/e2e_evidence.py``), which previously each
carried a verbatim copy because scripts cannot import one another. Scripts CAN
import packages, so the bridge lives here and the copies are gone.

Confirmation semantics follow ``docs/dispatch-confirmation-contract.md``:
``publish_frame`` is enqueue-only on core NATS; ``confirm_publishes`` is the
server ``flush`` round-trip that makes a dispatch batch a confirmed hand-off.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

from loguru import logger
from rtech_frames.codec import decode_frame, encode_frame

from .transport import FrameConfirmError, FramePublishError, FrameSubscribeError

if TYPE_CHECKING:
    from fleet_nats import FleetBus
    from nats.aio.msg import Msg
    from rtech_frames.codec import Frame

    from .transport import FrameHandler, UnsubscribeFn

CONFIRM_TIMEOUT_S = 5


class _RawSubscription(Protocol):
    """The retire surface a raw NATS subscription exposes (``nats.aio.Subscription``)."""

    async def unsubscribe(self) -> None:
        """Cancel the live subscription."""


def _idempotent_unsubscribe(subject: str, subscription: _RawSubscription) -> UnsubscribeFn:
    """Wrap a raw NATS subscription in an idempotent async retire callable.

    The frame-turn runner retires a sibling/steer lane exactly once at task
    completion, but a redriven ``SubscriptionSet.retire()`` may call
    the handle again; the second call must be a clean no-op, not a double
    ``unsubscribe`` on an already-closed raw subscription. So the returned
    callable retires the live handle on first use and short-circuits thereafter.
    A real transport failure on the first retire still PROPAGATES (the lane is
    still live and the caller must see it), leaving the handle un-retired so a
    later retry re-drives it.
    """

    retired = False
    retire_lock = asyncio.Lock()

    async def _unsubscribe() -> None:
        nonlocal retired
        async with retire_lock:
            if retired:
                logger.debug("typedbus.unsubscribe | already retired; no-op | subject={}", subject)
                return
            await subscription.unsubscribe()
            retired = True
            logger.debug("typedbus.unsubscribe | retired | subject={}", subject)

    return _unsubscribe


class _NatsSubscription:
    """Adapt a raw NATS subscription to the typed-bus ``Subscription`` protocol.

    Carries the live ``nats.aio`` subscription so the
    :func:`~rtech_frames.subs.subscribe_all` primitive can retire a
    partially-established set. ``unsubscribe`` is idempotent: the
    same shared retire callable the plain :meth:`TypedFrameBus.subscribe_frame`
    return uses, so a redriven ``retire()`` never double-cancels the raw handle.
    """

    def __init__(self, subject: str, subscription: _RawSubscription) -> None:
        """Wrap the live raw subscription handle behind an idempotent retire."""
        self._unsubscribe = _idempotent_unsubscribe(subject, subscription)

    async def unsubscribe(self) -> None:
        """Retire the live NATS subscription (idempotent)."""
        await self._unsubscribe()


class TypedFrameBus:
    """Typed publish/subscribe/confirm over a connected ``fleet_nats.FleetBus``.

    The orchestrator and ``FrameTurnRunner`` depend on the typed seam, while
    the live transport carries raw bytes. Every decode and every handler
    invocation is guarded so one bad frame never kills a subscription.
    """

    def __init__(self, bus: FleetBus, *, confirm_timeout_s: int = CONFIRM_TIMEOUT_S) -> None:
        """Wrap an already-connected ``FleetBus``.

        Args:
            bus: A connected fleet-nats bus for this role.
            confirm_timeout_s: Server round-trip timeout (whole seconds — the
                raw NATS client's ``flush`` takes an int) for
                ``confirm_publishes``.
        """
        self._bus = bus
        self._confirm_timeout_s = confirm_timeout_s

    async def publish_frame(self, subject: str, frame: Frame) -> None:
        """Encode and enqueue a typed frame on the live bus.

        Success means the PUB command was accepted client-side, NOT that the
        server received it — call ``confirm_publishes`` before treating the
        publish as a hand-off (dispatch-confirmation contract).

        Args:
            subject: The NATS subject to publish on.
            frame: The typed frame to publish.

        Raises:
            FramePublishError: If the transport publish failed. Swallowing
                this would let the orchestrator record a dispatch that never
                happened, so the failure is typed and
                loud.
        """
        data = encode_frame(frame)
        try:
            self._bus._guard_payload_size(subject, data)  # noqa: SLF001
            await self._bus._nc.publish(subject, data)  # noqa: SLF001
        except Exception as exc:
            logger.error(
                "typedbus.publish | subject={} failed error_type={}",
                subject,
                type(exc).__name__,
            )
            raise FramePublishError(f"publish to {subject} failed: {type(exc).__name__}") from exc
        logger.debug("typedbus.publish | subject={}", subject)

    async def publish_frame_checked(self, subject: str, frame: Frame) -> None:
        """Publish a typed frame and RE-RAISE on transport failure.

        This shared bus already propagates on :meth:`publish_frame`, so the checked
        twin — required by the agentd ``FrameBus`` structural Protocol both buses
        must satisfy — has identical propagating semantics: encode, guard, publish,
        and raise :class:`FramePublishError` on any transport failure. The wake tool
        (agentd) relies on the propagation to return a typed ``publish_failed``
        error instead of a false success.

        Raises:
            FramePublishError: If the transport publish failed.
        """
        data = encode_frame(frame)
        try:
            self._bus._guard_payload_size(subject, data)  # noqa: SLF001
            await self._bus._nc.publish(subject, data)  # noqa: SLF001
        except Exception as exc:
            logger.error(
                "typedbus.publish_checked | subject={} failed error_type={}",
                subject,
                type(exc).__name__,
            )
            raise FramePublishError(f"publish to {subject} failed: {type(exc).__name__}") from exc
        logger.debug("typedbus.publish_checked | subject={}", subject)

    async def confirm_publishes(self) -> None:
        """Confirm enqueued publishes via a server ``flush`` round-trip.

        Returns only after the server has processed every PUB enqueued on this
        connection — including publishes buffered across a reconnect. This is
        the confirmed hand-off boundary the dispatch record depends on.

        Raises:
            FrameConfirmError: If the round-trip failed or timed out; the
                enqueued publishes are UNCONFIRMED.
        """
        try:
            await self._bus._nc.flush(timeout=self._confirm_timeout_s)  # noqa: SLF001
        except Exception as exc:
            logger.error("typedbus.confirm | flush failed error_type={}", type(exc).__name__)
            raise FrameConfirmError(
                f"publish confirmation flush failed: {type(exc).__name__}"
            ) from exc
        logger.debug("typedbus.confirm | flush ok")

    async def subscribe_frame(self, subject: str, handler: FrameHandler) -> UnsubscribeFn:
        """Subscribe a typed handler on the live bus, returning its retire handle.

        Returns an idempotent async callable that retires this NATS subscription
        (stops handler delivery). The agentd ``FrameTurnRunner`` retires a
        sibling/steer lane through this handle when the task completes; before
        this returned a bare ``None``, so ``_BusSubscription`` had nothing to
        retire and sibling lanes stayed live past the round cap, reinjecting
        turns forever (the debate reinjection bug). Callers that only need the
        subscription — the orchestrator's task-ingress lane — may ignore it.

        Args:
            subject: The NATS subject to subscribe to.
            handler: The async callable invoked with each decoded frame.

        Returns:
            An idempotent async callable retiring this subscription.

        Raises:
            FrameSubscribeError: If the transport subscribe failed; a silently
                missing lane subscription would drop every frame on it.
        """
        return _idempotent_unsubscribe(subject, await self._raw_subscribe(subject, handler))

    async def subscribe_frame_tracked(
        self, subject: str, handler: FrameHandler
    ) -> _NatsSubscription:
        """Subscribe a typed handler and return its retirable subscription handle.

        Same delivery semantics as :meth:`subscribe_frame`, but the returned
        handle wraps the live NATS subscription so a caller — the typed-bus
        :func:`~rtech_frames.subs.subscribe_all` primitive — can retire it to
        roll a partially-established set back.

        Raises:
            FrameSubscribeError: If the transport subscribe failed; a silently
                missing lane subscription would drop every frame on it.
        """
        return _NatsSubscription(subject, await self._raw_subscribe(subject, handler))

    async def _raw_subscribe(self, subject: str, handler: FrameHandler) -> _RawSubscription:
        """Subscribe a guarded decode+dispatch callback, returning the raw handle.

        Every decode and handler invocation is wrapped so one bad frame never
        kills the subscription; the raw NATS subscription is returned so the
        tracked path can retire it.
        """

        async def _on_message(msg: Msg) -> None:
            try:
                frame = decode_frame(msg.data)
            except Exception as exc:
                logger.error(
                    "typedbus.decode | subject={} failed error_type={}",
                    subject,
                    type(exc).__name__,
                )
                return
            try:
                await handler(frame)
            except Exception as exc:
                logger.error(
                    "typedbus.handler | subject={} failed error_type={}",
                    subject,
                    type(exc).__name__,
                )

        try:
            sub: _RawSubscription = await self._bus._nc.subscribe(  # noqa: SLF001
                subject, cb=_on_message
            )
        except Exception as exc:
            logger.error(
                "typedbus.subscribe | subject={} failed error_type={}",
                subject,
                type(exc).__name__,
            )
            raise FrameSubscribeError(
                f"subscribe to {subject} failed: {type(exc).__name__}"
            ) from exc
        logger.debug("typedbus.subscribe | subject={}", subject)
        return sub
