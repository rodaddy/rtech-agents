"""Own the atomic all-or-nothing subscription-set primitive for the typed bus.

Three surfaces (orchestrator PR #73, steer lane PR #76, chatd edge PR #79) have
independently hand-rolled "subscribe N subjects sequentially" and independently
reproduced the same defect: a failure partway through leaves earlier
subscriptions live but untracked, so nothing ever retires them. This module is
the structural fix -- ONE primitive owns the sequence:

* :func:`subscribe_all` subscribes each subject in order; if ANY subscribe
  fails it rolls back everything already made and raises
  :class:`FrameSubscribeError`. Rollback is HONEST: any subscription whose
  unsubscribe fails during rollback is still live, so it is retained on a
  :class:`SubscriptionSet` carried by ``FrameSubscribeError.leaked`` -- the
  caller either holds a handle covering every live subscription or holds
  nothing, but a live subscription NEVER escapes without a handle.
* The returned :class:`SubscriptionSet` is the single retire handle: one
  ``retire()`` unsubscribes all, never raises, never lies. A subscription whose
  unsubscribe fails is RETAINED (not forgotten) so a later ``retire()`` retries
  it; ``retire()`` returns True only when the set is fully retired and False
  (with a critical log) while any handle remains live.

Consumers hand in their bus's ``subscribe`` callable, so any seam whose
subscribe returns an object with ``async unsubscribe()`` can adopt the
primitive without changing its handler type.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

__all__ = ["FrameSubscribeError", "Subscription", "SubscriptionSet", "subscribe_all"]


class Subscription(Protocol):
    """Represent the retire surface a bus subscription must expose."""

    async def unsubscribe(self) -> None:
        """Cancel the subscription."""


class SubscriptionSet:
    """Hold a complete subscription set behind ONE honest retire handle.

    Instances are only ever built by :func:`subscribe_all` (or by a rollback
    path wrapping the subscriptions already made), so a live set is complete by
    construction -- there is no partially-populated state to leak.

    Retirement is serialized across callers. A caller waiting behind a failed
    attempt re-drives the retained handles as a new attempt; a caller waiting
    behind a successful attempt observes the retired no-op. Every caller gets
    the honest result of the attempt it observes, and the set is marked retired
    only once the last handle is gone.
    """

    def __init__(self, subscriptions: Sequence[tuple[str, Subscription]]) -> None:
        """Wrap an already-complete ``(subject, subscription)`` sequence.

        An empty set is born already retired: there is nothing to sweep.
        """

        self._subscriptions: list[tuple[str, Subscription]] = list(subscriptions)
        self._retired = not self._subscriptions
        self._retire_lock = asyncio.Lock()

    def __len__(self) -> int:
        """Return the number of still-live subscriptions (0 once fully retired)."""

        return len(self._subscriptions)

    @property
    def retired(self) -> bool:
        """Return whether the set is fully retired (no live handle remains)."""

        return self._retired

    @property
    def subjects(self) -> tuple[str, ...]:
        """Return the still-live subjects (for escalation logs)."""

        return tuple(subject for subject, _ in self._subscriptions)

    async def retire(self) -> bool:
        """Unsubscribe every subscription in the set; never raise, never lie.

        Sweeps each ``(subject, subscription)`` pair. On an unsubscribe failure
        the pair is RETAINED in the set (logged at error level, handle held for
        retry) instead of being discarded, so a later :meth:`retire` call
        retries the retained handles. The set is marked retired only once the
        last handle unsubscribes cleanly.

        Returns:
            True when the set is fully retired (including the already-retired
            no-op), False when at least one subscription is still held (also
            logged at critical level listing the remaining subjects -- the
            caller must retry or escalate).
        """

        async with self._retire_lock:
            if self._retired:
                logger.debug("frames.subs.retire | already retired; no-op")
                return True
            pending = self._subscriptions
            retained: list[tuple[str, Subscription]] = []
            for subject, subscription in pending:
                try:
                    await subscription.unsubscribe()
                except Exception as exc:
                    logger.error(
                        "frames.subs.retire | unsubscribe failed subject={} error_type={}; "
                        "handle retained for retry",
                        subject,
                        type(exc).__name__,
                    )
                    retained.append((subject, subscription))
            self._subscriptions = retained
            if retained:
                logger.critical(
                    "frames.subs.retire | {} subscription(s) still live subjects={}; "
                    "caller must retry or escalate",
                    len(retained),
                    self.subjects,
                )
                return False
            self._retired = True
            logger.debug("frames.subs.retire | retired count={}", len(pending))
            return True


class FrameSubscribeError(RuntimeError):
    """Signal that establishing a subscription set failed after attempted rollback.

    Rollback was attempted for every subscription already made. Any subscription
    that could NOT be rolled back is still live and rides on :attr:`leaked` (an
    empty :class:`SubscriptionSet` when rollback was clean). Callers must never
    discard a non-empty ``leaked`` set -- it is the only handle to the still-live
    subscriptions; retire it (retrying until it returns True) or escalate.
    """

    def __init__(self, message: str, *, leaked: SubscriptionSet | None = None) -> None:
        """Attach the still-live ``leaked`` set (empty when rollback was clean)."""

        super().__init__(message)
        self.leaked: SubscriptionSet = leaked if leaked is not None else SubscriptionSet([])


async def subscribe_all[HandlerT](
    subscribe: Callable[[str, HandlerT], Awaitable[Subscription]],
    entries: Sequence[tuple[str, HandlerT]],
) -> SubscriptionSet:
    """Subscribe every ``(subject, handler)`` entry atomically.

    Subscribes sequentially in the given order. If ANY subscribe fails, every
    subscription already made is rolled back and :class:`FrameSubscribeError` is
    raised. Rollback is honest: any subscription whose unsubscribe fails during
    rollback is still live and rides on ``FrameSubscribeError.leaked`` -- the
    caller either holds a handle covering every live subscription or holds
    nothing, but a live-but-untracked subscription never escapes.

    Args:
        subscribe: The bus's subscribe callable
            (``subject, handler -> Subscription``).
        entries: Ordered ``(subject, handler)`` pairs to subscribe.

    Returns:
        A :class:`SubscriptionSet` covering every entry -- the single retire
        handle.

    Raises:
        FrameSubscribeError: If any subscribe failed; rollback of the earlier
            subscriptions has been attempted and any that could not be rolled
            back rides on the error's ``leaked`` set (never discarded).
    """

    made: list[tuple[str, Subscription]] = []
    for subject, handler in entries:
        try:
            made.append((subject, await subscribe(subject, handler)))
        except Exception as exc:
            # Log/raise the exception TYPE only, never its message: a broker
            # failure can carry connection/credential detail. The safe subject +
            # count keep the escalation actionable; the original exc rides on the
            # `raise ... from exc` chain for the caller's own handling.
            logger.error(
                "frames.subs.subscribe_all | subject={} failed after {} of {}; "
                "rolling back error_type={}",
                subject,
                len(made),
                len(entries),
                type(exc).__name__,
            )
            rollback = SubscriptionSet(made)
            if not await rollback.retire():
                logger.critical(
                    "frames.subs.subscribe_all | rollback incomplete subjects={}; "
                    "handles ride on FrameSubscribeError.leaked -- caller must not discard",
                    rollback.subjects,
                )
            raise FrameSubscribeError(
                f"subscribe to {subject} failed: {type(exc).__name__}", leaked=rollback
            ) from exc
    logger.debug("frames.subs.subscribe_all | subscribed count={}", len(made))
    return SubscriptionSet(made)
