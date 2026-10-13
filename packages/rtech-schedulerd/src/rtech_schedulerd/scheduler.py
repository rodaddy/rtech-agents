"""Mint TaskFrames on cron occurrences and publish them to task ingress.

The scheduler is a dumb producing edge. For each enabled schedule it computes the
next occurrence (in the schedule's timezone), waits until that wall-clock moment,
mints an ordinary :class:`~rtech_frames.TaskFrame`, and publishes it on the
canonical ``rtech.task.new`` ingress subject — identical in shape to an
intake-minted task, so downstream agents cannot tell it was scheduled.

Determinism and idempotency:

* The per-occurrence task id is ``sched-<schedule_id>-<epoch>`` where ``epoch`` is
  the integer UTC epoch-seconds of the occurrence. A process restart that
  re-evaluates the SAME occurrence mints the SAME id, so redelivery/recovery
  cannot double-mint a duplicate task for one fire. That deterministic id is
  absorbed by the ORCHESTRATOR's existing schedule-agnostic ingress dedup —
  :meth:`rtech_orchestrator.orchestrator.Orchestrator._on_task` keys off
  ``task.id`` (``self._store.get(task.id)``) and no-ops a same-id/same-payload
  redelivery — so no scheduler-side persistence is needed for at-most-once at the
  consumer. The scheduler keeps NO durable state; the bus + that boundary are the
  only coordination truth. An in-memory guard (bounded, FIFO-evicted) also drops
  a same-occurrence re-fire within a single process run; it is a fast-path only,
  never the durable dedup.

Transport honesty: the bus provides AT-LEAST-ONCE publish (an ambiguous
publish failure may have landed and is retained as a tombstone, never
republished in-process). Exactly-once at the consumer comes from the downstream
TaskFrame-id dedup above, NOT from the transport — this daemon never claims
exactly-once delivery on the wire.
* The clock and the sleep are injected seams (``now`` / ``sleep``), so tests
  drive occurrences deterministically with NO real system-clock sleeps.

Misfire policy is skip-missed: each schedule is armed at the next occurrence
strictly AFTER startup ``now``, so occurrences that would have fired while the
daemon was down are skipped, never fired late in a burst. No misfire state is
persisted — the bus is the only coordination truth.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from loguru import logger
from rtech_frames import TaskFrame
from rtech_orchestrator import subjects

from .config import ScheduleEntry, SchedulerConfig
from .cron import CronParseError

Now = Callable[[], datetime]
Sleep = Callable[[float], Awaitable[None]]

# Deterministic task-id prefix marking a scheduled origin in the id itself.
_TASK_ID_PREFIX = "sched"

# Occurrence reservation states (local dedup, in-process only — the durable
# at-most-once boundary is the downstream consumer's TaskFrame-id dedup).
#
# * ``RESERVED``  — claimed pre-publish; no other concurrent call may publish it.
# * ``PUBLISHED`` — publish + confirm succeeded; a definite duplicate hereafter.
# * ``TOMBSTONE`` — publish was ATTEMPTED and failed AMBIGUOUSLY (the server may
#   or may not have accepted it). Retained, not released, so a retry cannot
#   double-send a possibly-accepted occurrence. Downstream TaskFrame-id dedup is
#   the safety net if the ambiguous publish did land.
_RESERVED = "reserved"
_PUBLISHED = "published"
_TOMBSTONE = "tombstone"

# Hard cap on the in-process reservation map so a long-lived daemon never grows
# unbounded (a per-minute schedule would otherwise add 1440 entries/day forever).
# The map's ONLY job is to drop a same-occurrence re-fire within one process
# (a late-wake overshoot or a concurrent duplicate) — those collide within
# seconds, so a bounded most-recent window is sufficient. Cross-restart and
# ambiguous-publish idempotency rests on the deterministic TaskFrame id plus the
# downstream consumer-boundary dedup, NOT on this map, so evicting the oldest
# entry can never cause a real duplicate: a re-fire of an evicted occurrence is
# absorbed downstream. Eviction is deterministic: strict insertion-order
# (FIFO) oldest-first, so behavior is reproducible and never time/hash dependent.
_MAX_RESERVATIONS = 8192


class FramePublisher(Protocol):
    """The minimal publish seam the scheduler needs (typed-bus subset)."""

    async def publish_frame(self, subject: str, frame: TaskFrame) -> None:
        """Publish a typed frame to a subject.

        Raises:
            Exception: If the transport publish failed (a typed publish error).
        """
        ...

    async def confirm_publishes(self) -> None:
        """Confirm every enqueued publish with the server (flush round-trip).

        Raises:
            Exception: If the confirmation round-trip failed or timed out.
        """
        ...


def occurrence_task_id(schedule_id: str, occurrence: datetime) -> str:
    """Return the deterministic task id for one schedule occurrence.

    ``sched-<schedule_id>-<epoch>``, where ``epoch`` is the integer UTC
    epoch-seconds of the occurrence. Deterministic across process restarts so a
    re-evaluated occurrence yields the SAME id (idempotency) — the id itself is
    the occurrence's identity, never a fresh uuid.

    Args:
        schedule_id: The stable schedule identity.
        occurrence: The occurrence moment (any tz-aware datetime).

    Returns:
        The deterministic, subject-safe task id for this occurrence.
    """
    epoch = int(occurrence.astimezone(UTC).timestamp())
    return f"{_TASK_ID_PREFIX}-{schedule_id}-{epoch}"


@dataclass
class CronScheduler:
    """Arm each enabled schedule and mint a TaskFrame at every occurrence.

    Args:
        config: The validated schedule table plus stack-scope stamp.
        bus: The typed publish seam (``publish_frame`` + ``confirm_publishes``).
        now: Wall-clock source (injected for deterministic tests).
        sleep: Async sleep (injected; tests supply a no-real-time stub).
    """

    config: SchedulerConfig
    bus: FramePublisher
    now: Now = field(default_factory=lambda: _now_utc)
    sleep: Sleep = field(default_factory=lambda: asyncio.sleep)
    # task_id -> reservation state (_RESERVED | _PUBLISHED | _TOMBSTONE),
    # insertion-ordered and bounded to ``_MAX_RESERVATIONS`` with FIFO
    # oldest-first eviction (see :meth:`_remember`) so the map never grows
    # unbounded over a long-lived daemon.
    _reservations: OrderedDict[str, str] = field(default_factory=OrderedDict, init=False)
    # Guards the check-then-reserve so two concurrent calls for one occurrence
    # cannot both claim it — the reservation is atomic under this lock.
    _reserve_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _stop: asyncio.Event = field(default_factory=asyncio.Event, init=False)

    def request_stop(self) -> None:
        """Signal every schedule loop to exit at its next wake (clean shutdown)."""
        logger.info("scheduler.stop | stop requested")
        self._stop.set()

    def _remember(self, task_id: str, state: str) -> None:
        """Record ``task_id``'s reservation state, evicting the oldest if capped.

        Must be called under ``_reserve_lock``. Sets (or updates) the state and
        moves the id to the most-recent end, then drops strictly-oldest entries
        until the map is within ``_MAX_RESERVATIONS`` — deterministic FIFO
        eviction. Evicting an old occurrence is safe: a re-fire of an evicted id
        is absorbed by the downstream TaskFrame-id dedup, never a real duplicate.
        """
        self._reservations[task_id] = state
        self._reservations.move_to_end(task_id)
        while len(self._reservations) > _MAX_RESERVATIONS:
            self._reservations.popitem(last=False)

    async def _reserve(self, entry: ScheduleEntry, task_id: str) -> bool:
        """Atomically claim ``task_id`` for publishing, or reject a duplicate.

        Returns ``True`` iff this call won the reservation (state set to
        ``_RESERVED``). Any prior state — an in-flight ``_RESERVED`` by a
        concurrent call, a completed ``_PUBLISHED``, or an ambiguous
        ``_TOMBSTONE`` — makes this a duplicate that must NOT publish (returns
        ``False``). The check-and-set runs under ``_reserve_lock`` so it is
        atomic against concurrent occurrences of the same id.
        """
        async with self._reserve_lock:
            existing = self._reservations.get(task_id)
            if existing is not None:
                logger.info(
                    "scheduler.fire | duplicate occurrence skipped | schedule={} task_id={} "
                    "state={}",
                    entry.schedule_id,
                    task_id,
                    existing,
                )
                return False
            self._remember(task_id, _RESERVED)
            return True

    async def mint_and_publish(self, entry: ScheduleEntry, occurrence: datetime) -> bool:
        """Mint the occurrence's TaskFrame and publish it to task ingress.

        Idempotent per occurrence with an atomic reservation:

        1. Reserve the deterministic id under a lock. A concurrent call for the
           same occurrence loses the race and returns ``False`` — the two can
           never both publish.
        2. Construct the frame. A construction failure here is DEFINITELY
           pre-publish, so the reservation is released and a genuine retry can
           try again.
        3. Publish + confirm. On failure the outcome is AMBIGUOUS (the server may
           have accepted the frame before the error), so the reservation is
           converted to a bounded tombstone and RETAINED — a retry will not
           double-send. Downstream TaskFrame-id dedup absorbs the duplicate if
           the ambiguous publish did land.

        Args:
            entry: The schedule whose occurrence fired.
            occurrence: The occurrence moment.

        Returns:
            ``True`` when a frame was published and confirmed, ``False`` when the
            occurrence was a duplicate, a pre-publish failure, or an ambiguous
            publish failure.
        """
        task_id = occurrence_task_id(entry.schedule_id, occurrence)
        if not await self._reserve(entry, task_id):
            return False
        try:
            frame = TaskFrame(
                id=task_id,
                origin_edge=entry.origin_edge,
                requester=entry.requester,
                text=entry.text,
                scope=entry.scope,
                participants=list(entry.participants),
                stack=self.config.stack_id,
                created_at=occurrence.astimezone(UTC),
            )
        except Exception as exc:
            # DEFINITELY pre-publish: construction failed before any transport
            # call, so releasing is safe — nothing could have been accepted. A
            # real retry may then succeed. Log the exception TYPE only.
            async with self._reserve_lock:
                self._reservations.pop(task_id, None)
            logger.error(
                "scheduler.fire | frame construction failed; reservation released | "
                "schedule={} task_id={} error_type={}",
                entry.schedule_id,
                task_id,
                type(exc).__name__,
            )
            return False
        try:
            await self.bus.publish_frame(subjects.TASK_INGRESS, frame)
            await self.bus.confirm_publishes()
        except Exception as exc:
            # AMBIGUOUS: the publish was attempted; the server may already have
            # accepted the frame. Retain a bounded tombstone so a retry cannot
            # double-send. Sanitized: schedule id, task id, and exception TYPE
            # only — never the task text, requester, frame payload, or the raw
            # exception message (a transport error string can echo the subject,
            # the URL, or payload bytes).
            async with self._reserve_lock:
                self._remember(task_id, _TOMBSTONE)
            logger.error(
                "scheduler.fire | publish failed; occurrence tombstoned | schedule={} "
                "task_id={} error_type={}",
                entry.schedule_id,
                task_id,
                type(exc).__name__,
            )
            return False
        async with self._reserve_lock:
            self._remember(task_id, _PUBLISHED)
        logger.info(
            "scheduler.fire | task minted | schedule={} task_id={} scope={} subject={}",
            entry.schedule_id,
            task_id,
            entry.scope,
            subjects.TASK_INGRESS,
        )
        return True

    async def _run_schedule(self, entry: ScheduleEntry) -> None:
        """Arm one schedule and fire each occurrence until stop is requested.

        Skip-missed misfire policy: each next occurrence is computed strictly
        after the CURRENT ``now`` at arm time, so a downtime gap is skipped. The
        wait is chunked against the stop event so shutdown is prompt even for a
        far-future occurrence; sleeps are the injected seam (no real-clock sleep
        in tests).

        After the wait returns, the clock is RE-READ: if wake was late — the
        current minute is already past the occurrence's whole-minute window (the
        wait over-slept, or the loop was starved) — the stale occurrence is NOT
        published; the schedule is re-armed and recomputes the next fresh one.
        This closes the gap where a chunk-sleep overshoot would publish a task
        for a minute that has already elapsed.
        """
        while not self._stop.is_set():
            reference = self.now()
            try:
                occurrence = entry.cron.next_after(reference)
            except CronParseError as exc:
                logger.error(
                    "scheduler.arm | no future occurrence; retiring schedule | schedule={} "
                    "error_type={}",
                    entry.schedule_id,
                    type(exc).__name__,
                )
                return
            logger.debug(
                "scheduler.arm | armed | schedule={} next={}",
                entry.schedule_id,
                occurrence.isoformat(),
            )
            if not await self._wait_until(occurrence):
                return
            if self._is_stale(occurrence):
                logger.warning(
                    "scheduler.fire | late wake; occurrence skipped, recomputing | schedule={} "
                    "planned={}",
                    entry.schedule_id,
                    occurrence.isoformat(),
                )
                continue
            await self.mint_and_publish(entry, occurrence)

    def _is_stale(self, occurrence: datetime) -> bool:
        """Return whether ``now`` is past ``occurrence``'s whole-minute window.

        Cron fires on whole minutes; an occurrence is fresh only while the clock
        is still within its minute (``[occurrence, occurrence + 60s)``). A wake
        at or after ``occurrence + 60s`` is late — the planned minute has
        elapsed, so the occurrence is stale and must be skipped, not published.
        """
        return (self.now() - occurrence) >= timedelta(seconds=_OCCURRENCE_WINDOW_SECONDS)

    async def _wait_until(self, moment: datetime) -> bool:
        """Sleep until ``moment`` (injected sleep), honoring the stop event.

        Returns ``False`` as soon as stop is requested, ``True`` when ``moment``
        is reached. Each wait chunk RACES the injected sleeper against the stop
        event (``asyncio.wait(FIRST_COMPLETED)``): a :meth:`request_stop` during
        an active wait wins the race immediately, so shutdown never waits out the
        remaining sleep. The losing awaitable is cancelled and awaited so no task
        is left pending (no leak); a cancellation of this coroutine itself
        propagates after both children are cancelled and awaited.

        The chunk is still bounded by ``_POLL_SECONDS`` so a far-future
        occurrence never parks on one unbounded sleep, and — because the fake
        clock in tests advances only inside ``sleep`` — the sleeper wins each
        deterministic chunk while a real stop event pre-empts it.
        """
        while not self._stop.is_set():
            remaining = (moment - self.now()).total_seconds()
            if remaining <= 0:
                return True
            if await self._sleep_or_stop(min(remaining, _POLL_SECONDS)):
                return False
        return False

    async def _sleep_or_stop(self, seconds: float) -> bool:
        """Race the injected sleeper against the stop event for one chunk.

        Returns ``True`` if the stop event won (stop requested mid-wait), ``False``
        if the sleeper completed the chunk first. Whichever awaitable loses is
        cancelled and awaited so neither is left pending; if THIS coroutine is
        cancelled while racing, both children are cancelled and awaited before the
        cancellation propagates, so a shutdown cancel never leaks a task.
        """
        sleeper = asyncio.ensure_future(self.sleep(seconds))
        stopper = asyncio.ensure_future(self._stop.wait())
        try:
            done, pending = await asyncio.wait(
                {sleeper, stopper}, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            # This coroutine was cancelled mid-race (e.g. gather cancellation on
            # shutdown): cancel and await both children so nothing leaks, then
            # let the cancellation propagate to the caller.
            await _cancel_and_await(sleeper, stopper)
            raise
        await _cancel_and_await(*pending)
        return stopper in done

    async def run(self) -> None:
        """Run every ENABLED schedule concurrently until :meth:`request_stop`.

        Disabled schedules are logged and never armed. All schedule loops run as
        sibling tasks; on stop (or all loops retiring) the gather resolves and the
        method returns for a clean shutdown.

        A valid table with NOTHING to arm — empty, or every schedule disabled — is
        not an error: the daemon stays alive and idle, parked on the stop event,
        so it can be enabled later without a restart and so a lingering unit does
        not flap-exit. It returns only when :meth:`request_stop` is signalled, the
        same clean-shutdown path as an armed run.
        """
        armed = [entry for entry in self.config.schedules if entry.enabled]
        for entry in self.config.schedules:
            if not entry.enabled:
                logger.info(
                    "scheduler.run | schedule disabled; not armed | schedule={}",
                    entry.schedule_id,
                )
        if not armed:
            logger.warning("scheduler.run | no enabled schedules; idling until stop requested")
            await self._stop.wait()
            logger.info("scheduler.run | stop requested; idle scheduler exiting")
            return
        logger.info("scheduler.run | arming | enabled={}", len(armed))
        await asyncio.gather(*(self._run_schedule(entry) for entry in armed))
        logger.info("scheduler.run | all schedule loops exited")


# Chunk length (seconds) for the stop-aware wait: bounds shutdown latency and
# keeps a far-future occurrence from parking on one unbounded sleep.
_POLL_SECONDS = 30.0

# A cron occurrence is a whole minute; it is fresh only within its own minute.
# A wake at or after this many seconds past the occurrence is a late/stale wake.
_OCCURRENCE_WINDOW_SECONDS = 60


async def _cancel_and_await(*tasks: asyncio.Future[Any]) -> None:
    """Cancel each task and await it, swallowing its cancellation.

    Awaiting a cancelled task drains it so no "task was destroyed but it is
    pending" leak survives the race in :meth:`CronScheduler._sleep_or_stop`. An
    already-finished task is a no-op; a still-pending one is cancelled and its
    :class:`asyncio.CancelledError` is absorbed here (the caller's own
    cancellation, if any, is re-raised by the caller, not swallowed).
    """
    for task in tasks:
        task.cancel()
    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass


def _now_utc() -> datetime:
    """Return the current UTC time (injected in tests for determinism)."""
    return datetime.now(UTC)


__all__ = ["CronScheduler", "FramePublisher", "occurrence_task_id"]
