"""Test the cron scheduler: exact frames, idempotency, failures, and shutdown.

All timing is driven through injected ``now``/``sleep`` seams — no real
system-clock sleeps run. The fake clock advances only when the scheduler awaits
``sleep``, so occurrences fire deterministically.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from rtech_frames import TaskFrame
from rtech_orchestrator import subjects
from rtech_schedulerd.config import ScheduleEntry, SchedulerConfig, load_schedules
from rtech_schedulerd.scheduler import CronScheduler, occurrence_task_id

pytestmark = pytest.mark.asyncio


class FakeClock:
    """A monotonic fake clock advanced only by the injected ``sleep`` seam."""

    def __init__(self, start: datetime) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    async def sleep(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


class RecordingBus:
    """Capture published frames; optionally fail the publish for one test."""

    def __init__(self, *, fail: bool = False) -> None:
        self.published: list[tuple[str, TaskFrame]] = []
        self.confirms = 0
        self._fail = fail

    async def publish_frame(self, subject: str, frame: TaskFrame) -> None:
        if self._fail:
            raise RuntimeError("transport down")
        self.published.append((subject, frame))

    async def confirm_publishes(self) -> None:
        self.confirms += 1


def _config(*entries: ScheduleEntry, stack_id: str = "dev-stack") -> SchedulerConfig:
    return SchedulerConfig(stack_id=stack_id, schedules=entries)


def _entry(**over: object) -> ScheduleEntry:
    raw: dict[str, object] = {
        "schedule_id": "morning",
        "cron": "0 8 * * *",
        "text": "Summarize overnight",
    }
    raw.update(over)
    return load_schedules([raw], stack_id="dev-stack").schedules[0]


async def _run_until_stop(scheduler: CronScheduler, clock: FakeClock, *, max_wakes: int) -> None:
    """Drive the scheduler, requesting stop after ``max_wakes`` clock advances.

    Wraps the fake sleep so that after a bounded number of advances the stop
    event fires — bounding the test even though the schedule loop is otherwise
    infinite. No real time elapses.
    """
    advances = 0
    real_sleep = clock.sleep

    async def counting_sleep(seconds: float) -> None:
        nonlocal advances
        await real_sleep(seconds)
        advances += 1
        if advances >= max_wakes:
            scheduler.request_stop()

    scheduler.sleep = counting_sleep
    await scheduler.run()


async def test_fire_mints_exact_taskframe_on_ingress() -> None:
    """One occurrence mints an ordinary TaskFrame on the canonical ingress subject."""
    clock = FakeClock(datetime(2026, 1, 1, 7, 59, tzinfo=UTC))
    bus = RecordingBus()
    scheduler = CronScheduler(config=_config(_entry()), bus=bus, now=clock.now, sleep=clock.sleep)
    await _run_until_stop(scheduler, clock, max_wakes=200)

    assert len(bus.published) == 1
    subject, frame = bus.published[0]
    assert subject == subjects.TASK_INGRESS == "rtech.task.new"
    assert isinstance(frame, TaskFrame)
    assert frame.text == "Summarize overnight"
    assert frame.scope == "single"
    assert frame.stack == "dev-stack"
    assert frame.origin_edge == "scheduler"
    assert frame.requester == "scheduler"
    # created_at is the occurrence moment in UTC (08:00 local == 08:00 UTC here).
    assert frame.created_at == datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
    assert bus.confirms == 1


async def test_task_id_is_deterministic_per_occurrence() -> None:
    """The minted id matches the deterministic occurrence id (idempotency key)."""
    clock = FakeClock(datetime(2026, 1, 1, 7, 59, tzinfo=UTC))
    bus = RecordingBus()
    entry = _entry()
    scheduler = CronScheduler(config=_config(entry), bus=bus, now=clock.now, sleep=clock.sleep)
    await _run_until_stop(scheduler, clock, max_wakes=200)

    occurrence = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
    expected = occurrence_task_id(entry.schedule_id, occurrence)
    assert bus.published[0][1].id == expected
    assert expected == f"sched-morning-{int(occurrence.timestamp())}"


async def test_same_occurrence_not_double_minted() -> None:
    """A re-fire of the same occurrence within one process mints no duplicate."""
    clock = FakeClock(datetime(2026, 1, 1, 7, 59, tzinfo=UTC))
    bus = RecordingBus()
    entry = _entry()
    scheduler = CronScheduler(config=_config(entry), bus=bus, now=clock.now, sleep=clock.sleep)

    occurrence = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
    first = await scheduler.mint_and_publish(entry, occurrence)
    second = await scheduler.mint_and_publish(entry, occurrence)
    assert first is True
    assert second is False
    assert len(bus.published) == 1


async def test_restart_recovers_same_id_for_occurrence() -> None:
    """A fresh scheduler (restart) computes the SAME id for the same occurrence."""
    entry = _entry()
    occurrence = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
    id_a = occurrence_task_id(entry.schedule_id, occurrence)
    id_b = occurrence_task_id(entry.schedule_id, occurrence)
    assert id_a == id_b  # deterministic across process boundaries


async def test_disabled_schedule_never_fires() -> None:
    """A disabled schedule is not armed and mints nothing.

    With nothing armed the scheduler stays alive awaiting stop (it does not
    exit), so the run is ended explicitly via ``request_stop`` and asserted to
    resolve cleanly with no frames published.
    """
    clock = FakeClock(datetime(2026, 1, 1, 7, 59, tzinfo=UTC))
    bus = RecordingBus()
    scheduler = CronScheduler(
        config=_config(_entry(enabled=False)), bus=bus, now=clock.now, sleep=clock.sleep
    )
    run_task = asyncio.ensure_future(scheduler.run())
    # Yield so run() reaches its idle wait, then signal stop; run() must resolve.
    await asyncio.sleep(0)
    scheduler.request_stop()
    await asyncio.wait_for(run_task, timeout=1.0)
    assert bus.published == []


async def test_all_disabled_schedules_stay_alive_until_stop() -> None:
    """An all-disabled (valid) table keeps the daemon alive, not flap-exiting.

    ``run()`` must NOT return on its own when nothing is armed — a lingering unit
    would flap-restart. It parks on the stop event and returns only once stop is
    requested, mirroring the empty-table liveness contract.
    """
    clock = FakeClock(datetime(2026, 1, 1, 7, 59, tzinfo=UTC))
    bus = RecordingBus()
    a = _entry(schedule_id="a", enabled=False)
    b_raw = {"schedule_id": "b", "cron": "0 8 * * *", "text": "x", "enabled": False}
    b = load_schedules([b_raw], stack_id="dev-stack").schedules[0]
    scheduler = CronScheduler(config=_config(a, b), bus=bus, now=clock.now, sleep=clock.sleep)
    run_task = asyncio.ensure_future(scheduler.run())
    await asyncio.sleep(0)
    assert not run_task.done()  # still alive: parked on stop, not exited
    scheduler.request_stop()
    await asyncio.wait_for(run_task, timeout=1.0)
    assert bus.published == []


async def test_empty_schedule_table_stays_alive_until_stop() -> None:
    """A valid EMPTY schedule table keeps the daemon alive awaiting stop."""
    clock = FakeClock(datetime(2026, 1, 1, 7, 59, tzinfo=UTC))
    bus = RecordingBus()
    scheduler = CronScheduler(config=_config(), bus=bus, now=clock.now, sleep=clock.sleep)
    run_task = asyncio.ensure_future(scheduler.run())
    await asyncio.sleep(0)
    assert not run_task.done()
    scheduler.request_stop()
    await asyncio.wait_for(run_task, timeout=1.0)
    assert bus.published == []


async def test_enabled_and_disabled_mix_only_enabled_fire() -> None:
    """With a mix, only the enabled schedule mints frames."""
    clock = FakeClock(datetime(2026, 1, 1, 7, 59, tzinfo=UTC))
    bus = RecordingBus()
    on = _entry(schedule_id="on", cron="0 8 * * *")
    off_raw = {"schedule_id": "off", "cron": "0 8 * * *", "text": "x", "enabled": False}
    off = load_schedules([off_raw], stack_id="dev-stack").schedules[0]
    scheduler = CronScheduler(config=_config(on, off), bus=bus, now=clock.now, sleep=clock.sleep)
    await _run_until_stop(scheduler, clock, max_wakes=200)
    assert {frame.text for _, frame in bus.published} == {"Summarize overnight"}
    assert all(frame.id.startswith("sched-on-") for _, frame in bus.published)


async def test_publish_failure_is_contained() -> None:
    """A transport failure returns False and never raises into the loop."""
    clock = FakeClock(datetime(2026, 1, 1, 7, 59, tzinfo=UTC))
    bus = RecordingBus(fail=True)
    entry = _entry()
    scheduler = CronScheduler(config=_config(entry), bus=bus, now=clock.now, sleep=clock.sleep)
    occurrence = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
    result = await scheduler.mint_and_publish(entry, occurrence)
    assert result is False
    assert bus.published == []


async def test_ambiguous_publish_failure_tombstones_no_retry_double_send() -> None:
    """An attempted-then-failed publish is tombstoned; a retry cannot double-send.

    The publish was ATTEMPTED, so the server may have accepted the frame. The
    occurrence is retained as a tombstone (never released), so a retry of the
    same occurrence is a no-op — downstream TaskFrame-id dedup is the safety net
    if the ambiguous publish did land.
    """
    bus = RecordingBus(fail=True)
    entry = _entry()
    scheduler = CronScheduler(config=_config(entry), bus=bus)
    occurrence = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
    task_id = occurrence_task_id(entry.schedule_id, occurrence)

    first = await scheduler.mint_and_publish(entry, occurrence)
    assert first is False
    assert scheduler._reservations[task_id] == "tombstone"

    # A retry does not attempt a second publish — the tombstone blocks it.
    retry = await scheduler.mint_and_publish(entry, occurrence)
    assert retry is False
    assert bus.published == []


async def test_pre_publish_construction_failure_releases_reservation() -> None:
    """A construction failure BEFORE any publish releases the reservation.

    Nothing could have been accepted, so releasing is safe and a genuine retry
    can then succeed. A frozen ``ScheduleEntry`` with an invalid scope forces a
    ``TaskFrame`` construction error while the transport is never touched.
    """
    bus = RecordingBus()
    entry = _entry()
    occurrence = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
    task_id = occurrence_task_id(entry.schedule_id, occurrence)

    # Bypass config validation to inject a scope TaskFrame will reject at
    # construction (definitely pre-publish): object.__setattr__ on the frozen
    # entry, so the failure path under test is the frame constructor, not config.
    object.__setattr__(entry, "scope", "not-a-scope")
    scheduler = CronScheduler(config=_config(entry), bus=bus)

    result = await scheduler.mint_and_publish(entry, occurrence)
    assert result is False
    assert bus.published == []
    # Released, not tombstoned: nothing was sent, so a retry may try again.
    assert task_id not in scheduler._reservations


async def test_concurrent_calls_publish_at_most_once() -> None:
    """Two concurrent calls for one occurrence publish exactly one frame.

    The atomic reservation means only one call wins the claim; the other sees an
    in-flight reservation and returns without publishing — they can never both
    send.
    """
    import asyncio

    class SlowBus(RecordingBus):
        async def publish_frame(self, subject: str, frame: TaskFrame) -> None:
            await asyncio.sleep(0)  # yield so the sibling call interleaves
            await super().publish_frame(subject, frame)

    bus = SlowBus()
    entry = _entry()
    scheduler = CronScheduler(config=_config(entry), bus=bus)
    occurrence = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)

    results = await asyncio.gather(
        scheduler.mint_and_publish(entry, occurrence),
        scheduler.mint_and_publish(entry, occurrence),
    )
    assert sorted(results) == [False, True]
    assert len(bus.published) == 1


async def test_publish_failure_logs_are_sanitized(caplog: pytest.LogCaptureFixture) -> None:
    """The publish-failure log carries ids only — never the task text/requester."""
    import logging

    from loguru import logger

    handler_id = logger.add(caplog.handler, format="{message}", level="DEBUG")
    try:
        with caplog.at_level(logging.DEBUG):
            bus = RecordingBus(fail=True)
            entry = _entry(text="SECRET private summary content", requester="alice-private")
            scheduler = CronScheduler(config=_config(entry), bus=bus)
            await scheduler.mint_and_publish(entry, datetime(2026, 1, 1, 8, 0, tzinfo=UTC))
    finally:
        logger.remove(handler_id)

    logged = caplog.text
    assert "publish failed" in logged
    assert "SECRET private summary content" not in logged
    assert "alice-private" not in logged


async def test_reservation_map_is_bounded_with_fifo_eviction() -> None:
    """The reservation map never exceeds its cap; oldest ids evict first (FIFO).

    A long-lived daemon fires one occurrence per minute for years, so the
    in-process reservation map must be bounded. Publishing more distinct
    occurrences than the cap keeps the map at exactly the cap and evicts the
    strictly-oldest occurrences first — deterministic, never time/hash dependent.
    """
    from rtech_schedulerd import scheduler as scheduler_mod

    bus = RecordingBus()
    entry = _entry()
    sched = CronScheduler(config=_config(entry), bus=bus)
    cap = scheduler_mod._MAX_RESERVATIONS

    base = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
    overflow = 5
    ids = []
    for i in range(cap + overflow):
        occurrence = base + timedelta(minutes=i)
        ids.append(occurrence_task_id(entry.schedule_id, occurrence))
        assert await sched.mint_and_publish(entry, occurrence) is True

    assert len(sched._reservations) == cap
    # The oldest `overflow` ids evicted first; the newest `cap` are retained.
    for evicted in ids[:overflow]:
        assert evicted not in sched._reservations
    for kept in ids[overflow:]:
        assert sched._reservations[kept] == "published"


async def test_shutdown_stops_promptly_before_occurrence() -> None:
    """request_stop ends run() cleanly even with a far-future occurrence pending."""
    clock = FakeClock(datetime(2026, 1, 1, 0, 0, tzinfo=UTC))
    bus = RecordingBus()
    # Far-future daily schedule; stop before it ever fires.
    scheduler = CronScheduler(config=_config(_entry()), bus=bus, now=clock.now, sleep=clock.sleep)
    scheduler.request_stop()
    await scheduler.run()  # already stopped: returns without minting
    assert bus.published == []


async def test_request_stop_interrupts_active_real_wait_immediately() -> None:
    """A stop during a genuinely long wait returns at once — not after the sleep.

    The injected sleeper is a REAL ``asyncio.sleep`` of an hour, so a
    poll/chunk design that waited it out would hang the test. ``request_stop``
    fired from a sibling task must win the race in ``_wait_until`` immediately;
    the whole wait resolves in well under the sleeper's duration. No real
    wall-clock hour elapses.
    """
    import asyncio

    bus = RecordingBus()
    scheduler = CronScheduler(config=_config(_entry()), bus=bus, sleep=asyncio.sleep)
    far_future = datetime(2030, 1, 1, 0, 0, tzinfo=UTC)

    async def _stop_soon() -> None:
        await asyncio.sleep(0)  # let the wait enter its race first
        scheduler.request_stop()

    async with asyncio.timeout(5):
        reached, _ = await asyncio.gather(scheduler._wait_until(far_future), _stop_soon())
    # False == stop won the race (the far-future moment was never reached).
    assert reached is False


async def test_request_stop_wait_leaves_no_pending_task() -> None:
    """The interrupted wait cancels its losing child; nothing is left pending.

    After a stop-interrupted wait, the only task still alive is this test's own
    task — the sleeper spawned inside the race was cancelled and awaited, so it
    is not among the live tasks (no "task was destroyed but pending" leak).
    """
    import asyncio

    bus = RecordingBus()
    scheduler = CronScheduler(config=_config(_entry()), bus=bus, sleep=asyncio.sleep)
    scheduler.request_stop()  # stop already set: the race resolves on the stopper
    before = {id(t) for t in asyncio.all_tasks()}
    reached = await scheduler._wait_until(datetime(2030, 1, 1, tzinfo=UTC))
    assert reached is False
    leftover = {id(t) for t in asyncio.all_tasks()} - before - {id(asyncio.current_task())}
    assert leftover == set()


async def test_wait_cancellation_propagates_without_leak() -> None:
    """Cancelling the wait coroutine propagates the cancel and leaks no child.

    A sibling cancels the running wait task while its real sleeper is pending.
    The cancellation must reach the awaiting caller (the task ends cancelled),
    and the sleeper spawned inside the race must not survive as a pending task.
    """
    import asyncio

    bus = RecordingBus()
    scheduler = CronScheduler(config=_config(_entry()), bus=bus, sleep=asyncio.sleep)
    wait_task = asyncio.ensure_future(scheduler._wait_until(datetime(2030, 1, 1, tzinfo=UTC)))
    await asyncio.sleep(0)  # let the wait enter its race and spawn the sleeper
    live_during = {t for t in asyncio.all_tasks() if t is not wait_task}
    wait_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await wait_task
    # Every task that was alive during the race (besides ours) has settled.
    for task in live_during:
        assert task.done() or task is asyncio.current_task()


async def test_timezone_occurrence_created_at_is_utc() -> None:
    """A tz schedule mints created_at in real UTC, proving tz correctness."""
    clock = FakeClock(datetime(2026, 1, 1, 0, 0, tzinfo=UTC))
    bus = RecordingBus()
    entry = _entry(schedule_id="ny", cron="0 8 * * *", timezone="America/New_York")
    scheduler = CronScheduler(config=_config(entry), bus=bus, now=clock.now, sleep=clock.sleep)
    await _run_until_stop(scheduler, clock, max_wakes=2000)
    # 08:00 America/New_York on 2026-01-01 (EST, UTC-5) == 13:00 UTC.
    assert bus.published[0][1].created_at == datetime(2026, 1, 1, 13, 0, tzinfo=UTC)


async def test_missed_occurrences_are_skipped_not_replayed() -> None:
    """Skip-missed: arming after several missed fires produces only the next one."""
    # Start AFTER 08:00 on Jan 3 — Jan 1 and Jan 2 08:00 occurrences are missed.
    clock = FakeClock(datetime(2026, 1, 3, 9, 0, tzinfo=UTC))
    bus = RecordingBus()
    entry = _entry()
    scheduler = CronScheduler(config=_config(entry), bus=bus, now=clock.now, sleep=clock.sleep)
    await _run_until_stop(scheduler, clock, max_wakes=5000)
    # First (and only, before stop) fire is Jan 4 08:00 — the missed Jan 1-3
    # occurrences are NOT replayed.
    first = bus.published[0][1]
    assert first.created_at == datetime(2026, 1, 4, 8, 0, tzinfo=UTC)


async def test_late_wake_skips_stale_occurrence_and_recomputes() -> None:
    """A wake past the occurrence minute skips the stale fire and re-arms.

    A jumping clock advances by whole days per sleep, so the wait for the first
    08:00 occurrence overshoots into a later day — the planned minute has
    elapsed. That stale occurrence must NOT be published; the schedule recomputes
    and only ever mints a frame whose minute the clock actually lands on.
    """

    class JumpingClock(FakeClock):
        """A clock whose ``sleep`` overshoots by a full day, forcing late wakes."""

        async def sleep(self, seconds: float) -> None:  # noqa: ARG002
            # Always advance a full day regardless of the requested chunk, so the
            # wait lands well past every 08:00 whole-minute window.
            self._now = self._now + timedelta(days=1)

    clock = JumpingClock(datetime(2026, 1, 1, 8, 30, tzinfo=UTC))
    bus = RecordingBus()
    entry = _entry()  # 0 8 * * *
    scheduler = CronScheduler(config=_config(entry), bus=bus, now=clock.now, sleep=clock.sleep)
    await _run_until_stop(scheduler, clock, max_wakes=5)
    # Every wake overshoots the 08:00 minute, so no stale frame is ever published.
    assert bus.published == []


async def test_on_time_wake_is_not_treated_as_stale() -> None:
    """A wake landing exactly in the occurrence minute fires normally (no false skip)."""
    clock = FakeClock(datetime(2026, 1, 1, 7, 59, tzinfo=UTC))
    bus = RecordingBus()
    entry = _entry()
    scheduler = CronScheduler(config=_config(entry), bus=bus, now=clock.now, sleep=clock.sleep)
    await _run_until_stop(scheduler, clock, max_wakes=200)
    assert len(bus.published) == 1
    assert bus.published[0][1].created_at == datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
