"""Deterministic-clock tests for the edge's Telegram send pacer.

No real time passes: a mutable float is the monotonic clock and a fake sleeper
BOTH records the requested sleep AND advances that clock by the slept amount, so
the ``acquire`` retry loop converges exactly as it would in wall time. This
drives every budget (per-chat 1/s, group 20/min, global 30/s), the FloodWait
ledger (chat-scoped and global), penalty expiry, remaining-penalty reporting,
and the LRU eviction bound — all without touching :func:`asyncio.sleep`.
"""

from __future__ import annotations

import asyncio

from rtech_chatd.telegram_rate_limiter import (
    GLOBAL_WINDOW_MAX,
    GROUP_WINDOW_MAX,
    TelegramRateLimiter,
)


class FakeClock:
    """A mutable monotonic clock paired with a clock-advancing sleeper.

    ``sleep(d)`` records ``d`` and advances ``now`` by ``d`` — modelling wall
    time passing during the wait so the pacer's re-check loop terminates.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _limiter(fake: FakeClock, *, max_tracked_chats: int = 4096) -> TelegramRateLimiter:
    return TelegramRateLimiter(
        clock=fake.clock, sleeper=fake.sleep, max_tracked_chats=max_tracked_chats
    )


# --------------------------------------------------------------- per-chat 1/s


def test_first_send_is_immediate() -> None:
    fake = FakeClock()
    limiter = _limiter(fake)
    asyncio.run(limiter.acquire("chat-a", is_group=False))
    assert fake.sleeps == []  # no wait for the very first send


def test_second_send_same_chat_waits_remainder() -> None:
    fake = FakeClock()
    limiter = _limiter(fake)
    asyncio.run(limiter.acquire("chat-a", is_group=False))
    fake.advance(0.3)  # only 0.3s elapsed since the first send
    asyncio.run(limiter.acquire("chat-a", is_group=False))
    # Waited the remaining ~0.7s of the 1s per-chat interval.
    assert len(fake.sleeps) == 1
    assert abs(fake.sleeps[0] - 0.7) < 1e-9


def test_different_chats_do_not_wait_on_each_other() -> None:
    fake = FakeClock()
    limiter = _limiter(fake)
    asyncio.run(limiter.acquire("chat-a", is_group=False))
    # Immediately send to a DIFFERENT chat — the 1/s pacing is per-chat.
    asyncio.run(limiter.acquire("chat-b", is_group=False))
    assert fake.sleeps == []


# --------------------------------------------------------------- group 20/min


def test_group_budget_waits_on_twenty_first_send() -> None:
    fake = FakeClock()
    limiter = _limiter(fake)
    group = "-100999"
    # Fire the full minute budget, stepping past the 1/s per-chat gate each time.
    for _ in range(GROUP_WINDOW_MAX):
        asyncio.run(limiter.acquire(group, is_group=True))
        fake.advance(1.0)
    before = len(fake.sleeps)
    # The 21st send must wait until the OLDEST of the 20 ages out of the minute.
    asyncio.run(limiter.acquire(group, is_group=True))
    assert len(fake.sleeps) > before
    # Oldest send was at t=+0 relative to the 20 sends; 20 sends spanned 20s, so
    # the wait is ~ (60 - elapsed_since_oldest).
    assert fake.sleeps[-1] > 0.0


def test_non_group_send_skips_group_budget() -> None:
    fake = FakeClock()
    limiter = _limiter(fake)
    chat = "555"
    # Even 25 sends never trip a group budget when is_group is False; only the
    # 1/s and 30/s budgets apply. Step past 1/s each time.
    for _ in range(25):
        asyncio.run(limiter.acquire(chat, is_group=False))
        fake.advance(1.0)
    # No group-window wait was ever imposed (all waits, if any, were 1/s = 0 here
    # because we advanced a full second between each).
    assert fake.sleeps == []


# --------------------------------------------------------------- global 30/s


def test_global_budget_across_different_chats() -> None:
    fake = FakeClock()
    limiter = _limiter(fake)
    # 30 sends to 30 DISTINCT chats in the same instant: none hit the per-chat
    # 1/s gate, but the 31st trips the global 30/s window.
    for i in range(GLOBAL_WINDOW_MAX):
        asyncio.run(limiter.acquire(f"chat-{i}", is_group=False))
    assert fake.sleeps == []  # first 30 all immediate
    asyncio.run(limiter.acquire("chat-overflow", is_group=False))
    assert len(fake.sleeps) == 1
    # Oldest global send was at t=0; the window is 1s, so wait ~1s.
    assert abs(fake.sleeps[0] - 1.0) < 1e-9


# --------------------------------------------------------------- FloodWait ledger


def test_apply_flood_wait_chat_scoped_blocks_only_that_chat() -> None:
    fake = FakeClock()
    limiter = _limiter(fake)
    limiter.apply_flood_wait("chat-a", 5.0)
    # chat-a's next acquire waits ~5s...
    asyncio.run(limiter.acquire("chat-a", is_group=False))
    assert len(fake.sleeps) == 1
    assert abs(fake.sleeps[0] - 5.0) < 1e-9
    sleeps_before = len(fake.sleeps)
    # ...while a DIFFERENT chat proceeds without waiting on that penalty.
    asyncio.run(limiter.acquire("chat-b", is_group=False))
    assert len(fake.sleeps) == sleeps_before  # no new wait for chat-b


def test_apply_flood_wait_global_blocks_all_chats() -> None:
    fake = FakeClock()
    limiter = _limiter(fake)
    limiter.apply_flood_wait(None, 4.0)  # global penalty, no chat scope
    asyncio.run(limiter.acquire("chat-a", is_group=False))
    assert len(fake.sleeps) == 1
    assert abs(fake.sleeps[0] - 4.0) < 1e-9


def test_flood_wait_penalty_expiry_clears() -> None:
    fake = FakeClock()
    limiter = _limiter(fake)
    limiter.apply_flood_wait("chat-a", 5.0)
    fake.advance(5.0)  # the whole penalty window has elapsed
    asyncio.run(limiter.acquire("chat-a", is_group=False))
    assert fake.sleeps == []  # expired penalty imposes no wait


def test_flood_wait_remaining_reporting() -> None:
    fake = FakeClock()
    limiter = _limiter(fake)
    assert limiter.flood_wait_remaining("chat-a") == 0.0
    limiter.apply_flood_wait("chat-a", 10.0)
    assert abs(limiter.flood_wait_remaining("chat-a") - 10.0) < 1e-9
    fake.advance(4.0)
    assert abs(limiter.flood_wait_remaining("chat-a") - 6.0) < 1e-9
    fake.advance(6.0)
    assert limiter.flood_wait_remaining("chat-a") == 0.0  # never negative
    # A global penalty is reported under the None scope.
    limiter.apply_flood_wait(None, 3.0)
    assert abs(limiter.flood_wait_remaining() - 3.0) < 1e-9
    assert limiter.flood_wait_remaining("chat-a") == 0.0  # chat scope unaffected


# --------------------------------------------------------------- LRU eviction


def test_lru_eviction_readmits_evicted_chat_at_full_budget() -> None:
    fake = FakeClock()
    limiter = _limiter(fake, max_tracked_chats=2)
    # Send to chat-a, then fill the two-slot LRU with two OTHER chats, evicting
    # chat-a's pacing record (it was the oldest-touched).
    asyncio.run(limiter.acquire("chat-a", is_group=False))
    asyncio.run(limiter.acquire("chat-b", is_group=False))
    asyncio.run(limiter.acquire("chat-c", is_group=False))  # evicts chat-a
    # chat-a re-admits at FULL budget: an immediate re-send imposes no per-chat
    # wait even though under 1s has passed, because its record was evicted.
    sleeps_before = len(fake.sleeps)
    asyncio.run(limiter.acquire("chat-a", is_group=False))
    assert len(fake.sleeps) == sleeps_before  # no 1/s wait — record was evicted


# ------------------------------------------- gauntlet fix regressions


def test_acquire_prunes_expired_flood_entries_for_other_chats() -> None:
    # Expired penalties for OTHER chats (and the global lane) are swept on any
    # acquire — without the sweep, one historical 429 per distinct chat pins a
    # ledger entry forever (expiry was only checked per exact chat).
    fake = FakeClock()
    limiter = _limiter(fake)
    limiter.apply_flood_wait("chat-a", 5.0)
    limiter.apply_flood_wait("chat-b", 5.0)
    limiter.apply_flood_wait(None, 5.0)
    fake.advance(10.0)
    asyncio.run(limiter.acquire("chat-z", is_group=False))
    assert limiter._flood_until == {}  # whole ledger swept, not just chat-z


def test_flood_ledger_is_bounded_and_folds_into_global() -> None:
    # a 429 storm across many distinct chats must not
    # grow the FloodWait ledger without limit. Eviction is fail-closed: the
    # evicted chat penalty folds into the global lane.
    fake = FakeClock()
    limiter = _limiter(fake, max_tracked_chats=2)
    limiter.apply_flood_wait("chat-a", 5.0)
    limiter.apply_flood_wait("chat-b", 10.0)
    limiter.apply_flood_wait("chat-c", 15.0)  # over the bound; chat-a evicted
    assert len(limiter._flood_until) <= 2
    # The evicted (earliest-expiring) penalty landed on the global lane, so no
    # send anywhere fires before chat-a's window would have expired.
    assert limiter.flood_wait_remaining() >= 5.0
    assert limiter.flood_wait_remaining("chat-c") >= 15.0  # newest survives
