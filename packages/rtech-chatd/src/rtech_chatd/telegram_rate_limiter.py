"""Edge-owned Telegram send pacing: the three Bot API budgets + FloodWait ledger.

Ports the ``TelegramRateLimiter`` prior art from ``b1x-telegram-admin``
(``src/bullish_admin_bot/utils/rate_limiter.py`` — complete but dead code
there, so validated by this port's unit suite). It enforces Telegram's actual
send budgets BEFORE a request goes out, so the normal path never trips a 429:

* ~1 message per second to the same chat;
* ~20 messages per minute to the same group/supergroup;
* ~30 messages per second across all chats;
* FloodWait penalties (a served ``retry_after``) honored per-chat or globally.

Two deliberate deviations from the prior art:

1. **The lock is never held across a sleep.** The b1x implementation slept
   while holding its asyncio lock, which serialized EVERY chat behind the
   single slowest wait. Here the wait is COMPUTED under the lock, slept
   OUTSIDE it, and re-checked — concurrent sends to unrelated chats never
   queue behind one chat's penalty.
2. **No module-level singleton.** The limiter is constructed and injected by
   the composition layer (one per sender pool), with an injectable clock and
   sleeper so the unit suite drives time deterministically.

FloodWait POLICY lives with the caller: when a send DOES receive a 429,
the sender records the penalty here (:meth:`apply_flood_wait`) and re-raises to
the render loop — this limiter only guarantees that no FOLLOW-UP send fires
into a known-active penalty window. It never blind-retries.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable

from loguru import logger

# Telegram's documented practical budgets (Bot API FAQ). The per-chat and
# global windows are seconds; the group window is a minute.
PER_CHAT_MIN_INTERVAL_S = 1.0
GROUP_WINDOW_S = 60.0
GROUP_WINDOW_MAX = 20
GLOBAL_WINDOW_S = 1.0
GLOBAL_WINDOW_MAX = 30

# Bound on the per-chat bookkeeping maps so an unbounded stream of distinct
# chat ids (an adversarial or misbehaving caller) cannot grow memory without
# limit. Oldest-touched entries are evicted first; a chat's stale pacing
# record simply re-admits at full budget, which is the safe direction.
DEFAULT_MAX_TRACKED_CHATS = 4096

# Key used for the GLOBAL FloodWait penalty lane (a 429 with no chat scope).
_GLOBAL = "__global__"

Clock = Callable[[], float]
Sleeper = Callable[[float], Awaitable[None]]


class TelegramRateLimiter:
    """Pace outbound Bot API calls inside Telegram's send budgets.

    All state is monotonic-clock based and in-process: pacing never needs to
    survive a restart (a fresh process starts at full budget, and Telegram
    itself is the durable authority via 429s, which the caller feeds back in
    through :meth:`apply_flood_wait`).
    """

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        sleeper: Sleeper | None = None,
        max_tracked_chats: int = DEFAULT_MAX_TRACKED_CHATS,
    ) -> None:
        if max_tracked_chats < 1:
            raise ValueError("max_tracked_chats must be >= 1")
        self._clock = clock or time.monotonic
        self._sleep: Sleeper = sleeper or asyncio.sleep
        self._max_tracked_chats = max_tracked_chats
        # chat_id -> monotonic time of the last recorded send (1/s pacing).
        self._last_send: OrderedDict[str, float] = OrderedDict()
        # chat_id -> recent send times inside the group minute window.
        self._group_sends: OrderedDict[str, deque[float]] = OrderedDict()
        # Recent send times across ALL chats (30/s window).
        self._global_sends: deque[float] = deque()
        # chat_id (or _GLOBAL) -> monotonic time a FloodWait penalty expires.
        self._flood_until: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, chat_id: str, *, is_group: bool) -> None:
        """Wait until a send to ``chat_id`` fits every budget, then record it.

        Computes the required wait under the lock, sleeps OUTSIDE it, and
        re-checks — a long penalty on one chat never blocks another chat's
        acquire. Returns only after the send slot is recorded, so the caller
        should dispatch immediately.
        """
        while True:
            async with self._lock:
                now = self._clock()
                self._prune_flood(now)
                wait, reasons = self._required_wait(chat_id, now, is_group=is_group)
                if wait <= 0:
                    self._record(chat_id, now, is_group=is_group)
                    return
            logger.info(
                "chatd.tg.ratelimit | waiting {:.2f}s | chat={} reasons={}",
                wait,
                chat_id,
                reasons,
            )
            await self._sleep(wait)

    def apply_flood_wait(self, chat_id: str | None, retry_after_s: float) -> None:
        """Record a served FloodWait penalty (chat-scoped, or global on None).

        Called by the SENDER when Telegram answers 429: the penalty window is
        remembered so no follow-up send fires into it. The triggering error
        itself is re-raised to the render loop by the caller — recording here
        is bookkeeping, not retry policy.
        """
        key = chat_id if chat_id is not None else _GLOBAL
        self._flood_until[key] = self._clock() + max(0.0, retry_after_s)
        # Bound the ledger like the pacing maps: a 429 storm across many
        # distinct chats must not grow it without limit. Eviction is
        # FAIL-CLOSED — the evicted (earliest-expiring) chat penalty folds
        # into the global lane, so no send ever fires into a known penalty
        # window; the cost is other chats briefly sharing the shortest
        # remaining wait.
        while len(self._flood_until) > self._max_tracked_chats:
            chat_keys = [k for k in self._flood_until if k != _GLOBAL]
            if not chat_keys:
                break
            victim = min(chat_keys, key=lambda k: self._flood_until[k])
            folded = self._flood_until.pop(victim)
            self._flood_until[_GLOBAL] = max(self._flood_until.get(_GLOBAL, 0.0), folded)
        logger.warning(
            "chatd.tg.ratelimit | FloodWait penalty recorded | scope={} retry_after={:.1f}s",
            "global" if chat_id is None else chat_id,
            retry_after_s,
        )

    def flood_wait_remaining(self, chat_id: str | None = None) -> float:
        """Return the remaining penalty seconds for a chat (or global), else 0."""
        key = chat_id if chat_id is not None else _GLOBAL
        until = self._flood_until.get(key)
        if until is None:
            return 0.0
        return max(0.0, until - self._clock())

    def _prune_flood(self, now: float) -> None:
        """Drop every expired FloodWait entry (caller holds the lock).

        Without this, a long-lived sender that ever received one 429 in each
        of many distinct chats retains an entry per chat forever — expiry was
        otherwise only checked on a later acquire for that exact chat.
        """
        expired = [key for key, until in self._flood_until.items() if now >= until]
        for key in expired:
            del self._flood_until[key]

    def _required_wait(
        self, chat_id: str, now: float, *, is_group: bool
    ) -> tuple[float, list[str]]:
        """Return (seconds to wait, human reasons) for a send to ``chat_id`` at ``now``."""
        wait = 0.0
        reasons: list[str] = []

        for key, label in ((chat_id, "chat-floodwait"), (_GLOBAL, "global-floodwait")):
            until = self._flood_until.get(key)
            if until is None:
                continue
            if now >= until:
                del self._flood_until[key]
                continue
            wait = max(wait, until - now)
            reasons.append(label)

        last = self._last_send.get(chat_id)
        if last is not None and (now - last) < PER_CHAT_MIN_INTERVAL_S:
            wait = max(wait, PER_CHAT_MIN_INTERVAL_S - (now - last))
            reasons.append("chat-1/s")

        if is_group:
            sends = self._group_sends.get(chat_id)
            if sends is not None:
                while sends and (now - sends[0]) >= GROUP_WINDOW_S:
                    sends.popleft()
                if len(sends) >= GROUP_WINDOW_MAX:
                    wait = max(wait, GROUP_WINDOW_S - (now - sends[0]))
                    reasons.append("group-20/min")

        while self._global_sends and (now - self._global_sends[0]) >= GLOBAL_WINDOW_S:
            self._global_sends.popleft()
        if len(self._global_sends) >= GLOBAL_WINDOW_MAX:
            wait = max(wait, GLOBAL_WINDOW_S - (now - self._global_sends[0]))
            reasons.append("global-30/s")

        return wait, reasons

    def _record(self, chat_id: str, now: float, *, is_group: bool) -> None:
        """Record a send slot for ``chat_id`` at ``now`` (caller holds the lock)."""
        self._last_send[chat_id] = now
        self._last_send.move_to_end(chat_id)
        if is_group:
            sends = self._group_sends.get(chat_id)
            if sends is None:
                sends = deque()
                self._group_sends[chat_id] = sends
            sends.append(now)
            self._group_sends.move_to_end(chat_id)
        self._global_sends.append(now)
        self._evict_over_capacity()

    def _evict_over_capacity(self) -> None:
        """Bound the per-chat maps, evicting oldest-touched chats first.

        An evicted chat re-admits at full budget — the safe direction: the
        worst case is one send arriving slightly early for a chat idle long
        enough to age out of a 4096-chat LRU, not a dropped or delayed send.
        """
        while len(self._last_send) > self._max_tracked_chats:
            self._last_send.popitem(last=False)
        while len(self._group_sends) > self._max_tracked_chats:
            self._group_sends.popitem(last=False)


__all__ = [
    "DEFAULT_MAX_TRACKED_CHATS",
    "GLOBAL_WINDOW_MAX",
    "GLOBAL_WINDOW_S",
    "GROUP_WINDOW_MAX",
    "GROUP_WINDOW_S",
    "PER_CHAT_MIN_INTERVAL_S",
    "TelegramRateLimiter",
]
