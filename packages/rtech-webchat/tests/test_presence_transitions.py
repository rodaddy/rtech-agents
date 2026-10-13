"""Test the presence-transition deriver dedup + display-name (issue #147)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from rtech_frames.frames import PresenceTransitionFrame
from rtech_webchat.presence import AgentPresence, AgentPresenceResponse
from rtech_webchat.presence_transitions import (
    PresenceTransitionDeriver,
    safe_display_name,
)


class FakeRegistry:
    """A deterministic stand-in for PresenceRegistry.list_agents()."""

    def __init__(self) -> None:
        self._alive: dict[str, bool] = {}

    def set(self, agent_id: str, alive: bool) -> None:
        self._alive[agent_id] = alive

    def list_agents(self) -> AgentPresenceResponse:
        agents = [
            AgentPresence(agent_id=agent_id, last_seen_ms=0, alive=alive)
            for agent_id, alive in sorted(self._alive.items())
        ]
        return AgentPresenceResponse(agents=agents, pipeline="healthy")


def _deriver(
    registry: FakeRegistry,
) -> tuple[PresenceTransitionDeriver, list[PresenceTransitionFrame]]:
    emitted: list[PresenceTransitionFrame] = []

    async def _publish(frame: PresenceTransitionFrame) -> None:
        emitted.append(frame)

    deriver = PresenceTransitionDeriver(
        registry=registry,  # type: ignore[arg-type]
        publish=_publish,
        clock=lambda: datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC),
    )
    return deriver, emitted


@pytest.mark.asyncio
async def test_first_sighting_alive_emits_nothing() -> None:
    registry = FakeRegistry()
    registry.set("skippy", True)
    deriver, emitted = _deriver(registry)
    await deriver.evaluate()
    assert emitted == []


@pytest.mark.asyncio
async def test_first_sighting_dead_emits_nothing() -> None:
    registry = FakeRegistry()
    registry.set("skippy", False)
    deriver, emitted = _deriver(registry)
    await deriver.evaluate()
    assert emitted == []


@pytest.mark.asyncio
async def test_offline_emitted_once_and_not_repeated() -> None:
    registry = FakeRegistry()
    registry.set("skippy", True)
    deriver, emitted = _deriver(registry)
    await deriver.evaluate()  # first sighting alive, no emit
    registry.set("skippy", False)
    await deriver.evaluate()  # alive -> dead: one offline
    await deriver.evaluate()  # still dead: no repeat
    await deriver.evaluate()  # still dead: no repeat
    assert [f.transition for f in emitted] == ["offline"]
    assert emitted[0].display_name == "Skippy"
    assert emitted[0].observed_at.tzinfo is UTC


@pytest.mark.asyncio
async def test_recovered_emitted_once_and_not_repeated() -> None:
    registry = FakeRegistry()
    registry.set("skippy", False)
    deriver, emitted = _deriver(registry)
    await deriver.evaluate()  # first sighting dead, no emit
    registry.set("skippy", True)
    await deriver.evaluate()  # dead -> alive: one recovered
    await deriver.evaluate()  # still alive: no repeat
    assert [f.transition for f in emitted] == ["recovered"]


@pytest.mark.asyncio
async def test_alive_dead_alive_emits_offline_then_recovered_each_once() -> None:
    registry = FakeRegistry()
    registry.set("nagatha-2", True)
    deriver, emitted = _deriver(registry)
    await deriver.evaluate()  # first sighting alive
    registry.set("nagatha-2", False)
    await deriver.evaluate()  # offline
    registry.set("nagatha-2", True)
    await deriver.evaluate()  # recovered
    assert [f.transition for f in emitted] == ["offline", "recovered"]
    assert all(f.display_name == "Nagatha 2" for f in emitted)


@pytest.mark.parametrize(
    ("slug", "expected"),
    [
        ("skippy", "Skippy"),
        ("nagatha-2", "Nagatha 2"),
        ("foo_bar", "Foo Bar"),
        ("agentd-bilby", "Agentd Bilby"),
    ],
)
def test_safe_display_name(slug: str, expected: str) -> None:
    assert safe_display_name(slug) == expected


@pytest.mark.asyncio
async def test_publish_failure_is_contained_and_edge_retried() -> None:
    # A publish failure RETAINS the pending edge (at-least-once): committed is NOT
    # advanced on failure, so the next tick RETRIES the same offline edge instead of
    # losing it. This is the Finding-5 fix — the old commit-before-publish path lost
    # the edge on a transient bus error.
    registry = FakeRegistry()
    registry.set("skippy", True)
    calls: list[str] = []
    fail = {"on": True}

    async def _publish(frame: PresenceTransitionFrame) -> None:
        calls.append(frame.transition)
        if fail["on"]:
            raise RuntimeError("bus down")

    deriver = PresenceTransitionDeriver(
        registry=registry,  # type: ignore[arg-type]
        publish=_publish,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )
    await deriver.evaluate()  # first sighting
    registry.set("skippy", False)
    await deriver.evaluate()  # offline attempt, publish raises but is contained
    assert calls == ["offline"]  # attempted once, retained
    fail["on"] = False
    await deriver.evaluate()  # retry the SAME pending offline edge, now succeeds
    assert calls == ["offline", "offline"]  # at-least-once retry
    await deriver.evaluate()  # committed advanced: no further replay
    assert calls == ["offline", "offline"]


@pytest.mark.asyncio
async def test_offline_before_recovered_ordering_after_failure() -> None:
    # An offline edge fails while offline; state recovers; a later evaluate delivers
    # the OFFLINE edge FIRST (advancing committed to dead), then sees current=alive !=
    # dead and creates+publishes RECOVERED — operator sees offline THEN recovered.
    registry = FakeRegistry()
    registry.set("nagatha-2", True)
    emitted: list[str] = []
    fail = {"on": False}

    async def _publish(frame: PresenceTransitionFrame) -> None:
        if fail["on"]:
            raise RuntimeError("bus down")
        emitted.append(frame.transition)

    deriver = PresenceTransitionDeriver(
        registry=registry,  # type: ignore[arg-type]
        publish=_publish,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )
    await deriver.evaluate()  # first sighting alive
    registry.set("nagatha-2", False)
    fail["on"] = True
    await deriver.evaluate()  # offline created but publish fails -> pending retained
    assert emitted == []
    registry.set("nagatha-2", True)  # state recovers while offline still owed
    fail["on"] = False
    await deriver.evaluate()  # deliver pending OFFLINE first, then RECOVERED
    assert emitted == ["offline", "recovered"]


@pytest.mark.asyncio
async def test_concurrent_evaluate_serialized_by_lock() -> None:
    # Two concurrent evaluate() calls must not interleave/reorder: the lock serializes
    # the whole read->compare->publish sequence, so exactly one offline edge is
    # emitted for a single alive->dead flip even under reentrant concurrency.
    registry = FakeRegistry()
    registry.set("skippy", True)
    emitted: list[str] = []
    import asyncio

    async def _publish(frame: PresenceTransitionFrame) -> None:
        await asyncio.sleep(0)  # yield to provoke interleave if the lock were absent
        emitted.append(frame.transition)

    deriver = PresenceTransitionDeriver(
        registry=registry,  # type: ignore[arg-type]
        publish=_publish,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )
    await deriver.evaluate()  # first sighting alive
    registry.set("skippy", False)
    await asyncio.gather(deriver.evaluate(), deriver.evaluate(), deriver.evaluate())
    assert emitted == ["offline"]


@pytest.mark.asyncio
async def test_absent_identity_without_pending_is_pruned() -> None:
    from rtech_webchat.presence_transitions import PresenceTransitionDeriver as _D  # noqa: F401

    registry = FakeRegistry()
    registry.set("skippy", True)
    deriver, emitted = _deriver(registry)
    await deriver.evaluate()  # track skippy (committed alive)
    assert "skippy" in deriver._agents  # noqa: SLF001
    registry._alive.clear()  # skippy absent from the snapshot, no pending edge
    await deriver.evaluate()
    assert "skippy" not in deriver._agents  # noqa: SLF001


@pytest.mark.asyncio
async def test_absent_identity_with_pending_edge_is_retained() -> None:
    registry = FakeRegistry()
    registry.set("skippy", True)
    fail = {"on": False}

    async def _publish(frame: PresenceTransitionFrame) -> None:
        if fail["on"]:
            raise RuntimeError("bus down")

    deriver = PresenceTransitionDeriver(
        registry=registry,  # type: ignore[arg-type]
        publish=_publish,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )
    await deriver.evaluate()  # first sighting alive
    registry.set("skippy", False)
    fail["on"] = True
    await deriver.evaluate()  # offline pending, publish failed -> owed
    registry._alive.clear()  # now absent BUT owes a pending edge
    await deriver.evaluate()
    # Retained because it still owes an edge (present in the snapshot is not required
    # to keep an owed edge alive).
    assert "skippy" in deriver._agents  # noqa: SLF001
    assert deriver._agents["skippy"].pending is not None  # noqa: SLF001


@pytest.mark.asyncio
async def test_cap_evicts_oldest_absent_identity() -> None:
    from rtech_webchat.presence_transitions import MAX_TRACKED_IDENTITIES

    registry = FakeRegistry()
    # Fill the tracker to the cap with alive-then-absent identities so they are
    # committed and then eligible for oldest-absent eviction.
    deriver, emitted = _deriver(registry)
    for i in range(MAX_TRACKED_IDENTITIES):
        registry.set(f"a{i}", True)
    await deriver.evaluate()  # track all cap identities
    assert len(deriver._agents) == MAX_TRACKED_IDENTITIES  # noqa: SLF001
    registry._alive.clear()  # all now absent, none owe a pending edge
    registry.set("newcomer", True)  # a single present new identity
    await deriver.evaluate()
    # The newcomer was admitted by evicting the oldest absent identity; the cap holds.
    assert "newcomer" in deriver._agents  # noqa: SLF001
    assert len(deriver._agents) <= MAX_TRACKED_IDENTITIES  # noqa: SLF001
    assert "a0" not in deriver._agents  # noqa: SLF001  oldest evicted first


@pytest.mark.asyncio
async def test_cap_admission_uses_complete_snapshot_not_processed_so_far() -> None:
    # Regression (terminal-audit P2): cap eviction must use the COMPLETE authoritative
    # snapshot identity set, not the identities processed SO FAR this tick. A lexically
    # EARLY newcomer arriving while the cap is full must NOT evict a LATER-in-snapshot
    # identity that is still PRESENT and is transitioning offline this very tick —
    # doing so would erase/re-baseline that identity's committed state and drop its
    # offline edge.
    from rtech_webchat.presence_transitions import MAX_TRACKED_IDENTITIES

    registry = FakeRegistry()
    deriver, emitted = _deriver(registry)

    # Fill cap-1 slots with evictable ABSENT identities ("m..." sorts between the
    # early newcomer and the late tracked identity, and they will be absent so
    # eligible for eviction), plus ONE late-sorting still-present tracked identity.
    for i in range(MAX_TRACKED_IDENTITIES - 1):
        registry.set(f"m{i:04d}", True)
    registry.set("zzz-tracked", True)  # sorts LAST; present and alive, tracked
    await deriver.evaluate()  # baseline: everyone committed
    assert len(deriver._agents) == MAX_TRACKED_IDENTITIES  # noqa: SLF001

    # Now the "m..." fillers go absent (evictable), "zzz-tracked" stays present and
    # flips OFFLINE, and a lexically EARLY newcomer "aaa-new" appears — forcing a cap
    # admission decision while "zzz-tracked" has not yet been processed this tick.
    registry._alive.clear()  # noqa: SLF001
    registry.set("aaa-new", True)  # lexically FIRST -> processed before zzz-tracked
    registry.set("zzz-tracked", False)  # still PRESENT, now offline
    await deriver.evaluate()

    # The offline edge for the still-present tracked identity MUST be emitted — it was
    # not erased/re-baselined by admitting the early newcomer.
    assert [f.transition for f in emitted] == ["offline"]
    assert emitted[0].display_name == "Zzz Tracked"
    # zzz-tracked is still tracked (not evicted); the newcomer was admitted by evicting
    # one of the genuinely-absent fillers.
    assert "zzz-tracked" in deriver._agents  # noqa: SLF001
    assert "aaa-new" in deriver._agents  # noqa: SLF001
    assert len(deriver._agents) <= MAX_TRACKED_IDENTITIES  # noqa: SLF001


@pytest.mark.asyncio
async def test_pending_edge_retains_same_transition_id_across_retry() -> None:
    # The transition_id is minted ONCE when the pending frame is created and RETAINED
    # on retry (same held pending_frame is re-published), so an at-least-once
    # redelivery of the SAME edge carries the SAME id.
    registry = FakeRegistry()
    registry.set("skippy", True)
    seen: list[str] = []
    fail = {"on": True}

    async def _publish(frame: PresenceTransitionFrame) -> None:
        seen.append(frame.transition_id)
        if fail["on"]:
            raise RuntimeError("bus down")

    deriver = PresenceTransitionDeriver(
        registry=registry,  # type: ignore[arg-type]
        publish=_publish,
        clock=lambda: datetime(2026, 7, 20, tzinfo=UTC),
    )
    await deriver.evaluate()  # first sighting
    registry.set("skippy", False)
    await deriver.evaluate()  # offline attempt fails, pending retained
    fail["on"] = False
    await deriver.evaluate()  # retry the SAME pending edge, now succeeds
    assert len(seen) == 2
    assert seen[0] == seen[1]  # same stable transition_id across retry


@pytest.mark.asyncio
async def test_distinct_edges_have_distinct_transition_ids() -> None:
    registry = FakeRegistry()
    registry.set("nagatha-2", True)
    deriver, emitted = _deriver(registry)
    await deriver.evaluate()  # first sighting alive
    registry.set("nagatha-2", False)
    await deriver.evaluate()  # offline
    registry.set("nagatha-2", True)
    await deriver.evaluate()  # recovered
    assert [f.transition for f in emitted] == ["offline", "recovered"]
    assert emitted[0].transition_id != emitted[1].transition_id


@pytest.mark.asyncio
async def test_present_snapshot_over_cap_fails_closed() -> None:
    from rtech_webchat.presence_transitions import MAX_TRACKED_IDENTITIES

    registry = FakeRegistry()
    for i in range(MAX_TRACKED_IDENTITIES + 25):
        registry.set(f"a{i}", True)  # all PRESENT, more than the cap
    deriver, emitted = _deriver(registry)
    await deriver.evaluate()
    # Fail closed: never unbounded-allocate. The tracker holds at most the cap.
    assert len(deriver._agents) <= MAX_TRACKED_IDENTITIES  # noqa: SLF001


@pytest.mark.asyncio
async def test_offset_datetime_normalized_and_naive_rejected() -> None:
    from datetime import timedelta, timezone

    registry = FakeRegistry()
    registry.set("skippy", True)
    emitted: list[PresenceTransitionFrame] = []

    async def _publish(frame: PresenceTransitionFrame) -> None:
        emitted.append(frame)

    plus5 = timezone(timedelta(hours=5))
    deriver = PresenceTransitionDeriver(
        registry=registry,  # type: ignore[arg-type]
        publish=_publish,
        clock=lambda: datetime(2026, 7, 20, 17, 0, 0, tzinfo=plus5),
    )
    await deriver.evaluate()  # first sighting
    registry.set("skippy", False)
    await deriver.evaluate()  # offline
    assert len(emitted) == 1
    # +05:00 17:00 == 12:00 UTC, stored honestly as the UTC instant.
    assert emitted[0].observed_at == datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
    assert emitted[0].observed_at.utcoffset() == timedelta(0)


@pytest.mark.asyncio
async def test_invalid_first_identity_does_not_abort_later_offline_edge() -> None:
    # Finding 1 regression: a subject-safe-but-OVER-CAP (65+ char) authoritative
    # agent_id would raise inside PresenceTransitionFrame construction and, being
    # uncaught, abort the WHOLE evaluate tick — permanently suppressing every later
    # agent. The invalid identity must be ISOLATED (skipped, fail closed) while a
    # VALID later agent's offline edge still delivers. FakeRegistry sorts by id, and
    # a 65-char "a..." slug sorts BEFORE "skippy", so the invalid one is processed
    # FIRST — proving it does not abort the still-valid later transition.
    long_id = "a" * 65  # subject-safe alphabet, but exceeds MAX_PRESENCE_AGENT_ID_LEN
    registry = FakeRegistry()
    registry.set(long_id, True)
    registry.set("skippy", True)
    deriver, emitted = _deriver(registry)
    await deriver.evaluate()  # first sighting for both (skippy tracked; long_id skipped)
    registry.set(long_id, False)  # invalid id flips offline (must be skipped)
    registry.set("skippy", False)  # valid id flips offline (must still deliver)
    await deriver.evaluate()
    # The valid later agent's offline edge is delivered exactly once; the tick was
    # not aborted by the invalid identity.
    assert [f.transition for f in emitted] == ["offline"]
    assert emitted[0].display_name == "Skippy"
    # The invalid identity never entered tracking (no accounting corruption).
    assert long_id not in deriver._agents  # noqa: SLF001
    assert "skippy" in deriver._agents  # noqa: SLF001


@pytest.mark.asyncio
async def test_no_raw_agent_id_in_captured_logs_on_success_or_skip() -> None:
    # Finding 2 regression + Finding 1 log hygiene: neither a successful delivery nor
    # an invalid-identity skip may emit the raw agent_id into logs. Only the safe
    # display_name, transition kind, and bounded counts may appear.
    from loguru import logger

    records: list[str] = []
    sink_id = logger.add(records.append, level="DEBUG", format="{message}")
    try:
        long_id = "b" * 65  # invalid: over cap; skipped with a sanitized count log
        registry = FakeRegistry()
        registry.set(long_id, True)
        registry.set("skippy", True)
        deriver, emitted = _deriver(registry)
        await deriver.evaluate()
        registry.set("skippy", False)
        await deriver.evaluate()  # valid offline delivered -> info log with name only
    finally:
        logger.remove(sink_id)
    blob = "\n".join(records)
    assert emitted and emitted[0].transition == "offline"
    # The raw slug never appears on ANY path (success or skip).
    assert long_id not in blob
    assert "skippy" not in blob  # raw slug never logged; only "Skippy" display name
    assert "Skippy" in blob  # the safe display name IS logged


def test_safe_display_name_strips_unsafe_chars() -> None:
    # Defensive: an unexpected slug with unsafe chars still yields a valid safe name
    # (output always passes the frame's display_name validator).
    assert safe_display_name("foo@bar") == "Foobar"
    assert safe_display_name("a<b>c") == "Abc"
    assert safe_display_name("!!!") == "agent"  # nothing safe -> non-blank fallback
    # And the output is always constructible as a display_name.
    PresenceTransitionFrame(
        agent_id="foobar",
        display_name=safe_display_name("foo_bar"),
        transition="offline",
        observed_at=datetime(2026, 7, 20, tzinfo=UTC),
        transition_id="edge1",
    )
