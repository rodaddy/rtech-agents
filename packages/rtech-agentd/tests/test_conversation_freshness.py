"""Verify bounded local conversation freshness classification."""

from __future__ import annotations

import pytest
from rtech_agentd.conversation_freshness import (
    DEFAULT_IDLE_THRESHOLD_S,
    ConversationFreshnessConfig,
    ConversationFreshnessTracker,
    FreshnessClassification,
)
from rtech_frames import ChannelOrigin


class _Clock:
    """Provide a manually advanced monotonic test clock."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _origin(channel: str, *, thread: str | None = None) -> ChannelOrigin:
    return ChannelOrigin(
        guild_id="guild",
        channel_id=channel,
        thread_id=thread,
    )


def test_first_contact_then_warm_followup() -> None:
    clock = _Clock()
    tracker = ConversationFreshnessTracker(clock=clock)
    origin = _origin("channel")

    assert (
        tracker.classify(origin_edge="telegram", origin=origin)
        is FreshnessClassification.FIRST_CONTACT
    )
    clock.now = DEFAULT_IDLE_THRESHOLD_S - 1
    assert (
        tracker.classify(origin_edge="telegram", origin=origin)
        is FreshnessClassification.WARM_FOLLOWUP
    )


def test_idle_threshold_is_cold_and_then_resets_activity() -> None:
    clock = _Clock()
    tracker = ConversationFreshnessTracker(clock=clock)
    origin = _origin("channel")
    tracker.classify(origin_edge="telegram", origin=origin)

    clock.now = DEFAULT_IDLE_THRESHOLD_S
    assert (
        tracker.classify(origin_edge="telegram", origin=origin)
        is FreshnessClassification.COLD_AFTER_IDLE
    )
    clock.now += 1
    assert (
        tracker.classify(origin_edge="telegram", origin=origin)
        is FreshnessClassification.WARM_FOLLOWUP
    )


def test_edge_guild_channel_and_thread_are_distinct_keys() -> None:
    tracker = ConversationFreshnessTracker(clock=_Clock())

    assert (
        tracker.classify(origin_edge="telegram", origin=_origin("c", thread="a")) == "first_contact"
    )
    assert (
        tracker.classify(origin_edge="telegram", origin=_origin("c", thread="b")) == "first_contact"
    )
    assert (
        tracker.classify(origin_edge="discord", origin=_origin("c", thread="a")) == "first_contact"
    )


def test_originless_tasks_are_always_cold_and_not_retained() -> None:
    tracker = ConversationFreshnessTracker(clock=_Clock())

    assert tracker.classify(origin_edge="web", origin=None) == "first_contact"
    assert tracker.classify(origin_edge="web", origin=None) == "first_contact"
    assert tracker.tracked_count == 0


def test_lru_eviction_is_bounded_and_refreshes_recency() -> None:
    clock = _Clock()
    tracker = ConversationFreshnessTracker(
        ConversationFreshnessConfig(max_conversations=2),
        clock=clock,
    )
    first = _origin("first")
    second = _origin("second")
    third = _origin("third")

    tracker.classify(origin_edge="web", origin=first)
    tracker.classify(origin_edge="web", origin=second)
    tracker.classify(origin_edge="web", origin=first)
    tracker.classify(origin_edge="web", origin=third)

    assert tracker.tracked_count == 2
    assert tracker.classify(origin_edge="web", origin=second) == "first_contact"


@pytest.mark.parametrize(
    ("idle_threshold_s", "max_conversations"),
    [(0.0, 1), (-1.0, 1), (1.0, 0)],
)
def test_invalid_config_rejected(idle_threshold_s: float, max_conversations: int) -> None:
    with pytest.raises(ValueError):
        ConversationFreshnessConfig(
            idle_threshold_s=idle_threshold_s,
            max_conversations=max_conversations,
        )
