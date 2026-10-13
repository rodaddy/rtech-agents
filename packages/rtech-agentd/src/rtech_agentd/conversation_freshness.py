"""Classify local conversation freshness from typed channel origins."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from loguru import logger
from rtech_frames import ChannelOrigin

DEFAULT_IDLE_THRESHOLD_S = 15 * 60.0
DEFAULT_MAX_CONVERSATIONS = 1024


class FreshnessClassification(StrEnum):
    """Name the recency state observed at the start of a task turn."""

    FIRST_CONTACT = "first_contact"
    COLD_AFTER_IDLE = "cold_after_idle"
    WARM_FOLLOWUP = "warm_followup"


@dataclass(frozen=True)
class ConversationFreshnessConfig:
    """Configure the local bounded freshness tracker."""

    idle_threshold_s: float = DEFAULT_IDLE_THRESHOLD_S
    max_conversations: int = DEFAULT_MAX_CONVERSATIONS

    def __post_init__(self) -> None:
        if self.idle_threshold_s <= 0:
            raise ValueError("idle_threshold_s must be greater than zero")
        if self.max_conversations < 1:
            raise ValueError("max_conversations must be at least one")


@dataclass(frozen=True)
class ConversationKey:
    """Identify one channel or thread without message- or member-level data."""

    origin_edge: str
    guild_id: str | None
    channel_id: str
    thread_id: str | None


class ConversationFreshnessTracker:
    """Track bounded per-origin activity with monotonic timestamps."""

    def __init__(
        self,
        config: ConversationFreshnessConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Build an empty tracker.

        Args:
            config: Idle threshold and LRU capacity.
            clock: Injected monotonic clock for deterministic classification.
        """
        self._config = config or ConversationFreshnessConfig()
        self._clock = clock
        self._last_activity: OrderedDict[ConversationKey, float] = OrderedDict()

    @property
    def tracked_count(self) -> int:
        """Return the current bounded LRU entry count."""
        return len(self._last_activity)

    def classify(
        self,
        *,
        origin_edge: str,
        origin: ChannelOrigin | None,
    ) -> FreshnessClassification:
        """Record activity and return the origin's conversation freshness.

        Origin-less tasks are always cold and are intentionally not retained:
        without a typed conversational identity, a later task cannot prove it
        is a follow-up.

        Args:
            origin_edge: Typed ingress edge carried by the task frame.
            origin: Typed chat origin, or ``None`` for non-conversational tasks.

        Returns:
            ``first_contact``, ``cold_after_idle``, or ``warm_followup``.
        """
        if origin is None:
            logger.debug("conversation.freshness | first_contact | origin=absent")
            return FreshnessClassification.FIRST_CONTACT

        key = ConversationKey(
            origin_edge=origin_edge,
            guild_id=origin.guild_id,
            channel_id=origin.channel_id,
            thread_id=origin.thread_id,
        )
        now = self._clock()
        previous = self._last_activity.pop(key, None)
        self._last_activity[key] = now
        while len(self._last_activity) > self._config.max_conversations:
            self._last_activity.popitem(last=False)

        if previous is None:
            classification = FreshnessClassification.FIRST_CONTACT
        elif now - previous >= self._config.idle_threshold_s:
            classification = FreshnessClassification.COLD_AFTER_IDLE
        else:
            classification = FreshnessClassification.WARM_FOLLOWUP
        logger.debug(
            "conversation.freshness | {} | tracked={}",
            classification.value,
            len(self._last_activity),
        )
        return classification


__all__ = [
    "DEFAULT_IDLE_THRESHOLD_S",
    "DEFAULT_MAX_CONVERSATIONS",
    "ConversationFreshnessConfig",
    "ConversationFreshnessTracker",
    "ConversationKey",
    "FreshnessClassification",
]
