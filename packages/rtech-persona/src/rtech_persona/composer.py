"""Compose the ordered system-prompt slots into a single string.

Slot order (from the governing spec, Phase 1):

    identity (SOUL) -> kernel -> user profile -> memory block ->
    platform hint -> timestamp/model line

Slots are joined with a blank line. Absent optional slots are skipped and
logged. The identity (SOUL) slot is always present because a :class:`Persona`
cannot be constructed without it.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from .loader import Persona

_SLOT_SEPARATOR = "\n\n"


def _timestamp_line(now: datetime, model: str) -> str:
    """Build the trailing date + model line.

    Date-only precision (not minute) keeps the composed prompt byte-stable for
    the whole day, which preserves upstream prefix caches across turns.

    Args:
        now: The reference time.
        model: The engine model identifier.

    Returns:
        A two-line "Conversation started" + "Model" block.
    """
    return f"Conversation started: {now.strftime('%A, %B %d, %Y')}\nModel: {model}"


def compose(
    persona: Persona,
    *,
    memory_block: str | None,
    platform_hint: str | None,
    now: datetime,
    model: str,
) -> str:
    """Compose a persona + runtime slots into one system-prompt string.

    Args:
        persona: The loaded persona (identity + optional kernel/user).
        memory_block: Open Brain recall block, or None when unavailable.
        platform_hint: Platform-specific behavioral hint, or None.
        now: Reference time for the timestamp line.
        model: Engine model identifier for the timestamp line.

    Returns:
        The composed system prompt, slots joined by a blank line.
    """
    logger.debug(
        "persona.compose | model={} memory={} platform_hint={} kernel={} user={}",
        model,
        "present" if memory_block else "absent",
        "present" if platform_hint else "absent",
        "present" if persona.kernel else "absent",
        "present" if persona.user else "absent",
    )

    # Ordered (slot_name, value) pairs; None/blank values are dropped + logged.
    ordered: list[tuple[str, str | None]] = [
        ("identity", persona.soul),
        ("kernel", persona.kernel),
        ("user_profile", persona.user),
        ("memory_block", memory_block),
        ("platform_hint", platform_hint),
        ("timestamp", _timestamp_line(now, model)),
    ]

    parts: list[str] = []
    for slot_name, value in ordered:
        if value is None or not value.strip():
            logger.debug("persona.compose | slot absent, skipping | slot={}", slot_name)
            continue
        parts.append(value.strip())

    composed = _SLOT_SEPARATOR.join(parts)
    logger.info("persona.compose | ok | slots={} len={}", len(parts), len(composed))
    return composed
