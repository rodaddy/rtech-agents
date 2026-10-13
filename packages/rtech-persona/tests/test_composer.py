"""Functional I/O tests for the persona composer.

Covers: slot order, optional-slot skipping, timestamp/model line, and a golden
full-composition assertion.
"""

from __future__ import annotations

from datetime import datetime

from rtech_persona.composer import compose
from rtech_persona.loader import Persona

_NOW = datetime(2026, 7, 7, 22, 10, 0)
_MODEL = "claude-sonnet-5"


def _full_persona() -> Persona:
    return Persona(soul="IAM-SOUL", user="IAM-USER", kernel="IAM-KERNEL")


def test_slot_order_all_present() -> None:
    composed = compose(
        _full_persona(),
        memory_block="IAM-MEMORY",
        platform_hint="IAM-PLATFORM",
        now=_NOW,
        model=_MODEL,
    )
    # identity -> kernel -> user -> memory -> platform -> timestamp
    order = [
        "IAM-SOUL",
        "IAM-KERNEL",
        "IAM-USER",
        "IAM-MEMORY",
        "IAM-PLATFORM",
        "Conversation started",
    ]
    positions = [composed.index(token) for token in order]
    assert positions == sorted(positions), composed


def test_absent_optional_slots_are_skipped() -> None:
    persona = Persona(soul="ONLY-SOUL", user=None, kernel=None)
    composed = compose(
        persona,
        memory_block=None,
        platform_hint=None,
        now=_NOW,
        model=_MODEL,
    )
    assert "ONLY-SOUL" in composed
    assert "IAM-USER" not in composed
    assert "IAM-KERNEL" not in composed
    # Only identity + timestamp slots remain.
    assert composed.count("\n\n") == 1


def test_blank_memory_block_is_skipped() -> None:
    composed = compose(
        _full_persona(),
        memory_block="   ",  # whitespace-only -> treated as absent
        platform_hint=None,
        now=_NOW,
        model=_MODEL,
    )
    # Memory slot dropped; kernel is immediately followed by user (no empty gap).
    assert "IAM-KERNEL\n\nIAM-USER" in composed


def test_timestamp_and_model_line() -> None:
    composed = compose(
        Persona(soul="S", user=None, kernel=None),
        memory_block=None,
        platform_hint=None,
        now=_NOW,
        model=_MODEL,
    )
    assert "Conversation started: Tuesday, July 07, 2026" in composed
    assert f"Model: {_MODEL}" in composed


def test_golden_full_composition() -> None:
    composed = compose(
        _full_persona(),
        memory_block="IAM-MEMORY",
        platform_hint="IAM-PLATFORM",
        now=_NOW,
        model=_MODEL,
    )
    expected = (
        "IAM-SOUL\n\n"
        "IAM-KERNEL\n\n"
        "IAM-USER\n\n"
        "IAM-MEMORY\n\n"
        "IAM-PLATFORM\n\n"
        "Conversation started: Tuesday, July 07, 2026\n"
        "Model: claude-sonnet-5"
    )
    assert composed == expected


def test_slots_are_stripped_before_join() -> None:
    persona = Persona(soul="  padded-soul  ", user=None, kernel=None)
    composed = compose(
        persona,
        memory_block=None,
        platform_hint=None,
        now=_NOW,
        model=_MODEL,
    )
    assert composed.startswith("padded-soul\n\n")
