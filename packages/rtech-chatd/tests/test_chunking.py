"""Functional I/O tests for the pure chunking module.

Covers: fit-in-one, exact-2000 boundary, multi-chunk indicators, code-fence
spanning a split (close + reopen with language), inline-code avoidance, and
unicode payloads.
"""

from __future__ import annotations

import pytest
from rtech_chatd.chunking import DISCORD_MAX_LENGTH, chunk_message


def test_short_message_single_chunk_unchanged() -> None:
    assert chunk_message("hello world") == ["hello world"]


def test_exact_max_length_single_chunk() -> None:
    body = "a" * DISCORD_MAX_LENGTH
    result = chunk_message(body)
    assert result == [body]
    assert len(result) == 1


def test_one_over_max_splits_into_two() -> None:
    body = "a" * (DISCORD_MAX_LENGTH + 1)
    result = chunk_message(body)
    assert len(result) == 2
    # Each chunk (incl. indicator) stays within the ceiling.
    for chunk in result:
        assert len(chunk) <= DISCORD_MAX_LENGTH


def test_multi_chunk_has_indicators() -> None:
    body = ("word " * 1500).strip()  # ~7500 chars
    result = chunk_message(body)
    assert len(result) >= 2
    total = len(result)
    for i, chunk in enumerate(result):
        assert chunk.endswith(f"({i + 1}/{total})")


def test_prefers_newline_split() -> None:
    # A clean newline near the boundary should be the split point.
    line = "a" * 1900
    body = line + "\n" + ("b" * 400)
    result = chunk_message(body)
    assert len(result) == 2
    # First chunk carries the a-line; the b-run starts the second chunk.
    assert "a" * 100 in result[0]
    assert result[1].startswith("b")


def test_code_fence_spanning_split_is_closed_and_reopened() -> None:
    # A python fence with a body long enough to force a split mid-block.
    inner = "\n".join(f"line_{i} = {i}" for i in range(400))
    body = f"```python\n{inner}\n```"
    result = chunk_message(body)
    assert len(result) >= 2
    # First chunk closes its orphaned fence (before the trailing " (n/m)").
    first_core = result[0].rsplit(" (", 1)[0]
    assert first_core.rstrip().endswith("```")
    # Second chunk reopens with the language tag.
    assert result[1].startswith("```python\n")
    # Fence balance per chunk: even count of ``` in each (indicator stripped).
    for chunk in result:
        core = chunk.rsplit(" (", 1)[0]
        assert core.count("```") % 2 == 0


def test_plain_language_fence_reopens_without_tag() -> None:
    inner = "x" * 4000
    body = f"```\n{inner}\n```"
    result = chunk_message(body)
    assert len(result) >= 2
    # Reopened with a bare fence (no language).
    assert result[1].startswith("```\n")


def test_inline_code_not_split_mid_span() -> None:
    # Build content where a naive split would land inside an inline `code` span.
    prefix = "a" * 1990
    body = prefix + " `important_token_value_here` trailing text " * 30
    result = chunk_message(body)
    for chunk in result:
        # Strip the trailing indicator before counting backticks.
        core = chunk.rsplit(" (", 1)[0]
        assert core.count("`") % 2 == 0, core[-80:]


def test_unicode_payload_chunks_without_crash() -> None:
    body = ("héllo wörld 日本語 🚀 " * 300).strip()
    result = chunk_message(body)
    assert len(result) >= 2
    # Rejoining the bodies (minus indicators) recovers all the source glyphs.
    assert "日本語" in "".join(result)
    assert "🚀" in "".join(result)


def test_max_length_too_small_raises() -> None:
    with pytest.raises(ValueError):
        chunk_message("some content that is long enough to need chunking here", max_length=5)


def test_custom_small_max_length_works() -> None:
    body = "one two three four five six seven eight nine ten"
    result = chunk_message(body, max_length=20)
    assert len(result) >= 2
    for chunk in result:
        assert len(chunk) <= 20
