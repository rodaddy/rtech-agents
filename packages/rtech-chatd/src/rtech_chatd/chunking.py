"""Pure message chunking for Discord's 2000-char limit.

Ports the Hermes ``truncate_message`` (gateway/platforms/base.py:4804-4900):

* split at ``max_length`` (2000 for Discord), preferring a newline then a space
  boundary over a hard cut;
* append ``(n/m)`` continuation indicators when the output spans >1 chunk;
* close a triple-backtick code fence left open at a split and reopen it (with
  its language tag) at the start of the next chunk, so each chunk renders on its
  own;
* avoid splitting inside an inline ``code`` span (odd backtick count), which
  would leave an unpaired backtick and mangle rendering.

No discord.py import — the function is pure and fully unit-testable.
"""

from __future__ import annotations

from loguru import logger

# Discord's hard per-message ceiling.
DISCORD_MAX_LENGTH = 2000

# Room reserved for a trailing " (XX/XX)" indicator.
_INDICATOR_RESERVE = 10
# The closing fence appended when a chunk ends mid-code-block.
_FENCE_CLOSE = "\n```"


def _ends_open_fence(chunk_body: str, carry_lang: str | None) -> tuple[bool, str]:
    """Walk ``chunk_body`` to decide if it ends inside an open code fence.

    Args:
        chunk_body: The body text of the chunk (excludes any reopen prefix).
        carry_lang: The language tag carried in from a prior open fence, or
            None when the chunk did not start inside a fence.

    Returns:
        ``(in_code, lang)`` — whether the chunk ends inside a fence, and the
        active language tag to reopen with.
    """
    in_code = carry_lang is not None
    lang = carry_lang or ""
    for line in chunk_body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                in_code = False
                lang = ""
            else:
                in_code = True
                tag = stripped[3:].strip()
                lang = tag.split()[0] if tag else ""
    return in_code, lang


def _split_point(remaining: str, limit: int) -> int:
    """Choose a split offset within ``remaining`` up to ``limit`` chars.

    Prefers the last newline, then the last space; falls back to a hard cut at
    ``limit`` when neither sits past the halfway mark. Then backs off further if
    the cut would fall inside an inline ``code`` span (odd backtick count).
    """
    region = remaining[:limit]
    split_at = region.rfind("\n")
    if split_at < limit // 2:
        split_at = region.rfind(" ")
    if split_at < 1:
        split_at = limit

    # Avoid splitting inside an inline code span (unescaped backticks odd).
    candidate = remaining[:split_at]
    backtick_count = candidate.count("`") - candidate.count("\\`")
    if backtick_count % 2 == 1:
        last_bt = candidate.rfind("`")
        while last_bt > 0 and candidate[last_bt - 1] == "\\":
            last_bt = candidate.rfind("`", 0, last_bt)
        if last_bt > 0:
            safe_split = candidate.rfind(" ", 0, last_bt)
            nl_split = candidate.rfind("\n", 0, last_bt)
            safe_split = max(safe_split, nl_split)
            if safe_split > limit // 4:
                split_at = safe_split
    return split_at


def chunk_message(content: str, max_length: int = DISCORD_MAX_LENGTH) -> list[str]:
    """Split ``content`` into Discord-safe chunks.

    Args:
        content: The full message text.
        max_length: Per-chunk ceiling (default: Discord's 2000).

    Returns:
        A list of chunks, each within ``max_length`` (including any indicator),
        with code fences balanced per chunk. A message that already fits returns
        as a single-element list unchanged (no indicator).

    Raises:
        ValueError: If ``max_length`` is too small to make progress.
    """
    if max_length <= _INDICATOR_RESERVE + len(_FENCE_CLOSE) + 1:
        raise ValueError(f"max_length too small to chunk: {max_length}")

    if len(content) <= max_length:
        return [content]

    logger.debug("chunking.chunk_message | len={} max={}", len(content), max_length)

    chunks: list[str] = []
    remaining = content
    carry_lang: str | None = None

    while remaining:
        prefix = f"```{carry_lang}\n" if carry_lang is not None else ""

        headroom = max_length - _INDICATOR_RESERVE - len(prefix) - len(_FENCE_CLOSE)
        if headroom < 1:
            headroom = max_length // 2

        # Everything left fits in one final chunk (with its reopen prefix).
        if len(prefix) + len(remaining) <= max_length - _INDICATOR_RESERVE:
            chunks.append(prefix + remaining)
            break

        split_at = _split_point(remaining, headroom)
        chunk_body = remaining[:split_at]
        remaining = remaining[split_at:].lstrip()

        full_chunk = prefix + chunk_body
        in_code, lang = _ends_open_fence(chunk_body, carry_lang)
        if in_code:
            full_chunk += _FENCE_CLOSE
            carry_lang = lang
        else:
            carry_lang = None

        chunks.append(full_chunk)

    if len(chunks) > 1:
        total = len(chunks)
        chunks = [f"{chunk} ({i + 1}/{total})" for i, chunk in enumerate(chunks)]

    logger.info("chunking.chunk_message | chunks={}", len(chunks))
    return chunks


__all__ = [
    "DISCORD_MAX_LENGTH",
    "chunk_message",
]
