"""Outbound leak guard — sanitize a fully-buffered reply before it is published.

Ported from ``rtech_hermes_runtime.outbound_guard`` (marker detection with a
unicode-normalization defense) and reduced to what a NATS-native agent needs:

1. **Buffer-complete-then-sanitize.** The reply path hands a WHOLE turn's text
   here before any publish — streaming deltas never bypass this. The guard is
   unconditional on the reply path.
2. **Internal marker / header-line stripping.** Lines that are internal engine
   scaffolding (``[[marker]]`` control tokens, ``x-internal:``-style headers)
   are removed so they never reach a chat surface or a coordination lane.
3. **Credential-shape redaction.** Obvious credential-shaped strings (bearer
   tokens, ``sk-``/``ghp_`` keys, long hex/base64 secrets, ``password=...``) are
   replaced with ``[redacted]`` — a defense in depth against a model echoing a
   secret it saw.

Detection normalizes for homoglyph/zero-width evasion (an attacker echoing a
control token via Cyrillic look-alikes must not slip a marker past the strip).
Output preserves the original casing/formatting of the surviving text.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from loguru import logger

GuardAction = Literal["pass", "strip", "redact"]

# --- Unicode-normalization defense (ported intent from _detection_utils) ------

_ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿­͏؜᠎‎‏‪-‮⁦-⁩￹-￻ㅤﾠ ︀-️̀-ͯ]")
_SUPPLEMENTARY_INVISIBLE_RE = re.compile("[\U000e0001-\U000e007f\U000e0100-\U000e01ef]")

# Minimal Cyrillic/Greek → Latin homoglyph fold, enough to defeat control-token
# spoofing on the small marker vocabulary this guard recognises.
_HOMOGLYPH_TO_LATIN = {
    "А": "A",
    "а": "a",
    "В": "B",
    "Е": "E",
    "е": "e",
    "К": "K",
    "М": "M",
    "О": "O",
    "о": "o",
    "Р": "P",
    "р": "p",
    "С": "C",
    "с": "c",
    "Т": "T",
    "Х": "X",
    "х": "x",
    "Ѕ": "S",
    "і": "i",
    "ԁ": "d",
    "ո": "n",
    "Α": "A",
    "α": "a",
    "Β": "B",
    "Ε": "E",
    "Ο": "O",
    "ο": "o",
    "Ρ": "P",
    "Τ": "T",
    "Χ": "X",
    "Ι": "I",
    "Κ": "K",
    "Μ": "M",
    "Ν": "N",
}
_HOMOGLYPH_TABLE = str.maketrans(_HOMOGLYPH_TO_LATIN)


def _normalize_for_detection(text: str) -> str:
    """Fold zero-width/homoglyph evasion for detection only (never for output)."""
    text = unicodedata.normalize("NFD", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _SUPPLEMENTARY_INVISIBLE_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_HOMOGLYPH_TABLE)
    return text.lower()


# --- Internal marker / header patterns to strip whole lines -------------------

# ALL `[[...]]` control tokens are internal scaffolding and never reach a chat
# surface or coordination lane. There is no sanctioned trailer: agentd carries no
# done signal (done is orchestrator-derived from the typed frame graph).
_CONTROL_TOKEN_RE = re.compile(r"\[\[[^\]]*\]\]")

# Internal header lines the engine/scaffold may emit (never for a human).
_INTERNAL_HEADER_RE = re.compile(
    r"^\s*(?:x-internal|x-agent|x-trace|system-note|internal)\s*:",
    re.IGNORECASE,
)

# --- Credential-shape redaction -----------------------------------------------

_CRED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{16,}\b"),  # OpenAI-style keys
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),  # GitHub tokens
    re.compile(
        r"(?<![A-Za-z0-9_-])sk_(?:live|test)_[A-Za-z0-9]{24,}(?![A-Za-z0-9_-])"
    ),  # Stripe secret keys
    re.compile(
        r"(?<![A-Za-z0-9_-])pk_(?:live|test)_[A-Za-z0-9]{24,}(?![A-Za-z0-9_-])"
    ),  # Stripe publishable keys
    re.compile(
        r"(?<![A-Za-z0-9_-])rk_(?:live|test)_[A-Za-z0-9]{24,}(?![A-Za-z0-9_-])"
    ),  # Stripe restricted keys
    re.compile(r"(?<![A-Za-z0-9_-])glpat-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"),
    re.compile(r"(?<![A-Za-z0-9_-])xox[baprs]-[A-Za-z0-9-]{10,}(?![A-Za-z0-9_-])"),
    # AWS access-key IDs are exactly 20 uppercase-alphanumeric chars. Redact
    # before punctuation/whitespace/underscore/hyphen suffixes, but a longer
    # uninterrupted uppercase-alphanumeric run is not a valid key — leave it.
    re.compile(r"(?<![A-Za-z0-9_-])AKIA[A-Z0-9]{16}(?![A-Z0-9])"),
    re.compile(r"(?<![A-Za-z0-9_-])npm_[A-Za-z0-9]{24,}(?![A-Za-z0-9_-])"),
    re.compile(r"(?<![A-Za-z0-9_-])hf_[A-Za-z0-9]{24,}(?![A-Za-z0-9_-])"),
    re.compile(r"(?<![A-Za-z0-9_-])sbp_[A-Za-z0-9]{24,}(?![A-Za-z0-9_-])"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE),
    re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{4,}\b"),  # JWT
    re.compile(r"(?i)\b(?:password|passwd|secret|token|api[_-]?key)\s*[=:]\s*\S{6,}"),
)
_REDACTED = "[redacted]"


@dataclass(frozen=True)
class GuardDecision:
    """The guard's verdict for a piece of outbound text."""

    text: str
    action: GuardAction
    platform: str | None = None


def _strip_control_tokens(text: str) -> tuple[str, bool]:
    """Remove ALL ``[[...]]`` control tokens. Returns (text, changed)."""
    normalized = _normalize_for_detection(text)
    if "[[" not in normalized:
        return text, False
    stripped = _CONTROL_TOKEN_RE.sub("", text)
    return stripped, stripped != text


def _drop_internal_header_lines(text: str) -> tuple[str, bool]:
    """Drop whole lines that are internal engine/scaffold headers."""
    kept: list[str] = []
    changed = False
    for line in text.splitlines(keepends=True):
        if _INTERNAL_HEADER_RE.match(line):
            changed = True
            continue
        kept.append(line)
    return "".join(kept), changed


def _redact_credentials(text: str) -> tuple[str, bool]:
    """Replace credential-shaped substrings with ``[redacted]``."""
    changed = False
    out = text
    for pat in _CRED_PATTERNS:
        out, n = pat.subn(_REDACTED, out)
        if n:
            changed = True
    return out, changed


def sanitize_outbound_text(
    text: object,
    *,
    platform: str | None = None,
    return_decision: bool = False,
) -> str | GuardDecision:
    """Sanitize a fully-buffered reply before it leaves the agent.

    Order: drop internal header lines → strip all ``[[...]]`` control tokens →
    redact credential-shaped strings → collapse the whitespace the removals may
    leave. A non-string payload is dropped to an empty string (never published
    raw).

    Args:
        text: The buffered reply text (any non-string is dropped).
        platform: Optional target surface name, for logging only.
        return_decision: When True, return the :class:`GuardDecision`; else the
            sanitized string.

    Returns:
        The sanitized text, or the full :class:`GuardDecision`.
    """
    if not isinstance(text, str):
        logger.warning(
            "outbound_guard drop | non-string payload type={} platform={}",
            type(text).__name__,
            platform,
        )
        decision = GuardDecision(text="", action="redact", platform=platform)
        return decision if return_decision else decision.text

    working, header_changed = _drop_internal_header_lines(text)
    working, token_changed = _strip_control_tokens(working)
    working, cred_changed = _redact_credentials(working)

    changed = header_changed or token_changed or cred_changed
    if changed:
        # Collapse runs of blank lines the strips may have created; trim edges.
        working = re.sub(r"\n{3,}", "\n\n", working).strip()
        action: GuardAction = "redact" if cred_changed else "strip"
        logger.warning(
            "outbound_guard {} | header={} token={} cred={} platform={} in_len={} out_len={}",
            action,
            header_changed,
            token_changed,
            cred_changed,
            platform,
            len(text),
            len(working),
        )
    else:
        action = "pass"
        logger.debug("outbound_guard pass | platform={} len={}", platform, len(text))

    decision = GuardDecision(text=working, action=action, platform=platform)
    return decision if return_decision else decision.text
