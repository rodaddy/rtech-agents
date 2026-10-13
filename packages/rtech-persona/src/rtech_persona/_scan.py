"""Content scan for persona files (injection / control-char guard).

Ports the *intent* of Hermes ``_scan_context_content``: a persona file enters
the system prompt verbatim and the user has no chance to intervene, so
suspicious content is flagged before it is loaded. This is a focused
reimplementation (control characters + prompt-include-style marker headers),
not a copy of the upstream threat-pattern library.

The scan returns a list of finding labels. An empty list means "clean". Callers
decide the disposition (loader rejects on any finding).
"""

from __future__ import annotations

import re
import unicodedata

# Control characters that never legitimately appear in a markdown persona file.
# Tab (\t), newline (\n), carriage return (\r) are explicitly allowed.
_ALLOWED_CONTROL = {"\t", "\n", "\r"}

# Prompt-include-style marker headers: attempts to smuggle a fake system/tool
# boundary or an "include another file / ignore above" directive into the
# identity slot. Case-insensitive, matched anywhere in the content.
_MARKER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "chatml-marker",
        re.compile(r"<\|(?:im_start|im_end|system|user|assistant|endoftext)\|>", re.IGNORECASE),
    ),
    (
        "role-header-injection",
        re.compile(r"^\s*#{0,6}\s*(?:system|assistant|tool)\s*:", re.IGNORECASE | re.MULTILINE),
    ),
    (
        "instruction-override",
        re.compile(
            r"\bignore\s+(?:all\s+)?(?:the\s+)?(?:above|previous|prior)\b"
            r"[^\n]*\b(?:instruction|prompt|rule|directive)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt-include-directive",
        re.compile(
            r"\{\{\s*(?:include|import|inject|load)\b|"
            r"\[\[\s*(?:include|import|inject|load)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "fake-tool-fence",
        re.compile(r"^\s*```(?:tool_code|tool_call|function_call)\b", re.IGNORECASE | re.MULTILINE),
    ),
)


def _find_control_chars(content: str) -> str | None:
    """Return a label if disallowed control characters are present, else None."""
    for ch in content:
        if ch in _ALLOWED_CONTROL:
            continue
        category = unicodedata.category(ch)
        # Cc = control, Cf = format (e.g. zero-width / bidi), Co = private use.
        if category in {"Cc", "Cf", "Co"}:
            return f"control-char:U+{ord(ch):04X}"
    return None


def scan_persona_content(content: str) -> list[str]:
    """Scan persona-file content for injection / control-char signals.

    Args:
        content: Raw file content to inspect.

    Returns:
        A list of finding labels. Empty means the content is clean.
    """
    findings: list[str] = []

    control_finding = _find_control_chars(content)
    if control_finding is not None:
        findings.append(control_finding)

    for label, pattern in _MARKER_PATTERNS:
        if pattern.search(content):
            findings.append(label)

    return findings
