"""The ``EngineSession`` protocol — the managed-session seam agentd owns.

An agent is ONE persistent, inline-steerable engine session. The rest of agentd
(supervisor, NATS ingress, outbound guard) talks to that session only through
this protocol, so the concrete engine (Claude Agent SDK today, Codex later) is a
swappable adapter, never wired into coordination code.

Event model: a turn is a stream of typed :class:`EngineEvent`s ending in a
:class:`TurnDone` (or an :class:`EngineError`). ``events()`` is a single
long-lived async iterator over the session's whole life; turn boundaries are
marked by :class:`TurnDone`, not by the iterator ending. ``send()`` while a turn
is active is inline steering; ``interrupt()`` aborts the active turn.

Everything here is pure typing + dataclasses — no engine import — so it stays
trivially testable and lets a ``FakeEngine`` satisfy the same contract.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from rtech_frames import Stance, VerificationReport


def _default_stance() -> Stance:
    """Return the neutral stance the engine fills when it states none itself.

    ``TurnFrame.stance`` is a required typed field, so every completed turn must
    carry one. The engine sets this explicitly — it is NEVER scraped from turn
    text — and overrides it when the concrete engine surfaces a real position.
    """

    return Stance(position="responded", because="engine completed the turn")


def unverified_verification() -> VerificationReport:
    """Return the deterministic report used when an engine provides none.

    Verification is a required engine-boundary field even when a provider cannot
    produce a well-formed structural report.  This default makes that absence
    explicit rather than allowing a missing value to look like grounded output.
    """
    return VerificationReport(
        status="unverified",
        claim_kind="mixed",
        ob_checked=False,
        limitations=["verification was not reported"],
    )


@dataclass(frozen=True)
class ToolsPolicy:
    """Allow/deny tool lists handed to the engine at session start.

    Attributes:
        allow: Tool names the engine may use. Empty means "no explicit allow
            list" (engine default), NOT "deny everything".
        deny: Tool names the engine must never use. Takes precedence over allow.
    """

    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()


# --- Typed event union -------------------------------------------------------

EventKind = Literal["text_delta", "tool_use", "turn_done", "error"]


@dataclass(frozen=True)
class TextDelta:
    """A chunk of assistant-visible text produced during a turn."""

    text: str
    kind: EventKind = "text_delta"


@dataclass(frozen=True)
class ToolUse:
    """The engine invoked a tool during a turn (observability, not execution)."""

    tool_id: str
    name: str
    tool_input: dict[str, Any] = field(default_factory=dict)
    kind: EventKind = "tool_use"


@dataclass(frozen=True)
class TurnDone:
    """A turn reached its terminal boundary.

    Attributes:
        text: The full assistant text for the turn, concatenated in order. The
            reply path uses THIS (buffered-complete) — never the deltas — so the
            outbound guard sees a whole message.
        session_ref: The engine session id captured for resume, if known.
        is_error: Whether the engine reported the turn itself as an error result.
        stance: The typed stance the engine states for this turn. It rides
            straight onto the outbound ``TurnFrame.stance`` — the ingress never
            scrapes a stance from ``text``. Defaults to a neutral stance the
            engine fills when it surfaces none of its own.
        verification: The typed evidence report for this turn. Malformed or
            missing provider output gets the deterministic ``unverified``
            report; verification is never inferred from assistant prose.
    """

    text: str
    session_ref: str | None = None
    is_error: bool = False
    stance: Stance = field(default_factory=_default_stance)
    verification: VerificationReport = field(default_factory=unverified_verification)
    kind: EventKind = "turn_done"


@dataclass(frozen=True)
class EngineError:
    """A failure surfaced from the engine (never swallowed silently)."""

    message: str
    kind: EventKind = "error"


# A chat-safe failure detail is ALLOWLISTED, not blocklisted: only short runs
# of plain ASCII words/punctuation. No `/` or `.` (kills URLs, domains, and
# paths — scheme-less included), no `=`/`&`/`%`/`@`/`#` (query strings,
# encodings, handles), and by charset omission no newlines, tabs, NUL, ANSI
# escapes, or Unicode line/bidi controls. Anything outside this shape is
# payload, not a sentence, and stays in logs.
_SAFE_FAILURE_DETAIL_RE = re.compile(r"[A-Za-z0-9 _'(),:;!?-]{1,200}")

# Secrets that fit the charset (e.g. a bare provider API key) are long unbroken
# runs; no vendor error code or human word is this long. The outbound redaction
# guard at the publish sites is the second net, but it only knows prefixed
# credential shapes, so this cap is load-bearing.
_MAX_DETAIL_TOKEN_LEN = 25


def _safe_failure_detail(value: str) -> bool:
    """Whether a candidate detail is a short plain sentence, not payload.

    Applied to EVERY detail candidate — JSON-extracted ``type``/``code`` values
    included, since a vendor payload can smuggle payload material inside those
    fields just as easily as in a plain message.
    """

    return _SAFE_FAILURE_DETAIL_RE.fullmatch(value) is not None and all(
        len(token) <= _MAX_DETAIL_TOKEN_LEN for token in value.split()
    )


def failure_visible_text(raw: str) -> str:
    """Return the one human-shaped line a failed turn shows in chat.

    An engine terminal error often carries the vendor's RAW error body (JSON
    with URLs, request ids, config fragments). Chat surfaces are treated as
    leaked, so a failed turn's ``visible_text`` is always a summary built here;
    the raw payload belongs in logs only (the publish sites log it once).

    Best effort, never raising: a JSON payload contributes its error ``type``/
    ``code`` and HTTP ``status`` when present; a detail (JSON-extracted or the
    whole plain message) reaches chat only if it fits the allowlisted
    plain-sentence shape (``_SAFE_FAILURE_DETAIL_RE``); anything else stays
    generic.
    """
    text = raw.strip()
    detail: str | None = None
    status: int | None = None
    if text.startswith("{"):
        try:
            payload: Any = json.loads(text)
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            error = payload.get("error")
            error = error if isinstance(error, dict) else payload
            for key in ("type", "code"):
                value = error.get(key)
                if isinstance(value, str) and value != "error" and _safe_failure_detail(value):
                    detail = value
                    break
            for source, key in ((payload, "status"), (payload, "statusCode"), (error, "status")):
                value = source.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    status = value
                    break
    elif _safe_failure_detail(text):
        detail = text
    line = "engine error" if detail is None else f"engine error: {detail}"
    if status is not None:
        line += f" (status {status})"
    return line + " — details in agentd logs"


EngineEvent = TextDelta | ToolUse | TurnDone | EngineError


@dataclass(frozen=True)
class EngineHealth:
    """A point-in-time health snapshot of the managed session."""

    connected: bool
    session_ref: str | None
    active_turn: bool
    detail: str = ""


@runtime_checkable
class EngineSession(Protocol):
    """One managed, persistent, inline-steerable engine session.

    Lifecycle: :meth:`start` (or :meth:`resume`) → interleaved :meth:`send` /
    :meth:`interrupt` while consuming :meth:`events` → :meth:`close`.
    """

    async def start(self, system_prompt: str, tools_policy: ToolsPolicy) -> None:
        """Open a fresh session with the composed system prompt + tools policy."""
        ...

    async def resume(self, session_ref: str) -> None:
        """Reopen a prior session by its captured ``session_ref``."""
        ...

    async def send(self, text: str) -> None:
        """Send a user turn, OR steer inline if a turn is already active."""
        ...

    async def interrupt(self) -> None:
        """Abort the active turn (no-op if none is active)."""
        ...

    def events(self) -> AsyncIterator[EngineEvent]:
        """The session's long-lived typed event stream (turns end in TurnDone)."""
        ...

    async def health(self) -> EngineHealth:
        """A current health snapshot."""
        ...

    async def close(self) -> None:
        """Tear the session down. Safe to call when not started."""
        ...
