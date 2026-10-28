"""Typed-stance strategy for the codex engine via per-turn ``outputSchema``.

The codex app-server ``turn/start`` (and ``turn/steer``) accepts an
``outputSchema`` — a JSON Schema that constrains the turn's final assistant
message to strict JSON. That is the codex analogue of Claude's ``stance`` MCP
tool: the engine fills a typed stance rather than the ingress scraping it from
prose. The adapter passes :data:`STANCE_OUTPUT_SCHEMA` on every turn, then parses
the final ``agentMessage.text`` back into a :class:`~rtech_frames.Stance` plus the
human-visible ``reply``.

The Protocol invariant is honored by construction: a stance is NEVER regexed out
of turn text. When the JSON is missing or malformed we fall back to the neutral
default and log — we do not guess a position from the words.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger
from pydantic import ValidationError
from rtech_frames import Stance, VerificationReport

from .engine import unverified_verification

# JSON Schema constraining a turn's final message to a strict stance object. The
# ``reply`` field carries the human-visible text; ``position``/``because``/
# ``needs_from`` map straight onto the typed :class:`Stance`; ``verification``
# maps to the bounded :class:`VerificationReport` that rides ``TurnDone``.
#
# Strict-schema rule (enforced server-side by the codex app-server): ``required``
# must list EVERY key in ``properties`` — strict mode has no optional keys, so
# "no needs" is expressed as an empty ``needs_from`` array, never an absent one.
# Omitting a property from ``required`` fails the whole turn with a 400
# ``invalid_json_schema`` before the model runs (observed live, 2026-07-17).
STANCE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["position", "because", "needs_from", "reply", "verification"],
    "properties": {
        "position": {"type": "string", "minLength": 1},
        "because": {"type": "string", "minLength": 1},
        "needs_from": {"type": "array", "items": {"type": "string"}},
        "reply": {"type": "string"},
        "verification": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "status",
                "claim_kind",
                "ob_checked",
                "sources",
                "limitations",
                "methods",
            ],
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["grounded", "partial", "unverified", "not_applicable"],
                },
                "claim_kind": {
                    "type": "string",
                    "enum": ["factual", "mixed", "opinion", "social", "action"],
                },
                "ob_checked": {"type": "boolean"},
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["source_id", "kind", "label", "locator", "authority"],
                        "properties": {
                            "source_id": {"type": "string", "minLength": 1, "maxLength": 64},
                            "kind": {
                                "type": "string",
                                "enum": [
                                    "open_brain",
                                    "repo",
                                    "file",
                                    "url",
                                    "live",
                                    "user_context",
                                ],
                            },
                            "label": {"type": "string", "minLength": 1, "maxLength": 200},
                            "locator": {"type": ["string", "null"], "maxLength": 512},
                            "authority": {
                                "type": "string",
                                "enum": ["live", "source", "memory", "user", "unknown"],
                            },
                        },
                    },
                },
                "limitations": {"type": "array", "items": {"type": "string"}},
                "methods": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}

# Appended to the system prompt so the model knows the final message must be the
# stance JSON (the schema enforces shape; this makes the intent explicit).
STANCE_DIRECTIVE = (
    "\n\nEnd every turn with a single JSON object matching the required output "
    "schema: `position` (your stance), `because` (why), `needs_from` (a list of "
    "AGENT IDS you need input from — bare ids like `agentd-birk`, NOT "
    "sentences; empty array if none), `verification` (the required typed evidence "
    "report), and `reply` (your human-visible message). ONLY `reply` is shown to "
    "people, so put your whole readable answer there; never put prose in the "
    "other fields."
)


# Parse outcome classification at the parser/engine boundary. This is the typed
# control result the finding requires: the engine decides how to publish a turn
# from ``outcome``, never from re-inspecting prose.
#
# * ``stance`` — a clean strict-schema stance object. Publish a normal
#   substantive terminal/non-terminal turn with the parsed stance + reply.
# * ``prose`` — a legitimate plain-text final message (the model ignored the
#   schema and just answered). Ships verbatim as a normal turn with the neutral
#   stance; the plain-prose fallback is preserved.
# * ``blocked`` — a STANCE-SHAPED response (JSON that is, or tries to be, the
#   stance object) that failed to parse or validate: a truncated/invalid blob, a
#   non-object JSON payload, a missing/blank ``position``/``because``, a
#   non-string ``reply``, an unusable ``needs_from``, or a pydantic rejection.
#   This can never be trusted as a substantive answer AND must never leak raw
#   JSON into chat. The engine turns it into a typed engine-error/failure turn
#   (``TurnDone(is_error=True)``) so it fails SAFE through the human-one-liner
#   failure path instead of publishing a false substantive terminal
#   turn or hanging a single-scope task on an invented non-terminal stance.
StanceOutcome = Literal["stance", "prose", "blocked"]


@dataclass(frozen=True)
class StanceParse:
    """The typed parse-validity result at the parser/engine boundary.

    ``outcome`` is the control signal (never re-derived from prose downstream):
    ``stance``/``prose`` are valid replies to publish normally; ``blocked`` is a
    malformed stance-shaped response the engine must fail safely as a typed
    error. ``stance`` carries the parsed (or neutral) stance; ``verification``
    carries the typed evidence report (the deterministic ``unverified`` report
    for the ``prose``/``blocked`` outcomes, where no provider report is trusted);
    ``reply`` is the human-visible text for the valid outcomes and is EMPTY for
    ``blocked`` (no raw JSON ever ships). ``reason`` is a short, JSON-free label
    for logs/the failure line — never the raw payload.
    """

    outcome: StanceOutcome
    stance: Stance
    verification: VerificationReport
    reply: str
    reason: str = ""

    @property
    def is_blocked(self) -> bool:
        """Whether this response was a malformed stance-shaped blob (fail safe)."""
        return self.outcome == "blocked"


def _looks_stance_shaped(stripped: str) -> bool:
    """Whether ``stripped`` is an attempt at the stance object (vs plain prose).

    A JSON-shaped blob that mentions ANY stance-schema field was trying to be the
    schema-constrained object. Requiring ``"reply"`` specifically would let an
    object missing that required field (or truncated before it) leak as ordinary
    prose. Anything without a stance field remains legitimate plain prose/non-
    stance JSON.
    """
    if not stripped.startswith(("{", "[")):
        return False
    return any(f'"{field}"' in stripped for field in STANCE_OUTPUT_SCHEMA["properties"])


def parse_stance_result(text: str) -> StanceParse:
    """Classify a turn's final message into a typed :class:`StanceParse`.

    The text is expected to be the strict JSON produced under
    :data:`STANCE_OUTPUT_SCHEMA`. The outcome is the boundary control signal:

    * A clean object → ``stance`` (parsed stance + reply).
    * Plain prose / non-stance JSON → ``prose`` (neutral stance, text shipped).
    * A stance-SHAPED blob that fails any structural check → ``blocked`` (no
      reply, so no raw JSON leaks; the engine fails it safe as a typed error).

    Args:
        text: The final ``agentMessage`` text for the turn.

    Returns:
        The typed parse-validity result. ``stance`` carries the parsed (or
        neutral) stance and ``verification`` the typed evidence report: a clean
        stance-shaped object with a valid ``verification`` yields both; a
        stance-shaped object whose ``verification`` is present but malformed
        falls back all-or-nothing to neutral prose (no field is partially
        trusted); a missing ``verification`` keeps the parsed stance with the
        deterministic ``unverified`` report.
    """
    stripped = text.strip()
    if not stripped:
        # An empty final message is not a stance-shaped failure — it is a benign
        # no-content turn; ships as neutral prose (the reply is empty text).
        return _prose(text)
    stance_shaped = _looks_stance_shaped(stripped)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        # A blob that STARTS as a stance object yet fails to parse (truncation, a
        # trailing comma) would leak schema JSON: fail it SAFE as a typed error,
        # never dumping the fragment into chat. Non-stance non-JSON is legit prose.
        if stance_shaped:
            logger.warning("codex_stance.parse | stance-shaped blob failed JSON parse; blocked")
            return _blocked("malformed stance JSON")
        logger.debug("codex_stance.parse | non-JSON final message; prose")
        return _prose(text)
    if not isinstance(data, dict):
        if stance_shaped:
            logger.warning("codex_stance.parse | stance-shaped JSON is not an object; blocked")
            return _blocked("stance payload is not an object")
        logger.debug("codex_stance.parse | JSON not an object; prose")
        return _prose(text)
    position = data.get("position")
    because = data.get("because")
    reply = data.get("reply")
    # Strict: any structural violation on a stance-shaped object → ``blocked``
    # (never partial trust, never a substantive terminal turn built from a field
    # we could not read). A parsed object that is NOT stance-shaped (no ``reply``
    # key) is treated as non-stance JSON prose so a legitimate structured answer
    # is never blocked.
    if not stance_shaped:
        logger.debug("codex_stance.parse | object is not stance-shaped; prose")
        return _prose(text)
    if not isinstance(position, str) or not position.strip():
        logger.warning("codex_stance.parse | missing/blank position; blocked")
        return _blocked("missing stance position")
    if not isinstance(because, str) or not because.strip():
        logger.warning("codex_stance.parse | missing/blank because; blocked")
        return _blocked("missing stance rationale")
    if not isinstance(reply, str):
        logger.warning("codex_stance.parse | non-string reply; blocked")
        return _blocked("malformed stance reply")
    # An unparseable ``needs_from`` must NEVER become a substantive terminal turn:
    # the model DID populate the field (it wanted collaboration), and an empty
    # ``needs_from`` is precisely the typed signal for TERMINAL (``is_done``
    # derives done from ``needs_from`` being empty). Coercing malformed input to
    # ``[]`` invents a false terminality claim; keeping the position while
    # dropping the field publishes an unverified substantive terminal turn. Both
    # are unsafe, so the whole stance-shaped-but-malformed response fails SAFE as
    # a typed engine error — the human-readable failure line ships, the raw JSON
    # never does, and a single-scope task does not silently hang.
    needs_from = _validate_needs_from(data.get("needs_from"))
    if needs_from is None:
        logger.warning("codex_stance.parse | unusable needs_from; blocked (fail safe)")
        return _blocked("malformed stance needs_from")
    try:
        stance = Stance(position=position.strip(), because=because.strip(), needs_from=needs_from)
    except Exception as exc:  # noqa: BLE001 - pydantic validation → typed blocked error
        logger.warning("codex_stance.parse | Stance validation failed: {}; blocked", exc)
        return _blocked("stance validation failed")
    # Verification is validated all-or-nothing (schema-v8 typed evidence): a
    # present-but-malformed report is never partially trusted — the whole
    # stance-shaped object falls back to neutral prose so no field survives on
    # its own. An ABSENT verification keeps the parsed stance with the
    # deterministic ``unverified`` report (older stance objects have no
    # verification key; they are still a valid stance, just unverified).
    if "verification" not in data:
        return StanceParse(
            outcome="stance",
            stance=stance,
            verification=unverified_verification(),
            reply=reply,
        )
    try:
        verification = VerificationReport.model_validate(data["verification"], strict=True)
    except ValidationError:
        logger.debug("codex_stance.parse | malformed verification; default outputs")
        return _prose(text)
    return StanceParse(outcome="stance", stance=stance, verification=verification, reply=reply)


def _prose(text: str) -> StanceParse:
    """Build the ``prose`` result: neutral stance, unverified report, text shipped."""
    return StanceParse(
        outcome="prose",
        stance=_default_stance(),
        verification=unverified_verification(),
        reply=text,
    )


def _blocked(reason: str) -> StanceParse:
    """Build the ``blocked`` result: failed stance, no reply, JSON-free reason."""
    return StanceParse(
        outcome="blocked",
        stance=_failed_stance(),
        verification=unverified_verification(),
        reply="",
        reason=reason,
    )


def parse_stance_output(text: str) -> tuple[Stance, VerificationReport, str]:
    """Parse a turn's final message into ``(stance, verification, reply)``.

    Thin wrapper over :func:`parse_stance_result` preserving the historical
    tuple surface for the valid paths: a clean stance or plain prose returns its
    ``(stance, verification, reply)``. A ``blocked`` (malformed stance-shaped)
    response returns the failed stance with an EMPTY reply so no raw JSON ever
    leaks — callers that need to fail such a turn as a typed error use
    :func:`parse_stance_result` and branch on :attr:`StanceParse.is_blocked`.
    """
    result = parse_stance_result(text)
    return result.stance, result.verification, result.reply


def _validate_needs_from(raw: Any) -> list[str] | None:
    """Strictly validate a ``needs_from`` value into a clean agent-id list.

    A list whose every element is already a non-blank string
    with no interior whitespace (a plausible agent id) → that list, stripped.
    Any other shape (non-list, a non-string element, a blank/whitespace-only
    element) → ``None`` to signal the caller to fall back to the neutral stance.
    Never coerces a non-string (e.g. a dict) into ``str`` — that would smuggle
    garbage into a terminal turn's collaboration needs.
    """
    if not isinstance(raw, list):
        return None
    validated: list[str] = []
    for element in raw:
        if not isinstance(element, str):
            return None
        stripped = element.strip()
        if not stripped or stripped != element or any(c.isspace() for c in element):
            return None
        validated.append(stripped)
    return validated


def _default_stance() -> Stance:
    """The neutral stance the adapter fills when the engine states none."""
    return Stance(position="responded", because="engine completed the turn")


def _failed_stance() -> Stance:
    """The terminal FAILED stance a blocked (malformed) stance-shaped turn takes.

    Mirrors the codex engine's terminal-error stance so a blocked response
    renders through the same typed failure-turn path (``is_error=True``): it is
    terminal (empty ``needs_from``) so a single-scope task never hangs, and it
    carries no invented position/target.
    """
    return Stance(position="failed", because="engine produced an unreadable stance")
