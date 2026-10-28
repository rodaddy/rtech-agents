"""The codex stance ``outputSchema`` must satisfy strict-schema rules.

Observed live (2026-07-17, AI Test Group debate smoke): the codex app-server
rejected every worker turn with a 400 ``invalid_json_schema`` because
``required`` did not list ``needs_from`` — strict mode demands ``required``
include EVERY key in ``properties``. Both codex agents failed terminally
before the model ever ran.
"""

from __future__ import annotations

import json

from rtech_agentd.codex_stance import (
    STANCE_OUTPUT_SCHEMA,
    parse_stance_output,
    parse_stance_result,
)


def test_required_lists_every_property_key() -> None:
    # The strict-schema invariant that broke live: required == properties keys.
    assert set(STANCE_OUTPUT_SCHEMA["required"]) == set(STANCE_OUTPUT_SCHEMA["properties"])
    verification = STANCE_OUTPUT_SCHEMA["properties"]["verification"]
    assert set(verification["required"]) == set(verification["properties"])
    source = verification["properties"]["sources"]["items"]
    assert set(source["required"]) == set(source["properties"])


def test_schema_forbids_additional_properties() -> None:
    assert STANCE_OUTPUT_SCHEMA["additionalProperties"] is False


def test_parse_accepts_empty_needs_from() -> None:
    # Strict mode makes needs_from always-present; "no needs" is the empty
    # array and must parse to a clean stance, not the neutral fallback.
    text = json.dumps(
        {
            "position": "dashboard-first",
            "because": "fewer surfaces to secure",
            "needs_from": [],
            "verification": {
                "status": "grounded",
                "claim_kind": "factual",
                "ob_checked": True,
                "sources": [],
                "limitations": [],
                "methods": ["open-brain"],
            },
            "reply": "move it to the dashboard",
        }
    )
    stance, verification, reply = parse_stance_output(text)
    assert stance.position == "dashboard-first"
    assert stance.needs_from == []
    assert verification.status == "grounded"
    assert reply == "move it to the dashboard"


def test_free_text_needs_from_is_blocked_at_the_parse_boundary() -> None:
    # Observed live (2026-07-17): the model put full SENTENCES in needs_from
    # ("Birk should identify..."). This is a malformed STANCE-SHAPED response:
    # the field cannot be coerced (an empty needs_from would invent a false
    # TERMINAL claim ``is_done`` derives from) and the position cannot be trusted
    # as a substantive terminal turn. The parse-validity boundary classifies it
    # ``blocked`` so the engine fails it SAFE as a typed error — no raw JSON ever
    # ships, and the reply is EMPTY (the engine renders the human one-liner).
    raw = json.dumps(
        {
            "position": "Dash should own control",
            "because": "one authoritative home",
            "needs_from": ["Birk should identify unique Discord value."],
            "reply": "My position: Dash should own the operation.",
        }
    )
    result = parse_stance_result(raw)
    assert result.is_blocked is True
    assert result.outcome == "blocked"
    # No schema JSON — and no untrusted position/reply — leaks through.
    assert result.reply == ""
    assert '"position"' not in result.reply
    # Terminal failed stance so a single-scope task cannot hang.
    assert result.stance.position == "failed"
    assert result.stance.needs_from == []
    # The compat tuple shim keeps an empty reply too (never the raw blob).
    stance, _verification, reply = parse_stance_output(raw)
    assert reply == ""
    assert stance.position == "failed"


def test_valid_needs_from_preserves_non_terminal_stance() -> None:
    # A well-formed needs_from (bare agent ids) is trusted verbatim: the stance
    # stays NON-terminal (needs_from populated) so the debate does not falsely
    # derive done, and the model's real position/because carry through.
    raw = json.dumps(
        {
            "position": "Dash should own control",
            "because": "one authoritative home",
            "needs_from": ["agentd-birk", "agentd-sprocket"],
            "reply": "My position: Dash should own the operation.",
        }
    )
    stance, _verification, reply = parse_stance_output(raw)
    assert reply == "My position: Dash should own the operation."
    assert stance.position == "Dash should own control"
    assert stance.needs_from == ["agentd-birk", "agentd-sprocket"]


def test_stance_shaped_blob_that_fails_to_parse_is_blocked() -> None:
    # A truncated/invalid stance object (starts '{', mentions "reply") must not
    # dump schema JSON into chat — it is BLOCKED (typed error), not shipped raw.
    truncated = '{"position":"x","because":"y","needs_from":[],"reply":"hi"'  # no closing brace
    result = parse_stance_result(truncated)
    assert result.is_blocked is True
    assert result.reply == ""
    assert "position" not in result.reply
    assert result.stance.position == "failed"
    # Compat shim still withholds the raw blob.
    _stance, _verification, reply = parse_stance_output(truncated)
    assert reply == ""


def test_stance_object_missing_reply_is_blocked() -> None:
    # Missing the required reply field is still stance-shaped. It must not fall
    # through as ordinary JSON prose and become a normal substantive turn.
    raw = json.dumps(
        {
            "position": "ship",
            "because": "the boundary is sound",
            "needs_from": [],
        }
    )
    result = parse_stance_result(raw)
    assert result.is_blocked is True
    assert result.reply == ""
    assert result.stance.position == "failed"


def test_plain_prose_final_message_still_ships() -> None:
    # The model ignoring the schema and just answering is legitimate: the prose
    # ships verbatim as a valid (non-blocked) turn (only stance-shaped blobs fail).
    result = parse_stance_result("Just answering directly, no JSON here.")
    assert result.outcome == "prose"
    assert result.is_blocked is False
    assert result.reply == "Just answering directly, no JSON here."


def test_parse_malformed_verification_falls_back_all_or_nothing() -> None:
    text = json.dumps(
        {
            "position": "grounded",
            "because": "the schema accepted it",
            "needs_from": [],
            "verification": {
                "status": "grounded",
                "claim_kind": "factual",
                "ob_checked": True,
                "sources": [],
                "limitations": [],
                "methods": [1],
            },
            "reply": "This must not be partially trusted.",
        }
    )

    stance, verification, reply = parse_stance_output(text)

    assert stance.position == "responded"
    assert verification.status == "unverified"
    assert reply == text
