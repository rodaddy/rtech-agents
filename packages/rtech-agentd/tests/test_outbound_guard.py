"""Outbound guard: adversarial marker/header/credential cases."""

from __future__ import annotations

from rtech_agentd.outbound_guard import GuardDecision, sanitize_outbound_text


def _clean(text: object) -> str:
    result = sanitize_outbound_text(text)
    assert isinstance(result, str)
    return result


def test_plain_text_passes_unchanged() -> None:
    text = "Hello, here is your answer. Nothing to strip."
    assert _clean(text) == text


def test_non_string_is_dropped() -> None:
    decision = sanitize_outbound_text({"not": "a string"}, return_decision=True)
    assert isinstance(decision, GuardDecision)
    assert decision.action == "redact"
    assert decision.text == ""


def test_internal_control_token_stripped() -> None:
    out = _clean("Real answer.\n[[internal:trace-id=abc]]")
    assert "[[internal" not in out
    assert "Real answer." in out


def test_done_trailer_is_stripped() -> None:
    # There is no sanctioned trailer anymore: agentd carries no done signal, so
    # a [[done...]] token is stripped like any other internal control token.
    out = _clean("Task complete [[done:completed]]")
    assert "[[done" not in out
    assert "Task complete" in out


def test_internal_header_line_dropped() -> None:
    out = _clean("x-internal: secret-context\nVisible reply line.")
    assert "x-internal" not in out
    assert "Visible reply line." in out


def test_credential_shapes_redacted() -> None:
    cases = [
        "key is sk-ABCDEFGHIJKLMNOP1234567890",
        "token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
        "Authorization: Bearer abcdefghij0123456789KLMNOP",
        "password=hunter2supersecret",
        "slack xoxb-1234567890-abcdefghij",
    ]
    for case in cases:
        out = _clean(case)
        assert "[redacted]" in out, case


def test_bare_provider_keys_redacted() -> None:
    cases = [
        "sk_live_" + "A1b2C3d4E5f6G7h8I9j0K1l2",
        "sk_test_" + "A1b2C3d4E5f6G7h8I9j0K1l2",
        "pk_live_" + "A1b2C3d4E5f6G7h8I9j0K1l2",
        "pk_test_" + "A1b2C3d4E5f6G7h8I9j0K1l2",
        "rk_live_" + "A1b2C3d4E5f6G7h8I9j0K1l2",
        "rk_test_" + "A1b2C3d4E5f6G7h8I9j0K1l2",
        "glpat-" + "ABCDEF1234567890abcd",
        "xoxb-" + "1234567890-abcdefghij",
        "xoxp-" + "1234567890-abcdefghij",
        "xoxa-" + "1234567890-abcdefghij",
        "xoxr-" + "1234567890-abcdefghij",
        "AKIA" + "ABCDEF1234567890",
        "npm_" + "A1b2C3d4E5f6G7h8I9j0K1l2",
        "hf_" + "A1b2C3d4E5f6G7h8I9j0K1l2",
        "sbp_" + "A1b2C3d4E5f6G7h8I9j0K1l2",
    ]
    for key in cases:
        out = _clean(f"credential {key}")
        assert key not in out
        assert "[redacted]" in out


def test_stripe_keys_redact_at_delimiters_without_consuming_them() -> None:
    first = "sk_live_" + "A1b2C3d4E5f6G7h8I9j0K1l2"
    second = "rk_test_" + "Z9y8X7w6V5u4T3s2R1q0P9o8"
    out = _clean(f'keys: ["{first}", "{second}"] next')
    assert out == 'keys: ["[redacted]", "[redacted]"] next'
    assert _clean(f"before {first} next") == "before [redacted] next"


def test_aws_access_key_redacts_before_adjacent_suffixes() -> None:
    # AWS access-key IDs are exactly 20 uppercase-alphanumeric chars: redact the
    # key before punctuation/whitespace/underscore/hyphen suffixes without
    # consuming the delimiter.
    key = "AKIA" + "ABCDEF1234567890"
    for suffix in ("_suffix", "-suffix", ",", ".", " tail"):
        assert _clean(f"credential {key}{suffix}") == f"credential [redacted]{suffix}"


def test_aws_access_key_longer_lookalike_passes_unchanged() -> None:
    # A 21-char (or longer) uninterrupted uppercase-alphanumeric run is not a
    # valid AWS key; the guard must not partially redact it.
    key = "AKIA" + "ABCDEF1234567890"
    for extra in ("X", "XY", "0", "1234"):
        lookalike = f"identifier {key}{extra} end"
        assert _clean(lookalike) == lookalike


def test_benign_url_and_stripe_lookalikes_pass_unchanged() -> None:
    cases = [
        "https://docs.example.com/webhooks/retry-policy",
        "sk_live_documentation",
        "pk_test_reference_20260718",
        "rk_test_reference_20260718",
        "npm_documentation_reference",
    ]
    for case in cases:
        assert _clean(case) == case


def test_provider_key_lookalikes_pass_unchanged() -> None:
    cases = [
        "sk_live_documentation_identifier",
        "sk_live_ABCDEF",
        "build-sk_live_reference_20260718",
        "rk_live_documentation_identifier",
        "rk_live_ABCDEF",
        "build-rk_live_reference_20260718",
    ]
    for lookalike in cases:
        assert _clean(lookalike) == lookalike


def test_homoglyph_control_token_still_stripped() -> None:
    # Cyrillic look-alike letters inside a control token must not evade the strip.
    # 'о' and 'а' below are Cyrillic homoglyphs.
    spoofed = "answer [[іnternаl:x]]"
    out = _clean(spoofed)
    assert "[[" not in out


def test_jwt_redacted() -> None:
    jwt = "eyJhbGciOiJI.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF2QT4"
    out = _clean(f"here: {jwt}")
    assert jwt not in out
    assert "[redacted]" in out
