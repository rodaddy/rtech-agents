"""ChatdConfig validation + token-ref indirection resolution."""

from __future__ import annotations

import pytest
from rtech_chatd.config import ChatdConfig, resolve_token, validate_token_ref


def test_valid_config_builds() -> None:
    cfg = ChatdConfig(
        nats_url="nats://10.0.0.1:4222",
        token_ref="env:RTECH_CHATD_BOT_TOKEN",
        inbox_channel="123",
    )
    assert cfg.origin_edge == "discord"
    assert cfg.inbox_channel == "123"


def test_bad_nats_url_raises() -> None:
    with pytest.raises(ValueError):
        ChatdConfig(nats_url="http://nope")


def test_empty_token_ref_raises() -> None:
    with pytest.raises(ValueError):
        ChatdConfig(token_ref="")


def test_resolve_env_ref_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOMEVAR", "value-here")
    assert resolve_token("env:SOMEVAR") == "value-here"


def test_resolve_env_ref_unset_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOMEVAR", raising=False)
    assert resolve_token("env:SOMEVAR") is None


def test_resolve_unknown_scheme_is_none() -> None:
    assert resolve_token("mystery:x") is None


def test_resolve_empty_ref_is_none() -> None:
    assert resolve_token("") is None


def test_validate_token_ref_accepts_environment_indirection() -> None:
    # The supported scheme returns None (no raise).
    assert validate_token_ref("env:RTECH_CHATD_BOT_TOKEN") is None


@pytest.mark.parametrize(
    "bad",
    ["", "raw-literal-token-value", "env:", "env:NOT-VALID", "vaultwarden:chatd-token"],
)
def test_validate_token_ref_rejects_non_indirection(bad: str) -> None:
    # Empty, a bare literal, and an empty-name scheme all fail closed.
    with pytest.raises(ValueError):
        validate_token_ref(bad)


def test_validate_token_ref_never_echoes_rejected_credential() -> None:
    # CRITICAL: a mistakenly configured literal token must NEVER reach the error
    # (or any log) — the fail-closed path cannot leak the credential to journald.
    with pytest.raises(ValueError) as excinfo:
        validate_token_ref("raw-literal-token-value")
    assert "raw-literal-token-value" not in str(excinfo.value)


def test_validate_token_ref_never_echoes_invalid_environment_ref() -> None:
    invalid_ref = "env:NOT-VALID"
    with pytest.raises(ValueError) as excinfo:
        validate_token_ref(invalid_ref)
    assert invalid_ref not in str(excinfo.value)


def test_config_literal_token_ref_raises_without_echo() -> None:
    with pytest.raises(ValueError) as excinfo:
        ChatdConfig(token_ref="some-literal")
    assert "some-literal" not in str(excinfo.value)


# ---------------------------------------------------------- TG identity map


def test_identity_map_unset_is_empty() -> None:
    from rtech_chatd.config import parse_identity_map

    assert parse_identity_map(None) == {}


def test_identity_map_parses_pairs_and_preserves_refs() -> None:
    from rtech_chatd.config import parse_identity_map

    raw = "agentd-nimbus=env:TG_TOKEN_NIMBUS, agentd-birk=env:TG_TOKEN_BIRK"
    assert parse_identity_map(raw) == {
        "agentd-nimbus": "env:TG_TOKEN_NIMBUS",
        "agentd-birk": "env:TG_TOKEN_BIRK",
    }


def test_identity_map_set_but_empty_fails_closed() -> None:
    from rtech_chatd.config import parse_identity_map

    with pytest.raises(ValueError, match="no usable pair"):
        parse_identity_map(" , ,")


def test_identity_map_rejects_bare_agent_name() -> None:
    from rtech_chatd.config import parse_identity_map

    with pytest.raises(ValueError, match="agentd-<name> convention"):
        parse_identity_map("nimbus=env:TG_TOKEN_NIMBUS")


def test_identity_map_rejects_literal_token_without_echoing() -> None:
    from rtech_chatd.config import parse_identity_map

    with pytest.raises(ValueError) as excinfo:
        parse_identity_map("agentd-nimbus=1234567:AAH-literal-token")
    assert "1234567" not in str(excinfo.value)


def test_identity_map_rejects_duplicate_agent() -> None:
    from rtech_chatd.config import parse_identity_map

    with pytest.raises(ValueError, match="twice"):
        parse_identity_map("agentd-nimbus=env:A,agentd-nimbus=env:B")


def test_identity_map_rejects_malformed_pair() -> None:
    from rtech_chatd.config import parse_identity_map

    with pytest.raises(ValueError, match="malformed"):
        parse_identity_map("agentd-nimbus")


# ------------------------------------------- gauntlet fix regressions


def test_identity_map_malformed_entry_withholds_both_sides() -> None:
    from rtech_chatd.config import parse_identity_map

    # A literal token pasted with no '=' must never land in the raised error;
    # entries are identified by position only.
    with pytest.raises(ValueError) as excinfo:
        parse_identity_map("agentd-nimbus=env:A,1234567:AAH-pasted-literal")
    message = str(excinfo.value)
    assert "entry #2" in message
    assert "1234567" not in message
    assert "AAH-pasted-literal" not in message


def test_identity_map_convention_error_withholds_agent_side() -> None:
    from rtech_chatd.config import parse_identity_map

    # A literal token pasted on the LEFT of '=' fails the convention check
    # without being echoed.
    with pytest.raises(ValueError) as excinfo:
        parse_identity_map("1234567:AAH-pasted-literal=env:X")
    message = str(excinfo.value)
    assert "entry #1" in message
    assert "1234567" not in message


def test_legacy_debate_lane_accepts_positive_integer_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rtech_chatd.config import _env_lane_tuple

    # A forum topic id is always a positive integer; a valid entry parses through.
    monkeypatch.setenv("RTECH_CHATD_DEBATE_LANES", "agentd-nimbus-dev=94")
    assert _env_lane_tuple("RTECH_CHATD_DEBATE_LANES") == (("agentd-nimbus-dev", "94"),)


def test_legacy_debate_lane_skips_non_integer_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rtech_chatd.config import _env_lane_tuple

    # Consistent with the chat-scoped twin: a non-integer topic id can never
    # route, so the entry is skipped and the agent degrades to a per-turn thread.
    monkeypatch.setenv("RTECH_CHATD_DEBATE_LANES", "agentd-nimbus-dev=legacy-94")
    assert _env_lane_tuple("RTECH_CHATD_DEBATE_LANES") == ()


def test_legacy_debate_lane_skips_zero_and_negative_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rtech_chatd.config import _env_lane_tuple

    # Zero and signed topic ids are rejected exactly like the chat-scoped parser
    # (a forum topic id is always >= 1).
    monkeypatch.setenv("RTECH_CHATD_DEBATE_LANES", "agentd-a=0,agentd-b=-5,agentd-c=7")
    assert _env_lane_tuple("RTECH_CHATD_DEBATE_LANES") == (("agentd-c", "7"),)


def test_legacy_debate_lane_valid_entries_survive_among_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rtech_chatd.config import _env_lane_tuple

    # A single bad entry never fails config construction; the valid lanes remain.
    monkeypatch.setenv("RTECH_CHATD_DEBATE_LANES", "agentd-a=12,agentd-b=nope,agentd-c=34")
    assert _env_lane_tuple("RTECH_CHATD_DEBATE_LANES") == (
        ("agentd-a", "12"),
        ("agentd-c", "34"),
    )
