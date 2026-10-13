"""IntakeConfig env plumbing (#93) + the triage-agent subject builder."""

from __future__ import annotations

import pytest
from rtech_chatd import subjects
from rtech_chatd.config import IntakeConfig, IntakeMode


def test_from_env_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "RTECH_CHATD_INTAKE_ENABLED",
        "RTECH_CHATD_INTAKE_MODE",
        "RTECH_CHATD_TRIAGE_AGENT",
        "RTECH_CHATD_INTAKE_MEMBER_IDS",
        "RTECH_CHATD_INTAKE_NAME_TRIGGERS",
        "RTECH_CHATD_INTAKE_TRIGGER_PREFIXES",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = IntakeConfig.from_env()
    assert cfg.enabled is False
    assert cfg.mode is IntakeMode.LISTED
    # No bare-name default: a deployed agentd subscribes as agentd-<name>, so a
    # default here would publish into the void. Disabled config leaves it blank.
    assert cfg.triage_agent == ""
    assert cfg.member_ids == frozenset()


def test_from_env_parses_all_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RTECH_CHATD_INTAKE_ENABLED", "true")
    monkeypatch.setenv("RTECH_CHATD_INTAKE_MODE", "all-channels")
    monkeypatch.setenv("RTECH_CHATD_TRIAGE_AGENT", "agentd-bilby")
    monkeypatch.setenv("RTECH_CHATD_INTAKE_MEMBER_IDS", "77, 88 ,99")
    monkeypatch.setenv("RTECH_CHATD_INTAKE_NAME_TRIGGERS", "nagatha,bilby")
    monkeypatch.setenv("RTECH_CHATD_INTAKE_TRIGGER_PREFIXES", "!ask,!triage")
    cfg = IntakeConfig.from_env()
    assert cfg.enabled is True
    assert cfg.mode is IntakeMode.ALL_CHANNELS
    assert cfg.triage_agent == "agentd-bilby"
    assert cfg.member_ids == frozenset({"77", "88", "99"})
    assert cfg.name_triggers == frozenset({"nagatha", "bilby"})
    assert cfg.trigger_prefixes == frozenset({"!ask", "!triage"})


def test_from_env_unknown_flag_value_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RTECH_CHATD_INTAKE_ENABLED", "maybe")
    assert IntakeConfig.from_env().enabled is False


def test_from_env_blank_triage_disabled_stays_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    # Disabled config never dispatches, so a blank/whitespace triage id is inert
    # (no default is minted — a default would publish into the void when enabled).
    monkeypatch.setenv("RTECH_CHATD_TRIAGE_AGENT", "   ")
    monkeypatch.delenv("RTECH_CHATD_INTAKE_ENABLED", raising=False)
    assert IntakeConfig.from_env().triage_agent == ""


def test_from_env_enabled_unset_triage_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    # Enabled with RTECH_CHATD_TRIAGE_AGENT unset must fail loud, naming the var
    # and the agentd-<name> convention — never silently default to a bare name.
    monkeypatch.setenv("RTECH_CHATD_INTAKE_ENABLED", "true")
    monkeypatch.delenv("RTECH_CHATD_TRIAGE_AGENT", raising=False)
    with pytest.raises(ValueError, match="RTECH_CHATD_TRIAGE_AGENT"):
        IntakeConfig.from_env()


def test_enabled_blank_triage_agent_fails_closed() -> None:
    with pytest.raises(ValueError, match="RTECH_CHATD_TRIAGE_AGENT"):
        IntakeConfig(enabled=True, triage_agent="  ")


def test_enabled_bare_triage_agent_fails_agentd_convention() -> None:
    with pytest.raises(ValueError, match="full agentd-<name> convention"):
        IntakeConfig(enabled=True, triage_agent="bilby")


@pytest.mark.parametrize("triage_agent", ["", "bilby"])
def test_disabled_triage_agent_is_not_validated(triage_agent: str) -> None:
    # Disabled config never dispatches, so legacy placeholder values remain inert.
    cfg = IntakeConfig(enabled=False, triage_agent=triage_agent)
    assert cfg.enabled is False


# --------------------------------------------------------------- debate participants


def test_debate_participants_default_empty_and_no_scope() -> None:
    # Unset debate participants: single-scope path, no debate scope.
    cfg = IntakeConfig()
    assert cfg.debate_participants == ()
    assert cfg.debate_scope is None


def test_from_env_parses_debate_participants_ordered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RTECH_CHATD_INTAKE_ENABLED", "true")
    monkeypatch.setenv("RTECH_CHATD_TRIAGE_AGENT", "agentd-nagatha")
    monkeypatch.setenv(
        "RTECH_CHATD_DEBATE_PARTICIPANTS", "agentd-nagatha, agentd-bilby ,agentd-iris"
    )
    cfg = IntakeConfig.from_env()
    # Order is preserved (typed identity), NOT a set.
    assert cfg.debate_participants == ("agentd-nagatha", "agentd-bilby", "agentd-iris")
    assert cfg.debate_scope == "trio"


def test_from_env_unset_debate_is_single_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RTECH_CHATD_INTAKE_ENABLED", "true")
    monkeypatch.setenv("RTECH_CHATD_TRIAGE_AGENT", "agentd-nagatha")
    monkeypatch.delenv("RTECH_CHATD_DEBATE_PARTICIPANTS", raising=False)
    cfg = IntakeConfig.from_env()
    assert cfg.debate_participants == ()
    assert cfg.debate_scope is None


def test_duo_debate_scope() -> None:
    cfg = IntakeConfig(
        enabled=True,
        triage_agent="agentd-nagatha",
        debate_participants=("agentd-nagatha", "agentd-bilby"),
    )
    assert cfg.debate_scope == "duo"


def test_debate_participant_count_one_fails_loud() -> None:
    # A single participant is not a debate; must fail loud (use single-scope).
    with pytest.raises(ValueError, match="2 \\(duo\\) or 3 \\(trio\\)"):
        IntakeConfig(
            enabled=True,
            triage_agent="agentd-nagatha",
            debate_participants=("agentd-nagatha",),
        )


def test_debate_participant_count_four_fails_loud() -> None:
    with pytest.raises(ValueError, match="2 \\(duo\\) or 3 \\(trio\\)"):
        IntakeConfig(
            enabled=True,
            triage_agent="agentd-a",
            debate_participants=("agentd-a", "agentd-b", "agentd-c", "agentd-d"),
        )


def test_debate_participant_unsafe_id_fails_loud() -> None:
    # A non-subject-safe id (a dot) fails the shared wire convention fail-loud.
    with pytest.raises(ValueError, match="safe NATS subject token"):
        IntakeConfig(
            enabled=True,
            triage_agent="agentd-nagatha",
            debate_participants=("agentd-nagatha", "bad.id"),
        )


def test_debate_duplicate_participant_fails_loud() -> None:
    with pytest.raises(ValueError, match="unique"):
        IntakeConfig(
            enabled=True,
            triage_agent="agentd-nagatha",
            debate_participants=("agentd-nagatha", "agentd-nagatha"),
        )


def test_debate_triage_not_a_participant_fails_loud() -> None:
    # The triage agent must be one of the debate voices, not a bystander.
    with pytest.raises(ValueError, match="must be one of"):
        IntakeConfig(
            enabled=True,
            triage_agent="agentd-nagatha",
            debate_participants=("agentd-bilby", "agentd-iris"),
        )


def test_debate_triage_is_a_participant_ok() -> None:
    cfg = IntakeConfig(
        enabled=True,
        triage_agent="agentd-nagatha",
        debate_participants=("agentd-nagatha", "agentd-bilby", "agentd-iris"),
    )
    assert cfg.triage_agent in cfg.debate_participants


def test_debate_bare_name_participant_fails_agentd_convention() -> None:
    """D6 repro: a subject-safe BUT bare-name participant fails the agentd-<name>
    convention (the same trap fixed in wake F3).

    ``nagatha`` passes ``validate_participants`` (subject-safe) yet addresses
    ``rtech.agent.nagatha.task`` — a subject no deployed agentd subscribes — so the
    fanout would silently vanish. The convention check fails it loud at config
    build, naming the env var and the required pattern.
    """
    with pytest.raises(ValueError, match="agentd-<name> convention"):
        IntakeConfig(
            enabled=True,
            triage_agent="agentd-nagatha",
            debate_participants=("agentd-nagatha", "bilby"),
        )


def test_debate_convention_enforced_even_when_disabled() -> None:
    """A configured (but disabled) debate roster with a bare id still fails loud."""
    with pytest.raises(ValueError, match="agentd-<name> convention"):
        IntakeConfig(
            enabled=False,
            triage_agent="",
            debate_participants=("agentd-nagatha", "bilby"),
        )


def test_debate_all_agentd_prefixed_participants_pass_convention() -> None:
    """The other direction: a fully agentd-<name> roster is accepted (D6)."""
    cfg = IntakeConfig(
        enabled=True,
        triage_agent="agentd-nagatha",
        debate_participants=("agentd-nagatha", "agentd-bilby"),
    )
    assert cfg.debate_participants == ("agentd-nagatha", "agentd-bilby")


# --------------------------------------------------------------- subject builder


def test_agent_task_subject_matches_ratified_scheme() -> None:
    # The triage id is the deployed agentd-<name>, so the dispatch subject
    # matches the agentd's RTECH_AGENT_ID subscription lane.
    assert subjects.agent_task("agentd-nagatha") == "rtech.agent.agentd-nagatha.task"


def test_agent_task_rejects_blank_agent_id() -> None:
    with pytest.raises(ValueError, match="agent_id must be a non-empty string"):
        subjects.agent_task("  ")
