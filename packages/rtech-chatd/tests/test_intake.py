"""Functional tests for the deterministic intake hit-gate.

Covers the hit matrix (member-id mention, name trigger, trigger prefix, non-hit
chatter, bot-author humans-only floor, DM exclusion) and the channel-scope
modes (listed preserves today's scoping; all-channels widens except ignored).
No LLM classification: every branch is pure string/id matching.
"""

from __future__ import annotations

from rtech_chatd.channel import InboundMessage
from rtech_chatd.config import AllowBots, GateConfig, IntakeConfig, IntakeMode
from rtech_chatd.gates import intake_hit


def _msg(**overrides: object) -> InboundMessage:
    base: dict[str, object] = {
        "channel_id": "chan1",
        "thread_id": None,
        "author": "user1",
        "text": "hello",
        "is_bot": False,
        "is_dm": False,
        "mentions_me": False,
        "is_reply": False,
        "mentions_other_bot": False,
        "mentions_human": False,
        "guild_id": "g1",
        "message_id": "m1",
        "mentioned_ids": frozenset(),
    }
    base.update(overrides)
    return InboundMessage(**base)  # type: ignore[arg-type]


def _intake(**overrides: object) -> IntakeConfig:
    # Enabled intake requires an explicit triage id (no default); the hit-gate
    # under test never reads it, but construction fails closed without it.
    base: dict[str, object] = {
        "enabled": True,
        "mode": IntakeMode.ALL_CHANNELS,
        "triage_agent": "agentd-nimbus",
    }
    base.update(overrides)
    return IntakeConfig(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------- master switch


def test_disabled_intake_never_hits() -> None:
    cfg = IntakeConfig(enabled=False, member_ids=frozenset({"77"}))
    assert intake_hit(_msg(mentioned_ids=frozenset({"77"})), GateConfig(), cfg) is False


# ---------------------------------------------------------------- hit types


def test_member_id_mention_is_hit() -> None:
    cfg = _intake(member_ids=frozenset({"77", "88"}))
    assert intake_hit(_msg(mentioned_ids=frozenset({"88"})), GateConfig(), cfg) is True


def test_trigger_prefix_is_hit() -> None:
    cfg = _intake(trigger_prefixes=frozenset({"!ask"}))
    assert intake_hit(_msg(text="!ask what is the deploy status"), GateConfig(), cfg) is True


def test_trigger_prefix_case_insensitive() -> None:
    cfg = _intake(trigger_prefixes=frozenset({"!ask"}))
    assert intake_hit(_msg(text="!ASK something"), GateConfig(), cfg) is True


def test_trigger_prefix_requires_word_boundary() -> None:
    # D4: a prefix hit needs the prefix followed by end-of-string or whitespace,
    # so ``!ask`` fires on ``!ask ...`` and bare ``!ask`` but NOT ``!asking``.
    cfg = _intake(trigger_prefixes=frozenset({"!ask"}))
    # Positive: exact token and token+space.
    assert intake_hit(_msg(text="!ask"), GateConfig(), cfg) is True
    assert intake_hit(_msg(text="!ask what is the deploy status"), GateConfig(), cfg) is True
    # Negative: the prefix is only the head of a longer word.
    assert intake_hit(_msg(text="!asking about lunch"), GateConfig(), cfg) is False


def test_name_trigger_whole_word_is_hit() -> None:
    cfg = _intake(name_triggers=frozenset({"nimbus"}))
    assert intake_hit(_msg(text="hey Nimbus can you help"), GateConfig(), cfg) is True


def test_name_trigger_requires_whole_word() -> None:
    # "nimbuss" must NOT match the token "nimbus" (whole-word only).
    cfg = _intake(name_triggers=frozenset({"nimbus"}))
    assert intake_hit(_msg(text="the nimbuss are coming"), GateConfig(), cfg) is False


def test_non_hit_chatter_ignored() -> None:
    cfg = _intake(
        member_ids=frozenset({"77"}),
        name_triggers=frozenset({"nimbus"}),
        trigger_prefixes=frozenset({"!ask"}),
    )
    assert intake_hit(_msg(text="just some random chatter"), GateConfig(), cfg) is False


# ---------------------------------------------------------------- humans-only floor


def test_bot_message_ignored_under_default_allow_bots() -> None:
    cfg = _intake(member_ids=frozenset({"77"}))
    msg = _msg(is_bot=True, mentioned_ids=frozenset({"77"}))
    # GateConfig default allow_bots is NONE — humans only (baseline).
    assert intake_hit(msg, GateConfig(), cfg) is False


def test_bot_message_allowed_when_allow_bots_all() -> None:
    cfg = _intake(member_ids=frozenset({"77"}))
    msg = _msg(is_bot=True, mentioned_ids=frozenset({"77"}))
    assert intake_hit(msg, GateConfig(allow_bots=AllowBots.ALL), cfg) is True


def test_self_message_never_hits() -> None:
    cfg = _intake(member_ids=frozenset({"77"}))
    msg = _msg(mentioned_ids=frozenset({"77"}))
    assert intake_hit(msg, GateConfig(), cfg, is_self=True) is False


def test_user_not_in_allowlist_rejected() -> None:
    cfg = _intake(member_ids=frozenset({"77"}))
    gate = GateConfig(allowed_users=frozenset({"vip"}))
    msg = _msg(author="rando", mentioned_ids=frozenset({"77"}))
    assert intake_hit(msg, gate, cfg) is False


def test_dm_is_not_an_intake_source() -> None:
    cfg = _intake(member_ids=frozenset({"77"}))
    msg = _msg(is_dm=True, guild_id=None, mentioned_ids=frozenset({"77"}))
    assert intake_hit(msg, GateConfig(), cfg) is False


# ---------------------------------------------------------------- channel scope


def test_all_channels_mode_hits_any_channel() -> None:
    cfg = _intake(member_ids=frozenset({"77"}), mode=IntakeMode.ALL_CHANNELS)
    msg = _msg(channel_id="never-listed", mentioned_ids=frozenset({"77"}))
    assert intake_hit(msg, GateConfig(), cfg) is True


def test_all_channels_mode_still_honors_ignored() -> None:
    cfg = _intake(member_ids=frozenset({"77"}), mode=IntakeMode.ALL_CHANNELS)
    gate = GateConfig(ignored_channels=frozenset({"chan1"}))
    msg = _msg(channel_id="chan1", mentioned_ids=frozenset({"77"}))
    assert intake_hit(msg, gate, cfg) is False


def test_listed_mode_requires_configured_channel() -> None:
    cfg = _intake(member_ids=frozenset({"77"}), mode=IntakeMode.LISTED)
    # Empty allow/free lists in LISTED mode = no widening = no channel in scope.
    msg = _msg(channel_id="chan1", mentioned_ids=frozenset({"77"}))
    assert intake_hit(msg, GateConfig(), cfg) is False


def test_listed_mode_hits_in_allowed_channel() -> None:
    cfg = _intake(member_ids=frozenset({"77"}), mode=IntakeMode.LISTED)
    gate = GateConfig(allowed_channels=frozenset({"chan1"}))
    msg = _msg(channel_id="chan1", mentioned_ids=frozenset({"77"}))
    assert intake_hit(msg, gate, cfg) is True


def test_listed_mode_hits_in_free_response_channel() -> None:
    cfg = _intake(member_ids=frozenset({"77"}), mode=IntakeMode.LISTED)
    gate = GateConfig(free_response_channels=frozenset({"chan1"}))
    msg = _msg(channel_id="chan1", mentioned_ids=frozenset({"77"}))
    assert intake_hit(msg, gate, cfg) is True


def test_thread_matches_parent_channel_in_listed_scope() -> None:
    cfg = _intake(member_ids=frozenset({"77"}), mode=IntakeMode.LISTED)
    gate = GateConfig(allowed_channels=frozenset({"chan1"}))
    msg = _msg(channel_id="chan1", thread_id="t1", mentioned_ids=frozenset({"77"}))
    assert intake_hit(msg, gate, cfg) is True
