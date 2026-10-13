"""Functional I/O tests for the pure gate contract.

Covers the full should_respond matrix (allow_bots x mention x free-response x
DM x allowlists) and every thread_decision no-thread coupling. This is the
deliverable's core.
"""

from __future__ import annotations

import pytest
from rtech_chatd.channel import InboundMessage
from rtech_chatd.config import AllowBots, GateConfig, ThreadConfig
from rtech_chatd.gates import should_respond, thread_decision


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
    }
    base.update(overrides)
    return InboundMessage(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------- self / bots


def test_self_message_never_answered() -> None:
    assert should_respond(_msg(mentions_me=True), GateConfig(), is_self=True) is False


def test_bot_none_ignored_even_when_mentioned() -> None:
    cfg = GateConfig(allow_bots=AllowBots.NONE)
    assert should_respond(_msg(is_bot=True, mentions_me=True), cfg) is False


def test_bot_mentions_requires_mention() -> None:
    cfg = GateConfig(allow_bots=AllowBots.MENTIONS)
    assert should_respond(_msg(is_bot=True, mentions_me=False), cfg) is False
    assert should_respond(_msg(is_bot=True, mentions_me=True), cfg) is True


def test_bot_all_accepts_unmentioned_bot() -> None:
    # allow_bots=all + free channel so the mention gate does not re-block it.
    cfg = GateConfig(allow_bots=AllowBots.ALL, free_response_channels=frozenset({"chan1"}))
    assert should_respond(_msg(is_bot=True, mentions_me=False), cfg) is True


def test_bot_filter_runs_before_user_allowlist() -> None:
    # An allowed bot must NOT be rejected for being absent from allowed_users.
    cfg = GateConfig(
        allow_bots=AllowBots.ALL,
        allowed_users=frozenset({"someone-else"}),
        free_response_channels=frozenset({"chan1"}),
    )
    assert should_respond(_msg(is_bot=True, author="botX", mentions_me=False), cfg) is True


# ---------------------------------------------------------------- user allowlist


def test_empty_allowlist_allows_everyone() -> None:
    assert should_respond(_msg(mentions_me=True), GateConfig()) is True


def test_user_not_in_allowlist_rejected() -> None:
    cfg = GateConfig(allowed_users=frozenset({"vip"}))
    assert should_respond(_msg(author="rando", mentions_me=True), cfg) is False


def test_user_in_allowlist_passes() -> None:
    cfg = GateConfig(allowed_users=frozenset({"vip"}))
    assert should_respond(_msg(author="vip", mentions_me=True), cfg) is True


def test_role_authorized_bypasses_user_allowlist() -> None:
    cfg = GateConfig(allowed_users=frozenset({"vip"}), allowed_roles=frozenset({"role1"}))
    assert should_respond(_msg(author="rando", mentions_me=True), cfg, role_authorized=True) is True


def test_role_configured_no_match_no_user_rejected() -> None:
    cfg = GateConfig(allowed_roles=frozenset({"role1"}))
    msg = _msg(author="rando", mentions_me=True)
    assert should_respond(msg, cfg, role_authorized=False) is False


# ---------------------------------------------------------------- DMs


def test_dm_never_gated_even_without_mention() -> None:
    cfg = GateConfig(require_mention=True, allow_dms=True)
    assert should_respond(_msg(is_dm=True, mentions_me=False), cfg) is True


def test_dm_still_honors_user_allowlist() -> None:
    cfg = GateConfig(allow_dms=True, allowed_users=frozenset({"vip"}))
    assert should_respond(_msg(is_dm=True, author="rando"), cfg) is False


def test_dm_dropped_by_default() -> None:
    # allow_dms defaults False (fail closed): a DM is a free-response surface
    # outside the designated inbox, forbidden unless an operator opts in.
    assert should_respond(_msg(is_dm=True, mentions_me=True), GateConfig()) is False


# ---------------------------------------------------------------- mention gate


def test_guild_requires_mention_by_default() -> None:
    assert should_respond(_msg(mentions_me=False), GateConfig()) is False
    assert should_respond(_msg(mentions_me=True), GateConfig()) is True


def test_require_mention_false_answers_without_mention() -> None:
    cfg = GateConfig(require_mention=False)
    assert should_respond(_msg(mentions_me=False), cfg) is True


def test_free_response_channel_waives_mention() -> None:
    cfg = GateConfig(free_response_channels=frozenset({"chan1"}))
    assert should_respond(_msg(mentions_me=False), cfg) is True


def test_free_response_wildcard_waives_mention() -> None:
    cfg = GateConfig(free_response_channels=frozenset({"*"}))
    assert should_respond(_msg(mentions_me=False), cfg) is True


# ---------------------------------------------------------------- thread mention behavior


def test_thread_message_still_requires_mention() -> None:
    cfg = GateConfig()
    msg = _msg(thread_id="t1", mentions_me=False)
    assert should_respond(msg, cfg) is False


def test_thread_message_can_use_explicit_no_mention_config() -> None:
    msg = _msg(thread_id="t1", mentions_me=False)
    assert should_respond(msg, GateConfig(require_mention=False)) is True


# ---------------------------------------------------------------- channel lists


def test_allowed_channels_whitelist_blocks_others() -> None:
    cfg = GateConfig(allowed_channels=frozenset({"chanX"}))
    assert should_respond(_msg(mentions_me=True), cfg) is False


def test_allowed_channels_whitelist_permits_member() -> None:
    cfg = GateConfig(allowed_channels=frozenset({"chan1"}))
    assert should_respond(_msg(mentions_me=True), cfg) is True


def test_ignored_channel_blocks_even_when_mentioned() -> None:
    cfg = GateConfig(ignored_channels=frozenset({"chan1"}))
    assert should_respond(_msg(mentions_me=True), cfg) is False


def test_thread_matches_parent_channel_in_lists() -> None:
    # An ignored PARENT channel blocks a message inside its thread.
    cfg = GateConfig(ignored_channels=frozenset({"chan1"}))
    msg = _msg(channel_id="chan1", thread_id="t1", mentions_me=True)
    assert should_respond(msg, cfg) is False


# ---------------------------------------------------------------- multi-agent filter


def test_other_bot_mentioned_not_us_stays_silent() -> None:
    cfg = GateConfig()
    msg = _msg(mentions_me=False, mentions_other_bot=True)
    assert should_respond(msg, cfg) is False


def test_other_bot_mentioned_but_us_too_answers() -> None:
    cfg = GateConfig()
    msg = _msg(mentions_me=True, mentions_other_bot=True)
    assert should_respond(msg, cfg) is True


def test_human_mentioned_not_us_stays_silent() -> None:
    cfg = GateConfig()
    msg = _msg(mentions_me=False, mentions_human=True)
    assert should_respond(msg, cfg) is False


def test_free_response_overrides_multi_agent_filter() -> None:
    cfg = GateConfig(free_response_channels=frozenset({"chan1"}))
    msg = _msg(mentions_me=False, mentions_other_bot=True)
    assert should_respond(msg, cfg) is True


# ---------------------------------------------------------------- thread_decision


def test_thread_decision_creates_on_trigger() -> None:
    d = thread_decision(_msg(text="Fix the deploy bug please"), ThreadConfig())
    assert d.create is True
    assert d.name == "Fix the deploy bug please"


def test_thread_decision_skipped_when_disabled() -> None:
    d = thread_decision(_msg(), ThreadConfig(auto_thread=False))
    assert d.create is False
    assert d.reason == "auto-thread-disabled"


def test_thread_decision_skipped_in_existing_thread() -> None:
    d = thread_decision(_msg(thread_id="t1"), ThreadConfig())
    assert d.create is False
    assert d.reason == "already-in-thread"


def test_thread_decision_skipped_in_dm() -> None:
    d = thread_decision(_msg(is_dm=True), ThreadConfig())
    assert d.create is False
    assert d.reason == "dm"


def test_thread_decision_free_response_implies_no_thread() -> None:
    cfg = ThreadConfig(free_response_channels=frozenset({"chan1"}))
    d = thread_decision(_msg(), cfg)
    assert d.create is False
    assert d.reason == "free-response-channel"


def test_thread_decision_no_thread_channel() -> None:
    cfg = ThreadConfig(no_thread_channels=frozenset({"chan1"}))
    d = thread_decision(_msg(), cfg)
    assert d.create is False
    assert d.reason == "no-thread-channel"


def test_thread_decision_reply_implies_no_thread() -> None:
    d = thread_decision(_msg(is_reply=True), ThreadConfig())
    assert d.create is False
    assert d.reason == "reply"


# ---------------------------------------------------------------- thread name


def test_thread_name_strips_mention_markup() -> None:
    d = thread_decision(_msg(text="<@123> hey <#456> look at <@&789> this"), ThreadConfig())
    assert d.create is True
    assert "<@" not in d.name
    assert "<#" not in d.name
    assert d.name == "hey look at this"


def test_thread_name_capped_with_ellipsis() -> None:
    long = "x" * 200
    cfg = ThreadConfig(thread_name_max=80)
    d = thread_decision(_msg(text=long), cfg)
    assert len(d.name) == 80
    assert d.name.endswith("...")


def test_thread_name_falls_back_when_empty() -> None:
    cfg = ThreadConfig(default_thread_name="chat")
    d = thread_decision(_msg(text="<@123> <#456>"), cfg)
    assert d.create is True
    assert d.name == "chat"


def test_thread_name_exactly_at_cap_not_truncated() -> None:
    cfg = ThreadConfig(thread_name_max=10)
    d = thread_decision(_msg(text="abcde fghi"), cfg)  # 10 chars
    assert d.create is True
    assert d.name == "abcde fghi"
    assert "..." not in d.name


@pytest.mark.parametrize(
    ("allow", "is_bot", "mentions_me", "expect"),
    [
        (AllowBots.NONE, True, True, False),
        (AllowBots.NONE, False, True, True),
        (AllowBots.MENTIONS, True, True, True),
        (AllowBots.MENTIONS, True, False, False),
        (AllowBots.ALL, True, False, True),
    ],
)
def test_allow_bots_matrix(allow: AllowBots, is_bot: bool, mentions_me: bool, expect: bool) -> None:
    # free channel neutralises the mention gate so we isolate the bot policy.
    cfg = GateConfig(allow_bots=allow, free_response_channels=frozenset({"chan1"}))
    assert should_respond(_msg(is_bot=is_bot, mentions_me=mentions_me), cfg) is expect
