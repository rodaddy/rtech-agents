"""Argument parsing + bot-list splitting for the provision-debate-group CLI.

The network dispatch itself is not exercised here (it needs a live account); only
the pure argparse layer and the ``--bots`` comma-split helper are — the same
test discipline the tg-admin CLI test uses.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
import rtech_chatd.provision_group as provision_group
from rtech_chatd.provision_group import _parse_args, _parse_bots, _run_provision
from rtech_chatd.user_session.provisioning import GroupAudit, ProvisionResult


def test_parse_args_requires_name_and_bots() -> None:
    args = _parse_args(["--name", "Trio Debate", "--bots", "skippy,nagatha,bilby"])
    assert args.name == "Trio Debate"
    assert args.bots == "skippy,nagatha,bilby"
    assert args.env == "dev"  # default


def test_parse_args_accepts_env_and_about() -> None:
    args = _parse_args(
        ["--name", "X", "--bots", "skippy", "--env", "prod", "--about", "the debate"]
    )
    assert args.env == "prod"
    assert args.about == "the debate"


def test_parse_args_opt_in_invite_link_is_explicit() -> None:
    assert _parse_args(["--name", "X", "--bots", "skippy", "--invite-link"]).invite_link is True


def test_parse_args_missing_name_exits() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--bots", "skippy"])


def test_parse_args_missing_bots_exits() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--name", "X"])


def test_parse_bots_splits_and_strips() -> None:
    assert _parse_bots("skippy, nagatha ,bilby") == ["skippy", "nagatha", "bilby"]


def test_parse_bots_drops_empty_segments() -> None:
    assert _parse_bots("skippy,,bilby,") == ["skippy", "bilby"]


def test_parse_bots_all_blank_is_empty() -> None:
    assert _parse_bots("  , ,") == []


def _result(
    *,
    privacy_blocked: list[str] | None = None,
    invite_failed: dict[str, str] | None = None,
    promotions: dict[str, str] | None = None,
) -> ProvisionResult:
    return ProvisionResult(
        chat_id="-100123",
        title="Trio Debate",
        invite_link=None,
        created=False,
        forum_enabled_now=False,
        invited=["@skippy"],
        invite_privacy_blocked=privacy_blocked or [],
        invite_failed=invite_failed or {},
        promotions=promotions or {"@skippy": "ok"},
        audit=GroupAudit(
            chat_id="-100123",
            title="Trio Debate",
            is_forum=True,
            admins=["1"],
            member_count=2,
        ),
    )


@pytest.mark.parametrize(
    "result",
    [
        _result(privacy_blocked=["@nagatha"]),
        _result(invite_failed={"@nagatha": "RuntimeError"}),
        _result(promotions={"@skippy": "ok", "@nagatha": "RuntimeError"}),
    ],
)
async def test_run_provision_exits_nonzero_but_prints_coordinates_when_incomplete(
    result: ProvisionResult,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    @asynccontextmanager
    async def fake_connected_user_client(*, env: str) -> AsyncIterator[object]:
        yield object()

    async def fake_provision(
        client: object, name: str, bots: list[str], *, about: str, mint_invite_link: bool
    ) -> ProvisionResult:
        return result

    monkeypatch.setattr(provision_group, "connected_user_client", fake_connected_user_client)
    monkeypatch.setattr(provision_group, "provision_debate_group", fake_provision)

    exit_code = await _run_provision("dev", "Trio Debate", ["skippy"], "")

    assert exit_code == 1
    assert capsys.readouterr().out == "-100123\n"


async def test_run_provision_exits_zero_when_all_bots_seated_and_promoted(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = _result()

    @asynccontextmanager
    async def fake_connected_user_client(*, env: str) -> AsyncIterator[object]:
        yield object()

    async def fake_provision(
        client: object, name: str, bots: list[str], *, about: str, mint_invite_link: bool
    ) -> ProvisionResult:
        return result

    monkeypatch.setattr(provision_group, "connected_user_client", fake_connected_user_client)
    monkeypatch.setattr(provision_group, "provision_debate_group", fake_provision)

    exit_code = await _run_provision("dev", "Trio Debate", ["skippy"], "")

    assert exit_code == 0
    assert capsys.readouterr().out == "-100123\n"
