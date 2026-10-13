"""Argument parsing for the rtech-tg-admin CLI (issue #110).

Locks the footgun that --env must work on EITHER side of the subcommand, and
that ensure-forum requires --chat. The network dispatch itself is not exercised
here (it needs a live account); only the pure argparse layer is.
"""

from __future__ import annotations

import pytest
from rtech_chatd.tg_admin import _parse_args


def test_list_accepts_env_after_subcommand() -> None:
    args = _parse_args(["list", "--env", "prod"])
    assert args.command == "list"
    assert args.env == "prod"


def test_list_accepts_env_before_subcommand() -> None:
    args = _parse_args(["--env", "prod", "list"])
    assert args.command == "list"
    assert args.env == "prod"


def test_env_defaults_to_dev() -> None:
    assert _parse_args(["list"]).env == "dev"


def test_ensure_forum_parses_chat() -> None:
    args = _parse_args(["ensure-forum", "--chat", "-1001234567890", "--env", "dev"])
    assert args.command == "ensure-forum"
    assert args.chat == "-1001234567890"


def test_ensure_forum_requires_chat() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["ensure-forum"])


def test_create_group_parses_title_and_members() -> None:
    args = _parse_args(
        [
            "create-group",
            "--title",
            "Trio Debate",
            "--member",
            "@Bilby_example_bot",
            "--member",
            "@Skippy_example_bot",
            "--env",
            "dev",
        ]
    )
    assert args.command == "create-group"
    assert args.title == "Trio Debate"
    assert args.members == ["@Bilby_example_bot", "@Skippy_example_bot"]


def test_create_group_requires_title() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["create-group"])


def test_create_group_members_default_empty() -> None:
    args = _parse_args(["create-group", "--title", "X"])
    assert args.members == []


def test_promote_admins_parses_chat_and_members() -> None:
    args = _parse_args(
        [
            "promote-admins",
            "--chat",
            "-1001111111111",
            "--member",
            "@Bilby_example_bot",
            "--member",
            "5000000001",
        ]
    )
    assert args.command == "promote-admins"
    assert args.chat == "-1001111111111"
    assert args.members == ["@Bilby_example_bot", "5000000001"]


def test_promote_admins_requires_member() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["promote-admins", "--chat", "-1009"])


def test_subcommand_required() -> None:
    with pytest.raises(SystemExit):
        _parse_args([])
