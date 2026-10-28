"""CLI for the Telegram user-session admin ops.

Wraps :mod:`rtech_chatd.user_session.admin` in a small argparse front end,
installed as ``rtech-tg-admin`` (also ``python -m rtech_chatd.tg_admin``).
Subcommands:

* ``list`` — print the account's groups/channels (id, type, is_forum, title) so
  you can find a chat's canonical ``-100…`` id by name. READ-ONLY.
* ``create-group --title <t> [--member <@h> ...]`` — create a fresh
  forum-enabled supergroup and invite members. Prints the new ``-100…`` id.
* ``ensure-forum --chat <id|@handle>`` — turn Topics ON for an existing
  supergroup (idempotent).

Run from the repo root so ``secrets/local/<env>/`` resolves. All human output
goes to STDERR via loguru with a ``TG_ADMIN:`` prefix; the new group id (and the
``list`` table) are the only things written to STDOUT so they can be piped.

Pattern/Convention:
    uv run python -m rtech_chatd.tg_admin list --env dev
    uv run python -m rtech_chatd.tg_admin create-group --env dev \
        --title "Trio Debate" --member @Birk_example_bot --member @Sprocket_example_bot
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from loguru import logger

from .user_session.admin import (
    connected_user_client,
    create_supergroup,
    ensure_forum,
    list_dialogs,
    promote_admins,
)


async def _run_list(env: str) -> int:
    """Print the account's group/channel dialogs to stdout as a table."""
    async with connected_user_client(env=env) as client:
        dialogs = await list_dialogs(client)
    # STDOUT: pipe-friendly, tab-separated so `... | grep -i 'ai test'` works.
    for d in dialogs:
        forum = "forum" if d.is_forum else "-"
        sys.stdout.write(f"{d.chat_id}\t{d.kind}\t{forum}\t{d.title}\n")
    sys.stdout.flush()
    return 0


async def _run_ensure_forum(env: str, chat: str) -> int:
    """Enable Topics on ``chat`` (idempotent)."""
    async with connected_user_client(env=env) as client:
        flipped = await ensure_forum(client, chat)
    if flipped:
        logger.info("TG_ADMIN: Topics turned ON for {}", chat)
    else:
        logger.info("TG_ADMIN: {} already had Topics on; no change", chat)
    return 0


async def _run_create_group(env: str, title: str, members: list[str], about: str) -> int:
    """Create a fresh forum-enabled trio supergroup and invite ``members``.

    Prints the NEW supergroup id to STDOUT (the one thing on stdout) so it can be
    captured and pasted into RTECH_TG_ALLOWED_GROUPS.
    """
    async with connected_user_client(env=env) as client:
        created = await create_supergroup(client, title, members=members, about=about, forum=True)
    logger.info(
        "TG_ADMIN: created {!r} id={} — added {}, failed {}. Set "
        "RTECH_TG_ALLOWED_GROUPS to the id above.",
        created.title,
        created.chat_id,
        created.added,
        created.failed or "none",
    )
    sys.stdout.write(created.chat_id + "\n")
    sys.stdout.flush()
    return 0 if not created.failed else 1


async def _run_promote_admins(env: str, chat: str, members: list[str]) -> int:
    """Promote ``members`` to bot-admin (manage_topics) in ``chat``."""
    async with connected_user_client(env=env) as client:
        results = await promote_admins(client, chat, members)
    bad = {m: r for m, r in results.items() if r != "ok"}
    logger.info("TG_ADMIN: promote-admins results: {}", results)
    return 0 if not bad else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the admin CLI arguments."""
    # --env is shared via a parent parser so it is accepted on EITHER side of the
    # subcommand (`--env dev list` and `list --env dev` both work); with plain
    # subparsers a parent-only option must precede the subcommand, an easy
    # footgun. The child copies use default=SUPPRESS so that OMITTING --env on
    # the subcommand does NOT overwrite a value already parsed by the parent
    # (argparse's parent/child default-collision trap); the real default lives
    # on the parent only.
    child_env = argparse.ArgumentParser(add_help=False)
    child_env.add_argument(
        "--env",
        default=argparse.SUPPRESS,
        help="Environment slug selecting secrets/local/<env>/ (default: dev).",
    )

    parser = argparse.ArgumentParser(
        prog="rtech-tg-admin",
        description="Telegram user-session admin ops (list dialogs, enable Topics).",
    )
    parser.add_argument(
        "--env",
        default="dev",
        help="Environment slug selecting secrets/local/<env>/ (default: dev).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "list",
        parents=[child_env],
        help="List the account's groups/channels (read-only).",
    )

    forum = sub.add_parser(
        "ensure-forum",
        parents=[child_env],
        help="Turn Topics ON for a supergroup (idempotent).",
    )
    forum.add_argument(
        "--chat",
        required=True,
        help="Target chat: a -100… id, numeric id, or @handle.",
    )

    create = sub.add_parser(
        "create-group",
        parents=[child_env],
        help="Create a fresh forum-enabled supergroup and invite members.",
    )
    create.add_argument("--title", required=True, help="New group title.")
    create.add_argument(
        "--member",
        action="append",
        default=[],
        dest="members",
        help="A member to invite (@handle or id). Repeatable.",
    )
    create.add_argument("--about", default="", help="Optional group description.")

    promote = sub.add_parser(
        "promote-admins",
        parents=[child_env],
        help="Grant members bot-admin rights (manage_topics) in a supergroup.",
    )
    promote.add_argument(
        "--chat",
        required=True,
        help="Target supergroup: a -100… id, numeric id, or @handle.",
    )
    promote.add_argument(
        "--member",
        action="append",
        default=[],
        dest="members",
        required=True,
        help="A member to promote (@handle or id). Repeatable.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: dispatch to the requested admin subcommand.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` on success, ``1`` on a handled failure).
    """
    args = _parse_args(argv)
    try:
        if args.command == "list":
            return asyncio.run(_run_list(args.env))
        if args.command == "ensure-forum":
            return asyncio.run(_run_ensure_forum(args.env, args.chat))
        if args.command == "create-group":
            return asyncio.run(_run_create_group(args.env, args.title, args.members, args.about))
        if args.command == "promote-admins":
            return asyncio.run(_run_promote_admins(args.env, args.chat, args.members))
    except (FileNotFoundError, ValueError) as exc:
        logger.error("TG_ADMIN: {}", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 — surface any Telethon RPC failure.
        logger.error("TG_ADMIN: admin op failed: {}", exc, exc_info=True)
        return 1
    return 1  # unreachable: subparsers are required.


if __name__ == "__main__":
    raise SystemExit(main())
