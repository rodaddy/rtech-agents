"""CLI for idempotent trio debate-group provisioning.

The automation-tier entry point, installed as ``provision-debate-group`` (also
``python -m rtech_chatd.provision_group``). It drives the one command the
acceptance criteria name:

    provision-debate-group --name X --bots sprocket,nimbus,birk

It loads the Telethon user session through the SAME fail-loud lifecycle the rest
of the edge uses (:func:`rtech_chatd.user_session.admin.connected_user_client`),
runs the idempotent recipe in
:func:`rtech_chatd.user_session.provisioning.provision_debate_group`, and prints
the canonical group id to STDOUT so it can be piped/captured:

    <-100… group id>

With ``--invite-link`` it also mints a BOUNDED, revocable human-join link and
emits ``<-100… group id>\t<invite link>`` instead. By default no link is minted:
the bots are seated by direct add and a Bot API account cannot use an invite link,
so a durable link would just be a standing bearer-credential leak.

Everything human — the created/reused flag, the per-bot invite/promote results,
the post-run audit — goes to STDERR via loguru with a ``TG_PROVISION:`` prefix,
so ``$(provision-debate-group --name X --bots sprocket)`` cleanly yields the id
alone. A re-run with the same ``--name`` reuses the existing group (prints
``reused`` in the human line) and re-emits the same id.

Run from the repo root so ``secrets/local/<env>/`` resolves. Credentials are
never logged; a revoked session / rate limit fails loud with an ACTION REQUIRED
message and a non-zero exit — the credential itself never reaches a log line.

Pattern/Convention:
    uv run provision-debate-group --name "Trio Debate" --bots sprocket,nimbus,birk
    uv run python -m rtech_chatd.provision_group --name "Trio Debate" \
        --bots sprocket,nimbus,birk --env dev
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from loguru import logger

from .user_session.admin import connected_user_client
from .user_session.provisioning import (
    ProvisioningError,
    ProvisionResult,
    provision_debate_group,
)


def _parse_bots(raw: str) -> list[str]:
    """Split the ``--bots`` comma list into non-blank identifiers.

    ``"sprocket, nimbus ,birk"`` → ``["sprocket", "nimbus", "birk"]``. Empty
    segments (a trailing comma) are dropped; an all-blank value yields ``[]`` so
    the downstream resolver fails loud with a clear ACTION REQUIRED.
    """
    return [part.strip() for part in raw.split(",") if part.strip()]


async def _run_provision(
    env: str, name: str, bots: list[str], about: str, *, mint_invite_link: bool = False
) -> int:
    """Connect, run the idempotent recipe, and print the canonical id to stdout."""
    async with connected_user_client(env=env) as client:
        result = await provision_debate_group(
            client, name, bots, about=about, mint_invite_link=mint_invite_link
        )
    _log_human_summary(result)
    # STDOUT: the canonical group id (always), tab-followed by a bounded invite link
    # ONLY when one was explicitly requested. Bots are seated by direct add, so the
    # default emits the id alone and never a durable link.
    if result.invite_link is not None:
        sys.stdout.write(f"{result.chat_id}\t{result.invite_link}\n")
    else:
        sys.stdout.write(f"{result.chat_id}\n")
    sys.stdout.flush()
    incomplete = bool(
        result.invite_privacy_blocked
        or result.invite_failed
        or any(status != "ok" for status in result.promotions.values())
    )
    return 1 if incomplete else 0


def _log_human_summary(result: ProvisionResult) -> None:
    """Emit the human-readable provisioning summary to STDERR (loguru)."""
    logger.info(
        "TG_PROVISION: {} group {!r} id={} — {}",
        "created" if result.created else "reused",
        result.title,
        result.chat_id,
        "Topics turned ON" if result.forum_enabled_now else "Topics already on",
    )
    logger.info(
        "TG_PROVISION: invited={} privacy-blocked={} invite-failed={} promotions={}",
        result.invited,
        result.invite_privacy_blocked,
        result.invite_failed or "none",
        result.promotions,
    )
    logger.info(
        "TG_PROVISION: audit — forum={} admins={} members={}",
        result.audit.is_forum,
        len(result.audit.admins),
        result.audit.member_count,
    )
    if result.invite_privacy_blocked:
        logger.warning(
            "TG_PROVISION: {} bot(s) are privacy-blocked; a bot CANNOT join via an "
            "invite link — an admin must add them manually. See invite-failed for the "
            "per-bot remediation.",
            len(result.invite_privacy_blocked),
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the provision-debate-group CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="provision-debate-group",
        description=(
            "Idempotently provision the trio debate supergroup: create-or-reuse, "
            "enable Topics, invite + promote the bots, mint an invite link."
        ),
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Group title (also the idempotency key: same name reuses the group).",
    )
    parser.add_argument(
        "--bots",
        required=True,
        help=(
            "Comma-separated configured fleet bot slugs (sprocket,nimbus,birk) or their "
            "configured @handles."
        ),
    )
    parser.add_argument(
        "--env",
        default="dev",
        help="Environment slug selecting secrets/local/<env>/ (default: dev).",
    )
    parser.add_argument(
        "--about",
        default="",
        help="Optional group description (only applied on first creation).",
    )
    parser.add_argument(
        "--invite-link",
        action="store_true",
        help=(
            "Also mint a BOUNDED, revocable human-join invite link (time-boxed + "
            "use-capped). Off by default: bots are seated by direct add and cannot "
            "use a link, so no durable link is emitted unless you ask for one."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run the idempotent debate-group provisioning recipe.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: ``0`` only when every bot was directly seated and
        promoted; ``1`` on handled errors or any incomplete seating/promotion.
    """
    args = _parse_args(argv)
    bots = _parse_bots(args.bots)
    try:
        return asyncio.run(
            _run_provision(args.env, args.name, bots, args.about, mint_invite_link=args.invite_link)
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("TG_PROVISION: {}", exc)
        return 1
    except ProvisioningError as exc:
        # A fail-loud auth/flood/peer-flood class — the message is credential-free
        # and carries the ACTION REQUIRED remediation.
        logger.error("TG_PROVISION: {} ({})", exc, exc.kind)
        return 1
    except Exception as exc:  # noqa: BLE001 — surface any other Telethon RPC failure.
        logger.error("TG_PROVISION: provisioning failed: {}", type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
