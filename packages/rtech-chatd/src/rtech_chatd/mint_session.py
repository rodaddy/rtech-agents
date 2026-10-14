"""Mint a Telethon user-session ``StringSession`` interactively.

This is the in-app generator for a Telegram *user* session — the CLI entry point
``python -m rtech_chatd.mint_session`` (also installed as ``rtech-tg-session``).
It reads ``api_id`` / ``api_hash`` from :func:`load_telegram_user_config`, runs
the interactive Telethon login (phone/code/2FA), prints the resulting
``StringSession`` inside a security banner, and — with ``--write`` — persists the
session back into the same gitignored per-env secrets JSON.

Architecture:
    The interactive login is deliberately thin and pushes every reusable, pure
    piece behind a seam so the untestable parts (a real phone, a real code) are
    the ONLY untested surface:

    * :func:`load_telegram_user_config` supplies ``api_id`` / ``api_hash`` (the
      storage seam — Option 3 backend swap lands there, not here).
    * :func:`render_session_banner` is a pure string builder — a SECRET-FREE
      confirmation (redacted fingerprint plus the "one IP at a time" security
      note) written to stderr; the raw session goes to stdout only. Unit-tested
      without any network.
    * :func:`persist_session_string` is a pure JSON merge that preserves
      ``api_id`` / ``api_hash`` and only rewrites ``session_string``.
    * :func:`mint_session` owns the Telethon connect/login/disconnect lifecycle;
      it is the only network-touching function and is NOT unit-tested (a real
      login needs a human).

Key Components:
    - mint_session: async interactive login → StringSession (network).
    - render_session_banner: pure banner text (session + security note).
    - persist_session_string: pure secrets-JSON merge for ``--write``.
    - main: argparse CLI (``--env`` / ``--phone`` / ``--write``).

Pattern/Convention:
    Run from the repo root so ``secrets/local/<env>/`` resolves::

        uv run python -m rtech_chatd.mint_session --env dev
        uv run python -m rtech_chatd.mint_session --env dev --write

    The verification code and 2FA password are ALWAYS entered interactively; a
    phone number MAY be supplied via ``--phone`` but otherwise is prompted too.
    The raw ``StringSession`` is written to STDOUT (pipe-friendly); all human
    log output goes to STDERR via loguru with a ``TG_SESSION:`` prefix. The
    stderr banner is redacted — it never embeds the raw session.

Example:
    >>> from rtech_chatd.mint_session import render_session_banner
    >>> secret = "1AsELgTHEFULLSECRETSESSIONVALUE"
    >>> banner = render_session_banner(secret)
    >>> secret in banner  # the raw session is NEVER in the stderr banner
    False
    >>> "ONE IP address at a time" in banner
    True

See Also:
    - rtech_chatd.user_session.config: the storage seam + fail-loud loader.
    - rtech_chatd.telegram_adapter: the bot-token inbound edge.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys
from pathlib import Path

from loguru import logger
from telethon import TelegramClient
from telethon.errors import (
    AuthKeyUnregisteredError,
    FloodWaitError,
    SessionPasswordNeededError,
)
from telethon.sessions import StringSession

from .user_session.config import (
    TelegramUserConfig,
    load_telegram_user_config,
    secrets_path_for_env,
)

_BANNER_RULE = "=" * 60


def render_session_banner(session_string: str) -> str:
    """Build the human banner wrapping a minted ``StringSession``.

    Pure (no I/O) so it is unit-testable. Carries the same security note the b1x
    generator prints: a user session is single-IP and must not be shared across
    deployments. The raw session is NEVER embedded — only a redacted fingerprint
    (length + short tail) — because this banner is written to the stderr log sink.

    Args:
        session_string: The minted Telethon ``StringSession`` value (used only
            to derive the redacted fingerprint; not embedded verbatim).

    Returns:
        The multi-line, SECRET-FREE banner text (fingerprint + security note).
    """
    # The session string itself is NEVER embedded here: this banner is written
    # to the loguru stderr sink (which terminal/system log capture may record),
    # so it must stay secret-free. The raw session goes to STDOUT only (see
    # ``main``). We surface only a redacted fingerprint so the operator can
    # confirm a value was produced without the value leaking into logs.
    fingerprint = (
        f"len={len(session_string)} tail=…{session_string[-6:]}" if session_string else "EMPTY"
    )
    return "\n".join(
        [
            _BANNER_RULE,
            "SESSION MINTED — the raw string was written to STDOUT only.",
            f"  ({fingerprint})",
            "Copy it from STDOUT into your gitignored telegram_user.json.",
            _BANNER_RULE,
            "IMPORTANT:",
            "- This session can only be active from ONE IP address at a time",
            "- Do NOT use the same session for two concurrent deployments",
            "- Store this session string only in a gitignored secrets file",
            _BANNER_RULE,
        ]
    )


def persist_session_string(path: Path, session_string: str) -> None:
    """Merge ``session_string`` into the secrets JSON, preserving credentials.

    A pure JSON round-trip (read → set one key → write). ``api_id`` / ``api_hash``
    and any other keys are preserved untouched; only ``session_string`` is
    rewritten. The file must already exist (the generator loads it first), so a
    missing file here is a programmer error surfaced as ``FileNotFoundError``.

    Args:
        path: Path to the env's ``telegram_user.json``.
        session_string: The minted session value to persist.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"CRITICAL: cannot persist session, secrets file missing at {path}. "
            "ACTION REQUIRED: run without --write, or create the file first."
        )
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    data["session_string"] = session_string
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    logger.info("TG_SESSION: persisted session_string to {}", path)


async def mint_session(config: TelegramUserConfig, phone: str | None = None) -> str:
    """Run the interactive Telethon login and return a new ``StringSession``.

    Connects with an empty ``StringSession`` and drives the phone/code/2FA login.
    The verification code and 2FA password are ALWAYS interactive; ``phone`` may
    be supplied (else prompted). The client is disconnected in ``finally`` so a
    failed login never leaks a live connection.

    Args:
        config: Loaded credential (supplies ``api_id`` / ``api_hash``).
        phone: Optional phone number (with country code). Prompted when ``None``.

    Returns:
        The minted ``StringSession`` string.

    Raises:
        SessionPasswordNeededError: 2FA required but no password path completed.
        AuthKeyUnregisteredError: The auth key was revoked; re-mint required.
        FloodWaitError: Telegram rate-limited the login; retry after the wait.
    """

    def _prompt(message: str) -> str:
        # Prompt on STDERR, not STDOUT: STDOUT is the machine-readable
        # session-only stream (see ``main``), so any prompt text there would
        # corrupt a piped ``… > session.txt``. Writing the prompt to stderr
        # keeps the two streams cleanly separated.
        sys.stderr.write(message)
        sys.stderr.flush()
        return input()

    def phone_callback() -> str:
        if phone:
            logger.info("TG_SESSION: using provided phone (value withheld)")
            return phone
        return _prompt("Enter your phone number (with country code, e.g. +15551234567): ")

    def code_callback() -> str:
        # Always interactive: the code is generated only after connect. Read via
        # getpass so the one-time code is NOT echoed into terminal scrollback or
        # a PTY recording; the prompt goes to stderr, keeping stdout clean.
        return getpass.getpass(
            "Enter the verification code sent to your Telegram app: ", stream=sys.stderr
        )

    def password_callback() -> str:
        # Always interactive: never accept a 2FA password via argv (leaks in ps).
        # getpass suppresses echo so the password never hits terminal scrollback.
        return getpass.getpass("Enter your 2FA password: ", stream=sys.stderr)

    client = TelegramClient(StringSession(), config.api_id, config.api_hash)
    try:
        await client.connect()
        logger.info("TG_SESSION: connected to Telegram")

        if not await client.is_user_authorized():
            await client.start(
                phone=phone_callback,
                code_callback=code_callback,
                password=password_callback,
            )

        me = await client.get_me()
        # ``get_me`` returns a User here (authorized above); id is safe to log.
        logger.info("TG_SESSION: authenticated as user_id={}", getattr(me, "id", "?"))

        # StringSession.save returns Any (telethon is stub-less); the saved value
        # is documented to be the session string.
        session_string = str(StringSession.save(client.session))
        logger.info("TG_SESSION: session minted")
        return session_string
    except SessionPasswordNeededError:
        logger.error(
            "TG_SESSION: 2FA password required but login did not complete. "
            "ACTION REQUIRED: re-run and enter the account's 2FA password when "
            "prompted.",
            exc_info=True,
        )
        raise
    except AuthKeyUnregisteredError:
        logger.error(
            "TG_SESSION: auth key unregistered (session revoked). ACTION REQUIRED: "
            "re-mint a fresh session for this account.",
            exc_info=True,
        )
        raise
    except FloodWaitError as exc:
        logger.error(
            "TG_SESSION: Telegram rate-limited the login (FloodWait {}s). "
            "ACTION REQUIRED: wait the stated interval, then re-run.",
            getattr(exc, "seconds", "?"),
            exc_info=True,
        )
        raise
    finally:
        await client.disconnect()
        logger.debug("TG_SESSION: disconnected")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the mint-session CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="rtech-tg-session",
        description="Mint a Telethon user-session StringSession for the chatd edge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Run from the repo root so secrets/local/<env>/ resolves.\n"
            "The verification code and 2FA password are always entered "
            "interactively."
        ),
    )
    parser.add_argument(
        "--env",
        default="dev",
        help="Environment slug selecting secrets/local/<env>/ (default: dev).",
    )
    parser.add_argument(
        "--phone",
        default=None,
        help=(
            "Phone number with country code. Prompted interactively if omitted. "
            "NOTE: a value passed here is visible in the process list (ps) and "
            "shell history; omit it to be prompted privately instead."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist the minted session back into the env's telegram_user.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: load config, mint a session, print/persist it.

    Loads the per-env credential (fails loud on a missing/placeholder file), runs
    the interactive login, writes the raw ``StringSession`` to STDOUT, and — with
    ``--write`` — persists it back into the secrets JSON.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` on success, ``1`` on a handled failure).
    """
    args = _parse_args(argv)
    try:
        config = load_telegram_user_config(args.env)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("TG_SESSION: cannot load config: {}", exc)
        return 1

    logger.info("TG_SESSION: minting session for env={}", args.env)
    try:
        session_string = asyncio.run(mint_session(config, phone=args.phone))
    except KeyboardInterrupt:
        logger.warning("TG_SESSION: cancelled by user")
        return 1
    except Exception as exc:  # noqa: BLE001 — surface any Telethon failure cleanly.
        logger.error("TG_SESSION: session minting failed: {}", exc, exc_info=True)
        return 1

    # Human banner to STDERR (loguru), machine-readable session to STDOUT so the
    # value can be piped without the surrounding noise.
    logger.info("TG_SESSION: minted session\n{}", render_session_banner(session_string))
    sys.stdout.write(session_string + "\n")
    sys.stdout.flush()

    if args.write:
        path = secrets_path_for_env(args.env)
        persist_session_string(path, session_string)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
