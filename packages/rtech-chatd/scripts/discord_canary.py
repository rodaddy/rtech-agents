#!/usr/bin/env python
"""Live Discord round-trip canary: mention -> auto-thread -> reply.

Rico-gated. This is the ONLY code path that opens a live Discord connection.
It reads the bot token indirectly through an ``env:VAR`` ``*_ref`` lookup and
NEVER accepts or prints a literal token. The operator may source the secret from
Vaultwarden, but injects the VALUE into the env var the ref names — only
``env:VAR`` refs resolve at runtime (a ``vaultwarden:`` ref is unsupported).

When the token is not provisioned it exits 0 with a clear
"token not provisioned" SKIP message — so CI stays green until Rico wires the
bot for the test agent (spec Phase 6).

Usage:
    uv run python packages/rtech-chatd/scripts/discord_canary.py \
        --token-ref env:RTECH_CHATD_BOT_TOKEN \
        --channel <channel_id>

Token ref forms:
    env:VARNAME          - read from an environment variable (indirection only);
                           this is the ONLY ref shape resolve_token accepts. To
                           use a Vaultwarden-held secret, the operator fetches the
                           VALUE and writes it into VARNAME, then configures
                           ``env:VARNAME`` — a ``vaultwarden:`` ref is not resolved.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from loguru import logger
from rtech_chatd.config import resolve_token

# SKIP is a clean, Rico-gated exit — not a failure.
_SKIP_EXIT = 0
_FAIL_EXIT = 1


async def _run_roundtrip(token: str, channel_id: str) -> int:
    """Open a live connection and prove mention -> thread -> reply.

    Imported lazily so the SKIP path never needs discord.py at import time.
    """
    import discord
    from rtech_chatd.channel import SendTarget
    from rtech_chatd.config import GateConfig, ThreadConfig
    from rtech_chatd.discord_adapter import DiscordChannelAdapter

    def provider() -> str:
        return token

    adapter = DiscordChannelAdapter(
        token_provider=provider,
        gate_config=GateConfig(),
        thread_config=ThreadConfig(),
    )

    result = {"ok": False}

    async def _consume() -> None:
        async for msg in adapter.events():
            logger.info("canary | inbound | thread={} text={!r}", msg.thread_id, msg.text[:80])
            target = SendTarget(channel_id=msg.channel_id, thread_id=msg.thread_id)
            await adapter.send(target, "canary ack: round-trip proven", [])
            result["ok"] = True
            break

    consumer = asyncio.create_task(_consume())
    try:
        await asyncio.wait_for(adapter.start(), timeout=60.0)
    except (TimeoutError, discord.DiscordException) as exc:
        logger.error("canary | live connection failed | reason={}", exc)
    finally:
        consumer.cancel()
        await adapter.close()

    if result["ok"]:
        logger.info("canary | PASS | round-trip proven")
        return 0
    logger.error("canary | FAIL | no round-trip observed")
    return _FAIL_EXIT


def main(argv: list[str] | None = None) -> int:
    if os.environ.get("DISCORD_LIVE") != "1":
        logger.info("canary | SKIP | live canary disabled (set DISCORD_LIVE=1)")
        return _SKIP_EXIT

    parser = argparse.ArgumentParser(description="Discord round-trip canary (Rico-gated).")
    parser.add_argument(
        "--token-ref",
        default=os.environ.get("RTECH_CHATD_TOKEN_REF", "env:RTECH_CHATD_BOT_TOKEN"),
        help=(
            "Indirect token ref; runtime accepts env:VAR only. "
            "Vaultwarden-held values must be injected into the named "
            "environment variable. Never a literal."
        ),
    )
    parser.add_argument("--channel", default=os.environ.get("RTECH_CHATD_CANARY_CHANNEL", ""))
    args = parser.parse_args(argv)

    token = resolve_token(args.token_ref)
    if not token:
        logger.info(
            "canary | SKIP | token not provisioned (ref={}); wire the test-agent bot "
            "token before running the live canary (spec Phase 6).",
            args.token_ref,
        )
        return _SKIP_EXIT

    if not args.channel:
        logger.info(
            "canary | SKIP | no --channel provided; live round-trip needs a target channel."
        )
        return _SKIP_EXIT

    return asyncio.run(_run_roundtrip(token, args.channel))


if __name__ == "__main__":
    sys.exit(main())
