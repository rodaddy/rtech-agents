"""Run the chatd **Telegram** long-poll edge as a service (``rtech-tgd``).

The standalone Telegram production entrypoint: builds the
:class:`~rtech_chatd.telegram_service.TgdServiceConfig` from the environment
(fail-closed on a missing bot ref, group allowlist, or stack id) and runs the
Telegram-only edge until its poll loop dies. It composes ONLY the Telegram
surface — no Discord adapter, no orchestrator, no agentd. Long polling means
there is no inbound port, webhook, or public domain.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from .telegram_service import ORIGIN_EDGE_TELEGRAM, TgdServiceConfig, run_service


def main() -> None:
    """Build the Telegram-edge config from the environment and run the edge."""
    config = TgdServiceConfig.from_env()
    logger.info(
        "chatd.tgd.main | startup stack={} origin={} groups={} identities={} intake={}",
        config.stack,
        ORIGIN_EDGE_TELEGRAM,
        len(config.telegram.allowed_groups),
        len(config.identity_refs),
        config.intake.enabled,
    )
    asyncio.run(run_service(config))


if __name__ == "__main__":
    main()
