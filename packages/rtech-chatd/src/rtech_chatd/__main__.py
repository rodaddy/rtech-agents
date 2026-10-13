"""Run the chatd Discord edge as a service (`python -m rtech_chatd`)."""

from __future__ import annotations

import asyncio

from loguru import logger

from .service import ChatdServiceConfig, run_service


def main() -> None:
    """Build the service config from the environment and run the edge."""

    config = ChatdServiceConfig.from_env()
    logger.info(
        "chatd.main | startup stack={} inbox={} participants={} allow_bots={}",
        config.stack,
        config.edge.inbox_channel,
        list(config.participants),
        config.allow_bots.value,
    )
    asyncio.run(run_service(config))


if __name__ == "__main__":
    main()
