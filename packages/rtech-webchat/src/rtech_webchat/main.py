"""Run the rtech-webchat ASGI app."""

from __future__ import annotations

import uvicorn

from .app import create_app
from .config import WebchatConfig


def main() -> None:
    """Start uvicorn with loopback binding by default."""

    config = WebchatConfig.from_env()
    uvicorn.run(create_app(config=config), host=config.host, port=config.port)
