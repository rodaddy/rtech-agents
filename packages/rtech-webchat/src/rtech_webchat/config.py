"""Load immutable runtime configuration for the web-chat edge."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_NATS_URL = "nats://127.0.0.1:4222"
DEFAULT_ENV = "dev"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_ORIGIN_EDGE = "web-chat"
DEFAULT_REPLAY_FRAME_LIMIT = 200
DEFAULT_REPLAY_TTL_SECONDS = 900
DEFAULT_REPLAY_MAX_TASKS = 1000
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30.0
# Env var naming the stack id stamped on every TaskFrame this edge publishes.
# The orchestrator's ingress guard only consumes matching tasks.
# There is NO default: a shared literal is fail-open producer addressing — two
# unconfigured edges on the shared bus would stamp the same address — so the
# stack id is required, mirroring orchestratord's loud missing-id behavior.
ENV_STACK_ID = "RTECH_STACK_ID"
ALLOWED_ORIGIN_EDGES = frozenset({"web-chat", "dash-dash"})


@dataclass(frozen=True)
class WebchatConfig:
    """Hold web-chat runtime settings."""

    stack: str
    nats_url: str = DEFAULT_NATS_URL
    env: str = DEFAULT_ENV
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    origin_edge: str = DEFAULT_ORIGIN_EDGE
    api_key: str | None = None
    replay_frame_limit: int = DEFAULT_REPLAY_FRAME_LIMIT
    replay_ttl_seconds: int = DEFAULT_REPLAY_TTL_SECONDS
    replay_max_tasks: int = DEFAULT_REPLAY_MAX_TASKS
    heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS

    def __post_init__(self) -> None:
        if not self.nats_url.startswith("nats://"):
            raise ValueError(f"nats_url must be a nats:// URL, got {self.nats_url!r}")
        if not self.env:
            raise ValueError("env must be non-empty")
        if not self.host:
            raise ValueError("host must be non-empty")
        if self.port <= 0:
            raise ValueError(f"port must be positive, got {self.port}")
        if self.origin_edge not in ALLOWED_ORIGIN_EDGES:
            raise ValueError(f"origin_edge must be allowlisted, got {self.origin_edge!r}")
        if not self.stack or not self.stack.strip():
            raise ValueError(
                "stack must be non-empty: this edge stamps every TaskFrame with its "
                f"stack id; set {ENV_STACK_ID} — there is no default"
            )
        if self.replay_frame_limit <= 0:
            raise ValueError(f"replay_frame_limit must be positive, got {self.replay_frame_limit}")
        if self.replay_ttl_seconds <= 0:
            raise ValueError(f"replay_ttl_seconds must be positive, got {self.replay_ttl_seconds}")
        if self.replay_max_tasks <= 0:
            raise ValueError(f"replay_max_tasks must be positive, got {self.replay_max_tasks}")
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError(
                "heartbeat_interval_seconds must be positive, "
                f"got {self.heartbeat_interval_seconds}"
            )

    @classmethod
    def from_env(cls) -> WebchatConfig:
        """Build config from environment variables.

        Raises:
            ValueError: If ``RTECH_STACK_ID`` is unset or blank — the stack
                stamp is security-floor addressing config with no default.
        """

        stack = os.getenv(ENV_STACK_ID, "").strip()
        if not stack:
            raise ValueError(
                f"{ENV_STACK_ID} must be set to this deployment's stack id "
                "; the edge never defaults to a shared literal"
            )
        raw_port = os.getenv("RTECH_WEBCHAT_PORT")
        port = int(raw_port) if raw_port is not None else DEFAULT_PORT
        raw_replay_frame_limit = os.getenv("RTECH_WEBCHAT_REPLAY_FRAME_LIMIT")
        replay_frame_limit = (
            int(raw_replay_frame_limit)
            if raw_replay_frame_limit is not None
            else DEFAULT_REPLAY_FRAME_LIMIT
        )
        raw_replay_ttl_seconds = os.getenv("RTECH_WEBCHAT_REPLAY_TTL_SECONDS")
        replay_ttl_seconds = (
            int(raw_replay_ttl_seconds)
            if raw_replay_ttl_seconds is not None
            else DEFAULT_REPLAY_TTL_SECONDS
        )
        raw_replay_max_tasks = os.getenv("RTECH_WEBCHAT_REPLAY_MAX_TASKS")
        replay_max_tasks = (
            int(raw_replay_max_tasks)
            if raw_replay_max_tasks is not None
            else DEFAULT_REPLAY_MAX_TASKS
        )
        raw_heartbeat_interval_seconds = os.getenv("RTECH_WEBCHAT_HEARTBEAT_INTERVAL_SECONDS")
        heartbeat_interval_seconds = (
            float(raw_heartbeat_interval_seconds)
            if raw_heartbeat_interval_seconds is not None
            else DEFAULT_HEARTBEAT_INTERVAL_SECONDS
        )
        return cls(
            stack=stack,
            nats_url=os.getenv("RTECH_WEBCHAT_NATS_URL", DEFAULT_NATS_URL),
            env=os.getenv("RTECH_WEBCHAT_ENV", DEFAULT_ENV),
            host=os.getenv("RTECH_WEBCHAT_HOST", DEFAULT_HOST),
            port=port,
            origin_edge=os.getenv("RTECH_WEBCHAT_ORIGIN_EDGE", DEFAULT_ORIGIN_EDGE),
            api_key=os.getenv("RTECH_AGENTS_API_KEY"),
            replay_frame_limit=replay_frame_limit,
            replay_ttl_seconds=replay_ttl_seconds,
            replay_max_tasks=replay_max_tasks,
            heartbeat_interval_seconds=heartbeat_interval_seconds,
        )
