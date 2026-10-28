"""Load immutable runtime configuration for the Hermes bridge.

The bridge straddles two subject namespaces on the fleet NATS bus:

* the **rtech** side — typed ``rtech_frames`` frames on ``rtech.*`` subjects
  (``rtech.agent.<id>.task`` inbound, ``rtech.task.<id>.turn`` outbound), and
* the **Hermes/fleet** side — ``fleet_nats.Envelope`` messages on the
  ``{env}.agent.*`` subjects the gateway's ``fleet_subscriber`` adapter reads.

In the current deployment both sides live on the *same* server (the fleet bus,
``nats://127.0.0.1:4222``); the two URLs are kept as distinct config fields so a
future split (e.g. an isolated rtech bus) needs no code change. Only the subject
namespace differs, never the wire host by default.

Secret env VALUES are never logged; the bus URLs are host:port only, not secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_RTECH_NATS_URL = "nats://127.0.0.1:4222"
DEFAULT_FLEET_NATS_URL = "nats://127.0.0.1:4222"
DEFAULT_FLEET_ENV = "dev"
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30.0
DEFAULT_TASK_TIMEOUT_SECONDS = 900.0
DEFAULT_MAX_QUEUE_DEPTH = 64
_VALID_FLEET_ENVS = frozenset({"dev", "prod"})

ENV_FLEET_IDS = "RTECH_BRIDGE_FLEET_IDS"
ENV_RTECH_NATS_URL = "RTECH_BRIDGE_RTECH_NATS_URL"
ENV_FLEET_NATS_URL = "RTECH_BRIDGE_FLEET_NATS_URL"
ENV_FLEET_ENV = "RTECH_BRIDGE_FLEET_ENV"
ENV_HEARTBEAT_INTERVAL = "RTECH_BRIDGE_HEARTBEAT_INTERVAL_SECONDS"
ENV_DEFAULT_CHANNEL = "RTECH_BRIDGE_DEFAULT_CHANNEL"
ENV_TASK_TIMEOUT = "RTECH_BRIDGE_TASK_TIMEOUT_SECONDS"
ENV_MAX_QUEUE_DEPTH = "RTECH_BRIDGE_MAX_QUEUE_DEPTH"


@dataclass(frozen=True)
class BridgeConfig:
    """Hold Hermes-bridge runtime settings.

    Attributes:
        fleet_ids: The fleet agent ids this bridge fronts (e.g. ``["birk"]``).
            One bridge process can front several ids; each gets its own
            ``rtech.agent.<id>.task`` subscription and its own heartbeat.
        rtech_nats_url: The bus carrying ``rtech.*`` typed frames.
        fleet_nats_url: The bus carrying ``{env}.agent.*`` fleet envelopes (the
            gateway's bus). Defaults to the same live fleet bus as the rtech side.
        fleet_env: The fleet subject env prefix (``dev``/``prod``) — must match
            the gateway's ``FLEET_ENV``.
        heartbeat_interval_seconds: Cadence for the per-fleet-id liveness beat
            that lists the bridged agent in the web-chat picker.
        default_channel: Fallback Discord channel stamped on the outbound Hermes
            envelope when a task carries none. The gateway's mapping skips any
            turn with no resolvable channel, so a task with no channel and no
            default is dropped at the Hermes side.
        task_timeout_seconds: How long a dispatched task may stay in flight
            without a typed completion before the bridge expires it — the
            reply-lane subscription is dropped, a ``StatusFrame(blocked)`` is
            published, and the next queued task for the agent dispatches.
        max_queue_depth: How many tasks may wait behind an agent's active task
            (round-2 H6). Admission past this depth is shed loudly with a typed
            ``StatusFrame(blocked)`` so a slow agent never grows bridge memory
            or latency without bound.
    """

    fleet_ids: tuple[str, ...]
    rtech_nats_url: str = DEFAULT_RTECH_NATS_URL
    fleet_nats_url: str = DEFAULT_FLEET_NATS_URL
    fleet_env: str = DEFAULT_FLEET_ENV
    heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS
    default_channel: str | None = None
    task_timeout_seconds: float = DEFAULT_TASK_TIMEOUT_SECONDS
    max_queue_depth: int = DEFAULT_MAX_QUEUE_DEPTH
    # Reserved for future per-id metadata; kept typed to avoid a bare dict later.
    _reserved: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.fleet_ids:
            raise ValueError("fleet_ids must name at least one fleet agent id")
        if any(not fid or not fid.strip() for fid in self.fleet_ids):
            raise ValueError("every fleet id must be a non-empty string")
        if len(set(self.fleet_ids)) != len(self.fleet_ids):
            raise ValueError(f"fleet_ids must be unique, got {self.fleet_ids!r}")
        if not self.rtech_nats_url.startswith("nats://"):
            raise ValueError(f"rtech_nats_url must be a nats:// URL, got {self.rtech_nats_url!r}")
        if not self.fleet_nats_url.startswith("nats://"):
            raise ValueError(f"fleet_nats_url must be a nats:// URL, got {self.fleet_nats_url!r}")
        if self.fleet_env not in _VALID_FLEET_ENVS:
            raise ValueError(
                f"fleet_env must be one of {sorted(_VALID_FLEET_ENVS)}, got {self.fleet_env!r}"
            )
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError(
                "heartbeat_interval_seconds must be positive, "
                f"got {self.heartbeat_interval_seconds}"
            )
        if self.task_timeout_seconds <= 0:
            raise ValueError(
                f"task_timeout_seconds must be positive, got {self.task_timeout_seconds}"
            )
        if self.max_queue_depth <= 0:
            raise ValueError(f"max_queue_depth must be positive, got {self.max_queue_depth}")
        if self.default_channel is not None and not self.default_channel.strip():
            raise ValueError("default_channel, when set, must be non-empty")

    @classmethod
    def from_env(cls) -> BridgeConfig:
        """Build config from environment variables.

        ``RTECH_BRIDGE_FLEET_IDS`` is a comma-separated list of fleet agent ids
        (required — the bridge fronts nothing by default). All other settings
        fall back to the live fleet bus defaults.

        Returns:
            The validated :class:`BridgeConfig`.

        Raises:
            ValueError: If ``RTECH_BRIDGE_FLEET_IDS`` is unset/empty or any field
                fails validation.
        """
        raw_ids = os.getenv(ENV_FLEET_IDS, "")
        fleet_ids = tuple(fid.strip() for fid in raw_ids.split(",") if fid.strip())
        if not fleet_ids:
            raise ValueError(
                f"{ENV_FLEET_IDS} must name at least one fleet agent id "
                "(comma-separated, e.g. 'birk' or 'birk,sprocket')"
            )
        raw_interval = os.getenv(ENV_HEARTBEAT_INTERVAL)
        heartbeat_interval = (
            float(raw_interval) if raw_interval is not None else DEFAULT_HEARTBEAT_INTERVAL_SECONDS
        )
        raw_timeout = os.getenv(ENV_TASK_TIMEOUT)
        task_timeout = (
            float(raw_timeout) if raw_timeout is not None else DEFAULT_TASK_TIMEOUT_SECONDS
        )
        raw_channel = os.getenv(ENV_DEFAULT_CHANNEL)
        default_channel = raw_channel.strip() if raw_channel and raw_channel.strip() else None
        raw_depth = os.getenv(ENV_MAX_QUEUE_DEPTH)
        max_queue_depth = int(raw_depth) if raw_depth is not None else DEFAULT_MAX_QUEUE_DEPTH
        return cls(
            fleet_ids=fleet_ids,
            rtech_nats_url=os.getenv(ENV_RTECH_NATS_URL, DEFAULT_RTECH_NATS_URL),
            fleet_nats_url=os.getenv(ENV_FLEET_NATS_URL, DEFAULT_FLEET_NATS_URL),
            fleet_env=os.getenv(ENV_FLEET_ENV, DEFAULT_FLEET_ENV),
            heartbeat_interval_seconds=heartbeat_interval,
            default_channel=default_channel,
            task_timeout_seconds=task_timeout,
            max_queue_depth=max_queue_depth,
        )
