"""Publish periodic agent liveness heartbeats on the fleet heartbeat lane.

This emits the shared fleet-core :class:`~fleet_core.spec.Heartbeat` on the fleet
subject ``{env}.agent.heartbeat.<id>`` so rtech agents are visible in every fleet
client's presence view — not a private ``rtech.agent.<id>.heartbeat`` dialect.

rtech's ``active_turn`` liveness bit rides as a typed extra field on the
Heartbeat (``extra="allow"`` on the fleet-core model), never a prose trailer or a
side channel. The beat carries a monotonic ``seq`` the orchestrator uses to spot
gaps, so it is sequence-aware by construction.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from fleet_nats import Envelope, heartbeat_envelope, subjects
from loguru import logger

Publisher = Callable[[str, Envelope], Awaitable[None]]
Sleep = Callable[[float], Awaitable[None]]
ActiveTurnProbe = Callable[[], bool]

# The rtech-specific liveness bit carried as a typed extra on the fleet Heartbeat.
ACTIVE_TURN_KEY = "active_turn"
DEFAULT_ROLE = "worker"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_id() -> str:
    return uuid4().hex


def heartbeat_subject(env: str, agent_id: str) -> str:
    """Return the fleet heartbeat subject for one agent.

    Delegates to :func:`fleet_nats.subjects.agent_heartbeat` so rtech never
    hand-formats a fleet subject — the fleet naming convention stays the single
    source of truth.
    """
    if not env:
        raise ValueError("env must be non-empty")
    if not agent_id:
        raise ValueError("agent_id must be non-empty")
    return subjects.agent_heartbeat(env, agent_id)


@dataclass(frozen=True)
class HeartbeatConfig:
    """Heartbeat publication settings."""

    agent_id: str
    env: str
    interval_s: float = 30.0
    role: str = DEFAULT_ROLE
    lanes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("agent_id must be non-empty")
        if not self.env:
            raise ValueError("env must be non-empty")
        if self.interval_s <= 0:
            raise ValueError("interval_s must be positive")
        if not self.role:
            raise ValueError("role must be non-empty")


class HeartbeatPublisher:
    """Publish fleet-core heartbeat envelopes at a configured cadence."""

    def __init__(
        self,
        *,
        config: HeartbeatConfig,
        publisher: Publisher,
        active_turn: ActiveTurnProbe,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._config = config
        self._publish = publisher
        self._active_turn = active_turn
        self._sleep = sleep
        self._subject = heartbeat_subject(config.env, config.agent_id)
        self._seq = 0

    @property
    def subject(self) -> str:
        """Return the fleet heartbeat subject this publisher emits to."""
        return self._subject

    async def run(self) -> None:
        """Publish until cancelled."""
        logger.debug("heartbeat.run | entry | subject={}", self._subject)
        try:
            while True:
                await self.publish_once()
                await self._sleep(self._config.interval_s)
        except asyncio.CancelledError:
            logger.info("heartbeat.run | cancelled | subject={}", self._subject)
            raise
        except Exception as exc:
            logger.error("heartbeat.run | failed | subject={} reason={}", self._subject, exc)
            raise

    async def publish_once(self) -> None:
        """Publish one fleet-core heartbeat envelope.

        Builds the beat through :func:`fleet_nats.heartbeat_envelope` (the shared
        fleet builder) and binds rtech's ``active_turn`` as a typed extra on the
        carried Heartbeat and its payload, so the fleet payload shape stays
        authoritative and the rtech signal remains a typed field — never prose.
        """
        seq = self._seq
        active_turn = bool(self._active_turn())
        envelope = heartbeat_envelope(
            msg_id=_new_id(),
            ts=_now_iso(),
            sender=self._config.agent_id,
            role=self._config.role,
            lanes=self._config.lanes,
            seq=seq,
            last_turn_ts=None,
        )
        heartbeat = envelope.typed_signal
        if heartbeat is not None:
            setattr(heartbeat, ACTIVE_TURN_KEY, active_turn)
        envelope.payload[ACTIVE_TURN_KEY] = active_turn
        try:
            await self._publish(self._subject, envelope)
            logger.debug(
                "heartbeat.publish | sent | subject={} seq={} active_turn={}",
                self._subject,
                seq,
                active_turn,
            )
        except Exception as exc:
            logger.error("heartbeat.publish | failed | subject={} reason={}", self._subject, exc)
            raise
        self._seq = seq + 1
