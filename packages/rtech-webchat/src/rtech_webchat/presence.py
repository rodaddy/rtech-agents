"""Track live agent heartbeat presence for the web-chat API."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from fleet_nats import Envelope, parse_heartbeat
from loguru import logger
from pydantic import BaseModel

STALE_AFTER_INTERVALS = 3
MILLISECONDS_PER_SECOND = 1000
Clock = Callable[[], float]
# Typed liveness probe: returns True while the heartbeat subscription's transport
# is still connected. The presence read consults it so a subscription proven
# healthy once can still re-derive ``subscription_failed`` if the connection is
# later severed (issue #145). It is NOT an agent-count or heartbeat-presence
# check — a healthy-but-empty edge (zero agents, feed up) stays healthy.
LivenessProbe = Callable[[], bool]

# Typed presence-pipeline health (issue #145). This is the ONE signal Dash needs
# to tell a reachable-but-empty edge apart from a broken heartbeat feed:
#   - "healthy":             the heartbeat subscription is established, so the
#                            agent list is authoritative — an empty list means
#                            genuinely zero registered agents, not a broken feed.
#   - "subscription_failed": the heartbeat subscription could not be established,
#                            so the agent list is not trustworthy as liveness.
# Dash already derives edge-unavailable from HTTP failure; this field only splits
# the two cases that both answer HTTP 200 with an empty list. No diagnostics.
PresencePipeline = Literal["healthy", "subscription_failed"]


class AgentPresence(BaseModel):
    """Serialize one agent presence row."""

    agent_id: str
    last_seen_ms: int
    alive: bool


class AgentPresenceResponse(BaseModel):
    """Serialize the agent presence list with typed pipeline health.

    ``pipeline`` is the minimal typed health verdict (issue #145): ``"healthy"``
    means the agent list is authoritative (empty == zero agents), whereas
    ``"subscription_failed"`` means the heartbeat feed is down and the list is
    not liveness truth. Known stale agents still appear here with ``alive=False``
    (Offline) regardless of pipeline state.
    """

    agents: list[AgentPresence]
    pipeline: PresencePipeline


@dataclass(frozen=True)
class PresenceConfig:
    """Configure heartbeat freshness calculation."""

    heartbeat_interval_seconds: float

    def __post_init__(self) -> None:
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")

    @property
    def stale_after_ms(self) -> int:
        """Return the heartbeat freshness cutoff in milliseconds."""

        return int(
            self.heartbeat_interval_seconds * STALE_AFTER_INTERVALS * MILLISECONDS_PER_SECOND
        )


class PresenceRegistry:
    """Keep the latest heartbeat receipt time per agent."""

    def __init__(self, *, config: PresenceConfig, clock: Clock = time.time) -> None:
        """Create an empty registry.

        The pipeline starts as ``subscription_failed``: presence is not trustworthy
        until the heartbeat subscription is proven established (issue #145). The
        edge lifespan flips it to ``healthy`` only after :meth:`mark_subscribed`,
        and the read path re-derives ``subscription_failed`` if the subscription's
        transport is later severed (see :meth:`_current_pipeline`).
        """

        self._config = config
        self._clock = clock
        self._last_seen_by_agent: dict[str, int] = {}
        self._last_seq_by_agent: dict[str, int] = {}
        self._pipeline: PresencePipeline = "subscription_failed"
        self._liveness: LivenessProbe | None = None

    def mark_subscribed(self, *, liveness: LivenessProbe | None = None) -> None:
        """Record that the heartbeat subscription is established (feed is healthy).

        ``liveness`` is a typed probe over the subscription's transport. When
        supplied it is consulted on every read so a later connection drop or
        subscription retirement re-derives ``subscription_failed`` instead of the
        pipeline remaining healthy forever (issue #145). Omitting it preserves the
        one-shot verdict for callers with no live transport to probe.
        """

        self._pipeline = "healthy"
        self._liveness = liveness
        logger.info("webchat.presence.pipeline | subscribed pipeline=healthy")

    def mark_subscription_failed(self) -> None:
        """Record that the heartbeat subscription could not be established."""

        self._pipeline = "subscription_failed"
        self._liveness = None
        logger.warning("webchat.presence.pipeline | pipeline=subscription_failed")

    def _current_pipeline(self) -> PresencePipeline:
        """Re-derive the health verdict from the last mark plus live transport.

        A subscription marked healthy once is only *still* healthy while its
        transport remains connected: the probe closes the gap where a severed
        connection would otherwise leave the pipeline healthy indefinitely
        (issue #145). No probe (or an already-failed pipeline) returns the stored
        verdict unchanged. This never inspects agent count — a connected feed with
        zero agents is a valid healthy-empty edge.
        """

        if self._pipeline != "healthy" or self._liveness is None:
            return self._pipeline
        if self._liveness():
            return "healthy"
        logger.warning("webchat.presence.pipeline | subscription lost pipeline=subscription_failed")
        return "subscription_failed"

    def record_heartbeat(self, agent_id: str, *, seen_ms: int | None = None) -> None:
        """Record a heartbeat receipt for one agent."""

        if not agent_id:
            logger.warning("webchat.presence.record | ignored empty agent_id")
            return
        resolved_seen_ms = seen_ms if seen_ms is not None else self._now_ms()
        self._last_seen_by_agent[agent_id] = resolved_seen_ms
        logger.debug("webchat.presence.record | agent_id={} seen_ms={}", agent_id, resolved_seen_ms)

    def list_agents(self) -> AgentPresenceResponse:
        """Return all known agents with current alive state."""

        logger.debug("webchat.presence.list | entry")
        now_ms = self._now_ms()
        agents = [
            AgentPresence(
                agent_id=agent_id,
                last_seen_ms=last_seen_ms,
                alive=now_ms - last_seen_ms <= self._config.stale_after_ms,
            )
            for agent_id, last_seen_ms in sorted(self._last_seen_by_agent.items())
        ]
        pipeline = self._current_pipeline()
        logger.info("webchat.presence.list | count={} pipeline={}", len(agents), pipeline)
        return AgentPresenceResponse(agents=agents, pipeline=pipeline)

    async def handle_heartbeat(self, data: bytes) -> None:
        """Decode and record one fleet heartbeat envelope from the bus.

        Parses via :func:`fleet_nats.parse_heartbeat` — the shared strict fleet
        parser — so web-chat reads the SAME typed :class:`~fleet_core.spec.Heartbeat`
        every fleet emitter publishes. A malformed beat fails loud (logged and
        skipped), never silently counted as liveness. The read is sequence-aware:
        an out-of-order or duplicate ``seq`` from an already-seen agent is dropped
        so a replayed/stale beat cannot revive a dead agent.
        """

        logger.debug("webchat.presence.heartbeat | entry byte_count={}", len(data))
        try:
            envelope = Envelope.from_bytes(data)
        except Exception as exc:
            logger.error("webchat.presence.heartbeat | decode failed reason={}", exc)
            raise
        if envelope.kind != "heartbeat":
            logger.warning(
                "webchat.presence.heartbeat | ignored kind={} sender={}",
                envelope.kind,
                envelope.sender,
            )
            return
        try:
            heartbeat = parse_heartbeat(envelope)
        except ValueError as exc:
            logger.warning(
                "webchat.presence.heartbeat | malformed beat skipped sender={} reason={}",
                envelope.sender,
                exc,
            )
            return
        agent_id = heartbeat.agent_id
        last_seq = self._last_seq_by_agent.get(agent_id)
        if last_seq is not None and heartbeat.seq <= last_seq:
            logger.debug(
                "webchat.presence.heartbeat | stale seq dropped agent_id={} seq={} last_seq={}",
                agent_id,
                heartbeat.seq,
                last_seq,
            )
            return
        self._last_seq_by_agent[agent_id] = heartbeat.seq
        self.record_heartbeat(agent_id)
        logger.info(
            "webchat.presence.heartbeat | recorded agent_id={} seq={}", agent_id, heartbeat.seq
        )

    def _now_ms(self) -> int:
        return int(self._clock() * MILLISECONDS_PER_SECOND)
