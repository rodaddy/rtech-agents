"""Derive typed presence-transition frames from PresenceRegistry deltas (issue #147).

Presence/liveness truth is owned by :class:`rtech_webchat.presence.PresenceRegistry`
(issue #145). This module is a THIN observer over that ONE store: it does NOT add a
second presence store, poller, queue, database, or any enterprise machinery. On each
evaluation tick it reads ``registry.list_agents()`` and, per agent, compares the
current ``alive`` flag against the last DELIVERED (``committed``) state it recorded. A
genuine edge emits exactly one typed
:class:`~rtech_frames.frames.PresenceTransitionFrame`:

- committed alive and now dead  -> emit ONE ``offline`` frame, advance committed to
  dead on successful delivery.
- committed dead  and now alive -> emit ONE ``recovered`` frame, advance committed to
  alive on successful delivery.
- no change (steady state)       -> emit NOTHING. This transition-only dedup is the
  ONLY dedup mechanism — there are NO time windows.

First sighting emits nothing: a newly-seen agent records ``committed`` = current alive
with no frame and no pending edge (it was never in a prior state to transition from).

DELIVERY SEMANTICS (at-least-once). The committed (last DELIVERED) state is kept
SEPARATE from AT MOST ONE ``pending`` edge (a frame created-but-not-yet-delivered).
A publish failure RETAINS the pending edge and does NOT advance committed, so a
transient bus error never LOSES an edge: the next tick RETRIES the pending edge first,
then — if the state has moved again — creates the next edge in order. Delivery is
therefore AT-LEAST-ONCE: a duplicate delivery is possible only under an ambiguous
transport ack (we published but never learned the ack), so connectors MUST be
idempotent if the bus redelivers the same typed frame.

Each edge carries a STABLE typed ``transition_id`` (issue #147) minted ONCE when the
pending frame is created and RETAINED on retry (the same held ``pending_frame`` is
re-published), so an ambiguous-ack redelivery of the SAME edge carries the SAME id.
The consumer keys a bounded seen-id cache on it to render/send that edge exactly
once, while two DISTINCT edges (distinct ids) still deliver in order. The id is a
delivery-idempotency key only — the NATS typed frame stays truth, and the id is never
rendered to an operator surface.

Concurrency: the whole read->compare->publish sequence of each :meth:`evaluate` is
serialized under ONE ``asyncio.Lock`` so reentrant/concurrent calls cannot interleave
or reorder edges.

Bounding: tracked identities are pruned when absent from the authoritative snapshot
(unless they still owe a pending edge) and capped at
:data:`MAX_TRACKED_IDENTITIES` with DETERMINISTIC oldest-absent eviction; a present
snapshot larger than the cap FAILS CLOSED (sanitized bounded counts logged, never raw
ids) rather than growing unbounded.

The deriver stamps a clean, safe ``display_name`` (from the persona slug) and a
tz-aware UTC ``observed_at`` from an injected clock, then hands the frame to a publish
callback. It NEVER logs private content, raw payloads, raw agent ids, or credential-
bearing subjects — human-facing logs carry the transition kind + display_name and
bounded counts only.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import uuid4

from loguru import logger
from rtech_frames.frames import PresenceTransitionFrame, is_valid_presence_agent_id

from .presence import PresenceRegistry

# Publish one derived frame onto the bus (the app wires this to ``encode_frame`` +
# ``bus.publish`` on the PRESENCE_TRANSITION subject). Kept as an injected callback so
# the deriver stays pure and the whole edge logic is testable without a live NATS.
PublishTransition = Callable[[PresenceTransitionFrame], Awaitable[None]]
# Returns a tz-aware UTC ``datetime`` — the observed transition time stamped onto each
# frame. Injected so tests are deterministic and no naive datetime ever rides the wire.
TransitionClock = Callable[[], datetime]

# Hard cap on tracked identities (committed state + at-most-one pending edge each).
# A deterministic backstop so a churn of one-shot slugs can never grow the deriver's
# memory without bound; eviction is oldest-absent-first (never random).
MAX_TRACKED_IDENTITIES = 256

# Safe display-name vocabulary (mirrors rtech_frames._DISPLAY_NAME_PATTERN): letters,
# digits, spaces, dot, hyphen, underscore. safe_display_name() defensively strips any
# character outside this set so its output ALWAYS passes the frame's display_name
# validator, even if fed an unexpected slug.
_SAFE_DISPLAY_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ._-"
)


def safe_display_name(agent_id: str) -> str:
    """Clean a persona slug into a safe, title-cased human display name.

    The slug is normally the NATS-subject-safe token (``[A-Za-z0-9_-]+``, e.g.
    ``skippy``, ``nagatha-2``, ``foo_bar``). ``-``/``_`` become spaces, each word is
    title-cased, and surrounding whitespace is stripped: ``skippy`` -> ``Skippy``,
    ``nagatha-2`` -> ``Nagatha 2``, ``foo_bar`` -> ``Foo Bar``.

    DEFENSIVE: any character OUTSIDE the safe display vocabulary (letters, digits,
    space, ``.``, ``-``, ``_``) is dropped, so the output ALWAYS passes the frame's
    ``display_name`` validator even if fed an unexpected slug. A slug that reduces to
    nothing safe falls back to ``"agent"`` so the name is never blank. Pure and
    unit-tested; it is the ONLY identity a connector renders.

    Args:
        agent_id: The persona slug token.

    Returns:
        The cleaned, title-cased display name (always a valid safe display name).
    """

    spaced = agent_id.replace("-", " ").replace("_", " ")
    stripped = "".join(ch for ch in spaced if ch in _SAFE_DISPLAY_CHARS)
    name = " ".join(word for word in stripped.split()).title().strip()
    return name or "agent"


@dataclass
class _AgentState:
    """Per-agent committed state plus at-most-one pending (undelivered) edge.

    ``committed`` is the last alive-bool whose edge was successfully delivered (or the
    first-sighting baseline). ``pending`` is the alive-bool of the ONE frame created
    but not yet confirmed delivered (``None`` = no owed edge). The pending frame itself
    is held so a retry re-publishes the SAME typed frame (idempotent redelivery).
    """

    committed: bool
    pending: bool | None = None
    pending_frame: PresenceTransitionFrame | None = None


class PresenceTransitionDeriver:
    """Observe PresenceRegistry alive-state deltas and emit one frame per edge.

    Holds a reference to the EXISTING registry (never a copy of its state), a publish
    callback, and a UTC clock. Per-agent committed/pending state lives in ONE
    insertion-ordered dict; the ONLY dedup is the committed-vs-current comparison in
    :meth:`evaluate` — no timers. Delivery is at-least-once (a failed publish retains
    the pending edge for retry), and the whole evaluate sequence is serialized under
    one lock.
    """

    def __init__(
        self,
        *,
        registry: PresenceRegistry,
        publish: PublishTransition,
        clock: TransitionClock,
    ) -> None:
        """Create the deriver over an existing registry (no second store)."""

        self._registry = registry
        self._publish = publish
        self._clock = clock
        # Insertion-ordered so deterministic oldest-absent eviction is possible.
        self._agents: OrderedDict[str, _AgentState] = OrderedDict()
        # Serializes the whole read->compare->publish sequence of each evaluate() so
        # reentrant/concurrent calls cannot interleave or reorder edges.
        self._lock = asyncio.Lock()

    async def evaluate(self) -> None:
        """Read the registry once and publish one frame per genuine presence edge.

        The WHOLE read->compare->publish sequence runs under one lock so concurrent or
        reentrant calls cannot interleave/reorder edges. For each agent in the
        authoritative snapshot, in order:

        1. A ``pending`` edge is RETRIED first: on success advance committed and clear
           pending; on failure retain pending and move on (no new edge this tick).
        2. After a delivered pending, compare committed against the SAME current alive;
           if it moved again, create+publish the NEXT edge in order (offline before
           recovered), same success/failure handling.
        3. With no pending and committed != current alive, create+publish the true
           edge; success advances committed and clears pending, failure retains it.
        4. First sighting records committed = current alive, no frame, no pending.

        After processing, tracked identities absent from the snapshot with NO pending
        edge are pruned. A present snapshot larger than the cap fails closed (bounded
        sanitized counts logged, never raw ids). Delivery is at-least-once: a duplicate
        is possible only under an ambiguous transport ack, so connectors must be
        idempotent.
        """

        async with self._lock:
            logger.debug("webchat.presence.transition | evaluate entry")
            snapshot = self._registry.list_agents().agents
            emitted = 0
            processed = 0
            # The COMPLETE authoritative identity set for this tick, precomputed
            # BEFORE any processing. Cap admission evicts only identities absent from
            # THIS complete set (and with no pending edge) — never an identity that is
            # present-but-not-yet-processed later in the snapshot. Using the
            # processed-so-far set instead would let a lexically-early newcomer evict a
            # still-present tracked identity whose offline edge is about to be derived,
            # erasing/re-baselining that edge.
            # Only VALID authoritative identities count as "present" for prune/evict:
            # an invalid slug never enters tracking, so it must not shield a same-named
            # tracked identity from pruning/eviction (it never can collide, but keeping
            # the present-set to valid ids keeps the invariant explicit).
            snapshot_ids = {
                agent.agent_id for agent in snapshot if is_valid_presence_agent_id(agent.agent_id)
            }
            over_cap = 0
            invalid = 0
            for agent in snapshot:
                agent_id = agent.agent_id
                # Isolate an invalid authoritative identity BEFORE any frame is built:
                # a subject-safe-but-over-cap (65+ char) or otherwise-unsupported slug
                # would raise inside PresenceTransitionFrame construction and, being
                # uncaught, abort the WHOLE evaluate tick — permanently suppressing every
                # later agent's transition. The shared identity predicate is checked here
                # so ONE malformed/unsupported identity is skipped (fail closed) and never
                # enters tracking, while every later transition still processes. Sanitized
                # bounded count only — never the raw id.
                if not is_valid_presence_agent_id(agent_id):
                    invalid += 1
                    continue
                # Fail closed on a present snapshot larger than the cap: never
                # unbounded-allocate. Process up to the cap deterministically (an
                # already-tracked agent still processes; only NEW slugs past the cap
                # are refused) and drop the rest with a sanitized bounded log.
                if agent_id not in self._agents and len(self._agents) >= MAX_TRACKED_IDENTITIES:
                    if not self._evict_one_absent(snapshot_ids):
                        over_cap += 1
                        continue
                processed += 1
                if await self._process_agent(agent_id, agent.alive):
                    emitted += 1
            self._prune_absent(snapshot_ids)
            if invalid:
                # Bounded, sanitized: count only, NEVER the raw id(s). One malformed
                # identity was skipped so later transitions still delivered this tick.
                logger.error(
                    "webchat.presence.transition | skipped invalid authoritative "
                    "identities skipped={}",
                    invalid,
                )
            if over_cap:
                # Bounded, sanitized: counts only, NEVER raw ids or payloads.
                logger.error(
                    "webchat.presence.transition | snapshot exceeded cap; refused new "
                    "identities snapshot_size={} cap={} refused={}",
                    len(snapshot),
                    MAX_TRACKED_IDENTITIES,
                    over_cap,
                )
            logger.debug(
                "webchat.presence.transition | evaluate exit emitted={} processed={}",
                emitted,
                processed,
            )

    async def _process_agent(self, agent_id: str, alive: bool) -> bool:
        """Advance one agent's committed/pending state; return whether a frame emitted.

        Runs under the evaluate lock. Retries a pending edge first, then creates the
        next edge if committed still differs from the current authoritative ``alive``.
        A publish failure retains the pending edge (at-least-once) rather than
        advancing committed.
        """

        state = self._agents.get(agent_id)
        if state is None:
            # First sighting: record committed baseline, no frame, no pending.
            self._agents[agent_id] = _AgentState(committed=alive)
            self._agents.move_to_end(agent_id)
            return False
        self._agents.move_to_end(agent_id)

        emitted = False
        # 1. Retry a pending edge FIRST.
        if state.pending is not None and state.pending_frame is not None:
            if await self._deliver(state.pending_frame):
                state.committed = state.pending
                state.pending = None
                state.pending_frame = None
                emitted = True
            else:
                # Retain pending; do not advance committed, do not create a new edge
                # this tick.
                return False

        # 2/3. With no pending, create the next true edge if committed moved again.
        if state.pending is None and state.committed != alive:
            frame = self._build_frame(agent_id, alive)
            state.pending = alive
            state.pending_frame = frame
            if await self._deliver(frame):
                state.committed = alive
                state.pending = None
                state.pending_frame = None
                emitted = True
            # On failure the pending edge is retained for the next tick's retry.
        return emitted

    def _build_frame(self, agent_id: str, alive: bool) -> PresenceTransitionFrame:
        """Build one typed edge frame (offline on ->dead, recovered on ->alive).

        Mints the stable ``transition_id`` ONCE here, when the pending frame is
        created (the repo's ``uuid4().hex`` idiom). Because the built frame is held as
        the agent's ``pending_frame`` and the SAME frame instance is re-published on a
        retry, the id is RETAINED across retries automatically — an at-least-once
        redelivery of this edge carries the SAME id, so a consumer renders it once.
        """

        transition: Literal["offline", "recovered"] = "recovered" if alive else "offline"
        return PresenceTransitionFrame(
            agent_id=agent_id,
            display_name=safe_display_name(agent_id),
            transition=transition,
            observed_at=self._clock(),
            transition_id=uuid4().hex,
        )

    async def _deliver(self, frame: PresenceTransitionFrame) -> bool:
        """Publish one derived frame; return True on success, False on contained error.

        Logs the transition kind + already-safe display_name ONLY — NEVER the raw
        agent_id, the raw payload, or a subject that could carry credentials (the raw
        agent_id is never emitted on any path, including successful delivery). A publish
        failure is contained (returns False) so the caller retains the pending edge for
        retry — the edge is never swallowed-and-advanced.
        """

        try:
            await self._publish(frame)
        except Exception as exc:
            logger.error(
                "webchat.presence.transition | publish failed (edge retained) "
                "transition={} name={} error_type={}",
                frame.transition,
                frame.display_name,
                type(exc).__name__,
            )
            return False
        logger.info(
            "webchat.presence.transition | emitted transition={} name={}",
            frame.transition,
            frame.display_name,
        )
        return True

    def _prune_absent(self, snapshot_ids: set[str]) -> None:
        """Drop tracked identities absent from the snapshot with no pending edge.

        ``snapshot_ids`` is the COMPLETE authoritative identity set for this tick (not
        a processed-so-far subset), so an identity that IS present later in the
        snapshot is never treated as absent. An absent identity that still owes a
        pending edge is RETAINED (the edge is still owed); only absent identities with
        no pending are dropped. Runs under the evaluate lock.
        """

        for agent_id in [aid for aid in self._agents if aid not in snapshot_ids]:
            state = self._agents[agent_id]
            if state.pending is None:
                del self._agents[agent_id]

    def _evict_one_absent(self, snapshot_ids: set[str]) -> bool:
        """Evict the OLDEST identity absent-from-the-complete-snapshot, no pending edge.

        ``snapshot_ids`` is the COMPLETE authoritative identity set for this tick,
        precomputed before processing — so an identity that is present-but-not-yet-
        processed later in the snapshot is NEVER evicted to make room for a lexically-
        early newcomer. Deterministic (insertion-ordered), never random. An identity
        in the complete snapshot or owing a pending edge is never evicted. Returns True
        when one was evicted (room made for a new identity), False when none is
        evictable. Runs under the evaluate lock.
        """

        for agent_id, state in self._agents.items():
            if agent_id not in snapshot_ids and state.pending is None:
                del self._agents[agent_id]
                return True
        return False


__all__ = [
    "MAX_TRACKED_IDENTITIES",
    "PresenceTransitionDeriver",
    "PublishTransition",
    "TransitionClock",
    "safe_display_name",
]
