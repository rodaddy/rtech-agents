"""Wire the Hermes bridge to the two live buses and run the translation loop.

The bridge fronts one or more existing Hermes gateway agents on the fleet bus so
each appears in the rtech web-chat picker and answers in its own voice, without
deploying agentd per box.

Two bus namespaces (see :mod:`config`):

* **rtech side** — typed ``rtech_frames`` frames as raw JSON bytes on ``rtech.*``.
  The bridge subscribes ``rtech.agent.<fleet_id>.task`` (inbound tasks), publishes
  ``rtech.task.<task_id>.turn`` (the terminal reply) and
  ``rtech.task.<task_id>.status`` (typed in-flight state), and heartbeats each
  fleet id on ``rtech.agent.<fleet_id>.heartbeat`` so the picker lists it.
* **fleet side** — ``fleet_nats.Envelope`` on ``{env}.agent.*``. The bridge
  publishes a ``dispatch`` envelope on ``{env}.agent.dispatch.<fleet_id>`` and
  subscribes the reply lane ``{env}.agent.status.<task_id>``.

Reply protocol (matches the gateway's ``fleet_subscriber.send()`` exactly): the
status lane carries zero or more ``kind="status"`` envelopes (stance-only
structural signals and/or visible reply text), then one ``kind="done"`` envelope
carrying the typed ``fleet_core.spec.Done``. The bridge is CONTENT-ONLY on the
rtech side (docs/bridged-participant-lifecycle.md): each reply
publishes an immediate content ``TurnFrame`` (typed passthrough of text +
gateway stance) plus a ``StatusFrame(state="working")`` for liveness. The typed
``Done`` is consumed here for LIFECYCLE ONLY — unsubscribe, drop state, advance
the queue — and never becomes a frame; terminal semantics for the bridged
participant belong to the orchestrator, which consumes the same typed ``Done``
on its own fleet-side subscription and derives done in one place.

Task lifecycle (bounded, serialized):

* One in-flight task per fleet id at a time. The gateway's fleet-subscriber keys
  its reply correlation on the channel and OVERWRITES it per dispatch, so
  overlapping dispatches to the same agent/channel would cross-route replies.
  Later tasks queue FIFO per fleet id and dispatch when the current one ends;
  the queue is BOUNDED (``max_queue_depth``) and sheds loudly (a typed
  ``StatusFrame(state="blocked")``) when a slow agent backs it up.
* The reply-lane subscription is established BEFORE the dispatch publish; if it
  cannot be established the dispatch is rolled back (no publish without a
  consumer — a completed reply must never be black-holed).
* Every task's reply subscription is unsubscribed and its state dropped when the
  task completes, fails, or exceeds ``task_timeout_seconds`` (a sweeper expires
  stale tasks loudly, publishing ``StatusFrame(state="blocked")``). Finalization
  is claimed atomically under the lock, so a completion racing the sweeper can
  never double-finalize or emit a false blocked status. Finished task ids are
  remembered in a bounded window for idempotent redelivery.

Every external call (connect, publish, subscribe) is guarded so one failure never
kills the loop. Secret env VALUES are never logged; bus URLs are host:port only.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from fleet_nats import BusConfig, Envelope, FleetBus
from loguru import logger
from rtech_frames import StatusFrame, TaskFrame
from rtech_frames.codec import FrameDecodeError, decode_frame, encode_frame
from rtech_orchestrator import subjects as rtech_subjects

from ._rtech_frame_bus import RtechFrameBus
from .config import BridgeConfig
from .heartbeat import BridgeHeartbeat
from .translate import (
    ReplyCompletion,
    ReplyText,
    build_content_turn,
    build_inbound_envelope,
    classify_reply,
    reply_status_subject,
)

if TYPE_CHECKING:
    from nats.aio.msg import Msg
    from nats.aio.subscription import Subscription

# The rtech-side bus participant name/id for this bridge connection. Distinct
# from the fronted fleet ids — this is the bridge process's own identity on the
# rtech bus, used only for the connection (heartbeats are stamped per fleet id).
_RTECH_BUS_AGENT_ID = "hermes-bridge"
_RTECH_BUS_ENV = "dev"

# How many finished task ids to remember for idempotent redelivery. Bounded so
# sustained traffic cannot grow bridge memory without limit; the window is far
# wider than any realistic redelivery horizon.
_COMPLETED_MEMORY = 4096

# Expiry sweep cadence bounds (seconds). The sweep is cheap (a dict scan).
_SWEEP_MIN_S = 1.0
_SWEEP_MAX_S = 30.0


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_id() -> str:
    return uuid4().hex


@dataclass
class _PendingTask:
    """The bridge's per-in-flight-task state (lifecycle handles + counters)."""

    fleet_id: str
    subscription: Subscription | None
    deadline: float
    published_turns: int = 0


class HermesBridge:
    """Bridge rtech task frames to Hermes gateway agents on the fleet bus."""

    def __init__(
        self,
        *,
        config: BridgeConfig,
        rtech_bus: FleetBus,
        fleet_bus: FleetBus,
    ) -> None:
        """Wire the bridge to two already-constructed (not yet connected) buses.

        Args:
            config: The validated bridge config (fleet ids, urls, channel, flags).
            rtech_bus: The bus carrying ``rtech.*`` traffic. Task/turn frames ride
                it as raw bytes via :class:`RtechFrameBus`; heartbeats ride it as
                fleet envelopes via ``FleetBus.publish`` (the picker reads the
                heartbeat envelope's ``sender``).
            fleet_bus: The bus carrying ``{env}.agent.*`` fleet envelopes.
        """
        self._config = config
        self._rtech_bus = rtech_bus
        self._rtech_frames = RtechFrameBus(rtech_bus)
        self._fleet_bus = fleet_bus
        # Correlation + lifecycle state (bounded — see module docstring):
        #   task_id -> in-flight state (reply buffer, subscription, deadline).
        self._pending: dict[str, _PendingTask] = {}
        #   fleet_id -> the task id currently dispatched to it (serialization).
        self._active: dict[str, str] = {}
        #   fleet_id -> tasks waiting for the active one to finish (FIFO).
        self._queues: dict[str, deque[TaskFrame]] = {}
        #   Finished task ids (bounded window) for idempotent redelivery.
        self._completed: OrderedDict[str, None] = OrderedDict()
        self._lock = asyncio.Lock()
        self._heartbeats: list[BridgeHeartbeat] = []
        self._heartbeat_tasks: list[asyncio.Task[None]] = []
        self._sweeper_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Subscribe inbound task lanes, start heartbeats and the expiry sweeper."""
        logger.info(
            "hermes_bridge.start | fleet_ids={} env={} task_timeout_s={}",
            self._config.fleet_ids,
            self._config.fleet_env,
            self._config.task_timeout_seconds,
        )
        for fleet_id in self._config.fleet_ids:
            await self._subscribe_task_lane(fleet_id)
            self._start_heartbeat(fleet_id)
        self._sweeper_task = asyncio.create_task(self._run_sweeper(), name="bridge-expiry-sweep")
        logger.info("hermes_bridge.start | ready | fronting={} agents", len(self._config.fleet_ids))

    async def stop(self) -> None:
        """Stop heartbeats and the sweeper. Bus connections are closed by the caller."""
        for hb in self._heartbeats:
            hb.stop()
        tasks = list(self._heartbeat_tasks)
        if self._sweeper_task is not None:
            tasks.append(self._sweeper_task)
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error("hermes_bridge.stop | task join failed reason={}", exc)
        logger.info("hermes_bridge.stop | stopped")

    # --- inbound: rtech TaskFrame -> Hermes dispatch --------------------------

    async def _subscribe_task_lane(self, fleet_id: str) -> None:
        """Subscribe this fleet id's rtech task lane to the dispatch handler."""
        subject = rtech_subjects.agent_task(fleet_id)

        async def _on_message(msg: Msg) -> None:
            await self._on_task_bytes(fleet_id, msg.data)

        try:
            await self._rtech_frames.raw_subscribe(subject, _on_message)
        except Exception as exc:
            logger.error(
                "hermes_bridge.subscribe | task lane failed | fleet_id={} subject={} reason={}",
                fleet_id,
                subject,
                exc,
            )
            raise
        logger.info(
            "hermes_bridge.subscribe | task lane | fleet_id={} subject={}", fleet_id, subject
        )

    async def _on_task_bytes(self, fleet_id: str, data: bytes) -> None:
        """Decode an inbound rtech frame and enqueue it for dispatch."""
        try:
            frame = decode_frame(data)
        except FrameDecodeError as exc:
            logger.error(
                "hermes_bridge.inbound | undecodable frame | fleet_id={} reason={}", fleet_id, exc
            )
            return
        if not isinstance(frame, TaskFrame):
            logger.warning(
                "hermes_bridge.inbound | non-task frame on task lane | fleet_id={} type={}",
                fleet_id,
                type(frame).__name__,
            )
            return
        await self._enqueue_task(fleet_id, frame)

    async def _enqueue_task(self, fleet_id: str, task: TaskFrame) -> None:
        """Admit a task once, serialized per fleet id (one in flight at a time).

        The gateway's fleet-subscriber stores ONE channel->task correlation and
        overwrites it per dispatch, so overlapping dispatches to the same
        agent/channel would cross-route replies. Later tasks wait in a BOUNDED
        FIFO; past ``max_queue_depth`` the task is shed loudly with a typed
        blocked status (round-2 H6 — no unbounded admission).
        """
        admission: Literal["dispatch", "queued", "shed"]
        async with self._lock:
            if task.id in self._pending or task.id in self._completed:
                logger.info(
                    "hermes_bridge.dispatch | idempotent skip (already seen) | task_id={}",
                    task.id,
                )
                return
            queue = self._queues.setdefault(fleet_id, deque())
            if fleet_id in self._active:
                if any(queued.id == task.id for queued in queue):
                    logger.info(
                        "hermes_bridge.dispatch | idempotent skip (already queued) | task_id={}",
                        task.id,
                    )
                    return
                if len(queue) >= self._config.max_queue_depth:
                    logger.error(
                        "hermes_bridge.dispatch | queue full — task shed | "
                        "task_id={} fleet_id={} depth={}",
                        task.id,
                        fleet_id,
                        len(queue),
                    )
                    admission = "shed"
                else:
                    queue.append(task)
                    logger.info(
                        "hermes_bridge.dispatch | queued behind active task | "
                        "task_id={} fleet_id={} active={} depth={}",
                        task.id,
                        fleet_id,
                        self._active[fleet_id],
                        len(queue),
                    )
                    admission = "queued"
            else:
                self._active[fleet_id] = task.id
                admission = "dispatch"
        if admission == "shed":
            # The typed blocked status publishes outside the lock.
            await self._publish_status(task.id, fleet_id, "blocked")
            return
        if admission == "queued":
            return
        await self._dispatch_task(fleet_id, task)

    async def _dispatch_task(self, fleet_id: str, task: TaskFrame) -> None:
        """Subscribe the reply lane, then publish the dispatch (in that order).

        The subscription MUST exist before the publish: the gateway can complete
        real work fast, and a reply with no consumer is permanently lost. A
        subscription failure therefore rolls the dispatch back (no publish) so a
        redelivered task can retry; a publish failure unsubscribes and rolls back.
        """
        subject = reply_status_subject(self._config.fleet_env, task.id)

        async def _on_reply(envelope: Envelope) -> None:
            await self._on_reply_envelope(fleet_id, task.id, envelope)

        try:
            subscription = await self._fleet_bus.subscribe(subject, _on_reply)
        except Exception as exc:
            logger.error(
                "hermes_bridge.dispatch | reply subscribe failed — dispatch aborted | "
                "task_id={} subject={} reason={}",
                task.id,
                subject,
                exc,
            )
            await self._rollback_dispatch(fleet_id, task.id)
            return
        deadline = asyncio.get_running_loop().time() + self._config.task_timeout_seconds
        async with self._lock:
            self._pending[task.id] = _PendingTask(
                fleet_id=fleet_id, subscription=subscription, deadline=deadline
            )
        built = build_inbound_envelope(
            task=task,
            fleet_id=fleet_id,
            fleet_env=self._config.fleet_env,
            default_channel=self._config.default_channel,
            msg_id=_new_id(),
            ts=_now_iso(),
        )
        try:
            await self._fleet_bus.publish(built.subject, built.envelope)
        except Exception as exc:
            logger.error(
                "hermes_bridge.dispatch | publish failed | task_id={} subject={} reason={}",
                task.id,
                built.subject,
                exc,
            )
            await self._unsubscribe(task.id, subscription)
            async with self._lock:
                self._pending.pop(task.id, None)
            await self._rollback_dispatch(fleet_id, task.id)
            return
        logger.info(
            "hermes_bridge.dispatch | dispatched | task_id={} fleet_id={} subject={}",
            task.id,
            fleet_id,
            built.subject,
        )

    async def _rollback_dispatch(self, fleet_id: str, task_id: str) -> None:
        """Release a failed dispatch so redelivery can retry, then advance the queue."""
        async with self._lock:
            if self._active.get(fleet_id) == task_id:
                self._active.pop(fleet_id, None)
        await self._advance_queue(fleet_id)

    async def _advance_queue(self, fleet_id: str) -> None:
        """Dispatch the next queued task for a fleet id, if any."""
        async with self._lock:
            if fleet_id in self._active:
                return
            queue = self._queues.get(fleet_id)
            if not queue:
                return
            task = queue.popleft()
            self._active[fleet_id] = task.id
        logger.info(
            "hermes_bridge.dispatch | dequeued next task | task_id={} fleet_id={}",
            task.id,
            fleet_id,
        )
        await self._dispatch_task(fleet_id, task)

    # --- outbound: Hermes reply -> rtech frames --------------------------------

    async def _on_reply_envelope(self, fleet_id: str, task_id: str, envelope: Envelope) -> None:
        """Publish reply content immediately; consume typed completion as lifecycle."""
        result = classify_reply(envelope=envelope, fleet_id=fleet_id, task_id=task_id)
        if result is None:
            return
        async with self._lock:
            pending = self._pending.get(task_id)
        if pending is None:
            logger.warning(
                "hermes_bridge.reply | reply for unknown/finished task — dropped | task_id={}",
                task_id,
            )
            return
        if isinstance(result, ReplyText):
            await self._publish_content_turn(task_id, pending, result)
            return
        await self._on_completion(task_id, result)

    async def _publish_content_turn(
        self, task_id: str, pending: _PendingTask, reply: ReplyText
    ) -> None:
        """Publish one reply as a content ``TurnFrame`` plus a working status.

        Content only: the turn's stance is the gateway's typed stance passed
        through verbatim; terminal semantics stay with the orchestrator's own
        typed-``Done`` consumption.
        """
        frame = build_content_turn(
            fleet_id=pending.fleet_id,
            task_id=task_id,
            text=reply.text,
            stance=reply.stance,
        )
        subject = rtech_subjects.task_turn(task_id)
        try:
            await self._rtech_frames.raw_publish(subject, encode_frame(frame))
        except Exception as exc:
            logger.error(
                "hermes_bridge.reply | content turn publish failed | "
                "task_id={} subject={} reason={}",
                task_id,
                subject,
                exc,
            )
            return
        pending.published_turns += 1
        logger.info(
            "hermes_bridge.reply | content turn published | task_id={} fleet_id={} count={}",
            task_id,
            pending.fleet_id,
            pending.published_turns,
        )
        await self._publish_status(task_id, pending.fleet_id, "working")

    async def _on_completion(self, task_id: str, completion: ReplyCompletion) -> None:
        """End the task's LOCAL lifecycle on the typed completion — never a frame.

        The orchestrator consumes the same typed ``Done`` on its own fleet-side
        subscription and records the participant's terminal evidence there; the
        bridge only releases its per-task resources. A completion that arrives
        with no content turn ever published is surfaced as a typed ``blocked``
        status so the failure is loud instead of a silently un-derivable task.
        """
        pending = await self._claim_finalize(task_id)
        if pending is None:
            logger.debug(
                "hermes_bridge.reply | completion for already-finalized task | task_id={}",
                task_id,
            )
            return
        if pending.published_turns == 0:
            logger.error(
                "hermes_bridge.reply | typed completion with no published content | "
                "task_id={} fleet_id={} outcome={}",
                task_id,
                pending.fleet_id,
                completion.outcome,
            )
            await self._publish_status(task_id, pending.fleet_id, "blocked")
        else:
            logger.info(
                "hermes_bridge.reply | typed completion consumed (lifecycle only) | "
                "task_id={} fleet_id={} outcome={}",
                task_id,
                pending.fleet_id,
                completion.outcome,
            )
        await self._release_task(task_id, pending)

    async def _claim_finalize(self, task_id: str) -> _PendingTask | None:
        """Atomically claim a task for finalization; ``None`` if already claimed.

        Completion and the expiry sweeper can race; the single pop under the
        lock guarantees exactly one of them finalizes (and only the claimer may
        publish a blocked status), fixing the round-2 snapshot-then-act gap.
        """
        async with self._lock:
            pending = self._pending.pop(task_id, None)
            if pending is None:
                return None
            if self._active.get(pending.fleet_id) == task_id:
                self._active.pop(pending.fleet_id, None)
            self._completed[task_id] = None
            while len(self._completed) > _COMPLETED_MEMORY:
                self._completed.popitem(last=False)
        return pending

    async def _release_task(self, task_id: str, pending: _PendingTask) -> None:
        """Unsubscribe the claimed task's reply lane and advance the agent's queue."""
        await self._unsubscribe(task_id, pending.subscription)
        logger.info(
            "hermes_bridge.lifecycle | task finished | task_id={} fleet_id={}",
            task_id,
            pending.fleet_id,
        )
        await self._advance_queue(pending.fleet_id)

    async def _unsubscribe(self, task_id: str, subscription: Subscription | None) -> None:
        """Drop a task's reply-lane subscription; a failure is logged, never fatal."""
        if subscription is None:
            return
        try:
            await subscription.unsubscribe()
        except Exception as exc:
            logger.error(
                "hermes_bridge.lifecycle | unsubscribe failed | task_id={} reason={}",
                task_id,
                exc,
            )

    async def _publish_status(
        self, task_id: str, fleet_id: str, state: Literal["working", "input_required", "blocked"]
    ) -> None:
        """Publish a typed StatusFrame on the rtech status lane; guarded.

        ``fleet_id`` is the bridged participant the status speaks for — the
        same identity the content turns carry — so per-participant done
        derivation classifies the status against ITS author.
        """
        frame = StatusFrame(task_id=task_id, agent_id=fleet_id, state=state)
        subject = rtech_subjects.task_status(task_id)
        try:
            await self._rtech_frames.raw_publish(subject, encode_frame(frame))
        except Exception as exc:
            logger.error(
                "hermes_bridge.status | publish failed | task_id={} state={} reason={}",
                task_id,
                state,
                exc,
            )
            return
        logger.debug("hermes_bridge.status | published | task_id={} state={}", task_id, state)

    # --- expiry ----------------------------------------------------------------

    async def _run_sweeper(self) -> None:
        """Periodically expire in-flight tasks past their deadline."""
        interval = min(_SWEEP_MAX_S, max(_SWEEP_MIN_S, self._config.task_timeout_seconds / 10))
        while True:
            await asyncio.sleep(interval)
            try:
                await self._sweep_expired()
            except Exception as exc:
                logger.error("hermes_bridge.sweep | sweep failed reason={}", exc)

    async def _sweep_expired(self) -> None:
        """Expire every pending task whose deadline has passed, loudly.

        Each expiry re-claims through :meth:`_claim_finalize`, so a completion
        landing between the snapshot and the sweep wins the claim and the
        sweeper neither double-finalizes nor emits a false blocked status.
        """
        now = asyncio.get_running_loop().time()
        async with self._lock:
            expired_ids = [
                task_id for task_id, pending in self._pending.items() if pending.deadline <= now
            ]
        for task_id in expired_ids:
            pending = await self._claim_finalize(task_id)
            if pending is None:
                continue
            logger.error(
                "hermes_bridge.sweep | task expired after {}s without typed completion | "
                "task_id={} fleet_id={} published_turns={}",
                self._config.task_timeout_seconds,
                task_id,
                pending.fleet_id,
                pending.published_turns,
            )
            await self._publish_status(task_id, pending.fleet_id, "blocked")
            await self._release_task(task_id, pending)

    # --- heartbeat ------------------------------------------------------------

    def _start_heartbeat(self, fleet_id: str) -> None:
        """Start a liveness heartbeat for one fronted fleet id on the rtech bus."""
        hb = BridgeHeartbeat(
            bus=self._rtech_bus,
            fleet_id=fleet_id,
            interval_s=self._config.heartbeat_interval_seconds,
        )
        self._heartbeats.append(hb)
        self._heartbeat_tasks.append(asyncio.create_task(hb.run(), name=f"bridge-hb-{fleet_id}"))
        logger.info("hermes_bridge.heartbeat | started | fleet_id={}", fleet_id)


def build_buses(config: BridgeConfig) -> tuple[FleetBus, FleetBus]:
    """Construct (not connect) the rtech and fleet buses from config.

    Args:
        config: The validated bridge config.

    Returns:
        A ``(rtech_bus, fleet_bus)`` tuple. When both urls are the same host, two
        distinct connections are still opened (one per namespace) — the subject
        namespaces differ, and a single participant identity per connection keeps
        the fleet-side ``from`` stamping unambiguous.
    """
    rtech_bus = FleetBus(
        BusConfig(
            url=config.rtech_nats_url,
            agent_id=_RTECH_BUS_AGENT_ID,
            env=_RTECH_BUS_ENV,
            name="rtech-hermes-bridge-rtech",
        )
    )
    fleet_bus = FleetBus(
        BusConfig(
            url=config.fleet_nats_url,
            agent_id=_RTECH_BUS_AGENT_ID,
            env=config.fleet_env,
            name="rtech-hermes-bridge-fleet",
        )
    )
    return rtech_bus, fleet_bus
