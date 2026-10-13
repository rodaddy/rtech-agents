"""Replay per-task web-chat edge frames for late stream attach."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from loguru import logger
from rtech_frames.codec import decode_frame, encode_frame
from rtech_frames.frames import StatusFrame, TurnFrame

from . import subjects
from .bus import FrameBus, FrameHandler, Subscription
from .registry import RenderedPayload, TaskRegistry

ReplayEventKind = Literal["frame", "result"]


@dataclass(frozen=True)
class ReplayEvent:
    """Carry one recorded stream event."""

    kind: ReplayEventKind
    payload: RenderedPayload
    data: bytes | None = None


@dataclass(frozen=True)
class ReplayAttachment:
    """Carry initial replay events and an optional live-tail queue."""

    snapshot: Sequence[ReplayEvent]
    queue: asyncio.Queue[ReplayEvent] | None


@dataclass
class TaskReplayBuffer:
    """Hold a bounded event buffer and live listeners for one task."""

    task_id: str
    capacity: int
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    events: deque[ReplayEvent] = field(default_factory=deque)
    listeners: set[asyncio.Queue[ReplayEvent]] = field(default_factory=set)
    result_received: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self.events = deque(maxlen=self.capacity)

    async def append(self, event: ReplayEvent) -> None:
        """Append an event and fan it out to live listeners."""

        async with self.lock:
            self.events.append(event)
            self.updated_at = time.monotonic()
            if event.kind == "result":
                self.result_received = True
            for queue in self.listeners:
                queue.put_nowait(event)

    async def attach(self) -> ReplayAttachment:
        """Return a replay snapshot and, when incomplete, a live-tail queue."""

        async with self.lock:
            snapshot = tuple(self.events)
            if self.result_received:
                return ReplayAttachment(snapshot=snapshot, queue=None)
            queue: asyncio.Queue[ReplayEvent] = asyncio.Queue()
            self.listeners.add(queue)
            return ReplayAttachment(snapshot=snapshot, queue=queue)

    async def detach(self, queue: asyncio.Queue[ReplayEvent] | None) -> None:
        """Remove a live-tail queue from this buffer."""

        if queue is None:
            return
        async with self.lock:
            self.listeners.discard(queue)

    def expired(self, now: float, ttl_seconds: float) -> bool:
        """Return whether the buffer has aged past its retention TTL."""

        return now - self.updated_at > ttl_seconds


@dataclass
class ReplayHub:
    """Manage per-task replay subscriptions and bounded buffers.

    Two subscription lifetimes are tracked separately because they answer to
    different truths. ``subscriptions`` holds the per-task turn/status interest
    that feeds the *evictable* replay buffer: it exists only to bound SSE replay
    memory and is torn down whenever the buffer is evicted for TTL/LRU. The
    result subscription lives in ``result_subscriptions`` and is *not* evictable
    — it is the connector's mirror of orchestrator-derived completion, so it
    must outlive buffer eviction. If it were torn down on eviction, a later
    core-NATS result would be missed and the completed dash-dash task would stay
    authoritatively active forever (issue #144). It is retained until the result
    actually arrives (its natural terminal) or the bus connection drops.
    """

    bus: FrameBus
    registry: TaskRegistry
    frame_limit: int
    ttl_seconds: int
    max_tasks: int
    buffers: dict[str, TaskReplayBuffer] = field(default_factory=dict)
    subscriptions: dict[str, list[Subscription]] = field(default_factory=dict)
    result_subscriptions: dict[str, Subscription] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def start_task(self, task_id: str) -> None:
        """Start the POST-time task subscriptions used by replay attach.

        The result subscription is created at most once per task and survives
        buffer eviction; the turn/status subscriptions and buffer are recreated
        on re-attach after eviction. Both creations are idempotent so a re-
        attach after eviction never double-subscribes.
        """

        await self.sweep()
        await self._ensure_result_subscription(task_id)
        async with self.lock:
            if task_id in self.buffers:
                return
            self.buffers[task_id] = TaskReplayBuffer(task_id=task_id, capacity=self.frame_limit)
        try:
            subscriptions = [
                await self.bus.subscribe(
                    subjects.task_turn(task_id),
                    self._build_frame_handler(task_id),
                ),
                await self.bus.subscribe(
                    subjects.task_status(task_id),
                    self._build_frame_handler(task_id),
                ),
            ]
        except Exception as exc:
            logger.error("webchat.replay.start | failed task_id={} reason={}", task_id, exc)
            async with self.lock:
                self.buffers.pop(task_id, None)
            raise
        async with self.lock:
            self.subscriptions[task_id] = subscriptions
        logger.info("webchat.replay.start | subscribed task_id={}", task_id)
        await self.sweep()

    async def _ensure_result_subscription(self, task_id: str) -> None:
        """Subscribe the eviction-independent result interest exactly once.

        The result subject feeds registry completion truth. If the task's
        result already arrived, no interest is (re)created — the terminal is
        past. The subscription is stored under the lock only after a successful
        subscribe, and a lost race is unsubscribed rather than leaked.
        """

        async with self.lock:
            if task_id in self.result_subscriptions:
                return
            if self.registry.result_received(task_id):
                return
        subscription = await self.bus.subscribe(
            subjects.task_result(task_id),
            self._build_result_handler(task_id),
        )
        async with self.lock:
            if task_id in self.result_subscriptions:
                await subscription.unsubscribe()
                return
            self.result_subscriptions[task_id] = subscription
        logger.debug("webchat.replay.result | subscribed task_id={}", task_id)

    async def attach(self, task_id: str) -> ReplayAttachment:
        """Attach a stream to replayed events, then live-tail if incomplete."""

        await self.sweep()
        async with self.lock:
            buffer = self.buffers.get(task_id)
        if buffer is None:
            if self.registry.result_received(task_id):
                return ReplayAttachment(snapshot=(), queue=None)
            await self.start_task(task_id)
            async with self.lock:
                buffer = self.buffers[task_id]
        return await buffer.attach()

    async def detach(self, task_id: str, queue: asyncio.Queue[ReplayEvent] | None) -> None:
        """Detach a live-tail stream from a task buffer."""

        async with self.lock:
            buffer = self.buffers.get(task_id)
        if buffer is not None:
            await buffer.detach(queue)

    async def sweep(self) -> None:
        """Evict expired and excess task buffers."""

        now = time.monotonic()
        async with self.lock:
            expired = [
                task_id
                for task_id, buffer in self.buffers.items()
                if buffer.expired(now, self.ttl_seconds)
            ]
            excess_count = max(0, len(self.buffers) - len(expired) - self.max_tasks)
            retained = [
                (task_id, buffer)
                for task_id, buffer in self.buffers.items()
                if task_id not in expired
            ]
            retained.sort(key=lambda item: item[1].updated_at)
            evicted = [task_id for task_id, _ in retained[:excess_count]]
        for task_id in [*expired, *evicted]:
            await self.evict(task_id, reason="ttl" if task_id in expired else "max_tasks")

    async def evict(self, task_id: str, *, reason: str) -> None:
        """Evict one replay buffer and cancel its turn/status subscriptions.

        The result subscription is deliberately NOT cancelled here: buffer
        eviction bounds SSE replay memory, but the result interest is the
        connector's completion mirror and must survive so a later core-NATS
        result is still observed and the dash-dash active record is truthfully
        cleared (issue #144). It is torn down only when the result arrives.
        """

        async with self.lock:
            buffer = self.buffers.pop(task_id, None)
            subscriptions = self.subscriptions.pop(task_id, [])
        if buffer is None:
            return
        for subscription in subscriptions:
            await subscription.unsubscribe()
        logger.debug("webchat.replay.evict | task_id={} reason={}", task_id, reason)

    def _build_frame_handler(self, task_id: str) -> FrameHandler:
        async def on_frame(data: bytes) -> None:
            frame = decode_frame(data)
            if not isinstance(frame, TurnFrame | StatusFrame):
                logger.warning(
                    "webchat.replay.frame | ignored frame type={} task_id={}",
                    type(frame).__name__,
                    task_id,
                )
                return
            if frame.task_id != task_id:
                logger.warning(
                    "webchat.replay.frame | ignored mismatched task_id={} expected={}",
                    frame.task_id,
                    task_id,
                )
                return
            payload = self.registry.record_frame(frame)
            if payload is None:
                return
            await self._append(
                task_id,
                ReplayEvent(kind="frame", payload=payload, data=encode_frame(frame)),
            )

        return on_frame

    def _build_result_handler(self, task_id: str) -> FrameHandler:
        async def on_result(_: bytes) -> None:
            payload = self.registry.record_result_arrival(task_id)
            if payload is None:
                return
            await self._append(task_id, ReplayEvent(kind="result", payload=payload))
            # Completion is the result subscription's natural terminal: the task
            # is no longer active, so retaining the interest would only leak it.
            await self._teardown_result_subscription(task_id)

        return on_result

    async def _teardown_result_subscription(self, task_id: str) -> None:
        """Cancel and drop the eviction-independent result subscription."""

        async with self.lock:
            subscription = self.result_subscriptions.pop(task_id, None)
        if subscription is None:
            return
        await subscription.unsubscribe()
        logger.debug("webchat.replay.result | unsubscribed task_id={}", task_id)

    async def _append(self, task_id: str, event: ReplayEvent) -> None:
        async with self.lock:
            buffer = self.buffers.get(task_id)
        if buffer is None:
            logger.warning("webchat.replay.append | missing buffer task_id={}", task_id)
            return
        await buffer.append(event)
