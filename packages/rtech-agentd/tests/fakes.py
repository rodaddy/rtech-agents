"""Test doubles: a scripted FakeEngine and an in-process FleetBus-publish stub.

FakeEngine implements the ``EngineSession`` protocol with an internal event queue
you push events onto — so tests drive turn/steer/interrupt ordering
deterministically without the Claude SDK. It records every ``send`` (with a flag
for whether a turn was active = a steer) so a test can assert a steer landed
INSIDE an active turn.

The publisher stub records ``(subject, envelope)`` publishes for assertions.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from fleet_nats import Envelope
from rtech_agentd.engine import (
    EngineEvent,
    EngineHealth,
    ToolsPolicy,
    TurnDone,
)
from rtech_agentd.frame_turn import FrameHandler
from rtech_frames.codec import Frame


@dataclass
class SendRecord:
    """One recorded ``send`` on the fake engine."""

    text: str
    during_active_turn: bool


class FakeEngine:
    """A scripted ``EngineSession`` for supervisor/ingress/steering tests."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[EngineEvent | None] = asyncio.Queue()
        self.sends: list[SendRecord] = []
        self.interrupts: int = 0
        self.stream_failures: int = 0
        self.started: bool = False
        self.start_count: int = 0
        self.resumed_ref: str | None = None
        self.resume_refs: list[str] = []
        self.closed: bool = False
        self._active_turn: bool = False
        self._session_ref: str | None = "fake-session-1"

    # --- scripting API (test-only) -------------------------------------------

    async def push(self, event: EngineEvent) -> None:
        """Enqueue an engine event for ``events()`` to yield."""
        if isinstance(event, TurnDone):
            self._active_turn = False
            if event.session_ref:
                self._session_ref = event.session_ref
        await self._queue.put(event)

    def mark_turn_active(self) -> None:
        """Simulate a turn being in-flight (so a following send counts as steer)."""
        self._active_turn = True

    async def end_stream(self) -> None:
        """Signal ``events()`` to stop (end the async iterator)."""
        await self._queue.put(None)

    # --- EngineSession protocol ----------------------------------------------

    async def start(self, system_prompt: str, tools_policy: ToolsPolicy) -> None:
        self.started = True
        self.start_count += 1

    async def resume(self, session_ref: str) -> None:
        self.resumed_ref = session_ref
        self.resume_refs.append(session_ref)
        self.started = True
        self._session_ref = session_ref

    async def send(self, text: str) -> None:
        self.sends.append(SendRecord(text=text, during_active_turn=self._active_turn))
        self._active_turn = True

    async def interrupt(self) -> None:
        self.interrupts += 1
        self._active_turn = False

    async def events(self) -> AsyncIterator[EngineEvent]:
        if self.stream_failures > 0:
            self.stream_failures -= 1
            raise RuntimeError("simulated engine stream crash")
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    async def health(self) -> EngineHealth:
        return EngineHealth(
            connected=not self.closed,
            session_ref=self._session_ref,
            active_turn=self._active_turn,
        )

    async def close(self) -> None:
        self.closed = True
        await self._queue.put(None)


@dataclass
class RecordingPublisher:
    """Records every publish for assertions; satisfies the ``Publisher`` seam."""

    published: list[tuple[str, Envelope]] = field(default_factory=list)

    async def __call__(self, subject: str, envelope: Envelope) -> None:
        self.published.append((subject, envelope))

    def by_kind(self, kind: str) -> list[tuple[str, Envelope]]:
        return [(s, e) for s, e in self.published if e.kind == kind]


class FakeFrameBus:
    """A typed-frame publish/subscribe double for FrameTurnRunner tests.

    Records every publish as ``(subject, frame)`` and stores EVERY handler
    subscribed to a subject — live NATS delivers to all subscriptions on a
    subject, and collapsing them to one masked duplicate-callback bugs (#76).
    :meth:`subscribe_frame` returns an unsubscribe handle mirroring the live
    bus; :meth:`deliver` invokes all still-subscribed handlers.
    """

    def __init__(self) -> None:
        self.published: list[tuple[str, Frame]] = []
        self.subscribed: dict[str, list[FrameHandler]] = {}

    async def publish_frame(self, subject: str, frame: Frame) -> None:
        """Record a published ``(subject, frame)`` pair."""
        self.published.append((subject, frame))

    async def publish_frame_checked(self, subject: str, frame: Frame) -> None:
        """Record a published ``(subject, frame)`` pair, propagating on failure.

        The checked twin of :meth:`publish_frame` (issue #93). The base fake never
        raises, so this behaves identically; a test that needs a raising checked
        publish overrides this method (as the wake-service publish-failure test
        does) so the wake tool's typed ``publish_failed`` path is exercised.
        """
        self.published.append((subject, frame))

    async def subscribe_frame(
        self, subject: str, handler: FrameHandler
    ) -> Callable[[], Awaitable[None]]:
        """Register a handler for a subject and return its unsubscribe handle."""
        self.subscribed.setdefault(subject, []).append(handler)

        async def _unsubscribe() -> None:
            handlers = self.subscribed.get(subject, [])
            if handler in handlers:
                handlers.remove(handler)
            if not handlers:
                self.subscribed.pop(subject, None)

        return _unsubscribe

    def handlers_on(self, subject: str) -> list[FrameHandler]:
        """Return the live handlers registered for a subject (empty if none)."""
        return list(self.subscribed.get(subject, []))

    async def deliver(self, subject: str, frame: Frame) -> None:
        """Invoke every handler registered for ``subject`` with ``frame``.

        Raises:
            KeyError: If no handler is subscribed — a test delivering to a
                retired lane should fail loudly, mirroring "nothing listens".
        """
        handlers = self.subscribed[subject]
        for handler in list(handlers):
            await handler(frame)

    def frames_on(self, subject: str) -> list[Frame]:
        """Return frames published on a subject, in publish order."""
        return [f for s, f in self.published if s == subject]

    def by_type(self, frame_type: type[Frame]) -> list[tuple[str, Frame]]:
        """Return ``(subject, frame)`` pairs whose frame is of ``frame_type``."""
        return [(s, f) for s, f in self.published if isinstance(f, frame_type)]
