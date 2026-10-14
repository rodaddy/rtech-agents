"""A deterministic in-process ``EngineSession`` with no external I/O.

Extracted from ``scripts/e2e_evidence.py`` so both the engine-selection helper
(:mod:`rtech_agentd.engine_select`) and the local dev stack can import the same
class.

Completion contract: every ``send`` opens a turn that COMPLETES on
its own with a terminal ``TurnDone`` — a task send never depends on a later steer
"flush" to finish. The first send is the task turn; a subsequent send (during a
now-idle session) is a steer turn whose ``TurnDone`` text reflects the steer.
Two completed turns is exactly the steer-lands / second-turn signal the e2e proof
already keys on, so the proof holds without any turn being held open.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from loguru import logger
from rtech_frames import Stance

from .engine import (
    EngineEvent,
    EngineHealth,
    TextDelta,
    ToolsPolicy,
    TurnDone,
)


class ScriptedEngine:
    """A deterministic ``EngineSession``: every send completes its own turn.

    A task send opens a turn (``TextDelta``) and completes it with a terminal
    ``TurnDone`` in the same call — no steer flush required. A later
    send is treated as a steer: it opens and completes a second turn whose
    ``TurnDone`` text reflects the steer, so the frame graph shows two completed
    turns (task then steer) without ever holding a turn open.
    """

    def __init__(self) -> None:
        """Create the scripted engine with an empty event queue."""
        self._queue: asyncio.Queue[EngineEvent | None] = asyncio.Queue()
        self._task_turn_done = False
        self._session_ref = "e2e-scripted-session"
        self._closed = False

    async def start(self, system_prompt: str, tools_policy: ToolsPolicy) -> None:
        """Open a fresh scripted session (no external I/O)."""
        logger.info("scripted.start | system_prompt_len={}", len(system_prompt))

    async def resume(self, session_ref: str) -> None:
        """Reopen a prior scripted session (no external I/O)."""
        logger.info("scripted.resume | ref={}", session_ref)

    async def send(self, text: str) -> None:
        """Open AND complete a turn: task turn first, steer turn thereafter.

        The turn always terminates in this same call with a ``TurnDone`` — the
        completion contract forbids holding a turn open waiting for a steer.
        """
        if not self._task_turn_done:
            self._task_turn_done = True
            logger.info("scripted.send | task turn opened+completed len={}", len(text))
            await self._queue.put(TextDelta(text="working on it..."))
            await self._queue.put(
                TurnDone(
                    text="working on it... done.",
                    session_ref=self._session_ref,
                    stance=Stance(position="responded", because="completed the task turn"),
                )
            )
            return
        logger.info("scripted.send | steer turn opened+completed len={}", len(text))
        await self._queue.put(TextDelta(text="adjusting..."))
        await self._queue.put(
            TurnDone(
                text=f"acknowledged steer: {text}",
                session_ref=self._session_ref,
                stance=Stance(position="responded", because="completed after steer"),
            )
        )

    async def interrupt(self) -> None:
        """Abort the active turn (no-op for the scripted engine)."""
        logger.info("scripted.interrupt | called")

    async def events(self) -> AsyncIterator[EngineEvent]:
        """Yield queued events until the sentinel ``None`` pulls the stream shut."""
        while True:
            event = await self._queue.get()
            if event is None:
                logger.info("scripted.events | stream ended")
                return
            yield event

    async def health(self) -> EngineHealth:
        """Return a point-in-time health snapshot.

        The scripted engine completes every turn within its ``send`` call, so a
        turn is never held open between calls; ``active_turn`` is always False.
        """
        return EngineHealth(
            connected=not self._closed,
            session_ref=self._session_ref,
            active_turn=False,
        )

    async def close(self) -> None:
        """Tear down: push the stream-ending sentinel."""
        if self._closed:
            return
        self._closed = True
        await self._queue.put(None)
        logger.info("scripted.close | closed")
