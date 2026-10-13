"""Rotation-during-debate overlap stress (issue #81 x #96, graduated from the
post-#84 integration harness S3b/S3c).

These are cross-slice integration tests, not unit tests: the REAL Orchestrator,
REAL Supervisor (with its REAL event pump ``sup.run()``), and REAL
FrameTurnRunner run in-process over a loopback typed-frame bus
(:class:`DeliveringBus`) that mirrors live NATS core — a publish delivers to
EVERY handler still subscribed to the subject, so sibling fanout is genuine. The
single-agent idle-only rotation guard is already pinned by
``test_lifecycle_policy.test_task_arriving_*_aborts_rotation_and_serves_task``;
what those do NOT exercise is a rotation that COMPLETES (session swap) while a
live multi-agent debate crosses the same turn boundary with real sibling fanout.
That overlap is what these two tests own.

The delivery is genuinely DECOUPLED, exactly as production requires (the engine
contract in ``frame_turn.py`` warns that a ``TurnDone`` delivered synchronously
inside ``session.send`` is the anti-pattern). :class:`QueueEngine` NEVER delivers
a ``TurnDone`` from inside ``send``: ``send`` only ENQUEUES engine events, and
the supervisor's own pump (``sup.run()`` → ``_pump`` → ``events()`` →
``_handle_event``) drains them on a later tick. Because the pump owns delivery,
the rotation-summary turn's ``TurnDone`` is intercepted by the supervisor's
``_rotation_pending`` path (supervisor.py ``_handle_event``) and NEVER fanned to
the runner sink — so ``_complete_rotation`` actually runs and no spurious summary
``TurnFrame`` is published (contrast the retired synchronous fake, which drove
``runner._on_engine_event`` from inside ``send`` and manufactured a duplicate
follow-up turn — see the test module's git history / the F3 finding).

The loopback bus and the queue engine are load-bearing scaffolding the committed
``fakes`` deliberately do not fully provide: ``FakeFrameBus`` requires an explicit
:meth:`deliver` and never loops a publish back to co-subscribers, so a debate
could not fan out through it; ``FakeEngine`` is a fine decoupled engine but has
no per-``send`` follow-up scripting, which the debate needs.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from rtech_agentd.engine import (
    EngineEvent,
    EngineHealth,
    TextDelta,
    ToolsPolicy,
    TurnDone,
)
from rtech_agentd.frame_turn import FrameTurnRunner
from rtech_agentd.lifecycle_policy import RotationPolicy
from rtech_agentd.registry import ActiveSessionRegistry
from rtech_agentd.supervisor import ROTATION_SUMMARY_DIRECTIVE, Supervisor, SupervisorConfig
from rtech_frames import Stance, TaskFrame, TurnFrame
from rtech_frames.codec import Frame
from rtech_orchestrator import Orchestrator, OrchestratorConfig, subjects

from .fakes import RecordingPublisher

FrameHandler = Callable[[Frame], Awaitable[None]]

_STACK = "stress-stack"

# The one sibling-triggered follow-up each agent posts when it SEES a sibling's
# non-terminal turn. It is TERMINAL (empty ``needs_from``) so the debate settles
# below the sibling-round cap: exactly one follow-up per agent, then consensus.
_FOLLOWUP_TEXT = "revised position"


class _Clock:
    """A hand-advanced monotonic clock so rotation fires DETERMINISTICALLY once.

    The rotation is AGE-triggered (not turn-triggered) so exactly ONE swap
    happens: the opener's ``TurnDone`` is delivered after the clock is advanced
    past ``max_age_s``, so ``_maybe_rotate`` fires; ``on_rotated`` then resets the
    session-start stamp to the (frozen) current time, so alice's later follow-up
    on the fresh session is well within the age bound and never re-rotates. A
    turn-count trigger (``max_turns=1``) would instead re-rotate on every
    post-swap turn — a real ``max_turns=1`` cascade, but not the single-overlap
    scenario these tests own.
    """

    def __init__(self) -> None:
        self._t = 0.0

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


class _Sub:
    """Retirable subscription handle for the loopback bus."""

    def __init__(self, unsub: Callable[[], Awaitable[None]]) -> None:
        self._unsub = unsub

    async def unsubscribe(self) -> None:
        await self._unsub()


class DeliveringBus:
    """In-process typed-frame bus that loops publishes back to subscribers.

    Mirrors live NATS core: a publish delivers to EVERY handler still subscribed
    to the subject — the fanout that makes an inter-agent debate real. Draining
    is queue-driven (not recursive) with a hard fuse so an unbounded fanout
    becomes a loud failure, not a hang. The fuse RAISING is the only overlap
    fuse assertion the tests need — a test that also asserted
    ``delivery_count <= max`` after the fact would be tautological, so it does
    not.
    """

    def __init__(self, max_deliveries: int = 500) -> None:
        self.published: list[tuple[str, Frame]] = []
        self.subscribed: dict[str, list[FrameHandler]] = {}
        self._pending: list[tuple[str, Frame]] = []
        self._draining = False
        self.delivery_count = 0
        self._max = max_deliveries

    async def publish_frame(self, subject: str, frame: Frame) -> None:
        self.published.append((subject, frame))
        self._pending.append((subject, frame))
        if self._draining:
            return
        self._draining = True
        try:
            while self._pending:
                subj, fr = self._pending.pop(0)
                self.delivery_count += 1
                if self.delivery_count > self._max:
                    raise AssertionError(
                        f"DeliveringBus fuse tripped: unbounded fanout (> {self._max})"
                    )
                for handler in list(self.subscribed.get(subj, [])):
                    await handler(fr)
        finally:
            self._draining = False

    async def publish_frame_checked(self, subject: str, frame: Frame) -> None:
        await self.publish_frame(subject, frame)

    async def subscribe_frame(
        self, subject: str, handler: FrameHandler
    ) -> Callable[[], Awaitable[None]]:
        self.subscribed.setdefault(subject, []).append(handler)

        async def _unsub() -> None:
            handlers = self.subscribed.get(subject, [])
            if handler in handlers:
                handlers.remove(handler)
            if not handlers:
                self.subscribed.pop(subject, None)

        return _unsub

    async def subscribe_frame_tracked(self, subject: str, handler: FrameHandler) -> _Sub:
        """Subscribe and return a retirable handle (the orchestrator's seam)."""
        unsub = await self.subscribe_frame(subject, handler)
        return _Sub(unsub)

    async def confirm_publishes(self) -> None:
        """Loopback delivers synchronously; nothing to flush."""
        return None

    def frames_on(self, subject: str) -> list[Frame]:
        return [f for s, f in self.published if s == subject]

    def by_type(self, frame_type: type[Frame]) -> list[tuple[str, Frame]]:
        return [(s, f) for s, f in self.published if isinstance(f, frame_type)]

    def turn_frames_by_agent(self, agent_id: str, task_id: str) -> list[TurnFrame]:
        """Return an agent's TurnFrames published on the TASK TURN LANE only.

        Scoped to ``subjects.task_turn(task_id)`` so the orchestrator's derived-
        done republish of the terminal turn onto the RESULT lane is never
        miscounted as a second agent turn — the mistake that would otherwise
        inflate the exact per-agent turn counts.
        """
        turn_subject = subjects.task_turn(task_id)
        return [
            f
            for s, f in self.published
            if s == turn_subject and isinstance(f, TurnFrame) and f.agent_id == agent_id
        ]


class QueueEngine:
    """A genuinely DECOUPLED scripted ``EngineSession`` (no external I/O).

    Contract (issue #51 completion + the engine ordering invariant): every
    ``send`` opens a turn that COMPLETES with a terminal ``TurnDone`` — but the
    ``TurnDone`` is ENQUEUED, never delivered inside ``send``. The supervisor's
    own pump (``events()`` → ``_handle_event``) drains the queue on a later tick,
    so delivery is strictly decoupled from ``send``. This is exactly what
    ``frame_turn._emit_closing_turn_if_hung`` and the supervisor's rotation
    interception require and what the retired synchronous fake violated.

    Per-``send`` scripting drives the debate deterministically by the text:

    * The rotation summary directive enqueues ONE terminal summary ``TurnDone``
      — the supervisor's ``_rotation_pending`` path intercepts it (never a sink
      fan-out, never a ``TurnFrame``) and completes the swap.
    * A sibling-turn injection (a ``[sibling turn from ...]`` render) enqueues
      ONE terminal follow-up ``TurnDone`` — this agent's single debate rebuttal.
    * Any other ``send`` (the task dispatch) enqueues nothing on its own: the
      opener is driven explicitly by the test so the rotation boundary is crossed
      exactly where intended.
    """

    def __init__(self, agent_id: str = "engine") -> None:
        self._queue: asyncio.Queue[EngineEvent | None] = asyncio.Queue()
        self._agent_id = agent_id
        self._session_ref = f"{agent_id}-session"
        self._closed = False
        self.sends: list[str] = []
        self.start_count = 0
        self._followups = 0

    def set_ref(self, ref: str) -> None:
        self._session_ref = ref

    async def start(self, system_prompt: str, tools_policy: ToolsPolicy) -> None:
        self.start_count += 1

    async def resume(self, session_ref: str) -> None:
        self._session_ref = session_ref

    async def push(self, event: EngineEvent | None) -> None:
        """Enqueue an engine event for the supervisor pump to drain (test seam)."""
        await self._queue.put(event)

    async def send(self, text: str) -> None:
        self.sends.append(text)
        if text == ROTATION_SUMMARY_DIRECTIVE:
            # The bounded summary turn: terminal, decoupled. The supervisor
            # intercepts this TurnDone via _rotation_pending and completes the
            # swap — it is never fanned to the runner sink.
            await self._queue.put(TextDelta(text="summarizing..."))
            await self._queue.put(
                TurnDone(
                    text=f"{self._agent_id} rotation summary",
                    session_ref=self._session_ref,
                    stance=Stance(position="summarized", because="context rotation"),
                )
            )
            return
        if text.startswith("[sibling turn from "):
            # A live sibling was seen: post one TERMINAL follow-up so the debate
            # settles below the cap. Enqueued (decoupled), drained by the pump.
            self._followups += 1
            await self._queue.put(TextDelta(text="revising..."))
            await self._queue.put(
                TurnDone(
                    text=_FOLLOWUP_TEXT,
                    session_ref=self._session_ref,
                    stance=Stance(position="revised", because="sibling raised a point"),
                )
            )
            return
        # The task dispatch send: the opener is driven explicitly by the test.

    async def interrupt(self) -> None:
        pass

    async def events(self) -> AsyncIterator[EngineEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    async def health(self) -> EngineHealth:
        return EngineHealth(
            connected=not self._closed,
            session_ref=self._session_ref,
            active_turn=False,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._queue.put(None)


def _task(task_id: str, *, scope: str, participants: list[str]) -> TaskFrame:
    return TaskFrame(
        id=task_id,
        origin_edge="web",
        requester="rico",
        text="do the thing",
        scope=scope,
        stack=_STACK,
        participants=participants,
        created_at=datetime.now(UTC),
    )


def _make_agent(
    agent: str,
    bus: DeliveringBus,
    home: Path,
    *,
    rotation: RotationPolicy | None = None,
    engine_factory: Callable[[], QueueEngine] | None = None,
) -> tuple[QueueEngine, Supervisor, FrameTurnRunner]:
    engine = QueueEngine(agent_id=agent)
    sup = Supervisor(
        session=engine,
        config=SupervisorConfig(
            agent=agent,
            env="dev",
            # Long non-firing loops: the pump is what we exercise; the heartbeat
            # and watchdog must never interfere with the deterministic drain.
            heartbeat_interval_s=1000.0,
            stall_timeout_s=1000.0,
        ),
        home=home,
        publisher=RecordingPublisher(),
        system_prompt="SYS",
        tools_policy=ToolsPolicy(),
        # A fresh per-supervisor active registry keeps each test's agent claims
        # isolated (the default is a process-wide singleton that would reject a
        # second test re-claiming the same agent id), mirroring the lifecycle
        # suite's ``_sup`` helper.
        active_registry=ActiveSessionRegistry(),
        rotation_policy=rotation,
        engine_factory=engine_factory,  # type: ignore[arg-type]
    )
    runner = FrameTurnRunner(agent_id=agent, supervisor=sup, bus=bus)
    return engine, sup, runner


async def _settle(*, ticks: int = 200) -> None:
    """Yield the event loop until every pump has drained its queue.

    Deterministic (no wall-clock sleep): the pumps deliver TurnDones and the
    loopback bus fans siblings within a bounded number of cooperative ticks.
    Yielding ``ticks`` times is far more than the fixed debate depth needs and
    keeps the 5x repeat clean without racing a timer.
    """
    for _ in range(ticks):
        await asyncio.sleep(0)


async def test_rotation_completes_across_live_sibling_debate(tmp_path: Path) -> None:
    """S3b: a turn-bound rotation that COMPLETES (session swap) while a sibling
    round is genuinely live must swap exactly once, close the old session, seed a
    fresh one, and still let the debate converge to orchestrator-derived done.

    Timeline (all through the REAL supervisor pump + real fanout):

    1. alice opens NON-terminal (``needs_from=[bob]``) — the loopback fans this
       to bob's live sibling lane; bob is now OPEN (un-terminal) — the overlap.
    2. alice's opener TurnDone crosses her ``max_turns=1`` rotation bound. Because
       the summary send's TurnDone is DECOUPLED (enqueued, drained by the pump
       and intercepted by ``_rotation_pending``), ``_complete_rotation`` runs:
       ``alice_factory`` is called exactly ONCE, the fresh session is started and
       swapped in, and the OLD session is closed.
    3. bob's sibling follow-up (seen alice's opener) posts a TERMINAL rebuttal;
       alice — now on her FRESH rotated session — sees bob's turn and posts her
       own TERMINAL follow-up. Both agents' latest turns are terminal.
    4. The orchestrator derives done and publishes the result.

    Pins the rotation swap OUTCOME (factory-call count, session identity, old
    session closed) rather than internal arming state, plus exact per-agent turn
    counts and convergence — the multi-agent overlap the single-agent idle-only
    guard tests do not cover.
    """
    bus = DeliveringBus()
    parts = ["alice", "bob"]
    orch = Orchestrator(
        bus=bus, config=OrchestratorConfig(single_agent_id="alice", stack_id=_STACK)
    )

    factory_calls = {"n": 0}
    fresh_engine = QueueEngine(agent_id="alice")
    fresh_engine.set_ref("alice-rotated-ref")

    def alice_factory() -> QueueEngine:
        factory_calls["n"] += 1
        return fresh_engine

    clock = _Clock()
    alice_eng, alice_sup, alice = _make_agent(
        "alice",
        bus,
        tmp_path / "s3b-alice",
        rotation=RotationPolicy(
            max_turns=1000, max_age_s=100.0, summary_timeout_s=1000.0, now=clock
        ),
        engine_factory=alice_factory,
    )
    bob_eng, bob_sup, bob = _make_agent("bob", bus, tmp_path / "s3b-bob")

    for sup in (alice_sup, bob_sup):
        await sup.start()
    await alice.start()
    await bob.start()
    await orch.start()

    # Run the REAL supervisor pumps: TurnDones are delivered decoupled from send.
    runs = [asyncio.create_task(alice_sup.run()), asyncio.create_task(bob_sup.run())]

    task = _task("T-rot", scope="duo", participants=parts)
    await bus.publish_frame(subjects.TASK_INGRESS, task)
    await _settle()

    # alice opens NON-terminal on her ORIGINAL session. Enqueue the opener as a
    # real engine event so the pump delivers it: the sink publishes alice's
    # opener TurnFrame (fanning to bob's LIVE lane), THEN _maybe_rotate fires the
    # rotation whose summary TurnDone the pump later intercepts.
    #
    # bob's sibling round is genuinely OPEN here: bob has published no terminal
    # turn yet when alice's rotation bound is crossed — the overlap under test.
    assert "bob" not in _terminal_agents(bus, "T-rot"), "bob was terminal before the overlap"
    # Cross the age bound so the opener's TurnDone triggers exactly one rotation.
    clock.advance(200.0)
    await alice_eng.push(
        TurnDone(
            text="alice opens",
            stance=Stance(position="opened", because="kickoff", needs_from=["bob"]),
            session_ref="ref-alice-1",
        )
    )
    await _settle()

    # --- rotation OUTCOME (observable, not internal arming state) -------------
    assert factory_calls["n"] == 1, "rotation must swap exactly once (factory called once)"
    assert alice_sup.session is fresh_engine, "the fresh rotated session must be serving"
    assert alice_sup.session is not alice_eng, "the pre-rotation session must be retired"
    assert alice_eng._closed is True, "the old session must be closed on swap"  # noqa: SLF001
    assert fresh_engine.start_count == 1, "the fresh session must be started exactly once"
    assert alice_sup._rotation_pending is None, "no rotation left dangling"  # noqa: SLF001

    # --- exact turn counts per agent -----------------------------------------
    # alice: opener (non-terminal) + one follow-up rebuttal (terminal) = 2.
    # bob:   one follow-up rebuttal (terminal, seen alice's opener) = 1.
    # The rotation summary turn publishes NO TurnFrame (intercepted by the
    # supervisor before any sink) — that absence is the F3 fix, pinned by count.
    alice_turns = bus.turn_frames_by_agent("alice", "T-rot")
    bob_turns = bus.turn_frames_by_agent("bob", "T-rot")
    assert len(alice_turns) == 2, f"alice must publish exactly 2 turns, got {len(alice_turns)}"
    assert len(bob_turns) == 1, f"bob must publish exactly 1 turn, got {len(bob_turns)}"
    assert [t.stance.needs_from for t in alice_turns] == [["bob"], []], (
        "alice opener non-terminal then follow-up terminal"
    )
    assert bob_turns[0].stance.needs_from == [], "bob's follow-up is terminal"
    # No rotation-summary text ever rode out as a TurnFrame.
    assert not any("rotation summary" in t.visible_text for t in alice_turns), (
        "the rotation summary turn must NOT publish a TurnFrame (F3 regression guard)"
    )

    # OVERLAP PROOF: alice's follow-up came from the FRESH rotated engine, not the
    # retired one — so bob's sibling round was still LIVE (un-terminal) when the
    # rotation crossed, and it drove a turn on alice's post-swap session. Had the
    # debate settled before the swap, the rotated engine would have served no
    # follow-up. This pins the interleave observably, not by internal arming.
    assert alice_eng._followups == 0, "the retired session must NOT have served the round"  # noqa: SLF001
    assert fresh_engine._followups == 1, "the rotated session served alice's next round"  # noqa: SLF001
    # And the ordering on the wire shows it: bob's live turn lands AFTER the
    # rotation swap and alice's terminal rebuttal lands last.
    lane = subjects.task_turn("T-rot")
    order = [f.visible_text for s, f in bus.published if s == lane]
    assert order == ["alice opens", "revised position", "revised position"], (
        f"turn-lane order must show opener -> bob live turn -> rotated rebuttal, got {order}"
    )

    # --- debate still converges to orchestrator-derived done -----------------
    results = bus.frames_on(subjects.task_result("T-rot"))
    assert len(results) == 1, "the debate must derive done exactly once under rotation overlap"

    # No sibling lane outlives the settled debate on either agent.
    assert "T-rot" not in alice._sibling_subscriptions  # noqa: SLF001
    assert "T-rot" not in bob._sibling_subscriptions  # noqa: SLF001

    for sup in (alice_sup, bob_sup):
        await sup.stop()
    for run in runs:
        run.cancel()


async def test_rotation_boundary_swaps_without_losing_a_turn(tmp_path: Path) -> None:
    """S3c: a rotation bound crossed at the exact TurnDone that OPENS a live
    sibling round must complete the swap yet lose no turn — the debate still
    converges with each agent's exact expected turns and no double-dispatch.

    Sharp race: alice's ``max_turns=1`` rotation becomes eligible at the SAME
    idle-opener TurnDone whose non-terminal stance drives bob's sibling
    follow-up. Because delivery is decoupled through the real pump, the rotation
    completes on alice's ORIGINAL session (idle at that TurnDone), and the debate
    then flows onto her FRESH session — no follow-up is dropped or duplicated
    across the swap.

    This pins the same swap OUTCOME as S3b at the exact boundary, and pins the
    NO-double-dispatch claim by EXACT turn counts (the retired test asserted it
    by absence with a weak ``assert turns`` — the F3 finding).
    """
    bus = DeliveringBus()
    parts = ["alice", "bob"]
    orch = Orchestrator(
        bus=bus, config=OrchestratorConfig(single_agent_id="alice", stack_id=_STACK)
    )

    swaps = {"n": 0}
    fresh_engine = QueueEngine(agent_id="alice")
    fresh_engine.set_ref("alice-race-rotated")

    def alice_factory() -> QueueEngine:
        swaps["n"] += 1
        return fresh_engine

    clock = _Clock()
    alice_eng, alice_sup, alice = _make_agent(
        "alice",
        bus,
        tmp_path / "s3c-alice",
        rotation=RotationPolicy(
            max_turns=1000, max_age_s=100.0, summary_timeout_s=1000.0, now=clock
        ),
        engine_factory=alice_factory,
    )
    bob_eng, bob_sup, bob = _make_agent("bob", bus, tmp_path / "s3c-bob")

    for sup in (alice_sup, bob_sup):
        await sup.start()
    await alice.start()
    await bob.start()
    await orch.start()

    runs = [asyncio.create_task(alice_sup.run()), asyncio.create_task(bob_sup.run())]

    task = _task("T-race", scope="duo", participants=parts)
    await bus.publish_frame(subjects.TASK_INGRESS, task)
    await _settle()

    # Open the debate with a non-terminal alice stance so a live sibling round is
    # driven at the exact TurnDone that crosses the rotation bound.
    clock.advance(200.0)
    await alice_eng.push(
        TurnDone(
            text="alice opens",
            stance=Stance(position="opened", because="kickoff", needs_from=["bob"]),
            session_ref="ref-alice-race",
        )
    )
    await _settle()

    # Swap happened exactly once, at the boundary, on the ORIGINAL (idle) session.
    assert swaps["n"] == 1, "rotation must swap exactly once at the turn boundary"
    assert alice_sup.session is fresh_engine
    assert alice_eng._closed is True  # noqa: SLF001
    assert alice_sup._rotation_pending is None  # noqa: SLF001

    # Exact turn counts prove no turn was lost and none double-dispatched across
    # the swap: alice opener + one follow-up = 2; bob one follow-up = 1.
    alice_turns = bus.turn_frames_by_agent("alice", "T-race")
    bob_turns = bus.turn_frames_by_agent("bob", "T-race")
    assert len(alice_turns) == 2, f"alice must publish exactly 2 turns, got {len(alice_turns)}"
    assert len(bob_turns) == 1, f"bob must publish exactly 1 turn, got {len(bob_turns)}"
    # Not one extra follow-up (double-dispatch would show a 2nd alice follow-up).
    assert sum(1 for t in alice_turns if t.stance.position == "revised") == 1, (
        "alice posted exactly one follow-up — no double-dispatch across the swap"
    )
    assert not any("rotation summary" in t.visible_text for t in alice_turns), (
        "the rotation summary turn must NOT publish a TurnFrame (F3 regression guard)"
    )

    # OVERLAP PROOF (same as S3b): the rebuttal ran on the FRESH rotated session,
    # so bob's round was still live/open when the boundary was crossed and the
    # swap completed; the wire order confirms the swap-then-live-sibling interleave.
    assert alice_eng._followups == 0, "the retired session must NOT have served the round"  # noqa: SLF001
    assert fresh_engine._followups == 1, "the rotated session served alice's next round"  # noqa: SLF001
    lane = subjects.task_turn("T-race")
    order = [f.visible_text for s, f in bus.published if s == lane]
    assert order == ["alice opens", "revised position", "revised position"], (
        f"turn-lane order must show opener -> bob live turn -> rotated rebuttal, got {order}"
    )

    # The debate derives done across the rotation/sibling boundary.
    results = bus.frames_on(subjects.task_result("T-race"))
    assert len(results) == 1, "done must derive exactly once across the boundary"

    # No sibling lane leaks past the bounded debate on either agent.
    assert "T-race" not in alice._sibling_subscriptions  # noqa: SLF001
    assert "T-race" not in bob._sibling_subscriptions  # noqa: SLF001

    for sup in (alice_sup, bob_sup):
        await sup.stop()
    for run in runs:
        run.cancel()


def _terminal_agents(bus: DeliveringBus, task_id: str) -> set[str]:
    """Return agents whose LATEST turn on ``task_id`` is terminal (empty needs_from)."""
    latest: dict[str, TurnFrame] = {}
    for _s, f in bus.published:
        if isinstance(f, TurnFrame) and f.task_id == task_id:
            latest[f.agent_id] = f
    return {a for a, f in latest.items() if not f.stance.needs_from}
