"""Prove the shared ``TypedFrameBus`` gives ``FrameTurnRunner`` a REAL retire handle.

Root cause (the debate reinjection bug): ``TypedFrameBus.subscribe_frame``
awaited ``_raw_subscribe`` but returned ``None``. ``FrameTurnRunner`` wraps that
return in ``_BusSubscription``; a ``None`` handle makes ``unsubscribe`` a logged
no-op ("no unsubscribe handle to retire lane"). Over a REAL NATS debate the
sibling turn lane therefore stayed live past the round cap: every post-cap
sibling turn kept being DELIVERED to ``_on_sibling_turn`` (and, once the
lane-scoped counter is cleared, could be reinjected), so agents never stopped
seeing each other despite the "round budget exhausted" logs.

This is a cross-boundary test on purpose: it wires the ACTUAL orchestrator-owned
:class:`~rtech_orchestrator.typedbus.TypedFrameBus` (not a fake bus) to the
ACTUAL agentd :class:`~rtech_agentd.frame_turn.FrameTurnRunner`, over a raw-NATS
stub that models core-NATS delivery honestly — a publish reaches every callback
still subscribed to the subject, and a raw ``unsubscribe`` genuinely removes the
callback. The proof: once the runner retires the sibling lane through the bus's
now-real unsubscribe handle, a subsequently published sibling turn reaches ZERO
callbacks — it is neither delivered nor reinjected.

Bus-first is preserved: the fanout is a typed ``TurnFrame`` on the shared task
turn lane, retirement is bus-owned (the raw subscription), and there is no
connector state, no done token, and no prose trailer anywhere in the path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from rtech_agentd.engine import EngineEvent, EngineHealth, ToolsPolicy
from rtech_agentd.frame_turn import FrameTurnRunner
from rtech_agentd.registry import ActiveSessionRegistry
from rtech_agentd.supervisor import Supervisor, SupervisorConfig
from rtech_frames import Stance, TurnFrame
from rtech_orchestrator import subjects
from rtech_orchestrator.typedbus import TypedFrameBus

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from pathlib import Path

    from fleet_nats import Envelope, FleetBus
    from rtech_frames.codec import Frame

_AGENT = "alice"
_TASK = "T-boundary"


class _RawSub:
    """A raw ``nats.aio``-style subscription whose ``unsubscribe`` truly retires.

    Removing the callback from the delivery table is exactly what a live NATS
    ``Subscription.unsubscribe`` does: no message published afterward reaches it.
    """

    def __init__(self, table: list[Callable[[object], Awaitable[None]]], cb: object) -> None:
        self._table = table
        self._cb = cb

    async def unsubscribe(self) -> None:
        cb = cast("Callable[[object], Awaitable[None]]", self._cb)
        if cb in self._table:
            self._table.remove(cb)


class _RawNats:
    """Core-NATS stub: ``publish`` fans bytes to every live subject callback.

    Models the ONE property the fix rides on — a retired subscription no longer
    receives publishes — without a live server. ``subscribe`` registers a raw
    callback and hands back a :class:`_RawSub` whose ``unsubscribe`` removes it.
    ``publish`` invokes exactly the callbacks still registered for the subject,
    so a delivery after retirement provably reaches none.
    """

    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[[object], Awaitable[None]]]] = {}
        self.delivered: dict[str, int] = {}

    async def subscribe(self, subject: str, cb: Callable[[object], Awaitable[None]]) -> _RawSub:
        table = self._subs.setdefault(subject, [])
        table.append(cb)
        return _RawSub(table, cb)

    async def publish(self, subject: str, data: bytes) -> None:
        msg = _Msg(data)
        callbacks = list(self._subs.get(subject, []))
        self.delivered[subject] = self.delivered.get(subject, 0) + len(callbacks)
        for cb in callbacks:
            await cb(msg)

    def live_callbacks(self, subject: str) -> int:
        """Return how many callbacks are still subscribed to a subject."""
        return len(self._subs.get(subject, []))


class _Msg:
    """A raw NATS message carrying the encoded frame bytes (``.data``)."""

    def __init__(self, data: bytes) -> None:
        self.data = data


class _StubFleetBus:
    """FleetBus stub exposing the private seams ``TypedFrameBus`` touches."""

    def __init__(self, nc: _RawNats) -> None:
        self._nc = nc

    def _guard_payload_size(self, subject: str, data: bytes) -> None:
        return None


class _ScriptedEngine:
    """A minimal decoupled ``EngineSession``: records every ``send`` for assertions.

    The runner's sibling-injection path calls ``supervisor.session.send`` and
    nothing else here, so this records sends and never delivers events from
    inside ``send`` (the decoupling the engine contract requires).
    """

    def __init__(self) -> None:
        self.sends: list[str] = []
        self._closed = False

    async def start(self, system_prompt: str, tools_policy: ToolsPolicy) -> None:
        return None

    async def resume(self, session_ref: str) -> None:
        return None

    async def send(self, text: str) -> None:
        self.sends.append(text)

    async def interrupt(self) -> None:
        return None

    async def events(self) -> AsyncIterator[EngineEvent]:
        # Never yields: this test drives the sibling seam directly, not the pump.
        if False:  # pragma: no cover - unreachable generator body
            yield cast("EngineEvent", None)

    async def health(self) -> EngineHealth:
        return EngineHealth(connected=not self._closed, session_ref="scripted", active_turn=False)

    async def close(self) -> None:
        self._closed = True


class _RecordingPublisher:
    """Satisfies the ``Publisher`` seam; records nothing this test asserts on."""

    async def __call__(self, subject: str, envelope: Envelope) -> None:
        return None


def _sibling_turn(*, author: str, needs_from: list[str]) -> TurnFrame:
    return TurnFrame(
        agent_id=author,
        task_id=_TASK,
        visible_text="a live sibling position",
        stance=Stance(position="p", because="b", needs_from=needs_from),
    )


def _make_runner(nc: _RawNats, home: Path) -> tuple[FrameTurnRunner, _ScriptedEngine]:
    engine = _ScriptedEngine()
    sup = Supervisor(
        session=cast("object", engine),  # type: ignore[arg-type]
        config=SupervisorConfig(agent=_AGENT, env="dev"),
        home=home,
        publisher=_RecordingPublisher(),
        system_prompt="SYS",
        tools_policy=ToolsPolicy(),
        # Fresh per-test active registry: the default is a process-wide singleton
        # that would reject re-claiming the agent id across tests.
        active_registry=ActiveSessionRegistry(),
    )
    bus = TypedFrameBus(cast("FleetBus", _StubFleetBus(nc)))
    runner = FrameTurnRunner(agent_id=_AGENT, supervisor=sup, bus=cast("object", bus))  # type: ignore[arg-type]
    return runner, engine


async def test_runner_retires_sibling_lane_over_real_typedbus(tmp_path: Path) -> None:
    """A runner on the REAL ``TypedFrameBus`` retires its sibling lane for good.

    Before the fix this test would fail at the final assertion: the bus returned
    ``None``, ``_BusSubscription.unsubscribe`` no-oped, and the raw callback
    stayed in the delivery table — so the post-retire publish still reached the
    runner. With the bus returning an idempotent async retire handle, the lane's
    raw subscription is genuinely removed and the later sibling turn is delivered
    to nobody.
    """
    nc = _RawNats()
    turn_subject = subjects.task_turn(_TASK)
    runner, engine = _make_runner(nc, tmp_path)

    # A duo task fans this agent onto the shared task turn lane. Seed the sibling
    # bookkeeping the runner's fanout path expects for a live fanned task, then
    # subscribe the lane through the REAL bus.
    runner._sibling_subscriptions[_TASK] = None  # type: ignore[assignment]  # noqa: SLF001
    runner._sibling_participants[_TASK] = frozenset({_AGENT, "bob"})  # noqa: SLF001
    runner._sibling_latest_terminal[_TASK] = {}  # noqa: SLF001
    await runner._subscribe_siblings(_TASK)  # noqa: SLF001

    assert nc.live_callbacks(turn_subject) == 1, "the sibling lane must be live on the bus"

    # A live sibling turn arrives and is seen (non-terminal keeps the debate open,
    # so it drives a follow-up send — the engine records it).
    await nc.publish(turn_subject, _encode(_sibling_turn(author="bob", needs_from=[_AGENT])))
    assert engine.sends, "the live sibling turn must reach the session before retirement"
    sends_before = len(engine.sends)

    # Retire the fanout lane the way the round cap does. This is the seam the fix
    # targets: retirement flows bus.subscribe_frame -> _BusSubscription -> the
    # bus's now-real idempotent unsubscribe -> the raw subscription is removed.
    await runner._retire_exhausted_sibling_lane(_TASK)  # noqa: SLF001

    # The lane is gone from the bus AND from the runner's live registry.
    assert nc.live_callbacks(turn_subject) == 0, (
        "retire must remove the raw subscription (the None-return no-op regression)"
    )
    assert _TASK not in runner._sibling_subscriptions  # noqa: SLF001

    # A subsequently published sibling turn reaches ZERO callbacks: not delivered,
    # so it cannot be reinjected into the session. This is the anti-reinjection
    # proof the cap depends on.
    delivered_before = nc.delivered.get(turn_subject, 0)
    await nc.publish(turn_subject, _encode(_sibling_turn(author="bob", needs_from=[_AGENT])))
    assert nc.delivered.get(turn_subject, 0) == delivered_before, (
        "no callback may receive a post-retire sibling turn"
    )
    assert len(engine.sends) == sends_before, (
        "a retired lane must not reinject any further sibling turn into the session"
    )

    # Retirement is idempotent: a redriven retire is a clean no-op, never a double
    # raw-unsubscribe error.
    await runner._retire_exhausted_sibling_lane(_TASK)  # noqa: SLF001
    assert nc.live_callbacks(turn_subject) == 0


def _encode(frame: Frame) -> bytes:
    from rtech_frames.codec import encode_frame

    return encode_frame(frame)
