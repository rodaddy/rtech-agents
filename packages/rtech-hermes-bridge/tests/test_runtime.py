"""Test the bridge runtime loop end to end with wire-faithful fake buses.

Every fleet envelope a test delivers passes through the real wire codec
(``Envelope.to_bytes`` -> ``from_bytes``) inside the fake bus, so these tests
prove the bridge's actual wire contract: typed signals must survive
serialization, every reply becomes an immediate CONTENT turn, and the typed
completion produces NO frame — it only ends the bridge's local lifecycle
(PR #73 redesign: terminal semantics live in the orchestrator).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fleet_core.spec import Stance as FleetStance
from fleet_nats import Envelope
from fleet_nats.sentinel import done_envelope
from rtech_frames import Stance, StatusFrame, TaskFrame, TurnFrame
from rtech_frames.codec import decode_frame, encode_frame
from rtech_hermes_bridge.config import BridgeConfig
from rtech_hermes_bridge.runtime import HermesBridge

if TYPE_CHECKING:
    from .conftest import FakeFleetBus


def _config(**overrides: object) -> BridgeConfig:
    kwargs: dict[str, object] = {"fleet_ids": ("bilby",), "default_channel": "chan-1"}
    kwargs.update(overrides)
    return BridgeConfig(**kwargs)  # type: ignore[arg-type]


def _task(task_id: str = "task-1") -> TaskFrame:
    return TaskFrame(
        id=task_id,
        origin_edge="web-chat",
        requester="rico",
        text="status?",
        scope="single",
        participants=["bilby"],
    )


def _status_reply(
    text: str,
    *,
    task_id: str = "task-1",
    sender: str = "bilby",
    stance: FleetStance | None = None,
    msg_id: str = "r1",
) -> Envelope:
    return Envelope.new(
        msg_id=msg_id,
        ts="2026-07-11T00:00:01+00:00",
        sender=sender,
        kind="status",
        task_id=task_id,
        channel="chan-1",
        payload={"text": text},
        typed_signal=stance,
    )


def _done(
    *, task_id: str = "task-1", sender: str = "bilby", outcome: str = "completed"
) -> Envelope:
    return done_envelope(
        msg_id="d1",
        ts="2026-07-11T00:00:02+00:00",
        sender=sender,
        task_id=task_id,
        outcome=outcome,
        channel="chan-1",
    )


async def _started_bridge(
    fake_bus_cls: type[FakeFleetBus],
    config: BridgeConfig | None = None,
) -> tuple[HermesBridge, FakeFleetBus, FakeFleetBus]:
    rtech_bus = fake_bus_cls()
    fleet_bus = fake_bus_cls()
    bridge = HermesBridge(
        config=config or _config(),
        rtech_bus=rtech_bus,  # type: ignore[arg-type]
        fleet_bus=fleet_bus,  # type: ignore[arg-type]
    )
    await bridge.start()
    return bridge, rtech_bus, fleet_bus


def _frames_on(rtech_bus: FakeFleetBus, subject: str) -> list[TurnFrame | StatusFrame]:
    return [
        decode_frame(data)  # type: ignore[misc]
        for published_subject, data in rtech_bus.raw_published()
        if published_subject == subject
    ]


async def test_start_subscribes_task_lane_and_beats(fake_bus_cls: type[FakeFleetBus]) -> None:
    bridge, rtech_bus, _fleet_bus = await _started_bridge(fake_bus_cls)
    assert rtech_bus.has_raw_sub("rtech.agent.bilby.task")
    await bridge.stop()


async def test_inbound_task_becomes_fleet_dispatch(fake_bus_cls: type[FakeFleetBus]) -> None:
    bridge, rtech_bus, fleet_bus = await _started_bridge(fake_bus_cls)
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task()))
    assert len(fleet_bus.published) == 1
    subject, envelope = fleet_bus.published[0]
    assert subject == "dev.agent.dispatch.bilby"
    assert envelope.kind == "dispatch"
    assert envelope.task_id == "task-1"
    assert envelope.payload["text"] == "status?"
    # The reply lane for the task is subscribed BEFORE publishing.
    assert fleet_bus.has_env_sub("dev.agent.status.task-1")
    await bridge.stop()


async def test_reply_publishes_immediate_content_turn(fake_bus_cls: type[FakeFleetBus]) -> None:
    """Each reply ships as a content turn right away plus a typed working status."""
    bridge, rtech_bus, fleet_bus = await _started_bridge(fake_bus_cls)
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task()))
    await fleet_bus.deliver_envelope("dev.agent.status.task-1", _status_reply("all green"))
    turns = _frames_on(rtech_bus, "rtech.task.task-1.turn")
    assert len(turns) == 1
    frame = turns[0]
    assert isinstance(frame, TurnFrame)
    assert frame.agent_id == "bilby"
    assert frame.visible_text == "all green"
    assert frame.stance.position == "responded"
    status_frames = _frames_on(rtech_bus, "rtech.task.task-1.status")
    statuses = [frame for frame in status_frames if isinstance(frame, StatusFrame)]
    assert [frame.state for frame in statuses] == ["working"]
    # The paired status carries the SAME authorship as the content turn, so
    # per-participant done derivation classifies it against its author.
    assert [frame.agent_id for frame in statuses] == ["bilby"]
    await bridge.stop()


async def test_typed_completion_produces_no_frame(fake_bus_cls: type[FakeFleetBus]) -> None:
    """Frame provenance: the typed Done never originates ANY rtech frame.

    Every frame on the turn lane comes from a ``status`` reply; the frame count
    is unchanged by the completion — the doctrine boundary the round-1/round-2
    shapes violated.
    """
    bridge, rtech_bus, fleet_bus = await _started_bridge(fake_bus_cls)
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task()))
    await fleet_bus.deliver_envelope("dev.agent.status.task-1", _status_reply("all green"))
    published_before = list(rtech_bus.raw_published())
    await fleet_bus.deliver_envelope("dev.agent.status.task-1", _done())
    assert rtech_bus.raw_published() == published_before
    await bridge.stop()


async def test_every_reply_ships_not_only_the_last(fake_bus_cls: type[FakeFleetBus]) -> None:
    """No buffering: statuses stream through as content turns in arrival order."""
    bridge, rtech_bus, fleet_bus = await _started_bridge(fake_bus_cls)
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task()))
    stance = FleetStance(position="agree", fact_pull=True, because="looks right")
    await fleet_bus.deliver_envelope(
        "dev.agent.status.task-1", _status_reply("structural stance", stance=stance, msg_id="r1")
    )
    await fleet_bus.deliver_envelope(
        "dev.agent.status.task-1", _status_reply("the real final answer", msg_id="r2")
    )
    await fleet_bus.deliver_envelope("dev.agent.status.task-1", _done())
    turns = _frames_on(rtech_bus, "rtech.task.task-1.turn")
    assert [turn.visible_text for turn in turns if isinstance(turn, TurnFrame)] == [
        "structural stance",
        "the real final answer",
    ]
    await bridge.stop()


async def test_typed_stance_survives_wire_onto_content_turn(
    fake_bus_cls: type[FakeFleetBus],
) -> None:
    """A typed fleet stance rides through verbatim — position, because, needs_from."""
    bridge, rtech_bus, fleet_bus = await _started_bridge(fake_bus_cls)
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task()))
    stance = FleetStance(
        position="dissent", fact_pull=False, because="numbers disagree", needs_from=("rico",)
    )
    await fleet_bus.deliver_envelope(
        "dev.agent.status.task-1", _status_reply("I disagree", stance=stance)
    )
    turns = _frames_on(rtech_bus, "rtech.task.task-1.turn")
    assert len(turns) == 1
    frame = turns[0]
    assert isinstance(frame, TurnFrame)
    assert frame.stance.position == "dissent"
    assert frame.stance.because == "numbers disagree"
    assert frame.stance.needs_from == ["rico"]
    await bridge.stop()


async def test_done_without_typed_signal_fails_closed(fake_bus_cls: type[FakeFleetBus]) -> None:
    """A hand-rolled kind=done with no typed Done signal must not end the task."""
    bridge, rtech_bus, fleet_bus = await _started_bridge(fake_bus_cls)
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task()))
    await fleet_bus.deliver_envelope("dev.agent.status.task-1", _status_reply("all green"))
    fake_done = Envelope.new(
        msg_id="d1",
        ts="2026-07-11T00:00:02+00:00",
        sender="bilby",
        kind="done",
        task_id="task-1",
        payload={"outcome": "completed"},
    )
    await fleet_bus.deliver_envelope("dev.agent.status.task-1", fake_done)
    # The task is still in flight: subscription intact, state kept.
    assert fleet_bus.has_env_sub("dev.agent.status.task-1")
    assert "task-1" in bridge._pending  # noqa: SLF001 — forged done must not finalize
    await bridge.stop()


async def test_spoofed_sender_is_dropped(fake_bus_cls: type[FakeFleetBus]) -> None:
    """An envelope from another identity on the task subject is dropped entirely."""
    bridge, rtech_bus, fleet_bus = await _started_bridge(fake_bus_cls)
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task()))
    await fleet_bus.deliver_envelope(
        "dev.agent.status.task-1", _status_reply("evil answer", sender="mallory")
    )
    await fleet_bus.deliver_envelope("dev.agent.status.task-1", _done(sender="mallory"))
    assert _frames_on(rtech_bus, "rtech.task.task-1.turn") == []
    assert "task-1" in bridge._pending  # noqa: SLF001 — spoofed done must not finalize
    # The real agent still answers afterwards.
    await fleet_bus.deliver_envelope("dev.agent.status.task-1", _status_reply("real answer"))
    turns = _frames_on(rtech_bus, "rtech.task.task-1.turn")
    assert len(turns) == 1
    assert isinstance(turns[0], TurnFrame)
    assert turns[0].visible_text == "real answer"
    await bridge.stop()


async def test_mismatched_task_id_is_dropped(fake_bus_cls: type[FakeFleetBus]) -> None:
    """An envelope whose own task_id field disagrees with the subject is dropped."""
    bridge, rtech_bus, fleet_bus = await _started_bridge(fake_bus_cls)
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task()))
    await fleet_bus.deliver_envelope(
        "dev.agent.status.task-1", _status_reply("wrong task", task_id="task-9")
    )
    await fleet_bus.deliver_envelope("dev.agent.status.task-1", _done(task_id="task-9"))
    assert _frames_on(rtech_bus, "rtech.task.task-1.turn") == []
    await bridge.stop()


async def test_completion_without_reply_publishes_blocked(
    fake_bus_cls: type[FakeFleetBus],
) -> None:
    """Typed done with no content ever published ends loudly as blocked, no turn."""
    bridge, rtech_bus, fleet_bus = await _started_bridge(fake_bus_cls)
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task()))
    await fleet_bus.deliver_envelope("dev.agent.status.task-1", _done())
    assert _frames_on(rtech_bus, "rtech.task.task-1.turn") == []
    statuses = _frames_on(rtech_bus, "rtech.task.task-1.status")
    assert [frame.state for frame in statuses if isinstance(frame, StatusFrame)] == ["blocked"]
    await bridge.stop()


async def test_dispatch_is_idempotent(fake_bus_cls: type[FakeFleetBus]) -> None:
    bridge, rtech_bus, fleet_bus = await _started_bridge(fake_bus_cls)
    frame_bytes = encode_frame(_task())
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", frame_bytes)
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", frame_bytes)
    # At-least-once delivery must not double-run the gateway.
    assert len(fleet_bus.published) == 1
    await bridge.stop()


async def test_finished_task_redelivery_is_idempotent(fake_bus_cls: type[FakeFleetBus]) -> None:
    bridge, rtech_bus, fleet_bus = await _started_bridge(fake_bus_cls)
    frame_bytes = encode_frame(_task())
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", frame_bytes)
    await fleet_bus.deliver_envelope("dev.agent.status.task-1", _status_reply("all green"))
    await fleet_bus.deliver_envelope("dev.agent.status.task-1", _done())
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", frame_bytes)
    assert len(fleet_bus.published) == 1
    assert len(_frames_on(rtech_bus, "rtech.task.task-1.turn")) == 1
    await bridge.stop()


async def test_subscribe_failure_aborts_dispatch(fake_bus_cls: type[FakeFleetBus]) -> None:
    """No reply consumer means NO dispatch publish — the answer must never be lost."""
    bridge, rtech_bus, fleet_bus = await _started_bridge(fake_bus_cls)
    fleet_bus.fail_subscribe = True
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task()))
    assert fleet_bus.published == []
    # The claim is rolled back: once subscribing works, redelivery dispatches.
    fleet_bus.fail_subscribe = False
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task()))
    assert len(fleet_bus.published) == 1
    assert fleet_bus.has_env_sub("dev.agent.status.task-1")
    await bridge.stop()


async def test_publish_failure_rolls_back_and_unsubscribes(
    fake_bus_cls: type[FakeFleetBus],
) -> None:
    bridge, rtech_bus, fleet_bus = await _started_bridge(fake_bus_cls)
    fleet_bus.fail_publish = True
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task()))
    assert fleet_bus.published == []
    assert all(sub.unsubscribed for sub in fleet_bus.subscriptions)
    fleet_bus.fail_publish = False
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task()))
    assert len(fleet_bus.published) == 1
    await bridge.stop()


async def test_concurrent_tasks_same_agent_serialize(fake_bus_cls: type[FakeFleetBus]) -> None:
    """The gateway holds ONE channel->task correlation; overlap must not cross-route."""
    bridge, rtech_bus, fleet_bus = await _started_bridge(fake_bus_cls)
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task("task-1")))
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task("task-2")))
    # Only the first dispatches; the second waits for the typed completion.
    assert [env.task_id for _, env in fleet_bus.published] == ["task-1"]
    await fleet_bus.deliver_envelope("dev.agent.status.task-1", _status_reply("answer one"))
    await fleet_bus.deliver_envelope("dev.agent.status.task-1", _done())
    assert [env.task_id for _, env in fleet_bus.published] == ["task-1", "task-2"]
    assert fleet_bus.has_env_sub("dev.agent.status.task-2")
    await fleet_bus.deliver_envelope(
        "dev.agent.status.task-2", _status_reply("answer two", task_id="task-2", msg_id="r2")
    )
    await fleet_bus.deliver_envelope("dev.agent.status.task-2", _done(task_id="task-2"))
    turns_one = _frames_on(rtech_bus, "rtech.task.task-1.turn")
    turns_two = _frames_on(rtech_bus, "rtech.task.task-2.turn")
    assert len(turns_one) == 1 and len(turns_two) == 1
    assert isinstance(turns_one[0], TurnFrame) and turns_one[0].visible_text == "answer one"
    assert isinstance(turns_two[0], TurnFrame) and turns_two[0].visible_text == "answer two"
    await bridge.stop()


async def test_queue_overflow_sheds_loudly(fake_bus_cls: type[FakeFleetBus]) -> None:
    """Admission past max_queue_depth sheds with a typed blocked status (H6)."""
    bridge, rtech_bus, fleet_bus = await _started_bridge(fake_bus_cls, _config(max_queue_depth=1))
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task("task-1")))
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task("task-2")))
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task("task-3")))
    # task-1 active, task-2 queued, task-3 shed.
    assert [env.task_id for _, env in fleet_bus.published] == ["task-1"]
    statuses = _frames_on(rtech_bus, "rtech.task.task-3.status")
    assert [frame.state for frame in statuses if isinstance(frame, StatusFrame)] == ["blocked"]
    # The shed task never dispatches, even after the queue drains.
    await fleet_bus.deliver_envelope("dev.agent.status.task-1", _status_reply("one"))
    await fleet_bus.deliver_envelope("dev.agent.status.task-1", _done())
    assert [env.task_id for _, env in fleet_bus.published] == ["task-1", "task-2"]
    await bridge.stop()


async def test_completed_task_unsubscribes_and_drops_state(
    fake_bus_cls: type[FakeFleetBus],
) -> None:
    """No permanent subscriptions or map entries after a task finishes."""
    bridge, rtech_bus, fleet_bus = await _started_bridge(fake_bus_cls)
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task()))
    await fleet_bus.deliver_envelope("dev.agent.status.task-1", _status_reply("all green"))
    await fleet_bus.deliver_envelope("dev.agent.status.task-1", _done())
    assert all(sub.unsubscribed for sub in fleet_bus.subscriptions)
    assert not fleet_bus.has_env_sub("dev.agent.status.task-1")
    assert bridge._pending == {}  # noqa: SLF001 — lifecycle state must be dropped
    assert bridge._active == {}  # noqa: SLF001
    await bridge.stop()


async def test_finalize_claim_is_exactly_once(fake_bus_cls: type[FakeFleetBus]) -> None:
    """Completion racing the sweeper: only one side wins the atomic claim."""
    bridge, rtech_bus, _fleet_bus = await _started_bridge(fake_bus_cls)
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task()))
    first = await bridge._claim_finalize("task-1")  # noqa: SLF001 — race seam under test
    second = await bridge._claim_finalize("task-1")  # noqa: SLF001
    assert first is not None
    assert second is None
    await bridge.stop()


async def test_expired_task_is_swept_blocked_and_queue_advances(
    fake_bus_cls: type[FakeFleetBus],
) -> None:
    """A task with no typed completion expires loudly and frees the agent."""
    bridge, rtech_bus, fleet_bus = await _started_bridge(
        fake_bus_cls, _config(task_timeout_seconds=0.01)
    )
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task("task-1")))
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(_task("task-2")))
    assert [env.task_id for _, env in fleet_bus.published] == ["task-1"]
    await asyncio.sleep(0.02)
    await bridge._sweep_expired()  # noqa: SLF001 — deterministic sweep under test
    statuses = _frames_on(rtech_bus, "rtech.task.task-1.status")
    assert [frame.state for frame in statuses if isinstance(frame, StatusFrame)] == ["blocked"]
    assert _frames_on(rtech_bus, "rtech.task.task-1.turn") == []
    # The queued task dispatched once the expired one released the agent.
    assert [env.task_id for _, env in fleet_bus.published] == ["task-1", "task-2"]
    # A late reply for the expired task has no consumer and produces nothing.
    assert not fleet_bus.has_env_sub("dev.agent.status.task-1")
    await bridge.stop()


async def test_undecodable_task_frame_is_dropped(fake_bus_cls: type[FakeFleetBus]) -> None:
    bridge, rtech_bus, fleet_bus = await _started_bridge(fake_bus_cls)
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", b"not a frame")
    assert fleet_bus.published == []
    await bridge.stop()


async def test_non_task_frame_on_task_lane_ignored(fake_bus_cls: type[FakeFleetBus]) -> None:
    bridge, rtech_bus, fleet_bus = await _started_bridge(fake_bus_cls)
    turn = TurnFrame(
        agent_id="x",
        task_id="task-1",
        visible_text="not a task",
        stance=Stance(position="p", because="b", needs_from=[]),
        refs=[],
    )
    await rtech_bus.deliver_raw("rtech.agent.bilby.task", encode_frame(turn))
    assert fleet_bus.published == []
    await bridge.stop()
