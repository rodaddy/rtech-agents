"""Test the pure frame <-> fleet-envelope translation and typed classification."""

from __future__ import annotations

from fleet_core.spec import Stance as FleetStance
from fleet_nats import Envelope
from fleet_nats.sentinel import done_envelope
from rtech_frames import TaskFrame
from rtech_hermes_bridge.translate import (
    ReplyCompletion,
    ReplyText,
    build_content_turn,
    build_inbound_envelope,
    classify_reply,
    reply_status_subject,
    task_to_dispatch_subject,
)


def _task(text: str = "what is the status?") -> TaskFrame:
    return TaskFrame(
        id="task-1",
        origin_edge="web-chat",
        requester="rico",
        text=text,
        scope="single",
        participants=["bilby"],
    )


def _wire(envelope: Envelope) -> Envelope:
    """Round-trip an envelope through the real wire codec (typed-signal faithful)."""
    return Envelope.from_bytes(envelope.to_bytes())


def _status_reply(
    text: str,
    *,
    task_id: str = "task-1",
    sender: str = "bilby",
    stance: FleetStance | None = None,
) -> Envelope:
    return _wire(
        Envelope.new(
            msg_id="r1",
            ts="2026-07-11T00:00:01+00:00",
            sender=sender,
            kind="status",
            task_id=task_id,
            channel="chan",
            payload={"text": text},
            typed_signal=stance,
        )
    )


def test_task_to_dispatch_subject_is_bare_lane() -> None:
    assert task_to_dispatch_subject("dev", "bilby") == "dev.agent.dispatch.bilby"


def test_reply_status_subject_keys_on_task_id() -> None:
    assert reply_status_subject("dev", "task-1") == "dev.agent.status.task-1"


def test_task_frame_to_dispatch_envelope() -> None:
    built = build_inbound_envelope(
        task=_task(),
        fleet_id="bilby",
        fleet_env="dev",
        default_channel="100000000000000001",
        msg_id="m1",
        ts="2026-07-11T00:00:00+00:00",
    )
    assert built.subject == "dev.agent.dispatch.bilby"
    env = built.envelope
    assert env.kind == "dispatch"
    assert env.sender == "bilby"
    assert env.task_id == "task-1"
    assert env.channel == "100000000000000001"
    assert env.payload["text"] == "what is the status?"
    # The envelope must round-trip through the sanctioned fleet encoder.
    assert Envelope.from_bytes(env.to_bytes()).payload["text"] == "what is the status?"


def test_task_frame_to_dispatch_envelope_no_channel() -> None:
    built = build_inbound_envelope(
        task=_task(),
        fleet_id="bilby",
        fleet_env="dev",
        default_channel=None,
        msg_id="m1",
        ts="2026-07-11T00:00:00+00:00",
    )
    assert built.envelope.channel is None


def test_classify_status_is_reply_text() -> None:
    result = classify_reply(envelope=_status_reply("all green"), fleet_id="bilby", task_id="task-1")
    assert result == ReplyText(text="all green", stance=None)


def test_classify_status_carries_typed_stance_from_wire() -> None:
    stance = FleetStance(position="agree", fact_pull=True, because="checked it")
    result = classify_reply(
        envelope=_status_reply("looks right", stance=stance),
        fleet_id="bilby",
        task_id="task-1",
    )
    assert isinstance(result, ReplyText)
    assert result.stance is not None
    assert str(result.stance.position) == "agree"
    assert result.stance.because == "checked it"


def test_classify_typed_done_is_completion() -> None:
    done = _wire(
        done_envelope(
            msg_id="d1",
            ts="2026-07-11T00:00:02+00:00",
            sender="bilby",
            task_id="task-1",
            outcome="completed",
        )
    )
    result = classify_reply(envelope=done, fleet_id="bilby", task_id="task-1")
    assert result == ReplyCompletion(outcome="completed")


def test_classify_done_without_typed_signal_fails_closed() -> None:
    fake_done = _wire(
        Envelope.new(
            msg_id="d1",
            ts="2026-07-11T00:00:02+00:00",
            sender="bilby",
            kind="done",
            task_id="task-1",
            payload={"outcome": "completed"},
        )
    )
    assert classify_reply(envelope=fake_done, fleet_id="bilby", task_id="task-1") is None


def test_classify_done_with_mismatched_typed_task_fails_closed() -> None:
    done = _wire(
        done_envelope(
            msg_id="d1",
            ts="2026-07-11T00:00:02+00:00",
            sender="bilby",
            task_id="task-9",
            outcome="completed",
        )
    )
    # Envelope.task_id forged to match; the typed signal's own task_id disagrees.
    forged = done.model_copy(update={"task_id": "task-1"})
    assert classify_reply(envelope=forged, fleet_id="bilby", task_id="task-1") is None


def test_classify_rejects_spoofed_sender() -> None:
    assert (
        classify_reply(
            envelope=_status_reply("evil", sender="mallory"), fleet_id="bilby", task_id="task-1"
        )
        is None
    )


def test_classify_rejects_mismatched_task_id() -> None:
    assert (
        classify_reply(
            envelope=_status_reply("wrong", task_id="task-9"), fleet_id="bilby", task_id="task-1"
        )
        is None
    )


def test_classify_skips_empty_text() -> None:
    assert classify_reply(envelope=_status_reply("   "), fleet_id="bilby", task_id="task-1") is None


def test_classify_skips_unexpected_kind() -> None:
    other = _wire(
        Envelope.new(
            msg_id="x1",
            ts="2026-07-11T00:00:03+00:00",
            sender="bilby",
            kind="heartbeat",
            task_id="task-1",
            payload={"text": "noise"},
        )
    )
    assert classify_reply(envelope=other, fleet_id="bilby", task_id="task-1") is None


def test_content_turn_without_stance_states_bridge_attribution() -> None:
    frame = build_content_turn(fleet_id="bilby", task_id="task-1", text="all green", stance=None)
    assert frame.agent_id == "bilby"
    assert frame.task_id == "task-1"
    assert frame.visible_text == "all green"
    assert frame.stance.position == "responded"
    assert frame.stance.because == "hermes-bridge"
    assert frame.stance.needs_from == []
    assert frame.refs == ["task-1"]


def test_content_turn_passes_typed_stance_through_verbatim() -> None:
    stance = FleetStance(position="dissent", fact_pull=False, because="numbers disagree")
    frame = build_content_turn(fleet_id="bilby", task_id="task-1", text="I disagree", stance=stance)
    assert frame.stance.position == "dissent"
    assert frame.stance.because == "numbers disagree"
    assert frame.stance.needs_from == []


def test_content_turn_passes_needs_from_through_verbatim() -> None:
    # The gateway's own typed needs_from is CONTENT passthrough — the bridge
    # neither strips nor invents it (terminality for bridged participants is
    # decided only by the orchestrator's typed-Done evidence).
    stance = FleetStance(position="agree", fact_pull=False, because="ok", needs_from=("rico",))
    frame = build_content_turn(fleet_id="bilby", task_id="task-1", text="need input", stance=stance)
    assert frame.stance.needs_from == ["rico"]


def test_content_turn_stance_without_reason_falls_back() -> None:
    stance = FleetStance(position="neutral", fact_pull=False)
    frame = build_content_turn(fleet_id="bilby", task_id="task-1", text="noted", stance=stance)
    assert frame.stance.position == "neutral"
    assert frame.stance.because == "hermes-bridge"
