"""Functional I/O tests for the typed-frame chat edge bridge."""

from __future__ import annotations

import pytest
from rtech_chatd.bridge import (
    DEFAULT_INTAKE_ORIGIN_EDGE,
    DEFAULT_ORIGIN_EDGE,
    inbound_to_intake_task_frame,
    inbound_to_task_frame,
    intake_reply_target,
    status_frame_to_send,
    turn_frame_to_send,
    turn_frame_to_tldr,
)
from rtech_chatd.channel import InboundMessage, SendTarget
from rtech_frames import (
    EvidenceSource,
    Stance,
    StatusFrame,
    TaskFrame,
    TurnFrame,
    VerificationReport,
)


def _inbound(**overrides: object) -> InboundMessage:
    base: dict[str, object] = {
        "channel_id": "chan1",
        "thread_id": "t1",
        "author": "user1",
        "text": "hello agent",
        "is_dm": False,
    }
    base.update(overrides)
    return InboundMessage(**base)  # type: ignore[arg-type]


def _target(**overrides: object) -> SendTarget:
    base: dict[str, object] = {
        "channel_id": "chan1",
        "thread_id": "t1",
        "reply_to": None,
    }
    base.update(overrides)
    return SendTarget(**base)  # type: ignore[arg-type]


def test_inbound_mints_task_frame_only() -> None:
    frame = inbound_to_task_frame(_inbound(), task_id="task-1", stack="dev")

    assert isinstance(frame, TaskFrame)
    assert frame.id == "task-1"
    assert frame.origin_edge == DEFAULT_ORIGIN_EDGE
    assert frame.requester == "user1"
    assert frame.text == "hello agent"
    assert frame.scope == "single"


def test_inbound_uses_explicit_edge_identity_and_scope() -> None:
    frame = inbound_to_task_frame(
        _inbound(text="pair on this"),
        task_id="task-2",
        stack="dev",
        origin_edge="discord:chan1:t1",
        scope="duo",
        participants=["codex-a", "opus-b"],
    )

    assert frame.origin_edge == "discord:chan1:t1"
    assert frame.scope == "duo"
    assert frame.participants == ["codex-a", "opus-b"]
    assert frame.stack == "dev"
    assert frame.text == "pair on this"


def test_intake_mint_stamps_created_at_and_channel_identity() -> None:
    msg = _inbound(
        channel_id="chan9",
        thread_id="t9",
        author="asker42",
        text="what is the deploy status",
        guild_id="guild1",
        message_id="msg99",
    )
    frame = inbound_to_intake_task_frame(
        msg,
        task_id="intake-1",
        stack="dev",
        triage_agent="nimbus",
    )

    assert isinstance(frame, TaskFrame)
    assert frame.id == "intake-1"
    assert frame.origin_edge == f"{DEFAULT_INTAKE_ORIGIN_EDGE}:nimbus"
    assert frame.requester == "asker42"
    assert frame.text == "what is the deploy status"
    assert frame.scope == "single"
    assert frame.stack == "dev"
    # created_at is always stamped (fail-closed downstream).
    assert frame.created_at is not None
    # channel identity rides the frame.
    assert frame.origin is not None
    assert frame.origin.guild_id == "guild1"
    assert frame.origin.channel_id == "chan9"
    assert frame.origin.thread_id == "t9"
    assert frame.origin.message_id == "msg99"
    assert frame.origin.asker == "asker42"
    # A single-scope mint carries no debate arbiter role.
    assert frame.arbiter is None


def test_intake_debate_mint_stamps_arbiter_among_participants() -> None:
    # A debate mint stamps the arbiter role authority (the triage agent), which
    # must be one of the participants — TaskFrame validates that invariant.
    trio = ["agentd-nimbus", "agentd-birk", "agentd-iris"]
    frame = inbound_to_intake_task_frame(
        _inbound(guild_id="g", message_id="m"),
        task_id="deb-1",
        stack="dev",
        triage_agent="agentd-nimbus",
        scope="trio",
        participants=trio,
        debate=True,
        arbiter="agentd-nimbus",
    )
    assert isinstance(frame, TaskFrame)
    assert frame.scope == "trio"
    assert frame.participants == trio
    assert frame.arbiter == "agentd-nimbus"
    assert frame.arbiter in frame.participants
    assert frame.origin is not None
    assert frame.origin.debate is True


def test_intake_mint_rejects_blank_stack() -> None:
    with pytest.raises(ValueError, match="stack must be non-empty"):
        inbound_to_intake_task_frame(
            _inbound(guild_id="g", message_id="m"),
            task_id="intake-x",
            stack="  ",
            triage_agent="nimbus",
        )


def test_intake_reply_target_returns_source_channel() -> None:
    frame = inbound_to_intake_task_frame(
        _inbound(channel_id="chanZ", thread_id=None, message_id="msgZ", guild_id="g"),
        task_id="intake-3",
        stack="dev",
        triage_agent="nimbus",
    )
    target = intake_reply_target(frame)
    assert target is not None
    assert target.channel_id == "chanZ"
    assert target.thread_id is None
    assert target.reply_to == "msgZ"
    # The reply lands in the SAME channel the question came from.
    assert target.resolve() == "chanZ"


def test_intake_reply_target_none_for_non_intake_frame() -> None:
    frame = inbound_to_task_frame(_inbound(), task_id="plain-1", stack="dev")
    assert frame.origin is None
    assert intake_reply_target(frame) is None


def test_turn_frame_renders_visible_text_verbatim() -> None:
    frame = TurnFrame(
        agent_id="nova",
        task_id="task-1",
        visible_text="visible answer\nwith exact spacing",
        stance=Stance(
            position="do-not-render-position",
            because="do-not-render-because",
            needs_from=["do-not-render-need"],
        ),
        refs=["do-not-render-ref"],
    )

    send = turn_frame_to_send(frame, target=_target())

    assert send.text == "visible answer\nwith exact spacing"
    assert "do-not-render" not in send.text
    assert send.target.resolve() == "t1"
    assert send.suppressed is False


def test_turn_frame_filter_can_transform_or_suppress_visible_text() -> None:
    frame = TurnFrame(
        agent_id="nova",
        task_id="task-1",
        visible_text="TOKEN=abc leaked",
        stance=Stance(position="neutral", because="typed only"),
    )

    redacted = turn_frame_to_send(
        frame,
        target=_target(thread_id=None),
        outbound_filter=lambda text: text.replace("abc", "[redacted]"),
    )
    suppressed = turn_frame_to_send(frame, target=_target(), outbound_filter=lambda _text: "")

    assert redacted.text == "TOKEN=[redacted] leaked"
    assert redacted.target.resolve() == "chan1"
    assert suppressed.suppressed is True


def test_turn_frame_renders_typed_sources_and_partial_limitations_once() -> None:
    source = EvidenceSource(
        source_id="repo-1",
        kind="repo",
        label="Current config",
        locator="packages/example/config.py",
        authority="source",
    )
    frame = TurnFrame(
        agent_id="nova",
        task_id="task-1",
        visible_text="The configured value is 15 minutes.",
        stance=Stance(position="neutral", because="typed only"),
        verification=VerificationReport(
            status="partial",
            claim_kind="factual",
            ob_checked=True,
            sources=[source, source],
            limitations=["Live runtime was not contacted.", "Live runtime was not contacted."],
        ),
    )

    send = turn_frame_to_send(frame, target=_target())

    assert send.text == (
        "The configured value is 15 minutes.\n\n"
        "Sources checked\n"
        "- Current config — packages/example/config.py\n\n"
        "Limitations\n"
        "- Live runtime was not contacted."
    )


@pytest.mark.parametrize(
    ("status", "claim_kind", "limitation"),
    [
        ("grounded", "factual", "Live state may change after this response."),
        ("partial", "factual", "Only the repository source was available."),
        ("unverified", "factual", "No authoritative source was reachable."),
        ("not_applicable", "social", "This response reflects a stated preference."),
    ],
)
def test_turn_frame_renders_limitations_for_every_verification_status(
    status: str,
    claim_kind: str,
    limitation: str,
) -> None:
    frame = TurnFrame(
        agent_id="nova",
        task_id="task-1",
        visible_text="Answer.",
        stance=Stance(position="neutral", because="typed only"),
        verification=VerificationReport(
            status=status,
            claim_kind=claim_kind,
            ob_checked=True,
            limitations=[limitation],
        ),
    )

    send = turn_frame_to_send(frame, target=_target())

    assert send.text == f"Answer.\n\nLimitations\n- {limitation}"


def test_turn_frame_filters_complete_typed_evidence_render() -> None:
    frame = TurnFrame(
        agent_id="nova",
        task_id="task-1",
        visible_text="Answer.",
        stance=Stance(position="neutral", because="typed only"),
        verification=VerificationReport(
            status="grounded",
            claim_kind="factual",
            ob_checked=True,
            sources=[
                EvidenceSource(
                    source_id="file-1",
                    kind="file",
                    label="TOKEN source",
                    locator="safe/TOKEN.txt",
                    authority="source",
                )
            ],
        ),
    )

    send = turn_frame_to_send(
        frame,
        target=_target(),
        outbound_filter=lambda text: text.replace("TOKEN", "[redacted]"),
    )

    assert send.text == "Answer.\n\nSources checked\n- [redacted] source — safe/[redacted].txt"


def test_debate_tldr_does_not_render_typed_source_details() -> None:
    frame = TurnFrame(
        agent_id="nova",
        task_id="task-1",
        visible_text="Full response.",
        stance=Stance(position="support", because="The typed reason."),
        verification=VerificationReport(
            status="grounded",
            claim_kind="factual",
            ob_checked=True,
            sources=[
                EvidenceSource(
                    source_id="repo-1",
                    kind="repo",
                    label="Current config",
                    locator="packages/example/config.py",
                    authority="source",
                )
            ],
        ),
    )

    assert turn_frame_to_tldr(frame) == "**nova** — support: The typed reason."


def test_status_frame_renders_minimal_indicator() -> None:
    frame = StatusFrame(task_id="task-1", agent_id="agent-a", state="working")

    send = status_frame_to_send(frame, target=_target())

    assert send.text == "Working..."


def test_status_frame_verifying_renders_immediate_acknowledgement() -> None:
    frame = StatusFrame(
        task_id="task-1",
        agent_id="agent-a",
        state="working",
        phase="verifying",
    )

    send = status_frame_to_send(frame, target=_target())

    assert send.text == "Got it — I’m verifying a few things before I answer."


def test_status_frame_responding_preserves_working_indicator() -> None:
    frame = StatusFrame(
        task_id="task-1",
        agent_id="agent-a",
        state="working",
        phase="responding",
    )

    send = status_frame_to_send(frame, target=_target())

    assert send.text == "Working..."


def test_inbound_rejects_blank_stack() -> None:
    # The stack stamp is required addressing config: a
    # blank id would mint tasks every stack-guarded orchestrator drops.
    with pytest.raises(ValueError, match="stack must be non-empty"):
        inbound_to_task_frame(_inbound(), task_id="task-3", stack="  ")


def test_tldr_named_default_prefixes_agent() -> None:
    # Single-bot surfaces: the name prefix IS the attribution (earlier shape).
    frame = TurnFrame(
        agent_id="agentd-nimbus",
        task_id="t1",
        visible_text="long prose",
        stance=Stance(position="ship", because="green"),
    )
    assert turn_frame_to_tldr(frame) == "**agentd-nimbus** — ship: green"


def test_tldr_unnamed_drops_prefix_for_identity_send() -> None:
    # the posting bot IS the attribution — no ventriloquism prefix.
    frame = TurnFrame(
        agent_id="agentd-nimbus",
        task_id="t1",
        visible_text="long prose",
        stance=Stance(position="ship", because="green"),
    )
    assert turn_frame_to_tldr(frame, named=False) == "ship: green"


def test_tldr_because_cap_applies_in_both_shapes() -> None:
    long_because = "x" * 400
    frame = TurnFrame(
        agent_id="a",
        task_id="t1",
        visible_text="p",
        stance=Stance(position="hold", because=long_because),
    )
    named = turn_frame_to_tldr(frame)
    unnamed = turn_frame_to_tldr(frame, named=False)
    assert named.endswith("…") and unnamed.endswith("…")
    assert unnamed == named.removeprefix("**a** — ")
