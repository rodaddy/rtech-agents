"""Test typed frame JSON codec behavior."""

from __future__ import annotations

import json

import pytest
import rtech_frames
from rtech_frames.codec import FrameDecodeError, decode_frame, encode_frame
from rtech_frames.frames import (
    MAX_EVIDENCE_SOURCES,
    SCHEMA_VERSION,
    ChannelOrigin,
    EvidenceSource,
    LifecycleFrame,
    Stance,
    StatusFrame,
    SteerFrame,
    TaskFrame,
    TurnFrame,
    VerificationReport,
)


def test_package_entrypoint_loads() -> None:
    assert rtech_frames.__version__


def test_broadcast_target_is_wire_locked_and_exported() -> None:
    """The broadcast token is owned here and its wire value is locked (#76)."""
    assert rtech_frames.BROADCAST_TARGET == "all"
    assert "BROADCAST_TARGET" in rtech_frames.__all__


@pytest.mark.parametrize(
    "frame",
    [
        TaskFrame(
            id="task-1",
            origin_edge="discord",
            requester="rico",
            text="Route this task.",
            scope="single",
        ),
        TurnFrame(
            agent_id="codex-a",
            task_id="task-1",
            visible_text="I will inspect the package.",
            stance=Stance(position="agree", because="The scope is bounded."),
            refs=["docs/decision-record-002-typed-frames.md"],
        ),
        SteerFrame(task_id="task-1", text="Keep this bus-first.", target_agent="codex-a"),
        SteerFrame(
            task_id="task-1",
            text="Stop and redo.",
            target_agent="codex-a",
            interrupt=True,
        ),
        StatusFrame(task_id="task-1", agent_id="codex-a", state="working"),
    ],
)
def test_frame_round_trip(frame: TaskFrame | TurnFrame | SteerFrame | StatusFrame) -> None:
    decoded = decode_frame(encode_frame(frame))

    assert decoded == frame
    assert decoded.schema_version == SCHEMA_VERSION


@pytest.mark.parametrize(
    ("frame", "frame_type"),
    [
        (
            TaskFrame(
                id="task-1",
                origin_edge="discord",
                requester="rico",
                text="Route this task.",
                scope="duo",
                participants=["codex-a", "opus-b"],
            ),
            "task",
        ),
        (
            TurnFrame(
                agent_id="codex-a",
                task_id="task-1",
                visible_text="I will inspect the package.",
                stance=Stance(position="agree", because="The scope is bounded."),
            ),
            "turn",
        ),
        (
            SteerFrame(task_id="task-1", text="Keep this bus-first.", target_agent="codex-a"),
            "steer",
        ),
        (StatusFrame(task_id="task-1", agent_id="codex-a", state="blocked"), "status"),
    ],
)
def test_encoded_frame_includes_wire_type(
    frame: TaskFrame | TurnFrame | SteerFrame | StatusFrame, frame_type: str
) -> None:
    payload = json.loads(encode_frame(frame).decode("utf-8"))

    assert payload["frame_type"] == frame_type
    assert payload["schema_version"] == SCHEMA_VERSION


def test_task_participants_default_empty_and_round_trips() -> None:
    default = TaskFrame(
        id="task-1",
        origin_edge="discord",
        requester="rico",
        text="Route this task.",
        scope="single",
    )
    assert default.participants == []

    duo = TaskFrame(
        id="task-1",
        origin_edge="discord",
        requester="rico",
        text="Pair on this.",
        scope="duo",
        participants=["codex-a", "opus-b"],
    )
    payload = json.loads(encode_frame(duo).decode("utf-8"))
    assert payload["participants"] == ["codex-a", "opus-b"]

    decoded = decode_frame(encode_frame(duo))
    assert isinstance(decoded, TaskFrame)
    assert decoded.participants == ["codex-a", "opus-b"]


def test_task_stack_round_trips_and_legacy_wire_decodes() -> None:
    # New wire: the typed stack scope rides the frame (issue #75).
    stamped = TaskFrame(
        id="task-1",
        origin_edge="discord",
        requester="rico",
        text="Route this task.",
        scope="single",
        stack="mini-m4-dev",
    )
    payload = json.loads(encode_frame(stamped).decode("utf-8"))
    assert payload["stack"] == "mini-m4-dev"
    decoded = decode_frame(encode_frame(stamped))
    assert isinstance(decoded, TaskFrame)
    assert decoded.stack == "mini-m4-dev"

    # Legacy wire: pre-#75 bytes carry no stack key and decode as unscoped.
    legacy_payload = {
        "frame_type": "task",
        "schema_version": SCHEMA_VERSION,
        "id": "task-1",
        "origin_edge": "discord",
        "requester": "rico",
        "text": "Route this task.",
        "scope": "single",
        "participants": [],
    }
    legacy = decode_frame(json.dumps(legacy_payload).encode("utf-8"))
    assert isinstance(legacy, TaskFrame)
    assert legacy.stack == ""


def test_task_origin_round_trips_and_legacy_wire_decodes() -> None:
    # New wire (issue #93 intake): the channel origin rides the frame.
    stamped = TaskFrame(
        id="intake-1",
        origin_edge="chatd-intake:nagatha",
        requester="asker42",
        text="what is the deploy status",
        scope="single",
        stack="dev",
        origin=ChannelOrigin(
            channel_id="chan9",
            guild_id="guild1",
            thread_id="t9",
            message_id="msg99",
            asker="asker42",
        ),
    )
    decoded = decode_frame(encode_frame(stamped))
    assert isinstance(decoded, TaskFrame)
    assert decoded == stamped
    assert decoded.origin is not None
    assert decoded.origin.channel_id == "chan9"
    assert decoded.origin.guild_id == "guild1"

    # Legacy wire: a non-intake frame carries no origin key and decodes as None.
    legacy_payload = {
        "frame_type": "task",
        "schema_version": SCHEMA_VERSION,
        "id": "task-1",
        "origin_edge": "discord",
        "requester": "rico",
        "text": "Route this task.",
        "scope": "single",
        "participants": [],
    }
    legacy = decode_frame(json.dumps(legacy_payload).encode("utf-8"))
    assert isinstance(legacy, TaskFrame)
    assert legacy.origin is None


def test_origin_debate_default_false_omits_key_for_legacy_decoder() -> None:
    """Wire-compat: a non-debate origin omits the ``debate`` key entirely.

    ``ChannelOrigin`` is ``extra="forbid"``; an old decoder without the ``debate``
    field would reject any origin carrying ``debate: false``. The dropped-on-default
    serializer must therefore omit the key when False, keeping the single-scope
    intake origin byte-identical to the pre-debate wire. A True debate marker rides
    normally and round-trips.
    """
    from pydantic import BaseModel, ConfigDict

    class _OldForbiddingOrigin(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        channel_id: str
        guild_id: str | None = None
        thread_id: str | None = None
        message_id: str | None = None
        asker: str | None = None

    single = TaskFrame(
        id="task-1",
        origin_edge="chatd-intake:nagatha",
        requester="asker",
        text="q",
        scope="single",
        origin=ChannelOrigin(channel_id="c1"),
    )
    payload = json.loads(encode_frame(single).decode("utf-8"))
    assert "debate" not in payload["origin"]
    # The old forbidding origin decoder accepts the non-debate origin bytes.
    _OldForbiddingOrigin.model_validate(payload["origin"])

    debate = TaskFrame(
        id="task-2",
        origin_edge="chatd-intake:nagatha",
        requester="asker",
        text="q",
        scope="trio",
        participants=["a", "b", "c"],
        origin=ChannelOrigin(channel_id="c1", debate=True),
    )
    debate_payload = json.loads(encode_frame(debate).decode("utf-8"))
    assert debate_payload["origin"]["debate"] is True
    decoded = decode_frame(encode_frame(debate))
    assert isinstance(decoded, TaskFrame)
    assert decoded == debate
    assert decoded.origin is not None
    assert decoded.origin.debate is True

    # A legacy origin payload with no debate key decodes as the False default.
    legacy = decode_frame(encode_frame(single))
    assert isinstance(legacy, TaskFrame)
    assert legacy.origin is not None
    assert legacy.origin.debate is False


def test_origin_none_omits_key_so_old_forbidding_decoder_accepts() -> None:
    """Wire-compat (D1): a frame without origin encodes with NO ``origin`` key.

    The origin/main decoder is ``extra="forbid"`` and has no ``origin`` field, so
    an ``origin: null`` key would make it reject EVERY new-encoded TaskFrame. The
    new encoder must therefore omit the key entirely when origin is None, leaving
    other Optional None fields (``created_at``) untouched.
    """
    from pydantic import BaseModel, ConfigDict, Field

    class _OldForbiddingTaskFrame(BaseModel):
        """Stand-in for the origin/main extra=forbid decoder (no origin field)."""

        model_config = ConfigDict(extra="forbid", frozen=True)

        frame_type: str
        schema_version: str
        id: str
        origin_edge: str
        requester: str
        text: str
        scope: str
        participants: list[str] = Field(default_factory=list)
        stack: str = ""
        created_at: str | None = None

    frame = TaskFrame(
        id="task-1",
        origin_edge="discord",
        requester="rico",
        text="Route this task.",
        scope="single",
    )
    payload = json.loads(encode_frame(frame).decode("utf-8"))

    # The origin key is absent (not present-as-null), and created_at is untouched.
    assert "origin" not in payload
    assert "created_at" in payload
    assert payload["created_at"] is None

    # The old forbidding decoder (no origin field) accepts the new-encoded bytes.
    _OldForbiddingTaskFrame.model_validate(payload)

    # And a frame WITH origin still emits the key and round-trips.
    stamped = TaskFrame(
        id="task-2",
        origin_edge="chatd-intake:nagatha",
        requester="asker",
        text="q",
        scope="single",
        origin=ChannelOrigin(channel_id="c1"),
    )
    stamped_payload = json.loads(encode_frame(stamped).decode("utf-8"))
    assert stamped_payload["origin"]["channel_id"] == "c1"
    decoded = decode_frame(encode_frame(stamped))
    assert isinstance(decoded, TaskFrame)
    assert decoded == stamped


def test_arbiter_round_trips_and_must_be_a_participant() -> None:
    """The debate arbiter rides the wire and validates as one of participants."""
    frame = TaskFrame(
        id="deb-1",
        origin_edge="chatd-intake:nagatha",
        requester="asker",
        text="debate this",
        scope="trio",
        participants=["agentd-nagatha", "agentd-bilby", "agentd-iris"],
        arbiter="agentd-nagatha",
    )
    payload = json.loads(encode_frame(frame).decode("utf-8"))
    assert payload["arbiter"] == "agentd-nagatha"
    decoded = decode_frame(encode_frame(frame))
    assert isinstance(decoded, TaskFrame)
    assert decoded == frame
    assert decoded.arbiter == "agentd-nagatha"

    # A non-participant arbiter is a mint bug and fails closed at validation.
    with pytest.raises(ValueError, match="arbiter .* must be one of participants"):
        TaskFrame(
            id="deb-2",
            origin_edge="chatd-intake:nagatha",
            requester="asker",
            text="debate this",
            scope="trio",
            participants=["agentd-nagatha", "agentd-bilby", "agentd-iris"],
            arbiter="agentd-stranger",
        )


def test_arbiter_none_omits_key_so_old_forbidding_decoder_accepts() -> None:
    """Wire-compat: a frame without arbiter encodes with NO ``arbiter`` key.

    The pre-#arbiter decoder is ``extra="forbid"`` with no ``arbiter`` field, so
    an ``arbiter: null`` key would make it reject EVERY new-encoded TaskFrame.
    The encoder omits the key entirely when arbiter is None (mirroring
    ``origin``), so a non-debate frame differs from the pre-arbiter wire ONLY in
    the ``schema_version`` stamp ("6" -> "7") — the review-corrected contract:
    NOT byte-identical, but key-set-identical, and an old forbidding decoder
    without a version pin accepts it. A stamped arbiter rides normally.
    """
    from pydantic import BaseModel, ConfigDict, Field

    class _OldForbiddingTaskFrame(BaseModel):
        """Stand-in for the pre-arbiter extra=forbid decoder (no arbiter field)."""

        model_config = ConfigDict(extra="forbid", frozen=True)

        frame_type: str
        schema_version: str
        id: str
        origin_edge: str
        requester: str
        text: str
        scope: str
        participants: list[str] = Field(default_factory=list)
        stack: str = ""
        created_at: str | None = None

    frame = TaskFrame(
        id="task-1",
        origin_edge="discord",
        requester="rico",
        text="Route this task.",
        scope="single",
    )
    payload = json.loads(encode_frame(frame).decode("utf-8"))
    assert "arbiter" not in payload
    # Pin the REAL legacy contract: the "6"-era wire had exactly these keys, so
    # apart from the schema_version stamp the key set is unchanged — an old
    # decoder sees no new key, only the version string it must not pin.
    assert payload["schema_version"] == SCHEMA_VERSION
    assert set(payload) == set(_OldForbiddingTaskFrame.model_fields)
    # The old forbidding decoder (no arbiter field) accepts the new-encoded bytes.
    _OldForbiddingTaskFrame.model_validate(payload)

    # And a legacy payload with no arbiter key decodes as the None default.
    legacy = decode_frame(encode_frame(frame))
    assert isinstance(legacy, TaskFrame)
    assert legacy.arbiter is None

    # A stray arbiter on a frame with NO participants fails closed (review
    # finding: it previously validated and serialized).
    with pytest.raises(ValueError, match="arbiter .* must be one of participants"):
        TaskFrame(
            id="task-2",
            origin_edge="discord",
            requester="rico",
            text="Route this task.",
            scope="single",
            arbiter="agentd-stranger",
        )


def test_turn_origin_round_trips_and_legacy_wire_decodes() -> None:
    # New wire (issue #93 reply render): the channel origin rides the reply turn.
    stamped = TurnFrame(
        agent_id="nagatha",
        task_id="intake-1",
        visible_text="the deploy is green",
        stance=Stance(position="answered", because="I checked the pipeline."),
        origin=ChannelOrigin(
            channel_id="chan9",
            guild_id="guild1",
            thread_id="t9",
            message_id="msg99",
            asker="asker42",
        ),
    )
    decoded = decode_frame(encode_frame(stamped))
    assert isinstance(decoded, TurnFrame)
    assert decoded == stamped
    assert decoded.origin is not None
    assert decoded.origin.channel_id == "chan9"
    assert decoded.origin.message_id == "msg99"

    # Legacy wire: a non-intake turn carries no origin key and decodes as None.
    legacy_payload = {
        "frame_type": "turn",
        "schema_version": SCHEMA_VERSION,
        "agent_id": "agent-a",
        "task_id": "task-1",
        "visible_text": "hi",
        "stance": {"position": "working", "because": "active", "needs_from": []},
        "refs": [],
    }
    legacy = decode_frame(json.dumps(legacy_payload).encode("utf-8"))
    assert isinstance(legacy, TurnFrame)
    assert legacy.origin is None


def test_turn_origin_none_omits_key_so_old_forbidding_decoder_accepts() -> None:
    """Wire-compat: a TurnFrame without origin encodes with NO ``origin`` key.

    Mirrors :func:`test_origin_none_omits_key_so_old_forbidding_decoder_accepts`
    for the TurnFrame: the origin/main decoder is ``extra="forbid"`` with no
    ``origin`` field, so an ``origin: null`` key would make it reject EVERY
    new-encoded TurnFrame. The new encoder omits the key entirely when origin is
    None, so a legacy-shaped turn is byte-identical to the pre-#93 wire.
    """
    from pydantic import BaseModel, ConfigDict, Field

    class _OldForbiddingTurnFrame(BaseModel):
        """Stand-in for the origin/main extra=forbid decoder (no origin field)."""

        model_config = ConfigDict(extra="forbid", frozen=True)

        frame_type: str
        schema_version: str
        agent_id: str
        task_id: str
        visible_text: str
        stance: dict[str, object]
        refs: list[str] = Field(default_factory=list)

    frame = TurnFrame(
        agent_id="agent-a",
        task_id="task-1",
        visible_text="hi",
        stance=Stance(position="working", because="active"),
    )
    payload = json.loads(encode_frame(frame).decode("utf-8"))

    # The origin key is absent (not present-as-null).
    assert "origin" not in payload

    # The old forbidding decoder (no origin field) accepts the new-encoded bytes.
    _OldForbiddingTurnFrame.model_validate(payload)

    # And a turn WITH origin still emits the key and round-trips.
    stamped = TurnFrame(
        agent_id="nagatha",
        task_id="task-2",
        visible_text="done",
        stance=Stance(position="answered", because="checked"),
        origin=ChannelOrigin(channel_id="c1"),
    )
    stamped_payload = json.loads(encode_frame(stamped).decode("utf-8"))
    assert stamped_payload["origin"]["channel_id"] == "c1"
    decoded = decode_frame(encode_frame(stamped))
    assert isinstance(decoded, TurnFrame)
    assert decoded == stamped


def test_turn_verification_round_trips_and_legacy_wire_decodes() -> None:
    # New wire (evidence-gated responses): a full report rides the answer turn.
    report = VerificationReport(
        status="grounded",
        claim_kind="factual",
        ob_checked=True,
        sources=[
            EvidenceSource(
                source_id="OB1",
                kind="open_brain",
                label="deploy-status note",
                locator="ob://entity/deploy-status",
                authority="memory",
            ),
        ],
        limitations=["OB recall timed out on the second lane"],
        methods=["open_brain_recall"],
    )
    stamped = TurnFrame(
        agent_id="nagatha",
        task_id="intake-1",
        visible_text="the deploy is green",
        stance=Stance(position="answered", because="I checked the pipeline."),
        verification=report,
    )
    decoded = decode_frame(encode_frame(stamped))
    assert isinstance(decoded, TurnFrame)
    assert decoded == stamped
    assert decoded.verification is not None
    assert decoded.verification.status == "grounded"
    assert decoded.verification.sources[0].locator == "ob://entity/deploy-status"

    _assert_legacy_turn_has_no_verification()


def _assert_legacy_turn_has_no_verification() -> None:
    # Legacy wire: a pre-evidence turn carries no verification key => decodes None.
    legacy_payload = {
        "frame_type": "turn",
        "schema_version": SCHEMA_VERSION,
        "agent_id": "agent-a",
        "task_id": "task-1",
        "visible_text": "hi",
        "stance": {"position": "working", "because": "active", "needs_from": []},
        "refs": [],
    }
    legacy = decode_frame(json.dumps(legacy_payload).encode("utf-8"))
    assert isinstance(legacy, TurnFrame)
    assert legacy.verification is None


def test_turn_verification_none_omits_key_so_old_forbidding_decoder_accepts() -> None:
    """Wire-compat: a TurnFrame without verification encodes with NO ``verification`` key.

    Mirrors the origin omission: the old decoder is ``extra="forbid"`` with no
    ``verification`` field, so a ``verification: null`` key would make it reject
    EVERY new-encoded TurnFrame. The encoder omits the key entirely when None.
    """
    from pydantic import BaseModel, ConfigDict, Field

    class _OldForbiddingTurnFrame(BaseModel):
        """Stand-in for the pre-evidence extra=forbid decoder (no verification field)."""

        model_config = ConfigDict(extra="forbid", frozen=True)

        frame_type: str
        schema_version: str
        agent_id: str
        task_id: str
        visible_text: str
        stance: dict[str, object]
        refs: list[str] = Field(default_factory=list)

    frame = TurnFrame(
        agent_id="agent-a",
        task_id="task-1",
        visible_text="hi",
        stance=Stance(position="working", because="active"),
    )
    payload = json.loads(encode_frame(frame).decode("utf-8"))
    assert "verification" not in payload
    _OldForbiddingTurnFrame.model_validate(payload)


def test_status_phase_round_trips_and_legacy_wire_decodes() -> None:
    # New wire (evidence-gated responses): the cold-ack phase rides the status.
    stamped = StatusFrame(
        task_id="task-1",
        agent_id="codex-a",
        state="working",
        phase="verifying",
    )
    payload = json.loads(encode_frame(stamped).decode("utf-8"))
    assert payload["phase"] == "verifying"
    decoded = decode_frame(encode_frame(stamped))
    assert isinstance(decoded, StatusFrame)
    assert decoded == stamped
    assert decoded.phase == "verifying"

    # Legacy wire: a pre-evidence status carries no phase key => decodes None.
    legacy_payload = {
        "frame_type": "status",
        "schema_version": SCHEMA_VERSION,
        "task_id": "task-1",
        "agent_id": "codex-a",
        "state": "working",
    }
    legacy = decode_frame(json.dumps(legacy_payload).encode("utf-8"))
    assert isinstance(legacy, StatusFrame)
    assert legacy.phase is None
    assert legacy.state == "working"


def test_status_phase_none_omits_key_so_old_forbidding_decoder_accepts() -> None:
    """Wire-compat: a StatusFrame without phase encodes with NO ``phase`` key.

    The old decoder is ``extra="forbid"`` with no ``phase`` field, so a
    ``phase: null`` key would make it reject EVERY new-encoded StatusFrame. The
    encoder omits the key entirely when phase is None.
    """
    from pydantic import BaseModel, ConfigDict

    class _OldForbiddingStatusFrame(BaseModel):
        """Stand-in for the pre-evidence extra=forbid decoder (no phase field)."""

        model_config = ConfigDict(extra="forbid", frozen=True)

        frame_type: str
        schema_version: str
        task_id: str
        agent_id: str
        state: str

    frame = StatusFrame(task_id="task-1", agent_id="codex-a", state="working")
    payload = json.loads(encode_frame(frame).decode("utf-8"))
    assert "phase" not in payload
    _OldForbiddingStatusFrame.model_validate(payload)


def test_steer_interrupt_defaults_false_and_round_trips() -> None:
    default = SteerFrame(task_id="task-1", text="carry on", target_agent="codex-a")
    assert default.interrupt is False

    flagged = SteerFrame(task_id="task-1", text="abort", target_agent="codex-a", interrupt=True)
    payload = json.loads(encode_frame(flagged).decode("utf-8"))
    assert payload["interrupt"] is True

    decoded = decode_frame(encode_frame(flagged))
    assert isinstance(decoded, SteerFrame)
    assert decoded.interrupt is True


def test_lifecycle_frame_round_trips_with_wire_type() -> None:
    """LifecycleFrame carries reason/initiator/correlation_id on the wire (#84)."""
    frame = LifecycleFrame(
        agent_id="agentd-bilby",
        reason="shutdown",
        initiator="signal:SIGTERM",
        correlation_id="corr-1",
        detail="operator ctrl-c",
    )
    payload = json.loads(encode_frame(frame).decode("utf-8"))
    assert payload["frame_type"] == "lifecycle"
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["reason"] == "shutdown"
    assert payload["initiator"] == "signal:SIGTERM"
    assert payload["correlation_id"] == "corr-1"

    decoded = decode_frame(encode_frame(frame))
    assert isinstance(decoded, LifecycleFrame)
    assert decoded == frame


def test_decode_rejects_unknown_frame_type() -> None:
    payload = {
        "frame_type": "mystery",
        "schema_version": SCHEMA_VERSION,
        "task_id": "task-1",
    }

    with pytest.raises(FrameDecodeError, match="unknown frame_type"):
        decode_frame(json.dumps(payload).encode("utf-8"))


def test_decode_rejects_missing_required_field() -> None:
    payload = {
        "frame_type": "task",
        "schema_version": SCHEMA_VERSION,
        "id": "task-1",
        "origin_edge": "discord",
        "requester": "rico",
        "scope": "single",
    }

    with pytest.raises(FrameDecodeError, match="invalid task frame"):
        decode_frame(json.dumps(payload).encode("utf-8"))


def test_decode_rejects_v1_status_without_agent_id() -> None:
    # Wire change (PR #73 round 5): StatusFrame gained a required agent_id in
    # schema_version 2. A v1 status (no authorship) fails closed at decode —
    # deployed stacks must re-sync on merge.
    payload = {
        "frame_type": "status",
        "schema_version": "1",
        "task_id": "task-1",
        "state": "working",
    }

    with pytest.raises(FrameDecodeError, match="invalid status frame"):
        decode_frame(json.dumps(payload).encode("utf-8"))


def test_decode_rejects_missing_frame_type() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": "task-1",
        "state": "working",
    }

    with pytest.raises(FrameDecodeError, match="frame_type"):
        decode_frame(json.dumps(payload).encode("utf-8"))


def _valid_source(source_id: str = "OB1") -> EvidenceSource:
    return EvidenceSource(source_id=source_id, kind="open_brain", label="a note")


def _report(**overrides: object) -> VerificationReport:
    base: dict[str, object] = {
        "status": "grounded",
        "claim_kind": "factual",
        "ob_checked": True,
    }
    base.update(overrides)
    return VerificationReport(**base)  # type: ignore[arg-type]


def test_verification_report_bounds_reject_oversize_and_blank() -> None:
    # A blank limitation entry is rejected by the after-validator.
    with pytest.raises(ValueError):
        _report(limitations=["   "])
    # A limitation over 300 chars is rejected.
    with pytest.raises(ValueError):
        _report(limitations=["x" * 301])
    # More than MAX_EVIDENCE_SOURCES sources is rejected by the field bound.
    with pytest.raises(ValueError):
        _report(sources=[_valid_source(f"S{i}") for i in range(MAX_EVIDENCE_SOURCES + 1)])
    # A blank method entry is rejected by the after-validator.
    with pytest.raises(ValueError):
        _report(methods=["  "])
