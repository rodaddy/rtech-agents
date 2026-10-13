"""Codec and validation coverage for the supervised-Start frames (#146).

Split out of ``test_codec.py`` (750-line cap): these are the supervised-Start
codec/validation tests added by the supervised-agent-start work. The round-trip
cases for :class:`StartAgentFrame`, :class:`StartAllAgentsFrame`, and
:class:`SupervisedAgentStateFrame` moved here verbatim from the shared
``test_frame_round_trip`` parametrization, alongside the two dedicated
supervised-Start typing/validation tests. Every assertion and every collected case
is preserved; unrelated legacy codec tests stay in ``test_codec.py``.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from rtech_frames.codec import FrameDecodeError, FrameEncodeError, decode_frame, encode_frame
from rtech_frames.frames import (
    MAX_REQUEST_ID_LEN,
    MAX_SOURCE_ID_LEN,
    SCHEMA_VERSION,
    EvidenceSource,
    Stance,
    StartAgentFrame,
    StartAllAgentsFrame,
    SupervisedAgentStateFrame,
    TurnFrame,
    VerificationReport,
)


@pytest.mark.parametrize(
    "frame",
    [
        StartAgentFrame(request_id="start-1", agent_id="agentd-bilby", requester_id="operator-1"),
        StartAllAgentsFrame(request_id="start-all-1", requester_id="operator-1"),
        SupervisedAgentStateFrame(request_id="start-1", agent_id="agentd-bilby", state="Online"),
    ],
)
def test_supervised_start_frame_round_trip(
    frame: StartAgentFrame | StartAllAgentsFrame | SupervisedAgentStateFrame,
) -> None:
    decoded = decode_frame(encode_frame(frame))

    assert decoded == frame
    assert decoded.schema_version == SCHEMA_VERSION


def test_supervised_start_frames_are_typed_and_failure_guarded() -> None:
    """Start control carries no command/config and failures stay enumerated."""
    request = StartAgentFrame(
        request_id="start-1",
        agent_id="agentd-bilby",
        requester_id="operator-1",
    )
    assert json.loads(encode_frame(request).decode("utf-8"))["frame_type"] == "start_agent"

    failed = SupervisedAgentStateFrame(
        request_id="start-1",
        agent_id="agentd-bilby",
        state="Failed",
        failure="guard_rejected",
    )
    assert decode_frame(encode_frame(failed)) == failed

    with pytest.raises(ValueError, match="requires a typed failure code"):
        SupervisedAgentStateFrame(request_id="start-1", agent_id="agentd-bilby", state="Failed")


def test_liveness_unconfirmed_is_a_typed_terminal_failure() -> None:
    """The v9 contract enumerates liveness_unconfirmed as a typed Failed code (#146).

    A start action that returned but was not affirmatively confirmed alive fails
    closed with this typed code — never a false Online. It round-trips as a valid
    terminal Failed receipt and stays inside the enumerated failure vocabulary.
    """
    unconfirmed = SupervisedAgentStateFrame(
        request_id="start-1",
        agent_id="agentd-bilby",
        state="Failed",
        failure="liveness_unconfirmed",
    )
    assert decode_frame(encode_frame(unconfirmed)) == unconfirmed
    # A non-Failed state may never carry it (or any failure code).
    with pytest.raises(ValueError, match="only failed supervised state may carry a failure code"):
        SupervisedAgentStateFrame(
            request_id="start-1",
            agent_id="agentd-bilby",
            state="Online",
            failure="liveness_unconfirmed",
        )


# The three Start models bound ``request_id`` to the ONE shared
# :data:`MAX_REQUEST_ID_LEN` (terminal-audit finding): each is built with a
# request_id EXACTLY at the bound (accepted) and one char over (rejected), so an
# oversized id can never enter an in-flight/completed key or ride a terminal
# receipt. Each entry is (label, factory) where factory(request_id) mints the model.
_START_MODEL_FACTORIES = [
    (
        "StartAgentFrame",
        lambda request_id: StartAgentFrame(
            request_id=request_id, agent_id="agentd-bilby", requester_id="operator-1"
        ),
    ),
    (
        "StartAllAgentsFrame",
        lambda request_id: StartAllAgentsFrame(request_id=request_id, requester_id="operator-1"),
    ),
    (
        "SupervisedAgentStateFrame",
        lambda request_id: SupervisedAgentStateFrame(
            request_id=request_id, agent_id="agentd-bilby", state="Online"
        ),
    ),
]


@pytest.mark.parametrize(("label", "factory"), _START_MODEL_FACTORIES)
def test_request_id_at_max_len_is_accepted_and_round_trips(label, factory) -> None:
    """A request_id EXACTLY at MAX_REQUEST_ID_LEN is accepted and round-trips (#146)."""
    max_id = "a" * MAX_REQUEST_ID_LEN
    frame = factory(max_id)

    assert frame.request_id == max_id
    assert len(frame.request_id) == MAX_REQUEST_ID_LEN
    # Codec behavior: the boundary-length id survives an encode/decode round trip.
    assert decode_frame(encode_frame(frame)) == frame


@pytest.mark.parametrize(("label", "factory"), _START_MODEL_FACTORIES)
def test_request_id_over_max_len_is_rejected(label, factory) -> None:
    """A request_id ONE char over MAX_REQUEST_ID_LEN is rejected at construction (#146)."""
    over_id = "a" * (MAX_REQUEST_ID_LEN + 1)

    with pytest.raises(ValidationError):
        factory(over_id)


@pytest.mark.parametrize(("label", "factory"), _START_MODEL_FACTORIES)
def test_oversized_request_id_is_rejected_at_decode(label, factory) -> None:
    """The codec rejects an over-max request_id on the wire before it can key state.

    An oversized id is hand-crafted into wire bytes (bypassing the model
    constructor) so the DECODE path itself is exercised: it must fail closed with a
    :class:`FrameDecodeError`, never construct a frame that could enter an
    in-flight/completed key or ride a terminal receipt (terminal-audit finding).
    """
    max_frame = factory("a" * MAX_REQUEST_ID_LEN)
    payload = json.loads(encode_frame(max_frame).decode("utf-8"))
    payload["request_id"] = "a" * (MAX_REQUEST_ID_LEN + 1)
    oversized_bytes = json.dumps(payload).encode("utf-8")

    with pytest.raises(FrameDecodeError):
        decode_frame(oversized_bytes)


# ``model_copy(update=...)`` (and any post-construction mutation) BYPASSES Pydantic
# field validation, so a frame can hold an out-of-bounds field the constructor would
# have rejected. The generic ``encode_frame`` revalidation boundary must catch this
# for EVERY constrained frame — no per-field/per-frame special-casing — and must
# NEVER emit bytes for an invalid frame. These regressions prove the boundary for all
# three supervised-Start frame types with an oversized ``request_id`` (the #146
# terminal-audit finding) plus a nested-constraint case on a different field.
@pytest.mark.parametrize(("label", "factory"), _START_MODEL_FACTORIES)
def test_encode_rejects_model_copy_oversized_request_id(label, factory) -> None:
    """A frame copied with an over-max ``request_id`` fails encode, emitting no bytes.

    ``model_copy(update=...)`` skips validation, so the oversized id rides an
    otherwise-valid instance straight into ``encode_frame``. The revalidation
    boundary must reject it with :class:`FrameEncodeError` before serialization, so
    invalid wire bytes for a key-able ``request_id`` never leave the encoder.
    """
    valid = factory("start-1")
    tampered = valid.model_copy(update={"request_id": "a" * (MAX_REQUEST_ID_LEN + 1)})

    # The bypass really happened: the instance holds the out-of-bounds value.
    assert len(tampered.request_id) == MAX_REQUEST_ID_LEN + 1

    with pytest.raises(FrameEncodeError):
        encode_frame(tampered)


def test_encode_rejects_model_copy_oversized_nested_source_id() -> None:
    """A DIFFERENT constrained field, nested one level deep, also fails encode.

    ``TurnFrame.verification.sources[*].source_id`` is bounded to
    :data:`MAX_SOURCE_ID_LEN`. A copy that swaps in an oversized nested
    ``EvidenceSource`` bypasses validation the same way an oversized ``request_id``
    does. The encoder revalidates the WHOLE instance (nested models included), so the
    single generic boundary — not a ``request_id`` special-case — protects every
    constrained frame. Encode must fail with :class:`FrameEncodeError` and emit no
    bytes.
    """
    good_source = EvidenceSource(source_id="OB1", kind="open_brain", label="a note")
    report = VerificationReport(
        status="grounded", claim_kind="factual", ob_checked=True, sources=[good_source]
    )
    turn = TurnFrame(
        agent_id="agentd-bilby",
        task_id="task-1",
        visible_text="answer",
        stance=Stance(position="agree", because="checked"),
        verification=report,
    )

    oversized_source = good_source.model_copy(update={"source_id": "a" * (MAX_SOURCE_ID_LEN + 1)})
    oversized_report = report.model_copy(update={"sources": [oversized_source]})
    tampered = turn.model_copy(update={"verification": oversized_report})

    assert len(tampered.verification.sources[0].source_id) == MAX_SOURCE_ID_LEN + 1

    with pytest.raises(FrameEncodeError):
        encode_frame(tampered)


def test_encode_preserves_wire_bytes_for_a_valid_frame() -> None:
    """The revalidation boundary is byte-transparent for a valid frame (#146).

    Adding revalidation must not change the wire for a well-formed frame: a normal
    valid :class:`StartAgentFrame` encodes to the EXACT pre-fix expected bytes.
    """
    valid = StartAgentFrame(
        request_id="start-1", agent_id="agentd-bilby", requester_id="operator-1"
    )
    expected = (
        b'{"agent_id":"agentd-bilby","frame_type":"start_agent",'
        b'"request_id":"start-1","requester_id":"operator-1","schema_version":"9"}'
    )

    assert encode_frame(valid) == expected


def test_supervised_start_symbols_are_a_single_identity_across_surfaces() -> None:
    """The moved supervised-Start family has ONE identity from every import surface.

    The family now lives in :mod:`rtech_frames.supervised` and is re-exported from
    :mod:`rtech_frames.frames` and the package root. A move must not fork the model
    classes: the codec registry keys on ``type(frame)``, so if the frames-module
    re-export or the package-root export resolved to a DIFFERENT class object than
    the codec imports, encode/decode would silently break. This pins that every
    public surface yields the SAME object.
    """
    import rtech_frames
    from rtech_frames import frames as frames_module
    from rtech_frames import supervised as supervised_module

    # Re-exported from the frames module (the historical
    # ``from rtech_frames.frames import ...`` surface). ``StartFailure`` was only
    # ever public via the frames module, never the package root, so it is checked
    # here and NOT against ``rtech_frames`` — the move preserves that exact surface.
    for name in (
        "StartAgentFrame",
        "StartAllAgentsFrame",
        "SupervisedAgentStateFrame",
        "StartFailure",
        "MAX_REQUEST_ID_LEN",
    ):
        owned = getattr(supervised_module, name)
        assert getattr(frames_module, name) is owned, f"frames.{name} re-export forked"

    # Exported from the package root (unchanged public surface). ``StartFailure`` is
    # deliberately absent — it was never a root export.
    for name in (
        "StartAgentFrame",
        "StartAllAgentsFrame",
        "SupervisedAgentStateFrame",
        "MAX_REQUEST_ID_LEN",
    ):
        assert getattr(rtech_frames, name) is getattr(supervised_module, name), (
            f"root {name} export forked"
        )
    assert not hasattr(rtech_frames, "StartFailure"), "StartFailure must not be a root export"
