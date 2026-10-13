"""Test the PresenceTransitionFrame wire contract (issue #147)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError
from rtech_frames import (
    FrameEncodeError,
    PresenceTransitionFrame,
    decode_frame,
    encode_frame,
)
from rtech_frames.codec import FrameDecodeError


def _frame() -> PresenceTransitionFrame:
    return PresenceTransitionFrame(
        agent_id="nagatha-2",
        display_name="Nagatha 2",
        transition="offline",
        observed_at=datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC),
        transition_id="abc123def456",
    )


def test_round_trips_through_encode_decode() -> None:
    frame = _frame()
    decoded = decode_frame(encode_frame(frame))
    assert isinstance(decoded, PresenceTransitionFrame)
    assert decoded == frame
    assert decoded.transition == "offline"
    assert decoded.observed_at == frame.observed_at


def test_recovered_transition_round_trips() -> None:
    frame = PresenceTransitionFrame(
        agent_id="skippy",
        display_name="Skippy",
        transition="recovered",
        observed_at=datetime(2026, 7, 20, 13, 30, 0, tzinfo=UTC),
        transition_id="recover1",
    )
    decoded = decode_frame(encode_frame(frame))
    assert isinstance(decoded, PresenceTransitionFrame)
    assert decoded.transition == "recovered"


def test_malformed_bytes_raise_frame_decode_error() -> None:
    with pytest.raises(FrameDecodeError):
        decode_frame(b"not json at all")


def test_unknown_frame_type_raises_frame_decode_error() -> None:
    with pytest.raises(FrameDecodeError):
        decode_frame(b'{"frame_type":"nope","display_name":"x"}')


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        PresenceTransitionFrame(
            agent_id="skippy",
            display_name="Skippy",
            transition="offline",
            observed_at=datetime(2026, 7, 20, tzinfo=UTC),
            transition_id="edge1",
            surprise="x",  # type: ignore[call-arg]
        )


def test_missing_required_field_rejected() -> None:
    with pytest.raises(ValidationError):
        PresenceTransitionFrame(
            agent_id="skippy",
            display_name="Skippy",
            transition="offline",
            # observed_at missing
        )  # type: ignore[call-arg]


def test_invalid_transition_literal_rejected() -> None:
    with pytest.raises(ValidationError):
        PresenceTransitionFrame(
            agent_id="skippy",
            display_name="Skippy",
            transition="degraded",  # type: ignore[arg-type]
            observed_at=datetime(2026, 7, 20, tzinfo=UTC),
            transition_id="edge1",
        )


def test_blank_display_name_rejected() -> None:
    with pytest.raises(ValidationError):
        PresenceTransitionFrame(
            agent_id="skippy",
            display_name="",
            transition="offline",
            observed_at=datetime(2026, 7, 20, tzinfo=UTC),
            transition_id="edge1",
        )


# --- Finding 1: agent_id subject-safe + bounded; display_name safe vocabulary ---


@pytest.mark.parametrize(
    "agent_id",
    [
        "foo@bar",  # mention-shaped
        "foo.bar",  # subject-token '.'
        "foo>bar",  # NATS '>'
        "foo*bar",  # NATS '*'
        "foo bar",  # whitespace
        "../etc",  # path traversal
        "a" * 65,  # over the 64-char cap
    ],
)
def test_unsafe_or_overcap_agent_id_rejected(agent_id: str) -> None:
    with pytest.raises(ValidationError):
        PresenceTransitionFrame(
            agent_id=agent_id,
            display_name="Skippy",
            transition="offline",
            observed_at=datetime(2026, 7, 20, tzinfo=UTC),
            transition_id="edge1",
        )


@pytest.mark.parametrize(
    ("agent_id", "valid"),
    [
        ("skippy", True),
        ("nagatha-2", True),
        ("foo_bar", True),
        ("a" * 64, True),  # exactly at the cap
        ("a" * 65, False),  # over the cap: subject-safe but rejected (Finding 1)
        ("foo@bar", False),  # mention-shaped
        ("foo.bar", False),  # subject-token '.'
        ("foo bar", False),  # whitespace
        ("../etc", False),  # path traversal
        ("", False),  # empty
    ],
)
def test_is_valid_presence_agent_id_matches_frame_constraints(agent_id: str, valid: bool) -> None:
    # The shared identity predicate the deriver calls BEFORE frame construction must
    # agree EXACTLY with what the frame validator accepts/rejects (Finding 1): a
    # subject-safe-but-over-cap 65-char id is invalid, so the deriver skips it instead
    # of raising inside construction and aborting the whole tick.
    from rtech_frames import is_valid_presence_agent_id

    assert is_valid_presence_agent_id(agent_id) is valid


@pytest.mark.parametrize(
    "display_name",
    [
        "<@123>",  # Discord mention
        "[x](http://evil)",  # Markdown link
        "@everyone",  # mass mention
        "`code`",  # backticks
        "a(b)c",  # parentheses
        "a/b",  # slash
        "a\\b",  # backslash
        "a\x00b",  # control char
        " Skippy",  # leading whitespace
        "Skippy ",  # trailing whitespace
        "x" * 65,  # over the 64-char cap
    ],
)
def test_unsafe_or_overcap_display_name_rejected(display_name: str) -> None:
    with pytest.raises(ValidationError):
        PresenceTransitionFrame(
            agent_id="skippy",
            display_name=display_name,
            transition="offline",
            observed_at=datetime(2026, 7, 20, tzinfo=UTC),
            transition_id="edge1",
        )


@pytest.mark.parametrize("display_name", ["Nagatha 2", "Foo.Bar", "Skippy", "foo_bar"])
def test_normal_display_names_accepted(display_name: str) -> None:
    frame = PresenceTransitionFrame(
        agent_id="skippy",
        display_name=display_name,
        transition="offline",
        observed_at=datetime(2026, 7, 20, tzinfo=UTC),
        transition_id="edge1",
    )
    assert frame.display_name == display_name


def test_boundary_survives_encode_decode() -> None:
    # A constructed frame with a safe agent_id + safe display_name round-trips.
    frame = PresenceTransitionFrame(
        agent_id="agentd-nagatha",
        display_name="Nagatha 2",
        transition="recovered",
        observed_at=datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC),
        transition_id="boundary1",
    )
    decoded = decode_frame(encode_frame(frame))
    assert isinstance(decoded, PresenceTransitionFrame)
    assert decoded == frame


def test_decode_rejects_malicious_agent_id_bytes() -> None:
    # A crafted wire payload with a subject-unsafe agent_id fails closed at decode.
    good = encode_frame(_frame())
    hostile = good.replace(b'"nagatha-2"', b'"../etc/passwd"')
    with pytest.raises(FrameDecodeError):
        decode_frame(hostile)


# --- Finding 2: observed_at UTC normalization + naive rejection ----------------


def test_offset_observed_at_normalized_to_utc() -> None:
    plus5 = timezone(timedelta(hours=5))
    frame = PresenceTransitionFrame(
        agent_id="skippy",
        display_name="Skippy",
        transition="offline",
        observed_at=datetime(2026, 7, 20, 17, 0, 0, tzinfo=plus5),
        transition_id="edge1",
    )
    # 17:00 +05:00 == 12:00 UTC, stored as the true UTC instant.
    assert frame.observed_at == datetime(2026, 7, 20, 12, 0, 0, tzinfo=UTC)
    assert frame.observed_at.utcoffset() == timedelta(0)


def test_naive_observed_at_rejected_at_construction() -> None:
    with pytest.raises(ValidationError):
        PresenceTransitionFrame(
            agent_id="skippy",
            display_name="Skippy",
            transition="offline",
            observed_at=datetime(2026, 7, 20, 12, 0, 0),  # naive
            transition_id="edge1",
        )


def test_naive_observed_at_rejected_at_decode() -> None:
    good = encode_frame(_frame())
    # Strip the trailing 'Z' (UTC designator) from the wire timestamp -> naive on
    # decode, which the field validator rejects.
    naive = good.replace(b"12:00:00Z", b"12:00:00")
    with pytest.raises(FrameDecodeError):
        decode_frame(naive)


# --- Finding 3: encode revalidation raises FrameEncodeError, emits no bytes ----


def test_encode_revalidation_rejects_invalid_transition() -> None:
    frame = _frame()
    mutated = frame.model_copy(update={"transition": "degraded"})
    with pytest.raises(FrameEncodeError):
        encode_frame(mutated)


def test_encode_revalidation_rejects_naive_timestamp() -> None:
    frame = _frame()
    mutated = frame.model_copy(update={"observed_at": datetime(2026, 7, 20, 12, 0, 0)})
    with pytest.raises(FrameEncodeError):
        encode_frame(mutated)


def test_encode_revalidation_rejects_unsafe_agent_id() -> None:
    frame = _frame()
    mutated = frame.model_copy(update={"agent_id": "../evil"})
    with pytest.raises(FrameEncodeError):
        encode_frame(mutated)


def test_encode_revalidation_rejects_overcap_agent_id() -> None:
    frame = _frame()
    mutated = frame.model_copy(update={"agent_id": "a" * 65})
    with pytest.raises(FrameEncodeError):
        encode_frame(mutated)


def test_encode_revalidation_rejects_unsafe_display_name() -> None:
    frame = _frame()
    mutated = frame.model_copy(update={"display_name": "<@everyone>"})
    with pytest.raises(FrameEncodeError):
        encode_frame(mutated)


def test_valid_frame_still_encodes_byte_identically() -> None:
    # Revalidation must leave a valid frame's wire bytes unchanged.
    frame = _frame()
    first = encode_frame(frame)
    assert first == encode_frame(frame)
    assert isinstance(first, bytes) and len(first) > 0


# --- Delivery idempotency: transition_id bounded/safe + round-trips + revalidated --


def test_transition_id_round_trips_through_encode_decode() -> None:
    frame = _frame()
    decoded = decode_frame(encode_frame(frame))
    assert isinstance(decoded, PresenceTransitionFrame)
    assert decoded.transition_id == frame.transition_id


def test_missing_transition_id_rejected() -> None:
    with pytest.raises(ValidationError):
        PresenceTransitionFrame(
            agent_id="skippy",
            display_name="Skippy",
            transition="offline",
            observed_at=datetime(2026, 7, 20, tzinfo=UTC),
            # transition_id missing
        )  # type: ignore[call-arg]


def test_blank_transition_id_rejected() -> None:
    with pytest.raises(ValidationError):
        PresenceTransitionFrame(
            agent_id="skippy",
            display_name="Skippy",
            transition="offline",
            observed_at=datetime(2026, 7, 20, tzinfo=UTC),
            transition_id="",
        )


@pytest.mark.parametrize(
    "transition_id",
    [
        "has space",  # whitespace
        "has.dot",  # subject-token '.'
        "has>gt",  # NATS '>'
        "has*star",  # NATS '*'
        "../evil",  # path traversal shape
        "id@x",  # mention-shaped
        "z" * 65,  # over the 64-char cap
    ],
)
def test_unsafe_or_overcap_transition_id_rejected(transition_id: str) -> None:
    with pytest.raises(ValidationError):
        PresenceTransitionFrame(
            agent_id="skippy",
            display_name="Skippy",
            transition="offline",
            observed_at=datetime(2026, 7, 20, tzinfo=UTC),
            transition_id=transition_id,
        )


def test_uuid_hex_transition_id_accepted() -> None:
    # The deriver mints uuid4().hex (32 hex chars); it must be a valid transition_id.
    frame = PresenceTransitionFrame(
        agent_id="skippy",
        display_name="Skippy",
        transition="offline",
        observed_at=datetime(2026, 7, 20, tzinfo=UTC),
        transition_id="0f9c2b7a4e1d4a6b8c3e5f7a9b1d2c3e",
    )
    assert len(frame.transition_id) == 32


def test_encode_revalidation_rejects_unsafe_transition_id() -> None:
    # model_copy(update=...) bypasses field validation; encode must fail closed.
    frame = _frame()
    mutated = frame.model_copy(update={"transition_id": "../evil"})
    with pytest.raises(FrameEncodeError):
        encode_frame(mutated)


def test_encode_revalidation_rejects_overcap_transition_id() -> None:
    frame = _frame()
    mutated = frame.model_copy(update={"transition_id": "z" * 65})
    with pytest.raises(FrameEncodeError):
        encode_frame(mutated)


def test_decode_rejects_unsafe_transition_id_bytes() -> None:
    # A crafted wire payload with an unsafe transition_id fails closed at decode.
    good = encode_frame(_frame())
    hostile = good.replace(b'"abc123def456"', b'"../evil"')
    with pytest.raises(FrameDecodeError):
        decode_frame(hostile)
