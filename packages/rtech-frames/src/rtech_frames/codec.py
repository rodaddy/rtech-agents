"""Encode and decode typed frames for the NATS JSON wire."""

from __future__ import annotations

import json

from loguru import logger
from pydantic import ValidationError

from .frames import (
    LifecycleFrame,
    PresenceTransitionFrame,
    StatusFrame,
    SteerFrame,
    TaskFrame,
    TurnFrame,
)
from .supervised import (
    StartAgentFrame,
    StartAllAgentsFrame,
    SupervisedAgentStateFrame,
)

type Frame = (
    TaskFrame
    | TurnFrame
    | SteerFrame
    | StatusFrame
    | LifecycleFrame
    | PresenceTransitionFrame
    | StartAgentFrame
    | StartAllAgentsFrame
    | SupervisedAgentStateFrame
)

FRAME_TYPE_FIELD = "frame_type"

_FRAME_TYPES: dict[str, type[Frame]] = {
    "task": TaskFrame,
    "turn": TurnFrame,
    "steer": SteerFrame,
    "status": StatusFrame,
    "lifecycle": LifecycleFrame,
    "presence_transition": PresenceTransitionFrame,
    "start_agent": StartAgentFrame,
    "start_all_agents": StartAllAgentsFrame,
    "supervised_agent_state": SupervisedAgentStateFrame,
}

_FRAME_NAMES: dict[type[Frame], str] = {
    TaskFrame: "task",
    TurnFrame: "turn",
    SteerFrame: "steer",
    StatusFrame: "status",
    LifecycleFrame: "lifecycle",
    PresenceTransitionFrame: "presence_transition",
    StartAgentFrame: "start_agent",
    StartAllAgentsFrame: "start_all_agents",
    SupervisedAgentStateFrame: "supervised_agent_state",
}


class FrameDecodeError(ValueError):
    """Report invalid frame bytes with a stable error type."""


class FrameEncodeError(ValueError):
    """Report a frame that fails revalidation before it can reach the wire."""


def encode_frame(frame: Frame) -> bytes:
    """Encode a frame as UTF-8 JSON bytes for NATS.

    Every concrete frame instance is revalidated against its exact runtime model
    class immediately before serialization. ``model_copy(update=...)`` and other
    mutation paths bypass Pydantic field validation, so a frame carrying an
    out-of-bounds field (e.g. an oversized ``request_id``, a nested evidence
    ``source_id``, an invalid presence transition, a naive ``observed_at``, or an
    over-cap ``agent_id``/unsafe ``display_name``) could otherwise be serialized
    into invalid wire bytes before a downstream ``decode_frame`` rejects it.
    Revalidation here fails closed on any constrained frame — no special-casing per
    field or per frame type — and never emits bytes for an invalid frame.

    Args:
        frame: The typed frame to encode.

    Returns:
        UTF-8 JSON bytes with a frame type discriminator.

    Raises:
        FrameEncodeError: If the frame fails revalidation against its model class.
    """

    logger.debug("encoding frame", frame_type=type(frame).__name__)
    frame_model = type(frame)
    try:
        validated = frame_model.model_validate(frame.model_dump(mode="python"))
    except ValidationError as exc:
        logger.error(
            "frame revalidation failed",
            frame_type=frame_model.__name__,
            error_count=len(exc.errors()),
        )
        raise FrameEncodeError(f"invalid {frame_model.__name__} frame") from exc

    payload = validated.model_dump(mode="json")
    payload[FRAME_TYPE_FIELD] = _FRAME_NAMES[frame_model]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    logger.info("encoded frame", frame_type=payload[FRAME_TYPE_FIELD], byte_count=len(encoded))
    return encoded


def decode_frame(data: bytes) -> Frame:
    """Decode UTF-8 JSON bytes into a typed frame.

    Args:
        data: UTF-8 JSON bytes received from NATS.

    Returns:
        A validated typed frame.

    Raises:
        FrameDecodeError: If JSON, frame type, or frame fields are invalid.
    """

    logger.debug("decoding frame", byte_count=len(data))
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.error("failed to decode frame JSON", error_type=type(exc).__name__)
        raise FrameDecodeError("frame payload must be valid UTF-8 JSON") from exc

    if not isinstance(payload, dict):
        logger.warning("frame payload rejected", reason="not_object")
        raise FrameDecodeError("frame payload must be a JSON object")

    frame_type = payload.pop(FRAME_TYPE_FIELD, None)
    if not isinstance(frame_type, str):
        logger.warning("frame payload rejected", reason="missing_frame_type")
        raise FrameDecodeError("frame payload must include a string frame_type")

    frame_model = _FRAME_TYPES.get(frame_type)
    if frame_model is None:
        logger.warning("frame payload rejected", reason="unknown_frame_type", frame_type=frame_type)
        raise FrameDecodeError(f"unknown frame_type: {frame_type}")

    try:
        frame = frame_model.model_validate(payload)
    except ValidationError as exc:
        logger.error(
            "frame validation failed",
            frame_type=frame_type,
            error_count=len(exc.errors()),
        )
        raise FrameDecodeError(f"invalid {frame_type} frame") from exc

    logger.info("decoded frame", frame_type=frame_type)
    return frame
