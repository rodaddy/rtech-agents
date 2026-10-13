"""Expose typed NATS wire frames and codecs."""

from __future__ import annotations

from .codec import FrameEncodeError, decode_frame, encode_frame
from .done import TerminalEvidence, is_done
from .frames import (
    BROADCAST_TARGET,
    MAX_EVIDENCE_SOURCES,
    MAX_SOURCE_ID_LEN,
    MAX_SOURCE_LABEL_LEN,
    MAX_SOURCE_LOCATOR_LEN,
    MAX_VERIFICATION_LIMITATIONS,
    MAX_VERIFICATION_METHODS,
    SCHEMA_VERSION,
    SCOPE_PARTICIPANT_COUNT,
    ChannelOrigin,
    EvidenceSource,
    LifecycleFrame,
    PresenceTransitionFrame,
    Stance,
    StatusFrame,
    SteerFrame,
    TaskFrame,
    TurnFrame,
    VerificationReport,
    is_valid_presence_agent_id,
    stance_to_fleet,
    validate_participants,
)
from .subs import (
    FrameSubscribeError,
    Subscription,
    SubscriptionSet,
    subscribe_all,
)
from .supervised import (
    MAX_REQUEST_ID_LEN,
    StartAgentFrame,
    StartAllAgentsFrame,
    SupervisedAgentStateFrame,
)

__version__ = "0.0.1"

__all__ = [
    "BROADCAST_TARGET",
    "MAX_EVIDENCE_SOURCES",
    "MAX_REQUEST_ID_LEN",
    "MAX_SOURCE_ID_LEN",
    "MAX_SOURCE_LABEL_LEN",
    "MAX_SOURCE_LOCATOR_LEN",
    "MAX_VERIFICATION_LIMITATIONS",
    "MAX_VERIFICATION_METHODS",
    "SCHEMA_VERSION",
    "SCOPE_PARTICIPANT_COUNT",
    "ChannelOrigin",
    "EvidenceSource",
    "FrameEncodeError",
    "FrameSubscribeError",
    "LifecycleFrame",
    "PresenceTransitionFrame",
    "StartAgentFrame",
    "StartAllAgentsFrame",
    "Stance",
    "StatusFrame",
    "SteerFrame",
    "SupervisedAgentStateFrame",
    "Subscription",
    "SubscriptionSet",
    "TaskFrame",
    "TerminalEvidence",
    "TurnFrame",
    "VerificationReport",
    "decode_frame",
    "encode_frame",
    "is_done",
    "is_valid_presence_agent_id",
    "stance_to_fleet",
    "subscribe_all",
    "validate_participants",
]
