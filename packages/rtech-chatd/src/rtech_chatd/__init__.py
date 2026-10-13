"""rtech-chatd: chat ingress/egress daemon (ChannelAdapter + Discord v1).

Public API:
    ChannelAdapter, InboundMessage, SendTarget  - the chat-surface abstraction.
    should_respond, thread_decision             - the pure Discord gate contract.
    chunk_message                               - pure 2000-char fence-aware split.
    inbound_to_task_frame, turn_frame_to_send   - typed bus frame edge mapping.

The discord.py wiring lives in :mod:`rtech_chatd.discord_adapter` and is not
re-exported here so importing the pure core never pulls in discord.py.
"""

from __future__ import annotations

from .bridge import (
    DEFAULT_ORIGIN_EDGE,
    DEFAULT_SCOPE,
    OutboundFilter,
    OutboundSend,
    inbound_to_task_frame,
    status_frame_to_send,
    turn_frame_to_send,
)
from .bus import FrameBus, NatsFrameBus
from .channel import ChannelAdapter, InboundMessage, SendTarget
from .chunking import DISCORD_MAX_LENGTH, chunk_message
from .config import (
    HOME_ADVISORY,
    AllowBots,
    ChatdConfig,
    GateConfig,
    HomeChannel,
    ThreadConfig,
    resolve_token,
)
from .edge import DiscordEdge
from .gates import ThreadDecision, should_respond, thread_decision
from .outbound_guard import GuardDecision, sanitize_outbound_text
from .presence_notify import (
    PresenceNotifyConsumer,
    dash_link_url,
    render_notification,
    validate_dash_base_url,
)
from .subjects import (
    PRESENCE_TRANSITION,
    TASK_INGRESS,
    task_result,
    task_status,
    task_steer,
    task_turn,
)

__version__ = "0.0.1"

__all__ = [
    "DEFAULT_ORIGIN_EDGE",
    "DEFAULT_SCOPE",
    "DISCORD_MAX_LENGTH",
    "HOME_ADVISORY",
    "PRESENCE_TRANSITION",
    "TASK_INGRESS",
    "AllowBots",
    "ChannelAdapter",
    "ChatdConfig",
    "DiscordEdge",
    "FrameBus",
    "GateConfig",
    "GuardDecision",
    "HomeChannel",
    "InboundMessage",
    "NatsFrameBus",
    "OutboundFilter",
    "OutboundSend",
    "PresenceNotifyConsumer",
    "SendTarget",
    "ThreadConfig",
    "ThreadDecision",
    "__version__",
    "chunk_message",
    "inbound_to_task_frame",
    "dash_link_url",
    "render_notification",
    "resolve_token",
    "sanitize_outbound_text",
    "should_respond",
    "status_frame_to_send",
    "task_result",
    "task_status",
    "task_steer",
    "task_turn",
    "thread_decision",
    "turn_frame_to_send",
    "validate_dash_base_url",
]
