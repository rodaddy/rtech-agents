"""Assemble the runnable chatd Discord edge service.

This module wires the pure edge (:class:`rtech_chatd.edge.DiscordEdge`) to the
discord.py adapter and the NATS frame bus into a single fail-closed process.
Importing it pulls in discord.py (via :mod:`rtech_chatd.discord_adapter`), so the
pure-core package :mod:`rtech_chatd` does NOT re-export these symbols — keeping
the promise that importing the core never drags in discord.py.

Security floor: the bot token is resolved from a ``*_ref``
indirection ONLY, never a literal and never logged; a blank inbox, blank stack,
or ``allow_bots=all`` fails closed BEFORE any connection is opened.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Protocol

from loguru import logger

from .bus import NatsFrameBus
from .channel import SendTarget
from .config import (
    AllowBots,
    ChatdConfig,
    GateConfig,
    IntakeConfig,
    ThreadConfig,
    resolve_token,
)
from .discord_adapter import DiscordChannelAdapter
from .edge import DiscordEdge, debate_chat_lanes_from_triples, debate_lanes_from_pairs
from .intake_sink import IntakeSink


class RunnableAdapter(Protocol):
    """The adapter lifecycle surface the service runner drives (test seam)."""

    async def start(self) -> None:
        """Connect and run the chat client until it closes."""
        ...

    async def close(self) -> None:
        """Tear the client down."""
        ...


class LifecycleBus(Protocol):
    """The bus lifecycle surface the service runner drives (test seam)."""

    async def connect(self) -> None:
        """Open the bus connection."""
        ...

    async def close(self) -> None:
        """Close the bus connection."""
        ...


ENV_STACK_ID = "RTECH_STACK_ID"
ENV_PARTICIPANTS = "RTECH_CHATD_PARTICIPANTS"
ENV_ALLOW_BOTS = "RTECH_CHATD_ALLOW_BOTS"


@dataclass(frozen=True)
class ChatdServiceConfig:
    """Hold the fully-resolved runnable-service configuration.

    ``edge`` carries the transport/token-ref settings; ``stack`` is the required
    stack stamp; ``participants`` is the typed participant set whose
    count derives the task scope; ``allow_bots``/``require_mention`` are the gate
    policy. Every invariant fails closed in :meth:`__post_init__` so a
    misconfigured service never reaches a live connection.
    """

    edge: ChatdConfig
    stack: str
    participants: tuple[str, ...] = ()
    allow_bots: AllowBots = AllowBots.NONE
    require_mention: bool = True
    intake: IntakeConfig = field(default_factory=IntakeConfig)

    def __post_init__(self) -> None:
        if not self.stack or not self.stack.strip():
            raise ValueError(
                "stack must be non-empty: chatd stamps every TaskFrame with its "
                f"stack id; set {ENV_STACK_ID} — there is no default"
            )
        if self.edge.inbox_channel is None or not self.edge.inbox_channel.strip():
            raise ValueError(
                "inbox_channel is required: the designated free-response inbox is "
                "the ONLY channel chatd answers freely on (set RTECH_CHATD_INBOX_CHANNEL)"
            )
        if self.allow_bots is AllowBots.ALL:
            raise ValueError(
                "allow_bots=all is rejected (security floor: treat Discord as leaked); "
                "the bot policy must be at least as strict as 'mentions'"
            )

    @classmethod
    def from_env(cls) -> ChatdServiceConfig:
        """Build the service config from environment variables (refs only).

        Raises:
            ValueError: If ``RTECH_STACK_ID`` is unset/blank, or
                ``RTECH_CHATD_ALLOW_BOTS`` is ``all`` (fail closed, never
                silently downgraded), or any invariant in
                :meth:`__post_init__` fails.
        """

        stack = os.getenv(ENV_STACK_ID, "").strip()
        if not stack:
            raise ValueError(
                f"{ENV_STACK_ID} must be set to this deployment's stack id "
                "; the edge never defaults to a shared literal"
            )
        edge = ChatdConfig.from_env()
        raw_participants = os.getenv(ENV_PARTICIPANTS, "")
        participants = tuple(p.strip() for p in raw_participants.split(",") if p.strip())
        raw_allow_bots = os.getenv(ENV_ALLOW_BOTS)
        if raw_allow_bots is not None and raw_allow_bots.strip().lower() == "all":
            raise ValueError(
                f"{ENV_ALLOW_BOTS}=all is rejected (security floor: treat Discord as "
                "leaked); use 'none' or 'mentions'"
            )
        allow_bots = AllowBots.parse(raw_allow_bots)
        return cls(
            edge=edge,
            stack=stack,
            participants=participants,
            allow_bots=allow_bots,
            intake=IntakeConfig.from_env(),
        )

    @property
    def inbox(self) -> str:
        """Return the non-None inbox channel (guaranteed by __post_init__)."""

        inbox = self.edge.inbox_channel
        if inbox is None:  # pragma: no cover - __post_init__ already fails closed
            raise ValueError("inbox_channel is required")
        return inbox

    def gate_config(self) -> GateConfig:
        """Build the inbound gate config; the inbox is the only free-response channel."""

        return GateConfig(
            require_mention=self.require_mention,
            allow_bots=self.allow_bots,
            free_response_channels=frozenset({self.inbox}),
        )

    def thread_config(self) -> ThreadConfig:
        """Build the thread config; the inbox is the only free-response channel."""

        return ThreadConfig(free_response_channels=frozenset({self.inbox}))


def resolve_required_token(config: ChatdServiceConfig) -> str:
    """Resolve the bot token from its ref, failing closed when unprovisioned.

    Fails BEFORE any connection is opened. The token VALUE is never logged; only
    the ref string is surfaced on failure — safe because ``ChatdConfig`` already
    rejected any non-indirection shape (``validate_token_ref``) without echoing
    it, so a mistakenly configured literal token can never reach this log line.

    Raises:
        RuntimeError: If the ref resolves to no value (unprovisioned).
    """

    token = resolve_token(config.edge.token_ref)
    if not token:
        logger.error("chatd.service | bot token not provisioned ref={}", config.edge.token_ref)
        raise RuntimeError(f"chatd: bot token not provisioned (ref={config.edge.token_ref})")
    return token


async def _run_adapter_loud(adapter: RunnableAdapter) -> None:
    """Run the chat client and fail LOUD when it returns normally.

    ``edge.run_inbound()`` blocks forever on the adapter's event queue, so a
    client that disconnects and returns would leave the TaskGroup waiting
    eternally — the service would neither exit for a systemd restart nor tear
    down. Raising converts that silent hang into a loud exit.

    Raises:
        RuntimeError: Always, when ``start()`` returns instead of running
            forever (a clean client exit is still a dead edge).
    """

    await adapter.start()
    logger.error("chatd.service | chat client exited without error; failing loud for restart")
    raise RuntimeError("chatd: chat client exited; the edge cannot run without it")


async def run_components(adapter: RunnableAdapter, bus: LifecycleBus, edge: DiscordEdge) -> None:
    """Connect the bus, run the client + inbound drain, always tear down.

    Extracted from :func:`run_service` as the injectable-seam runner so the
    service lifecycle (client exit → TaskGroup teardown → loud failure) is
    unit-testable without discord.py or a live NATS.
    """

    await bus.connect()
    try:
        async with asyncio.TaskGroup() as group:
            group.create_task(_run_adapter_loud(adapter))
            group.create_task(edge.run_inbound())
    finally:
        try:
            await edge.close()
        except Exception as exc:
            # Never render the raw exception message: a teardown failure can
            # carry connection/credential detail. Log the safe stage + type only.
            logger.error(
                "chatd.service | edge close failed stage=edge error_type={}", type(exc).__name__
            )
        try:
            await adapter.close()
        except Exception as exc:
            logger.error(
                "chatd.service | adapter close failed stage=adapter error_type={}",
                type(exc).__name__,
            )
        try:
            await bus.close()
        except Exception as exc:
            logger.error(
                "chatd.service | bus close failed stage=bus error_type={}", type(exc).__name__
            )
        logger.info("chatd.service | exit")


async def run_service(config: ChatdServiceConfig) -> None:
    """Assemble the live edge from config and run it until the client dies.

    A discord client that stops — even by returning cleanly — fails LOUD so
    systemd restarts the service instead of leaving it hanging half-dead. All
    teardown (edge/adapter/bus close) is handled by :func:`run_components`.
    """

    logger.info(
        "chatd.service | entry stack={} inbox={} participants={} allow_bots={}",
        config.stack,
        config.edge.inbox_channel,
        list(config.participants),
        config.allow_bots.value,
    )
    token = resolve_required_token(config)
    gate_config = config.gate_config()
    bus = NatsFrameBus(config.edge.nats_url)
    # The intake sink reuses the SAME bus publisher and the SAME gate
    # config the responder path uses, so its responder-yield decision is identical
    # to what the responder itself computes. Only built when intake is enabled;
    # existing deploys pass no sink and see no intake behavior.
    # The instant debate-mint ack (UX v2) sends via the adapter, which is built
    # AFTER the sink (the sink is an adapter ctor arg) — late-bind through a cell
    # the adapter is dropped into below. Before binding (impossible in practice:
    # the gateway only delivers hits after start) the ack is a silent no-op.
    adapter_cell: list[DiscordChannelAdapter] = []

    async def _ack_send(target: SendTarget, text: str) -> str | None:
        if not adapter_cell:
            return None
        return await adapter_cell[0].send(target, text, [])

    intake_sink = (
        IntakeSink(
            bus=bus,
            intake_config=config.intake,
            gate_config=gate_config,
            stack=config.stack,
            ack_send=_ack_send,
        )
        if config.intake.enabled
        else None
    )
    adapter = DiscordChannelAdapter(
        token_provider=lambda: token,
        gate_config=gate_config,
        thread_config=config.thread_config(),
        intake_config=config.intake,
        intake_sink=intake_sink,
    )
    adapter_cell.append(adapter)
    edge = DiscordEdge(
        adapter,
        bus,
        stack=config.stack,
        participants=list(config.participants),
        attributed=True,
        origin_edge=config.edge.origin_edge,
        home_target=config.edge.home_send_target(),
        # Same opt-in as the intake sink: only a triage edge renders
        # EXTERNAL task answers (intake/wake) back into the asking channel.
        # Existing deploys leave intake off and stay byte-identical.
        intake_enabled=config.intake.enabled,
        debate_lanes=debate_lanes_from_pairs(config.intake.debate_lanes),
        debate_chat_lanes=debate_chat_lanes_from_triples(config.intake.debate_chat_lanes),
        # Presence notifications land on the operator home channel —
        # the SAME cross-platform/cron delivery target — so no new channel is
        # introduced. Absent home channel => no notify target => the edge simply
        # does not subscribe presence transitions (byte-identical to before).
        presence_notify_target=config.edge.home_send_target(),
        dash_base_url=config.edge.dash_base_url,
    )
    await run_components(adapter, bus, edge)


__all__ = [
    "ENV_ALLOW_BOTS",
    "ENV_PARTICIPANTS",
    "ENV_STACK_ID",
    "ChatdServiceConfig",
    "IntakeSink",
    "LifecycleBus",
    "RunnableAdapter",
    "resolve_required_token",
    "run_components",
    "run_service",
]
