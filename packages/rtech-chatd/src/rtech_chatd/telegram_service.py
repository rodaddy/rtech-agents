"""Assemble the runnable chatd **Telegram** long-poll edge service.

This is the production entrypoint for a STANDALONE Telegram edge (issue #112):
the long-poll adapter + multi-identity sender + intake sink wired to the NATS
frame bus and the existing dumb-edge lifecycle. It composes ONLY the Telegram
surface — it never builds a Discord adapter, an orchestrator, or an agentd. The
edge stays a dumb boundary: it mints a typed ``TaskFrame`` from each gated
inbound message and renders agent turns back; it carries no control signal and
never authors "done" (completion is orchestrator-derived).

The composition mirrors :func:`rtech_chatd.service.run_service` (the Discord
production root) and reuses the SAME identity-layer assembly the dev stack wires
in ``scripts/dev_stack.py`` — there is a single Telegram protocol implementation,
not a second divergent one. The multi-identity sender, intake sink ack/typing
hooks, and identity-stamped debate render (``identity_send=True``,
``attributed=False``) are preserved from the PR #122 baseline.

Security floor (issue #78 / #112): the edge bot token and every per-identity
token arrive as ``*_ref`` indirections ONLY, never literals and never logged; a
missing bot ref, missing/invalid group allowlist, or invalid identity map fails
closed BEFORE any connection is opened. Long polling means there is NO inbound
port, webhook, or public domain.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from loguru import logger

from .bus import NatsFrameBus
from .callback_sink import TelegramCallbackSink, TelegramTurnInteraction
from .channel import SendTarget
from .config import (
    ENV_TG_IDENTITY_MAP,
    GateConfig,
    IntakeConfig,
    TelegramConfig,
    parse_identity_map,
    resolve_token,
    validate_token_ref,
)
from .edge import DiscordEdge, debate_chat_lanes_from_triples, debate_lanes_from_pairs
from .intake_sink import IntakeSink
from .telegram_adapter import TelegramChannelAdapter
from .telegram_callback import TelegramCallbackRegistry
from .telegram_sender import TelegramSender


class RunnableAdapter(Protocol):
    """The adapter lifecycle surface the service runner drives (test seam)."""

    async def start(self) -> None:
        """Connect and run the poll loop until it closes."""
        ...

    async def close(self) -> None:
        """Tear the adapter (poll loop + sender pool + bot session) down."""
        ...


class InboundEdge(Protocol):
    """The edge inbound-drain + teardown surface the runner drives (test seam)."""

    async def run_inbound(self) -> None:
        """Drain the adapter's inbound queue onto the bus forever."""
        ...

    async def close(self) -> None:
        """Cancel outbound render streams and release edge resources."""
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
ENV_NATS_URL = "RTECH_CHATD_NATS_URL"
ENV_EDGE_TOKEN_REF = "RTECH_CHATD_TOKEN_REF"
# Presence-notify wiring (issue #147). Both are non-secret operator knobs (a public
# chat id and a public base URL), so — like the group allowlist — they are plain
# values, never ``*_ref`` indirections. UNSET/blank => no presence notify (the edge
# does not subscribe presence transitions), byte-identical to a pre-#147 deploy.
ENV_PRESENCE_NOTIFY_CHAT = "RTECH_TG_PRESENCE_NOTIFY_CHAT"
ENV_DASH_BASE_URL = "RTECH_CHATD_DASH_BASE_URL"
DEFAULT_EDGE_TOKEN_REF = "env:RTECH_CHATD_BOT_TOKEN"
DEFAULT_NATS_URL = "nats://127.0.0.1:4222"
ORIGIN_EDGE_TELEGRAM = "telegram"

# A Telegram presence-notify chat id must be a signed-integer chat id string (the
# canonical ``-100…`` supergroup id, or a smaller signed id), matching the
# allowlist shape. A set-but-malformed value fails closed rather than silently
# addressing a chat that could never receive.
_TELEGRAM_CHAT_ID_RE = re.compile(r"^-?\d+$")


@dataclass(frozen=True)
class TgdServiceConfig:
    """Hold the fully-resolved runnable Telegram-edge configuration.

    Every credential is an ``env:VAR`` indirection ref,
    never a literal — ``edge_token_ref`` and every value in ``identity_refs`` are
    validated fail-closed in :meth:`__post_init__`, WITHOUT echoing the value, so
    a mistakenly configured literal token never lands in an error or log line.

    Attributes:
        stack: Required stack stamp (issue #75); the edge stamps every minted
            TaskFrame with it and the orchestrator consumes only matching tasks.
        nats_url: NATS coordination URL (``nats://`` only).
        edge_token_ref: Indirection ref naming WHERE the polling bot token lives.
        identity_refs: ``agent_id -> token_ref`` map for the per-identity sender
            pool (multi-identity debate render). Empty = single-bot sends only.
        telegram: Non-secret Telegram knobs (group allowlist, allowed_updates,
            poll timeout). ``allowed_groups`` is REQUIRED non-empty here — a
            standalone deploy must scope its blast radius, never accept every
            group (the dev stack tolerates an empty allowlist; production does
            not).
        intake: Intake hit-gate config (issue #93); the debate roster and
            triage agent fail closed in :class:`IntakeConfig` when enabled.
        require_mention: Guild-channel mention requirement for the gate.
    """

    stack: str
    nats_url: str
    edge_token_ref: str
    telegram: TelegramConfig
    identity_refs: dict[str, str] = field(default_factory=dict)
    intake: IntakeConfig = field(default_factory=IntakeConfig)
    require_mention: bool = True
    # Presence-notify operator chat id (issue #147), or None when unset (no presence
    # notify). A public signed-integer chat id, NOT a secret — a plain value like the
    # group allowlist, never a ``*_ref``. A set-but-malformed value fails closed.
    presence_notify_chat: str | None = None
    # Public Dash live-page base URL for the notification link; empty renders the
    # notification without a link (fail safe, still notify). Public, not a secret.
    dash_base_url: str = ""

    def __post_init__(self) -> None:
        if not self.stack or not self.stack.strip():
            raise ValueError(
                "stack must be non-empty: the Telegram edge stamps every TaskFrame "
                f"with its stack id (issue #75); set {ENV_STACK_ID} — there is no default"
            )
        if not self.nats_url.startswith("nats://"):
            raise ValueError(f"nats_url must be a nats:// URL, got {self.nats_url!r}")
        # Fail closed WITHOUT echoing if the "ref" is actually a literal token.
        validate_token_ref(self.edge_token_ref)
        # A standalone production Telegram edge MUST scope which groups it accepts:
        # an empty allowlist would let the bot answer in ANY group it is added to,
        # which the security floor ("treat chat as leaked") forbids. The dev stack
        # tolerates an empty allowlist for a loopback smoke; production does not.
        if not self.telegram.allowed_groups:
            raise ValueError(
                "RTECH_TG_ALLOWED_GROUPS is required for the standalone Telegram edge: "
                "an empty group allowlist would answer in every group the bot is added "
                "to (security floor: treat chat as leaked). List at least one canonical "
                "'-100…' chat id."
            )
        # Every per-identity ref is already validate_token_ref-checked by
        # parse_identity_map at build time; re-assert here so a directly-constructed
        # config (tests / dev stack) cannot smuggle a literal into the pool.
        for ref in self.identity_refs.values():
            validate_token_ref(ref)
        if self.intake.debate_participants:
            missing = set(self.intake.debate_participants).difference(self.identity_refs)
            if missing:
                raise ValueError(
                    "every configured RTECH_CHATD_DEBATE_PARTICIPANTS member must have "
                    "a RTECH_TG_IDENTITY_MAP entry before Telegram ingress starts"
                )
        # A configured presence-notify chat id must be a signed-integer chat id: a
        # set-but-malformed value would address a chat that can never receive, so it
        # fails closed rather than silently never notifying. Unset (None) is the
        # no-presence-notify default and skips this check.
        if self.presence_notify_chat is not None and (
            _TELEGRAM_CHAT_ID_RE.match(self.presence_notify_chat) is None
        ):
            raise ValueError(
                f"{ENV_PRESENCE_NOTIFY_CHAT} must be a signed-integer chat id "
                r"(matching '^-?\d+$', e.g. '-1001234567890'); unset it entirely to "
                f"disable presence notifications, got {self.presence_notify_chat!r}"
            )

    @classmethod
    def from_env(cls) -> TgdServiceConfig:
        """Build the Telegram-edge service config from the environment (refs only).

        Vars (all refs/ids — never a literal token):
            RTECH_STACK_ID: REQUIRED stack stamp (issue #75); no default.
            RTECH_CHATD_NATS_URL: NATS URL (default the live fleet bus).
            RTECH_CHATD_TOKEN_REF: polling-bot token ref (default
                ``env:RTECH_CHATD_BOT_TOKEN``); a literal fails closed.
            RTECH_TG_IDENTITY_MAP: ``agentd-<name>=<token_ref>`` pairs for the
                per-identity sender pool. Unset = no pool (single-bot sends).
            RTECH_TG_ALLOWED_GROUPS: REQUIRED canonical ``-100…`` group
                allowlist (see :class:`TelegramConfig`); unset/empty fails closed.
            RTECH_TG_POLL_TIMEOUT_S: getUpdates long-poll hold (see config).
            RTECH_CHATD_INTAKE_* / RTECH_CHATD_DEBATE_*: intake + debate roster.
            RTECH_TG_PRESENCE_NOTIFY_CHAT: operator chat id for presence
                notifications (issue #147); UNSET disables them. A signed-integer
                chat id — a set-but-malformed value fails closed.
            RTECH_CHATD_DASH_BASE_URL: public Dash live-page base for the
                notification link; empty renders the notification without a link.

        Raises:
            ValueError: On a missing/blank stack, a literal/absent token ref, an
                invalid identity map, or a missing/invalid group allowlist — each
                fails closed WITHOUT echoing any credential value.
        """
        stack = os.getenv(ENV_STACK_ID, "").strip()
        if not stack:
            raise ValueError(
                f"{ENV_STACK_ID} must be set to this deployment's stack id "
                "(issue #75); the Telegram edge never defaults to a shared literal"
            )
        return cls(
            stack=stack,
            nats_url=os.getenv(ENV_NATS_URL, DEFAULT_NATS_URL),
            edge_token_ref=os.getenv(ENV_EDGE_TOKEN_REF, DEFAULT_EDGE_TOKEN_REF),
            identity_refs=parse_identity_map(os.getenv(ENV_TG_IDENTITY_MAP)),
            telegram=TelegramConfig.from_env(),
            intake=IntakeConfig.from_env(),
            presence_notify_chat=(os.getenv(ENV_PRESENCE_NOTIFY_CHAT, "").strip() or None),
            dash_base_url=(os.getenv(ENV_DASH_BASE_URL, "") or "").strip(),
        )

    def gate_config(self) -> GateConfig:
        """Build the inbound gate config (mention policy for group channels)."""
        return GateConfig(
            require_mention=self.require_mention,
            allowed_channels=self.telegram.allowed_groups,
        )


@dataclass
class TelegramEdgeComponents:
    """The assembled Telegram edge's live resources, for run + teardown."""

    adapter: TelegramChannelAdapter
    edge: DiscordEdge
    bus: NatsFrameBus


def build_telegram_components(config: TgdServiceConfig) -> TelegramEdgeComponents:
    """Assemble the Telegram trio edge from a resolved config (no I/O here).

    This is the SINGLE Telegram assembly used by both this production entrypoint
    and ``scripts/dev_stack.py`` — the dev stack delegates here so there is one
    protocol implementation, not two divergent ones. The identity layer is
    preserved from the PR #122 baseline: the intake sink's ack/typing hooks
    late-bind to the adapter through a cell, the sender's per-identity pool
    resolves token REFS at first use, and the edge renders debate turns
    identity-stamped (``identity_send=True``, ``attributed=False`` — the posting
    bot IS the attribution).

    No network is opened here: the bus is constructed unconnected and the
    adapter builds its aiogram ``Bot`` lazily at :meth:`start`. Fail-closed
    credential validation already ran in :meth:`TgdServiceConfig.__post_init__`.
    """
    return assemble_telegram_edge(
        stack=config.stack,
        nats_url=config.nats_url,
        edge_token_ref=config.edge_token_ref,
        identity_refs=config.identity_refs,
        telegram=config.telegram,
        intake=config.intake,
        gate_config=config.gate_config(),
        presence_notify_chat=config.presence_notify_chat,
        dash_base_url=config.dash_base_url,
    )


def assemble_telegram_edge(
    *,
    stack: str,
    nats_url: str,
    edge_token_ref: str,
    identity_refs: dict[str, str],
    telegram: TelegramConfig,
    intake: IntakeConfig,
    gate_config: GateConfig,
    presence_notify_chat: str | None = None,
    dash_base_url: str = "",
) -> TelegramEdgeComponents:
    """Wire the Telegram adapter + sender + intake sink + edge from resolved parts.

    The lowest common assembly, shared verbatim by the production entrypoint
    (:func:`build_telegram_components`) and ``scripts/dev_stack.py`` so there is a
    SINGLE Telegram protocol implementation, not two. Callers own their own
    fail-closed policy (the production config requires a group allowlist; the dev
    stack tolerates an empty one) and pass an already-resolved parts set here;
    this function only wires them. It opens no network — the bus is unconnected
    and the adapter builds its ``Bot`` lazily at :meth:`start`.

    Interactive steer layer (issue #117): when intake is enabled, ONE shared
    :class:`~rtech_chatd.telegram_callback.TelegramCallbackRegistry` backs both the
    inbound :class:`~rtech_chatd.callback_sink.TelegramCallbackSink` (wired into
    the adapter — it authorizes and consumes taps into typed steers) and the
    outbound :class:`~rtech_chatd.callback_sink.TelegramTurnInteraction` (wired
    into the edge — it mints actor-bound capabilities on a display-only keyboard
    attached to each landed debate TLDR, on the polled default bot). A tap becomes
    a typed ``SteerFrame`` on the bus, never a control signal in the connector.
    """

    def _edge_token() -> str:
        token = resolve_token(edge_token_ref)
        if not token:
            raise RuntimeError("chatd.tgd: edge bot token not provisioned (ref withheld)")
        return token

    def _identity_provider(ref: str) -> str:
        token = resolve_token(ref)
        if not token:
            # The ref is withheld: an identity ref value must never reach a log
            # or error, even by name, so a misconfigured pool cannot leak.
            raise RuntimeError("chatd.tgd: identity bot token not provisioned (ref withheld)")
        return token

    def _provider(ref: str) -> Callable[[], str]:
        def _resolve() -> str:
            return _identity_provider(ref)

        return _resolve

    sender = TelegramSender(
        identity_token_providers={agent: _provider(ref) for agent, ref in identity_refs.items()},
    )

    bus = NatsFrameBus(nats_url)

    # ONE shared callback registry backs BOTH the inbound sink (which authorizes
    # and consumes taps) and the outbound interaction (which mints capabilities on
    # the buttons it attaches): a capability issued by the interaction must be the
    # SAME record the sink later reserves/commits, so they cannot own separate
    # registries. Only wired when intake is enabled — the interactive steer layer
    # rides the debate render, which only a triage/intake edge runs (issue #117).
    callback_sink: TelegramCallbackSink | None = None
    turn_interaction: TelegramTurnInteraction | None = None
    if intake.enabled:
        registry = TelegramCallbackRegistry(clock=time.monotonic)
        callback_sink = TelegramCallbackSink(bus=bus, sender=sender, registry=registry)
        turn_interaction = TelegramTurnInteraction(sink=callback_sink, sender=sender)

    # The intake sink's instant-ack + typing hooks send via the adapter, which is
    # built AFTER the sink (the sink is an adapter ctor arg). Late-bind through a
    # cell the adapter is dropped into below; before binding the hooks no-op.
    adapter_cell: list[TelegramChannelAdapter] = []

    async def _ack_send(target: SendTarget, text: str) -> str | None:
        if not adapter_cell:
            return None
        return await adapter_cell[0].send(target, text, [])

    async def _typing_start(target: SendTarget) -> None:
        if adapter_cell:
            await adapter_cell[0].typing(target, True)

    intake_sink = (
        IntakeSink(
            bus=bus,
            intake_config=intake,
            gate_config=gate_config,
            stack=stack,
            ack_send=_ack_send,
            typing_start=_typing_start,
            # Identity routing owns the voice: the ack posts under the triage
            # agent's own bot, unprefixed.
            voiced=False,
        )
        if intake.enabled
        else None
    )
    adapter = TelegramChannelAdapter(
        token_provider=_edge_token,
        gate_config=gate_config,
        telegram_config=telegram,
        intake_config=intake,
        intake_sink=intake_sink,
        callback_sink=callback_sink,
        sender=sender,
    )
    adapter_cell.append(adapter)
    # Presence notifications (issue #147) land in the configured operator chat as a
    # plain chat-id target — no thread, no per-identity voice (the default polled
    # bot posts them). Unset => None target => the edge does not subscribe presence
    # transitions (byte-identical to a pre-#147 deploy).
    presence_notify_target = (
        SendTarget(channel_id=presence_notify_chat) if presence_notify_chat is not None else None
    )
    edge = DiscordEdge(
        adapter,
        bus,
        stack=stack,
        origin_edge=ORIGIN_EDGE_TELEGRAM,
        attributed=False,
        intake_enabled=intake.enabled,
        identity_send=True,
        debate_lanes=debate_lanes_from_pairs(intake.debate_lanes),
        debate_chat_lanes=debate_chat_lanes_from_triples(intake.debate_chat_lanes),
        turn_interaction=turn_interaction,
        presence_notify_target=presence_notify_target,
        dash_base_url=dash_base_url,
    )
    return TelegramEdgeComponents(adapter=adapter, edge=edge, bus=bus)


async def _run_adapter_loud(adapter: RunnableAdapter) -> None:
    """Run the poll loop and fail LOUD when it returns normally.

    ``edge.run_inbound()`` blocks forever on the adapter's inbound queue, so an
    adapter whose poll loop exits and returns would leave the TaskGroup waiting
    eternally — the service would neither exit for a systemd restart nor tear
    down. Raising converts that silent hang into a loud exit.

    Raises:
        RuntimeError: Always, when ``start()`` returns instead of running forever
            (a clean poll-loop exit is still a dead edge).
    """
    await adapter.start()
    logger.error("chatd.tgd | poll loop exited without error; failing loud for restart")
    raise RuntimeError("chatd.tgd: Telegram poll loop exited; the edge cannot run without it")


async def run_components(adapter: RunnableAdapter, bus: LifecycleBus, edge: InboundEdge) -> None:
    """Connect the bus, run the poll loop + inbound drain, always tear down.

    The injectable-seam runner: the lifecycle (poll-loop exit → TaskGroup
    teardown → loud failure) is unit-testable without aiogram or a live NATS.
    Every external resource (edge render streams, adapter poll loop + sender
    pool + bot session, bus connection) is closed in ``finally``, each guarded so
    one teardown failure never masks another or aborts the rest.
    """
    try:
        await bus.connect()
        async with asyncio.TaskGroup() as group:
            group.create_task(_run_adapter_loud(adapter))
            group.create_task(edge.run_inbound())
    finally:
        # Teardown failures render only the exception TYPE: a bus (or bus-backed
        # edge/adapter) close failure can carry the credentialed NATS URL
        # (nats://host?token=...) in its message text, and rendering exc here would
        # leak it into the logs.
        try:
            await edge.close()
        except Exception as exc:
            logger.error("chatd.tgd | edge close failed error_type={}", type(exc).__name__)
        try:
            await adapter.close()
        except Exception as exc:
            logger.error("chatd.tgd | adapter close failed error_type={}", type(exc).__name__)
        try:
            await bus.close()
        except Exception as exc:
            logger.error("chatd.tgd | bus close failed error_type={}", type(exc).__name__)
        logger.info("chatd.tgd | exit")


def _preflight_credentials(config: TgdServiceConfig) -> None:
    """Resolve every configured Telegram credential before opening a connection."""
    refs = (config.edge_token_ref, *config.identity_refs.values())
    for ref in refs:
        if not resolve_token(ref):
            raise RuntimeError("chatd.tgd: configured bot token not provisioned (ref withheld)")


async def run_service(config: TgdServiceConfig) -> None:
    """Assemble the live Telegram edge from config and run it until the poll dies.

    A poll loop that stops — even by returning cleanly — fails LOUD so systemd
    restarts the service instead of leaving it hanging half-dead. If assembly or
    the bus connect fails BEFORE the run loop, the partially-built components are
    still torn down (bus close is idempotent when never connected).
    """
    logger.info(
        "chatd.tgd | entry stack={} origin={} groups={} identities={} intake={} debate={}",
        config.stack,
        ORIGIN_EDGE_TELEGRAM,
        len(config.telegram.allowed_groups),
        len(config.identity_refs),
        config.intake.enabled,
        list(config.intake.debate_participants),
    )
    _preflight_credentials(config)
    components = build_telegram_components(config)
    await run_components(components.adapter, components.bus, components.edge)


__all__ = [
    "DEFAULT_EDGE_TOKEN_REF",
    "DEFAULT_NATS_URL",
    "ENV_EDGE_TOKEN_REF",
    "ENV_DASH_BASE_URL",
    "ENV_NATS_URL",
    "ENV_PRESENCE_NOTIFY_CHAT",
    "ENV_STACK_ID",
    "ORIGIN_EDGE_TELEGRAM",
    "InboundEdge",
    "LifecycleBus",
    "RunnableAdapter",
    "TelegramEdgeComponents",
    "TgdServiceConfig",
    "assemble_telegram_edge",
    "build_telegram_components",
    "run_components",
    "run_service",
]
