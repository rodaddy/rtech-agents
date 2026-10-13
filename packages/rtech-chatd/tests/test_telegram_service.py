"""Tests for the standalone Telegram edge entrypoint (issue #112).

Assembly + fail-closed config for ``rtech-tgd``: the entrypoint composes ONLY the
Telegram surface (adapter + sender + intake sink + edge), never a Discord adapter
/ orchestrator / agentd; it fails closed on missing auth, identity, or group
config; it subscribes ``allowed_updates`` including every currently-supported
update; and it closes every resource on startup failure and shutdown. No network
is opened anywhere in this module.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from rtech_chatd import telegram_service
from rtech_chatd.channel import SendTarget
from rtech_chatd.config import (
    DEFAULT_TELEGRAM_ALLOWED_UPDATES,
    GateConfig,
    IntakeConfig,
    TelegramConfig,
)
from rtech_chatd.edge import DiscordEdge
from rtech_chatd.telegram_adapter import TelegramChannelAdapter
from rtech_chatd.telegram_service import (
    ORIGIN_EDGE_TELEGRAM,
    TelegramEdgeComponents,
    TgdServiceConfig,
    assemble_telegram_edge,
    build_telegram_components,
    run_components,
)

GROUP_ID = "-1001234567890"


def _set_min_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimum viable production Telegram env: stack, token ref, one group."""
    monkeypatch.setenv("RTECH_STACK_ID", "canary")
    monkeypatch.setenv("RTECH_CHATD_TOKEN_REF", "env:RTECH_CHATD_BOT_TOKEN")
    monkeypatch.setenv("RTECH_TG_ALLOWED_GROUPS", GROUP_ID)
    monkeypatch.delenv("RTECH_TG_IDENTITY_MAP", raising=False)
    monkeypatch.delenv("RTECH_CHATD_INTAKE_ENABLED", raising=False)


# --- config: fail-closed on missing auth / identity / group ------------------


def test_from_env_fails_closed_on_missing_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_min_env(monkeypatch)
    monkeypatch.delenv("RTECH_STACK_ID", raising=False)
    with pytest.raises(ValueError, match="RTECH_STACK_ID"):
        TgdServiceConfig.from_env()


def test_from_env_fails_closed_on_blank_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_min_env(monkeypatch)
    monkeypatch.setenv("RTECH_STACK_ID", "   ")
    with pytest.raises(ValueError, match="RTECH_STACK_ID"):
        TgdServiceConfig.from_env()


def test_from_env_fails_closed_on_missing_group_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_min_env(monkeypatch)
    monkeypatch.delenv("RTECH_TG_ALLOWED_GROUPS", raising=False)
    with pytest.raises(ValueError, match="RTECH_TG_ALLOWED_GROUPS is required"):
        TgdServiceConfig.from_env()


def test_from_env_fails_closed_on_empty_group_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    # Set-but-empty is a configured-but-meaningless guard: config parsing itself
    # rejects it (never silently accept every group).
    _set_min_env(monkeypatch)
    monkeypatch.setenv("RTECH_TG_ALLOWED_GROUPS", " , ,")
    with pytest.raises(ValueError, match="RTECH_TG_ALLOWED_GROUPS"):
        TgdServiceConfig.from_env()


def test_from_env_rejects_literal_edge_token_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    # A literal token where an indirection ref belongs fails closed WITHOUT the
    # value ever reaching the error/log (no credential leak to journald).
    _set_min_env(monkeypatch)
    monkeypatch.setenv("RTECH_CHATD_TOKEN_REF", "literal-bot-token-value")
    with pytest.raises(ValueError) as excinfo:
        TgdServiceConfig.from_env()
    assert "literal-bot-token-value" not in str(excinfo.value)


def test_from_env_rejects_literal_identity_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_min_env(monkeypatch)
    monkeypatch.setenv("RTECH_TG_IDENTITY_MAP", "agentd-nagatha=literal-identity-token")
    with pytest.raises(ValueError) as excinfo:
        TgdServiceConfig.from_env()
    assert "literal-identity-token" not in str(excinfo.value)


def test_from_env_fails_closed_on_intake_without_triage(monkeypatch: pytest.MonkeyPatch) -> None:
    # Intake enabled but no triage agent id → IntakeConfig fails closed.
    _set_min_env(monkeypatch)
    monkeypatch.setenv("RTECH_CHATD_INTAKE_ENABLED", "1")
    monkeypatch.delenv("RTECH_CHATD_TRIAGE_AGENT", raising=False)
    with pytest.raises(ValueError, match="RTECH_CHATD_TRIAGE_AGENT"):
        TgdServiceConfig.from_env()


def test_direct_config_rejects_literal_identity_ref() -> None:
    # A directly-constructed config (dev stack / tests) cannot smuggle a literal
    # into the identity pool: __post_init__ re-validates every ref.
    with pytest.raises(ValueError):
        TgdServiceConfig(
            stack="canary",
            nats_url="nats://x:4222",
            edge_token_ref="env:RTECH_CHATD_BOT_TOKEN",
            telegram=TelegramConfig(allowed_groups=frozenset({GROUP_ID})),
            identity_refs={"agentd-nagatha": "literal-token"},
        )


def test_direct_config_rejects_non_nats_url() -> None:
    with pytest.raises(ValueError, match="nats://"):
        TgdServiceConfig(
            stack="canary",
            nats_url="http://x:4222",
            edge_token_ref="env:RTECH_CHATD_BOT_TOKEN",
            telegram=TelegramConfig(allowed_groups=frozenset({GROUP_ID})),
        )


def test_from_env_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_min_env(monkeypatch)
    monkeypatch.setenv("RTECH_CHATD_NATS_URL", "nats://127.0.0.1:4223")
    monkeypatch.setenv(
        "RTECH_TG_IDENTITY_MAP",
        "agentd-nagatha=env:TG_NAG,agentd-bilby=env:TG_BILBY",
    )
    config = TgdServiceConfig.from_env()
    assert config.stack == "canary"
    assert config.nats_url == "nats://127.0.0.1:4223"
    assert config.telegram.allowed_groups == frozenset({GROUP_ID})
    assert config.identity_refs == {"agentd-nagatha": "env:TG_NAG", "agentd-bilby": "env:TG_BILBY"}


# --- allowed_updates: current supported update kinds -------------------------


def test_config_allowed_updates_include_current_supported_kinds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_min_env(monkeypatch)
    config = TgdServiceConfig.from_env()
    updates = config.telegram.allowed_updates
    # The full inbound surface plus the opt-in reaction feed (never auto-included
    # by the Bot API — omitting it blinds the bot to reactions).
    assert "message" in updates
    assert "edited_message" in updates
    assert "message_reaction" in updates
    assert set(updates) == set(DEFAULT_TELEGRAM_ALLOWED_UPDATES)


def test_assembled_adapter_carries_allowed_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_min_env(monkeypatch)
    config = TgdServiceConfig.from_env()
    components = build_telegram_components(config)
    assert (
        components.adapter._telegram_config.allowed_updates  # noqa: SLF001
        == DEFAULT_TELEGRAM_ALLOWED_UPDATES
    )


# --- assembly: Telegram only, no Discord / orchestrator / agentd -------------


def _build(monkeypatch: pytest.MonkeyPatch) -> TelegramEdgeComponents:
    _set_min_env(monkeypatch)
    monkeypatch.setenv("RTECH_CHATD_INTAKE_ENABLED", "1")
    monkeypatch.setenv("RTECH_CHATD_TRIAGE_AGENT", "agentd-nagatha")
    monkeypatch.setenv(
        "RTECH_CHATD_DEBATE_PARTICIPANTS", "agentd-nagatha,agentd-bilby,agentd-skippy"
    )
    monkeypatch.setenv(
        "RTECH_TG_IDENTITY_MAP",
        "agentd-nagatha=env:TG_NAG,agentd-bilby=env:TG_BILBY,agentd-skippy=env:TG_SKIPPY",
    )
    config = TgdServiceConfig.from_env()
    return build_telegram_components(config)


def test_assembly_selects_telegram_adapter_only(monkeypatch: pytest.MonkeyPatch) -> None:
    components = _build(monkeypatch)
    assert isinstance(components.adapter, TelegramChannelAdapter)
    assert isinstance(components.edge, DiscordEdge)
    # No Discord adapter type anywhere in the composed edge's inbound source.
    from rtech_chatd.discord_adapter import DiscordChannelAdapter

    assert not isinstance(components.adapter, DiscordChannelAdapter)


def test_assembly_builds_no_orchestrator_or_agentd(monkeypatch: pytest.MonkeyPatch) -> None:
    # The components dataclass exposes EXACTLY adapter/edge/bus — no supervisor,
    # runner, or orchestrator handle. A dumb edge carries no control-plane role.
    components = _build(monkeypatch)
    assert set(vars(components)) == {"adapter", "edge", "bus"}


def test_assembly_uses_identity_stamped_debate_render(monkeypatch: pytest.MonkeyPatch) -> None:
    components = _build(monkeypatch)
    edge = components.edge
    assert edge._origin_edge == ORIGIN_EDGE_TELEGRAM  # noqa: SLF001
    assert edge._identity_send is True  # noqa: SLF001
    # attributed=False: the posting bot IS the attribution (multi-identity).
    assert edge._attributed is False  # noqa: SLF001
    assert edge._intake_enabled is True  # noqa: SLF001


def test_assembly_wires_multi_identity_sender(monkeypatch: pytest.MonkeyPatch) -> None:
    components = _build(monkeypatch)
    # The adapter received a sender (the multi-identity pool), preserved from the
    # PR #122 baseline — not the #108 stub.
    assert components.adapter._sender is not None  # noqa: SLF001


def test_assembly_wires_intake_sink_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    components = _build(monkeypatch)
    assert components.adapter._intake_sink is not None  # noqa: SLF001


# --- presence-notify config + wiring (issue #147) ----------------------------


def test_from_env_presence_notify_unset_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_min_env(monkeypatch)
    monkeypatch.delenv("RTECH_TG_PRESENCE_NOTIFY_CHAT", raising=False)
    monkeypatch.delenv("RTECH_CHATD_DASH_BASE_URL", raising=False)
    config = TgdServiceConfig.from_env()
    assert config.presence_notify_chat is None
    assert config.dash_base_url == ""


def test_from_env_presence_notify_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_min_env(monkeypatch)
    monkeypatch.setenv("RTECH_TG_PRESENCE_NOTIFY_CHAT", "-1009999")
    monkeypatch.setenv("RTECH_CHATD_DASH_BASE_URL", "https://dash.example.internal/")
    config = TgdServiceConfig.from_env()
    assert config.presence_notify_chat == "-1009999"
    assert config.dash_base_url == "https://dash.example.internal/"


def test_config_fails_closed_on_malformed_presence_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_min_env(monkeypatch)
    monkeypatch.setenv("RTECH_TG_PRESENCE_NOTIFY_CHAT", "not-an-id")
    with pytest.raises(ValueError, match="RTECH_TG_PRESENCE_NOTIFY_CHAT"):
        TgdServiceConfig.from_env()


def test_assembly_no_presence_target_leaves_edge_unsubscribed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = _build(monkeypatch)
    # No RTECH_TG_PRESENCE_NOTIFY_CHAT set by _build => no notify target => the edge
    # holds no presence consumer.
    assert components.edge._presence_consumer is None  # noqa: SLF001
    assert components.edge._presence_notify_target is None  # noqa: SLF001


def test_assembly_presence_target_builds_consumer(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_min_env(monkeypatch)
    monkeypatch.setenv("RTECH_TG_PRESENCE_NOTIFY_CHAT", "-1009999")
    monkeypatch.setenv("RTECH_CHATD_DASH_BASE_URL", "https://dash.example.internal")
    config = TgdServiceConfig.from_env()
    components = build_telegram_components(config)
    edge = components.edge
    assert edge._presence_consumer is not None  # noqa: SLF001
    assert edge._presence_notify_target == SendTarget(channel_id="-1009999")  # noqa: SLF001
    assert edge._dash_base_url == "https://dash.example.internal"  # noqa: SLF001


def test_assembly_wires_callback_sink_and_interaction_when_intake_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The interactive steer layer (issue #117) is wired end-to-end in production
    # composition: the adapter receives a callback sink (so a tap becomes a typed
    # steer) and the edge receives a turn-interaction hook (so a landed debate TLDR
    # gets an actor-bound keyboard). Both ride the SAME shared registry.
    from rtech_chatd.callback_sink import TelegramCallbackSink, TelegramTurnInteraction

    components = _build(monkeypatch)
    assert isinstance(components.adapter._callback_sink, TelegramCallbackSink)  # noqa: SLF001
    interaction = components.edge._turn_interaction  # noqa: SLF001
    assert isinstance(interaction, TelegramTurnInteraction)
    # One shared registry: the sink that consumes and the interaction that issues
    # must be the same record store, or a minted capability could never authorize.
    assert (
        interaction._sink._registry  # noqa: SLF001
        is components.adapter._callback_sink._registry  # noqa: SLF001
    )


def test_assembly_omits_interactive_layer_when_intake_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A non-intake deploy runs no debate render, so no interactive steer layer is
    # wired — the adapter still answers taps (spinner-stop) but produces no steer,
    # and the edge attaches no markup.
    _set_min_env(monkeypatch)
    monkeypatch.delenv("RTECH_CHATD_INTAKE_ENABLED", raising=False)
    config = TgdServiceConfig.from_env()
    components = build_telegram_components(config)
    assert components.adapter._callback_sink is None  # noqa: SLF001
    assert components.edge._turn_interaction is None  # noqa: SLF001


def test_listed_intake_scope_covers_configured_telegram_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_min_env(monkeypatch)
    monkeypatch.setenv("RTECH_CHATD_INTAKE_ENABLED", "1")
    monkeypatch.setenv("RTECH_CHATD_TRIAGE_AGENT", "agentd-nagatha")
    config = TgdServiceConfig.from_env()
    assert config.gate_config().allowed_channels == frozenset({GROUP_ID})


def test_debate_participants_require_identity_map_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_min_env(monkeypatch)
    monkeypatch.setenv("RTECH_CHATD_INTAKE_ENABLED", "1")
    monkeypatch.setenv("RTECH_CHATD_TRIAGE_AGENT", "agentd-nagatha")
    monkeypatch.setenv("RTECH_CHATD_DEBATE_PARTICIPANTS", "agentd-nagatha,agentd-bilby")
    monkeypatch.setenv("RTECH_TG_IDENTITY_MAP", "agentd-nagatha=env:TG_NAG")
    with pytest.raises(ValueError, match="every configured RTECH_CHATD_DEBATE_PARTICIPANTS"):
        TgdServiceConfig.from_env()


def test_assembly_no_intake_sink_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_min_env(monkeypatch)
    monkeypatch.delenv("RTECH_CHATD_INTAKE_ENABLED", raising=False)
    config = TgdServiceConfig.from_env()
    components = build_telegram_components(config)
    assert components.adapter._intake_sink is None  # noqa: SLF001


def test_shared_assembler_used_by_dev_stack_shape() -> None:
    # The low-level assembler both entrypoints share: passing resolved parts wires
    # a Telegram-only edge with no network opened.
    components = assemble_telegram_edge(
        stack="dev",
        nats_url="nats://x:4222",
        edge_token_ref="env:RTECH_CHATD_BOT_TOKEN",
        identity_refs={},
        telegram=TelegramConfig(allowed_groups=frozenset({GROUP_ID})),
        intake=IntakeConfig(),
        gate_config=GateConfig(require_mention=True),
    )
    assert isinstance(components.adapter, TelegramChannelAdapter)
    assert components.edge._origin_edge == ORIGIN_EDGE_TELEGRAM  # noqa: SLF001


# --- end-to-end: a debate TLDR gets an actor-bound keyboard; a tap steers -----


class _E2ESender:
    """A fake TelegramSender exercising the real edge/adapter/sender OUTBOUND seam.

    Records every send (returning an incrementing message id so the interaction's
    mint-after-id path runs), every attached/cleared inline keyboard, and every
    callback answer — all without aiogram or a network.
    """

    def __init__(self) -> None:
        # Numeric ids, exactly like a real Telegram message id (the id space the
        # inbound callback normalizes to via str(message.message_id)).
        self._next_id = 500
        self.sends: list[tuple[str, str, str | None, str | None]] = []
        self.attached: dict[str, object] = {}
        self.cleared: list[tuple[str, str, str | None]] = []
        self.answers: list[str] = []

    def bind_default_bot(self, provider: object) -> None:
        return None

    async def send(self, target: SendTarget, text: str, files: list[Path]) -> str | None:
        message_id = str(self._next_id)
        self._next_id += 1
        self.sends.append((target.channel_id, text, target.reply_to, target.as_agent))
        return message_id

    async def edit_reply_markup(
        self, target: SendTarget, message_id: str, *, reply_markup: object = None
    ) -> bool:
        if reply_markup is not None:
            self.attached[message_id] = reply_markup
        else:
            self.cleared.append((target.channel_id, message_id, target.as_agent))
        return True

    async def answer_callback(
        self, callback_query_id: str, *, channel_id: str | None = None
    ) -> bool:
        self.answers.append(callback_query_id)
        return True

    async def typing(self, target: SendTarget, on: bool) -> None:
        return None

    async def react_receipt(self, channel_id: str, message_id: str) -> None:
        return None

    async def create_thread(self, channel_id: str, message_id: str, name: str) -> str | None:
        return None

    async def close(self) -> None:
        return None


def _compose_e2e_edge(
    sender: _E2ESender,
) -> tuple[object, TelegramChannelAdapter, object]:
    """Wire the real edge + interaction + callback sink + adapter as production does.

    Uses the real :class:`~rtech_chatd.callback_sink.TelegramCallbackSink`,
    :class:`~rtech_chatd.callback_sink.TelegramTurnInteraction`, real
    :class:`TelegramChannelAdapter`, and real :class:`DiscordEdge` — only the
    sender's aiogram Bot and the NATS bus are faked, so the whole interaction
    flows through the same seams the standalone service composes.
    """
    import time

    from rtech_chatd.callback_sink import TelegramCallbackSink, TelegramTurnInteraction
    from rtech_chatd.telegram_callback import TelegramCallbackRegistry

    class _Bus:
        def __init__(self) -> None:
            self.published: list[tuple[str, bytes]] = []

        async def publish(self, subject: str, data: bytes) -> None:
            self.published.append((subject, data))

        async def subscribe(self, subject: str, handler: object) -> object:
            return None

    bus = _Bus()
    registry = TelegramCallbackRegistry(clock=time.monotonic)
    sink = TelegramCallbackSink(bus=bus, sender=sender, registry=registry)  # type: ignore[arg-type]
    interaction = TelegramTurnInteraction(sink=sink, sender=sender)  # type: ignore[arg-type]
    adapter = TelegramChannelAdapter(
        token_provider=lambda: "x",
        gate_config=GateConfig(require_mention=True),
        callback_sink=sink,
        sender=sender,  # type: ignore[arg-type]
    )
    edge = DiscordEdge(
        adapter,
        bus,  # type: ignore[arg-type]
        stack="dev",
        origin_edge=ORIGIN_EDGE_TELEGRAM,
        intake_enabled=True,
        identity_send=True,
        turn_interaction=interaction,
    )
    return edge, adapter, bus


def test_e2e_debate_tldr_keyboard_lets_the_requester_send_a_typed_steer() -> None:
    from rtech_chatd import subjects
    from rtech_frames.codec import decode_frame, encode_frame
    from rtech_frames.frames import ChannelOrigin, Stance, SteerFrame, TurnFrame

    sender = _E2ESender()
    edge, adapter, bus = _compose_e2e_edge(sender)

    # The original requester (Telegram user id "99") asked in a debate; a
    # participant's terminal turn arrives on the shared turn wildcard. The asker id
    # is the SAME id space as an inbound callback's author (both str(from_user.id)),
    # so the capability binds to the exact user who can tap.
    origin = ChannelOrigin(
        channel_id="-100777",
        message_id="ask-1",
        asker="99",
        debate=True,
    )
    turn = TurnFrame(
        agent_id="agentd-bilby",
        task_id="deb-e2e",
        visible_text="Full reasoning lands under the identity bot.",
        stance=Stance(position="ship", because="the gate is green", needs_from=[]),
        origin=origin,
    )
    asyncio.run(edge._on_external_turn(encode_frame(turn)))  # noqa: SLF001

    # The debate TLDR was posted under the AGENT identity bot; the interaction then
    # posted a default-bot ack (unstamped) carrying the steer keyboard.
    tldr_sends = [s for s in sender.sends if s[3] == "agentd-bilby"]
    ack_sends = [s for s in sender.sends if s[1] == "Steer this direction:"]
    assert tldr_sends, "the agent TLDR must post under its own identity bot"
    assert ack_sends and ack_sends[0][3] is None, "the ack keyboard rides the polled default bot"
    assert len(sender.attached) == 1
    ack_message_id, markup = next(iter(sender.attached.items()))
    buttons = [b for row in markup.inline_keyboard for b in row]  # type: ignore[attr-defined]
    approve_data = buttons[0].callback_data

    # The ORIGINAL requester taps Approve through the REAL adapter callback seam.
    tap = _FakeCallbackQuery(
        callback_id="cq-e2e",
        chat_id=-100777,
        message_id=int(ack_message_id),
        user_id=99,  # matches origin.asker "99"
        data=approve_data,
    )
    asyncio.run(adapter._on_callback(tap))  # noqa: SLF001

    # A typed SteerFrame — never a prose trailer — landed on the bus for the task.
    steers = [
        decode_frame(data)
        for subject, data in bus.published
        if subject == subjects.task_steer("deb-e2e")
    ]
    assert len(steers) == 1
    steer = steers[0]
    assert isinstance(steer, SteerFrame)
    assert steer.task_id == "deb-e2e"
    assert steer.target_agent == "agentd-bilby"
    assert steer.interrupt is False  # approve does not interrupt
    # The tap was answered and the spent keyboard cleared on the default bot.
    assert sender.answers == ["cq-e2e"]
    assert sender.cleared == [("-100777", ack_message_id, None)]


def test_e2e_intruder_tap_on_requester_keyboard_fails_closed() -> None:
    from rtech_chatd import subjects
    from rtech_frames.codec import encode_frame
    from rtech_frames.frames import ChannelOrigin, Stance, TurnFrame

    sender = _E2ESender()
    edge, adapter, bus = _compose_e2e_edge(sender)
    origin = ChannelOrigin(channel_id="-100777", message_id="ask-1", asker="99", debate=True)
    turn = TurnFrame(
        agent_id="agentd-bilby",
        task_id="deb-e2e",
        visible_text="Full reasoning.",
        stance=Stance(position="ship", because="green", needs_from=[]),
        origin=origin,
    )
    asyncio.run(edge._on_external_turn(encode_frame(turn)))  # noqa: SLF001
    ack_message_id, markup = next(iter(sender.attached.items()))
    approve_data = [b for row in markup.inline_keyboard for b in row][0].callback_data  # type: ignore[attr-defined]

    # A DIFFERENT chat member taps the requester's button: no steer, still answered,
    # keyboard NOT cleared (it stays live for the real requester).
    tap = _FakeCallbackQuery(
        callback_id="cq-intruder",
        chat_id=-100777,
        message_id=int(ack_message_id),
        user_id=7,  # not human-99
        data=approve_data,
    )
    asyncio.run(adapter._on_callback(tap))  # noqa: SLF001
    assert [s for s in bus.published if s[0] == subjects.task_steer("deb-e2e")] == []
    assert sender.answers == ["cq-intruder"]
    assert sender.cleared == []


class _FakeMessageChat:
    def __init__(self, chat_id: int) -> None:
        self.id = chat_id


class _FakeCallbackMessage:
    def __init__(self, chat_id: int, message_id: int) -> None:
        self.chat = _FakeMessageChat(chat_id)
        self.message_id = message_id


class _FakeCallbackUser:
    def __init__(self, user_id: int) -> None:
        self.id = user_id


class _FakeCallbackQuery:
    """The minimal aiogram CallbackQuery surface ``map_callback`` reads."""

    def __init__(
        self, *, callback_id: str, chat_id: int, message_id: int, user_id: int, data: str
    ) -> None:
        self.id = callback_id
        self.message = _FakeCallbackMessage(chat_id, message_id)
        self.from_user = _FakeCallbackUser(user_id)
        self.data = data


# --- lifecycle: closes resources on startup failure and shutdown -------------


class _CleanExitAdapter:
    """A poll loop that returns immediately (a dead-but-clean edge)."""

    def __init__(self) -> None:
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    async def send(self, target: SendTarget, text: str, files: list[Path]) -> None:
        return None

    async def typing(self, target: SendTarget, on: bool) -> None:
        return None


class _BlockingInboundEdge:
    """An edge whose inbound drain blocks forever; records close()."""

    def __init__(self) -> None:
        self.closed = False

    async def run_inbound(self) -> None:
        await asyncio.Event().wait()

    async def close(self) -> None:
        self.closed = True


class _LifecycleBus:
    def __init__(self) -> None:
        self.connected = False
        self.closed = False

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.closed = True


class _PartialConnectBus(_LifecycleBus):
    async def connect(self) -> None:
        self.connected = True
        raise RuntimeError("partial connect boom")


async def test_poll_loop_clean_exit_fails_loud_and_tears_down() -> None:
    adapter = _CleanExitAdapter()
    edge = _BlockingInboundEdge()
    bus = _LifecycleBus()

    caught = False
    try:
        await run_components(adapter, bus, edge)  # type: ignore[arg-type]
    except* RuntimeError:
        caught = True

    assert caught is True
    assert bus.connected is True
    # Every external resource is closed even on the loud failure path.
    assert bus.closed is True
    assert adapter.closed is True
    assert edge.closed is True


class _RaisingCloseAdapter(_CleanExitAdapter):
    async def close(self) -> None:
        self.closed = True
        raise RuntimeError("adapter close boom")


async def test_teardown_continues_when_one_close_raises() -> None:
    # One teardown failure must not abort the rest: the bus still closes even
    # when the adapter's close() raises.
    adapter = _RaisingCloseAdapter()
    edge = _BlockingInboundEdge()
    bus = _LifecycleBus()

    try:
        await run_components(adapter, bus, edge)  # type: ignore[arg-type]
    except* RuntimeError:
        pass

    assert adapter.closed is True
    assert edge.closed is True
    assert bus.closed is True


async def test_partial_bus_connect_failure_tears_down_all_resources() -> None:
    adapter = _CleanExitAdapter()
    edge = _BlockingInboundEdge()
    bus = _PartialConnectBus()

    with pytest.raises(RuntimeError, match="partial connect boom"):
        await run_components(adapter, bus, edge)  # type: ignore[arg-type]

    assert bus.connected is True
    assert bus.closed is True
    assert adapter.closed is True
    assert edge.closed is True


class _EventsAdapter:
    """Adapter whose inbound events() blocks; supports the run_service path."""

    def __init__(self) -> None:
        self.closed = False

    async def start(self) -> None:
        raise RuntimeError("connect boom")

    async def close(self) -> None:
        self.closed = True

    async def events(self) -> AsyncIterator[object]:
        await asyncio.Event().wait()
        if False:  # pragma: no cover
            yield object()

    async def send(self, target: SendTarget, text: str, files: list[Path]) -> None:
        return None

    async def typing(self, target: SendTarget, on: bool) -> None:
        return None


async def test_startup_failure_tears_down_all_resources() -> None:
    # A poll loop that raises on connect still closes the bus/edge/adapter.
    adapter = _EventsAdapter()
    edge = _BlockingInboundEdge()
    bus = _LifecycleBus()

    try:
        await run_components(adapter, bus, edge)  # type: ignore[arg-type]
    except* RuntimeError:
        pass

    assert bus.closed is True
    assert edge.closed is True
    assert adapter.closed is True


# --- teardown log leak guard: a close failure never renders the raw exception --

# A sentinel whose substrings must never survive into a rendered teardown log line;
# a bus (or bus-backed edge/adapter) close failure can carry the credentialed NATS
# URL in its message text.
_TEARDOWN_SECRET = "nats://host?token=SUPER_SECRET requester=123"


def _capture_rendered_errors() -> tuple[list[str], int]:
    """Attach a sink capturing fully RENDERED ERROR log lines; return (lines, id).

    ``str(message)`` is the formatted line with ``{}`` already filled in — what a
    real log destination sees — so it would expose a leak living in the arguments.
    """
    from loguru import logger

    lines: list[str] = []
    sink_id = logger.add(lambda m: lines.append(str(m)), level="ERROR")
    return lines, sink_id


class _SecretCloseBus(_LifecycleBus):
    """A bus whose close() raises an exception carrying the credentialed URL."""

    async def close(self) -> None:
        self.closed = True
        raise RuntimeError(_TEARDOWN_SECRET)


class _SecretCloseEdge(_BlockingInboundEdge):
    async def close(self) -> None:
        self.closed = True
        raise RuntimeError(_TEARDOWN_SECRET)


class _SecretCloseAdapter(_CleanExitAdapter):
    async def close(self) -> None:
        self.closed = True
        raise RuntimeError(_TEARDOWN_SECRET)


async def test_teardown_close_failures_omit_credentialed_url_in_logs() -> None:
    # Every teardown close (edge, adapter, bus) raises an exception carrying the
    # credentialed NATS URL. The finally block must log only the exception TYPE and
    # a safe stage label — never render the raw exception into journald.
    adapter = _SecretCloseAdapter()
    edge = _SecretCloseEdge()
    bus = _SecretCloseBus()

    lines, sink_id = _capture_rendered_errors()
    try:
        try:
            await run_components(adapter, bus, edge)  # type: ignore[arg-type]
        except* RuntimeError:
            pass
    finally:
        from loguru import logger

        logger.remove(sink_id)

    blob = "\n".join(lines)
    # All three close paths ran and were logged.
    assert adapter.closed is True
    assert edge.closed is True
    assert bus.closed is True
    assert "chatd.tgd | edge close failed" in blob
    assert "chatd.tgd | adapter close failed" in blob
    assert "chatd.tgd | bus close failed" in blob
    # None of the credential-bearing substrings may appear anywhere.
    assert "SUPER_SECRET" not in blob
    assert "token=" not in blob
    assert "nats://host" not in blob
    assert "requester=123" not in blob
    # The safe shape survives: the exception type name renders instead.
    assert "RuntimeError" in blob


async def test_run_service_preflights_credentials_before_component_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TgdServiceConfig(
        stack="canary",
        nats_url="nats://x:4222",
        edge_token_ref="env:TG_CUSTOM_EDGE_TOKEN",
        telegram=TelegramConfig(allowed_groups=frozenset({GROUP_ID})),
    )
    assembled = False

    def _unexpected_assembly(_: TgdServiceConfig) -> TelegramEdgeComponents:
        nonlocal assembled
        assembled = True
        raise AssertionError("credentials should fail before assembly")

    monkeypatch.delenv("TG_CUSTOM_EDGE_TOKEN", raising=False)
    monkeypatch.setattr(telegram_service, "build_telegram_components", _unexpected_assembly)
    with pytest.raises(RuntimeError) as excinfo:
        await telegram_service.run_service(config)

    assert "TG_CUSTOM_EDGE_TOKEN" not in str(excinfo.value)
    assert assembled is False


async def test_run_service_preflights_every_debate_identity_before_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TgdServiceConfig(
        stack="canary",
        nats_url="nats://x:4222",
        edge_token_ref="env:TG_EDGE_TOKEN",
        identity_refs={
            "agentd-nagatha": "env:TG_NAG_TOKEN",
            "agentd-bilby": "env:TG_BILBY_TOKEN",
        },
        telegram=TelegramConfig(allowed_groups=frozenset({GROUP_ID})),
        intake=IntakeConfig(
            enabled=True,
            triage_agent="agentd-nagatha",
            debate_participants=("agentd-nagatha", "agentd-bilby"),
        ),
    )
    assembled = False

    def _unexpected_assembly(_: TgdServiceConfig) -> TelegramEdgeComponents:
        nonlocal assembled
        assembled = True
        raise AssertionError("credentials should fail before assembly")

    monkeypatch.setenv("TG_EDGE_TOKEN", "edge-token")
    monkeypatch.setenv("TG_NAG_TOKEN", "nag-token")
    monkeypatch.delenv("TG_BILBY_TOKEN", raising=False)
    monkeypatch.setattr(telegram_service, "build_telegram_components", _unexpected_assembly)
    with pytest.raises(RuntimeError) as excinfo:
        await telegram_service.run_service(config)

    assert "TG_BILBY_TOKEN" not in str(excinfo.value)
    assert assembled is False
