"""Thin smoke tests for the discord.py adapter — no network.

Uses fake discord objects (client / channel / message) that satisfy the
isinstance checks by borrowing the real classes via ``object.__new__`` (their
real ``__init__`` needs a live connection). These prove the wiring: message ->
InboundMessage mapping, gate application, auto-thread creation, chunked send
with thread>channel resolution, and the typing keepalive toggle.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import discord
import pytest
from rtech_chatd.channel import InboundMessage, SendTarget
from rtech_chatd.config import GateConfig, IntakeConfig, IntakeMode, ThreadConfig
from rtech_chatd.discord_adapter import (
    DiscordChannelAdapter,
    _map_message,
)


class FakeUser:
    def __init__(self, uid: int, *, bot: bool = False) -> None:
        self.id = uid
        self.bot = bot

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeUser) and other.id == self.id

    def __hash__(self) -> int:
        return hash(self.id)


def _dm_channel(cid: int) -> discord.DMChannel:
    ch = object.__new__(discord.DMChannel)
    ch.id = cid  # type: ignore[misc]
    return ch


class FakeTextChannel(discord.abc.Messageable):
    """A minimal messageable text channel (isinstance(..., Messageable) passes)."""

    def __init__(self, cid: int) -> None:
        self.id = cid
        self.sent: list[dict[str, object]] = []
        self.created_thread: FakeThread | None = None

    async def send(self, content=None, reference=None, files=None):  # noqa: ANN001
        self.sent.append({"content": content, "reference": reference, "files": files})

    async def typing(self) -> None:  # type: ignore[override]
        return None


class FakeThread(discord.Thread):
    """A real ``discord.Thread`` subtype (so isinstance passes) built without
    a live connection via ``__new__``. ``__init__`` needs the gateway, so we
    skip it and set only the attributes the adapter reads."""

    def __new__(cls, tid: int, parent_id: int) -> FakeThread:
        obj = object.__new__(cls)
        obj.id = tid
        obj.parent_id = parent_id
        obj.sent = []
        return obj

    def __init__(self, tid: int, parent_id: int) -> None:  # noqa: D401
        # Intentionally do NOT call super().__init__ (needs a live connection).
        pass

    async def send(self, content=None, reference=None, files=None):  # type: ignore[override]  # noqa: ANN001
        self.sent.append({"content": content, "reference": reference, "files": files})


class FakeGuild:
    def __init__(self, gid: int) -> None:
        self.id = gid


class FakeMessage:
    def __init__(
        self,
        *,
        author: FakeUser,
        channel: object,
        content: str,
        mentions: list[FakeUser] | None = None,
        reference: object | None = None,
        mtype: discord.MessageType = discord.MessageType.default,
        mid: int = 7000,
        guild: FakeGuild | None = None,
    ) -> None:
        self.author = author
        self.channel = channel
        self.content = content
        self.mentions = mentions or []
        self.reference = reference
        self.type = mtype
        self.id = mid
        self.guild = guild
        self.created_thread: FakeThread | None = None

    async def create_thread(self, name: str, auto_archive_duration: int = 1440) -> FakeThread:
        self.created_thread = FakeThread(999, getattr(self.channel, "id", 0))
        return self.created_thread


class FakeClient:
    """A stand-in for discord.Client (the adapter's ``client`` seam)."""

    def __init__(self, user: FakeUser) -> None:
        self.user = user
        self._channels: dict[int, object] = {}

    def event(self, func):  # noqa: ANN001
        # The adapter registers on_message via @client.event; keep a handle.
        self.on_message = func
        return func

    def get_channel(self, cid: int):  # noqa: ANN001
        return self._channels.get(cid)

    def register_channel(self, cid: int, channel: object) -> None:
        self._channels[cid] = channel


def _adapter(client: FakeClient, **kw: object) -> DiscordChannelAdapter:
    return DiscordChannelAdapter(
        token_provider=lambda: "unused-in-test",
        gate_config=kw.get("gate", GateConfig()),  # type: ignore[arg-type]
        thread_config=kw.get("thread", ThreadConfig()),  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------- map_message


def test_map_message_strips_own_mention() -> None:
    me = FakeUser(1)
    ch = FakeTextChannel(100)
    msg = FakeMessage(author=FakeUser(2), channel=ch, content="<@1> hello", mentions=[me])
    mapped = _map_message(msg, me)  # type: ignore[arg-type]
    assert mapped.text == "hello"
    assert mapped.mentions_me is True
    assert mapped.channel_id == "100"
    assert mapped.thread_id is None


def test_map_message_dm_sets_flags() -> None:
    me = FakeUser(1)
    msg = FakeMessage(author=FakeUser(2), channel=_dm_channel(50), content="hi")
    mapped = _map_message(msg, me)  # type: ignore[arg-type]
    assert mapped.is_dm is True
    assert mapped.mentions_me is True  # implicit in DM


def test_map_message_thread_resolves_parent() -> None:
    me = FakeUser(1)
    th = FakeThread(500, parent_id=100)
    msg = FakeMessage(author=FakeUser(2), channel=th, content="in thread")
    mapped = _map_message(msg, me)  # type: ignore[arg-type]
    assert mapped.thread_id == "500"
    assert mapped.channel_id == "100"


def test_map_message_reply_flag() -> None:
    me = FakeUser(1)
    ch = FakeTextChannel(100)
    msg = FakeMessage(author=FakeUser(2), channel=ch, content="re", reference=object())
    mapped = _map_message(msg, me)  # type: ignore[arg-type]
    assert mapped.is_reply is True


# --------------------------------------------------------------- on_message + gate


def test_on_message_enqueues_when_mentioned() -> None:
    me = FakeUser(1)
    client = FakeClient(me)
    adapter = _adapter(client, thread=ThreadConfig(auto_thread=False))
    ch = FakeTextChannel(100)
    msg = FakeMessage(author=FakeUser(2), channel=ch, content="<@1> hey", mentions=[me])

    asyncio.run(adapter._on_message(msg))  # type: ignore[arg-type]
    assert adapter._inbound.qsize() == 1


def test_on_message_ignores_self() -> None:
    me = FakeUser(1)
    client = FakeClient(me)
    adapter = _adapter(client)
    ch = FakeTextChannel(100)
    msg = FakeMessage(author=me, channel=ch, content="<@1> me", mentions=[me])
    asyncio.run(adapter._on_message(msg))  # type: ignore[arg-type]
    assert adapter._inbound.qsize() == 0


def test_on_message_drops_unmentioned_in_guild() -> None:
    me = FakeUser(1)
    client = FakeClient(me)
    adapter = _adapter(client)
    ch = FakeTextChannel(100)
    msg = FakeMessage(author=FakeUser(2), channel=ch, content="just chatting")
    asyncio.run(adapter._on_message(msg))  # type: ignore[arg-type]
    assert adapter._inbound.qsize() == 0


def test_on_message_creates_thread_and_marks() -> None:
    me = FakeUser(1)
    client = FakeClient(me)
    adapter = _adapter(client)  # auto_thread default True
    ch = FakeTextChannel(100)
    msg = FakeMessage(author=FakeUser(2), channel=ch, content="<@1> start work", mentions=[me])

    asyncio.run(adapter._on_message(msg))  # type: ignore[arg-type]
    # Thread was created and the enqueued message points at it.
    assert msg.created_thread is not None
    queued = adapter._inbound.get_nowait()
    assert queued.thread_id == "999"


# --------------------------------------------------------------- send / typing


def test_send_targets_thread_over_channel() -> None:
    me = FakeUser(1)
    client = FakeClient(me)
    th = FakeThread(999, parent_id=100)
    client.register_channel(999, th)
    adapter = _adapter(client)

    asyncio.run(adapter.send(SendTarget(channel_id="100", thread_id="999"), "hello", []))
    assert len(th.sent) == 1
    assert th.sent[0]["content"] == "hello"


class _AnchorChannel(FakeTextChannel):
    """A text channel that can fetch an anchor message for thread creation."""

    def __init__(self, cid: int, anchor: FakeMessage | None) -> None:
        super().__init__(cid)
        self._anchor = anchor

    async def fetch_message(self, mid: int):  # noqa: ANN001, ANN201
        if self._anchor is None:
            raise discord.DiscordException("no such message")
        return self._anchor


def test_create_thread_anchors_to_message_and_returns_id() -> None:
    """D1: create_thread opens a thread anchored to the ask and returns its id."""
    me = FakeUser(1)
    client = FakeClient(me)
    anchor = FakeMessage(author=FakeUser(2), channel=None, content="ask", mid=7000)
    ch = _AnchorChannel(100, anchor)
    client.register_channel(100, ch)
    adapter = _adapter(client)

    tid = asyncio.run(adapter.create_thread("100", "7000", "debate T-1"))
    assert tid == "999"  # FakeMessage.create_thread mints thread 999
    assert anchor.created_thread is not None


def test_create_thread_fails_soft_on_unresolvable_channel() -> None:
    """D1: an unresolvable channel returns None (degrade), never raises."""
    me = FakeUser(1)
    client = UnresolvableClient(me)  # get_channel misses, fetch_channel raises
    adapter = _adapter(client)

    tid = asyncio.run(adapter.create_thread("404", "7000", "debate"))
    assert tid is None


def test_create_thread_fails_soft_when_anchor_missing() -> None:
    """D1: a missing anchor message returns None (degrade), never raises."""
    me = FakeUser(1)
    client = FakeClient(me)
    ch = _AnchorChannel(100, anchor=None)  # fetch_message raises NotFound
    client.register_channel(100, ch)
    adapter = _adapter(client)

    tid = asyncio.run(adapter.create_thread("100", "7000", "debate"))
    assert tid is None


class _IdReturningChannel(FakeTextChannel):
    """A text channel whose send() returns a message object with an id."""

    def __init__(self, cid: int) -> None:
        super().__init__(cid)
        self._next_id = 5000

    async def send(self, content=None, reference=None, files=None):  # noqa: ANN001
        await super().send(content=content, reference=reference, files=files)
        self._next_id += 1
        return SimpleNamespace(id=self._next_id)


def test_send_returns_first_chunk_message_id() -> None:
    """UX v2 anchor contract: send() returns the FIRST sent chunk's message id."""
    me = FakeUser(1)
    client = FakeClient(me)
    ch = _IdReturningChannel(100)
    client.register_channel(100, ch)
    adapter = _adapter(client)

    long = "x" * 5000  # forces chunking; the FIRST chunk's id must come back
    result = asyncio.run(adapter.send(SendTarget(channel_id="100"), long, []))
    assert len(ch.sent) >= 2
    assert result == "5001"


def test_send_chunks_long_text() -> None:
    me = FakeUser(1)
    client = FakeClient(me)
    ch = FakeTextChannel(100)
    client.register_channel(100, ch)
    adapter = _adapter(client)

    long = "x" * 5000
    asyncio.run(adapter.send(SendTarget(channel_id="100"), long, []))
    assert len(ch.sent) >= 2  # chunked


def test_typing_toggle_starts_and_stops() -> None:
    me = FakeUser(1)
    client = FakeClient(me)
    ch = FakeTextChannel(100)
    client.register_channel(100, ch)
    adapter = _adapter(client)

    async def drive() -> None:
        target = SendTarget(channel_id="100")
        await adapter.typing(target, True)
        assert "100" in adapter._typing_tasks
        await asyncio.sleep(0)  # let the loop tick once
        await adapter.typing(target, False)
        assert "100" not in adapter._typing_tasks

    asyncio.run(drive())


# --------------------------------------------------------------- map_message intake fields


def test_map_message_captures_guild_and_mentioned_ids() -> None:
    me = FakeUser(1)
    other = FakeUser(42)
    ch = FakeTextChannel(100)
    msg = FakeMessage(
        author=FakeUser(2),
        channel=ch,
        content="<@42> ping",
        mentions=[other],
        mid=9001,
        guild=FakeGuild(555),
    )
    mapped = _map_message(msg, me)  # type: ignore[arg-type]
    assert mapped.guild_id == "555"
    assert mapped.message_id == "9001"
    assert mapped.mentioned_ids == frozenset({"42"})


def test_map_message_dm_has_no_guild() -> None:
    me = FakeUser(1)
    msg = FakeMessage(author=FakeUser(2), channel=_dm_channel(50), content="hi")
    mapped = _map_message(msg, me)  # type: ignore[arg-type]
    assert mapped.guild_id is None


# --------------------------------------------------------------- intake wiring


def _intake_cfg(**kw: object) -> IntakeConfig:
    # Enabled intake requires an explicit triage id (the deployed agentd-<name>);
    # there is no bare-name default.
    base: dict[str, object] = {
        "enabled": True,
        "mode": IntakeMode.ALL_CHANNELS,
        "triage_agent": "agentd-nagatha",
    }
    base.update(kw)
    return IntakeConfig(**base)  # type: ignore[arg-type]


def test_intake_hit_fires_sink_with_channel_identity() -> None:
    me = FakeUser(1)
    trio = FakeUser(77)
    client = FakeClient(me)
    hits: list[InboundMessage] = []
    adapter = DiscordChannelAdapter(
        token_provider=lambda: "unused",
        gate_config=GateConfig(),
        thread_config=ThreadConfig(auto_thread=False),
        intake_config=_intake_cfg(member_ids=frozenset({"77"})),
        intake_sink=hits.append,
        client=client,  # type: ignore[arg-type]
    )
    ch = FakeTextChannel(100)
    msg = FakeMessage(
        author=FakeUser(2),
        channel=ch,
        content="<@77> can you look at this",
        mentions=[trio],
        mid=8080,
        guild=FakeGuild(200),
    )
    asyncio.run(adapter._on_message(msg))  # type: ignore[arg-type]
    assert len(hits) == 1
    hit = hits[0]
    assert hit.guild_id == "200"
    assert hit.channel_id == "100"
    assert hit.message_id == "8080"
    assert hit.author == "2"


def test_intake_multi_guild_hits_are_independent() -> None:
    # A second guild's channel produces its OWN hit with its own identity.
    me = FakeUser(1)
    trio = FakeUser(77)
    client = FakeClient(me)
    hits: list[InboundMessage] = []
    adapter = DiscordChannelAdapter(
        token_provider=lambda: "unused",
        gate_config=GateConfig(),
        thread_config=ThreadConfig(auto_thread=False),
        intake_config=_intake_cfg(member_ids=frozenset({"77"})),
        intake_sink=hits.append,
        client=client,  # type: ignore[arg-type]
    )
    msg_a = FakeMessage(
        author=FakeUser(2),
        channel=FakeTextChannel(100),
        content="<@77> guild A",
        mentions=[trio],
        mid=1,
        guild=FakeGuild(200),
    )
    msg_b = FakeMessage(
        author=FakeUser(3),
        channel=FakeTextChannel(300),
        content="<@77> guild B",
        mentions=[trio],
        mid=2,
        guild=FakeGuild(400),
    )
    asyncio.run(adapter._on_message(msg_a))  # type: ignore[arg-type]
    asyncio.run(adapter._on_message(msg_b))  # type: ignore[arg-type]
    assert [(h.guild_id, h.channel_id) for h in hits] == [("200", "100"), ("400", "300")]


def test_intake_disabled_never_fires_sink() -> None:
    me = FakeUser(1)
    trio = FakeUser(77)
    client = FakeClient(me)
    hits: list[InboundMessage] = []
    adapter = DiscordChannelAdapter(
        token_provider=lambda: "unused",
        gate_config=GateConfig(),
        thread_config=ThreadConfig(),
        intake_config=IntakeConfig(enabled=False, member_ids=frozenset({"77"})),
        intake_sink=hits.append,
        client=client,  # type: ignore[arg-type]
    )
    ch = FakeTextChannel(100)
    msg = FakeMessage(
        author=FakeUser(2), channel=ch, content="<@77> hey", mentions=[trio], guild=FakeGuild(200)
    )
    asyncio.run(adapter._on_message(msg))  # type: ignore[arg-type]
    assert hits == []


def test_intake_bot_message_ignored() -> None:
    # A bot author is never an intake hit under the default AllowBots.NONE.
    me = FakeUser(1)
    trio = FakeUser(77)
    client = FakeClient(me)
    hits: list[InboundMessage] = []
    adapter = DiscordChannelAdapter(
        token_provider=lambda: "unused",
        gate_config=GateConfig(),
        thread_config=ThreadConfig(auto_thread=False),
        intake_config=_intake_cfg(member_ids=frozenset({"77"})),
        intake_sink=hits.append,
        client=client,  # type: ignore[arg-type]
    )
    ch = FakeTextChannel(100)
    msg = FakeMessage(
        author=FakeUser(2, bot=True),
        channel=ch,
        content="<@77> hey",
        mentions=[trio],
        guild=FakeGuild(200),
    )
    asyncio.run(adapter._on_message(msg))  # type: ignore[arg-type]
    assert hits == []


def test_intake_sink_failure_does_not_crash_gateway() -> None:
    me = FakeUser(1)
    trio = FakeUser(77)
    client = FakeClient(me)

    def boom(_msg: InboundMessage) -> None:
        raise RuntimeError("sink exploded")

    adapter = DiscordChannelAdapter(
        token_provider=lambda: "unused",
        gate_config=GateConfig(),
        thread_config=ThreadConfig(auto_thread=False),
        intake_config=_intake_cfg(member_ids=frozenset({"77"})),
        intake_sink=boom,
        client=client,  # type: ignore[arg-type]
    )
    ch = FakeTextChannel(100)
    msg = FakeMessage(
        author=FakeUser(2), channel=ch, content="<@77> hey", mentions=[trio], guild=FakeGuild(200)
    )
    # Must not raise.
    asyncio.run(adapter._on_message(msg))  # type: ignore[arg-type]


def test_intake_async_sink_is_awaited_and_publishes() -> None:
    # D5: the in-repo bus publisher is async. An async def sink must be AWAITED at
    # the call site — otherwise its coroutine is never run and it publishes
    # nothing (a silent "coroutine never awaited").
    me = FakeUser(1)
    trio = FakeUser(77)
    client = FakeClient(me)
    published: list[InboundMessage] = []

    async def async_sink(m: InboundMessage) -> None:
        # Yield control to prove the coroutine actually executed on the loop.
        await asyncio.sleep(0)
        published.append(m)

    adapter = DiscordChannelAdapter(
        token_provider=lambda: "unused",
        gate_config=GateConfig(),
        thread_config=ThreadConfig(auto_thread=False),
        intake_config=_intake_cfg(member_ids=frozenset({"77"})),
        intake_sink=async_sink,
        client=client,  # type: ignore[arg-type]
    )
    ch = FakeTextChannel(100)
    msg = FakeMessage(
        author=FakeUser(2),
        channel=ch,
        content="<@77> hey",
        mentions=[trio],
        mid=8081,
        guild=FakeGuild(200),
    )
    asyncio.run(adapter._on_message(msg))  # type: ignore[arg-type]
    assert len(published) == 1
    assert published[0].message_id == "8081"


def test_intake_async_raising_sink_is_contained_by_boundary() -> None:
    # D5: an async sink that raises is contained by the SAME error boundary as a
    # sync one — the awaited coroutine's exception never crashes the gateway.
    me = FakeUser(1)
    trio = FakeUser(77)
    client = FakeClient(me)

    async def async_boom(_msg: InboundMessage) -> None:
        await asyncio.sleep(0)
        raise RuntimeError("async sink exploded")

    adapter = DiscordChannelAdapter(
        token_provider=lambda: "unused",
        gate_config=GateConfig(),
        thread_config=ThreadConfig(auto_thread=False),
        intake_config=_intake_cfg(member_ids=frozenset({"77"})),
        intake_sink=async_boom,
        client=client,  # type: ignore[arg-type]
    )
    ch = FakeTextChannel(100)
    msg = FakeMessage(
        author=FakeUser(2), channel=ch, content="<@77> hey", mentions=[trio], guild=FakeGuild(200)
    )
    # Must not raise.
    asyncio.run(adapter._on_message(msg))  # type: ignore[arg-type]


def test_intake_and_responder_are_independent() -> None:
    # A mention of THIS bot: responder enqueues AND intake fires (bot is a
    # configured member). They do not interfere.
    me = FakeUser(1)
    client = FakeClient(me)
    hits: list[InboundMessage] = []
    adapter = DiscordChannelAdapter(
        token_provider=lambda: "unused",
        gate_config=GateConfig(),
        thread_config=ThreadConfig(auto_thread=False),
        intake_config=_intake_cfg(member_ids=frozenset({"1"})),
        intake_sink=hits.append,
        client=client,  # type: ignore[arg-type]
    )
    ch = FakeTextChannel(100)
    msg = FakeMessage(
        author=FakeUser(2),
        channel=ch,
        content="<@1> please help",
        mentions=[me],
        guild=FakeGuild(200),
    )
    asyncio.run(adapter._on_message(msg))  # type: ignore[arg-type]
    assert adapter._inbound.qsize() == 1  # responder path unaffected
    assert len(hits) == 1  # intake path fired


def test_existing_default_config_has_no_intake() -> None:
    # An adapter built without intake config (existing deploys) never fires.
    me = FakeUser(1)
    trio = FakeUser(77)
    client = FakeClient(me)
    adapter = _adapter(client, thread=ThreadConfig(auto_thread=False))
    ch = FakeTextChannel(100)
    msg = FakeMessage(
        author=FakeUser(2), channel=ch, content="<@77> hey", mentions=[trio], guild=FakeGuild(200)
    )
    # No sink, no intake_config → only the responder path runs; no crash.
    asyncio.run(adapter._on_message(msg))  # type: ignore[arg-type]
    assert adapter._intake_config.enabled is False


def test_start_rejects_empty_token() -> None:
    me = FakeUser(1)
    client = FakeClient(me)
    adapter = DiscordChannelAdapter(
        token_provider=lambda: "",
        gate_config=GateConfig(),
        thread_config=ThreadConfig(),
        client=client,  # type: ignore[arg-type]
    )
    with pytest.raises(RuntimeError):
        asyncio.run(adapter.start())


class UnresolvableClient(FakeClient):
    """A client that cannot resolve the destination channel at all."""

    async def fetch_channel(self, cid: int):  # noqa: ANN001
        raise discord.DiscordException("unknown channel")


def test_send_unresolvable_destination_raises() -> None:
    # get_channel returns None (not registered) and fetch_channel raises: the
    # destination is unresolvable, so send() must fail LOUD rather than silently
    # record a delivery that never happened.
    me = FakeUser(1)
    client = UnresolvableClient(me)
    adapter = _adapter(client)

    with pytest.raises(RuntimeError):
        asyncio.run(adapter.send(SendTarget(channel_id="404"), "hello", []))


def test_dm_intent_follows_gate_config_default_false() -> None:
    # When the adapter builds its own client (client=None), the DM intent tracks
    # gate_config.allow_dms — default GateConfig disables DMs.
    adapter = DiscordChannelAdapter(
        token_provider=lambda: "unused",
        gate_config=GateConfig(),
        thread_config=ThreadConfig(),
    )
    assert adapter._client.intents.dm_messages is False


def test_dm_intent_follows_gate_config_enabled_true() -> None:
    adapter = DiscordChannelAdapter(
        token_provider=lambda: "unused",
        gate_config=GateConfig(allow_dms=True),
        thread_config=ThreadConfig(),
    )
    assert adapter._client.intents.dm_messages is True
