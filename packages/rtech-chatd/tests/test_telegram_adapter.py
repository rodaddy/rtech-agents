"""Fake-transport tests for the aiogram Telegram adapter (inbound).

No live token, no network: aiogram's ``Update`` / ``Message`` types are pydantic
models constructible directly, so we build real update objects and drive
:meth:`TelegramChannelAdapter._on_message` / :meth:`_handle_update` against them.
A tiny fake ``Bot`` covers the poll-loop path. These prove the wiring: update ->
InboundMessage normalization (incl. ``-100…`` canonicalization, thread id,
mention/mentioned_id duality, DM addressing), composite ``(chat_id, message_id)``
dedup, edited/migration handling, gate + intake integration, and protocol
conformance.
"""

from __future__ import annotations

import asyncio

import pytest
from aiogram.types import CallbackQuery, Chat, Message, MessageEntity, Update, User
from loguru import logger
from rtech_chatd import telegram_adapter
from rtech_chatd.channel import ChannelAdapter, InboundMessage, SendTarget
from rtech_chatd.config import GateConfig, IntakeConfig, IntakeMode, TelegramConfig
from rtech_chatd.telegram_adapter import (
    InboundCallback,
    TelegramChannelAdapter,
    canonical_chat_id,
    map_callback,
    map_message,
)

# A canonical -100 supergroup id (what the Bot API delivers for a supergroup).
SUPERGROUP_ID = -1001234567890
SUPERGROUP_STR = "-1001234567890"


# --------------------------------------------------------------- builders


def _user(uid: int, *, is_bot: bool = False, username: str | None = None) -> User:
    return User(id=uid, is_bot=is_bot, first_name="U", username=username)


def _supergroup_chat(cid: int = SUPERGROUP_ID) -> Chat:
    return Chat(id=cid, type="supergroup", title="Test Group")


def _private_chat(cid: int = 555) -> Chat:
    return Chat(id=cid, type="private")


def _message(
    *,
    chat: Chat,
    author: User | None,
    text: str | None = None,
    message_id: int = 7000,
    thread_id: int | None = None,
    is_topic_message: bool | None = None,
    entities: list[MessageEntity] | None = None,
    reply_to: Message | None = None,
    migrate_to_chat_id: int | None = None,
    migrate_from_chat_id: int | None = None,
    sender_chat: Chat | None = None,
) -> Message:
    return Message(
        message_id=message_id,
        date=0,
        chat=chat,
        from_user=author,
        text=text,
        message_thread_id=thread_id,
        is_topic_message=is_topic_message,
        entities=entities,
        reply_to_message=reply_to,
        migrate_to_chat_id=migrate_to_chat_id,
        migrate_from_chat_id=migrate_from_chat_id,
        sender_chat=sender_chat,
    )


def _mention_entity(text: str, handle: str) -> MessageEntity:
    """Build a ``@handle`` mention entity anchored at its offset in ``text``."""
    at = f"@{handle}"
    offset = text.index(at)
    return MessageEntity(type="mention", offset=offset, length=len(at))


def _text_mention_entity(text: str, span: str, user: User) -> MessageEntity:
    offset = text.index(span)
    return MessageEntity(type="text_mention", offset=offset, length=len(span), user=user)


def _adapter(**kw: object) -> TelegramChannelAdapter:
    base: dict[str, object] = {
        "token_provider": lambda: "unused-in-test",
        "gate_config": GateConfig(),
    }
    base.update(kw)
    return TelegramChannelAdapter(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------- normalization


def test_canonical_chat_id_stringifies_supergroup() -> None:
    assert canonical_chat_id(SUPERGROUP_ID) == SUPERGROUP_STR


def test_map_message_supergroup_fields() -> None:
    msg = _message(
        chat=_supergroup_chat(),
        author=_user(2),
        text="hello world",
        message_id=42,
    )
    mapped = map_message(msg, self_username="mybot", self_user_id="1")
    assert mapped.channel_id == SUPERGROUP_STR  # -100 canonicalization
    assert mapped.message_id == "42"
    assert mapped.author == "2"
    assert mapped.text == "hello world"
    assert mapped.thread_id is None
    assert mapped.is_dm is False
    assert mapped.guild_id is None  # no guild tier on Telegram
    assert mapped.is_bot is False


def test_map_message_thread_id_from_forum_topic() -> None:
    # A real forum topic: is_topic_message is set alongside message_thread_id.
    msg = _message(
        chat=_supergroup_chat(),
        author=_user(2),
        text="in topic",
        thread_id=99,
        is_topic_message=True,
    )
    mapped = map_message(msg, self_username="mybot", self_user_id="1")
    assert mapped.thread_id == "99"


def test_map_message_reply_thread_without_topic_flag_has_no_thread_id() -> None:
    # An ORDINARY reply thread carries message_thread_id too, but NOT
    # is_topic_message. aiogram gates topic routing on is_topic_message, so this
    # deliberately does not set thread_id (it surfaces as is_reply instead) —
    # conflating it with a forum topic would mis-target the topic sends.
    msg = _message(
        chat=_supergroup_chat(),
        author=_user(2),
        text="a reply",
        thread_id=99,
        is_topic_message=None,
    )
    mapped = map_message(msg, self_username="mybot", self_user_id="1")
    assert mapped.thread_id is None


def test_map_message_reply_flag() -> None:
    parent = _message(chat=_supergroup_chat(), author=_user(3), text="parent", message_id=1)
    msg = _message(
        chat=_supergroup_chat(), author=_user(2), text="re", message_id=2, reply_to=parent
    )
    mapped = map_message(msg, self_username="mybot", self_user_id="1")
    assert mapped.is_reply is True


def test_map_message_username_mention_normalized_lowercase_no_at() -> None:
    text = "hey @Nagatha take a look"
    msg = _message(
        chat=_supergroup_chat(),
        author=_user(2),
        text=text,
        entities=[_mention_entity(text, "Nagatha")],
    )
    mapped = map_message(msg, self_username="mybot", self_user_id="1")
    # mention entity -> lowercase handle WITHOUT the @.
    assert mapped.mentioned_ids == frozenset({"nagatha"})


def test_map_message_text_mention_uses_user_id() -> None:
    text = "hey Bob take a look"
    target = _user(777)
    msg = _message(
        chat=_supergroup_chat(),
        author=_user(2),
        text=text,
        entities=[_text_mention_entity(text, "Bob", target)],
    )
    mapped = map_message(msg, self_username="mybot", self_user_id="1")
    # text_mention entity -> numeric user id (author id space).
    assert mapped.mentioned_ids == frozenset({"777"})


def test_map_message_mixed_mention_entities() -> None:
    text = "@handleone and Bob"
    target = _user(888)
    msg = _message(
        chat=_supergroup_chat(),
        author=_user(2),
        text=text,
        entities=[
            _mention_entity(text, "handleone"),
            _text_mention_entity(text, "Bob", target),
        ],
    )
    mapped = map_message(msg, self_username="mybot", self_user_id="1")
    assert mapped.mentioned_ids == frozenset({"handleone", "888"})


def test_map_message_private_chat_is_dm_and_mentions_me() -> None:
    msg = _message(chat=_private_chat(), author=_user(2), text="hi")
    mapped = map_message(msg, self_username="mybot", self_user_id="1")
    assert mapped.is_dm is True
    assert mapped.mentions_me is True  # implicit in private chat


def test_map_message_mentions_me_by_username_in_group() -> None:
    text = "@mybot ping"
    msg = _message(
        chat=_supergroup_chat(),
        author=_user(2),
        text=text,
        entities=[_mention_entity(text, "mybot")],
    )
    mapped = map_message(msg, self_username="mybot", self_user_id="1")
    assert mapped.mentions_me is True


def test_map_message_not_mentions_me_when_other_handle() -> None:
    text = "@someoneelse ping"
    msg = _message(
        chat=_supergroup_chat(),
        author=_user(2),
        text=text,
        entities=[_mention_entity(text, "someoneelse")],
    )
    mapped = map_message(msg, self_username="mybot", self_user_id="1")
    assert mapped.mentions_me is False


def test_map_message_bot_author_flag() -> None:
    msg = _message(chat=_supergroup_chat(), author=_user(9, is_bot=True), text="beep")
    mapped = map_message(msg, self_username="mybot", self_user_id="1")
    assert mapped.is_bot is True


def test_map_message_caption_and_caption_entities() -> None:
    # A media message carries body/entities on caption, not text.
    caption = "@mybot look at this photo"
    msg = Message(
        message_id=5,
        date=0,
        chat=_supergroup_chat(),
        from_user=_user(2),
        caption=caption,
        caption_entities=[_mention_entity(caption, "mybot")],
    )
    mapped = map_message(msg, self_username="mybot", self_user_id="1")
    assert mapped.text == caption
    assert mapped.mentions_me is True


# --------------------------------------------------------------- dedup


def test_dedup_fresh_then_repeat_suppressed() -> None:
    now = [1000.0]
    adapter = _adapter(clock=lambda: now[0])
    author = _user(2, username="human")
    ch = _supergroup_chat()
    # Mention this bot so should_respond enqueues (require_mention default).
    text = "@mybot hey"
    ent = [_mention_entity(text, "mybot")]
    adapter._self_username = "mybot"
    adapter._self_user_id = "1"

    m1 = _message(chat=ch, author=author, text=text, entities=ent, message_id=100)
    asyncio.run(adapter._on_message(m1))
    assert adapter._inbound.qsize() == 1

    # Same (chat, message_id) redelivered inside the horizon -> suppressed.
    m2 = _message(chat=ch, author=author, text=text, entities=ent, message_id=100)
    asyncio.run(adapter._on_message(m2))
    assert adapter._inbound.qsize() == 1  # not enqueued again


def test_dedup_horizon_expiry_readmits() -> None:
    now = [1000.0]
    adapter = _adapter(clock=lambda: now[0], dedup_horizon_s=100.0)
    author = _user(2, username="human")
    ch = _supergroup_chat()
    text = "@mybot hey"
    ent = [_mention_entity(text, "mybot")]
    adapter._self_username = "mybot"
    adapter._self_user_id = "1"

    asyncio.run(
        adapter._on_message(
            _message(chat=ch, author=author, text=text, entities=ent, message_id=100)
        )
    )
    assert adapter._inbound.qsize() == 1

    now[0] += 200.0  # past the horizon
    asyncio.run(
        adapter._on_message(
            _message(chat=ch, author=author, text=text, entities=ent, message_id=100)
        )
    )
    assert adapter._inbound.qsize() == 2  # re-admitted after expiry


def test_dedup_cross_chat_same_message_id_both_admit() -> None:
    # Telegram message ids are PER-CHAT: the same id in two chats is two messages.
    now = [1000.0]
    adapter = _adapter(clock=lambda: now[0])
    author = _user(2, username="human")
    text = "@mybot hey"
    ent = [_mention_entity(text, "mybot")]
    adapter._self_username = "mybot"
    adapter._self_user_id = "1"

    chat_a = _supergroup_chat(-1001111111111)
    chat_b = _supergroup_chat(-1002222222222)
    asyncio.run(
        adapter._on_message(
            _message(chat=chat_a, author=author, text=text, entities=ent, message_id=100)
        )
    )
    asyncio.run(
        adapter._on_message(
            _message(chat=chat_b, author=author, text=text, entities=ent, message_id=100)
        )
    )
    assert adapter._inbound.qsize() == 2  # both admitted


# --------------------------------------------------------------- edited / migration


def test_edited_message_logged_and_dropped() -> None:
    adapter = _adapter()
    adapter._self_username = "mybot"
    adapter._self_user_id = "1"
    edited = _message(chat=_supergroup_chat(), author=_user(2), text="@mybot edited")
    update = Update(update_id=1, edited_message=edited)
    asyncio.run(adapter._handle_update(update))
    assert adapter._inbound.qsize() == 0  # never enqueued


def test_migration_service_message_emits_nothing() -> None:
    adapter = _adapter()
    adapter._self_username = "mybot"
    adapter._self_user_id = "1"
    # migrate_to_chat_id on the OLD basic group -> logged, nothing emitted.
    migrate = _message(
        chat=Chat(id=-500, type="group", title="Old"),
        author=_user(2),
        text="",
        migrate_to_chat_id=SUPERGROUP_ID,
    )
    asyncio.run(adapter._on_message(migrate))
    assert adapter._inbound.qsize() == 0


def test_migrate_from_service_message_emits_nothing() -> None:
    adapter = _adapter()
    adapter._self_username = "mybot"
    adapter._self_user_id = "1"
    migrate = _message(
        chat=_supergroup_chat(),
        author=_user(2),
        text="",
        migrate_from_chat_id=-500,
    )
    asyncio.run(adapter._on_message(migrate))
    assert adapter._inbound.qsize() == 0


# --------------------------------------------------------------- callback


class _FakeAnsweringSender:
    """Capture safe callback acknowledgements without Telegram."""

    def __init__(self) -> None:
        self.answers: list[tuple[str, str | None]] = []

    async def answer_callback(
        self, callback_query_id: str, *, channel_id: str | None = None
    ) -> bool:
        self.answers.append((callback_query_id, channel_id))
        return True

    def bind_default_bot(self, provider: object) -> None:
        # The adapter binds the polling bot into the sender at construction; the
        # callback tests never send, so this is a captured no-op.
        return None


def _callback_query(
    *, data: str | None = "v1|approve|nonce|sig", message_id: int = 42
) -> CallbackQuery:
    msg = _message(chat=_supergroup_chat(), author=_user(1, is_bot=True), message_id=message_id)
    return CallbackQuery(
        id="cq-1",
        from_user=_user(2),
        chat_instance="ci-1",
        message=msg,
        data=data,
    )


def test_map_callback_normalizes_surface_coordinates() -> None:
    mapped = map_callback(_callback_query(data="opaque-token", message_id=99))
    assert mapped.callback_query_id == "cq-1"
    assert mapped.channel_id == SUPERGROUP_STR  # -100 canonicalization
    assert mapped.message_id == "99"
    assert mapped.author == "2"
    assert mapped.data == "opaque-token"


def test_callback_update_routes_to_sink() -> None:
    seen: list[InboundCallback] = []

    async def sink(callback: InboundCallback) -> None:
        seen.append(callback)

    adapter = _adapter(callback_sink=sink)
    update = Update(update_id=1, callback_query=_callback_query(data="tap-data"))
    asyncio.run(adapter._handle_update(update))
    assert len(seen) == 1
    assert seen[0].data == "tap-data"
    assert seen[0].channel_id == SUPERGROUP_STR
    assert seen[0].message_id == "42"


def test_callback_without_sink_answers_only() -> None:
    sender = _FakeAnsweringSender()
    adapter = _adapter(sender=sender)  # no callback_sink wired
    update = Update(update_id=1, callback_query=_callback_query())
    asyncio.run(adapter._handle_update(update))
    # Spinner stopped (query answered), but nothing steered.
    assert sender.answers == [("cq-1", SUPERGROUP_STR)]


def test_callback_sink_failure_still_answers() -> None:
    sender = _FakeAnsweringSender()

    async def failing_sink(callback: InboundCallback) -> None:
        raise RuntimeError("sink blew up")

    adapter = _adapter(sender=sender, callback_sink=failing_sink)
    update = Update(update_id=1, callback_query=_callback_query())
    # The gateway must not crash on a sink failure, and the human's button
    # must still stop spinning.
    asyncio.run(adapter._handle_update(update))
    assert sender.answers == [("cq-1", SUPERGROUP_STR)]


# --------------------------------------------------------------- gate integration


def test_gated_out_message_never_emitted() -> None:
    # Unmentioned message in a group with require_mention -> dropped by the gate.
    adapter = _adapter()
    adapter._self_username = "mybot"
    adapter._self_user_id = "1"
    msg = _message(chat=_supergroup_chat(), author=_user(2, username="human"), text="just chatting")
    asyncio.run(adapter._on_message(msg))
    assert adapter._inbound.qsize() == 0


def test_group_allowlist_drops_out_of_scope_chat() -> None:
    adapter = _adapter(
        telegram_config=TelegramConfig(allowed_groups=frozenset({"-1009999999999"})),
    )
    adapter._self_username = "mybot"
    adapter._self_user_id = "1"
    text = "@mybot hey"
    ent = [_mention_entity(text, "mybot")]
    msg = _message(chat=_supergroup_chat(), author=_user(2), text=text, entities=ent)
    asyncio.run(adapter._on_message(msg))
    assert adapter._inbound.qsize() == 0  # chat not in allowlist


# --------------------------------------------------------------- intake integration


def _intake_cfg(**kw: object) -> IntakeConfig:
    base: dict[str, object] = {
        "enabled": True,
        "mode": IntakeMode.ALL_CHANNELS,
        "triage_agent": "agentd-nagatha",
    }
    base.update(kw)
    return IntakeConfig(**base)  # type: ignore[arg-type]


def test_intake_hit_fires_sink_with_channel_identity() -> None:
    hits: list[InboundMessage] = []
    adapter = _adapter(
        intake_config=_intake_cfg(member_ids=frozenset({"nagatha"})),
        intake_sink=hits.append,
    )
    adapter._self_username = "mybot"
    adapter._self_user_id = "1"
    text = "@nagatha can you look at this"
    ent = [_mention_entity(text, "nagatha")]
    msg = _message(
        chat=_supergroup_chat(),
        author=_user(2, username="human"),
        text=text,
        entities=ent,
        message_id=8080,
    )
    asyncio.run(adapter._on_message(msg))
    assert len(hits) == 1
    assert hits[0].channel_id == SUPERGROUP_STR
    assert hits[0].message_id == "8080"
    assert hits[0].author == "2"


def test_intake_disabled_never_fires_sink() -> None:
    hits: list[InboundMessage] = []
    adapter = _adapter(
        intake_config=IntakeConfig(enabled=False, member_ids=frozenset({"nagatha"})),
        intake_sink=hits.append,
    )
    adapter._self_username = "mybot"
    adapter._self_user_id = "1"
    text = "@nagatha hey"
    ent = [_mention_entity(text, "nagatha")]
    msg = _message(chat=_supergroup_chat(), author=_user(2), text=text, entities=ent)
    asyncio.run(adapter._on_message(msg))
    assert hits == []


def test_intake_sink_failure_does_not_crash() -> None:
    def boom(_msg: InboundMessage) -> None:
        raise RuntimeError("sink exploded")

    adapter = _adapter(
        intake_config=_intake_cfg(member_ids=frozenset({"nagatha"})),
        intake_sink=boom,
    )
    adapter._self_username = "mybot"
    adapter._self_user_id = "1"
    text = "@nagatha hey"
    ent = [_mention_entity(text, "nagatha")]
    msg = _message(chat=_supergroup_chat(), author=_user(2), text=text, entities=ent)
    # Must not raise.
    asyncio.run(adapter._on_message(msg))


def test_intake_async_sink_is_awaited() -> None:
    published: list[InboundMessage] = []

    async def async_sink(m: InboundMessage) -> None:
        await asyncio.sleep(0)
        published.append(m)

    adapter = _adapter(
        intake_config=_intake_cfg(member_ids=frozenset({"nagatha"})),
        intake_sink=async_sink,
    )
    adapter._self_username = "mybot"
    adapter._self_user_id = "1"
    text = "@nagatha hey"
    ent = [_mention_entity(text, "nagatha")]
    msg = _message(
        chat=_supergroup_chat(), author=_user(2), text=text, entities=ent, message_id=8081
    )
    asyncio.run(adapter._on_message(msg))
    assert len(published) == 1
    assert published[0].message_id == "8081"


def test_intake_bot_author_ignored() -> None:
    # A bot author is never an intake hit under the default AllowBots.NONE.
    hits: list[InboundMessage] = []
    adapter = _adapter(
        intake_config=_intake_cfg(member_ids=frozenset({"nagatha"})),
        intake_sink=hits.append,
    )
    adapter._self_username = "mybot"
    adapter._self_user_id = "1"
    text = "@nagatha hey"
    ent = [_mention_entity(text, "nagatha")]
    msg = _message(chat=_supergroup_chat(), author=_user(9, is_bot=True), text=text, entities=ent)
    asyncio.run(adapter._on_message(msg))
    assert hits == []


# --------------------------------------------------------------- poll loop / stubs


class _FakeSession:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FakeBot:
    """A minimal aiogram-Bot stand-in for the poll loop (no network).

    Records the last getUpdates kwargs so tests can assert wire propagation
    (allowed_updates, request_timeout vs timeout).
    """

    def __init__(self, batches: list[list[Update]]) -> None:
        self._batches = batches
        self.session = _FakeSession()
        self.calls: list[dict[str, object]] = []

    async def get_me(self) -> User:
        return _user(1, username="mybot")

    async def get_updates(
        self,
        offset: int | None = None,
        timeout: int = 0,
        allowed_updates: object = None,
        request_timeout: int | None = None,
    ) -> list[Update]:
        self.calls.append(
            {
                "offset": offset,
                "timeout": timeout,
                "allowed_updates": allowed_updates,
                "request_timeout": request_timeout,
            }
        )
        if self._batches:
            return self._batches.pop(0)
        await asyncio.sleep(0)
        return []


def test_poll_loop_processes_and_stops() -> None:
    author = _user(2, username="human")
    text = "@mybot hey"
    ent = [_mention_entity(text, "mybot")]
    msg = _message(chat=_supergroup_chat(), author=author, text=text, entities=ent, message_id=10)
    update = Update(update_id=555, message=msg)
    bot = _FakeBot(batches=[[update]])
    adapter = _adapter(bot=bot)  # type: ignore[arg-type]

    async def drive() -> None:
        task = asyncio.create_task(adapter.start())
        # let the loop consume the batch
        for _ in range(10):
            await asyncio.sleep(0)
            if adapter._inbound.qsize() >= 1:
                break
        await adapter.close()
        await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(drive())
    assert adapter._inbound.qsize() == 1
    assert adapter._offset == 556  # update_id + 1
    assert bot.session.closed is True


class _FlakyBot(_FakeBot):
    """A fake bot whose FIRST poll raises (dead transport), then recovers."""

    def __init__(self, batches: list[list[Update]]) -> None:
        super().__init__(batches)
        self.failures = 0

    async def get_updates(
        self,
        offset: int | None = None,
        timeout: int = 0,
        allowed_updates: object = None,
        request_timeout: int | None = None,
    ) -> list[Update]:
        if self.failures == 0:
            self.failures += 1
            raise RuntimeError("transport down")
        return await super().get_updates(
            offset, timeout, allowed_updates, request_timeout=request_timeout
        )


def test_poll_failure_backs_off_and_loop_survives(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed poll pauses (never hot-spins) and the loop keeps polling after.

    Regression: a dead transport makes
    get_updates raise INSTANTLY (no long-poll hold spent), so without the
    failure backoff the loop spins flooding the error log. The backoff sleep
    lives on the failure branch only; a recovered transport still delivers.
    """
    # Shrink the real backoff so the test exercises the sleep without waiting.
    monkeypatch.setattr(telegram_adapter, "POLL_FAILURE_BACKOFF_SECONDS", 0.001)

    author = _user(2, username="human")
    text = "@mybot hey"
    ent = [_mention_entity(text, "mybot")]
    msg = _message(chat=_supergroup_chat(), author=author, text=text, entities=ent, message_id=11)
    update = Update(update_id=777, message=msg)
    bot = _FlakyBot(batches=[[update]])
    adapter = _adapter(bot=bot)  # type: ignore[arg-type]

    async def drive() -> None:
        task = asyncio.create_task(adapter.start())
        for _ in range(200):
            await asyncio.sleep(0.001)
            if adapter._inbound.qsize() >= 1:
                break
        await adapter.close()
        await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(drive())
    assert bot.failures == 1  # the failure branch ran
    assert adapter._inbound.qsize() == 1  # ...and the loop survived to deliver
    assert adapter._offset == 778


class _NeverReturnsBot(_FakeBot):
    """A fake bot whose get_updates NEVER returns on its own (parked long-poll).

    Models a live long-poll parked on the server-side hold: without the close-race
    in ``_poll_once`` a ``close()`` could not interrupt it. It honours cancellation
    (asyncio.wait cancels the poll future on close), so the loop unwinds promptly.
    """

    def __init__(self) -> None:
        super().__init__(batches=[])

    async def get_updates(
        self,
        offset: int | None = None,
        timeout: int = 0,
        allowed_updates: object = None,
        request_timeout: int | None = None,
    ) -> list[Update]:
        self.calls.append({"parked": True})
        await asyncio.Event().wait()  # never set → parks until cancelled
        return []


def test_request_timeout_exceeds_poll_timeout() -> None:
    # request_timeout (the aiogram HTTP timeout) MUST be larger than the
    # server-side long-poll hold (timeout), or a poll_timeout_s >= the HTTP
    # timeout would raise before the server ever responds. Regression for the
    # 60s-HTTP-timeout bricking finding.
    bot = _FakeBot(batches=[[]])
    adapter = _adapter(bot=bot, telegram_config=TelegramConfig(poll_timeout_s=30))  # type: ignore[arg-type]

    async def drive() -> None:
        task = asyncio.create_task(adapter.start())
        for _ in range(20):
            await asyncio.sleep(0)
            if bot.calls:
                break
        await adapter.close()
        await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(drive())
    call = bot.calls[0]
    assert call["request_timeout"] == 40  # poll_timeout_s (30) + 10 buffer
    assert call["request_timeout"] > call["timeout"]


def test_allowed_updates_propagate_to_get_updates() -> None:
    # The load-bearing gotcha: message_reaction (and message) MUST reach
    # getUpdates or the bot is blind to the events. Assert both are on the wire.
    bot = _FakeBot(batches=[[]])
    adapter = _adapter(bot=bot)  # type: ignore[arg-type]

    async def drive() -> None:
        task = asyncio.create_task(adapter.start())
        for _ in range(20):
            await asyncio.sleep(0)
            if bot.calls:
                break
        await adapter.close()
        await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(drive())
    allowed = bot.calls[0]["allowed_updates"]
    assert isinstance(allowed, list)
    assert "message_reaction" in allowed
    assert "message" in allowed


def test_logged_poll_exception_is_token_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    # A transport error whose str embeds the full Bot API URL (token in the path)
    # must be redacted before it reaches a log line — never the raw token.
    monkeypatch.setattr(telegram_adapter, "POLL_FAILURE_BACKOFF_SECONDS", 0.001)
    captured: list[str] = []
    sink_id = logger.add(lambda m: captured.append(m.record["message"]), level="ERROR")

    class _LeakyBot(_FakeBot):
        async def get_updates(
            self,
            offset: int | None = None,
            timeout: int = 0,
            allowed_updates: object = None,
            request_timeout: int | None = None,
        ) -> list[Update]:
            self.calls.append({"leak": True})
            raise RuntimeError(
                "connection failed to https://api.telegram.org/bot123:SECRET/getUpdates timed out"
            )

    bot = _LeakyBot(batches=[])
    adapter = _adapter(bot=bot)  # type: ignore[arg-type]

    async def drive() -> None:
        task = asyncio.create_task(adapter.start())
        for _ in range(50):
            await asyncio.sleep(0.001)
            if bot.calls:
                break
        await adapter.close()
        await asyncio.wait_for(task, timeout=1.0)

    try:
        asyncio.run(drive())
    finally:
        logger.remove(sink_id)

    joined = "\n".join(captured)
    assert "SECRET" not in joined
    assert "<redacted>" in joined


def test_close_interrupts_parked_long_poll_quickly() -> None:
    # A live long-poll parked on the server-side hold must be interruptible by
    # close(): the start() task completes promptly (well under any real backoff/
    # long-poll horizon), because _poll_once races the poll against _closed.
    bot = _NeverReturnsBot()
    adapter = _adapter(bot=bot)  # type: ignore[arg-type]

    async def drive() -> float:
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(adapter.start())
        for _ in range(50):
            await asyncio.sleep(0)
            if bot.calls:
                break
        t0 = loop.time()
        await adapter.close()
        await asyncio.wait_for(task, timeout=1.0)
        return loop.time() - t0

    elapsed = asyncio.run(drive())
    assert elapsed < 1.0  # no 5s/30s stall


def test_close_interrupts_failure_backoff() -> None:
    # The failure backoff is interruptible: after a poll error, close() ends the
    # backoff wait early instead of waiting out POLL_FAILURE_BACKOFF_SECONDS.
    class _AlwaysFailsBot(_FakeBot):
        async def get_updates(
            self,
            offset: int | None = None,
            timeout: int = 0,
            allowed_updates: object = None,
            request_timeout: int | None = None,
        ) -> list[Update]:
            self.calls.append({"fail": True})
            raise RuntimeError("transport down")

    # A large real backoff: only an interruptible wait finishes the test fast.
    import rtech_chatd.telegram_adapter as ta

    orig = ta.POLL_FAILURE_BACKOFF_SECONDS
    ta.POLL_FAILURE_BACKOFF_SECONDS = 30.0
    try:
        bot = _AlwaysFailsBot(batches=[])
        adapter = _adapter(bot=bot)  # type: ignore[arg-type]

        async def drive() -> None:
            task = asyncio.create_task(adapter.start())
            for _ in range(50):
                await asyncio.sleep(0)
                if bot.calls:
                    break
            await adapter.close()
            await asyncio.wait_for(task, timeout=1.0)  # would time out if backoff blocked

        asyncio.run(drive())
    finally:
        ta.POLL_FAILURE_BACKOFF_SECONDS = orig


def test_get_me_failure_leaves_identity_unset_and_loop_survives() -> None:
    # A get_me failure is best-effort: identity stays unset (mention-by-username
    # disabled) and the poll loop still runs and delivers.
    author = _user(2, username="human")
    text = "@mybot hey"
    ent = [_mention_entity(text, "mybot")]
    msg = _message(chat=_supergroup_chat(), author=author, text=text, entities=ent, message_id=12)
    update = Update(update_id=900, message=msg)

    class _NoIdentityBot(_FakeBot):
        async def get_me(self) -> User:
            raise RuntimeError("get_me failed")

    bot = _NoIdentityBot(batches=[[update]])
    adapter = _adapter(bot=bot)  # type: ignore[arg-type]

    async def drive() -> None:
        task = asyncio.create_task(adapter.start())
        for _ in range(20):
            await asyncio.sleep(0)
            if adapter._inbound.qsize() >= 1 or bot.calls:
                break
        await adapter.close()
        await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(drive())
    assert adapter._self_username is None
    assert adapter._self_user_id is None
    # DM-less group message not mentioning a resolved bot won't enqueue, but the
    # loop ran and offset advanced — the failure did not tear it down.
    assert adapter._offset == 901


def test_events_terminates_on_close() -> None:
    # events() must END (not block forever) once the adapter closes, draining any
    # already-queued message first.
    author = _user(2, username="human")
    text = "@mybot hey"
    ent = [_mention_entity(text, "mybot")]
    adapter = _adapter()
    adapter._self_username = "mybot"
    adapter._self_user_id = "1"

    async def run() -> list[str]:
        msg = _message(chat=_supergroup_chat(), author=author, text=text, entities=ent)
        await adapter._on_message(msg)  # enqueue one
        await adapter.close()
        seen: list[str] = []
        async for m in adapter.events():
            seen.append(m.message_id)
        return seen

    seen = asyncio.run(run())
    assert len(seen) == 1  # drained the backlog, then terminated


def test_bounded_inbound_queue_uses_constant() -> None:
    # The inbound queue is bounded by DEFAULT_INBOUND_QUEUE_MAX (backpressure:
    # a full queue stalls the poll loop, so the offset is not confirmed and
    # Telegram redelivers server-side — nothing is dropped).
    adapter = _adapter()
    assert adapter._inbound.maxsize == telegram_adapter.DEFAULT_INBOUND_QUEUE_MAX


def test_bounded_queue_full_put_stalls_then_unblocks() -> None:
    # Backpressure unit: a maxsize-1 queue blocks the second put until a consumer
    # drains one — the poll-loop stall is real, lossless flow control.
    adapter = TelegramChannelAdapter(
        token_provider=lambda: "unused",
        gate_config=GateConfig(),
    )
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
    adapter._inbound = q  # type: ignore[assignment]

    async def run() -> bool:
        await q.put("a")  # fills it
        put_task = asyncio.create_task(q.put("b"))  # must block
        await asyncio.sleep(0)
        blocked = not put_task.done()
        assert q.get_nowait() == "a"  # drain → unblocks the pending put
        await asyncio.wait_for(put_task, timeout=1.0)
        return blocked

    assert asyncio.run(run()) is True


def test_close_frees_poll_loop_parked_on_full_queue_put() -> None:
    # Fix-delta verification finding: the bounded queue made _on_message's put a
    # parked await close() previously could NOT interrupt — a stalled consumer +
    # full queue deadlocked shutdown. The put now races _closed like every other
    # await: close() must free the parked put promptly, releasing the message
    # un-enqueued (Telegram redelivers the unconfirmed batch on next start).
    author = _user(2, username="human")
    text = "@mybot hey"
    ent = [_mention_entity(text, "mybot")]
    adapter = _adapter()
    adapter._self_username = "mybot"
    adapter._self_user_id = "1"
    q: asyncio.Queue[InboundMessage] = asyncio.Queue(maxsize=1)
    adapter._inbound = q

    async def run() -> None:
        first = _message(
            chat=_supergroup_chat(), author=author, text=text, entities=ent, message_id=1
        )
        await adapter._on_message(first)  # fills the maxsize-1 queue
        second = _message(
            chat=_supergroup_chat(), author=author, text=text, entities=ent, message_id=2
        )
        parked = asyncio.create_task(adapter._on_message(second))
        await asyncio.sleep(0)
        assert not parked.done()  # genuinely parked on the full-queue put
        await adapter.close()
        await asyncio.wait_for(parked, timeout=1.0)  # freed promptly, no deadlock

    asyncio.run(run())
    assert q.qsize() == 1  # the parked message was released, not enqueued


def test_anonymous_admin_derives_author_from_sender_chat() -> None:
    # An anonymous supergroup admin arrives as the fake GroupAnonymousBot sender
    # with the real author in sender_chat: author = sender_chat.id, is_bot False.
    sender_chat = _supergroup_chat(-1005555555555)
    anon = _user(telegram_adapter.GROUP_ANONYMOUS_BOT_ID, is_bot=True, username="GroupAnonymousBot")
    msg = _message(
        chat=_supergroup_chat(),
        author=anon,
        text="admin speaking",
        sender_chat=sender_chat,
    )
    mapped = map_message(msg, self_username="mybot", self_user_id="1")
    assert mapped.author == "-1005555555555"
    assert mapped.is_bot is False  # a human admin, not a bot


def test_channel_post_no_from_user_author_empty() -> None:
    # A channel post has no from_user and no sender_chat here → author "".
    msg = _message(chat=_supergroup_chat(), author=None, text="posted")
    mapped = map_message(msg, self_username="mybot", self_user_id="1")
    assert mapped.author == ""
    assert mapped.is_bot is False


def test_start_after_close_raises() -> None:
    adapter = _adapter()

    async def run() -> None:
        await adapter.close()
        await adapter.start()

    with pytest.raises(RuntimeError, match="already-closed"):
        asyncio.run(run())


def test_dedup_not_touched_when_allowlist_drops_chat() -> None:
    # Ordering guard: the allowlist drop precedes dedup, so a dropped chat leaves
    # the dedup store empty (never records an id it also refused).
    adapter = _adapter(
        telegram_config=TelegramConfig(allowed_groups=frozenset({"-1009999999999"})),
    )
    adapter._self_username = "mybot"
    adapter._self_user_id = "1"
    text = "@mybot hey"
    ent = [_mention_entity(text, "mybot")]
    msg = _message(chat=_supergroup_chat(), author=_user(2), text=text, entities=ent)
    asyncio.run(adapter._on_message(msg))
    assert len(adapter._dedup) == 0


def test_utf16_mention_offset_extracts_correct_handle() -> None:
    # UTF-16 regression: an emoji (non-BMP, 2 UTF-16 code units) BEFORE the mention
    # shifts the Bot API's UTF-16 offset past a naive Python-codepoint slice. The
    # entity offset is computed in REAL UTF-16 code units; extract_from must still
    # pull "@alice". A raw text[offset:offset+length] slice fails this.
    prefix = "😀😀 "
    text = f"{prefix}@alice hi"
    offset = len(prefix.encode("utf-16-le")) // 2  # UTF-16 code units before @
    length = len("@alice")
    ent = MessageEntity(type="mention", offset=offset, length=length)
    msg = _message(chat=_supergroup_chat(), author=_user(2), text=text, entities=[ent])
    mapped = map_message(msg, self_username="mybot", self_user_id="1")
    assert mapped.mentioned_ids == frozenset({"alice"})


def test_start_rejects_empty_token() -> None:
    adapter = TelegramChannelAdapter(
        token_provider=lambda: "",
        gate_config=GateConfig(),
    )
    with pytest.raises(RuntimeError):
        asyncio.run(adapter.start())


def test_outbound_stubs_are_failsoft() -> None:
    adapter = _adapter()
    target = SendTarget(channel_id=SUPERGROUP_STR)
    assert asyncio.run(adapter.send(target, "hi", [])) is None
    assert asyncio.run(adapter.typing(target, True)) is None
    assert asyncio.run(adapter.create_thread(SUPERGROUP_STR, "1", "x")) is None


# --------------------------------------------------------------- protocol conformance


def test_adapter_satisfies_channel_adapter_protocol() -> None:
    adapter = _adapter()
    assert isinstance(adapter, ChannelAdapter)
