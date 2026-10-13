"""Fake-bot tests for the multi-identity Telegram sender (issue #109, outbound).

No live token, no network: a :class:`FakeBot` records every Bot API call and the
``bot_factory``/``default_bot_provider`` seams hand the sender fakes instead of
real aiogram clients. Time is deterministic — a real :class:`TelegramRateLimiter`
is injected with a fake clock/sleeper (the sender type-hints the concrete
limiter, so a real one keeps ``mypy --strict`` happy while staying wall-clock
free). Covers identity routing, chunked HTML sends, thread/reply resolution,
FloodWait penalty recording + re-raise, the entity-parse plain-resend degrade,
forum-topic creation vs reply degrade, receipt reactions, and typing keepalive.
"""

from __future__ import annotations

import asyncio

import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.methods import SendMessage
from aiogram.types import InlineKeyboardMarkup, ReactionTypeEmoji
from rtech_chatd import telegram_sender
from rtech_chatd.channel import ChunkProgress, SendTarget, chunk_progress_from
from rtech_chatd.telegram_rate_limiter import TelegramRateLimiter
from rtech_chatd.telegram_sender import (
    RECEIPT_REACTION,
    InteractiveMarkupIdentityError,
    TelegramSender,
    UnknownAgentIdentityError,
)


class FakeClock:
    """Mutable monotonic clock + clock-advancing sleeper for the limiter.

    ``sleep`` advances the clock by the slept amount so a limiter wait converges
    without any real time passing — a real :func:`asyncio.sleep` on this frozen
    clock would spin forever once a budget wait is required.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


def _fake_limiter() -> TelegramRateLimiter:
    """A real limiter driven by a fake clock/sleeper pair (wall-clock free)."""
    fake = FakeClock()
    return TelegramRateLimiter(clock=fake.clock, sleeper=fake.sleep)


class _FakeSession:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _SentMessage:
    def __init__(self, message_id: int) -> None:
        self.message_id = message_id


class _FakeChat:
    def __init__(self, is_forum: bool) -> None:
        self.is_forum = is_forum


class _FakeTopic:
    def __init__(self, message_thread_id: int) -> None:
        self.message_thread_id = message_thread_id


class FakeBot:
    """Records every Bot API call the sender makes; no network.

    ``send_message`` returns an object with an incrementing ``message_id`` so a
    multi-chunk send yields distinct ids and the FIRST is asserted. Behaviour
    hooks (``send_raises``, ``is_forum``, ``get_chat_raises`` …) let a test steer
    one call path without subclassing.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        send_raises: BaseException | None = None,
        second_send_raises: BaseException | None = None,
        send_failures: dict[int, BaseException] | None = None,
        is_forum: bool = False,
        get_chat_raises: bool = False,
        create_topic_id: int = 4242,
        react_raises: bool = False,
        callback_raises: BaseException | None = None,
        markup_raises: BaseException | None = None,
        edit_raises: BaseException | None = None,
    ) -> None:
        self.token = token
        self.session = _FakeSession()
        self._next_id = 100
        self._send_raises = send_raises
        self._second_send_raises = second_send_raises
        self._send_failures = dict(send_failures or {})
        self._is_forum = is_forum
        self._get_chat_raises = get_chat_raises
        self._create_topic_id = create_topic_id
        self._react_raises = react_raises
        self._callback_raises = callback_raises
        self._markup_raises = markup_raises
        self._edit_raises = edit_raises
        self.send_message_calls: list[dict[str, object]] = []
        self.send_document_calls: list[dict[str, object]] = []
        self.chat_action_calls: list[dict[str, object]] = []
        self.reaction_calls: list[dict[str, object]] = []
        self.create_topic_calls: list[tuple[int, str]] = []
        self.callback_calls: list[dict[str, object]] = []
        self.markup_calls: list[dict[str, object]] = []
        self.edit_calls: list[dict[str, object]] = []

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        parse_mode: str | None = None,
        message_thread_id: int | None = None,
        reply_parameters: object = None,
        reply_markup: object = None,
    ) -> _SentMessage:
        self.send_message_calls.append(
            {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "message_thread_id": message_thread_id,
                "reply_parameters": reply_parameters,
                "reply_markup": reply_markup,
            }
        )
        failure = self._send_failures.get(len(self.send_message_calls))
        if failure is not None:
            raise failure
        # First call may raise; a queued second-raise fires on the resend path.
        if len(self.send_message_calls) == 1 and self._send_raises is not None:
            raise self._send_raises
        if len(self.send_message_calls) == 2 and self._second_send_raises is not None:
            raise self._second_send_raises
        self._next_id += 1
        return _SentMessage(self._next_id)

    async def send_document(
        self, chat_id: int, document: object, *, message_thread_id: int | None = None
    ) -> _SentMessage:
        self.send_document_calls.append(
            {"chat_id": chat_id, "document": document, "message_thread_id": message_thread_id}
        )
        self._next_id += 1
        return _SentMessage(self._next_id)

    async def send_chat_action(
        self, chat_id: int, action: str, *, message_thread_id: int | None = None
    ) -> None:
        self.chat_action_calls.append(
            {"chat_id": chat_id, "action": action, "message_thread_id": message_thread_id}
        )

    async def get_chat(self, chat_id: int) -> _FakeChat:
        if self._get_chat_raises:
            raise RuntimeError("get_chat failed")
        return _FakeChat(self._is_forum)

    async def create_forum_topic(self, chat_id: int, name: str) -> _FakeTopic:
        self.create_topic_calls.append((chat_id, name))
        return _FakeTopic(self._create_topic_id)

    async def set_message_reaction(
        self, chat_id: int, message_id: int, *, reaction: object = None
    ) -> None:
        if self._react_raises:
            raise RuntimeError("reaction failed")
        self.reaction_calls.append(
            {"chat_id": chat_id, "message_id": message_id, "reaction": reaction}
        )

    async def answer_callback_query(
        self, callback_query_id: str, *, text: str | None = None, show_alert: bool = False
    ) -> None:
        if self._callback_raises is not None:
            raise self._callback_raises
        self.callback_calls.append(
            {"callback_query_id": callback_query_id, "text": text, "show_alert": show_alert}
        )

    async def edit_message_reply_markup(
        self, *, chat_id: int, message_id: int, reply_markup: object = None
    ) -> None:
        if self._markup_raises is not None:
            raise self._markup_raises
        self.markup_calls.append(
            {"chat_id": chat_id, "message_id": message_id, "reply_markup": reply_markup}
        )

    async def edit_message_text(
        self,
        text: str,
        *,
        chat_id: int,
        message_id: int,
        parse_mode: str | None = None,
        reply_markup: object = None,
    ) -> None:
        if self._edit_raises is not None:
            raise self._edit_raises
        self.edit_calls.append(
            {
                "text": text,
                "chat_id": chat_id,
                "message_id": message_id,
                "parse_mode": parse_mode,
                "reply_markup": reply_markup,
            }
        )


def _flood(retry_after: int = 7) -> TelegramRetryAfter:
    return TelegramRetryAfter(
        method=SendMessage(chat_id=1, text="x"), message="flood", retry_after=retry_after
    )


def _bad_request(message: str) -> TelegramBadRequest:
    return TelegramBadRequest(method=SendMessage(chat_id=1, text="x"), message=message)


def _sender(
    *,
    nagatha: FakeBot | None = None,
    bilby: FakeBot | None = None,
    default: FakeBot | None = None,
    limiter: TelegramRateLimiter | None = None,
) -> tuple[TelegramSender, dict[str, FakeBot]]:
    """Build a sender wired to fakes; return it plus the built-bot registry.

    The ``bot_factory`` keys built bots by the token it is handed, so a test can
    assert WHICH identity's token produced the client that sent a message.
    """
    built: dict[str, FakeBot] = {}
    n_bot = nagatha or FakeBot()
    b_bot = bilby or FakeBot()
    d_bot = default or FakeBot()

    def factory(token: str) -> FakeBot:
        chosen = {"tok-n": n_bot, "tok-b": b_bot}[token]
        chosen.token = token
        built[token] = chosen
        return chosen  # type: ignore[return-value]

    sender = TelegramSender(
        identity_token_providers={
            "agentd-nagatha": lambda: "tok-n",
            "agentd-bilby": lambda: "tok-b",
        },
        rate_limiter=limiter or _fake_limiter(),
        bot_factory=factory,  # type: ignore[arg-type]
        default_bot_provider=lambda: d_bot,  # type: ignore[arg-type,return-value]
    )
    return sender, built


# --------------------------------------------------------------- identity routing


def test_send_as_agent_uses_that_agents_bot() -> None:
    nagatha = FakeBot()
    sender, built = _sender(nagatha=nagatha)
    target = SendTarget(channel_id="555", as_agent="agentd-nagatha")
    asyncio.run(sender.send(target, "hi", []))
    assert "tok-n" in built  # nagatha's token built the client
    assert len(nagatha.send_message_calls) == 1


def test_send_as_agent_none_uses_default_bot() -> None:
    default = FakeBot()
    sender, built = _sender(default=default)
    target = SendTarget(channel_id="555", as_agent=None)
    asyncio.run(sender.send(target, "hi", []))
    assert built == {}  # no identity client built
    assert len(default.send_message_calls) == 1


def test_send_unknown_agent_raises_and_sends_nothing() -> None:
    nagatha = FakeBot()
    default = FakeBot()
    sender, _ = _sender(nagatha=nagatha, default=default)
    target = SendTarget(channel_id="555", as_agent="agentd-ghost")
    with pytest.raises(UnknownAgentIdentityError):
        asyncio.run(sender.send(target, "hi", []))
    assert nagatha.send_message_calls == []
    assert default.send_message_calls == []  # nothing sent under any identity


# --------------------------------------------------------------- chunking / ids


def test_send_returns_first_chunk_message_id_as_str() -> None:
    default = FakeBot()
    sender, _ = _sender(default=default)
    result = asyncio.run(sender.send(SendTarget(channel_id="555"), "hi", []))
    # FakeBot starts at 100, first send returns 101.
    assert result == "101"
    assert isinstance(result, str)


def test_multi_chunk_send_all_html_parse_mode() -> None:
    default = FakeBot()
    sender, _ = _sender(default=default)
    long_text = "\n".join(f"line {i} filler content to force multiple chunks" for i in range(400))
    result = asyncio.run(sender.send(SendTarget(channel_id="555"), long_text, []))
    assert len(default.send_message_calls) > 1  # multiple chunks sent
    assert all(call["parse_mode"] == "HTML" for call in default.send_message_calls)
    assert result == "101"  # first chunk's id


def test_send_serializes_reply_markup_on_first_chunk_only() -> None:
    default = FakeBot()
    sender, _ = _sender(default=default)
    markup = InlineKeyboardMarkup(inline_keyboard=[])
    long_text = "\n".join(f"line {i} filler content to force multiple chunks" for i in range(400))
    asyncio.run(sender.send(SendTarget(channel_id="555"), long_text, [], reply_markup=markup))
    assert default.send_message_calls[0]["reply_markup"] == markup
    assert all(call["reply_markup"] is None for call in default.send_message_calls[1:])


def test_interactive_markup_refused_on_agent_stamped_send() -> None:
    # An agent-identity bot runs no getUpdates poll loop, so a button under it
    # could never deliver its tap. The sender refuses loud and sends nothing —
    # no dead button ever reaches a human.
    nagatha = FakeBot()
    default = FakeBot()
    sender, _ = _sender(nagatha=nagatha, default=default)
    markup = InlineKeyboardMarkup(inline_keyboard=[])
    target = SendTarget(channel_id="555", as_agent="agentd-nagatha")
    with pytest.raises(InteractiveMarkupIdentityError):
        asyncio.run(sender.send(target, "decide", [], reply_markup=markup))
    assert nagatha.send_message_calls == []
    assert default.send_message_calls == []


def test_interactive_markup_allowed_on_default_identity() -> None:
    # The polled default identity DOES receive taps, so interactive markup rides
    # it unchanged (the unstamped send is the pre-#117 contract).
    default = FakeBot()
    sender, _ = _sender(default=default)
    markup = InlineKeyboardMarkup(inline_keyboard=[])
    asyncio.run(sender.send(SendTarget(channel_id="555"), "decide", [], reply_markup=markup))
    assert default.send_message_calls[0]["reply_markup"] == markup


def test_third_chunk_flood_wait_keeps_type_and_exposes_two_landed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial FloodWait retains the exact exception and a typed cursor."""

    monkeypatch.setattr(telegram_sender, "telegram_chunks", lambda _: ["one", "two", "three"])
    flood = _flood(7)
    default = FakeBot(send_failures={3: flood})
    limiter = _fake_limiter()
    sender, _ = _sender(default=default, limiter=limiter)

    with pytest.raises(TelegramRetryAfter) as excinfo:
        asyncio.run(sender.send(SendTarget(channel_id="555"), "logical message", []))

    assert excinfo.value is flood
    assert limiter.flood_wait_remaining("555") > 0.0
    progress = chunk_progress_from(excinfo.value)
    assert progress is not None
    assert progress.chunks_landed == 2
    assert progress.chunk_count == 3
    assert progress.plan_fingerprint

    # Redelivery uses the exact plan checkpoint, so only the failed third
    # chunk transmits; the first two are never duplicated.
    asyncio.run(
        sender.send(
            SendTarget(channel_id="555", chunk_resume=progress),
            "logical message",
            [],
        )
    )
    assert [call["text"] for call in default.send_message_calls] == ["one", "two", "three", "three"]


def test_chunk_plan_mismatch_restarts_at_first_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale checkpoint cannot silently skip a changed chunk plan."""

    monkeypatch.setattr(telegram_sender, "telegram_chunks", lambda _: ["one", "two", "three"])
    default = FakeBot()
    sender, _ = _sender(default=default)
    stale = ChunkProgress(chunks_landed=2, chunk_count=3, plan_fingerprint="different-plan")

    asyncio.run(sender.send(SendTarget(channel_id="555", chunk_resume=stale), "changed", []))

    assert [call["text"] for call in default.send_message_calls] == ["one", "two", "three"]


# --------------------------------------------------------------- thread / reply


def test_numeric_thread_id_passed_as_message_thread_id_int() -> None:
    default = FakeBot()
    sender, _ = _sender(default=default)
    target = SendTarget(channel_id="-100999", thread_id="77")
    asyncio.run(sender.send(target, "hi", []))
    assert default.send_message_calls[0]["message_thread_id"] == 77


def test_reply_marker_thread_id_sets_reply_parameters_first_chunk_only() -> None:
    default = FakeBot()
    sender, _ = _sender(default=default)
    long_text = "\n".join(f"line {i} filler content to force multiple chunks" for i in range(400))
    target = SendTarget(channel_id="-100999", thread_id="reply:123")
    asyncio.run(sender.send(target, long_text, []))
    calls = default.send_message_calls
    assert len(calls) > 1
    # reply_parameters on the FIRST chunk only.
    first_rp = calls[0]["reply_parameters"]
    assert first_rp is not None
    assert getattr(first_rp, "message_id", None) == 123
    assert all(call["reply_parameters"] is None for call in calls[1:])


def test_explicit_reply_to_wins_over_marker() -> None:
    default = FakeBot()
    sender, _ = _sender(default=default)
    target = SendTarget(channel_id="-100999", thread_id="reply:123", reply_to="999")
    asyncio.run(sender.send(target, "hi", []))
    rp = default.send_message_calls[0]["reply_parameters"]
    assert getattr(rp, "message_id", None) == 999  # explicit reply_to wins


# --------------------------------------------------------------- FloodWait


def test_flood_wait_records_penalty_and_propagates() -> None:
    default = FakeBot(send_raises=_flood(7))
    limiter = _fake_limiter()
    sender, _ = _sender(default=default, limiter=limiter)
    target = SendTarget(channel_id="555")
    with pytest.raises(TelegramRetryAfter):
        asyncio.run(sender.send(target, "hi", []))
    # The penalty was recorded in the limiter under the chat scope.
    assert limiter.flood_wait_remaining("555") > 0.0


def test_callback_answer_records_flood_wait_and_propagates() -> None:
    default = FakeBot(callback_raises=_flood(7))
    limiter = _fake_limiter()
    sender, _ = _sender(default=default, limiter=limiter)
    with pytest.raises(TelegramRetryAfter):
        asyncio.run(sender.answer_callback("cq-1", channel_id="555"))
    assert limiter.flood_wait_remaining("555") > 0.0


def test_callback_answer_failure_is_contained() -> None:
    default = FakeBot(callback_raises=RuntimeError("answer failed"))
    sender, _ = _sender(default=default)
    assert asyncio.run(sender.answer_callback("cq-1", channel_id="555")) is False


# --------------------------------------------------------------- parse degrade


def test_parse_error_triggers_plain_resend_and_returns_its_id() -> None:
    default = FakeBot(send_raises=_bad_request("Bad Request: can't parse entities: bad tag"))
    sender, _ = _sender(default=default)
    result = asyncio.run(sender.send(SendTarget(channel_id="555"), "hi", []))
    calls = default.send_message_calls
    assert len(calls) == 2  # HTML attempt (raised) + plain resend
    assert calls[0]["parse_mode"] == "HTML"
    assert calls[1]["parse_mode"] is None  # resend has NO parse_mode
    assert result == "101"  # the resend's message id


def test_bad_request_without_marker_propagates() -> None:
    default = FakeBot(send_raises=_bad_request("Bad Request: chat not found"))
    sender, _ = _sender(default=default)
    with pytest.raises(TelegramBadRequest):
        asyncio.run(sender.send(SendTarget(channel_id="555"), "hi", []))
    assert len(default.send_message_calls) == 1  # no resend for a non-parse 400


# --------------------------------------------------------------- create_thread


def test_create_thread_forum_creates_topic_returns_id() -> None:
    default = FakeBot(is_forum=True, create_topic_id=808)
    sender, _ = _sender(default=default)
    result = asyncio.run(sender.create_thread("-100999", "42", "Debate"))
    assert result == "808"
    assert default.create_topic_calls == [(-100999, "Debate")]


def test_create_thread_non_forum_returns_reply_marker_no_topic() -> None:
    default = FakeBot(is_forum=False)
    sender, _ = _sender(default=default)
    result = asyncio.run(sender.create_thread("-100999", "42", "Debate"))
    assert result == "reply:42"
    assert default.create_topic_calls == []  # never attempted a topic


def test_create_thread_get_chat_failure_returns_none() -> None:
    default = FakeBot(get_chat_raises=True)
    sender, _ = _sender(default=default)
    result = asyncio.run(sender.create_thread("-100999", "42", "Debate"))
    assert result is None


# --------------------------------------------------------------- react_receipt


def test_react_receipt_sets_eyes_reaction() -> None:
    default = FakeBot()
    sender, _ = _sender(default=default)
    asyncio.run(sender.react_receipt("-100999", "42"))
    assert len(default.reaction_calls) == 1
    reaction = default.reaction_calls[0]["reaction"]
    assert isinstance(reaction, list)
    assert isinstance(reaction[0], ReactionTypeEmoji)
    assert reaction[0].emoji == RECEIPT_REACTION


def test_react_receipt_failure_is_contained() -> None:
    default = FakeBot(react_raises=True)
    sender, _ = _sender(default=default)
    # Must not raise despite the bot raising.
    asyncio.run(sender.react_receipt("-100999", "42"))


def test_callback_markup_and_text_edits_use_typed_markup() -> None:
    default = FakeBot()
    sender, _ = _sender(default=default)
    markup = InlineKeyboardMarkup(inline_keyboard=[])
    target = SendTarget(channel_id="-100999")
    assert asyncio.run(sender.edit_reply_markup(target, "42", reply_markup=markup)) is True
    assert (
        asyncio.run(sender.edit_text(target, "42", "<b>updated</b>", reply_markup=markup)) is True
    )
    assert default.markup_calls[0]["reply_markup"] == markup
    assert default.edit_calls[0]["parse_mode"] == "HTML"
    assert default.edit_calls[0]["reply_markup"] == markup


def test_callback_edits_contain_failure_but_propagate_flood_wait() -> None:
    target = SendTarget(channel_id="555")
    sender, _ = _sender(default=FakeBot(markup_raises=RuntimeError("edit failed")))
    assert asyncio.run(sender.edit_reply_markup(target, "42")) is False
    limiter = _fake_limiter()
    sender, _ = _sender(default=FakeBot(edit_raises=_flood(9)), limiter=limiter)
    with pytest.raises(TelegramRetryAfter):
        asyncio.run(sender.edit_text(target, "42", "updated"))
    assert limiter.flood_wait_remaining("555") > 0.0


# --------------------------------------------------------------- typing


async def test_typing_on_spawns_keepalive_then_off_cancels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Shrink the refresh so a real event loop drives >=2 chat actions fast.
    monkeypatch.setattr(telegram_sender, "TYPING_REFRESH_INTERVAL_S", 0.01)
    default = FakeBot()
    sender, _ = _sender(default=default)
    target = SendTarget(channel_id="555")

    await sender.typing(target, True)
    # Let the keepalive fire a few times.
    for _ in range(50):
        await asyncio.sleep(0.005)
        if len(default.chat_action_calls) >= 2:
            break
    assert len(default.chat_action_calls) >= 2
    assert sender._typing_tasks  # task is live

    await sender.typing(target, False)
    await asyncio.sleep(0)  # let the cancellation settle
    assert sender._typing_tasks == {}  # task removed


async def test_typing_double_on_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telegram_sender, "TYPING_REFRESH_INTERVAL_S", 0.01)
    default = FakeBot()
    sender, _ = _sender(default=default)
    target = SendTarget(channel_id="555")
    await sender.typing(target, True)
    await sender.typing(target, True)  # second on-call is a no-op
    assert len(sender._typing_tasks) == 1
    await sender.typing(target, False)
    await asyncio.sleep(0)


async def test_typing_expires_when_no_terminal_turn_stops_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lost terminal TurnFrame releases the sender-owned typing lease."""

    monkeypatch.setattr(telegram_sender, "TYPING_REFRESH_INTERVAL_S", 0.005)
    monkeypatch.setattr(telegram_sender, "TYPING_MAX_LIFETIME_S", 0.02)
    default = FakeBot()
    sender, _ = _sender(default=default)
    target = SendTarget(channel_id="555")

    await sender.typing(target, True)
    await asyncio.sleep(0.06)

    assert default.chat_action_calls
    assert sender._typing_tasks == {}


async def test_close_cancels_typing_and_closes_pool_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telegram_sender, "TYPING_REFRESH_INTERVAL_S", 0.01)
    nagatha = FakeBot()
    default = FakeBot()
    sender, _ = _sender(nagatha=nagatha, default=default)
    # Build the nagatha pool client via a send, and start a typing loop.
    await sender.send(SendTarget(channel_id="555", as_agent="agentd-nagatha"), "hi", [])
    await sender.typing(SendTarget(channel_id="555"), True)
    await asyncio.sleep(0.005)
    assert sender._typing_tasks

    await sender.close()
    await asyncio.sleep(0)
    assert sender._typing_tasks == {}  # typing loops cancelled
    assert nagatha.session.closed is True  # pool client session closed


# ------------------------------------------- gauntlet fix regressions (PR #122)


def test_plain_resend_flood_wait_records_penalty_and_propagates() -> None:
    # The HTML attempt trips the entity parser; the PLAIN resend then serves a
    # 429. The resend goes through the SAME FloodWait-recording path as the
    # original send — the gauntlet caught the degrade path bypassing the ledger.
    default = FakeBot(
        send_raises=_bad_request("Bad Request: can't parse entities: bad tag"),
        second_send_raises=_flood(9),
    )
    limiter = _fake_limiter()
    sender, _ = _sender(default=default, limiter=limiter)
    with pytest.raises(TelegramRetryAfter):
        asyncio.run(sender.send(SendTarget(channel_id="555"), "hi", []))
    assert len(default.send_message_calls) == 2  # HTML attempt + plain resend
    assert limiter.flood_wait_remaining("555") > 0.0  # penalty recorded


def test_transport_error_is_sanitized_and_token_redacted() -> None:
    # aiohttp-style errors can stringify to the full request URL with the bot
    # token embedded; the raised replacement must carry only redacted text and
    # the original exception must never travel upward.
    boom = RuntimeError(
        "Cannot connect to https://api.telegram.org/bot1234567:AAH-secret/sendMessage"
    )
    default = FakeBot(send_raises=boom)
    sender, _ = _sender(default=default)
    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(sender.send(SendTarget(channel_id="555"), "hi", []))
    raised = excinfo.value
    assert raised is not boom  # replaced, not re-raised
    assert "AAH-secret" not in str(raised)
    assert "/bot<redacted>" in str(raised)


async def test_typing_restart_survives_cancelled_predecessor_unwind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # typing(off) -> typing(on) installs a NEW task under the same key before
    # the cancelled loop unwinds; the old loop's finally must not pop (orphan)
    # the fresh task.
    monkeypatch.setattr(telegram_sender, "TYPING_REFRESH_INTERVAL_S", 0.01)
    default = FakeBot()
    sender, _ = _sender(default=default)
    target = SendTarget(channel_id="555")
    await sender.typing(target, True)
    await sender.typing(target, False)
    await sender.typing(target, True)  # restart before the old task unwinds
    await asyncio.sleep(0.05)  # let the cancelled predecessor run its finally
    task = sender._typing_tasks.get(("555", "", ""))
    assert task is not None  # fresh task still registered
    assert not task.cancelled()
    calls_before = len(default.chat_action_calls)
    await asyncio.sleep(0.05)
    assert len(default.chat_action_calls) > calls_before  # keepalive still live
    await sender.typing(target, False)
    await asyncio.sleep(0)


async def test_typing_keys_are_per_topic_within_one_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two topics of ONE chat keep independent indicator loops; stopping one
    # never cancels the other (the key carries the thread dimension).
    monkeypatch.setattr(telegram_sender, "TYPING_REFRESH_INTERVAL_S", 0.01)
    default = FakeBot()
    sender, _ = _sender(default=default)
    topic_a = SendTarget(channel_id="-100999", thread_id="77")
    topic_b = SendTarget(channel_id="-100999", thread_id="88")
    await sender.typing(topic_a, True)
    await sender.typing(topic_b, True)
    assert len(sender._typing_tasks) == 2  # one loop per topic
    await sender.typing(topic_a, False)
    await asyncio.sleep(0)
    assert list(sender._typing_tasks) == [("-100999", "88", "")]
    await sender.typing(topic_b, False)
    await asyncio.sleep(0)


class _YieldingBot(FakeBot):
    """A FakeBot whose send yields to the loop first — makes interleaving of
    concurrent senders POSSIBLE, so the per-(chat, identity) lock is what the
    contiguity assertion actually exercises."""

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        parse_mode: str | None = None,
        message_thread_id: int | None = None,
        reply_parameters: object = None,
        reply_markup: object = None,
    ) -> _SentMessage:
        await asyncio.sleep(0)
        return await super().send_message(
            chat_id,
            text,
            parse_mode=parse_mode,
            message_thread_id=message_thread_id,
            reply_parameters=reply_parameters,
            reply_markup=reply_markup,
        )


async def test_concurrent_multichunk_sends_do_not_interleave() -> None:
    # Two concurrent multi-chunk sends from the SAME identity to the SAME chat
    # must serialize: without the send lock the chunks arrive A1, B1, A2 and
    # split A's continuation in the surface.
    default = _YieldingBot()
    sender, _ = _sender(default=default)
    text_a = "\n".join(f"alpha {i} filler content to force multiple chunks" for i in range(400))
    text_b = "\n".join(f"beta {i} filler content to force multiple chunks" for i in range(400))
    target = SendTarget(channel_id="555")
    await asyncio.gather(sender.send(target, text_a, []), sender.send(target, text_b, []))
    kinds = ["a" if "alpha" in str(call["text"]) else "b" for call in default.send_message_calls]
    assert len(kinds) > 2  # both messages actually chunked
    switches = sum(1 for prev, cur in zip(kinds, kinds[1:], strict=False) if prev != cur)
    assert switches == 1  # exactly one contiguous block per logical message
