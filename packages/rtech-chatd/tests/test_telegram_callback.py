"""Functional tests for Telegram callback authorization and typed steering."""

from __future__ import annotations

import asyncio
from pathlib import Path

from aiogram.types import InlineKeyboardMarkup
from loguru import logger
from rtech_chatd import subjects
from rtech_chatd.callback_sink import TelegramCallbackSink, TelegramTurnInteraction
from rtech_chatd.channel import SendTarget
from rtech_chatd.telegram_adapter import InboundCallback
from rtech_chatd.telegram_callback import TelegramCallbackRegistry
from rtech_frames.codec import decode_frame
from rtech_frames.frames import SteerFrame


class FakeBus:
    """Capture callback publications without a NATS connection."""

    def __init__(self) -> None:
        self.published: list[tuple[str, bytes]] = []

    async def publish(self, subject: str, data: bytes) -> None:
        self.published.append((subject, data))

    async def subscribe(self, subject: str, handler: object) -> object:
        raise NotImplementedError


class _FlakyBus(FakeBus):
    """Fail the first ``fail_times`` publishes, then behave like FakeBus."""

    def __init__(self, *, fail_times: int) -> None:
        super().__init__()
        self._fail_times = fail_times

    async def publish(self, subject: str, data: bytes) -> None:
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("bus down")
        await super().publish(subject, data)


class FakeSender:
    """Capture safe callback acknowledgements + markup clears without Telegram."""

    def __init__(self) -> None:
        self.answers: list[tuple[str, str | None]] = []
        # (channel_id, message_id, as_agent) of every markup edit — a committed tap
        # clears its spent keyboard so it cannot invite a confusing replay.
        self.markup_edits: list[tuple[str, str, str | None]] = []

    async def answer_callback(
        self, callback_query_id: str, *, channel_id: str | None = None
    ) -> bool:
        self.answers.append((callback_query_id, channel_id))
        return True

    async def edit_reply_markup(
        self, target: SendTarget, message_id: str, *, reply_markup: object = None
    ) -> bool:
        self.markup_edits.append((target.channel_id, message_id, target.as_agent))
        return True


class Clock:
    """Provide a deterministic monotonic source for callback expiry."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _callback(
    data: str | None, *, channel_id: str = "-10099", message_id: str = "42"
) -> InboundCallback:
    return InboundCallback(
        callback_query_id="cq-1",
        channel_id=channel_id,
        message_id=message_id,
        author="human-1",
        data=data,
    )


def _intruder_callback(data: str | None, *, author: str) -> InboundCallback:
    """A tap on the same message/chat but from a DIFFERENT Telegram user."""
    return InboundCallback(
        callback_query_id="cq-1",
        channel_id="-10099",
        message_id="42",
        author=author,
        data=data,
    )


def _sink(clock: Clock) -> tuple[TelegramCallbackSink, FakeBus, FakeSender]:
    bus = FakeBus()
    sender = FakeSender()
    registry = TelegramCallbackRegistry(secret=b"test-secret", clock=clock)
    return TelegramCallbackSink(bus=bus, sender=sender, registry=registry), bus, sender  # type: ignore[arg-type]


def test_valid_callback_publishes_only_a_typed_steer_frame() -> None:
    clock = Clock()
    sink, bus, sender = _sink(clock)
    data = sink.issue(
        action="revise",
        task_id="task-123",
        target_agent="agentd-bilby",
        channel_id="-10099",
        message_id="42",
        actor="human-1",
    )

    asyncio.run(sink(_callback(data)))

    assert sender.answers == [("cq-1", "-10099")]
    assert [subject for subject, _data in bus.published] == [subjects.task_steer("task-123")]
    frame = decode_frame(bus.published[0][1])
    assert isinstance(frame, SteerFrame)
    assert frame.task_id == "task-123"
    assert frame.target_agent == "agentd-bilby"
    assert frame.interrupt is True
    assert "done" not in frame.text.lower()
    # A committed tap clears its spent keyboard on the POLLED default bot
    # (as_agent None) so it cannot invite a confusing replay tap.
    assert sender.markup_edits == [("-10099", "42", None)]


def test_malformed_unknown_and_tampered_callbacks_ack_without_publish() -> None:
    clock = Clock()
    sink, bus, sender = _sink(clock)
    issued = sink.issue(
        action="approve",
        task_id="task-123",
        target_agent="agentd-bilby",
        channel_id="-10099",
        message_id="42",
        actor="human-1",
    )

    for data in (
        None,
        "v1|approve",
        "v1|unknown|nonce|signature",
        issued.replace("approve", "stop"),
    ):
        asyncio.run(sink(_callback(data)))

    assert bus.published == []
    assert len(sender.answers) == 4
    # No tap committed, so no keyboard was cleared — an invalid tap never disables
    # a still-live button.
    assert sender.markup_edits == []


def test_expired_mismatched_and_replayed_callbacks_fail_closed() -> None:
    clock = Clock()
    sink, bus, sender = _sink(clock)
    expired = sink.issue(
        action="approve",
        task_id="expired-task",
        target_agent="agentd-bilby",
        channel_id="-10099",
        message_id="42",
        actor="human-1",
    )
    clock.now += 901.0
    asyncio.run(sink(_callback(expired)))

    valid = sink.issue(
        action="stop",
        task_id="task-123",
        target_agent="agentd-nagatha",
        channel_id="-10099",
        message_id="42",
        actor="human-1",
    )
    asyncio.run(sink(_callback(valid, channel_id="-10000")))
    asyncio.run(sink(_callback(valid, message_id="43")))
    asyncio.run(sink(_callback(valid)))
    asyncio.run(sink(_callback(valid)))

    assert len(bus.published) == 1
    frame = decode_frame(bus.published[0][1])
    assert isinstance(frame, SteerFrame)
    assert frame.task_id == "task-123"
    assert frame.target_agent == "agentd-nagatha"
    assert len(sender.answers) == 5


def test_capability_is_bound_to_its_issued_actor() -> None:
    clock = Clock()
    sink, bus, sender = _sink(clock)
    data = sink.issue(
        action="approve",
        task_id="task-123",
        target_agent="agentd-bilby",
        channel_id="-10099",
        message_id="42",
        actor="human-1",
    )

    # A DIFFERENT chat member taps the same valid button: fails closed, still
    # acknowledged (spinner stops), never publishes a steer.
    asyncio.run(sink(_intruder_callback(data, author="intruder-9")))
    assert bus.published == []
    assert sender.answers == [("cq-1", "-10099")]

    # The intended actor can still act on it afterwards — the intruder's tap did
    # not spend the capability.
    asyncio.run(sink(_callback(data)))
    assert len(bus.published) == 1
    frame = decode_frame(bus.published[0][1])
    assert isinstance(frame, SteerFrame)
    assert frame.target_agent == "agentd-bilby"


def test_failed_publish_rolls_back_so_the_tap_is_retryable() -> None:
    clock = Clock()
    bus = _FlakyBus(fail_times=1)
    sender = FakeSender()
    registry = TelegramCallbackRegistry(secret=b"test-secret", clock=clock)
    sink = TelegramCallbackSink(bus=bus, sender=sender, registry=registry)  # type: ignore[arg-type]
    data = sink.issue(
        action="stop",
        task_id="task-123",
        target_agent="agentd-bilby",
        channel_id="-10099",
        message_id="42",
        actor="human-1",
    )

    # First tap: publish raises. The reservation must roll back, so nothing lands
    # yet the human is acknowledged and the capability stays live. A rolled-back
    # (retryable) tap must NOT clear its keyboard — the human still needs it.
    asyncio.run(sink(_callback(data)))
    assert bus.published == []
    assert sender.answers == [("cq-1", "-10099")]
    assert sender.markup_edits == []

    # Retry: the bus is healthy now, and because the failed tap rolled back its
    # reservation, the retry is NOT rejected as replayed — the steer lands, and
    # only NOW is the spent keyboard cleared.
    asyncio.run(sink(_callback(data)))
    assert len(bus.published) == 1
    frame = decode_frame(bus.published[0][1])
    assert isinstance(frame, SteerFrame)
    assert frame.task_id == "task-123"
    assert sender.markup_edits == [("-10099", "42", None)]

    # A THIRD tap is now a genuine replay of the committed capability: rejected,
    # and the keyboard is not cleared a second time.
    asyncio.run(sink(_callback(data)))
    assert len(bus.published) == 1
    assert sender.markup_edits == [("-10099", "42", None)]


# The sentinel string a leaked publish exception would smuggle into a log line if
# the sink logged the raw exception text instead of only its type name.
_SECRET_SENTINEL = "SUPER_SECRET_NATS_URL"


class _SecretLeakBus(FakeBus):
    """Raise a publish exception whose message carries the NATS URL sentinel."""

    async def publish(self, subject: str, data: bytes) -> None:
        raise RuntimeError(f"connection to nats://{_SECRET_SENTINEL}:4222 refused")


def test_publish_failure_never_logs_raw_exception_text() -> None:
    clock = Clock()
    bus = _SecretLeakBus()
    sender = FakeSender()
    registry = TelegramCallbackRegistry(secret=b"test-secret", clock=clock)
    sink = TelegramCallbackSink(bus=bus, sender=sender, registry=registry)  # type: ignore[arg-type]
    data = sink.issue(
        action="stop",
        task_id="task-123",
        target_agent="agentd-bilby",
        channel_id="-10099",
        message_id="42",
        actor="human-1",
    )

    captured: list[str] = []
    handler_id = logger.add(captured.append, level="TRACE", format="{message}")
    try:
        asyncio.run(sink(_callback(data)))
    finally:
        logger.remove(handler_id)

    joined = "".join(captured)
    # The raw exception text — and the sentinel URL inside it — must never reach a
    # log line; only the exception type name plus safe task context may.
    assert _SECRET_SENTINEL not in joined
    assert "nats://" not in joined
    assert "RuntimeError" in joined
    # Containment still holds: the failed publish rolled back, so the human is
    # acknowledged and nothing landed on the bus.
    assert bus.published == []
    assert sender.answers == [("cq-1", "-10099")]
    assert sender.markup_edits == []

    # The rollback left the capability live: a healthy retry still lands the steer.
    healthy = FakeBus()
    retry_sink = TelegramCallbackSink(bus=healthy, sender=sender, registry=registry)  # type: ignore[arg-type]
    asyncio.run(retry_sink(_callback(data)))
    assert [subject for subject, _ in healthy.published] == [subjects.task_steer("task-123")]


# --- outbound interaction: attach an actor-bound steer keyboard (issue #117) ---


class InteractionSender(FakeSender):
    """A FakeSender that also captures debate-TLDR ack sends and attached markup."""

    def __init__(self, *, ack_message_id: str | None = "500") -> None:
        super().__init__()
        self._ack_message_id = ack_message_id
        # (channel_id, text, reply_to) of every ack send the interaction posts.
        self.sends: list[tuple[str, str, str | None]] = []
        # The InlineKeyboardMarkup attached by a successful edit_reply_markup, and
        # whether the attach succeeded (the interaction's attach vs the sink's clear
        # both route through edit_reply_markup; the attach carries a non-None markup).
        self.attached: list[InlineKeyboardMarkup] = []
        self.attach_ok = True

    async def send(self, target: SendTarget, text: str, files: list[Path]) -> str | None:
        self.sends.append((target.channel_id, text, target.reply_to))
        return self._ack_message_id

    async def edit_reply_markup(
        self,
        target: SendTarget,
        message_id: str,
        *,
        reply_markup: InlineKeyboardMarkup | None = None,
    ) -> bool:
        if reply_markup is not None:
            self.attached.append(reply_markup)
            if not self.attach_ok:
                return False
        else:
            self.markup_edits.append((target.channel_id, message_id, target.as_agent))
        return True


def _interaction(
    clock: Clock, *, sender: InteractionSender
) -> tuple[TelegramTurnInteraction, TelegramCallbackSink, FakeBus]:
    bus = FakeBus()
    registry = TelegramCallbackRegistry(secret=b"test-secret", clock=clock)
    sink = TelegramCallbackSink(bus=bus, sender=sender, registry=registry)  # type: ignore[arg-type]
    interaction = TelegramTurnInteraction(sink=sink, sender=sender)  # type: ignore[arg-type]
    return interaction, sink, bus


def test_interaction_mints_actor_bound_data_after_the_ack_id_is_known() -> None:
    clock = Clock()
    sender = InteractionSender(ack_message_id="500")
    interaction, sink, bus = _interaction(clock, sender=sender)

    asyncio.run(
        interaction.offer_debate_tldr(
            target=SendTarget(channel_id="-10099"),
            anchor_message_id="42",
            task_id="task-123",
            agent_id="agentd-bilby",
            asker="human-1",
        )
    )

    # The ack is posted UNDER the polled default bot (unstamped) as a reply to the
    # agent's TLDR anchor (42), and its message id (500) — not the anchor — is what
    # the keyboard's callback data binds to.
    assert sender.sends == [("-10099", "Steer this direction:", "42")]
    assert len(sender.attached) == 1
    buttons = [b for row in sender.attached[0].inline_keyboard for b in row]
    assert [b.text for b in buttons] == ["✅ Approve", "✏️ Revise", "🛑 Stop"]

    # The capability is bound to the ACK message id (500), NOT the anchor (42): a
    # tap on the anchor id fails closed, a tap on the ack id authorizes.
    approve_data = buttons[0].callback_data
    asyncio.run(
        sink(
            InboundCallback(
                callback_query_id="cq-anchor",
                channel_id="-10099",
                message_id="42",
                author="human-1",
                data=approve_data,
            )
        )
    )
    assert bus.published == []  # bound to 500, not the anchor 42

    asyncio.run(
        sink(
            InboundCallback(
                callback_query_id="cq-ack",
                channel_id="-10099",
                message_id="500",
                author="human-1",
                data=approve_data,
            )
        )
    )
    assert [subject for subject, _ in bus.published] == [subjects.task_steer("task-123")]


def test_interaction_round_trip_original_requester_steers_then_markup_clears() -> None:
    clock = Clock()
    sender = InteractionSender(ack_message_id="500")
    interaction, sink, bus = _interaction(clock, sender=sender)

    asyncio.run(
        interaction.offer_debate_tldr(
            target=SendTarget(channel_id="-10099"),
            anchor_message_id="42",
            task_id="task-123",
            agent_id="agentd-nagatha",
            asker="human-1",
        )
    )
    revise_data = [b for row in sender.attached[0].inline_keyboard for b in row][1].callback_data

    # The ORIGINAL requester taps Revise: a typed SteerFrame lands on the bus and
    # the spent keyboard is cleared on the ack message under the default bot.
    asyncio.run(
        sink(
            InboundCallback(
                callback_query_id="cq-3",
                channel_id="-10099",
                message_id="500",
                author="human-1",
                data=revise_data,
            )
        )
    )
    assert [subject for subject, _ in bus.published] == [subjects.task_steer("task-123")]
    frame = decode_frame(bus.published[0][1])
    assert isinstance(frame, SteerFrame)
    assert frame.task_id == "task-123"
    assert frame.target_agent == "agentd-nagatha"
    assert frame.interrupt is True
    assert sender.markup_edits == [("-10099", "500", None)]


def test_interaction_intruder_and_wrong_message_fail_closed() -> None:
    clock = Clock()
    sender = InteractionSender(ack_message_id="500")
    interaction, sink, bus = _interaction(clock, sender=sender)
    asyncio.run(
        interaction.offer_debate_tldr(
            target=SendTarget(channel_id="-10099"),
            anchor_message_id="42",
            task_id="task-123",
            agent_id="agentd-bilby",
            asker="human-1",
        )
    )
    approve_data = [b for row in sender.attached[0].inline_keyboard for b in row][0].callback_data

    # A different chat member taps the requester's button: fails closed.
    asyncio.run(
        sink(
            InboundCallback(
                callback_query_id="cq-x",
                channel_id="-10099",
                message_id="500",
                author="intruder-9",
                data=approve_data,
            )
        )
    )
    # The same actor but a different message id: fails closed (message binding).
    asyncio.run(
        sink(
            InboundCallback(
                callback_query_id="cq-y",
                channel_id="-10099",
                message_id="999",
                author="human-1",
                data=approve_data,
            )
        )
    )
    assert bus.published == []
    assert sender.markup_edits == []


def test_interaction_declines_when_origin_has_no_asker() -> None:
    clock = Clock()
    sender = InteractionSender(ack_message_id="500")
    interaction, _sink, _bus = _interaction(clock, sender=sender)

    asyncio.run(
        interaction.offer_debate_tldr(
            target=SendTarget(channel_id="-10099"),
            anchor_message_id="42",
            task_id="task-123",
            agent_id="agentd-bilby",
            asker=None,
        )
    )
    # No asker to bind to: no ack posted, no capability minted, no keyboard.
    assert sender.sends == []
    assert sender.attached == []


def test_interaction_leaves_plain_ack_when_keyboard_attach_fails() -> None:
    clock = Clock()
    sender = InteractionSender(ack_message_id="500")
    sender.attach_ok = False
    interaction, _sink, _bus = _interaction(clock, sender=sender)

    asyncio.run(
        interaction.offer_debate_tldr(
            target=SendTarget(channel_id="-10099"),
            anchor_message_id="42",
            task_id="task-123",
            agent_id="agentd-bilby",
            asker="human-1",
        )
    )
    # The ack landed but the keyboard attach failed: a plain ack is left, never a
    # half-wired dead button.
    assert sender.sends == [("-10099", "Steer this direction:", "42")]
    assert len(sender.attached) == 1  # attempted once, returned False


def test_interaction_skips_keyboard_when_ack_has_no_message_id() -> None:
    clock = Clock()
    sender = InteractionSender(ack_message_id=None)
    interaction, _sink, _bus = _interaction(clock, sender=sender)

    asyncio.run(
        interaction.offer_debate_tldr(
            target=SendTarget(channel_id="-10099"),
            anchor_message_id="42",
            task_id="task-123",
            agent_id="agentd-bilby",
            asker="human-1",
        )
    )
    # No id to bind to: the ack was attempted but no keyboard is minted/attached.
    assert sender.sends == [("-10099", "Steer this direction:", "42")]
    assert sender.attached == []
