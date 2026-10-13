"""Wire tests for DiscordEdge using a fake adapter + fake bus (no discord.py)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from loguru import logger
from rtech_chatd import subjects
from rtech_chatd.channel import ChunkProgress, InboundMessage, SendTarget, attach_chunk_progress
from rtech_chatd.edge import DiscordEdge
from rtech_frames.codec import decode_frame, encode_frame
from rtech_frames.frames import (
    ChannelOrigin,
    EvidenceSource,
    Stance,
    StatusFrame,
    TaskFrame,
    TurnFrame,
    VerificationReport,
)
from rtech_frames.subs import FrameSubscribeError

FrameHandler = Callable[[bytes], Awaitable[None]]


@dataclass
class FakeSubscription:
    """Record unsubscribe calls."""

    unsubscribed: bool = False

    async def unsubscribe(self) -> None:
        self.unsubscribed = True


@dataclass
class FakeFrameBus:
    """Record publishes and let the test invoke subscribed handlers."""

    published: list[tuple[str, bytes]] = field(default_factory=list)
    handlers: dict[str, FrameHandler] = field(default_factory=dict)
    subs_by_subject: dict[str, FakeSubscription] = field(default_factory=dict)
    fail_subjects: set[str] = field(default_factory=set)

    async def publish(self, subject: str, data: bytes) -> None:
        self.published.append((subject, data))

    async def subscribe(self, subject: str, handler: FrameHandler) -> FakeSubscription:
        if subject in self.fail_subjects:
            raise RuntimeError(f"subscribe boom for {subject}")
        sub = FakeSubscription()
        self.handlers[subject] = handler
        self.subs_by_subject[subject] = sub
        return sub

    async def emit(self, subject: str, data: bytes) -> None:
        """Deliver bytes to a captured handler."""
        await self.handlers[subject](data)

    def subscriptions_for(self, task_id: str) -> list[FakeSubscription]:
        return [sub for subject, sub in self.subs_by_subject.items() if f".{task_id}." in subject]


@dataclass
class FakeAdapter:
    """Yield scripted inbound messages and record send() + create_thread() calls."""

    inbound: list[InboundMessage] = field(default_factory=list)
    sent: list[tuple[SendTarget, str]] = field(default_factory=list)
    # Thread ids handed back per create_thread call; None entries model a
    # create failure (the edge degrades full text to the parent channel).
    thread_ids: list[str | None] = field(default_factory=list)
    thread_calls: list[tuple[str, str, str]] = field(default_factory=list)
    typing_calls: list[tuple[SendTarget, bool]] = field(default_factory=list)
    _next_thread: int = 0

    async def events(self) -> AsyncIterator[InboundMessage]:
        for msg in self.inbound:
            yield msg

    async def send(self, target: SendTarget, text: str, files: list[Path]) -> str | None:
        self.sent.append((target, text))
        # Deterministic per-send message id (the ChannelAdapter thread-anchor
        # contract): "sent-<1-based index into self.sent>".
        return f"sent-{len(self.sent)}"

    async def typing(self, target: SendTarget, on: bool) -> None:
        self.typing_calls.append((target, on))
        return None

    async def create_thread(self, channel_id: str, message_id: str, name: str) -> str | None:
        self.thread_calls.append((channel_id, message_id, name))
        # Default: a deterministic thread id anchored to the task; tests that need
        # a create failure seed ``thread_ids`` with a None.
        if self._next_thread < len(self.thread_ids):
            tid = self.thread_ids[self._next_thread]
            self._next_thread += 1
            return tid
        return f"thread-for-{message_id}"


def _turn_bytes(task_id: str, visible_text: str) -> bytes:
    return encode_frame(
        TurnFrame(
            agent_id="agent-a",
            task_id=task_id,
            visible_text=visible_text,
            stance=Stance(position="working", because="The task is active."),
        )
    )


def _status_bytes(task_id: str, state: str) -> bytes:
    frame = StatusFrame(task_id=task_id, agent_id="agent-a", state=state)  # type: ignore[arg-type]
    return encode_frame(frame)


async def test_inbound_publishes_one_task_frame() -> None:
    msg = InboundMessage(
        channel_id="chan-1",
        thread_id="thread-9",
        author="user-42",
        text="Build the edge.",
    )
    adapter = FakeAdapter(inbound=[msg])
    bus = FakeFrameBus()
    edge = DiscordEdge(adapter, bus, stack="dev", id_factory=lambda: "task-1")

    await edge.run_inbound()

    ingress = [p for p in bus.published if p[0] == subjects.TASK_INGRESS]
    assert len(ingress) == 1
    frame = decode_frame(ingress[0][1])
    assert isinstance(frame, TaskFrame)
    assert frame.created_at is not None
    assert frame.model_copy(update={"created_at": None}) == TaskFrame(
        id="task-1",
        origin_edge="discord",
        requester="user-42",
        text="Build the edge.",
        scope="single",
        stack="dev",
    )


async def test_turn_renders_to_original_thread_target() -> None:
    msg = InboundMessage(
        channel_id="chan-1",
        thread_id="thread-9",
        author="user-42",
        text="hello",
    )
    adapter = FakeAdapter(inbound=[msg])
    bus = FakeFrameBus()
    edge = DiscordEdge(adapter, bus, stack="dev", id_factory=lambda: "task-1")
    await edge.run_inbound()

    await bus.emit(subjects.task_turn("task-1"), _turn_bytes("task-1", "Visible answer."))

    assert len(adapter.sent) == 1
    target, text = adapter.sent[0]
    assert target.resolve() == "thread-9"  # thread_id wins over channel_id
    assert text == "Visible answer."


async def test_turn_goes_through_guard_whole_text() -> None:
    msg = InboundMessage(
        channel_id="chan-1",
        thread_id=None,
        author="user-42",
        text="hello",
    )
    adapter = FakeAdapter(inbound=[msg])
    bus = FakeFrameBus()
    edge = DiscordEdge(adapter, bus, stack="dev", id_factory=lambda: "task-1")
    await edge.run_inbound()

    leak = "leak sk-ABCDEFGHIJKLMNOP1234567890"
    await bus.emit(subjects.task_turn("task-1"), _turn_bytes("task-1", leak))

    assert len(adapter.sent) == 1
    _, text = adapter.sent[0]
    assert "sk-ABCDEFGHIJKLMNOP1234567890" not in text
    assert "[redacted]" in text


async def test_status_working_renders_indicator() -> None:
    msg = InboundMessage(
        channel_id="chan-1",
        thread_id="thread-9",
        author="user-42",
        text="hello",
    )
    adapter = FakeAdapter(inbound=[msg])
    bus = FakeFrameBus()
    edge = DiscordEdge(adapter, bus, stack="dev", id_factory=lambda: "task-1")
    await edge.run_inbound()

    await bus.emit(subjects.task_status("task-1"), _status_bytes("task-1", "working"))

    assert adapter.sent == [(SendTarget(channel_id="chan-1", thread_id="thread-9"), "Working...")]


async def test_inbound_never_posts_home_advisory() -> None:
    # Regression: an ordinary inbound + turn render produces exactly ONE send
    # (the turn), never an extra home-advisory post — even with no home channel.
    from rtech_chatd.config import HOME_ADVISORY

    msg = InboundMessage(channel_id="chan-1", thread_id="thread-9", author="user-42", text="hello")
    adapter = FakeAdapter(inbound=[msg])
    bus = FakeFrameBus()
    edge = DiscordEdge(adapter, bus, stack="dev", id_factory=lambda: "task-1", home_target=None)
    await edge.run_inbound()
    await bus.emit(subjects.task_turn("task-1"), _turn_bytes("task-1", "Answer."))

    assert len(adapter.sent) == 1
    assert all(HOME_ADVISORY not in text for _, text in adapter.sent)


async def test_edge_rejects_blank_stack_at_construction() -> None:
    """A blank stack id fails closed at startup (issue #75 round 2)."""
    adapter = FakeAdapter([])
    bus = FakeFrameBus()
    with pytest.raises(ValueError, match="stack must be non-empty"):
        DiscordEdge(adapter, bus, stack="  ", id_factory=lambda: "task-1")


def _turn_bytes_for(task_id: str, agent_id: str, visible_text: str) -> bytes:
    return encode_frame(
        TurnFrame(
            agent_id=agent_id,
            task_id=task_id,
            visible_text=visible_text,
            stance=Stance(position="working", because="The task is active."),
        )
    )


async def test_participants_mint_trio_task_frame() -> None:
    msg = InboundMessage(
        channel_id="chan-1",
        thread_id="thread-9",
        author="user-42",
        text="Debate this.",
    )
    adapter = FakeAdapter(inbound=[msg])
    bus = FakeFrameBus()
    edge = DiscordEdge(
        adapter,
        bus,
        stack="dev",
        id_factory=lambda: "task-1",
        participants=["agent-a", "agent-b", "agent-c"],
    )

    await edge.run_inbound()

    ingress = [p for p in bus.published if p[0] == subjects.TASK_INGRESS]
    assert len(ingress) == 1
    frame = decode_frame(ingress[0][1])
    assert isinstance(frame, TaskFrame)
    assert frame.scope == "trio"
    assert frame.participants == ["agent-a", "agent-b", "agent-c"]


async def test_attributed_renders_three_agents_as_prefixed_sends() -> None:
    msg = InboundMessage(
        channel_id="chan-1",
        thread_id="thread-9",
        author="user-42",
        text="hello",
    )
    adapter = FakeAdapter(inbound=[msg])
    bus = FakeFrameBus()
    edge = DiscordEdge(
        adapter,
        bus,
        stack="dev",
        id_factory=lambda: "task-1",
        participants=["agent-a", "agent-b", "agent-c"],
        attributed=True,
    )
    await edge.run_inbound()

    await bus.emit(subjects.task_turn("task-1"), _turn_bytes_for("task-1", "agent-a", "Alpha."))
    await bus.emit(subjects.task_turn("task-1"), _turn_bytes_for("task-1", "agent-b", "Bravo."))
    await bus.emit(subjects.task_turn("task-1"), _turn_bytes_for("task-1", "agent-c", "Charlie."))

    texts = [text for _, text in adapter.sent]
    assert texts == [
        "**agent-a**: Alpha.",
        "**agent-b**: Bravo.",
        "**agent-c**: Charlie.",
    ]


async def test_result_lane_renders_done_marker_once_then_drops() -> None:
    msg = InboundMessage(
        channel_id="chan-1",
        thread_id="thread-9",
        author="user-42",
        text="hello",
    )
    adapter = FakeAdapter(inbound=[msg])
    bus = FakeFrameBus()
    edge = DiscordEdge(
        adapter,
        bus,
        stack="dev",
        id_factory=lambda: "task-1",
        participants=["agent-a", "agent-b", "agent-c"],
        attributed=True,
    )
    await edge.run_inbound()

    result_subject = subjects.task_result("task-1")
    terminal = _turn_bytes_for("task-1", "agent-a", "Final answer.")
    # No prior turn render: the result lane renders the terminal turn FIRST,
    # then the done marker (two sends).
    await bus.emit(result_subject, terminal)

    assert [text for _, text in adapter.sent] == [
        "**agent-a**: Final answer.",
        "✅ done — agent-a",
    ]

    # Second emit renders nothing more (once-guard).
    await bus.emit(result_subject, terminal)
    assert len(adapter.sent) == 2

    # The task's per-task subscriptions were unsubscribed and the target dropped.
    assert all(sub.unsubscribed for sub in bus.subscriptions_for("task-1"))
    assert "task-1" not in edge._targets


async def test_result_lane_done_marker_unattributed() -> None:
    msg = InboundMessage(channel_id="chan-1", thread_id=None, author="user-42", text="hello")
    adapter = FakeAdapter(inbound=[msg])
    bus = FakeFrameBus()
    edge = DiscordEdge(adapter, bus, stack="dev", id_factory=lambda: "task-1")
    await edge.run_inbound()

    await bus.emit(subjects.task_result("task-1"), _turn_bytes_for("task-1", "agent-a", "done."))

    # No prior turn render: the terminal turn renders first, then the marker.
    assert [text for _, text in adapter.sent] == ["done.", "✅ done"]


async def test_attributed_status_renders_with_agent_prefix() -> None:
    msg = InboundMessage(
        channel_id="chan-1",
        thread_id="thread-9",
        author="user-42",
        text="hello",
    )
    adapter = FakeAdapter(inbound=[msg])
    bus = FakeFrameBus()
    edge = DiscordEdge(
        adapter,
        bus,
        stack="dev",
        id_factory=lambda: "task-1",
        participants=["agent-a", "agent-b", "agent-c"],
        attributed=True,
    )
    await edge.run_inbound()

    await bus.emit(subjects.task_status("task-1"), _status_bytes("task-1", "working"))

    assert adapter.sent == [
        (SendTarget(channel_id="chan-1", thread_id="thread-9"), "**agent-a**: Working..."),
    ]


def test_invalid_participants_fail_at_construction() -> None:
    adapter = FakeAdapter([])
    bus = FakeFrameBus()
    # Four ids map to no scope.
    with pytest.raises(ValueError):
        DiscordEdge(adapter, bus, stack="dev", participants=["a", "b", "c", "d"])
    # Duplicate ids are rejected.
    with pytest.raises(ValueError):
        DiscordEdge(adapter, bus, stack="dev", participants=["a", "a"])


@dataclass
class FastTurnFrameBus(FakeFrameBus):
    """On the ingress publish, immediately deliver a turn via the live handler.

    This proves the reply subscriptions are installed BEFORE the publish: the
    turn handler must already exist at publish time for this to render.
    """

    fast_turn: bytes = b""
    handler_existed_at_publish: bool = False

    async def publish(self, subject: str, data: bytes) -> None:
        self.published.append((subject, data))
        if subject == subjects.TASK_INGRESS:
            handler = self.handlers.get(subjects.task_turn("task-1"))
            self.handler_existed_at_publish = handler is not None
            if handler is not None:
                await handler(self.fast_turn)


async def test_replies_subscribed_before_publish() -> None:
    msg = InboundMessage(channel_id="chan-1", thread_id="thread-9", author="user-42", text="hi")
    adapter = FakeAdapter(inbound=[msg])
    bus = FastTurnFrameBus(fast_turn=_turn_bytes("task-1", "Fast turn."))
    edge = DiscordEdge(adapter, bus, stack="dev", id_factory=lambda: "task-1")

    await edge.run_inbound()

    assert bus.handler_existed_at_publish is True
    assert [text for _, text in adapter.sent] == ["Fast turn."]


@dataclass
class FailingPublishFrameBus(FakeFrameBus):
    """Raise on publish so the edge must roll back its subscriptions."""

    async def publish(self, subject: str, data: bytes) -> None:
        raise RuntimeError("publish boom")


async def test_failed_publish_rolls_back_subscriptions() -> None:
    msg = InboundMessage(channel_id="chan-1", thread_id="thread-9", author="user-42", text="hi")
    adapter = FakeAdapter(inbound=[msg])
    bus = FailingPublishFrameBus()
    edge = DiscordEdge(adapter, bus, stack="dev", id_factory=lambda: "task-1")

    with pytest.raises(RuntimeError, match="publish boom"):
        await edge.run_inbound()

    assert all(sub.unsubscribed for sub in bus.subscriptions_for("task-1"))
    assert "task-1" not in edge._targets


# The sentinel a raising bus.publish would smuggle into an ingest log line if the
# edge logged the raw exception text instead of only its type name. It carries a
# credentialed NATS URL — exactly the class of secret the edge must never render.
_INGEST_SECRET_SENTINEL = "SUPER_SECRET_NATS_URL"


@dataclass
class LeakingPublishFrameBus(FakeFrameBus):
    """Raise a sentinel-credentialed exception on publish (post-subscribe leak)."""

    async def publish(self, subject: str, data: bytes) -> None:
        raise RuntimeError(
            f"publish failed talking to nats://user:token@{_INGEST_SECRET_SENTINEL}:4222"
        )


async def test_ingest_publish_failure_never_logs_raw_exception_text() -> None:
    # The verified P1 end-to-end leak: bus.publish re-raises an exception whose
    # message carries a credentialed URL, and _ingest's setup-failure log used to
    # render that raw text (reason={} with exc). Inject the sentinel through the
    # NORMAL _ingest publish-failure path and assert it never reaches any rendered
    # log line, while the same exception still propagates and the task is rolled
    # back (subscriptions retired, target/lock/set dropped).
    msg = InboundMessage(channel_id="chan-1", thread_id="thread-9", author="user-42", text="hi")
    adapter = FakeAdapter(inbound=[msg])
    bus = LeakingPublishFrameBus()
    edge = DiscordEdge(adapter, bus, stack="dev", id_factory=lambda: "task-1")

    captured: list[str] = []
    handler_id = logger.add(captured.append, level="TRACE", format="{message}")
    try:
        # The same exception propagates out of run_inbound (cleanup runs on it).
        with pytest.raises(RuntimeError):
            await edge.run_inbound()
    finally:
        logger.remove(handler_id)

    joined = "".join(captured)
    # The raw exception text — and the credentialed URL inside it — must never
    # reach a log line; only the exception type name plus safe task context may.
    assert _INGEST_SECRET_SENTINEL not in joined
    assert "nats://" not in joined
    assert "token" not in joined
    assert "RuntimeError" in joined
    assert "task-1" in joined
    # Cleanup still ran on the propagating exception: subscriptions retired, no
    # target/lock/tracked set survives.
    assert all(sub.unsubscribed for sub in bus.subscriptions_for("task-1"))
    assert "task-1" not in edge._targets
    assert "task-1" not in edge._task_locks
    assert edge._task_subscriptions == {}


async def test_result_after_turn_renders_only_marker() -> None:
    msg = InboundMessage(channel_id="chan-1", thread_id="thread-9", author="user-42", text="hi")
    adapter = FakeAdapter(inbound=[msg])
    bus = FakeFrameBus()
    edge = DiscordEdge(
        adapter,
        bus,
        stack="dev",
        id_factory=lambda: "task-1",
        participants=["agent-a", "agent-b", "agent-c"],
        attributed=True,
    )
    await edge.run_inbound()

    terminal = _turn_bytes_for("task-1", "agent-a", "Final answer.")
    # Terminal turn arrives on the TURN lane first, then republished on RESULT.
    await bus.emit(subjects.task_turn("task-1"), terminal)
    await bus.emit(subjects.task_result("task-1"), terminal)

    # Exactly two sends total: the turn (once) and the marker (no duplicate turn).
    assert [text for _, text in adapter.sent] == [
        "**agent-a**: Final answer.",
        "✅ done — agent-a",
    ]


@dataclass
class FailOnceAdapter(FakeAdapter):
    """Raise on the first send() call, then succeed on subsequent calls."""

    calls: int = 0

    async def send(self, target: SendTarget, text: str, files: list[Path]) -> str | None:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("send boom")
        self.sent.append((target, text))
        return f"sent-{len(self.sent)}"


async def test_marker_send_failure_keeps_task_live_for_retry() -> None:
    msg = InboundMessage(channel_id="chan-1", thread_id=None, author="user-42", text="hi")
    adapter = FailOnceAdapter(inbound=[msg])
    bus = FakeFrameBus()
    edge = DiscordEdge(adapter, bus, stack="dev", id_factory=lambda: "task-1")
    await edge.run_inbound()

    terminal = _turn_bytes_for("task-1", "agent-a", "Final answer.")
    result_subject = subjects.task_result("task-1")

    # First emit: the terminal-turn render is the first dispatch and it raises.
    # Nothing was recorded, so the task stays live for retry.
    with pytest.raises(RuntimeError, match="send boom"):
        await bus.emit(result_subject, terminal)

    assert adapter.sent == []
    assert "task-1" not in edge._done_tasks
    assert "task-1" in edge._targets
    assert not any(sub.unsubscribed for sub in bus.subscriptions_for("task-1"))

    # Second emit: retry renders the terminal turn then the marker, then drops.
    await bus.emit(result_subject, terminal)

    assert [text for _, text in adapter.sent] == ["Final answer.", "✅ done"]
    assert "task-1" in edge._done_tasks
    assert "task-1" not in edge._targets
    assert all(sub.unsubscribed for sub in bus.subscriptions_for("task-1"))


async def test_partial_subscribe_rollback_status_fails() -> None:
    # STATUS is the SECOND of three subscriptions. When it fails, subscribe_all
    # rolls the whole set back: the TURN subscription that succeeded is
    # unsubscribed, the error propagates out of run_inbound, and nothing leaks.
    msg = InboundMessage(channel_id="chan-1", thread_id="thread-9", author="user-42", text="hi")
    adapter = FakeAdapter(inbound=[msg])
    bus = FakeFrameBus(fail_subjects={subjects.task_status("task-1")})
    edge = DiscordEdge(adapter, bus, stack="dev", id_factory=lambda: "task-1")

    with pytest.raises(FrameSubscribeError):
        await edge.run_inbound()

    # The turn subscription that succeeded before status failed was rolled back.
    turn_sub = bus.subs_by_subject[subjects.task_turn("task-1")]
    assert turn_sub.unsubscribed is True
    # No ingress publish happened (subscribe failed before publish).
    assert [p for p in bus.published if p[0] == subjects.TASK_INGRESS] == []
    # No leak: neither the target nor a tracked subscription set survives.
    assert "task-1" not in edge._targets
    assert edge._task_subscriptions == {}
    assert "task-1" not in edge._task_locks


async def test_partial_subscribe_rollback_result_fails() -> None:
    # RESULT is the THIRD subscription. Its failure rolls back both TURN and
    # STATUS (the two that already succeeded).
    msg = InboundMessage(channel_id="chan-1", thread_id="thread-9", author="user-42", text="hi")
    adapter = FakeAdapter(inbound=[msg])
    bus = FakeFrameBus(fail_subjects={subjects.task_result("task-1")})
    edge = DiscordEdge(adapter, bus, stack="dev", id_factory=lambda: "task-1")

    with pytest.raises(FrameSubscribeError):
        await edge.run_inbound()

    assert bus.subs_by_subject[subjects.task_turn("task-1")].unsubscribed is True
    assert bus.subs_by_subject[subjects.task_status("task-1")].unsubscribed is True
    assert "task-1" not in edge._targets
    assert edge._task_subscriptions == {}


@dataclass
class BlockingSendAdapter(FakeAdapter):
    # Block on an Event on the FIRST send() call, then send normally.

    gate: asyncio.Event = field(default_factory=asyncio.Event)
    entered_first: asyncio.Event = field(default_factory=asyncio.Event)
    calls: int = 0

    async def send(self, target: SendTarget, text: str, files: list[Path]) -> str | None:
        self.calls += 1
        if self.calls == 1:
            self.entered_first.set()
            await self.gate.wait()
        self.sent.append((target, text))
        return f"sent-{len(self.sent)}"


async def _ingest_one(edge: DiscordEdge) -> None:
    # Drive a single ingest through the edge machinery (installs the lock).
    msg = InboundMessage(channel_id="chan-1", thread_id=None, author="user-42", text="hi")
    adapter_msgs = FakeAdapter(inbound=[msg])
    async for m in adapter_msgs.events():
        await edge._ingest(m)


async def test_turn_inflight_result_concurrent_single_terminal_and_marker() -> None:
    # Terminal turn delivery blocks inside dispatch (first send). Concurrently
    # the result frame (same terminal turn republished) is delivered. When the
    # gate opens, the terminal text is sent EXACTLY ONCE, the done marker once
    # AFTER it, and no send after the drop.
    adapter = BlockingSendAdapter()
    bus = FakeFrameBus()
    edge = DiscordEdge(adapter, bus, stack="dev", id_factory=lambda: "task-1")
    await _ingest_one(edge)

    terminal = _turn_bytes_for("task-1", "agent-a", "Final answer.")

    turn_task = asyncio.ensure_future(bus.emit(subjects.task_turn("task-1"), terminal))
    await adapter.entered_first.wait()
    result_task = asyncio.ensure_future(bus.emit(subjects.task_result("task-1"), terminal))
    await asyncio.sleep(0)  # let result_task reach the lock and park
    assert not result_task.done()

    adapter.gate.set()
    await asyncio.gather(turn_task, result_task)

    texts = [text for _, text in adapter.sent]
    assert texts == ["Final answer.", "✅ done"]
    assert "task-1" not in edge._targets
    assert all(sub.unsubscribed for sub in bus.subscriptions_for("task-1"))


async def test_result_inflight_turn_concurrent_no_duplicate() -> None:
    # Reverse interleaving: the result lane renders the terminal first and blocks
    # in dispatch; the turn lane is delivered concurrently. No duplicate terminal
    # send, marker once, and the concurrent turn renders nothing after the drop.
    adapter = BlockingSendAdapter()
    bus = FakeFrameBus()
    edge = DiscordEdge(adapter, bus, stack="dev", id_factory=lambda: "task-1")
    await _ingest_one(edge)

    terminal = _turn_bytes_for("task-1", "agent-a", "Final answer.")

    result_task = asyncio.ensure_future(bus.emit(subjects.task_result("task-1"), terminal))
    await adapter.entered_first.wait()
    turn_task = asyncio.ensure_future(bus.emit(subjects.task_turn("task-1"), terminal))
    await asyncio.sleep(0)
    assert not turn_task.done()

    adapter.gate.set()
    await asyncio.gather(result_task, turn_task)

    texts = [text for _, text in adapter.sent]
    assert texts == ["Final answer.", "✅ done"]
    assert "task-1" not in edge._targets


async def test_late_turn_after_done_is_noop() -> None:
    # After a completed result flow (task dropped, lock popped), a late turn
    # frame delivered to the still-captured handler renders nothing / no crash.
    msg = InboundMessage(channel_id="chan-1", thread_id=None, author="user-42", text="hi")
    adapter = FakeAdapter(inbound=[msg])
    bus = FakeFrameBus()
    edge = DiscordEdge(adapter, bus, stack="dev", id_factory=lambda: "task-1")
    await edge.run_inbound()

    terminal = _turn_bytes_for("task-1", "agent-a", "Final answer.")
    await bus.emit(subjects.task_result("task-1"), terminal)
    assert "task-1" not in edge._targets

    before = list(adapter.sent)
    await bus.emit(subjects.task_turn("task-1"), _turn_bytes("task-1", "Late turn."))
    assert adapter.sent == before


@dataclass
class TransientUnsubSubscription:
    """A subscription whose FIRST unsubscribe fails, then succeeds on retry."""

    calls: int = 0
    unsubscribed: bool = False

    async def unsubscribe(self) -> None:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("unsubscribe boom (transient)")
        self.unsubscribed = True


@dataclass
class LeakyRollbackFrameBus(FakeFrameBus):
    """Fail a later subscribe AND make an earlier sub's unsubscribe fail.

    The first subscribe succeeds but returns a subscription that raises on
    unsubscribe permanently, so rollback cannot retire it -- it must ride out on
    FrameSubscribeError.leaked. ``fail_after`` counts successful subscribes
    before the next one raises.
    """

    fail_after: int = 1
    _ok: int = 0

    async def subscribe(self, subject: str, handler: FrameHandler) -> FakeSubscription:
        if self._ok >= self.fail_after:
            raise RuntimeError(f"subscribe boom for {subject}")
        self._ok += 1

        class _PermFailSub:
            unsubscribed = False

            async def unsubscribe(self_inner) -> None:
                raise RuntimeError("rollback unsubscribe boom")

        sub = _PermFailSub()  # type: ignore[assignment]
        self.handlers[subject] = handler
        self.subs_by_subject[subject] = sub  # type: ignore[assignment]
        return sub  # type: ignore[return-value]


async def test_subscribe_rollback_leak_held_for_escalation() -> None:
    # Second subscribe fails; the first sub's rollback-unsubscribe fails too, so
    # rollback leaves a live subscription. The task is refused (target/lock gone)
    # and the leaked set is held in _leaked_subscriptions for retry-at-close.
    msg = InboundMessage(channel_id="chan-1", thread_id="thread-9", author="user-42", text="hi")
    adapter = FakeAdapter(inbound=[msg])
    bus = LeakyRollbackFrameBus(fail_after=1)
    edge = DiscordEdge(adapter, bus, stack="dev", id_factory=lambda: "task-1")

    with pytest.raises(FrameSubscribeError):
        await edge.run_inbound()

    # Task refused: no ingress publish, no target, no lock, no tracked set.
    assert [p for p in bus.published if p[0] == subjects.TASK_INGRESS] == []
    assert "task-1" not in edge._targets
    assert "task-1" not in edge._task_locks
    assert edge._task_subscriptions == {}
    # The live-but-unretireable subscription is held for escalation, not lost.
    assert "task-1" in edge._leaked_subscriptions
    assert len(edge._leaked_subscriptions["task-1"]) == 1


@dataclass
class TransientUnsubFrameBus(FakeFrameBus):
    """A bus whose subscriptions fail their FIRST unsubscribe, then succeed."""

    def subscriptions_for(self, task_id: str) -> list[TransientUnsubSubscription]:  # type: ignore[override]
        return [
            sub  # type: ignore[misc]
            for subject, sub in self.subs_by_subject.items()
            if f".{task_id}." in subject
        ]

    async def subscribe(self, subject: str, handler: FrameHandler) -> FakeSubscription:
        if subject in self.fail_subjects:
            raise RuntimeError(f"subscribe boom for {subject}")
        sub = TransientUnsubSubscription()
        self.handlers[subject] = handler
        self.subs_by_subject[subject] = sub  # type: ignore[assignment]
        return sub  # type: ignore[return-value]


async def test_drop_task_retire_failure_holds_leaked_set_and_close_retries() -> None:
    # Drive a task to result-done where each unsubscribe fails transiently on
    # the first attempt. _drop_task's retire returns False, so the set lands in
    # _leaked_subscriptions; close() retries it (second unsubscribe succeeds)
    # and clears it.
    msg = InboundMessage(channel_id="chan-1", thread_id=None, author="user-42", text="hi")
    adapter = FakeAdapter(inbound=[msg])
    bus = TransientUnsubFrameBus()
    edge = DiscordEdge(adapter, bus, stack="dev", id_factory=lambda: "task-1")
    await edge.run_inbound()

    terminal = _turn_bytes_for("task-1", "agent-a", "Final answer.")
    await bus.emit(subjects.task_result("task-1"), terminal)

    # The done flow ran (marker rendered) but retire failed on first sweep, so
    # the set is held for retry rather than discarded.
    assert "task-1" in edge._leaked_subscriptions
    assert "task-1" not in edge._task_subscriptions

    await edge.close()

    # close() retried the leaked set; its unsubscribes now succeeded and it is
    # cleared.
    assert edge._leaked_subscriptions == {}
    assert all(sub.unsubscribed for sub in bus.subscriptions_for("task-1"))


# --- External-turn reply render (issue #93) -------------------------------

TURN_WILDCARD = subjects.TASK_TURN_WILDCARD


def _external_turn_bytes(
    task_id: str,
    visible_text: str,
    *,
    origin: ChannelOrigin | None,
    needs_from: list[str] | None = None,
    agent_id: str = "nagatha",
    position: str = "answered",
    because: str = "checked",
    verification: VerificationReport | None = None,
) -> bytes:
    """Build an EXTERNAL reply TurnFrame (terminal unless needs_from is set)."""
    return encode_frame(
        TurnFrame(
            agent_id=agent_id,
            task_id=task_id,
            visible_text=visible_text,
            stance=Stance(
                position=position,
                because=because,
                needs_from=needs_from or [],
            ),
            origin=origin,
            verification=verification,
        )
    )


def _intake_edge(adapter: FakeAdapter, bus: FakeFrameBus) -> DiscordEdge:
    return DiscordEdge(adapter, bus, stack="dev", id_factory=lambda: "self-1", intake_enabled=True)


def _external_seen_key(
    task_id: str,
    visible_text: str,
    *,
    origin: ChannelOrigin | None,
    agent_id: str = "nagatha",
    position: str = "answered",
    because: str = "checked",
) -> tuple[str, ...]:
    """The (task_id, agent_id, digest) seen-set key an external turn maps to.

    Mirrors the edge's ``_turn_identity_digest`` (SHA-256 over the canonical wire
    encoding of the full typed frame) so a test can assert exactly which key a
    given turn shares or splits from.
    """
    from rtech_chatd.edge import _turn_identity_digest

    frame = TurnFrame(
        agent_id=agent_id,
        task_id=task_id,
        visible_text=visible_text,
        stance=Stance(position=position, because=because, needs_from=[]),
        origin=origin,
    )
    return (task_id, agent_id, _turn_identity_digest(frame))


async def test_external_terminal_turn_posts_to_origin_channel() -> None:
    """A terminal external turn with origin, for an untracked task, is posted."""
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = ChannelOrigin(channel_id="chan-Q", thread_id="thread-Q", message_id="msg-Q")
    await bus.emit(TURN_WILDCARD, _external_turn_bytes("ext-1", "The answer.", origin=origin))

    assert len(adapter.sent) == 1
    target, text = adapter.sent[0]
    assert target.channel_id == "chan-Q"
    assert target.thread_id == "thread-Q"
    assert target.reply_to == "msg-Q"
    assert text == "The answer."


async def test_external_self_minted_task_is_not_double_posted() -> None:
    """A turn for a task this edge minted keeps the per-task path, not external."""
    msg = InboundMessage(channel_id="chan-1", thread_id="thread-9", author="u", text="hi")
    adapter = FakeAdapter(inbound=[msg])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()  # mints self-1, subscribes its per-task lane

    origin = ChannelOrigin(channel_id="chan-Q", message_id="msg-Q")
    # The SAME turn arrives on the wildcard; because self-1 is tracked it is not
    # re-rendered by the external path.
    await bus.emit(TURN_WILDCARD, _external_turn_bytes("self-1", "own answer", origin=origin))
    assert adapter.sent == []


async def test_external_non_terminal_turn_is_ignored() -> None:
    """A non-terminal external turn (needs_from set) is not posted (no streaming)."""
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = ChannelOrigin(channel_id="chan-Q", message_id="msg-Q")
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes("ext-2", "partial", origin=origin, needs_from=["bob"]),
    )
    assert adapter.sent == []


async def test_external_turn_without_origin_is_ignored() -> None:
    """A terminal external turn with no origin has nowhere to route; ignored."""
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    await bus.emit(TURN_WILDCARD, _external_turn_bytes("ext-3", "answer", origin=None))
    assert adapter.sent == []


async def test_external_turn_redelivery_posts_once() -> None:
    """A redelivered external turn (same task + text) posts exactly once."""
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = ChannelOrigin(channel_id="chan-Q", message_id="msg-Q")
    payload = _external_turn_bytes("ext-4", "dup answer", origin=origin)
    await bus.emit(TURN_WILDCARD, payload)
    await bus.emit(TURN_WILDCARD, payload)

    assert len(adapter.sent) == 1


async def test_external_render_failure_is_contained() -> None:
    """A send failure on an external turn is logged and contained, not raised."""

    class _BoomAdapter(FakeAdapter):
        async def send(self, target: SendTarget, text: str, files: list[Path]) -> str | None:
            raise RuntimeError("send blew up")

    adapter = _BoomAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = ChannelOrigin(channel_id="chan-Q", message_id="msg-Q")
    # Must not raise. And because the post failed, the seen-set stays empty so a
    # redelivery can retry.
    await bus.emit(TURN_WILDCARD, _external_turn_bytes("ext-5", "boom", origin=origin))
    assert edge._external_seen == {}


async def test_disabled_intake_installs_no_wildcard_subscription() -> None:
    """With intake disabled the edge subscribes NO turn wildcard (byte-identical)."""
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = DiscordEdge(adapter, bus, stack="dev", id_factory=lambda: "self-1")
    await edge.run_inbound()

    assert TURN_WILDCARD not in bus.handlers
    assert edge._external_subscription is None


async def test_disabled_intake_ignores_external_turn_even_if_delivered() -> None:
    """A default (non-intake) edge never renders an external turn."""
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = DiscordEdge(adapter, bus, stack="dev", id_factory=lambda: "self-1")
    await edge.run_inbound()

    # No handler is registered on the wildcard, so there is nothing to invoke.
    assert TURN_WILDCARD not in bus.handlers
    assert adapter.sent == []


async def test_external_subscription_retired_on_close() -> None:
    """close() unsubscribes the external turn wildcard subscription."""
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()
    sub = bus.subs_by_subject[TURN_WILDCARD]

    await edge.close()

    assert sub.unsubscribed is True
    assert edge._external_subscription is None


async def test_external_concurrent_deliveries_post_once() -> None:
    """R3: two interleaved deliveries of the SAME external turn post exactly once.

    The dedup mark must land BEFORE the await. A send that yields lets a second
    delivery of the identical turn run its check while the first is mid-dispatch;
    under the old check-then-await-then-record split both passed the check and
    double-posted. Marking up front makes the guard atomic.
    """

    gate = asyncio.Event()

    class _GatedAdapter(FakeAdapter):
        gated: bool = False

        async def send(self, target: SendTarget, text: str, files: list[Path]) -> str | None:
            if self.gated:
                await gate.wait()
            self.sent.append((target, text))
            return f"sent-{len(self.sent)}"

    adapter = _GatedAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = ChannelOrigin(channel_id="chan-Q", message_id="msg-Q")
    payload = _external_turn_bytes("ext-cc", "same answer", origin=origin)
    adapter.gated = True

    # Fire both deliveries concurrently; both reach the gate before either posts.
    gathered = asyncio.gather(bus.emit(TURN_WILDCARD, payload), bus.emit(TURN_WILDCARD, payload))
    await asyncio.sleep(0)
    gate.set()
    await gathered

    assert len(adapter.sent) == 1


async def test_external_two_agents_same_text_both_post() -> None:
    """R4: two participants' terminal turns with identical text on one task post twice.

    The dedup key is ``(task_id, agent_id)``, so distinct authors are not
    collapsed and each attribution survives.
    """
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = ChannelOrigin(channel_id="chan-Q", message_id="msg-Q")
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes("ext-multi", "agreed", origin=origin, agent_id="nagatha"),
    )
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes("ext-multi", "agreed", origin=origin, agent_id="bilby"),
    )

    assert len(adapter.sent) == 2


async def test_debate_same_agent_changed_final_renders_distinctly() -> None:
    """A CHANGED debate final for the same (task, agent) is a DISTINCT turn.

    The dedup key is (task_id, agent_id, digest) over the full typed frame, so a
    genuinely different later terminal turn (a re-worded, distinct substantive
    answer) hashes differently and renders as its OWN turn rather than being
    suppressed by an author-only key. Each distinct turn posts its TLDR + full,
    both landing in the agent's (reused) thread. A BYTE-IDENTICAL redelivery is
    covered by ``test_debate_exact_duplicate_collapses_to_one``.
    """
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = _debate_origin()
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes("deb-rw", "first wording", origin=origin, agent_id="nagatha"),
    )
    assert len(adapter.sent) == 2  # first turn: TLDR + full
    # Same (task, agent), DIFFERENT text: a distinct terminal turn, so it renders
    # its own TLDR + full (the digest key no longer collapses distinct content).
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes("deb-rw", "reworded final", origin=origin, agent_id="nagatha"),
    )
    assert len(adapter.sent) == 4  # second distinct turn: another TLDR + full
    # The full text of the second turn is the reworded content (not the first's).
    fulls = [txt for t, txt in adapter.sent if t.thread_id is not None]
    assert "reworded final" in fulls


def _debate_origin(**overrides: object) -> ChannelOrigin:
    """A debate-marked origin (TLDR-to-channel + full-to-thread render shape)."""
    base: dict[str, object] = {
        "channel_id": "chan-Q",
        "thread_id": "thread-Q",
        "message_id": "msg-Q",
        "debate": True,
    }
    base.update(overrides)
    return ChannelOrigin(**base)  # type: ignore[arg-type]


async def test_debate_three_participants_render_tldr_and_per_agent_threads() -> None:
    """3 debate participants -> 3 plain channel TLDRs, each anchoring its OWN thread.

    UX v2: each participant's final terminal turn posts exactly once (per-agent
    collapse) as a stance-derived TLDR sent PLAIN to the parent channel (never a
    reply), and that TLDR message anchors the agent's OWN thread holding its full
    visible_text — one thread per (task, agent), named for the agent, never one
    shared debate thread and never anchored on the ask message.
    """
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = _debate_origin()
    for agent, pos, why, full in (
        ("nagatha", "ship", "the pipeline is green", "Full reasoning from nagatha."),
        ("bilby", "hold", "coverage dipped", "Full reasoning from bilby."),
        ("iris", "ship", "risk is bounded", "Full reasoning from iris."),
    ):
        await bus.emit(
            TURN_WILDCARD,
            _external_turn_bytes(
                "deb-1", full, origin=origin, agent_id=agent, position=pos, because=why
            ),
        )

    # THREE threads: one per agent, each anchored on that agent's OWN TLDR
    # message (never the ask message, never another agent's), named for the
    # agent. Sequential renders: agent k's TLDR is send 2k -> id "sent-<2k+1>",
    # so the exact (anchor, name) pairing proves per-agent anchoring — a
    # cross-keyed anchor write would break this even if the id SETS matched.
    assert adapter.thread_calls == [
        ("chan-Q", "sent-1", "nagatha — full thought"),
        ("chan-Q", "sent-3", "bilby — full thought"),
        ("chan-Q", "sent-5", "iris — full thought"),
    ]
    assert adapter.sent[0][1].startswith("**nagatha**")  # sent-1 IS nagatha's TLDR
    assert adapter.sent[2][1].startswith("**bilby**")  # sent-3 IS bilby's TLDR
    assert adapter.sent[4][1].startswith("**iris**")  # sent-5 IS iris's TLDR
    # 3 participants -> 6 dispatches (a channel TLDR + an in-thread full each).
    assert len(adapter.sent) == 6
    channel_posts = [(t, txt) for t, txt in adapter.sent if t.thread_id is None]
    thread_posts = [(t, txt) for t, txt in adapter.sent if t.thread_id is not None]
    assert len(channel_posts) == 3
    assert len(thread_posts) == 3
    # TLDRs are short, stance-derived, agent-prefixed, and PLAIN (no reply ref).
    tldr_texts = {txt for _, txt in channel_posts}
    assert "**nagatha** — ship: the pipeline is green" in tldr_texts
    assert "**bilby** — hold: coverage dipped" in tldr_texts
    assert all(t.reply_to is None for t, _ in channel_posts)
    # Each full landed in its OWN thread (three distinct thread ids), each thread
    # anchored on that agent's TLDR message.
    assert len({t.thread_id for t, _ in thread_posts}) == 3
    assert {txt for _, txt in thread_posts} == {
        "Full reasoning from nagatha.",
        "Full reasoning from bilby.",
        "Full reasoning from iris.",
    }


async def test_debate_verification_metadata_renders_only_in_full_thought() -> None:
    """Keep verification detail out of the TLDR and render it once in-thread."""
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    verification = VerificationReport(
        status="partial",
        claim_kind="factual",
        ob_checked=True,
        sources=[
            EvidenceSource(
                source_id="repo-edge",
                kind="repo",
                label="Edge implementation",
                locator="packages/rtech-chatd/src/rtech_chatd/edge.py",
                authority="source",
            )
        ],
        limitations=["Runtime delivery was not independently verified."],
        methods=["repo inspection"],
    )
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes(
            "deb-evidence",
            "Full verification-aware reasoning.",
            origin=_debate_origin(),
            agent_id="nagatha",
            position="ship",
            because="the source supports the change",
            verification=verification,
        ),
    )

    assert len(adapter.sent) == 2
    parent_text = adapter.sent[0][1]
    full_text = adapter.sent[1][1]
    assert "Sources checked" not in parent_text
    assert "Limitations" not in parent_text
    assert full_text.count("Sources checked") == 1
    assert full_text.count("Limitations") == 1
    assert full_text.count("Edge implementation") == 1
    assert full_text.count("Runtime delivery was not independently verified.") == 1


async def test_debate_cap_closing_turn_renders_one_closing_post() -> None:
    """A cap-closing turn (distinct closing author) adds exactly one closing post.

    The closing turn carries a distinct ``agent_id`` (the closing author), so it
    keys distinctly and renders as +1 (its own TLDR + in-thread full) on top of the
    participants' finals — the bounded 'N + at most 1 closing' channel post count.
    """
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = _debate_origin()
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes("deb-2", "n final", origin=origin, agent_id="nagatha"),
    )
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes("deb-2", "b final", origin=origin, agent_id="bilby"),
    )
    channel_before = [t for t, _ in adapter.sent if t.thread_id is None]
    assert len(channel_before) == 2
    # The cap-closing turn: a distinct author id => a distinct (task, agent) key.
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes(
            "deb-2",
            "This position stands as final: budget reached.",
            origin=origin,
            agent_id="nagatha-closing",
            position="ship",
            because="round budget reached",
        ),
    )
    channel_after = [t for t, _ in adapter.sent if t.thread_id is None]
    assert len(channel_after) == 3  # 2 participants + 1 closing TLDR


async def test_debate_non_terminal_round_posts_nothing() -> None:
    """A non-terminal intermediate debate round renders zero posts (no trickle)."""
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = _debate_origin()
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes(
            "deb-3", "round one", origin=origin, agent_id="nagatha", needs_from=["bilby"]
        ),
    )
    assert adapter.sent == []


async def test_debate_terminates_bounded_channel_posts() -> None:
    """Loop/bound safety: a full trio debate settles to a bounded channel count.

    Three participants each post one final (per-agent collapse absorbs any
    redelivery) and the debate emits at most one closing turn: channel posts are
    bounded by N + 1. No new bound is introduced here — MAX_SIBLING_ROUNDS already
    bounds the debate; this asserts the channel render honors that bound.
    """
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = _debate_origin()
    participants = ("nagatha", "bilby", "iris")
    # Each participant's final, plus a redelivery of each (must collapse).
    for agent in participants:
        payload = _external_turn_bytes("deb-4", f"{agent} final", origin=origin, agent_id=agent)
        await bus.emit(TURN_WILDCARD, payload)
        await bus.emit(TURN_WILDCARD, payload)
    # One closing turn.
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes("deb-4", "final stands", origin=origin, agent_id="nagatha-closing"),
    )
    channel_posts = [t for t, _ in adapter.sent if t.thread_id is None]
    # N participants + at most 1 closing == 4, never more.
    assert len(channel_posts) == len(participants) + 1


async def test_debate_thread_create_failure_degrades_full_to_channel() -> None:
    """A thread-create failure posts the full text to the parent channel (D1).

    When an agent's lazy thread creation fails (adapter returns None), that
    agent's full turn text is never dropped: it degrades (loudly) to the parent
    channel as the last resort, alongside the TLDR. The failed create is cached
    per (task, agent) so it is not retried on redelivery — while a DIFFERENT
    agent still gets its own create attempt (UX v2: one thread per agent).
    """
    # Seed a None so the FIRST create_thread call (nagatha's) fails.
    adapter = FakeAdapter(inbound=[], thread_ids=[None])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = _debate_origin()
    payload = _external_turn_bytes("deb-5", "full body", origin=origin, agent_id="nagatha")
    await bus.emit(TURN_WILDCARD, payload)
    # TLDR + full, both to the parent channel (thread create failed).
    assert len(adapter.sent) == 2
    assert all(t.thread_id is None for t, _ in adapter.sent)
    texts = {txt for _, txt in adapter.sent}
    assert "full body" in texts
    assert len(adapter.thread_calls) == 1

    # A redelivery of the SAME agent's turn is fully collapsed — and even if a
    # later same-key piece were retried, the cached failed create is not re-run.
    await bus.emit(TURN_WILDCARD, payload)
    assert len(adapter.thread_calls) == 1
    assert len(adapter.sent) == 2

    # A SECOND participant gets its OWN create attempt (per-agent threads); the
    # default fake create succeeds, so bilby's full lands in bilby's thread.
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes("deb-5", "more body", origin=origin, agent_id="bilby"),
    )
    assert len(adapter.thread_calls) == 2
    bilby_thread_posts = [txt for t, txt in adapter.sent if t.thread_id is not None]
    assert bilby_thread_posts == ["more body"]


async def test_non_debate_same_agent_changed_terminal_renders_distinctly() -> None:
    """A CHANGED non-debate terminal for the same task+agent is a DISTINCT turn.

    The dedup key is (task_id, agent_id, digest) over the full typed frame, so a
    genuinely different later terminal answer for the same (task, agent) hashes
    differently and renders as its own turn — an author-only key would have
    silently suppressed the distinct later answer. A byte-identical redelivery
    still collapses (``test_non_debate_exact_duplicate_collapses_to_one``).
    """
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = ChannelOrigin(channel_id="chan-Q", message_id="msg-Q")  # debate=False
    # First terminal answer 'main' lands.
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes("mainmain-1", "main", origin=origin, agent_id="nagatha"),
    )
    assert [txt for _, txt in adapter.sent] == ["main"]
    # Same (task, agent), CHANGED text 'mainmain': a distinct terminal turn, so it
    # renders (the digest differs from the first).
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes("mainmain-1", "mainmain", origin=origin, agent_id="nagatha"),
    )
    assert [txt for _, txt in adapter.sent] == ["main", "mainmain"]
    # A BYTE-IDENTICAL redelivery of the first collapses (shares its digest key).
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes("mainmain-1", "main", origin=origin, agent_id="nagatha"),
    )
    assert [txt for _, txt in adapter.sent] == ["main", "mainmain"]


async def test_non_debate_exact_duplicate_collapses_to_one() -> None:
    """A byte-identical non-debate redelivery collapses to exactly one render.

    The (task_id, agent_id, digest) key over the full typed frame is shared by a
    byte-identical redelivery, so the exact duplicate never double-posts.
    """
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = ChannelOrigin(channel_id="chan-Q", message_id="msg-Q")  # debate=False
    payload = _external_turn_bytes("dup-1", "same answer", origin=origin, agent_id="nagatha")
    await bus.emit(TURN_WILDCARD, payload)
    await bus.emit(TURN_WILDCARD, payload)
    await bus.emit(TURN_WILDCARD, payload)
    assert [txt for _, txt in adapter.sent] == ["same answer"]
    # The shared key carries a single completed record.
    key = _external_seen_key("dup-1", "same answer", origin=origin, agent_id="nagatha")
    assert edge._external_seen[key].pieces == {"full"}


async def test_debate_exact_duplicate_collapses_to_one() -> None:
    """A byte-identical DEBATE redelivery posts exactly one TLDR + one full."""
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = _debate_origin()
    payload = _external_turn_bytes("deb-dup", "final answer", origin=origin, agent_id="nagatha")
    await bus.emit(TURN_WILDCARD, payload)
    assert len(adapter.sent) == 2  # TLDR + full
    await bus.emit(TURN_WILDCARD, payload)
    assert len(adapter.sent) == 2  # exact duplicate adds nothing


async def test_non_debate_guard_suppression_releases_external_dedup_reservation() -> None:
    """A suppressed terminal answer cannot poison a later valid same-key answer."""

    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = DiscordEdge(
        adapter,
        bus,
        stack="dev",
        id_factory=lambda: "self-1",
        intake_enabled=True,
        outbound_filter=lambda text: "" if text == "suppressed" else text,
    )
    await edge.run_inbound()

    origin = ChannelOrigin(channel_id="chan-Q", message_id="msg-Q")
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes("suppressed-1", "suppressed", origin=origin, agent_id="nagatha"),
    )
    assert adapter.sent == []
    assert edge._external_seen == {}

    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes("suppressed-1", "valid", origin=origin, agent_id="nagatha"),
    )
    assert [text for _, text in adapter.sent] == ["valid"]


async def test_changed_terminal_frame_keys_distinctly_from_original() -> None:
    """The content digest key SPLITS a changed terminal turn from the original.

    Directly asserts the identity contract behind the render fix: two turns that
    differ in any typed field (here ``visible_text``) map to DIFFERENT seen-set
    keys, so a distinct later terminal turn can never share (and be suppressed by)
    the original's progress record; a byte-identical turn maps to the SAME key.
    """
    origin = _debate_origin()
    first = _external_seen_key("k-1", "main", origin=origin, agent_id="nagatha")
    changed = _external_seen_key("k-1", "mainmain", origin=origin, agent_id="nagatha")
    same = _external_seen_key("k-1", "main", origin=origin, agent_id="nagatha")
    assert first != changed  # changed content → distinct key → renders distinctly
    assert first == same  # byte-identical → shared key → collapses
    # task_id stays key[0] (eviction cleanup) and agent_id stays key[1].
    assert first[0] == "k-1"
    assert first[1] == "nagatha"


async def test_non_debate_send_failure_redelivery_retries() -> None:
    """dup-external: partial-render retry survives the collapse.

    If the FIRST non-debate send fails before the piece lands, nothing is marked
    complete, so the key's progress record is dropped and a redelivery RETRIES the
    send. Once the send lands, a subsequent (even mutated-text) duplicate must NOT
    post again.
    """

    class _FailOnceExternalAdapter(FakeAdapter):
        fail_next: int = 0

        async def send(self, target: SendTarget, text: str, files: list[Path]) -> str | None:
            if self.fail_next > 0:
                self.fail_next -= 1
                raise RuntimeError("external send blew up")
            self.sent.append((target, text))
            return f"sent-{len(self.sent)}"

    adapter = _FailOnceExternalAdapter(inbound=[])
    adapter.fail_next = 1
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = ChannelOrigin(channel_id="chan-Q", message_id="msg-Q")  # debate=False

    # First delivery: the send raises before the piece lands. Contained, and the
    # seen-set stays empty so a redelivery can retry.
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes("nd-retry", "main", origin=origin, agent_id="nagatha"),
    )
    assert adapter.sent == []
    assert edge._external_seen == {}

    # Redelivery: the send now lands (retry semantics preserved).
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes("nd-retry", "main", origin=origin, agent_id="nagatha"),
    )
    assert [txt for _, txt in adapter.sent] == ["main"]

    # A later BYTE-IDENTICAL redelivery must NOT post: the piece already landed
    # under this turn's shared digest key.
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes("nd-retry", "main", origin=origin, agent_id="nagatha"),
    )
    assert [txt for _, txt in adapter.sent] == ["main"]


async def test_chunked_full_redelivery_resumes_monotonically_and_clears_cursor() -> None:
    """Partial external sends resume only the missing chunks, then dedup normally."""

    class _ProgressAdapter(FakeAdapter):
        def __init__(self) -> None:
            super().__init__(inbound=[])
            self.fail_at = [1, 2]
            self.landed_chunks: list[int] = []
            self.resume_cursors: list[int | None] = []

        async def send(self, target: SendTarget, text: str, files: list[Path]) -> str | None:
            resume = target.chunk_resume
            self.resume_cursors.append(None if resume is None else resume.chunks_landed)
            start = resume.chunks_landed if resume is not None else 0
            for index in range(start, 3):
                if self.fail_at and self.fail_at[0] == index:
                    self.fail_at.pop(0)
                    exc = RuntimeError("chunk send failed")
                    attach_chunk_progress(
                        exc,
                        ChunkProgress(
                            chunks_landed=index,
                            chunk_count=3,
                            plan_fingerprint="plan-v1",
                        ),
                    )
                    raise exc
                self.landed_chunks.append(index)
            self.sent.append((target, text))
            return f"sent-{len(self.sent)}"

    adapter = _ProgressAdapter()
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()
    origin = ChannelOrigin(channel_id="chan-Q", message_id="msg-Q")
    payload = _external_turn_bytes("chunk-retry", "full body", origin=origin, agent_id="nagatha")
    key = _external_seen_key("chunk-retry", "full body", origin=origin, agent_id="nagatha")

    await bus.emit(TURN_WILDCARD, payload)
    progress = edge._external_seen[key]
    assert progress.chunk_cursors["full"].chunks_landed == 1

    await bus.emit(TURN_WILDCARD, payload)
    progress = edge._external_seen[key]
    assert progress.chunk_cursors["full"].chunks_landed == 2

    await bus.emit(TURN_WILDCARD, payload)
    progress = edge._external_seen[key]
    assert adapter.landed_chunks == [0, 1, 2]
    assert adapter.resume_cursors == [None, 1, 2]
    assert progress.chunk_cursors == {}
    assert progress.pieces == {"full"}

    await bus.emit(TURN_WILDCARD, payload)
    assert adapter.landed_chunks == [0, 1, 2]


async def test_chunked_full_first_chunk_failure_retains_zero_cursor() -> None:
    """A first-chunk FloodWait checkpoint remains zero rather than claiming a send."""

    class _FirstChunkFailsAdapter(FakeAdapter):
        async def send(self, target: SendTarget, text: str, files: list[Path]) -> str | None:
            exc = RuntimeError("chunk send failed")
            attach_chunk_progress(
                exc,
                ChunkProgress(chunks_landed=0, chunk_count=3, plan_fingerprint="plan-v1"),
            )
            raise exc

    adapter = _FirstChunkFailsAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()
    origin = ChannelOrigin(channel_id="chan-Q", message_id="msg-Q")
    key = _external_seen_key("chunk-zero", "full body", origin=origin, agent_id="nagatha")

    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes("chunk-zero", "full body", origin=origin, agent_id="nagatha"),
    )

    progress = edge._external_seen[key]
    assert progress.pieces == set()
    assert progress.chunk_cursors["full"].chunks_landed == 0


async def test_debate_full_send_failure_redelivery_reposts_only_full() -> None:
    """D4/D5 repro: TLDR lands, full (thread) send fails -> redelivery reposts full only.

    Split render-progress marks (tldr-sent / full-sent): a partial failure retires
    only the missing piece. The TLDR that landed is NEVER re-posted on redelivery;
    the failed full is retried, so the full text eventually lands and the TLDR is
    not duplicated (content is never lost, nothing double-posts).
    """

    class _FullFailsOnceAdapter(FakeAdapter):
        fail_thread_sends: int = 0

        async def send(self, target: SendTarget, text: str, files: list[Path]) -> str | None:
            # A thread-targeted send is the FULL piece; fail it the first N times.
            if target.thread_id is not None and self.fail_thread_sends > 0:
                self.fail_thread_sends -= 1
                raise RuntimeError("thread send blew up")
            self.sent.append((target, text))
            return f"sent-{len(self.sent)}"

    adapter = _FullFailsOnceAdapter(inbound=[])
    adapter.fail_thread_sends = 1
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = _debate_origin()
    payload = _external_turn_bytes("deb-pf", "full body", origin=origin, agent_id="nagatha")

    # First delivery: TLDR lands (parent channel), full (thread) raises.
    await bus.emit(TURN_WILDCARD, payload)
    landed = adapter.sent
    assert [t.thread_id for t, _ in landed] == [None]  # only the TLDR landed
    tldr_text = landed[0][1]

    # Redelivery: TLDR is already marked (not re-posted); only the full retries and
    # now lands in the thread.
    await bus.emit(TURN_WILDCARD, payload)
    assert len(adapter.sent) == 2
    # The TLDR was posted exactly once (not duplicated).
    assert [txt for _, txt in adapter.sent].count(tldr_text) == 1
    # The full text landed in the created thread.
    thread_posts = [(t, txt) for t, txt in adapter.sent if t.thread_id is not None]
    assert len(thread_posts) == 1
    assert thread_posts[0][1] == "full body"


async def test_debate_full_failure_then_thread_unusable_falls_back_to_channel() -> None:
    """D5 repro: full send keeps failing in-thread; last-resort parent fallback lands it.

    Even when the thread is unusable (every in-thread send raises), the full text is
    never lost: the TLDR lands once, and a redelivery re-routes the still-missing
    full to the parent channel as the documented last resort — the TLDR is not
    duplicated.
    """

    class _ThreadAlwaysFailsAdapter(FakeAdapter):
        async def send(self, target: SendTarget, text: str, files: list[Path]) -> str | None:
            if target.thread_id is not None:
                raise RuntimeError("thread permanently unusable")
            self.sent.append((target, text))
            return f"sent-{len(self.sent)}"

    # No thread ids seeded -> create returns thread-for-msg-Q; but every in-thread
    # send raises, so the full must eventually degrade to the parent channel.
    adapter = _ThreadAlwaysFailsAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = _debate_origin()
    payload = _external_turn_bytes("deb-tf", "full body", origin=origin, agent_id="nagatha")

    # First delivery: TLDR lands, thread full raises -> full stays unmarked.
    await bus.emit(TURN_WILDCARD, payload)
    assert [t.thread_id for t, _ in adapter.sent] == [None]  # only TLDR
    tldr_text = adapter.sent[0][1]

    # A second delivery still fails the thread send. To model the last-resort
    # parent fallback we now make the thread unavailable (create returns None):
    # the full degrades to the parent channel and lands.
    edge._debate_threads[("deb-tf", "nagatha")] = None  # unusable -> parent fallback
    await bus.emit(TURN_WILDCARD, payload)
    # TLDR posted exactly once; the full text landed in the parent channel.
    assert [txt for _, txt in adapter.sent].count(tldr_text) == 1
    assert any(t.thread_id is None and txt == "full body" for t, txt in adapter.sent)


async def test_external_evicted_key_reposts_bounded_best_effort() -> None:
    """R3: after cap-eviction the same turn posts again on redelivery (documented).

    The seen-set cap evicts oldest-first; an already-posted turn whose key was
    evicted under redelivery pressure re-posts. This is the bounded-best-effort
    property the dedup docstring documents — exact within the cap, best-effort past
    it.
    """
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    # A cap of 1 makes the eviction deterministic: any second key evicts the first.
    edge = DiscordEdge(
        adapter,
        bus,
        stack="dev",
        id_factory=lambda: "self-1",
        intake_enabled=True,
        external_seen_max=1,
    )
    await edge.run_inbound()

    origin = ChannelOrigin(channel_id="chan-Q", message_id="msg-Q")
    first = _external_turn_bytes("ext-eA", "answer A", origin=origin)
    first_key = _external_seen_key("ext-eA", "answer A", origin=origin)
    # Post A -> its key is the sole entry.
    await bus.emit(TURN_WILDCARD, first)
    assert len(adapter.sent) == 1
    # A different turn posts and evicts A's key (cap=1).
    await bus.emit(TURN_WILDCARD, _external_turn_bytes("ext-eB", "answer B", origin=origin))
    assert len(adapter.sent) == 2
    assert first_key not in edge._external_seen
    # A redelivered -> its key was evicted, so it re-posts (bounded-best-effort).
    await bus.emit(TURN_WILDCARD, first)
    assert len(adapter.sent) == 3


async def test_debate_concurrent_agents_each_get_own_thread() -> None:
    """Concurrent first-renders from two agents create exactly TWO threads (UX v2).

    One thread per (task, agent): two participants racing their first renders
    must each end with their OWN thread, each anchored on their own TLDR message,
    with each full text in its author's thread — the creation lock serializes the
    per-key check-create-cache sequence without collapsing distinct agents onto
    one thread.
    """

    class _YieldingAdapter(FakeAdapter):
        async def create_thread(self, channel_id: str, message_id: str, name: str) -> str | None:
            # Yield twice so the second concurrent render reaches the creation
            # path while this create is still in flight.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return await super().create_thread(channel_id, message_id, name)

    adapter = _YieldingAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = _debate_origin()
    await asyncio.gather(
        bus.emit(
            TURN_WILDCARD,
            _external_turn_bytes("deb-race", "nagatha full", origin=origin, agent_id="nagatha"),
        ),
        bus.emit(
            TURN_WILDCARD,
            _external_turn_bytes("deb-race", "bilby full", origin=origin, agent_id="bilby"),
        ),
    )
    assert len(adapter.thread_calls) == 2
    # Each thread is anchored on ITS OWN agent's TLDR message: resolve every
    # anchor id back to the recorded send and check the TLDR names the same
    # agent the thread is named for (cross-anchoring would flunk this).
    for _, anchor_id, name in adapter.thread_calls:
        agent = name.split(" — ")[0]
        anchored_target, anchored_text = adapter.sent[int(anchor_id.split("-")[1]) - 1]
        assert anchored_target.thread_id is None  # anchored on a channel TLDR
        assert anchored_text.startswith(f"**{agent}**")
    thread_posts = [(t.thread_id, txt) for t, txt in adapter.sent if t.thread_id is not None]
    assert {txt for _, txt in thread_posts} == {"nagatha full", "bilby full"}
    assert len({tid for tid, _ in thread_posts}) == 2


async def test_debate_concurrent_same_agent_deliveries_create_one_thread() -> None:
    """Concurrent duplicate deliveries of ONE agent's turn still thread correctly.

    The adversarial window: delivery A is mid-TLDR-send (yielded on the network)
    when duplicate B reaches the FULL piece — the anchor is not recorded yet, so
    without the in-flight guard B degrades the full text to the channel and
    poisons the thread cache with ``None`` forever. The fake yields inside BOTH
    send and create_thread so the window is actually open. With the guard: B
    defers, A lands the TLDR then renders FULL into the agent's real thread —
    one TLDR, one create, full text IN-THREAD, never degraded.
    """

    class _YieldingAdapter(FakeAdapter):
        async def send(self, target: SendTarget, text: str, files: list[Path]) -> str | None:
            # Yield BEFORE recording, opening the mid-send window for the
            # concurrent duplicate to observe "TLDR claimed, anchor missing".
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return await super().send(target, text, files)

        async def create_thread(self, channel_id: str, message_id: str, name: str) -> str | None:
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return await super().create_thread(channel_id, message_id, name)

    adapter = _YieldingAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = _debate_origin()
    payload = _external_turn_bytes("deb-dup", "nagatha full", origin=origin, agent_id="nagatha")
    await asyncio.gather(bus.emit(TURN_WILDCARD, payload), bus.emit(TURN_WILDCARD, payload))
    assert len(adapter.thread_calls) == 1
    assert len(adapter.sent) == 2  # one TLDR + one full
    # The full text landed IN the created thread (not degraded to the channel),
    # and the thread is anchored on the TLDR message that actually landed.
    tldr_anchor = adapter.thread_calls[0][1]
    full_sends = [(t, txt) for t, txt in adapter.sent if txt == "nagatha full"]
    assert full_sends == [
        (SendTarget(channel_id="chan-Q", thread_id=f"thread-for-{tldr_anchor}"), "nagatha full")
    ]
    # The cache holds a REAL thread id for this key — never a poisoned None.
    assert edge._debate_threads[("deb-dup", "nagatha")] == f"thread-for-{tldr_anchor}"


async def test_debate_fifth_distinct_agent_tldr_suppressed_full_still_routes() -> None:
    """D8 boundary+1: the 5th distinct agent gets NO TLDR but its full text routes.

    The distinct-agent TLDR cap is 4 (trio + closing). A 5th novel agent id on the
    same debate task is warned + suppressed on the TLDR piece; with no TLDR of its
    own to anchor a thread on (UX v2), its full text degrades to the parent
    channel — routed loudly, never lost, never a 5th TLDR.
    """
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = _debate_origin()
    agents = ["nagatha", "bilby", "iris", "closing"]
    for agent in agents:
        await bus.emit(
            TURN_WILDCARD,
            _external_turn_bytes("deb-cap", f"{agent} full", origin=origin, agent_id=agent),
        )
    channel_before = [t for t, _ in adapter.sent if t.thread_id is None]
    assert len(channel_before) == len(agents)  # 4 TLDRs, all within cap
    assert len(adapter.thread_calls) == len(agents)  # one thread per agent

    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes("deb-cap", "fifth full", origin=origin, agent_id="fifth"),
    )
    channel_texts_after = [txt for t, txt in adapter.sent if t.thread_id is None]
    # No 5th TLDR rendered (no "**fifth**" stance line)...
    assert not any(txt.startswith("**fifth**") for txt in channel_texts_after)
    # ...but the full text still routed (degraded to the parent channel: the 5th
    # agent has no TLDR message to anchor its thread on).
    assert "fifth full" in channel_texts_after
    assert len(adapter.thread_calls) == len(agents)  # no 5th create attempt


async def test_identity_send_debate_stamps_agent_and_drops_name_prefix() -> None:
    """identity_send=True: every debate piece is voiced as its author (#109).

    The TLDR and the in-thread full both carry ``as_agent=<turn agent_id>`` so a
    multi-identity surface posts them under that agent's OWN bot; the TLDR drops
    its ``**agent** —`` prefix (the posting identity IS the attribution); and the
    agent's typing indicator is stopped when its terminal turn renders.
    """
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = DiscordEdge(
        adapter,
        bus,
        stack="dev",
        id_factory=lambda: "self-1",
        intake_enabled=True,
        identity_send=True,
    )
    await edge.run_inbound()

    origin = _debate_origin()
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes(
            "deb-id",
            "Full reasoning from nagatha.",
            origin=origin,
            agent_id="agentd-nagatha",
            position="ship",
            because="the pipeline is green",
        ),
    )

    assert len(adapter.sent) == 2
    tldr_target, tldr_text = adapter.sent[0]
    full_target, full_text = adapter.sent[1]
    # Both pieces voiced as the turn's author.
    assert tldr_target.as_agent == "agentd-nagatha"
    assert full_target.as_agent == "agentd-nagatha"
    # TLDR is stance-derived and UNNAMED (no ventriloquism prefix).
    assert tldr_text == "ship: the pipeline is green"
    assert full_text == "Full reasoning from nagatha."
    # The agent's typing indicator was stopped at terminal render, keyed to the
    # same voiced identity in the origin channel.
    stops = [(t, on) for t, on in adapter.typing_calls if on is False]
    assert len(stops) == 1
    assert stops[0][0].channel_id == "chan-Q"
    assert stops[0][0].as_agent == "agentd-nagatha"


async def test_identity_send_off_keeps_named_tldr_and_unstamped_targets() -> None:
    """Default (identity_send=False) stays byte-identical: named TLDR, no stamp."""
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = _intake_edge(adapter, bus)
    await edge.run_inbound()

    origin = _debate_origin()
    await bus.emit(
        TURN_WILDCARD,
        _external_turn_bytes(
            "deb-off",
            "Full reasoning.",
            origin=origin,
            agent_id="agentd-nagatha",
            position="ship",
            because="green",
        ),
    )

    assert len(adapter.sent) == 2
    tldr_target, tldr_text = adapter.sent[0]
    assert tldr_target.as_agent is None
    assert adapter.sent[1][0].as_agent is None
    assert tldr_text == "**agentd-nagatha** — ship: green"
    # No typing stop is issued on the default path.
    assert adapter.typing_calls == []


# --- turn-interaction hook failure log safety (issue #117) ----------------

# The sentinel a raising interaction hook would smuggle into a log line if the
# edge logged the raw exception text instead of only its type name.
_EDGE_SECRET_SENTINEL = "SUPER_SECRET_NATS_URL"


class _LeakingTurnInteraction:
    """A turn-interaction hook that raises an exception carrying the sentinel."""

    async def offer_debate_tldr(
        self,
        *,
        target: SendTarget,
        anchor_message_id: str,
        task_id: str,
        agent_id: str,
        asker: str | None,
    ) -> None:
        raise RuntimeError(f"hook exploded talking to nats://{_EDGE_SECRET_SENTINEL}:4222")


async def test_turn_interaction_hook_failure_never_logs_raw_exception_text() -> None:
    adapter = FakeAdapter(inbound=[])
    bus = FakeFrameBus()
    edge = DiscordEdge(
        adapter,
        bus,
        stack="dev",
        id_factory=lambda: "task-1",
        turn_interaction=_LeakingTurnInteraction(),  # type: ignore[arg-type]
    )
    frame = TurnFrame(
        agent_id="agentd-bilby",
        task_id="task-9",
        visible_text="Full reasoning.",
        stance=Stance(position="ship", because="green"),
        origin=_debate_origin(asker="human-1"),
    )

    captured: list[str] = []
    handler_id = logger.add(captured.append, level="TRACE", format="{message}")
    try:
        # A raising hook must be contained: the coroutine returns normally (never
        # propagates), so the render path it decorates can never crash.
        await edge._offer_turn_interaction(frame, SendTarget(channel_id="chan-Q"), "anchor-500")
    finally:
        logger.remove(handler_id)

    joined = "".join(captured)
    # The raw exception text — and the sentinel URL inside it — must never reach a
    # log line; only the exception type name plus safe task/agent context may.
    assert _EDGE_SECRET_SENTINEL not in joined
    assert "nats://" not in joined
    assert "RuntimeError" in joined
    assert "task-9" in joined
    assert "agentd-bilby" in joined
