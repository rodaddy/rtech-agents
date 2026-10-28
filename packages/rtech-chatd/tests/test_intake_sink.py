"""Functional tests for the intake SINK (slice 2).

The sink turns a confirmed intake hit into a bus publish toward the triage agent.
These cover the four behaviors the sink owns:

* dedup by ``message_id`` (same id twice → ONE publish; horizon + capacity),
* responder-yield (direct @mention of the triage agent → no intake publish;
  a member-config hit for a DIFFERENT agent → intake publishes),
* publish payload correctness (subject via ``subjects.agent_task``, the frame
  decodes, channel origin is carried, ``created_at`` is stamped),
* bounded memory (LRU eviction by count).

A fake in-process bus captures ``(subject, bytes)`` pairs; no NATS, no discord.py.
"""

from __future__ import annotations

import asyncio

import pytest
from loguru import logger
from rtech_chatd import subjects
from rtech_chatd.channel import InboundMessage, SendTarget
from rtech_chatd.config import GateConfig, IntakeConfig, IntakeMode
from rtech_chatd.intake_sink import IntakeSink, _MessageIdDedup
from rtech_frames.codec import decode_frame
from rtech_frames.frames import TaskFrame


class FakeBus:
    """Capture published ``(subject, data)`` pairs (the FrameBus publish surface)."""

    def __init__(self) -> None:
        self.published: list[tuple[str, bytes]] = []

    async def publish(self, subject: str, data: bytes) -> None:
        self.published.append((subject, data))

    async def subscribe(self, subject, handler):  # noqa: ANN001, ANN201 - unused by the sink
        raise NotImplementedError


class BoomBus(FakeBus):
    """A bus whose publish always fails (loud-failure boundary test)."""

    async def publish(self, subject: str, data: bytes) -> None:
        raise RuntimeError("publish exploded")


def _msg(**overrides: object) -> InboundMessage:
    base: dict[str, object] = {
        "channel_id": "chan1",
        "thread_id": None,
        "author": "user1",
        "text": "hey can you look at this",
        "is_bot": False,
        "is_dm": False,
        "mentions_me": False,
        "is_reply": False,
        "mentions_other_bot": False,
        "mentions_human": False,
        "guild_id": "g1",
        "message_id": "m1",
        "mentioned_ids": frozenset(),
    }
    base.update(overrides)
    return InboundMessage(**base)  # type: ignore[arg-type]


def _intake(**overrides: object) -> IntakeConfig:
    base: dict[str, object] = {
        "enabled": True,
        "mode": IntakeMode.ALL_CHANNELS,
        # The deployed triage id is the full agentd-<name> (RTECH_AGENT_ID);
        # a bare name would publish to a subject no agentd subscribes to.
        "triage_agent": "agentd-nimbus",
        "member_ids": frozenset({"77"}),
    }
    base.update(overrides)
    return IntakeConfig(**base)  # type: ignore[arg-type]


def _sink(bus: FakeBus, **kw: object) -> IntakeSink:
    return IntakeSink(
        bus=bus,  # type: ignore[arg-type]
        intake_config=kw.get("intake", _intake()),  # type: ignore[arg-type]
        gate_config=kw.get("gate", GateConfig()),  # type: ignore[arg-type]
        stack=kw.get("stack", "rtech-dev"),  # type: ignore[arg-type]
        id_factory=kw.get("id_factory"),  # type: ignore[arg-type]
        should_respond_fn=kw.get("should_respond_fn"),  # type: ignore[arg-type]
        clock=kw.get("clock"),  # type: ignore[arg-type]
        dedup_max_entries=kw.get("dedup_max_entries", 65536),  # type: ignore[arg-type]
        dedup_horizon_s=kw.get("dedup_horizon_s", 3600.0),  # type: ignore[arg-type]
        ack_send=kw.get("ack_send"),  # type: ignore[arg-type]
        typing_start=kw.get("typing_start"),  # type: ignore[arg-type]
        voiced=kw.get("voiced", True),  # type: ignore[arg-type]
    )


# --------------------------------------------------------------- construction


def test_blank_stack_fails_closed() -> None:
    with pytest.raises(ValueError, match="stack must be non-empty"):
        IntakeSink(
            bus=FakeBus(),  # type: ignore[arg-type]
            intake_config=_intake(),
            gate_config=GateConfig(),
            stack="   ",
        )


# --------------------------------------------------------------- publish payload


def test_publish_subject_and_frame_correctness() -> None:
    bus = FakeBus()
    sink = _sink(bus, id_factory=lambda: "task-abc")
    msg = _msg(
        channel_id="100",
        thread_id="t9",
        guild_id="200",
        message_id="8080",
        author="42",
        mentioned_ids=frozenset({"77"}),
    )
    asyncio.run(sink(msg))

    assert len(bus.published) == 1
    subject, data = bus.published[0]
    # Subject built via the subjects module → the triage agent's task lane. The
    # triage id is the deployed agentd-<name>, so the subject is the lane that
    # agentd's RTECH_AGENT_ID actually subscribes on.
    assert subject == subjects.agent_task("agentd-nimbus")
    assert subject == "rtech.agent.agentd-nimbus.task"

    frame = decode_frame(data)
    assert isinstance(frame, TaskFrame)
    assert frame.id == "task-abc"
    assert frame.stack == "rtech-dev"
    assert frame.requester == "42"
    assert frame.text == "hey can you look at this"
    # Channel origin carried so the triage agent can answer in-channel.
    assert frame.origin is not None
    assert frame.origin.channel_id == "100"
    assert frame.origin.guild_id == "200"
    assert frame.origin.thread_id == "t9"
    assert frame.origin.message_id == "8080"
    assert frame.origin.asker == "42"
    # created_at is always stamped (agentd dedup/retry horizon fails closed
    # on unstamped frames).
    assert frame.created_at is not None
    # origin_edge names the triage destination.
    assert frame.origin_edge == "chatd-intake:agentd-nimbus"


def test_triage_agent_from_config_is_the_subject() -> None:
    bus = FakeBus()
    sink = _sink(bus, intake=_intake(triage_agent="agentd-birk"))
    asyncio.run(sink(_msg(mentioned_ids=frozenset({"77"}))))
    assert bus.published[0][0] == "rtech.agent.agentd-birk.task"


# --------------------------------------------------------------- debate fanout


def test_unset_debate_participants_is_byte_identical_single_scope() -> None:
    # No debate participants: the mint is EXACTLY today's single-scope publish —
    # same subject (direct to the triage lane), scope, empty participants, no
    # debate marker. The disabled path defaults IntakeConfig with no
    # debate_participants, so its published subject/scope are unchanged.
    bus = FakeBus()
    sink = _sink(bus, id_factory=lambda: "task-x")
    asyncio.run(sink(_msg(mentioned_ids=frozenset({"77"}))))
    subject, data = bus.published[-1]
    assert subject == subjects.agent_task("agentd-nimbus")
    frame = decode_frame(data)
    assert isinstance(frame, TaskFrame)
    assert frame.scope == "single"
    assert frame.participants == []
    assert frame.arbiter is None
    assert frame.origin is not None
    assert frame.origin.debate is False


def test_trio_mint_fans_via_ingress_mirroring_orchestrator() -> None:
    # A configured trio debate mints ONE trio TaskFrame published to TASK_INGRESS
    # (the orchestrator then fans it verbatim to each participant lane —
    # rtech_orchestrator _route_multi/_dispatch_batch: one ingress frame ->
    # orchestrator publishes to every agent_task(participant) lane). The sink does
    # NOT reinvent the fanout; it hands the orchestrator ONE multi-participant frame.
    bus = FakeBus()
    trio = ("agentd-nimbus", "agentd-birk", "agentd-iris")
    cfg = _intake(triage_agent="agentd-nimbus", debate_participants=trio)
    sink = _sink(bus, intake=cfg, id_factory=lambda: "deb-task")
    asyncio.run(sink(_msg(mentioned_ids=frozenset({"77"}))))

    assert len(bus.published) == 1
    subject, data = bus.published[0]
    assert subject == subjects.TASK_INGRESS
    frame = decode_frame(data)
    assert isinstance(frame, TaskFrame)
    assert frame.id == "deb-task"
    assert frame.scope == "trio"
    assert frame.participants == list(trio)  # order preserved
    # The debate mint stamps the arbiter role authority = the triage agent, and
    # it is one of the participants (TaskFrame validates that invariant).
    assert frame.arbiter == "agentd-nimbus"
    assert frame.arbiter in frame.participants
    assert frame.stack == "rtech-dev"
    assert frame.created_at is not None
    assert frame.origin is not None
    assert frame.origin.debate is True
    assert frame.origin.channel_id == "chan1"


def test_duo_mint_scope_matches_participant_count() -> None:
    bus = FakeBus()
    duo = ("agentd-nimbus", "agentd-birk")
    cfg = _intake(triage_agent="agentd-nimbus", debate_participants=duo)
    sink = _sink(bus, intake=cfg, id_factory=lambda: "duo-task")
    asyncio.run(sink(_msg(mentioned_ids=frozenset({"77"}))))
    frame = decode_frame(bus.published[0][1])
    assert isinstance(frame, TaskFrame)
    assert frame.scope == "duo"
    assert frame.participants == list(duo)
    assert frame.arbiter == "agentd-nimbus"


def test_debate_still_yields_to_responder() -> None:
    # The responder-yield precedes the mint regardless of scope: if the responder
    # answers this message, no debate frame is minted.
    bus = FakeBus()
    cfg = _intake(
        triage_agent="agentd-nimbus",
        debate_participants=("agentd-nimbus", "agentd-birk"),
    )
    sink = _sink(bus, intake=cfg, should_respond_fn=lambda _m, _c: True)
    asyncio.run(sink(_msg(mentions_me=True, mentioned_ids=frozenset({"77"}))))
    assert bus.published == []


def test_debate_deduped_by_message_id() -> None:
    # The same message id mints one debate frame within the window (redelivery).
    bus = FakeBus()
    cfg = _intake(
        triage_agent="agentd-nimbus",
        debate_participants=("agentd-nimbus", "agentd-birk", "agentd-iris"),
    )
    sink = _sink(bus, intake=cfg)
    msg = _msg(message_id="dup-deb", mentioned_ids=frozenset({"77"}))
    asyncio.run(sink(msg))
    asyncio.run(sink(msg))
    assert len(bus.published) == 1


# --------------------------------------------------------------- dedup


def test_same_message_id_twice_publishes_once() -> None:
    bus = FakeBus()
    sink = _sink(bus)
    msg = _msg(message_id="dup-1", mentioned_ids=frozenset({"77"}))
    asyncio.run(sink(msg))
    asyncio.run(sink(msg))
    assert len(bus.published) == 1


def test_distinct_message_ids_each_publish() -> None:
    bus = FakeBus()
    sink = _sink(bus)
    asyncio.run(sink(_msg(message_id="a", mentioned_ids=frozenset({"77"}))))
    asyncio.run(sink(_msg(message_id="b", mentioned_ids=frozenset({"77"}))))
    assert len(bus.published) == 2


def test_dedup_expires_after_horizon() -> None:
    # A repeat after the retention window mints again (window fully elapsed).
    ticks = iter([0.0, 10.0, 100.0])
    bus = FakeBus()
    sink = _sink(bus, clock=lambda: next(ticks), dedup_horizon_s=50.0)
    msg = _msg(message_id="m", mentioned_ids=frozenset({"77"}))
    asyncio.run(sink(msg))  # t=0 → mint
    asyncio.run(sink(msg))  # t=10 (<=50) → suppressed
    asyncio.run(sink(msg))  # t=100 (>50 since t=0) → mint again
    assert len(bus.published) == 2


def test_composite_key_same_id_different_channels_both_mint() -> None:
    # The dedup key is the COMPOSITE channel_id:message_id (built at the call
    # site). Telegram message ids are per-chat, so the same id in two different
    # channels is two distinct messages — both must mint (no cross-chat collision).
    bus = FakeBus()
    sink = _sink(bus)
    asyncio.run(sink(_msg(channel_id="chanA", message_id="1", mentioned_ids=frozenset({"77"}))))
    asyncio.run(sink(_msg(channel_id="chanB", message_id="1", mentioned_ids=frozenset({"77"}))))
    assert len(bus.published) == 2


def test_composite_key_same_channel_and_id_suppressed() -> None:
    # The same channel + id (a genuine redelivery in one chat) is suppressed.
    bus = FakeBus()
    sink = _sink(bus)
    msg = _msg(channel_id="chanA", message_id="1", mentioned_ids=frozenset({"77"}))
    asyncio.run(sink(msg))
    asyncio.run(sink(msg))
    assert len(bus.published) == 1


def test_missing_message_id_is_not_deduped() -> None:
    # A message with no id cannot be keyed; it always mints (never silently
    # collapses distinct id-less messages into one).
    bus = FakeBus()
    sink = _sink(bus)
    asyncio.run(sink(_msg(message_id=None, mentioned_ids=frozenset({"77"}))))
    asyncio.run(sink(_msg(message_id=None, mentioned_ids=frozenset({"77"}))))
    assert len(bus.published) == 2


# --------------------------------------------------------------- responder-yield


def test_direct_mention_of_triage_agent_yields_to_responder() -> None:
    # should_respond True ⇒ the responder path answers this message; intake
    # yields (no publish) regardless of mention state.
    bus = FakeBus()
    sink = _sink(bus, should_respond_fn=lambda _m, _c: True)
    msg = _msg(mentions_me=True, mentioned_ids=frozenset({"77"}))
    asyncio.run(sink(msg))
    assert bus.published == []


def test_free_response_no_mention_name_trigger_yields_to_responder() -> None:
    # The inbox is free-response: the REAL responder answers a name-trigger with
    # NO mention (should_respond step 7 waives the mention in free-response), so
    # this is a responder hit AND an intake hit. Intake must NOT also mint — the
    # exactly-one-handler property. Yielding on mentions_me alone would double-fire.
    bus = FakeBus()
    gate = GateConfig(free_response_channels=frozenset({"chan1"}))
    sink = _sink(bus, gate=gate)  # real should_respond
    msg = _msg(channel_id="chan1", mentions_me=False, mentioned_ids=frozenset({"77"}))
    asyncio.run(sink(msg))
    assert bus.published == []


def test_mention_required_channel_no_mention_intake_mints() -> None:
    # SAME message, but the channel is mention-required (not free-response) and
    # the bot is not mentioned: the responder DECLINES (step 7 mention gate), so
    # intake is the sole handler and mints.
    bus = FakeBus()
    gate = GateConfig(require_mention=True)  # no free-response channel
    sink = _sink(bus, gate=gate)  # real should_respond
    msg = _msg(channel_id="chan1", mentions_me=False, mentioned_ids=frozenset({"77"}))
    asyncio.run(sink(msg))
    assert len(bus.published) == 1


def test_allowed_dm_yields_to_responder() -> None:
    # An allowed DM is a free-response responder surface (should_respond step 4);
    # the responder answers it, so intake yields even with no explicit mention
    # flag driving the decision. Guards the DM half of the double-fire close.
    bus = FakeBus()
    gate = GateConfig(allow_dms=True)
    sink = _sink(bus, gate=gate)  # real should_respond
    msg = _msg(is_dm=True, mentions_me=True, mentioned_ids=frozenset({"77"}))
    asyncio.run(sink(msg))
    assert bus.published == []


def test_responder_declines_still_publishes() -> None:
    # should_respond False (e.g. ignored channel for the responder): the responder
    # does NOT handle it, so intake must still mint (the builder's intended case).
    bus = FakeBus()
    sink = _sink(bus, should_respond_fn=lambda _m, _c: False)
    msg = _msg(mentions_me=True, mentioned_ids=frozenset({"77"}))
    asyncio.run(sink(msg))
    assert len(bus.published) == 1


def test_default_should_respond_yields_on_mention() -> None:
    # With the REAL should_respond (default), a guild @mention of this bot is a
    # responder hit under a default GateConfig → intake yields.
    bus = FakeBus()
    sink = _sink(bus)  # real should_respond
    msg = _msg(mentions_me=True, mentioned_ids=frozenset({"77"}))
    asyncio.run(sink(msg))
    assert bus.published == []


# --------------------------------------------------------------- loud failure


def test_publish_failure_is_logged_once_and_contained() -> None:
    # The sink logs ONCE with full context then CONTAINS the failure (returns
    # normally, does not re-raise), so a real publish failure is not double-logged
    # at the adapter boundary. Nothing lands on the bus.
    bus = BoomBus()
    sink = _sink(bus)
    result = asyncio.run(sink(_msg(mentioned_ids=frozenset({"77"}))))
    assert result is None
    assert bus.published == []


# A publish exception whose message carries a credential-shaped secret: the sink
# must render the exception TYPE, never this text, into its containment log.
_PUBLISH_SECRET = "nats://user:s3cr3t-t0ken-DEADBEEF@bus.internal:4222"  # noqa: S105


class SecretBoomBus(FakeBus):
    """A bus whose publish raises with a credential-bearing message."""

    async def publish(self, subject: str, data: bytes) -> None:
        raise RuntimeError(f"connection refused for {_PUBLISH_SECRET}")


def test_publish_failure_log_omits_raw_exception_message() -> None:
    # Regression (finding #3): the publish-containment log must render the safe
    # exception TYPE and safe context, never the raw bus exception message, which
    # can carry a connection/credential secret. Containment is unchanged: the sink
    # returns and nothing lands on the bus.
    captured: list[str] = []
    sink_id = logger.add(lambda m: captured.append(m.record["message"]), level="ERROR")
    try:
        bus = SecretBoomBus()
        sink = _sink(bus)
        result = asyncio.run(sink(_msg(mentioned_ids=frozenset({"77"}))))
    finally:
        logger.remove(sink_id)
    assert result is None
    assert bus.published == []
    joined = "\n".join(captured)
    assert "publish failed" in joined
    assert "error_type=RuntimeError" in joined
    assert _PUBLISH_SECRET not in joined
    assert "s3cr3t" not in joined


# --------------------------------------------------------------- bounded memory


def _capture_warnings() -> tuple[list[str], int]:
    """Attach a WARNING-level loguru sink; return (captured messages, sink id)."""
    captured: list[str] = []
    sink_id = logger.add(lambda m: captured.append(m.record["message"]), level="WARNING")
    return captured, sink_id


def test_capacity_eviction_of_unexpired_id_warns_window_degraded() -> None:
    # at-most-once is EXACT within normal volume (the default 65536 cap sits far
    # above any channel's per-horizon count) and BOUNDED-BEST-EFFORT under an
    # id-flood: forcing a tiny cap here so capacity eviction removes an id still
    # inside the horizon. That degrades the dedup window, so the store logs a loud
    # warning naming cap + horizon, and the evicted id can re-mint inside the
    # window (its suppression record was capacity-evicted, not expired).
    captured, sink_id = _capture_warnings()
    try:
        store = _MessageIdDedup(max_entries=2, horizon_s=1e9, clock=lambda: 0.0)
        assert store.check_and_remember("a") is True
        assert store.check_and_remember("b") is True
        # "c" over capacity → evicts "a", which is NOT expired (horizon 1e9) → warn.
        assert store.check_and_remember("c") is True
        # The capacity-evicted, not-yet-expired id re-mints inside the window.
        assert store.check_and_remember("a") is True
    finally:
        logger.remove(sink_id)
    assert any("dedup window degraded" in m and "cap=2" in m for m in captured)


def test_capacity_eviction_of_expired_id_does_not_warn() -> None:
    # Normal case: the oldest over-capacity entry is already past the horizon, so
    # eviction is benign (expired-first) and no degraded-window warning fires.
    captured, sink_id = _capture_warnings()
    try:
        ticks = iter([0.0, 100.0, 200.0])
        store = _MessageIdDedup(max_entries=2, horizon_s=10.0, clock=lambda: next(ticks))
        assert store.check_and_remember("a") is True  # t=0
        assert store.check_and_remember("b") is True  # t=100
        assert store.check_and_remember("c") is True  # t=200; "a" & "b" expired
    finally:
        logger.remove(sink_id)
    assert not any("dedup window degraded" in m for m in captured)


def test_dedup_store_stays_within_capacity() -> None:
    clock = iter(float(i) for i in range(1000))
    store = _MessageIdDedup(max_entries=8, horizon_s=1e9, clock=lambda: next(clock))
    for i in range(50):
        store.check_and_remember(f"id-{i}")
    assert len(store) == 8


def test_dedup_repeat_within_window_refreshes_recency() -> None:
    # A steadily-repeated id keeps its suppression across other inserts: repeats
    # refresh recency so the id is not LRU-evicted out from under suppression.
    clock = iter(float(i) for i in range(1000))
    store = _MessageIdDedup(max_entries=2, horizon_s=1e9, clock=lambda: next(clock))
    assert store.check_and_remember("keep") is True
    assert store.check_and_remember("x") is True
    assert store.check_and_remember("keep") is False  # repeat, refreshes recency
    assert store.check_and_remember("y") is True  # evicts "x" (older than refreshed "keep")
    assert store.check_and_remember("keep") is False  # still suppressed


# --------------------------------------------------------------- instant ack (UX v2)


def test_debate_mint_posts_instant_ack_voiced_as_triage_agent() -> None:
    # A debate mint immediately posts a deterministic plain channel message voiced
    # as the triage agent, naming the other participants as drafting — the human
    # sees life within a second, long before the first engine turn.
    bus = FakeBus()
    acks: list[tuple[SendTarget, str]] = []

    async def ack_send(target: SendTarget, text: str) -> str | None:
        acks.append((target, text))
        return "ack-msg-1"

    trio = ("agentd-nimbus", "agentd-birk", "agentd-sprocket")
    cfg = _intake(triage_agent="agentd-nimbus", debate_participants=trio)
    sink = _sink(bus, intake=cfg, ack_send=ack_send)
    asyncio.run(sink(_msg(mentioned_ids=frozenset({"77"}))))

    assert len(bus.published) == 1  # the mint still published exactly once
    assert len(acks) == 1
    target, text = acks[0]
    assert target == SendTarget(channel_id="chan1", thread_id=None)  # plain, no reply
    assert text.startswith("**nimbus**")
    assert "birk" in text and "sprocket" in text
    assert "agentd-" not in text  # display names, not raw convention ids


def test_single_scope_mint_posts_no_ack() -> None:
    # No debate participants configured: the single-scope mint stays byte-identical
    # in behavior — no ack post.
    bus = FakeBus()
    acks: list[tuple[SendTarget, str]] = []

    async def ack_send(target: SendTarget, text: str) -> str | None:
        acks.append((target, text))
        return None

    sink = _sink(bus, ack_send=ack_send)
    asyncio.run(sink(_msg(mentioned_ids=frozenset({"77"}))))
    assert len(bus.published) == 1
    assert acks == []


def test_ack_failure_is_contained_and_publish_stands() -> None:
    # An ack send failure is logged + contained: the already-published mint is
    # unaffected and the sink does not raise.
    bus = FakeBus()

    async def ack_send(target: SendTarget, text: str) -> str | None:
        raise RuntimeError("ack surface exploded")

    trio = ("agentd-nimbus", "agentd-birk", "agentd-sprocket")
    cfg = _intake(triage_agent="agentd-nimbus", debate_participants=trio)
    sink = _sink(bus, intake=cfg, ack_send=ack_send)
    asyncio.run(sink(_msg(mentioned_ids=frozenset({"77"}))))  # must not raise
    assert len(bus.published) == 1


def test_ack_failure_log_omits_raw_exception_message() -> None:
    secret = "https://surface:ack-s3cr3t@chat.internal/send"  # noqa: S105
    captured, sink_id = _capture_warnings()

    async def ack_send(target: SendTarget, text: str) -> str | None:
        raise RuntimeError(f"ack rejected by {secret}")

    trio = ("agentd-nimbus", "agentd-birk", "agentd-sprocket")
    cfg = _intake(triage_agent="agentd-nimbus", debate_participants=trio)
    try:
        bus = FakeBus()
        sink = _sink(bus, intake=cfg, ack_send=ack_send, id_factory=lambda: "task-ack")
        result = asyncio.run(sink(_msg(mentioned_ids=frozenset({"77"}))))
    finally:
        logger.remove(sink_id)

    assert result is None
    assert len(bus.published) == 1
    joined = "\n".join(captured)
    assert "ack failed" in joined
    assert "channel=chan1 task_id=task-ack error_type=RuntimeError" in joined
    assert secret not in joined
    assert "ack-s3cr3t" not in joined


def test_debate_ack_lands_in_origin_thread_when_ask_was_in_thread() -> None:
    # An ask that arrived inside a thread acks back into that thread.
    bus = FakeBus()
    acks: list[tuple[SendTarget, str]] = []

    async def ack_send(target: SendTarget, text: str) -> str | None:
        acks.append((target, text))
        return None

    duo = ("agentd-nimbus", "agentd-birk")
    cfg = _intake(triage_agent="agentd-nimbus", debate_participants=duo)
    sink = _sink(bus, intake=cfg, ack_send=ack_send)
    asyncio.run(sink(_msg(thread_id="t42", mentioned_ids=frozenset({"77"}))))
    assert acks[0][0] == SendTarget(channel_id="chan1", thread_id="t42")


def test_redelivered_debate_message_acks_only_once() -> None:
    # Dedup and responder-yield short-circuit BEFORE the ack: a redelivered
    # message (same message_id) neither re-publishes nor re-acks.
    bus = FakeBus()
    acks: list[tuple[SendTarget, str]] = []

    async def ack_send(target: SendTarget, text: str) -> str | None:
        acks.append((target, text))
        return None

    trio = ("agentd-nimbus", "agentd-birk", "agentd-sprocket")
    cfg = _intake(triage_agent="agentd-nimbus", debate_participants=trio)
    sink = _sink(bus, intake=cfg, ack_send=ack_send)

    async def deliver_twice() -> None:
        msg = _msg(mentioned_ids=frozenset({"77"}))
        await sink(msg)
        await sink(msg)  # redelivery: same message_id

    asyncio.run(deliver_twice())
    assert len(bus.published) == 1
    assert len(acks) == 1


def test_publish_failure_posts_no_ack() -> None:
    # A contained publish failure must not ack: nothing was minted, so
    # announcing a debate would be a lie the human waits on forever.
    bus = BoomBus()
    acks: list[tuple[SendTarget, str]] = []

    async def ack_send(target: SendTarget, text: str) -> str | None:
        acks.append((target, text))
        return None

    trio = ("agentd-nimbus", "agentd-birk", "agentd-sprocket")
    cfg = _intake(triage_agent="agentd-nimbus", debate_participants=trio)
    sink = _sink(bus, intake=cfg, ack_send=ack_send)
    asyncio.run(sink(_msg(mentioned_ids=frozenset({"77"}))))  # contained
    assert acks == []


def test_debate_mint_starts_typing_for_every_participant() -> None:
    # a debate mint starts a typing indicator for EVERY
    # participant identity in the origin chat (targets stamped as_agent), so the
    # gap between ack and first TLDR never looks dead.
    bus = FakeBus()
    typing: list[SendTarget] = []

    async def typing_start(target: SendTarget) -> None:
        typing.append(target)

    trio = ("agentd-nimbus", "agentd-birk", "agentd-sprocket")
    cfg = _intake(triage_agent="agentd-nimbus", debate_participants=trio)
    sink = _sink(bus, intake=cfg, typing_start=typing_start)
    asyncio.run(sink(_msg(mentioned_ids=frozenset({"77"}))))

    assert len(bus.published) == 1
    assert [t.as_agent for t in typing] == list(trio)
    assert all(t.channel_id == "chan1" for t in typing)


def test_typing_start_failure_is_contained_per_participant() -> None:
    # A typing failure for one participant logs and continues: the remaining
    # participants still start, and the mint is unaffected.
    bus = FakeBus()
    typing: list[SendTarget] = []

    async def typing_start(target: SendTarget) -> None:
        if target.as_agent == "agentd-birk":
            raise RuntimeError("typing exploded")
        typing.append(target)

    trio = ("agentd-nimbus", "agentd-birk", "agentd-sprocket")
    cfg = _intake(triage_agent="agentd-nimbus", debate_participants=trio)
    sink = _sink(bus, intake=cfg, typing_start=typing_start)
    asyncio.run(sink(_msg(mentioned_ids=frozenset({"77"}))))

    assert len(bus.published) == 1
    assert [t.as_agent for t in typing] == ["agentd-nimbus", "agentd-sprocket"]


def test_typing_start_failure_log_omits_raw_exception_message() -> None:
    secret = "bot-token:typing-s3cr3t-DEADBEEF"  # noqa: S105
    captured, sink_id = _capture_warnings()
    typing: list[SendTarget] = []

    async def typing_start(target: SendTarget) -> None:
        if target.as_agent == "agentd-birk":
            raise RuntimeError(f"typing rejected with {secret}")
        typing.append(target)

    trio = ("agentd-nimbus", "agentd-birk", "agentd-sprocket")
    cfg = _intake(triage_agent="agentd-nimbus", debate_participants=trio)
    try:
        bus = FakeBus()
        sink = _sink(
            bus,
            intake=cfg,
            typing_start=typing_start,
            id_factory=lambda: "task-typing",
        )
        result = asyncio.run(sink(_msg(mentioned_ids=frozenset({"77"}))))
    finally:
        logger.remove(sink_id)

    assert result is None
    assert len(bus.published) == 1
    assert [target.as_agent for target in typing] == ["agentd-nimbus", "agentd-sprocket"]
    joined = "\n".join(captured)
    assert "typing start failed" in joined
    assert "agent=agentd-birk task_id=task-typing error_type=RuntimeError" in joined
    assert secret not in joined
    assert "typing-s3cr3t" not in joined


def test_single_scope_mint_starts_no_typing() -> None:
    # No debate participants: single-scope behavior stays byte-identical.
    bus = FakeBus()
    typing: list[SendTarget] = []

    async def typing_start(target: SendTarget) -> None:
        typing.append(target)

    sink = _sink(bus, typing_start=typing_start)
    asyncio.run(sink(_msg(mentioned_ids=frozenset({"77"}))))

    assert len(bus.published) == 1
    assert typing == []


def test_unvoiced_ack_is_identity_stamped_and_unprefixed() -> None:
    # voiced=False sends the ack under the triage agent's OWN bot —
    # the target carries as_agent and the text drops the name prefix.
    bus = FakeBus()
    acks: list[tuple[SendTarget, str]] = []

    async def ack_send(target: SendTarget, text: str) -> str | None:
        acks.append((target, text))
        return "ack-msg-1"

    trio = ("agentd-nimbus", "agentd-birk", "agentd-sprocket")
    cfg = _intake(triage_agent="agentd-nimbus", debate_participants=trio)
    sink = _sink(bus, intake=cfg, ack_send=ack_send, voiced=False)
    asyncio.run(sink(_msg(mentioned_ids=frozenset({"77"}))))

    assert len(acks) == 1
    target, text = acks[0]
    assert target.as_agent == "agentd-nimbus"
    assert not text.startswith("**")
    assert text.startswith("On it.")
    assert "birk" in text and "sprocket" in text
    assert "agentd-" not in text
