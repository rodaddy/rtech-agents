"""Exercise the Telegram E2E harness without live credentials or network access.

These mocked tests pin the CORRELATED-observer contract the live suite depends
on, and fail closed on the exact regressions this harness corrected:

- a subscription installed AFTER the action (a plain post-action pub/sub can miss
  the frames the action causes) — :func:`test_observer_subscribes_before_action`;
- an invented ``rtech.task.<id>.frames`` subject or a predicted task id instead of
  discovery from the minted ``TaskFrame.origin.message_id`` —
  :func:`test_task_id_is_discovered_not_invented`;
- a fake dedup that replays two DISTINCT inbound messages rather than the SAME
  terminal ``TurnFrame`` on the real per-task turn subject —
  :func:`test_dedup_replays_same_terminal_turn_not_two_messages`;
- a scenario that never reads the Telegram surface —
  :func:`test_live_actor_reads_both_surfaces`;
- a scenario that never cleans up (unsubscribe + bus close) —
  :func:`test_observer_close_unsubscribes_and_closes_bus`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from loguru import logger
from rtech_chatd import subjects
from rtech_chatd.tg_e2e import (
    FrameCapture,
    LiveE2EConfig,
    PollPolicy,
    PollTimeoutError,
    RunMarker,
    SurfaceMessage,
    TelegramAccountConfig,
    TelegramGroupConfig,
    drive_action,
    live_skip_reason,
    poll_frames,
    scenario_id,
    task_id,
)
from rtech_chatd.tg_e2e_live import (
    API_HASH_ENV,
    API_ID_ENV,
    FORUM_GROUP_ID_ENV,
    NON_FORUM_GROUP_ID_ENV,
    CorrelatedObserver,
    LiveTargets,
    LiveTelethonActor,
    TelethonClientWrapper,
    connected_telethon_actor,
    deferred_telethon_client,
    resolve_live_targets,
    resolve_session_material,
)
from rtech_frames import ChannelOrigin, Stance, SteerFrame, TaskFrame, TurnFrame
from rtech_frames.codec import encode_frame

LIVE_OPT_IN_ENV = "RTECH_TG_E2E_LIVE"
GROUP_REF_ENV = "RTECH_TG_E2E_GROUP_REF"
NATS_URL_REF_ENV = "RTECH_TG_E2E_NATS_URL_REF"


def _live_env() -> dict[str, str]:
    return {
        LIVE_OPT_IN_ENV: "true",
        GROUP_REF_ENV: "telegram-e2e-group-ref",
        NATS_URL_REF_ENV: "telegram-e2e-nats-url-ref",
        "RTECH_TG_E2E_ARBITER_SESSION_REF": "telegram-e2e-arbiter-session-ref",
        "RTECH_TG_E2E_ARBITER_USERNAME": "arbiter",
        "RTECH_TG_E2E_ARBITER_IDENTITY_BOT_SENDER_ID": "101",
        "RTECH_TG_E2E_ARBITER_TOPIC_REF": "telegram-e2e-arbiter-topic-ref",
        "RTECH_TG_E2E_PARTICIPANT_ONE_SESSION_REF": "telegram-e2e-one-session-ref",
        "RTECH_TG_E2E_PARTICIPANT_ONE_USERNAME": "one",
        "RTECH_TG_E2E_PARTICIPANT_ONE_IDENTITY_BOT_SENDER_ID": "102",
        "RTECH_TG_E2E_PARTICIPANT_ONE_TOPIC_REF": "telegram-e2e-one-topic-ref",
        "RTECH_TG_E2E_PARTICIPANT_TWO_SESSION_REF": "telegram-e2e-two-session-ref",
        "RTECH_TG_E2E_PARTICIPANT_TWO_USERNAME": "two",
        "RTECH_TG_E2E_PARTICIPANT_TWO_IDENTITY_BOT_SENDER_ID": "103",
        "RTECH_TG_E2E_PARTICIPANT_TWO_TOPIC_REF": "telegram-e2e-two-topic-ref",
    }


_OPAQUE_TASK_ID = "3f2b9c1a-4d5e-4f60-8a71-9b0c1d2e3f40"


def _task(
    task_id_value: str, *, message_id: str | None = None, text: str = "arbitrate"
) -> TaskFrame:
    origin = None if message_id is None else ChannelOrigin(channel_id="c1", message_id=message_id)
    return TaskFrame(
        id=task_id_value,
        origin_edge="telegram",
        requester="tester",
        text=text,
        scope="trio",
        participants=["agentd-arbiter", "agentd-one", "agentd-two"],
        arbiter="agentd-arbiter",
        origin=origin,
    )


def _turn(
    task_id_value: str, *, agent_id: str = "agentd-arbiter", text: str = "acknowledged"
) -> TurnFrame:
    return TurnFrame(
        agent_id=agent_id,
        task_id=task_id_value,
        visible_text=text,
        stance=Stance(position="neutral", because="canary", needs_from=[]),
    )


def _steer(task_id_value: str, *, target_agent: str = "agentd-arbiter") -> SteerFrame:
    return SteerFrame(task_id=task_id_value, text="steer via reaction", target_agent=target_agent)


class _FakeSub:
    def __init__(self, log: list[str], subject: str, *, fail: bool = False) -> None:
        self._log = log
        self._subject = subject
        self._fail = fail

    async def unsubscribe(self) -> None:
        self._log.append(f"unsub:{self._subject}")
        if self._fail:
            raise RuntimeError(f"unsubscribe failed for {self._subject}")


class _ScriptedBus:
    """A fake bus that records subscribe/publish/close order and emits given frames.

    ``emit`` is called with each subscribed subject's handler at subscribe time so
    a test can drive which subjects see which frames — this lets a test PROVE the
    ingress/turn/steer subscriptions are installed BEFORE any action publishes.
    """

    def __init__(
        self,
        emit: dict[str, Sequence[tuple[str, object]]] | None = None,
        *,
        fail_subscribe: str | None = None,
        fail_flush: bool = False,
        failing_unsubscribe: str | None = None,
    ) -> None:
        self._emit = emit or {}
        self._fail_subscribe = fail_subscribe
        self._fail_flush = fail_flush
        self._failing_unsubscribe = failing_unsubscribe
        self.log: list[str] = []
        self.subscribed: list[str] = []
        self.published: list[tuple[str, bytes]] = []
        self.closed = False
        self._handlers: dict[str, object] = {}

    async def subscribe(self, subject: str, handler: object) -> _FakeSub:
        if subject == self._fail_subscribe:
            raise RuntimeError(f"subscribe failed for {subject}")
        self.subscribed.append(subject)
        self.log.append(f"sub:{subject}")
        self._handlers[subject] = handler
        for actual_subject, frame in self._emit.get(subject, ()):  # replay seeded frames
            await handler(actual_subject, encode_frame(frame))  # type: ignore[operator]
        return _FakeSub(self.log, subject, fail=subject == self._failing_unsubscribe)

    async def flush(self) -> None:
        self.log.append("flush")
        if self._fail_flush:
            raise RuntimeError("flush failed")

    async def publish(self, subject: str, data: bytes) -> None:
        self.published.append((subject, data))
        self.log.append(f"pub:{subject}")
        handler = self._handlers.get(subject)
        if handler is not None:
            await handler(data)  # type: ignore[operator]

    async def close(self) -> None:
        self.closed = True
        self.log.append("close")


def _connector_for(bus: object) -> object:
    async def connect(_url: str) -> object:
        return bus

    return connect


# Reference-only config gating (unchanged contract).


def test_config_requires_explicit_opt_in_and_all_reference_settings() -> None:
    with pytest.raises(ValueError, match=LIVE_OPT_IN_ENV):
        LiveE2EConfig.from_env({})
    env = _live_env()
    del env[NATS_URL_REF_ENV]
    with pytest.raises(ValueError, match="nats_url_ref"):
        LiveE2EConfig.from_env(env)


def test_config_exposes_three_identities_and_topics() -> None:
    config = LiveE2EConfig.from_env(_live_env())
    assert [account.role for account in config.accounts] == [
        "arbiter",
        "participant_one",
        "participant_two",
    ]
    assert len(config.group.topic_refs) == 3


def test_zero_network_skip_gating_happens_before_client_construction() -> None:
    assert live_skip_reason({}) == f"{LIVE_OPT_IN_ENV} is not explicitly enabled"
    constructed = False

    def factory(session_string: str, api_id: int, api_hash: str) -> object:
        nonlocal constructed
        constructed = True
        return object()

    wrapper = TelethonClientWrapper(
        TelegramAccountConfig("session-ref", "arbiter", "arbiter", 101), factory
    )
    assert constructed is False
    assert wrapper.account.role == "arbiter"


def test_account_repr_redacts_session_reference() -> None:
    account = TelegramAccountConfig("sensitive-reference-name", "arbiter", "arbiter", 101)
    rendered = repr(account)
    assert "sensitive-reference-name" not in rendered
    assert "<redacted>" in rendered


def test_config_rejects_duplicate_usernames() -> None:
    env = _live_env()
    env["RTECH_TG_E2E_PARTICIPANT_ONE_USERNAME"] = "arbiter"
    with pytest.raises(ValueError, match="usernames must be distinct"):
        LiveE2EConfig.from_env(env)


def test_config_rejects_duplicate_session_refs() -> None:
    env = _live_env()
    env["RTECH_TG_E2E_PARTICIPANT_TWO_SESSION_REF"] = "telegram-e2e-arbiter-session-ref"
    with pytest.raises(ValueError, match="session_string_refs must be distinct"):
        LiveE2EConfig.from_env(env)


def test_config_rejects_zero_or_duplicate_identity_bot_sender_ids() -> None:
    zero = _live_env()
    zero["RTECH_TG_E2E_ARBITER_IDENTITY_BOT_SENDER_ID"] = "0"
    with pytest.raises(ValueError, match="non-zero"):
        LiveE2EConfig.from_env(zero)
    duplicate = _live_env()
    duplicate["RTECH_TG_E2E_PARTICIPANT_TWO_IDENTITY_BOT_SENDER_ID"] = "101"
    with pytest.raises(ValueError, match="sender_ids must be distinct"):
        LiveE2EConfig.from_env(duplicate)


def test_group_rejects_duplicate_topic_refs() -> None:
    with pytest.raises(ValueError, match="topic_refs must be distinct"):
        TelegramGroupConfig(group_ref="group", topic_refs=("t", "t", "u"))


# Marker/id helpers: a marker is TEXT-only, it never invents a subject or id.


def test_run_marker_is_text_only_and_invents_no_subject_or_task_id() -> None:
    one = RunMarker("arbitration-ack", "nonce-a")
    two = RunMarker("arbitration-ack", "nonce-b")
    assert one.marker != two.marker
    assert one.marker in one.trigger_text("body")
    assert one.matches(one.trigger_text("body"))
    # A RunMarker exposes ONLY a text marker — no invented .subject / .task_id that
    # would let a scenario predict the opaque product id or a frames subject.
    assert not hasattr(one, "subject")
    assert not hasattr(one, "task_id")


def test_scenario_and_task_ids_are_deterministic() -> None:
    assert scenario_id("arbitration ack") == scenario_id("arbitration ack")
    assert task_id("arbitration ack", 3) == task_id("arbitration ack", 3)
    assert task_id("arbitration ack", 3) != task_id("arbitration ack", 4)


# FrameCapture: typed task/turn/steer capture and assertion.


def test_frame_capture_asserts_typed_task_turn_and_steer_frames() -> None:
    capture = FrameCapture()
    task = _task(_OPAQUE_TASK_ID)
    turn = _turn(_OPAQUE_TASK_ID)
    steer = _steer(_OPAQUE_TASK_ID)
    capture.capture(task, subjects.TASK_INGRESS)
    capture.capture(turn, subjects.task_turn(_OPAQUE_TASK_ID))
    capture.capture(steer, subjects.task_steer(_OPAQUE_TASK_ID))

    assert (
        capture.assert_task(
            task_id_value=_OPAQUE_TASK_ID,
            participants=("agentd-arbiter", "agentd-one", "agentd-two"),
            arbiter="agentd-arbiter",
        )
        is task
    )
    assert capture.assert_turn(task_id_value=_OPAQUE_TASK_ID, agent_id="agentd-arbiter") is turn
    assert (
        capture.assert_steer(task_id_value=_OPAQUE_TASK_ID, target_agent="agentd-arbiter") is steer
    )


def test_frame_capture_rejects_untyped_observations() -> None:
    capture = FrameCapture()
    with pytest.raises(TypeError, match="TaskFrame, TurnFrame, or SteerFrame"):
        capture.capture(object(), subjects.TASK_INGRESS)  # type: ignore[arg-type]


async def test_poll_frames_times_out_at_bounded_attempt_count() -> None:
    reads = 0
    sleeps: list[float] = []

    async def read_frames() -> Sequence[TaskFrame | TurnFrame]:
        nonlocal reads
        reads += 1
        return ()

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    with pytest.raises(PollTimeoutError, match="3 attempts"):
        await poll_frames(
            read_frames,
            lambda _: False,
            policy=PollPolicy(attempts=3, interval_s=0.25),
            sleep=fake_sleep,
        )
    assert reads == 3
    assert sleeps == [0.25, 0.25]


# Resolved material / targets (live-run env resolution).


def test_resolve_session_material_requires_complete_material() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        resolve_session_material("arbiter", {})
    material = resolve_session_material(
        "arbiter",
        {
            "RTECH_TG_E2E_ARBITER_SESSION_STRING": "session-value",
            API_ID_ENV: "42",
            API_HASH_ENV: "hash-value",
        },
    )
    assert material.api_id == 42
    assert "session-value" not in repr(material)
    assert "hash-value" not in repr(material)


def test_deferred_client_construction_uses_resolved_material() -> None:
    seen: dict[str, object] = {}

    def factory(session_string: str, api_id: int, api_hash: str) -> object:
        seen["session_string"] = session_string
        seen["api_id"] = api_id
        return object()

    account = TelegramAccountConfig("session-ref", "arbiter", "arbiter", 101)
    client = deferred_telethon_client(
        account,
        factory=factory,
        env={
            "RTECH_TG_E2E_ARBITER_SESSION_STRING": "session-value",
            API_ID_ENV: "7",
            API_HASH_ENV: "hash-value",
        },
    )
    assert client is not None
    assert seen == {"session_string": "session-value", "api_id": 7}


def test_resolve_live_targets_requires_distinct_forum_and_non_forum() -> None:
    env = {
        FORUM_GROUP_ID_ENV: "1001",
        NON_FORUM_GROUP_ID_ENV: "1001",
        "RTECH_TG_E2E_ARBITER_TOPIC_ID": "10",
        "RTECH_TG_E2E_PARTICIPANT_ONE_TOPIC_ID": "20",
        "RTECH_TG_E2E_PARTICIPANT_TWO_TOPIC_ID": "30",
    }
    with pytest.raises(ValueError, match="DISTINCT group"):
        resolve_live_targets(env)


def test_resolve_live_targets_fails_closed_on_missing_id() -> None:
    with pytest.raises(ValueError, match="must be resolved"):
        resolve_live_targets({})


def test_resolve_live_targets_resolves_all_targets() -> None:
    env = {
        FORUM_GROUP_ID_ENV: "1001",
        NON_FORUM_GROUP_ID_ENV: "2002",
        "RTECH_TG_E2E_ARBITER_TOPIC_ID": "10",
        "RTECH_TG_E2E_PARTICIPANT_ONE_TOPIC_ID": "20",
        "RTECH_TG_E2E_PARTICIPANT_TWO_TOPIC_ID": "30",
    }
    targets: LiveTargets = resolve_live_targets(env)
    assert targets.forum_group_id == 1001
    assert targets.non_forum_group_id == 2002
    assert targets.topic_ids["arbiter"] == 10


# LiveTelethonActor: forwards to the real client AND reads BOTH surfaces.


async def test_connected_actor_wraps_deferred_client_and_returns_real_sent_id() -> None:
    class _FakeSent:
        id = 5150

    class _FakeClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def connect(self) -> None:
            self.calls.append("connect")

        async def disconnect(self) -> None:
            self.calls.append("disconnect")

        async def send_message(
            self, chat_id: int, text: str, *, reply_to: int | None = None
        ) -> _FakeSent:
            self.calls.append(f"send:{chat_id}:{reply_to}")
            return _FakeSent()

    fake = _FakeClient()

    def factory(session_string: str, api_id: int, api_hash: str) -> object:
        return fake

    account = TelegramAccountConfig("session-ref", "arbiter", "arbiter", 101)
    actor = await connected_telethon_actor(
        account,
        factory=factory,
        env={
            "RTECH_TG_E2E_ARBITER_SESSION_STRING": "session-value",
            API_ID_ENV: "7",
            API_HASH_ENV: "hash-value",
        },
    )
    assert isinstance(actor, LiveTelethonActor)
    await actor.connect()
    sent_id = await actor.send_message(99, "hi", reply_to=5)
    await actor.disconnect()
    # The adapter forwards connect/send/disconnect AND returns the REAL sent id so
    # the observer can correlate the minted TaskFrame by origin.message_id.
    assert fake.calls == ["connect", "send:99:5", "disconnect"]
    assert sent_id == 5150


async def test_live_actor_reads_both_surfaces() -> None:
    class _Msg:
        def __init__(self, mid: int, sender: int | None, top: int | None, text: str) -> None:
            self.id = mid
            self.sender_id = sender
            self.message = text
            self.reply_to = type("R", (), {"reply_to_top_id": top})() if top is not None else None

    class _FakeClient:
        async def get_messages(self, chat_id: int, *, limit: int = 50) -> list[object]:
            return [_Msg(1, 900, 77, "full text under topic"), _Msg(2, None, None, "no sender")]

    actor = LiveTelethonActor(_FakeClient())
    surface = await actor.read_messages(4004, limit=50)
    # Reads the ACTUAL Telegram surface into typed rows: sender/topic/text.
    assert surface[0] == SurfaceMessage(
        message_id=1, sender_id=900, topic_id=77, text="full text under topic"
    )
    assert surface[1].sender_id is None and surface[1].topic_id is None


# CorrelatedObserver: subscribe-before-action, discovery, dedup, cleanup, steer.


async def test_observer_subscribes_before_action() -> None:
    """Open installs the ingress + turn + steer subscriptions BEFORE any action.

    Regression guard: a post-action subscription would miss the frames the action
    caused. The observer subscribes exactly the REAL product subjects up front and
    NEVER an invented ``rtech.task.<id>.frames`` subject.
    """
    bus = _ScriptedBus()
    observer = CorrelatedObserver("nats://ignored", connect=_connector_for(bus))
    await observer.open()
    # Exactly the three real product subjects, all installed before any publish.
    assert bus.subscribed == [
        subjects.TASK_INGRESS,
        subjects.TASK_TURN_WILDCARD,
        subjects.TASK_STEER_WILDCARD,
    ]
    assert not any(entry.startswith("pub:") for entry in bus.log[: len(bus.subscribed)])
    assert not any(".frames" in subject for subject in bus.subscribed)
    assert bus.log[-1] == "flush"
    await observer.close()


async def test_task_id_is_discovered_not_invented() -> None:
    """The opaque task id is DISCOVERED from the minted TaskFrame.origin.message_id.

    A scenario cannot predict it. The observer subscribes ingress up front; when
    the minted TaskFrame arrives carrying the real sent message id, discovery
    returns the OPAQUE product id — not any value the test chose in advance.
    """
    mark = RunMarker("arbitration-ack", "nonce")
    sent_id = 24680
    minted = _task(_OPAQUE_TASK_ID, message_id=str(sent_id), text=mark.trigger_text("go"))
    bus = _ScriptedBus(emit={subjects.TASK_INGRESS: [(subjects.TASK_INGRESS, minted)]})
    observer = CorrelatedObserver("nats://ignored", connect=_connector_for(bus))
    await observer.open()
    try:
        discovered = await observer.discover_task_id(
            sent_id, marker=mark.marker, policy=PollPolicy(attempts=2, interval_s=0.0)
        )
        assert discovered == _OPAQUE_TASK_ID
        # A wrong (unminted) message id must NOT resolve to any task.
        with pytest.raises(PollTimeoutError):
            await observer.discover_task_id(
                99999, marker=mark.marker, policy=PollPolicy(attempts=1, interval_s=0.0)
            )
    finally:
        await observer.close()


async def test_discover_rejects_frame_without_matching_marker() -> None:
    """Discovery fails closed when the marker does not match the minted frame text."""
    minted = _task(_OPAQUE_TASK_ID, message_id="777", text="some other run [x]")
    bus = _ScriptedBus(emit={subjects.TASK_INGRESS: [(subjects.TASK_INGRESS, minted)]})
    observer = CorrelatedObserver("nats://ignored", connect=_connector_for(bus))
    await observer.open()
    try:
        with pytest.raises(PollTimeoutError):
            await observer.discover_task_id(
                777, marker="[not-this-run]", policy=PollPolicy(attempts=1, interval_s=0.0)
            )
    finally:
        await observer.close()


async def test_turns_and_steers_are_filtered_by_discovered_id() -> None:
    """Turn/steer wildcards are filtered by the DISCOVERED opaque id, not a guess."""
    other = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    bus = _ScriptedBus(
        emit={
            subjects.TASK_TURN_WILDCARD: [
                (
                    subjects.task_turn(_OPAQUE_TASK_ID),
                    _turn(_OPAQUE_TASK_ID, agent_id="agentd-one"),
                ),
                (subjects.task_turn(other), _turn(other, agent_id="agentd-one")),
                (subjects.task_turn(other), _turn(_OPAQUE_TASK_ID, agent_id="spoofed")),
                (
                    subjects.task_turn(_OPAQUE_TASK_ID),
                    _turn(_OPAQUE_TASK_ID, agent_id="agentd-one", text="   "),
                ),
            ],
            subjects.TASK_STEER_WILDCARD: [
                (subjects.task_steer(_OPAQUE_TASK_ID), _steer(_OPAQUE_TASK_ID)),
                (subjects.task_steer(other), _steer(other)),
                (subjects.task_steer(other), _steer(_OPAQUE_TASK_ID)),
            ],
        }
    )
    observer = CorrelatedObserver("nats://ignored", connect=_connector_for(bus))
    await observer.open()
    try:
        turns = observer.turns_for(_OPAQUE_TASK_ID)
        assert [t.agent_id for t in turns] == ["agentd-one"]  # blank + other-task excluded
        steers = observer.steers_for(_OPAQUE_TASK_ID)
        assert len(steers) == 1 and steers[0].task_id == _OPAQUE_TASK_ID
    finally:
        await observer.close()


async def test_dedup_replays_same_terminal_turn_not_two_messages() -> None:
    """Dedup replays the SAME terminal TurnFrame on the REAL per-task turn subject.

    Regression guard: a fake dedup that sent two DISTINCT inbound Telegram messages
    would prove nothing. The real boundary is a byte-identical replay of the
    terminal ``TurnFrame`` onto ``rtech.task.<id>.turn`` — this test proves the
    replay targets that ACTUAL subject with the SAME frame bytes.
    """
    terminal = _turn(_OPAQUE_TASK_ID, agent_id="agentd-one", text="final answer")
    bus = _ScriptedBus()
    observer = CorrelatedObserver("nats://ignored", connect=_connector_for(bus))
    await observer.open()
    try:
        turn_subject = subjects.task_turn(_OPAQUE_TASK_ID)
        await observer.publish(turn_subject, terminal)
        await observer.publish(turn_subject, terminal)
        # Both replays go to the ACTUAL per-task turn subject, byte-identical.
        assert [subject for subject, _ in bus.published] == [turn_subject, turn_subject]
        assert bus.published[0][1] == bus.published[1][1] == encode_frame(terminal)
        # And they are NOT an invented frames subject nor two distinct messages.
        assert all(".frames" not in subject for subject, _ in bus.published)
    finally:
        await observer.close()


async def test_publish_requires_open_bus() -> None:
    observer = CorrelatedObserver("nats://ignored", connect=_connector_for(_ScriptedBus()))
    with pytest.raises(RuntimeError, match="not open"):
        await observer.publish(subjects.task_turn(_OPAQUE_TASK_ID), _turn(_OPAQUE_TASK_ID))


async def test_observer_close_unsubscribes_and_closes_bus() -> None:
    """Cleanup unsubscribes EVERY subscription and closes the bus (finally path)."""
    bus = _ScriptedBus()
    observer = CorrelatedObserver("nats://ignored", connect=_connector_for(bus))
    await observer.open()
    await observer.close()
    # Every installed subscription is unsubscribed and the bus is closed.
    assert bus.log.count("unsub:" + subjects.TASK_INGRESS) == 1
    assert bus.log.count("unsub:" + subjects.TASK_TURN_WILDCARD) == 1
    assert bus.log.count("unsub:" + subjects.TASK_STEER_WILDCARD) == 1
    assert bus.closed is True
    assert bus.log[-1] == "close"


async def test_open_rolls_back_partial_subscriptions_and_closes_bus() -> None:
    """A setup failure removes earlier subscriptions and closes the connected bus."""
    bus = _ScriptedBus(fail_subscribe=subjects.TASK_TURN_WILDCARD)
    observer = CorrelatedObserver("nats://ignored", connect=_connector_for(bus))
    with pytest.raises(RuntimeError, match="subscribe failed"):
        await observer.open()
    assert bus.log == [
        f"sub:{subjects.TASK_INGRESS}",
        f"unsub:{subjects.TASK_INGRESS}",
        "close",
    ]
    assert bus.closed is True


async def test_open_rolls_back_all_subscriptions_when_flush_fails() -> None:
    """The server-readiness flush is part of setup and rolls back on failure."""
    bus = _ScriptedBus(fail_flush=True)
    observer = CorrelatedObserver("nats://ignored", connect=_connector_for(bus))
    with pytest.raises(RuntimeError, match="flush failed"):
        await observer.open()
    assert bus.log.count("unsub:" + subjects.TASK_INGRESS) == 1
    assert bus.log.count("unsub:" + subjects.TASK_TURN_WILDCARD) == 1
    assert bus.log.count("unsub:" + subjects.TASK_STEER_WILDCARD) == 1
    assert bus.closed is True


async def test_drive_action_disconnects_after_connect_failure() -> None:
    """Actor cleanup runs even when connecting the Telegram action fails."""
    calls: list[str] = []

    class _FailingActor:
        async def connect(self) -> None:
            calls.append("connect")
            raise RuntimeError("connect failed")

        async def disconnect(self) -> None:
            calls.append("disconnect")

    async def _action(_actor: object) -> None:
        raise AssertionError("action must not run")

    with pytest.raises(RuntimeError, match="connect failed"):
        await drive_action(_FailingActor(), _action)  # type: ignore[arg-type]
    assert calls == ["connect", "disconnect"]


async def test_expect_no_task_passes_when_absent_and_fails_when_present() -> None:
    """expect_no_task proves ABSENCE and fails closed when a task is minted."""
    absent_bus = _ScriptedBus()
    observer = CorrelatedObserver("nats://ignored", connect=_connector_for(absent_bus))
    await observer.open()
    try:
        # No minted frame for this id within the window -> passes.
        await observer.expect_no_task(
            555, marker="[none]", policy=PollPolicy(attempts=2, interval_s=0.0)
        )
    finally:
        await observer.close()

    mark = RunMarker("no-trigger", "nonce")
    present = _task(_OPAQUE_TASK_ID, message_id="555", text=mark.trigger_text("oops"))
    present_bus = _ScriptedBus(emit={subjects.TASK_INGRESS: [(subjects.TASK_INGRESS, present)]})
    observer = CorrelatedObserver("nats://ignored", connect=_connector_for(present_bus))
    await observer.open()
    try:
        with pytest.raises(AssertionError, match="unexpectedly minted"):
            await observer.expect_no_task(
                555, marker=mark.marker, policy=PollPolicy(attempts=1, interval_s=0.0)
            )
    finally:
        await observer.close()


async def test_wait_task_times_out_when_expectation_unmet() -> None:
    bus = _ScriptedBus()
    observer = CorrelatedObserver("nats://ignored", connect=_connector_for(bus))
    await observer.open()
    try:
        with pytest.raises(PollTimeoutError, match="2 attempts"):
            await observer.wait_task(
                lambda _cap: False, policy=PollPolicy(attempts=2, interval_s=0.0)
            )
    finally:
        await observer.close()


async def test_connect_frame_bus_closes_partial_bus_when_connect_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connect() that opens resources then raises must not leak a half-open bus.

    The caller has no handle to a bus that failed mid-connect, so _connect_frame_bus
    must close the partially initialized local bus before re-raising.
    """
    from rtech_chatd import tg_e2e

    closed = False
    connect_exc = RuntimeError("resources opened then connect failed")
    credential = "super-secret-token"
    rendered_logs: list[str] = []
    sink_id = logger.add(lambda message: rendered_logs.append(str(message)), level="WARNING")

    class _PartialBus:
        def __init__(self, _url: str) -> None:
            pass

        async def connect(self) -> None:
            raise connect_exc

        async def close(self) -> None:
            nonlocal closed
            closed = True
            raise RuntimeError(f"nats://user:{credential}@nats.internal:4222")

    monkeypatch.setattr(tg_e2e, "_NatsObservableBus", _PartialBus)
    try:
        with pytest.raises(RuntimeError, match="connect failed") as raised:
            await tg_e2e._connect_frame_bus("nats://ignored")
    finally:
        logger.remove(sink_id)
    assert closed is True, "partially initialized bus was not closed after a failed connect"
    assert raised.value is connect_exc
    assert rendered_logs
    assert credential not in "".join(rendered_logs)


async def test_connect_frame_bus_closes_after_connect_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled connect must close its partial bus and preserve cancellation."""
    from rtech_chatd import tg_e2e

    closed = False
    connect_exc = asyncio.CancelledError()

    class _PartialBus:
        def __init__(self, _url: str) -> None:
            pass

        async def connect(self) -> None:
            raise connect_exc

        async def close(self) -> None:
            nonlocal closed
            closed = True

    monkeypatch.setattr(tg_e2e, "_NatsObservableBus", _PartialBus)
    with pytest.raises(asyncio.CancelledError) as raised:
        await tg_e2e._connect_frame_bus("nats://ignored")

    assert closed is True, "partially initialized bus was not closed after cancellation"
    assert raised.value is connect_exc


async def test_connect_frame_bus_preserves_cancellation_when_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cleanup failure must be type-logged without masking connect cancellation."""
    from rtech_chatd import tg_e2e

    closed = False
    connect_exc = asyncio.CancelledError()
    credential = "super-secret-token"
    close_message = f"nats://user:{credential}@nats.internal:4222"
    rendered_logs: list[str] = []
    sink_id = logger.add(lambda message: rendered_logs.append(str(message)), level="WARNING")

    class _PartialBus:
        def __init__(self, _url: str) -> None:
            pass

        async def connect(self) -> None:
            raise connect_exc

        async def close(self) -> None:
            nonlocal closed
            closed = True
            raise RuntimeError(close_message)

    monkeypatch.setattr(tg_e2e, "_NatsObservableBus", _PartialBus)
    try:
        with pytest.raises(asyncio.CancelledError) as raised:
            await tg_e2e._connect_frame_bus("nats://ignored")
    finally:
        logger.remove(sink_id)

    assert closed is True, "partially initialized bus was not closed after cancellation"
    assert raised.value is connect_exc
    logs = "".join(rendered_logs)
    assert "close_error_type=RuntimeError" in logs
    assert close_message not in logs
    assert credential not in logs


async def test_connect_frame_bus_preserves_connect_failure_when_close_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled cleanup must be type-logged without masking connect failure."""
    from rtech_chatd import tg_e2e

    connect_exc = RuntimeError("connect failed")
    rendered_logs: list[str] = []
    sink_id = logger.add(lambda message: rendered_logs.append(str(message)), level="WARNING")

    class _PartialBus:
        def __init__(self, _url: str) -> None:
            pass

        async def connect(self) -> None:
            raise connect_exc

        async def close(self) -> None:
            raise asyncio.CancelledError("private close failure")

    monkeypatch.setattr(tg_e2e, "_NatsObservableBus", _PartialBus)
    try:
        with pytest.raises(RuntimeError, match="connect failed") as raised:
            await tg_e2e._connect_frame_bus("nats://ignored")
    finally:
        logger.remove(sink_id)

    assert raised.value is connect_exc
    logs = "".join(rendered_logs)
    assert "close_error_type=CancelledError" in logs
    assert "private close failure" not in logs


def test_non_forum_capability_error_matches_only_specific_rejection() -> None:
    """Only the specific Telethon CHANNEL_FORUM_MISSING rejection is proof of no topics."""
    from rtech_chatd.tg_e2e_live import is_non_forum_capability_error
    from telethon.errors import ChannelForumMissingError

    assert is_non_forum_capability_error(ChannelForumMissingError(request=None)) is True
    # An arbitrary exception whose text merely contains "forum" is NOT proof.
    assert is_non_forum_capability_error(RuntimeError("forum topics timed out")) is False
    assert is_non_forum_capability_error(TimeoutError("forum")) is False
