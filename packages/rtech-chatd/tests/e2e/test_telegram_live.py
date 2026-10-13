"""Drive authorization-gated real-account Telegram canary scenarios.

Collection fails closed: without explicit opt-in AND complete reference-only
config the whole module skips BEFORE any Telethon import or network object exists
(see :func:`live_skip_reason`). When authorized, every scenario:

1. Opens a :class:`CorrelatedObserver` FIRST — it subscribes the REAL product
   subjects (``TASK_INGRESS``, the per-task turn wildcard, and the per-task steer
   wildcard) and confirms them with a server flush BEFORE any Telegram action, so
   a plain NATS pub/sub cannot miss a frame the action causes.
2. Connects the intended real account and performs the actual Telegram action;
   :meth:`send_message` returns the REAL sent message id.
3. Discovers the OPAQUE product task id from the minted ``TaskFrame`` on
   ``TASK_INGRESS`` by ``origin.message_id`` == that sent id — a scenario cannot
   predict the id or invent a ``rtech.task.<id>.frames`` subject.
4. Filters ``TurnFrame`` / ``SteerFrame`` from the wildcards by the discovered id.
5. Asserts BOTH surfaces: the typed bus frames AND the actual Telegram surface
   (messages/topics read back with the actor), then disconnects and closes in
   ``finally``.

Done is orchestrator-derived from the frame graph; nothing here emits or scrapes
a prose completion marker or trailer. State is a typed frame field only.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Awaitable, Callable

import pytest
from rtech_chatd import subjects
from rtech_chatd.tg_e2e import (
    REACTION_LIVE_SKIP_REASON,
    LiveE2EConfig,
    PollPolicy,
    RunMarker,
    SurfaceMessage,
    TelegramActor,
    live_skip_reason,
)
from rtech_chatd.tg_e2e_live import (
    NATS_URL_ENV,
    CorrelatedObserver,
    LiveTargets,
    connected_telethon_actor,
    is_non_forum_capability_error,
    resolve_live_targets,
)
from rtech_frames import ChannelOrigin, Stance, TurnFrame

pytestmark = pytest.mark.e2e

_SKIP_REASON = live_skip_reason(os.environ)
if _SKIP_REASON is not None:
    pytest.skip(_SKIP_REASON, allow_module_level=True)

# Bounded observation budgets. The discover/turn budget must outlast the trio's
# real ack round-trip; the absence budget is short — a non-trigger must NOT mint
# a task within a comparable window.
_DISCOVER_POLICY = PollPolicy(attempts=30, interval_s=1.0)
_TURN_POLICY = PollPolicy(attempts=45, interval_s=1.0)
_ABSENCE_POLICY = PollPolicy(attempts=8, interval_s=1.0)
# The maximum acceptable ack latency: the trio's first turn must land within this
# many seconds of the trigger send. Derived from the turn budget so the assertion
# fails a stall rather than hanging forever.
_MAX_ACK_LATENCY_S = float(_TURN_POLICY.attempts * _TURN_POLICY.interval_s)


@pytest.fixture(scope="module")
def live_config() -> LiveE2EConfig:
    """Load validated reference-only config after collection-time authorization."""
    return LiveE2EConfig.from_env()


@pytest.fixture(scope="module")
def targets() -> LiveTargets:
    """Resolve concrete forum/topic/non-forum target ids for the live actions."""
    return resolve_live_targets(os.environ)


@pytest.fixture(scope="module")
def nats_url() -> str:
    """Resolve the authorized bus URL a live run populated from Vaultwarden."""
    url = os.environ.get(NATS_URL_ENV, "").strip()
    if not url:
        pytest.skip(f"{NATS_URL_ENV} must be resolved for a live bus observation")
    return url


def _marker(scenario: str, ordinal: int = 0) -> RunMarker:
    """Mint a UNIQUE per-run marker so caused frames cannot be pre-existing."""
    return RunMarker(scenario=scenario, nonce=uuid.uuid4().hex, ordinal=ordinal)


async def _observed_action(
    observer: CorrelatedObserver,
    actor: TelegramActor,
    action: Callable[[TelegramActor], Awaitable[int]],
    *,
    marker: str,
) -> str:
    """Drive an action on a connected actor and discover the opaque task id.

    The observer is already open (and server-confirmed) before this runs. The
    caller connected the actor, then this drives ``action`` (which returns the
    REAL sent message id) and discovers the opaque product task id from the
    minted ``TaskFrame`` by ``origin.message_id``. Cleanup remains with the
    caller's ``finally``.
    """
    sent_id = await action(actor)
    return await observer.discover_task_id(sent_id, marker=marker, policy=_DISCOVER_POLICY)


def _new_messages(
    surface: list[SurfaceMessage], baseline: list[SurfaceMessage]
) -> list[SurfaceMessage]:
    """Return only messages whose real Telegram ids were absent before the action."""
    baseline_ids = {message.message_id for message in baseline}
    return [message for message in surface if message.message_id not in baseline_ids]


async def _cleanup(observer: CorrelatedObserver, actor: TelegramActor | None) -> None:
    """Close both run resources even if the actor connection or action failed."""
    try:
        if actor is not None:
            await actor.disconnect()
    finally:
        await observer.close()


def test_arbitration_ack_three_identities_and_topics(
    live_config: LiveE2EConfig, targets: LiveTargets, nats_url: str
) -> None:
    """Trigger the trio; assert ack latency, three distinct-bot TLDRs, three topics."""
    arbiter = live_config.accounts[0]
    mark = _marker("arbitration-ack")
    participants = tuple(account.username for account in live_config.accounts)
    topic = targets.topic_ids[arbiter.role]

    async def _action(actor: TelegramActor) -> int:
        return await actor.send_message(
            targets.forum_group_id, mark.trigger_text("arbitrate this"), reply_to=topic
        )

    async def _run() -> None:
        observer = CorrelatedObserver(nats_url)
        actor: TelegramActor | None = None
        try:
            await observer.open()
            actor = await connected_telethon_actor(arbiter)
            await actor.connect()
            baseline = await actor.read_messages(targets.forum_group_id, limit=100)
            started = asyncio.get_running_loop().time()
            task_id = await _observed_action(observer, actor, _action, marker=mark.marker)

            # Bus surface: the trio TaskFrame and per-agent ack TurnFrames arrive
            # on the discovered opaque id's real turn subject, within ack latency.
            await observer.wait_task(
                lambda _cap: len(observer.turns_for(task_id)) >= 3, policy=_TURN_POLICY
            )
            ack_latency = asyncio.get_running_loop().time() - started
            assert ack_latency <= _MAX_ACK_LATENCY_S, f"ack latency {ack_latency:.1f}s too slow"

            observer.capture.assert_task(
                task_id_value=task_id, participants=participants, arbiter=arbiter.username
            )
            turns = observer.turns_for(task_id)
            turn_agents = {turn.agent_id for turn in turns}
            assert turn_agents == set(participants), f"expected trio turns, got {turn_agents!r}"

            # Telegram surface: three TLDRs from three DISTINCT bot sender ids, and
            # three full-text topic messages actually landed in the forum group.
            surface = await actor.read_messages(targets.forum_group_id, limit=100)
            new_surface = _new_messages(surface, baseline)
            expected_sender_ids = {
                account.identity_bot_sender_id for account in live_config.accounts
            }
            tldrs = [
                message
                for message in new_surface
                if message.sender_id in expected_sender_ids and message.topic_id is None
            ]
            assert len(tldrs) == 3, f"expected three post-baseline TLDRs, got {len(tldrs)}"
            assert {message.sender_id for message in tldrs} == expected_sender_ids

            expected_topics = {
                account.username: targets.topic_ids[account.role]
                for account in live_config.accounts
            }
            for turn in turns:
                expected_topic = expected_topics[turn.agent_id]
                assert any(
                    message.topic_id == expected_topic and turn.visible_text.strip() in message.text
                    for message in new_surface
                ), f"missing full thought for {turn.agent_id!r} in topic {expected_topic}"
        finally:
            await _cleanup(observer, actor)

    asyncio.run(_run())


def test_redelivery_dedup(live_config: LiveE2EConfig, targets: LiveTargets, nats_url: str) -> None:
    """Publish one external terminal frame twice; Telegram renders it exactly once."""
    participant = live_config.accounts[1]
    visible_marker = f"[tg-e2e-external-redelivery-{uuid.uuid4().hex}]"
    terminal = TurnFrame(
        agent_id=participant.username,
        task_id=f"tg-e2e-external-{uuid.uuid4()}",
        visible_text=f"{visible_marker} external terminal reply",
        stance=Stance(position="answered", because="e2e dedup", needs_from=[]),
        origin=ChannelOrigin(
            channel_id=str(targets.forum_group_id),
            message_id=f"tg-e2e-origin-{uuid.uuid4()}",
        ),
    )

    async def _run() -> None:
        observer = CorrelatedObserver(nats_url)
        actor: TelegramActor | None = None
        try:
            await observer.open()
            actor = await connected_telethon_actor(participant)
            await actor.connect()
            baseline = await actor.read_messages(targets.forum_group_id, limit=100)

            def _render_count(surface: list[SurfaceMessage]) -> int:
                return sum(1 for message in surface if visible_marker in message.text)

            async def _current_count() -> int:
                return _render_count(
                    _new_messages(
                        await actor.read_messages(targets.forum_group_id, limit=100), baseline
                    )
                )

            assert _render_count(baseline) == 0, (
                "external marker unexpectedly existed before replay"
            )

            turn_subject = subjects.task_turn(terminal.task_id)

            # 1) Publish ONCE and prove exactly one REAL Telegram render appears.
            # A first-delivery drop (nothing ever renders) fails here rather than
            # being masked by a dedup that suppresses a never-rendered replay.
            await observer.publish(turn_subject, terminal)
            first = 0
            for attempt in range(_TURN_POLICY.attempts):
                first = await _current_count()
                if first >= 1:
                    break
                if attempt + 1 < _TURN_POLICY.attempts:
                    await asyncio.sleep(_TURN_POLICY.interval_s)
            assert first == 1, (
                f"first delivery must render exactly once before replay, observed {first}"
            )

            # 2) Replay the BYTE-IDENTICAL terminal frame on its real turn subject,
            # then watch the FULL bounded window: the render count must stay one
            # for every attempt. A late second render at any point fails.
            await observer.publish(turn_subject, terminal)
            after = first
            for attempt in range(_TURN_POLICY.attempts):
                after = await _current_count()
                if after != 1:
                    raise AssertionError(
                        f"replayed identical terminal frame changed render count to {after}"
                    )
                if attempt + 1 < _TURN_POLICY.attempts:
                    await asyncio.sleep(_TURN_POLICY.interval_s)
            assert after == 1, "external terminal frame did not remain rendered exactly once"
        finally:
            await _cleanup(observer, actor)

    asyncio.run(_run())


def test_trigger_and_no_trigger(
    live_config: LiveE2EConfig, targets: LiveTargets, nats_url: str
) -> None:
    """A trigger mints a task; an ordinary message mints none within a bounded window."""
    participant = live_config.accounts[2]
    topic = targets.topic_ids[participant.role]
    triggered = _marker("trigger-gate", 1)
    ignored = _marker("no-trigger", 2)

    async def _trigger(actor: TelegramActor) -> int:
        return await actor.send_message(
            targets.forum_group_id, triggered.trigger_text("please arbitrate"), reply_to=topic
        )

    async def _run() -> None:
        observer = CorrelatedObserver(nats_url)
        actor: TelegramActor | None = None
        try:
            await observer.open()
            actor = await connected_telethon_actor(participant)
            await actor.connect()
            task_id = await _observed_action(observer, actor, _trigger, marker=triggered.marker)
            observer.capture.assert_task(
                task_id_value=task_id,
                participants=tuple(account.username for account in live_config.accounts),
                arbiter=live_config.accounts[0].username,
            )

            # Ordinary (non-trigger) message: prove ABSENCE — it mints no TaskFrame
            # within a bounded window. The ingress subscription is already installed.
            ignored_id = await actor.send_message(
                targets.forum_group_id, f"just chatting {ignored.marker}", reply_to=topic
            )
            await observer.expect_no_task(ignored_id, marker=ignored.marker, policy=_ABSENCE_POLICY)

            # Telegram surface: both messages actually landed in the group.
            surface = await actor.read_messages(targets.forum_group_id, limit=100)
            assert any(triggered.marker in m.text for m in surface), "trigger did not land"
            assert any(ignored.marker in m.text for m in surface), "ordinary msg did not land"
        finally:
            await _cleanup(observer, actor)

    asyncio.run(_run())


@pytest.mark.skip(reason=REACTION_LIVE_SKIP_REASON)
def test_reaction_typed_steer() -> None:
    """Record that a Telegram reaction produces no SteerFrame to observe.

    A ``message_reaction`` update IS received off the wire, but chatd has no
    reaction-to-steer producer, so the update is intentionally dropped and no
    typed SteerFrame is ever published. There is nothing to assert on the bus, so
    this path is recorded as intentionally skipped rather than asserting absence.
    """


def test_non_forum_degrade(live_config: LiveE2EConfig, targets: LiveTargets, nats_url: str) -> None:
    """Send into a DISTINCT non-forum group; assert degraded non-forum output."""
    arbiter = live_config.accounts[0]
    mark = _marker("non-forum-degrade")

    async def _action(actor: TelegramActor) -> int:
        # Target the explicitly configured NON-forum group, never the forum group,
        # and post with no topic reply_to (non-forum groups have no topics).
        return await actor.send_message(
            targets.non_forum_group_id, mark.trigger_text("no topics here")
        )

    async def _run() -> None:
        observer = CorrelatedObserver(nats_url)
        actor: TelegramActor | None = None
        try:
            await observer.open()
            actor = await connected_telethon_actor(arbiter)
            await actor.connect()
            baseline = await actor.read_messages(targets.non_forum_group_id, limit=100)
            task_id = await _observed_action(observer, actor, _action, marker=mark.marker)
            await observer.wait_task(
                lambda _cap: bool(observer.turns_for(task_id)), policy=_TURN_POLICY
            )

            # Telegram answers GetForumTopics on a genuine non-forum group with
            # the SPECIFIC RPC error 400 CHANNEL_FORUM_MISSING (Telethon
            # ChannelForumMissingError). Only that exact typed capability
            # rejection proves the group has no topics; any other exception
            # (timeout, auth, bug) is NOT proof and must propagate.
            try:
                topics = await actor.read_forum_topics(targets.non_forum_group_id, limit=50)
            except Exception as exc:
                if not is_non_forum_capability_error(exc):
                    raise
                topics = []
            assert not topics, "a non-forum group must expose no forum topics"
            surface = _new_messages(
                await actor.read_messages(targets.non_forum_group_id, limit=100), baseline
            )
            assert any(
                message.topic_id is None and turn.visible_text.strip() in message.text
                for turn in observer.turns_for(task_id)
                for message in surface
            ), "a rendered TurnFrame visible_text did not land without a topic"
        finally:
            await _cleanup(observer, actor)

    asyncio.run(_run())
