"""Own the network-capable live layer of the Telegram E2E canary.

Split out of :mod:`rtech_chatd.tg_e2e` (750-line cap): this module holds the parts
that ONLY matter on an authorized live run — resolving credential/target material,
constructing a deferred Telethon client and its actor, and the bus correlator that
subscribes the REAL product subjects BEFORE a Telegram action so a plain NATS
pub/sub cannot miss the frames the action causes.

The correlation is the crux of done right: the product mints an OPAQUE
``uuid4`` task id and publishes ``TaskFrame`` on ``rtech.task.new`` and ``TurnFrame``
on ``rtech.task.<opaque-id>.turn`` (:mod:`rtech_chatd.subjects`). A scenario cannot
predict the id or invent a ``rtech.task.<id>.frames`` subject. So the observer
subscribes ``TASK_INGRESS`` and ``TASK_TURN_WILDCARD`` up front, the actor's
``send_message`` returns the REAL Telegram message id, and the observer discovers
the opaque task id by matching the minted ``TaskFrame.origin.message_id`` to that
sent id. All Telethon imports stay deferred to the last responsible moment; nothing
network-capable exists at import or collection time.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from rtech_frames import SteerFrame, TaskFrame, TurnFrame
from rtech_frames.codec import encode_frame

from rtech_chatd import subjects

from .tg_e2e import (
    DEFAULT_POLL_POLICY,
    AccountRole,
    BusConnector,
    Frame,
    FrameCapture,
    PollPolicy,
    PollTimeoutError,
    Sleep,
    SurfaceMessage,  # noqa: F401  (re-exported for scenario imports)
    SurfaceTopic,  # noqa: F401  (re-exported for scenario imports)
    TelegramAccountConfig,
    TelegramActor,
    TelethonFactory,
    _connect_frame_bus,
    _decode_capturable_frame,
)

# Resolved live-target env vars (populated ONLY on an authorized run). These name
# concrete Telegram target ids the action layer sends into: the forum group and
# per-role topic ids the scenarios post to, plus a DISTINCT non-forum group id so
# the degrade scenario targets a real non-forum group, never the forum group. The
# reference-only config never holds these; a live run resolves them at the last
# responsible moment. Absent on any non-opt-in run.
FORUM_GROUP_ID_ENV = "RTECH_TG_E2E_FORUM_GROUP_ID"
NON_FORUM_GROUP_ID_ENV = "RTECH_TG_E2E_NON_FORUM_GROUP_ID"
_TOPIC_ID_ENV_BY_ROLE: dict[AccountRole, str] = {
    "arbiter": "RTECH_TG_E2E_ARBITER_TOPIC_ID",
    "participant_one": "RTECH_TG_E2E_PARTICIPANT_ONE_TOPIC_ID",
    "participant_two": "RTECH_TG_E2E_PARTICIPANT_TWO_TOPIC_ID",
}

# Resolved-material env vars (populated ONLY on an authorized run from the
# Vaultwarden references the config names). The config never holds these values;
# a live run resolves them at the last responsible moment for a real Telethon
# construction and a real bus connection. Absent on any non-opt-in run.
API_ID_ENV = "RTECH_TG_E2E_API_ID"
API_HASH_ENV = "RTECH_TG_E2E_API_HASH"
NATS_URL_ENV = "RTECH_TG_E2E_NATS_URL"
_SESSION_STRING_ENV_BY_ROLE: dict[AccountRole, str] = {
    "arbiter": "RTECH_TG_E2E_ARBITER_SESSION_STRING",
    "participant_one": "RTECH_TG_E2E_PARTICIPANT_ONE_SESSION_STRING",
    "participant_two": "RTECH_TG_E2E_PARTICIPANT_TWO_SESSION_STRING",
}


class TelethonClientWrapper:
    """Defer Telethon import and client construction until an authorized canary runs."""

    def __init__(
        self, account: TelegramAccountConfig, factory: TelethonFactory | None = None
    ) -> None:
        self._account = account
        self._factory = factory

    def construct(self, *, session_string: str, api_id: int, api_hash: str) -> object:
        """Construct a Telethon client only after the caller resolved authorized material."""
        if not session_string or api_id <= 0 or not api_hash:
            raise ValueError("authorized Telethon session material is required")
        factory = self._factory or _telethon_factory
        return factory(session_string, api_id, api_hash)

    @property
    def account(self) -> TelegramAccountConfig:
        """Return the reference-only account configuration."""
        return self._account


def _telethon_factory(session_string: str, api_id: int, api_hash: str) -> object:
    """Import Telethon at the last responsible moment for a live canary."""
    from telethon import TelegramClient
    from telethon.sessions import StringSession

    return TelegramClient(StringSession(session_string), api_id, api_hash)


def is_non_forum_capability_error(exc: BaseException) -> bool:
    """Return whether ``exc`` is the SPECIFIC non-forum capability rejection.

    A genuine non-forum group answers ``GetForumTopics`` with Telegram RPC error
    400 ``CHANNEL_FORUM_MISSING`` (Telethon ``ChannelForumMissingError``) — that
    exact typed rejection is the only proof the group has no topics. An arbitrary
    exception whose text merely contains the word "forum" (a timeout, an auth
    error, a bug) is NOT proof and must propagate. The Telethon import stays
    deferred so importing or collecting this module needs no Telethon or network;
    if Telethon is unavailable the classifier fails closed (returns False) and the
    exception propagates.
    """
    try:
        from telethon.errors import ChannelForumMissingError
    except ImportError:
        return False
    return isinstance(exc, ChannelForumMissingError)


@dataclass(frozen=True)
class ResolvedSessionMaterial:
    """Hold authorized Telethon material resolved at live-run time only."""

    session_string: str = field(repr=False)
    api_id: int
    api_hash: str = field(repr=False)

    def __repr__(self) -> str:
        """Return a diagnostic representation carrying no credential value."""
        return f"ResolvedSessionMaterial(session_string='<redacted>', api_id={self.api_id})"


def resolve_session_material(
    role: AccountRole, env: Mapping[str, str] | None = None
) -> ResolvedSessionMaterial:
    """Resolve authorized Telethon material for a role from the live-run env.

    Reads the resolved-material env vars a live run populates from Vaultwarden.
    Fails closed when any value is missing so a live test cannot silently degrade
    to a config-only assertion.
    """
    source = os.environ if env is None else env
    session_env = _SESSION_STRING_ENV_BY_ROLE[role]
    session_string = source.get(session_env, "").strip()
    api_hash = source.get(API_HASH_ENV, "").strip()
    raw_api_id = source.get(API_ID_ENV, "").strip()
    if not session_string or not api_hash or not raw_api_id:
        raise ValueError(f"resolved Telethon material for {role!r} is incomplete")
    try:
        api_id = int(raw_api_id)
    except ValueError as exc:
        raise ValueError(f"{API_ID_ENV} must be an integer") from exc
    if api_id <= 0:
        raise ValueError(f"{API_ID_ENV} must be a positive integer")
    return ResolvedSessionMaterial(session_string=session_string, api_id=api_id, api_hash=api_hash)


def deferred_telethon_client(
    account: TelegramAccountConfig,
    *,
    factory: TelethonFactory | None = None,
    env: Mapping[str, str] | None = None,
) -> object:
    """Construct a real deferred Telethon client for one authorized account.

    Resolves credential material from the live-run env, then constructs through
    the deferred wrapper — so Telethon is imported and a network-capable client
    built only on an authorized run, never at import or collection time.
    """
    material = resolve_session_material(account.role, env)
    wrapper = TelethonClientWrapper(account, factory)
    return wrapper.construct(
        session_string=material.session_string,
        api_id=material.api_id,
        api_hash=material.api_hash,
    )


class LiveTelethonActor:
    """Adapt a real deferred Telethon client to the :class:`TelegramActor` surface.

    Wraps the network-capable client built by :func:`deferred_telethon_client`.
    ``connect``/``disconnect``/``send_message`` map to native Telethon coroutines;
    ``send_message`` returns the REAL sent message id (the acceptance correlates the
    minted ``TaskFrame`` by ``origin.message_id`` == this id, then discovers the
    opaque product task id from that frame). ``read_messages``/``read_forum_topics``
    read the live Telegram surface so a scenario asserts BOTH surfaces.
    """

    def __init__(self, client: object) -> None:
        self._client = client

    async def connect(self) -> None:
        await self._client.connect()  # type: ignore[attr-defined]

    async def disconnect(self) -> None:
        result = self._client.disconnect()  # type: ignore[attr-defined]
        if result is not None:
            await result

    async def send_message(self, chat_id: int, text: str, *, reply_to: int | None = None) -> int:
        sent = await self._client.send_message(  # type: ignore[attr-defined]
            chat_id, text, reply_to=reply_to
        )
        return int(sent.id)

    async def read_messages(self, chat_id: int, *, limit: int = 50) -> list[SurfaceMessage]:
        """Read the most recent ``limit`` messages from a chat as typed surface rows.

        Reads the ACTUAL Telegram surface (native ``get_messages``) so a scenario
        asserts what landed in the group — sender bot ids, forum topic, and text —
        never only the bus. ``reply_to.reply_to_top_id`` is the forum topic id a
        message sits under (absent in a non-forum group).
        """
        surface: list[SurfaceMessage] = []
        messages = await self._client.get_messages(chat_id, limit=limit)  # type: ignore[attr-defined]
        for message in messages:
            reply_to = getattr(message, "reply_to", None)
            topic_id = getattr(reply_to, "reply_to_top_id", None) if reply_to is not None else None
            surface.append(
                SurfaceMessage(
                    message_id=int(message.id),
                    sender_id=None if message.sender_id is None else int(message.sender_id),
                    topic_id=None if topic_id is None else int(topic_id),
                    text=message.message or "",
                )
            )
        return surface

    async def read_forum_topics(self, chat_id: int, *, limit: int = 50) -> list[SurfaceTopic]:
        """Read a forum group's topics via the real Telegram forum-topics RPC."""
        from telethon.tl.functions.messages import GetForumTopicsRequest
        from telethon.tl.types import ForumTopic

        result = await self._client(  # type: ignore[operator]
            GetForumTopicsRequest(
                peer=chat_id,
                offset_date=None,
                offset_id=0,
                offset_topic=0,
                limit=limit,
            )
        )
        topics: list[SurfaceTopic] = []
        for topic in result.topics:
            if isinstance(topic, ForumTopic):
                topics.append(SurfaceTopic(topic_id=int(topic.id), title=topic.title))
        return topics


async def connected_telethon_actor(
    account: TelegramAccountConfig,
    *,
    factory: TelethonFactory | None = None,
    env: Mapping[str, str] | None = None,
) -> TelegramActor:
    """Build a live actor over a deferred Telethon client for one account.

    Constructs the client at the last responsible moment (still deferred until an
    authorized run) and returns it wrapped as an actor. The caller connects,
    acts, and disconnects; construction alone is NOT an action.
    """
    client = deferred_telethon_client(account, factory=factory, env=env)
    return LiveTelethonActor(client)


@dataclass(frozen=True)
class LiveTargets:
    """Hold the concrete Telegram target ids the action layer sends into.

    Resolved from dedicated authorized-run env vars, never from the reference-only
    config. ``non_forum_group_id`` is a DISTINCT group so the degrade scenario
    targets a real non-forum group, not the forum group.
    """

    forum_group_id: int
    topic_ids: dict[AccountRole, int]
    non_forum_group_id: int

    def __post_init__(self) -> None:
        if self.forum_group_id == self.non_forum_group_id:
            raise ValueError("non-forum group must be a DISTINCT group from the forum group")
        if set(self.topic_ids) != set(_TOPIC_ID_ENV_BY_ROLE):
            raise ValueError("topic ids must cover exactly the arbiter and both participant roles")


def _resolve_int(source: Mapping[str, str], env_name: str) -> int:
    """Resolve one required non-zero integer target id, failing closed if absent."""
    raw = source.get(env_name, "").strip()
    if not raw:
        raise ValueError(f"{env_name} must be resolved for a live action")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be an integer") from exc
    if value == 0:
        raise ValueError(f"{env_name} must be a non-zero target id")
    return value


def resolve_live_targets(env: Mapping[str, str] | None = None) -> LiveTargets:
    """Resolve concrete forum/topic/non-forum target ids for a live action run.

    Fails closed on any missing id so a live action cannot silently skip its
    Telegram target and degrade to a passive frame-only observation.
    """
    source = os.environ if env is None else env
    topic_ids = {
        role: _resolve_int(source, env_name) for role, env_name in _TOPIC_ID_ENV_BY_ROLE.items()
    }
    return LiveTargets(
        forum_group_id=_resolve_int(source, FORUM_GROUP_ID_ENV),
        topic_ids=topic_ids,
        non_forum_group_id=_resolve_int(source, NON_FORUM_GROUP_ID_ENV),
    )


class CorrelatedObserver:
    """Subscribe the REAL product subjects BEFORE an action, then correlate frames.

    This is the correction at the heart of this harness. The product publishes
    ``TaskFrame`` on ``TASK_INGRESS`` (``rtech.task.new``) with an OPAQUE ``uuid4``
    id, then ``TurnFrame`` on ``rtech.task.<opaque-id>.turn``; the id cannot be
    predicted and no ``rtech.task.<id>.frames`` subject exists. So this observer:

    1. On :meth:`open`, subscribes ``TASK_INGRESS`` AND ``TASK_TURN_WILDCARD``
       (``rtech.task.*.turn``) up front — BEFORE any Telegram action — so a plain
       NATS pub/sub cannot miss frames the action causes.
    2. The caller drives the Telegram action; ``send_message`` returns the REAL
       sent message id.
    3. :meth:`discover_task_id` matches the minted ``TaskFrame.origin.message_id``
       to that sent id to recover the opaque product task id.
    4. :meth:`wait_turn` / :meth:`wait_task` poll the captured frames for that id.

    It also exposes :meth:`publish` so scenario 2 can replay the exact same typed
    terminal ``TurnFrame`` on its real turn subject — the actual render-dedup
    boundary — and count Telegram renders. Subscriptions are unsubscribed and the
    bus closed in :meth:`close` (always call it from a ``finally``).
    """

    def __init__(
        self,
        nats_url: str,
        *,
        connect: BusConnector | None = None,
    ) -> None:
        self._nats_url = nats_url
        self._connect = connect or _connect_frame_bus
        self._bus: object | None = None
        self._subscriptions: list[object] = []
        self.capture = FrameCapture()

    async def __aenter__(self) -> CorrelatedObserver:
        await self.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def open(self) -> None:
        """Connect and install the ingress + turn/steer-wildcard subscriptions up front.

        Subscribes ``TASK_INGRESS`` (to discover the minted TaskFrame's opaque id),
        ``TASK_TURN_WILDCARD`` (per-task turns), AND ``TASK_STEER_WILDCARD`` for
        complete typed-wire observation before any Telegram action. A server flush
        confirms all subscriptions before the caller can send its action.
        """
        if self._bus is not None:
            raise RuntimeError("observer bus is already open")
        bus = await self._connect(self._nats_url)
        self._bus = bus

        async def _on_frame(actual_subject: str, data: bytes) -> None:
            frame = _decode_capturable_frame(data)
            if frame is not None:
                self.capture.capture(frame, actual_subject)

        try:
            for subject in (
                subjects.TASK_INGRESS,
                subjects.TASK_TURN_WILDCARD,
                subjects.TASK_STEER_WILDCARD,
            ):
                self._subscriptions.append(await bus.subscribe(subject, _on_frame))
            await bus.flush()
        except Exception:
            await self.close()
            raise

    async def publish(self, subject: str, frame: Frame) -> None:
        """Publish an encoded frame on the connected bus (render-dedup replay)."""
        if self._bus is None:
            raise RuntimeError("observer bus is not open")
        await self._bus.publish(subject, encode_frame(frame))  # type: ignore[attr-defined]

    async def close(self) -> None:
        """Unsubscribe every subscription and close the bus (call from ``finally``)."""
        failure: Exception | None = None
        subscriptions, self._subscriptions = self._subscriptions, []
        for subscription in subscriptions:
            try:
                await subscription.unsubscribe()  # type: ignore[attr-defined]
            except Exception as exc:
                if failure is None:
                    failure = exc
        bus, self._bus = self._bus, None
        if bus is not None:
            try:
                await bus.close()  # type: ignore[attr-defined]
            except Exception as exc:
                if failure is None:
                    failure = exc
        if failure is not None:
            raise failure

    async def discover_task_id(
        self,
        message_id: int,
        *,
        marker: str | None = None,
        policy: PollPolicy = DEFAULT_POLL_POLICY,
        sleep: Sleep = asyncio.sleep,
    ) -> str:
        """Recover the OPAQUE product task id from the minted TaskFrame.

        Polls the ingress capture for the ``TaskFrame`` this run's inbound message
        minted, matched on ``origin.message_id`` == the REAL ``message_id`` the
        actor's ``send_message`` returned (and, when given, the ``marker`` carried
        in the frame text as a secondary guard). Returns ``frame.id`` — the opaque
        ``uuid4`` the product minted, which the caller then uses to build the real
        per-task turn subject. Fails closed if no such frame is observed in-window.
        """
        target = str(message_id)

        def _match() -> TaskFrame | None:
            for frame in self.capture.task_frames:
                origin = frame.origin
                if origin is None or origin.message_id != target:
                    continue
                if marker is not None and marker not in frame.text:
                    continue
                return frame
            return None

        for attempt in range(policy.attempts):
            found = _match()
            if found is not None:
                return found.id
            if attempt + 1 < policy.attempts:
                await sleep(policy.interval_s)
        raise PollTimeoutError(
            f"no minted TaskFrame carried origin.message_id={target!r} after "
            f"{policy.attempts} attempts"
        )

    async def wait_task(
        self,
        predicate: Callable[[FrameCapture], bool],
        *,
        policy: PollPolicy = DEFAULT_POLL_POLICY,
        sleep: Sleep = asyncio.sleep,
    ) -> FrameCapture:
        """Poll the captured frames until ``predicate`` holds or the budget ends."""
        for attempt in range(policy.attempts):
            if predicate(self.capture):
                return self.capture
            if attempt + 1 < policy.attempts:
                await sleep(policy.interval_s)
        raise PollTimeoutError(f"frame expectation not observed after {policy.attempts} attempts")

    def turns_for(self, task_id_value: str) -> list[TurnFrame]:
        """Return every captured non-blank TurnFrame for one discovered task id."""
        return [
            captured.frame
            for captured in self.capture.frames
            if isinstance(captured.frame, TurnFrame)
            and captured.subject == subjects.task_turn(task_id_value)
            and captured.frame.task_id == task_id_value
            and captured.frame.visible_text.strip()
        ]

    def steers_for(self, task_id_value: str) -> list[SteerFrame]:
        """Return every captured SteerFrame for one discovered task id.

        Keeps only frames delivered on the discovered task's concrete steer
        subject; a wildcard delivery from any other subject is not evidence.
        """
        return [
            captured.frame
            for captured in self.capture.frames
            if isinstance(captured.frame, SteerFrame)
            and captured.subject == subjects.task_steer(task_id_value)
            and captured.frame.task_id == task_id_value
        ]

    async def expect_no_task(
        self,
        message_id: int,
        *,
        marker: str | None = None,
        policy: PollPolicy = DEFAULT_POLL_POLICY,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        """Prove ABSENCE: no TaskFrame is minted for ``message_id`` within a window.

        The ingress subscription is already installed (from :meth:`open`), so this
        watches the whole bounded budget and fails closed if a task ever appears
        for the ordinary (non-trigger) message — never a passive pre-action check.
        """
        for attempt in range(policy.attempts):
            with_id = await _find_task_id(self, message_id, marker)
            if with_id is not None:
                raise AssertionError(
                    "an ordinary non-trigger message unexpectedly minted a task "
                    f"(message_id={message_id})"
                )
            if attempt + 1 < policy.attempts:
                await sleep(policy.interval_s)


async def _find_task_id(
    observer: CorrelatedObserver, message_id: int, marker: str | None
) -> str | None:
    """Return the opaque task id for ``message_id`` if already captured, else None."""
    target = str(message_id)
    for frame in observer.capture.task_frames:
        origin = frame.origin
        if origin is None or origin.message_id != target:
            continue
        if marker is not None and marker not in frame.text:
            continue
        return frame.id
    return None
