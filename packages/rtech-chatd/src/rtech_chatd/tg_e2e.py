"""Define the authorization-gated Telegram live-canary harness.

This harness names the reference-only configuration a real-account Telegram
canary needs and asserts observations against the typed frame contract. It holds
no credential VALUES (only Vaultwarden/env reference names), defers every
Telethon import and client construction until an authorized run, and gates on
explicit operator opt-in BEFORE any network-capable object exists — so importing
or collecting it without credentials or network is safe.

Observations are asserted only against typed :class:`TaskFrame` /
:class:`TurnFrame` frames read off the bus: state is a typed frame field, never a
scraped prose trailer, sentinel, or connector-held signal.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from loguru import logger
from rtech_frames import SteerFrame, TaskFrame, TurnFrame

from .subjects import task_steer, task_turn

LIVE_OPT_IN_ENV = "RTECH_TG_E2E_LIVE"
GROUP_REF_ENV = "RTECH_TG_E2E_GROUP_REF"
NATS_URL_REF_ENV = "RTECH_TG_E2E_NATS_URL_REF"
REACTION_LIVE_SKIP_REASON = (
    "message_reaction updates are received but intentionally dropped: chatd has no "
    "reaction-to-steer producer, so a reaction emits no SteerFrame to observe"
)

# The resolved live-target and Telethon-material env vars a live run populates
# (never the reference-only config) live with the network layer in
# :mod:`rtech_chatd.tg_e2e_live`: FORUM_GROUP_ID_ENV / NON_FORUM_GROUP_ID_ENV /
# per-role topic + session ids, API_ID_ENV / API_HASH_ENV / NATS_URL_ENV. This
# module holds only the reference-only config, the frame primitives, and the
# abstract action contract — nothing network-capable.

type AccountRole = Literal["arbiter", "participant_one", "participant_two"]
# The capturable frame universe includes every frame the observer can encounter
# on its subscribed subjects. SteerFrame capture remains typed even though the
# product currently has no Telegram reaction-to-steer live scenario.
type Frame = TaskFrame | TurnFrame | SteerFrame
type FrameReader = Callable[[], Awaitable[Sequence[Frame]]]
type Sleep = Callable[[float], Awaitable[None]]
type TelethonFactory = Callable[[str, int, str], object]
type RawFrameHandler = Callable[[str, bytes], Awaitable[None]]


class _Subscription(Protocol):
    """Represent the unsubscribe surface a bus subscription exposes."""

    async def unsubscribe(self) -> None:
        """Cancel the subscription."""


class _ObservableBus(Protocol):
    """Represent the raw-frame publish/subscribe/close surface used for observation.

    ``publish`` is required for the render-dedup boundary (scenario 2): replaying
    the exact same terminal ``TurnFrame`` on its real per-task turn subject and
    counting Telegram renders is the actual dedup test, so the observer bus must
    be able to publish the byte-identical frame back onto the wire.
    """

    async def publish(self, subject: str, data: bytes) -> None:
        """Publish raw encoded frame bytes."""

    async def subscribe(self, subject: str, handler: RawFrameHandler) -> _Subscription:
        """Subscribe to raw encoded frame bytes."""

    async def flush(self) -> None:
        """Confirm pending subscriptions with the NATS server."""

    async def close(self) -> None:
        """Close the underlying connection."""


type BusConnector = Callable[[str], Awaitable[_ObservableBus]]

_ACCOUNT_ENV_BY_ROLE: dict[AccountRole, tuple[str, str, str]] = {
    "arbiter": (
        "RTECH_TG_E2E_ARBITER_SESSION_REF",
        "RTECH_TG_E2E_ARBITER_USERNAME",
        "RTECH_TG_E2E_ARBITER_TOPIC_REF",
    ),
    "participant_one": (
        "RTECH_TG_E2E_PARTICIPANT_ONE_SESSION_REF",
        "RTECH_TG_E2E_PARTICIPANT_ONE_USERNAME",
        "RTECH_TG_E2E_PARTICIPANT_ONE_TOPIC_REF",
    ),
    "participant_two": (
        "RTECH_TG_E2E_PARTICIPANT_TWO_SESSION_REF",
        "RTECH_TG_E2E_PARTICIPANT_TWO_USERNAME",
        "RTECH_TG_E2E_PARTICIPANT_TWO_TOPIC_REF",
    ),
}

_IDENTITY_BOT_SENDER_ID_ENV_BY_ROLE: dict[AccountRole, str] = {
    "arbiter": "RTECH_TG_E2E_ARBITER_IDENTITY_BOT_SENDER_ID",
    "participant_one": "RTECH_TG_E2E_PARTICIPANT_ONE_IDENTITY_BOT_SENDER_ID",
    "participant_two": "RTECH_TG_E2E_PARTICIPANT_TWO_IDENTITY_BOT_SENDER_ID",
}

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _required_name(value: str, field_name: str) -> str:
    """Return a non-blank reference or identity name."""
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty reference or identity name")
    return normalized


def _reject_duplicates(values: Sequence[str], field_name: str) -> None:
    """Reject repeated identity or reference names across a three-distinct set."""
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must be distinct across the trio, got {list(values)!r}")


def _required_nonzero_id(value: str, field_name: str) -> int:
    """Return one non-zero opaque Telegram identity id."""
    try:
        parsed = int(value.strip())
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a non-zero integer") from exc
    if parsed == 0:
        raise ValueError(f"{field_name} must be a non-zero integer")
    return parsed


@dataclass(frozen=True)
class TelegramAccountConfig:
    """Name one live Telegram account without holding its session value."""

    session_string_ref: str = field(repr=False)
    role: AccountRole
    username: str
    identity_bot_sender_id: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "session_string_ref",
            _required_name(self.session_string_ref, "session_string_ref"),
        )
        if self.role not in _ACCOUNT_ENV_BY_ROLE:
            raise ValueError(f"unsupported Telegram E2E role {self.role!r}")
        object.__setattr__(self, "username", _required_name(self.username, "username"))
        if self.identity_bot_sender_id == 0:
            raise ValueError("identity_bot_sender_id must be a non-zero integer")

    def __repr__(self) -> str:
        """Return a safe diagnostic representation with no credential reference."""
        return (
            "TelegramAccountConfig(session_string_ref='<redacted>', "
            f"role={self.role!r}, username={self.username!r}, "
            f"identity_bot_sender_id={self.identity_bot_sender_id})"
        )


@dataclass(frozen=True)
class TelegramGroupConfig:
    """Name the required forum group and its three expected topic references."""

    group_ref: str
    topic_refs: tuple[str, str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_ref", _required_name(self.group_ref, "group_ref"))
        if len(self.topic_refs) != 3:
            raise ValueError("topic_refs must contain exactly three topic reference names")
        object.__setattr__(
            self,
            "topic_refs",
            tuple(_required_name(ref, "topic_ref") for ref in self.topic_refs),
        )
        _reject_duplicates(self.topic_refs, "topic_refs")


@dataclass(frozen=True)
class NatsConfig:
    """Name the NATS connection setting without carrying its URL."""

    url_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "url_ref", _required_name(self.url_ref, "nats_url_ref"))


@dataclass(frozen=True)
class LiveE2EConfig:
    """Hold the reference-only configuration required to authorize a canary."""

    accounts: tuple[TelegramAccountConfig, TelegramAccountConfig, TelegramAccountConfig]
    group: TelegramGroupConfig
    nats: NatsConfig

    def __post_init__(self) -> None:
        roles = tuple(account.role for account in self.accounts)
        if roles != tuple(_ACCOUNT_ENV_BY_ROLE):
            raise ValueError(
                "accounts must be arbiter, participant_one, participant_two in that order"
            )
        _reject_duplicates(
            tuple(account.username for account in self.accounts), "account usernames"
        )
        _reject_duplicates(
            tuple(account.session_string_ref for account in self.accounts),
            "account session_string_refs",
        )
        sender_ids = tuple(account.identity_bot_sender_id for account in self.accounts)
        if len(set(sender_ids)) != len(sender_ids):
            raise ValueError(
                f"identity_bot_sender_ids must be distinct across the trio, got {sender_ids!r}"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LiveE2EConfig:
        """Load reference names after explicit live authorization is present."""
        source = os.environ if env is None else env
        if not live_opted_in(source):
            raise ValueError(f"{LIVE_OPT_IN_ENV} must be explicitly truthy before a live canary")
        accounts: list[TelegramAccountConfig] = []
        topic_refs: list[str] = []
        for role, (session_env, username_env, topic_env) in _ACCOUNT_ENV_BY_ROLE.items():
            accounts.append(
                TelegramAccountConfig(
                    session_string_ref=source.get(session_env, ""),
                    role=role,
                    username=source.get(username_env, ""),
                    identity_bot_sender_id=_required_nonzero_id(
                        source.get(_IDENTITY_BOT_SENDER_ID_ENV_BY_ROLE[role], ""),
                        _IDENTITY_BOT_SENDER_ID_ENV_BY_ROLE[role],
                    ),
                )
            )
            topic_refs.append(source.get(topic_env, ""))
        return cls(
            accounts=(accounts[0], accounts[1], accounts[2]),
            group=TelegramGroupConfig(
                group_ref=source.get(GROUP_REF_ENV, ""),
                topic_refs=(topic_refs[0], topic_refs[1], topic_refs[2]),
            ),
            nats=NatsConfig(url_ref=source.get(NATS_URL_REF_ENV, "")),
        )


@dataclass(frozen=True)
class PollPolicy:
    """Bound a canary observation loop."""

    attempts: int = 10
    interval_s: float = 1.0

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be >= 1")
        if self.interval_s < 0:
            raise ValueError("interval_s must be >= 0")


class PollTimeoutError(TimeoutError):
    """Report a bounded poll that did not observe its expected state."""


DEFAULT_POLL_POLICY = PollPolicy()


def live_opted_in(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the operator explicitly authorized live activity."""
    source = os.environ if env is None else env
    return source.get(LIVE_OPT_IN_ENV, "").strip().lower() in _TRUTHY


def live_skip_reason(env: Mapping[str, str] | None = None) -> str | None:
    """Return an import-safe skip reason before any network-capable object exists."""
    source = os.environ if env is None else env
    if not live_opted_in(source):
        return f"{LIVE_OPT_IN_ENV} is not explicitly enabled"
    try:
        LiveE2EConfig.from_env(source)
    except ValueError as exc:
        return f"live Telegram E2E configuration is incomplete: {exc}"
    return None


def scenario_id(scenario: str) -> str:
    """Return a deterministic, NATS-safe scenario identifier."""
    normalized = _required_name(scenario, "scenario").lower().replace(" ", "-")
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:12]
    return f"tg-e2e-{normalized}-{digest}"


def task_id(scenario: str, ordinal: int = 0) -> str:
    """Return a deterministic task identifier for one scenario attempt."""
    if ordinal < 0:
        raise ValueError("ordinal must be >= 0")
    return f"{scenario_id(scenario)}-{ordinal}"


async def poll_frames(
    read_frames: FrameReader,
    predicate: Callable[[Sequence[Frame]], bool],
    *,
    policy: PollPolicy = DEFAULT_POLL_POLICY,
    sleep: Sleep = asyncio.sleep,
) -> Sequence[Frame]:
    """Poll a typed frame reader for a bounded number of attempts."""
    last: Sequence[Frame] = ()
    for attempt in range(policy.attempts):
        last = await read_frames()
        if predicate(last):
            return last
        if attempt + 1 < policy.attempts:
            await sleep(policy.interval_s)
    raise PollTimeoutError(f"frame expectation not observed after {policy.attempts} attempts")


@dataclass(frozen=True)
class CapturedFrame:
    """Pair one decoded frame with the actual NATS subject that delivered it."""

    subject: str
    frame: Frame


@dataclass
class FrameCapture:
    """Capture and assert only typed TaskFrame, TurnFrame, and SteerFrame observations.

    Every capture retains its actual NATS subject so wildcard observations cannot
    be mistaken for a discovered task's concrete turn or steer wire.
    """

    frames: list[CapturedFrame] = field(default_factory=list)

    @property
    def task_frames(self) -> list[TaskFrame]:
        """Return captured TaskFrames while retaining their subjects in ``frames``."""
        return [captured.frame for captured in self.frames if isinstance(captured.frame, TaskFrame)]

    @property
    def turn_frames(self) -> list[TurnFrame]:
        """Return captured TurnFrames while retaining their subjects in ``frames``."""
        return [captured.frame for captured in self.frames if isinstance(captured.frame, TurnFrame)]

    @property
    def steer_frames(self) -> list[SteerFrame]:
        """Return captured SteerFrames while retaining their subjects in ``frames``."""
        return [
            captured.frame for captured in self.frames if isinstance(captured.frame, SteerFrame)
        ]

    def capture(self, frame: Frame, subject: str) -> None:
        """Record a typed frame or reject an unrelated observation."""
        if isinstance(frame, TaskFrame):
            self.frames.append(CapturedFrame(subject=subject, frame=frame))
            return
        if isinstance(frame, TurnFrame):
            self.frames.append(CapturedFrame(subject=subject, frame=frame))
            return
        if isinstance(frame, SteerFrame):
            self.frames.append(CapturedFrame(subject=subject, frame=frame))
            return
        raise TypeError(f"expected TaskFrame, TurnFrame, or SteerFrame, got {type(frame).__name__}")

    def assert_task(
        self, *, task_id_value: str, participants: tuple[str, str, str], arbiter: str
    ) -> TaskFrame:
        """Return the asserted arbitration TaskFrame from this capture."""
        matches = [frame for frame in self.task_frames if frame.id == task_id_value]
        if len(matches) != 1:
            raise AssertionError(
                f"expected exactly one TaskFrame for {task_id_value!r}, got {len(matches)}"
            )
        frame = matches[0]
        if tuple(frame.participants) != participants or frame.arbiter != arbiter:
            raise AssertionError("TaskFrame arbitration identities did not match the expected trio")
        return frame

    def assert_turn(self, *, task_id_value: str, agent_id: str) -> TurnFrame:
        """Return one non-blank TurnFrame for the stated task and agent."""
        matches = [
            captured.frame
            for captured in self.frames
            if isinstance(captured.frame, TurnFrame)
            and captured.subject == task_turn(task_id_value)
            and captured.frame.task_id == task_id_value
            and captured.frame.agent_id == agent_id
            and captured.frame.visible_text.strip()
        ]
        if len(matches) != 1:
            raise AssertionError(
                "expected exactly one non-blank TurnFrame for "
                f"{task_id_value!r}/{agent_id!r}, got {len(matches)}"
            )
        return matches[0]

    def assert_steer(self, *, task_id_value: str, target_agent: str) -> SteerFrame:
        """Return one typed SteerFrame for the stated task and target agent.

        Requires the discovered task's concrete steer subject, so a wildcard
        capture from another subject fails closed even when its payload names the
        same task and target agent.
        """
        matches = [
            captured.frame
            for captured in self.frames
            if isinstance(captured.frame, SteerFrame)
            and captured.subject == task_steer(task_id_value)
            and captured.frame.task_id == task_id_value
            and captured.frame.target_agent == target_agent
        ]
        if len(matches) != 1:
            raise AssertionError(
                "expected exactly one SteerFrame for "
                f"{task_id_value!r}/{target_agent!r}, got {len(matches)}"
            )
        return matches[0]


@dataclass(frozen=True)
class SurfaceMessage:
    """One Telegram message read back off the live surface via Telethon.

    A typed, minimal view of what actually landed in the group — asserted by the
    scenarios ALONGSIDE the bus surface. ``sender_id`` is the opaque author id (the
    posting bot's id for a rendered TLDR — distinct bot ids prove per-identity
    render); ``topic_id`` is the forum topic the message sits in (``None`` in a
    non-forum group); ``text`` is the message body. No display names, no secrets.
    """

    message_id: int
    sender_id: int | None
    topic_id: int | None
    text: str


@dataclass(frozen=True)
class SurfaceTopic:
    """One forum topic read back off the live surface via Telethon."""

    topic_id: int
    title: str


class TelegramActor(Protocol):
    """Model the minimal Telegram action + read surface a live scenario drives.

    A live run wraps a real connected Telethon client; a mocked unit test supplies
    a fake. Every action begins with :meth:`connect` and the caller MUST
    :meth:`disconnect` in a ``finally`` — a scenario that constructs a client but
    never connects, sends, or reacts is not a real action and must not pass.

    :meth:`send_message` returns the REAL sent message id so a scenario correlates
    the minted ``TaskFrame`` by ``origin.message_id`` and discovers the opaque
    product task id; :meth:`read_messages`/:meth:`read_forum_topics` read the live
    Telegram surface so every scenario asserts BOTH the Telegram and bus surfaces.
    """

    async def connect(self) -> None:
        """Connect the intended real account before any action."""

    async def disconnect(self) -> None:
        """Disconnect the account; always called from a ``finally``."""

    async def send_message(self, chat_id: int, text: str, *, reply_to: int | None = None) -> int:
        """Send an ordinary or trigger message; return the real sent message id."""

    async def read_messages(self, chat_id: int, *, limit: int = 50) -> list[SurfaceMessage]:
        """Read recent messages off the live Telegram surface as typed rows."""

    async def read_forum_topics(self, chat_id: int, *, limit: int = 50) -> list[SurfaceTopic]:
        """Read a forum group's topics off the live Telegram surface."""


type ActorFactory = Callable[[TelegramAccountConfig], Awaitable[TelegramActor]]


@dataclass(frozen=True)
class RunMarker:
    """Bind a scenario run to a unique correlation MARKER carried in message text.

    The marker rides the outbound Telegram action text ONLY: it makes this run's
    inbound message — and therefore the ``TaskFrame`` the product mints FROM it —
    distinguishable from any pre-existing traffic another process emitted. It does
    NOT invent a task id or a bus subject. The product mints an OPAQUE ``uuid4``
    task id and publishes ``TaskFrame`` on :data:`subjects.TASK_INGRESS` and
    ``TurnFrame`` on ``rtech.task.<opaque-id>.turn``; a scenario cannot predict
    that id. It is discovered by correlating the minted ``TaskFrame`` on the REAL
    ``origin.message_id`` returned by :meth:`TelegramActor.send_message` (the
    marker is a secondary text guard), then used to build the real per-task turn/
    steer subject — see :class:`rtech_chatd.tg_e2e_live.CorrelatedObserver`.
    ``nonce`` is caller-supplied so mocked tests stay deterministic while live
    runs pass a unique value per run.
    """

    scenario: str
    nonce: str
    ordinal: int = 0

    def __post_init__(self) -> None:
        _required_name(self.scenario, "scenario")
        _required_name(self.nonce, "nonce")
        if self.ordinal < 0:
            raise ValueError("ordinal must be >= 0")

    @property
    def marker(self) -> str:
        """Return the unique correlation marker the outbound action text carries."""
        base = scenario_id(self.scenario)
        digest = hashlib.sha256(f"{self.nonce}:{self.ordinal}".encode()).hexdigest()[:12]
        return f"[{base}-{self.ordinal}-{digest}]"

    def trigger_text(self, body: str) -> str:
        """Return trigger-message text that carries the correlation marker."""
        return f"{self.marker} {body}"

    def matches(self, text: str) -> bool:
        """Return whether ``text`` carries this run's marker (frame-text guard)."""
        return self.marker in text


async def drive_action(
    actor: TelegramActor, action: Callable[[TelegramActor], Awaitable[None]]
) -> None:
    """Connect, run the real action, and always disconnect in ``finally``.

    The single owned action-driver: a scenario installs its bus subscriptions
    BEFORE calling this (so a plain NATS pub/sub cannot miss the frames the action
    causes), drives the action here, and disconnects in ``finally`` even on error.
    """
    try:
        await actor.connect()
        await action(actor)
    finally:
        await actor.disconnect()


async def observe_frames(
    nats_url: str,
    subject: str,
    predicate: Callable[[FrameCapture], bool],
    *,
    policy: PollPolicy = DEFAULT_POLL_POLICY,
    sleep: Sleep = asyncio.sleep,
    connect: BusConnector | None = None,
) -> FrameCapture:
    """Observe typed frames off the live bus into a FrameCapture.

    Connects to the referenced NATS URL, subscribes to ``subject``, decodes each
    raw frame into a typed :class:`TaskFrame`/:class:`TurnFrame`, and captures it.
    Polls until ``predicate`` is satisfied or the bounded budget is exhausted.
    Bus state is read as typed frames — never a scraped prose trailer.
    """
    connector = connect or _connect_frame_bus
    capture = FrameCapture()

    async def _on_frame(actual_subject: str, data: bytes) -> None:
        frame = _decode_capturable_frame(data)
        if frame is not None:
            capture.capture(frame, actual_subject)

    bus = await connector(nats_url)
    subscription = await bus.subscribe(subject, _on_frame)
    try:
        for attempt in range(policy.attempts):
            if predicate(capture):
                return capture
            if attempt + 1 < policy.attempts:
                await sleep(policy.interval_s)
        raise PollTimeoutError(
            f"typed frame expectation not observed after {policy.attempts} attempts"
        )
    finally:
        await subscription.unsubscribe()
        await bus.close()


class _NatsObservableBus:
    """Expose raw NATS message subjects for the E2E observer only."""

    def __init__(self, nats_url: str) -> None:
        """Build the product bus adapter without connecting it."""
        from rtech_chatd.bus import NatsFrameBus

        self._frame_bus = NatsFrameBus(nats_url)

    async def connect(self) -> None:
        """Connect the underlying product bus."""
        await self._frame_bus.connect()

    async def publish(self, subject: str, data: bytes) -> None:
        """Publish bytes through the product bus adapter."""
        await self._frame_bus.publish(subject, data)

    async def subscribe(self, subject: str, handler: RawFrameHandler) -> _Subscription:
        """Subscribe at the raw boundary so each delivered subject is retained."""
        from nats.aio.msg import Msg

        async def _on_message(message: Msg) -> None:
            await handler(message.subject, message.data)

        return await self._frame_bus._bus.raw_subscribe(subject, _on_message)  # noqa: SLF001

    async def flush(self) -> None:
        """Confirm the observer's subscriptions reached the NATS server."""
        await self._frame_bus._bus._nc.flush()  # noqa: SLF001

    async def close(self) -> None:
        """Close the product bus connection."""
        await self._frame_bus.close()


async def _connect_frame_bus(nats_url: str) -> _ObservableBus:
    """Connect a raw-subject-preserving observer bus at live-run time.

    If ``connect`` raises AFTER the underlying client opened resources, close the
    partially initialized local bus before re-raising so a failed connect never
    leaks a half-open connection to the caller (who has no handle to close it).
    """
    bus = _NatsObservableBus(nats_url)
    try:
        await bus.connect()
    except BaseException:
        try:
            await bus.close()
        except BaseException as close_exc:
            logger.warning(
                "tg_e2e.connect_frame_bus | stage=close_after_failed_connect | close_error_type={}",
                type(close_exc).__name__,
            )
        raise
    return bus


def _decode_capturable_frame(data: bytes) -> Frame | None:
    """Decode raw bus bytes into a capturable TaskFrame/TurnFrame/SteerFrame or drop it."""
    from rtech_frames.codec import decode_frame

    decoded = decode_frame(data)
    if isinstance(decoded, TaskFrame | TurnFrame | SteerFrame):
        return decoded
    return None
