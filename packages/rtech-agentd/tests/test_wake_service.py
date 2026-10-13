"""WakeService: allow-list enforcement, minted-frame correctness, containment (#93).

The triage agent (Nagatha) can WAKE an allow-listed teammate: mint + publish a
fresh TaskFrame carrying the triggering task's ChannelOrigin, and get back the
minted task_id. These tests pin the locked design (issue #93):

* allow-list: allowed publishes; disallowed / self / blank -> typed error, NO publish;
* minted-frame correctness: fresh id + created_at, origin carried verbatim, subject,
  requester, origin_edge provenance, stack, single-scope one-participant;
* the ``from_env`` opt-in: unset/empty allow-list -> None (capability not exposed);
* publish-failure containment: a raising bus -> typed error, turn never crashes.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from rtech_agentd.wake_service import (
    ENV_WAKE_ALLOWED_AGENTS,
    WakeConfig,
    WakeError,
    WakeResult,
    WakeService,
)
from rtech_frames import ChannelOrigin, TaskFrame
from rtech_orchestrator import subjects

from .fakes import FakeFrameBus

_TRIAGE = "agentd-nagatha"
_BILBY = "agentd-bilby"
_SKIPPY = "agentd-skippy"
_STACK = "dev-stack"
_FIXED_NOW = datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC)

_ORIGIN = ChannelOrigin(
    channel_id="chan-1",
    guild_id="guild-1",
    thread_id="thread-1",
    message_id="msg-1",
    asker="author-1",
)


def _service(
    bus: FakeFrameBus,
    *,
    allowed: frozenset[str] = frozenset({_BILBY, _SKIPPY}),
    origin: ChannelOrigin | None = _ORIGIN,
    ids: list[str] | None = None,
) -> WakeService:
    minted = list(ids or ["wake-task-1"])
    return WakeService(
        config=WakeConfig(triage_agent=_TRIAGE, allowed_agents=allowed, stack=_STACK),
        bus=bus,
        origin_provider=lambda: origin,
        id_factory=lambda: minted.pop(0),
        now=lambda: _FIXED_NOW,
    )


async def test_allowed_wake_publishes_and_returns_task_id() -> None:
    bus = FakeFrameBus()
    svc = _service(bus)

    outcome = await svc.wake(_BILBY, "what's the k-board status?", "bilby owns the k-board")

    assert isinstance(outcome, WakeResult)
    assert outcome.agent_id == _BILBY
    assert outcome.task_id == "wake-task-1"
    published = bus.by_type(TaskFrame)
    assert len(published) == 1
    subject, frame = published[0]
    assert subject == subjects.agent_task(_BILBY)
    assert isinstance(frame, TaskFrame)


async def test_minted_frame_correctness() -> None:
    bus = FakeFrameBus()
    svc = _service(bus)

    await svc.wake(_BILBY, "answer this", "reason")

    _subject, frame = bus.by_type(TaskFrame)[0]
    assert isinstance(frame, TaskFrame)
    # Fresh identity + stamp (passes the #83 fail-closed horizon).
    assert frame.id == "wake-task-1"
    assert frame.created_at == _FIXED_NOW
    # Provenance + routing fields.
    assert frame.origin_edge == f"wake:{_TRIAGE}"
    assert frame.requester == _TRIAGE
    assert frame.stack == _STACK
    assert frame.scope == "single"
    assert frame.participants == [_BILBY]
    assert frame.text == "answer this"
    # Origin carried VERBATIM so a FUTURE wired reply path can route in-channel;
    # today the reply is inert on main (chatd renders only task_ids it minted).
    assert frame.origin == _ORIGIN


async def test_origin_absent_when_triggering_task_had_none() -> None:
    bus = FakeFrameBus()
    svc = _service(bus, origin=None)

    await svc.wake(_BILBY, "q", "r")

    _subject, frame = bus.by_type(TaskFrame)[0]
    assert isinstance(frame, TaskFrame)
    assert frame.origin is None


async def test_disallowed_agent_returns_typed_error_no_publish() -> None:
    bus = FakeFrameBus()
    svc = _service(bus, allowed=frozenset({_BILBY}))

    outcome = await svc.wake(_SKIPPY, "q", "r")

    assert isinstance(outcome, WakeError)
    assert outcome.reason == "not_allowed"
    assert not bus.by_type(TaskFrame), "a disallowed wake must never publish"


async def test_self_wake_refused_no_publish() -> None:
    bus = FakeFrameBus()
    svc = _service(bus)

    outcome = await svc.wake(_TRIAGE, "q", "r")

    assert isinstance(outcome, WakeError)
    assert outcome.reason == "self_wake"
    assert not bus.by_type(TaskFrame)


async def test_blank_agent_refused_no_publish() -> None:
    bus = FakeFrameBus()
    svc = _service(bus)

    outcome = await svc.wake("   ", "q", "r")

    assert isinstance(outcome, WakeError)
    assert outcome.reason == "blank_agent"
    assert not bus.by_type(TaskFrame)


async def test_publish_failure_contained_as_typed_error() -> None:
    # The wake tool publishes via the PROPAGATING publish_frame_checked (issue #93),
    # so a raising checked publish is the transport-failure path under test.
    class _RaisingBus(FakeFrameBus):
        async def publish_frame_checked(self, subject: str, frame: object) -> None:  # type: ignore[override]
            raise RuntimeError("bus down")

    bus = _RaisingBus()
    svc = _service(bus)

    # Must NOT raise — a publish failure is contained as a typed error.
    outcome = await svc.wake(_BILBY, "q", "r")

    assert isinstance(outcome, WakeError)
    assert outcome.reason == "publish_failed"
    # No false success: a raising checked publish records ZERO published frames.
    assert not bus.by_type(TaskFrame)


async def test_swallowing_publish_would_not_mask_a_raise() -> None:
    """Regression pin (#93): wake uses publish_frame_checked, not publish_frame.

    A bus whose publish_frame SWALLOWS (returns normally) but whose
    publish_frame_checked RAISES proves the wake tool takes the checked path — the
    swallow can no longer turn a failed publish into a WakeResult false success.
    """

    class _SwallowButCheckedRaisesBus(FakeFrameBus):
        async def publish_frame(self, subject: str, frame: object) -> None:  # type: ignore[override]
            # Swallow like the production TypedFrameBus.publish_frame does.
            return None

        async def publish_frame_checked(self, subject: str, frame: object) -> None:  # type: ignore[override]
            raise RuntimeError("transport down")

    bus = _SwallowButCheckedRaisesBus()
    svc = _service(bus)

    outcome = await svc.wake(_BILBY, "q", "r")

    assert isinstance(outcome, WakeError)
    assert outcome.reason == "publish_failed"
    assert not bus.by_type(TaskFrame)


async def test_from_env_unset_returns_none() -> None:
    cfg = WakeConfig.from_env(triage_agent=_TRIAGE, stack=_STACK, env={})
    assert cfg is None


async def test_from_env_blank_returns_none() -> None:
    cfg = WakeConfig.from_env(
        triage_agent=_TRIAGE, stack=_STACK, env={ENV_WAKE_ALLOWED_AGENTS: "  ,  "}
    )
    assert cfg is None


async def test_from_env_only_self_returns_none() -> None:
    # An allow-list naming ONLY this agent strips to empty -> not a waker.
    cfg = WakeConfig.from_env(
        triage_agent=_TRIAGE, stack=_STACK, env={ENV_WAKE_ALLOWED_AGENTS: _TRIAGE}
    )
    assert cfg is None


async def test_from_env_parses_and_strips_self() -> None:
    cfg = WakeConfig.from_env(
        triage_agent=_TRIAGE,
        stack=_STACK,
        env={ENV_WAKE_ALLOWED_AGENTS: f"{_BILBY}, {_SKIPPY} ,{_TRIAGE},{_BILBY}"},
    )
    assert cfg is not None
    # Self dropped, blanks stripped, de-duped.
    assert cfg.allowed_agents == frozenset({_BILBY, _SKIPPY})
    assert cfg.stack == _STACK
    assert cfg.triage_agent == _TRIAGE


async def test_config_rejects_blank_stack() -> None:
    with pytest.raises(ValueError, match="stack must be non-empty"):
        WakeConfig(triage_agent=_TRIAGE, allowed_agents=frozenset({_BILBY}), stack="  ")


async def test_config_rejects_empty_allow_list() -> None:
    with pytest.raises(ValueError, match="allowed_agents must be non-empty"):
        WakeConfig(triage_agent=_TRIAGE, allowed_agents=frozenset(), stack=_STACK)


async def test_config_rejects_self_in_allow_list() -> None:
    with pytest.raises(ValueError, match="must not be in its own wake allow-list"):
        WakeConfig(triage_agent=_TRIAGE, allowed_agents=frozenset({_TRIAGE}), stack=_STACK)


async def test_config_rejects_subject_unsafe_allow_list_id() -> None:
    # An id with a '.' would pass the allow-list `in` check yet raise inside
    # TaskFrame construction (mid-turn) and target a wrong subject — so it fails
    # CLOSED at config build instead.
    with pytest.raises(ValueError, match="not a safe NATS subject token"):
        WakeConfig(triage_agent=_TRIAGE, allowed_agents=frozenset({"agentd.bilby"}), stack=_STACK)


async def test_from_env_rejects_subject_unsafe_id() -> None:
    with pytest.raises(ValueError, match="not a safe NATS subject token"):
        WakeConfig.from_env(
            triage_agent=_TRIAGE, stack=_STACK, env={ENV_WAKE_ALLOWED_AGENTS: "agent one"}
        )


async def test_config_accepts_agentd_convention_id() -> None:
    # A deployed-convention id (``agentd-<name>``) passes both the wire-safety
    # check AND the convention regex.
    cfg = WakeConfig(triage_agent=_TRIAGE, allowed_agents=frozenset({_BILBY}), stack=_STACK)
    assert cfg.allowed_agents == frozenset({_BILBY})


async def test_config_rejects_non_agentd_convention_id() -> None:
    # ``bilby`` is subject-safe but not ``agentd-<name>``: it would target a subject
    # no deployed agentd subscribes, so it must fail CLOSED at build with the env
    # var, the required convention, and an example named in the message.
    with pytest.raises(ValueError) as excinfo:
        WakeConfig(triage_agent=_TRIAGE, allowed_agents=frozenset({"bilby"}), stack=_STACK)
    message = str(excinfo.value)
    assert ENV_WAKE_ALLOWED_AGENTS in message
    assert "agentd-" in message
    assert "agentd-bilby" in message
