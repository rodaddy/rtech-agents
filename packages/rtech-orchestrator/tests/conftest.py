"""Local pytest fixtures for the ``test_supervised_start*`` suite.

The external-supervisor Start-only lifecycle coverage is split across four
sibling modules. Rather than import shared doubles with a plain relative import
from a bare ``tests`` package -- which collides across the workspace's multiple
sibling ``tests`` packages during a full-repository collection -- the doubles,
the ``request`` frame builder, and the ``supervisor`` factory are exposed here as
a single typed fixture object. Nothing in this module is a test.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from rtech_frames import StartAgentFrame, SupervisedAgentStateFrame
from rtech_orchestrator.supervised_start import ExternalStartSupervisor, KnownAgent


class _Authorizer:
    """Record the principal the supervisor authorizes on (never a frame)."""

    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed
        self.principals: list[str | None] = []

    async def authorize(self, principal: str | None) -> bool:
        self.principals.append(principal)
        return self.allowed


class _FrameProbingAuthorizer:
    """A DELIBERATELY WRONG authorizer that TRIES to read a frame.

    It proves the trust seam is structurally frame-free: the supervisor calls it
    with only a principal, so the extra positional argument it expects is never
    supplied. Under the old (frame-carrying) contract this authorizer would have
    received the frame and could have authorized from ``requester_id``; under the
    hardened contract the supervisor cannot even call it that way.
    """

    def __init__(self, trusted_requester: str) -> None:
        self.trusted = trusted_requester
        self.calls: list[tuple[object, ...]] = []

    async def authorize(self, *args: object) -> bool:
        self.calls.append(args)
        # If a frame were ever passed we could forge trust from requester_id.
        for arg in args:
            if getattr(arg, "requester_id", None) == self.trusted:
                return True
        # Only a principal (or None) is ever passed: authorize on it, never a frame.
        return bool(args) and isinstance(args[0], str)


class _Guard:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed
        self.agents: list[KnownAgent] = []

    async def allows(self, agent: KnownAgent) -> bool:
        self.agents.append(agent)
        return self.allowed


class _Starter:
    def __init__(self, fail: bool = False, delay: float = 0.0) -> None:
        self.fail = fail
        self.delay = delay
        self.started: list[KnownAgent] = []
        # Only successful start ACTIONS -- used by _ConfirmingLiveness so a failed
        # attempt is never mistaken for a live process.
        self.succeeded: list[KnownAgent] = []

    async def start(self, agent: KnownAgent) -> None:
        self.started.append(agent)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("start failed")
        self.succeeded.append(agent)


class _Publisher:
    def __init__(self, fail_states: tuple[str, ...] = (), fail_all: bool = False) -> None:
        self.frames: list[SupervisedAgentStateFrame] = []
        self.subjects: list[str] = []
        self.subscriptions: list[str] = []
        self._fail_states = fail_states
        self.fail_all = fail_all
        self.attempts: list[SupervisedAgentStateFrame] = []

    async def publish_frame(self, subject: str, frame: SupervisedAgentStateFrame) -> None:
        self.attempts.append(frame)
        if self.fail_all or frame.state in self._fail_states:
            raise RuntimeError("publish failed")
        self.subjects.append(subject)
        self.frames.append(frame)

    async def subscribe_frame(self, subject: str, _handler: object) -> None:
        self.subscriptions.append(subject)


class _Principals:
    """Resolve a trusted transport principal WITHOUT receiving the request.

    The resolver takes no arguments: the trusted principal comes from transport
    context, never the request body an untrusted publisher controls.
    """

    def __init__(self, principal: str | None = "trusted-operator", fail: bool = False) -> None:
        self.principal = principal
        self.fail = fail
        self.calls = 0

    async def resolve(self) -> str | None:
        self.calls += 1
        if self.fail:
            raise RuntimeError("resolver down")
        return self.principal


class _Liveness:
    """A fixed liveness verdict for both pre-start and post-start probes."""

    def __init__(self, online: bool = False, fail: bool = False) -> None:
        self.online = online
        self.fail = fail
        self.probed: list[KnownAgent] = []

    async def is_online(self, agent: KnownAgent) -> bool:
        self.probed.append(agent)
        if self.fail:
            raise RuntimeError("probe down")
        return self.online


class _ConfirmingLiveness:
    """Report DOWN until the tracked starter has started the agent, then UP.

    This models the ONLY correct liveness for a real start: the agent is not
    already online (so the pre-start ``_already_online`` short-circuit does not
    fire and a real start happens), and it becomes affirmatively alive after the
    start action, so the required post-start confirmation succeeds and ``Online``
    may publish (finding P1 liveness).
    """

    def __init__(self, starter: _Starter) -> None:
        self._starter = starter
        self.probed: list[KnownAgent] = []

    async def is_online(self, agent: KnownAgent) -> bool:
        self.probed.append(agent)
        return any(ok.agent_id == agent.agent_id for ok in self._starter.succeeded)


class _NeverAliveLiveness:
    """Confirm the agent is NEVER alive, even after a successful start action.

    Models a start action that returned but whose process is not ready or has
    already exited: the required post-start liveness confirmation reports DOWN, so
    no ``Online`` may be published (finding P1 liveness).
    """

    def __init__(self) -> None:
        self.probed: list[KnownAgent] = []

    async def is_online(self, agent: KnownAgent) -> bool:
        self.probed.append(agent)
        return False


def _request(agent_id: str = "agentd-birk", request_id: str = "req-1") -> StartAgentFrame:
    return StartAgentFrame(request_id=request_id, agent_id=agent_id, requester_id="operator-1")


def _supervisor(
    *,
    authorizer: object | None = None,
    guard: _Guard | None = None,
    starter: _Starter | None = None,
    publisher: _Publisher | None = None,
    liveness: object | None = None,
    principals: _Principals | None = None,
) -> tuple[ExternalStartSupervisor, _Starter, _Publisher]:
    actual_starter = starter or _Starter()
    actual_publisher = publisher or _Publisher()
    # A real start requires an affirmative post-start liveness confirmation
    # (finding P1 liveness). Default to a probe that confirms the agent alive only
    # AFTER the start action, so happy-path tests exercise the confirmed Online.
    actual_liveness = liveness if liveness is not None else _ConfirmingLiveness(actual_starter)
    return (
        ExternalStartSupervisor(
            known_agents=(
                KnownAgent(agent_id="agentd-birk", service_ref="rtech-agentd@agentd-birk"),
                KnownAgent(agent_id="agentd-sprocket", service_ref="rtech-agentd@agentd-sprocket"),
            ),
            starter=actual_starter,
            bus=actual_publisher,
            authorizer=authorizer,
            guard=guard,
            liveness=actual_liveness,
            principals=principals,
        ),
        actual_starter,
        actual_publisher,
    )


@dataclass(frozen=True)
class SupervisedStartKit:
    """Typed handle onto the supervised-start doubles and factories.

    Exposes the test doubles as constructor attributes and the ``request`` frame
    builder and ``supervisor`` factory as callables, so the four split modules
    consume one fixture instead of a shared relative-import helper module.
    """

    Authorizer: type[_Authorizer]
    FrameProbingAuthorizer: type[_FrameProbingAuthorizer]
    Guard: type[_Guard]
    Starter: type[_Starter]
    Publisher: type[_Publisher]
    Principals: type[_Principals]
    Liveness: type[_Liveness]
    ConfirmingLiveness: type[_ConfirmingLiveness]
    NeverAliveLiveness: type[_NeverAliveLiveness]
    request: staticmethod
    supervisor: staticmethod


@pytest.fixture
def supervised_start_kit() -> SupervisedStartKit:
    return SupervisedStartKit(
        Authorizer=_Authorizer,
        FrameProbingAuthorizer=_FrameProbingAuthorizer,
        Guard=_Guard,
        Starter=_Starter,
        Publisher=_Publisher,
        Principals=_Principals,
        Liveness=_Liveness,
        ConfirmingLiveness=_ConfirmingLiveness,
        NeverAliveLiveness=_NeverAliveLiveness,
        request=staticmethod(_request),
        supervisor=staticmethod(_supervisor),
    )
