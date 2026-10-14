"""Mint + publish a wake TaskFrame toward an allow-listed teammate.

The triage agent (Nagatha) handles an intake :class:`~rtech_frames.TaskFrame` and
can either just ANSWER (the existing TurnFrame reply path — nothing new) or WAKE a
teammate. Waking means minting a FRESH ``TaskFrame`` on the teammate's task
subject that CARRIES the original :class:`~rtech_frames.ChannelOrigin`, and
returning the minted ``task_id`` so the triage agent can reference it in her reply.

The carried origin is NOT YET RENDERED BACK ON MAIN: chatd today only renders
task_ids it minted itself, so a woken agent's answer is INERT on main — the origin
is carried for WHEN the reply path is wired, but today it does not route the woken
agent's answer back to the originating channel.

This service is the engine-agnostic seam owned by the agentd frame-turn/supervisor
layer. It is exposed to :class:`~rtech_agentd.claude_engine.ClaudeEngine` via the
in-process SDK MCP wake tool (:mod:`rtech_agentd.wake_tool`), exactly as the
``stance`` tool exposes stance capture. Design invariants:

* **Allow-list, fail-loud.** Wakeable ids come from config/env
  (:data:`ENV_WAKE_ALLOWED_AGENTS`) — a comma list of ``agentd-<name>`` ids with
  NO default. An empty/unset allow-list means the wake capability is not exposed
  at all (the caller does not construct this service, so the tool never appears).
  A wake for a non-allowed id returns a typed error and NEVER publishes.
* **Loop safety.** The allow-list env being set IS the opt-in that this agent is a
  waker (only Nagatha's deploy sets it), which bounds wake-chaining: a woken agent
  is not itself a waker. Additionally, a wake targeting THIS agent is refused.
* **Minted frame correctness.** Fresh ``id`` + ``created_at`` (so the wake frame
  passes the fail-closed horizon), ``stack`` from config, ``origin`` copied
  verbatim from the triggering frame when present, ``origin_edge`` marking
  provenance (``wake:<triage-agent-id>``), and ``requester`` = the triage agent id.
  The subject is built ONLY via :func:`rtech_orchestrator.subjects.agent_task`.
* **Publish failures contained.** A publish failure returns a typed error to the
  engine (logged once with context); it never crashes the turn.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from loguru import logger
from rtech_frames import ChannelOrigin, TaskFrame, validate_participants
from rtech_orchestrator import subjects

from .frame_turn import FrameBus

# Comma-separated list of ``agentd-<name>`` ids THIS agent may wake. NO default:
# unset/empty means the wake capability is never exposed (see WakeConfig.from_env).
# Only the triage (waker) deploy — Nagatha — sets it; that opt-in is itself the
# loop-safety bound (a woken agent has no allow-list, so cannot wake-chain).
ENV_WAKE_ALLOWED_AGENTS = "RTECH_WAKE_ALLOWED_AGENTS"

# The wake TaskFrame is a single triage-driven dispatch: scope "single", one
# named participant (the woken teammate answers as itself, no fanout debate).
_WAKE_SCOPE: Literal["single"] = "single"

# Origin_edge provenance prefix stamped on a minted wake frame: "wake:<triage-id>".
_WAKE_ORIGIN_EDGE_PREFIX = "wake"

# Deployed agent ids follow the ``agentd-<name>`` convention (deploy/kboard/
# install.sh AGENT_INSTANCES, e.g. ``agentd-bilby``). An allow-list entry that is
# merely subject-safe (``bilby``) would pass ``validate_participants`` yet target
# ``rtech.agent.bilby.task`` — a subject no deployed agentd subscribes — so the
# wake would silently vanish. This convention regex fails such an id CLOSED at
# config build (a loud startup error) alongside the wire-safety check.
_ALLOWED_AGENT_CONVENTION = re.compile(r"^agentd-[a-z0-9][a-z0-9-]*$")

IdFactory = Callable[[], str]
Now = Callable[[], datetime]
# Supplies the ChannelOrigin of the task currently driving the engine session, so
# a wake carries the SAME origin. The origin is carried for a FUTURE wired reply
# path — today it is INERT on main (chatd renders only task_ids it minted, so the
# woken agent's answer is not yet routed back in-channel). ``None`` when the
# triggering task carried no origin (non-intake frame).
OriginProvider = Callable[[], ChannelOrigin | None]


@dataclass(frozen=True)
class WakeConfig:
    """Typed wake configuration for the triage (waker) agent.

    Attributes:
        triage_agent: This agent's id — the ``requester`` and ``origin_edge``
            provenance stamped on a minted wake frame, and the id a self-wake is
            refused against.
        allowed_agents: The frozen set of ``agentd-<name>`` ids this agent may
            wake. NEVER empty for a constructed :class:`WakeService` (an empty
            allow-list means the capability is not exposed at all).
        stack: The stack-scope stamp minted onto every wake frame so a
            stack-guarded orchestrator consumes it. Blank fails closed at build.
    """

    triage_agent: str
    allowed_agents: frozenset[str]
    stack: str

    def __post_init__(self) -> None:
        """Reject a config that could mint undeliverable or self-referential wakes."""
        if not self.triage_agent or not self.triage_agent.strip():
            raise ValueError("triage_agent must be non-empty")
        if not self.stack or not self.stack.strip():
            raise ValueError("stack must be non-empty; there is no default stack id")
        if not self.allowed_agents:
            raise ValueError(
                "allowed_agents must be non-empty; an empty allow-list means the wake "
                "capability must not be exposed at all (do not construct WakeService)"
            )
        if self.triage_agent in self.allowed_agents:
            raise ValueError(
                f"triage agent {self.triage_agent!r} must not be in its own wake allow-list "
                "(a self-wake is always refused)"
            )
        # Every allowed id becomes a NATS subject token (``rtech.agent.<id>.task``)
        # AND a ``TaskFrame.participants`` entry, so each MUST be subject-safe. An
        # operator id with a ``.`` / space would pass the allow-list ``in`` check
        # yet raise inside ``TaskFrame`` construction (and could target a wrong /
        # undeliverable subject) — so it fails CLOSED here at config build, not
        # mid-turn. ``validate_participants`` is the one shared wire-safety check.
        for candidate in sorted(self.allowed_agents):
            try:
                validate_participants([candidate], "single")
            except ValueError as exc:
                raise ValueError(
                    f"wake allow-list id {candidate!r} is not a safe NATS subject token: {exc}"
                ) from exc
            # Subject-safe is necessary but not sufficient: a deployed agentd only
            # subscribes ``rtech.agent.<agentd-name>.task`` (deploy convention), so
            # an id like ``bilby`` would target a subject nobody listens on. Enforce
            # the ``agentd-<name>`` convention so a mis-set allow-list fails loud at
            # build, not as a silently-vanished wake mid-turn.
            if _ALLOWED_AGENT_CONVENTION.match(candidate) is None:
                raise ValueError(
                    f"wake allow-list id {candidate!r} does not follow the required "
                    f"{ENV_WAKE_ALLOWED_AGENTS} convention: every id must match "
                    r"'^agentd-[a-z0-9][a-z0-9-]*$' (e.g. 'agentd-bilby')"
                )

    @classmethod
    def from_env(
        cls,
        *,
        triage_agent: str,
        stack: str,
        env: dict[str, str] | None = None,
    ) -> WakeConfig | None:
        """Build a wake config from ``RTECH_WAKE_ALLOWED_AGENTS``, or ``None``.

        Returns ``None`` when the allow-list env is unset OR resolves to an empty
        list after stripping blanks and this agent's own id — that is the signal
        the caller must NOT expose the wake capability. A present allow-list with
        at least one other id yields a validated config (fail-loud on a blank
        stack/triage id via ``__post_init__``).

        Args:
            triage_agent: This agent's id (self-wakes are stripped from the list).
            stack: The stack-scope stamp for minted wake frames.
            env: Environment mapping (defaults to ``os.environ``).

        Returns:
            A validated :class:`WakeConfig`, or ``None`` to signal "do not expose".
        """
        source = os.environ if env is None else env
        raw = source.get(ENV_WAKE_ALLOWED_AGENTS)
        if raw is None or not raw.strip():
            logger.info(
                "wake.config | {} unset/empty; wake capability NOT exposed | agent={}",
                ENV_WAKE_ALLOWED_AGENTS,
                triage_agent,
            )
            return None
        # Parse the comma list: strip blanks, drop this agent's own id (a self-wake
        # is never allowed), de-dup while preserving order for a stable log.
        parsed: list[str] = []
        for token in raw.split(","):
            candidate = token.strip()
            if not candidate or candidate == triage_agent or candidate in parsed:
                continue
            parsed.append(candidate)
        if not parsed:
            logger.warning(
                "wake.config | {} resolved to an empty allow-list after stripping blanks/self; "
                "wake capability NOT exposed | agent={} raw_len={}",
                ENV_WAKE_ALLOWED_AGENTS,
                triage_agent,
                len(raw),
            )
            return None
        logger.info(
            "wake.config | wake capability ENABLED | agent={} allowed={}",
            triage_agent,
            parsed,
        )
        return cls(triage_agent=triage_agent, allowed_agents=frozenset(parsed), stack=stack)


@dataclass(frozen=True)
class WakeResult:
    """A successful wake: the minted teammate task id, returned to the engine."""

    agent_id: str
    task_id: str
    ok: Literal[True] = True


@dataclass(frozen=True)
class WakeError:
    """A typed wake failure returned to the engine (never raised into the turn).

    ``reason`` names the closed failure kind so the engine tool can report it back
    to the model without leaking an exception. ``detail`` is short human context.
    """

    reason: Literal["not_allowed", "self_wake", "blank_agent", "publish_failed"]
    detail: str
    ok: Literal[False] = False


WakeOutcome = WakeResult | WakeError


@dataclass
class WakeService:
    """Mint + publish a wake TaskFrame toward an allow-listed teammate.

    Owned by the agentd frame-turn/supervisor layer and exposed to the Claude
    engine via the in-process wake MCP tool. One instance per triage agentd; its
    mere existence (a non-``None`` :meth:`WakeConfig.from_env`) is the opt-in that
    this agent is a waker.

    Args:
        config: Validated wake config (triage id, allow-list, stack).
        bus: The typed-frame publish seam (the same one FrameTurnRunner uses).
        origin_provider: Supplies the ChannelOrigin of the task currently driving
            the engine, so the wake carries it verbatim; ``None`` when the
            triggering task had no origin.
        id_factory: Fresh task-id minter (uuid4 hex — the repo idiom).
        now: Wall-clock stamp source (injected for deterministic tests).
    """

    config: WakeConfig
    bus: FrameBus
    origin_provider: OriginProvider
    id_factory: IdFactory = field(default_factory=lambda: _new_task_id)
    now: Now = field(default_factory=lambda: _now_utc)

    async def wake(self, agent_id: str, task_text: str, reason: str) -> WakeOutcome:
        """Mint + publish a wake TaskFrame for ``agent_id``; return a typed outcome.

        Guards, in order (each returns a typed :class:`WakeError`, never raises):
        the id must be non-blank; it must not be THIS agent (self-wake); it must be
        in the allow-list. A surviving wake mints a fresh frame carrying the
        triggering task's origin verbatim and publishes it to the teammate's task
        subject; a publish failure is logged once and returned as a typed error.

        Args:
            agent_id: The ``agentd-<name>`` teammate to wake.
            task_text: The question/task text the woken agent should answer.
            reason: Short free context for why the teammate is being woken (logged).

        Returns:
            A :class:`WakeResult` with the minted ``task_id`` on success, or a
            typed :class:`WakeError` (no publish) on any guard/publish failure.
        """
        target = agent_id.strip() if agent_id else ""
        if not target:
            logger.warning(
                "wake.service | blank target agent refused | triage={}", self.config.triage_agent
            )
            return WakeError(reason="blank_agent", detail="agent_id must be non-blank")
        if target == self.config.triage_agent:
            logger.warning(
                "wake.service | self-wake refused | triage={} target={}",
                self.config.triage_agent,
                target,
            )
            return WakeError(reason="self_wake", detail=f"{target} cannot wake itself")
        if target not in self.config.allowed_agents:
            logger.warning(
                "wake.service | non-allowed target refused | triage={} target={} allowed={}",
                self.config.triage_agent,
                target,
                sorted(self.config.allowed_agents),
            )
            return WakeError(
                reason="not_allowed",
                detail=f"{target} is not in this agent's wake allow-list",
            )

        text = (
            task_text
            if task_text and task_text.strip()
            else f"(woken by {self.config.triage_agent})"
        )
        task_id = self.id_factory()
        origin = self.origin_provider()
        # Mint AND publish inside one guard: the allow-list is subject-validated at
        # config build, so a construction error here is unexpected — but it must
        # still return a typed error, never crash the triage turn (invariant 5).
        try:
            frame = TaskFrame(
                id=task_id,
                origin_edge=f"{_WAKE_ORIGIN_EDGE_PREFIX}:{self.config.triage_agent}",
                requester=self.config.triage_agent,
                text=text,
                scope=_WAKE_SCOPE,
                participants=[target],
                stack=self.config.stack,
                created_at=self.now(),
                origin=origin,
            )
            subject = subjects.agent_task(target)
            # Use the PROPAGATING publish: publish_frame swallows a
            # raising NATS publish and returns normally, which would let the wake
            # tool report a false success with zero publication. publish_frame_checked
            # re-raises so a real transport failure becomes a typed publish_failed.
            await self.bus.publish_frame_checked(subject, frame)
        except Exception as exc:
            logger.error(
                "wake.service | mint/publish failed | triage={} target={} task_id={} "
                "subject={} reason={}",
                self.config.triage_agent,
                target,
                task_id,
                subjects.agent_task(target),
                exc,
            )
            return WakeError(reason="publish_failed", detail=f"wake of {target} failed")
        logger.info(
            "wake.service | woke teammate | triage={} target={} task_id={} has_origin={} why={}",
            self.config.triage_agent,
            target,
            task_id,
            origin is not None,
            reason,
        )
        return WakeResult(agent_id=target, task_id=task_id)


def _new_task_id() -> str:
    """Return a fresh opaque task id (uuid4 hex — the repo idiom)."""
    return uuid.uuid4().hex


def _now_utc() -> datetime:
    """Return the current UTC time (injected in tests for determinism)."""
    return datetime.now(UTC)


__all__ = [
    "ENV_WAKE_ALLOWED_AGENTS",
    "OriginProvider",
    "WakeConfig",
    "WakeError",
    "WakeOutcome",
    "WakeResult",
    "WakeService",
]
