"""Immutable domain types, protocols, and request-key/cache helpers for Start.

This sibling of :mod:`rtech_orchestrator.supervised_start` owns the pieces the
supervisor composes but does not itself orchestrate: the typed request/handler
aliases, the immutable :class:`KnownAgent`, the request-id-independent internal
:class:`_StartOutcome`, every server-owned dependency :class:`Protocol`, and the
principal-bound request-key and bounded completed-receipt-cache helpers.

Splitting these out keeps :mod:`supervised_start` below the file cap while it
gains the principal-bound in-flight keying and detached-task caching redesign.
Public imports from ``rtech_orchestrator.supervised_start`` stay
compatible: that module re-exports every name defined here.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, Protocol

from rtech_frames import (
    StartAgentFrame,
    StartAllAgentsFrame,
    SupervisedAgentStateFrame,
)
from rtech_frames.codec import Frame
from rtech_frames.frames import StartFailure

type StartRequest = StartAgentFrame | StartAllAgentsFrame
type FrameHandler = Callable[[Frame], Awaitable[None]]


@dataclass(frozen=True)
class _PrincipalIdentity:
    """Structurally distinguish a resolved principal from a resolver failure.

    A resolver that legitimately returns ``None`` (no authenticated principal on
    the transport, yet the resolve itself SUCCEEDED) and a resolver that RAISED
    both leave the authorizer with a ``None`` principal — but they are NOT the same
    request identity. Keying the in-flight/completed maps on the raw ``str | None``
    principal alone would collapse them onto one key, so a legitimately-resolved
    ``None`` that authorized and cached ``Online`` could be REPLAYED by a later
    resolver FAILURE for the same request/agent without re-authorization (finding
    P1). This immutable identity binds an explicit ``resolved`` flag alongside the
    principal so ``resolved(None)`` and ``resolution_failed()`` mint DISTINCT keys.

    The authorizer never sees this identity — only the raw resolved ``principal``
    value — so the trust seam contract is unchanged; the identity is an internal
    keying concern only.
    """

    principal: str | None
    resolved: bool

    @classmethod
    def resolved_as(cls, principal: str | None) -> _PrincipalIdentity:
        """Identity for a principal the resolver SUCCESSFULLY produced (may be None)."""
        return cls(principal=principal, resolved=True)

    @classmethod
    def resolution_failed(cls) -> _PrincipalIdentity:
        """Identity for a resolver that RAISED: fails closed, distinct from resolved(None)."""
        return cls(principal=None, resolved=False)


# The principal-bound request key that identifies ONE logical Start request. It
# binds the trusted principal IDENTITY (not the raw ``str | None``), so a
# legitimately-resolved ``None`` and a resolver FAILURE mint DISTINCT keys and a
# resolver failure can never replay a resolved-None receipt (finding P1). Two
# concurrent principals A and B sharing a request-id/agent also NEVER share one
# in-flight authorization result — each resolves and authorizes independently. The
# SAME key is reused for both the in-flight dedup map and the completed receipt
# cache, so at-most-one terminal publication holds across concurrent AND
# sequential redelivery for that principal-bound pair.
type RequestKey = tuple[_PrincipalIdentity, str, str]

# Bounded completed-request receipt cache size (finding request-pair idempotency).
# A terminal receipt for a :data:`RequestKey` ``(principal, request_id, agent_id)``
# is recorded so a later SEQUENTIAL redelivery returns the one cached terminal
# instead of republishing. Small explicit cap; deterministic oldest-first (FIFO)
# eviction, never time- or hash-based. Receipt idempotency, NOT RBAC — no
# enterprise machinery.
COMPLETED_RECEIPT_CACHE_CAP: Final[int] = 256


def request_key(identity: _PrincipalIdentity, request_id: str, agent_id: str) -> RequestKey:
    """Build the principal-bound key for one logical Start request.

    The key BINDS the trusted principal IDENTITY (which carries both the resolved
    principal value and the resolved/failed distinction), never the frame's
    self-asserted ``requester_id``. A cached or in-flight decision for principal A
    can never grant principal B authority, because B resolves its own principal and
    mints its own distinct key; and a resolver FAILURE mints an identity distinct
    from a legitimately-resolved ``None``, so a failure can never replay a
    resolved-None receipt (findings P1 trust, request-pair idempotency).
    """
    return (identity, request_id, agent_id)


def record_completed(
    completed: OrderedDict[RequestKey, SupervisedAgentStateFrame],
    key: RequestKey,
    terminal: SupervisedAgentStateFrame,
) -> None:
    """Record one terminal receipt, bounded FIFO oldest-first at the cap.

    The completed cache stays at :data:`COMPLETED_RECEIPT_CACHE_CAP`; the OLDEST
    inserted key is evicted first (deterministic, never time- or hash-based).
    """
    completed[key] = terminal
    completed.move_to_end(key)
    while len(completed) > COMPLETED_RECEIPT_CACHE_CAP:
        completed.popitem(last=False)


@dataclass(frozen=True)
class KnownAgent:
    """Name one server-configured agent that an external owner may start."""

    agent_id: str
    service_ref: str


@dataclass(frozen=True)
class _StartOutcome:
    """Carry the request-id-independent verdict of one shared start machine.

    The shared per-agent start machine (:meth:`ExternalStartSupervisor._run_start`)
    computes exactly one of these. It is deliberately NOT a
    :class:`SupervisedAgentStateFrame`: a single shared process start may be
    awaited by many distinct requests, and each of those requests must mint and
    publish its OWN terminal receipt carrying ITS OWN ``request_id`` (finding P2).
    Threading a frame (which pins one ``request_id``) out of the shared machine
    would mis-attribute every other waiter's receipt to the first requester.

    ``online`` is ``True`` only when the process start was affirmatively confirmed
    alive by the authoritative :class:`LivenessProbe`; ``failure`` is the typed
    terminal code otherwise. Exactly one of (``online``, ``failure``) is set.
    """

    online: bool
    failure: StartFailure | None


class StartAuthorizer(Protocol):
    """Authorize a Start request against a server-resolved trusted principal.

    The authorizer receives ONLY the server-resolved trusted ``principal`` — the
    identity produced by the trusted server-side transport boundary (the NATS
    publisher boundary), never the request frame. It is ``None`` when the
    transport supplied no authenticated principal. The authorizer is structurally
    incapable of reading the frame's self-asserted ``requester_id``, because it is
    never handed the frame: trust can never rest on a forgeable frame field
    (finding P1). This product has no per-agent RBAC requirement, so the contract
    is deliberately principal-only — no request scope or capability metadata is
    passed.
    """

    async def authorize(self, principal: str | None) -> bool:
        """Return whether the resolved principal may invoke the supervisor."""
        ...


class PrincipalResolver(Protocol):
    """Resolve the trusted server-side principal for the inbound transport.

    This is the seam where an authenticated NATS publisher identity (when the
    transport can supply one) crosses into the supervisor. It receives NO request
    frame: the trusted principal comes from the transport context alone, never
    from a frame body an untrusted publisher controls (finding P1). It is
    structurally incapable of reading ``requester_id`` because it is never handed
    the request. A resolver may return ``None`` to mean "no authenticated
    principal available", leaving authorization to rest on the trusted-publisher
    lane and the server-side authorizer's own policy.
    """

    async def resolve(self) -> str | None:
        """Return the trusted transport principal, or ``None``."""
        ...


class StartGuard(Protocol):
    """Apply the server-side guard before a configured agent is started."""

    async def allows(self, agent: KnownAgent) -> bool:
        """Return whether this known agent is currently safe to start."""
        ...


class ExternalStarter(Protocol):
    """Start a known service without exposing a command to the request frame."""

    async def start(self, agent: KnownAgent) -> None:
        """Ask the external process owner to start the configured service."""
        ...


class LivenessProbe(Protocol):
    """Report authoritative server-side liveness for a known agent.

    This is the ONLY source of an "Online" claim — both the "already alive"
    short-circuit AND the post-start confirmation. The supervisor never
    synthesizes liveness from its own start history: a process whose
    :meth:`ExternalStarter.start` action returned must still report Online only
    if this authoritative probe confirms it is currently alive. ``start``
    returning is an action receipt, not proof of readiness (finding P1 liveness).
    """

    async def is_online(self, agent: KnownAgent) -> bool:
        """Return whether the agent is authoritatively alive right now."""
        ...


class SupervisedStartBus(Protocol):
    """Carry typed Start requests and external-supervisor state receipts."""

    async def publish_frame(self, subject: str, frame: SupervisedAgentStateFrame) -> None:
        """Publish a typed state frame to its explicitly selected lane."""
        ...

    async def subscribe_frame(self, subject: str, handler: FrameHandler) -> object:
        """Subscribe a typed request handler to a fixed ingress lane."""
        ...
