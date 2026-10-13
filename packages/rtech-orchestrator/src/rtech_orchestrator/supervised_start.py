"""Own authenticated Start-only requests at the external-supervisor boundary."""

from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict
from collections.abc import Callable, Sequence

from loguru import logger
from rtech_frames import (
    StartAgentFrame,
    StartAllAgentsFrame,
    SupervisedAgentStateFrame,
)
from rtech_frames.codec import Frame
from rtech_frames.frames import StartFailure

from . import subjects
from .supervised_start_state import (
    COMPLETED_RECEIPT_CACHE_CAP,
    ExternalStarter,
    FrameHandler,
    KnownAgent,
    LivenessProbe,
    PrincipalResolver,
    RequestKey,
    StartAuthorizer,
    StartGuard,
    StartRequest,
    SupervisedStartBus,
    _PrincipalIdentity,
    _StartOutcome,
    record_completed,
    request_key,
)

# Re-export the domain surface so existing importers of
# ``rtech_orchestrator.supervised_start`` keep working after the split.
# The owning definitions now live in :mod:`supervised_start_state`.
__all__ = [
    "COMPLETED_RECEIPT_CACHE_CAP",
    "ExternalStartSupervisor",
    "ExternalStarter",
    "FrameHandler",
    "KnownAgent",
    "LivenessProbe",
    "PrincipalResolver",
    "StartAuthorizer",
    "StartGuard",
    "StartRequest",
    "SupervisedStartBus",
]


# Short correlation-token width (hex chars) for a request_id in logs. The
# ``request_id`` is caller-supplied text, so it MUST NOT be interpolated raw into a
# log line (terminal-audit finding: logs are a possibly-leaked surface and the raw
# id can carry caller-controlled content). Instead every log site emits a
# NON-REVERSIBLE deterministic token derived purely from the id via SHA-256 and
# truncated to this width — enough entropy to correlate the receipt lifecycle of
# one request across log lines without ever exposing the id itself.
_REQUEST_TOKEN_LEN = 12


def _request_token(request_id: str) -> str:
    """Derive a short non-reversible correlation token from a request_id.

    The token is a truncated hex SHA-256 digest of the already-bounded
    ``request_id`` and NOTHING else — it never includes caller text, agent ids, or
    any other field. It is deterministic (the same id always yields the same token,
    so one request's lifecycle correlates across log lines) and non-reversible (a
    digest cannot be turned back into the caller-supplied id), so it is safe to emit
    on a possibly-leaked log surface where the raw id must never appear
    (terminal-audit finding).
    """
    return hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:_REQUEST_TOKEN_LEN]


class ExternalStartSupervisor:
    """Start known offline agents through authenticated, guarded typed requests.

    The class deliberately owns only Start. It receives typed request frames,
    resolves the immutable known-agent registry locally, and delegates the
    actual process action to an external starter. Agentd receives neither a
    restart command nor a mutable configuration path.

    Trust boundary (finding P1 trust): the :class:`StartAuthorizer`
    is handed ONLY a server-resolved trusted ``principal`` and the
    :class:`PrincipalResolver` is handed NO request frame, so neither trust seam
    can read the frame's forgeable ``requester_id``. Every principal resolves and
    authorizes INDEPENDENTLY: the in-flight key and the completed cache both bind
    the trusted principal IDENTITY (:class:`_PrincipalIdentity`, which structurally
    separates a legitimately-resolved ``None`` from a resolver failure), so
    concurrent principals A and B sharing one request-id/agent never share one
    authorization result — one may succeed while the other is denied — and a
    resolver FAILURE can never replay a resolved-``None`` receipt, while the
    per-agent process start stays shared only across authorized requests.

    Liveness boundary (finding P1 liveness):
    :meth:`ExternalStarter.start` RETURNING is only an action receipt, never proof
    of Online. A successful start action is followed by exactly one authoritative
    post-start :class:`LivenessProbe` confirmation; ``Online`` publishes only when
    that probe affirmatively confirms alive. A missing/failing/DOWN probe yields
    the typed ``liveness_unconfirmed`` terminal failure — never a false ``Online``.
    One bounded confirmation, no unbounded polling.

    Correlation contract (finding P2): start work is reserved per
    **agent id**, not per (request_id, agent_id), so two distinct requests for the
    same agent — including a bulk request racing a single one — share the one
    in-flight start machine and never start the agent twice. The machine yields a
    request-id-INDEPENDENT :class:`_StartOutcome`, so every distinct waiting
    ``request_id`` mints and publishes its OWN terminal receipt with its own
    ``request_id``; the SAME ``request_id`` concurrently dedupes onto one receipt.

    Bounded-state and recovery contract: the in-flight map is pruned on every
    terminal outcome, so it stays bounded by the agents with a start in flight. A
    failed start is not retained, so a later request may safely retry it. A
    confirmed-and-online start whose per-request ``Online`` receipt has NOT yet
    published is recorded in a bounded ``_pending_online`` set ONLY to make
    publish-failure recovery safe: a retry consults the authoritative probe first,
    drops the stale claim on a DOWN probe and fails toward a real guarded start,
    and drops the agent once ``Online`` publishes — never a sticky "alive forever"
    claim. The shared start machine runs detached; waiters await it under a shield,
    so cancelling one waiter never cancels the shared machine.

    Request-pair idempotency (finding request-pair idempotency):
    ``_inflight_requests`` dedups CONCURRENT duplicates of a principal-bound pair
    but prunes on terminal, so a bounded completed-request receipt cache
    (``_completed``) closes the sequential-redelivery gap: EVERY terminal outcome
    for a pair — authorization, guard, unknown-agent, start, liveness, and publish
    failures, plus ``Online`` — is recorded keyed by the principal-bound
    ``(identity, request_id, agent_id)``, so the SAME pair publishes at most ONE
    terminal across concurrent AND sequential redelivery. The whole resolve-plus-
    dispatch lifecycle runs in a detached task the public caller merely shields, so
    cancelling the sole waiter while ``resolve()`` is still pending cannot skip
    resolution, terminal publication, or caching; the completed frame is recorded
    INSIDE the detached principal-bound request task before it returns, so a
    redelivery after cancellation returns the cached terminal without a second
    publish. One shared
    process start still remains per agent: the cache gates receipt REPUBLICATION,
    never the machine. It is bounded by :data:`COMPLETED_RECEIPT_CACHE_CAP` with
    deterministic oldest-first (FIFO) eviction. Principal-safety: the key binds the
    trusted principal, never ``requester_id``; the cache is consulted only for that
    principal's own key, so a cached decision from A can never grant B authority.
    """

    def __init__(
        self,
        *,
        known_agents: Sequence[KnownAgent],
        starter: ExternalStarter,
        bus: SupervisedStartBus,
        authorizer: StartAuthorizer | None = None,
        guard: StartGuard | None = None,
        liveness: LivenessProbe | None = None,
        principals: PrincipalResolver | None = None,
    ) -> None:
        """Initialize the Start-only supervisor with server-owned dependencies."""
        registry = {agent.agent_id: agent for agent in known_agents}
        if len(registry) != len(known_agents) or any(not agent.agent_id for agent in known_agents):
            raise ValueError("known agents must have unique non-empty ids")
        self._known_agents = registry
        self._starter = starter
        self._bus = bus
        self._authorizer = authorizer
        self._guard = guard
        self._liveness = liveness
        self._principals = principals
        self._lock = asyncio.Lock()
        self._inflight: dict[str, asyncio.Task[_StartOutcome]] = {}
        # Per-request receipt dedup keyed by the PRINCIPAL-bound
        # (principal, request_id, agent_id): concurrent duplicates of the SAME
        # request for the SAME agent under the SAME trusted principal share ONE
        # terminal receipt and never flood, while DISTINCT principals (or request
        # ids) each get their own independently-authorized correlated receipt
        # (findings P1, P2). Bounded by inflight distinct principal-bound requests.
        self._inflight_requests: dict[RequestKey, asyncio.Task[SupervisedAgentStateFrame]] = {}
        self._pending_online: set[str] = set()
        # Bounded completed-request receipt cache keyed by the same PRINCIPAL-bound
        # pair; ordered for deterministic oldest-first (FIFO) eviction at
        # COMPLETED_RECEIPT_CACHE_CAP. See the class docstring's request-pair
        # idempotency contract (findings idempotency, P1 trust).
        self._completed: OrderedDict[RequestKey, SupervisedAgentStateFrame] = OrderedDict()

    async def handle(self, request: StartRequest) -> tuple[SupervisedAgentStateFrame, ...]:
        """Dispatch one typed per-agent or bulk Start request."""
        if isinstance(request, StartAgentFrame):
            return (await self.start_agent(request),)
        return await self.start_all(request)

    async def start(self) -> None:
        """Subscribe to the fixed Start-only typed control-plane lanes."""
        await self._bus.subscribe_frame(subjects.SUPERVISED_START_INGRESS, self._on_frame)
        await self._bus.subscribe_frame(subjects.SUPERVISED_START_ALL_INGRESS, self._on_frame)
        logger.info("supervised_start.subscribe | ready")

    async def _on_frame(self, frame: Frame) -> None:
        """Accept only typed Start frames on this Start-only boundary."""
        if isinstance(frame, StartAgentFrame | StartAllAgentsFrame):
            await self.handle(frame)
            return
        logger.warning(
            "supervised_start.frame | ignored non-start frame type={}", type(frame).__name__
        )

    async def start_agent(self, request: StartAgentFrame) -> SupervisedAgentStateFrame:
        """Start one requested known agent, idempotently across redelivery.

        The WHOLE lifecycle — principal resolution, terminal computation, publish,
        and completed caching — runs inside a detached task that the public caller
        merely SHIELDS (finding P2). Cancelling the sole waiter during a still-
        pending ``resolve()`` therefore cannot leave the request without a terminal
        or a cache entry, so a redelivery returns the cache and never republishes.
        """
        return await asyncio.shield(
            asyncio.ensure_future(self._resolve_and_request(request.request_id, request.agent_id))
        )

    async def start_all(
        self, request: StartAllAgentsFrame
    ) -> tuple[SupervisedAgentStateFrame, ...]:
        """Start every server-configured agent, preserving per-agent idempotence.

        The trusted principal is resolved ONCE inside a single detached lifecycle
        task that the public caller shields, then each agent runs the
        principal-bound terminal path (finding P2). Cancelling the sole bulk waiter
        during resolution cannot cancel resolution, terminal publication, or
        completed caching.
        """
        return await asyncio.shield(
            asyncio.ensure_future(self._resolve_and_request_all(request.request_id))
        )

    async def _resolve_and_request(
        self, request_id: str, agent_id: str
    ) -> SupervisedAgentStateFrame:
        """Resolve the principal, then own the one per-agent terminal, detached.

        Concurrent duplicate calls may EACH resolve their own principal, but after
        resolution they converge on the SAME principal-bound in-flight request task
        (keyed by identity), so exactly one terminal is published and cached.
        """
        identity = await self._resolve_principal()
        return await self._request_terminal(identity, request_id, agent_id)

    async def _resolve_and_request_all(
        self, request_id: str
    ) -> tuple[SupervisedAgentStateFrame, ...]:
        """Resolve once, then own every per-agent terminal, detached."""
        agent_ids = tuple(self._known_agents)
        identity = await self._resolve_principal()
        return tuple(
            await asyncio.gather(
                *(self._request_terminal(identity, request_id, agent_id) for agent_id in agent_ids)
            )
        )

    async def _request_terminal(
        self,
        identity: _PrincipalIdentity,
        request_id: str,
        agent_id: str,
    ) -> SupervisedAgentStateFrame:
        """Return one principal-bound terminal receipt, deduped across redelivery.

        A pair that already reached a terminal for THIS trusted principal identity
        returns the ONE cached terminal — no second terminal is published across
        concurrent or sequential redelivery (any failure or success). Authorization
        is evaluated from the trusted principal and never short-circuited by a cache
        entry keyed to a DIFFERENT principal, nor by a resolver-FAILURE identity
        replaying a resolved-None receipt (findings request-pair idempotency, P1).

        The cache read is a fast pre-check; the detached request task itself records
        the completed frame BEFORE it returns, so cancelling the sole waiter here
        can never skip caching (finding cancellation-caching). Concurrent duplicates
        of the same principal-bound pair share the one in-flight task.
        """
        key = request_key(identity, request_id, agent_id)
        cached = self._completed.get(key)
        if cached is not None:
            logger.info(
                "supervised_start.idempotent | replay agent={} request_token={} state={}",
                agent_id,
                _request_token(request_id),
                cached.state,
            )
            return cached
        # Dedup the WHOLE terminal (authorization, guard, start, liveness, publish)
        # by the principal-bound key, so concurrent redelivery of a denied pair
        # shares ONE Failed receipt exactly like a started pair. Sequential
        # redelivery is served by the cache the detached task itself populated.
        return await self._run_deduped(key, identity, request_id, agent_id)

    async def _resolve_principal(self) -> _PrincipalIdentity:
        """Resolve the trusted transport principal into a keying identity.

        The resolver is handed NO request frame (finding P1). Returns a
        :class:`_PrincipalIdentity`: a successful resolve (even to ``None``) binds
        ``resolved_as``, while a resolver exception fails closed and binds the
        distinct ``resolution_failed`` identity. That distinction keeps a resolver
        FAILURE from ever sharing a completed-cache key with a legitimately-resolved
        ``None`` — so a failure cannot replay a resolved-None ``Online`` receipt
        (finding P1).
        """
        if self._principals is None:
            return _PrincipalIdentity.resolved_as(None)
        try:
            return _PrincipalIdentity.resolved_as(await self._principals.resolve())
        except Exception as exc:
            logger.error(
                "supervised_start.principal | failed error_type={}",
                type(exc).__name__,
            )
            return _PrincipalIdentity.resolution_failed()

    async def _authorize(self, identity: _PrincipalIdentity) -> StartFailure | None:
        """Fail closed when authorization is missing, denied, or unavailable.

        The authorizer is handed ONLY the pre-resolved trusted ``principal`` value,
        never the identity wrapper and never a request frame — neither trust seam
        can read the forgeable ``requester_id`` (finding P1). A resolver failure
        (``not identity.resolved``) fails closed: we cannot authorize an unknown
        principal.
        """
        if self._authorizer is None:
            logger.error("supervised_start.authorize | missing authorizer")
            return "auth_unavailable"
        if not identity.resolved:
            return "auth_unavailable"
        try:
            authorized = await self._authorizer.authorize(identity.principal)
        except Exception as exc:
            logger.error(
                "supervised_start.authorize | failed error_type={}",
                type(exc).__name__,
            )
            return "auth_unavailable"
        if not authorized:
            logger.warning("supervised_start.authorize | denied")
            return "unauthorized"
        return None

    async def _run_deduped(
        self,
        key: RequestKey,
        identity: _PrincipalIdentity,
        request_id: str,
        agent_id: str,
    ) -> SupervisedAgentStateFrame:
        """Dedup one principal-bound request, return its terminal receipt.

        Concurrent duplicates of the SAME principal-bound pair — including two
        detached lifecycle tasks that each resolved the same principal — CONVERGE on
        ONE per-request task and one terminal receipt, whether authorization
        failure, guard failure, or a started/Online outcome. DISTINCT principals (or
        request ids) each reserve their own task and independently-authorized
        correlated receipt (findings P1, P2). The task is shielded so cancelling one
        waiter never cancels the shared work, which always reaches a terminal AND
        records it in the completed cache before returning.
        """
        async with self._lock:
            existing_request = self._inflight_requests.get(key)
            if existing_request is not None:
                task = existing_request
            else:
                task = asyncio.create_task(self._run_request(key, identity, request_id, agent_id))
                self._inflight_requests[key] = task
                task.add_done_callback(self._make_request_pruner(key))
        return await asyncio.shield(task)

    def _make_request_pruner(
        self, key: RequestKey
    ) -> Callable[[asyncio.Task[SupervisedAgentStateFrame]], None]:
        """Build the done-callback that prunes the per-request entry on terminal."""

        def _prune(_task: asyncio.Task[SupervisedAgentStateFrame]) -> None:
            self._inflight_requests.pop(key, None)

        return _prune

    async def _run_request(
        self,
        key: RequestKey,
        identity: _PrincipalIdentity,
        request_id: str,
        agent_id: str,
    ) -> SupervisedAgentStateFrame:
        """Authorize, execute, publish, and RECORD one Start request's terminal.

        This detached task OWNS the whole principal-bound request lifecycle:
        authorization (evaluated INSIDE the task from the trusted principal so
        concurrent duplicates of a denied pair share ONE Failed receipt, finding
        P1), THIS request's own correlated terminal receipt (finding P2), and —
        critically — recording that terminal in the completed cache BEFORE
        returning. Because caching happens inside the shielded task, not after the
        caller's await, cancelling the sole waiter can never skip caching (finding
        cancellation-caching): a redelivery returns the cached terminal, no second
        publish.
        """
        terminal = await self._resolve_terminal(request_id, agent_id, identity)
        record_completed(self._completed, key, terminal)
        return terminal

    async def _resolve_terminal(
        self,
        request_id: str,
        agent_id: str,
        identity: _PrincipalIdentity,
    ) -> SupervisedAgentStateFrame:
        """Compute THIS request's terminal receipt on a bounded safe path.

        Every terminal path uses the bounded safe publication path so a broken bus
        returns a truthful ``state_publish_failed`` terminal instead of raising or
        retrying forever (finding safe-publish). The receipt is always returned,
        never raised, so the shielded waiter always gets a terminal tracked outcome.
        """
        failure = await self._authorize(identity)
        if failure is not None:
            return await self._safe_publish_failed(request_id, agent_id, failure)
        agent = self._known_agents.get(agent_id)
        if agent is None:
            return await self._safe_publish_failed(request_id, agent_id, "unknown_agent")
        if self._guard is None:
            logger.error("supervised_start.guard | missing guard agent={}", agent_id)
            return await self._safe_publish_failed(request_id, agent_id, "guard_unavailable")
        try:
            allowed = await self._guard.allows(agent)
        except Exception as exc:
            logger.error(
                "supervised_start.guard | failed agent={} error_type={}",
                agent_id,
                type(exc).__name__,
            )
            return await self._safe_publish_failed(request_id, agent_id, "guard_unavailable")
        if not allowed:
            logger.warning("supervised_start.guard | rejected agent={}", agent_id)
            return await self._safe_publish_failed(request_id, agent_id, "guard_rejected")

        already = await self._already_online(request_id, agent)
        if already is not None:
            return already

        # Publish THIS request's own Starting breadcrumb before joining the shared
        # machine. Every distinct request gets its own correlated Starting, and a
        # failed Starting publish is this request's own terminal outcome — no
        # process start is attempted because Starting never confirmed.
        starting = SupervisedAgentStateFrame(
            request_id=request_id,
            agent_id=agent.agent_id,
            state="Starting",
        )
        try:
            await self._publish_state(starting)
        except Exception as exc:
            logger.error(
                "supervised_start.publish | starting failed agent={} request_token={} "
                "error_type={}",
                agent.agent_id,
                _request_token(request_id),
                type(exc).__name__,
            )
            return await self._safe_publish_state_publish_failed(request_id, agent.agent_id)

        task = await self._reserve(agent)
        # Shield the shared machine: cancelling THIS waiter must not cancel the
        # detached start task, which always reaches a terminal outcome and prunes
        # itself. The outcome is request-id-independent, so THIS request mints and
        # publishes its own correlated terminal receipt from it.
        outcome = await asyncio.shield(task)
        return await self._publish_outcome(request_id, agent.agent_id, outcome)

    async def _publish_outcome(
        self, request_id: str, agent_id: str, outcome: _StartOutcome
    ) -> SupervisedAgentStateFrame:
        """Publish THIS request's own terminal receipt from a shared outcome.

        Every distinct waiter on a shared start machine mints its own terminal
        receipt with its own ``request_id`` (finding P2). The receipt is published
        bus-first on a bounded safe path; if the per-request publish fails, the
        typed ``state_publish_failed`` terminal receipt is still returned
        truthfully.
        """
        if outcome.failure is not None:
            return await self._safe_publish_failed(request_id, agent_id, outcome.failure)
        online = SupervisedAgentStateFrame(
            request_id=request_id,
            agent_id=agent_id,
            state="Online",
        )
        try:
            await self._publish_state(online)
        except Exception as exc:
            logger.error(
                "supervised_start.publish | online failed agent={} request_token={} error_type={}",
                agent_id,
                _request_token(request_id),
                type(exc).__name__,
            )
            # Bus-first terminal guarantee: the confirmed-alive start's Online
            # receipt failed, so the wire may not show Starting then silence.
            # Publish state_publish_failed on the safe path; ownership stays in
            # _pending_online so a retry re-confirms and re-publishes without a
            # duplicate start.
            async with self._lock:
                self._pending_online.add(agent_id)
            return await self._safe_publish_state_publish_failed(request_id, agent_id)
        # Online receipt is on the bus for THIS request: drop the confirmed-start
        # ownership so future liveness is the probe's job, never a sticky claim.
        async with self._lock:
            self._pending_online.discard(agent_id)
        logger.info(
            "supervised_start.start | online agent={} request_token={}",
            agent_id,
            _request_token(request_id),
        )
        return online

    async def _probe_online(self, agent: KnownAgent) -> bool | None:
        """Return the authoritative liveness verdict, or ``None`` on failure.

        Liveness is NEVER synthesized from start history: only the injected
        :class:`LivenessProbe` may assert an agent is alive. ``None`` means the
        probe could not be consulted (missing probe or a probe exception) — the
        caller MUST treat that as unconfirmed and fail closed, never as alive.
        ``True``/``False`` are the probe's affirmative verdicts.
        """
        if self._liveness is None:
            return None
        try:
            return await self._liveness.is_online(agent)
        except Exception as exc:
            logger.error(
                "supervised_start.liveness | failed agent={} error_type={}",
                agent.agent_id,
                type(exc).__name__,
            )
            return None

    async def _already_online(
        self, request_id: str, agent: KnownAgent
    ) -> SupervisedAgentStateFrame | None:
        """Return an authoritative already-online receipt, or ``None`` to start.

        Liveness is never synthesized from start history: only the injected
        :class:`LivenessProbe` reporting affirmatively alive short-circuits into an
        already-online receipt. Without a probe, on a probe failure, or when the
        probe reports not-alive, this returns ``None`` and the caller proceeds to
        the (deduplicated) start path. A probe failure fails toward starting: it is
        safer to attempt a guarded, deduplicated start than to falsely claim a
        crashed agent is Online.
        """
        if await self._probe_online(agent) is not True:
            return None
        online_frame = SupervisedAgentStateFrame(
            request_id=request_id,
            agent_id=agent.agent_id,
            state="Online",
        )
        try:
            await self._publish_state(online_frame)
        except Exception as exc:
            # Pre-start probe confirmed Online but the already-online receipt failed
            # to publish: publish the SAME single bounded state_publish_failed
            # terminal on the safe path and return it truthfully — never raise a raw
            # transport error out through this fast path (finding already-online).
            logger.error(
                "supervised_start.publish | already-online failed agent={} request_token={} "
                "error_type={}",
                agent.agent_id,
                _request_token(request_id),
                type(exc).__name__,
            )
            return await self._safe_publish_state_publish_failed(request_id, agent.agent_id)
        return online_frame

    async def _reserve(self, agent: KnownAgent) -> asyncio.Task[_StartOutcome]:
        """Reserve the one in-flight start task for this agent id.

        Reservation is keyed on ``agent.agent_id`` alone, so distinct request
        ids — including a bulk request racing a single one — dedupe onto the same
        detached task and never start the agent twice. The task yields a
        request-id-independent :class:`_StartOutcome`; each waiter publishes its
        own correlated receipt (finding P2).
        """
        async with self._lock:
            existing = self._inflight.get(agent.agent_id)
            if existing is not None:
                return existing
            task = asyncio.create_task(self._run_start(agent))
            self._inflight[agent.agent_id] = task
            # Prune on terminal outcome so the map stays bounded and a completed
            # (including failed) start does not block a later safe retry.
            task.add_done_callback(self._make_pruner(agent.agent_id))
            return task

    def _make_pruner(self, agent_id: str) -> Callable[[asyncio.Task[_StartOutcome]], None]:
        """Build the done-callback that prunes the in-flight entry for one agent."""

        def _prune(_task: asyncio.Task[_StartOutcome]) -> None:
            self._inflight.pop(agent_id, None)

        return _prune

    async def _run_start(self, agent: KnownAgent) -> _StartOutcome:
        """Run one shared start action, confirm liveness, and return the outcome.

        This is the request-id-INDEPENDENT shared machine (finding P2): it returns
        a :class:`_StartOutcome` and publishes NO frame at all — each waiting
        request mints its own correlated ``Starting`` and terminal receipt from it,
        so no waiter's receipt is mis-attributed to another's ``request_id``.

        Liveness (finding P1 liveness): :meth:`ExternalStarter.start` RETURNING is
        only an action receipt. The start is followed by exactly one authoritative
        post-start :class:`LivenessProbe` confirmation; the outcome is Online only
        when the probe confirms alive. A missing/failing/DOWN probe yields a
        ``liveness_unconfirmed`` outcome — never a false Online.
        """
        # A prior attempt whose Online receipt failed to publish left the start
        # confirmed and pending its receipt. Pending ownership is NOT itself a
        # liveness claim: consult the authoritative probe first (finding P1).
        # Re-confirm Online without re-starting only when the probe confirms alive;
        # a DOWN/absent/failing probe drops the unverifiable pending claim and falls
        # toward a real, guarded start rather than synthesize Online.
        if agent.agent_id in self._pending_online:
            if await self._probe_online(agent) is True:
                return _StartOutcome(online=True, failure=None)
            async with self._lock:
                self._pending_online.discard(agent.agent_id)

        try:
            await self._starter.start(agent)
        except Exception as exc:
            logger.error(
                "supervised_start.start | failed agent={} error_type={}",
                agent.agent_id,
                type(exc).__name__,
            )
            return _StartOutcome(online=False, failure="start_failed")

        # start() RETURNING is only an action receipt — NOT proof of Online. A
        # single authoritative post-start liveness confirmation gates the Online
        # outcome (finding P1 liveness). Missing/failing/DOWN liveness fails closed
        # with the typed ``liveness_unconfirmed`` code — never a false Online.
        if await self._probe_online(agent) is not True:
            logger.warning("supervised_start.liveness | unconfirmed agent={}", agent.agent_id)
            return _StartOutcome(online=False, failure="liveness_unconfirmed")

        # Record the confirmed-and-alive start BEFORE any waiter publishes Online,
        # so a per-request publish failure cannot lose the confirmed start and a
        # retry never double-starts.
        async with self._lock:
            self._pending_online.add(agent.agent_id)
        return _StartOutcome(online=True, failure=None)

    async def _safe_publish_failed(
        self, request_id: str, agent_id: str, failure: StartFailure
    ) -> SupervisedAgentStateFrame:
        """Publish a typed failure on a bounded path that never raises a receipt.

        A waiter MUST always return a terminal tracked outcome (finding
        6) rather than raise it out through the shielded waiter. Every terminal
        failure path — authorization, unknown-agent, guard, start, and liveness —
        publishes through here, so a broken bus degrades to a truthful typed
        ``state_publish_failed`` frame recorded once in the completed cache, never a
        raised or retried-forever raw exception (finding safe-publish). On a failed
        publish it makes exactly ONE further bounded attempt and does not recurse.
        """
        frame = SupervisedAgentStateFrame(
            request_id=request_id,
            agent_id=agent_id,
            state="Failed",
            failure=failure,
        )
        try:
            await self._publish_state(frame)
        except Exception as exc:
            logger.error(
                "supervised_start.publish | failed receipt agent={} error_type={}",
                agent_id,
                type(exc).__name__,
            )
            return await self._safe_publish_state_publish_failed(request_id, agent_id)
        return frame

    async def _safe_publish_state_publish_failed(
        self, request_id: str, agent_id: str
    ) -> SupervisedAgentStateFrame:
        """Publish the terminal ``state_publish_failed`` receipt, bus-first.

        A prior state publish (``Starting``, ``Online``, a typed failure, or an
        already-online receipt) failed, so the wire may not be left showing that
        state then silence. This makes ONE bounded attempt to publish the typed
        terminal and does NOT recurse if that also fails: it returns the same
        truthful frame without re-attempting, so a broken transport cannot loop.
        """
        frame = SupervisedAgentStateFrame(
            request_id=request_id,
            agent_id=agent_id,
            state="Failed",
            failure="state_publish_failed",
        )
        try:
            await self._publish_state(frame)
        except Exception as exc:
            logger.error(
                "supervised_start.publish | terminal receipt failed agent={} error_type={}",
                agent_id,
                type(exc).__name__,
            )
        return frame

    async def _publish_state(self, frame: SupervisedAgentStateFrame) -> None:
        """Publish state only to the supervisor-owned per-agent NATS lane."""
        await self._bus.publish_frame(subjects.agent_supervised_state(frame.agent_id), frame)
