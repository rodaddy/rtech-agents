"""Request-pair idempotency and the already-online publish-failure fast path.

The same logical (principal, request_id, agent) pair publishes AT MOST ONE terminal
receipt across concurrent AND sequential redelivery — for authorization, guard,
unknown-agent, start, liveness, and publish failures as well as ``Online`` — while
one shared process start remains per agent. The bounded completed-request cache is
principal-bound, so a cached decision for one principal never grants another
principal authority. An affirmative pre-start liveness probe whose already-online
publish then fails uses the same single bounded ``state_publish_failed`` terminal
and returns truthfully. Shared doubles come from the ``supervised_start_kit``
fixture in ``conftest``.
"""

from __future__ import annotations

import asyncio

import pytest
from rtech_frames import StartAllAgentsFrame
from rtech_orchestrator.supervised_start import (
    COMPLETED_RECEIPT_CACHE_CAP,
    _PrincipalIdentity,
)

from .conftest import SupervisedStartKit


@pytest.mark.asyncio
async def test_sequential_redelivery_of_online_pair_publishes_one_terminal(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """A sequential redelivery of a completed Online pair republishes NOTHING.

    The first delivery starts, confirms, and publishes ``Online``; an identical
    redelivery of the SAME pair returns the ONE cached terminal without a second
    process start and without a second terminal on the bus.
    """
    kit = supervised_start_kit
    supervisor, starter, publisher = kit.supervisor(authorizer=kit.Authorizer(), guard=kit.Guard())

    first = await supervisor.start_agent(kit.request(request_id="req-1"))
    for _ in range(4):
        await asyncio.sleep(0)
    second = await supervisor.start_agent(kit.request(request_id="req-1"))

    assert first.state == "Online"
    # The cached terminal is returned verbatim — no re-start, no second receipt.
    assert second is first
    assert [agent.agent_id for agent in starter.started] == ["agentd-bilby"]
    online = [f for f in publisher.frames if f.state == "Online"]
    assert len(online) == 1
    starting = [f for f in publisher.frames if f.state == "Starting"]
    assert len(starting) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "failure"),
    [
        ("unauthorized", "unauthorized"),
        ("auth_unavailable", "auth_unavailable"),
        ("guard_rejected", "guard_rejected"),
        ("guard_unavailable", "guard_unavailable"),
        ("unknown_agent", "unknown_agent"),
        ("start_failed", "start_failed"),
        ("liveness_unconfirmed", "liveness_unconfirmed"),
    ],
)
async def test_sequential_redelivery_of_failure_pair_publishes_one_terminal(
    case: str,
    failure: str,
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Every terminal FAILURE kind is republished at most once on redelivery.

    Authorization, guard, unknown-agent, start, and liveness failures each publish
    exactly one Failed terminal on the first delivery; a sequential redelivery of
    the same pair returns the ONE cached terminal and never republishes.
    """
    kit = supervised_start_kit
    agent_id = "agentd-bilby"
    authorizer: object | None = kit.Authorizer()
    guard = kit.Guard()
    starter = kit.Starter()
    liveness: object | None = None

    if case == "unauthorized":
        authorizer = kit.Authorizer(allowed=False)
    elif case == "auth_unavailable":
        authorizer = None
    elif case == "guard_rejected":
        guard = kit.Guard(allowed=False)
    elif case == "guard_unavailable":
        guard = None  # type: ignore[assignment]
    elif case == "unknown_agent":
        agent_id = "agentd-unknown"
    elif case == "start_failed":
        starter = kit.Starter(fail=True)
        liveness = kit.ConfirmingLiveness(starter)
    elif case == "liveness_unconfirmed":
        liveness = kit.NeverAliveLiveness()

    supervisor, _starter, publisher = kit.supervisor(
        authorizer=authorizer, guard=guard, starter=starter, liveness=liveness
    )

    first = await supervisor.start_agent(kit.request(request_id="req-1", agent_id=agent_id))
    for _ in range(4):
        await asyncio.sleep(0)
    second = await supervisor.start_agent(kit.request(request_id="req-1", agent_id=agent_id))

    assert first.state == "Failed"
    assert first.failure == failure
    # Redelivery returns the ONE cached terminal — never a second terminal.
    assert second is first
    failed = [f for f in publisher.frames if f.state == "Failed" and f.failure == failure]
    assert len(failed) == 1


@pytest.mark.asyncio
async def test_sequential_redelivery_of_state_publish_failed_publishes_one_terminal(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """A publish-failure terminal is not republished on sequential redelivery.

    The confirmed-alive start's Online publish fails, leaving the ONE
    ``state_publish_failed`` terminal on the bus. A redelivery of the SAME pair
    returns the cached terminal — it neither re-starts nor publishes a second
    terminal — even though pending-online ownership remains for a DISTINCT-request
    probe-gated retry.
    """
    kit = supervised_start_kit
    starter = kit.Starter()
    publisher = kit.Publisher(fail_states=("Online",))
    supervisor, _starter, _publisher = kit.supervisor(
        authorizer=kit.Authorizer(),
        guard=kit.Guard(),
        starter=starter,
        publisher=publisher,
        liveness=kit.ConfirmingLiveness(starter),
    )

    first = await supervisor.start_agent(kit.request(request_id="req-1"))
    for _ in range(4):
        await asyncio.sleep(0)
    second = await supervisor.start_agent(kit.request(request_id="req-1"))

    assert first.failure == "state_publish_failed"
    assert second is first
    # Exactly one terminal Failed publish attempt for this pair across redelivery.
    failed_attempts = [f for f in publisher.attempts if f.state == "Failed"]
    assert len(failed_attempts) == 1
    assert [agent.agent_id for agent in starter.started] == ["agentd-bilby"]


@pytest.mark.asyncio
async def test_concurrent_and_sequential_redelivery_share_one_terminal(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Concurrent duplicates AND a later redelivery all resolve to one terminal.

    Two concurrent deliveries of the same pair share one in-flight task and one
    terminal; a third, sequential redelivery after they drain returns the same
    cached terminal. One process start; one Online terminal on the bus.
    """
    kit = supervised_start_kit
    starter = kit.Starter(delay=0.02)
    supervisor, _starter, publisher = kit.supervisor(
        authorizer=kit.Authorizer(), guard=kit.Guard(), starter=starter
    )

    first, duplicate = await asyncio.gather(
        supervisor.start_agent(kit.request(request_id="req-1")),
        supervisor.start_agent(kit.request(request_id="req-1")),
    )
    for _ in range(4):
        await asyncio.sleep(0)
    third = await supervisor.start_agent(kit.request(request_id="req-1"))

    assert first == duplicate == third
    assert [agent.agent_id for agent in starter.started] == ["agentd-bilby"]
    assert len([f for f in publisher.frames if f.state == "Online"]) == 1


@pytest.mark.asyncio
async def test_cache_is_principal_bound_and_never_grants_cross_principal_authority(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """A cached decision for one principal never authorizes a DIFFERENT principal.

    Principal A is authorized and its Online terminal is cached under A's key. A
    replay of A's SAME ``request_id`` under a DIFFERENT trusted principal B (who is
    NOT authorized) misses A's key, re-authorizes from B's own principal, and is
    denied — a cached grant for A can never leak authority to B.
    """
    kit = supervised_start_kit

    class _PerPrincipalAuthorizer:
        """Authorize only the one trusted principal; deny everyone else."""

        def __init__(self, allowed_principal: str) -> None:
            self.allowed_principal = allowed_principal
            self.principals: list[str | None] = []

        async def authorize(self, principal: str | None) -> bool:
            self.principals.append(principal)
            return principal == self.allowed_principal

    class _SwitchableResolver:
        """Resolve whichever trusted principal is currently set on the transport."""

        def __init__(self, principal: str) -> None:
            self.principal = principal

        async def resolve(self) -> str | None:
            return self.principal

    authorizer = _PerPrincipalAuthorizer(allowed_principal="operator-a")
    resolver = _SwitchableResolver("operator-a")
    supervisor, starter, publisher = kit.supervisor(
        authorizer=authorizer,  # type: ignore[arg-type]
        guard=kit.Guard(),
        principals=resolver,  # type: ignore[arg-type]
    )

    # Principal A is authorized; its Online terminal is cached under A's key.
    a = await supervisor.start_agent(kit.request(request_id="shared-req"))
    assert a.state == "Online"

    # The SAME request_id now arrives under a different trusted principal B who is
    # not authorized. The A-keyed cache entry must NOT grant B authority.
    resolver.principal = "operator-b"
    b = await supervisor.start_agent(kit.request(request_id="shared-req"))

    assert b.state == "Failed"
    assert b.failure == "unauthorized"
    # B was actually re-authorized from its OWN principal, not A's cached decision.
    assert authorizer.principals == ["operator-a", "operator-b"]
    # Exactly one real start (A's); B never started the agent.
    assert [agent.agent_id for agent in starter.started] == ["agentd-bilby"]
    assert ("Failed", "unauthorized") in {(f.state, f.failure) for f in publisher.frames}


@pytest.mark.asyncio
async def test_completed_receipt_cache_is_bounded_with_fifo_eviction(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """The completed-receipt cache is capped with deterministic oldest-first eviction.

    Recording more than :data:`COMPLETED_RECEIPT_CACHE_CAP` distinct terminals keeps
    the cache at the cap and evicts the OLDEST entry first (FIFO). An evicted pair's
    redelivery is no longer a cache hit; a still-cached recent pair is.
    """
    kit = supervised_start_kit
    supervisor, _starter, _publisher = kit.supervisor(
        authorizer=kit.Authorizer(), guard=kit.Guard()
    )

    # Record cap + 1 distinct unknown-agent terminals directly to exercise bounded
    # FIFO eviction without standing up thousands of real starts.
    overflow = COMPLETED_RECEIPT_CACHE_CAP + 1
    for i in range(overflow):
        await supervisor.start_agent(kit.request(request_id=f"req-{i}", agent_id="agentd-unknown"))

    assert len(supervisor._completed) == COMPLETED_RECEIPT_CACHE_CAP  # noqa: SLF001
    # Oldest-first eviction: the very first key is gone, the newest keys remain.
    # No resolver configured → the key binds the resolved-None identity.
    identity = _PrincipalIdentity.resolved_as(None)
    assert (identity, "req-0", "agentd-unknown") not in supervisor._completed  # noqa: SLF001
    assert (identity, f"req-{overflow - 1}", "agentd-unknown") in supervisor._completed  # noqa: SLF001


@pytest.mark.asyncio
async def test_already_online_publish_failure_returns_state_publish_failed_truthfully(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """An affirmative pre-start probe whose Online publish fails returns truthfully.

    The already-online fast path confirms the agent alive but its ``Online`` publish
    fails. It must use the SAME single bounded ``state_publish_failed`` terminal and
    return it — never raise a raw transport error — and never start the process.
    """
    kit = supervised_start_kit
    starter = kit.Starter()
    publisher = kit.Publisher(fail_states=("Online",))
    supervisor, _starter, _publisher = kit.supervisor(
        authorizer=kit.Authorizer(),
        guard=kit.Guard(),
        starter=starter,
        publisher=publisher,
        liveness=kit.Liveness(online=True),
    )

    result = await supervisor.start_agent(kit.request(request_id="req-1"))

    assert result.state == "Failed"
    assert result.failure == "state_publish_failed"
    assert result.request_id == "req-1"
    # The affirmative pre-start probe short-circuited the start: no process started.
    assert starter.started == []
    # Exactly one bounded terminal-receipt attempt landed on the bus, bus-first.
    failed = [f for f in publisher.frames if f.state == "Failed"]
    assert len(failed) == 1
    assert failed[0].failure == "state_publish_failed"


@pytest.mark.asyncio
async def test_already_online_publish_failure_terminal_also_fails_returns_truthfully(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """When the fast-path terminal ALSO cannot publish, still fail truthfully once.

    The already-online Online publish fails and the ``state_publish_failed`` terminal
    itself cannot publish either. The supervisor returns the truthful typed failure
    with exactly ONE terminal-receipt attempt — no recursive failure loop, no raise.
    """
    kit = supervised_start_kit
    starter = kit.Starter()
    publisher = kit.Publisher(fail_states=("Online", "Failed"))
    supervisor, _starter, _publisher = kit.supervisor(
        authorizer=kit.Authorizer(),
        guard=kit.Guard(),
        starter=starter,
        publisher=publisher,
        liveness=kit.Liveness(online=True),
    )

    result = await supervisor.start_agent(kit.request(request_id="req-1"))

    assert result.state == "Failed"
    assert result.failure == "state_publish_failed"
    assert starter.started == []
    # Bounded: exactly one terminal Failed publish attempt (no recursion).
    failed_attempts = [f for f in publisher.attempts if f.state == "Failed"]
    assert len(failed_attempts) == 1
    assert publisher.frames == []


@pytest.mark.asyncio
async def test_concurrent_different_principals_authorize_independently(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Concurrent principals A and B on ONE request-id/agent never share authority.

    The in-flight key binds the trusted principal, so two concurrent deliveries of
    the SAME request_id and agent under DIFFERENT resolved principals do not share
    one in-flight authorization result. A (authorized) reaches Online while B
    (denied) reaches Failed unauthorized — one succeeds while the other is denied.
    The per-agent process start remains shared only across authorized requests, so
    exactly one real start happens (A's); B never started the agent.
    """
    kit = supervised_start_kit

    class _PerPrincipalAuthorizer:
        def __init__(self, allowed_principal: str) -> None:
            self.allowed_principal = allowed_principal
            self.principals: list[str | None] = []

        async def authorize(self, principal: str | None) -> bool:
            self.principals.append(principal)
            return principal == self.allowed_principal

    class _AlternatingResolver:
        """Hand out principal A on the first resolve, B on the second (by order)."""

        def __init__(self, principals: tuple[str, ...]) -> None:
            self._principals = principals
            self.calls = 0

        async def resolve(self) -> str | None:
            principal = self._principals[self.calls]
            self.calls += 1
            return principal

    authorizer = _PerPrincipalAuthorizer(allowed_principal="operator-a")
    resolver = _AlternatingResolver(("operator-a", "operator-b"))
    starter = kit.Starter(delay=0.02)
    supervisor, _starter, publisher = kit.supervisor(
        authorizer=authorizer,  # type: ignore[arg-type]
        guard=kit.Guard(),
        starter=starter,
        principals=resolver,  # type: ignore[arg-type]
        liveness=kit.ConfirmingLiveness(starter),
    )

    a, b = await asyncio.gather(
        supervisor.start_agent(kit.request(request_id="shared-req")),
        supervisor.start_agent(kit.request(request_id="shared-req")),
    )

    # Each principal independently resolved and authorized: A allowed, B denied.
    assert a.state == "Online"
    assert b.state == "Failed"
    assert b.failure == "unauthorized"
    assert authorizer.principals == ["operator-a", "operator-b"]
    # Distinct principal-bound keys — the pair was NOT shared across principals.
    key_a = (_PrincipalIdentity.resolved_as("operator-a"), "shared-req", "agentd-bilby")
    key_b = (_PrincipalIdentity.resolved_as("operator-b"), "shared-req", "agentd-bilby")
    assert key_a in supervisor._completed  # noqa: SLF001
    assert key_b in supervisor._completed  # noqa: SLF001
    # One shared process start (A's only); B contributed no start.
    assert [agent.agent_id for agent in starter.started] == ["agentd-bilby"]
    online = [f for f in publisher.frames if f.state == "Online"]
    assert len(online) == 1


@pytest.mark.asyncio
async def test_cancelled_sole_waiter_caches_terminal_and_redelivery_does_not_republish(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Cancelling the SOLE waiter still records the terminal; redelivery is cached.

    The detached principal-bound request task records the completed terminal BEFORE
    it returns, so cancelling the only waiter cannot prevent caching. A sequential
    redelivery of the same pair after cancellation returns the ONE cached terminal
    with NO second publish and NO second process start.
    """
    kit = supervised_start_kit
    starter = kit.Starter(delay=0.05)
    supervisor, _starter, publisher = kit.supervisor(
        authorizer=kit.Authorizer(), guard=kit.Guard(), starter=starter
    )

    # The SOLE waiter for this pair; cancel it while the shared start is in flight.
    waiter = asyncio.create_task(supervisor.start_agent(kit.request(request_id="req-1")))
    await asyncio.sleep(0.01)  # let the detached request + start tasks begin
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    # Let the detached (shielded) request task reach its terminal, publish, and
    # record the completed frame before redelivery. The shared start carries a real
    # delay, so wait past it, then drain the done-callbacks.
    await asyncio.sleep(0.1)
    for _ in range(8):
        await asyncio.sleep(0)

    # The cancelled pair's terminal was still published once and cached.
    online_first = [f for f in publisher.frames if f.state == "Online"]
    assert len(online_first) == 1
    assert ("req-1", "agentd-bilby") == (online_first[0].request_id, online_first[0].agent_id)
    key = (_PrincipalIdentity.resolved_as(None), "req-1", "agentd-bilby")
    assert key in supervisor._completed  # noqa: SLF001

    # A sequential redelivery returns the cached terminal — no second publish, no
    # second start.
    redelivered = await supervisor.start_agent(kit.request(request_id="req-1"))
    assert redelivered.state == "Online"
    assert redelivered is online_first[0]
    assert [agent.agent_id for agent in starter.started] == ["agentd-bilby"]
    assert len([f for f in publisher.frames if f.state == "Online"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "expected_pre_terminal"),
    [
        ("unauthorized", "unauthorized"),
        ("unknown_agent", "unknown_agent"),
        ("guard_rejected", "guard_rejected"),
        ("guard_unavailable", "guard_unavailable"),
    ],
)
async def test_broken_bus_pre_start_failure_returns_state_publish_failed_bounded(
    case: str,
    expected_pre_terminal: str,
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """A broken bus turns every pre-start failure into a bounded truthful terminal.

    Authorization, unknown-agent, and guard failures all publish through the
    bounded safe path. With a totally broken bus the raw publish raises, so the
    supervisor degrades to a truthful ``state_publish_failed`` terminal recorded
    once — never a raised raw exception and never a repeated/looping publish. The
    process is never started because the terminal is reached before any start.
    """
    kit = supervised_start_kit
    agent_id = "agentd-bilby"
    authorizer: object | None = kit.Authorizer()
    guard = kit.Guard()

    if case == "unauthorized":
        authorizer = kit.Authorizer(allowed=False)
    elif case == "unknown_agent":
        agent_id = "agentd-unknown"
    elif case == "guard_rejected":
        guard = kit.Guard(allowed=False)
    elif case == "guard_unavailable":
        guard = None  # type: ignore[assignment]

    publisher = kit.Publisher(fail_all=True)
    starter = kit.Starter()
    supervisor, _starter, _publisher = kit.supervisor(
        authorizer=authorizer, guard=guard, starter=starter, publisher=publisher
    )

    result = await supervisor.start_agent(kit.request(request_id="req-1", agent_id=agent_id))

    # Truthful terminal, never a raised raw exception from the broken bus.
    assert result.state == "Failed"
    assert result.failure == "state_publish_failed"
    assert result.request_id == "req-1"
    # No process start on a pre-start failure.
    assert starter.started == []
    # Bounded: the pre-start typed failure and the terminal receipt are each
    # attempted at most once — no repeated/looping publish on a broken bus.
    attempts = [f.failure for f in publisher.attempts if f.state == "Failed"]
    assert attempts == [expected_pre_terminal, "state_publish_failed"]
    # Nothing durably landed; the truthful terminal is still recorded once.
    assert publisher.frames == []
    key = (_PrincipalIdentity.resolved_as(None), "req-1", agent_id)
    assert key in supervisor._completed  # noqa: SLF001


@pytest.mark.asyncio
async def test_resolver_failure_cannot_replay_a_resolved_none_online_receipt(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Finding P1: a resolver FAILURE never replays a legitimate resolved-None Online.

    A resolver that FIRST legitimately resolves ``None`` (no authenticated principal
    on the transport, yet the resolve itself SUCCEEDED) authorizes and caches an
    ``Online`` terminal. When the resolver then RAISES on the SAME request/agent, the
    failure must mint a DISTINCT resolution-failed key — so redelivery fails closed
    with ``auth_unavailable`` instead of replaying the cached ``Online``. Keying on
    the raw ``None`` principal alone would collapse both onto one key and wrongly
    replay the Online receipt without re-authorization.
    """
    kit = supervised_start_kit

    class _NoneThenFailingResolver:
        """Resolve ``None`` successfully first, then RAISE on every later resolve."""

        def __init__(self) -> None:
            self.calls = 0

        async def resolve(self) -> str | None:
            self.calls += 1
            if self.calls == 1:
                return None
            raise RuntimeError("resolver down")

    # An authorizer that allows the resolved-None principal, so the first delivery
    # legitimately reaches Online and caches it.
    resolver = _NoneThenFailingResolver()
    supervisor, starter, publisher = kit.supervisor(
        authorizer=kit.Authorizer(allowed=True),
        guard=kit.Guard(),
        principals=resolver,  # type: ignore[arg-type]
    )

    first = await supervisor.start_agent(kit.request(request_id="req-1"))
    assert first.state == "Online"
    online_key = (_PrincipalIdentity.resolved_as(None), "req-1", "agentd-bilby")
    assert online_key in supervisor._completed  # noqa: SLF001

    # The resolver now RAISES for the SAME request/agent. The distinct
    # resolution-failed identity must NOT hit the resolved-None Online cache entry.
    second = await supervisor.start_agent(kit.request(request_id="req-1"))

    assert resolver.calls == 2
    assert second.state == "Failed"
    assert second.failure == "auth_unavailable"
    assert second is not first
    # The failed redelivery cached under its OWN distinct resolution-failed key.
    failed_key = (_PrincipalIdentity.resolution_failed(), "req-1", "agentd-bilby")
    assert failed_key in supervisor._completed  # noqa: SLF001
    # Exactly one real start (the first Online); the resolver failure started nothing.
    assert [agent.agent_id for agent in starter.started] == ["agentd-bilby"]
    assert len([f for f in publisher.frames if f.state == "Online"]) == 1


class _BlockingResolver:
    """Block on an event until released, so a waiter can be cancelled mid-resolve.

    Models a slow trusted-principal resolve: the resolve is PENDING when the sole
    single/bulk waiter is cancelled, exercising the detached resolve-plus-dispatch
    lifecycle (finding P2). Once released it resolves the fixed principal.
    """

    def __init__(self, principal: str | None = None) -> None:
        self.principal = principal
        self.release = asyncio.Event()
        self.calls = 0

    async def resolve(self) -> str | None:
        self.calls += 1
        await self.release.wait()
        return self.principal


@pytest.mark.asyncio
async def test_cancelled_single_during_blocked_resolution_then_redelivery_no_republish(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Finding P2: cancelling the sole single waiter DURING resolution still caches.

    The resolver is BLOCKED (still pending) when the sole ``start_agent`` waiter is
    cancelled. Because the whole resolve-plus-dispatch lifecycle runs inside a
    shielded detached task, cancelling the waiter cancels neither resolution,
    terminal publication, nor completed caching. After the resolver is released the
    lifecycle reaches ONE terminal and caches it; a redelivery returns the cache with
    NO second terminal and NO second process start.
    """
    kit = supervised_start_kit
    resolver = _BlockingResolver(principal=None)
    supervisor, starter, publisher = kit.supervisor(
        authorizer=kit.Authorizer(allowed=True),
        guard=kit.Guard(),
        principals=resolver,  # type: ignore[arg-type]
    )

    # The SOLE waiter; cancel it while resolution is still blocked/pending.
    waiter = asyncio.create_task(supervisor.start_agent(kit.request(request_id="req-1")))
    while resolver.calls == 0:
        await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    # Cancellation must NOT have cancelled resolution: release it and let the
    # detached lifecycle reach its terminal, publish, and cache.
    resolver.release.set()
    for _ in range(12):
        await asyncio.sleep(0)

    # Exactly one terminal published and cached for the cancelled pair.
    online_first = [f for f in publisher.frames if f.state == "Online"]
    assert len(online_first) == 1
    key = (_PrincipalIdentity.resolved_as(None), "req-1", "agentd-bilby")
    assert key in supervisor._completed  # noqa: SLF001

    # Redelivery returns the cached terminal — no second publish, no second start.
    redelivered = await supervisor.start_agent(kit.request(request_id="req-1"))
    assert redelivered is online_first[0]
    assert [agent.agent_id for agent in starter.started] == ["agentd-bilby"]
    assert len([f for f in publisher.frames if f.state == "Online"]) == 1


@pytest.mark.asyncio
async def test_cancelled_bulk_during_blocked_resolution_then_redelivery_no_republish(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Finding P2: cancelling the sole bulk waiter DURING resolution still caches.

    A bulk ``start_all`` resolves the principal ONCE inside a shielded detached
    lifecycle. Cancelling the sole bulk waiter while that single resolve is still
    pending cancels neither resolution, per-agent terminal publication, nor
    completed caching. After release every known agent reaches ONE terminal and is
    cached; a bulk redelivery returns the cached terminals with NO second terminal
    and NO second process start per agent.
    """
    kit = supervised_start_kit
    resolver = _BlockingResolver(principal=None)
    supervisor, starter, publisher = kit.supervisor(
        authorizer=kit.Authorizer(allowed=True),
        guard=kit.Guard(),
        principals=resolver,  # type: ignore[arg-type]
    )

    request = StartAllAgentsFrame(request_id="bulk-1", requester_id="op")
    waiter = asyncio.create_task(supervisor.start_all(request))
    while resolver.calls == 0:
        await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    # The single bulk resolve must survive the cancellation: release and drain.
    resolver.release.set()
    for _ in range(16):
        await asyncio.sleep(0)

    # Resolution ran exactly once for the whole bulk; each known agent has ONE
    # Online terminal cached under the resolved-None identity.
    assert resolver.calls == 1
    online_first = {(f.request_id, f.agent_id) for f in publisher.frames if f.state == "Online"}
    assert online_first == {("bulk-1", "agentd-bilby"), ("bulk-1", "agentd-skippy")}
    identity = _PrincipalIdentity.resolved_as(None)
    assert (identity, "bulk-1", "agentd-bilby") in supervisor._completed  # noqa: SLF001
    assert (identity, "bulk-1", "agentd-skippy") in supervisor._completed  # noqa: SLF001

    # Bulk redelivery returns the cached terminals — no second terminal, one start
    # per agent total.
    redelivered = await supervisor.start_all(request)
    assert {(f.request_id, f.agent_id) for f in redelivered} == online_first
    assert sorted(agent.agent_id for agent in starter.started) == [
        "agentd-bilby",
        "agentd-skippy",
    ]
    assert len([f for f in publisher.frames if f.state == "Online"]) == 2
