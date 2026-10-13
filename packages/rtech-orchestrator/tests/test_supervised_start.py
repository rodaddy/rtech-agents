"""Cover external-supervisor Start-only lifecycle requests.

Happy-path start/bulk/subscribe plus the P2 dedup and correlated-receipt seams.
Liveness, publish-failure, and trust coverage live in the sibling
``test_supervised_start_*`` modules; the shared doubles come from the
``supervised_start_kit`` fixture in ``conftest``.
"""

from __future__ import annotations

import asyncio

import pytest
from rtech_frames import StartAllAgentsFrame
from rtech_orchestrator import subjects
from rtech_orchestrator.supervised_start import KnownAgent

from .conftest import SupervisedStartKit


@pytest.mark.asyncio
async def test_start_known_agent_publishes_starting_then_online(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    kit = supervised_start_kit
    supervisor, starter, publisher = kit.supervisor(authorizer=kit.Authorizer(), guard=kit.Guard())

    result = await supervisor.start_agent(kit.request())

    assert result.state == "Online"
    assert [frame.state for frame in publisher.frames] == ["Starting", "Online"]
    assert publisher.subjects == [
        subjects.agent_supervised_state("agentd-bilby"),
        subjects.agent_supervised_state("agentd-bilby"),
    ]
    assert starter.started == [KnownAgent("agentd-bilby", "rtech-agentd@agentd-bilby")]


@pytest.mark.asyncio
async def test_bulk_start_uses_only_server_side_known_agents(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    kit = supervised_start_kit
    supervisor, starter, publisher = kit.supervisor(authorizer=kit.Authorizer(), guard=kit.Guard())

    result = await supervisor.start_all(
        StartAllAgentsFrame(request_id="bulk-1", requester_id="operator-1")
    )

    assert [frame.agent_id for frame in result] == ["agentd-bilby", "agentd-skippy"]
    assert sorted(agent.agent_id for agent in starter.started) == ["agentd-bilby", "agentd-skippy"]
    assert sorted(frame.state for frame in publisher.frames) == [
        "Online",
        "Online",
        "Starting",
        "Starting",
    ]


@pytest.mark.asyncio
async def test_start_subscribes_only_the_typed_start_lanes(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    kit = supervised_start_kit
    supervisor, _starter, publisher = kit.supervisor(authorizer=kit.Authorizer(), guard=kit.Guard())

    await supervisor.start()

    assert publisher.subscriptions == [
        subjects.SUPERVISED_START_INGRESS,
        subjects.SUPERVISED_START_ALL_INGRESS,
    ]


@pytest.mark.asyncio
async def test_duplicate_same_request_id_does_not_flood_receipts(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Finding P2: identical (request_id, agent) duplicates share ONE receipt.

    Two concurrent requests with the SAME request_id for the same agent must not
    flood duplicate terminal receipts.
    """
    kit = supervised_start_kit
    starter = kit.Starter(delay=0.02)
    supervisor, _starter, publisher = kit.supervisor(
        authorizer=kit.Authorizer(), guard=kit.Guard(), starter=starter
    )

    first, duplicate = await asyncio.gather(
        supervisor.start_agent(kit.request()),
        supervisor.start_agent(kit.request()),
    )

    assert first == duplicate
    assert [agent.agent_id for agent in starter.started] == ["agentd-bilby"]
    # One shared receipt for the identical request id: no duplicate flood.
    assert [frame.state for frame in publisher.frames] == ["Starting", "Online"]


@pytest.mark.asyncio
async def test_distinct_request_ids_each_get_their_own_terminal_receipt(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Finding P2: one shared start, but a correlated receipt PER request id.

    Two distinct request ids racing on the same agent share ONE process start,
    yet each must receive AND publish its own terminal ``Online`` receipt carrying
    its own ``request_id`` — never one requester's receipt reused for both. The
    old implementation returned the first requester's shared frame to both waiters
    and this assertion on distinct correlated receipts fails it.
    """
    kit = supervised_start_kit
    starter = kit.Starter(delay=0.02)
    supervisor, _starter, publisher = kit.supervisor(
        authorizer=kit.Authorizer(), guard=kit.Guard(), starter=starter
    )

    a, b = await asyncio.gather(
        supervisor.start_agent(kit.request(request_id="req-a")),
        supervisor.start_agent(kit.request(request_id="req-b")),
    )

    # Exactly one real start.
    assert [agent.agent_id for agent in starter.started] == ["agentd-bilby"]
    # Each waiter got its OWN terminal Online receipt correlated to its request id.
    assert a.state == "Online" and a.request_id == "req-a"
    assert b.state == "Online" and b.request_id == "req-b"
    assert a != b
    # Both correlated Online terminals landed on the bus (plus each Starting).
    online_ids = sorted(f.request_id for f in publisher.frames if f.state == "Online")
    assert online_ids == ["req-a", "req-b"]
    starting_ids = sorted(f.request_id for f in publisher.frames if f.state == "Starting")
    assert starting_ids == ["req-a", "req-b"]


@pytest.mark.asyncio
async def test_bulk_racing_single_starts_each_agent_once_with_own_receipts(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Finding P2: a bulk request racing single ones — one start, correlated receipts.

    A bulk request racing per-agent single requests shares one start per agent,
    and every distinct (request_id, agent) pair gets its own terminal receipt.
    """
    kit = supervised_start_kit
    starter = kit.Starter(delay=0.02)
    supervisor, _starter, publisher = kit.supervisor(
        authorizer=kit.Authorizer(), guard=kit.Guard(), starter=starter
    )

    bulk, single_bilby, single_skippy = await asyncio.gather(
        supervisor.start_all(StartAllAgentsFrame(request_id="bulk-1", requester_id="op")),
        supervisor.start_agent(kit.request(request_id="single-1", agent_id="agentd-bilby")),
        supervisor.start_agent(kit.request(request_id="single-2", agent_id="agentd-skippy")),
    )

    assert sorted(agent.agent_id for agent in starter.started) == [
        "agentd-bilby",
        "agentd-skippy",
    ]
    # Each distinct request id got its own correlated terminal for its agent.
    bulk_by_agent = {f.agent_id: f for f in bulk}
    assert bulk_by_agent["agentd-bilby"].request_id == "bulk-1"
    assert bulk_by_agent["agentd-skippy"].request_id == "bulk-1"
    assert single_bilby.request_id == "single-1" and single_bilby.agent_id == "agentd-bilby"
    assert single_skippy.request_id == "single-2" and single_skippy.agent_id == "agentd-skippy"
    # Every distinct (request_id, agent) pair has its own Online terminal on the bus.
    online = {(f.request_id, f.agent_id) for f in publisher.frames if f.state == "Online"}
    assert online == {
        ("bulk-1", "agentd-bilby"),
        ("bulk-1", "agentd-skippy"),
        ("single-1", "agentd-bilby"),
        ("single-2", "agentd-skippy"),
    }


@pytest.mark.asyncio
async def test_inflight_cache_is_pruned_after_each_start(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """The in-flight map is bounded — pruned on terminal outcome."""
    kit = supervised_start_kit
    supervisor, _starter, _publisher = kit.supervisor(
        authorizer=kit.Authorizer(), guard=kit.Guard()
    )

    await supervisor.start_agent(kit.request(request_id="req-1"))
    await supervisor.start_agent(kit.request(request_id="req-2", agent_id="agentd-skippy"))

    for _ in range(4):
        await asyncio.sleep(0)
    assert supervisor._inflight == {}  # noqa: SLF001
    assert supervisor._inflight_requests == {}  # noqa: SLF001


@pytest.mark.asyncio
async def test_failed_start_is_not_cached_so_retry_reattempts(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """A failed start is not retained; a later request retries safely."""
    kit = supervised_start_kit
    starter = kit.Starter(fail=True)
    supervisor, _starter, publisher = kit.supervisor(
        authorizer=kit.Authorizer(),
        guard=kit.Guard(),
        starter=starter,
        liveness=kit.ConfirmingLiveness(starter),
    )

    first = await supervisor.start_agent(kit.request(request_id="req-1"))
    assert first.failure == "start_failed"
    assert supervisor._inflight == {}  # noqa: SLF001

    # A fresh request retries the start (proving no stale dedup entry blocks it).
    starter.fail = False
    second = await supervisor.start_agent(kit.request(request_id="req-2"))
    assert second.state == "Online"
    assert [agent.agent_id for agent in starter.started] == ["agentd-bilby", "agentd-bilby"]


@pytest.mark.asyncio
async def test_cancelling_one_waiter_does_not_cancel_the_shared_start(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Cancelling one waiter leaves EXACTLY one terminal per pair, maps pruned.

    Cancelling the outer ``start_agent`` for the cancelled pair must not cancel its
    shielded per-request task: that task still reaches a terminal and publishes the
    cancelled pair's OWN single correlated receipt. After all callbacks drain there
    is exactly ONE terminal bus receipt for the cancelled pair AND exactly one for
    the surviving pair — no duplicates — and every bounded map is empty.
    """
    kit = supervised_start_kit
    starter = kit.Starter(delay=0.05)
    supervisor, _starter, publisher = kit.supervisor(
        authorizer=kit.Authorizer(), guard=kit.Guard(), starter=starter
    )

    waiter = asyncio.create_task(supervisor.start_agent(kit.request(request_id="req-1")))
    await asyncio.sleep(0.01)  # let the shared start task begin
    survivor = asyncio.create_task(supervisor.start_agent(kit.request(request_id="req-2")))
    await asyncio.sleep(0)
    waiter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiter

    # The shared machine still completes with a terminal Online outcome.
    result = await survivor
    assert result.state == "Online"
    assert result.request_id == "req-2"
    # Exactly one real process start was shared by both pairs.
    assert [agent.agent_id for agent in starter.started] == ["agentd-bilby"]

    # Let the cancelled pair's shielded task, its terminal publish, and every
    # done-callback fully drain before counting receipts.
    for _ in range(8):
        await asyncio.sleep(0)

    # Exactly one terminal bus receipt per pair — the cancelled pair AND the
    # survivor each land ONE Online terminal, with no duplicates.
    terminals = [(f.request_id, f.agent_id) for f in publisher.frames if f.state == "Online"]
    assert sorted(terminals) == [("req-1", "agentd-bilby"), ("req-2", "agentd-bilby")]
    assert len(terminals) == len(set(terminals))
    # Each pair also got exactly one Starting breadcrumb — no duplicate flood.
    startings = [(f.request_id, f.agent_id) for f in publisher.frames if f.state == "Starting"]
    assert sorted(startings) == [("req-1", "agentd-bilby"), ("req-2", "agentd-bilby")]
    # All bounded maps are pruned once the terminal outcomes drain.
    assert supervisor._inflight == {}  # noqa: SLF001
    assert supervisor._inflight_requests == {}  # noqa: SLF001
    assert supervisor._pending_online == set()  # noqa: SLF001
