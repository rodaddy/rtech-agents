"""Bus-first publish-failure and pending-online recovery seams.

A failed state publish must still put a typed terminal receipt on the wire — never
Starting then silence — and a pending-online claim after a publish failure never
overrides an authoritative DOWN probe. Shared doubles come from the
``supervised_start_kit`` fixture in ``conftest``.
"""

from __future__ import annotations

import pytest

from .conftest import SupervisedStartKit


@pytest.mark.asyncio
async def test_online_publish_failure_publishes_terminal_receipt_bus_first(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """An Online publish failure publishes the typed terminal receipt bus-first.

    The wire may not show Starting then silence: the confirmed-alive start's Online
    publish failed, so the ``state_publish_failed`` terminal receipt itself is
    published to the bus.
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

    result = await supervisor.start_agent(kit.request(request_id="req-1"))

    assert result.state == "Failed"
    assert result.failure == "state_publish_failed"
    assert result.request_id == "req-1"
    assert [agent.agent_id for agent in starter.started] == ["agentd-birk"]
    # Bus-first: Starting AND the terminal state_publish_failed receipt are on the
    # wire — never Starting then silence.
    assert [(f.state, f.failure) for f in publisher.frames] == [
        ("Starting", None),
        ("Failed", "state_publish_failed"),
    ]
    # Ownership survives the publish failure for a probe-gated retry.
    assert supervisor._pending_online == {"agentd-birk"}  # noqa: SLF001


@pytest.mark.asyncio
async def test_online_publish_failure_recovers_when_probe_confirms_alive(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Finding P1/P2: a retry re-publishes Online only when the probe confirms alive.

    With the process still authoritatively alive, the pending-online ownership is
    trusted and the retry re-publishes Online WITHOUT starting the process again.
    """
    kit = supervised_start_kit
    starter = kit.Starter()
    publisher = kit.Publisher(fail_states=("Online",))
    liveness = kit.ConfirmingLiveness(starter)
    supervisor, _starter, _publisher = kit.supervisor(
        authorizer=kit.Authorizer(),
        guard=kit.Guard(),
        starter=starter,
        publisher=publisher,
        liveness=liveness,
    )

    # First attempt starts for real, confirms alive, but its Online publish fails,
    # leaving pending ownership recorded.
    first = await supervisor.start_agent(kit.request(request_id="req-1"))
    assert first.failure == "state_publish_failed"
    assert [agent.agent_id for agent in starter.started] == ["agentd-birk"]
    assert supervisor._pending_online == {"agentd-birk"}  # noqa: SLF001

    # The probe still confirms the process is alive; the retry short-circuits at
    # _already_online and re-publishes Online with no re-start.
    publisher._fail_states = ()  # noqa: SLF001
    second = await supervisor.start_agent(kit.request(request_id="req-2"))
    assert second.state == "Online"
    assert second.request_id == "req-2"
    assert [agent.agent_id for agent in starter.started] == ["agentd-birk"]


@pytest.mark.asyncio
async def test_pending_online_does_not_override_probe_down_after_crash(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Finding P1: a stale pending-online claim never overrides a DOWN probe.

    A confirmed start whose Online receipt failed leaves pending ownership. If the
    process then crashes (probe reports DOWN), a retry MUST NOT synthesize Online
    from that stale ownership — it must drop it and start for real (then re-confirm).
    """
    kit = supervised_start_kit
    starter = kit.Starter()
    publisher = kit.Publisher(fail_states=("Online",))
    confirming = kit.ConfirmingLiveness(starter)
    supervisor, _starter, _publisher = kit.supervisor(
        authorizer=kit.Authorizer(),
        guard=kit.Guard(),
        starter=starter,
        publisher=publisher,
        liveness=confirming,
    )

    first = await supervisor.start_agent(kit.request(request_id="req-1"))
    assert first.failure == "state_publish_failed"
    assert supervisor._pending_online == {"agentd-birk"}  # noqa: SLF001

    # Process has crashed; probe now stays DOWN. The retry drops the stale claim
    # and does a REAL reconciliation start, re-confirmed alive afterward.
    publisher._fail_states = ()  # noqa: SLF001
    supervisor._liveness = kit.ConfirmingLiveness(starter)  # noqa: SLF001 — DOWN then UP again
    # Model the crash: forget the earlier start so the confirming probe reports
    # DOWN until a fresh start action runs.
    starter.started.clear()
    starter.succeeded.clear()
    second = await supervisor.start_agent(kit.request(request_id="req-2"))

    assert second.state == "Online"
    assert second.request_id == "req-2"
    # A REAL reconciliation start happened — the stale pending claim did not
    # short-circuit into a synthesized Online.
    assert [agent.agent_id for agent in starter.started] == ["agentd-birk"]
    assert supervisor._pending_online == set()  # noqa: SLF001


@pytest.mark.asyncio
async def test_pending_online_does_not_override_absent_probe(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Finding P1: with a probe that cannot confirm, pending cannot synthesize Online.

    When the authoritative probe cannot confirm the process is alive, a retry after
    a publish failure must do a real start rather than trust the unverifiable
    pending claim — and that real start still requires post-start confirmation.
    """
    kit = supervised_start_kit
    starter = kit.Starter()
    publisher = kit.Publisher(fail_states=("Online",))
    confirming = kit.ConfirmingLiveness(starter)
    supervisor, _starter, _publisher = kit.supervisor(
        authorizer=kit.Authorizer(),
        guard=kit.Guard(),
        starter=starter,
        publisher=publisher,
        liveness=confirming,
    )

    first = await supervisor.start_agent(kit.request(request_id="req-1"))
    assert first.failure == "state_publish_failed"
    assert supervisor._pending_online == {"agentd-birk"}  # noqa: SLF001

    # Swap in a probe that never confirms alive → the pending claim is unverifiable
    # and the post-start confirmation fails → liveness_unconfirmed, no false Online.
    publisher._fail_states = ()  # noqa: SLF001
    supervisor._liveness = kit.NeverAliveLiveness()  # noqa: SLF001
    second = await supervisor.start_agent(kit.request(request_id="req-2"))

    assert second.state == "Failed"
    assert second.failure == "liveness_unconfirmed"
    # The stale pending claim was dropped and a real reconciliation start attempted.
    assert [agent.agent_id for agent in starter.started] == ["agentd-birk", "agentd-birk"]
    assert supervisor._pending_online == set()  # noqa: SLF001


@pytest.mark.asyncio
async def test_starting_publish_failure_publishes_terminal_receipt_bus_first(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """A failed Starting publish still publishes the terminal receipt.

    The wire may not show a dropped Starting then silence: the typed
    state_publish_failed terminal receipt is published to the bus, and no start is
    attempted because Starting never confirmed.
    """
    kit = supervised_start_kit
    starter = kit.Starter()
    publisher = kit.Publisher(fail_states=("Starting",))
    supervisor, _starter, _publisher = kit.supervisor(
        authorizer=kit.Authorizer(), guard=kit.Guard(), starter=starter, publisher=publisher
    )

    result = await supervisor.start_agent(kit.request(request_id="req-1"))

    assert result.state == "Failed"
    assert result.failure == "state_publish_failed"
    assert result.request_id == "req-1"
    # The Starting publish failed, so no process start was attempted...
    assert starter.started == []
    # ...but the terminal receipt reached the bus (Starting never confirmed).
    assert [(f.state, f.failure) for f in publisher.frames] == [("Failed", "state_publish_failed")]
    # No pending-online ownership without a confirmed start.
    assert supervisor._pending_online == set()  # noqa: SLF001


@pytest.mark.asyncio
async def test_online_terminal_receipt_secondary_publish_failure_returns_truthfully(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """If the terminal receipt ALSO cannot publish, fail truthfully.

    When the Online publish fails and the state_publish_failed receipt itself
    cannot publish, the supervisor returns the truthful typed failure without a
    recursive failure loop. Exactly ONE terminal-receipt publish is attempted.
    """
    kit = supervised_start_kit
    starter = kit.Starter()
    publisher = kit.Publisher(fail_states=("Online", "Failed"))
    supervisor, _starter, _publisher = kit.supervisor(
        authorizer=kit.Authorizer(),
        guard=kit.Guard(),
        starter=starter,
        publisher=publisher,
        liveness=kit.ConfirmingLiveness(starter),
    )

    result = await supervisor.start_agent(kit.request(request_id="req-1"))

    assert result.state == "Failed"
    assert result.failure == "state_publish_failed"
    assert [agent.agent_id for agent in starter.started] == ["agentd-birk"]
    # Bounded: exactly one terminal Failed publish was attempted (no recursion).
    failed_attempts = [f for f in publisher.attempts if f.state == "Failed"]
    assert len(failed_attempts) == 1
    # Nothing durably landed on the bus for this attempt beyond Starting.
    assert [f.state for f in publisher.frames] == ["Starting"]
    # Ownership survives so a later probe-gated retry can still recover.
    assert supervisor._pending_online == {"agentd-birk"}  # noqa: SLF001


@pytest.mark.asyncio
async def test_starting_terminal_receipt_secondary_publish_failure_returns_truthfully(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """A totally broken bus still returns a truthful terminal frame.

    With every publish failing, the Starting publish fails and the terminal
    receipt fails too; the supervisor returns the typed failure and attempts the
    terminal receipt exactly once — no recursive loop.
    """
    kit = supervised_start_kit
    starter = kit.Starter()
    publisher = kit.Publisher(fail_all=True)
    supervisor, _starter, _publisher = kit.supervisor(
        authorizer=kit.Authorizer(), guard=kit.Guard(), starter=starter, publisher=publisher
    )

    result = await supervisor.start_agent(kit.request(request_id="req-1"))

    assert result.state == "Failed"
    assert result.failure == "state_publish_failed"
    assert starter.started == []
    # Exactly two publish attempts total: Starting, then the one terminal receipt.
    assert [f.state for f in publisher.attempts] == ["Starting", "Failed"]
    assert publisher.frames == []
    assert supervisor._pending_online == set()  # noqa: SLF001
