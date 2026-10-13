"""Finding P1 liveness: Online is published only for a probe-confirmed start (#146).

A ``start()`` action returning is not proof the process is alive; every Online
requires an affirmative post-start liveness confirmation. Shared doubles come
from the ``supervised_start_kit`` fixture in ``conftest``.
"""

from __future__ import annotations

import pytest
from rtech_orchestrator import subjects
from rtech_orchestrator.supervised_start import KnownAgent

from .conftest import SupervisedStartKit


@pytest.mark.asyncio
async def test_liveness_probe_reports_online_without_starting(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """An authoritative liveness probe short-circuits the start (already alive)."""
    kit = supervised_start_kit
    starter = kit.Starter()
    liveness = kit.Liveness(online=True)
    supervisor, _starter, publisher = kit.supervisor(
        authorizer=kit.Authorizer(), guard=kit.Guard(), starter=starter, liveness=liveness
    )

    result = await supervisor.start_agent(kit.request())

    assert result.state == "Online"
    assert result.failure is None
    assert starter.started == []
    assert liveness.probed == [KnownAgent("agentd-bilby", "rtech-agentd@agentd-bilby")]
    # The already-online receipt is published at the source.
    assert publisher.frames == [result]


@pytest.mark.asyncio
async def test_liveness_down_falls_through_to_a_real_confirmed_start(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """A probe reporting not-online pre-start drives a real, deduplicated start.

    The default confirming probe reports DOWN before the start and UP after, so a
    real start happens and its Online is post-start confirmed.
    """
    kit = supervised_start_kit
    supervisor, starter, publisher = kit.supervisor(authorizer=kit.Authorizer(), guard=kit.Guard())

    result = await supervisor.start_agent(kit.request())

    assert result.state == "Online"
    assert [agent.agent_id for agent in starter.started] == ["agentd-bilby"]
    assert [frame.state for frame in publisher.frames] == ["Starting", "Online"]


@pytest.mark.asyncio
async def test_missing_probe_never_publishes_online_after_start(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Finding P1 liveness: with NO probe a started agent is never falsely Online.

    ``start()`` returning is not proof of Online. Without an authoritative probe
    the supervisor cannot confirm the process is alive, so it must fail closed with
    ``liveness_unconfirmed`` rather than publish a false ``Online``. The old
    implementation published ``Online`` the instant ``start()`` returned and this
    assertion fails it.
    """
    kit = supervised_start_kit
    starter = kit.Starter()
    supervisor, _starter, publisher = kit.supervisor(
        authorizer=kit.Authorizer(), guard=kit.Guard(), starter=starter, liveness=None
    )
    # Explicitly no probe at all.
    supervisor._liveness = None  # noqa: SLF001

    result = await supervisor.start_agent(kit.request())

    assert result.state == "Failed"
    assert result.failure == "liveness_unconfirmed"
    # The action ran, but no false Online was published.
    assert [agent.agent_id for agent in starter.started] == ["agentd-bilby"]
    assert [frame.state for frame in publisher.frames] == ["Starting", "Failed"]
    assert "Online" not in {f.state for f in publisher.frames}
    # No pending-online ownership without a CONFIRMED start.
    assert supervisor._pending_online == set()  # noqa: SLF001


@pytest.mark.asyncio
async def test_false_post_start_liveness_never_publishes_online(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Finding P1 liveness: a started-but-not-ready process is never Online.

    The start action returns but the required post-start liveness confirmation
    reports the process DOWN (not ready, or already exited). The supervisor must
    fail closed with ``liveness_unconfirmed`` and never publish ``Online``.
    """
    kit = supervised_start_kit
    starter = kit.Starter()
    supervisor, _starter, publisher = kit.supervisor(
        authorizer=kit.Authorizer(),
        guard=kit.Guard(),
        starter=starter,
        liveness=kit.NeverAliveLiveness(),
    )

    result = await supervisor.start_agent(kit.request())

    assert result.state == "Failed"
    assert result.failure == "liveness_unconfirmed"
    assert [agent.agent_id for agent in starter.started] == ["agentd-bilby"]
    assert [frame.state for frame in publisher.frames] == ["Starting", "Failed"]
    assert supervisor._pending_online == set()  # noqa: SLF001


@pytest.mark.asyncio
async def test_failing_post_start_liveness_probe_never_publishes_online(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Finding P1 liveness: a probe that ERRORS post-start fails closed, no Online.

    A liveness probe exception cannot be read as "alive": the supervisor treats an
    unconfirmable probe as unconfirmed and fails closed.
    """
    kit = supervised_start_kit
    starter = kit.Starter()
    supervisor, _starter, publisher = kit.supervisor(
        authorizer=kit.Authorizer(),
        guard=kit.Guard(),
        starter=starter,
        liveness=kit.Liveness(fail=True),
    )

    result = await supervisor.start_agent(kit.request())

    assert result.state == "Failed"
    assert result.failure == "liveness_unconfirmed"
    assert [agent.agent_id for agent in starter.started] == ["agentd-bilby"]
    assert "Online" not in {f.state for f in publisher.frames}
    assert supervisor._pending_online == set()  # noqa: SLF001


@pytest.mark.asyncio
async def test_online_after_confirmed_start_survives_process_exit_semantics(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Finding P1 liveness: Online is published only for a probe-confirmed start.

    A confirmed start records pending ownership; a later request with the probe
    reporting DOWN (process exited) must never synthesize Online from that history.
    """
    kit = supervised_start_kit
    starter = kit.Starter()
    confirming = kit.ConfirmingLiveness(starter)
    supervisor, _starter, publisher = kit.supervisor(
        authorizer=kit.Authorizer(), guard=kit.Guard(), starter=starter, liveness=confirming
    )

    first = await supervisor.start_agent(kit.request(request_id="req-1"))
    assert first.state == "Online"
    assert supervisor._pending_online == set()  # noqa: SLF001 — dropped after publish

    # Simulate the process exiting: the confirming probe still returns True because
    # it tracks the starter history, so swap in a DOWN probe to model the crash.
    supervisor._liveness = kit.NeverAliveLiveness()  # noqa: SLF001
    second = await supervisor.start_agent(kit.request(request_id="req-2"))

    # No probe-confirmed liveness on retry → fail closed, never a synthesized Online.
    assert second.state == "Failed"
    assert second.failure == "liveness_unconfirmed"


@pytest.mark.asyncio
async def test_already_online_receipt_is_published_by_the_wire_handler(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """The wire handler does not discard the already-online receipt."""
    kit = supervised_start_kit
    liveness = kit.Liveness(online=True)
    supervisor, _starter, publisher = kit.supervisor(
        authorizer=kit.Authorizer(), guard=kit.Guard(), liveness=liveness
    )

    await supervisor._on_frame(kit.request())  # noqa: SLF001

    assert [frame.state for frame in publisher.frames] == ["Online"]
    assert publisher.subjects == [subjects.agent_supervised_state("agentd-bilby")]
