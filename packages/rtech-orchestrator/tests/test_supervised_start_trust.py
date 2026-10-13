"""Finding P1 trust plus fail-closed auth/guard seams (#146).

Authorization runs on a server-resolved transport principal, never the request
frame, so a forged ``requester_id`` can reach neither the authorizer nor the
resolver; missing or rejected auth/guard fails closed with a typed failure. Shared
doubles come from the ``supervised_start_kit`` fixture in ``conftest``.
"""

from __future__ import annotations

import pytest

from .conftest import SupervisedStartKit


@pytest.mark.asyncio
async def test_authorizer_receives_trusted_principal_not_a_frame(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Finding P1 trust: authorization uses a server-resolved principal, no frame.

    The authorizer is handed only the trusted principal from the resolver; it is
    NEVER handed the request frame, so it cannot read ``requester_id``.
    """
    kit = supervised_start_kit
    authorizer = kit.Authorizer()
    principals = kit.Principals(principal="trusted-operator")
    supervisor, _starter, _publisher = kit.supervisor(
        authorizer=authorizer, guard=kit.Guard(), principals=principals
    )

    await supervisor.start_agent(kit.request())

    # The authorizer saw the trusted principal from the resolver, independent of
    # the frame's self-asserted requester_id.
    assert authorizer.principals == ["trusted-operator"]
    assert principals.calls == 1  # resolver was consulted at the boundary


@pytest.mark.asyncio
async def test_resolver_is_never_handed_the_request_frame(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Finding P1 trust: the principal resolver takes NO frame argument.

    The resolver resolves the trusted transport principal without the request
    body, so a forged ``requester_id`` can never reach it. Calling it with an
    argument would raise; the supervisor calls it argument-free.
    """
    kit = supervised_start_kit
    calls: list[tuple[object, ...]] = []

    class _StrictResolver:
        async def resolve(self, *args: object) -> str | None:
            calls.append(args)
            return "trusted-operator"

    authorizer = kit.Authorizer()
    supervisor, _starter, _publisher = kit.supervisor(
        authorizer=authorizer,
        guard=kit.Guard(),
        principals=_StrictResolver(),  # type: ignore[arg-type]
    )

    await supervisor.start_agent(kit.request(request_id="req-forge"))

    # The resolver was called with NO positional arguments (no frame smuggled in).
    assert calls == [()]
    assert authorizer.principals == ["trusted-operator"]


@pytest.mark.asyncio
async def test_forged_requester_id_cannot_reach_either_trust_seam(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Finding P1 trust: a forged requester_id never reaches authorizer or resolver.

    A frame-probing authorizer that WOULD forge trust from ``requester_id`` is
    never handed the frame, so its forge path is structurally dead: it can only
    authorize on the principal the resolver produced. A distinct forged
    requester_id therefore has no effect on authorization.
    """
    kit = supervised_start_kit
    forged = kit.request(request_id="req-x")
    object.__setattr__(forged, "requester_id", "admin-impersonation")
    authorizer = kit.FrameProbingAuthorizer(trusted_requester="admin-impersonation")
    principals = kit.Principals(principal="trusted-operator")
    supervisor, _starter, _publisher = kit.supervisor(
        authorizer=authorizer, guard=kit.Guard(), principals=principals
    )

    result = await supervisor.start_agent(forged)

    # Authorization succeeded on the PRINCIPAL, never the forged requester_id.
    assert result.state == "Online"
    # The authorizer only ever received the principal — never the frame.
    assert authorizer.calls == [("trusted-operator",)]
    assert all("admin-impersonation" not in call for call in authorizer.calls)


@pytest.mark.asyncio
async def test_principal_resolver_failure_fails_closed(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    """Finding P1: a resolver failure fails closed with a typed failure."""
    kit = supervised_start_kit
    principals = kit.Principals(fail=True)
    supervisor, starter, publisher = kit.supervisor(
        authorizer=kit.Authorizer(), guard=kit.Guard(), principals=principals
    )

    result = await supervisor.start_agent(kit.request())

    assert result.state == "Failed"
    assert result.failure == "auth_unavailable"
    assert starter.started == []
    assert publisher.frames == [result]


@pytest.mark.asyncio
async def test_unknown_agent_reports_typed_failure_without_starting(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    kit = supervised_start_kit
    supervisor, starter, publisher = kit.supervisor(authorizer=kit.Authorizer(), guard=kit.Guard())

    result = await supervisor.start_agent(kit.request("agentd-unknown"))

    assert result.state == "Failed"
    assert result.failure == "unknown_agent"
    assert starter.started == []
    assert publisher.frames == [result]


@pytest.mark.asyncio
async def test_start_failure_reports_starting_then_typed_failed(
    supervised_start_kit: SupervisedStartKit,
) -> None:
    kit = supervised_start_kit
    starter = kit.Starter(fail=True)
    supervisor, _starter, publisher = kit.supervisor(
        authorizer=kit.Authorizer(),
        guard=kit.Guard(),
        starter=starter,
        liveness=kit.ConfirmingLiveness(starter),
    )

    result = await supervisor.start_agent(kit.request())

    assert result.failure == "start_failed"
    assert [frame.state for frame in publisher.frames] == ["Starting", "Failed"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("authorizer_spec", "guard_spec", "failure"),
    [
        (None, "allow", "auth_unavailable"),
        ("deny", "allow", "unauthorized"),
        ("allow", None, "guard_unavailable"),
        ("allow", "deny", "guard_rejected"),
    ],
)
async def test_missing_or_rejected_auth_guard_fails_closed(
    authorizer_spec: str | None,
    guard_spec: str | None,
    failure: str,
    supervised_start_kit: SupervisedStartKit,
) -> None:
    kit = supervised_start_kit
    authorizer = (
        None if authorizer_spec is None else kit.Authorizer(allowed=authorizer_spec == "allow")
    )
    guard = None if guard_spec is None else kit.Guard(allowed=guard_spec == "allow")
    supervisor, starter, publisher = kit.supervisor(authorizer=authorizer, guard=guard)

    result = await supervisor.start_agent(kit.request())

    assert result.failure == failure
    assert starter.started == []
    assert publisher.frames == [result]
