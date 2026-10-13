"""Assert subject builders emit the exact ratified strings."""

from __future__ import annotations

import pytest
from rtech_orchestrator import subjects


def test_task_ingress_constant() -> None:
    """Ingress subject must be the ratified literal."""
    assert subjects.TASK_INGRESS == "rtech.task.new"
    assert subjects.SUPERVISED_START_INGRESS == "rtech.supervisor.start"
    assert subjects.SUPERVISED_START_ALL_INGRESS == "rtech.supervisor.start-all"


def test_per_task_builders() -> None:
    """Per-task lane builders must match the ratified scheme."""
    assert subjects.task_turn("t1") == "rtech.task.t1.turn"
    assert subjects.task_status("t1") == "rtech.task.t1.status"
    assert subjects.task_steer("t1") == "rtech.task.t1.steer"
    assert subjects.task_result("t1") == "rtech.task.t1.result"


def test_agent_task_builder() -> None:
    """Agent dispatch builder must match the ratified scheme."""
    assert subjects.agent_task("agentd") == "rtech.agent.agentd.task"


def test_agent_lifecycle_builder() -> None:
    """Per-agent lifecycle lane builder must match the ratified scheme (#84)."""
    assert subjects.agent_lifecycle("agentd") == "rtech.agent.agentd.lifecycle"


def test_agent_supervised_state_builder() -> None:
    """External Start state has a distinct per-agent typed lane (#146)."""
    assert subjects.agent_supervised_state("agentd") == "rtech.agent.agentd.supervised-state"


@pytest.mark.parametrize(
    "builder",
    [
        subjects.task_turn,
        subjects.task_status,
        subjects.task_steer,
        subjects.task_result,
        subjects.agent_task,
        subjects.agent_lifecycle,
        subjects.agent_supervised_state,
    ],
)
def test_empty_id_raises(builder: object) -> None:
    """Empty ids must raise ValueError in every builder."""
    with pytest.raises(ValueError):
        builder("")  # type: ignore[operator]
    with pytest.raises(ValueError):
        builder("   ")  # type: ignore[operator]
