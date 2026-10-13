"""Build the ratified NATS subject strings for the orchestrator.

The subject scheme is fixed wire contract; the strings here MUST match the
ratified scheme exactly. Task ingress is a module constant; per-task and
per-agent lanes are built from an id via typed builders that reject empty ids.
"""

from __future__ import annotations

from loguru import logger

TASK_INGRESS = "rtech.task.new"
SUPERVISED_START_INGRESS = "rtech.supervisor.start"
SUPERVISED_START_ALL_INGRESS = "rtech.supervisor.start-all"


def _require(value: str, name: str) -> str:
    """Return a non-empty id or raise ``ValueError``.

    Args:
        value: The candidate id.
        name: The parameter name, for the error message.

    Returns:
        The stripped-safe original value when non-empty.

    Raises:
        ValueError: If the value is empty or whitespace-only.
    """
    if not value or not value.strip():
        logger.error("rejected empty subject id", parameter=name)
        raise ValueError(f"{name} must be a non-empty string")
    return value


def task_turn(task_id: str) -> str:
    """Return the per-task turn lane subject."""
    return f"rtech.task.{_require(task_id, 'task_id')}.turn"


def task_status(task_id: str) -> str:
    """Return the per-task status lane subject."""
    return f"rtech.task.{_require(task_id, 'task_id')}.status"


def task_steer(task_id: str) -> str:
    """Return the per-task steer lane subject."""
    return f"rtech.task.{_require(task_id, 'task_id')}.steer"


def task_result(task_id: str) -> str:
    """Return the per-task result lane subject (orchestrator-only publish)."""
    return f"rtech.task.{_require(task_id, 'task_id')}.result"


def agent_task(agent_id: str) -> str:
    """Return the per-agent dispatch subject."""
    return f"rtech.agent.{_require(agent_id, 'agent_id')}.task"


def agent_lifecycle(agent_id: str) -> str:
    """Return the per-agent lifecycle lane subject (issue #84).

    Daemon shutdown/restart and supervisor recovery publish a ``LifecycleFrame``
    here so the transition is legible on the bus, beside the per-agent dispatch
    lane.
    """
    return f"rtech.agent.{_require(agent_id, 'agent_id')}.lifecycle"


def agent_supervised_state(agent_id: str) -> str:
    """Return the external-supervisor Start state lane for one agent."""
    return f"rtech.agent.{_require(agent_id, 'agent_id')}.supervised-state"
