"""Build ratified web-chat NATS subjects in one place."""

from __future__ import annotations

from fleet_nats import subjects as fleet_subjects

TASK_INGRESS = "rtech.task.new"
TASK_TURN_TEMPLATE = "rtech.task.{task_id}.turn"
TASK_STATUS_TEMPLATE = "rtech.task.{task_id}.status"
TASK_STEER_TEMPLATE = "rtech.task.{task_id}.steer"
TASK_RESULT_TEMPLATE = "rtech.task.{task_id}.result"
AGENT_TASK_TEMPLATE = "rtech.agent.{agent_id}.task"
# The one presence-transition subject: the deriver publishes a typed
# PresenceTransitionFrame here on each genuine online<->offline edge, and a chatd
# consumer subscribes it to render an operator Offline/Recovered notification. A
# flat constant (like TASK_INGRESS) — the subject carries no per-task token.
PRESENCE_TRANSITION = "rtech.presence.transition"


def agent_heartbeat_wildcard(env: str) -> str:
    """Return the fleet heartbeat subscription wildcard for an env.

    Delegates to :func:`fleet_nats.subjects.agent_heartbeat_wildcard` so the
    web-chat presence lane reads the SAME ``{env}.agent.heartbeat.*`` subjects
    every fleet client publishes — not a private rtech dialect.
    """
    if not env:
        raise ValueError("env must be non-empty")
    return fleet_subjects.agent_heartbeat_wildcard(env)


def task_turn(task_id: str) -> str:
    """Return the per-task turn subject."""

    return TASK_TURN_TEMPLATE.format(task_id=task_id)


def task_status(task_id: str) -> str:
    """Return the per-task status subject."""

    return TASK_STATUS_TEMPLATE.format(task_id=task_id)


def task_steer(task_id: str) -> str:
    """Return the per-task steer subject."""

    return TASK_STEER_TEMPLATE.format(task_id=task_id)


def task_result(task_id: str) -> str:
    """Return the per-task result subject."""

    return TASK_RESULT_TEMPLATE.format(task_id=task_id)


def agent_task(agent_id: str) -> str:
    """Return the agent dispatch subject."""

    return AGENT_TASK_TEMPLATE.format(agent_id=agent_id)
