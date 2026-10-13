"""Build ratified chatd NATS subjects in one place.

chatd is a dumb edge: it publishes human input as a ``TaskFrame`` on
:data:`TASK_INGRESS` and subscribes the per-task ``turn``/``status`` subjects to
render replies back into the originating chat surface. It carries no control
signal — done is orchestrator-derived from the typed frame graph, never inferred
at this edge.
"""

from __future__ import annotations

TASK_INGRESS = "rtech.task.new"
TASK_TURN_TEMPLATE = "rtech.task.{task_id}.turn"
# The NATS single-token wildcard over every per-task turn subject. A triage edge
# (reply render) subscribes it to see EXTERNAL task turns — turns for
# tasks the edge did not mint itself (an intake/wake answer) — and render the
# terminal, origin-carrying ones back into the asking channel. ``*`` matches
# exactly one token (a task id), so it never widens to status/steer/result.
TASK_TURN_WILDCARD = "rtech.task.*.turn"
TASK_STATUS_TEMPLATE = "rtech.task.{task_id}.status"
TASK_STEER_TEMPLATE = "rtech.task.{task_id}.steer"
# The NATS single-token wildcard over every per-task steer subject. A live
# reaction-steer canary cannot predict the opaque task id BEFORE it
# reacts, so it subscribes this wildcard UP FRONT to capture the first-class
# typed ``SteerFrame`` a reaction produces on ``rtech.task.<opaque-id>.steer``,
# then filters by the discovered id. ``*`` matches exactly one token (a task id),
# so it never widens to turn/status/result. The steer wire stays first-class and
# typed — the canary observes a ``SteerFrame`` field, never a scraped prose steer.
TASK_STEER_WILDCARD = "rtech.task.*.steer"
TASK_RESULT_TEMPLATE = "rtech.task.{task_id}.result"
AGENT_TASK_TEMPLATE = "rtech.agent.{agent_id}.task"
# The one presence-transition subject. The web-chat presence deriver
# publishes a typed ``PresenceTransitionFrame`` here on each genuine
# online<->offline edge; a triage edge subscribes it to render an operator
# Offline/Recovered notification. It MUST match the producer's flat constant
# (``rtech_webchat.subjects.PRESENCE_TRANSITION``) byte-for-byte — the subject
# carries no per-task token, so it is a flat constant like ``TASK_INGRESS``.
PRESENCE_TRANSITION = "rtech.presence.transition"


def agent_task(agent_id: str) -> str:
    """Return the per-agent dispatch subject (intake → triage agent).

    Mirrors the ratified per-agent scheme exactly (``rtech.agent.<id>.task``, the
    same string ``rtech_orchestrator.subjects.agent_task`` produces): a triage
    TaskFrame minted by the intake sink is published here so the always-on triage
    agent (the deployed ``agentd-<name>``) wakes on its own task lane. chatd is a
    dumb edge —
    it owns its ratified subject strings locally, matching the module's existing
    ``TASK_*`` builders, rather than importing the orchestrator package.

    Raises:
        ValueError: If ``agent_id`` is empty or whitespace-only — an unaddressed
            triage dispatch would publish to ``rtech.agent..task`` and never wake
            the agent, so the edge fails closed rather than mint a dead subject.
    """
    if not agent_id or not agent_id.strip():
        raise ValueError("agent_id must be a non-empty string")
    return AGENT_TASK_TEMPLATE.format(agent_id=agent_id)


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
    """Return the per-task result subject (orchestrator-derived done, publish-side)."""

    return TASK_RESULT_TEMPLATE.format(task_id=task_id)
