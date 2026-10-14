"""Track edge-minted web-chat task observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from loguru import logger
from pydantic import BaseModel
from rtech_frames.frames import StatusFrame, TaskFrame, TurnFrame

TaskScope = Literal["single", "duo", "trio"]
RenderedPayload = dict[str, object]

# The dashboard origin whose still-active tasks the Dash recovery endpoint
# lists. Only tasks minted with this origin edge are eligible; web-chat and any
# other origin are excluded, and Telegram/Discord tasks never enter this
# connector's registry at all.
DASH_ORIGIN_EDGE = "dash-dash"

# StatusFrame carries no terminal state (``rtech_frames`` limits it to
# working/input_required/blocked); a task with no observed status yet is
# reported with this minted sentinel. Completion is NOT a status — it is the
# orchestrator-derived result-subject arrival, which excludes the task instead.
MINTED_STATE = "minted"


class ActiveKingTaskItem(BaseModel):
    """Report one still-active dash-dash task for reload recovery.

    Carries only minimal typed state derived from the frame graph/orchestrator
    truth: the task id, its (dash-dash) origin edge, the derived scope, and the
    last observed typed status. It deliberately omits the private task text,
    requester, participants, and any frame history so the recovery surface
    leaks no private payload.
    """

    task_id: str
    origin_edge: str
    scope: TaskScope
    status: str


class TaskListItem(BaseModel):
    """Render one edge-minted task in the operator list."""

    task_id: str
    requester: str
    text: str
    scope: TaskScope
    participants: list[str]
    status: str
    result_received: bool
    frames: list[RenderedPayload]


@dataclass
class TaskRecord:
    """Hold connector-local observations for one minted task."""

    task: TaskFrame
    frames: list[RenderedPayload] = field(default_factory=list)
    status: str = MINTED_STATE
    result_received: bool = False


@dataclass
class TaskRegistry:
    """Keep edge-minted tasks and observed frame payloads."""

    records: dict[str, TaskRecord] = field(default_factory=dict)

    def add(self, task: TaskFrame) -> None:
        """Record a newly minted task."""

        self.records[task.id] = TaskRecord(task=task)

    def has(self, task_id: str) -> bool:
        """Return whether this edge minted the task."""

        return task_id in self.records

    def result_received(self, task_id: str) -> bool:
        """Return whether this edge observed a result for the task."""

        record = self.records.get(task_id)
        return record.result_received if record is not None else False

    def list_items(self) -> list[TaskListItem]:
        """Return task list items in mint order."""

        return [
            TaskListItem(
                task_id=record.task.id,
                requester=record.task.requester,
                text=record.task.text,
                scope=record.task.scope,
                participants=list(record.task.participants),
                status="result" if record.result_received else record.status,
                result_received=record.result_received,
                frames=list(record.frames),
            )
            for record in self.records.values()
        ]

    def list_active_dash_items(self) -> list[ActiveKingTaskItem]:
        """Return still-active dash-dash tasks with minimal typed state.

        A record is included only when its minted origin edge is
        ``dash-dash`` and the orchestrator has NOT derived done for it — done
        arrives as the result-subject publish this connector observes via
        ``record_result_arrival``, so ``result_received`` is the connector's
        mirror of orchestrator-derived completion. Completed, web-chat, and
        every other-origin task is excluded.

        The returned items carry no task text, requester, participants, or
        frame history: only the id, origin edge, derived scope, and last
        observed typed status.
        """

        return [
            ActiveKingTaskItem(
                task_id=record.task.id,
                origin_edge=record.task.origin_edge,
                scope=record.task.scope,
                status=record.status,
            )
            for record in self.records.values()
            if record.task.origin_edge == DASH_ORIGIN_EDGE and not record.result_received
        ]

    def record_frame(self, frame: TurnFrame | StatusFrame) -> RenderedPayload | None:
        """Store and render a typed frame observed on a task stream."""

        record = self.records.get(frame.task_id)
        if record is None:
            logger.warning(
                "webchat.registry.frame | ignored unknown task_id={}",
                frame.task_id,
            )
            return None
        payload = render_frame(frame)
        record.frames.append(payload)
        if isinstance(frame, StatusFrame):
            record.status = frame.state
        return payload

    def record_result_arrival(self, task_id: str) -> RenderedPayload | None:
        """Store result-subject arrival without decoding connector-side meaning."""

        record = self.records.get(task_id)
        if record is None:
            logger.warning("webchat.registry.result | ignored unknown task_id={}", task_id)
            return None
        record.result_received = True
        payload: RenderedPayload = {"type": "result", "task_id": task_id}
        record.frames.append(payload)
        return payload


def render_frame(frame: TurnFrame | StatusFrame) -> RenderedPayload:
    """Shape typed frames into JSON payloads for the browser."""

    if isinstance(frame, TurnFrame):
        return {
            "type": "turn",
            "task_id": frame.task_id,
            "agent_id": frame.agent_id,
            "visible_text": frame.visible_text,
            "stance": frame.stance.model_dump(mode="json"),
            "refs": frame.refs,
        }
    return {
        "type": "status",
        "task_id": frame.task_id,
        "agent_id": frame.agent_id,
        "state": frame.state,
    }
