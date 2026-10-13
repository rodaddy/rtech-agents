"""Expose the bearer-protected web-chat edge API."""

from __future__ import annotations

import hmac
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field
from rtech_frames.codec import encode_frame
from rtech_frames.frames import BROADCAST_TARGET, SteerFrame, TaskFrame, validate_participants

from . import subjects
from .config import ALLOWED_ORIGIN_EDGES, WebchatConfig
from .presence import AgentPresenceResponse
from .registry import ActiveKingTaskItem

if TYPE_CHECKING:
    from .app import AppState

TaskScope = Literal["single", "duo", "trio"]
DEFAULT_TASK_SCOPE: TaskScope = "single"
# The broadcast wire token is owned by rtech-frames (issue #76): this edge only
# re-exposes it as its steer default, never re-declares the literal.
DEFAULT_STEER_TARGET = BROADCAST_TARGET

# Participant count -> derived scope. 0 or 1 participant is a single-scope task;
# more than 3 participants has no scope and is rejected at the edge.
_COUNT_TO_SCOPE: dict[int, TaskScope] = {0: "single", 1: "single", 2: "duo", 3: "trio"}
_MAX_PARTICIPANTS = 3


class ApiTaskRequest(BaseModel):
    """Validate API task ingress JSON.

    ``scope`` is optional and DERIVED server-side from ``participants`` count; a
    client-supplied ``scope`` is only accepted when it matches the derived value.
    """

    text: str = Field(min_length=1)
    requester: str = Field(min_length=1)
    scope: TaskScope | None = Field(default=None)
    participants: list[str] = Field(default_factory=list)
    origin_edge: str | None = Field(default=None, min_length=1)


def _derive_scope(participants: list[str], requested: TaskScope | None) -> TaskScope:
    """Derive scope from participant count and validate any client-sent scope.

    Args:
        participants: The requested participant agent ids.
        requested: The client-supplied scope, or ``None`` to derive silently.

    Returns:
        The scope implied by the participant count.

    Raises:
        HTTPException: 400 when there are more than three participants, when the
            participant ids are duplicated/blank/subject-unsafe, or when a
            client-supplied ``scope`` disagrees with the derived value.
    """
    count = len(participants)
    derived = _COUNT_TO_SCOPE.get(count)
    if derived is None:
        logger.warning("webchat.api.scope | too many participants count={}", count)
        raise HTTPException(
            status_code=400,
            detail=f"at most {_MAX_PARTICIPANTS} participants are supported, got {count}",
        )
    try:
        # Duplicate/blank/unsafe ids are rejected at the edge (400), before the
        # frame's own invariant would fail them server-side (issue #75 round 2).
        validate_participants(participants, derived)
    except ValueError as exc:
        logger.warning("webchat.api.scope | invalid participants reason={}", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if requested is not None and requested != derived:
        logger.warning(
            "webchat.api.scope | scope/participants mismatch requested={} derived={}",
            requested,
            derived,
        )
        raise HTTPException(
            status_code=400,
            detail=f"scope {requested!r} does not match {count} participants (derived {derived!r})",
        )
    logger.debug("webchat.api.scope | derived scope={} count={}", derived, count)
    return derived


class ApiTaskResponse(BaseModel):
    """Return the minted task id."""

    task_id: str


class ApiSteerRequest(BaseModel):
    """Validate API steer ingress JSON."""

    text: str = Field(min_length=1)
    target_agent: str | None = Field(default=None, min_length=1)
    interrupt: bool = False


class ApiSteerResponse(BaseModel):
    """Report whether steer was accepted for publish."""

    accepted: bool


async def _empty_auth_dependency() -> None:
    """Allow local development when no API key is configured."""


def register_api_routes(app: FastAPI, state: AppState, config: WebchatConfig) -> None:
    """Register the `/api` task and stream routes."""

    router = APIRouter(
        prefix="/api",
        dependencies=[Depends(_build_auth_dependency(config.api_key))],
    )

    @router.post("/tasks", response_model=ApiTaskResponse)
    async def create_api_task(
        request: ApiTaskRequest,
        x_rtech_origin_edge: str | None = Header(default=None),
    ) -> ApiTaskResponse:
        logger.debug("webchat.api.task | entry requester_len={}", len(request.requester))
        origin_edge = _resolve_origin_edge(
            configured=config.origin_edge,
            requested=request.origin_edge,
            header=x_rtech_origin_edge,
        )
        scope = _derive_scope(request.participants, request.scope)
        task = TaskFrame(
            id=state.id_factory(),
            origin_edge=origin_edge,
            requester=request.requester,
            text=request.text,
            scope=scope,
            participants=request.participants,
            stack=config.stack,
            created_at=datetime.now(UTC),
        )
        state.registry.add(task)
        await state.replay.start_task(task.id)
        await state.bus.publish(subjects.TASK_INGRESS, encode_frame(task))
        logger.info("webchat.api.task | published task_id={} origin={}", task.id, origin_edge)
        return ApiTaskResponse(task_id=task.id)

    @router.get("/tasks/{task_id}/stream")
    async def stream_api_task(task_id: str) -> StreamingResponse:
        logger.debug("webchat.api.stream | entry task_id={}", task_id)
        if not state.registry.has(task_id):
            logger.warning("webchat.api.stream | unknown task_id={}", task_id)
            raise HTTPException(status_code=404, detail="unknown task")
        return StreamingResponse(
            _stream_task_frames(state, task_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @router.post("/tasks/{task_id}/steer", response_model=ApiSteerResponse)
    async def steer_api_task(task_id: str, request: ApiSteerRequest) -> ApiSteerResponse:
        logger.debug("webchat.api.steer | entry task_id={}", task_id)
        frame = SteerFrame(
            task_id=task_id,
            text=request.text,
            target_agent=request.target_agent or DEFAULT_STEER_TARGET,
            interrupt=request.interrupt,
        )
        await state.bus.publish(subjects.task_steer(task_id), encode_frame(frame))
        logger.info(
            "webchat.api.steer | published task_id={} target={} interrupt={}",
            task_id,
            frame.target_agent,
            request.interrupt,
        )
        return ApiSteerResponse(accepted=True)

    @router.get("/agents", response_model=AgentPresenceResponse)
    async def list_api_agents() -> AgentPresenceResponse:
        logger.debug("webchat.api.agents | entry")
        response = state.presence.list_agents()
        logger.info("webchat.api.agents | rendered count={}", len(response.agents))
        return response

    @router.get("/dash/active-tasks", response_model=list[ActiveKingTaskItem])
    async def list_dash_active_tasks() -> list[ActiveKingTaskItem]:
        # Bearer-gated recovery surface for a reloaded Dash dashboard: only
        # still-active dash-dash tasks, with minimal typed state derived from
        # the frame graph/orchestrator truth. Completion is orchestrator-
        # derived (result-subject arrival), so completed tasks are filtered out
        # rather than reported; no history, archive, or private text (issue
        # #144). The dumb edge renders truth, it never owns it.
        logger.debug("webchat.api.dash.active | entry")
        items = state.registry.list_active_dash_items()
        logger.info("webchat.api.dash.active | rendered count={}", len(items))
        return items

    app.include_router(router)


def _build_auth_dependency(api_key: str | None) -> Callable[..., Awaitable[None]]:
    """Return a dependency that enforces bearer auth when configured."""

    if api_key is None:
        return _empty_auth_dependency

    async def require_bearer_auth(request: Request) -> None:
        logger.debug("webchat.api.auth | entry")
        authorization = request.headers.get("authorization")
        if authorization is None:
            logger.warning("webchat.api.auth | missing bearer")
            raise _unauthorized()
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(token, api_key):
            logger.warning("webchat.api.auth | rejected bearer")
            raise _unauthorized()
        logger.info("webchat.api.auth | accepted")

    return require_bearer_auth


def _unauthorized() -> HTTPException:
    """Return a stable bearer authentication failure."""

    return HTTPException(
        status_code=401,
        detail="unauthorized",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _resolve_origin_edge(
    *,
    configured: str,
    requested: str | None,
    header: str | None,
) -> str:
    """Resolve an allowlisted origin edge without accepting freeform input."""

    candidates = [value for value in (header, requested, configured) if value is not None]
    origin_edge = candidates[0]
    if origin_edge not in ALLOWED_ORIGIN_EDGES:
        logger.warning("webchat.api.origin | rejected origin={}", origin_edge)
        raise HTTPException(status_code=422, detail="origin_edge is not allowlisted")
    logger.debug("webchat.api.origin | resolved origin={}", origin_edge)
    return origin_edge


async def _stream_task_frames(state: AppState, task_id: str) -> AsyncIterator[str]:
    """Yield encoded TurnFrame and StatusFrame bytes as SSE data lines."""

    attachment = await state.replay.attach(task_id)
    try:
        logger.info("webchat.api.stream | attached task_id={}", task_id)
        for event in attachment.snapshot:
            if event.kind == "result":
                logger.info("webchat.api.stream | replayed result close task_id={}", task_id)
                return
            if event.data is not None:
                yield f"data: {event.data.decode('utf-8')}\n\n"
        if attachment.queue is None:
            logger.info("webchat.api.stream | completed replay close task_id={}", task_id)
            return
        while True:
            event = await attachment.queue.get()
            if event.kind == "result":
                logger.info("webchat.api.stream | result close task_id={}", task_id)
                return
            if event.data is not None:
                yield f"data: {event.data.decode('utf-8')}\n\n"
    finally:
        await state.replay.detach(task_id, attachment.queue)
        logger.info("webchat.api.stream | closed task_id={}", task_id)
