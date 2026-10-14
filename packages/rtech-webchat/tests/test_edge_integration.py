"""Prove the web-chat HTTP/SSE edge contract end-to-end against a fake bus.

Hermetic, in-process, no live network: every test drives the real FastAPI
surface through :class:`fastapi.testclient.TestClient` and an in-memory
:class:`FakeFrameBus`. Typed NATS frames remain the source of truth — the edge
only mints, transports, and closes; ``done`` stays orchestrator-derived (the
stream closes on the bus result frame, it never emits a ``done`` frame). This is
the edge-integration coverage, layered over the unit-level
assertions already in ``test_app.py`` by exercising the actual streaming SSE
route over HTTP rather than the ``_stream_task_frames`` generator directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from collections.abc import AsyncIterator, Callable

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient
from httpx_sse import ServerSentEvent, aconnect_sse
from loguru import logger
from rtech_frames.codec import decode_frame, encode_frame
from rtech_frames.frames import (
    SCHEMA_VERSION,
    Stance,
    StatusFrame,
    SteerFrame,
    TaskFrame,
    TurnFrame,
)
from rtech_webchat import subjects
from rtech_webchat.app import AppState, create_app
from rtech_webchat.config import WebchatConfig

# Reuse the existing fakes/helpers rather than re-declaring the bus/heartbeat.
from .test_app import FakeFrameBus, _heartbeat

DASH_DASH = "dash-dash"
API_KEY = "edge-secret-token"


@pytest.fixture(autouse=True)
def _stack_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the REQUIRED stack id; dev harness only."""

    monkeypatch.setenv("RTECH_STACK_ID", "dev")


def _keyed_config() -> WebchatConfig:
    """Build a bearer-protected dev config fronting the dash-dash origin edge."""

    return WebchatConfig(stack="dev", api_key=API_KEY, origin_edge=DASH_DASH)


def _auth() -> dict[str, str]:
    """Return the valid bearer header for the keyed edge."""

    return {"Authorization": f"Bearer {API_KEY}"}


# --------------------------------------------------------------------------- #
# POST /api/tasks — transitional HTTP/SSE origin + derived scope on the frame  #
# --------------------------------------------------------------------------- #


def test_post_api_task_returns_id_and_publishes_dash_dash_task_frame() -> None:
    """POST returns task_id and publishes a TaskFrame with dash-dash origin/derived scope."""

    bus = FakeFrameBus()
    app = create_app(config=_keyed_config(), bus=bus, id_factory=lambda: "task-integ")

    with TestClient(app) as client:
        response = client.post(
            "/api/tasks",
            headers=_auth(),
            json={
                "text": "Coordinate the duo.",
                "requester": "dash",
                "participants": ["codex-a", "opus-b"],
            },
        )

    assert response.status_code == 200
    assert response.json() == {"task_id": "task-integ"}
    assert len(bus.published) == 1
    subject, data = bus.published[0]
    assert subject == subjects.TASK_INGRESS
    frame = decode_frame(data)
    assert isinstance(frame, TaskFrame)
    # origin_edge is transitional HTTP/SSE provenance, scope is DERIVED from count.
    assert frame.origin_edge == DASH_DASH
    assert frame.scope == "duo"
    assert frame.participants == ["codex-a", "opus-b"]
    assert frame.schema_version == SCHEMA_VERSION


def test_post_api_task_header_overrides_to_dash_dash_origin() -> None:
    """The X-Rtech-Origin-Edge header selects an allowlisted transitional origin."""

    bus = FakeFrameBus()
    app = create_app(
        config=WebchatConfig(stack="dev", api_key=API_KEY),
        bus=bus,
        id_factory=lambda: "task-hdr",
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/tasks",
            headers={**_auth(), "X-Rtech-Origin-Edge": DASH_DASH},
            json={"text": "Single via header.", "requester": "dash"},
        )

    assert response.status_code == 200
    frame = decode_frame(bus.published[0][1])
    assert isinstance(frame, TaskFrame)
    assert frame.origin_edge == DASH_DASH
    assert frame.scope == "single"


# --------------------------------------------------------------------------- #
# GET /api/tasks/{id}/stream — buffered replay AND live tail over real HTTP    #
# --------------------------------------------------------------------------- #


def test_sse_route_replays_buffered_frames_over_http_and_closes_without_done() -> None:
    """The real HTTP SSE route replays buffered StatusFrame/TurnFrame, then closes clean.

    Drives the ACTUAL streaming route over TestClient. A turn then a status are
    buffered on the bus BEFORE the client attaches; the bus result frame marks the
    task complete. Attaching then reads the whole SSE body: replay order is
    edge-owned (turn before status), each frame rides as ONE schema-v8 JSON object
    per ``data:`` line, and the stream closes on the bus result WITHOUT emitting a
    ``done``/``result`` frame — completion stays orchestrator/bus-derived.

    This covers ONLY the buffered-replay half of the route: every frame is on the
    bus before attach, so the sync ``TestClient`` can drain the whole body without
    deadlocking. The synchronous ``TestClient`` fully drives the streaming app
    before returning the response, so it cannot interleave an emit-after-attach; the
    HTTP-boundary LIVE-tail half (a frame delivered over HTTP *after* attach) is
    proven separately by
    ``test_sse_route_live_tails_post_attach_frames_over_http_and_closes_without_done``,
    which runs the real app on a loopback socket via its shipped uvicorn server.
    """

    bus = FakeFrameBus()
    app = create_app(config=_keyed_config(), bus=bus, id_factory=lambda: "task-stream")

    with TestClient(app) as client:
        client.post(
            "/api/tasks",
            headers=_auth(),
            json={"text": "Stream me.", "requester": "dash"},
        )
        # Buffer a turn, then a status, then the bus result — all before attach.
        # Frame ``task_id`` must match the stream's task or replay drops it.
        turn = TurnFrame(
            agent_id="agent-a",
            task_id="task-stream",
            visible_text="Buffered turn.",
            stance=Stance(
                position="working", because="The task is active.", needs_from=["agent-b"]
            ),
            refs=["ref-a"],
        )
        bus.emit(subjects.task_turn("task-stream"), _encode(turn))
        bus.emit(
            subjects.task_status("task-stream"),
            _encode(StatusFrame(task_id="task-stream", agent_id="agent-a", state="working")),
        )
        bus.emit(subjects.task_result("task-stream"), b"arrived")

        with client.stream(
            "GET",
            "/api/tasks/task-stream/stream",
            headers={**_auth(), "Accept": "text/event-stream"},
        ) as stream:
            assert stream.status_code == 200
            assert stream.headers["content-type"].startswith("text/event-stream")
            body = "".join(stream.iter_text())

    frames = _sse_data_objects(body)
    # Exactly the buffered turn then status replayed, in edge-owned order.
    assert [frame["frame_type"] for frame in frames] == ["turn", "status"]
    assert frames[0]["visible_text"] == "Buffered turn."
    assert frames[0]["schema_version"] == SCHEMA_VERSION
    assert frames[1]["state"] == "working"
    assert frames[1]["schema_version"] == SCHEMA_VERSION
    # Clean closure: the bus result closes the stream and emits NO done frame.
    assert "done" not in body
    assert '"frame_type":"result"' not in body.replace(" ", "")


async def test_sse_route_live_tails_post_attach_frames_over_http_and_closes_without_done() -> None:
    """The real HTTP SSE route delivers frames emitted AFTER attach, then closes clean.

    This is the HTTP-boundary LIVE-tail half the buffered-replay test cannot reach:
    nothing is on the bus at attach, so the stream enters its live-tail loop
    (``queue is not None``) and blocks on the queue. Only THEN — proven by a live
    listener registering on the replay buffer — does the test emit a turn and a
    status, and each must arrive over the actual SSE HTTP response in emit order,
    as ONE schema-v8 JSON object per ``data:`` line. The bus result then closes the
    stream WITHOUT any ``done``/``result`` frame; completion stays bus-derived.

    Runs the real app on a loopback socket via its shipped uvicorn server (already a
    production dependency), so the streaming route and the emitter run concurrently
    over genuine HTTP — the interleaving the synchronous ``TestClient`` cannot do.
    Synchronization is bounded and sleep-free: waits key off real replay/stream
    state (a registered live listener, the next SSE event) under ``asyncio``
    timeouts, never a fixed delay.
    """

    bus = FakeFrameBus()
    app = create_app(config=_keyed_config(), bus=bus, id_factory=lambda: "task-live")

    async with _serve_over_loopback(app) as base_url:
        async with httpx.AsyncClient(base_url=base_url) as client:
            minted = await client.post(
                "/api/tasks",
                headers=_auth(),
                json={"text": "Live-tail me.", "requester": "dash"},
            )
            assert minted.status_code == 200
            state: AppState = app.state.webchat

            async with aconnect_sse(
                client,
                "GET",
                "/api/tasks/task-live/stream",
                headers={**_auth(), "Accept": "text/event-stream"},
            ) as event_source:
                response = event_source.response
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                events = event_source.aiter_sse()

                # The stream has no buffered frames, so it is now parked in the
                # live-tail loop: a live listener on the replay buffer is the proof
                # that attach happened and the queue is being awaited.
                await _wait_for_live_listener(state, "task-live")

                # Emit a turn, then a status, AFTER attach — the live-tail path.
                turn = TurnFrame(
                    agent_id="agent-a",
                    task_id="task-live",
                    visible_text="Live turn.",
                    stance=Stance(
                        position="working",
                        because="The task is active.",
                        needs_from=["agent-b"],
                    ),
                    refs=["ref-a"],
                )
                await bus.emit_async(subjects.task_turn("task-live"), encode_frame(turn))
                turn_event = await _next_sse(events)
                await bus.emit_async(
                    subjects.task_status("task-live"),
                    encode_frame(
                        StatusFrame(task_id="task-live", agent_id="agent-a", state="working")
                    ),
                )
                status_event = await _next_sse(events)

                # Each post-attach frame rode over HTTP in emit order, one schema-v8
                # JSON object per ``data:`` line — no done/result frame in either.
                turn_frame = json.loads(turn_event)
                status_frame = json.loads(status_event)
                assert turn_frame["frame_type"] == "turn"
                assert turn_frame["visible_text"] == "Live turn."
                assert turn_frame["schema_version"] == SCHEMA_VERSION
                assert status_frame["frame_type"] == "status"
                assert status_frame["state"] == "working"
                assert status_frame["schema_version"] == SCHEMA_VERSION

                # The bus result closes the stream: iteration ends with NO further
                # data line — completion stays orchestrator/bus-derived.
                await bus.emit_async(subjects.task_result("task-live"), b"arrived")
                assert await _drain_remaining_sse(events) == []


def test_sse_stream_unknown_task_is_404() -> None:
    """An unminted task has no registry entry; the stream route 404s, never 200-empties."""

    bus = FakeFrameBus()
    app = create_app(config=_keyed_config(), bus=bus)

    with TestClient(app) as client:
        response = client.get("/api/tasks/nope/stream", headers=_auth())

    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# POST steer — typed steer + interrupt, broadcast-default target               #
# --------------------------------------------------------------------------- #


def test_post_steer_publishes_typed_interrupt_frame() -> None:
    """Steer mints a typed SteerFrame carrying the interrupt control field."""

    bus = FakeFrameBus()
    app = create_app(config=_keyed_config(), bus=bus)

    with TestClient(app) as client:
        response = client.post(
            "/api/tasks/task-steer/steer",
            headers=_auth(),
            json={"text": "Refocus.", "target_agent": "agent-a", "interrupt": True},
        )

    assert response.status_code == 200
    assert response.json() == {"accepted": True}
    subject, data = bus.published[0]
    assert subject == subjects.task_steer("task-steer")
    frame = decode_frame(data)
    assert isinstance(frame, SteerFrame)
    assert frame == SteerFrame(
        task_id="task-steer",
        text="Refocus.",
        target_agent="agent-a",
        interrupt=True,
    )


def test_post_steer_defaults_to_broadcast_target() -> None:
    """Omitting target_agent broadcasts: the frame carries the frames-owned constant."""

    from rtech_frames import BROADCAST_TARGET

    bus = FakeFrameBus()
    app = create_app(config=_keyed_config(), bus=bus)

    with TestClient(app) as client:
        response = client.post(
            "/api/tasks/task-steer/steer",
            headers=_auth(),
            json={"text": "Everyone hold.", "interrupt": False},
        )

    assert response.status_code == 200
    frame = decode_frame(bus.published[0][1])
    assert isinstance(frame, SteerFrame)
    assert frame.target_agent == BROADCAST_TARGET
    assert frame.interrupt is False


# --------------------------------------------------------------------------- #
# GET /api/agents — typed presence rows                                        #
# --------------------------------------------------------------------------- #


def test_api_agents_returns_typed_presence_rows() -> None:
    """A heartbeat surfaces a typed presence row and a healthy pipeline verdict.

    The successful lifespan subscribe marks the pipeline ``healthy``,
    so Dash reads a non-empty list as authoritative liveness. The body carries
    exactly the typed rows plus the typed ``pipeline`` field — no extra keys.
    """

    bus = FakeFrameBus()
    app = create_app(config=_keyed_config(), bus=bus)

    with TestClient(app) as client:
        bus.emit(subjects.agent_heartbeat_wildcard("dev"), _heartbeat("agent-a"))
        response = client.get("/api/agents", headers=_auth())

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"agents", "pipeline"}
    assert body["pipeline"] == "healthy"
    assert len(body["agents"]) == 1
    row = body["agents"][0]
    assert set(row) == {"agent_id", "last_seen_ms", "alive"}
    assert row["agent_id"] == "agent-a"
    assert row["alive"] is True
    assert isinstance(row["last_seen_ms"], int) and row["last_seen_ms"] > 0


def test_api_agents_keyed_edge_reports_subscription_failed_over_http() -> None:
    """A down heartbeat feed surfaces subscription_failed through the keyed edge.

    Unit coverage in ``test_app.py`` proves the verdict on the open edge; this
    layers the bearer-gated dash-dash origin edge on top: a failed subscribe still
    answers HTTP 200 with a valid bearer, so Dash reads the typed
    ``subscription_failed`` verdict rather than a proxy failure or a
    silent "zero agents" — the empty list is explicitly NOT liveness truth here.
    """

    bus = FakeFrameBus()

    async def _raise_on_subscribe(subject: str, handler: Callable[[bytes], object]) -> object:
        raise RuntimeError("heartbeat subscription rejected")

    bus.subscribe = _raise_on_subscribe  # type: ignore[method-assign]
    app = create_app(config=_keyed_config(), bus=bus)

    with TestClient(app) as client:
        response = client.get("/api/agents", headers=_auth())

    assert response.status_code == 200
    assert response.json() == {"agents": [], "pipeline": "subscription_failed"}


# --------------------------------------------------------------------------- #
# Bearer auth — 401 for missing/bad key when configured                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("post", "/api/tasks", {"text": "x", "requester": "dash"}),
        ("get", "/api/tasks/task-x/stream", None),
        ("post", "/api/tasks/task-x/steer", {"text": "x"}),
        ("get", "/api/agents", None),
    ],
)
def test_keyed_routes_401_without_valid_bearer(
    method: str, path: str, payload: dict[str, str] | None
) -> None:
    """Every /api route is bearer-gated: 401 on missing AND wrong key when configured."""

    bus = FakeFrameBus()
    app = create_app(config=_keyed_config(), bus=bus, id_factory=lambda: "task-x")

    with TestClient(app) as client:
        caller = getattr(client, method)
        kwargs = {"json": payload} if payload is not None else {}
        missing = caller(path, **kwargs)
        wrong = caller(path, headers={"Authorization": "Bearer nope"}, **kwargs)

    assert missing.status_code == 401
    assert wrong.status_code == 401
    # Auth fails closed BEFORE any publish reaches the bus.
    assert bus.published == []


# --------------------------------------------------------------------------- #
# Security floor — no secrets or raw private payloads in logs                  #
# --------------------------------------------------------------------------- #


def test_edge_does_not_log_bearer_key_or_raw_task_text() -> None:
    """The edge logs sanitized summaries only: no bearer key, no raw task text."""

    bus = FakeFrameBus()
    app = create_app(config=_keyed_config(), bus=bus, id_factory=lambda: "task-log")
    private_body = "this is the private task body that must not be logged"

    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(message), level="DEBUG")
    try:
        with TestClient(app) as client:
            accepted = client.post(
                "/api/tasks",
                headers=_auth(),
                json={"text": private_body, "requester": "dash"},
            )
            rejected = client.post(
                "/api/tasks",
                headers={"Authorization": "Bearer wrong-edge-key"},
                json={"text": private_body, "requester": "dash"},
            )
    finally:
        logger.remove(sink_id)

    assert accepted.status_code == 200
    assert rejected.status_code == 401
    joined = "".join(records)
    # The bearer key is never rendered into a log line...
    assert API_KEY not in joined
    assert "wrong-edge-key" not in joined
    # ...nor is the raw task body — only sanitized summaries (ids, lengths).
    assert private_body not in joined


# Bounded synchronization budget for the real-HTTP live-tail test: every await
# below is capped so a broken live path fails fast instead of hanging the suite.
_LIVE_TAIL_TIMEOUT_SECONDS = 5.0


def _free_loopback_port() -> int:
    """Reserve and release an ephemeral loopback port for the test server."""

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])
    finally:
        probe.close()


@contextlib.asynccontextmanager
async def _serve_over_loopback(app: object) -> AsyncIterator[str]:
    """Serve ``app`` on a loopback socket via its shipped uvicorn server.

    Yields the base URL. The server runs as a task on the test's own event loop,
    so the ``FakeFrameBus`` handlers, the streaming route, and the emitter all
    share one loop and interleave over a genuine HTTP socket. Startup/shutdown are
    bounded: no fixed sleeps, only state polls under a timeout.
    """

    port = _free_loopback_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    serve_task = asyncio.create_task(server.serve())
    try:
        await asyncio.wait_for(_await_true(lambda: server.started), _LIVE_TAIL_TIMEOUT_SECONDS)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(serve_task, _LIVE_TAIL_TIMEOUT_SECONDS)


async def _await_true(predicate: Callable[[], bool]) -> None:
    """Yield the loop until ``predicate`` holds (caller bounds this with a timeout)."""

    while not predicate():
        await asyncio.sleep(0)


async def _wait_for_live_listener(state: AppState, task_id: str) -> None:
    """Wait until the stream has attached and parked in its live-tail loop.

    A live listener on the task's replay buffer proves the generator ran
    ``attach()`` and is now awaiting the queue — i.e. the ``queue is not None``
    live-tail branch, not buffered replay. Bounded by a timeout, sleep-free.
    """

    def _attached() -> bool:
        buffer = state.replay.buffers.get(task_id)
        return buffer is not None and bool(buffer.listeners)

    await asyncio.wait_for(_await_true(_attached), _LIVE_TAIL_TIMEOUT_SECONDS)


async def _next_sse(events: AsyncIterator[ServerSentEvent]) -> str:
    """Return the next SSE event's ``data`` payload, bounded by a timeout."""

    event = await asyncio.wait_for(events.__anext__(), _LIVE_TAIL_TIMEOUT_SECONDS)
    return event.data


async def _drain_remaining_sse(events: AsyncIterator[ServerSentEvent]) -> list[str]:
    """Return any SSE data payloads emitted before the stream closes.

    A clean bus-driven close ends iteration with no further event, so the expected
    result is an empty list; anything else means a done/result frame leaked.
    """

    payloads: list[str] = []
    try:
        while True:
            payloads.append(await _next_sse(events))
    except StopAsyncIteration:
        return payloads


def _encode(frame: StatusFrame | TurnFrame) -> bytes:
    """Encode a frame to wire bytes (thin wrapper to keep call sites terse)."""

    return encode_frame(frame)


def _sse_data_objects(body: str) -> list[dict[str, object]]:
    """Parse an SSE body into one decoded JSON object per ``data:`` line.

    Proves the wire shape the browser relies on: exactly one frame per ``data:``
    line, blank-line separated, and no bare ``done``/sentinel line.
    """

    frames: list[dict[str, object]] = []
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload:
            continue
        frames.append(json.loads(payload))
    return frames
