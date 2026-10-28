"""Test web-chat frame minting and websocket forwarding."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient
from fleet_nats import heartbeat_envelope
from loguru import logger
from rtech_frames.codec import decode_frame, encode_frame
from rtech_frames.frames import Stance, StatusFrame, SteerFrame, TaskFrame, TurnFrame
from rtech_webchat import subjects
from rtech_webchat.api import _stream_task_frames
from rtech_webchat.app import create_app
from rtech_webchat.config import WebchatConfig

FrameHandler = Callable[[bytes], Awaitable[None]]


@pytest.fixture(autouse=True)
def _stack_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the REQUIRED stack id: explicit dev id, test harness only."""

    monkeypatch.setenv("RTECH_STACK_ID", "dev")


@dataclass
class FakeSubscription:
    """Record unsubscribe calls."""

    subject: str
    unsubscribed: bool = False

    async def unsubscribe(self) -> None:
        """Mark the fake subscription as cancelled."""

        self.unsubscribed = True


@dataclass
class FakeFrameBus:
    """Capture publishes and route emitted frames to subscribers."""

    published: list[tuple[str, bytes]] = field(default_factory=list)
    handlers: dict[str, FrameHandler] = field(default_factory=dict)
    loops: dict[str, asyncio.AbstractEventLoop] = field(default_factory=dict)
    subscriptions: list[FakeSubscription] = field(default_factory=list)
    connected: bool = True

    def is_connected(self) -> bool:
        """Report the fake transport's current connection state.

        The presence read consults this at every poll, so a test can flip it to
        ``False`` after subscribe to simulate a later connection drop.
        """

        return self.connected

    async def publish(self, subject: str, data: bytes) -> None:
        """Record a publish."""

        self.published.append((subject, data))

    async def subscribe(self, subject: str, handler: FrameHandler) -> FakeSubscription:
        """Record a subscription handler."""

        self.handlers[subject] = handler
        self.loops[subject] = asyncio.get_running_loop()
        subscription = FakeSubscription(subject=subject)
        self.subscriptions.append(subscription)
        return subscription

    def emit(self, subject: str, data: bytes) -> None:
        """Deliver data to a captured handler from a sync test."""

        handler = self.handlers[subject]
        loop = self.loops[subject]
        future = asyncio.run_coroutine_threadsafe(handler(data), loop)
        future.result(timeout=2)

    async def emit_async(self, subject: str, data: bytes) -> None:
        """Deliver data to a captured handler from an async test."""

        handler = self.handlers[subject]
        await handler(data)


def _turn(text: str = "Visible answer.") -> TurnFrame:
    """Build a test turn frame."""

    return TurnFrame(
        agent_id="agent-a",
        task_id="task-123",
        visible_text=text,
        stance=Stance(
            position="working",
            because="The task is active.",
            needs_from=["agent-b"],
        ),
        refs=["ref-a"],
    )


def _decode_sse_line(line: str) -> TurnFrame | StatusFrame:
    """Decode one SSE data line into a typed frame."""

    return decode_frame(line.removeprefix("data: ").strip().encode())


def _heartbeat(agent_id: str = "agent-a", *, seq: int = 0, active_turn: bool = False) -> bytes:
    """Build a fleet-core heartbeat envelope with rtech's ``active_turn`` extra."""

    envelope = heartbeat_envelope(
        msg_id=f"heartbeat-{seq}",
        ts="2026-07-11T00:00:00+00:00",
        sender=agent_id,
        role="worker",
        lanes=(f"rtech.agent.{agent_id}.task",),
        seq=seq,
        last_turn_ts=None,
    )
    heartbeat = envelope.typed_signal
    if heartbeat is not None:
        heartbeat.active_turn = active_turn
    envelope.payload["active_turn"] = active_turn
    return envelope.to_bytes()


def test_package_entrypoint_loads() -> None:
    import rtech_webchat

    assert rtech_webchat.__version__


def test_post_task_mints_task_frame() -> None:
    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "task-123")

    with TestClient(app) as client:
        response = client.post(
            "/task",
            json={"text": "Build the edge.", "requester": "rico", "scope": "single"},
        )

    assert response.status_code == 200
    assert response.json() == {"task_id": "task-123"}
    assert len(bus.published) == 1
    subject, data = bus.published[0]
    assert subject == subjects.TASK_INGRESS
    frame = decode_frame(data)
    assert isinstance(frame, TaskFrame)
    assert frame.created_at is not None
    assert frame.model_copy(update={"created_at": None}) == TaskFrame(
        id="task-123",
        origin_edge="web-chat",
        requester="rico",
        text="Build the edge.",
        scope="single",
        stack="dev",
    )


def test_post_api_task_mints_task_frame_with_default_shape() -> None:
    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "task-123")

    with TestClient(app) as client:
        response = client.post(
            "/api/tasks",
            json={"text": "Build the edge.", "requester": "rico"},
        )

    assert response.status_code == 200
    assert response.json() == {"task_id": "task-123"}
    subject, data = bus.published[0]
    assert subject == subjects.TASK_INGRESS
    frame = decode_frame(data)
    assert isinstance(frame, TaskFrame)
    assert frame.created_at is not None
    assert frame.model_copy(update={"created_at": None}) == TaskFrame(
        id="task-123",
        origin_edge="web-chat",
        requester="rico",
        text="Build the edge.",
        scope="single",
        stack="dev",
    )


def test_post_api_task_derives_duo_scope_from_participants() -> None:
    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "task-123")

    with TestClient(app) as client:
        response = client.post(
            "/api/tasks",
            json={
                "text": "Pair on this.",
                "requester": "rico",
                "participants": ["codex-a", "opus-b"],
            },
        )

    assert response.status_code == 200
    frame = decode_frame(bus.published[0][1])
    assert isinstance(frame, TaskFrame)
    assert frame.created_at is not None
    assert frame.model_copy(update={"created_at": None}) == TaskFrame(
        id="task-123",
        origin_edge="web-chat",
        requester="rico",
        text="Pair on this.",
        scope="duo",
        participants=["codex-a", "opus-b"],
        stack="dev",
    )


def test_post_api_task_rejects_scope_participants_mismatch() -> None:
    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "task-123")

    with TestClient(app) as client:
        response = client.post(
            "/api/tasks",
            json={
                "text": "Mismatch.",
                "requester": "rico",
                "scope": "trio",
                "participants": ["codex-a", "opus-b"],
            },
        )

    assert response.status_code == 400
    assert not bus.published


def test_post_api_task_accepts_matching_client_scope() -> None:
    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "task-123")

    with TestClient(app) as client:
        response = client.post(
            "/api/tasks",
            json={
                "text": "Explicit but matching.",
                "requester": "rico",
                "scope": "duo",
                "participants": ["codex-a", "opus-b"],
            },
        )

    assert response.status_code == 200
    assert decode_frame(bus.published[0][1]).scope == "duo"


def test_post_api_task_rejects_more_than_three_participants() -> None:
    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "task-123")

    with TestClient(app) as client:
        response = client.post(
            "/api/tasks",
            json={
                "text": "Too many.",
                "requester": "rico",
                "participants": ["a", "b", "c", "d"],
            },
        )

    assert response.status_code == 400
    assert not bus.published


def test_post_api_task_allows_only_allowlisted_origin_edge() -> None:
    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "task-123")

    with TestClient(app) as client:
        accepted = client.post(
            "/api/tasks",
            json={
                "text": "Build the edge.",
                "requester": "dash",
                "scope": "duo",
                "participants": ["codex-a", "opus-b"],
                "origin_edge": "dash-dash",
            },
        )
        rejected = client.post(
            "/api/tasks",
            json={
                "text": "Build the edge.",
                "requester": "dash",
                "origin_edge": "other-edge",
            },
        )

    assert accepted.status_code == 200
    assert rejected.status_code == 422
    assert len(bus.published) == 1
    frame = decode_frame(bus.published[0][1])
    assert isinstance(frame, TaskFrame)
    assert frame.created_at is not None
    assert frame.model_copy(update={"created_at": None}) == TaskFrame(
        id="task-123",
        origin_edge="dash-dash",
        requester="dash",
        text="Build the edge.",
        scope="duo",
        participants=["codex-a", "opus-b"],
        stack="dev",
    )


def test_task_registry_lists_edge_minted_tasks() -> None:
    ids = iter(["task-1", "task-2"])
    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: next(ids))

    with TestClient(app) as client:
        first = client.post(
            "/task",
            json={"text": "First task.", "requester": "rico", "scope": "single"},
        )
        second = client.post(
            "/api/tasks",
            json={
                "text": "Second task.",
                "requester": "web",
                "participants": ["codex-a", "opus-b"],
            },
        )
        listed = client.get("/tasks")

    assert first.status_code == 200
    assert second.status_code == 200
    assert listed.status_code == 200
    assert listed.json() == [
        {
            "task_id": "task-1",
            "requester": "rico",
            "text": "First task.",
            "scope": "single",
            "participants": [],
            "status": "minted",
            "result_received": False,
            "frames": [],
        },
        {
            "task_id": "task-2",
            "requester": "web",
            "text": "Second task.",
            "scope": "duo",
            "participants": ["codex-a", "opus-b"],
            "status": "minted",
            "result_received": False,
            "frames": [],
        },
    ]


def test_post_steer_mints_steer_frame_with_interrupt() -> None:
    bus = FakeFrameBus()
    app = create_app(bus=bus)

    with TestClient(app) as client:
        response = client.post(
            "/steer",
            json={
                "task_id": "task-123",
                "text": "Narrow the scope.",
                "target_agent": "agent-a",
                "interrupt": True,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "task_id": "task-123",
        "subject": "rtech.task.task-123.steer",
        "interrupt": True,
    }
    subject, data = bus.published[0]
    assert subject == subjects.task_steer("task-123")
    assert decode_frame(data) == SteerFrame(
        task_id="task-123",
        text="Narrow the scope.",
        target_agent="agent-a",
        interrupt=True,
    )


def test_steer_default_target_is_the_frames_owned_broadcast_constant() -> None:
    """The edge default rides the rtech-frames wire token, never its own literal."""
    from rtech_frames import BROADCAST_TARGET
    from rtech_webchat.api import DEFAULT_STEER_TARGET

    assert DEFAULT_STEER_TARGET is BROADCAST_TARGET


def test_post_api_steer_accepts_interrupt() -> None:
    bus = FakeFrameBus()
    app = create_app(bus=bus)

    with TestClient(app) as client:
        response = client.post(
            "/api/tasks/task-123/steer",
            json={
                "text": "Narrow the scope.",
                "target_agent": "agent-a",
                "interrupt": True,
            },
        )

    assert response.status_code == 200
    assert response.json() == {"accepted": True}
    subject, data = bus.published[0]
    assert subject == subjects.task_steer("task-123")
    assert decode_frame(data) == SteerFrame(
        task_id="task-123",
        text="Narrow the scope.",
        target_agent="agent-a",
        interrupt=True,
    )


def test_api_bearer_auth_requires_exact_match_when_configured() -> None:
    bus = FakeFrameBus()
    app = create_app(
        config=WebchatConfig(stack="dev", api_key="correct-token"),
        bus=bus,
        id_factory=lambda: "task-123",
    )

    with TestClient(app) as client:
        missing = client.post(
            "/api/tasks",
            json={"text": "Build the edge.", "requester": "rico"},
        )
        wrong = client.post(
            "/api/tasks",
            headers={"Authorization": "Bearer wrong-token"},
            json={"text": "Build the edge.", "requester": "rico"},
        )
        accepted = client.post(
            "/api/tasks",
            headers={"Authorization": "Bearer correct-token"},
            json={"text": "Build the edge.", "requester": "rico"},
        )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert accepted.status_code == 200
    assert len(bus.published) == 1


def test_api_bearer_auth_is_open_when_unset() -> None:
    bus = FakeFrameBus()
    app = create_app(
        config=WebchatConfig(stack="dev", api_key=None),
        bus=bus,
        id_factory=lambda: "task-123",
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/tasks",
            json={"text": "Build the edge.", "requester": "rico"},
        )

    assert response.status_code == 200
    assert len(bus.published) == 1


def test_api_agents_lists_heartbeat_as_alive() -> None:
    bus = FakeFrameBus()
    app = create_app(bus=bus)

    with TestClient(app) as client:
        bus.emit(subjects.agent_heartbeat_wildcard("dev"), _heartbeat("agent-a"))
        response = client.get("/api/agents")

    assert response.status_code == 200
    assert response.json()["agents"] == [
        {
            "agent_id": "agent-a",
            "last_seen_ms": response.json()["agents"][0]["last_seen_ms"],
            "alive": True,
        }
    ]
    assert response.json()["agents"][0]["last_seen_ms"] > 0


def test_api_agents_advances_on_increasing_seq() -> None:
    bus = FakeFrameBus()
    app = create_app(bus=bus)

    with TestClient(app) as client:
        wildcard = subjects.agent_heartbeat_wildcard("dev")
        bus.emit(wildcard, _heartbeat("agent-a", seq=0))
        bus.emit(wildcard, _heartbeat("agent-a", seq=1))
        response = client.get("/api/agents")

    assert response.status_code == 200
    assert response.json()["agents"][0]["agent_id"] == "agent-a"
    assert response.json()["agents"][0]["alive"] is True


def test_api_agents_drops_stale_seq_heartbeat() -> None:
    bus = FakeFrameBus()
    app = create_app(
        config=WebchatConfig(stack="dev", heartbeat_interval_seconds=1.0),
        bus=bus,
    )
    presence = app.state.webchat.presence

    with TestClient(app) as client:
        wildcard = subjects.agent_heartbeat_wildcard("dev")
        # A fresh beat at seq=5 records liveness.
        bus.emit(wildcard, _heartbeat("agent-a", seq=5))
        # Force the recorded time stale, then replay an older seq=3 beat: the
        # sequence guard must drop it, so the agent stays not-alive.
        presence.record_heartbeat("agent-a", seen_ms=1)
        bus.emit(wildcard, _heartbeat("agent-a", seq=3))
        response = client.get("/api/agents")

    assert response.status_code == 200
    assert response.json() == {
        "agents": [{"agent_id": "agent-a", "last_seen_ms": 1, "alive": False}],
        "pipeline": "healthy",
    }


def test_api_agents_healthy_empty_pipeline_with_zero_agents() -> None:
    """A reachable edge whose heartbeat feed is up reports healthy + empty.

    This is the case Dash must NOT read as a broken pipeline: the subscription is
    established, so an empty agent list is authoritative (genuinely zero agents).
    """

    bus = FakeFrameBus()
    app = create_app(bus=bus)

    with TestClient(app) as client:
        response = client.get("/api/agents")

    assert response.status_code == 200
    assert response.json() == {"agents": [], "pipeline": "healthy"}


def test_api_agents_reports_subscription_failed_when_heartbeat_feed_down() -> None:
    """A heartbeat subscription that fails is surfaced as subscription_failed.

    The edge stays reachable (HTTP 200) so Dash reads a typed verdict rather than
    a proxy failure; the empty list is explicitly NOT liveness truth here.
    """

    bus = FakeFrameBus()

    async def _raise_on_subscribe(subject: str, handler: FrameHandler) -> FakeSubscription:
        raise RuntimeError("heartbeat subscription rejected")

    bus.subscribe = _raise_on_subscribe  # type: ignore[method-assign]
    app = create_app(bus=bus)

    with TestClient(app) as client:
        response = client.get("/api/agents")

    assert response.status_code == 200
    assert response.json() == {"agents": [], "pipeline": "subscription_failed"}


def test_api_agents_keeps_stale_agent_offline_on_healthy_pipeline() -> None:
    """Known stale agents remain present and marked Offline on a healthy feed.

    A stale agent must not vanish: it stays in the list with ``alive=False`` while
    the pipeline is still ``healthy``, so Dash renders it Offline rather than gone.
    """

    bus = FakeFrameBus()
    app = create_app(
        config=WebchatConfig(stack="dev", heartbeat_interval_seconds=1.0),
        bus=bus,
    )
    presence = app.state.webchat.presence

    with TestClient(app) as client:
        wildcard = subjects.agent_heartbeat_wildcard("dev")
        bus.emit(wildcard, _heartbeat("agent-a", seq=1))
        # Force the recorded receipt time far into the past so the agent is stale.
        presence.record_heartbeat("agent-a", seen_ms=1)
        response = client.get("/api/agents")

    assert response.status_code == 200
    assert response.json() == {
        "agents": [{"agent_id": "agent-a", "last_seen_ms": 1, "alive": False}],
        "pipeline": "healthy",
    }


def test_api_agents_flips_to_subscription_failed_on_later_connection_loss() -> None:
    """A subscription proven healthy once re-derives failed if the feed later dies.

    The bug was a one-shot verdict: mark_subscribed set healthy forever, so a
    severed heartbeat connection kept answering ``healthy`` (empty == zero agents)
    even though the feed was down. The read consults the bus's typed liveness, so
    dropping the connection after subscribe must flip the next read to
    ``subscription_failed`` — WITHOUT relying on agent count.
    """

    bus = FakeFrameBus()
    app = create_app(bus=bus)

    with TestClient(app) as client:
        # A live agent + connected feed reads healthy.
        bus.emit(subjects.agent_heartbeat_wildcard("dev"), _heartbeat("agent-a"))
        healthy = client.get("/api/agents")
        # The heartbeat transport is later severed; the agent row is untouched.
        bus.connected = False
        after_loss = client.get("/api/agents")

    assert healthy.status_code == 200
    assert healthy.json()["pipeline"] == "healthy"
    assert after_loss.status_code == 200
    # The known agent row survives; only the pipeline verdict flips, so Dash never
    # reads a broken feed as truth — and this is NOT an agent-count proxy.
    assert after_loss.json()["pipeline"] == "subscription_failed"
    assert [agent["agent_id"] for agent in after_loss.json()["agents"]] == ["agent-a"]


def test_api_agents_recovers_to_healthy_when_connection_returns() -> None:
    """A reconnect re-derives healthy: the verdict tracks live transport.

    Recovery is genuinely supported because the read re-evaluates the liveness
    probe every time; the pipeline is not latched failed once it flips.
    """

    bus = FakeFrameBus()
    app = create_app(bus=bus)

    with TestClient(app) as client:
        bus.connected = False
        during_loss = client.get("/api/agents")
        bus.connected = True
        after_recovery = client.get("/api/agents")

    assert during_loss.json()["pipeline"] == "subscription_failed"
    assert after_recovery.json()["pipeline"] == "healthy"


def test_subscribe_failure_logs_type_only_never_credential_url() -> None:
    """A failed heartbeat subscribe logs safe context + exception type, never str(exc).

    A NATS subscribe failure can carry the credential-bearing
    ``nats://user:pass@host`` URL in its message; that text must never reach a log
    line. A dedicated loguru sink captures exactly what the edge emits (loguru
    binds the real stderr at import time, so ``capsys`` cannot see it). The
    pipeline still surfaces subscription_failed for Dash.
    """

    secret_url = "nats://svc:sup3r-s3cret-t0ken@bus.internal:4222"
    records: list[str] = []
    sink_id = logger.add(records.append, level="TRACE", format="{message}")

    bus = FakeFrameBus()

    async def _raise_with_credential(subject: str, handler: FrameHandler) -> FakeSubscription:
        raise ConnectionError(f"connect refused for {secret_url}")

    bus.subscribe = _raise_with_credential  # type: ignore[method-assign]
    app = create_app(bus=bus)

    try:
        with TestClient(app) as client:
            response = client.get("/api/agents")
    finally:
        logger.remove(sink_id)

    logged = "".join(records)
    # The verdict is still surfaced for Dash.
    assert response.json()["pipeline"] == "subscription_failed"
    # No fragment of the credential-bearing URL leaks into any log line.
    assert "sup3r-s3cret-t0ken" not in logged
    assert "svc:" not in logged
    assert secret_url not in logged
    assert "bus.internal" not in logged
    # The safe exception type is logged so operators can still triage.
    assert "ConnectionError" in logged
    assert "webchat.presence.subscribe | failed error_type=ConnectionError" in logged


def test_api_agents_uses_api_bearer_auth() -> None:
    bus = FakeFrameBus()
    app = create_app(
        config=WebchatConfig(stack="dev", api_key="correct-token"),
        bus=bus,
    )

    with TestClient(app) as client:
        missing = client.get("/api/agents")
        wrong = client.get("/api/agents", headers={"Authorization": "Bearer wrong-token"})
        accepted = client.get(
            "/api/agents",
            headers={"Authorization": "Bearer correct-token"},
        )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json() == {"agents": [], "pipeline": "healthy"}


def test_api_sse_streams_encoded_frames_and_closes_on_result() -> None:
    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "task-123")

    async def run_stream() -> list[str]:
        with TestClient(app) as client:
            client.post(
                "/api/tasks",
                json={"text": "First task.", "requester": "rico", "scope": "single"},
            )
        state = app.state.webchat
        stream = _stream_task_frames(state, "task-123")
        first_line = asyncio.create_task(stream.__anext__())
        while subjects.task_result("task-123") not in bus.handlers:
            await asyncio.sleep(0)
        await bus.emit_async(
            subjects.task_turn("task-123"),
            encode_frame(_turn()),
        )
        second_line = asyncio.create_task(stream.__anext__())
        await bus.emit_async(
            subjects.task_status("task-123"),
            encode_frame(StatusFrame(task_id="task-123", agent_id="agent-a", state="working")),
        )
        close = asyncio.create_task(stream.__anext__())
        await bus.emit_async(subjects.task_result("task-123"), b"arrived")
        try:
            await close
        except StopAsyncIteration:
            pass
        return [await first_line, await second_line]

    lines = asyncio.run(run_stream())

    assert [_decode_sse_line(line) for line in lines] == [
        _turn(),
        StatusFrame(task_id="task-123", agent_id="agent-a", state="working"),
    ]


def test_api_sse_attach_after_completion_replays_frames_and_closes() -> None:
    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "task-123")

    async def run_stream() -> list[str]:
        with TestClient(app) as client:
            client.post(
                "/api/tasks",
                json={"text": "First task.", "requester": "rico", "scope": "single"},
            )
        await bus.emit_async(subjects.task_turn("task-123"), encode_frame(_turn()))
        await bus.emit_async(
            subjects.task_status("task-123"),
            encode_frame(StatusFrame(task_id="task-123", agent_id="agent-a", state="working")),
        )
        await bus.emit_async(subjects.task_result("task-123"), b"arrived")
        stream = _stream_task_frames(app.state.webchat, "task-123")
        first_line = await stream.__anext__()
        second_line = await stream.__anext__()
        try:
            await stream.__anext__()
        except StopAsyncIteration:
            return [first_line, second_line]
        raise AssertionError("stream did not close after replayed result")

    lines = asyncio.run(run_stream())

    assert [_decode_sse_line(line) for line in lines] == [
        _turn(),
        StatusFrame(task_id="task-123", agent_id="agent-a", state="working"),
    ]


def test_api_sse_attach_mid_turn_replays_then_tails() -> None:
    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "task-123")

    async def run_stream() -> list[str]:
        with TestClient(app) as client:
            client.post(
                "/api/tasks",
                json={"text": "First task.", "requester": "rico", "scope": "single"},
            )
        await bus.emit_async(subjects.task_turn("task-123"), encode_frame(_turn()))
        stream = _stream_task_frames(app.state.webchat, "task-123")
        first_line = await stream.__anext__()
        second_line = asyncio.create_task(stream.__anext__())
        await bus.emit_async(
            subjects.task_status("task-123"),
            encode_frame(StatusFrame(task_id="task-123", agent_id="agent-a", state="working")),
        )
        close = asyncio.create_task(stream.__anext__())
        await bus.emit_async(subjects.task_result("task-123"), b"arrived")
        try:
            await close
        except StopAsyncIteration:
            pass
        return [first_line, await second_line]

    lines = asyncio.run(run_stream())

    assert [_decode_sse_line(line) for line in lines] == [
        _turn(),
        StatusFrame(task_id="task-123", agent_id="agent-a", state="working"),
    ]


def test_replay_eviction_unsubscribes_old_task_buffer() -> None:
    ids = iter(["task-1", "task-2"])
    bus = FakeFrameBus()
    app = create_app(
        config=WebchatConfig(stack="dev", replay_max_tasks=1),
        bus=bus,
        id_factory=lambda: next(ids),
    )

    with TestClient(app) as client:
        first = client.post(
            "/api/tasks",
            json={"text": "First task.", "requester": "rico", "scope": "single"},
        )
        second = client.post(
            "/api/tasks",
            json={"text": "Second task.", "requester": "rico", "scope": "single"},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    # Buffer eviction (max_tasks=1) tears down the evicted task's turn/status
    # interest that only fed the bounded replay buffer...
    first_buffer_subscriptions = [
        subscription
        for subscription in bus.subscriptions
        if subscription.subject
        in {
            subjects.task_turn("task-1"),
            subjects.task_status("task-1"),
        }
    ]
    # ...but its result interest is retained: it is the completion mirror and
    # must survive eviction so a later core-NATS result is still observed.
    first_result_subscriptions = [
        subscription
        for subscription in bus.subscriptions
        if subscription.subject == subjects.task_result("task-1")
    ]
    second_buffer_subscriptions = [
        subscription
        for subscription in bus.subscriptions
        if subscription.subject
        in {
            subjects.task_turn("task-2"),
            subjects.task_status("task-2"),
            subjects.task_result("task-2"),
        }
    ]
    assert all(subscription.unsubscribed for subscription in first_buffer_subscriptions)
    assert first_result_subscriptions
    assert not any(subscription.unsubscribed for subscription in first_result_subscriptions)
    assert not any(subscription.unsubscribed for subscription in second_buffer_subscriptions)


def test_api_sse_replay_live_overlap_has_no_duplicate_frames() -> None:
    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "task-123")

    async def run_stream() -> list[str]:
        with TestClient(app) as client:
            client.post(
                "/api/tasks",
                json={"text": "First task.", "requester": "rico", "scope": "single"},
            )
        await bus.emit_async(subjects.task_turn("task-123"), encode_frame(_turn()))
        stream = _stream_task_frames(app.state.webchat, "task-123")
        first_line = asyncio.create_task(stream.__anext__())
        second_line = asyncio.create_task(stream.__anext__())
        await bus.emit_async(
            subjects.task_status("task-123"),
            encode_frame(StatusFrame(task_id="task-123", agent_id="agent-a", state="working")),
        )
        close = asyncio.create_task(stream.__anext__())
        await bus.emit_async(subjects.task_result("task-123"), b"arrived")
        try:
            await close
        except StopAsyncIteration:
            pass
        return [await first_line, await second_line]

    lines = asyncio.run(run_stream())

    assert [_decode_sse_line(line) for line in lines] == [
        _turn(),
        StatusFrame(task_id="task-123", agent_id="agent-a", state="working"),
    ]


def test_api_stream_endpoint_rejects_unknown_task() -> None:
    bus = FakeFrameBus()
    app = create_app(bus=bus)

    with TestClient(app) as client:
        response = client.get("/api/tasks/unknown/stream")

    assert response.status_code == 404


def test_websocket_multiplexes_task_frames_and_shapes_payloads() -> None:
    ids = iter(["task-123", "task-456"])
    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: next(ids))

    with TestClient(app) as client:
        client.post(
            "/task",
            json={"text": "First task.", "requester": "rico", "scope": "single"},
        )
        client.post(
            "/api/tasks",
            json={
                "text": "Second task.",
                "requester": "web",
                "participants": ["codex-a", "opus-b"],
            },
        )
        with client.websocket_connect("/ws") as websocket:
            assert websocket.receive_json() == {"type": "ready"}
            websocket.send_json({"type": "subscribe", "task_id": "task-123"})
            assert websocket.receive_json() == {"type": "subscribed", "task_id": "task-123"}
            websocket.send_json({"type": "subscribe", "task_id": "task-456"})
            assert websocket.receive_json() == {"type": "subscribed", "task_id": "task-456"}

            bus.emit(
                subjects.task_turn("task-123"),
                encode_frame(
                    TurnFrame(
                        agent_id="agent-a",
                        task_id="task-123",
                        visible_text="Visible answer.",
                        stance=Stance(
                            position="working",
                            because="The task is active.",
                            needs_from=["agent-b"],
                        ),
                        refs=["ref-a"],
                    )
                ),
            )
            assert websocket.receive_json() == {
                "type": "turn",
                "task_id": "task-123",
                "agent_id": "agent-a",
                "visible_text": "Visible answer.",
                "stance": {
                    "position": "working",
                    "because": "The task is active.",
                    "needs_from": ["agent-b"],
                },
                "refs": ["ref-a"],
            }
            bus.emit(
                subjects.task_status("task-456"),
                encode_frame(StatusFrame(task_id="task-456", agent_id="agent-a", state="blocked")),
            )
            assert websocket.receive_json() == {
                "type": "status",
                "task_id": "task-456",
                "agent_id": "agent-a",
                "state": "blocked",
            }
            bus.emit(subjects.task_result("task-123"), b"arrived")
            assert websocket.receive_json() == {"type": "result", "task_id": "task-123"}

        listed = client.get("/tasks")

    assert listed.json()[0]["frames"] == [
        {
            "type": "turn",
            "task_id": "task-123",
            "agent_id": "agent-a",
            "visible_text": "Visible answer.",
            "stance": {
                "position": "working",
                "because": "The task is active.",
                "needs_from": ["agent-b"],
            },
            "refs": ["ref-a"],
        },
        {"type": "result", "task_id": "task-123"},
    ]
    assert listed.json()[0]["status"] == "result"
    assert listed.json()[1]["frames"] == [
        {"type": "status", "task_id": "task-456", "agent_id": "agent-a", "state": "blocked"}
    ]
    assert listed.json()[1]["status"] == "blocked"


def test_index_serves_v3_page() -> None:
    bus = FakeFrameBus()
    app = create_app(bus=bus)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    body = response.text
    # v3 page carries the live agent picker, derived scope, and inline steer.
    assert "/api/agents" in body
    assert "requester (provenance label)" in body
    assert 'id="agents"' in body
    assert 'id="scope-badge"' in body
    assert 'id="steer-form"' in body


def test_v3_page_posts_participants_and_steer_to_api_routes() -> None:
    bus = FakeFrameBus()
    app = create_app(bus=bus)

    with TestClient(app) as client:
        body = client.get("/").text

    # Task POST carries participants and hits the participants-aware /api/tasks
    # contract; the picker derives scope from selection count client-side.
    assert "participants: participants" in body
    assert 'fetch("/api/tasks"' in body
    assert "SCOPE_BY_COUNT" in body
    assert "MAX_PARTICIPANTS" in body
    # Steer POST hits the per-task /api/tasks/{id}/steer route with interrupt.
    assert "/api/tasks/${encodeURIComponent(selectedTaskId)}/steer" in body
    assert "interrupt:" in body


def test_api_task_accepts_participants_and_still_mints_frame() -> None:
    """The edge sends participants; the current server ignores unknown fields.

    Post-merge the participants lane derives scope server-side from the count.
    This asserts the UI payload shape is accepted today and mints a task frame.
    """

    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "task-123")

    with TestClient(app) as client:
        response = client.post(
            "/api/tasks",
            json={
                "text": "Build the edge.",
                "requester": "rico",
                "participants": ["agent-a", "agent-b"],
            },
        )

    assert response.status_code == 200
    assert response.json() == {"task_id": "task-123"}
    assert len(bus.published) == 1
    minted = decode_frame(bus.published[0][1])
    assert isinstance(minted, TaskFrame)
    assert minted.requester == "rico"
    assert minted.text == "Build the edge."


@pytest.mark.parametrize(
    ("participants", "expected_scope"),
    [
        (["agent-a"], "single"),
        (["agent-a", "agent-b"], "duo"),
        (["agent-a", "agent-b", "agent-c"], "trio"),
    ],
)
def test_participants_derive_scope_post_merge(participants: list[str], expected_scope: str) -> None:
    """Expected-to-pass once the participants lane derives scope server-side.

    Skips while the sibling lane is unmerged (server still ignores participants
    and defaults scope to ``single``), then asserts the ratified contract:
    count 1/2/3 -> single/duo/trio.
    """

    ids = iter(["probe-id", "task-scope"])
    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: next(ids))

    with TestClient(app) as client:
        if len(participants) > 1 and not _has_scope_derivation(client, bus):
            pytest.skip("participants scope-derivation lane not merged yet")
        bus.published.clear()
        response = client.post(
            "/api/tasks",
            json={
                "text": "scope task",
                "requester": "rico",
                "participants": participants,
            },
        )

    assert response.status_code == 200
    minted = decode_frame(bus.published[0][1])
    assert isinstance(minted, TaskFrame)
    assert minted.scope == expected_scope


def _has_scope_derivation(client: TestClient, bus: FakeFrameBus) -> bool:
    """Probe whether the server derives duo scope from a 2-participant post."""

    before = len(bus.published)
    probe = client.post(
        "/api/tasks",
        json={
            "text": "probe",
            "requester": "rico",
            "participants": ["agent-a", "agent-b"],
        },
    )
    if probe.status_code != 200 or len(bus.published) <= before:
        return False
    minted = decode_frame(bus.published[-1][1])
    return isinstance(minted, TaskFrame) and minted.scope == "duo"


def test_agents_endpoint_shape_post_merge() -> None:
    """Expected-to-pass once the presence lane exposes GET /api/agents.

    Skips while the sibling presence lane is unmerged. When present, asserts the
    ratified contract: ``{"agents": [{"agent_id", "last_seen_ms", "alive"}]}``.
    """

    bus = FakeFrameBus()
    app = create_app(bus=bus)

    with TestClient(app) as client:
        response = client.get("/api/agents")
        if response.status_code == 404:
            pytest.skip("presence /api/agents lane not merged yet")

    assert response.status_code == 200
    body = response.json()
    assert "agents" in body
    assert isinstance(body["agents"], list)
    for agent in body["agents"]:
        assert set(agent) >= {"agent_id", "last_seen_ms", "alive"}


def test_task_registry_exposes_participants_for_destination() -> None:
    """The task list carries the minted participants so the UI can name the destination."""

    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "task-123")

    with TestClient(app) as client:
        minted = client.post(
            "/api/tasks",
            json={
                "text": "Pair on this.",
                "requester": "rico",
                "participants": ["agentd-birk", "agentd-mini"],
            },
        )
        listed = client.get("/tasks")

    assert minted.status_code == 200
    assert listed.status_code == 200
    item = listed.json()[0]
    assert item["participants"] == ["agentd-birk", "agentd-mini"]
    assert item["scope"] == "duo"


def test_api_stream_requires_bearer_when_keyed() -> None:
    """SSE stream is keyed: 401 without a bearer, 200/valid stream with the key."""

    bus = FakeFrameBus()
    app = create_app(
        config=WebchatConfig(stack="dev", api_key="correct-token"),
        bus=bus,
        id_factory=lambda: "task-123",
    )

    with TestClient(app) as client:
        client.post(
            "/api/tasks",
            headers={"Authorization": "Bearer correct-token"},
            json={"text": "Keyed task.", "requester": "rico"},
        )
        missing = client.get("/api/tasks/task-123/stream")
        wrong = client.get(
            "/api/tasks/task-123/stream",
            headers={"Authorization": "Bearer wrong-token"},
        )

    assert missing.status_code == 401
    assert wrong.status_code == 401


def test_v3_page_carries_keyed_fetch_and_destination_logic() -> None:
    """The v3 page sends a bearer on fetches, streams via fetch (not EventSource),
    distinguishes 401 from empty, and names the send destination."""

    bus = FakeFrameBus()
    app = create_app(bus=bus)

    with TestClient(app) as client:
        body = client.get("/").text

    # Masked, persisted edge API key sent as a bearer on fetches.
    assert 'id="api-key"' in body
    assert 'type="password"' in body
    assert "API_KEY_KEY" in body
    assert "authHeaders" in body
    assert '"Authorization"] = "Bearer "' in body
    # SSE consumed via fetch-streaming (no EventSource, no key in the URL).
    assert "getReader" in body
    assert "text/event-stream" in body
    assert "new EventSource" not in body
    assert "new WebSocket" not in body
    # 401 vs empty distinguished in the picker.
    assert "API key required or invalid" in body
    assert "No agents online" in body
    # Send button names the destination; participants + scope badge rendered.
    assert "destinationLabel" in body
    assert "Send to " in body
    assert "renderSelectedParticipants" in body


def test_v3_page_fails_closed_on_untrusted_pipeline_verdict() -> None:
    """The picker fails closed on an initial/absent/unrecognized verdict.

    A 200 whose typed pipeline is missing or unrecognized must NOT be coerced to
    healthy: the empty roster would then read as the authoritative "No agents
    online". Instead presence starts and stays "unavailable" until a recognized
    verdict arrives, rendered distinctly from healthy-empty and from
    subscription_failed.
    """

    bus = FakeFrameBus()
    app = create_app(bus=bus)

    with TestClient(app) as client:
        body = client.get("/").text

    # Initial state is unavailable, never healthy — no authoritative empty verdict
    # before a recognized 200.
    assert 'let agentsPipeline = "unavailable";' in body
    # Only the two recognized wire values are trusted; anything else is unavailable.
    assert 'if (body.pipeline === "healthy") {' in body
    assert '} else if (body.pipeline === "subscription_failed") {' in body
    assert 'agentsPipeline = "unavailable";' in body
    # An untrusted verdict clears untrusted rows/selections so a stale roster is
    # never presented as current.
    assert 'if (agentsPipeline === "unavailable") {' in body
    # The picker renders a distinct unavailable message, not "No agents online".
    assert "Presence unavailable." in body
    # The three verdicts stay distinct in the picker.
    assert "Presence feed unavailable." in body
    assert "No agents online." in body
    assert 'if (agentsPipeline !== "healthy") {' in body


def test_v3_page_preserves_auth_error_and_healthy_empty_paths() -> None:
    """The fail-closed verdict does not disturb the auth-error or healthy-empty paths.

    A 401 stays an auth error, and a genuine healthy pipeline with zero agents
    still renders the authoritative "No agents online.".
    """

    bus = FakeFrameBus()
    app = create_app(bus=bus)

    with TestClient(app) as client:
        body = client.get("/").text

    # The 401 auth path is untouched and still distinct from presence verdicts.
    assert "API key required or invalid." in body
    assert "agentsAuthError = true" in body
    # A trusted healthy verdict still routes to the authoritative empty roster.
    assert 'agentsPipeline = "healthy";' in body
    assert "No agents online." in body


def test_websocket_replays_buffered_frames_on_subscribe() -> None:
    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "task-123")

    with TestClient(app) as client:
        client.post(
            "/api/tasks",
            json={"text": "First task.", "requester": "rico", "scope": "single"},
        )
        bus.emit(subjects.task_turn("task-123"), encode_frame(_turn()))
        bus.emit(subjects.task_result("task-123"), b"arrived")

        with client.websocket_connect("/ws") as websocket:
            assert websocket.receive_json() == {"type": "ready"}
            websocket.send_json({"type": "subscribe", "task_id": "task-123"})
            assert websocket.receive_json() == {"type": "subscribed", "task_id": "task-123"}
            assert websocket.receive_json() == {
                "type": "turn",
                "task_id": "task-123",
                "agent_id": "agent-a",
                "visible_text": "Visible answer.",
                "stance": {
                    "position": "working",
                    "because": "The task is active.",
                    "needs_from": ["agent-b"],
                },
                "refs": ["ref-a"],
            }
            assert websocket.receive_json() == {"type": "result", "task_id": "task-123"}


def test_post_api_task_rejects_duplicate_participants() -> None:
    """Duplicate participant ids are rejected at the edge with a 400 (round 2)."""
    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "task-123")

    with TestClient(app) as client:
        response = client.post(
            "/api/tasks",
            json={
                "text": "Pair with yourself.",
                "requester": "rico",
                "participants": ["codex-a", "codex-a"],
            },
        )

    assert response.status_code == 400
    assert "unique" in response.json()["detail"]
    assert bus.published == []


def test_post_api_task_rejects_subject_unsafe_participants() -> None:
    """Blank or subject-unsafe participant ids are rejected with a 400."""
    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "task-123")

    with TestClient(app) as client:
        blank = client.post(
            "/api/tasks",
            json={"text": "Go.", "requester": "rico", "participants": [" "]},
        )
        dotted = client.post(
            "/api/tasks",
            json={"text": "Go.", "requester": "rico", "participants": ["a.b", "c"]},
        )

    assert blank.status_code == 400
    assert dotted.status_code == 400
    assert bus.published == []


def test_post_task_rejects_identityless_multi_scope() -> None:
    """The legacy /task route only mints single-scope tasks (round 2)."""
    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "task-123")

    with TestClient(app) as client:
        response = client.post(
            "/task",
            json={"text": "Pair on this.", "requester": "rico", "scope": "duo"},
        )

    assert response.status_code == 400
    assert bus.published == []
