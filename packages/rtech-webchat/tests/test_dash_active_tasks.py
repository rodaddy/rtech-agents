"""Test the bearer-gated active Dash task recovery endpoint.

Every case is hermetic: a ``FakeFrameBus`` captures publishes and routes
emitted frames to the connector's own subscribers, so completion is driven by
the orchestrator-derived result-subject arrival exactly as it would be on the
live bus — never by connector-invented state.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from rtech_frames.codec import encode_frame
from rtech_frames.frames import StatusFrame
from rtech_webchat import subjects
from rtech_webchat.app import AppState, create_app
from rtech_webchat.config import WebchatConfig
from rtech_webchat.replay import ReplayHub

FrameHandler = Callable[[bytes], Awaitable[None]]

DASH_ACTIVE_URL = "/api/dash/active-tasks"


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


def _mint_dash_task(client: TestClient, *, text: str, requester: str) -> str:
    """Mint one dash-dash task and return its id."""

    response = client.post(
        "/api/tasks",
        json={
            "text": text,
            "requester": requester,
            "participants": ["agentd-sprocket"],
            "origin_edge": "dash-dash",
        },
    )
    assert response.status_code == 200
    return response.json()["task_id"]


def _mint_webchat_task(client: TestClient, *, text: str, requester: str) -> str:
    """Mint one web-chat-origin task and return its id."""

    response = client.post(
        "/task",
        json={"text": text, "requester": requester, "scope": "single"},
    )
    assert response.status_code == 200
    return response.json()["task_id"]


def test_lists_only_active_dash_dash_tasks() -> None:
    """Only dash-dash origin tasks appear; web-chat origin is excluded."""

    ids = iter(["dash-1", "web-1"])
    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: next(ids))

    with TestClient(app) as client:
        _mint_dash_task(client, text="Dash secret.", requester="dash")
        _mint_webchat_task(client, text="Web task.", requester="rico")
        listed = client.get(DASH_ACTIVE_URL)

    assert listed.status_code == 200
    assert listed.json() == [
        {
            "task_id": "dash-1",
            "origin_edge": "dash-dash",
            "scope": "single",
            "status": "minted",
        }
    ]


def test_excludes_completed_dash_tasks() -> None:
    """A dash task whose orchestrator-derived result arrived is excluded."""

    ids = iter(["dash-done", "dash-live"])
    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: next(ids))

    with TestClient(app) as client:
        _mint_dash_task(client, text="Finished.", requester="dash")
        _mint_dash_task(client, text="Still running.", requester="dash")
        # The result subject is the orchestrator's derived-done publish; its
        # arrival is the ONLY completion signal the connector observes.
        bus.emit(subjects.task_result("dash-done"), b"arrived")
        listed = client.get(DASH_ACTIVE_URL)

    assert listed.status_code == 200
    assert [item["task_id"] for item in listed.json()] == ["dash-live"]


def test_reports_last_observed_typed_status() -> None:
    """The active item carries the last typed StatusFrame state, not prose."""

    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "dash-1")

    with TestClient(app) as client:
        _mint_dash_task(client, text="Blocked task.", requester="dash")
        bus.emit(
            subjects.task_status("dash-1"),
            encode_frame(
                StatusFrame(task_id="dash-1", agent_id="agentd-sprocket", state="blocked")
            ),
        )
        listed = client.get(DASH_ACTIVE_URL)

    assert listed.status_code == 200
    assert listed.json() == [
        {
            "task_id": "dash-1",
            "origin_edge": "dash-dash",
            "scope": "single",
            "status": "blocked",
        }
    ]


def test_item_shape_carries_no_private_payload() -> None:
    """The recovery item exposes id/origin/scope/status only — no text, no history."""

    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "dash-1")

    with TestClient(app) as client:
        _mint_dash_task(client, text="Private task text.", requester="secret-requester")
        listed = client.get(DASH_ACTIVE_URL)

    assert listed.status_code == 200
    item = listed.json()[0]
    assert set(item) == {"task_id", "origin_edge", "scope", "status"}
    # No private payload leaks through the recovery surface.
    serialized = listed.text
    assert "Private task text." not in serialized
    assert "secret-requester" not in serialized
    assert "agentd-sprocket" not in serialized
    assert "frames" not in item
    assert "participants" not in item
    assert "text" not in item
    assert "requester" not in item


def test_endpoint_requires_bearer_when_keyed() -> None:
    """The recovery endpoint is bearer-gated: 401 without/wrong, 200 with the key."""

    bus = FakeFrameBus()
    app = create_app(
        config=WebchatConfig(stack="dev", api_key="correct-token"),
        bus=bus,
        id_factory=lambda: "dash-1",
    )

    with TestClient(app) as client:
        client.post(
            "/api/tasks",
            headers={"Authorization": "Bearer correct-token"},
            json={
                "text": "Keyed dash task.",
                "requester": "dash",
                "participants": ["agentd-sprocket"],
                "origin_edge": "dash-dash",
            },
        )
        missing = client.get(DASH_ACTIVE_URL)
        wrong = client.get(DASH_ACTIVE_URL, headers={"Authorization": "Bearer wrong-token"})
        accepted = client.get(DASH_ACTIVE_URL, headers={"Authorization": "Bearer correct-token"})

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert accepted.status_code == 200
    assert [item["task_id"] for item in accepted.json()] == ["dash-1"]


def test_empty_when_no_dash_tasks() -> None:
    """With only non-dash tasks the recovery list is empty, not an error."""

    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "web-1")

    with TestClient(app) as client:
        _mint_webchat_task(client, text="Web only.", requester="rico")
        listed = client.get(DASH_ACTIVE_URL)

    assert listed.status_code == 200
    assert listed.json() == []


def _active_dash_ids(client: TestClient) -> list[str]:
    """Return the recovery endpoint's active dash-dash task ids."""

    listed = client.get(DASH_ACTIVE_URL)
    assert listed.status_code == 200
    return [item["task_id"] for item in listed.json()]


def _replay_hub(app: FastAPI) -> ReplayHub:
    """Return the app's live replay hub for deterministic state assertions."""

    state: AppState = app.state.webchat
    return state.replay


def _age_buffer_updated_at(app: FastAPI, task_id: str, seconds: float) -> None:
    """Age a replay buffer's ``updated_at`` back so the next sweep expires it.

    Reaches into the live hub state deliberately: it simulates real monotonic
    clock advance without a blocking sleep, so TTL eviction is deterministic.
    """

    _replay_hub(app).buffers[task_id].updated_at -= seconds


def test_ttl_eviction_of_buffer_still_clears_completed_dash_task() -> None:
    """After TTL buffer eviction, a later core-NATS result still clears active.

    The replay buffer is the evictable SSE-bound surface; the result interest
    that mirrors orchestrator-derived completion must outlive it. If it did not,
    the completed dash-dash task would stay authoritatively active forever.
    """

    ids = iter(["dash-ttl", "dash-sweep"])
    bus = FakeFrameBus()
    app = create_app(
        config=WebchatConfig(stack="dev", replay_ttl_seconds=1),
        bus=bus,
        id_factory=lambda: next(ids),
    )

    with TestClient(app) as client:
        _mint_dash_task(client, text="Long runner.", requester="dash")
        assert _active_dash_ids(client) == ["dash-ttl"]

        # Age the buffer past its TTL, then trigger a sweep by minting another
        # task; start_task sweeps, so the aged buffer is evicted for TTL.
        _age_buffer_updated_at(app, "dash-ttl", seconds=5.0)
        _mint_dash_task(client, text="Sweep trigger.", requester="dash")
        assert "dash-ttl" not in app.state.webchat.replay.buffers

        # The orchestrator's result arrives AFTER buffer eviction. The retained
        # result subscription observes it, so the task is truthfully cleared.
        bus.emit(subjects.task_result("dash-ttl"), b"arrived")
        assert "dash-ttl" not in _active_dash_ids(client)


def test_max_task_eviction_of_buffer_still_clears_completed_dash_task() -> None:
    """After max-task LRU buffer eviction, a later result still clears active."""

    ids = iter(["dash-evicted", "dash-newer"])
    bus = FakeFrameBus()
    app = create_app(
        config=WebchatConfig(stack="dev", replay_max_tasks=1),
        bus=bus,
        id_factory=lambda: next(ids),
    )

    with TestClient(app) as client:
        _mint_dash_task(client, text="Older runner.", requester="dash")
        # Minting a second task exceeds max_tasks=1 and LRU-evicts the first
        # buffer during the sweep.
        _mint_dash_task(client, text="Newer runner.", requester="dash")
        assert "dash-evicted" not in app.state.webchat.replay.buffers
        assert set(_active_dash_ids(client)) == {"dash-evicted", "dash-newer"}

        # Result for the evicted-buffer task arrives on core NATS; the surviving
        # result subscription clears it despite the buffer being gone.
        bus.emit(subjects.task_result("dash-evicted"), b"arrived")
        assert _active_dash_ids(client) == ["dash-newer"]


def test_normal_completion_clears_active_without_eviction() -> None:
    """The baseline path: a result with an intact buffer clears the task."""

    bus = FakeFrameBus()
    app = create_app(bus=bus, id_factory=lambda: "dash-1")

    with TestClient(app) as client:
        _mint_dash_task(client, text="Runs to completion.", requester="dash")
        assert _active_dash_ids(client) == ["dash-1"]
        bus.emit(subjects.task_result("dash-1"), b"arrived")
        assert _active_dash_ids(client) == []


def test_reload_recovery_lists_still_active_after_eviction() -> None:
    """A reloaded Dash dashboard recovers a still-active task past eviction.

    Buffer eviction (memory bound) must not misreport a still-running task as
    completed: with no result yet, the task remains active on the recovery
    surface even though its replay buffer was evicted.
    """

    ids = iter(["dash-live", "dash-sweep"])
    bus = FakeFrameBus()
    app = create_app(
        config=WebchatConfig(stack="dev", replay_ttl_seconds=1),
        bus=bus,
        id_factory=lambda: next(ids),
    )

    with TestClient(app) as client:
        _mint_dash_task(client, text="Still running.", requester="dash")
        _age_buffer_updated_at(app, "dash-live", seconds=5.0)
        _mint_dash_task(client, text="Sweep trigger.", requester="dash")
        assert "dash-live" not in app.state.webchat.replay.buffers

        # No result arrived: the reloaded dashboard must still see it active.
        assert "dash-live" in _active_dash_ids(client)
