"""Build the FastAPI web-chat edge."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from loguru import logger
from pydantic import BaseModel, Field
from rtech_frames.codec import encode_frame
from rtech_frames.frames import PresenceTransitionFrame, SteerFrame, TaskFrame

from . import subjects
from .api import register_api_routes
from .bus import FrameBus, NatsFrameBus, Subscription
from .config import WebchatConfig
from .presence import PresenceConfig, PresenceRegistry
from .presence_transitions import PresenceTransitionDeriver
from .registry import RenderedPayload, TaskListItem, TaskRegistry
from .replay import ReplayAttachment, ReplayHub

TaskScope = Literal["single", "duo", "trio"]
IdFactory = Callable[[], str]


class TaskRequest(BaseModel):
    """Validate task ingress JSON."""

    text: str = Field(min_length=1)
    requester: str = Field(min_length=1)
    scope: TaskScope


class TaskResponse(BaseModel):
    """Return the minted task id."""

    task_id: str


class SteerRequest(BaseModel):
    """Validate steer ingress JSON."""

    task_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    target_agent: str = Field(min_length=1)
    interrupt: bool = False


class SteerResponse(BaseModel):
    """Report the steer frame target subject."""

    task_id: str
    subject: str
    interrupt: bool


@dataclass(frozen=True)
class AppState:
    """Hold app dependencies."""

    bus: FrameBus
    id_factory: IdFactory
    registry: TaskRegistry
    replay: ReplayHub
    presence: PresenceRegistry


def create_app(
    *,
    config: WebchatConfig | None = None,
    bus: FrameBus | None = None,
    id_factory: IdFactory | None = None,
) -> FastAPI:
    """Create the FastAPI application."""

    resolved_config = config or WebchatConfig.from_env()
    resolved_bus = bus or NatsFrameBus(resolved_config.nats_url, env=resolved_config.env)
    registry = TaskRegistry()
    replay = ReplayHub(
        bus=resolved_bus,
        registry=registry,
        frame_limit=resolved_config.replay_frame_limit,
        ttl_seconds=resolved_config.replay_ttl_seconds,
        max_tasks=resolved_config.replay_max_tasks,
    )
    presence = PresenceRegistry(
        config=PresenceConfig(heartbeat_interval_seconds=resolved_config.heartbeat_interval_seconds)
    )

    async def _publish_transition(frame: PresenceTransitionFrame) -> None:
        await resolved_bus.publish(subjects.PRESENCE_TRANSITION, encode_frame(frame))

    presence_deriver = PresenceTransitionDeriver(
        registry=presence,
        publish=_publish_transition,
        clock=lambda: datetime.now(UTC),
    )
    state = AppState(
        bus=resolved_bus,
        id_factory=id_factory or _new_task_id,
        registry=registry,
        replay=replay,
        presence=presence,
    )

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        heartbeat_subscription: Subscription | None = None
        transition_task: asyncio.Task[None] | None = None
        if isinstance(resolved_bus, NatsFrameBus):
            await resolved_bus.connect()
        try:
            heartbeat_subscription = await resolved_bus.subscribe(
                subjects.agent_heartbeat_wildcard(resolved_config.env),
                state.presence.handle_heartbeat,
            )
        except Exception as exc:
            # Only the subscribe call is guarded here: connect() above is not, so a
            # connect failure fails app startup outright. When connect succeeded but
            # the heartbeat subscribe fails, the edge stays reachable so Dash can
            # still read a typed verdict — surfaced as subscription_failed (issue
            # #145), never silently rendered as "zero agents". Log only the
            # exception *type*, never str(exc): a NATS subscribe failure can embed
            # the credential-bearing nats://user:pass@host URL, which must never
            # reach a log line.
            logger.error(
                "webchat.presence.subscribe | failed error_type={}",
                type(exc).__name__,
            )
            state.presence.mark_subscription_failed()
        else:
            # Consult the bus's typed liveness at every read so a later connection
            # drop or subscription retirement re-derives subscription_failed
            # instead of the pipeline staying healthy forever (issue #145).
            state.presence.mark_subscribed(liveness=resolved_bus.is_connected)
        # One derivation-tick task over the EXISTING registry (issue #147): it emits a
        # typed PresenceTransitionFrame per genuine online<->offline edge. This is a
        # single evaluation loop, not a second presence store or poller.
        transition_task = asyncio.create_task(
            _run_transition_ticks(
                presence_deriver,
                resolved_config.heartbeat_interval_seconds,
            )
        )
        try:
            yield
        finally:
            if transition_task is not None:
                transition_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await transition_task
            if heartbeat_subscription is not None:
                await heartbeat_subscription.unsubscribe()
            if isinstance(resolved_bus, NatsFrameBus):
                await resolved_bus.close()

    app = FastAPI(title="rtech-webchat", lifespan=lifespan)
    app.state.webchat = state
    register_api_routes(app, state, resolved_config)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        logger.debug("webchat.index | entry")
        logger.info("webchat.index | rendered")
        return HTML_PAGE

    @app.post("/task", response_model=TaskResponse)
    async def create_task(request: TaskRequest) -> TaskResponse:
        logger.debug("webchat.task | entry requester_len={}", len(request.requester))
        if request.scope != "single":
            # This route carries no participants and an identity-less duo/trio
            # task can never derive done (issue #75 round 2); named
            # participants go through /api/tasks.
            logger.warning("webchat.task | rejected identity-less scope={}", request.scope)
            raise HTTPException(
                status_code=400,
                detail="scope must be 'single' here; use /api/tasks with participants",
            )
        task = TaskFrame(
            id=state.id_factory(),
            origin_edge="web-chat",
            requester=request.requester,
            text=request.text,
            scope=request.scope,
            stack=resolved_config.stack,
            created_at=datetime.now(UTC),
        )
        state.registry.add(task)
        await state.replay.start_task(task.id)
        await state.bus.publish(subjects.TASK_INGRESS, encode_frame(task))
        logger.info("webchat.task | published task_id={}", task.id)
        return TaskResponse(task_id=task.id)

    @app.get("/tasks", response_model=list[TaskListItem])
    async def list_tasks() -> list[TaskListItem]:
        logger.debug("webchat.tasks | entry")
        items = state.registry.list_items()
        logger.info("webchat.tasks | rendered count={}", len(items))
        return items

    @app.post("/steer", response_model=SteerResponse)
    async def steer(request: SteerRequest) -> SteerResponse:
        logger.debug("webchat.steer | entry task_id={}", request.task_id)
        frame = SteerFrame(
            task_id=request.task_id,
            text=request.text,
            target_agent=request.target_agent,
            interrupt=request.interrupt,
        )
        subject = subjects.task_steer(request.task_id)
        await state.bus.publish(subject, encode_frame(frame))
        logger.info(
            "webchat.steer | published task_id={} target={} interrupt={}",
            request.task_id,
            request.target_agent,
            request.interrupt,
        )
        return SteerResponse(task_id=request.task_id, subject=subject, interrupt=request.interrupt)

    @app.websocket("/ws")
    async def websocket(websocket: WebSocket) -> None:
        logger.debug("webchat.ws | entry")
        await websocket.accept()
        queue: asyncio.Queue[RenderedPayload] = asyncio.Queue()
        forwarders: dict[str, asyncio.Task[None]] = {}
        receiver: asyncio.Task[None] | None = None
        sender: asyncio.Task[None] | None = None
        try:
            await websocket.send_json({"type": "ready"})
            receiver = asyncio.create_task(
                _receive_ws_commands(websocket, state, queue, forwarders)
            )
            sender = asyncio.create_task(_stream_payloads(websocket, queue))
            done, pending = await asyncio.wait(
                {receiver, sender},
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for task in pending:
                task.cancel()
            for task in done:
                task.result()
        except WebSocketDisconnect:
            logger.info("webchat.ws | disconnected")
        finally:
            for maybe_task in (receiver, sender):
                if maybe_task is not None and not maybe_task.done():
                    maybe_task.cancel()
            for forwarder in forwarders.values():
                forwarder.cancel()
            for forwarder in forwarders.values():
                with contextlib.suppress(asyncio.CancelledError):
                    await forwarder
            logger.info("webchat.ws | closed")

    return app


async def _receive_ws_commands(
    websocket: WebSocket,
    state: AppState,
    queue: asyncio.Queue[RenderedPayload],
    forwarders: dict[str, asyncio.Task[None]],
) -> None:
    """Receive multiplex subscription commands from the browser."""

    while True:
        message = await websocket.receive_json()
        if message.get("type") != "subscribe":
            continue
        task_id = message.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            await websocket.send_json({"type": "error", "message": "task_id required"})
            continue
        if not state.registry.has(task_id):
            await websocket.send_json(
                {"type": "error", "task_id": task_id, "message": "unknown task"}
            )
            continue
        await websocket.send_json({"type": "subscribed", "task_id": task_id})
        if task_id not in forwarders:
            attachment = await state.replay.attach(task_id)
            forwarders[task_id] = asyncio.create_task(
                _forward_replay_payloads(state, task_id, attachment, queue)
            )


async def _forward_replay_payloads(
    state: AppState,
    task_id: str,
    attachment: ReplayAttachment,
    queue: asyncio.Queue[RenderedPayload],
) -> None:
    """Replay buffered payloads, then forward live-tail payloads."""

    try:
        for event in attachment.snapshot:
            await queue.put(event.payload)
        if attachment.queue is None:
            return
        while True:
            event = await attachment.queue.get()
            await queue.put(event.payload)
            if event.kind == "result":
                return
    finally:
        await state.replay.detach(task_id, attachment.queue)


async def _stream_payloads(
    websocket: WebSocket,
    queue: asyncio.Queue[RenderedPayload],
) -> None:
    """Forward rendered frame payloads from queue to websocket."""

    while True:
        await websocket.send_json(await queue.get())


async def _run_transition_ticks(
    deriver: PresenceTransitionDeriver,
    interval_seconds: float,
) -> None:
    """Call ``deriver.evaluate()`` once per interval until cancelled (issue #147).

    One evaluation loop over the EXISTING presence registry — the deriver holds all
    edge/dedup logic. A per-tick failure is logged sanitized and the loop continues so
    a transient error never kills presence derivation; ``CancelledError`` propagates so
    the lifespan tears the task down cleanly.
    """

    logger.info("webchat.presence.transition | tick loop start interval={}", interval_seconds)
    try:
        while True:
            try:
                await deriver.evaluate()
            except Exception as exc:
                logger.error(
                    "webchat.presence.transition | tick failed error_type={}",
                    type(exc).__name__,
                )
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        logger.info("webchat.presence.transition | tick loop cancelled")
        raise


def _new_task_id() -> str:
    """Return a new task id."""

    return str(uuid.uuid4())


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>rtech webchat v3</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
      --bg: #0d1117;
      --panel: #12181f;
      --panel-2: #161d26;
      --line: #232c38;
      --text: #e6edf3;
      --muted: #8b98a8;
      --accent: #2f9e6e;
      --accent-dim: #1b5b41;
      --warn: #c9861f;
      --danger: #d05a5a;
      --alive: #3fb950;
      --dead: #4b5563;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); }
    main {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(300px, 380px) minmax(0, 1fr);
    }
    aside, .workspace { padding: 20px; }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      display: grid;
      gap: 16px;
      align-content: start;
      overflow: auto;
      max-height: 100vh;
    }
    h1, h2, h3 { margin: 0; letter-spacing: -0.01em; }
    h1 { font-size: 1.2rem; }
    h2 { font-size: .95rem; }
    h3 { font-size: .82rem; color: var(--muted); font-weight: 600;
         text-transform: uppercase; letter-spacing: .06em; }
    form, .stack { display: grid; gap: 12px; }
    label { display: grid; gap: 6px; font-size: .82rem; color: var(--muted); }
    textarea, input, button { font: inherit; min-width: 0; }
    textarea, input {
      width: 100%;
      padding: 10px;
      border-radius: 7px;
      border: 1px solid var(--line);
      background: var(--bg);
      color: var(--text);
    }
    textarea { min-height: 84px; resize: vertical; }
    input:focus, textarea:focus { outline: none; border-color: var(--accent); }
    button {
      min-height: 38px;
      padding: 9px 14px;
      border-radius: 7px;
      border: 1px solid transparent;
      background: var(--accent);
      color: #06110b;
      font-weight: 600;
      cursor: pointer;
      white-space: nowrap;
    }
    button:disabled { background: var(--accent-dim); color: var(--muted); cursor: not-allowed; }
    button.secondary { background: var(--panel-2); color: var(--text); border-color: var(--line); }
    /* agent picker */
    .picker-head {
      display: flex; align-items: center; justify-content: space-between; gap: 8px;
    }
    .agents { display: grid; gap: 6px; }
    .agent {
      display: flex; align-items: center; gap: 10px;
      padding: 8px 10px; border: 1px solid var(--line);
      border-radius: 7px; background: var(--bg); cursor: pointer;
      font-size: .86rem;
    }
    .agent[data-checked="true"] { border-color: var(--accent); background: var(--panel-2); }
    .agent input { width: auto; accent-color: var(--accent); }
    .agent .dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
    .dot.alive { background: var(--alive); box-shadow: 0 0 6px var(--alive); }
    .dot.dead { background: var(--dead); }
    .agent .id { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .agent .seen { color: var(--muted); font-size: .72rem; }
    .scope-line { display: flex; align-items: center; gap: 8px; font-size: .8rem; }
    .scope-line { color: var(--muted); }
    .scope-line.over { color: var(--danger); }
    .agents-error {
      color: var(--danger); font-size: .82rem;
      border: 1px solid var(--danger); border-radius: 7px;
      padding: 8px 10px; background: rgba(208, 90, 90, .08);
    }
    .participants {
      display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
    }
    /* task list */
    .task-list { display: grid; gap: 8px; }
    .task-row {
      width: 100%; display: grid; grid-template-columns: 1fr auto;
      gap: 4px 10px; text-align: left; align-items: center;
      background: var(--bg); color: var(--text);
      border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px;
      cursor: pointer;
    }
    .task-row[aria-selected="true"] { border-color: var(--accent); background: var(--panel-2); }
    .task-title {
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 600;
    }
    .task-row .meta { grid-column: 1; }
    .task-row .pill { grid-column: 2; grid-row: 1 / span 2; }
    .meta { color: var(--muted); font-size: .74rem; }
    /* pills + chips */
    .pill, .chip, .badge {
      display: inline-flex; align-items: center; gap: 5px;
      width: max-content; max-width: 100%;
      border-radius: 999px; padding: 3px 9px;
      font-size: .72rem; line-height: 1.4;
      background: var(--panel-2); color: var(--text);
      border: 1px solid var(--line);
    }
    .pill.working, .badge.working { border-color: var(--accent); color: var(--accent); }
    .pill.input_required, .badge.input_required { border-color: var(--warn); color: var(--warn); }
    .pill.blocked, .badge.blocked { border-color: var(--danger); color: var(--danger); }
    .pill.result, .badge.result {
      border-color: var(--accent); background: var(--accent-dim); color: var(--text);
    }
    .scope-badge {
      display: inline-flex; align-items: center;
      border-radius: 999px; padding: 3px 10px; font-size: .72rem;
      background: var(--accent-dim); color: var(--text); border: 1px solid var(--accent);
    }
    /* workspace */
    .workspace {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 16px; min-width: 0; max-height: 100vh;
    }
    .task-head {
      display: flex; justify-content: space-between; gap: 12px;
      align-items: flex-start; border-bottom: 1px solid var(--line); padding-bottom: 14px;
    }
    .task-head .meta { margin-top: 6px; word-break: break-all; }
    .timeline {
      display: grid; align-content: start; gap: 10px;
      min-height: 200px; overflow: auto; padding-right: 4px;
    }
    .frame {
      display: grid; gap: 8px;
      border: 1px solid var(--line); border-radius: 9px;
      padding: 12px 14px; background: var(--panel);
    }
    .frame .fhead { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .frame .agent-id { font-weight: 650; font-size: .88rem; }
    .frame p { margin: 0; white-space: pre-wrap; line-height: 1.5; }
    .chips, .refs { display: flex; flex-wrap: wrap; gap: 6px; }
    .refs { font-size: .74rem; color: var(--muted); }
    .refs a { color: #6cb6ff; overflow-wrap: anywhere; }
    .status-line {
      display: flex; align-items: center; gap: 8px;
      font-size: .78rem; color: var(--muted);
      border-left: 2px solid var(--line); padding: 4px 0 4px 12px;
    }
    .completed-banner {
      display: flex; align-items: center; gap: 8px;
      border: 1px solid var(--accent); background: var(--accent-dim);
      border-radius: 9px; padding: 12px 14px; color: var(--text); font-weight: 600;
    }
    /* steer */
    .steer {
      border-top: 1px solid var(--line); padding-top: 14px;
      display: grid; gap: 10px;
    }
    .steer-row {
      display: grid; grid-template-columns: minmax(150px, 220px) 1fr auto;
      gap: 10px; align-items: end;
    }
    .check { display: flex; align-items: center; gap: 8px; color: var(--text); font-size: .82rem; }
    .check input { width: auto; accent-color: var(--warn); }
    .empty { color: var(--muted); padding: 24px 4px; font-size: .88rem; }
    @media (max-width: 820px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); max-height: none; }
      .workspace { max-height: none; }
      .steer-row { grid-template-columns: 1fr; }
      .task-head { display: grid; }
    }
  </style>
</head>
<body>
  <main>
    <aside>
      <h1>rtech webchat</h1>
      <form id="task-form" class="stack">
        <label>edge API key (sent as Bearer; blank for open local edge)
          <input id="api-key" type="password" autocomplete="off" placeholder="unset">
        </label>
        <label>requester (provenance label)
          <input id="requester" autocomplete="off" required>
        </label>
        <section class="stack">
          <div class="picker-head">
            <h3>Agents</h3>
            <span id="scope-badge" class="scope-badge">no scope</span>
          </div>
          <div id="agents" class="agents"></div>
          <div id="agents-error" class="agents-error" hidden></div>
          <div id="scope-line" class="scope-line">Select 1&ndash;3 agents.</div>
        </section>
        <label>Task <textarea id="task-text" required></textarea></label>
        <button id="send-task" type="submit" disabled>Select an agent</button>
      </form>
      <section class="stack">
        <h3>Tasks</h3>
        <div id="task-list" class="task-list"></div>
      </section>
    </aside>
    <section class="workspace">
      <header class="task-head">
        <div>
          <h2 id="selected-title">No task selected</h2>
          <div id="selected-participants" class="participants"></div>
          <div id="selected-meta" class="meta"></div>
        </div>
        <div id="selected-status" class="pill">idle</div>
      </header>
      <div id="timeline" class="timeline">
        <div class="empty">Select or mint a task.</div>
      </div>
      <form id="steer-form" class="steer">
        <div class="steer-row">
          <label>Target agent <input id="target-agent" placeholder="all" autocomplete="off"></label>
          <label>Steer <textarea id="steer-text" required></textarea></label>
          <button type="submit">Send steer</button>
        </div>
        <label class="check"><input id="interrupt" type="checkbox"> Interrupt current turn</label>
      </form>
    </section>
  </main>
  <script>
    const REQUESTER_KEY = "rtech-webchat.requester";
    const API_KEY_KEY = "rtech-webchat.api-key";
    const SCOPE_BY_COUNT = {0: "no scope", 1: "single", 2: "duo", 3: "trio"};
    const MAX_PARTICIPANTS = 3;
    const AGENT_POLL_MS = 5000;
    const tasks = new Map();
    const agents = new Map();
    const selectedAgents = new Set();
    const subscribed = new Set();
    let selectedTaskId = "";
    let agentsAuthError = false;
    // Fail closed: until a 200 carries a recognized typed verdict, presence is
    // "unavailable" — never the authoritative "No agents online" (issue #145).
    // Initial load, an absent field (older/misbehaving edge), or an
    // unrecognized value all render presence unavailable, not healthy-empty.
    let agentsPipeline = "unavailable";

    const apiKeyEl = document.querySelector("#api-key");
    const requesterEl = document.querySelector("#requester");
    const agentsEl = document.querySelector("#agents");
    const scopeBadgeEl = document.querySelector("#scope-badge");
    const scopeLineEl = document.querySelector("#scope-line");
    const agentsErrorEl = document.querySelector("#agents-error");
    const sendTaskEl = document.querySelector("#send-task");
    const taskListEl = document.querySelector("#task-list");
    const timelineEl = document.querySelector("#timeline");
    const selectedTitleEl = document.querySelector("#selected-title");
    const selectedParticipantsEl = document.querySelector("#selected-participants");
    const selectedMetaEl = document.querySelector("#selected-meta");
    const selectedStatusEl = document.querySelector("#selected-status");

    requesterEl.value = localStorage.getItem(REQUESTER_KEY) || "web";
    requesterEl.addEventListener("input", () => {
      localStorage.setItem(REQUESTER_KEY, requesterEl.value);
    });

    // ---- edge API key: masked, persisted, sent as Bearer on every fetch ----
    apiKeyEl.value = localStorage.getItem(API_KEY_KEY) || "";
    apiKeyEl.addEventListener("input", () => {
      localStorage.setItem(API_KEY_KEY, apiKeyEl.value);
      // Re-probe presence immediately so the picker reflects the new key.
      pollAgents();
    });
    function apiKey() { return (apiKeyEl.value || "").trim(); }
    function authHeaders(base) {
      const headers = base ? {...base} : {};
      const key = apiKey();
      if (key) headers["Authorization"] = "Bearer " + key;
      return headers;
    }

    // ---- agent picker: poll /api/agents, derive scope from selection count ----
    function seenLabel(ms) {
      if (!ms) return "";
      const delta = Date.now() - ms;
      if (delta < 0) return "now";
      const s = Math.floor(delta / 1000);
      if (s < 60) return s + "s ago";
      const m = Math.floor(s / 60);
      if (m < 60) return m + "m ago";
      return Math.floor(m / 60) + "h ago";
    }
    function selectionCount() { return selectedAgents.size; }
    function derivedScope() { return SCOPE_BY_COUNT[selectionCount()] || "over"; }
    // Name the destination on the Send button so the operator always sees where
    // a message goes: single -> the agent id; 2-3 -> count + derived scope.
    function destinationLabel() {
      const chosen = [...selectedAgents];
      if (!chosen.length) return "Select an agent";
      if (chosen.length === 1) return "Send to " + chosen[0];
      return "Send to " + chosen.length + " agents (" + derivedScope() + ")";
    }
    function updateScopeUi() {
      const count = selectionCount();
      const over = count > MAX_PARTICIPANTS;
      scopeBadgeEl.textContent = over ? "too many" : derivedScope();
      scopeLineEl.classList.toggle("over", over);
      if (over) {
        scopeLineEl.textContent = "Pick at most 3 agents (" + count + " selected).";
      } else if (count === 0) {
        scopeLineEl.textContent = "Select 1–3 agents.";
      } else {
        scopeLineEl.textContent = count + " selected → " + derivedScope();
      }
      sendTaskEl.disabled = count === 0 || over;
      sendTaskEl.textContent = over ? "Too many agents" : destinationLabel();
    }
    function toggleAgent(agentId, checked) {
      if (checked) selectedAgents.add(agentId);
      else selectedAgents.delete(agentId);
      renderAgents();
      updateScopeUi();
    }
    function renderAgents() {
      agentsEl.replaceChildren();
      // 401 and empty-but-authed are distinct states; never conflate them.
      if (agentsAuthError) {
        agentsErrorEl.hidden = false;
        agentsErrorEl.textContent = "API key required or invalid.";
        return;
      }
      // subscription_failed is distinct from healthy-but-empty: the heartbeat
      // feed is down, so the (empty) list is NOT liveness truth (issue #145).
      if (agentsPipeline === "subscription_failed") {
        agentsErrorEl.hidden = false;
        agentsErrorEl.textContent = "Presence feed unavailable.";
        return;
      }
      // Fail closed: an initial/absent/unrecognized verdict is NOT trusted as
      // healthy-empty, so the (empty) list is never rendered as the
      // authoritative "No agents online" (issue #145). Presence is unknown until
      // a 200 carries a recognized typed verdict.
      if (agentsPipeline !== "healthy") {
        agentsErrorEl.hidden = false;
        agentsErrorEl.textContent = "Presence unavailable.";
        return;
      }
      agentsErrorEl.hidden = true;
      if (!agents.size) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No agents online.";
        agentsEl.append(empty);
        return;
      }
      for (const agent of agents.values()) {
        const checked = selectedAgents.has(agent.agent_id);
        const row = document.createElement("label");
        row.className = "agent";
        row.setAttribute("data-checked", String(checked));
        const box = document.createElement("input");
        box.type = "checkbox";
        box.checked = checked;
        box.addEventListener("change", () => toggleAgent(agent.agent_id, box.checked));
        const dot = document.createElement("span");
        dot.className = "dot " + (agent.alive ? "alive" : "dead");
        const id = document.createElement("span");
        id.className = "id";
        id.textContent = agent.agent_id;
        const seen = document.createElement("span");
        seen.className = "seen";
        seen.textContent = seenLabel(agent.last_seen_ms);
        row.append(box, dot, id, seen);
        agentsEl.append(row);
      }
    }
    async function pollAgents() {
      try {
        const response = await fetch("/api/agents", {headers: authHeaders()});
        if (response.status === 401) {
          // Keyed edge rejected us: surface as an auth error, not "no agents".
          agentsAuthError = true;
          agents.clear();
          selectedAgents.clear();
          renderAgents();
          updateScopeUi();
          return;
        }
        agentsAuthError = false;
        if (!response.ok) return;
        const body = await response.json();
        // Fail closed on the typed pipeline verdict: ONLY the two recognized
        // wire values are trusted. An absent or unrecognized field (older or
        // misbehaving edge) is "unavailable", never coerced to healthy — a 200
        // with an invalid verdict must not read as authoritative
        // "No agents online" (issue #145).
        if (body.pipeline === "healthy") {
          agentsPipeline = "healthy";
        } else if (body.pipeline === "subscription_failed") {
          agentsPipeline = "subscription_failed";
        } else {
          agentsPipeline = "unavailable";
        }
        // An untrusted verdict carries no liveness truth: drop any rows and
        // selections so a stale roster is never presented as current.
        if (agentsPipeline === "unavailable") {
          agents.clear();
          selectedAgents.clear();
          renderAgents();
          updateScopeUi();
          return;
        }
        const seen = new Set();
        for (const agent of body.agents || []) {
          agents.set(agent.agent_id, agent);
          seen.add(agent.agent_id);
        }
        for (const id of [...agents.keys()]) if (!seen.has(id)) agents.delete(id);
        for (const id of [...selectedAgents]) if (!agents.has(id)) selectedAgents.delete(id);
        renderAgents();
        updateScopeUi();
      } catch (err) {
        /* dumb edge: presence is best-effort, keep last known */
      }
    }

    // ---- keyed live-tail over SSE via fetch-streaming ----
    // EventSource cannot set an Authorization header, so we consume the keyed
    // /api/tasks/{id}/stream route with fetch() streaming instead. The key rides
    // in the Bearer header, never in the URL or a query param — nothing to leak.
    function badgeText(task) {
      if (!task) return "idle";
      return task.result_received ? "result" : (task.status || "minted");
    }
    function upsertTask(task) {
      const existing = tasks.get(task.task_id) || {frames: []};
      tasks.set(task.task_id, {
        ...existing, ...task,
        frames: task.frames || existing.frames || []
      });
    }
    // Normalize a wire frame ({frame_type}) into the shape the timeline renders
    // ({type}). The SSE route closes on result rather than emitting one.
    function normalizeFrame(taskId, wire) {
      const type = wire.frame_type || wire.type;
      const frame = {...wire, type, task_id: wire.task_id || taskId};
      delete frame.frame_type;
      return frame;
    }
    function receiveFrame(frame) {
      const task = tasks.get(frame.task_id);
      if (!task) return;
      task.frames.push(frame);
      if (frame.type === "status") task.status = frame.state;
      if (frame.type === "result") task.result_received = true;
      renderTasks();
      if (selectedTaskId === frame.task_id) renderSelected();
    }
    function markResult(taskId) {
      const task = tasks.get(taskId);
      if (!task || task.result_received) return;
      task.result_received = true;
      task.frames.push({type: "result", task_id: taskId});
      renderTasks();
      if (selectedTaskId === taskId) renderSelected();
    }
    async function subscribeTask(taskId) {
      if (subscribed.has(taskId)) return;
      subscribed.add(taskId);
      try {
        await streamTask(taskId);
      } finally {
        subscribed.delete(taskId);
      }
    }
    async function streamTask(taskId) {
      const url = `/api/tasks/${encodeURIComponent(taskId)}/stream`;
      let response;
      try {
        response = await fetch(url, {headers: authHeaders({Accept: "text/event-stream"})});
      } catch (err) {
        return; // dumb edge: transient network error, caller may resubscribe
      }
      if (!response.ok || !response.body) return;
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        let chunk;
        try {
          chunk = await reader.read();
        } catch (err) {
          break;
        }
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, {stream: true});
        let sep;
        while ((sep = buffer.indexOf("\\n\\n")) !== -1) {
          const event = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);
          for (const line of event.split("\\n")) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (!data) continue;
            try {
              receiveFrame(normalizeFrame(taskId, JSON.parse(data)));
            } catch (err) { /* skip malformed line */ }
          }
        }
      }
      // The keyed stream closes when the task completes; mirror that as a result.
      markResult(taskId);
    }

    // ---- task list (left) ----
    function firstWords(text) {
      const words = (text || "").trim().split(/\\s+/).slice(0, 8).join(" ");
      return words || "(empty task)";
    }
    // Compact destination summary from the minted TaskFrame the POST returned.
    function participantsOf(task) {
      return (task && Array.isArray(task.participants)) ? task.participants : [];
    }
    function destinationText(task) {
      const to = participantsOf(task);
      const scope = (task && task.scope) || "single";
      if (!to.length) return scope;
      return "to: " + to.join(" + ") + " · " + scope;
    }
    function renderTasks() {
      taskListEl.replaceChildren();
      if (!tasks.size) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No tasks minted.";
        taskListEl.append(empty);
        return;
      }
      for (const task of tasks.values()) {
        const row = document.createElement("button");
        row.type = "button";
        row.className = "task-row";
        row.setAttribute("aria-selected", String(task.task_id === selectedTaskId));
        row.addEventListener("click", () => selectTask(task.task_id));
        const title = document.createElement("div");
        title.className = "task-title";
        title.textContent = firstWords(task.text);
        const meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = `${task.requester} · ${destinationText(task)}`;
        const pill = document.createElement("span");
        pill.className = `pill ${badgeText(task)}`;
        pill.textContent = badgeText(task);
        row.append(title, meta, pill);
        taskListEl.append(row);
      }
    }
    function selectTask(taskId) {
      selectedTaskId = taskId;
      subscribeTask(taskId);
      renderTasks();
      renderSelected();
    }

    // ---- timeline (right) ----
    // Header participants + scope badge, sourced from the minted TaskFrame.
    function renderSelectedParticipants(task) {
      selectedParticipantsEl.replaceChildren();
      if (!task) return;
      const to = participantsOf(task);
      if (to.length) {
        const label = document.createElement("span");
        label.className = "meta";
        label.textContent = "to:";
        selectedParticipantsEl.append(label);
        for (const agentId of to) {
          const chip = document.createElement("span");
          chip.className = "chip";
          chip.textContent = agentId;
          selectedParticipantsEl.append(chip);
        }
      }
      const badge = document.createElement("span");
      badge.className = "scope-badge";
      badge.textContent = task.scope || "single";
      selectedParticipantsEl.append(badge);
    }
    function renderSelected() {
      const task = tasks.get(selectedTaskId);
      selectedTitleEl.textContent = task ? firstWords(task.text) : "No task selected";
      renderSelectedParticipants(task);
      selectedMetaEl.textContent = task
        ? `${task.task_id} · ${task.requester}`
        : "";
      selectedStatusEl.className = `pill ${badgeText(task)}`;
      selectedStatusEl.textContent = badgeText(task);
      timelineEl.replaceChildren();
      if (!task || !task.frames.length) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = task ? "Waiting for frames." : "Select or mint a task.";
        timelineEl.append(empty);
        return;
      }
      for (const frame of task.frames) timelineEl.append(renderFrame(frame));
      timelineEl.scrollTop = timelineEl.scrollHeight;
    }
    function renderFrame(frame) {
      if (frame.type === "turn") return renderTurn(frame);
      if (frame.type === "status") return renderStatus(frame);
      return renderResult();
    }
    function renderTurn(frame) {
      const node = document.createElement("article");
      node.className = "frame";
      const head = document.createElement("div");
      head.className = "fhead";
      const who = document.createElement("span");
      who.className = "agent-id";
      who.textContent = frame.agent_id;
      const tag = document.createElement("span");
      tag.className = "meta";
      tag.textContent = "turn";
      head.append(who, tag);
      const text = document.createElement("p");
      text.textContent = frame.visible_text;
      const chips = document.createElement("div");
      chips.className = "chips";
      const stance = frame.stance || {};
      if (stance.position) addChip(chips, "position", stance.position);
      if (stance.because) addChip(chips, "because", stance.because);
      for (const need of stance.needs_from || []) addChip(chips, "needs", need);
      node.append(head, text, chips);
      const refs = frame.refs || [];
      if (refs.length) {
        const refBox = document.createElement("div");
        refBox.className = "refs";
        refBox.append(document.createTextNode("refs: "));
        for (const ref of refs) {
          const link = document.createElement("a");
          link.href = ref;
          link.textContent = ref;
          link.target = "_blank";
          link.rel = "noreferrer";
          refBox.append(link);
        }
        node.append(refBox);
      }
      return node;
    }
    function renderStatus(frame) {
      const node = document.createElement("div");
      node.className = "status-line";
      const pill = document.createElement("span");
      pill.className = `badge ${frame.state}`;
      pill.textContent = frame.state;
      node.append(pill, document.createTextNode("status"));
      return node;
    }
    function renderResult() {
      const node = document.createElement("div");
      node.className = "completed-banner";
      node.textContent = "✓ completed";
      return node;
    }
    function addChip(parent, name, value) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = `${name}: ${value}`;
      parent.append(chip);
    }

    // ---- ingress ----
    async function refreshTasks() {
      try {
        const response = await fetch("/tasks", {headers: authHeaders()});
        if (response.ok) {
          for (const task of await response.json()) upsertTask(task);
        }
      } catch (err) { /* keep last known */ }
      renderTasks();
      if (!selectedTaskId && tasks.size) selectTask(tasks.keys().next().value);
      renderSelected();
      for (const taskId of tasks.keys()) subscribeTask(taskId);
    }
    document.querySelector("#task-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const participants = [...selectedAgents];
      if (!participants.length || participants.length > MAX_PARTICIPANTS) return;
      const scope = SCOPE_BY_COUNT[participants.length];
      const response = await fetch("/api/tasks", {
        method: "POST",
        headers: authHeaders({"content-type": "application/json"}),
        body: JSON.stringify({
          requester: requesterEl.value,
          text: document.querySelector("#task-text").value,
          participants: participants
        })
      });
      if (!response.ok) return;
      const payload = await response.json();
      // Persist the minted destination locally so cards/header can show it even
      // before /tasks round-trips it back.
      upsertTask({
        task_id: payload.task_id,
        requester: requesterEl.value,
        text: document.querySelector("#task-text").value,
        scope: scope,
        participants: participants,
        status: "minted",
        result_received: false
      });
      await refreshTasks();
      selectTask(payload.task_id);
      document.querySelector("#task-text").value = "";
    });
    document.querySelector("#steer-form").addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!selectedTaskId) return;
      const target = document.querySelector("#target-agent").value.trim();
      const body = {
        text: document.querySelector("#steer-text").value,
        interrupt: document.querySelector("#interrupt").checked
      };
      if (target) body.target_agent = target;
      await fetch(`/api/tasks/${encodeURIComponent(selectedTaskId)}/steer`, {
        method: "POST",
        headers: authHeaders({"content-type": "application/json"}),
        body: JSON.stringify(body)
      });
      document.querySelector("#steer-text").value = "";
      document.querySelector("#interrupt").checked = false;
    });

    updateScopeUi();
    renderAgents();
    refreshTasks();
    pollAgents();
    setInterval(pollAgents, AGENT_POLL_MS);
  </script>
</body>
</html>
"""
