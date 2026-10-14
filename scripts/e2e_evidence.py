"""Prove a fully typed bus-first round-trip against the LIVE fleet NATS bus.

Boot all three bus-first roles IN-PROCESS against the live fleet NATS
server, drive a task + mid-turn steer through the web-chat edge, and write an
ordered, timestamped transcript of every frame that crossed the wire to
``.evidence/webchat-<ts>.json``.

Roles, each on its own live connection:

* **Web-chat edge** -- ``rtech_webchat.app.create_app`` driven via an httpx
  ``ASGITransport`` client, publishing on its own ``NatsFrameBus``. A fixed
  ``id_factory`` mints a known ``TASK_ID`` up front so every lane subject is
  known before anything is published.
* **Orchestrator** -- ``rtech_orchestrator.Orchestrator`` over a typed
  ``FrameBus`` seam, routing single-scope tasks to agentd and deriving done from
  the frame graph.
* **agentd** -- ``rtech_agentd.supervisor.Supervisor`` +
  ``rtech_agentd.frame_turn.FrameTurnRunner`` over the typed ``FrameBus`` seam,
  emitting ``StatusFrame(working)`` then the ``TurnFrame`` and subscribing the
  steer lane.

The typed ``FrameBus`` seam is bridged onto the raw-bytes ``fleet_nats.FleetBus``
live transport by :class:`TypedFrameBus`.

WS render proof: httpx's ``ASGITransport`` cannot upgrade a WebSocket, so the WS
render contract is proven by MIRRORING the app's own WS ``on_frame`` handler --
subscribing the edge bus for the task turn/status lanes and formatting exactly as
``rtech_webchat.app._stream_frames`` emits to the browser. This is the same
decode-and-forward the browser receives.

Exit 0 ONLY on a fully proven round-trip (run doctrine rule 4 / brief hard-rule 1).
Regenerate: ``uv run python scripts/e2e_evidence.py``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from fleet_nats import BusConfig, FleetBus
from loguru import logger
from rtech_agentd.engine import ToolsPolicy
from rtech_agentd.frame_turn import FrameTurnRunner
from rtech_agentd.scripted_engine import ScriptedEngine
from rtech_agentd.supervisor import Supervisor, SupervisorConfig
from rtech_frames.codec import Frame, decode_frame
from rtech_frames.frames import StatusFrame, TurnFrame
from rtech_orchestrator import Orchestrator, OrchestratorConfig
from rtech_orchestrator import subjects as osubj
from rtech_orchestrator.typedbus import TypedFrameBus
from rtech_webchat import subjects as wsubj
from rtech_webchat.app import create_app
from rtech_webchat.bus import NatsFrameBus
from rtech_webchat.config import WebchatConfig

# --- constants ---------------------------------------------------------------

DEFAULT_NATS_URL = "nats://127.0.0.1:4222"
ENV_NATS_URL = "RTECH_WEBCHAT_NATS_URL"
ENV_ENGINE = "E2E_ENGINE"
ENV_MODEL = "E2E_MODEL"
ENV_BASE = "ANTHROPIC_BASE_URL"
ENV_KEY = "ANTHROPIC_API_KEY"

AGENT_ID = "agentd"
# Stack id shared by the in-process webchat edge and orchestrator so the
# ingress guard accepts this run's own tasks.
STACK_ID = "e2e-evidence"
TASK_ID = uuid4().hex
BASE_URL = "http://e2e.local"
EVIDENCE_DIR = Path(__file__).resolve().parent.parent / ".evidence"
HOME_DIR = EVIDENCE_DIR / "e2e-home"
SYSTEM_PROMPT = "e2e scripted round-trip"

REACHABILITY_TIMEOUT_S = 2.0
SUB_SETTLE_S = 0.3
EVENT_TIMEOUT_S = 15.0

TASK_TEXT = "Prove the round-trip."
STEER_TEXT = "steer: focus the answer"

# Tokens that MUST NOT appear anywhere in the transcript (regression scan).
FORBIDDEN_TOKENS = ("[[done]]", "done_sentinel", "TurnSignal", "trailer.scrape")

# Env keys whose VALUES are redacted from the transcript when long enough.
_SECRET_KEY_MARKERS = ("TOKEN", "SECRET", "KEY", "PASSWORD", "CREDENTIAL", "API")
_MIN_SECRET_LEN = 8
_REDACTED = "«redacted»"

# Token-shape regexes redacted regardless of source.
_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{8,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{8,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
)


class _ProofFailure(Exception):
    """Raised when a proof step fails; the transcript is still written."""


# --- transcript records ------------------------------------------------------


@dataclass
class CapturedFrame:
    """One frame observed on a lane, in arrival order."""

    seq: int
    ts: str
    subject: str
    frame_type: str
    payload: dict[str, Any]


@dataclass
class Capture:
    """Accumulate captured frames + fire the milestone events off the wire."""

    frames: list[CapturedFrame] = field(default_factory=list)
    ws_messages: list[dict[str, Any]] = field(default_factory=list)
    turns_seen: int = 0
    _seq: int = 0
    task_frame_published: asyncio.Event = field(default_factory=asyncio.Event)
    working_status: asyncio.Event = field(default_factory=asyncio.Event)
    first_turn: asyncio.Event = field(default_factory=asyncio.Event)
    steered_turn: asyncio.Event = field(default_factory=asyncio.Event)
    result: asyncio.Event = field(default_factory=asyncio.Event)

    def record(self, subject: str, frame: Frame) -> CapturedFrame:
        """Append a captured frame with a monotonically increasing sequence."""
        self._seq += 1
        captured = CapturedFrame(
            seq=self._seq,
            ts=datetime.now(UTC).isoformat(),
            subject=subject,
            frame_type=type(frame).__name__,
            payload=frame.model_dump(mode="json"),
        )
        self.frames.append(captured)
        return captured


def _make_capture_handlers(cap: Capture) -> dict[str, Callable[[Frame], Awaitable[None]]]:
    """Build per-lane capture handlers bound to a :class:`Capture`."""

    async def cap_ingress(frame: Frame) -> None:
        cap.record(osubj.TASK_INGRESS, frame)
        if type(frame).__name__ == "TaskFrame":
            cap.task_frame_published.set()

    async def cap_agent(frame: Frame) -> None:
        cap.record(osubj.agent_task(AGENT_ID), frame)

    async def cap_turn(frame: Frame) -> None:
        cap.record(osubj.task_turn(TASK_ID), frame)
        cap.turns_seen += 1
        cap.first_turn.set()
        visible = getattr(frame, "visible_text", "") or ""
        if "steer" in visible.lower() or cap.turns_seen >= 2:
            cap.steered_turn.set()

    async def cap_status(frame: Frame) -> None:
        cap.record(osubj.task_status(TASK_ID), frame)
        if getattr(frame, "state", None) == "working":
            cap.working_status.set()

    async def cap_steer(frame: Frame) -> None:
        cap.record(osubj.task_steer(TASK_ID), frame)

    async def cap_result(frame: Frame) -> None:
        cap.record(osubj.task_result(TASK_ID), frame)
        cap.result.set()

    return {
        osubj.TASK_INGRESS: cap_ingress,
        osubj.agent_task(AGENT_ID): cap_agent,
        osubj.task_turn(TASK_ID): cap_turn,
        osubj.task_status(TASK_ID): cap_status,
        osubj.task_steer(TASK_ID): cap_steer,
        osubj.task_result(TASK_ID): cap_result,
    }


def _make_ws_mirror(cap: Capture) -> Callable[[bytes], Awaitable[None]]:
    """Return an ``on_frame`` mirror of the app's WS forward contract.

    httpx's ``ASGITransport`` cannot upgrade a WebSocket, so this mirrors the
    exact decode-and-forward the browser receives via ``_stream_frames``: turn
    frames become ``{"type":"turn", ...}`` and status frames
    ``{"type":"status", ...}``. Same bytes, same formatting, same lanes.
    """

    async def on_frame(data: bytes) -> None:
        try:
            frame = decode_frame(data)
        except Exception as exc:
            logger.error("ws_mirror.decode | failed reason={}", exc)
            return
        if isinstance(frame, TurnFrame):
            cap.ws_messages.append(
                {
                    "type": "turn",
                    "task_id": frame.task_id,
                    "agent_id": frame.agent_id,
                    "visible_text": frame.visible_text,
                    "refs": frame.refs,
                }
            )
            return
        if isinstance(frame, StatusFrame):
            cap.ws_messages.append(
                {"type": "status", "task_id": frame.task_id, "state": frame.state}
            )
            return
        logger.warning("ws_mirror | ignored frame type={}", type(frame).__name__)

    return on_frame


# --- redaction ---------------------------------------------------------------


def _secret_env_values() -> set[str]:
    """Collect concrete env VALUES to redact (secret-shaped keys, len>=8)."""
    values: set[str] = set()
    for key, value in os.environ.items():
        upper = key.upper()
        if not any(marker in upper for marker in _SECRET_KEY_MARKERS):
            continue
        if len(value) >= _MIN_SECRET_LEN:
            values.add(value)
    return values


def _redact_text(text: str, secret_values: set[str]) -> str:
    """Redact concrete secret values and token shapes from a string."""
    redacted = text
    for value in secret_values:
        if value and value in redacted:
            redacted = redacted.replace(value, _REDACTED)
    for pattern in _TOKEN_PATTERNS:
        redacted = pattern.sub(_REDACTED, redacted)
    return redacted


def _redact(obj: Any, secret_values: set[str]) -> Any:
    """Recursively redact secrets from a JSON-serializable structure."""
    if isinstance(obj, str):
        return _redact_text(obj, secret_values)
    if isinstance(obj, dict):
        return {k: _redact(v, secret_values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v, secret_values) for v in obj]
    return obj


# --- engine selection --------------------------------------------------------


def _build_engine() -> tuple[Any, str] | tuple[None, str]:
    """Build the engine per ``E2E_ENGINE``; return (engine, name) or (None, reason)."""
    kind = os.environ.get(ENV_ENGINE, "scripted").strip().lower()
    if kind == "scripted":
        return ScriptedEngine(), "scripted"
    if kind == "claude":
        return _build_claude_engine()
    logger.error("engine.select | unknown engine kind={}", kind)
    return None, f"unknown engine kind: {kind}"


def _build_claude_engine() -> tuple[Any, str] | tuple[None, str]:
    """Build the real Claude engine only if credentials are present."""
    if not (os.environ.get(ENV_KEY) or os.environ.get(ENV_BASE)):
        reason = "claude engine requires ANTHROPIC_API_KEY or ANTHROPIC_BASE_URL"
        logger.error("engine.select | {}", reason)
        return None, reason
    try:
        from rtech_agentd.claude_engine import ClaudeEngine
        from rtech_agentd.config import EngineConfig

        cfg = EngineConfig(
            kind="claude",
            model=os.environ.get(ENV_MODEL, "claude-sonnet-5"),
            litellm_base=os.environ.get(ENV_BASE),
            backup_model=None,
        )
        return ClaudeEngine(cfg), "claude"
    except Exception as exc:
        logger.error("engine.select | claude engine build failed reason={}", exc)
        return None, f"claude engine build failed: {exc}"


# --- reachability ------------------------------------------------------------


def _nats_reachable(url: str) -> bool:
    """Return whether the NATS host:port accepts a TCP connection."""
    without_scheme = url.split("://", 1)[-1]
    host_port = without_scheme.split("/", 1)[0]
    host, _, port_raw = host_port.partition(":")
    port = int(port_raw) if port_raw else 4222
    try:
        with socket.create_connection((host, port), REACHABILITY_TIMEOUT_S):
            return True
    except OSError as exc:
        logger.error("nats.reachability | unreachable host={} port={} reason={}", host, port, exc)
        return False


async def _connect_fleet_bus(url: str, agent_id: str) -> FleetBus:
    """Create and connect a fleet-nats bus for a role."""
    bus = FleetBus(BusConfig(url=url, agent_id=agent_id, env="dev"))
    await bus.connect()
    return bus


# --- proof steps -------------------------------------------------------------


async def _await_event(event: asyncio.Event, name: str) -> None:
    """Await a milestone event with a timeout, raising ``_ProofFailure``."""
    try:
        await asyncio.wait_for(event.wait(), EVENT_TIMEOUT_S)
    except TimeoutError as exc:
        raise _ProofFailure(f"timed out awaiting {name} ({EVENT_TIMEOUT_S}s)") from exc


async def _drive_proof(
    client: httpx.AsyncClient, cap: Capture, steps: list[dict[str, Any]]
) -> None:
    """Drive the task + steer proof sequence through the edge; record each step."""
    task_resp = await client.post(
        "/task",
        json={"text": TASK_TEXT, "requester": "e2e", "scope": "single"},
    )
    task_ok = task_resp.status_code == 200 and task_resp.json().get("task_id") == TASK_ID
    steps.append({"step": "post_task", "status": task_resp.status_code, "ok": task_ok})
    if not task_ok:
        raise _ProofFailure(f"POST /task failed status={task_resp.status_code}")

    await _await_event(cap.working_status, "working_status")
    steps.append({"step": "working_status", "ok": True})

    steer_resp = await client.post(
        "/steer",
        json={"task_id": TASK_ID, "text": STEER_TEXT, "target_agent": AGENT_ID},
    )
    steer_ok = steer_resp.status_code == 200
    steps.append({"step": "post_steer", "status": steer_resp.status_code, "ok": steer_ok})
    if not steer_ok:
        raise _ProofFailure(f"POST /steer failed status={steer_resp.status_code}")

    await _await_event(cap.steered_turn, "steered_turn")
    if not any(m["type"] == "turn" for m in cap.ws_messages):
        raise _ProofFailure("no WS turn message arrived after steer")
    steps.append({"step": "steered_turn", "ok": True})

    await _await_event(cap.result, "result")
    steps.append({"step": "result", "ok": True})


def _compute_assertions(cap: Capture) -> tuple[dict[str, Any], bool]:
    """Compute the pass/fail assertion block; return (assertions, all_passed)."""
    turn_rendered_ws = any(m["type"] == "turn" for m in cap.ws_messages)
    blob = json.dumps([c.payload for c in cap.frames])
    token_hits = [tok for tok in FORBIDDEN_TOKENS if tok in blob]
    assertions = {
        "task_frame_published": cap.task_frame_published.is_set(),
        "turn_rendered_ws": turn_rendered_ws,
        "steer_reflected": cap.steered_turn.is_set(),
        "done_derived_result_fired": cap.result.is_set(),
        "no_done_token_in_frames": not token_hits,
        "forbidden_token_hits": token_hits,
    }
    passed = all(
        assertions[k]
        for k in (
            "task_frame_published",
            "turn_rendered_ws",
            "steer_reflected",
            "done_derived_result_fired",
            "no_done_token_in_frames",
        )
    )
    return assertions, passed


def _write_transcript(
    cap: Capture, engine_name: str, url: str, steps: list[dict[str, Any]], result: str
) -> Path:
    """Redact and write the transcript; return its path."""
    secret_values = _secret_env_values()
    assertions, _ = _compute_assertions(cap)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    transcript = {
        "ts": ts,
        "issue": 22,
        "engine": engine_name,
        "transport": "live-nats",
        "nats_url": url,
        "agent": AGENT_ID,
        "task_id": TASK_ID,
        "steps": steps,
        "assertions": assertions,
        "ws_messages": _redact(cap.ws_messages, secret_values),
        "frames": [
            {
                "seq": c.seq,
                "ts": c.ts,
                "subject": c.subject,
                "frame_type": c.frame_type,
                "payload": _redact(c.payload, secret_values),
            }
            for c in cap.frames
        ],
        "result": result,
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / f"webchat-{ts}.json"
    path.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
    return path


# --- orchestration -----------------------------------------------------------


async def _run() -> int:
    """Boot the roles, drive the proof, write the transcript, return the exit code."""
    url = os.environ.get(ENV_NATS_URL, DEFAULT_NATS_URL)
    logger.info("e2e.run | starting | task_id={} nats_url={}", TASK_ID, url)

    if not _nats_reachable(url):
        logger.error("e2e.run | live NATS bus unreachable; aborting url={}", url)
        return 2

    engine, engine_name = _build_engine()
    if engine is None:
        logger.error("e2e.run | engine unavailable reason={}", engine_name)
        return 2
    logger.info("e2e.run | engine selected kind={}", engine_name)

    cap = Capture()
    steps: list[dict[str, Any]] = []

    edge_bus = NatsFrameBus(url)
    orch_bus_conn: FleetBus | None = None
    agent_bus_conn: FleetBus | None = None
    probe_bus_conn: FleetBus | None = None
    sup: Supervisor | None = None
    run_task: asyncio.Task[None] | None = None
    result = "FAIL"
    exit_code = 1

    try:
        await edge_bus.connect()
        orch_bus_conn = await _connect_fleet_bus(url, "orchestrator")
        agent_bus_conn = await _connect_fleet_bus(url, "agentd-e2e")
        probe_bus_conn = await _connect_fleet_bus(url, "e2e-probe")

        orch_bus = TypedFrameBus(orch_bus_conn)
        agent_bus = TypedFrameBus(agent_bus_conn)
        probe_bus = TypedFrameBus(probe_bus_conn)

        # Capture EVERY lane for TASK_ID before anything is published.
        for subject, handler in _make_capture_handlers(cap).items():
            await probe_bus.subscribe_frame(subject, handler)

        orch = Orchestrator(
            bus=orch_bus,
            config=OrchestratorConfig(single_agent_id=AGENT_ID, stack_id=STACK_ID),
        )
        sup = Supervisor(
            session=engine,
            config=SupervisorConfig(
                agent=AGENT_ID,
                env="dev",
                heartbeat_interval_s=999,
                stall_timeout_s=999,
            ),
            home=HOME_DIR,
            publisher=agent_bus_conn.publish,
            system_prompt=SYSTEM_PROMPT,
            tools_policy=ToolsPolicy(),
        )
        runner = FrameTurnRunner(agent_id=AGENT_ID, supervisor=sup, bus=agent_bus)

        HOME_DIR.mkdir(parents=True, exist_ok=True)
        await orch.start()
        await runner.start()
        await sup.start()
        run_task = asyncio.create_task(sup.run(), name="e2e-supervisor-run")

        app = create_app(
            config=WebchatConfig(nats_url=url, stack=STACK_ID),
            bus=edge_bus,
            id_factory=lambda: TASK_ID,
        )

        # Mirror the WS forward contract on the connected edge bus.
        ws_mirror = _make_ws_mirror(cap)
        await edge_bus.subscribe(wsubj.task_turn(TASK_ID), ws_mirror)
        await edge_bus.subscribe(wsubj.task_status(TASK_ID), ws_mirror)

        # Let subscriptions settle on the server before publishing.
        await asyncio.sleep(SUB_SETTLE_S)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            try:
                await _drive_proof(client, cap, steps)
            except _ProofFailure as exc:
                logger.error("e2e.run | proof failed reason={}", exc)
                steps.append({"step": "proof_failure", "ok": False, "reason": str(exc)})

        assertions, passed = _compute_assertions(cap)
        result = "PASS" if passed else "FAIL"
        exit_code = 0 if passed else 1
        logger.info("e2e.run | assertions={} result={}", assertions, result)
    except Exception as exc:
        logger.error("e2e.run | unexpected failure reason={}", exc)
        steps.append({"step": "unexpected_failure", "ok": False, "reason": str(exc)})
        result = "FAIL"
        exit_code = 1
    finally:
        # The scripted engine completes every turn within its own send,
        # so there is no held-open turn to flush at teardown.
        if sup is not None:
            try:
                await sup.stop()
            except Exception as exc:
                logger.error("e2e.teardown | supervisor stop failed reason={}", exc)
        if run_task is not None:
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error("e2e.teardown | run task join failed reason={}", exc)
        for bus in (orch_bus_conn, agent_bus_conn, probe_bus_conn):
            if bus is None:
                continue
            try:
                await bus.close()
            except Exception as exc:
                logger.error("e2e.teardown | fleet bus close failed reason={}", exc)
        try:
            await edge_bus.close()
        except Exception as exc:
            logger.error("e2e.teardown | edge bus close failed reason={}", exc)

    path = _write_transcript(cap, engine_name, url, steps, result)
    print(str(path))  # noqa: T201 -- the single allowed evidence-path line
    return exit_code


def main() -> int:
    """Entry point: run the async round-trip and return the process exit code."""
    return asyncio.run(_run())


if __name__ == "__main__":
    import sys

    sys.exit(main())
