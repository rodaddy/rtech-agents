"""Boot the whole bus-first stack long-running on this Mac against live NATS.

One command stands up the bus-first roles IN-PROCESS against the live fleet
NATS server and serves a chat edge until Ctrl-C:

* **Orchestrator** -- routes single-scope tasks to agentd, fans duo/trio tasks
  verbatim to every participant, and derives done from the typed frame graph.
* **agentd(s)** -- ``Supervisor`` + ``FrameTurnRunner`` per agent, each owning
  ONE managed engine session. Default: one agent on the env-selected engine.
  ``DEV_STACK_AGENTS`` boots several (the trio smoke: nimbus on the
  Claude engine, birk+sprocket on Codex).
* **Edge** -- ``DEV_STACK_EDGE=webchat`` (default) serves the FastAPI web-chat
  app exactly as before. ``DEV_STACK_EDGE=tg`` boots the Telegram edge instead:
  long-poll adapter + multi-identity sender + intake sink wired
  for the trio debate render (👀 receipt, instant unvoiced ack, per-agent
  typing, per-agent TLDR + topic/full).

Telegram mode env (all refs, never literal tokens; values never logged):
  RTECH_CHATD_TOKEN_REF        edge polling bot ref (default env:RTECH_CHATD_BOT_TOKEN)
  RTECH_TG_IDENTITY_MAP        agent=token_ref pairs for the per-identity pool
  RTECH_TG_ALLOWED_GROUPS      transport group allowlist (canonical ids)
  RTECH_CHATD_INTAKE_*         intake hit-gate config
  RTECH_CHATD_DEBATE_PARTICIPANTS  ordered trio for the debate fanout
  DEV_STACK_AGENTS             agent_id=engine pairs, e.g.
                               agentd-nimbus-dev=claude,agentd-birk-dev=codex

Run it: ``uv run python scripts/dev_stack.py``. Web-chat mode serves
``http://127.0.0.1:3800``. Secret env VALUES are never logged. Ctrl-C shuts
down cleanly.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from fleet_nats import BusConfig, FleetBus
from loguru import logger
from rtech_agentd.engine import EngineSession, ToolsPolicy
from rtech_agentd.engine_select import EngineSelectionError, build_engine
from rtech_agentd.frame_turn import FrameTurnConfig, FrameTurnRunner
from rtech_agentd.memory import build_agent_memory_lane
from rtech_agentd.supervisor import Supervisor, SupervisorConfig
from rtech_chatd.bus import NatsFrameBus as ChatdNatsFrameBus
from rtech_chatd.channel import SendTarget
from rtech_chatd.config import (
    ENV_TG_IDENTITY_MAP,
    GateConfig,
    IntakeConfig,
    TelegramConfig,
    parse_identity_map,
    resolve_token,
    validate_token_ref,
)
from rtech_chatd.edge import (
    DiscordEdge,
    debate_chat_lanes_from_triples,
    debate_lanes_from_pairs,
)
from rtech_chatd.intake_sink import IntakeSink
from rtech_chatd.telegram_adapter import TelegramChannelAdapter
from rtech_chatd.telegram_sender import TelegramSender
from rtech_frames import validate_participants
from rtech_orchestrator import Orchestrator, OrchestratorConfig
from rtech_orchestrator.typedbus import TypedFrameBus
from rtech_webchat.app import create_app
from rtech_webchat.bus import NatsFrameBus
from rtech_webchat.config import WebchatConfig

# --- constants ---------------------------------------------------------------

DEFAULT_NATS_URL = "nats://127.0.0.1:4222"
ENV_NATS_URL = "NATS_URL"
DEFAULT_PORT = 3800
ENV_PORT = "DEV_STACK_PORT"
ENV_EVIDENCE_GATE_ENABLED = "RTECH_EVIDENCE_GATE_ENABLED"
HOST = "127.0.0.1"

ENV_EDGE = "DEV_STACK_EDGE"
EDGE_WEBCHAT = "webchat"
EDGE_TELEGRAM = "tg"
ENV_AGENTS = "DEV_STACK_AGENTS"
ENV_EDGE_TOKEN_REF = "RTECH_CHATD_TOKEN_REF"
DEFAULT_EDGE_TOKEN_REF = "env:RTECH_CHATD_BOT_TOKEN"

# A per-host dev id so the local stack never collides with a deployed agentd on
# the shared fleet bus. The orchestrator target is set to the SAME id
# below, so the dev stack is self-contained: it answers only its own tasks.
# Lowercased: BusConfig.agent_id must be a canonical token, and macOS hostnames carry capitals.
HOSTNAME = socket.gethostname().split(".", 1)[0].lower() or "unknown"
AGENT_ID = f"agentd-{HOSTNAME}-dev"
# A per-host stack id so the orchestrator's ingress guard only
# consumes tasks stamped by THIS stack's edge — never a deployed edge's.
STACK_ID = f"{HOSTNAME}-dev"
SYSTEM_PROMPT = "rtech-agents local dev stack"
HOME_ROOT = Path(__file__).resolve().parent.parent / ".dev-stack"
# The legacy single-agent state home. Before ``DEV_STACK_AGENTS`` existed the
# lone dev agent persisted its session registry + completed-task dedup here
# (``.dev-stack/home``). Keep using it verbatim when the roster is IMPLICIT (env
# unset) so an existing dev checkout's durable dedup is not orphaned by moving to
# a per-id home. An EXPLICIT multi-agent roster instead isolates every agent
# under ``.dev-stack/<agent_id>`` so two supervisors never share one state home.
LEGACY_HOME_DIR = HOME_ROOT / "home"

REACHABILITY_TIMEOUT_S = 2.0
HEARTBEAT_INTERVAL_S = 30.0
STALL_TIMEOUT_S = 90.0


# --- environment + reachability ----------------------------------------------


def _resolve_url() -> str:
    """Return the NATS URL from ``NATS_URL`` (default the live fleet bus)."""
    return os.environ.get(ENV_NATS_URL, DEFAULT_NATS_URL)


def _resolve_port() -> int:
    """Return the edge port from ``DEV_STACK_PORT`` (default 3800)."""
    raw = os.environ.get(ENV_PORT)
    if raw is None:
        return DEFAULT_PORT
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "dev_stack.port | invalid {}={!r}, using default {}", ENV_PORT, raw, DEFAULT_PORT
        )
        return DEFAULT_PORT


def _resolve_edge_kind() -> str:
    """Return the selected edge kind, failing loud on an unknown value."""
    raw = os.environ.get(ENV_EDGE, EDGE_WEBCHAT).strip().lower() or EDGE_WEBCHAT
    if raw not in (EDGE_WEBCHAT, EDGE_TELEGRAM):
        raise ValueError(f"{ENV_EDGE} must be '{EDGE_WEBCHAT}' or '{EDGE_TELEGRAM}', got {raw!r}")
    return raw


@dataclass(frozen=True)
class AgentSpec:
    """One agentd to boot: typed id, engine kind, model, and its state home.

    ``home`` is the per-agent state home (session registry + completed-task
    dedup). It is the LEGACY ``.dev-stack/home`` for the implicit single-agent
    default (env unset) and an isolated ``.dev-stack/<agent_id>`` for every
    member of an explicit ``DEV_STACK_AGENTS`` roster.
    """

    agent_id: str
    engine_kind: str | None
    engine_model: str | None = None
    home: Path = LEGACY_HOME_DIR


def _agent_specs() -> list[AgentSpec]:
    """Parse ``DEV_STACK_AGENTS`` into the agent roster (default: one, as before).

    Shape: comma-separated ``agent_id=engine[:model]`` entries (engine:
    ``claude`` / ``codex`` / ``scripted``; model optional — needed when two
    agents share an engine kind but not a model, since the ``E2E_MODEL`` env
    override is process-global). Unset keeps today's single agent on the
    env-selected engine. Set-but-unusable fails loud — a half-parsed roster
    would boot a debate with missing voices.
    """
    raw = os.environ.get(ENV_AGENTS)
    if raw is None:
        # Legacy default: ONE agent on the env-selected engine, persisting to the
        # pre-roster ``.dev-stack/home`` so an existing checkout's durable dedup
        # survives (finding #4). AGENT_ID is host-derived and already subject-safe.
        return [AgentSpec(agent_id=AGENT_ID, engine_kind=None, home=LEGACY_HOME_DIR)]
    specs: list[AgentSpec] = []
    for token in (tok.strip() for tok in raw.split(",") if tok.strip()):
        agent_id, sep, engine_spec = token.partition("=")
        agent_id = agent_id.strip()
        engine, _, model = engine_spec.strip().partition(":")
        engine = engine.strip().lower()
        model = model.strip()
        if not sep or not agent_id or not engine:
            raise ValueError(
                f"{ENV_AGENTS} entries must be 'agent_id=engine[:model]'; got {token!r}"
            )
        # Every roster id becomes a NATS subject token (``rtech.agent.<id>.task``)
        # AND a ``.dev-stack/<id>`` state home. An id with a ``.``/``*``/``>``/
        # whitespace would build a wildcard or malformed subject (and could target
        # a wrong/undeliverable lane) yet slip past the empty-id check, so it must
        # fail CLOSED here at config parse — before any engine build or NATS
        # connection — using the one shared wire-safety check. The rejected id is
        # kept in the sanitized message (it is an operator-supplied roster id, not
        # a secret); the underlying validator withholds nothing sensitive.
        try:
            validate_participants([agent_id], "single")
        except ValueError as exc:
            raise ValueError(
                f"{ENV_AGENTS} agent id {agent_id!r} is not a safe NATS subject token: {exc}"
            ) from exc
        if any(spec.agent_id == agent_id for spec in specs):
            # Two supervisors on one agent id would share a NATS identity and
            # a .dev-stack home, racing on the same persisted session state.
            raise ValueError(f"{ENV_AGENTS} lists agent id {agent_id!r} twice")
        specs.append(
            AgentSpec(
                agent_id=agent_id,
                engine_kind=engine,
                engine_model=model or None,
                # Explicit roster: isolate every agent under its own id-scoped home
                # so two supervisors never race one state home (finding #4).
                home=HOME_ROOT / agent_id,
            )
        )
    if not specs:
        raise ValueError(f"{ENV_AGENTS} is set but contains no usable 'agent_id=engine' pair")
    return specs


def _evidence_gate_enabled() -> bool:
    """Return the dev-stack evidence-gate setting (enabled by default)."""
    return os.environ.get(ENV_EVIDENCE_GATE_ENABLED, "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


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


async def _close_fleet_bus(bus: FleetBus | None) -> None:
    """Close a fleet bus, logging (never raising) on failure."""
    if bus is None:
        return
    try:
        await bus.close()
    except Exception as exc:
        logger.error("dev_stack.teardown | fleet bus close failed reason={}", exc)


# --- signal handling ---------------------------------------------------------


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event) -> None:
    """Wire SIGINT/SIGTERM to set ``stop_event`` for a clean Ctrl-C shutdown."""

    def _request_stop() -> None:
        logger.info("dev_stack.signal | stop requested")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except (NotImplementedError, RuntimeError) as exc:
            logger.warning("dev_stack.signal | handler unavailable | sig={} reason={}", sig, exc)


# --- agent boot ----------------------------------------------------------------


@dataclass
class RunningAgent:
    """One booted agentd's live resources, for teardown."""

    agent_id: str
    bus: FleetBus
    supervisor: Supervisor
    run_task: asyncio.Task[None]


def _build_agent_engine(spec: AgentSpec) -> EngineSession:
    """Build one agent's engine from its spec (env-selected when unpinned).

    Called for EVERY spec up front, BEFORE any NATS connection or orchestrator
    start, so a missing CLI fails fast with exit 2 and zero side effects —
    the original single-agent contract.
    """
    engine_env = dict(os.environ)
    if spec.engine_kind is not None:
        # Set BOTH selection vars: build_engine prefers E2E_ENGINE, so an
        # ambient E2E_ENGINE in the parent env would otherwise silently
        # override every pinned spec and boot a single-engine "debate".
        engine_env["E2E_ENGINE"] = spec.engine_kind
        engine_env["DEV_STACK_ENGINE"] = spec.engine_kind
    if spec.engine_model is not None:
        engine_env["E2E_MODEL"] = spec.engine_model
    engine, engine_kind = build_engine(engine_env)
    logger.info("dev_stack.agent | engine selected | agent={} kind={}", spec.agent_id, engine_kind)
    return engine


async def _boot_agent(url: str, spec: AgentSpec, engine: EngineSession) -> RunningAgent:
    """Connect and start one agentd (supervisor + runner) on a prebuilt engine."""
    bus = await _connect_fleet_bus(url, spec.agent_id)
    typed_bus = TypedFrameBus(bus)
    # Legacy ``.dev-stack/home`` for the implicit single agent, isolated
    # ``.dev-stack/<agent_id>`` for an explicit roster (finding #4) — resolved on
    # the spec at parse time.
    home = spec.home
    home.mkdir(parents=True, exist_ok=True)
    sup = Supervisor(
        session=engine,
        config=SupervisorConfig(
            agent=spec.agent_id,
            env="dev",
            heartbeat_interval_s=HEARTBEAT_INTERVAL_S,
            stall_timeout_s=STALL_TIMEOUT_S,
        ),
        home=home,
        publisher=bus.publish,
        system_prompt=SYSTEM_PROMPT,
        tools_policy=ToolsPolicy(),
    )
    # Per-agent OB recall lane (Phase 4): the dev stack is ONE process, so a
    # token-bound agent must resolve its OWN token/namespace via the per-agent
    # suffixed env vars (OPENBRAIN_TOKEN_<SUFFIX> derived from the agent id)
    # rather than a single process-global OPENBRAIN_TOKEN — the shared factory
    # does that derivation and falls back to the global vars for a single-agent
    # dev run. A missing token disables recall for THIS agent only (warns, None).
    # Token values/lengths are never logged.
    memory = build_agent_memory_lane(spec.agent_id)
    runner = FrameTurnRunner(
        agent_id=spec.agent_id,
        supervisor=sup,
        bus=typed_bus,
        config=FrameTurnConfig(evidence_gate_enabled=_evidence_gate_enabled()),
        memory=memory,
    )
    await runner.start()
    await sup.start()
    run_task = asyncio.create_task(sup.run(), name=f"dev-stack-supervisor-{spec.agent_id}")
    return RunningAgent(agent_id=spec.agent_id, bus=bus, supervisor=sup, run_task=run_task)


async def _teardown_agent(agent: RunningAgent) -> None:
    """Stop one agentd, containing (never raising) each failure."""
    try:
        await agent.supervisor.stop()
    except Exception as exc:
        logger.error(
            "dev_stack.teardown | supervisor stop failed agent={} reason={}",
            agent.agent_id,
            exc,
        )
    agent.run_task.cancel()
    try:
        await agent.run_task
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.error(
            "dev_stack.teardown | run task join failed agent={} reason={}", agent.agent_id, exc
        )
    await _close_fleet_bus(agent.bus)


# --- telegram edge -------------------------------------------------------------


@dataclass
class TelegramEdgeRuntime:
    """The Telegram edge's live resources, for run + teardown."""

    adapter: TelegramChannelAdapter
    edge: DiscordEdge
    bus: ChatdNatsFrameBus


@dataclass(frozen=True)
class TelegramEdgeConfig:
    """Hold every pure Telegram edge setting needed before stack startup.

    The dev stack starts its orchestrator and managed agents before attaching an
    edge. Telegram configuration must therefore be entirely parsed and checked
    first: an invalid roster or transport option cannot be allowed to leave
    background NATS roles running without a usable edge.
    """

    edge_token_ref: str
    identity_refs: dict[str, str]
    intake_config: IntakeConfig
    telegram_config: TelegramConfig

    @classmethod
    def from_env(cls) -> TelegramEdgeConfig:
        """Build and validate Telegram settings without resolving credentials."""

        edge_token_ref = os.environ.get(ENV_EDGE_TOKEN_REF, DEFAULT_EDGE_TOKEN_REF)
        # Fail closed — WITHOUT echoing — if the configured "ref" is actually a
        # literal token (validate_token_ref withholds the value). Only after this
        # is it safe to interpolate the ref into an eventual missing-token error.
        validate_token_ref(edge_token_ref)
        identity_refs = parse_identity_map(os.environ.get(ENV_TG_IDENTITY_MAP))
        intake_config = IntakeConfig.from_env()
        missing_identities = sorted(set(intake_config.debate_participants) - identity_refs.keys())
        if missing_identities:
            raise ValueError(
                "RTECH_CHATD_DEBATE_PARTICIPANTS contains participant ids without a "
                f"matching {ENV_TG_IDENTITY_MAP} entry: {missing_identities!r}; every "
                "debate voice needs its own configured Telegram identity"
            )
        return cls(
            edge_token_ref=edge_token_ref,
            identity_refs=identity_refs,
            intake_config=intake_config,
            telegram_config=TelegramConfig.from_env(),
        )


def _build_telegram_edge(config: TelegramEdgeConfig, url: str) -> TelegramEdgeRuntime:
    """Assemble the Telegram trio edge from validated settings.

    Mirrors ``rtech_chatd.service`` composition with the identity layer:
    the intake sink's ack and typing hooks late-bind to the adapter through a
    cell, the sender's per-identity pool resolves token REFS at first use, and
    the edge renders debate turns identity-stamped (``identity_send=True``,
    ``attributed=False`` — the posting bot is the attribution).
    """

    def _edge_token() -> str:
        token = resolve_token(config.edge_token_ref)
        if not token:
            raise RuntimeError(
                f"dev_stack: edge bot token not provisioned (ref={config.edge_token_ref})"
            )
        return token

    def _identity_provider(ref: str) -> str:
        token = resolve_token(ref)
        if not token:
            raise RuntimeError("dev_stack: identity bot token not provisioned (ref withheld)")
        return token

    sender = TelegramSender(
        identity_token_providers={
            agent: (lambda ref=ref: _identity_provider(ref))
            for agent, ref in config.identity_refs.items()
        },
    )

    gate_config = GateConfig(require_mention=True)
    bus = ChatdNatsFrameBus(url)

    adapter_cell: list[TelegramChannelAdapter] = []

    async def _ack_send(target: SendTarget, text: str) -> str | None:
        if not adapter_cell:
            return None
        return await adapter_cell[0].send(target, text, [])

    async def _typing_start(target: SendTarget) -> None:
        if adapter_cell:
            await adapter_cell[0].typing(target, True)

    intake_sink = (
        IntakeSink(
            bus=bus,
            intake_config=config.intake_config,
            gate_config=gate_config,
            stack=STACK_ID,
            ack_send=_ack_send,
            typing_start=_typing_start,
            # Identity routing owns the voice: the ack posts under the triage
            # agent's own bot, unprefixed.
            voiced=False,
        )
        if config.intake_config.enabled
        else None
    )
    adapter = TelegramChannelAdapter(
        token_provider=_edge_token,
        gate_config=gate_config,
        telegram_config=config.telegram_config,
        intake_config=config.intake_config,
        intake_sink=intake_sink,
        sender=sender,
    )
    adapter_cell.append(adapter)
    edge = DiscordEdge(
        adapter,
        bus,
        stack=STACK_ID,
        origin_edge="telegram",
        attributed=False,
        intake_enabled=config.intake_config.enabled,
        identity_send=True,
        debate_lanes=debate_lanes_from_pairs(config.intake_config.debate_lanes),
        debate_chat_lanes=debate_chat_lanes_from_triples(config.intake_config.debate_chat_lanes),
    )
    return TelegramEdgeRuntime(adapter=adapter, edge=edge, bus=bus)


# --- boot ---------------------------------------------------------------------


async def _run() -> int:
    """Boot the roles, serve the edge until Ctrl-C, then tear down. Return code."""
    # Fail-fast configuration + engine selection: everything here raises/exits
    # BEFORE any NATS connection or orchestrator start, preserving the original
    # exit-2-with-zero-side-effects contract for a missing CLI or bad env.
    try:
        url = _resolve_url()
        port = _resolve_port()
        edge_kind = _resolve_edge_kind()
        telegram_config = TelegramEdgeConfig.from_env() if edge_kind == EDGE_TELEGRAM else None
        specs = _agent_specs()
        engines = [_build_agent_engine(spec) for spec in specs]
    except (EngineSelectionError, ValueError) as exc:
        logger.error("dev_stack.run | configuration error | {}", exc)
        return 2
    logger.info(
        "dev_stack.run | starting | nats_url={} edge={} agents={}",
        url,
        edge_kind,
        [s.agent_id for s in specs],
    )

    if not _nats_reachable(url):
        logger.error("dev_stack.run | live NATS bus unreachable; aborting url={}", url)
        return 2

    orch_bus_conn: FleetBus | None = None
    agents: list[RunningAgent] = []
    serve_task: asyncio.Task[None] | None = None
    server: uvicorn.Server | None = None
    tg: TelegramEdgeRuntime | None = None
    tg_tasks: list[asyncio.Task[None]] = []
    exit_code = 0

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    _install_signal_handlers(loop, stop_event)

    try:
        orch_bus_conn = await _connect_fleet_bus(url, "orchestrator")
        orch = Orchestrator(
            bus=TypedFrameBus(orch_bus_conn),
            config=OrchestratorConfig(single_agent_id=specs[0].agent_id, stack_id=STACK_ID),
        )
        await orch.start()

        for spec, engine in zip(specs, engines, strict=True):
            agents.append(await _boot_agent(url, spec, engine))

        if edge_kind == EDGE_TELEGRAM:
            if telegram_config is None:  # pragma: no cover - selected edge always preflights above.
                raise RuntimeError(
                    "dev_stack: Telegram edge selected without validated configuration"
                )
            tg = _build_telegram_edge(telegram_config, url)
            await tg.bus.connect()
            tg_tasks = [
                asyncio.create_task(tg.adapter.start(), name="dev-stack-tg-adapter"),
                asyncio.create_task(tg.edge.run_inbound(), name="dev-stack-tg-inbound"),
            ]
            # A dead edge must surface, not hang the stack: the first edge task
            # to finish (either one — both are run-forever) requests stop.
            for task in tg_tasks:
                task.add_done_callback(lambda _t: stop_event.set())
            logger.info("dev_stack.run | telegram edge polling | stack={}", STACK_ID)
        else:
            # The edge owns its own NatsFrameBus connection via the app
            # lifespan; do NOT pre-connect it here.
            edge_bus = NatsFrameBus(url)
            app = create_app(
                config=WebchatConfig(nats_url=url, host=HOST, port=port, stack=STACK_ID),
                bus=edge_bus,
                id_factory=None,
            )
            server = uvicorn.Server(
                uvicorn.Config(app, host=HOST, port=port, log_level="info", lifespan="on")
            )
            serve_task = asyncio.create_task(server.serve(), name="dev-stack-uvicorn")
            logger.info("dev_stack.run | webchat serving | url=http://{}:{}", HOST, port)

        await stop_event.wait()
        logger.info("dev_stack.run | shutdown starting")
    except Exception as exc:
        logger.error("dev_stack.run | unexpected failure reason={}", exc)
        exit_code = 1
    finally:
        if server is not None:
            server.should_exit = True
        if serve_task is not None:
            try:
                await serve_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error("dev_stack.teardown | serve task join failed reason={}", exc)
        if tg is not None:
            try:
                await tg.edge.close()
            except Exception as exc:
                logger.error("dev_stack.teardown | tg edge close failed reason={}", exc)
            try:
                await tg.adapter.close()
            except Exception as exc:
                logger.error("dev_stack.teardown | tg adapter close failed reason={}", exc)
            for task in tg_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    logger.error("dev_stack.teardown | tg task join failed reason={}", exc)
            try:
                await tg.bus.close()
            except Exception as exc:
                logger.error("dev_stack.teardown | tg bus close failed reason={}", exc)
        for agent in agents:
            await _teardown_agent(agent)
        await _close_fleet_bus(orch_bus_conn)
        logger.info("dev_stack.run | shutdown complete exit_code={}", exit_code)

    return exit_code


def main() -> int:
    """Entry point: run the dev stack event loop and return the process exit code."""
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(main())
