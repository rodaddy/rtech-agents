"""Run agentd (supervisor + frame-turn runner) as a standalone service.

The local dev stack (``scripts/dev_stack.py``) boots the orchestrator, agentd,
and the web-chat edge together IN-PROCESS. On the k-board LXC each role is its
own systemd unit, so agentd needs a standalone entrypoint that owns exactly the
agentd wiring dev_stack does: select the managed engine, connect a fleet-nats
bus, bridge it to the typed ``FrameBus`` seam, start the :class:`Supervisor` and
:class:`FrameTurnRunner`, run the supervisor pump, and run until SIGINT/SIGTERM.

The engine defaults to the real Claude engine (subscription auth via an
authenticated ``claude`` CLI on PATH); set ``DEV_STACK_ENGINE=scripted`` for the
offline escape hatch. The ``TypedFrameBus`` bridge below is the same one
dev_stack carries; scripts are not importable from one another, so the
runner-owned copy keeps the deploy path free of any ``scripts/`` import. Run it:
``python -m rtech_agentd``. Configuration is read from the environment:

* ``NATS_URL`` -- the fleet NATS server (default the live fleet bus).
* ``RTECH_AGENT_ID`` -- this agent's id (default the collision-safe per-host
  ``agentd-<hostname>``; see :mod:`~rtech_agentd.identity`).
* ``DEV_STACK_ENGINE`` / ``E2E_ENGINE`` -- ``scripted`` for the offline engine.
* ``RTECH_STATE_HOME`` -- the managed session's state dir (session registry,
  engine home). Defaults to ``state/agentd-home`` relative to the working
  directory (the unit's ``WorkingDirectory=%h/rtech-agents`` under the rootless
  deploy), never a root-only ``/opt`` path (issue #55).

Secret env VALUES are never logged. The bus URL is host:port only, not a secret.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fleet_nats import BusConfig, FleetBus
from loguru import logger
from rtech_frames.codec import Frame, decode_frame, encode_frame

from .config import AgentConfig, ConfigError, load_agent_config
from .engine import EngineSession, ToolsPolicy
from .engine_select import (
    ENV_BASE,
    ENV_ENGINE,
    ENV_ENGINE_ALT,
    ENV_MODEL,
    EngineSelectionError,
    build_engine,
)
from .frame_turn import FrameBus, FrameTurnConfig, FrameTurnRunner
from .identity import resolve_agent_id
from .memory import AgentMemoryLane, build_agent_memory_lane
from .persona_glue import build_system_prompt
from .supervisor import CircuitOpenError, Supervisor, SupervisorConfig
from .wake_service import OriginProvider, WakeConfig, WakeService

# Env var naming this stack's id (issue #75). Owned by the orchestrator config; a
# minted wake TaskFrame must be stack-scoped, so agentd reads the same var when a
# triage agent is configured as a waker.
ENV_STACK_ID = "RTECH_STACK_ID"

if TYPE_CHECKING:
    from nats.aio.msg import Msg

# --- constants ---------------------------------------------------------------

DEFAULT_NATS_URL = "nats://127.0.0.1:4222"
ENV_NATS_URL = "NATS_URL"
BUS_ENV = "dev"
SYSTEM_PROMPT = "rtech-agents managed agent"
# Where the managed session's state (session registry, engine home) lives. Under
# the rootless deploy the units set WorkingDirectory=%h/rtech-agents, so the
# default resolves to ~/rtech-agents/state/agentd-home -- a user-owned path, not
# the old root-only /opt tree (issue #55). Override with RTECH_STATE_HOME for a
# non-default layout.
ENV_STATE_HOME = "RTECH_STATE_HOME"
DEFAULT_STATE_HOME = "state/agentd-home"
# Root under which per-agent homes live: agents/<RTECH_AGENT_ID>/agent.yaml (#84).
# Env-tunable so the deploy can point at an install-root agents tree; the default
# is relative to the unit's WorkingDirectory (resolved absolute at load).
ENV_AGENTS_ROOT = "RTECH_AGENTS_ROOT"
DEFAULT_AGENTS_ROOT = "agents"
ENV_AGENT_ID = "RTECH_AGENT_ID"
ENV_EVIDENCE_GATE_ENABLED = "RTECH_EVIDENCE_GATE_ENABLED"


def _resolve_state_home() -> Path:
    """Resolve the agentd state home directory.

    Reads ``RTECH_STATE_HOME`` when set; otherwise defaults to
    ``state/agentd-home`` relative to the current working directory (the unit's
    ``WorkingDirectory``). The path is resolved to an absolute location so a
    relative default binds to the install root rather than a transient CWD.

    Returns:
        The absolute state-home path for the managed session.
    """
    raw = os.getenv(ENV_STATE_HOME, DEFAULT_STATE_HOME)
    return Path(raw).resolve()


def _evidence_gate_enabled() -> bool:
    """Return the production-safe evidence-gate setting (enabled by default)."""
    raw = os.environ.get(ENV_EVIDENCE_GATE_ENABLED, "true").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    logger.warning(
        "agentd.config | invalid {}={!r}; defaulting enabled", ENV_EVIDENCE_GATE_ENABLED, raw
    )
    return True


HEARTBEAT_INTERVAL_S = 30.0
STALL_TIMEOUT_S = 90.0


def _resolve_manifest_path(root: Path, agent_id: str) -> Path | None:
    """Resolve the on-disk ``agent.yaml`` for ``agent_id`` under ``root`` (#84).

    Resolution order (first hit wins), loudly logged:

    1. ``<root>/<agent_id>/agent.yaml`` — an exact-id directory (wins when both
       an exact-id and a slug dir exist).
    2. ``<root>/<slug>/agent.yaml`` where ``slug`` strips a leading ``agentd-``
       prefix — the committed convention (dirs are slug-named ``agents/skippy``
       while the systemd instance id is ``agentd-skippy``).

    Returns:
        The resolved manifest path, or ``None`` when neither candidate exists.
    """
    exact = root / agent_id / "agent.yaml"
    if exact.is_file():
        logger.info("agentd.config | manifest resolved by exact id={} at {}", agent_id, exact)
        return exact
    slug = agent_id.removeprefix("agentd-")
    if slug != agent_id:
        slug_path = root / slug / "agent.yaml"
        if slug_path.is_file():
            logger.info(
                "agentd.config | manifest resolved by slug={} (from id={}) at {}",
                slug,
                agent_id,
                slug_path,
            )
            return slug_path
    return None


def _resolve_agent_config(agent_id: str) -> AgentConfig | None:
    """Resolve this agent's ``agent.yaml`` from ``RTECH_AGENT_ID`` (#84).

    Convention: manifests live under ``<RTECH_AGENTS_ROOT>`` (root env-tunable,
    default ``./agents`` under the unit's WorkingDirectory). Directories are
    SLUG-named (``agents/skippy``) while the systemd instance id is the full
    ``agentd-skippy``; resolution therefore tries the exact id first, then the
    slug with a leading ``agentd-`` stripped (see :func:`_resolve_manifest_path`).
    A MISSING yaml is NOT an error — the entrypoint falls back to env-only
    behavior so nothing that ships today breaks; the fallback is logged loudly.
    A PRESENT-but-broken yaml IS surfaced loudly and also falls back (a
    source-owned deploy defect must be visible, never silently masked).

    Args:
        agent_id: This agent's id (the systemd instance ``%i``).

    Returns:
        The loaded :class:`AgentConfig`, or ``None`` to signal env-only fallback.
    """
    root = Path(os.getenv(ENV_AGENTS_ROOT, DEFAULT_AGENTS_ROOT)).resolve()
    yaml_path = _resolve_manifest_path(root, agent_id)
    if yaml_path is None:
        logger.warning(
            "agentd.config | no agent.yaml for id={} under {} (tried exact-id and "
            "slug dirs) | FALLBACK to env-only config (hardcoded system prompt, "
            "env-selected engine)",
            agent_id,
            root,
        )
        return None
    try:
        config = load_agent_config(yaml_path)
    except ConfigError as exc:
        logger.error(
            "agentd.config | agent.yaml present but invalid | path={} reason={} | "
            "FALLBACK to env-only config",
            yaml_path,
            exc,
        )
        return None
    logger.info("agentd.config | loaded agent.yaml | path={} home={}", yaml_path, config.home)
    return config


def _engine_env_from_config(config: AgentConfig) -> dict[str, str]:
    """Overlay ``build_engine``'s env with this agent's engine selection (#84).

    ``build_engine`` reads the engine kind/model/base from the environment, so we
    thread ``config.engine`` by overlaying those exact keys ON TOP of the process
    environment (config wins where set). Existing env-only behavior is unchanged
    when no config resolves — this overlay is only built when a config is present.

    The manifest is AUTHORITATIVE for the engine kind. ``_selected_kind`` reads
    ``E2E_ENGINE`` (``ENV_ENGINE``) BEFORE ``DEV_STACK_ENGINE`` (``ENV_ENGINE_ALT``),
    so setting only the latter would let a stray ``E2E_ENGINE`` in the process
    env silently override the manifest. We therefore set BOTH spellings to the
    config kind, and loudly log any pre-existing conflicting env var we override.
    """
    env = dict(os.environ)
    for key in (ENV_ENGINE, ENV_ENGINE_ALT):
        prior = env.get(key)
        if prior is not None and prior.strip().lower() != config.engine.kind.strip().lower():
            logger.warning(
                "agentd.config | overriding stray {}={} with manifest engine kind={}",
                key,
                prior,
                config.engine.kind,
            )
        env[key] = config.engine.kind
    env[ENV_MODEL] = config.engine.model
    if config.engine.litellm_base:
        env[ENV_BASE] = config.engine.litellm_base
    return env


def _resolve_system_prompt(config: AgentConfig | None) -> str:
    """Compose the system prompt from the agent's persona, or fall back (#84).

    With a config, the prompt is composed from the persona at ``config.home``
    (SOUL/USER/KERNEL) via :func:`build_system_prompt`. Any persona failure
    (missing home/SOUL, scan finding) is logged loudly and falls back to the
    hardcoded default so a deploy defect is visible but never fatal to wiring.
    """
    if config is None:
        return SYSTEM_PROMPT
    try:
        return build_system_prompt(config.home, model=config.engine.model)
    except Exception as exc:
        logger.error(
            "agentd.persona | build_system_prompt failed | home={} reason={} | "
            "FALLBACK to default system prompt",
            config.home,
            exc,
        )
        return SYSTEM_PROMPT


def _build_memory_lane(agent_id: str) -> AgentMemoryLane | None:
    """Build the per-agent Open Brain recall lane, or ``None`` when disabled.

    Delegates to the shared :func:`~rtech_agentd.memory.build_agent_memory_lane`
    factory (the single owning memory/config boundary) reading the process
    environment. Standalone agentd runs one agent per process, so the factory's
    per-agent suffixed vars fall back to the process-global ``OPENBRAIN_*`` here
    exactly as before — this entrypoint's behavior is unchanged. The token is a
    secret and is NEVER logged (value or length).

    Args:
        agent_id: This agent's id (lane owner + namespace default).

    Returns:
        A wired :class:`~rtech_agentd.memory.AgentMemoryLane`, or ``None`` when
        recall is disabled (no OB token) or the lane could not be built.
    """
    return build_agent_memory_lane(agent_id)


# --- typed frame bus over the live raw transport -----------------------------


class TypedFrameBus:
    """Bridge the typed ``FrameBus`` seam onto a connected ``fleet_nats.FleetBus``.

    The frame-turn runner depends on ``publish_frame`` / ``subscribe_frame``
    (typed), while the live transport carries raw bytes. Every decode and every
    handler invocation is guarded so one bad frame never kills a subscription.
    Same bridge dev_stack carries; scripts are not importable from one another,
    so the runner keeps its own copy.
    """

    def __init__(self, bus: FleetBus) -> None:
        """Wrap an already-connected ``FleetBus``.

        Args:
            bus: A connected fleet-nats bus for this role.
        """
        self._bus = bus

    async def publish_frame(self, subject: str, frame: Frame) -> None:
        """Encode and publish a typed frame on the live bus.

        Args:
            subject: The NATS subject to publish on.
            frame: The typed frame to publish.
        """
        data = encode_frame(frame)
        try:
            self._bus._guard_payload_size(subject, data)  # noqa: SLF001
            await self._bus._nc.publish(subject, data)  # noqa: SLF001
        except Exception as exc:
            logger.error("typedbus.publish | subject={} failed reason={}", subject, exc)
            return
        logger.debug("typedbus.publish | subject={}", subject)

    async def publish_frame_checked(self, subject: str, frame: Frame) -> None:
        """Encode and publish a typed frame, RE-RAISING on transport failure.

        The propagating twin of :meth:`publish_frame` (issue #93): the wake tool
        must not record a false success on a raising NATS publish, so this variant
        does NOT swallow — the exception propagates to the wake service, which
        returns a typed ``publish_failed`` error.
        """
        data = encode_frame(frame)
        self._bus._guard_payload_size(subject, data)  # noqa: SLF001
        await self._bus._nc.publish(subject, data)  # noqa: SLF001
        logger.debug("typedbus.publish_checked | subject={}", subject)

    async def subscribe_frame(
        self, subject: str, handler: Callable[[Frame], Awaitable[None]]
    ) -> Callable[[], Awaitable[None]] | None:
        """Subscribe a typed handler on the live bus.

        Args:
            subject: The NATS subject to subscribe to.
            handler: The async callable invoked with each decoded frame.

        Returns:
            An awaitable retiring this NATS subscription (the frame-turn runner
            calls it when a task's steer lane ends — issue #76), or ``None``
            when the subscribe itself failed.
        """

        async def _on_message(msg: Msg) -> None:
            try:
                frame = decode_frame(msg.data)
            except Exception as exc:
                logger.error("typedbus.decode | subject={} failed reason={}", subject, exc)
                return
            try:
                await handler(frame)
            except Exception as exc:
                logger.error("typedbus.handler | subject={} failed reason={}", subject, exc)

        try:
            sub = await self._bus._nc.subscribe(subject, cb=_on_message)  # noqa: SLF001
        except Exception as exc:
            logger.error("typedbus.subscribe | subject={} failed reason={}", subject, exc)
            return None
        logger.debug("typedbus.subscribe | subject={}", subject)

        async def _unsubscribe() -> None:
            await sub.unsubscribe()

        return _unsubscribe


# --- environment -------------------------------------------------------------


def _resolve_url() -> str:
    """Return the NATS URL from ``NATS_URL`` (default the live fleet bus)."""
    return os.environ.get(ENV_NATS_URL, DEFAULT_NATS_URL)


@dataclass
class _StopSignal:
    """Records which signal requested shutdown, for the lifecycle initiator.

    The signal handler stamps ``name`` (e.g. ``"SIGTERM"``) so the shutdown
    lifecycle frame (#84) can name its initiator ``"signal:<name>"``.
    """

    name: str | None = None


def _install_signal_handlers(
    loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event, stop_signal: _StopSignal
) -> None:
    """Wire SIGINT/SIGTERM to set ``stop_event`` for a clean shutdown."""

    def _request_stop(sig: signal.Signals) -> None:
        stop_signal.name = sig.name
        logger.info("agentd.signal | stop requested | sig={}", sig.name)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop, sig)
        except (NotImplementedError, RuntimeError) as exc:
            logger.warning("agentd.signal | handler unavailable | sig={} reason={}", sig, exc)


def _build_wake_service(
    *,
    agent_id: str,
    bus: FrameBus,
    origin_provider: OriginProvider,
) -> WakeService | None:
    """Build the wake service for a triage (waker) agent, or ``None`` (issue #93).

    ``WakeConfig.from_env`` returns ``None`` when ``RTECH_WAKE_ALLOWED_AGENTS`` is
    unset/empty — that is the signal this agent is NOT a waker and the wake tool
    must not be exposed. When the allow-list IS set, the stack id is REQUIRED
    (a wake frame must be stack-scoped, issue #75); a missing stack while wake is
    requested fails closed loudly rather than minting undeliverable wakes.

    Args:
        agent_id: This agent's id (the triage id / self-wake refusal target).
        bus: The typed-frame publish seam wake frames ride.
        origin_provider: Supplies the active task's ChannelOrigin (the runner's
            ``current_origin``).

    Returns:
        A wired :class:`WakeService`, or ``None`` when this agent is not a waker.
    """
    stack = os.getenv(ENV_STACK_ID, "").strip()
    wake_config = WakeConfig.from_env(triage_agent=agent_id, stack=stack)
    if wake_config is None:
        return None
    return WakeService(config=wake_config, bus=bus, origin_provider=origin_provider)


def _attach_wake(engine: EngineSession, wake_service: WakeService | None) -> None:
    """Attach the wake service to a Claude engine; fail loud on codex (issue #93).

    Wake is a Claude-engine-only capability in this slice (conservative; the codex
    tool plumbing is deliberately deferred and additive later — see
    :mod:`rtech_agentd.codex_engine`). When wake is NOT requested this is a no-op
    for any engine. When wake IS requested but the selected engine is not a
    :class:`~rtech_agentd.claude_engine.ClaudeEngine`, we FAIL LOUD rather than
    silently drop the capability: a triage deploy that set the allow-list expects
    a live wake tool, and a silent drop would look like Nagatha simply choosing
    never to wake anyone.

    Args:
        engine: The engine session to attach to.
        wake_service: The wake service, or ``None`` when this agent is not a waker.

    Raises:
        EngineSelectionError: If wake is requested but the engine cannot host it.
    """
    if wake_service is None:
        return
    from .claude_engine import ClaudeEngine

    if not isinstance(engine, ClaudeEngine):
        raise EngineSelectionError(
            "wake capability is Claude-engine-only in this slice, but the selected "
            f"engine is {type(engine).__name__}. Unset RTECH_WAKE_ALLOWED_AGENTS on "
            "this non-Claude triage deploy, or select the claude engine (issue #93)."
        )
    engine.set_wake_service(wake_service)


async def _run() -> int:
    """Select the engine, connect the bus, run agentd until a stop signal."""
    url = _resolve_url()
    agent_id = resolve_agent_id().agent_id
    logger.info("agentd.run | starting | nats_url={} agent_id={}", url, agent_id)

    # Resolve the per-agent source-owned config (#84). Missing -> env-only
    # fallback (loudly logged), so nothing that ships today breaks.
    config = _resolve_agent_config(agent_id)
    engine_env = _engine_env_from_config(config) if config is not None else None

    try:
        engine, engine_kind = build_engine(engine_env)
    except EngineSelectionError as exc:
        logger.error("agentd.run | engine unavailable | {}", exc)
        return 2
    logger.info("agentd.run | engine selected kind={}", engine_kind)

    # The wake service (issue #93) is wired after the runner exists (it needs the
    # runner's active-origin provider), then attached to BOTH the initial engine
    # and any rotation-fresh engine. Held in a one-slot list so the factory closure
    # below reads the value set later without a forward-reference dance.
    wake_holder: list[WakeService | None] = [None]

    def _engine_factory() -> EngineSession:
        """Build a FRESH engine for a rotation swap (never reuse one object).

        Uses the SAME config-derived env overlay as the initial build so a
        rotation swap keeps the manifest's engine kind/model, not a stray
        process-env selection (#84). A wake-enabled triage agent re-attaches its
        wake service so the rotated session keeps the ``wake_agent`` tool (#93).
        """
        fresh, _kind = build_engine(engine_env)
        _attach_wake(fresh, wake_holder[0])
        return fresh

    system_prompt = _resolve_system_prompt(config)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    stop_signal = _StopSignal()
    _install_signal_handlers(loop, stop_event, stop_signal)

    bus_conn: FleetBus | None = None
    sup: Supervisor | None = None
    run_task: asyncio.Task[None] | None = None
    exit_code = 0
    try:
        bus_conn = FleetBus(BusConfig(url=url, agent_id=agent_id, env=BUS_ENV))
        await bus_conn.connect()
        typed_bus = TypedFrameBus(bus_conn)

        state_home = _resolve_state_home()
        sup = Supervisor(
            session=engine,
            config=SupervisorConfig(
                agent=agent_id,
                env=BUS_ENV,
                heartbeat_interval_s=HEARTBEAT_INTERVAL_S,
                stall_timeout_s=STALL_TIMEOUT_S,
            ),
            home=state_home,
            publisher=bus_conn.publish,
            system_prompt=system_prompt,
            # ToolsPolicy is left at the empty default: AgentConfig has no tools
            # field today, so there is nothing to thread from config yet (#84).
            tools_policy=ToolsPolicy(),
            engine_factory=_engine_factory,
        )
        logger.info("agentd.run | state home={}", state_home)
        state_home.mkdir(parents=True, exist_ok=True)
        memory = _build_memory_lane(agent_id)
        runner = FrameTurnRunner(
            agent_id=agent_id,
            supervisor=sup,
            bus=typed_bus,
            config=FrameTurnConfig(
                state_home=state_home,
                evidence_gate_enabled=_evidence_gate_enabled(),
            ),
            memory=memory,
        )
        # Wake capability (issue #93): only a triage agent whose deploy sets
        # RTECH_WAKE_ALLOWED_AGENTS becomes a waker. The service reads the runner's
        # active-origin so a wake carries the triggering task's channel origin, and
        # is attached to the initial engine BEFORE start() so the tool registers.
        wake_holder[0] = _build_wake_service(
            agent_id=agent_id,
            bus=typed_bus,
            origin_provider=runner.current_origin,
        )
        _attach_wake(engine, wake_holder[0])
        if wake_holder[0] is not None:
            # Wiring-level confirmation this agentd came up AS A WAKER (issue #93).
            # Guarded on a non-None service so THIS line stays silent for a non-waker
            # agent; when RTECH_WAKE_ALLOWED_AGENTS is absent WakeConfig.from_env still
            # logs its own "wake capability NOT exposed" info line (pre-existing).
            logger.info(
                "agentd.wake | waker mode active | agent={} targets={}",
                agent_id,
                sorted(wake_holder[0].config.allowed_agents),
            )
        await runner.start()
        await sup.start()
        run_task = asyncio.create_task(sup.run(), name="agentd-supervisor-run")
        logger.info("agentd.run | agentd started; awaiting tasks")

        # Wake on EITHER a stop signal OR the supervisor run loop ending. The run
        # loop ends cleanly on operator stop, but raises CircuitOpenError when the
        # restart budget is spent — that fatal marker must exit the process
        # non-zero so systemd's Restart=on-failure relaunches the unit with a fresh
        # budget. Racing both means a circuit-open death is reaped even without a
        # signal (the latent hang: nothing set the signal's stop_event on it).
        signal_wait = asyncio.create_task(stop_event.wait(), name="agentd-stop-wait")
        done, _pending = await asyncio.wait(
            {signal_wait, run_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if not signal_wait.done():
            signal_wait.cancel()
        logger.info("agentd.run | shutdown starting")
        if run_task in done:
            # The run loop ended on its own. A CircuitOpenError is the fatal
            # circuit-open marker -> non-zero exit; a clean return means the pump
            # stopped without a spent budget (nothing to escalate).
            try:
                run_task.result()
            except CircuitOpenError as exc:
                logger.error("agentd.run | restart circuit open; exiting non-zero: {}", exc)
                exit_code = 1
            run_task = None
    except Exception as exc:
        logger.error("agentd.run | unexpected failure reason={}", exc)
        exit_code = 1
    finally:
        if sup is not None:
            initiator = f"signal:{stop_signal.name}" if stop_signal.name else "operator"
            try:
                await sup.stop(reason="shutdown", initiator=initiator)
            except Exception as exc:
                logger.error("agentd.teardown | supervisor stop failed reason={}", exc)
        if run_task is not None:
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error("agentd.teardown | run task join failed reason={}", exc)
        if bus_conn is not None:
            try:
                await bus_conn.close()
            except Exception as exc:
                logger.error("agentd.teardown | bus close failed reason={}", exc)
        logger.info("agentd.run | shutdown complete exit_code={}", exit_code)
    return exit_code


def main() -> int:
    """Entry point: run the agentd event loop and return the exit code."""
    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(main())
