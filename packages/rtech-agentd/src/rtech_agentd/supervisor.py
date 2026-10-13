"""Own one managed engine session for an agent.

Responsibilities:

* **Start / restart with resume.** A process-local registry rejects duplicate
  live supervisors for the same agent. A per-agent disk record stores the latest
  ``session_ref`` for Agent SDK resume.
* **Heartbeat.** Publishes fleet-core liveness on the fleet heartbeat lane
  ``{env}.agent.heartbeat.<agent_id>`` so rtech agents show up in every fleet
  client's presence view.
* **Stall watchdog.** If an active turn produces no engine event for N seconds,
  it logs and emits a ``status`` envelope (the turn may be wedged) without
  killing the session.

The supervisor tracks turn liveness by observing the session's event stream; the
NATS ingress consumes the SAME stream for replies, so both share one pump owned
here (:meth:`run`), and the ingress registers an event sink.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fleet_nats import Envelope, subjects
from loguru import logger
from rtech_frames import LifecycleFrame, StatusFrame
from rtech_orchestrator import subjects as rtech_subjects

from .engine import EngineError, EngineEvent, EngineSession, ToolsPolicy, TurnDone
from .heartbeat import HeartbeatConfig, HeartbeatPublisher, Sleep
from .lifecycle_policy import RestartPolicy, RotationDecision, RotationPolicy
from .registry import ACTIVE_SESSIONS, ActiveSessionRegistry, DuplicateSessionError, SessionRegistry

__all__ = [
    "CircuitOpenError",
    "DuplicateSessionError",
    "EventSink",
    "Publisher",
    "SessionRegistry",
    "Supervisor",
    "SupervisorConfig",
]


class CircuitOpenError(RuntimeError):
    """The restart circuit opened: a fatal marker :meth:`Supervisor.run` raises.

    The bounded-inner-loop design (issue #84) only stops the *machine* from
    crash-looping if the process actually exits non-zero so systemd's outer
    ``Restart=on-failure`` loop fires. A clean ``run()`` return would let the
    entrypoint mistake a spent restart budget for a normal shutdown. So when the
    circuit opens the supervisor publishes the blocked status + circuit-open
    ``crash`` frame and then :meth:`run` raises THIS — the fatal signal
    ``__main__`` maps to a non-zero ``sys.exit``.

    Slice C dependency: the systemd unit's ``StartLimitIntervalSec`` /
    ``StartLimitBurst`` bound how often that outer relaunch may repeat; without
    them a genuinely wedged engine still restarts forever at the unit level. This
    error is only the inner half of the bound.
    """


# The directive the supervisor sends as the ONE bounded summary turn before it
# rotates a session. It rides the normal ``send`` path; the produced ``TurnDone``
# text becomes the new session's appended context.
ROTATION_SUMMARY_DIRECTIVE = (
    "Your context is being rotated to stay bounded. Produce a concise summary of "
    "this session's durable state — decisions made, work in progress, and any "
    "commitments — so a fresh session can continue without losing the thread. "
    "Reply with the summary only."
)

# An event sink the ingress registers to receive each engine event (for replies).
EventSink = Callable[[EngineEvent], Awaitable[None]]
# Builds a FRESH engine session for a rotation swap (the house injection pattern).
# ``__main__`` wires this from ``engine_select.build_engine``; when unset, rotation
# is disabled because in-place close()+start() on one engine object is racy against
# the real ClaudeEngine (stale reader/queue bleed).
EngineFactory = Callable[[], EngineSession]
# A publisher seam: (subject, envelope) -> None. FleetBus.publish satisfies it;
# tests pass a recording stub.
Publisher = Callable[[str, Envelope], Awaitable[None]]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_id() -> str:
    return uuid4().hex


# The lifecycle transition kinds this supervisor publishes; mirrors
# ``LifecycleFrame.reason`` so callers pick from the same closed set.
# ``requested_restart`` is reserved-and-used: its emitter lives one commit
# up-stack (slice A's RestartPolicy publishes it for a deliberate rotation), so
# it stays part of the closed vocabulary here but is not emitted below.
LifecycleReason = Literal["requested_restart", "crash", "supervisor_recovery", "shutdown"]


@dataclass(frozen=True)
class SupervisorConfig:
    """Tunables for the supervisor loops."""

    agent: str
    env: str
    heartbeat_interval_s: float = 30.0
    stall_timeout_s: float = 90.0
    registry_filename: str = "session-registry.json"

    def __post_init__(self) -> None:
        if not self.agent:
            raise ValueError("agent must be non-empty")
        if not self.env:
            raise ValueError("env must be non-empty")
        if self.heartbeat_interval_s <= 0:
            raise ValueError("heartbeat_interval_s must be positive")
        if self.stall_timeout_s <= 0:
            raise ValueError("stall_timeout_s must be positive")
        if not self.registry_filename:
            raise ValueError("registry_filename must be non-empty")


class Supervisor:
    """Owns the engine session lifecycle, heartbeat, watchdog, and event pump."""

    def __init__(
        self,
        *,
        session: EngineSession,
        config: SupervisorConfig,
        home: Path,
        publisher: Publisher,
        system_prompt: str,
        tools_policy: ToolsPolicy,
        active_registry: ActiveSessionRegistry = ACTIVE_SESSIONS,
        sleep: Sleep = asyncio.sleep,
        restart_policy: RestartPolicy | None = None,
        rotation_policy: RotationPolicy | None = None,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        self._session = session
        self._config = config
        self._home = home
        self._publish = publisher
        self._system_prompt = system_prompt
        self._tools_policy = tools_policy
        self._registry = SessionRegistry(home, config.registry_filename, config.agent)
        self._active_registry = active_registry
        self._sleep = sleep
        # Rotation builds a FRESH engine via this factory and atomically swaps the
        # reference the pump reads (never in-place close()+start() on one object,
        # which bleeds a stale reader/queue on the real ClaudeEngine). When unset,
        # rotation is DISABLED (conservative default) and logged once.
        self._engine_factory = engine_factory
        self._rotation_disabled_logged = False
        # Bounded restart (circuit breaker) + bounded context rotation. Injected
        # like sleep/active_registry; env-tunable defaults when not supplied.
        self._restart_policy = restart_policy or RestartPolicy.from_env()
        self._rotation_policy = rotation_policy or RotationPolicy.from_env()
        self._circuit_open = False
        self._sinks: list[EventSink] = []
        self._last_event_at: float | None = None
        self._active_turn = False
        self._in_flight_task_id: str | None = None
        self._stall_reported = False
        self._pump_task: asyncio.Task[None] | None = None
        self._hb_task: asyncio.Task[None] | None = None
        self._watchdog_task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()
        self._started = False
        self._stop_called = False
        # Rotation state machine. When a summary directive is in flight the NEXT
        # TurnDone is its result (not a task reply); ``_rotation_pending`` carries
        # the trigger reason across those two pump iterations. ``_rotating`` tells
        # the pump a stream-end was a rotation-driven close(), not a crash.
        self._rotation_pending: RotationDecision | None = None
        self._rotation_deadline_task: asyncio.Task[None] | None = None
        self._rotating = False
        # Set when task work lands DURING an in-flight summary turn: rotation may
        # only run when idle, so the arriving task aborts the attempt. The summary
        # turn's TurnDone is still consumed by the rotation state machine (never
        # correlated to the task); we do NOT swap sessions and retry at the next
        # idle TurnDone with the policy's backoff.
        self._rotation_aborted = False

    @property
    def session(self) -> EngineSession:
        return self._session

    @property
    def circuit_open(self) -> bool:
        """True once the restart budget is spent and the supervisor stays down.

        The entrypoint reads this to exit non-zero so systemd's outer
        ``Restart=on-failure`` loop reaps and re-launches the unit with a fresh
        attempt budget.
        """
        return self._circuit_open

    @property
    def in_flight_task_id(self) -> str | None:
        """The task_id of the active turn, or ``None`` when no turn is in flight.

        Registered by ``FrameTurnRunner`` at task send and cleared on the matching
        ``TurnDone``; the stall watchdog maps a wedged turn back to this id.
        """
        return self._in_flight_task_id

    def add_event_sink(self, sink: EventSink) -> None:
        """Register a consumer of engine events (e.g. the NATS reply path)."""
        self._sinks.append(sink)

    def track_in_flight_task(self, task_id: str) -> None:
        """Record the task whose turn is active so recovery can re-emit status.

        If a bounded-context summary turn is in flight (:attr:`_rotation_pending`),
        this arriving task work makes rotation NON-idle: it aborts the rotation
        attempt (:meth:`_abort_pending_rotation`) so sessions are never swapped
        under live task work and the task is served with correct correlation. The
        summary turn's own TurnDone is still consumed by the rotation state machine
        (never mapped to this task); rotation retries at the next idle TurnDone.
        """
        if not task_id:
            raise ValueError("task_id must be non-empty")
        if self._rotation_pending is not None:
            self._abort_pending_rotation()
        self._in_flight_task_id = task_id
        self._active_turn = True
        self._stall_reported = False
        logger.debug(
            "supervisor.track | task active | agent={} task_id={}", self._config.agent, task_id
        )

    def clear_in_flight_task(self, task_id: str) -> None:
        """Clear the tracked task if it matches the completed turn."""
        if self._in_flight_task_id == task_id:
            self._in_flight_task_id = None
            self._active_turn = False
            logger.debug(
                "supervisor.track | task cleared | agent={} task_id={}", self._config.agent, task_id
            )

    async def start(self) -> None:
        """Resume the prior session if one is on disk, else start fresh."""
        self._active_registry.claim(self._config.agent, self)
        try:
            await self._resume_or_start()
        except Exception:
            self._active_registry.release(self._config.agent, self)
            raise
        self._started = True

    async def run(self) -> None:
        """Run the event pump + heartbeat + watchdog until :meth:`stop`.

        Raises:
            CircuitOpenError: When the restart circuit opened (attempt budget
                spent). This is the fatal marker ``__main__`` maps to a non-zero
                ``sys.exit`` so systemd's outer ``Restart=on-failure`` loop reaps
                the unit; a clean operator :meth:`stop` returns normally instead.
        """
        self._pump_task = asyncio.create_task(self._pump(), name="supervisor-pump")
        self._hb_task = asyncio.create_task(self._heartbeat_loop(), name="supervisor-heartbeat")
        self._watchdog_task = asyncio.create_task(self._watchdog_loop(), name="supervisor-watchdog")
        await self._stopped.wait()
        if self._circuit_open:
            raise CircuitOpenError(
                f"restart circuit open for agent {self._config.agent!r}; "
                "attempt budget spent — exiting for the systemd outer loop"
            )

    async def stop(
        self,
        *,
        reason: LifecycleReason = "shutdown",
        initiator: str = "operator",
        correlation_id: str | None = None,
        detail: str = "",
    ) -> None:
        """Signal loops to stop, cancel them, and close the session.

        Publishes a lifecycle frame (issue #84) BEFORE tearing down loops/session
        so the bus is still connected. ``reason``/``initiator`` name the caller's
        intent (the signal path passes ``initiator="signal:<name>"``). The publish
        can never block or fail the shutdown — :meth:`_publish_lifecycle` swallows
        every error.

        Idempotent: a second ``stop()`` (e.g. a shutdown racing a signal handler)
        is a no-op — it publishes no second shutdown frame and re-tears nothing
        down — so exactly one shutdown frame is ever emitted.
        """
        if self._stop_called:
            return
        self._stop_called = True
        await self._publish_lifecycle(
            reason,
            initiator,
            correlation_id or _new_id(),
            detail,
        )
        self._stopped.set()
        self._cancel_rotation_deadline()
        for task in (self._pump_task, self._hb_task, self._watchdog_task):
            if task is not None:
                task.cancel()
        for task in (self._pump_task, self._hb_task, self._watchdog_task):
            if task is None:
                continue
            try:
                await task
            except asyncio.CancelledError:
                pass
        try:
            await self._session.close()
            logger.info("supervisor.stop | stopped | agent={}", self._config.agent)
        finally:
            if self._started:
                self._active_registry.release(self._config.agent, self)
                self._started = False

    # --- loops ----------------------------------------------------------------

    async def _pump(self) -> None:
        """Drain engine events: fan out to sinks + track turn liveness."""
        loop = asyncio.get_running_loop()
        while not self._stopped.is_set() and not self._circuit_open:
            try:
                async for event in self._session.events():
                    self._last_event_at = loop.time()
                    await self._handle_event(event)
                if self._stopped.is_set():
                    return
                if self._rotating:
                    # A rotation close() ended the old stream; the fresh session is
                    # already started. Re-enter events() on it — this is not a crash.
                    self._rotating = False
                    logger.info(
                        "supervisor.pump | rotation stream swap | agent={}", self._config.agent
                    )
                    continue
                logger.warning(
                    "supervisor.pump | event stream ended | agent={}", self._config.agent
                )
                await self._recover_dead_session("stream-ended")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("supervisor.pump | event stream error: {}", exc)
                await self._recover_dead_session("stream-error")
        if self._circuit_open:
            # Circuit open: stop pumping and stay down. The process exits and the
            # outer systemd Restart=on-failure loop reaps the unit — we do not spin
            # a tight restart loop in-process (that was the unbounded-loop bug).
            logger.error(
                "supervisor.pump | circuit open, pump halted | agent={}", self._config.agent
            )

    async def _heartbeat_loop(self) -> None:
        """Publish a liveness envelope at the configured interval."""
        heartbeat = HeartbeatPublisher(
            config=HeartbeatConfig(
                agent_id=self._config.agent,
                env=self._config.env,
                interval_s=self._config.heartbeat_interval_s,
            ),
            publisher=self._publish,
            active_turn=lambda: self._active_turn,
            sleep=self._sleep,
        )
        await heartbeat.run()

    async def _watchdog_loop(self) -> None:
        """Emit a status envelope if an active turn stalls (no event for N s)."""
        loop = asyncio.get_running_loop()
        # Poll at a fraction of the stall window so detection latency stays bounded
        # (floor keeps a fast test-configured window responsive without busy-spin).
        interval = max(0.01, self._config.stall_timeout_s / 3)
        try:
            while True:
                await self._sleep(interval)
                # Time-based healthy reset: a session that stays connected for
                # ``healthy_reset_s`` without needing a restart has proven healthy,
                # so refund the attempt budget even with NO TurnDone (an idle-but-
                # alive session would otherwise carry stale attempts into a much
                # later transient crash and lock out prematurely). Uses the policy's
                # injected monotonic clock.
                self._maybe_reset_healthy_budget()
                if not self._active_turn or self._last_event_at is None or self._stall_reported:
                    continue
                idle = loop.time() - self._last_event_at
                if idle >= self._config.stall_timeout_s:
                    self._stall_reported = True
                    logger.warning(
                        "supervisor.watchdog | stall detected | agent={} idle={:.1f}s",
                        self._config.agent,
                        idle,
                    )
                    await self._emit_stall_status(idle)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("supervisor.watchdog | loop error: {}", exc)

    def _maybe_reset_healthy_budget(self) -> None:
        """Refund the restart budget once ``healthy_reset_s`` connected has elapsed.

        The time-based half of the healthy reset (the other half is the TurnDone
        reset in :meth:`_handle_event`). Skips when the circuit is already open —
        that budget is spent for this unit lifetime and only a fresh systemd
        relaunch refills it.
        """
        if self._circuit_open:
            return
        if self._restart_policy.attempts == 0:
            return
        if self._restart_policy.seconds_since_connect() >= self._restart_policy.healthy_reset_s:
            logger.info(
                "supervisor.watchdog | healthy window elapsed, resetting restart budget | agent={}",
                self._config.agent,
            )
            self._restart_policy.on_healthy()

    async def _emit_stall_status(self, idle: float) -> None:
        task_id = self._in_flight_task_id
        if task_id is None:
            logger.warning(
                "supervisor.watchdog | stall status skipped, no task | agent={} idle={:.1f}s",
                self._config.agent,
                idle,
            )
            return
        frame = StatusFrame(task_id=task_id, agent_id=self._config.agent, state="blocked")
        subject = subjects.agent_status(self._config.env, task_id)
        env = Envelope.new(
            msg_id=_new_id(),
            ts=_now_iso(),
            sender=self._config.agent,
            kind="status",
            task_id=task_id,
            payload=frame.model_dump(mode="json"),
        )
        try:
            await self._publish(subject, env)
        except Exception as exc:
            logger.error("supervisor.watchdog | status publish failed: {}", exc)

    async def _publish_lifecycle(
        self,
        reason: LifecycleReason,
        initiator: str,
        correlation_id: str,
        detail: str = "",
    ) -> None:
        """Publish a lifecycle frame on the per-agent lifecycle lane (issue #84).

        This MUST NEVER block or fail the shutdown/recovery path: any failure to
        build or publish the frame is logged loudly and swallowed so the caller
        proceeds. ``correlation_id`` links a death and its recovery into one pair.
        """
        agent = self._config.agent
        try:
            frame = LifecycleFrame(
                agent_id=agent,
                reason=reason,
                initiator=initiator,
                correlation_id=correlation_id,
                detail=detail,
                ts=datetime.now(UTC),
            )
            subject = rtech_subjects.agent_lifecycle(agent)
            env = Envelope.new(
                msg_id=_new_id(),
                ts=_now_iso(),
                sender=agent,
                kind="lifecycle",
                task_id=self._in_flight_task_id,
                payload=frame.model_dump(mode="json"),
            )
            await self._publish(subject, env)
            logger.info(
                "supervisor.lifecycle | published | agent={} reason={} initiator={} corr={}",
                agent,
                reason,
                initiator,
                correlation_id,
            )
        except Exception as exc:
            logger.error(
                "supervisor.lifecycle | publish failed | agent={} reason={} corr={} error={}",
                agent,
                reason,
                correlation_id,
                exc,
            )

    async def _resume_or_start(self) -> None:
        prior = self._registry.load()
        if prior:
            logger.info("supervisor.start | resuming | agent={} ref={}", self._config.agent, prior)
            try:
                await self._session.resume(prior)
                return
            except Exception as exc:
                logger.warning(
                    "supervisor.start | resume failed, starting fresh | ref={} reason={}",
                    prior,
                    exc,
                )
        logger.info("supervisor.start | fresh session | agent={}", self._config.agent)
        await self._session.start(self._system_prompt, self._tools_policy)

    async def _handle_event(self, event: EngineEvent) -> None:
        # A summary turn is in flight: the NEXT TurnDone is ITS result, not a task
        # reply. Intercept it before normal handling (never fan to sinks, never
        # re-decide rotation, never correlate to a task) and complete the swap.
        # Interim deltas are ignored. If task work landed mid-summary the rotation
        # was aborted: the summary's TurnDone is still consumed HERE (discarded, no
        # swap), then normal task handling resumes on the following turns.
        if self._rotation_pending is not None:
            if isinstance(event, TurnDone):
                if self._rotation_aborted:
                    self._discard_aborted_summary()
                else:
                    await self._complete_rotation(event)
            return
        if isinstance(event, TurnDone):
            self._active_turn = False
            self._stall_reported = False
            self._in_flight_task_id = None
            if event.session_ref:
                try:
                    self._registry.save(event.session_ref)
                except OSError:
                    logger.warning(
                        "supervisor.pump | resume ref not persisted | agent={}",
                        self._config.agent,
                    )
            # A completed turn is proof the session is healthy: reset the restart
            # budget so a later transient crash gets the full attempt allowance.
            self._restart_policy.on_healthy()
        else:
            self._active_turn = True
            self._stall_reported = False
        for sink in self._sinks:
            try:
                await sink(event)
            except Exception as exc:
                logger.error("supervisor.pump | sink error: {}", exc)
        # Rotation is decided ONLY at TurnDone (never mid-turn) and only after the
        # reply has fanned to sinks, so the turn's output lands before any swap.
        if isinstance(event, TurnDone):
            await self._maybe_rotate()

    async def _recover_dead_session(self, detail: str = "") -> None:
        if self._stopped.is_set() or self._circuit_open:
            return
        # One id links the death and its recovery into a correlated pair (#84):
        # the "crash" frame marks the session dying, "supervisor_recovery" marks
        # the supervisor bringing it back. ``detail`` distinguishes stream-ended
        # from stream-error.
        correlation_id = _new_id()
        await self._publish_lifecycle("crash", "supervisor", correlation_id, detail)
        # Bound the restart: exponential backoff, capped attempts, circuit breaker.
        decision = self._restart_policy.on_dead_session()
        if not decision.should_restart:
            await self._open_circuit(detail)
            return
        if decision.backoff_s > 0:
            logger.warning(
                "supervisor.recover | backing off | agent={} attempt={} backoff={:.1f}s",
                self._config.agent,
                decision.attempt,
                decision.backoff_s,
            )
            await self._sleep(decision.backoff_s)
            if self._stopped.is_set():
                return
        try:
            ref = await self._resume_ref()
            if ref:
                await self._session.resume(ref)
                logger.warning(
                    "supervisor.recover | resumed dead session | agent={} ref={} attempt={}",
                    self._config.agent,
                    ref,
                    decision.attempt,
                )
            else:
                await self._session.start(self._system_prompt, self._tools_policy)
                logger.warning(
                    "supervisor.recover | restarted session | agent={} attempt={}",
                    self._config.agent,
                    decision.attempt,
                )
        except Exception:
            # The bounded restart attempt itself failed. Close the crash frame with
            # a matching 'crash'/recovery-failed frame under the SAME correlation id
            # (never an orphan) and re-raise so the pump routes it back through the
            # RestartPolicy for the next bounded attempt (or the circuit).
            await self._publish_lifecycle("crash", "supervisor", correlation_id, "recovery-failed")
            raise
        self._restart_policy.on_connected()
        await self._publish_lifecycle("supervisor_recovery", "supervisor", correlation_id, detail)
        await self._republish_working_status()

    async def _open_circuit(self, detail: str) -> None:
        """Stop retrying: publish blocked status + a circuit-open crash frame.

        After the restart budget is exhausted the supervisor stays down and lets
        the outer systemd ``Restart=on-failure`` loop reap the unit. Any in-flight
        task gets a terminal ``blocked`` status so the orchestrator/human is not
        left waiting on a session that will not come back inside this unit.
        """
        self._circuit_open = True
        logger.error(
            "supervisor.recover | circuit open, staying down | agent={} attempts={}",
            self._config.agent,
            self._restart_policy.attempts,
        )
        await self._emit_blocked_status()
        await self._publish_lifecycle("crash", "supervisor", _new_id(), "circuit_open")
        # Release run() so the entrypoint can exit non-zero and hand off to the
        # outer systemd Restart=on-failure loop; ``circuit_open`` stays True so the
        # exit code reflects the failure (not a clean operator shutdown).
        self._stopped.set()

    async def _emit_blocked_status(self) -> None:
        task_id = self._in_flight_task_id
        if task_id is None:
            return
        frame = StatusFrame(task_id=task_id, agent_id=self._config.agent, state="blocked")
        subject = subjects.agent_status(self._config.env, task_id)
        env = Envelope.new(
            msg_id=_new_id(),
            ts=_now_iso(),
            sender=self._config.agent,
            kind="status",
            task_id=task_id,
            payload=frame.model_dump(mode="json"),
        )
        try:
            await self._publish(subject, env)
        except Exception as exc:
            logger.error("supervisor.recover | blocked status publish failed: {}", exc)

    # --- bounded context rotation --------------------------------------------

    async def _maybe_rotate(self) -> None:
        """At a completed turn, start a bounded rotation if a bound is crossed.

        Consults the rotation policy (turn-count / age). When it fires, it marks
        the rotation in-flight (``_rotation_pending``) and arms a hard deadline
        BEFORE the summary send's await, then sends ONE summary directive down the
        SAME session; the next TurnDone (intercepted in :meth:`_handle_event`)
        completes the swap. The summary turn is a SINGLE attempt — never retried in
        a loop. A send failure rolls the mark back via :meth:`_fail_rotation`,
        preserving the live session.

        IDLE-ONLY: rotation may begin only when no task turn is in flight. The idle
        check below and the ``_rotation_pending`` mark form ONE synchronous section
        (no await between them), so a TaskFrame that lands inside the summary send's
        await window always sees the pending rotation in :meth:`track_in_flight_task`
        and aborts it — the session is never swapped under live work. A task that
        raced in between this TurnDone and the idle check (``_in_flight_task_id`` set
        by :meth:`track_in_flight_task`) is caught by that check and deferred. Either
        way the bound stays crossed, so the next idle TurnDone re-fires — with the
        policy's backoff via :meth:`RotationPolicy.defer`.
        """
        if self._circuit_open or self._rotation_pending is not None:
            return
        if self._engine_factory is None:
            # No factory: rotation is disabled (in-place close()+start() is racy on
            # the real engine). Log ONCE, loudly, then never rotate. The policy
            # counter is left untouched so nothing accumulates unbounded work here
            # beyond what the operator opted into by omitting the factory.
            if not self._rotation_disabled_logged:
                self._rotation_disabled_logged = True
                logger.warning(
                    "supervisor.rotate | rotation DISABLED (no engine_factory injected); "
                    "context will not be bounded by rotation | agent={}",
                    self._config.agent,
                )
            return
        decision = self._rotation_policy.on_turn_done()
        if not decision.should_rotate:
            return
        if self._in_flight_task_id is not None or self._active_turn:
            # Not idle: task work is queued. Defer to the next idle TurnDone rather
            # than rotate under live work (the desync bug).
            logger.info(
                "supervisor.rotate | bound crossed but not idle, deferring | agent={} task_id={}",
                self._config.agent,
                self._in_flight_task_id,
            )
            self._rotation_policy.defer()
            return
        logger.info(
            "supervisor.rotate | rotation triggered | agent={} reason={}",
            self._config.agent,
            decision.reason,
        )
        # Mark the rotation in-flight BEFORE the summary send's await. The idle
        # check above and this mark are one synchronous section: a TaskFrame that
        # lands inside the send() window (below) now always sees ``_rotation_pending``
        # in :meth:`track_in_flight_task` and aborts the attempt. Marking AFTER the
        # send was the race — a task arriving mid-send saw no pending rotation, so
        # rotation proceeded against a no-longer-idle session and swapped it under
        # live work. Arm the deadline here too so the window is bounded.
        self._rotation_pending = decision
        self._rotation_aborted = False
        self._arm_rotation_deadline()
        try:
            await self._session.send(ROTATION_SUMMARY_DIRECTIVE)
        except Exception as exc:
            # Roll the mark back: the live session is untouched, the ref unclobbered.
            # ``_fail_rotation`` clears ``_rotation_pending``/``_rotation_aborted``,
            # cancels the just-armed deadline, and defers the policy to the next turn.
            logger.error("supervisor.rotate | summary send failed: {}", exc)
            await self._fail_rotation(f"summary send failed: {exc}")
            return
        # The send completed. Honor an abort requested DURING the send window: a task
        # that arrived while the directive was in flight set ``_rotation_aborted`` (and
        # deferred the policy). The summary's own TurnDone is discarded in
        # :meth:`_handle_event` — no swap — and the task is served. No new work here;
        # the abort machinery in :meth:`_abort_pending_rotation` already ran.
        if self._rotation_aborted:
            logger.info(
                "supervisor.rotate | task arrived during summary send, abort honored | agent={}",
                self._config.agent,
            )

    def _arm_rotation_deadline(self) -> None:
        """Arm the single hard timeout for the in-flight summary turn."""

        async def _deadline() -> None:
            try:
                await self._sleep(self._rotation_policy.summary_timeout_s)
            except asyncio.CancelledError:
                raise
            if self._rotation_pending is None:
                return
            logger.error(
                "supervisor.rotate | summary timed out after {}s | agent={}",
                self._rotation_policy.summary_timeout_s,
                self._config.agent,
            )
            await self._fail_rotation("rotation_summary_failed")

        self._rotation_deadline_task = asyncio.create_task(
            _deadline(), name="supervisor-rotation-deadline"
        )

    def _abort_pending_rotation(self) -> None:
        """Abort an in-flight summary turn because task work arrived (idle-only).

        Called synchronously from :meth:`track_in_flight_task` the instant a task
        turn is registered while a summary directive is pending. Cancels the hard
        deadline, marks the attempt aborted so the summary's own TurnDone is
        DISCARDED (never correlated to the task) in :meth:`_handle_event`, defers
        the policy so rotation retries at the next idle TurnDone, and schedules a
        best-effort interrupt of the wedged summary turn so the task is served
        promptly. The session is NEVER swapped — the live session and its ref are
        preserved intact.
        """
        self._cancel_rotation_deadline()
        self._rotation_aborted = True
        self._rotation_policy.defer()
        logger.info(
            "supervisor.rotate | task arrived mid-summary, rotation aborted | agent={}",
            self._config.agent,
        )
        # Interrupt the wedged summary turn so the arriving task is served sooner.
        # Fire-and-forget (this entry point is sync); interrupt failures are logged
        # and never block task tracking.
        asyncio.create_task(  # noqa: RUF006
            self._interrupt_aborted_summary(), name="supervisor-rotation-abort-interrupt"
        )

    async def _interrupt_aborted_summary(self) -> None:
        """Best-effort interrupt of the summary turn aborted by arriving task work."""
        try:
            await self._session.interrupt()
        except Exception as exc:
            logger.error("supervisor.rotate | abort interrupt failed: {}", exc)

    def _discard_aborted_summary(self) -> None:
        """Consume the aborted summary turn's TurnDone without swapping sessions.

        The summary turn's terminal TurnDone must be consumed by the rotation state
        machine and NEVER correlated to the arriving task. This clears the rotation
        pending state; task turns that follow flow through normal handling.
        """
        self._rotation_pending = None
        self._rotation_aborted = False
        logger.info(
            "supervisor.rotate | aborted summary TurnDone consumed (no swap) | agent={}",
            self._config.agent,
        )

    async def _complete_rotation(self, summary: TurnDone) -> None:
        """Swap to a fresh session seeded with the summary; persist the new ref.

        Work is never lost: the old session_ref is only clobbered AFTER a fresh
        session is confirmed started. On any failure the rotation aborts with the
        old session/ref intact (:meth:`_fail_rotation`).
        """
        self._cancel_rotation_deadline()
        decision = self._rotation_pending
        self._rotation_pending = None
        self._rotation_aborted = False
        summary_text = summary.text.strip()
        if not summary_text:
            await self._fail_rotation("rotation_summary_failed")
            return
        # A factory is mandatory to reach here: ``_maybe_rotate`` disables rotation
        # when none is injected, so a pending rotation always has one.
        factory = self._engine_factory
        if factory is None:  # pragma: no cover - guarded by _maybe_rotate
            await self._fail_rotation("no engine_factory")
            return
        reason = decision.reason if decision is not None else ""
        fresh_prompt = f"{self._system_prompt}\n\n## Rotated context summary\n\n{summary_text}"
        old_session = self._session
        try:
            # Build a FRESH engine, start it, THEN atomically swap the reference the
            # pump reads. Never close()+start() one object in place (stale reader/
            # queue bleed on the real ClaudeEngine). Close the old engine last so a
            # fresh-start failure leaves the old session live and serving.
            fresh = factory()
            await fresh.start(fresh_prompt, self._tools_policy)
        except Exception as exc:
            logger.error("supervisor.rotate | fresh engine start failed: {}", exc)
            await self._fail_rotation(f"fresh start failed: {exc}")
            return
        # Fresh engine is up: swap the reference (pump re-enters events() on it),
        # tell the pump the old stream-end is a rotation swap, then close the old.
        self._rotating = True
        self._session = fresh
        try:
            await old_session.close()
        except Exception as exc:
            # The fresh engine is already live and swapped in; a failed close of the
            # retired engine is logged but must not fail the rotation.
            logger.error("supervisor.rotate | old engine close failed: {}", exc)
        # Confirmed fresh: NOW persist the new ref (never before this point).
        new_ref = await self._fresh_session_ref()
        if new_ref:
            try:
                self._registry.save(new_ref)
            except OSError:
                logger.warning(
                    "supervisor.rotate | new ref not persisted | agent={}", self._config.agent
                )
        self._rotation_policy.on_rotated()
        self._restart_policy.on_connected()
        logger.info(
            "supervisor.rotate | rotated | agent={} reason={} new_ref={}",
            self._config.agent,
            reason,
            new_ref,
        )
        await self._publish_lifecycle("requested_restart", "rotation", _new_id(), reason)

    async def _fail_rotation(self, detail: str) -> None:
        """Abort an in-flight rotation, preserving the live session and its ref.

        Emits a structured :class:`EngineError`-derived failure to sinks and a
        lifecycle frame, then defers the policy to the NEXT TurnDone (single
        attempt — the summary is NEVER retried immediately in a loop). The old
        session keeps serving turns meanwhile.
        """
        self._cancel_rotation_deadline()
        self._rotation_pending = None
        self._rotation_aborted = False
        self._rotation_policy.defer()
        failure = EngineError(message=f"rotation aborted: {detail}")
        for sink in self._sinks:
            try:
                await sink(failure)
            except Exception as exc:
                logger.error("supervisor.rotate | failure sink error: {}", exc)
        await self._publish_lifecycle("crash", "rotation", _new_id(), "rotation_summary_failed")

    def _cancel_rotation_deadline(self) -> None:
        task = self._rotation_deadline_task
        self._rotation_deadline_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _fresh_session_ref(self) -> str | None:
        """Read the freshly-started session's ref via the health snapshot."""
        try:
            health = await self._session.health()
        except Exception as exc:
            logger.warning(
                "supervisor.rotate | health unavailable | agent={} reason={}",
                self._config.agent,
                exc,
            )
            return None
        return health.session_ref

    async def _resume_ref(self) -> str | None:
        ref = self._registry.load()
        if ref:
            return ref
        try:
            health = await self._session.health()
        except Exception as exc:
            logger.warning(
                "supervisor.recover | health unavailable | agent={} reason={}",
                self._config.agent,
                exc,
            )
            return None
        return health.session_ref

    async def _republish_working_status(self) -> None:
        task_id = self._in_flight_task_id
        if task_id is None:
            logger.debug("supervisor.recover | no tracked task | agent={}", self._config.agent)
            return
        frame = StatusFrame(task_id=task_id, agent_id=self._config.agent, state="working")
        subject = subjects.agent_status(self._config.env, task_id)
        env = Envelope.new(
            msg_id=_new_id(),
            ts=_now_iso(),
            sender=self._config.agent,
            kind="status",
            task_id=task_id,
            payload=frame.model_dump(mode="json"),
        )
        try:
            await self._publish(subject, env)
            logger.info(
                "supervisor.recover | working status republished | agent={} task_id={}",
                self._config.agent,
                task_id,
            )
        except Exception as exc:
            logger.error(
                "supervisor.recover | status publish failed | agent={} task_id={} reason={}",
                self._config.agent,
                task_id,
                exc,
            )
