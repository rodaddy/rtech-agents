"""Bound the supervisor's restart and context-rotation loops.

Two supervisor collaborators, injected like ``sleep``/``active_registry`` so the
supervisor owns *when* to act and these own *whether* and *how much*:

* :class:`RestartPolicy` bounds :meth:`Supervisor._recover_dead_session`. Today
  that loop restarts a dead session with zero backoff and no attempt cap — a
  crash-looping engine spins forever. This policy makes it a bounded, backing-off
  circuit breaker: after ``max_attempts`` the circuit *opens* and the supervisor
  STOPS retrying, staying down for the outer systemd ``Restart=on-failure`` loop
  to reap. The attempt counter resets after a healthy period (a successful
  ``TurnDone`` or ``healthy_reset_s`` connected), so a transient blip does not
  burn the budget.

* :class:`RotationPolicy` bounds *context* growth. A session that never rotates
  retains its whole transcript forever (the Hermes disease: unbounded retained
  history re-triggering the same work). This policy fires a bounded rotation at a
  turn or age threshold — checked only at ``TurnDone``, never mid-turn — so the
  supervisor can summarize, close, and start fresh with a compact prompt.

Both are pure logic + injected clock: no I/O, no engine, no bus. The supervisor
performs every side effect; these only decide.

systemd interaction: the unit sets ``Restart=on-failure`` (the outer loop). When
:class:`RestartPolicy` opens its circuit the supervisor publishes a blocked
status + a ``crash`` lifecycle frame (``detail="circuit_open"``) and stays down;
the process then exits non-zero and systemd restarts the *unit*, which brings up
a fresh supervisor with a fresh attempt budget. We do NOT fight that outer loop —
the in-process budget bounds a single unit lifetime; systemd bounds the machine.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger

__all__ = [
    "RestartDecision",
    "RestartPolicy",
    "RotationDecision",
    "RotationPolicy",
]

# --- env keys ----------------------------------------------------------------

ENV_RESTART_MAX_ATTEMPTS = "RTECH_RESTART_MAX_ATTEMPTS"
ENV_RESTART_BACKOFF_BASE_S = "RTECH_RESTART_BACKOFF_BASE_S"
ENV_RESTART_BACKOFF_CAP_S = "RTECH_RESTART_BACKOFF_CAP_S"
ENV_RESTART_HEALTHY_RESET_S = "RTECH_RESTART_HEALTHY_RESET_S"

ENV_ROTATE_MAX_TURNS = "RTECH_ROTATE_MAX_TURNS"
ENV_ROTATE_MAX_AGE_S = "RTECH_ROTATE_MAX_AGE_S"
ENV_ROTATE_SUMMARY_TIMEOUT_S = "RTECH_ROTATE_SUMMARY_TIMEOUT_S"

# --- defaults ----------------------------------------------------------------

DEFAULT_RESTART_MAX_ATTEMPTS = 5
DEFAULT_RESTART_BACKOFF_BASE_S = 1.0
DEFAULT_RESTART_BACKOFF_CAP_S = 60.0
DEFAULT_RESTART_HEALTHY_RESET_S = 300.0

DEFAULT_ROTATE_MAX_TURNS = 200
DEFAULT_ROTATE_MAX_AGE_S = 24 * 60 * 60.0  # 24h
DEFAULT_ROTATE_SUMMARY_TIMEOUT_S = 120.0


def _env_int(name: str, default: int, *, minimum: int) -> int:
    """Return a positive int env override, or ``default`` when unset/invalid."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("lifecycle_policy | non-int {}={!r}, using default {}", name, raw, default)
        return default
    if value < minimum:
        logger.warning("lifecycle_policy | {}={} below min {}, using min", name, value, minimum)
        return minimum
    return value


def _env_float(name: str, default: float, *, minimum: float) -> float:
    """Return a positive float env override, or ``default`` when unset/invalid."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("lifecycle_policy | non-float {}={!r}, using default {}", name, raw, default)
        return default
    if value < minimum:
        logger.warning("lifecycle_policy | {}={} below min {}, using min", name, value, minimum)
        return minimum
    return value


# --- restart policy ----------------------------------------------------------


@dataclass(frozen=True)
class RestartDecision:
    """What the supervisor should do about one dead-session event.

    ``should_restart`` false means the circuit is OPEN: stop retrying, publish the
    blocked status + circuit-open lifecycle frame, and stay down. ``backoff_s`` is
    how long to sleep before the (allowed) restart. ``attempt`` is the 1-based
    attempt number this decision authorizes (for logging).
    """

    should_restart: bool
    backoff_s: float
    attempt: int


class RestartPolicy:
    """Bound restart attempts with exponential backoff + a circuit breaker.

    State: a single in-process attempt counter. :meth:`on_dead_session` advances
    it and returns a :class:`RestartDecision`; once ``max_attempts`` is exhausted
    the circuit opens (``should_restart=False``) and stays open until
    :meth:`on_healthy` resets the counter. Backoff is
    ``min(base * 2**(attempt-1), cap)``.

    The healthy reset has two triggers, both routed by the supervisor: a
    successful ``TurnDone`` (call :meth:`on_healthy`), or ``healthy_reset_s``
    elapsed since the last (re)connect while no restart was needed (the supervisor
    compares :meth:`seconds_since_connect` against :attr:`healthy_reset_s`).
    """

    def __init__(
        self,
        *,
        max_attempts: int = DEFAULT_RESTART_MAX_ATTEMPTS,
        backoff_base_s: float = DEFAULT_RESTART_BACKOFF_BASE_S,
        backoff_cap_s: float = DEFAULT_RESTART_BACKOFF_CAP_S,
        healthy_reset_s: float = DEFAULT_RESTART_HEALTHY_RESET_S,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if backoff_base_s <= 0:
            raise ValueError("backoff_base_s must be positive")
        if backoff_cap_s < backoff_base_s:
            raise ValueError("backoff_cap_s must be >= backoff_base_s")
        if healthy_reset_s <= 0:
            raise ValueError("healthy_reset_s must be positive")
        self.max_attempts = max_attempts
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self.healthy_reset_s = healthy_reset_s
        self._now = now
        self._attempts = 0
        self._last_connect_at = now()

    @classmethod
    def from_env(cls, *, now: Callable[[], float] = time.monotonic) -> RestartPolicy:
        """Build a policy from ``RTECH_RESTART_*`` env vars, falling back to defaults."""
        base = _env_float(ENV_RESTART_BACKOFF_BASE_S, DEFAULT_RESTART_BACKOFF_BASE_S, minimum=0.001)
        # The cap can never sit below the base (the constructor rejects it); clamp
        # an env cap up to the base so a misconfig degrades to "no backoff growth"
        # rather than crashing the daemon at boot.
        cap = max(
            base,
            _env_float(ENV_RESTART_BACKOFF_CAP_S, DEFAULT_RESTART_BACKOFF_CAP_S, minimum=0.001),
        )
        return cls(
            max_attempts=_env_int(
                ENV_RESTART_MAX_ATTEMPTS, DEFAULT_RESTART_MAX_ATTEMPTS, minimum=1
            ),
            backoff_base_s=base,
            backoff_cap_s=cap,
            healthy_reset_s=_env_float(
                ENV_RESTART_HEALTHY_RESET_S, DEFAULT_RESTART_HEALTHY_RESET_S, minimum=0.001
            ),
            now=now,
        )

    @property
    def attempts(self) -> int:
        """The number of restart attempts consumed since the last healthy reset."""
        return self._attempts

    def on_dead_session(self) -> RestartDecision:
        """Advance the attempt counter and decide whether to restart.

        Returns a decision authorizing the next restart with its backoff, or —
        once the budget is spent — a circuit-open decision (``should_restart``
        false) the supervisor turns into a blocked status + circuit-open frame.
        """
        if self._attempts >= self.max_attempts:
            logger.error(
                "restart_policy | circuit open | attempts={} max={}",
                self._attempts,
                self.max_attempts,
            )
            return RestartDecision(should_restart=False, backoff_s=0.0, attempt=self._attempts)
        self._attempts += 1
        backoff = min(self.backoff_base_s * (2 ** (self._attempts - 1)), self.backoff_cap_s)
        logger.warning(
            "restart_policy | restart authorized | attempt={} max={} backoff={:.1f}s",
            self._attempts,
            self.max_attempts,
            backoff,
        )
        return RestartDecision(should_restart=True, backoff_s=backoff, attempt=self._attempts)

    def on_connected(self) -> None:
        """Mark a fresh (re)connect: start the healthy-reset clock for this attempt."""
        self._last_connect_at = self._now()

    def seconds_since_connect(self) -> float:
        """Seconds since the last :meth:`on_connected` (the healthy-window elapsed)."""
        return self._now() - self._last_connect_at

    def on_healthy(self) -> None:
        """Reset the attempt budget after a proven-healthy period.

        Called on a successful ``TurnDone`` or once ``healthy_reset_s`` connected
        elapsed. A no-op when nothing was consumed, so it is cheap to call on
        every completed turn.
        """
        if self._attempts:
            logger.info("restart_policy | healthy, attempt budget reset from {}", self._attempts)
        self._attempts = 0
        self._last_connect_at = self._now()


# --- rotation policy ---------------------------------------------------------


@dataclass(frozen=True)
class RotationDecision:
    """Whether a completed turn should trigger a bounded context rotation.

    ``should_rotate`` true means the supervisor runs ONE bounded summary turn and
    (on success) swaps to a fresh session. ``reason`` names the trigger
    (``"turns"`` or ``"age"``) for logging and the lifecycle frame detail.
    """

    should_rotate: bool
    reason: str = ""


class RotationPolicy:
    """Trigger bounded context rotation at a turn or age threshold.

    State: a completed-turn counter and the session-start timestamp. The
    supervisor calls :meth:`on_turn_done` at each ``TurnDone`` (never mid-turn)
    and rotates when either bound is crossed. After a successful rotation the
    supervisor calls :meth:`on_rotated` to reset both counters for the fresh
    session. On a failed rotation the supervisor calls :meth:`defer` so the next
    trigger is the next ``TurnDone`` (the failed attempt is not retried in a loop
    — work is preserved and the old session keeps serving).
    """

    def __init__(
        self,
        *,
        max_turns: int = DEFAULT_ROTATE_MAX_TURNS,
        max_age_s: float = DEFAULT_ROTATE_MAX_AGE_S,
        summary_timeout_s: float = DEFAULT_ROTATE_SUMMARY_TIMEOUT_S,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if max_age_s <= 0:
            raise ValueError("max_age_s must be positive")
        if summary_timeout_s <= 0:
            raise ValueError("summary_timeout_s must be positive")
        self.max_turns = max_turns
        self.max_age_s = max_age_s
        self.summary_timeout_s = summary_timeout_s
        self._now = now
        self._completed_turns = 0
        self._session_started_at = now()
        # After a failed rotation, hold off until the NEXT turn crosses the bound
        # again rather than re-firing on every subsequent turn.
        self._deferred = False

    @classmethod
    def from_env(cls, *, now: Callable[[], float] = time.monotonic) -> RotationPolicy:
        """Build a policy from ``RTECH_ROTATE_*`` env vars, falling back to defaults."""
        return cls(
            max_turns=_env_int(ENV_ROTATE_MAX_TURNS, DEFAULT_ROTATE_MAX_TURNS, minimum=1),
            max_age_s=_env_float(ENV_ROTATE_MAX_AGE_S, DEFAULT_ROTATE_MAX_AGE_S, minimum=1.0),
            summary_timeout_s=_env_float(
                ENV_ROTATE_SUMMARY_TIMEOUT_S, DEFAULT_ROTATE_SUMMARY_TIMEOUT_S, minimum=1.0
            ),
            now=now,
        )

    @property
    def completed_turns(self) -> int:
        """Completed turns counted against the rotation bound for this session."""
        return self._completed_turns

    def on_turn_done(self) -> RotationDecision:
        """Count a completed turn and decide whether to rotate now.

        Called ONLY at ``TurnDone`` so rotation never fires mid-turn. Returns a
        rotate decision when the turn count or session age crosses its bound.
        """
        self._completed_turns += 1
        if self._completed_turns >= self.max_turns:
            return RotationDecision(should_rotate=True, reason="turns")
        age = self._now() - self._session_started_at
        if age >= self.max_age_s:
            return RotationDecision(should_rotate=True, reason="age")
        return RotationDecision(should_rotate=False)

    def on_rotated(self) -> None:
        """Reset both bounds for the freshly-started session."""
        self._completed_turns = 0
        self._session_started_at = self._now()
        self._deferred = False
        logger.info("rotation_policy | rotated, counters reset")

    def defer(self) -> None:
        """Record a failed rotation: back off the count bound by one turn.

        Keeps the age bound intact (an old session should still rotate on the next
        turn once summary succeeds) but drops the turn counter just under the bound
        so the NEXT ``TurnDone`` re-triggers rather than every turn re-firing.
        """
        self._deferred = True
        self._completed_turns = max(0, self.max_turns - 1)
        logger.warning("rotation_policy | rotation deferred to next turn")
