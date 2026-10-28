"""The frame-driven managed turn — the ratified rtech-frame seam.

This is the bus-first, typed-frame turn path, distinct from the fleet-envelope
:mod:`~rtech_agentd.nats_ingress`. Here the wire is pure typed frames:

* A :class:`~rtech_frames.TaskFrame` arrives on ``rtech.agent.<agent_id>.task``
  (``subjects.agent_task``) and drives the single managed engine session.
* At turn start a :class:`~rtech_frames.StatusFrame` with ``state="working"`` is
  published on ``rtech.task.<task_id>.status`` (``subjects.task_status``) — the
  in-flight / steer-wire signal that a turn is running.
* The completed engine turn is published as one
  :class:`~rtech_frames.TurnFrame` on ``rtech.task.<task_id>.turn``
  (``subjects.task_turn``).
* ``input_required`` / ``blocked`` StatusFrames reflect engine session states on
  the same status lane.
* While a task is active agentd subscribes its steer lane
  ``rtech.task.<task_id>.steer`` (``subjects.task_steer``). A
  :class:`~rtech_frames.SteerFrame` targeted at this agent — or at the broadcast
  target ``"all"``, the edge default when the human names no agent — for that
  active task is injected into the LIVE session (``session.send`` =
  streaming-input mid-run); a steer with ``interrupt=True`` aborts the active
  turn first, then injects. The
  interrupt intent is the typed ``SteerFrame.interrupt`` field, NEVER scraped from
  ``text`` — the steer wire carries no prose command channel. Each accepted steer
  is appended to the task's frame-graph refs, so the resulting ``TurnFrame.refs``
  references the steer causally. A steer for a non-active task or a non-matching
  explicit target is NEVER silently dropped: every ignore path logs a warning with
  the full typed context (target, self, task_id). Each lane handler
  is bound to its own task id — a payload whose ``task_id`` disagrees with the
  lane it rode in on is loudly rejected — and the subscription itself is retired
  (bus unsubscribe) when the task completes, so a stale lane can neither spoof
  nor double-deliver into a later task.

agentd NEVER emits a done signal — done is orchestrator-derived state, computed
from the typed frame graph, never a token agentd carries. ``visible_text`` is
guard-sanitized before publish; ``stance`` is the engine-filled typed
:class:`~rtech_frames.Stance` carried straight through, never scraped from prose.

Turn↔task correlation is the same single-serial-session rationale as the
fleet-envelope ingress ``PendingTurn`` deque: the managed session runs one turn
at a time, so a FIFO of pending ``task_id``s maps each :class:`TurnDone` back to
the task that triggered it.
"""

from __future__ import annotations

import asyncio
import os
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Literal, Protocol

from loguru import logger
from rtech_frames import (
    BROADCAST_TARGET,
    ChannelOrigin,
    Stance,
    StatusFrame,
    SteerFrame,
    TaskFrame,
    TurnFrame,
    VerificationReport,
)
from rtech_frames.codec import Frame
from rtech_frames.subs import FrameSubscribeError, Subscription, SubscriptionSet, subscribe_all
from rtech_orchestrator import subjects

from .completed_store import CompletedTaskStore
from .conversation_freshness import (
    ConversationFreshnessConfig,
    ConversationFreshnessTracker,
    FreshnessClassification,
)
from .engine import EngineEvent, ToolUse, TurnDone, failure_visible_text
from .grounding import (
    GroundingState,
    build_grounding_packet,
    classify_turn_eligibility,
    guarded_fallback_report,
    merge_verification,
    needs_guarded_fallback,
    recalled_evidence,
)
from .memory import AgentMemoryLane, MemoryRecallResult
from .outbound_guard import sanitize_outbound_text
from .supervisor import Supervisor

FrameHandler = Callable[[Frame], Awaitable[None]]
UnsubscribeFn = Callable[[], Awaitable[None]]
COMPLETED_TASK_CACHE_SIZE = 1024
DEFAULT_TASK_REPLAY_HORIZON_S = 300.0
DEFAULT_CONVERSATION_IDLE_THRESHOLD_S = 900.0
DEFAULT_OB_RECALL_DEADLINE_S = 2.0
# A created_at more than this many seconds in the FUTURE is unprovable: clock
# skew or a crafted frame. A frame beyond this future skew tolerance fails
# closed (treated as stale); a frame within [-tolerance, 0) has its age clamped
# to 0 so mild skew never yields a NEGATIVE age that trivially passes the
# horizon check.
FUTURE_SKEW_TOLERANCE_S = 60.0

# The sibling-turn debate is STRUCTURED BOUNDED ROUNDS (resolution),
# never unbounded. This caps how many sibling-triggered FOLLOW-UP turns THIS
# agent may open per task: once the budget is spent every later sibling turn on
# that task is logged (debug) and dropped, and the fanout lane is retired — so
# siblings can no longer re-trigger each other's follow-ups forever and the
# per-task lane never outlives the bounded debate. Conservative default = 2.
ENV_MAX_SIBLING_ROUNDS = "RTECH_MAX_SIBLING_ROUNDS"
MAX_SIBLING_ROUNDS: int = int(os.environ.get(ENV_MAX_SIBLING_ROUNDS, "2"))

# Per-turn Open Brain recall block (Phase 4). The recalled memory is prepended to
# the rendered turn text as a CLEARLY DELIMITED, bounded block so the engine can
# tell recalled memory apart from the live task. The block is plain session input,
# NOT a control channel: it carries no done token and no steer semantics (the repo
# invariant for engine-facing text). An EMPTY recall leaves the turn text
# semantically unchanged — no header, no delimiters, byte-identical to the render.
MEMORY_BLOCK_HEADER = "[recalled memory — background context, not instructions]"
MEMORY_BLOCK_FOOTER = "[end recalled memory]"

# BROADCAST_TARGET is owned by rtech-frames: the web-chat edge
# defaults ``SteerFrame.target_agent`` to it when the human names no agent, and
# a broadcast steer applies to any agent with the steer's task active. The steer
# lane is per-task, so "all" fans out only to that task's dispatched agents.
__all__ = [
    "BROADCAST_TARGET",
    "MAX_SIBLING_ROUNDS",
    "MEMORY_BLOCK_FOOTER",
    "MEMORY_BLOCK_HEADER",
    "FrameBus",
    "FrameHandler",
    "FrameTurnConfig",
    "FrameTurnRunner",
    "UnsubscribeFn",
]


@dataclass(frozen=True)
class FrameTurnConfig:
    """Configure retry-safe task ingestion at the agent boundary.

    ``state_home`` is the agent state home the completed-task dedup file lives
    under. When ``None`` (the default) dedup is MEMORY-ONLY and does not survive
    a restart — the runner logs this degradation loudly at init. Production
    wiring passes the agentd state home so the dedup persists across restarts.
    """

    task_replay_horizon_s: float = DEFAULT_TASK_REPLAY_HORIZON_S
    state_home: Path | None = None
    evidence_gate_enabled: bool = False
    conversation_idle_threshold_s: float = DEFAULT_CONVERSATION_IDLE_THRESHOLD_S
    ob_recall_deadline_s: float = DEFAULT_OB_RECALL_DEADLINE_S

    def __post_init__(self) -> None:
        """Reject a horizon that cannot provide a meaningful safety window."""
        if self.task_replay_horizon_s <= 0:
            raise ValueError("task_replay_horizon_s must be positive")
        if self.conversation_idle_threshold_s <= 0:
            raise ValueError("conversation_idle_threshold_s must be positive")
        if self.ob_recall_deadline_s <= 0:
            raise ValueError("ob_recall_deadline_s must be positive")


class FrameBus(Protocol):
    """Publish/subscribe seam for typed frames (mirrors the orchestrator seam)."""

    async def publish_frame(self, subject: str, frame: Frame) -> None:
        """Publish a typed frame to a subject.

        Args:
            subject: The NATS subject to publish on.
            frame: The typed frame to publish.
        """
        ...

    async def publish_frame_checked(self, subject: str, frame: Frame) -> None:
        """Publish a typed frame and RE-RAISE on transport failure.

        Unlike :meth:`publish_frame` (which swallows a raising publish and returns
        normally — the semantics the orchestrator/dev-stack turn path depends on),
        this variant lets a transport failure PROPAGATE so a caller that must not
        report a false success (the wake tool) can catch it and return a
        typed error instead of a silent no-op.

        Args:
            subject: The NATS subject to publish on.
            frame: The typed frame to publish.

        Raises:
            Exception: Whatever the underlying transport raises on publish.
        """
        ...

    async def subscribe_frame(self, subject: str, handler: FrameHandler) -> UnsubscribeFn | None:
        """Subscribe a handler to typed frames on a subject.

        Args:
            subject: The NATS subject to subscribe to.
            handler: The async callable invoked with each decoded frame.

        Returns:
            An awaitable that retires this subscription (stops handler delivery),
            or ``None`` when the bus cannot offer per-subscription retirement.
        """
        ...


class _BusSubscription:
    """Adapt the bus's ``UnsubscribeFn | None`` to the typed ``Subscription`` protocol.

    The frame bus's :meth:`FrameBus.subscribe_frame` returns an ``UnsubscribeFn``
    (or ``None`` when the transport cannot offer per-subscription retirement).
    The typed-bus :func:`~rtech_frames.subs.subscribe_all` primitive works with
    any object exposing ``async unsubscribe()``, so this wrapper carries the
    handle: ``unsubscribe`` delegates to it, or is a no-op when the bus offered
    none — mirroring the prior hand-rolled ``_retire_steer_subscription``.
    """

    def __init__(self, subject: str, unsubscribe: UnsubscribeFn | None) -> None:
        """Wrap the bus-returned unsubscribe handle (``None`` = not retirable)."""
        self._subject = subject
        self._unsubscribe = unsubscribe

    @property
    def retirable(self) -> bool:
        """Return whether the bus gave a real unsubscribe handle."""
        return self._unsubscribe is not None

    async def unsubscribe(self) -> None:
        """Retire the underlying subscription, or no-op when none was offered."""
        if self._unsubscribe is None:
            logger.warning(
                "frame_turn.steer | no unsubscribe handle to retire lane | subject={}",
                self._subject,
            )
            return
        await self._unsubscribe()


# Agentd deploy convention: participant ids are ``agentd-<name>`` optionally
# suffixed by the stack lane (``-dev``/``-prod``). The debate directive addresses
# each debater by the human-readable short name, mirroring how intake_sink derives
# ``arbiter = self.triage_agent.removeprefix("agentd-")``.
_AGENT_ID_PREFIX = "agentd-"
_AGENT_ID_SUFFIXES = ("-dev", "-prod")


def _short_name(agent_id: str) -> str:
    """Return the human-readable debate name for an ``agentd-<name>`` id.

    Strips the ``agentd-`` convention prefix and a ``-dev``/``-prod`` stack
    suffix, mirroring the intake edge's ack-name convention so a directive names
    ``nimbus`` rather than ``agentd-nimbus-prod``.
    """
    name = agent_id.removeprefix(_AGENT_ID_PREFIX)
    for suffix in _AGENT_ID_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _render_debate_task(frame: TaskFrame, agent_id: str) -> str:
    """Wrap a multi-participant debate task with a role directive (Rico #debate).

    On a duo/trio task each engine is told its ROLE before the task text so the
    debate is a real back-and-forth, not one agent writing the whole thing:

    * A PARTICIPANT (``agent_id`` is not the arbiter, or there is no arbiter) is
      told to argue ONLY its own position concisely and to engage sibling turns
      directly (agree / rebut / sharpen / concede) rather than summarize.
    * The ARBITER (``agent_id`` equals ``frame.arbiter``) is told to MODERATE — not
      argue the substance — and to close with a short TLDR ruling of the debaters'
      positions, not its own dissertation.

    Single-scope tasks (``len(participants) <= 1``) are returned UNCHANGED —
    ``frame.text`` byte-identical — so the non-debate path is untouched. The
    directive is plain session input, NOT a control channel: it carries no done
    token and no steer semantics (the repo invariant for engine-facing text).
    """
    participants = frame.participants
    if len(participants) <= 1:
        return frame.text
    names = [_short_name(pid) for pid in participants]
    count = len(names)
    roster = ", ".join(names)
    if frame.arbiter is not None and agent_id == frame.arbiter:
        directive = (
            "[debate task — you are the ARBITER] Moderate; do not argue the "
            "substance yourself. Open with at most one short framing line if "
            "useful. Wait for the debaters' positions (they arrive as sibling "
            "turns); ask a pointed question only if the debate stalls. Your FINAL "
            "turn is a short TLDR ruling/summary of THEIR positions — not your own "
            "dissertation."
        )
    else:
        me = _short_name(agent_id)
        directive = (
            f"[debate task] You are {me}, one of {count} debaters ({roster}). "
            "Speak ONLY for yourself, in your own voice. State your position "
            "concisely — a few short paragraphs, not an essay. When sibling turns "
            "arrive, engage them directly: agree, rebut, sharpen, concede. Do not "
            "summarize the whole debate; the arbiter does that."
        )
    return f"{directive}\n\n{frame.text}"


def _render_sibling_turn(frame: TurnFrame) -> str:
    """Render a co-participant's TurnFrame as session input (fanout).

    Presents the REAL sibling position — author, visible text, and the typed
    stance (``position``/``because``, plus any ``needs_from``) — so the model's
    next round can reference an actual sibling stance rather than role-playing
    one. This is plain session input, NOT a control channel: it carries no done
    token and no steer semantics.
    """
    stance = frame.stance
    lines = [
        f"[sibling turn from {frame.agent_id}]",
        frame.visible_text,
        f"(stance: {stance.position} — because {stance.because})",
    ]
    if stance.needs_from:
        lines.append(f"(needs from: {', '.join(stance.needs_from)})")
    return "\n".join(lines)


class FrameTurnRunner:
    """Drive the managed session from typed frames (TaskFrame in, TurnFrame out).

    Args:
        agent_id: The agent id this runner dispatches for (its task subject).
        supervisor: The supervisor owning the managed engine session + event pump.
        bus: The typed-frame publish/subscribe seam.
        config: Retry-safe task-ingestion config (dedup horizon + state home).
        memory: The per-agent Open Brain recall lane, or ``None`` when recall is
            disabled (no OB token). When present, each turn's ``frame.text`` is
            used as the recall query and a bounded memory block is prepended to
            the rendered turn text before ``session.send``. Recall is best-effort:
            an empty recall leaves the turn unchanged and any recall failure is
            contained and logged, so the original turn always proceeds.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        supervisor: Supervisor,
        bus: FrameBus,
        config: FrameTurnConfig | None = None,
        memory: AgentMemoryLane | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._supervisor = supervisor
        self._bus = bus
        self._config = config or FrameTurnConfig()
        self._memory = memory
        self._freshness = ConversationFreshnessTracker(
            ConversationFreshnessConfig(idle_threshold_s=self._config.conversation_idle_threshold_s)
        )
        self._grounding: dict[str, GroundingState] = {}
        if memory is None:
            logger.info(
                "frame_turn.init | no memory lane; per-turn OB recall disabled | agent={}",
                self._agent_id,
            )
        # FIFO turn↔task correlation: one managed session runs one turn at a time,
        # so each TurnDone maps back to the oldest still-pending task_id.
        self._pending: deque[str] = deque()
        # Per-pending-task ChannelOrigin, captured at dispatch, keyed
        # by task_id exactly like ``_steer_refs``. Presence of the KEY means the
        # task's origin is recorded; the VALUE may be ``None`` (a task can carry no
        # origin). The wake tool reads the EXECUTING turn's origin via
        # :meth:`current_origin` (the FIFO head of ``_pending``) so a wake carries
        # the SAME origin (for a FUTURE wired reply path — the carried origin is
        # inert on main today; see :mod:`rtech_agentd.wake_service`). A single slot
        # could not do this: a task pending mid-turn would overwrite the executing
        # turn's origin.
        self._task_origins: dict[str, ChannelOrigin | None] = {}
        # Per-active-task accumulated steer refs. Presence of a key = the task is
        # active and its steer lane is subscribed; the list is the extra frame-graph
        # refs (steer task_ids) folded into the eventual TurnFrame.refs.
        self._steer_refs: dict[str, list[str]] = {}
        # Per-task steer-lane subscription registry: key present =
        # a lane subscription exists (or is being established); the value is the
        # atomic ``SubscriptionSet`` handle installed via ``subscribe_all``,
        # used to RETIRE the real subscription when the task
        # completes — never just guard-state bookkeeping. One lane per task, so
        # the set is a single honest retire handle (a handle whose unsubscribe
        # fails is retained, not retried) in place of the prior hand-rolled
        # unsubscribe callable.
        self._steer_subscriptions: dict[str, SubscriptionSet] = {}
        # Un-retirable steer sets whose unsubscribe failed. This is a stash, not
        # a retry queue: nothing here is retried automatically. It keeps an
        # un-retirable live lane referenced and observable (logged at stash time)
        # instead of letting it keep delivering untracked.
        self._leaked_steer_subscriptions: dict[str, SubscriptionSet] = {}
        # Per-task sibling-turn subscription registry: a multi-agent
        # task subscribes THIS agent to the shared per-task turn lane so it SEES
        # its co-participants' TurnFrames — the fanout that makes the debate real.
        # Installed through the same atomic ``subscribe_all``/``SubscriptionSet``
        # primitive as the steer lane. Retired when the bounded-round
        # budget is exhausted or on a failed-send rollback — the lane
        # never outlives the bounded debate. One lane per task, so the set is a
        # single honest retire handle.
        self._sibling_subscriptions: dict[str, SubscriptionSet] = {}
        # Retirement-escalation store: sibling sets whose unsubscribe failed.
        # NEVER discarded — held so a stale lane can never keep delivering.
        self._leaked_sibling_subscriptions: dict[str, SubscriptionSet] = {}
        # Per-task count of sibling-triggered FOLLOW-UP turns THIS agent has
        # opened. The debate is STRUCTURED BOUNDED ROUNDS: once a
        # task reaches ``_max_sibling_rounds`` follow-ups here, further sibling
        # TurnFrames on it are dropped and the fanout lane is retired, so
        # siblings can never re-trigger each other's rounds without end. The cap
        # is snapshotted at init (env ``RTECH_MAX_SIBLING_ROUNDS``) so it is a
        # stable per-runner budget.
        self._sibling_round_counts: dict[str, int] = {}
        self._max_sibling_rounds: int = MAX_SIBLING_ROUNDS
        # Per-task participant identities of a fanned (multi-agent) task, captured
        # at dispatch. Presence keys the settle-detection and closing-turn state
        # to a live fanned task; cleared whenever the sibling lane retires.
        self._sibling_participants: dict[str, frozenset[str]] = {}
        # Per-task latest-observed terminality per participant: agentd
        # observes EVERY participant's TurnFrame on the shared task turn lane
        # (siblings via ``_on_sibling_turn``, its own via ``_publish_turn``). A
        # turn is terminal when its ``stance.needs_from`` is empty. When ALL
        # participants' latest observed turns are terminal the debate has settled
        # BELOW the cap (the common consensus case), so the fanout lane is retired
        # and its counter cleared — no lane/counter leak past a settled debate.
        self._sibling_latest_terminal: dict[str, dict[str, bool]] = {}
        # This agent's latest published stance per fanned task. The cap-induced
        # closing turn reuses its ``position``/``because`` so the terminus honestly
        # restates the position that stood, only flipping ``needs_from`` to empty.
        self._sibling_last_stance: dict[str, Stance] = {}
        # This agent's latest published, guard-sanitized ``visible_text`` per fanned
        # task. The cap-induced closing turn REUSES this REAL substantive text so a
        # full-thought debater's final budgeted answer is what renders in-channel —
        # terminality is communicated by the typed stance (``needs_from=[]``), never
        # by replacing the agent's words with a synthetic placeholder (a
        # follow-up). Captured alongside ``_sibling_last_stance`` in
        # :meth:`_publish_turn` so both survive lane retirement on a cap retire.
        self._sibling_last_visible_text: dict[str, str] = {}
        # Per-task marker: the round budget has been fully spent. Set when the
        # cap is reached and RETAINED past lane retirement so the final budgeted
        # follow-up's publish can still see "budget spent" and honor the closing
        # terminus. Cleared the moment that terminus is resolved.
        self._sibling_budget_spent: set[str] = set()
        # Per-task guard: a cap-induced closing TurnFrame is emitted at most once.
        # If this agent's final budgeted follow-up is still non-terminal, the cap
        # then drops all further sibling turns and orchestrator-derived done would
        # hang False forever; the frame-level terminus publishes one closing
        # terminal turn (to be superseded by a future arbiter). Cleared with the
        # rest of the per-task sibling state on retire.
        self._sibling_closing_emitted: set[str] = set()
        # Bounded per-agent idempotency memory. Keys include this runner's
        # participant identity explicitly so the invariant stays scoped to the
        # consumer that actually executed the task. Values are MONOTONIC
        # completion times, used only for in-process count-bounded eviction.
        self._completed_tasks: OrderedDict[tuple[str, str], float] = OrderedDict()
        # Durable dedup: persists (task_id, agent_id) -> completion WALL time so
        # a restart within the replay horizon does not start empty and
        # re-dispatch an already-completed redelivered frame.
        # ``None`` when no state home is configured — dedup is then MEMORY-ONLY,
        # logged loudly below so the degradation is never silent.
        self._completed_store: CompletedTaskStore | None = None
        # WALL-clock mirror of the persisted completions, consulted by the dedup
        # guard and kept in sync with the store on every completion.
        self._persisted_completions: dict[tuple[str, str], datetime] = {}
        state_home = self._config.state_home
        if state_home is None:
            logger.warning(
                "frame_turn.init | no state_home; completed-task dedup is MEMORY-ONLY "
                "and will NOT survive a restart | agent={}",
                self._agent_id,
            )
        else:
            self._completed_store = CompletedTaskStore(state_home, self._agent_id)
            # Evict-by-horizon on load: only within-horizon completions can still
            # dedup a redelivery, and a corrupt/missing file starts empty loudly.
            self._persisted_completions = self._completed_store.load(
                self._config.task_replay_horizon_s
            )
            logger.info(
                "frame_turn.init | loaded persisted dedup state | agent={} entries={} path={}",
                self._agent_id,
                len(self._persisted_completions),
                self._completed_store.path,
            )
        self._lock = asyncio.Lock()
        supervisor.add_event_sink(self._on_engine_event)

    async def start(self) -> None:
        """Subscribe the per-agent task subject to :meth:`_on_task_frame`."""
        subject = subjects.agent_task(self._agent_id)
        logger.info("frame_turn.start | agent={} subject={}", self._agent_id, subject)
        await self._bus.subscribe_frame(subject, self._on_task_frame)

    def current_origin(self) -> ChannelOrigin | None:
        """Return the ChannelOrigin of the task currently driving the session.

        The :class:`~rtech_agentd.wake_service.WakeService` reads this so a wake
        fired mid-turn carries the SAME origin as the triggering intake task (for a
        FUTURE wired reply path — the carried origin is inert on main today, since
        chatd renders only task_ids it minted). The executing turn is the FIFO
        head of ``_pending`` (the managed session runs one turn at a time), so its
        origin is the unambiguous "current" one. ``None`` when no task is executing
        or the executing task carried no origin (a non-intake frame).

        The head read is INTENTIONALLY lock-free: :class:`OriginProvider` is sync
        and WakeService's call site is sync, so acquiring the asyncio lock is not
        possible here. A single ``_pending`` head + ``_task_origins`` dict read is
        atomic enough for a best-effort origin — pop/append on the executing head
        is serialized by the single session, so WHICH task is head is correct; a
        stale-by-a-hair origin value is acceptable.
        """
        head = self._pending[0] if self._pending else None
        if head is None:
            return None
        return self._task_origins.get(head)

    async def _on_task_frame(self, frame: Frame) -> None:
        """Handle an inbound TaskFrame: emit working, then send the task to engine.

        Non-TaskFrame payloads on the task subject are logged and dropped.
        """
        if not isinstance(frame, TaskFrame):
            logger.warning(
                "frame_turn.task | non-task frame on task subject, ignoring | type={}",
                type(frame).__name__,
            )
            return
        logger.info("frame_turn.task | received | agent={} task_id={}", self._agent_id, frame.id)
        async with self._lock:
            task_key = (frame.id, self._agent_id)
            if (
                frame.id in self._steer_refs
                or task_key in self._completed_tasks
                or self._is_persisted_completed(task_key)
            ):
                logger.info(
                    "frame_turn.task | duplicate active/completed task, no-op | "
                    "agent={} task_id={}",
                    self._agent_id,
                    frame.id,
                )
                return
            if self._is_stale(frame):
                logger.warning(
                    "frame_turn.task | stale untracked task, fail-closed no-op | "
                    "agent={} task_id={} horizon_s={}",
                    self._agent_id,
                    frame.id,
                    self._config.task_replay_horizon_s,
                )
                return
            freshness = self._freshness.classify(
                origin_edge=frame.origin_edge,
                origin=frame.origin,
            )
            cold = freshness is not FreshnessClassification.WARM_FOLLOWUP
            self._pending.append(frame.id)
            self._steer_refs.setdefault(frame.id, [])
            # Record this task's channel origin keyed by task_id, right
            # alongside the steer-refs marker so exactly the tasks that pend get an
            # origin. A wake fired while this task is the executing FIFO head then
            # carries the SAME origin. Recorded AFTER the refuse/dedup/stale guards
            # above so no refused frame ever leaves an unmatched entry.
            self._task_origins[frame.id] = frame.origin
            if self._config.evidence_gate_enabled:
                self._grounding[frame.id] = GroundingState(
                    cold=cold, eligibility=classify_turn_eligibility(frame.text)
                )
            needs_subscribe = frame.id not in self._steer_subscriptions
            # A task with co-participants fans sibling turns to this agent:
            # more than one named participant means real siblings
            # exist on the shared per-task turn lane worth seeing.
            needs_siblings = (
                len(frame.participants) > 1 and frame.id not in self._sibling_subscriptions
            )
            if needs_subscribe:
                # Reserve the slot under the lock so a concurrent redelivery of
                # the same task id never races a second lane subscription. The
                # empty set is an already-retired placeholder until the real
                # subscription lands.
                self._steer_subscriptions[frame.id] = SubscriptionSet([])
            if needs_siblings:
                self._sibling_subscriptions[frame.id] = SubscriptionSet([])
                # Capture the participant identities so settle-detection knows
                # WHOSE latest turns must all be terminal for the debate to have
                # settled below the cap.
                self._sibling_participants[frame.id] = frozenset(frame.participants)
                self._sibling_latest_terminal[frame.id] = {}
        if needs_subscribe:
            await self._subscribe_steer(frame.id)
        if needs_siblings:
            await self._subscribe_siblings(frame.id)
        # Register the in-flight task with the supervisor BEFORE the send so the
        # stall watchdog maps a wedged turn back to this task_id (never "no task")
        # and recovery can re-emit its working status. The supervisor clears it on
        # the matching TurnDone. Part of the completion contract: a turn is bound
        # to its task from send to TurnDone.
        self._supervisor.track_in_flight_task(frame.id)
        await self.publish_status(
            frame.id,
            "working",
            phase=("verifying" if cold else "responding")
            if self._config.evidence_gate_enabled
            else None,
        )
        # Multi-participant (duo/trio) tasks get a debate-role directive prepended
        # so each engine argues its OWN position and the arbiter moderates+TLDRs,
        # instead of one agent writing the whole debate (Rico's live feedback). A
        # single-scope task renders ``frame.text`` byte-identically (no directive).
        rendered = _render_debate_task(frame, self._agent_id)
        # Phase 4: per-turn Open Brain recall happens HERE — immediately before the
        # session send — using the raw task text as the query, then prepends a
        # bounded, delimited memory block to the rendered turn. Empty recall or any
        # recall failure leaves ``rendered`` unchanged so the turn always proceeds.
        turn_text = await self._grounded_turn_text(frame.id, rendered, frame.text)
        try:
            await self._supervisor.session.send(turn_text)
        except Exception as exc:
            logger.error(
                "frame_turn.task | session send failed | task_id={} reason={}", frame.id, exc
            )
            # The turn never opened: retire the tracking and pending state so a
            # later TurnDone is not mis-correlated to this task and the watchdog
            # does not see a phantom in-flight turn.
            await self._retire_failed_task(frame.id)
            return
        logger.debug("frame_turn.task | dispatched to engine | task_id={}", frame.id)

    async def _grounded_turn_text(self, task_id: str, rendered: str, query: str) -> str:
        """Recall under a deadline and construct the evidence packet for a gated turn."""
        if not self._config.evidence_gate_enabled:
            return self._prepend_recall(rendered, query)
        state = self._grounding[task_id]
        if self._memory is None:
            state.limitations.append("Open Brain recall was unavailable")
            return build_grounding_packet(
                rendered=rendered,
                recall_block="",
                cold=state.cold,
                memory_header=MEMORY_BLOCK_HEADER,
                memory_footer=MEMORY_BLOCK_FOOTER,
            )
        # Bound the underlying OB HTTP timeout DOWN to the hard recall deadline
        # before the threaded recall: ``asyncio.wait_for`` cancels only the AWAIT,
        # not the worker thread, so without this the blocking request could keep
        # the thread occupied up to the client's own (much larger) timeout.
        self._memory.bound_http_timeout(self._config.ob_recall_deadline_s)
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._memory.recall_with_sources,
                    query,
                    limit=self._memory.max_items,
                ),
                timeout=self._config.ob_recall_deadline_s,
            )
            if result.status == "success":
                state.ob_checked = True
            else:
                state.limitations.append("Open Brain recall failed")
        except TimeoutError:
            result = MemoryRecallResult(block="")
            state.limitations.append("Open Brain recall timed out")
            logger.warning("frame_turn.recall | deadline exceeded | agent={}", self._agent_id)
        except Exception as exc:
            result = MemoryRecallResult(block="")
            state.limitations.append("Open Brain recall failed")
            logger.warning(
                "frame_turn.recall | recall failed | agent={} reason={}",
                self._agent_id,
                type(exc).__name__,
            )
        state.recalled_sources = recalled_evidence(result)
        return build_grounding_packet(
            rendered=rendered,
            recall_block=result.block,
            cold=state.cold,
            memory_header=MEMORY_BLOCK_HEADER,
            memory_footer=MEMORY_BLOCK_FOOTER,
        )

    def _prepend_recall(self, rendered: str, query: str) -> str:
        """Prepend a bounded Open Brain memory block to the rendered turn text.

        Per-turn recall (Phase 4): when a memory lane is configured, recall the
        agent's Open Brain memory keyed on the raw task ``query`` and prepend the
        result as a clearly delimited, bounded block above the rendered turn so
        the engine can distinguish recalled context from the live task. The block
        is plain session input — it carries no done token and no steer semantics.

        Recall is best-effort and NEVER fails the turn:

        * No memory lane configured -> ``rendered`` returned unchanged.
        * Empty recall (nothing found) -> ``rendered`` returned unchanged
          (semantically identical — no header, no delimiters).
        * Any recall failure -> contained and logged as a warning WITHOUT the
          query text, recalled content, or token; ``rendered`` returned unchanged.

        :meth:`AgentMemoryLane.recall` already redacts and bounds its output and
        swallows its own transport failures (returning an empty block), so a
        network/OB failure degrades to the empty-recall path here. The extra
        guard wraps any UNEXPECTED raise so the turn is contained regardless.

        Args:
            rendered: The fully rendered turn text (debate directive already
                applied for a multi-participant task).
            query: The raw task text used as the recall query. Never logged.

        Returns:
            ``rendered`` with a leading memory block when recall returned content,
            else ``rendered`` unchanged.
        """
        if self._memory is None:
            return rendered
        try:
            block = self._memory.recall(query, limit=self._memory.max_items)
        except Exception as exc:  # best-effort: never fail the turn on recall
            logger.warning(
                "frame_turn.recall | recall failed; sending turn without memory | "
                "agent={} reason={}",
                self._agent_id,
                type(exc).__name__,
            )
            return rendered
        if not block.strip():
            logger.debug(
                "frame_turn.recall | empty recall; turn unchanged | agent={}", self._agent_id
            )
            return rendered
        logger.info(
            "frame_turn.recall | prepended memory block | agent={} block_len={}",
            self._agent_id,
            len(block),
        )
        return f"{MEMORY_BLOCK_HEADER}\n{block}\n{MEMORY_BLOCK_FOOTER}\n\n{rendered}"

    def _is_stale(self, frame: TaskFrame) -> bool:
        """Return whether an untracked task must fail closed against the retry horizon.

        The dedup cache proves a *within-horizon* completed replay is never
        re-dispatched (its key is retained until it ages past the horizon). This
        guard is the symmetric half for untracked frames: a task the cache has
        never seen (or has evicted) is safe to dispatch ONLY if it can be proven
        within the horizon. A frame with no ``created_at`` cannot be proven
        within-horizon — it is indistinguishable from an evicted, arbitrarily
        old replay — so it FAILS CLOSED rather than fail open and
        risk a double-dispatch. Every live producer stamps ``created_at``
        (nats_ingress / chatd / webchat) and the orchestrator republishes the
        frame verbatim, so only legacy/untimestamped replays hit this branch.

        A FUTURE-dated ``created_at`` (clock skew or a crafted frame) yields a
        negative age that would trivially pass ``age > horizon`` and fail OPEN.
        Beyond ``FUTURE_SKEW_TOLERANCE_S`` in the future the age is unprovable,
        so the frame FAILS CLOSED; within the tolerance the age is clamped to 0
        so mild skew never dispatches off a negative age.

        DISTRUST WINDOW: if the durable dedup store loaded a CORRUPT
        file, the restart-surviving dedup authority is provably lost. For
        ``task_replay_horizon_s`` seconds after that load ANY untracked frame is
        refused (fail closed, loud per refusal) — an untracked frame cannot be
        proven a first delivery when the record that would have deduped it may
        have been in the corrupt file. After the window every redeliverable
        frame has aged past the horizon, so the guarantee is restored honestly.
        """
        if self._completed_store is not None and self._completed_store.in_distrust_window():
            logger.warning(
                "frame_turn.task | corrupt-store distrust window OPEN (until {}); refusing "
                "untracked frame, fail-closed no-op | agent={} task_id={} horizon_s={}",
                self._completed_store.distrust_until.isoformat()
                if self._completed_store.distrust_until is not None
                else "?",
                self._agent_id,
                frame.id,
                self._config.task_replay_horizon_s,
            )
            return True
        if frame.created_at is None:
            return True
        created_at = frame.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        age_s = (datetime.now(UTC) - created_at).total_seconds()
        if age_s < -FUTURE_SKEW_TOLERANCE_S:
            logger.warning(
                "frame_turn.task | created_at beyond future skew tolerance, fail-closed | "
                "agent={} task_id={} future_s={:.1f} tolerance_s={}",
                self._agent_id,
                frame.id,
                -age_s,
                FUTURE_SKEW_TOLERANCE_S,
            )
            return True
        if age_s < 0:
            age_s = 0.0
        return age_s > self._config.task_replay_horizon_s

    def _is_persisted_completed(self, task_key: tuple[str, str]) -> bool:
        """Return whether ``task_key`` completed within the wall-clock horizon.

        The durable (restart-surviving) half of the dedup: a completion persisted
        by a prior runner still deduplicates a redelivery as long as it is within
        the replay horizon by WALL clock (monotonic does not survive a restart).
        Caller holds ``self._lock``.
        """
        completed = self._persisted_completions.get(task_key)
        if completed is None:
            return False
        within = (datetime.now(UTC) - completed).total_seconds() <= (
            self._config.task_replay_horizon_s
        )
        if not within:
            # Aged out: drop it so the map stays bounded to the safety window.
            del self._persisted_completions[task_key]
        return within

    async def _retire_failed_task(self, task_id: str) -> None:
        """Roll back task state after a send that never opened a turn."""
        self._supervisor.clear_in_flight_task(task_id)
        async with self._lock:
            try:
                self._pending.remove(task_id)
            except ValueError:
                logger.debug("frame_turn.task | task_id not pending on rollback | {}", task_id)
            # The turn never opened: remove ONLY this failed task's origin entry.
            # Another task still at the pending head keeps its origin.
            self._task_origins.pop(task_id, None)
            self._steer_refs.pop(task_id, None)
            self._grounding.pop(task_id, None)
            self._clear_sibling_task_state(task_id)
            subscriptions = self._steer_subscriptions.pop(task_id, None)
            sibling_subscriptions = self._sibling_subscriptions.pop(task_id, None)
        await self._retire_steer_subscription(task_id, subscriptions)
        await self._retire_sibling_subscription(task_id, sibling_subscriptions)

    # --- steer wire (human-in-the-loop) --------------------------------------

    async def _subscribe_one(self, subject: str, handler: FrameHandler) -> Subscription:
        """Subscribe one subject and adapt the bus handle to a ``Subscription``.

        The seam for :func:`~rtech_frames.subs.subscribe_all`: it calls the
        frame bus's ``subscribe_frame`` and wraps the returned
        ``UnsubscribeFn | None`` in a :class:`_BusSubscription` so the atomic set
        carries a real retire handle. This wrapper only adapts the
        handle — it does not itself raise. The rollback / fail-closed machinery
        in ``subscribe_all`` (and the ``FrameSubscribeError`` branch in
        :meth:`_subscribe_steer`) only engages for bus implementations whose
        ``subscribe_frame`` raises; the production agentd bus returns ``None``
        rather than raising, so that path stays dormant there.
        """
        unsubscribe = await self._bus.subscribe_frame(subject, handler)
        return _BusSubscription(subject, unsubscribe)

    async def _subscribe_steer(self, task_id: str) -> None:
        """Subscribe the per-task steer lane so live steers reach this session.

        The handler is BOUND to this lane's task id: a frame whose
        payload ``task_id`` disagrees with the lane it arrived on is loudly
        rejected, so a stale or spoofed lane can never inject into another task.
        The lane is installed through the typed-bus
        :func:`~rtech_frames.subs.subscribe_all` primitive: a single
        atomic ``SubscriptionSet`` replaces the hand-rolled unsubscribe handle,
        so a failed subscribe leaves nothing live-but-untracked and the returned
        set is the one honest retire handle stored for completion.
        """
        subject = subjects.task_steer(task_id)

        async def _bound_handler(frame: Frame) -> None:
            await self._on_steer_frame(task_id, frame)

        try:
            subscriptions = await subscribe_all(self._subscribe_one, [(subject, _bound_handler)])
        except FrameSubscribeError as exc:
            logger.error("frame_turn.steer | subscribe failed | subject={} reason={}", subject, exc)
            # Rollback may have left a live handle on exc.leaked. Never discard
            # it: stash it (referenced + logged, no automatic retry) so the live
            # lane stays observable, then release the reserved guard slot.
            async with self._lock:
                reserved = self._steer_subscriptions.pop(task_id, None)
            if len(exc.leaked) > 0:
                self._leaked_steer_subscriptions[task_id] = exc.leaked
                logger.critical(
                    "frame_turn.steer | rollback incomplete; live subscription(s) held | "
                    "task_id={} subjects={}",
                    task_id,
                    exc.leaked.subjects,
                )
            # A reserved-but-empty placeholder is expected; only warn if a live
            # earlier set was somehow displaced (should never happen for one lane).
            if reserved is not None and len(reserved) > 0:
                logger.critical(
                    "frame_turn.steer | reserved set held live handles on failure | task_id={}",
                    task_id,
                )
                self._leaked_steer_subscriptions.setdefault(task_id, reserved)
            return
        async with self._lock:
            self._steer_subscriptions[task_id] = subscriptions
        logger.info("frame_turn.steer | subscribed | agent={} subject={}", self._agent_id, subject)

    async def _retire_steer_subscription(
        self, task_id: str, subscriptions: SubscriptionSet | None
    ) -> None:
        """Retire a task's live steer-lane subscription set, loudly on failure.

        ``SubscriptionSet.retire()`` never raises and never lies: a handle whose
        unsubscribe fails is RETAINED (not discarded) and the set reports
        incomplete, so the still-live lane is stashed in
        ``_leaked_steer_subscriptions`` — referenced and logged, not
        auto-retried — rather than escaping untracked.
        """
        if subscriptions is None:
            return
        if not await subscriptions.retire():
            self._leaked_steer_subscriptions[task_id] = subscriptions
            logger.critical(
                "frame_turn.steer | retire incomplete; live lane stashed (no auto-retry) | "
                "task_id={} subjects={}",
                task_id,
                subscriptions.subjects,
            )
            return
        logger.info(
            "frame_turn.steer | lane retired | agent={} task_id={}", self._agent_id, task_id
        )

    # --- sibling turn fanout (real inter-agent debate) -----------------------

    async def _subscribe_siblings(self, task_id: str) -> None:
        """Subscribe the shared per-task turn lane so co-participants are SEEN.

        This is the fanout wire. Every participant already PUBLISHES its
        ``TurnFrame`` to the ONE shared ``rtech.task.<task_id>.turn`` subject
        (``subjects.task_turn``); a multi-agent task additionally SUBSCRIBES
        this agent to that subject so a sibling's real position (visible text +
        typed stance) reaches the live session and the model can post a second
        round whose ``stance.because``/``needs_from`` reference a REAL sibling,
        not a role-played one. The lane is installed through the same atomic
        :func:`~rtech_frames.subs.subscribe_all` primitive as the steer lane:
        a failed subscribe leaves nothing live-but-untracked and
        the returned :class:`~rtech_frames.subs.SubscriptionSet` is the single
        retire handle. A missing sibling lane is non-fatal (the task still
        runs) — exactly like the steer lane.
        """
        subject = subjects.task_turn(task_id)

        async def _bound_handler(frame: Frame) -> None:
            await self._on_sibling_turn(task_id, frame)

        try:
            subscriptions = await subscribe_all(self._subscribe_one, [(subject, _bound_handler)])
        except FrameSubscribeError as exc:
            logger.error(
                "frame_turn.sibling | subscribe failed | subject={} reason={}", subject, exc
            )
            async with self._lock:
                self._sibling_subscriptions.pop(task_id, None)
            if len(exc.leaked) > 0:
                self._leaked_sibling_subscriptions[task_id] = exc.leaked
                logger.critical(
                    "frame_turn.sibling | rollback incomplete; live subscription(s) held | "
                    "task_id={} subjects={}",
                    task_id,
                    exc.leaked.subjects,
                )
            return
        async with self._lock:
            self._sibling_subscriptions[task_id] = subscriptions
        logger.info(
            "frame_turn.sibling | subscribed | agent={} subject={}", self._agent_id, subject
        )

    async def _retire_sibling_subscription(
        self, task_id: str, subscriptions: SubscriptionSet | None
    ) -> None:
        """Retire a task's live sibling-turn lane, loudly on failure.

        Mirrors :meth:`_retire_steer_subscription`: a handle whose unsubscribe
        fails is RETAINED by ``SubscriptionSet.retire()`` and moved into
        ``_leaked_sibling_subscriptions`` — referenced and logged, not
        auto-retried — rather than escaping untracked.
        """
        if subscriptions is None:
            return
        if not await subscriptions.retire():
            self._leaked_sibling_subscriptions[task_id] = subscriptions
            logger.critical(
                "frame_turn.sibling | retire incomplete; live lane stashed (no auto-retry) | "
                "task_id={} subjects={}",
                task_id,
                subscriptions.subjects,
            )
            return
        logger.info(
            "frame_turn.sibling | lane retired | agent={} task_id={}", self._agent_id, task_id
        )

    async def _retire_exhausted_sibling_lane(self, task_id: str, *, by_cap: bool = True) -> None:
        """Retire the fanout lane once the bounded debate is over.

        Idempotent: pops the live set under the lock and retires it through the
        shared retire path. A concurrent caller that already popped it finds
        nothing and no-ops, so the lane is retired exactly once even when the
        drop path and the budget-just-spent path both fire.

        ``by_cap`` distinguishes the two terminal conditions: the round budget
        was spent (``True``, the default) vs. the debate settled below the cap
        with every participant's latest turn terminal (``False``). Only the
        cap-exhaust path can leave this agent hung on a non-terminal final turn.

        On a SETTLED retire (``by_cap=False``) every participant's latest turn is
        already terminal, so ALL per-task state is cleared here — there is no hang
        to close. On a CAP retire (``by_cap=True``) the round counter and
        participant set (lane-scoped) are dropped, but a small
        ``_sibling_budget_spent`` marker plus the closing-decision state (last
        stance + latest-terminality) is RETAINED past lane retirement: this
        agent's final budgeted follow-up may not have PUBLISHED yet, and
        :meth:`_publish_turn` reads that marker to emit the frame-level closing
        terminus for the still-in-flight follow-up, then clears it. In
        production every ``session.send`` yields a TurnDone, so that publish always
        runs and the retained marker is always cleared.
        """
        async with self._lock:
            subscriptions = self._sibling_subscriptions.pop(task_id, None)
            # The round counter and participant set are lane-scoped: drop them the
            # moment the lane retires so neither outlives the bounded debate.
            self._sibling_round_counts.pop(task_id, None)
            self._sibling_participants.pop(task_id, None)
            if by_cap:
                # Remember the budget was spent so the final follow-up's publish
                # can honor the closing terminus; keep the last stance +
                # latest-terminality it reads. Cleared when that publish resolves.
                self._sibling_budget_spent.add(task_id)
            else:
                # Settled below the cap: every latest turn is terminal, no hang.
                self._clear_sibling_closing_state(task_id)
        if subscriptions is None:
            return
        await self._retire_sibling_subscription(task_id, subscriptions)

    def _clear_sibling_closing_state(self, task_id: str) -> None:
        """Drop the closing-decision bookkeeping (caller holds ``self._lock``).

        The budget-spent marker, latest-terminality map, last stance, and
        closing-turn guard together decide the cap-induced terminus; they are
        cleared as a unit once the terminus is resolved so none outlives the
        bounded debate. A fanned task's ``_task_origins`` entry (R1)
        is retired here too: for a fanned task the entry is kept alive past first
        TurnDone so the cap-closing turn can read it, so THIS unit-clear (settle,
        cap-terminus, or failed-send rollback via ``_clear_sibling_task_state``)
        is its final retirement point — callers that still need the origin (the
        cap terminus) snapshot it under the lock before calling this.
        """
        self._sibling_budget_spent.discard(task_id)
        self._sibling_latest_terminal.pop(task_id, None)
        self._sibling_last_stance.pop(task_id, None)
        self._sibling_last_visible_text.pop(task_id, None)
        self._sibling_closing_emitted.discard(task_id)
        self._task_origins.pop(task_id, None)

    def _clear_sibling_task_state(self, task_id: str) -> None:
        """Drop ALL per-task sibling bookkeeping (caller holds ``self._lock``).

        Used on the failed-send rollback where the task never really opened:
        the lane-scoped round counter + participant set AND the closing-decision
        state all go.
        """
        self._sibling_round_counts.pop(task_id, None)
        self._sibling_participants.pop(task_id, None)
        self._clear_sibling_closing_state(task_id)

    async def _record_observed_turn(self, frame: TurnFrame) -> bool:
        """Record a participant's latest-observed turn terminality; report settled.

        agentd observes EVERY participant's ``TurnFrame`` on the shared task turn
        lane (siblings here, its own via :meth:`_publish_turn`). A turn is
        terminal when ``stance.needs_from`` is empty. When ALL known participants'
        latest observed turns are terminal the debate has SETTLED below the cap
        — the caller retires the lane and clears the counter so a
        settled multi-agent task leaks neither its subscription nor its counter.

        Returns ``True`` when this observation settles the debate for a still-live
        fanned task, else ``False``. A task whose sibling lane is already gone
        (settled/exhausted/rolled back) never re-settles.
        """
        task_id = frame.task_id
        async with self._lock:
            latest = self._sibling_latest_terminal.get(task_id)
            if latest is None:
                # Not a fanned task, or its closing-decision state is already
                # cleared (settled / rolled back): nothing to record or settle.
                return False
            latest[frame.agent_id] = not frame.stance.needs_from
            # A debate can only be SETTLED while the lane is still live — once the
            # lane is retired (cap path) there is nothing left to retire and the
            # closing terminus, not settle-detection, owns termination. The record
            # above still lands so the cap-induced closing check sees this turn.
            if task_id not in self._sibling_subscriptions:
                return False
            participants = self._sibling_participants.get(task_id, frozenset())
            settled = bool(participants) and all(
                latest.get(participant, False) for participant in participants
            )
        return settled

    async def _emit_closing_turn_if_hung(self, task_id: str) -> None:
        """Publish one closing terminal turn when the cap left this agent hung.

        The cap dropped all further sibling turns for ``task_id``. If this agent's
        LATEST published turn is still non-terminal (``needs_from`` populated),
        orchestrator-derived ``is_done`` would stay False forever. The frame-level
        terminus publishes ONE closing ``TurnFrame`` through the normal publish
        path — same ``position``/``because`` as the latest stance,
        ``needs_from=[]`` (terminal), and honest ``visible_text`` that the
        position stands as final because the round budget was reached. Nimbus's
        arbiter (future work) supersedes this later.

        Guarded to fire at most once per task, only once the round budget is
        spent, and only for a still-hung agent: the closing turn is published via
        :meth:`_publish_closing_turn`, which does NOT re-pend or count against the
        budget, and the echo guard already skips this agent's own turn on the
        shared lane, so it cannot re-trigger fanout.

        ORDERING INVARIANT (engine contract): ``TurnDone`` must be delivered
        decoupled from ``session.send`` (the supervisor pump does this), so the
        final budgeted follow-up's :meth:`_publish_turn` runs strictly AFTER the
        cap-retire sets ``_sibling_budget_spent`` and resolves the closing
        decision here. An engine that delivers ``TurnDone`` synchronously inside
        ``send`` violates this: the early budget-not-spent return below would
        strand the closing-decision state and skip the closing turn. Test
        engines must model the decoupled pump (as ``AutoRespondEngine`` does).

        PUBLISH-THEN-CLEAR (D7): the closing-decision state is NOT
        cleared until the closing turn is CONFIRMED published. The prior order
        cleared the state (and this task's origin) BEFORE the publish, so a
        publish failure was caught, swallowed, and lost the terminus forever — the
        capped participant stayed non-terminal in the orchestrator graph. Now the
        one-shot guard is reserved, the frame is published via the CHECKED path,
        and only on success is the state cleared as a unit. On a publish FAILURE
        the reservation is fully rolled back (guard un-set, own-terminality
        restored to False, all decision state retained), so the NEXT observed
        turn's :meth:`_publish_turn` re-enters here and retries — exactly one
        closing turn is ever published (the success clear + the one-shot guard
        together prevent a double-publish).
        """
        async with self._lock:
            if task_id in self._sibling_closing_emitted:
                return
            if task_id not in self._sibling_budget_spent:
                # Budget not yet spent: more real rounds may still land a terminal
                # turn on their own; no cap-induced hang to close.
                return
            # Budget IS spent: this is the terminal decision point.
            latest = self._sibling_latest_terminal.get(task_id)
            own_terminal = latest.get(self._agent_id) if latest is not None else None
            stance = self._sibling_last_stance.get(task_id)
            # Reuse this agent's REAL last visible_text (follow-up): the
            # cap forces terminality via the stance, but the debater's full-thought
            # answer must remain the actual agent content, never a placeholder.
            last_visible_text = self._sibling_last_visible_text.get(task_id)
            emit = own_terminal is False and stance is not None
            # Snapshot the origin authority under the lock (R1) so the
            # closing turn carries it even though the ``_task_origins`` entry is only
            # retired on the SUCCESS clear below.
            closing_origin = self._task_origins.get(task_id)
            if not emit:
                # Nothing to publish (already terminal, or no stance): the decision
                # is final, so clear the state as a unit now — no publish to gate on.
                self._clear_sibling_closing_state(task_id)
                return
            # Reserve the one-shot guard and provisionally mark this agent terminal
            # under the same lock as the check. Both are ROLLED BACK on a publish
            # failure so a retry re-evaluates, and made permanent by the success
            # clear that retires the decision state.
            self._sibling_closing_emitted.add(task_id)
            if latest is not None:
                latest[self._agent_id] = True
        assert stance is not None
        published = await self._publish_closing_turn(
            task_id, stance, last_visible_text, closing_origin
        )
        async with self._lock:
            if published:
                # Confirmed on the wire: retire the decision state as a unit now
                # (D7) — publish happened, so exactly one closing turn
                # exists and the state has done its job.
                self._clear_sibling_closing_state(task_id)
            else:
                # Publish failed: roll the reservation back so the NEXT observed
                # turn retries. Keep every decision-state entry (budget_spent, last
                # stance, latest-terminality, origin) intact.
                self._sibling_closing_emitted.discard(task_id)
                latest = self._sibling_latest_terminal.get(task_id)
                if latest is not None:
                    latest[self._agent_id] = False

    async def _publish_closing_turn(
        self,
        task_id: str,
        last_stance: Stance,
        last_visible_text: str | None,
        origin: ChannelOrigin | None,
    ) -> bool:
        """Publish the cap-induced closing terminal TurnFrame.

        The cap forces terminality, but a full-thought debater's FINAL budgeted
        turn is real agent content — Sprocket/Birk wrote a substantive paragraph,
        they just still ``needs_from`` a sibling. The typed stance carries the
        terminality (``needs_from=[]``); the ``visible_text`` REUSES this agent's
        latest real substantive answer (``last_visible_text``, captured in
        :meth:`_publish_turn`), so the closing turn renders the agent's own words
        rather than a synthetic ``"the round budget was reached"`` placeholder
        (follow-up). The round-budget status stays where it belongs: in
        the typed stance and the logs, never in the visible prose.

        Only when NO real text was ever captured (an agent that never published a
        visible turn for this task — not the full-thought debater case) does the
        frame fall back to a minimal non-empty placeholder so ``visible_text``
        stays a valid, non-empty field.

        ``origin`` is the task's channel origin, snapshotted by the caller under
        the lock (R1) so a triage edge can render this terminal closing
        turn back in-channel like a normal reply.

        Uses the CHECKED publish (D7): a transport failure PROPAGATES so
        this method can report it. Returns ``True`` when the frame is confirmed on
        the wire, ``False`` when the publish raised (caller re-arms for retry).
        The failure is logged here exactly once.
        """
        closing_stance = Stance(
            position=last_stance.position,
            because=last_stance.because,
            needs_from=[],
        )
        # Reuse the agent's real substantive text; only fall back when there is
        # genuinely none (never for a debater who published a paragraph).
        visible_text = (
            last_visible_text if last_visible_text and last_visible_text.strip() else "(no output)"
        )
        frame = TurnFrame(
            agent_id=self._agent_id,
            task_id=task_id,
            visible_text=visible_text,
            stance=closing_stance,
            refs=[task_id],
            origin=origin,
            verification=(
                VerificationReport(
                    status="not_applicable",
                    claim_kind="action",
                    ob_checked=False,
                    limitations=["debate round budget reached"],
                    methods=["runtime_cap_closure"],
                )
                if self._config.evidence_gate_enabled
                else None
            ),
        )
        subject = subjects.task_turn(task_id)
        try:
            await self._bus.publish_frame_checked(subject, frame)
        except Exception as exc:
            logger.error(
                "frame_turn.sibling | closing turn publish failed; will retry on next "
                "observed turn | subject={} reason={}",
                subject,
                exc,
            )
            return False
        logger.info(
            "frame_turn.sibling | closing terminal turn published (cap terminus) | "
            "agent={} task_id={} cap={}",
            self._agent_id,
            task_id,
            self._max_sibling_rounds,
        )
        return True

    async def _on_sibling_turn(self, lane_task_id: str, frame: Frame) -> None:
        """Inject a co-participant's TurnFrame into the live session.

        Guards, in order: the payload must be a ``TurnFrame``; its ``task_id``
        must match ``lane_task_id`` (the task the invoking subscription is bound
        to, so a payload cannot ride a stale/foreign lane — same contract as the
        steer lane); and its ``agent_id`` must NOT be this agent — an agent never
        feeds its OWN just-published turn back into itself (that echo would be
        pointless self-talk and could loop). A surviving sibling turn is rendered
        (visible text + typed stance) and handed to the session so the next round
        can reference the REAL sibling position. Done stays orchestrator-derived
        from the frame graph: this only supplies context; whether a follow-up
        turn is terminal is still decided by that turn's own ``stance.needs_from``
        (``is_done`` retracts terminality on a later non-terminal turn), so the
        fanout never double-counts or hangs completion.
        """
        if not isinstance(frame, TurnFrame):
            logger.warning(
                "frame_turn.sibling | non-turn frame on turn lane, ignoring | "
                "type={} lane_task_id={} self={}",
                type(frame).__name__,
                lane_task_id,
                self._agent_id,
            )
            return
        if frame.task_id != lane_task_id:
            logger.warning(
                "frame_turn.sibling | lane/payload task mismatch, turn IGNORED | "
                "author={} self={} task_id={} lane_task_id={}",
                frame.agent_id,
                self._agent_id,
                frame.task_id,
                lane_task_id,
            )
            return
        if frame.agent_id == self._agent_id:
            logger.debug(
                "frame_turn.sibling | own turn echo, not re-injected | self={} task_id={}",
                self._agent_id,
                frame.task_id,
            )
            return
        # This agent OBSERVED a sibling's turn on the shared lane: record its
        # latest-observed terminality for settle-detection. Done so
        # BEFORE the budget check so a below-cap consensus is caught even on the
        # delivery that would otherwise be dropped.
        settled = await self._record_observed_turn(frame)
        if settled:
            logger.info(
                "frame_turn.sibling | debate settled below cap, retiring lane | "
                "agent={} task_id={}",
                self._agent_id,
                frame.task_id,
            )
            await self._retire_exhausted_sibling_lane(frame.task_id, by_cap=False)
            return
        # STRUCTURED BOUNDED ROUNDS: reserve the round slot under the
        # SAME lock acquisition as the '< cap' check, BEFORE the await, so
        # concurrent sibling deliveries cannot each pass the check before any
        # increment (the old soft-cap overshoot). A reserved-but-unspent slot is
        # rolled back if the session send fails. Once this agent has reserved its
        # cap of sibling-triggered follow-ups, every later sibling turn is logged
        # (debug) and dropped — never re-pended — so the debate terminates. The
        # lane is retired the moment the budget is spent so it no longer outlives
        # the bounded debate.
        reserved = False
        async with self._lock:
            spent = self._sibling_round_counts.get(frame.task_id, 0)
            # ``_sibling_budget_spent`` also fails-closed the reserve: the cap-retire
            # path clears the round counter (lane-scoped), so a late concurrent
            # reserve must not re-pass the '< cap' check off a cleared counter and
            # overshoot. The budget-spent marker is retained until the closing
            # terminus resolves, so it holds the line across that window.
            if spent < self._max_sibling_rounds and frame.task_id not in self._sibling_budget_spent:
                self._sibling_round_counts[frame.task_id] = spent + 1
                reserved = True
        if not reserved:
            logger.debug(
                "frame_turn.sibling | round budget exhausted, sibling turn dropped | "
                "agent={} task_id={} author={} cap={}",
                self._agent_id,
                frame.task_id,
                frame.agent_id,
                self._max_sibling_rounds,
            )
            await self._retire_exhausted_sibling_lane(frame.task_id)
            return
        logger.info(
            "frame_turn.sibling | injecting sibling turn | agent={} task_id={} author={}",
            self._agent_id,
            frame.task_id,
            frame.agent_id,
        )
        sibling_turn = _render_sibling_turn(frame)
        async with self._lock:
            opens_new_turn = frame.task_id not in self._pending
            if self._config.evidence_gate_enabled and opens_new_turn:
                # A sibling's visible answer is evidence/context, not the original
                # human request. Never let sibling prose classify its own follow-up
                # as social/opinion and thereby bypass factual verification.
                self._grounding[frame.task_id] = GroundingState(cold=False, eligibility="factual")
        if self._config.evidence_gate_enabled and opens_new_turn:
            sibling_turn = await self._grounded_turn_text(
                frame.task_id,
                sibling_turn,
                frame.visible_text,
            )
        try:
            await self._supervisor.session.send(sibling_turn)
        except Exception as exc:
            logger.error(
                "frame_turn.sibling | session send failed | task_id={} author={} reason={}",
                frame.task_id,
                frame.agent_id,
                exc,
            )
            # Roll the reservation back: the follow-up turn never opened, so this
            # sibling delivery did not spend a round of the budget.
            async with self._lock:
                current = self._sibling_round_counts.get(frame.task_id, 0)
                if current > 0:
                    self._sibling_round_counts[frame.task_id] = current - 1
                if opens_new_turn:
                    self._grounding.pop(frame.task_id, None)
            return
        # The turn opened. A sibling arriving after this agent already finished
        # its round starts a fresh follow-up turn (mirrors the ingress steer
        # "nothing in flight -> register a pending context" path): re-register
        # the task as pending so the resulting TurnDone correlates and publishes
        # a real second-round TurnFrame. When a turn is still in flight the
        # sibling folds into it and no extra pending entry is added. Mirror the
        # normal TaskFrame pend path PRECISELY so the follow-up round is as
        # active — and as steerable — as the first: restore ``_steer_refs`` (the
        # active-task marker) AND re-establish the steer lane the first round's
        # TurnDone retired, reserving the slot under the lock exactly as
        # ``_on_task_frame`` does, so a concurrent redelivery cannot race a second
        # lane. The bounded-round slot was already reserved atomically above.
        exhausted_now = False
        needs_steer_resubscribe = False
        async with self._lock:
            if frame.task_id not in self._pending:
                self._pending.append(frame.task_id)
            self._steer_refs.setdefault(frame.task_id, [])
            # Re-pend restores the active-task markers ONLY. It NEVER writes origin
            # (R2): a sibling's TurnFrame carries the sibling's OWN origin,
            # and trusting it would let a hostile/foreign sibling frame become THIS
            # task's reply authority and misroute the answer to the sibling's
            # channel. The local ``_task_origins`` entry — written once at TaskFrame
            # pend time (the sole authority) and now surviving until the task is
            # fully retired (R1) rather than popped at first TurnDone — is still live
            # for a fanned task's follow-up round, so no restore is needed or safe.
            if frame.task_id not in self._steer_subscriptions:
                # Reserve the slot under the lock (empty = already-retired
                # placeholder) so the real subscription can land, matching the
                # first-round pend path's reserved-slot-under-lock discipline.
                self._steer_subscriptions[frame.task_id] = SubscriptionSet([])
                needs_steer_resubscribe = True
            if self._sibling_round_counts.get(frame.task_id, 0) >= self._max_sibling_rounds:
                exhausted_now = True
        if needs_steer_resubscribe:
            await self._subscribe_steer(frame.task_id)
        if exhausted_now:
            await self._retire_exhausted_sibling_lane(frame.task_id)

    def _steer_guard_ok(self, lane_task_id: str, frame: Frame) -> bool:
        """Return True and record the ref if ``frame`` is an acceptable steer.

        Guards, in order: the payload must be a ``SteerFrame``; its ``task_id``
        must match ``lane_task_id`` — the task the invoking subscription is bound
        to, so a payload cannot ride a stale or foreign lane; its
        ``target_agent`` must be this agent or the first-class
        :data:`BROADCAST_TARGET` (``"all"``, the edge default); and
        its ``task_id`` must be an active task. Any guard trigger logs a warning
        with the full typed context (target, self, task_id) and returns False —
        an ignored steer is never silent. On success the steer's ``task_id`` is
        appended to that task's frame-graph refs so the eventual
        ``TurnFrame.refs`` references the steer causally.
        """
        if not isinstance(frame, SteerFrame):
            logger.warning(
                "frame_turn.steer | non-steer frame on steer lane, ignoring | "
                "type={} lane_task_id={} self={}",
                type(frame).__name__,
                lane_task_id,
                self._agent_id,
            )
            return False
        if frame.task_id != lane_task_id:
            logger.warning(
                "frame_turn.steer | lane/payload task mismatch, steer IGNORED | "
                "target={} self={} task_id={} lane_task_id={}",
                frame.target_agent,
                self._agent_id,
                frame.task_id,
                lane_task_id,
            )
            return False
        if frame.target_agent not in (self._agent_id, BROADCAST_TARGET):
            logger.warning(
                "frame_turn.steer | explicit target mismatch, steer IGNORED | "
                "target={} self={} task_id={}",
                frame.target_agent,
                self._agent_id,
                frame.task_id,
            )
            return False
        if frame.task_id not in self._steer_refs:
            logger.warning(
                "frame_turn.steer | steer for non-active task, ignoring | "
                "target={} self={} task_id={}",
                frame.target_agent,
                self._agent_id,
                frame.task_id,
            )
            return False
        self._steer_refs[frame.task_id].append(frame.task_id)
        return True

    async def _on_steer_frame(self, lane_task_id: str, frame: Frame) -> None:
        """Inject a SteerFrame targeted at this agent into the LIVE session.

        Args:
            lane_task_id: The task id the invoking lane subscription is bound to.
            frame: The decoded frame delivered on that lane.
        """
        if not self._steer_guard_ok(lane_task_id, frame):
            return
        assert isinstance(frame, SteerFrame)
        logger.info(
            "frame_turn.steer | injecting | agent={} task_id={} interrupt={} len={}",
            self._agent_id,
            frame.task_id,
            frame.interrupt,
            len(frame.text),
        )
        if frame.interrupt:
            await self._interrupt_session(frame.task_id)
        try:
            await self._supervisor.session.send(frame.text)
        except Exception as exc:
            logger.error(
                "frame_turn.steer | session send failed | task_id={} reason={}",
                frame.task_id,
                exc,
            )

    async def interrupt_active_turn(self, task_id: str) -> None:
        """Abort the active turn for ``task_id`` (typed steer-driven interrupt).

        Guard: the task must be active; otherwise the call logs and no-ops. This
        is the seam a steer with ``interrupt=True`` drives, and the direct entry
        point exercised by tests.
        """
        if task_id not in self._steer_refs:
            logger.warning(
                "frame_turn.steer | interrupt for non-active task, ignoring | self={} task_id={}",
                self._agent_id,
                task_id,
            )
            return
        await self._interrupt_session(task_id)

    async def _interrupt_session(self, task_id: str) -> None:
        """Call the engine interrupt, logging failures (never raising to the bus)."""
        try:
            await self._supervisor.session.interrupt()
            logger.info("frame_turn.steer | interrupted active turn | task_id={}", task_id)
        except Exception as exc:
            logger.error("frame_turn.steer | interrupt failed | task_id={} reason={}", task_id, exc)

    async def _on_engine_event(self, event: EngineEvent) -> None:
        """Engine event sink: on TurnDone, publish the completed TurnFrame."""
        if isinstance(event, ToolUse):
            async with self._lock:
                task_id = self._pending[0] if self._pending else None
                state = self._grounding.get(task_id) if task_id is not None else None
                if state is not None:
                    state.observe_tool(event.name, event.tool_input)
            return
        if not isinstance(event, TurnDone):
            return
        async with self._lock:
            task_id = self._pending.popleft() if self._pending else None
            grounding = self._grounding.pop(task_id, None) if task_id is not None else None
            # The turn ended: read this task's channel origin. The
            # captured origin is copied onto the reply TurnFrame so a triage edge
            # can render this agent's answer back into the SAME channel the question
            # came from.
            #
            # ORIGIN AUTHORITY LIFETIME (R1): ``_task_origins`` is the sole
            # authority, written once at TaskFrame pend and popped ONLY when the task
            # is fully retired — NOT at first TurnDone. A fanned (multi-round) task's
            # cap-closing turn is published LATER (``_publish_closing_turn``) and
            # re-reads this map; popping here made that read empty so the round-capped
            # answer lost its origin and never rendered. So: a FANNED task (one whose
            # sibling closing-decision state is still alive) KEEPS its origin here —
            # it is popped as a unit with that state in ``_clear_sibling_closing_state``
            # (settle / cap-terminus) or ``_clear_sibling_task_state`` (failed-send
            # rollback). A NON-fanned (single) task has no such retirement path, so it
            # is retired right here at TurnDone.
            #
            # ``current_origin`` (FIFO-head discipline) is unaffected: after a
            # single task's TurnDone ``_pending`` no longer holds it, so the head read
            # returns the next task's origin (or None) regardless of a lingering entry.
            reply_origin = self._task_origins.get(task_id) if task_id is not None else None
            if task_id is not None and task_id not in self._sibling_latest_terminal:
                # Single task: no sibling closing-state owns this entry, retire now.
                self._task_origins.pop(task_id, None)
            # Retire the task's steer state as its turn ends; carry the accumulated
            # steer refs into the outbound TurnFrame so the graph shows causality.
            steer_refs = self._steer_refs.pop(task_id, []) if task_id is not None else []
            subscriptions = (
                self._steer_subscriptions.pop(task_id, None) if task_id is not None else None
            )
            persist_completions: dict[tuple[str, str], datetime] | None = None
            if task_id is not None:
                task_key = (task_id, self._agent_id)
                completed_at = monotonic()
                self._completed_tasks[task_key] = completed_at
                self._completed_tasks.move_to_end(task_key)
                # Durable dedup: record the WALL completion time and snapshot the
                # horizon-pruned map for the post-lock disk write.
                if self._completed_store is not None:
                    now = datetime.now(UTC)
                    self._persisted_completions[task_key] = now
                    self._persisted_completions = {
                        key: completed
                        for key, completed in self._persisted_completions.items()
                        if (now - completed).total_seconds() <= self._config.task_replay_horizon_s
                    }
                    persist_completions = dict(self._persisted_completions)
                while (
                    len(self._completed_tasks) > COMPLETED_TASK_CACHE_SIZE
                    and completed_at - next(iter(self._completed_tasks.values()))
                    > self._config.task_replay_horizon_s
                ):
                    self._completed_tasks.popitem(last=False)
                if len(self._completed_tasks) > COMPLETED_TASK_CACHE_SIZE:
                    logger.warning(
                        "frame_turn.task | completed cache above soft count bound to honor "
                        "replay horizon | agent={} entries={} bound={} horizon_s={}",
                        self._agent_id,
                        len(self._completed_tasks),
                        COMPLETED_TASK_CACHE_SIZE,
                        self._config.task_replay_horizon_s,
                    )
        if persist_completions is not None and self._completed_store is not None:
            # Write-on-completion: persist the horizon-pruned dedup map so a
            # restart within the horizon reloads it and does not re-dispatch an
            # already-completed redelivery. A write failure is logged
            # (raised by the store) but MUST NOT drop the turn — the in-memory
            # dedup still holds for this process; only cross-restart dedup is at
            # risk, and the loud error is the signal.
            try:
                self._completed_store.save(persist_completions, self._config.task_replay_horizon_s)
            except OSError as exc:
                logger.error(
                    "frame_turn.task | persist dedup state failed; cross-restart dedup at risk | "
                    "agent={} task_id={} reason={}",
                    self._agent_id,
                    task_id,
                    exc,
                )
        if task_id is None:
            logger.warning("frame_turn.event | turn_done with no pending task; dropping")
            return
        if grounding is not None:
            verification = merge_verification(event.verification, grounding)
            if needs_guarded_fallback(
                verification, cold=grounding.cold, eligibility=grounding.eligibility
            ):
                event = replace(
                    event,
                    text=(
                        "I can’t provide a verified factual answer yet because "
                        "verification was insufficient."
                    ),
                    verification=guarded_fallback_report(verification),
                )
            else:
                event = replace(event, verification=verification)
        # Retire the LIVE steer lane, not just the guard state — a stale lane
        # must stop delivering entirely. The sibling-turn lane
        # deliberately OUTLIVES a single TurnDone: a co-participant
        # can publish its turn after this agent's round completes, and that
        # sibling must still reach the session to drive a real follow-up round.
        # The lane is NOT retired here — it is retired on a failed-send rollback
        # (``_retire_failed_task``) or when the bounded-round budget is exhausted
        # (``_retire_exhausted_sibling_lane``) — so no round of the
        # bounded debate is missed but the lane never outlives it. Done stays
        # orchestrator-derived; a live lane never affects the frame-graph derivation.
        await self._retire_steer_subscription(task_id, subscriptions)
        await self._publish_turn(task_id, event, steer_refs, reply_origin)

    async def _publish_turn(
        self,
        task_id: str,
        turn: TurnDone,
        steer_refs: list[str],
        origin: ChannelOrigin | None,
    ) -> None:
        """Publish one guarded, typed TurnFrame for a completed turn.

        agentd emits NO done — done is orchestrator-derived from the frame graph.
        ``visible_text`` is guard-sanitized; ``stance`` is the engine-filled typed
        field carried straight through. ``refs`` starts with the task_id and
        appends each steer that landed during the turn, so the frame graph shows
        the steer→turn causality. ``origin`` is the triggering task's channel
        origin: copied onto the reply frame when the answered task
        arrived from an intake/wake path so a triage edge can render this answer
        back in-channel; ``None`` for a non-intake task (the frame omits it and
        the wire stays legacy-shaped).
        """
        if turn.is_error:
            # A failed turn's text is often the vendor's raw error body (URLs,
            # request ids). Chat gets a human one-liner; the payload lands in
            # logs ONCE here.
            logger.error(
                "frame_turn.turn | failed turn; payload withheld from chat | task_id={} payload={}",
                task_id,
                turn.text,
            )
            outbound_text = failure_visible_text(turn.text)
        else:
            outbound_text = turn.text
        sanitized = sanitize_outbound_text(outbound_text, platform="nats")
        assert isinstance(sanitized, str)
        frame = TurnFrame(
            agent_id=self._agent_id,
            task_id=task_id,
            visible_text=sanitized or "(no output)",
            stance=turn.stance,
            refs=[task_id, *steer_refs],
            origin=origin,
            verification=turn.verification,
        )
        subject = subjects.task_turn(task_id)
        try:
            await self._bus.publish_frame(subject, frame)
        except Exception as exc:
            logger.error("frame_turn.turn | publish failed | subject={} reason={}", subject, exc)
            return
        logger.info(
            "frame_turn.turn | published | subject={} visible_len={}",
            subject,
            len(frame.visible_text),
        )
        # This agent's OWN turn never re-enters through the echo-guarded sibling
        # handler, so record its latest-observed terminality + stance here. For a
        # fanned task this both feeds the cap-induced closing-turn decision and can
        # settle the debate below the cap when it makes every participant's latest
        # turn terminal.
        async with self._lock:
            # Record while the closing-decision state exists — that outlives the
            # lane on a cap retire so the final follow-up's stance still lands.
            # Only real sanitized output replaces the cached visible text; the
            # synthetic per-turn fallback must not erase an earlier answer.
            if task_id in self._sibling_latest_terminal:
                self._sibling_last_stance[task_id] = frame.stance
                if sanitized.strip():
                    self._sibling_last_visible_text[task_id] = sanitized
        settled = await self._record_observed_turn(frame)
        if settled:
            logger.info(
                "frame_turn.sibling | debate settled below cap on own turn, retiring lane | "
                "agent={} task_id={}",
                self._agent_id,
                task_id,
            )
            await self._retire_exhausted_sibling_lane(task_id, by_cap=False)
            return
        # This agent published a non-terminal turn. If the round budget is already
        # spent, the cap will drop every further sibling turn and this agent would
        # never publish a terminal turn on its own — orchestrator-derived done
        # would hang False forever. Emit the one-shot frame-level closing terminus.
        await self._emit_closing_turn_if_hung(task_id)

    async def publish_status(
        self,
        task_id: str,
        state: Literal["working", "input_required", "blocked"],
        *,
        phase: Literal["responding", "verifying"] | None = None,
    ) -> None:
        """Publish a typed StatusFrame on the per-task status lane.

        Used for the working-at-turn-start emit and for input_required/blocked
        engine-state transitions.
        """
        subject = subjects.task_status(task_id)
        frame = StatusFrame(task_id=task_id, agent_id=self._agent_id, state=state, phase=phase)
        try:
            await self._bus.publish_frame(subject, frame)
        except Exception as exc:
            logger.error(
                "frame_turn.status | publish failed | subject={} state={} reason={}",
                subject,
                state,
                exc,
            )
            return
        logger.debug("frame_turn.status | published | subject={} state={}", subject, state)
