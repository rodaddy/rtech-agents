"""Coordinate tasks by deriving done from the typed frame graph.

The orchestrator subscribes to task ingress, routes ``scope=single`` tasks
directly to agentd, accumulates each task's frame graph, and derives completion
via ``rtech_frames.done.is_done``. On derived done it re-publishes the task's
terminal turn onto the result lane for edges to render — orchestrator-selected
publication, never a new "done" token or frame field.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from loguru import logger
from rtech_frames import StatusFrame, TaskFrame, TerminalEvidence, TurnFrame, is_done
from rtech_frames.codec import Frame
from rtech_frames.subs import FrameSubscribeError, SubscriptionSet, subscribe_all

from . import subjects
from .config import OrchestratorConfig
from .fleet_signals import FleetSignalSource, evidence_from_envelope
from .graph import GraphStore, InMemoryGraphStore, TaskGraph
from .transport import FrameBus, FrameTransportError

if TYPE_CHECKING:
    from fleet_nats import Envelope

SCOPE_SINGLE = "single"


class Orchestrator:
    """Derived-done coordinator over a ``FrameBus``."""

    def __init__(
        self,
        *,
        bus: FrameBus,
        config: OrchestratorConfig,
        store: GraphStore | None = None,
        fleet_signals: FleetSignalSource | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            bus: The transport seam for publish/subscribe.
            config: The routing config carrying the explicit single-scope target
                id. Required — there is no hardcoded default target.
            store: The graph store; defaults to an in-memory POC store.
            fleet_signals: The fleet-side completion seam for BRIDGED
                participants. Required whenever
                ``config.bridged_agents`` is non-empty and a task dispatches to
                one — a bridged dispatch with no signal source fails closed.
        """
        self._bus: FrameBus = bus
        self._store: GraphStore = store if store is not None else InMemoryGraphStore()
        self._config: OrchestratorConfig = config
        self._fleet_signals: FleetSignalSource | None = fleet_signals
        # task_id -> awaitable unsubscribe for that task's fleet done lane.
        self._fleet_unsubs: dict[str, Callable[[], Awaitable[None]]] = {}
        # ONE handle per task covering its turn+status reply lanes, installed
        # atomically via ``subscribe_all``: a partial-subscribe
        # failure rolls the whole set back so an earlier lane never survives
        # live-but-untracked. The lanes themselves live for the orchestrator's
        # lifetime (unchanged from the prior sequential subscribe — a late turn
        # or status after done must still land to be safely ignored, never error
        # on a retired lane); the retained ``SubscriptionSet`` is the single
        # honest retire handle, replacing the untracked per-lane subscribes.
        self._lane_subscriptions: dict[str, SubscriptionSet] = {}
        # Un-retirable live subscription sets whose rollback left a handle live.
        # This is a stash, not a retry queue: nothing here is retried
        # automatically. It exists so an un-retirable live lane stays referenced
        # and observable (logged at stash time) instead of escaping untracked.
        self._leaked_subscriptions: dict[str, SubscriptionSet] = {}
        self._finished: set[str] = set()
        self._route_by_scope: dict[str, Callable[[TaskFrame], Awaitable[None]]] = {
            SCOPE_SINGLE: self._route_single,
            "duo": self._route_multi,
            "trio": self._route_multi,
        }

    async def start(self) -> None:
        """Subscribe to task ingress and begin coordinating."""
        logger.info("orchestrator starting", ingress=subjects.TASK_INGRESS)
        await self._bus.subscribe_frame(subjects.TASK_INGRESS, self._on_task)

    async def _on_task(self, frame: Frame) -> None:
        """Handle a task-ingress frame: create graph, subscribe lanes, route."""
        if not isinstance(frame, TaskFrame):
            logger.warning("ignoring non-task frame on ingress", frame_type=type(frame).__name__)
            return
        task = frame
        if task.stack != self._config.stack_id:
            # Ingress stack guard: a stack only consumes tasks
            # addressed to it. Unscoped legacy frames ("") also fail closed.
            logger.warning(
                "ignoring task for foreign stack",
                task_id=task.id,
                task_stack=task.stack,
                stack_id=self._config.stack_id,
            )
            return
        existing = self._store.get(task.id)
        if existing is not None:
            # Ingress is idempotent: the graph is claimed
            # atomically below before any await, so a redelivery of the same
            # frame is a no-op — never a second subscription or dispatch — and
            # a same-id/different-payload frame is rejected loudly.
            if existing.task == task:
                logger.info("duplicate task delivery; no-op", task_id=task.id)
            else:
                logger.error(
                    "task id reuse with different payload; rejecting",
                    task_id=task.id,
                )
            return
        logger.info("received task", task_id=task.id, scope=task.scope, stack=task.stack)
        graph = self._store.create(task)
        try:
            await self._subscribe_lanes(task.id)
        except FrameSubscribeError as exc:
            # Atomic rollback ran; anything it could not retire rides on
            # ``exc.leaked`` and must never be discarded. Stash it (referenced +
            # logged, no automatic retry) so the live lane stays observable, then
            # fail the dispatch closed.
            if len(exc.leaked) > 0:
                self._leaked_subscriptions[task.id] = exc.leaked
                logger.critical(
                    "lane subscription rollback incomplete; live subscription(s) held",
                    task_id=task.id,
                    leaked=len(exc.leaked),
                    subjects=exc.leaked.subjects,
                )
            logger.error("lane subscription failed", task_id=task.id, error=str(exc))
            graph.mark_dispatch_failed()
            return
        route = self._route_by_scope.get(task.scope)
        if route is None:
            logger.warning("no route for scope", task_id=task.id, scope=task.scope)
            return
        await route(task)

    def _turn_handler(self, task_id: str) -> Callable[[Frame], Awaitable[None]]:
        """Return a turn-lane handler bound to a task id."""

        async def handle(frame: Frame) -> None:
            await self._on_turn(task_id, frame)

        return handle

    def _status_handler(self, task_id: str) -> Callable[[Frame], Awaitable[None]]:
        """Return a status-lane handler bound to a task id."""

        async def handle(frame: Frame) -> None:
            await self._on_status(task_id, frame)

        return handle

    async def _subscribe_lanes(self, task_id: str) -> None:
        """Install a task's turn + status reply lanes as ONE atomic set.

        The two ``(subject, handler)`` entries are handed to the typed-bus
        :func:`~rtech_frames.subs.subscribe_all` primitive, which subscribes them
        in order and rolls the whole set back on ANY failure (raising
        :class:`~rtech_frames.subs.FrameSubscribeError`). Replaces the prior
        hand-rolled sequential subscribe, where a status-lane failure left the
        turn lane live-but-untracked with no retire handle. The returned
        :class:`~rtech_frames.subs.SubscriptionSet` is the single retire handle
        kept for this task.

        Raises:
            FrameSubscribeError: If any lane subscribe failed; the primitive has
                already attempted rollback and any still-live handle rides on the
                error's ``leaked`` set.
        """
        entries = [
            (subjects.task_turn(task_id), self._turn_handler(task_id)),
            (subjects.task_status(task_id), self._status_handler(task_id)),
        ]
        self._lane_subscriptions[task_id] = await subscribe_all(
            self._bus.subscribe_frame_tracked, entries
        )
        logger.debug("lane subscriptions installed", task_id=task_id)

    async def _route_single(self, task: TaskFrame) -> None:
        """Route a single-scope task to the named participant, else the default.

        When the task names a participant, ``participants[0]`` is the target;
        otherwise the configured default single-agent id is used (unchanged
        behavior when no participant is named). The bridged fleet-done watch is
        installed BEFORE the publish: a fast gateway can emit
        its one-shot typed ``Done`` the moment the dispatch lands, and a
        subscription installed after the publish misses it forever. A failed
        or unconfirmed publish rolls the watch back and marks the task
        dispatch-failed. The dispatch record is written only AFTER confirmed
        server hand-off, so done can never derive against an agent
        that may never receive the task.
        """
        target = task.participants[0] if task.participants else self._config.single_agent_id
        subject = subjects.agent_task(target)
        logger.info("routing single task to agent", task_id=task.id, subject=subject, target=target)
        if not await self._watch_bridged(task.id, frozenset({target})):
            return
        dispatched = await self._dispatch_batch(task, (target,))
        if not dispatched:
            await self._drop_fleet_watch(task.id)
            return
        self._record_dispatch(task.id, frozenset({target}))
        await self._check_completion(task.id)

    async def _route_multi(self, task: TaskFrame) -> None:
        """Fan a duo/trio task to each named participant's agent lane.

        The ``TaskFrame`` is published verbatim to every ``participants`` agent's
        ``rtech.agent.<id>.task`` lane. Done is derived from the accumulated
        frame graph via ``is_done`` (duo/trio count semantics), so no completion
        signal is injected here.

        The bridged fleet-done watch for EVERY bridged participant is installed
        BEFORE the first publish: any published participant's
        gateway can emit its one-shot typed ``Done`` immediately, so the
        subscription must exist before any dispatch lands. The batch follows
        publish -> confirm -> record (dispatch-confirmation contract): every
        lane publish is enqueued, one server confirmation covers the batch,
        and only then is the full participant set recorded. Any publish or
        confirmation failure is an explicit partial failure: the task is
        marked dispatch-failed, the fleet watch is rolled back, and the task
        fails closed instead of waiting silently or completing on a subset.
        """
        if not task.participants:
            logger.warning(
                "multi-agent task has no participants; nothing to fan out",
                task_id=task.id,
                scope=task.scope,
            )
            return
        logger.info(
            "fanning task to participants",
            task_id=task.id,
            scope=task.scope,
            count=len(task.participants),
        )
        if not await self._watch_bridged(task.id, frozenset(task.participants)):
            return
        dispatched = await self._dispatch_batch(task, tuple(task.participants))
        if not dispatched:
            await self._drop_fleet_watch(task.id)
            return
        self._record_dispatch(task.id, frozenset(task.participants))
        await self._check_completion(task.id)

    async def _dispatch_batch(self, task: TaskFrame, targets: tuple[str, ...]) -> bool:
        """Publish a task to each target lane, then confirm the batch.

        Returns:
            ``True`` when every publish was enqueued AND the batch was
            confirmed with the server — only then may the caller write the
            dispatch record. On any typed transport failure the task is marked
            dispatch-failed (rolling back any recorded subset) and ``False``
            is returned.
        """
        try:
            for target in targets:
                subject = subjects.agent_task(target)
                logger.info(
                    "publishing task to agent lane",
                    task_id=task.id,
                    subject=subject,
                    target=target,
                )
                await self._bus.publish_frame(subject, task)
            await self._bus.confirm_publishes()
        except FrameTransportError as exc:
            logger.error(
                "dispatch hand-off unconfirmed; failing closed",
                task_id=task.id,
                targets=list(targets),
                error=str(exc),
            )
            self._mark_dispatch_failed(task.id)
            return False
        return True

    async def _on_turn(self, task_id: str, frame: Frame) -> None:
        """Handle a turn frame: accumulate, then check for derived done."""
        if not isinstance(frame, TurnFrame):
            logger.warning("ignoring non-turn frame on turn lane", task_id=task_id)
            return
        if frame.task_id != task_id:
            # Lane/payload identity must agree: a frame
            # delivered on task A's subject carrying task B's id never lands
            # in A's graph.
            logger.warning(
                "rejecting turn with mismatched task id",
                lane_task_id=task_id,
                frame_task_id=frame.task_id,
            )
            return
        graph = self._store.get(task_id)
        if graph is None:
            logger.warning("turn for unknown task", task_id=task_id)
            return
        graph.add_turn(frame)
        await self._maybe_finish(graph)

    async def _on_status(self, task_id: str, frame: Frame) -> None:
        """Handle a status frame: accumulate, then check for derived done."""
        if not isinstance(frame, StatusFrame):
            logger.warning("ignoring non-status frame on status lane", task_id=task_id)
            return
        if frame.task_id != task_id:
            logger.warning(
                "rejecting status with mismatched task id",
                lane_task_id=task_id,
                frame_task_id=frame.task_id,
            )
            return
        graph = self._store.get(task_id)
        if graph is None:
            logger.warning("status for unknown task", task_id=task_id)
            return
        graph.add_status(frame)
        await self._maybe_finish(graph)

    async def _watch_bridged(self, task_id: str, dispatched: frozenset[str]) -> bool:
        """Start consuming the typed fleet Done for a task's bridged participants.

        Called BEFORE any dispatch publish: the gateway's
        typed ``Done`` is one-shot, and a fast completion emitted the instant
        the dispatch lands must already have a live subscription to land on.
        The bridged subset of ``dispatched`` (per ``config.bridged_agents``) is
        recorded on the graph, and ONE fleet-done-lane subscription per task is
        established through the injected seam. Fails closed: a bridged dispatch
        with no signal source, or a failed subscription, marks the task
        dispatch-failed BEFORE anything is published — the orchestrator could
        never observe the participant's terminal evidence, so a loud failure
        beats an eternal silent wait.

        Args:
            task_id: The task about to be dispatched.
            dispatched: The participant ids the task will be dispatched to.

        Returns:
            True when the router may proceed to publish (watch installed, or no
            bridged participants); False when the task failed closed.
        """
        bridged = dispatched & self._config.bridged_agents
        if not bridged:
            return True
        graph = self._store.get(task_id)
        if graph is None:
            logger.warning("bridged watch for unknown task graph", task_id=task_id)
            return False
        graph.record_bridged(bridged)
        if self._fleet_signals is None:
            logger.error(
                "bridged participants dispatched but no fleet signal source; failing closed",
                task_id=task_id,
                bridged=sorted(bridged),
            )
            graph.mark_dispatch_failed()
            return False
        if task_id in self._fleet_unsubs:
            logger.debug("fleet done lane already watched", task_id=task_id)
            return True

        async def _on_envelope(envelope: Envelope) -> None:
            await self._on_fleet_envelope(task_id, envelope)

        try:
            self._fleet_unsubs[task_id] = await self._fleet_signals.subscribe_done(
                task_id, _on_envelope
            )
        except Exception as exc:
            logger.error(
                "fleet done subscription failed; failing closed",
                task_id=task_id,
                error=str(exc),
            )
            graph.mark_dispatch_failed()
            return False
        return True

    async def _on_fleet_envelope(self, task_id: str, envelope: Envelope) -> None:
        """Record typed fleet Done as terminal evidence, then re-derive done.

        The typed signal is consumed as a typed INPUT: validated fail-closed by
        :func:`evidence_from_envelope` and appended to the graph as
        ``TerminalEvidence`` — never re-authored as a ``TurnFrame``.
        """
        graph = self._store.get(task_id)
        if graph is None:
            logger.warning("fleet envelope for unknown task", task_id=task_id)
            return
        evidence = evidence_from_envelope(envelope=envelope, task_id=task_id, bridged=graph.bridged)
        if evidence is None:
            return
        graph.add_terminal_evidence(evidence)
        await self._maybe_finish(graph)

    def _record_dispatch(self, task_id: str, participants: frozenset[str]) -> None:
        """Record the dispatched participant set on the task's graph.

        The dispatched set is the typed identity source for participant-scoped
        done derivation.
        """
        graph = self._store.get(task_id)
        if graph is None:
            logger.warning("dispatch for unknown task graph", task_id=task_id)
            return
        graph.record_dispatch(participants)

    def _mark_dispatch_failed(self, task_id: str) -> None:
        """Latch the explicit dispatch-failure state on the task's graph."""
        graph = self._store.get(task_id)
        if graph is None:
            logger.warning("dispatch failure for unknown task graph", task_id=task_id)
            return
        graph.mark_dispatch_failed()

    async def _check_completion(self, task_id: str) -> None:
        """Re-derive done after the dispatch record lands (race-safe check).

        Over a real transport a participant's turn can arrive between the
        dispatch publish and the dispatch record; that turn's ``_maybe_finish``
        ran against an empty participant set and skipped. Re-checking here
        closes that window.
        """
        graph = self._store.get(task_id)
        if graph is None:
            logger.warning("completion check for unknown task graph", task_id=task_id)
            return
        await self._maybe_finish(graph)

    async def _maybe_finish(self, graph: TaskGraph) -> None:
        """Derive done; on completion publish the terminal turn to the result lane."""
        task = graph.task
        if task.id in self._finished:
            logger.debug("task already finished; skipping", task_id=task.id)
            return
        if graph.dispatch_failed:
            # Explicit partial/total dispatch failure: the task fails closed
            # and never derives done from whatever subset did respond.
            logger.debug("task dispatch failed; skipping done check", task_id=task.id)
            return
        if not graph.participants:
            # Fail closed: without a dispatched participant set
            # there is no identity to derive done against.
            logger.debug("task not dispatched yet; skipping done check", task_id=task.id)
            return
        if not is_done(task, graph.events, participants=graph.participants, bridged=graph.bridged):
            return
        terminal = self._latest_terminal_turn(graph)
        if terminal is None:
            # A bridged participant's typed Done can outrun its content turn
            # across subjects; the next turn arrival re-runs this check, so the
            # task never finishes without a result turn.
            logger.warning("derived done but no terminal turn to publish", task_id=task.id)
            return
        self._finished.add(task.id)
        await self._drop_fleet_watch(task.id)
        subject = subjects.task_result(task.id)
        logger.info("task derived done -> result published", task_id=task.id, subject=subject)
        await self._bus.publish_frame(subject, terminal)

    async def _drop_fleet_watch(self, task_id: str) -> None:
        """Unsubscribe a finished task's fleet done lane, if one was watched."""
        unsubscribe = self._fleet_unsubs.pop(task_id, None)
        if unsubscribe is None:
            return
        try:
            await unsubscribe()
        except Exception as exc:
            logger.error("fleet done unsubscribe failed", task_id=task_id, error=str(exc))

    @staticmethod
    def _latest_terminal_turn(graph: TaskGraph) -> TurnFrame | None:
        """Return the most recent turn from a terminally-evidenced participant.

        Turns from agents outside the dispatched set are never published as the
        task's result. Terminality is per participant kind: a
        native participant's turn counts when its ``needs_from`` is empty; a
        BRIDGED participant's turns are content-only, so its latest turn counts
        once its orchestrator-recorded ``TerminalEvidence`` exists
        — never on the turn's own stance fields.
        """
        evidenced = {
            event.agent_id
            for event in graph.events
            if isinstance(event, TerminalEvidence) and event.agent_id in graph.bridged
        }
        for turn in reversed(graph.turns):
            if turn.agent_id not in graph.participants:
                continue
            if turn.agent_id in graph.bridged:
                if turn.agent_id in evidenced:
                    return turn
                continue
            if not turn.stance.needs_from:
                return turn
        return None
