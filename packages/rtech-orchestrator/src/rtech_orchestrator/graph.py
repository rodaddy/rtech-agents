"""Accumulate and store the per-task frame graph.

``TaskGraph`` collects the turns and statuses for one task as a single flat,
time-ordered event log and exposes it via ``events`` for
``rtech_frames.done.is_done``. ``GraphStore`` is the Protocol seam so a durable
store can replace the POC-only ``InMemoryGraphStore`` without touching the
orchestrator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from loguru import logger
from rtech_frames import StatusFrame, TaskFrame, TerminalEvidence, TurnFrame

if TYPE_CHECKING:
    from collections.abc import Collection


class TaskGraph:
    """Per-task accumulator of turn and status frames."""

    def __init__(self, task: TaskFrame) -> None:
        """Initialize the graph from its originating task frame.

        Args:
            task: The task frame whose lanes this graph accumulates.
        """
        self.task: TaskFrame = task
        self._events: list[TurnFrame | StatusFrame | TerminalEvidence] = []
        self._participants: frozenset[str] = frozenset()
        self._bridged: frozenset[str] = frozenset()
        self._dispatch_failed: bool = False

    def record_dispatch(self, participants: Collection[str]) -> None:
        """Record the agent ids this task was actually dispatched to.

        The dispatched set is the typed identity source for participant-scoped
        done derivation (issue #75): ``is_done`` counts terminal turns only
        from these agents. Recording is cumulative so a multi-lane dispatch may
        record per publish or once with the full set.

        Args:
            participants: The agent ids the orchestrator routed the task to.
        """
        logger.debug("record_dispatch", task_id=self.task.id, participants=sorted(participants))
        self._participants = self._participants | frozenset(participants)

    @property
    def participants(self) -> frozenset[str]:
        """Return the dispatched participant set (empty until dispatch)."""
        return self._participants

    def record_bridged(self, participants: Collection[str]) -> None:
        """Record which dispatched participants are fronted by a bridge.

        Bridged participants (PR #73 redesign) complete via orchestrator-
        consumed :class:`~rtech_frames.TerminalEvidence`, never via their own
        turn terminality. Cumulative, like :meth:`record_dispatch`.

        Args:
            participants: The bridged agent ids this task dispatched to.
        """
        logger.debug("record_bridged", task_id=self.task.id, participants=sorted(participants))
        self._bridged = self._bridged | frozenset(participants)

    @property
    def bridged(self) -> frozenset[str]:
        """Return the bridged subset of the dispatched participants."""
        return self._bridged

    def add_terminal_evidence(self, evidence: TerminalEvidence) -> None:
        """Append orchestrator-recorded terminal evidence to the event log.

        Only the orchestrator creates evidence, from a typed fleet ``Done``
        signal it consumed on its own subscription; sequence position in the
        log is load-bearing exactly like a turn's.

        Args:
            evidence: The typed terminal-evidence record.
        """
        logger.debug(
            "add_terminal_evidence",
            task_id=self.task.id,
            agent_id=evidence.agent_id,
            source=evidence.source,
        )
        self._events.append(evidence)

    def mark_dispatch_failed(self) -> None:
        """Latch the dispatch-failure state and roll back any recorded subset.

        Set when a dispatch publish, publish confirmation, or lane
        subscription fails, including a PARTIAL multi-lane fan-out failure
        (issue #75 rounds 2-3): agents the task never CONFIRMABLY reached must
        not count toward done. Any participants recorded before the failure
        are rolled back (dispatch-confirmation contract) so the graph carries
        no identity a later turn could complete against, and the task fails
        closed loudly instead of waiting forever or completing on a subset.
        """
        logger.error(
            "dispatch failed; task fails closed",
            task_id=self.task.id,
            rolled_back=sorted(self._participants),
        )
        self._participants = frozenset()
        self._dispatch_failed = True

    @property
    def dispatch_failed(self) -> bool:
        """Return whether dispatch failed; a failed task never derives done."""
        return self._dispatch_failed

    def add_turn(self, turn: TurnFrame) -> None:
        """Append a turn frame to the event log.

        Args:
            turn: The turn frame to accumulate.
        """
        logger.debug("add_turn", task_id=self.task.id, agent_id=turn.agent_id)
        self._events.append(turn)

    def add_status(self, status: StatusFrame) -> None:
        """Append a status frame to the event log.

        Args:
            status: The status frame to accumulate.
        """
        logger.debug("add_status", task_id=self.task.id, state=status.state)
        self._events.append(status)

    @property
    def events(self) -> list[TurnFrame | StatusFrame | TerminalEvidence]:
        """Return the flat, time-ordered event log for ``is_done``.

        Turns, statuses, and terminal evidence are interleaved in true
        arrival order.
        """
        return list(self._events)

    @property
    def turns(self) -> list[TurnFrame]:
        """Return every accumulated turn, in arrival order."""
        return [e for e in self._events if isinstance(e, TurnFrame)]


class GraphStore(Protocol):
    """Storage seam for per-task graphs; durable stores implement this."""

    def get(self, task_id: str) -> TaskGraph | None:
        """Return the graph for a task id, or ``None`` if absent."""
        ...

    def create(self, task: TaskFrame) -> TaskGraph:
        """Create (idempotently) and return the graph for a task."""
        ...

    def has(self, task_id: str) -> bool:
        """Return whether a graph exists for the task id."""
        ...


class InMemoryGraphStore:
    """POC-only in-memory ``GraphStore`` backed by a dict."""

    def __init__(self) -> None:
        """Initialize the empty in-memory store."""
        self._graphs: dict[str, TaskGraph] = {}

    def get(self, task_id: str) -> TaskGraph | None:
        """Return the graph for a task id, or ``None`` if absent."""
        return self._graphs.get(task_id)

    def create(self, task: TaskFrame) -> TaskGraph:
        """Create the graph for a task, idempotently.

        Args:
            task: The originating task frame.

        Returns:
            The existing graph if the task id is already present, else a new
            ``TaskGraph`` that is stored and returned.
        """
        existing = self._graphs.get(task.id)
        if existing is not None:
            logger.debug("graph create is idempotent no-op", task_id=task.id)
            return existing
        logger.info("creating task graph", task_id=task.id, scope=task.scope)
        graph = TaskGraph(task)
        self._graphs[task.id] = graph
        return graph

    def has(self, task_id: str) -> bool:
        """Return whether a graph exists for the task id."""
        return task_id in self._graphs
