"""Derive orchestrator-owned done state from the frame graph.

Done is NOT a frame and NOT a prose trailer. Agents never emit done; the
orchestrator derives it from typed frame fields only.

Graph representation choice: ``Sequence[TurnFrame | StatusFrame]`` -- a flat,
time-ordered event log. Rationale: this matches codec output (each decoded
frame is one typed event) and the NATS bus semantics, where frames arrive in
publish order. Sequence position is therefore load-bearing: it orders turns and
statuses so supersession can be decided by "which came later", not adjacency.

Participant-scoped derivation: done derives ONLY from the task's
dispatched participant set -- the typed identities the orchestrator actually
routed the task to (or ``task.participants`` when the frame names them).
Counting distinct terminal agents against a scope COUNT is identity-blind: on a
shared bus a foreign stack's agent can land a terminal turn on the task's turn
lane and satisfy the count before the dispatched agent finishes (defect B).
Terminal turns from non-participants are ignored, and with no known participant
set the derivation FAILS CLOSED. ``task.scope`` remains a consistency check:
a participant set whose size does not match the scope count also fails closed.

Status authorship (frame contract amendment): ``StatusFrame``
carries a required ``agent_id`` mirroring ``TurnFrame``, so status is evaluated
PER PARTICIPANT. Rounds 3/4 proved the task-wide alternative unsound: one
``completion_by_evidence`` boolean classified every later ``working`` status by
whichever participant happened to complete the set, so mixed native+bridged
orderings either weakened native semantics or reopened the cross-subject
contradiction. Statuses from agents outside the dispatched participant set are
ignored (identity-scoped, exactly like turns --).

Sequence-aware supersession: all three ``StatusFrame`` states
(``working``/``input_required``/``blocked``) are non-terminal and there is NO
terminal status value -- agentd emits ``StatusFrame(state="working")`` at turn
START, before the terminal ``TurnFrame``. Treating the latest status
order-blind would make ``[working, terminal-turn]`` derive not-done forever, so
done would never be derived over the real wire. Instead we are SEQUENCE-AWARE
per participant: a non-terminal status blocks done only while it is LATER in
the log than its AUTHOR's terminal-establishing event (terminal turn or
terminal evidence). A status at or before its author's terminal event is
superseded by that completion and does not block; the pre-turn ``working``
status agentd emits is superseded this way, and a later ``working`` reopens
ONLY its author's terminality -- never another participant's. We do NOT add a
terminal status value or a completion sentinel -- the StatusFrame state enum is
unchanged.

Bridged participants (see docs/bridged-participant-lifecycle.md):
a participant fronted by a bus-side bridge (an existing Hermes gateway agent)
never emits its own rtech frames -- the bridge translates its reply CONTENT into
``TurnFrame``s whose stance fields are a verbatim copy of the gateway's typed
stance. Those turns carry no terminal semantics: for a bridged participant,
turn terminality is IGNORED in both directions (it can neither complete nor
retract). The participant's terminal evidence is ONLY a :class:`TerminalEvidence`
entry -- the orchestrator's typed in-graph record of the fleet-side typed
``Done`` signal it consumed directly. ``TerminalEvidence`` is deliberately NOT a
wire frame: it has no codec registration and can never be published; terminal
semantics for bridged participants therefore originate in exactly one place,
the orchestrator.

Evidence-established terminality vs status supersession:
``TerminalEvidence`` arrives on the orchestrator's OWN fleet-side connection,
while the bridge's frames (including the ``working`` status it pairs with each
content turn) ride the rtech connection. The merged log's relative order
between a participant's evidence and its OWN statuses is therefore transport
noise, not source order -- and at the source the gateway emits its typed
``Done`` strictly after every reply that produced a paired status. So for a
participant whose terminality is EVIDENCE-established, a later ``working``
status it authored is a cross-connection stale echo and does NOT block done --
otherwise the paired status either deadlocks the task or contradicts an
already-published result. ``blocked`` and ``input_required`` are escalations,
not liveness echoes, and still block in the fail-closed direction. A
participant whose terminality is TURN-established keeps the full
status-after-terminal block: its statuses share its turn's connection, so log
order is sound. Because the classification is keyed by the status author, the
derivation is order-independent across participants: another participant's
frames can never flip this participant's staleness classification (the round-4
mixed-ordering regression). Retraction via a later non-terminal TURN is
unaffected either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from .frames import SCOPE_PARTICIPANT_COUNT, StatusFrame, TurnFrame

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from .frames import TaskFrame

# States that mean the task is still in flight and cannot be done.
_NON_TERMINAL_STATES: frozenset[str] = frozenset({"working", "input_required", "blocked"})


@dataclass(frozen=True)
class TerminalEvidence:
    """Record the orchestrator's typed observation of a bridged participant's end.

    This is an in-graph event, NOT a wire frame: it is never encoded, never
    published, and only the orchestrator creates it -- from a typed
    ``fleet_core.spec.Done`` signal it consumed on its own fleet-side
    subscription. Its sequence position in the frame graph is load-bearing,
    exactly like a turn's.

    Attributes:
        agent_id: The bridged participant this evidence terminates.
        task_id: The rtech task the evidence belongs to.
        source: Where the typed signal came from (e.g. ``"fleet-done"``).
        outcome: The typed outcome the signal carried.
    """

    agent_id: str
    task_id: str
    source: str
    outcome: str = "completed"


@dataclass
class _ParticipantState:
    """Track one dispatched participant's derivation state.

    Attributes:
        terminal_index: Log position of the event establishing the
            participant's CURRENT terminality (terminal turn for native,
            terminal evidence for bridged); ``None`` while not terminal.
        by_evidence: Whether that terminality is evidence-established, which
            decides the participant's own staleness classification for later
            statuses.
        last_status_index: Log position of the participant's latest
            non-terminal status of any state.
        last_escalation_index: Log position of the participant's latest
            ``blocked``/``input_required`` escalation.
    """

    terminal_index: int | None = None
    by_evidence: bool = False
    last_status_index: int | None = None
    last_escalation_index: int | None = None


def _resolve_participants(
    task: TaskFrame, participants: Collection[str] | None
) -> frozenset[str] | None:
    """Resolve the required participant identity set, or ``None`` to fail closed.

    The explicit ``participants`` argument (the orchestrator's dispatched set)
    wins; otherwise the task frame's own ``participants`` field is used. With
    neither available there is no identity set to derive against -- the
    identity-blind count path was defect B and is removed.
    """
    if participants:
        return frozenset(participants)
    if task.participants:
        return frozenset(task.participants)
    logger.warning(
        "no participant identities for task; failing closed",
        task_id=task.id,
        scope=task.scope,
    )
    return None


def is_done(
    task: TaskFrame,
    frame_graph: Sequence[TurnFrame | StatusFrame | TerminalEvidence],
    *,
    participants: Collection[str] | None = None,
    bridged: Collection[str] | None = None,
) -> bool:
    """Derive whether a task is complete from its frame graph.

    Done is true when EVERY dispatched participant's LATEST ``TurnFrame`` for
    ``task.id`` is terminal (``stance.needs_from`` empty) — a later non-terminal
    turn retracts a participant's earlier terminality, so terminal state is
    never sticky — AND no participant is reopened by a non-terminal
    ``StatusFrame`` it authored later in the log than its own
    terminal-establishing event. A status at or before its author's terminal
    event is superseded by that completion and does not block; a later one
    reopens ONLY its author (per-participant, keyed by the
    status frame's ``agent_id``). One carve-out: a participant whose
    terminality is EVIDENCE-established is not reopened by its own later
    ``working`` status — evidence rides the orchestrator's own fleet
    connection, so its log order against bridge-published statuses is
    transport noise (see the module docstring) — while
    ``blocked``/``input_required`` escalations still block fail-closed. Turns
    and statuses from agents outside the participant set NEVER count.

    BRIDGED participants complete differently: their turns
    are content only -- terminality is decided EXCLUSIVELY by a
    :class:`TerminalEvidence` entry the orchestrator recorded from the typed
    fleet ``Done`` signal. A bridged participant's turn can neither complete
    nor retract; evidence naming a non-bridged agent is ignored (fails closed
    against spoofed or misrouted signals).

    Args:
        task: The task whose completion is being derived.
        frame_graph: Time-ordered event log of turn/status frames and
            orchestrator-recorded terminal evidence.
        participants: The dispatched participant agent ids, from the
            orchestrator's typed dispatch record. When omitted, the task
            frame's own ``participants`` field is used.
        bridged: The subset of ``participants`` fronted by a bridge, from the
            same dispatch record. Must be a subset of the resolved participant
            set; anything else fails closed.

    Returns:
        True if the task is derived done, otherwise False. Fails closed
        (returns False) on an unrecognized scope, on an empty/unknown
        participant set, on a participant set inconsistent with the scope
        count, or on a bridged set that is not a subset of the participants.
    """

    logger.debug("deriving done state", task_id=task.id, scope=task.scope)

    required_count = SCOPE_PARTICIPANT_COUNT.get(task.scope)
    if required_count is None:
        logger.warning("unrecognized task scope; failing closed", task_id=task.id, scope=task.scope)
        return False

    required = _resolve_participants(task, participants)
    if required is None:
        return False
    if len(required) != required_count:
        logger.warning(
            "participant set size does not match scope; failing closed",
            task_id=task.id,
            scope=task.scope,
            participant_count=len(required),
            required_count=required_count,
        )
        return False

    bridged_set = frozenset(bridged) if bridged else frozenset()
    if not bridged_set <= required:
        logger.warning(
            "bridged set is not a subset of participants; failing closed",
            task_id=task.id,
            bridged=sorted(bridged_set),
            participants=sorted(required),
        )
        return False

    states: dict[str, _ParticipantState] = {agent_id: _ParticipantState() for agent_id in required}

    for index, frame in enumerate(frame_graph):
        if frame.task_id != task.id:
            continue

        if isinstance(frame, TerminalEvidence):
            if frame.agent_id not in bridged_set:
                # Evidence may terminate ONLY a dispatched bridged participant;
                # anything else is a spoofed/misrouted signal and never counts.
                logger.warning(
                    "ignoring terminal evidence for non-bridged agent",
                    task_id=task.id,
                    agent_id=frame.agent_id,
                    source=frame.source,
                )
                continue
            state = states[frame.agent_id]
            state.terminal_index = index
            state.by_evidence = True
            continue

        if isinstance(frame, TurnFrame):
            if frame.agent_id not in required:
                logger.debug(
                    "ignoring turn from non-participant",
                    task_id=task.id,
                    agent_id=frame.agent_id,
                )
                continue
            if frame.agent_id in bridged_set:
                # Bridged turns are CONTENT only: their stance is the gateway's
                # typed stance passed through verbatim and never carries
                # terminal semantics -- it can neither complete nor retract.
                logger.debug(
                    "bridged participant turn is content-only",
                    task_id=task.id,
                    agent_id=frame.agent_id,
                )
                continue
            state = states[frame.agent_id]
            if frame.stance.needs_from:
                # Terminality is LATEST-TURN state, not sticky: a later
                # non-terminal turn (steer/retargeted follow-up) retracts a
                # participant's earlier terminal turn.
                if state.terminal_index is not None:
                    logger.debug(
                        "retracting terminality on later non-terminal turn",
                        task_id=task.id,
                        agent_id=frame.agent_id,
                    )
                state.terminal_index = None
                state.by_evidence = False
                continue
            state.terminal_index = index
            state.by_evidence = False
            continue

        if isinstance(frame, StatusFrame) and frame.state in _NON_TERMINAL_STATES:
            if frame.agent_id not in required:
                # Statuses are identity-scoped like turns: a
                # non-participant can never reopen a task it was not
                # dispatched to.
                logger.debug(
                    "ignoring status from non-participant",
                    task_id=task.id,
                    agent_id=frame.agent_id,
                    state=frame.state,
                )
                continue
            state = states[frame.agent_id]
            state.last_status_index = index
            if frame.state != "working":
                state.last_escalation_index = index

    for agent_id, state in states.items():
        if state.terminal_index is None:
            logger.debug(
                "task not done: participant outstanding",
                task_id=task.id,
                agent_id=agent_id,
            )
            return False
        # A TURN-established terminality is reopened by ANY later non-terminal
        # status its author published: turns and statuses share the rtech
        # connection, so log order is sound. An EVIDENCE-established
        # terminality rides the orchestrator's own fleet connection -- the
        # author's later `working` is a cross-connection stale echo (the
        # gateway's typed Done is strictly later than every paired working
        # status at the source) that must not block, or the paired status
        # either deadlocks the task or contradicts an already-published result.
        # blocked/input_required escalations still block
        # fail-closed.
        reopen_index = state.last_escalation_index if state.by_evidence else state.last_status_index
        if reopen_index is not None and reopen_index > state.terminal_index:
            logger.debug(
                "task not done: non-terminal status after participant terminality",
                task_id=task.id,
                agent_id=agent_id,
                terminal_index=state.terminal_index,
                status_index=reopen_index,
            )
            return False

    logger.info(
        "task derived done",
        task_id=task.id,
        terminal_count=len(states),
    )
    return True
