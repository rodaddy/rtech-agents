"""Test orchestrator-derived done state over the frame graph."""

from __future__ import annotations

import pytest
from rtech_frames.done import TerminalEvidence, is_done
from rtech_frames.frames import Stance, StatusFrame, TaskFrame, TurnFrame

# Default participant identities per scope; done is participant-scoped.
_SCOPE_PARTICIPANTS: dict[str, list[str]] = {
    "single": ["codex-a"],
    "duo": ["codex-a", "codex-b"],
}


def _task(scope: str, participants: list[str] | None = None) -> TaskFrame:
    resolved = participants if participants is not None else _SCOPE_PARTICIPANTS.get(scope, [])
    return TaskFrame(
        id="task-1",
        origin_edge="discord",
        requester="rico",
        text="Route this task.",
        scope=scope,  # type: ignore[arg-type]
        participants=resolved,
    )


def _terminal_turn(agent_id: str) -> TurnFrame:
    return TurnFrame(
        agent_id=agent_id,
        task_id="task-1",
        visible_text="Done from my side.",
        stance=Stance(position="agree", because="The scope is satisfied."),
    )


def _pending_turn(agent_id: str) -> TurnFrame:
    return TurnFrame(
        agent_id=agent_id,
        task_id="task-1",
        visible_text="I still need input.",
        stance=Stance(
            position="agree",
            because="I need a hand.",
            needs_from=["codex-b"],
        ),
    )


def _status(state: str, agent_id: str = "codex-a") -> StatusFrame:
    return StatusFrame(task_id="task-1", agent_id=agent_id, state=state)  # type: ignore[arg-type]


def test_module_entrypoint_loads() -> None:
    assert callable(is_done)


def test_single_agent_complete_is_done() -> None:
    task = _task("single")
    graph: list[TurnFrame | StatusFrame] = [_terminal_turn("codex-a")]

    assert is_done(task, graph) is True


def test_duo_incomplete_one_silent_is_not_done() -> None:
    task = _task("duo")
    graph: list[TurnFrame | StatusFrame] = [_terminal_turn("codex-a")]

    assert is_done(task, graph) is False


def test_duo_complete_two_distinct_agents_is_done() -> None:
    task = _task("duo")
    graph: list[TurnFrame | StatusFrame] = [
        _terminal_turn("codex-a"),
        _terminal_turn("codex-b"),
    ]

    assert is_done(task, graph) is True


def test_blocked_participant_is_not_done() -> None:
    task = _task("duo")
    graph: list[TurnFrame | StatusFrame] = [
        _terminal_turn("codex-a"),
        _terminal_turn("codex-b"),
        StatusFrame(task_id="task-1", agent_id="codex-a", state="blocked"),
    ]

    assert is_done(task, graph) is False


def test_needs_from_pending_turn_is_not_terminal() -> None:
    task = _task("duo")
    graph: list[TurnFrame | StatusFrame] = [
        _terminal_turn("codex-a"),
        _pending_turn("codex-b"),
    ]

    assert is_done(task, graph) is False


def test_unrecognized_scope_fails_closed() -> None:
    task = _task("single")
    object.__setattr__(task, "scope", "quartet")
    graph: list[TurnFrame | StatusFrame] = [_terminal_turn("codex-a")]

    assert is_done(task, graph) is False


@pytest.mark.parametrize("state", ["working", "input_required", "blocked"])
def test_non_terminal_status_after_completion_blocks_done(state: str) -> None:
    # Status arrives AFTER the completing terminal turn -> blocks done.
    task = _task("single")
    graph: list[TurnFrame | StatusFrame] = [
        _terminal_turn("codex-a"),
        StatusFrame(task_id="task-1", agent_id="codex-a", state=state),  # type: ignore[arg-type]
    ]

    assert is_done(task, graph) is False


def test_status_working_then_terminal_turn_is_done() -> None:
    # agentd emits working at turn START, before the terminal turn. The
    # pre-turn working status is superseded by completion -> done.
    task = _task("single")
    graph: list[TurnFrame | StatusFrame] = [
        _status("working"),
        _terminal_turn("codex-a"),
    ]

    assert is_done(task, graph) is True


def test_terminal_turn_then_blocked_is_not_done() -> None:
    # blocked AFTER completion blocks done.
    task = _task("single")
    graph: list[TurnFrame | StatusFrame] = [
        _terminal_turn("codex-a"),
        _status("blocked"),
    ]

    assert is_done(task, graph) is False


def test_terminal_turn_then_input_required_is_not_done() -> None:
    # input_required AFTER completion blocks done.
    task = _task("single")
    graph: list[TurnFrame | StatusFrame] = [
        _terminal_turn("codex-a"),
        _status("input_required"),
    ]

    assert is_done(task, graph) is False


def test_duo_interleaved_working_before_second_completion_is_done() -> None:
    # Each working is the turn-start emit BEFORE its author's terminal turn
    # -> superseded per participant -> done.
    task = _task("duo")
    graph: list[TurnFrame | StatusFrame] = [
        _status("working", agent_id="codex-a"),
        _terminal_turn("codex-a"),
        _status("working", agent_id="codex-b"),
        _terminal_turn("codex-b"),
    ]

    assert is_done(task, graph) is True


def test_duo_both_terminal_then_blocked_is_not_done() -> None:
    # Non-terminal status AFTER duo completion blocks done.
    task = _task("duo")
    graph: list[TurnFrame | StatusFrame] = [
        _terminal_turn("codex-a"),
        _terminal_turn("codex-b"),
        _status("blocked"),
    ]

    assert is_done(task, graph) is False


def test_terminal_status_after_non_terminal_allows_done() -> None:
    task = _task("single")
    graph: list[TurnFrame | StatusFrame] = [
        StatusFrame(task_id="task-1", agent_id="codex-a", state="working"),
        _terminal_turn("codex-a"),
        StatusFrame(task_id="task-1", agent_id="codex-a", state="input_required"),
        StatusFrame(task_id="task-2", agent_id="codex-a", state="blocked"),
    ]
    # task-1 input_required occurs AFTER the completing terminal turn -> still
    # not done under sequence-aware semantics; task-2 status is ignored.
    assert is_done(task, graph) is False


def test_other_task_frames_are_ignored() -> None:
    task = _task("single")
    graph: list[TurnFrame | StatusFrame] = [
        TurnFrame(
            agent_id="codex-x",
            task_id="task-2",
            visible_text="Different task.",
            stance=Stance(position="agree", because="Elsewhere."),
        ),
        _terminal_turn("codex-a"),
        StatusFrame(task_id="task-2", agent_id="codex-x", state="blocked"),
    ]

    assert is_done(task, graph) is True


# --- participant scoping (defect B) --------------------------------


def test_foreign_terminal_turn_never_counts() -> None:
    # Defect B shape: a foreign stack's agent lands a terminal turn on the
    # task's turn lane. Identity-blind counting derived done from it; the
    # participant-scoped derivation must not.
    task = _task("single")
    graph: list[TurnFrame | StatusFrame] = [_terminal_turn("agentd-bilby")]

    assert is_done(task, graph) is False


def test_done_waits_for_dispatched_participant() -> None:
    # Foreign terminal turn first, dispatched participant's turn later: done
    # derives only once the PARTICIPANT is terminal.
    task = _task("single")
    graph: list[TurnFrame | StatusFrame] = [
        _status("working"),
        _terminal_turn("agentd-bilby"),
    ]
    assert is_done(task, graph) is False

    graph.append(_terminal_turn("codex-a"))
    assert is_done(task, graph) is True


def test_duo_foreign_turn_does_not_substitute_for_participant() -> None:
    task = _task("duo")
    graph: list[TurnFrame | StatusFrame] = [
        _terminal_turn("codex-a"),
        _terminal_turn("agentd-bilby"),
    ]

    assert is_done(task, graph) is False


def test_explicit_participants_argument_wins() -> None:
    # The orchestrator's dispatched set is the identity source even when the
    # task frame itself names no participants (single-scope default routing).
    task = _task("single", participants=[])
    graph: list[TurnFrame | StatusFrame] = [_terminal_turn("codex-a")]

    assert is_done(task, graph, participants={"codex-a"}) is True
    assert is_done(task, graph, participants={"agentd-other"}) is False


def test_no_participant_identities_fails_closed() -> None:
    # Identity-blind derivation is removed: with no dispatched set and no
    # frame participants there is nothing to derive against.
    task = _task("single", participants=[])
    graph: list[TurnFrame | StatusFrame] = [_terminal_turn("codex-a")]

    assert is_done(task, graph) is False


def test_participant_count_scope_mismatch_fails_closed() -> None:
    # The wire now rejects a scope/participant-count mismatch outright, so the
    # mismatch is driven through the orchestrator's explicit dispatched set:
    # a single-scope task with a two-member dispatch record fails closed.
    task = _task("single", participants=[])
    graph: list[TurnFrame | StatusFrame] = [
        _terminal_turn("codex-a"),
        _terminal_turn("codex-b"),
    ]

    assert is_done(task, graph, participants={"codex-a", "codex-b"}) is False


def test_duplicate_participant_terminal_turns_count_once() -> None:
    # Duplicate terminal turns from one participant (duplicate .result root
    # cause) never satisfy a duo.
    task = _task("duo")
    graph: list[TurnFrame | StatusFrame] = [
        _terminal_turn("codex-a"),
        _terminal_turn("codex-a"),
    ]

    assert is_done(task, graph) is False


def test_terminality_retracted_by_later_non_terminal_turn() -> None:
    # Terminal state is latest-turn-derived, never sticky (round 2): a steer/
    # retargeted follow-up with a nonempty needs_from retracts the earlier
    # terminal turn, so a later peer terminal turn cannot derive done while
    # this participant is nonterminal again.
    task = _task("duo")
    graph: list[TurnFrame | StatusFrame] = [
        _terminal_turn("codex-a"),
        _pending_turn("codex-a"),
        _terminal_turn("codex-b"),
    ]

    assert is_done(task, graph) is False


def test_retracted_participant_completes_again() -> None:
    # After retraction, a NEW terminal turn from the same participant restores
    # completion — and the pre-existing working status earlier in the log is
    # superseded as usual.
    task = _task("duo")
    graph: list[TurnFrame | StatusFrame] = [
        _status("working"),
        _terminal_turn("codex-a"),
        _pending_turn("codex-a"),
        _terminal_turn("codex-b"),
        _terminal_turn("codex-a"),
    ]

    assert is_done(task, graph) is True


def test_single_retraction_then_completion() -> None:
    task = _task("single")
    graph: list[TurnFrame | StatusFrame] = [
        _terminal_turn("codex-a"),
        _pending_turn("codex-a"),
    ]
    assert is_done(task, graph) is False

    graph.append(_terminal_turn("codex-a"))
    assert is_done(task, graph) is True


# --- bridged participants -----------------------------------


def _evidence(agent_id: str, task_id: str = "task-1") -> TerminalEvidence:
    return TerminalEvidence(
        agent_id=agent_id, task_id=task_id, source="fleet-done", outcome="completed"
    )


def test_bridged_turn_terminality_is_ignored() -> None:
    # A bridged participant's turn is CONTENT only: even a structurally
    # terminal-looking stance (empty needs_from, passthrough) never completes.
    task = _task("single", participants=["bilby"])
    graph: list[TurnFrame | StatusFrame | TerminalEvidence] = [_terminal_turn("bilby")]

    assert is_done(task, graph, bridged={"bilby"}) is False


def test_bridged_completes_only_on_terminal_evidence() -> None:
    task = _task("single", participants=["bilby"])
    graph: list[TurnFrame | StatusFrame | TerminalEvidence] = [
        _status("working", agent_id="bilby"),
        _terminal_turn("bilby"),
        _evidence("bilby"),
    ]

    assert is_done(task, graph, bridged={"bilby"}) is True


def test_evidence_for_non_bridged_agent_never_counts() -> None:
    # Evidence may terminate ONLY bridged participants; for a native
    # participant it is a spoofed/misrouted signal and is ignored.
    task = _task("single")
    graph: list[TurnFrame | StatusFrame | TerminalEvidence] = [_evidence("codex-a")]

    assert is_done(task, graph) is False


def test_evidence_for_other_task_is_ignored() -> None:
    task = _task("single", participants=["bilby"])
    graph: list[TurnFrame | StatusFrame | TerminalEvidence] = [
        _evidence("bilby", task_id="task-9"),
    ]

    assert is_done(task, graph, bridged={"bilby"}) is False


def test_bridged_not_subset_of_participants_fails_closed() -> None:
    task = _task("single")
    graph: list[TurnFrame | StatusFrame | TerminalEvidence] = [_evidence("bilby")]

    assert is_done(task, graph, bridged={"bilby"}) is False


def test_mixed_duo_native_turn_plus_bridged_evidence() -> None:
    task = _task("duo", participants=["codex-a", "bilby"])
    graph: list[TurnFrame | StatusFrame | TerminalEvidence] = [
        _terminal_turn("codex-a"),
        _terminal_turn("bilby"),  # content only — must not complete the duo
    ]
    assert is_done(task, graph, bridged={"bilby"}) is False

    graph.append(_evidence("bilby"))
    assert is_done(task, graph, bridged={"bilby"}) is True


def test_blocked_status_after_evidence_blocks_done() -> None:
    # Sequence-aware supersession applies to evidence exactly like turns: the
    # bridge's blocked shed/expiry status after completion keeps the task
    # un-done, visibly.
    task = _task("single", participants=["bilby"])
    graph: list[TurnFrame | StatusFrame | TerminalEvidence] = [
        _terminal_turn("bilby"),
        _evidence("bilby"),
        _status("blocked", agent_id="bilby"),
    ]

    assert is_done(task, graph, bridged={"bilby"}) is False


def test_working_status_before_evidence_is_superseded() -> None:
    task = _task("single", participants=["bilby"])
    graph: list[TurnFrame | StatusFrame | TerminalEvidence] = [
        _status("working", agent_id="bilby"),
        _terminal_turn("bilby"),
        _evidence("bilby"),
    ]

    assert is_done(task, graph, bridged={"bilby"}) is True


def test_working_status_after_evidence_completion_does_not_block() -> None:
    # the bridge pairs each content turn with a working status
    # on the rtech connection, while evidence rides the orchestrator's own
    # fleet connection — a working status landing after the evidence in the
    # merged log is a stale cross-connection echo, never a blocker. Otherwise
    # it would contradict the already-published result.
    task = _task("single", participants=["bilby"])
    graph: list[TurnFrame | StatusFrame | TerminalEvidence] = [
        _terminal_turn("bilby"),
        _evidence("bilby"),
        _status("working", agent_id="bilby"),
    ]

    assert is_done(task, graph, bridged={"bilby"}) is True


def test_evidence_then_working_then_turn_completes() -> None:
    # Worst cross-connection ordering: a fast typed Done outruns BOTH halves
    # of the bridge's turn/working pair. The stale working echo must not
    # deadlock the task once the content turn lands.
    task = _task("single", participants=["bilby"])
    graph: list[TurnFrame | StatusFrame | TerminalEvidence] = [
        _evidence("bilby"),
        _status("working", agent_id="bilby"),
        _terminal_turn("bilby"),
    ]

    assert is_done(task, graph, bridged={"bilby"}) is True


def test_input_required_after_evidence_still_blocks() -> None:
    # Escalations are not liveness echoes: blocked/input_required after an
    # evidence-established completion keep the task un-done, fail-closed.
    task = _task("single", participants=["bilby"])
    graph: list[TurnFrame | StatusFrame | TerminalEvidence] = [
        _terminal_turn("bilby"),
        _evidence("bilby"),
        _status("input_required", agent_id="bilby"),
    ]

    assert is_done(task, graph, bridged={"bilby"}) is False


def test_mixed_duo_turn_established_completion_still_blocked_by_working() -> None:
    # A native participant's terminality is TURN-established: its own statuses
    # share its connection, so a later working it authored reopens it — the
    # full status-after-terminal block applies per participant.
    task = _task("duo", participants=["codex-a", "bilby"])
    graph: list[TurnFrame | StatusFrame | TerminalEvidence] = [
        _evidence("bilby"),
        _terminal_turn("codex-a"),
        _status("working", agent_id="codex-a"),
    ]

    assert is_done(task, graph, bridged={"bilby"}) is False


# --- per-participant status authorship ------------------------


def test_round4_native_terminal_then_evidence_then_native_working_blocks() -> None:
    # Round-4 failing sequence 1: [native terminal, bridged TerminalEvidence,
    # working]. The task-wide boolean classified the whole set as
    # evidence-established (the bridged evidence happened to complete it), so
    # the genuinely later NATIVE working was ignored. Keyed by author, the
    # native participant is turn-established and its own later working reopens
    # it.
    task = _task("duo", participants=["codex-a", "bilby"])
    graph: list[TurnFrame | StatusFrame | TerminalEvidence] = [
        _terminal_turn("codex-a"),
        _evidence("bilby"),
        _status("working", agent_id="codex-a"),
    ]

    assert is_done(task, graph, bridged={"bilby"}) is False

    # The reopened participant completes again on its next terminal turn.
    graph.append(_terminal_turn("codex-a"))
    assert is_done(task, graph, bridged={"bilby"}) is True


def test_round4_evidence_then_native_terminal_then_late_bridge_working_completes() -> None:
    # Round-4 failing sequence 2: [bridged TerminalEvidence, native terminal,
    # delayed paired bridge working]. The task-wide boolean classified the set
    # as turn-established (the native turn completed it), so the bridge's
    # stale cross-connection working echo flipped a full-graph re-derivation
    # to False AFTER the orchestrator could already have published and latched
    # the result. Keyed by author, the bridged participant is
    # evidence-established and its own working echo never blocks — derivation
    # is order-independent across the echo's arrival.
    task = _task("duo", participants=["codex-a", "bilby"])
    graph: list[TurnFrame | StatusFrame | TerminalEvidence] = [
        _evidence("bilby"),
        _terminal_turn("codex-a"),
    ]
    assert is_done(task, graph, bridged={"bilby"}) is True

    graph.append(_status("working", agent_id="bilby"))
    assert is_done(task, graph, bridged={"bilby"}) is True


def test_status_from_non_participant_never_blocks() -> None:
    # Statuses are identity-scoped like turns: a foreign stack's
    # status on the task's status lane cannot reopen the task.
    task = _task("single")
    graph: list[TurnFrame | StatusFrame] = [
        _terminal_turn("codex-a"),
        _status("blocked", agent_id="agentd-bilby"),
    ]

    assert is_done(task, graph) is True
