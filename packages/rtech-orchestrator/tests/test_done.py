"""Cover derived-done logic across scope, terminality, status, and identity."""

from __future__ import annotations

from rtech_frames import Stance, StatusFrame, TaskFrame, TurnFrame, is_done
from rtech_orchestrator import TaskGraph

# Dispatched participant sets per scope (done is participant-scoped).
_SCOPE_DISPATCH: dict[str, frozenset[str]] = {
    "single": frozenset({"a1"}),
    "duo": frozenset({"a1", "a2"}),
}


def _task(scope: str) -> TaskFrame:
    """Build a valid task frame for a given scope.

    Duo/trio frames must name their participants (wire
    invariant); ``_SCOPE_DISPATCH`` remains the dispatched identity source the
    derivation actually uses.
    """
    return TaskFrame(
        id="t1",
        origin_edge="web",
        requester="rico",
        text="do the thing",
        scope=scope,  # type: ignore[arg-type]
        participants=[] if scope == "single" else sorted(_SCOPE_DISPATCH[scope]),
    )


def _graph(scope: str) -> TaskGraph:
    """Build a task graph with the scope's participant set already dispatched."""
    graph = TaskGraph(_task(scope))
    graph.record_dispatch(_SCOPE_DISPATCH[scope])
    return graph


def _done(graph: TaskGraph) -> bool:
    """Derive done the way the orchestrator does: over the dispatched set."""
    return is_done(graph.task, graph.events, participants=graph.participants)


def _turn(agent_id: str, needs: list[str] | None = None) -> TurnFrame:
    """Build a turn frame; ``needs`` populates the outstanding cross-agent need."""
    return TurnFrame(
        agent_id=agent_id,
        task_id="t1",
        visible_text="here is my take",
        stance=Stance(position="p", because="b", needs_from=needs or []),
    )


def _status(state: str, agent_id: str = "a1") -> StatusFrame:
    """Build a status frame for a given machine state."""
    return StatusFrame(task_id="t1", agent_id=agent_id, state=state)  # type: ignore[arg-type]


def test_single_terminal_turn_is_done() -> None:
    """(a) single scope, one terminal turn => done."""
    graph = _graph("single")
    graph.add_turn(_turn("a1"))
    assert _done(graph) is True


def test_single_turn_with_need_not_done() -> None:
    """(b) single scope, turn still needs a peer => not done."""
    graph = _graph("single")
    graph.add_turn(_turn("a1", needs=["a2"]))
    assert _done(graph) is False


def test_single_terminal_but_blocked_not_done() -> None:
    """(c) single scope, terminal turn but latest status blocked => not done."""
    graph = _graph("single")
    graph.add_turn(_turn("a1"))
    graph.add_status(_status("blocked"))
    assert _done(graph) is False


def test_single_input_required_not_done() -> None:
    """(d) single scope, latest status input_required => not done."""
    graph = _graph("single")
    graph.add_turn(_turn("a1"))
    graph.add_status(_status("input_required"))
    assert _done(graph) is False


def test_single_working_status_not_done() -> None:
    """(e) single scope, terminal turn but latest status working => NOT done \
(working is non-terminal in the canonical contract)."""
    graph = _graph("single")
    graph.add_turn(_turn("a1"))
    graph.add_status(_status("working"))
    assert _done(graph) is False


def test_duo_one_of_two_not_done() -> None:
    """(f) duo scope, only one agent terminal => not done."""
    graph = _graph("duo")
    graph.add_turn(_turn("a1"))
    assert _done(graph) is False


def test_duo_both_terminal_is_done() -> None:
    """(g) duo scope, both agents terminal => done."""
    graph = _graph("duo")
    graph.add_turn(_turn("a1"))
    graph.add_turn(_turn("a2"))
    assert _done(graph) is True


def test_no_turns_not_done() -> None:
    """(h) no turns => not done."""
    graph = _graph("single")
    assert _done(graph) is False


def test_duo_two_turns_same_agent_counts_once() -> None:
    """(i) duo scope, two terminal turns from one agent counts as 1 => not done."""
    graph = _graph("duo")
    graph.add_turn(_turn("a1"))
    graph.add_turn(_turn("a1"))
    assert _done(graph) is False


def test_undispatched_graph_has_no_participants() -> None:
    """(j) before dispatch the graph carries no participant identities."""
    graph = TaskGraph(_task("single"))
    assert graph.participants == frozenset()
    graph.add_turn(_turn("a1"))
    # Without a dispatched set the orchestrator skips the done check entirely;
    # is_done itself also fails closed when handed no identities.
    assert is_done(graph.task, graph.events) is False


def test_foreign_agent_turn_does_not_count() -> None:
    """(k): a terminal turn from a non-dispatched agent never counts."""
    graph = _graph("single")
    graph.add_turn(_turn("agentd-bilby"))
    assert _done(graph) is False
    graph.add_turn(_turn("a1"))
    assert _done(graph) is True


def test_record_dispatch_is_cumulative() -> None:
    """(l) record_dispatch unions across calls (per-lane fan-out)."""
    graph = TaskGraph(_task("duo"))
    graph.record_dispatch({"a1"})
    graph.record_dispatch({"a2"})
    assert graph.participants == frozenset({"a1", "a2"})
