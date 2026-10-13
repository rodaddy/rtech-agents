# Bridged participants: lifecycle, ownership, done derivation

Design for (redesign after the stop-rule). Supersedes the
round-1/round-2 reverse-path shape, in which the bridge edge authored the
terminal `TurnFrame` (first from `kind=status`, then from the typed fleet
`Done`). Both shapes put terminal-turn authorship at the edge — a violation of
doctrine rules 1/2 at the root. This document moves terminal semantics for
bridged participants to the orchestrator and constrains the bridge to content
translation only.

Builds on participant-scoped done model (`is_done(task, graph,
participants=...)`, dispatch record on `TaskGraph`); this design builds on
that model.

## Who owns what

| Concern | Owner | Never |
|---|---|---|
| TaskFrame → fleet dispatch `Envelope` (content) | bridge | control fields |
| fleet reply text/stance → `TurnFrame` content fields | bridge | terminal semantics |
| In-flight liveness (`StatusFrame` working/blocked) | bridge | done/terminal state |
| Fleet-side lifecycle (reply subscription, per-agent FIFO, expiry) | bridge | rtech-side control emission |
| Consuming the typed `fleet_core.spec.Done` as **terminal evidence** | orchestrator | re-authoring it as a `TurnFrame` |
| Deriving done from participants' terminal evidence | orchestrator (`is_done`, one place) | any edge |
| Publishing the task result turn | orchestrator | any edge |

## The bridged-participant lifecycle

```
webchat edge          orchestrator                bridge (dumb edge)         hermes gateway
    |                      |                            |                         |
    |-- TaskFrame -------->|  rtech.task.new            |                         |
    |                      |-- TaskFrame -------------->|  rtech.agent.<id>.task  |
    |                      |   record_dispatch({id})    |                         |
    |                      |   record_bridged({id})     |-- Envelope(dispatch) -->|  {env}.agent.dispatch.<id>
    |                      |   subscribe fleet done     |                         |
    |                      |   lane for task            |                         |
    |                      |                            |<- Envelope(status) -----|  {env}.agent.status.<task>
    |                      |<- TurnFrame (CONTENT) -----|  rtech.task.<id>.turn   |
    |                      |<- StatusFrame(working) ----|  rtech.task.<id>.status |
    |                      |                            |<- Envelope(done, typed  |
    |                      |<==== typed fleet Done =====|====== Done signal) =====|  (same fleet lane; the
    |                      |   (orchestrator's OWN      |   bridge: lifecycle     |   orchestrator consumes it
    |                      |    fleet subscription)     |   only — NO frame       |   directly, typed input)
    |                      |   terminal evidence        |                         |
    |                      |   is_done() → done         |                         |
    |<- result (turn) -----|  rtech.task.<id>.result    |                         |
```

Key wire fact: the typed `Done` reaches the orchestrator on the **fleet**
namespace (`{env}.agent.status.<task_id>`), via the orchestrator's own
subscription — not through any bridge-authored rtech frame. The bridge observes
the same envelope purely to end its per-task lifecycle (unsubscribe, drop
state, advance the FIFO); it emits nothing on the rtech side for it.

## Terminal evidence model

`is_done` (rtech-frames `done.py`) remains the single derivation point. It now
distinguishes participant kinds recorded at dispatch:

- **Native participants** (agentd): terminal evidence is the LATEST `TurnFrame`
  being terminal (`stance.needs_from` empty), sequence-aware with retraction —
  unchanged from the earlier design.
- **Bridged participants**: terminal evidence is ONLY a recorded
  `TerminalEvidence` entry — the orchestrator's typed record of the fleet
  `Done` it consumed (agent id, task id, outcome, source). `TurnFrame`s from a
  bridged participant are **content only**: their `stance.needs_from` carries
  the gateway's own typed stance verbatim and is IGNORED for terminality in
  both directions (it can neither complete nor retract).

`TerminalEvidence` is a typed in-graph record, deliberately NOT a wire frame:
it has no codec registration, cannot be published, and exists only inside the
orchestrator's `TaskGraph` event log. This is what "consume the typed Done as a
typed input, do not re-author it" means concretely.

Fail-closed rules (all pre-existing rules from kept):

- Evidence whose agent id is not a **dispatched bridged participant** of the
  task is rejected at the graph boundary (spoof on the fleet lane).
- The fleet envelope must satisfy: `kind == "done"`, typed signal isinstance
  `fleet_core.spec.Done`, `envelope.sender` ∈ dispatched bridged set,
  `envelope.task_id == Done.task_id == task id`. Any mismatch → dropped loudly.
- A bridged participant dispatched while the orchestrator has **no fleet
  signal source configured** marks the task dispatch-failed (it could never
  observe terminal evidence — better loud than an eternal wait).
- Status classification is PER PARTICIPANT, keyed by the status frame's
  required `agent_id` (frame contract amendment, round 5): a non-terminal
  status reopens ONLY its author, and only when it is later in the log than
  that author's own terminal-establishing event. For an evidence-established
  (bridged) participant a later `working` it authored is a stale
  cross-connection echo and never blocks, while `blocked`/`input_required`
  escalations still block fail-closed — the bridge's `blocked` status on an
  empty-reply completion or expiry keeps the task un-done, visibly. Statuses
  from non-participants are ignored (identity-scoped, like turns).

## How the orchestrator knows a participant is bridged

From its own typed dispatch record. `OrchestratorConfig.bridged_agents` names
the fleet-fronted ids (the same ids the bridge fronts); when a dispatch routes
to one of them, the orchestrator records the participant as bridged on the
graph and subscribes the task's fleet done lane through an injected
`FleetSignalSource` seam (`__main__` wires a real fleet-bus connection; tests
inject a fake that round-trips `Envelope.to_bytes/from_bytes`). The
subscription is dropped when the task finishes.

## Result publishing

The result turn is the latest `TurnFrame` from a participant whose terminal
evidence is satisfied: native → its latest terminal turn; bridged → its latest
content turn once `TerminalEvidence` is recorded. If done derives before the
bridged content turn has landed (subject-ordering race), the orchestrator
simply finishes on the turn's arrival — `_maybe_finish` re-runs per frame; the
task is never finished without a result turn.

## What the bridge keeps from round 2

Typed named-field consumption (no prose anywhere, `_trailer_utils` stays
deleted), identity fail-closed classification, subscribe-before-publish with
rollback, one-in-flight-per-agent FIFO serialization, bounded completed-id
window, expiry sweeper. New in round 3: a bounded per-agent queue with loud
shed (`blocked` status) and an atomic claim in the sweeper so completion and
expiry can never double-finalize.
