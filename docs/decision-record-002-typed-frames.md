# Decision Record 002: Typed Bus Frames

## Status

Accepted.

## Context

The rtech-agents bus-first proof of concept needs a shared wire contract before
connectors and managed agents can coordinate safely. Connectors must stay dumb
edges: they translate edge payloads into bus frames and bus frames back into edge
messages, but they do not carry hidden control signals.

## Decision

Add `packages/rtech-frames` as the owner of typed Pydantic models and the JSON
codec used on the NATS wire.

Accepted frame models:

- `TaskFrame`: `schema_version`, `id`, `origin_edge`, `requester`, `text`,
  `scope`.
- `Stance`: `position`, `because`, `needs_from`.
- `TurnFrame`: `schema_version`, `agent_id`, `task_id`, `visible_text`,
  `stance`, `refs`.
- `SteerFrame`: `schema_version`, `task_id`, `text`, `target_agent`.
- `StatusFrame`: `schema_version`, `task_id`, `agent_id`, `state`.

The codec adds a wire-only `frame_type` discriminator when encoding JSON and
uses it for type-dispatched decode. `frame_type` is not part of the Pydantic
frame models so the model fields remain the domain contract rather than a
transport artifact.

`StatusFrame.state` is limited to `working`, `input_required`, and `blocked`.
There is no completion frame, completion sentinel, prose trailer, or scraped
signal. Completion is orchestrator-derived from the frame graph.

## Consequences

Frame validation now happens at the package boundary before connector or agent
code can consume a bus payload. Unknown frame types and missing required fields
fail closed through the codec.

## Amendment (steer wire)

`SteerFrame` gains a typed `interrupt: bool = False` field. The interrupt intent
is a frame field, never scraped from `text` — the steer wire carries no prose
command channel (doctrine rule 3). A steer with `interrupt=True` aborts the
target's active turn before injecting `text`; `interrupt=False` injects inline as
before. The field defaults to `False`, so existing steer payloads decode
unchanged. This keeps the human-in-the-loop steer wire first-class and reachable
directly from the bus (doctrine rule 5) rather than as a deferred connector
behavior.

## Amendment (sequence-aware done derivation)

Done derivation is **sequence-aware**. The original rule tracked the single
most-recent `StatusFrame` for a task and blocked done whenever that latest
status was non-terminal (`working` / `input_required` / `blocked`), order-blind.
That is wrong on the live bus path: agentd's `FrameTurnRunner` emits
`StatusFrame(state="working")` at turn **START** — before the terminal
`TurnFrame` — and all three status states are non-terminal (there is **no**
terminal status value). So the real-wire graph `[working-status, terminal-turn]`
derived `is_done=False` forever, and done was never derived over NATS
(empirically confirmed on the live bus).

The amended rule: a non-terminal `StatusFrame` blocks done only when it occurs
**after** the *participation-completing terminal turn* — the terminal turn (by
sequence position) at which the set of distinct terminal `agent_id`s first
reaches the scope's required count. A non-terminal status at or before that
completion position is naturally **superseded** by the completion and does not
block. The pre-turn `working` status agentd emits is superseded this way, so
`[working, terminal-turn]` now derives done; but a `blocked` / `input_required`
/ `working` that lands *after* completion still (correctly) blocks.

Doctrine held (anti-regression rule 2, "no `[[done]]` from agents; done is
derived by the orchestrator"):

- **StatusFrame enum unchanged** — states remain exactly
  `working | input_required | blocked`. No terminal status value, no completion
  frame, no sentinel was added.
- **agentd unchanged** — no agent-asserted completion status; agents still never
  assert done. Done stays orchestrator-derived from the typed frame graph, now
  using sequence position (which was always load-bearing on the ordered log).

## Amendment (task participants)

`TaskFrame` gains a typed `participants: list[str] = []` field. It names the
specific agent ids a multi-agent task fans out to — the UI must be able to pick
*which* agents form a duo/trio, and identity was previously unrepresentable
(`scope` names only a COUNT). Participants is a frame field, never scraped from
`text` (doctrine rule 3). The field defaults to empty, so existing task payloads
decode unchanged.

`scope` and `participants` are two views of the same fact: `scope` is the
required count, `participants` are the identities. The **edge derives scope
server-side from the participant count** — 0 or 1 → `single`, 2 → `duo`, 3 →
`trio`, more than 3 → HTTP 400. A client that still sends an explicit `scope`
must have it MATCH the derived value or the edge returns 400; the derived count
is authoritative, so the two can never disagree on the wire.

Orchestrator routing consumes the field:

- `scope=single` routes to `participants[0]` when given, else the configured
  default single-agent target (unchanged behavior when participants is empty).
- `scope=duo`/`trio` publishes the `TaskFrame` verbatim to each participant's
  `rtech.agent.<id>.task` lane. Done is derived from the frame graph via the
  existing sequence-aware `is_done` (duo/trio count semantics already
  implemented); the terminal turn is published once to the result lane.

Doctrine held: no new done token or completion frame, no scraped signal —
`participants` is a typed identity field and done stays orchestrator-derived
from the graph.

## Amendment (status authorship)

`StatusFrame` gains a **required** typed `agent_id: str` field naming the
status author, mirroring `TurnFrame`. Earlier review rounds proved authorless
statuses unsound once native and bridged participants mix on one task: done
derivation had to classify a later `working` status as "stale cross-connection
echo" or "genuine reopen" with one task-wide flag, so the classification
depended on which participant happened to complete the set, not on who
authored the status. With authorship on the frame, staleness is derived PER
PARTICIPANT: a non-terminal status reopens only its author, and only when it
is later in the log than that author's own terminal-establishing event
(terminal turn for native participants, `TerminalEvidence` for bridged ones).
Statuses from non-participants are ignored, identity-scoped exactly like turns.

This is a **breaking wire change**: `agent_id` has no default, so a v1 status
without authorship fails closed at decode, and `SCHEMA_VERSION` bumps `1` →
`2` (the constant stamped on every encoded frame; decode accepts any non-empty
version, so only the status shape itself gates). Deployed stacks must re-sync
on merge.

Doctrine held: signals remain typed frame fields (rule 3 — per-participant
state needs the participant on the frame), the status enum is unchanged, no
completion frame or sentinel was added, and done stays orchestrator-derived.
