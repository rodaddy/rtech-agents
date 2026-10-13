# Debate protocol reference

This document describes the current intended protocol contract, including frame
schema v8 and the typed evidence / status-phase behavior. It is a wire and
lifecycle reference, not a prompt convention.

## Two JSON wire worlds

The repository has two distinct NATS JSON contracts. They must not be decoded
as if they were the same wire shape.

The fleet-facing path uses a `fleet_nats.Envelope`. Routing and correlation
live in envelope fields such as `kind`, `to`, `task_id`, `channel`, and
`correlation_id`; `payload` is an opaque JSON object owned by the consumer.
Built-in coordination signals belong in the envelope's typed signal field, not
in payload prose. `rtech_agentd.nats_ingress` consumes these envelopes on fleet
subjects and translates accepted work into a local `TaskFrame` before sending
the task text to the engine. Its replies wrap a serialized `TurnFrame` object
inside an outbound fleet envelope's `payload`.

The frame-driven rtech path uses raw UTF-8 JSON bytes produced by
`rtech_frames.codec.encode_frame`. The top-level object has a `frame_type`
discriminator and the fields of exactly one `TaskFrame`, `StatusFrame`,
`TurnFrame`, `SteerFrame`, or `LifecycleFrame`; it is not wrapped in a fleet
`Envelope`. `rtech_orchestrator.typedbus` and the frame-driven
`rtech_agentd.frame_turn` path publish and decode this raw typed-frame wire on
the canonical `rtech.*` subjects below.

The current frame schema version is `8`. It adds typed evidence through
`EvidenceSource` and `VerificationReport`, the optional `TurnFrame.verification`
field, and the optional `StatusFrame.phase` field. These optional fields are
dropped from the wire when absent, but the schema-version stamp still changes; a
legacy-shaped v8 payload therefore keeps the prior key set apart from that
version stamp and is not byte-identical to a v7 payload. Every frame model
forbids unknown fields (`extra="forbid"`), so this is a strict
consumers-before-producers rollout: every consumer must be upgraded to a build
that knows `verification` and `phase` before any producer begins emitting them,
or the un-upgraded consumer rejects the new fields at decode.

Use fleet envelopes at the existing fleet-bus ingress, reply, heartbeat, and
bridge boundaries. Use raw rtech frame JSON inside the orchestrator/agentd
task graph. A bridge or edge may translate between the worlds, but wrapping a
frame in an envelope payload does not turn that envelope into the raw frame
wire.

## Task and turn lifecycle

1. An edge publishes a raw `TaskFrame` to `rtech.task.new`.
2. The orchestrator validates and records the task, subscribes atomically to
   `rtech.task.<task_id>.turn` and `rtech.task.<task_id>.status`, then publishes
   the same immutable `TaskFrame` to each selected
   `rtech.agent.<agent_id>.task` lane.
3. Agentd appends the task id to its pending FIFO, opens the task's steer lane,
   and, for a task with multiple named participants, opens the shared sibling
   turn lane. It publishes `StatusFrame(state="working")` to
   `rtech.task.<task_id>.status` before sending the task to the engine.
4. When the evidence gate is enabled, every conversational task performs bounded
   Open Brain recall and builds a grounding packet. Recalled memory is evidence,
   not instructions. Authority order is live/source state, then Open Brain, then
   conversation/model memory. The working status carries a typed `phase`:
   `phase="verifying"` for cold/first/stale contexts or `phase="responding"` for
   warm follow-ups. `phase` only distinguishes the cold acknowledgement; it does
   not change `state` (still `working`) and is never a control signal. A cold
   factual or mixed answer without validated evidence is replaced with a
   deterministic guarded fallback.
5. The managed engine runs one serial turn. Its eventual `TurnDone` is an
   internal engine event, not a NATS frame. Agentd pops the FIFO head to bind
   that event back to the originating task id.
6. Agentd sanitizes the complete output and publishes one typed `TurnFrame` to
   `rtech.task.<task_id>.turn`. The frame carries visible text, typed `stance`,
   causal `refs`, and, on a gated turn, an optional typed `verification` report.
   Reported Open Brain source ids are validated against that turn's recall;
   non-memory source locators are accepted only when the daemon observed a
   verification method and the locator passes the safe-locator rules. The
   verification report is submitted structurally by the engine and validated and
   bounded by the runtime before it rides the wire — it is never scraped from
   `visible_text`.
7. The orchestrator accumulates status and turn events and derives completion.
   When the graph is done, only the orchestrator publishes the selected result
   turn on `rtech.task.<task_id>.result`.

The other canonical task lane is `rtech.task.<task_id>.steer`, described below.
Agent lifecycle events use `rtech.agent.<agent_id>.lifecycle` and are separate
from debate completion.

## Current coordination-frame fields

The current application fields, excluding the shared `schema_version` and
`frame_type` discriminator, are:

| Frame | Fields |
| --- | --- |
| `TaskFrame` | `id`, `origin_edge`, `requester`, `text`, `scope`, `participants`, `stack`, `created_at`, `origin`, `arbiter` |
| `TurnFrame` | `agent_id`, `task_id`, `visible_text`, `stance`, `refs`, `origin`, `verification` |
| `StatusFrame` | `task_id`, `agent_id`, `state`, `phase` |
| `SteerFrame` | `task_id`, `text`, `target_agent`, `interrupt` |

`TurnFrame.refs` records frame-graph causality: it begins with the task id and
may include accepted steer references. It is not a citation list. Evidence
citations and verification limitations belong only in `TurnFrame.verification`.

`TurnFrame.verification` carries a `VerificationReport`, which records `status`,
`claim_kind`, `ob_checked` (whether Open Brain recall ran), a bounded list of
typed `sources`, `limitations`, and observed `methods`. Each `EvidenceSource`
carries a `source_id`, a typed `kind`, a display `label`, an optional safe
`locator`, and a typed `authority` tier. All list and string fields are
length-bounded so a report cannot blow the frame size. The connector may render
those typed sources and limitations, but it never extracts citations,
confidence, completion, or control state from answer prose.

## Terminality and derived done

For a native participant, a turn is terminal if and only if
`turn.stance.needs_from == []`. A populated `needs_from` list is an explicit
request for further participant input and is non-terminal. A later turn can
therefore reopen or re-establish that participant's terminality; terminality is
latest-turn state, not a prose claim and not a sticky flag.

All three `StatusFrame.state` values (`working`, `input_required`, and
`blocked`) are non-terminal. Status is sequence-aware and participant-scoped:
a status published before that participant's terminal turn is superseded, while
a later status reopens only its author. For orchestrator-owned terminal evidence
from a bridged participant, a later cross-connection `working` echo is treated
as stale, but a later `blocked` or `input_required` escalation still blocks
completion.

Done is not a frame. `rtech_frames/done.py` derives it from the task, the
ordered `StatusFrame`/`TurnFrame` graph, the confirmed dispatched participant
set, and any orchestrator-owned terminal evidence for bridged participants.
The orchestrator calls this derivation and publishes the result lane. Agentd
never emits done.

The string `[[done]]` is forbidden as a protocol mechanism. Completion must not
be encoded in visible text, trailers, payload strings, or prompt conventions.
Likewise, an engine event stream closing, a process exiting, or a connector
finishing a send is not task completion. Engine stream closure is handled as a
lifecycle/failure boundary; only an orchestrator result publication proves
protocol completion. A connector may render a human-facing completion marker
after receiving that result, but the marker is presentation, never control.

## Sibling fanout and bounded debate

A multi-participant `TaskFrame` may name an `arbiter`, which must also appear in
`participants`. Before engine dispatch, agentd prepends a typed-role directive:
the arbiter moderates the exchange and closes with a TLDR ruling, while each
other participant argues its own position and engages sibling turns. These role
directives shape model behavior; they do not change terminality or create a
connector control channel.

For a task with more than one named participant, every agent publishes to and
subscribes to the same `rtech.task.<task_id>.turn` lane. The subscription is
installed through `rtech_frames.subs.subscribe_all`, which treats the requested
subscriptions as one set: any subscribe failure rolls back the handles already
opened, and incomplete rollback remains tracked for escalation.

The sibling handler accepts only a `TurnFrame` whose payload `task_id` matches
the task id bound to the lane. It ignores the local agent's own published turn;
this echo guard prevents self-injection and feedback loops. An accepted sibling
turn is rendered from its visible text and typed stance and sent into the live
engine session.

Sibling-triggered follow-ups are bounded per agent and task by
`RTECH_MAX_SIBLING_ROUNDS`. The current default is `2`. The handler checks and
reserves a round slot under the same async lock, before awaiting the engine
send, so concurrent deliveries cannot overshoot the cap. A failed send rolls
back its reservation.

Retirement is settle-aware:

- Below the cap, the lane retires when every named participant's latest
  observed turn is terminal. The lane subscription, counter, and closing state
  are cleared.
- At the cap, the lane and lane-scoped counter retire immediately. If the local
  agent's latest published stance is still non-terminal, agentd publishes one
  substantive closing `TurnFrame`: it preserves the latest `position` and
  `because`, reuses the most recent non-empty visible output (falling back to
  `(no output)` only when none exists), and sets `needs_from=[]`. This is a typed
  terminal turn, not a done signal. The checked publish is one-shot on success
  and re-armed after failure.
- If the latest local turn is already terminal, no extra closing turn is
  emitted.

The engine contract requires `TurnDone` delivery to be decoupled from
`session.send`: the final follow-up's completion must arrive through the event
pump after the cap reservation and retirement decision. Synchronously emitting
`TurnDone` inside `send` violates that ordering.

A sibling follow-up that starts after the prior turn completed re-pends the
task, restores its active steer marker, and re-subscribes its steer lane. Human
steering therefore survives across sibling rounds. The sibling lane itself
outlives an individual `TurnDone` until settlement, cap retirement, or failed
task startup; it is not retired merely because one engine turn ended.

## Connector rendering

Connectors remain dumb translation and rendering edges. Render intent comes
from typed origin data such as `ChannelOrigin.debate`; a connector does not infer
a debate, terminality, or completion from prose.

On debate intake, chatd may send one best-effort acknowledgement before task
execution. For a debate `TurnFrame`, it renders a stance-derived TLDR to the
parent channel and routes the complete `visible_text` to the anchored
thread/topic for that agent's full thought. TLDR and full-text pieces have
separate idempotency marks so redelivery cannot duplicate a piece that already
landed. A single-scope turn renders its complete text in the target channel.
Typed verification sources and limitations attach to the full answer (rendered
from `TurnFrame.verification` as a compact "Sources checked" / "Limitations"
suffix) rather than expanding the parent-channel TLDR.

These placement and idempotency rules are connector-owned presentation
behavior only. Typed frames remain the source of truth, and done remains
orchestrator-derived.

## Replay and idempotency

The orchestrator claims a task id before awaiting subscriptions or dispatch.
Redelivery of the same immutable `TaskFrame` is a no-op; reuse of the same id
with different content is rejected. Dispatch participants are recorded only
after confirmed server hand-off, so completion cannot derive from an intended
but unconfirmed dispatch.

Agentd rejects duplicate active or recently completed `(task_id, agent_id)`
work. Completed-task records are bounded by the configured replay horizon
(default 300 seconds) and persist across restart only when `state_home` is
wired; without it, deduplication is memory-only. Untimestamped or stale
untracked tasks fail closed, and corrupt durable state opens a distrust window
that refuses otherwise unprovable replays rather than executing them twice.

At the connector edge, intake ids are deduplicated before a task is minted.
Turn rendering also tracks delivered pieces: debate TLDR and full-text delivery
have separate marks, so a partial failure retries only the missing piece. These
marks prevent duplicate presentation; they do not create terminality or replace
the orchestrator's result lane.

## SteerFrame

A `SteerFrame` rides `rtech.task.<task_id>.steer` and contains `task_id`, `text`,
`target_agent`, and `interrupt`. The payload task id must match the lane-bound
task id, the task must be active locally, and `target_agent` must be either the
local agent id or the canonical broadcast token `all`. Broadcast is limited by
the per-task lane: it reaches the active agents subscribed for that task, not
every agent globally.

With `interrupt=false`, accepted text is injected into the live session. With
`interrupt=true`, agentd first calls the engine interrupt and then injects the
text. Interrupt intent is always the typed boolean field and is never parsed
from prose.

Each accepted steer appends a causal task reference to that task's active ref
list. When `TurnDone` becomes a `TurnFrame`, its `refs` begins with the task id
and includes the accepted steer references accumulated during that turn. The
current `SteerFrame` has no separate steer-id or `refs` field.

## Source anchors

- `packages/rtech-frames/src/rtech_frames/codec.py` and `frames.py`: raw frame
  codec, frame fields, broadcast token, and stance contract.
- `packages/rtech-frames/src/rtech_frames/done.py`: participant-scoped,
  sequence-aware done derivation.
- `packages/rtech-frames/src/rtech_frames/subs.py`: atomic subscription-set
  installation and retirement.
- `packages/rtech-orchestrator/src/rtech_orchestrator/subjects.py` and
  `orchestrator.py`: canonical subjects, fanout, graph accumulation, and result
  publication.
- `packages/rtech-agentd/src/rtech_agentd/nats_ingress.py` and `frame_turn.py`:
  fleet-envelope translation, FIFO correlation, sibling rounds, steering, typed
  frame publication, and evidence-gated publication with the cold-phase status.
- `packages/rtech-agentd/src/rtech_agentd/grounding.py`: authority order,
  source validation, verification merging, and guarded factual fallback.
- `packages/rtech-agentd/src/rtech_agentd/completed_store.py`: restart-surviving
  completed-task replay protection and corrupt-store distrust behavior.
- `packages/rtech-agentd/src/rtech_agentd/engine.py`: internal `TurnDone` event.
- `packages/rtech-chatd/src/rtech_chatd/intake_sink.py`, `bridge.py`, and
  `edge.py`: best-effort acknowledgement, typed render translation, debate TLDR
  placement, full-thought routing, and per-piece idempotency.
- `packages/rtech-agentd/tests/test_frame_turn.py`,
  `test_frame_turn_grounding.py`, and `test_steer_wire.py`: cap, atomic reserve,
  settlement, closing terminus, evidence gating, steer survival, interrupt,
  broadcast, and ref behavior.
