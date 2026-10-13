# rtech-orchestrator

Minimal derived-done orchestrator for rtech-agents.

## What it does

- Subscribes to the task-ingress subject (`rtech.task.new`).
- For `scope=single` tasks, routes the `TaskFrame` **directly** to agentd
  (`rtech.agent.<agent_id>.task`) — no discussion round for single scope.
- Accumulates the per-task frame graph (turns + statuses) in a store.
- **Derives** completion via `rtech_frames.done.is_done` from the frame graph —
  never from a token, a frame field, or a scraped prose trailer.
- On derived done, re-publishes the task's terminal `TurnFrame` (for single
  scope, the single agent's terminal turn) onto `rtech.task.<id>.result` for
  edges to render. This is **orchestrator state** (the orchestrator *chose* to
  publish because done became true), not a new "done" claim by any agent.

## Ratified subject scheme

| Subject | Direction | Purpose |
|---|---|---|
| `rtech.task.new` | edge -> orchestrator | Task ingress |
| `rtech.task.<task_id>.turn` | agent -> bus | Per-task agent turns |
| `rtech.task.<task_id>.status` | agent -> bus | Per-task machine status |
| `rtech.task.<task_id>.steer` | human -> bus | Per-task human steering |
| `rtech.task.<task_id>.result` | orchestrator -> bus | Terminal render (orchestrator-only publish) |
| `rtech.agent.<agent_id>.task` | orchestrator -> agentd | Agent dispatch |

## Invariants

- **Done is DERIVED** from the frame graph, never token- or frame-asserted.
  No completion tokens, no sentinels, and no prose-trailer parsing anywhere.
- **Signals are typed frame fields only** (`StatusFrame.state`,
  `TurnFrame.stance.needs_from`), never prose.
- The in-memory graph store is **POC-only**. `GraphStore` is the Protocol seam
  so a durable store can replace it without touching the orchestrator.
- The transport is behind the `FrameBus` Protocol. A fleet-nats-backed adapter
  (`encode_frame`/`decode_frame` over FleetBus raw bytes) is a **later slice**;
  `FrameBus` is the seam that keeps the orchestrator transport-agnostic and
  testable.
