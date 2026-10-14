# Turn completion contract

A stop-rule redesign, not a patch. Two prior manifestations of the same root
(scripted stall; real-engine stall masked by a steer flush) shared one missing
contract: nothing bound *engine emits `TurnDone`* to *`frame_turn` publishes a
`TurnFrame`* to *the supervisor knows which task is in flight*. This note is the
ratified contract; the code enforces it.

## The contract

One agentd owns ONE managed engine session that runs ONE turn at a time. A turn
is bound to its task from the moment agentd sends it until the terminal
`TurnDone`.

1. **A task send opens exactly one turn that completes on its own.** The engine
   emits a terminal `TurnDone` (with `visible_text` and a typed `Stance`) when
   the turn is done — NEVER gated on a later steer "flush". A turn that only
   completes when a steer arrives is a contract violation.
2. **`frame_turn` publishes exactly one `TurnFrame` per completed turn**, on
   `rtech.task.<task_id>.turn` (`subjects.task_turn`). `visible_text` is
   guard-sanitized; `stance` is the engine-filled typed field carried straight
   through, never scraped from prose. agentd emits NO done token — done is
   orchestrator-derived from the frame graph.
3. **The in-flight task is registered with the supervisor for the whole turn.**
   `FrameTurnRunner._on_task_frame` calls `supervisor.track_in_flight_task(id)`
   before the send; the supervisor clears it on the matching `TurnDone`. This is
   what lets the stall watchdog map a wedged turn back to its `task_id` instead
   of logging "stall status skipped, no task" during a live turn.
4. **The steer subscription for a task closes when its turn ends.**
   `_on_engine_event` pops the task's steer state (refs + subscription marker)
   as it publishes the `TurnFrame`.

## Per-engine honoring

- **Scripted engine** (`scripted_engine.py`): every `send` opens AND completes a
  turn in the same call — the task send yields a `TurnDone` with no steer
  required. A subsequent send is a steer turn that also completes on its own, so
  the frame graph shows two completed turns (task, then steer). Deterministic;
  no held-open turn, so no teardown "flush" method exists.
- **Claude engine** (`claude_engine.py`): the turn completes on the SDK
  `ResultMessage`, which the reader translates to a `TurnDone` carrying the
  buffered turn text and the typed stance (from the `stance` tool, else a fixed
  position keyed on the SDK result). The supervisor pump fans that `TurnDone` to
  `frame_turn`, which publishes the `TurnFrame`.

## The regression this fixed

`FrameTurnRunner` sent the task to the engine but never registered the in-flight
task with the supervisor, so `supervisor._in_flight_task_id` stayed `None` and
the watchdog logged "stall status skipped, no task" during a live turn (the
mapping bug). Separately, the scripted engine held `TurnDone` open until a second
send, so a scripted task turn produced a `StatusFrame(working)` and then nothing
— the CT-271 host proof only got a `TurnFrame` via a steer flush, masking the
same missing completion semantics. Both are closed by the contract above.

## Tests that pin it

- `test_frame_turn.py::test_task_frame_tracks_in_flight_task_until_done` —
  the watchdog sees the active task while a turn runs; cleared on `TurnDone`.
- `test_frame_turn.py::test_scripted_task_completes_without_steer` (regression
  pin for the scenario) — a scripted task send drives a `TurnFrame` with no
  steer involved.
- `test_dev_stack_smoke.py` — the dev-stack wiring: task in → `TurnFrame` out,
  no steer, no flush call.
- `test_supervisor.py` — watchdog maps a stall to the tracked task.
- LIVE (scripted): `scripts/dev_stack.py` on a temp port → POST `/api/tasks` →
  SSE stream yields a `TurnFrame` within seconds, no steer.
