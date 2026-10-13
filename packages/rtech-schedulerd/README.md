# rtech-schedulerd

Cron-scheduled agent tasks: mint ordinary typed `TaskFrame`s on a schedule and
publish them on the canonical task-ingress subject (`rtech.task.new`).

Downstream agents do not know or care that a task was scheduled — a scheduled
task is byte-identical in shape to an intake-minted one. The bus stays the single
coordination truth; the scheduler is a dumb producing edge.

## What it is (and is not)

- **Is:** a small long-running daemon (own systemd/lingering unit, pattern)
  that evaluates a frozen schedule table, and at each occurrence mints a
  `TaskFrame` and publishes it to `TASK_INGRESS`.
- **Is not:** a distributed scheduler. There is **no Redis/database/distributed
  scheduler**, no connector-held scheduling state, no `done` frame, no prose
  sentinel/trailer, and no UI. The NATS frame remains coordination truth.

## Schedule storage

Schedules are a frozen config table loaded from a JSON file
(`RTECH_SCHEDULE_FILE`). Each entry is validated at startup and fails **closed**
(the daemon aborts) on any malformed schedule. There is no live-editing surface
in this slice.

## Determinism and delivery

- **Task id** per occurrence is deterministic:
  `sched-<schedule_id>-<epoch_seconds_of_occurrence>`. A process restart or
  redelivery that re-evaluates the same occurrence mints the **same** id, so
  retry/recovery cannot double-mint a duplicate task for one fire.
- **Transport is at-least-once, idempotency is downstream.** The bus provides
  at-least-once publish: an ambiguous publish failure may already have landed, so
  the occurrence is tombstoned and never republished in-process. Exactly-once at
  the consumer comes from the **downstream orchestrator's TaskFrame-id dedup**
  (it keys off `task.id` and no-ops a same-id redelivery), NOT from the wire —
  this daemon never claims exactly-once delivery. An in-process bounded reservation
  map is only a fast-path guard against a same-occurrence re-fire within one run.
- **Timezone** is explicit per schedule (IANA name, default `UTC`). Cron fields
  are evaluated in the schedule's own timezone; the minted `created_at` is UTC.
- **Misfire policy:** skip-missed. On startup the scheduler arms each schedule at
  the next occurrence strictly after "now" — occurrences that would have fired
  while the daemon was down are skipped, never fired late in a burst. No misfire
  state is persisted (bus is truth; the connector holds no scheduling state).

## Cron syntax

Standard 5-field cron: `minute hour day-of-month month day-of-week`. Supports
`*`, comma lists (`1,15`), ranges (`9-17`), steps (`*/15`, `0-30/10`, and the
wildcard-equivalent `*/1`), and Sunday as both `0` and `7` (in values, lists,
ranges, and steps).

Occurrence calculation is delegated to [`croniter`](https://pypi.org/project/croniter/)
behind a small owned typed wrapper (`rtech_schedulerd.cron`) — callers, config,
and tests never touch croniter directly. The wrapper holds the strict five-field
contract (exactly five fields; ascending ranges only; `@`-macros rejected) and
adds a generic timezone/DST guard around croniter's result; it does not
re-implement field parsing or occurrence scanning.

Day-of-month and day-of-week follow the **standard cron OR rule**: when BOTH day
fields are restricted a day matches if EITHER matches (`0 0 13 * 5` fires on the
13th *and* every Friday); when exactly one is `*` only the other constrains the
day.

**DST correctness:** occurrences are timezone-aware. A spring-forward
nonexistent wall time (e.g. 02:30 during a 02:00→03:00 jump) is skipped, never
fired at a phantom instant. A fall-back ambiguous wall time (e.g. 01:30 during a
02:00→01:00 fall) fires **at most once**, on its earlier (`fold=0`) real instant,
and never returns an occurrence at or before the queried instant — including when
queried during the second fold.

## Config env

- `RTECH_SCHEDULE_FILE` — path to the JSON schedule table (required).
- `RTECH_STACK_ID` — this stack's id, stamped on every minted frame's `stack`
  so a stack-guarded orchestrator consumes it. No default; fails
  closed when unset.
- `NATS_URL` — the fleet NATS server.
