# Dispatch-Confirmation Contract

This contract governs
when the orchestrator may record a participant as dispatched. It is the typed
identity source for participant-scoped done derivation, so it must reflect
reality: **dispatch is recorded ONLY after a confirmed hand-off to the server.**

## The defect this replaces

`await nc.publish(...)` on core NATS only enqueues the PUB command on the
client. The client may be reconnect-buffering: the coroutine returns, the
server never receives the frame, and the agent never sees the task. Recording
a participant on an unconfirmed publish therefore either waits forever for a
turn that can never come, or (worse) lies about who was dispatched.

## The rule

1. **Publish is enqueue, not delivery.** `publish_frame` success means the
   frame was accepted into the client's outbound buffer, nothing more.
2. **Confirmation is a server round-trip.** On core NATS a dispatch batch is
   confirmed by `publish(...)` per lane followed by ONE
   `await nc.flush(timeout)` round-trip (`FrameBus.confirm_publishes`). Flush
   returns only after the server has ponged, which orders it after every PUB
   enqueued on that connection — including publishes that were buffered across
   a reconnect.
3. **Record after confirm, never before.** The orchestrator publishes the full
   batch (one lane for `single`, one per participant for `duo`/`trio`), awaits
   `confirm_publishes()`, and only then records the participant set on the
   task graph. A turn arriving between publish and record is closed by the
   post-record completion re-check (unchanged from round 2).

## Failure semantics

- **Flush timeout / flush failure** → typed `FrameConfirmError` (a
  `FrameTransportError`). NO participant from the batch is recorded, any
  previously recorded subset for the task is rolled back
  (`TaskGraph.mark_dispatch_failed` clears the recorded set), and the task
  latches the typed partial-failure state `dispatch_failed`. A
  `dispatch_failed` task never derives done and never publishes a result —
  it fails closed, loudly.
- **Publish failure mid-batch** → typed `FramePublishError`; same rollback +
  `dispatch_failed` latch. The batch is all-or-nothing: a subset that
  confirmably reached some agents must still never satisfy the scope count.
- **Reconnect in flight** → publishes buffered by the client during a
  reconnect are NOT confirmed until a flush round-trip completes on the
  re-established connection. If flush cannot complete within the timeout, the
  dispatch is treated as failed exactly as above, even though the buffered
  frames may still drain later. That asymmetry is safe because of the retry
  rule below.

## Duplicate-dispatch protection on retry

A confirmation failure does **not** prove non-delivery: the frame may have
reached the server before the failure, or may drain from the reconnect buffer
afterward. Protection is identity-based, not delivery-count-based:

- A retry MUST re-publish the **same immutable `TaskFrame`** (same `id`).
  `FrameTurnRunner` enforces consumer idempotency by the
  `(task_id, participant identity)` key: a task already active or already
  completed by that agent is a typed no-op before status publication, steer
  subscription, or engine dispatch. Completed ids are retained for at least the
  configured `task_replay_horizon_s` (default 300 seconds, longer than the
  expected core-NATS reconnect buffer plus the operator retry window). The
  1,024-entry count is a soft memory bound only: the cache grows above it when
  required to preserve the horizon, and evicts an entry only when it is both
  over the count bound and older than the horizon.
- Every new producer stamps `TaskFrame.created_at`. An untracked frame older
  than `task_replay_horizon_s` fails closed at agent ingress before status
  publication, steer subscription, or engine dispatch. No `StatusFrame` is
  emitted for that stale no-op. Legacy timestamp-less frames remain decodable
  during wire migration, but cannot be retried safely after their first
  delivery; current repo producers never emit them.
- The orchestrator's ingress is idempotent (round-2 H3): a redelivered
  identical task on a claimed graph never re-subscribes or re-dispatches, and
  a same-id/different-payload frame is rejected.
- In this slice the orchestrator does NOT auto-retry: `dispatch_failed` is a
  latch, surfaced loudly for operator action. When automatic retry is added it
  must target only `dispatch_failed` graphs, reuse the original frame, and go
  through the same publish → confirm → record path.

## JetStream migration path (durable end-state)

Core-NATS flush confirms *server receipt*, not durable delivery to a consumer.
The durable end-state is JetStream: each dispatch lane publish becomes a
JetStream publish whose per-message **PubAck** is awaited before recording —
at-least-once delivery to a durable consumer, with the stream deduplicating on
`Nats-Msg-Id = <task_id>/<participant>` so the retry rule above stays safe.
That Msg-Id layer remains future belt-and-suspenders work in fleet-bus;
the implemented consumer invariant is the current retry-safety boundary.

The seam is already shaped for it: `confirm_publishes()` is the confirmation
boundary. A JetStream-backed `FrameBus` awaits the PubAck inside
`publish_frame` (each publish is individually confirmed) and implements
`confirm_publishes()` as a no-op. The orchestrator's
publish → confirm → record sequence is unchanged, so the migration touches
only the transport adapter, never the dispatch-recording logic. JetStream is
NOT built in this slice; this contract only guarantees forward compatibility.
