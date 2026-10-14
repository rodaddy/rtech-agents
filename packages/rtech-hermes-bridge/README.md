# rtech-hermes-bridge

A **content-only dumb edge** that bridges rtech-agents task frames to the
**existing Hermes gateway agents** already running on the shared fleet NATS bus.
Each fronted fleet agent (e.g. `bilby`) then appears in the rtech web-chat picker
and answers in its own voice — without deploying agentd per box.

The bridge translates at the boundary and carries **no control signal**: reply
text and typed stance ride through as content `TurnFrame`s, and terminal
semantics for a bridged participant live in the **orchestrator**, which consumes
the gateway's typed `fleet_core.spec.Done` on its own fleet-side subscription
and records it as in-graph terminal evidence. Full design:
`docs/bridged-participant-lifecycle.md` (repo root docs/).

## Pinned gateway bus contract (bilby, read-only inspection)

The Hermes gateway reaches the fleet bus via the **`fleet_subscriber` NATS
platform adapter** installed in its venv — **not** via `hermes_cli/gateway.py`
(that file is only the systemd/CLI wrapper). Running env: `FLEET_ENV=dev`,
`FLEET_AGENT_ID=bilby`, `FLEET_ROLE=worker`, `FLEET_NATS_URL=nats://127.0.0.1:4222`.

Clean source lives in the fleet-bus repo at `packages/fleet-subscriber`
(`adapter.py` / `mapping.py` / `completion.py`) — the on-host `hermes_cli` copy is
identifier-minified; the contract was read from the clean library.

### Inbound — what the gateway subscribes to (worker role)

| Lane | Subject |
|---|---|
| dispatch (bare) | `dev.agent.dispatch.<id>` |
| dispatch (per-channel) | `dev.agent.*.dispatch.<id>` |
| relay / wake / inbox | `dev.agent.{relay,wake,inbox}.<id>` |

Inbound is a fleet `Envelope`. `envelope_to_message` forges an internal agent turn
only when the envelope carries **`payload.text`** (non-empty) **and** a resolvable
channel (`envelope.channel` or the agent's configured `home_channel`); otherwise
the turn is skipped. `task_id` is carried so the reply keys back to the same id.

The bridge publishes on the **bare** `dev.agent.dispatch.<id>` lane (the gateway
subscribes it directly), `kind="dispatch"`, `payload.text` = the task text,
`task_id` = the rtech task id, `channel` = the configured default channel.

### Outbound — what the gateway emits, and what the bridge does with it

The adapter's `send()` publishes the agent's reply as `kind="status"` envelopes
on `dev.agent.status.<task_id>` with `payload.text` = the agent's own words and,
when present, the typed fleet-core `Stance` as the envelope's typed signal. When
the turn completes it emits one `kind="done"` envelope carrying the typed
`fleet_core.spec.Done`.

The bridge's reverse path is **content only**:

- each identity-checked `status` reply publishes an immediate content
  `TurnFrame` on `rtech.task.<task_id>.turn` — `visible_text` is the named
  `payload["text"]` verbatim, and the stance fields are a verbatim copy of the
  gateway's typed stance (fallback attribution `responded`/`hermes-bridge` when
  none rode the envelope). No prose parsing anywhere; every field is typed.
- the typed `Done` is consumed for **lifecycle only** (unsubscribe the reply
  lane, drop per-task state, advance the agent's queue). It never becomes a
  frame. The orchestrator consumes the same typed `Done` on its own
  subscription (`RTECH_BRIDGED_AGENTS` on the orchestrator side) and derives
  done in one place; a bridged participant's turn stance is IGNORED for
  terminality by contract.

Identity is fail-closed: an envelope whose `sender` or `task_id` (or typed
`Done.task_id`) does not match the dispatched task is dropped loudly.

### No agent-side change required

The gateway already accepts inbound work on its dispatch lane and replies on the
status lane. Bridging needs **zero** Hermes-side changes.

## Data flow (end to end)

```
web-chat  --TaskFrame-->  rtech.task.new
orchestrator  --TaskFrame-->  rtech.agent.<id>.task     (records dispatch,
  bridge  --Envelope(dispatch)-->  dev.agent.dispatch.<id>   marks <id> bridged,
    Hermes gateway runs a full agent turn                    subscribes fleet done lane)
  Hermes  --Envelope(status)-->  dev.agent.status.<task_id>
  bridge  --content TurnFrame-->  rtech.task.<task_id>.turn
  Hermes  --Envelope(done, typed Done)-->  dev.agent.status.<task_id>
    bridge: lifecycle only (no frame)
    orchestrator: typed Done -> TerminalEvidence -> derives done -> result lane
web-chat renders the result
```

The bridge also heartbeats each fronted fleet id on
`rtech.agent.<id>.heartbeat` (a fleet `Envelope`, `kind="heartbeat"`,
`sender=<id>`) so the web-chat presence picker lists it exactly like a native
agentd agent.

## Discord double-answer (host-side contract)

A bridged agent can answer **twice** if the bus-direct path AND the gateway's own
Discord voice-tee both fire for the same turn. The gateway's Discord behavior is
driven by its own env (`FLEET_VOICE_TEE` / `DISCORD_FREE_RESPONSE_CHANNELS`) and
by the `channel` on the inbound envelope. The bridge exposes **no switch** for
this and never edits the gateway's config (agents do not self-modify): the
operator keeps `FLEET_VOICE_TEE` off for a bridged agent, and the bridge stamps
`default_channel` only when one is configured.

## Task lifecycle (bounded, serialized)

- One in-flight task per fleet id (the gateway keys reply correlation per
  channel and overwrites it per dispatch); later tasks queue FIFO, bounded by
  `RTECH_BRIDGE_MAX_QUEUE_DEPTH` — past that, admission is shed loudly with a
  typed `StatusFrame(blocked)`.
- Reply-lane subscription is established BEFORE the dispatch publish; failures
  roll back so a redelivered task can retry.
- Tasks with no typed completion expire after
  `RTECH_BRIDGE_TASK_TIMEOUT_SECONDS` (blocked status, loud log). Finalization
  is claimed atomically, so completion and the expiry sweeper can never
  double-finalize.
- Finished task ids are remembered in a bounded window for idempotent
  redelivery.

## Configuration (environment)

| Var | Default | Meaning |
|---|---|---|
| `RTECH_BRIDGE_FLEET_IDS` | *(required)* | Comma-separated fleet ids to front, e.g. `bilby` |
| `RTECH_BRIDGE_FLEET_ENV` | `dev` | Fleet subject env prefix (match the gateway) |
| `RTECH_BRIDGE_RTECH_NATS_URL` | live fleet bus | Bus carrying `rtech.*` frames |
| `RTECH_BRIDGE_FLEET_NATS_URL` | live fleet bus | Bus carrying `{env}.agent.*` envelopes |
| `RTECH_BRIDGE_DEFAULT_CHANNEL` | *(unset)* | Fallback Discord channel for a task with none |
| `RTECH_BRIDGE_HEARTBEAT_INTERVAL_SECONDS` | `30` | Picker liveness cadence |
| `RTECH_BRIDGE_TASK_TIMEOUT_SECONDS` | `900` | In-flight expiry without typed completion |
| `RTECH_BRIDGE_MAX_QUEUE_DEPTH` | `64` | Per-agent FIFO bound; excess sheds loudly |

Run it: `python -m rtech_hermes_bridge`. The orchestrator pairs with it via
`RTECH_BRIDGED_AGENTS=<same ids>` so it consumes the typed fleet `Done` for
those participants.

## Dependencies

`fleet-nats` (the wire contract + `Envelope` encoder — never hand-rolled),
`rtech-frames` (typed frames + codec), `rtech-orchestrator` (the ratified rtech
subject builders). Local path sources for dev/CI; deploys pin fleet-bus as a git
dep, same as agentd.
