# Telegram live E2E (issue #111)

Authorization-gated real-account canary for the practical small-group debate
flow. It is reference-only: env vars name Vaultwarden references, never
credential values, and the harness holds no session strings. The suite is marked
`e2e` and **fails closed** — without explicit opt-in AND complete config, the
whole module skips at collection time, before any Telethon import or network
object exists.

## Run

```bash
# Default gate never runs it (marker deselected):
uv run pytest -q -m "not e2e"

# Authorized live run (opt-in + config present):
RTECH_TG_E2E_LIVE=1 uv run pytest -q -m e2e \
  packages/rtech-chatd/tests/e2e/test_telegram_live.py
```

## Authorization surface (reference names only — no secrets)

Config-time env vars name Vaultwarden references, never values:

- `RTECH_TG_E2E_LIVE` — explicit opt-in (`1`/`true`/`yes`/`on`).
- `RTECH_TG_E2E_GROUP_REF` — forum group reference.
- `RTECH_TG_E2E_NATS_URL_REF` — bus URL reference for frame observation.
- `RTECH_TG_E2E_<ROLE>_SESSION_REF` — Telethon session reference per role.
- `RTECH_TG_E2E_<ROLE>_USERNAME` — distinct identity per role.
- `RTECH_TG_E2E_<ROLE>_IDENTITY_BOT_SENDER_ID` — expected non-zero Telegram
  sender id for that identity's rendered TLDR.
- `RTECH_TG_E2E_<ROLE>_TOPIC_REF` — forum topic reference per role.

`<ROLE>` ∈ {`ARBITER`, `PARTICIPANT_ONE`, `PARTICIPANT_TWO`} — the three
usernames, three session references, and three topic references must each be
**distinct**; any repeat fails config construction closed (`must be distinct
across the trio`). The three identity-bot sender ids must also be non-zero and
distinct, so TLDR provenance is asserted from Telegram sender identity rather
than an inbound-message marker.

### Resolved material (values — populated ONLY on an authorized run)

The config holds no credential values. A live run resolves them from Vaultwarden
into these env vars at the last responsible moment, so a real deferred Telethon
client and a real bus connection are built only when authorized:

- `RTECH_TG_E2E_API_ID` / `RTECH_TG_E2E_API_HASH` — Telegram app credentials.
- `RTECH_TG_E2E_<ROLE>_SESSION_STRING` — resolved Telethon session string.
- `RTECH_TG_E2E_NATS_URL` — resolved bus URL for typed-frame observation.

Missing resolved material fails the scenario closed (`resolved Telethon material
… is incomplete`), never a config-only pass.

### Live target ids (values — populated ONLY on an authorized run)

Actions must send into concrete Telegram targets. A live run resolves these
dedicated env vars (`resolve_live_targets`); the reference-only config never holds
them. Each fails closed if missing (`must be resolved for a live action`):

- `RTECH_TG_E2E_FORUM_GROUP_ID` — the forum group the trigger/dedup/reaction
  scenarios post into.
- `RTECH_TG_E2E_<ROLE>_TOPIC_ID` — the per-role forum topic id an action targets
  via `reply_to`.
- `RTECH_TG_E2E_NON_FORUM_GROUP_ID` — a **DISTINCT** non-forum group the degrade
  scenario posts into; construction rejects it if it equals the forum group.

## Correlation model

The product mints an **OPAQUE `uuid4` task id**; a scenario cannot predict it and
cannot invent a `rtech.task.<id>.frames` subject. So each scenario correlates
rather than guesses:

1. Open a `CorrelatedObserver` **FIRST** — it subscribes the REAL product
   subjects (`rtech.task.new` ingress, the per-task turn wildcard
   `rtech.task.*.turn`, and the per-task steer wildcard `rtech.task.*.steer`)
   **BEFORE** any Telegram action and confirms them with a NATS flush, so a plain
   NATS pub/sub cannot miss the frames the action causes. Every capture retains
   the actual `Msg.subject`; turn/steer assertions require the discovered task's
   exact `rtech.task.<id>.turn` / `rtech.task.<id>.steer` subject.
2. Connect the intended real account and perform the action; `send_message`
   returns the **REAL** sent Telegram message id.
3. Discover the opaque task id via `discover_task_id`, matching the minted
   `TaskFrame.origin.message_id` to that sent id (a per-run `RunMarker` carried in
   the message text is a secondary guard so a pre-existing frame cannot match).
4. Filter `TurnFrame` / `SteerFrame` from the wildcards by the discovered id.

`RunMarker` is **text-only**: it makes this run's inbound message distinguishable
and invents no subject or task id. `LiveTelethonActor` maps
`connect`/`disconnect`/`send_message` (returning the real sent id)/`send_reaction`
and reads the live surface (`read_messages`/`read_forum_topics`) onto the real
Telethon client. The observer and actor are always torn down in `finally`
(`observer.close()` unsubscribes every subscription and closes the bus).

## Scenarios

Every scenario asserts **BOTH surfaces** — the typed bus frames AND the actual
Telegram surface read back with the actor. State is a typed frame field, never a
scraped prose trailer or a config-only assertion:

- **arbitration ack** — send a marked trigger into the arbiter's forum topic.
  Bus: the trio `TaskFrame` and one `TurnFrame` per participant within a bounded
  **ack-latency** budget. Telegram: three rendered TLDRs from three **distinct bot
  sender ids** after the pre-action baseline, plus three full typed thoughts
  matched to their expected topic ids.
- **duplicate-delivery suppression** — create a unique external terminal
  `TurnFrame` with origin, verify its safe visible marker has zero prior Telegram
  renders, then **replay that exact byte-identical frame twice on the real
  `rtech.task.<id>.turn` subject** while wildcard rendering is active and require
  exactly one matching Telegram render.
- **trigger gating** — a trigger mints a task; a following ordinary (non-trigger)
  message proves ABSENCE within a bounded window (`expect_no_task`). Both messages
  are confirmed on the Telegram surface.
- **reaction-driven steer** — explicitly skipped: the product no longer accepts
  `message_reaction` updates and has no reaction-to-steer producer, so a live
  reaction test would claim a product behavior that does not exist.
- **non-forum degradation** — post into the DISTINCT non-forum group (no topic
  `reply_to`); treat a `GetForumTopics` non-forum rejection as no topics, then
  require a new rendered `TurnFrame.visible_text` message with `topic_id is None`.

Provisioning reuses the merged `provision-debate-group` tooling.
