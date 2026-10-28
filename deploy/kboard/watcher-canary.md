# Watcher canary kit — trio-debate in a fresh no-agents channel

A paste-and-start runbook for proving the trio-intake debate path end to end, in **one fresh Discord channel with no Hermes agents
present**, driven by a **dedicated watcher bot** (its own Discord app, not any
agent bot). An intake hit fans out to a real duo/trio `TaskFrame`; each
participant's turn renders back as a stance **TLDR** in the channel plus **full
text in one lazily-created thread**, with a single closing marker.

This is additive to the rootless k-board deploy — it reuses the existing units
(`deploy/kboard/install.sh`, `rtech-chatd.service`, `rtech-agentd@.service`,
`rtech-orchestrator.service`) and adds only env. Nothing here touches
`k-board-api` / `caddy` / `postgres`. Env templates live in
`deploy/kboard/env/`; fill them into the shared `~/rtech-agents/.env`.

> **Env-var provenance.** Every var below is grepped against the source on
> `main`; see "Derived from (file:line per var)" at the end. Careful with the
> similarly-named vars: `RTECH_CHATD_PARTICIPANTS` (service.py:62) is chatd's
> own outbound task mint and is NOT the debate roster — the trio debate uses
> `RTECH_CHATD_DEBATE_PARTICIPANTS`. The stock `.env.example` now carries the
> intake/debate block too, but the templates in `deploy/kboard/env/` are the
> paste-and-start source for this kit.

---

## 1. Operator prereqs (do these by hand first)

These are console/Vaultwarden actions the rootless installer does NOT and CANNOT
do (mirrors `docs/canary-rollback-runbook.md` "Not automated").

- [ ] **Create a dedicated watcher Discord application + bot.** A NEW app in the
      Discord developer portal — distinct from Nimbus/Birk/Sprocket bots. This
      bot is the *edge*, not an agent.
- [ ] **Enable the privileged Message Content intent** on that bot. chatd
      declares `intents.message_content = True` and `intents.guild_messages =
      True` (`discord_adapter.py:145-147`); without Message Content the gateway
      delivers empty message bodies and no intake trigger can ever match. DM
      intent stays off unless `allow_dms` is set — leave it off.
- [ ] **Grant channel permissions** for the watcher bot in the test channel:
      **View Channel / Read Message History**, **Send Messages**, and **Create
      Public Threads** (+ Send Messages in Threads). The debate render posts a
      TLDR to the parent channel and full text into a thread it creates
      (`edge.py:122 _debate_thread_name`, `discord_adapter.py:335-343
      chatd.thread | created`). No Create-Public-Threads permission ⇒ thread
      creation fails and you get a wall of inline text.
- [ ] **Invite the watcher bot to the server and the ONE fresh channel.** Create
      a fresh channel for the test. **Exclude the Hermes agent bots from it**
      (no Nimbus/Birk/Sprocket Hermes presence) so nothing double-answers — the
      only bot in-channel is the watcher edge; the three participants act purely
      as bus consumers (agentd), never as in-channel Discord bots.
- [ ] **Store the watcher bot token in Vaultwarden.** Ref:
      `<your-secret-store-path>/RTECH_WATCHER_BOT_TOKEN`. The config expects an
      indirection ref in `RTECH_CHATD_TOKEN_REF` (default `env:RTECH_CHATD_BOT_TOKEN`);
      the operator fetches the VALUE from Vaultwarden and writes it into the env
      var that ref names, in `~/rtech-agents/.env` — **never** a literal token in
      any committed file. `validate_token_ref` fails closed on a literal without
      echoing it (`config.py:439-456`).
- [ ] **Capture the fresh channel id** — it becomes `RTECH_CHATD_INBOX_CHANNEL`.
- [ ] **Confirm the three participants have committed homes.** `agents/nimbus`,
      `agents/birk`, `agents/sprocket` each carry `agent.yaml` (+ `SOUL.md`);
      already on `main`. Their Discord `bot_token_ref`s in agent.yaml are for
      their OWN bots and are irrelevant to this watcher-only channel test.

---

## 2. Service topology (what runs where)

All on the k-board LXC (your deploy host), rootless systemd **user**
units, one shared `~/rtech-agents/.env`. Per `install.sh` reality the default
`RTECH_AGENT_INSTANCES` is exactly the three trio agents:

| Unit | Role | Count |
|---|---|---|
| `rtech-chatd.service` | the **watcher** edge: intake gate + debate fanout + TLDR/thread render-back | 1 |
| `rtech-orchestrator.service` | derived-done coordinator; `TASK_INGRESS` fans the debate frame to each lane | 1 |
| `rtech-agentd@agentd-nimbus` | triage participant (a debate voice) | 1 |
| `rtech-agentd@agentd-birk` | participant | 1 |
| `rtech-agentd@agentd-sprocket` | participant | 1 |

So: **1 chatd (watcher) + 1 orchestrator + 3 agentd = 5 units.** `install.sh`
enables orchestrator + the three agentd + webchat, and **copies but does not
enable** `rtech-chatd.service` (operator opt-in — enabling without a token/inbox/
stack would crash-loop). The webchat edge is not needed for this test.

**Atomic-ship requirement (README).** The trio-debate path needs the
orchestrator + agentd built from the branch with the `TASK_INGRESS` fan-out
consumer AND an rtech-frames build that decodes `origin`/`debate`
(SCHEMA_VERSION 6). On `main` today it does. Upgrade rtech-frames
consumers first, then orchestrator + agentd, **then** enable the chatd debate
participants — a debate frame published against an old orchestrator (no
`TASK_INGRESS` fan-out) or an old rtech-frames (`extra="forbid"` rejects the
`debate` key) is dropped and the debate never runs.

---

## 3. Exact env per unit

Shared file: `~/rtech-agents/.env` (all user units load the same
`EnvironmentFile`). Full templates: `deploy/kboard/env/*.env.example`. The
watcher canary uses a dedicated **`RTECH_STACK_ID=canary`** across all roles so
it never cross-consumes the live `dev` stack on the shared bus.

### 3a. Watcher chatd (`rtech-chatd.service`)

```
RTECH_CHATD_NATS_URL=nats://127.0.0.1:4223
RTECH_STACK_ID=canary
RTECH_CHATD_TOKEN_REF=env:RTECH_CHATD_BOT_TOKEN
RTECH_CHATD_BOT_TOKEN=<from Vaultwarden <your-secret-store-path>/RTECH_WATCHER_BOT_TOKEN>
RTECH_CHATD_INBOX_CHANNEL=<fresh channel id>
RTECH_CHATD_INTAKE_ENABLED=1
RTECH_CHATD_INTAKE_MODE=listed
RTECH_CHATD_TRIAGE_AGENT=agentd-nimbus
RTECH_CHATD_INTAKE_NAME_TRIGGERS=debate,trio
RTECH_CHATD_INTAKE_TRIGGER_PREFIXES=!debate
RTECH_CHATD_DEBATE_PARTICIPANTS=agentd-nimbus,agentd-birk,agentd-sprocket
# RTECH_CHATD_ALLOW_BOTS=none   # default; humans-only for the test
```

Notes derived from `rtech_chatd.config`:
- `RTECH_CHATD_TRIAGE_AGENT` MUST be one of `RTECH_CHATD_DEBATE_PARTICIPANTS`
  (the triage agent is a voice, not a bystander) and MUST be a deployed
  `agentd-<name>` id — a bare `nimbus` fails closed at config build.
- `RTECH_CHATD_DEBATE_PARTICIPANTS` count MUST be 2 (duo) or 3 (trio); UNSET
  collapses to single-scope intake (Nimbus only). Order is preserved.
- Discord ids (member/inbox) are opaque public ids, not secrets. The ONLY
  credential is the bot token, held as a ref.

### 3b. Nimbus agentd (`rtech-agentd@agentd-nimbus`)

```
NATS_URL=nats://127.0.0.1:4223
RTECH_STACK_ID=canary
# Optional — only if you want Nimbus's in-turn wake_agent tool live:
# RTECH_WAKE_ALLOWED_AGENTS=agentd-birk,agentd-sprocket
```

- `RTECH_AGENT_ID=agentd-nimbus` is set by the systemd instance name (`%i` in
  `rtech-agentd@.service`, `Environment=RTECH_AGENT_ID=%i`), NOT by `.env`.
- Engine `kind=claude` is pinned by `agents/nimbus/agent.yaml` (manifest is
  authoritative for engine kind); leave `DEV_STACK_ENGINE` UNSET for the real
  engine. Setting `RTECH_WAKE_ALLOWED_AGENTS` requires `RTECH_STACK_ID`; wake is
  Claude-engine-only, which matches.

### 3c. Birk / Sprocket agentd (`rtech-agentd@agentd-birk`, `@agentd-sprocket`)

```
NATS_URL=nats://127.0.0.1:4223
RTECH_STACK_ID=canary
```

- Engine `claude` from `agents/birk/agent.yaml` / `agents/sprocket/agent.yaml`.
  A **codex** engine option exists in-code (`rtech_agentd.codex_engine`) but is
  out of scope here — wake is Claude-only, and the canary uses `claude`.
- **Leave `RTECH_WAKE_ALLOWED_AGENTS` UNSET** on these two — a woken agent must
  not itself be a waker (bounds wake-chaining).

### 3d. Orchestrator (`rtech-orchestrator.service`)

```
NATS_URL=nats://127.0.0.1:4223
RTECH_STACK_ID=canary
RTECH_TARGET_AGENT_ID=agentd-nimbus
```

- `RTECH_STACK_ID` and a routing target are both REQUIRED — each fails loudly at
  startup if unset (`resolve_stack_id`, `resolve_target_agent_id`).

> **NATS auth on the authed cluster.** `:4223` is the authed 3-node fleet-js
> cluster (three nodes in the author's deploy). The legacy hub serves the
> unauthed `:4222` only — it is NOT a `:4223` node. fleet-nats resolves its
> credentials from its own env/creds ref (never a literal in these files);
> supply the creds to the process the same way the live `dev` deploy already
> does. Do not inline a credential anywhere in `.env` or a unit.
>
> **First-canary fallback:** if the host has no `:4223` credentials provisioned
> run the whole canary on the legacy
> `nats://127.0.0.1:4222` — every role on the SAME bus, never mixed. The
> first deploy ran on `:4222` for exactly this reason.

---

## 4. Bring it up

```sh
# On the k-board host, AS birk (never root/sudo). Assumes install.sh already ran
# once (units copied, tree synced). Fill ~/rtech-agents/.env from the templates:
$EDITOR ~/rtech-agents/.env    # paste sections 3a-3d, set RTECH_STACK_ID=canary,
                               # inbox channel id, watcher token from Vaultwarden

systemctl --user daemon-reload
# Participants + coordinator (already enabled by install.sh; restart to load env):
systemctl --user restart rtech-orchestrator \
  rtech-agentd@agentd-nimbus rtech-agentd@agentd-birk rtech-agentd@agentd-sprocket
# The watcher edge (operator opt-in — install.sh does NOT enable it):
systemctl --user enable --now rtech-chatd.service
```

---

## 5. The test

**In the fresh channel, type a role-assigned debate ask that hits the intake
gate.** The message must trip a trigger — begin with `!debate`, or contain a
whole-word `debate`/`trio` name trigger (from section 3a). Example (an
arbitration prompt in the operator's style):

```
!debate You are three siblings arbitrating one call: should the fleet run the
trio debate on the authed :4223 cluster or stay on :4222 for this canary?
Nimbus open, Birk counter, Sprocket break the tie. One paragraph each, then a
one-line closing verdict.
```

### What SHOULD appear

1. **N stance TLDR lines in the channel** — one short `**agentd-<name>**` TLDR
   per participant (stance-derived, ≤ ~200-char "because" clause;
   `bridge.py:234 turn_frame_to_tldr`). For a trio, up to 3 + 1 closing author =
   4 distinct TLDR authors (`edge.py:90 DEBATE_DISTINCT_AGENT_TLDR_CAP`).
2. **One thread** named `debate <task_id>` created off the trigger message, with
   each participant's **full position text** posted inside it
   (`edge.py:122 _debate_thread_name`, lazily created once).
3. **A single closing / `✅ done`** marker rendered exactly once (orchestrator-
   derived completion on the per-task result subject; the edge never authors
   "done" — README "chatd Discord edge").

### What deviations mean

| Symptom | Likely cause | Where to look |
|---|---|---|
| **No response at all** | intake gate never fired | Is `RTECH_CHATD_INTAKE_ENABLED=1`? Did the message trip a trigger? Journal for `gates.intake_hit \| drop=...` (bot/user/channel/no-trigger) vs `gates.intake_hit \| hit` |
| **Wall of full text, no thread** | thread creation failed | Watcher bot lacks **Create Public Threads**. Journal for `chatd.thread \| create failed` (`discord_adapter.py:335-341`) |
| **Only Nimbus answers, no debate** | `RTECH_CHATD_DEBATE_PARTICIPANTS` unset/empty → single-scope intake | Confirm the var is set to all 3 ids and `RTECH_STACK_ID` matches the orchestrator; journal `chatd.intake_sink \| published \| subject=...` should show `TASK_INGRESS`, not a single `rtech.agent.agentd-nimbus.task` |
| **Debate frame minted but silence from agents** | orchestrator/frames build mismatch (atomic-ship) or stack mismatch | Orchestrator journal: is it consuming ingress? Confirm every role's `RTECH_STACK_ID=canary` |
| **Turn posts but no closing** | result-subject / derived-done not observed | Orchestrator journal for the per-task result/done |

---

## 6. Verification commands

Read-only. Run on the k-board host as birk.

```sh
# Per-agent deploy check (config + persona + enabled + active; publishes nothing):
bash scripts/verify-agent-deploy.sh agentd-nimbus
bash scripts/verify-agent-deploy.sh agentd-birk
bash scripts/verify-agent-deploy.sh agentd-sprocket

# Watch the whole flow live across the five units:
journalctl --user -u rtech-chatd -u rtech-orchestrator \
  -u rtech-agentd@agentd-nimbus -u rtech-agentd@agentd-birk \
  -u rtech-agentd@agentd-sprocket -f
```

**Exact log lines to watch (source-verified):**

| Stage | Log line | Source |
|---|---|---|
| Intake hit | `gates.intake_hit \| hit channel=<id> thread=<id>` | `gates.py:340` |
| Debate mint + fanout publish | `chatd.intake_sink \| published \| subject=<TASK_INGRESS> triage=agentd-nimbus ...` | `intake_sink.py:321` |
| (single-scope drop-back) | subject shows `rtech.agent.agentd-nimbus.task` instead of the ingress token ⇒ participants unset | `intake_sink.py:286` |
| Thread render | `chatd.thread \| created \| channel=<id> thread=<id>` | `discord_adapter.py:343` |
| agentd clean start | `agentd.config \| loaded agent.yaml`, `engine selected kind=claude`, `agentd.run \| agentd started` | `__main__.py`, canary runbook §Verification |
| waker mode (if wake set) | `agentd.wake \| waker mode active \| agent=agentd-nimbus targets=[...]` | `__main__.py:516` |

Optional bus observation (needs the authed creds for `:4223`):

```sh
# Heartbeat + lifecycle per participant (see docs/canary-rollback-runbook.md):
nats --server nats://127.0.0.1:4223 sub 'dev.agent.agentd-nimbus.heartbeat'
nats --server nats://127.0.0.1:4223 sub 'rtech.agent.agentd-nimbus.lifecycle'
```

---

## 7. Rollback

Mirrors the existing runbook style — stop the units, nothing else touched.
`k-board-api` / `caddy` / `postgres` are never in scope.

```sh
# Stop the watcher edge (it was the operator opt-in):
systemctl --user disable --now rtech-chatd.service

# Optionally stop the participants + coordinator (leave them if the live 'dev'
# stack shares them — the canary is isolated only by RTECH_STACK_ID=canary):
systemctl --user stop rtech-orchestrator \
  rtech-agentd@agentd-nimbus rtech-agentd@agentd-birk rtech-agentd@agentd-sprocket

# To fully revert env, remove the watcher/canary lines from ~/rtech-agents/.env
# (back it up first) and restart whatever you left running. No unit files,
# no source, no other service is modified.
```

If a participant unit latches `failed` with `start request repeated too quickly`
(outer StartLimit), clear it deliberately:

```sh
systemctl --user reset-failed rtech-agentd@agentd-nimbus
systemctl --user start rtech-agentd@agentd-nimbus
```

---

## Telegram edge variant (`rtech-tgd`)

The SAME trio-debate canary, driven by the standalone **Telegram** edge
(`rtech-tgd.service`) instead of Discord. Everything downstream is identical —
orchestrator, the three agentd participants, the derived-done contract — only the
edge changes. Long polling means there is **no inbound port, no webhook, and no
public domain**: the edge reaches OUT to `api.telegram.org`; nothing reaches in,
so there is nothing to expose or proxy.

### Operator prereqs (Telegram)

- [ ] **Create a dedicated watcher Telegram bot via @BotFather** (its own app,
      distinct from the agent bots). This bot long-polls and posts the edge acks.
- [ ] **Create one per-agent bot for each debate voice** (nimbus/birk/sprocket) —
      the multi-identity render posts each turn under that agent's OWN bot. Store
      each token in Vaultwarden (`<your-secret-store-path>/RTECH_TG_TOKEN_<NAME>`).
- [ ] **Disable each bot's group privacy mode** (@BotFather → Bot Settings →
      Group Privacy → Turn OFF) so the polling bot receives non-command group
      messages; without it the gateway delivers empty bodies and no trigger ever
      matches (the Telegram twin of the Discord Message-Content intent).
- [ ] **Create a fresh supergroup, enable Topics (forum mode)** for the per-agent
      full-text lanes, and add all four bots. Capture the canonical `-100…` chat
      id — it becomes `RTECH_TG_ALLOWED_GROUPS` (required; empty fails closed).
- [ ] **Store all bot tokens in Vaultwarden**, then write their values into the
      corresponding env vars; only `env:<ENV_VAR>` refs live in `.env`, never a
      literal. `validate_token_ref` fails closed without echoing the supplied ref.

### Env (Telegram watcher)

Full template: `deploy/kboard/env/watcher-tgd.env.example`. Minimum:

```
RTECH_CHATD_NATS_URL=nats://127.0.0.1:4223
RTECH_STACK_ID=canary
RTECH_CHATD_TOKEN_REF=env:RTECH_CHATD_BOT_TOKEN
RTECH_CHATD_BOT_TOKEN=<from Vaultwarden <your-secret-store-path>/RTECH_TG_WATCHER_BOT_TOKEN>
RTECH_TG_ALLOWED_GROUPS=<fresh supergroup -100… id>
RTECH_TG_IDENTITY_MAP=agentd-nimbus=env:RTECH_TG_TOKEN_NIMBUS,agentd-birk=env:RTECH_TG_TOKEN_BIRK,agentd-sprocket=env:RTECH_TG_TOKEN_SPROCKET
RTECH_TG_TOKEN_NIMBUS=<from Vaultwarden>
RTECH_TG_TOKEN_BIRK=<from Vaultwarden>
RTECH_TG_TOKEN_SPROCKET=<from Vaultwarden>
RTECH_CHATD_INTAKE_ENABLED=1
RTECH_CHATD_INTAKE_MODE=listed
RTECH_CHATD_TRIAGE_AGENT=agentd-nimbus
RTECH_CHATD_INTAKE_NAME_TRIGGERS=debate,trio
RTECH_CHATD_INTAKE_TRIGGER_PREFIXES=!debate
RTECH_CHATD_DEBATE_PARTICIPANTS=agentd-nimbus,agentd-birk,agentd-sprocket
```

Bring it up (the agentd + orchestrator env is unchanged from sections 3b–3d):

```sh
$EDITOR ~/rtech-agents/.env    # paste the Telegram block above
systemctl --user daemon-reload
systemctl --user restart rtech-orchestrator \
  rtech-agentd@agentd-nimbus rtech-agentd@agentd-birk rtech-agentd@agentd-sprocket
# The Telegram watcher edge (operator opt-in — install.sh does NOT enable it).
# If a PRIOR fail-closed start (missing/invalid env) latched the unit `failed`
# with `start request repeated too quickly` (outer StartLimit), clear it BEFORE
# restarting — after correcting the env — or the fixed unit will not start:
systemctl --user reset-failed rtech-tgd.service
systemctl --user enable --now rtech-tgd.service
```

### The test (Telegram)

In the fresh supergroup, send a message that trips the intake gate — begin with
`!debate`, or include a whole-word `debate`/`trio`. A 👀 reaction acks the hit,
then each participant's stance TLDR posts under its own bot, with full text in a
per-agent forum topic (or a lazily-created thread when no lane is mapped).

### What deviations mean (Telegram)

| Symptom | Likely cause | Where to look |
|---|---|---|
| **No response at all** | intake gate never fired, or the group is not allow-listed | Is `RTECH_CHATD_INTAKE_ENABLED=1`? Is the group's `-100…` id in `RTECH_TG_ALLOWED_GROUPS`? Journal `gates.intake_hit \| drop=...` vs `... \| hit` |
| **Service crash-loops on start** | fail-closed config | Journal for `RTECH_STACK_ID must be set`, `RTECH_TG_ALLOWED_GROUPS is required`, `token_ref must be an indirection ref`, or `chatd.tgd: configured bot token not provisioned` |
| **Empty message bodies, no trigger ever matches** | bot group-privacy mode still ON | @BotFather → Bot Settings → Group Privacy → OFF for the polling bot |
| **Service crash-loops before ingress** | `RTECH_TG_IDENTITY_MAP` unset, partial, or unprovisioned | Confirm every participant has an `agentd-<name>=env:<ENV_VAR>` entry with a value; missing identities fail closed and never fall back to the edge bot |
| **Bot silently blind to reactions** | `allowed_updates` missing `message_reaction` | The edge always subscribes it (`TelegramConfig` fails closed otherwise); confirm no custom `allowed_updates` override |
| **Only Nimbus answers, no debate** | `RTECH_CHATD_DEBATE_PARTICIPANTS` unset/empty → single-scope intake | Confirm all 3 ids set and `RTECH_STACK_ID` matches the orchestrator; journal `chatd.intake_sink \| published \| subject=<TASK_INGRESS>` |
| **Debate frame minted but agents silent** | orchestrator/frames build mismatch (atomic-ship) or stack mismatch | Confirm `SCHEMA_VERSION >= 6` and every role's `RTECH_STACK_ID=canary` |
| **Turn posts but no closing** | result-subject / derived-done not observed | Orchestrator journal for the per-task result/done |

### Rollback (Telegram)

```sh
systemctl --user disable --now rtech-tgd.service
# Optionally stop the shared participants + orchestrator (as in section 7).
# To fully revert env, remove the RTECH_TG_* / RTECH_CHATD_* lines from
# ~/rtech-agents/.env (back it up first). No unit files or source are modified.
```

## Derived from (file:line per var)

Every env var this kit uses, traced to the source on `main`. Paths relative to
repo root.

| Var | Source (file:line) |
|---|---|
| `RTECH_CHATD_NATS_URL` | `packages/rtech-chatd/src/rtech_chatd/config.py:417` |
| `RTECH_CHATD_TOKEN_REF` | `packages/rtech-chatd/src/rtech_chatd/config.py:418` |
| `RTECH_CHATD_BOT_TOKEN` (env ref target) | `packages/rtech-chatd/src/rtech_chatd/config.py:390,470-472` |
| `RTECH_CHATD_INBOX_CHANNEL` | `packages/rtech-chatd/src/rtech_chatd/config.py:419` |
| `RTECH_CHATD_ALLOW_BOTS` | `packages/rtech-chatd/src/rtech_chatd/config.py:44` (AllowBots), README `deploy/kboard/README.md:259` |
| `RTECH_CHATD_INTAKE_ENABLED` | `packages/rtech-chatd/src/rtech_chatd/config.py:312` |
| `RTECH_CHATD_INTAKE_MODE` | `packages/rtech-chatd/src/rtech_chatd/config.py:313` |
| `RTECH_CHATD_TRIAGE_AGENT` | `packages/rtech-chatd/src/rtech_chatd/config.py:314` |
| `RTECH_CHATD_INTAKE_MEMBER_IDS` | `packages/rtech-chatd/src/rtech_chatd/config.py:315` |
| `RTECH_CHATD_INTAKE_NAME_TRIGGERS` | `packages/rtech-chatd/src/rtech_chatd/config.py:316` |
| `RTECH_CHATD_INTAKE_TRIGGER_PREFIXES` | `packages/rtech-chatd/src/rtech_chatd/config.py:317` |
| `RTECH_CHATD_DEBATE_PARTICIPANTS` | `packages/rtech-chatd/src/rtech_chatd/config.py:318` |
| `DISCORD_HOME_CHANNEL` | `packages/rtech-chatd/src/rtech_chatd/config.py:406` |
| `NATS_URL` (agentd + orchestrator) | `packages/rtech-agentd/src/rtech_agentd/__main__.py:71,331` |
| `RTECH_AGENT_ID` (via unit `%i`) | `deploy/kboard/rtech-agentd@.service:18`; `packages/rtech-agentd/src/rtech_agentd/__main__.py:86` |
| `RTECH_STACK_ID` (agentd waker) | `packages/rtech-agentd/src/rtech_agentd/__main__.py:63,385` |
| `RTECH_WAKE_ALLOWED_AGENTS` | `deploy/kboard/README.md:293`; consumed via `WakeConfig.from_env` (`__main__.py:386`) |
| `DEV_STACK_ENGINE` | `packages/rtech-agentd/src/rtech_agentd/__main__.py:20` (ENV_ENGINE_ALT); README `deploy/kboard/README.md:141` |
| `RTECH_STACK_ID` (orchestrator) | `packages/rtech-orchestrator/src/rtech_orchestrator/config.py:25,154-179` |
| `RTECH_TARGET_AGENT_ID` | `packages/rtech-orchestrator/src/rtech_orchestrator/config.py:22,112-151` |
| `RTECH_SINGLE_AGENT_ID` (back-compat) | `packages/rtech-orchestrator/src/rtech_orchestrator/config.py:23,130` |
| `RTECH_BRIDGED_AGENTS` | `packages/rtech-orchestrator/src/rtech_orchestrator/config.py:29,182-210` |
| `RTECH_AGENT_INSTANCES` (installer) | `deploy/kboard/install.sh:34` |
| `RTECH_STACK_ID` (tgd edge) | `packages/rtech-chatd/src/rtech_chatd/telegram_service.py:88,178` |
| `RTECH_CHATD_NATS_URL` (tgd edge) | `packages/rtech-chatd/src/rtech_chatd/telegram_service.py:89,186` |
| `RTECH_CHATD_TOKEN_REF` (tgd edge) | `packages/rtech-chatd/src/rtech_chatd/telegram_service.py:90,187` |
| `RTECH_TG_ALLOWED_GROUPS` | `packages/rtech-chatd/src/rtech_chatd/config.py:487,627` (parsed), `telegram_service.py:146` (required) |
| `RTECH_TG_IDENTITY_MAP` | `packages/rtech-chatd/src/rtech_chatd/config.py:726,729`; `telegram_service.py:188` |
| `RTECH_TG_POLL_TIMEOUT_S` | `packages/rtech-chatd/src/rtech_chatd/config.py:608` |
| `RTECH_CHATD_DEBATE_CHAT_LANES` | `packages/rtech-chatd/src/rtech_chatd/config.py:437` |
| `RTECH_CHATD_DEBATE_LANES` | `packages/rtech-chatd/src/rtech_chatd/config.py:436` |

**Naming trap.** `RTECH_CHATD_PARTICIPANTS` (`service.py:62`) is chatd's own
outbound task mint roster — a different, pre-existing feature. The trio debate
roster is `RTECH_CHATD_DEBATE_PARTICIPANTS` (`config.py:318`). The stock
`.env.example` documents both, with the intake block
(`RTECH_CHATD_INTAKE_*`, `RTECH_CHATD_TRIAGE_AGENT`) added alongside; this
kit's `deploy/kboard/env/` templates remain the paste-and-start source for the
watcher canary.
