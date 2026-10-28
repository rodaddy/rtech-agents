# Deploy: rtech-agents on the k-board LXC (single host)

> "k-board" is the author's deploy host: an LXC container that already runs an
> unrelated app (`k-board-api`, `caddy`, `postgres`). `birk` is the user that
> owns the install there. This runbook is a worked example for that one host;
> substitute your own host, user and secret-store paths.

Operator runbook for standing up the bus-first stack (orchestrator + agentd +
web-chat edge) as **additive, non-disruptive**, **rootless** systemd **user**
services on the EXISTING k-board LXC. That box already runs `k-board-api` +
`caddy` + `postgres`; this deploy must not touch them.

**Rootless mandate (global run-as rule).** Everything here runs AS
`birk` — zero root, zero sudo. The tree is birk-owned at `~birk/rtech-agents`,
units are systemd **user** units under `~/.config/systemd/user/`, and
`install.sh` refuses to run as root. The one step that may need admin privilege
(`loginctl enable-linger`) is detected and REPORTED, never escalated.

These are repo artifacts. Nothing here mutates the host until an operator runs
`install.sh` ON the box AS birk.

## What gets installed

| Unit | Role | Entry |
|---|---|---|
| `rtech-orchestrator.service` | derived-done coordinator | `python -m rtech_orchestrator` |
| `rtech-agentd@<id>.service` | managed engine session (templated per agent id) | `python -m rtech_agentd` |
| `rtech-webchat.service` | typed task-frame edge (loopback/LAN) | `rtech-webchat` |
| `rtech-chatd.service` (opt-in) | Discord edge: TaskFrame ingress + attributed turn render-back | `rtech-chatd` |
| `rtech-tgd.service` (opt-in) | Telegram long-poll edge: TaskFrame ingress + multi-identity turn render-back | `rtech-tgd` |

- Install root: `~/rtech-agents` (source tree, synced by `install.sh`).
- Env file: `~/rtech-agents/.env` (from `.env.example`, secrets from Vaultwarden).
- Ownership: birk-owned; no service user is created, nothing runs as root.
- Unit dir: `~/.config/systemd/user/`; controlled via `systemctl --user`.
- All units: `uv run --frozen`, `Restart=on-failure`, `After=network-online.target`,
  `WantedBy=default.target`.

### Clean engine env

`birk`'s `~/.cc-env` exports stale 4-6-era `ANTHROPIC_*` pins that route the SDK
engine through a LiteLLM proxy — the root cause of the real-engine failure. Every
unit's `ExecStart` is wrapped in `env -u` to strip these before launch:

```
ANTHROPIC_BASE_URL ANTHROPIC_API_KEY ANTHROPIC_MODEL
ANTHROPIC_DEFAULT_OPUS_MODEL ANTHROPIC_DEFAULT_SONNET_MODEL
```

With them unset the Agent SDK spawns the box's authenticated `claude` CLI and
uses its **subscription auth and own model defaults**. Do NOT add `ANTHROPIC_*`
to `.env` — the units read no such value from it. Verify the scrub after start
with `systemctl --user show`/`environ` (see [Verify](#verify)).

## Pre-flight: port-conflict check (MANDATORY)

The edge binds `RTECH_WEBCHAT_HOST:RTECH_WEBCHAT_PORT` (default `127.0.0.1:3800`).
Confirm nothing on the box already listens there BEFORE installing:

```sh
ss -ltn | grep -E ':3800\b' || echo "port 3800 free"
# Also confirm the existing services' ports are untouched by us:
ss -ltn
```

If `3800` is taken, pick a free port and set `RTECH_WEBCHAT_PORT` in `.env`
before starting `rtech-webchat.service`. The edge MUST bind the LAN address
(`127.0.0.1`), never `0.0.0.0` and never a public interface.

> **Cutover note.** While the OLD root install still owns `3800`, bring the
> rootless stack up on `RTECH_WEBCHAT_PORT=3801` to prove it side-by-side, then
> switch back to `3800` after the old install is decommissioned.

## Install (rootless, as birk)

```sh
# On the k-board LXC, AS birk (NOT root, NOT sudo), with a checked-out
# rtech-agents tree at $SRC. The frozen lock pins two EDITABLE path deps
# (fleet-bus, open-brain) as siblings of the install root, so their source
# checkouts must be stageable. By default install.sh looks for them next to
# $SRC (../fleet-bus, ../open-brain); pass explicit paths as the 2nd/3rd args
# if they live elsewhere:
sh deploy/kboard/install.sh "$SRC" [/path/to/fleet-bus] [/path/to/open-brain]
```

`install.sh` is idempotent and rootless: it refuses to run as root, rsyncs the
tree to `~/rtech-agents` (excluding `.git`, `.venv`, caches, and any existing
`.env`), **stages the editable path deps to `~/fleet-bus` and `~/open-brain`**
(the frozen `uv.lock` resolves them relative to the install root — a
runbook gap fixed here), runs `uv sync --frozen`, seeds `.env` from `.env.example` only if
absent, installs the three **user** units, and `systemctl --user enable --now`s
them. The agentd unit is started as `rtech-agentd@agentd-birk` — the instance
name becomes `RTECH_AGENT_ID`, a UNIQUE id on the shared fleet bus (never a bare
shared `agentd`, which collided across two agentds).

### Linger (persistence across logout/reboot)

User units only survive logout and start at boot when **lingering** is enabled:

```sh
loginctl show-user birk -p Linger    # install.sh reports this for you
```

Enabling it — `loginctl enable-linger birk` — needs admin privilege and is
therefore OUT of scope for the rootless installer. If linger is off, `install.sh`
still starts the units now but WARNS that they will not persist. An admin (or a
NOPASSWD entry) must run `loginctl enable-linger birk` **once**. Nothing
in this deploy escalates on its own.

### Agent identity

- `RTECH_AGENT_ID=agentd-birk` — this agentd's unique bus id. Absent an explicit
  value the standalone runner defaults to `agentd-<hostname>`, never bare
  `agentd`.
- `RTECH_TARGET_AGENT_ID=agentd-birk` — the orchestrator's single-scope routing
  target. Must equal `RTECH_AGENT_ID` for the single-agent POC. If it (and the
  back-compat `RTECH_SINGLE_AGENT_ID`, and any registry) is unset, the
  orchestrator **fails loudly at startup** rather than routing to a shared literal.

### Fill secrets (after first sync, before real use)

```sh
# .env was seeded from .env.example. Fill the edge API key from Vaultwarden:
#   Vaultwarden ref: <your-secret-store-path>/RTECH_AGENTS_API_KEY
$EDITOR ~/rtech-agents/.env            # set RTECH_AGENTS_API_KEY, adjust host/port
systemctl --user restart rtech-webchat.service
```

Never commit a filled `.env`. Secret values live only on the host, sourced from
Vaultwarden at install time.

## Engine mode: real Claude (default) vs. scripted

`.env.example` ships with `DEV_STACK_ENGINE` **unset (commented)** — the **real
Claude engine** is the deployed default. It uses subscription auth, NOT an API
key:

1. Log in ONCE as birk so the `claude` CLI carries subscription auth:

   ```sh
   claude   # opens browser login; do this once on the box, as birk
   ```

   The Agent SDK spawns this authenticated CLI. The units scrub `ANTHROPIC_*`
   so no API key or stale proxy pin interferes.

2. Leave `DEV_STACK_ENGINE` unset (the shipped default) for the real engine.

For an offline bus-wiring smoke test WITHOUT a real model, set
`DEV_STACK_ENGINE=scripted` in `~/rtech-agents/.env` and restart agentd:

```sh
systemctl --user restart rtech-agentd@agentd-birk.service
```

If the `claude` CLI is missing or unauthenticated, agentd fails closed with an
actionable message.

## Verify

```sh
systemctl --user status rtech-orchestrator rtech-agentd@agentd-birk rtech-webchat
journalctl --user -u rtech-orchestrator -u rtech-agentd@agentd-birk -u rtech-webchat -f
# Edge reachable on the LAN address only:
curl -s http://127.0.0.1:3800/api/tasks      # (3801 during cutover)

# Prove the ANTHROPIC_* scrub took effect in the running agentd:
#   the main PID's environ must contain NO ANTHROPIC_ entries.
tr '\0' '\n' < /proc/$(systemctl --user show -p MainPID --value rtech-agentd@agentd-birk)/environ \
  | grep ANTHROPIC_ || echo "clean: no ANTHROPIC_* in agentd env"
```

## Rollback (rootless stack)

Fully reverses THIS rootless install; leaves `k-board-api` / `caddy` / `postgres`
untouched. All as birk, no sudo:

```sh
# 1. stop + disable our user units (leave existing services alone)
systemctl --user disable --now rtech-webchat.service
systemctl --user disable --now rtech-agentd@agentd-birk.service
systemctl --user disable --now rtech-orchestrator.service

# 2. remove the unit files and reload
rm -f ~/.config/systemd/user/rtech-orchestrator.service \
      ~/.config/systemd/user/rtech-agentd@.service \
      ~/.config/systemd/user/rtech-webchat.service
systemctl --user daemon-reload

# 3. remove the install tree (this removes .env -- back it up first if needed)
#    plus the staged editable path deps:
rm -rf ~/rtech-agents ~/fleet-bus ~/open-brain
```

## Decommission the OLD root install

The superseded #7 install was performed AS ROOT with **system** units in
`/etc/systemd/system` and a tree at `/opt/rtech-agents` owned by a `rtech`
system user. Because root installed it, retiring it needs **root** — the
rootless flow above CANNOT and MUST NOT remove it. These commands are for an admin
or a NOPASSWD path; the rootless worker does not run them:

```sh
# --- run as root (or via an explicit NOPASSWD entry) ---
# 1. stop + disable the old SYSTEM units
systemctl disable --now rtech-webchat.service
systemctl disable --now rtech-agentd@agentd-birk.service
systemctl disable --now rtech-orchestrator.service

# 2. remove the old system unit files and reload
rm -f /etc/systemd/system/rtech-orchestrator.service \
      /etc/systemd/system/rtech-agentd@.service \
      /etc/systemd/system/rtech-webchat.service
systemctl daemon-reload

# 3. remove the old /opt install tree + staged path deps
rm -rf /opt/rtech-agents /opt/fleet-bus /opt/open-brain

# 4. remove the old dedicated system service user
userdel -r rtech
```

After decommission, switch the rootless edge back to `RTECH_WEBCHAT_PORT=3800`
(freed by the old install) and `systemctl --user restart rtech-webchat.service`.

## chatd Discord edge (opt-in)

`rtech-chatd.service` is the product's **primary chat surface**: a **dumb edge**
that mints a typed `TaskFrame` from each gated inbound Discord message, publishes
it on the bus, and renders agent turns back into the originating thread —
**attributed** (each participant's turn renders as its own `**<agent_id>**:`
reply). It carries NO control signal, and it never authors "done": completion is
orchestrator-derived and arrives on the per-task result subject, where the edge
renders a single `✅ done` marker exactly once.

`install.sh` **copies** the unit but does NOT `enable --now` it — enabling is an
operator opt-in. Without a bot token, inbox channel, AND stack id it **fail-closes
at startup** (by design), so auto-enabling would crash-loop a fresh install.

### Install (operator opt-in)

The unit is already copied to `~/.config/systemd/user/rtech-chatd.service` by
`install.sh`. If you installed BEFORE this change, copy it once:

```sh
cp ~/rtech-agents/deploy/kboard/rtech-chatd.service ~/.config/systemd/user/
```

Fill the `RTECH_CHATD_*` block plus `RTECH_STACK_ID` in `~/rtech-agents/.env`
(the bot token VALUE comes from Vaultwarden — ref
`<your-secret-store-path>/RTECH_CHATD_BOT_TOKEN`; only the `*_ref` lives in `.env`,
never the literal). At minimum set `RTECH_STACK_ID`, `RTECH_CHATD_INBOX_CHANNEL`,
and `RTECH_CHATD_BOT_TOKEN`. Then enable it:

```sh
$EDITOR ~/rtech-agents/.env    # set RTECH_STACK_ID, RTECH_CHATD_INBOX_CHANNEL,
                               # RTECH_CHATD_BOT_TOKEN, RTECH_CHATD_PARTICIPANTS
systemctl --user daemon-reload
systemctl --user enable --now rtech-chatd.service
```

`RTECH_CHATD_PARTICIPANTS` is a comma-separated agent-id list whose COUNT derives
the task scope (1=single, 2=duo, 3=trio); empty = a single-scope task routed to
the orchestrator's default target. `RTECH_CHATD_ALLOW_BOTS` is `none` or
`mentions` — `all` is rejected fail-closed (security floor: treat Discord as
leaked).

### Verify

```sh
journalctl --user -u rtech-chatd -f
```

When required config is missing the service fail-closes with an actionable
message and does NOT connect:

- `RTECH_STACK_ID must be set ...` when the stack id is unset/blank,
- `inbox_channel is required ...` when `RTECH_CHATD_INBOX_CHANNEL` is unset,
- `RTECH_CHATD_ALLOW_BOTS=all is rejected ...` when the bot policy is `all`,
- `chatd: bot token not provisioned ...` when the token ref resolves empty.

### Rollback

```sh
systemctl --user disable --now rtech-chatd.service
rm -f ~/.config/systemd/user/rtech-chatd.service
systemctl --user daemon-reload
```

## tgd Telegram edge (opt-in)

`rtech-tgd.service` is the standalone **Telegram** chat surface: a **dumb edge**
that long-polls ONE allow-listed supergroup, mints a typed `TaskFrame` from each
gated inbound message, publishes it on the bus, and renders agent turns back —
**multi-identity** (each debate participant's turn posts under its OWN Telegram
bot; `attributed=False`, the posting bot IS the attribution). It composes ONLY
the Telegram surface (adapter + multi-identity sender + intake sink + edge) — no
Discord adapter, no orchestrator, no agentd. It carries NO control signal and
never authors "done": completion is orchestrator-derived on the per-task result
subject.

**Long polling = no ingress.** The edge calls `getUpdates` OUTBOUND to
`api.telegram.org`; there is **no inbound port, no webhook, and no public
domain**. Nothing to open in the firewall, nothing behind a reverse proxy.

`install.sh` **copies** the unit but does NOT `enable --now` it — enabling is an
operator opt-in. Without an env token ref, group allowlist, stack id, AND (for
debate) every participant's mapped, provisioned identity it
**fail-closes at startup** (by design), so auto-enabling would crash-loop a fresh
install.

**One explicit unit, not a template.** `rtech-tgd.service` is a single
non-templated unit. The per-agent bot identities are the sender's own env-ref
pool (`RTECH_TG_IDENTITY_MAP`) resolved inside ONE process — a `%i`-template
would collapse the multi-identity edge into N single-identity processes racing
the same long-poll offset.

### Install (operator opt-in)

The unit is already copied to `~/.config/systemd/user/rtech-tgd.service` by
`install.sh`. If you installed BEFORE this change, copy it once:

```sh
cp ~/rtech-agents/deploy/kboard/rtech-tgd.service ~/.config/systemd/user/
```

Fill the `RTECH_CHATD_*` / `RTECH_TG_*` block plus `RTECH_STACK_ID` in
`~/rtech-agents/.env` (paste-and-start template: `deploy/kboard/env/watcher-tgd.env.example`).
Bot token values are retrieved from Vaultwarden by the operator, but the service
accepts only `env:<ENV_VAR>` refs in `.env`, never literals or vaultwarden refs.
At minimum set `RTECH_STACK_ID`, `RTECH_CHATD_BOT_TOKEN` (the polling bot), and
`RTECH_TG_ALLOWED_GROUPS`; debate also requires each participant's provisioned
identity-map entry. In `listed` intake mode, the configured Telegram groups are
the intake scope. Then enable it:

```sh
$EDITOR ~/rtech-agents/.env    # set RTECH_STACK_ID, RTECH_CHATD_BOT_TOKEN,
                               # RTECH_TG_ALLOWED_GROUPS, RTECH_TG_IDENTITY_MAP,
                               # RTECH_CHATD_TRIAGE_AGENT, RTECH_CHATD_DEBATE_PARTICIPANTS
systemctl --user daemon-reload
systemctl --user enable --now rtech-tgd.service
```

### Verify

```sh
journalctl --user -u rtech-tgd -f
```

When required config is missing the service fail-closes with an actionable
message and does NOT connect:

- `RTECH_STACK_ID must be set ...` when the stack id is unset/blank,
- `RTECH_TG_ALLOWED_GROUPS is required ...` when the group allowlist is
  unset/empty (an empty allowlist would answer in every group),
- `token_ref must be an indirection ref of shape 'env:VALID_ENV_VAR' ...` when a
  literal, vaultwarden, or invalid env ref is configured,
- `chatd.tgd: configured bot token not provisioned (ref withheld)` when a
  configured env ref resolves empty; neither ref nor token value is logged.

### Schema / version diagnostics (atomic-ship)

The Telegram trio-debate path is the SAME bus contract as the Discord one and
carries the SAME atomic-ship requirement: it needs the orchestrator + agentd
built from the branch with the `TASK_INGRESS` fan-out consumer AND an
rtech-frames build that decodes `origin`/`debate` (`SCHEMA_VERSION 6`). A debate
frame published against an old orchestrator (no `TASK_INGRESS` fan-out) or an old
rtech-frames (`extra="forbid"` rejects the `debate` key) is dropped and the
debate never runs. Confirm the frames schema version before enabling the roster:

```sh
# Frames schema version this checkout builds/decodes (must be >= 6):
uv run python -c "import rtech_frames; print('SCHEMA_VERSION', rtech_frames.SCHEMA_VERSION)"
# The tgd console entrypoint resolves (no network; fails closed on bad env):
uv run --frozen rtech-tgd --help 2>/dev/null \
  || echo "rtech-tgd present (no --help; it reads env and long-polls)"
```

Upgrade rtech-frames consumers first, then the orchestrator and agentd, then
enable `RTECH_CHATD_DEBATE_PARTICIPANTS` on tgd.

### Rollback

```sh
systemctl --user disable --now rtech-tgd.service
rm -f ~/.config/systemd/user/rtech-tgd.service
systemctl --user daemon-reload
```

## Wake (triage) capability

A triage agentd (Nimbus) can **wake** an allow-listed teammate — mint a fresh
`TaskFrame` on the teammate's task subject, carrying the original
`ChannelOrigin`. Enable it by setting `RTECH_WAKE_ALLOWED_AGENTS` (a comma list of
`agentd-<name>` ids) plus `RTECH_STACK_ID` in `~/rtech-agents/.env`, then restart
that agentd.

**Loop bound.** Setting `RTECH_WAKE_ALLOWED_AGENTS` IS the waker opt-in — **only
the triage deploy sets it**. A woken agent is **not** itself a waker (it has no
allow-list), so a wake cannot chain into an unbounded loop. Leave the var unset on
every non-triage agentd.

**Reply path is DEAD-UNTIL-WIRED.** A woken agent's reply is **inert on `main`
today**: chatd only renders task_ids it minted, so the origin the wake frame
carries does not yet route the woken agent's answer back to the originating
channel. The origin is carried for when that reply path is wired; until then the
woken agent runs and publishes its turn, but its answer is not rendered back.

## Trio-intake debate: orchestrator must ship atomically

The trio-intake debate path (chatd fans an intake hit to a duo/trio `TaskFrame`
carrying `ChannelOrigin.debate`, and renders per-agent channel TLDRs plus full
turn text in a lazily-created thread) requires the **orchestrator build from this
branch or later** — the one with the `TASK_INGRESS` consumer that fans a debate
frame verbatim to each participant's task lane, and the rtech-frames build that
decodes `origin`/`debate` (SCHEMA_VERSION `6`). **Ship the orchestrator + agentd
atomically with (or before) this chatd build:** a debate frame published against
an old orchestrator that has no `TASK_INGRESS` fan-out — or an old rtech-frames
that rejects the `debate` key at its `extra="forbid"` decode — is dropped, and the
debate never runs. Upgrade rtech-frames consumers first, then the orchestrator and
agentd, then enable `RTECH_CHATD_DEBATE_PARTICIPANTS` on chatd.
