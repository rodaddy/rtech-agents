# Canary + rollback runbook — per-agent agentd deploy (#84)

Staged migration of the trio (Bilby, Skippy, Nagatha) onto source-owned
per-agent homes (slug-named `agents/<slug>/agent.yaml` + `SOUL.md`) and the
rootless `rtech-agentd@<id>` systemd user units. This runbook covers the
canary-first rollout, per-agent verification, rollback, and the steps that are
explicitly **not** automated.

## Manifest resolution convention (slug vs instance id)

Manifest **directories are slug-named** (`agents/skippy`, `agents/bilby`,
`agents/nagatha`) while the systemd **instance id carries an `agentd-` prefix**
(`RTECH_AGENT_ID=agentd-skippy`, from the unit's `%i`). The entrypoint resolves
the manifest in this order (first hit wins, and it logs which one resolved):

1. `<RTECH_AGENTS_ROOT>/<id>/agent.yaml` — an exact-id directory, if present.
2. `<RTECH_AGENTS_ROOT>/<slug>/agent.yaml` where `slug` strips a leading
   `agentd-` prefix — the committed convention.

An exact-id dir wins over a slug dir when both exist. A miss under **both**
falls back to env-only behavior (loudly logged) so a typo degrades visibly.

All commands run **on the k-board host (your deploy host) AS the target
user** — never root, never sudo. The install is additive and idempotent
(`deploy/kboard/install.sh`); re-running it is safe.

## Migration order — canary first

Nagatha is the **canary**: the first agent migrated, verified in isolation before
any other agent is enabled (fleet decision). Only after Nagatha passes
verification do Skippy and then Bilby follow.

`deploy/kboard/install.sh` enables `RTECH_AGENT_INSTANCES` (default
`agentd-nagatha agentd-skippy agentd-bilby`). To stage the canary alone, override
the set for the first run:

```sh
RTECH_AGENT_INSTANCES=agentd-nagatha sh deploy/kboard/install.sh /path/to/rtech-agents-checkout
```

Verify Nagatha (below). Once green, enable the rest:

```sh
RTECH_AGENT_INSTANCES="agentd-skippy agentd-bilby" sh deploy/kboard/install.sh /path/to/rtech-agents-checkout
```

Each id **must** map to a committed slug-named `agents/<slug>/agent.yaml` (see
"Manifest resolution convention" above); the entrypoint falls back to env-only
behavior (loudly logged) for an id with no home, so a typo degrades visibly
rather than silently.

## Restart bounding — inner circuit + outer StartLimit

Two nested loops bound restarts; **both** are required for a true bound:

- **Inner (in-process) circuit.** The supervisor tracks its restart budget and,
  when spent, `Supervisor.run` raises `CircuitOpenError`; the entrypoint maps
  that to a **non-zero process exit** (#84 slice A). This bounds crash-restarts
  *within a single process lifetime*.
- **Outer (systemd) StartLimit.** `Restart=on-failure` would relaunch that
  non-zero exit forever, so `rtech-agentd@.service` sets
  `StartLimitIntervalSec=300` + `StartLimitBurst=5`: at most 5 starts per 300s
  window, after which systemd holds the unit **failed** and stops relaunching.

So a wedged agent climbs: inner circuit trips → process exits non-zero →
systemd restarts (RestartSec=5) → after 5 such starts in 5 minutes the outer
StartLimit latches the unit failed. Clearing it is a deliberate operator action:

```sh
systemctl --user reset-failed rtech-agentd@agentd-nagatha
systemctl --user start rtech-agentd@agentd-nagatha
```

A unit stuck in `failed` with `start request repeated too quickly` in the
journal means the outer bound latched — treat it as a real incident (the agent
could not stay up across five restarts), not a transient blip.

## Verification (per agent)

Run the read-only checker after enabling an instance. It starts/stops nothing and
publishes nothing; it exits non-zero if any check fails, so it gates the canary:

```sh
bash scripts/verify-agent-deploy.sh agentd-nagatha
```

It confirms: (1) the manifest + `SOUL.md` resolve under the install root using
the SAME order as the runtime entrypoint (exact-id `agents/<id>/` dir first, then
the `agentd-`-stripped `agents/<slug>/` dir; see "Manifest resolution convention"
above), (2) the unit is `is-enabled` and `is-active`, and (3) tails the unit
journal for the clean-start markers. If the script is unavailable, run these
exact commands:

1. **systemd status** — unit is loaded, enabled, active (running):

   ```sh
   systemctl --user status rtech-agentd@agentd-nagatha
   ```

   Expect `Active: active (running)` and, in the log, `agentd.config | loaded
   agent.yaml`, `engine_select.build | requested kind=claude`, and `agentd.run |
   agentd started`. A `restart circuit open` line or a non-zero exit means the
   in-process restart budget was spent — investigate before proceeding.

2. **Heartbeat lane** — the agent publishes to its fleet heartbeat subject
   (`rtech_agentd.heartbeat.heartbeat_subject(env, agent_id)`,
   `<env>.agent.<id>.heartbeat`). Observe it on the fleet bus:

   ```sh
   nats --server nats://127.0.0.1:4222 sub 'dev.agent.agentd-nagatha.heartbeat'
   ```

   Expect a fresh `StatusFrame` every ~30s (the `HEARTBEAT_INTERVAL_S` default).

3. **LifecycleFrame observation** — the daemon emits a typed `LifecycleFrame` on
   its per-agent lifecycle lane (`rtech_orchestrator.subjects.agent_lifecycle`,
   `rtech.agent.<id>.lifecycle`) at start/shutdown/recovery:

   ```sh
   nats --server nats://127.0.0.1:4222 sub 'rtech.agent.agentd-nagatha.lifecycle'
   ```

   A restart of the unit MUST produce a matching lifecycle event — that is the
   #84 legibility guarantee.

4. **Scripted-engine smoke turn** — prove the wiring end to end WITHOUT a real
   model. Because a resolved manifest is now **authoritative for the engine kind**
   (#84 slice C — the config kind overrides a stray `DEV_STACK_ENGINE`/`E2E_ENGINE`
   and logs when it does), point `RTECH_AGENTS_ROOT` at an empty dir so the run
   takes the env-only fallback and honors `DEV_STACK_ENGINE=scripted`:

   ```sh
   RTECH_AGENT_ID=agentd-nagatha RTECH_AGENTS_ROOT=$(mktemp -d) \
     DEV_STACK_ENGINE=scripted uv run --frozen python -m rtech_agentd
   ```

   The log should show the loud env-only fallback (no manifest under the empty
   root), `engine selected kind=scripted`, and the hardcoded fallback system
   prompt. Publish a task on the ingress subject and confirm a
   `TurnFrame`/result comes back. (To smoke the *persona-composed* prompt with the
   real manifest, drop `RTECH_AGENTS_ROOT`/`DEV_STACK_ENGINE` and run against the
   real engine.)

## Rollback (per agent)

Rollback is per-instance and non-destructive to the other agents and to the
existing k-board-api / caddy / postgres services.

1. **Stop and disable the unit** (leaves the source tree + config in place):

   ```sh
   systemctl --user disable --now rtech-agentd@agentd-nagatha
   ```

2. **Re-enable Hermes intake** for that agent. agentd taking a task and Hermes
   answering the same surface would double-respond, so agentd is enabled only
   after Hermes intake for that agent is scoped down (below). Rolling back means
   reversing that: re-enable the agent's Hermes intake path so coverage is not
   lost while agentd is down. This is a **Hermes-side** action (separate repo),
   not automated here — see "Not automated".

3. Confirm the unit is gone and the agent is served by Hermes again:

   ```sh
   systemctl --user is-active rtech-agentd@agentd-nagatha   # -> inactive
   ```

## Not automated (do these by hand)

- **Hermes-side mention scope-down.** Narrowing Hermes' mention/intake scope so
  it stops answering for a migrated agent (and re-widening it on rollback) lives
  in the Hermes repo and is a deliberate manual step. This runbook does not touch
  Hermes. There is **no runtime dependency on rtech-hermes** in this repo
  (enforced by `scripts/grep-gate.sh`, #84).
- **Discord invites / bot presence.** Adding or removing an agent's Discord bot
  in the target server/channels is a console action. `agent.yaml` carries only a
  `discord.bot_token_ref` indirection (a `<provider>:<name>` ref, never a token);
  minting/rotating the actual token in Vaultwarden and inviting the bot are
  manual.
- **Vaultwarden secret population.** `install.sh` seeds `.env` from
  `.env.example` but never fills secret values; the operator copies them from
  Vaultwarden (README).
- **`loginctl enable-linger`.** Persisting user units across reboot needs admin
  privilege and is out of scope for the rootless installer (it reports linger
  state only).
