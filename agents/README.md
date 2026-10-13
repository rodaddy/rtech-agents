# agents/ — source-owned per-agent homes (#84)

Each agent gets a **slug-named** directory here holding its deployment manifest
and persona:

```
agents/<slug>/
├── agent.yaml   # rtech_agentd.config.AgentConfig; creds are <provider>:<name> refs only
└── SOUL.md      # persona home (SOUL/USER/KERNEL), source-owned
```

Only `agents/_skeleton` ships in this export. Real agents are deployment config:
add your own `agents/<slug>/` (agent.yaml plus SOUL/USER/KERNEL) per host.

## Directory naming: slug, not instance id

Directories are **slug-named** (`skippy`) but the systemd instance id — and thus
`RTECH_AGENT_ID` (from the unit's `%i`) — carries an `agentd-` prefix
(`agentd-skippy`). The entrypoint (`rtech_agentd.__main__._resolve_manifest_path`)
resolves the manifest under `RTECH_AGENTS_ROOT` (default `./agents`) in this
order, logging which one resolved:

1. `<root>/<RTECH_AGENT_ID>/agent.yaml` — an exact-id dir, if present (wins).
2. `<root>/<slug>/agent.yaml` where `slug` strips a leading `agentd-` prefix —
   the committed convention.

A miss under **both** falls back to env-only behavior (loudly logged), so a typo
degrades visibly rather than silently. New agents follow the slug convention:
create `agents/<slug>/` and add `agentd-<slug>` to `RTECH_AGENT_INSTANCES`.

## No literal secrets

Every credential field in `agent.yaml` is a `<provider>:<name>` **ref**
(`env:`, `vaultwarden:`, `file:`) — never a literal token. `load_agent_config`
rejects anything that looks like a raw secret. See
`docs/canary-rollback-runbook.md` for rollout, restart bounding, and rollback.
