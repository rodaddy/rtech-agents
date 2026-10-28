# Contributing to rtech-agents

Ground rules for anyone, human or coding agent, changing this repo.

rtech-agents is a NATS-native managed-agent runtime. Each agent is a managed,
long-running, inline-steerable Claude/Codex session supervised by a per-agent
daemon. It talks in Discord, Telegram or a local web chat, coordinates over
NATS, composes identity from `SOUL.md`/`USER.md`/`KERNEL.md`, and uses Open
Brain as its primary memory. See `README.md` for the overview and setup.

New-repo rationale: `docs/decision-record-001-new-repo.md`.

## Packages layout

Each package is a `src`-layout uv package under `packages/`:

```text
packages/
  rtech-frames/         typed wire frames
  rtech-persona/        SOUL/USER/KERNEL loader (scan + cap) + prompt composer
  rtech-agentd/         managed engine session + NATS ingress/egress
  rtech-orchestrator/   derives done from the frame graph
  rtech-chatd/          ChannelAdapter protocol + Discord and Telegram edges
  rtech-webchat/        local web-chat edge
  rtech-schedulerd/     cron-scheduled task minter
  rtech-hermes-bridge/  optional adapter for an external Hermes gateway
```

The repo root `pyproject.toml` is a uv workspace over `packages/*`. Dev tools
(ruff/mypy/pytest) are declared once at the workspace root and shared by every
package. Cross-package imports use absolute imports; sibling imports within a
package are relative.

## Toolchain and quality gates

- `uv` only. Never `pip`.
- Ruff for formatting and linting; mypy for type checking; pytest for tests.
- Python `>=3.13`.

Gates (must all pass before a change is merged):

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy packages/
uv run pytest -q
```

## Coding standards

Read `docs/coding-standards.md` before touching code. In short:
`from __future__ import annotations` in every module; full type hints
(`str | None`, never `Optional`); Loguru only (never stdlib logging or
`print`); guard clauses and dispatch maps over nested `if`/`elif`; frozen
dataclasses for config; `try/except/finally` around external calls; log at
entry/exit/failure/fallback/guard-trigger; 750-line hard file cap.

## Security floor

- Private network (Tailscale/WireGuard) only; nothing public except (later) a
  webhook behind a reverse proxy with HMAC verification.
- A secret store for every credential; `*_ref` indirection in config, never
  literal secrets; gitleaks on push.
- "Treat Discord as leaked": `allow_bots >= mentions`, free-response only on the
  designated inbox channel, no secrets/member-ids in narration.
- Agents do not self-modify (no daemon restarts/config edits by the agent
  itself).
- Open Brain writes pass the redaction guard; spooled != saved is surfaced,
  never silent.
- Never log secrets, tokens, auth JSON, raw session DBs, or private user
  content. Log sanitized summaries only.
- Fail closed on missing auth or guard config.

## Git hooks

Run `bash scripts/install-hooks.sh` once after clone to activate the checked-in
pre-commit hook (it blocks commits on `main` and sets the checkout-local
`core.hooksPath`).

## Commits

Stage files explicitly by name. Commit messages use conventional style:
`feat:`, `fix:`, `chore:`, `docs:`, `test:`.
