# Project: rtech-agents

NATS-native managed-agent stack — a from-scratch agent product that replaces the
Hermes runtime for the fleet. Each agent is a managed, long-running,
inline-steerable Claude/Codex session supervised by a per-agent daemon. It talks
in Discord (WhatsApp next), coordinates over NATS, composes identity from
`SOUL.md`/`USER.md`/`KERNEL.md`, and uses Open Brain as its primary memory.

New-repo rationale: `docs/decision-record-001-new-repo.md`.


## Default branch

The default branch is `main`.

**Never work in `main`.** It is pass-through only: merge, pull, leave. After any
merge or pull, immediately switch to a work branch. If no known branch exists for
what's next, branch main to `wip/<topic>` while figuring things out, then branch
from there or rename it (`git branch -m`) once the work has a name. All changes
merge via PR.

Branch names describe the work, not the agent:

```text
feat/walking-skeleton
feat/agentd-nats-ingress
fix/composer-slot-order
chore/scaffold
```

## Packages layout

Four-pillar product; each pillar is a `src`-layout uv package under `packages/`.

```text
packages/
  rtech-persona/       SOUL/USER/KERNEL loader (scan + cap) + prompt composer
    src/rtech_persona/
    tests/
  rtech-agentd/        managed engine session + NATS ingress/egress
    src/rtech_agentd/
    tests/
  rtech-chatd/         ChannelAdapter protocol + Discord v1
    src/rtech_chatd/
    tests/
```

The repo root `pyproject.toml` is a uv workspace over `packages/*`. Dev tools
(ruff/mypy/pytest) are declared once at the workspace root and shared by every
package. Cross-package imports use absolute imports; sibling imports within a
package are relative.

## Toolchain and quality gates

- `uv` only. Never `pip`.
- Ruff for formatting and linting; mypy for type checking; pytest for tests.
- Python `>=3.13`.

Gates (must all pass before a PR is mergeable):

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy packages/
uv run pytest -q
```

## Coding standards

Read `docs/coding-standards.md` at the start of every session before touching
code. In short: `from __future__ import annotations` in every module; full type
hints (`str | None`, never `Optional`); Loguru only (never stdlib logging or
`print`); guard clauses and dispatch maps over nested `if`/`elif`; frozen
dataclasses for config; `try/except/finally` around external calls; log at
entry/exit/failure/fallback/guard-trigger; 750-line hard file cap.

## Security floor (inherited from the spec, non-negotiable)

- Private network (Tailscale/WireGuard) only; nothing public except (later) the WhatsApp Cloud
  webhook behind a reverse proxy with HMAC verification.
- A secret store for every credential; `*_ref` indirection in config, never
  literal secrets; gitleaks on push.
- "Treat Discord as leaked": `allow_bots >= mentions`, free-response only on the
  designated inbox channel, no secrets/member-ids in narration.
- Agents do not self-modify (no daemon restarts/config edits by the agent
  itself).
- OB writes pass the redaction guard; spooled != saved is surfaced, never
  silent.
- Never log secrets, tokens, auth JSON, raw session DBs, or private user
  content. Log sanitized summaries only.
- Fail closed on missing auth or guard config.

## Root instruction files: AGENTS.md is canonical

`AGENTS.md` is the only root instruction file. **Edit `AGENTS.md`.** The
`.githooks/pre-commit` hook blocks commits while a `CLAUDE.md` exists. No
symlinks anywhere.

Run `bash scripts/install-hooks.sh` once after clone to activate the checked-in
hooks (sets the checkout-local `core.hooksPath`).

## PR rules

Stage files explicitly by name. Do not blindly `git add .`. Commit messages use
conventional style: `feat:`, `fix:`, `chore:`, `docs:`, `test:`.

## Corrections

- **Open Brain is agent-readable** -- Use live Open Brain reads as the intended
  agent memory path. Continue protecting credentials and never log raw private
  content.

