# rtech-agents

A runtime for long-running, steerable AI agents that coordinate over NATS.

Each agent is a managed Claude (or Codex) session supervised by its own daemon.
It takes work from typed frames on a NATS bus, talks to people in Discord,
Telegram or a small web chat, composes its identity from three Markdown files,
and keeps memory in Open Brain.

rtech-agents is the runtime. Its sibling repo,
[fleet-bus](https://github.com/rodaddy/fleet-bus), is the bus it rides on:
the NATS library, the subject tree and the deterministic orchestrator.
rtech-agents imports fleet-bus packages directly.

## Why it exists

One-shot agent calls (`claude -p`) cannot be steered mid-turn, and chat-driven
coordination loses track of who owns a task. This runtime keeps one persistent
session per agent, lets a person steer it while it works, and decides when a
task is done from the graph of typed frames on the bus rather than from a magic
token in the model's reply.

## Packages

| Package | What it does |
|---|---|
| `rtech-frames` | Typed wire frames: `TaskFrame`, `TurnFrame`, `SteerFrame`, `StatusFrame`. |
| `rtech-agentd` | Per-agent supervisor: one persistent engine session, NATS ingress and egress, steering, Open Brain memory. |
| `rtech-orchestrator` | Derives "done" from the frame graph and fires the result lane. |
| `rtech-chatd` | Chat edges behind a `ChannelAdapter` protocol: Discord and Telegram. |
| `rtech-webchat` | A small web-chat edge for local use. |
| `rtech-persona` | Loads `SOUL.md` / `USER.md` / `KERNEL.md`, scans and caps them, composes the system prompt. |
| `rtech-schedulerd` | Cron-scheduled task minter. |
| `rtech-hermes-bridge` | Adapter that fronts existing Hermes gateway agents on the same bus. |

## Agents are config

Only `agents/_skeleton` ships here: a test persona used by the round-trip
scripts. A real agent is a directory you add per deployment,
`agents/<slug>/` with an `agent.yaml` and its persona files. Credentials in
`agent.yaml` are references (`env:NAME`, `file:path`, or a secret-store ref),
never literal values; the loader rejects anything that looks like a raw secret.
See `agents/README.md`.

## Run it

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/), and sibling
checkouts laid out like this, because the workspace uses local path
dependencies:

```text
parent/
  rtech-agents/
  fleet-bus/                          # github.com/rodaddy/fleet-bus
  open-brain/python/openbrain-memory  # github.com/rodaddy/open-brain
```

```bash
git clone https://github.com/rodaddy/rtech-agents.git
git clone https://github.com/rodaddy/fleet-bus.git
git clone https://github.com/rodaddy/open-brain.git
cd rtech-agents
```

The root `pyproject.toml` resolves `fleet-core`, `fleet-nats` and
`fleet-subscriber` from `../fleet-bus/packages/*`, and `openbrain-memory` from
`../open-brain/python/openbrain-memory`, as editable path sources (see
`[tool.uv.sources]`). Then:

```bash
uv sync --all-packages
uv run ruff check .
uv run mypy packages/
uv run pytest -q
```

Local stack (orchestrator, one agent, web-chat edge) against a NATS server you
run yourself:

```bash
NATS_URL=nats://127.0.0.1:4222 DEV_STACK_ENGINE=scripted uv run python scripts/dev_stack.py
# then open http://127.0.0.1:3800
```

Drop `DEV_STACK_ENGINE=scripted` to use the real Claude engine through an
authenticated `claude` CLI on PATH.

## Tests

`uv run pytest -q` runs the unit suite with no network. Tests that need a live
NATS server (`test_nats_live.py`) skip when it is unreachable; the Discord and
Telegram canaries in `packages/rtech-chatd/scripts/` and
`scripts/e2e_evidence.py` are manual integration runs.

Three tests are skipped because the author's agent configs and the `secrets/`
templates are not part of this export; the other skips are live-service smokes
gated on an env var or a reachable server.

## What is production-proven and what is not

- Proven: three agents ran on this runtime in the author's own fleet in 2026,
  on one host as rootless systemd user services (`deploy/kboard/`).
- Not proven: any other deployment. The systemd units and env examples are a
  worked example for one host, not an installer for yours.
- Not included: the author's agent personas, the k-board service, or the
  Open Brain server.

## Naming notes

- Bilby, Nagatha and Skippy are the author's agents and appear as example ids.
- "Dash" is a neutral stand-in for an external dashboard that renders agent
  presence; set `RTECH_CHATD_DASH_BASE_URL` to link your own, or leave it empty.
- Hostnames under `example.internal` are placeholders.

## Docs

- `docs/decision-record-002-typed-frames.md`: the typed-frame contract.
- `docs/turn-completion-contract.md`, `docs/dispatch-confirmation-contract.md`:
  how done and dispatch are decided.
- `docs/debate-protocol.md`: multi-agent debate lanes.
- `docs/canary-rollback-runbook.md`: rollout and rollback.

## License

MIT. See `LICENSE`.
