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
| `rtech-hermes-bridge` | Optional adapter that fronts agents running on an external Hermes gateway attached to the same NATS bus through fleet-bus's `fleet-subscriber` adapter. The Hermes gateway is not part of this repo; skip this package if you do not run one. |

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
mkdir parent && cd parent
git clone https://github.com/rodaddy/rtech-agents.git
git clone https://github.com/rodaddy/fleet-bus.git
git clone https://github.com/rodaddy/open-brain.git
cd rtech-agents
```

The three clones must sit side by side under the same parent directory; the
directory names must stay `rtech-agents`, `fleet-bus` and `open-brain`.

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

`uv run pytest -q` runs the unit suite with no network. On a fresh checkout
expect 7 skips:

- 3 because this export ships only `agents/_skeleton` and no `secrets/` tree
  (two tests in `rtech-agentd/tests/test_agent_config_resolve.py`, one in
  `rtech-chatd/tests/test_user_session_config.py`);
- 4 live smokes: `test_nats_live.py` (skips when no NATS server answers at the
  configured URL), `test_codex_live.py` (`RTECH_CODEX_LIVE=1`),
  `test_memory.py` (`OB_LIVE`), and `rtech-chatd/tests/e2e/test_telegram_live.py`
  (`RTECH_TG_E2E_LIVE`, see `docs/telegram-live-e2e.md`).

The Discord canary (`packages/rtech-chatd/scripts/discord_canary.py`) and
`scripts/e2e_evidence.py` are manual integration runs.

## What is production-proven and what is not

- Proven: three agents ran on this runtime in the author's own fleet in 2026,
  on one host as rootless systemd user services (`deploy/kboard/`).
- Not proven: any other deployment. The systemd units and env examples are a
  worked example for one host, not an installer for yours.
- Not included: the author's agent personas, the unrelated app that shares
  the author's deploy host (`deploy/kboard/` is named after it), the Open Brain
  server, or a Hermes gateway.

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
