# How rtech-agents was built

Sources for this page: the private development repository's full history
(205 commits, 138 of them non-merge, 197 of them in July 2026), its commit
messages and PR merge bodies, the decision records and contracts in `docs/`,
the plan of record written on 2026-07-10, and the test suite. This public
repository holds a 2026-10-13 export, so its own `git log` starts at that export.

## The problem and the first version (2026-07-07)

The author ran a small group of AI agents on a third-party agent gateway, with
the agents coordinating in Discord. An audit of that setup concluded that the
gateway's integration advantages were mostly gone (memory, personas and
coordination had already moved out of it) and that the hard problems all came
from coordinating through a chat surface: mention gates, bot-to-bot rules,
thread mechanics, missed hand-offs.

On 2026-07-07 the repository was scaffolded as a uv workspace with three
packages: a persona loader and prompt composer (`rtech-persona`), the core of
a Discord edge (`rtech-chatd`), and the per-agent daemon with a managed engine
session and NATS ingress (`rtech-agentd`), proven by a walking-skeleton
round-trip script.

## The design correction (2026-07-10 and 11)

The first spec, written on 2026-07-07 and 08, still let agents signal
completion by writing a `[[done]]` trailer in their reply, and it reused an
untyped message envelope from the bus library. The plan of record written on
2026-07-10 and merged on 2026-07-11 changed three things, and the rest of the
project hangs off them:

1. NATS is the only source of truth. Chat connectors are dumb edges with no
   coordination logic.
2. A new typed-frame contract (`rtech-frames`, Pydantic) is the keystone:
   `TaskFrame`, `TurnFrame`, `SteerFrame`, `StatusFrame`
   ([decision record 002](decision-record-002-typed-frames.md)).
3. `[[done]]` is deleted. The orchestrator derives completion from the frame
   graph. A model cannot forget to say "done", and nothing is scraped from
   prose.

The reasoning recorded in the plan: rules that live only as prose in a prompt,
with no enforcement on the wire, were the class of failure the setup had just
spent weeks removing.

## Timeline

| Date | Phase |
|---|---|
| 2026-07-07 | Workspace scaffold, persona loader, Discord edge core, agentd core and round-trip skeleton |
| 2026-07-11 | Plan of record; typed frames; done derived from the frame graph; agentd re-seated on frames; minimal orchestrator; steering into a live turn; web-chat edge; Open Brain memory lane; end-to-end evidence on a live NATS bus; one-command dev stack; first deployment |
| 2026-07-12 | Adopt the shared fleet-core base; bridge to agents on the external gateway; done scoped per participant |
| 2026-07-15 | Trio debate (agents see each other's turns); lifecycle frames; bounded restarts and idle-only context rotation; Codex app-server engine; deterministic intake listener with a triage agent and a wake tool |
| 2026-07-17 to 19 | Telegram edge with one bot identity per agent, inline steer buttons and a live end-to-end harness; responses gated on typed evidence; log-redaction hardening |
| 2026-07-19 to 20 | Web-chat presence; supervised Start for offline agents; cron-scheduled tasks |
| 2026-10-01 to 23 | Maintenance: CI runs affected tests on PRs and the full suite on `main`; shared test gate |
| 2026-10-13 | Public export with private agent names and hosts removed |

## Key design decisions

- **Typed frames, validated at both ends.** Every frame is revalidated at the
  encode boundary as well as on decode, so a malformed frame fails in the
  sender rather than in a consumer.
- **Done is derived, per participant.** A duo or trio task is done only when
  each named participant has a terminal turn and no later status from that
  participant reopens it. Turns from agents that were not dispatched never
  count. 42 cases in `packages/rtech-frames/tests/test_done.py` pin this.
- **Steering is a frame, not a command in text.** `SteerFrame` carries an
  explicit `interrupt` field; nothing is parsed out of the message body.
- **One engine protocol.** Claude Agent SDK and Codex app-server sessions sit
  behind one `EngineSession` protocol, selected per agent.
- **Fail closed on configuration.** For example, a triage agent id that does
  not match the `agentd-<name>` subject convention is a startup error, because
  a bare name publishes to a subject nobody subscribes to.
- **Use a library where one exists.** The scheduler first shipped a
  hand-written cron scanner. Repeated review findings on it led to replacing
  it with croniter behind the existing typed wrapper.

## How AI agents built, reviewed and tested it

The code was written by AI coding agents (Claude Code with Claude Opus),
working from the plan of record in numbered phases and
slices, each on its own branch and pull request. The commit history carries
the agents' co-author lines.

Larger changes went through adversarial review before merge, at times by two model families.
One commit records that two adversarial reviews, one by Claude Opus and one by an OpenAI Codex model, returned
HOLD on a slice of the intake work and lists the four fixes. Pull requests
record numbered review rounds: participant-scoped done derivation went
through five rounds, and the scheduler PR closed its findings across several.
Enforcement was put in code where possible: a grep-gate CI job scans for
regressions of the project's rules (for example, prose completion tokens),
and the pre-commit hook blocks commits on `main`.

Quality gates on every change: `ruff check`, `ruff format --check`,
`mypy --strict` and `pytest`. The export has 111 test modules; the suite runs
1,786 tests with 7 skips (live smokes and private-config tests) in about 12
seconds with no network. Live behaviour is checked separately: an
end-to-end evidence script against a real NATS bus, a Discord canary and an
authorization-gated Telegram live-account harness.

## What did not work

- **Prose completion signals.** The first spec's `[[done]]` trailer and
  stance scraped from text were dropped before they shipped widely, in favour
  of typed frames and derived completion.
- **The first gateway bridge.** The bridge to agents on the external gateway
  first authored terminal turns itself; it was reworked into a content-only
  edge so the orchestrator alone owns completion.
- **The hand-written cron scanner.** Replaced by croniter after repeated
  review findings on DST and day-of-month/day-of-week edge cases.
- **Error logs that could leak.** Review found that several failure paths
  logged raw exception text that could contain credentials; a series of
  commits on 2026-07-19 and 20 redacted bus, callback, intake and connect
  errors.
- **Sibling path dependencies in CI.** The workspace depends on fleet-bus and
  open-brain checkouts, which needed extra CI checkout steps and still make a
  fresh clone a three-repository setup.

## Current state

Experimental. Three agents ran on this runtime on one host as rootless
systemd user services. Active development stopped after July 2026; later
commits are CI and maintenance. The round-trip skeleton script
(`scripts/skeleton-roundtrip.sh`) currently reports FAIL on its reply check
and needs updating to the typed reply payload.
