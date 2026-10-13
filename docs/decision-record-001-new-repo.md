# Decision Record 001 — New repo, new product: `rodaddy/rtech-agents`

- **Status:** Accepted
- **Date:** 2026-07-07
- **Context refs:** rtech-hermes#458 (audit), governing spec
  `rtech-hermes/specs/rtech-agents-product-spec.html`

## Decision

Build the managed-agent stack as a **new, standalone product repo**,
`rodaddy/rtech-agents`, with `main` as a pass-through default branch and all work
on `feat/*` branches. The name was verified free on 2026-07-07.

## Rationale

Issue #458's audit found that Hermes's integration moat is mostly gone:

- **OB memory is portable** — HTTP/MCP API plus the packaged `openbrain-memory`
  client; zero server work required to reuse it.
- **Personas are markdown** — `SOUL.md`/`USER.md`/`KERNEL.md` plus a slot map and
  scan/cap behavior fully located in Hermes source; a pure port.
- **Coordination is being extracted** into fleet-bus/NATS deliberately, so the
  new product is just another NATS consumer.
- **The Hermes agent loop itself is the weakest link** — 4–5 primitive tool
  grinds where a managed Claude/Codex session does one.

What is genuinely Hermes-only is the gateway daemon plus the Discord adapter.
This is therefore a **runtime replacement, not a Hermes customization** — it
cannot live in `rtech-hermes`, whose charter is Hermes tooling.

## Why not extend Hermes

Rejected by the audit itself: the loop is the weak part, and everything portable
has already left the Hermes core. Bolting a managed-session engine onto the
Hermes loop would inherit the exact machinery the product exists to replace.

## Consequences

- `rtech-hermes` **keeps owning the incumbent fleet** (Bilby/Skippy/Nagatha on
  Hermes) until a separate, later migration decision. That migration happens
  only after the fresh test agent proves the new stack.
- This spec/plan lives in `rtech-hermes` because #458 lives there; the product
  code, standards, and CI live here in `rtech-agents`.
- Reuse-first: `fleet-nats` (transport), `openbrain-memory` (memory), Hermes
  source (persona + Discord behavioral contract), and monkeyproof/monkeyproof-a2a
  (managed-session patterns) are dependencies or reference sources, not rewrites.

## What stays in rtech-hermes

- The Hermes tooling/customization slice (install/update, plugins, patches,
  repair, runtime package) for the production trio.
- The governing spec and its verifier
  (`specs/rtech-agents-product-spec.{html,verify.sh}`).
- Ownership of the incumbent fleet until migration is separately decided.
