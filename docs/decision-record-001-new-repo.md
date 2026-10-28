# Decision Record 001 — New repo, new product: `rtech-agents`

- **Status:** Accepted
- **Date:** 2026-07-07
- **Context refs:** an earlier audit of the Hermes integration and the
  governing product spec, both kept in the author's private Hermes tooling repo

## Decision

Build the managed-agent stack as a **new, standalone product repo**,
`rtech-agents`, with `main` as a pass-through default branch and all work
on `feat/*` branches. The name was verified free on 2026-07-07.

## Rationale

The audit found that Hermes's integration moat is mostly gone:

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
cannot live in the Hermes tooling repo, whose charter is Hermes tooling.

## Why not extend Hermes

Rejected by the audit itself: the loop is the weak part, and everything portable
has already left the Hermes core. Bolting a managed-session engine onto the
Hermes loop would inherit the exact machinery the product exists to replace.

## Consequences

- The Hermes tooling repo **keeps owning the incumbent fleet** (Birk/Sprocket/Nimbus on
  Hermes) until a separate, later migration decision. That migration happens
  only after the fresh test agent proves the new stack.
- The spec/plan lives in the Hermes tooling repo beside the audit; the product
  code, standards, and CI live here in `rtech-agents`.
- Reuse-first: `fleet-nats` (transport), `openbrain-memory` (memory), Hermes
  source (persona + Discord behavioral contract), and monkeyproof/monkeyproof-a2a
  (managed-session patterns) are dependencies or reference sources, not rewrites.

## What stays in the Hermes tooling repo

- The Hermes tooling/customization slice (install/update, plugins, patches,
  repair, runtime package) for the production trio.
- The governing spec and its verifier
  (`specs/rtech-agents-product-spec.{html,verify.sh}`).
- Ownership of the incumbent fleet until migration is separately decided.
