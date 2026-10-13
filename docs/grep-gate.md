# grep-gate — doctrine anti-regression scan

`scripts/grep-gate.sh` is a fast, dependency-light gate that fails the build if
shipped code under `packages/` reintroduces the exact regressions doctrine rules
2 and 3 exist to prevent (the controller doctrine, kept outside this export).

## What it enforces

**Rule 2 — done is orchestrator-derived, never a sentinel.** Agents must not emit
`[[done]]`. Done state is computed by the orchestrator from the frame graph, not
carried as a token in code or output. Banned: `[[done]]`, `done_sentinel`.

**Rule 3 — signals are typed frame fields, never scraped prose.** Control signals
live on `TurnFrame` / `StatusFrame` fields, never parsed out of a prose trailer.
Banned: `TurnSignal` (a signal masquerading as a scraped token), `trailer.scrape`
(prose-trailer parsing).

**#84 — no runtime dependency on rtech-hermes.** The fleet stack is off Hermes;
`packages/` must not grow a live `import rtech_hermes` / `from rtech_hermes …`.
The scan is anchored to a line-leading `import`/`from`, so prose and docstring
mentions (e.g. "ported from `rtech_hermes_runtime`") do not trip it, and the
module boundary after `rtech_hermes` (a `.`, whitespace, or end of line) excludes
the legitimate, distinct `rtech_hermes_bridge` package.

If any of these tokens appear under `packages/`, the gate prints the offending
`file:line:match` and exits non-zero. Zero matches → exit 0 → shippable on this
axis.

## Single source of truth

Each banned pattern lives in exactly one place in `scripts/grep-gate.sh`: the
`GATE_PATTERN` constant (rules 2/3) and the `HERMES_IMPORT_PATTERN` constant
(#84). CI and local runs both invoke that script; there is no second copy of a
pattern to drift out of sync.

## Running it

```bash
./scripts/grep-gate.sh
```

zsh, uses `rg`, and resolves the repo root from its own location, so it runs from
any working directory.

## CI wiring

The `grep-gate` job in `.github/workflows/ci.yml` installs ripgrep and runs the
script on every push and pull request, in parallel with the existing `gates` job
(ruff / mypy / pytest). A regression fails the check before merge.
