#!/usr/bin/env zsh
# grep-gate: anti-regression scan for doctrine rules 2 and 3, plus the
# "no new runtime dependency on rtech-hermes" invariant.
#
# Rule 2: agents never emit `[[done]]`; done is orchestrator-derived from the
#         frame graph, never a sentinel token in shipped code.
# Rule 3: signals are typed frame fields, never scraped prose trailers.
# packages/ must not grow a live `import rtech_hermes` / `from
#         rtech_hermes` — the fleet stack is off Hermes; only prose/docstring
#         mentions and the distinct `rtech_hermes_bridge` package are allowed.
#
# The gate fails (non-zero exit, matches printed) if any banned token appears in
# SHIPPED code under packages/. Keep each pattern in ONE place: the constants
# below are the single source of truth.
#
# See docs/grep-gate.md for rationale.

set -euo pipefail

# Single source of truth for the banned-token pattern (doctrine rules 2/3).
GATE_PATTERN='\[\[done\]\]|done_sentinel|TurnSignal|trailer.scrape'

# Single source of truth for the banned rtech-hermes import pattern.
# Anchored to a line-leading `import`/`from` so prose/docstring mentions do not
# trip it, and the module boundary (`.`/space/end after `rtech_hermes`) excludes
# the legitimate distinct `rtech_hermes_bridge` package.
HERMES_IMPORT_PATTERN='^[[:space:]]*(import|from)[[:space:]]+rtech_hermes([.[:space:]]|$)'

# Repo root = parent of this script's directory, so the gate runs from anywhere.
SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h}"

SCAN_DIR="${REPO_ROOT}/packages"

if [[ ! -d "${SCAN_DIR}" ]]; then
  print -u2 "grep-gate: no packages/ directory at ${SCAN_DIR}; nothing to scan."
  exit 0
fi

if ! command -v rg >/dev/null 2>&1; then
  print -u2 "grep-gate: required command not found: rg"
  exit 1
fi

# rg exits 0 when it finds matches, 1 when it finds none. Invert that: matches
# are a regression (fail), zero matches are shippable (pass).
if matches="$(rg -n "${GATE_PATTERN}" "${SCAN_DIR}")"; then
  print -u2 "grep-gate: FAIL — banned doctrine tokens found in shipped code (rules 2/3):"
  print -u2 -- "${matches}"
  exit 1
fi

if hermes="$(rg -n "${HERMES_IMPORT_PATTERN}" "${SCAN_DIR}")"; then
  print -u2 "grep-gate: FAIL — live rtech_hermes import found in packages/ (no runtime dep on rtech-hermes):"
  print -u2 -- "${hermes}"
  exit 1
fi

print "grep-gate: PASS — no [[done]] / sentinel / trailer-scrape tokens and no live rtech_hermes imports in packages/."
exit 0
