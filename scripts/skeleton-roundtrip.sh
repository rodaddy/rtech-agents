#!/usr/bin/env bash
# Walking-skeleton round-trip proof (spec phases 2-3).
#
# Starts agentd with the _skeleton test persona, publishes a chat envelope, waits
# for the guarded chat_reply, asserts the agent self-identifies per SOUL.md, then
# injects a steer envelope mid-turn and asserts acknowledgement. Writes a
# transcript to .evidence/skeleton-<ts>.json and exits 0 only on full proof.
#
#   ENGINE=fake    (default) — scripted engine, no live model; proves the wiring.
#   ENGINE=claude            — real Claude Agent SDK engine (the genuine proof).
#
# Transport is a live FleetBus when NATS is reachable, else an in-process bus so
# the slice proves out off-network.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ENGINE="${ENGINE:-fake}"
echo "skeleton-roundtrip | ENGINE=${ENGINE} | repo=${REPO_ROOT}"

# Run the proof. The runner prints the evidence path on stdout (last line) and
# exits non-zero unless the round-trip AND the steer both leave evidence.
ENGINE="${ENGINE}" uv run python scripts/skeleton_runner.py
rc=$?

if [[ ${rc} -eq 0 ]]; then
  echo "skeleton-roundtrip | PASS (ENGINE=${ENGINE})"
else
  echo "skeleton-roundtrip | FAIL rc=${rc} (ENGINE=${ENGINE})" >&2
fi
exit ${rc}
