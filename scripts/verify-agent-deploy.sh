#!/usr/bin/env bash
# verify-agent-deploy: READ-ONLY post-deploy checks for one agentd instance (#84).
#
# Companion to docs/canary-rollback-runbook.md. Every check below only READS
# state (systemctl status/show, a source-file existence probe) — it NEVER starts,
# stops, restarts, or edits a unit, and it publishes nothing to the bus. Run it on
# the k-board host AS the target user after enabling an instance to confirm the
# staged migration landed. It exits non-zero if any check fails so it can gate a
# canary before the next agent is migrated.
#
# Usage (on the host, as the target user):
#   bash scripts/verify-agent-deploy.sh [agent-id]   # default: agentd-nagatha
#
# Manifest resolution MIRRORS the runtime entrypoint
# (rtech_agentd.__main__._resolve_manifest_path) EXACTLY so the operator gate and
# the daemon can never disagree. First hit wins, and the resolved path is logged:
#   1. <root>/agents/<agent-id>/agent.yaml — an exact-id directory (wins when
#      both an exact-id and a slug dir exist).
#   2. <root>/agents/<slug>/agent.yaml where <slug> strips a leading `agentd-`
#      prefix — the committed convention (dirs are slug-named `agents/skippy`
#      while the systemd instance id is `agentd-skippy`).
# A miss under BOTH is a FAIL (the daemon would fall back to env-only, which is
# not a verified per-agent deploy).

set -eu

AGENT_ID="${1:-agentd-nagatha}"
INSTALL_ROOT="${RTECH_AGENTS_ROOT_INSTALL:-$HOME/rtech-agents}"
UNIT="rtech-agentd@${AGENT_ID}.service"
rc=0

log() {
	printf '[verify] %s\n' "$1"
}
fail() {
	printf '[verify] FAIL: %s\n' "$1" >&2
	rc=1
}

# 1. Source-owned config + persona exist for this id (the #84 invariant: no
#    hardcoded agent — the id must map to a source-owned home + SOUL.md).
#    Manifest resolution mirrors rtech_agentd.__main__._resolve_manifest_path:
#    exact-id dir wins, then the `agentd-`-stripped slug dir.
AGENTS_ROOT="$INSTALL_ROOT/agents"
AGENT_HOME=""
EXACT_HOME="$AGENTS_ROOT/$AGENT_ID"
SLUG="${AGENT_ID#agentd-}"
SLUG_HOME="$AGENTS_ROOT/$SLUG"
if [ -f "$EXACT_HOME/agent.yaml" ]; then
	AGENT_HOME="$EXACT_HOME"
	log "manifest resolved by exact id=$AGENT_ID at $AGENT_HOME/agent.yaml"
elif [ "$SLUG" != "$AGENT_ID" ] && [ -f "$SLUG_HOME/agent.yaml" ]; then
	AGENT_HOME="$SLUG_HOME"
	log "manifest resolved by slug=$SLUG (from id=$AGENT_ID) at $AGENT_HOME/agent.yaml"
else
	fail "no agent.yaml for id=$AGENT_ID (tried exact $EXACT_HOME/agent.yaml and slug $SLUG_HOME/agent.yaml)"
fi

if [ -n "$AGENT_HOME" ]; then
	if [ -f "$AGENT_HOME/SOUL.md" ]; then
		log "persona present: $AGENT_HOME/SOUL.md"
	else
		fail "missing $AGENT_HOME/SOUL.md"
	fi
fi

# 2. systemd unit is enabled and active (read-only; is-active/is-enabled only).
if systemctl --user is-enabled "$UNIT" >/dev/null 2>&1; then
	log "unit enabled: $UNIT"
else
	fail "unit not enabled: $UNIT"
fi
if systemctl --user is-active "$UNIT" >/dev/null 2>&1; then
	log "unit active: $UNIT"
else
	fail "unit not active: $UNIT (see: systemctl --user status $UNIT)"
fi

# 3. The unit's recent journal shows a clean engine selection + config load and
#    no restart-circuit-open exit. Read-only journal tail.
log "recent journal (read-only; inspect for 'engine selected', 'loaded agent.yaml', no 'circuit open'):"
journalctl --user -u "$UNIT" -n 20 --no-pager 2>/dev/null || \
	log "journalctl unavailable in this context; inspect logs manually"

if [ "$rc" -eq 0 ]; then
	log "PASS: $UNIT verified (config + persona + enabled + active)."
else
	log "one or more checks FAILED for $UNIT; do NOT proceed to the next agent."
fi
exit "$rc"
