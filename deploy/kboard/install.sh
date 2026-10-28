#!/usr/bin/env sh
# Rootless, idempotent install of the rtech-agents stack onto one Linux deploy
# host as additive, non-disruptive systemd USER services. "k-board" is the
# author's host: an LXC container that also runs an unrelated app (k-board-api,
# caddy, postgres). Substitute your own host and user.
# Run ON the host AS THE TARGET USER (birk) -- NEVER as root, NEVER via sudo.
# Re-runnable: every step is create-if-missing or a safe rsync/sync. Existing
# k-board-api + caddy + postgres are NOT touched.
#
# Usage (on the host, as birk):
#   sh deploy/kboard/install.sh /path/to/rtech-agents-checkout \
#       [/path/to/fleet-bus] [/path/to/open-brain]
#
# The first argument is the path to a checked-out rtech-agents tree (the source
# to rsync into ~/rtech-agents). The frozen uv.lock pins two EDITABLE path deps
# as siblings of the install root -- ../fleet-bus and ../open-brain -- so
# `uv sync --frozen` needs those trees staged at ~/fleet-bus and ~/open-brain
# (runbook gap). Pass their source checkouts as the optional 2nd/3rd
# args; default source paths are the rtech-agents checkout's own siblings
# (../fleet-bus, ../open-brain). Staging is skipped for a dep whose source is
# absent AND whose destination already exists (idempotent re-runs).
# Secrets are NOT handled here: after install, the operator copies .env.example
# to ~/rtech-agents/.env and fills secret values from Vaultwarden (README.md).

set -eu

INSTALL_ROOT="$HOME/rtech-agents"
UNIT_DIR="$HOME/.config/systemd/user"
# The agentd ids to enable on the shared fleet bus. Each becomes the systemd
# instance name %i -> RTECH_AGENT_ID, and each MUST resolve to a source-owned
# home via the entrypoint's order (exact-id agents/<id>/agent.yaml, else the
# agentd--stripped slug agents/<slug>/agent.yaml; see the runbook's "Manifest
# resolution convention"). Default: the three trio agents. Override
# with RTECH_AGENT_INSTANCES (space-separated) for a different set/canary subset.
# Never a bare shared "agentd" -- that collided across two agentds.
AGENT_INSTANCES="${RTECH_AGENT_INSTANCES:-agentd-nimbus agentd-sprocket agentd-birk}"
SRC="${1:-}"
# Editable path-dep sources (frozen lock -> ../fleet-bus, ../open-brain). Default
# to the rtech-agents checkout's siblings; override via args 2 and 3.
FLEET_BUS_SRC="${2:-}"
OPEN_BRAIN_SRC="${3:-}"

log() {
	printf '[install] %s\n' "$1"
}

if [ -z "$SRC" ]; then
	log "ERROR: pass the path to a checked-out rtech-agents source tree"
	log "usage: sh deploy/kboard/install.sh /path/to/rtech-agents-checkout"
	exit 2
fi

if [ ! -d "$SRC" ]; then
	log "ERROR: source tree not found: $SRC"
	exit 2
fi

# Rootless mandate: refuse to run as root. This stack is birk-owned; root would
# recreate the /opt system-unit install this redo exists to retire.
if [ "$(id -u)" -eq 0 ]; then
	log "ERROR: do NOT run as root. This is a rootless, user-unit install."
	log "       run it AS the target user (birk): sh deploy/kboard/install.sh ..."
	exit 2
fi

# --- 1. sync source tree to the user-owned install root ----------------------
log "syncing source tree to $INSTALL_ROOT"
mkdir -p "$INSTALL_ROOT"
# Trailing slash on SRC copies contents. Exclude VCS, caches, and any local .env
# so an operator-filled secret file at the destination is never clobbered.
rsync -a --delete \
	--exclude '.git' \
	--exclude '.venv' \
	--exclude '__pycache__' \
	--exclude '.env' \
	--exclude 'state' \
	"$SRC"/ "$INSTALL_ROOT"/
mkdir -p "$INSTALL_ROOT/state"

# --- 1b. stage editable path deps the frozen lock needs ----------------------
# uv.lock pins fleet-nats/fleet-subscriber (../fleet-bus) and openbrain-memory
# (../open-brain) as EDITABLE path sources relative to the install root. Stage
# them next to ~/rtech-agents so `uv sync --frozen` resolves.
# Idempotent: rsync when a source is given/found; else require the dest present.
SRC_PARENT=$(dirname "$SRC")
: "${FLEET_BUS_SRC:=$SRC_PARENT/fleet-bus}"
: "${OPEN_BRAIN_SRC:=$SRC_PARENT/open-brain}"

stage_dep() {
	# $1 = human label, $2 = source path, $3 = destination path.
	dep_label="$1"
	dep_src="$2"
	dep_dst="$3"
	if [ -d "$dep_src" ]; then
		log "staging $dep_label: $dep_src -> $dep_dst"
		mkdir -p "$dep_dst"
		rsync -a --delete \
			--exclude '.git' \
			--exclude '.venv' \
			--exclude '__pycache__' \
			"$dep_src"/ "$dep_dst"/
	elif [ -d "$dep_dst" ]; then
		log "$dep_label source absent; keeping existing $dep_dst (idempotent)"
	else
		log "ERROR: $dep_label source not found ($dep_src) and no existing $dep_dst"
		log "       the frozen lock needs this editable path dep; pass its source path"
		exit 2
	fi
}

stage_dep "fleet-bus" "$FLEET_BUS_SRC" "$HOME/fleet-bus"
stage_dep "open-brain" "$OPEN_BRAIN_SRC" "$HOME/open-brain"

# --- 2. resolve dependencies (frozen) ----------------------------------------
log "running uv sync --frozen in $INSTALL_ROOT"
if ! command -v uv >/dev/null 2>&1; then
	log "ERROR: uv not found on PATH; install uv before running this script"
	exit 2
fi
( cd "$INSTALL_ROOT" && uv sync --frozen )

# --- 3. seed .env if absent (values filled by operator, never here) ----------
if [ -f "$INSTALL_ROOT/.env" ]; then
	log ".env already present; leaving operator secrets untouched"
else
	log "seeding .env from .env.example (fill secret values from Vaultwarden)"
	cp "$INSTALL_ROOT/deploy/kboard/.env.example" "$INSTALL_ROOT/.env"
	chmod 600 "$INSTALL_ROOT/.env"
fi

# --- 4. install systemd USER units -------------------------------------------
log "installing systemd user units to $UNIT_DIR"
mkdir -p "$UNIT_DIR"
# rtech-chatd.service (Discord edge) and rtech-tgd.service (Telegram edge) are
# COPIED but NOT enabled below -- both are operator opt-ins: without a bot ref,
# stack id, and (tgd) group allowlist they fail closed at startup, so
# auto-enabling would crash-loop a fresh install.
for unit in rtech-orchestrator.service rtech-agentd@.service rtech-webchat.service rtech-chatd.service rtech-tgd.service; do
	install -m 0644 "$INSTALL_ROOT/deploy/kboard/$unit" "$UNIT_DIR/$unit"
done
systemctl --user daemon-reload

# --- 5. linger check (report; NEVER sudo) ------------------------------------
# User units only survive logout / start at boot if lingering is enabled for the
# user. Enabling it (`loginctl enable-linger $USER`) needs admin privilege on
# most systems -- OUT of scope for this rootless installer. Detect and REPORT;
# do not escalate. If linger is off, the units still start now but will not
# persist across a full logout/reboot until an admin enables it.
LINGER=$(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null || echo "unknown")
if [ "$LINGER" = "yes" ]; then
	log "linger is enabled for $(id -un); user units will persist across reboot"
else
	log "WARNING: linger is '$LINGER' for $(id -un)."
	log "         user units start now but will NOT survive a full logout/reboot."
	log "         An admin must run once (NOT this rootless installer):"
	log "           loginctl enable-linger $(id -un)"
fi

# --- 6. enable + start (additive; does not touch existing services) ----------
log "enabling and starting rtech-agents user units"
systemctl --user enable --now rtech-orchestrator.service
# One agentd unit per instance. Instance name %i becomes RTECH_AGENT_ID -- unique
# per agent, never bare "agentd". Each id resolves to a slug-named home
# (agents/<slug>/agent.yaml, exact-id dir winning if present) at runtime.
# Idempotent: enable --now is a no-op when the instance is already active.
AGENTD_STATUS_UNITS=""
for AGENT_INSTANCE in $AGENT_INSTANCES; do
	log "enabling agentd instance: $AGENT_INSTANCE"
	systemctl --user enable --now "rtech-agentd@${AGENT_INSTANCE}.service"
	AGENTD_STATUS_UNITS="$AGENTD_STATUS_UNITS rtech-agentd@${AGENT_INSTANCE}"
done
systemctl --user enable --now rtech-webchat.service

log "install complete. Verify with: systemctl --user status rtech-orchestrator${AGENTD_STATUS_UNITS} rtech-webchat"
