#!/usr/bin/env bash
set -euo pipefail

# Activate the checked-in git hooks for this checkout.
#
# Sets the checkout-local core.hooksPath to .githooks/. This is per-checkout
# (not global) and takes precedence over any global hooks configuration.

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

git config core.hooksPath .githooks
chmod +x .githooks/* 2>/dev/null || true

echo "Hooks activated: core.hooksPath -> .githooks"
echo "  pre-commit: protected-branch block, CLAUDE.md -> AGENTS.md sync"
