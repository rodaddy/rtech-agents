"""Select and build the managed engine session from the environment.

The engine an agentd owns is decided ONCE here so every caller (the local dev
stack, the live-fleet evidence script) shares one selection policy instead of
each re-deriving it. Two engines exist:

* **claude** (default) -- the real Claude Agent SDK session. The SDK spawns the
  ``claude`` CLI as a subprocess, and that CLI carries the operator's
  *subscription* auth. So the selection deliberately does NOT require
  ``ANTHROPIC_API_KEY``: an authenticated CLI on PATH is sufficient. Selection
  fails with a clear, actionable message ONLY when the CLI binary is genuinely
  absent (there is nothing for the SDK to spawn). ``ANTHROPIC_BASE_URL`` /
  ``ANTHROPIC_API_KEY`` remain honored when present (LiteLLM / key-based auth),
  but neither is mandatory.
* **codex** -- an OpenAI ``codex app-server`` session (JSON-RPC over stdio, one
  child per session). Selected via ``E2E_ENGINE=codex`` /
  ``DEV_STACK_ENGINE=codex`` or an agent whose ``agent.yaml`` declares
  ``engine.kind: codex``. Auth rides in via the operator's authenticated
  ``codex`` CLI (subscription), so like the claude path it requires only an
  installed CLI, not an API key. Fails closed with an actionable message when the
  ``codex`` binary is genuinely absent.
* **scripted** -- a deterministic in-process engine with no external I/O, the
  escape hatch for wiring smoke tests and offline runs (``E2E_ENGINE=scripted``
  or ``DEV_STACK_ENGINE=scripted``).

This module resolves the CLI exactly the way the SDK's own transport does
(``shutil.which`` plus the SDK's fallback install locations) so a "claude engine
selected" result means the SDK will find the same binary at connect time.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from loguru import logger

from .config import EngineConfig
from .engine import EngineSession

ENV_ENGINE = "E2E_ENGINE"
ENV_ENGINE_ALT = "DEV_STACK_ENGINE"
ENV_MODEL = "E2E_MODEL"
ENV_BASE = "ANTHROPIC_BASE_URL"
ENV_KEY = "ANTHROPIC_API_KEY"
ENV_CODEX_BASE = "OPENAI_BASE_URL"
ENV_CODEX_KEY = "OPENAI_API_KEY"

DEFAULT_ENGINE = "claude"
DEFAULT_MODEL = "claude-sonnet-5"
# ``gpt-5-codex`` is a RETIRED model id — the app-server 400s on it on
# ChatGPT-account turns ("model not supported when using Codex with a ChatGPT
# account", issue #132). The fallback must itself be a supported tier. The live
# GPT-5.6 family is Luna/Sol/Terra (see the operator's model-routing notes); Sol is the
# routine-implementation default and matches the model the interactive codex CLI
# resolves for ChatGPT accounts (~/.codex/config.toml model = "gpt-5.6-sol").
DEFAULT_CODEX_MODEL = "gpt-5.6-sol"

_CLI_FALLBACK_LOCATIONS: tuple[Path, ...] = (
    Path.home() / ".npm-global/bin/claude",
    Path("/usr/local/bin/claude"),
    Path.home() / ".local/bin/claude",
    Path.home() / "node_modules/.bin/claude",
    Path.home() / ".yarn/bin/claude",
    Path.home() / ".claude/local/claude",
)

_CLI_MISSING_MSG = (
    "claude engine selected but the `claude` CLI was not found on PATH. The "
    "Agent SDK spawns that CLI (it carries your subscription auth), so it must be "
    "installed and authenticated. Install it with "
    "`npm install -g @anthropic-ai/claude-code`, run `claude` once to log in, or "
    "set E2E_ENGINE=scripted to run the stack without a real engine."
)

_CODEX_CLI_MISSING_MSG = (
    "codex engine selected but the `codex` CLI was not found on PATH. The engine "
    "spawns `codex app-server` (it carries your subscription auth), so it must be "
    "installed and authenticated. Install the OpenAI Codex CLI, run `codex login` "
    "once, or set E2E_ENGINE=scripted to run the stack without a real engine."
)

_CODEX_CLI_FALLBACK_LOCATIONS: tuple[Path, ...] = (
    Path.home() / ".npm-global/bin/codex",
    Path("/usr/local/bin/codex"),
    Path.home() / ".local/bin/codex",
    Path.home() / "node_modules/.bin/codex",
)


class EngineSelectionError(Exception):
    """Raised when the requested engine cannot be built (fail closed)."""


def find_claude_cli() -> str | None:
    """Return the resolved ``claude`` CLI path, or ``None`` if not installed.

    Mirrors the SDK transport's resolution order: ``shutil.which`` first, then
    the SDK's fixed fallback install locations. A returned path is a strong
    signal the SDK will find the same binary when it spawns the subprocess.
    """
    found = shutil.which("claude")
    if found:
        return found
    for candidate in _CLI_FALLBACK_LOCATIONS:
        try:
            if candidate.is_file():
                return str(candidate)
        except OSError as exc:  # pragma: no cover - stat failure is environmental
            logger.debug("engine_select.cli | stat failed | path={} reason={}", candidate, exc)
    return None


def find_codex_cli() -> str | None:
    """Return the resolved ``codex`` CLI path, or ``None`` if not installed.

    Mirrors :func:`find_claude_cli`: ``shutil.which`` first, then a fixed set of
    common install locations. A returned path is a strong signal the engine can
    spawn ``codex app-server`` at connect time.
    """
    found = shutil.which("codex")
    if found:
        return found
    for candidate in _CODEX_CLI_FALLBACK_LOCATIONS:
        try:
            if candidate.is_file():
                return str(candidate)
        except OSError as exc:  # pragma: no cover - stat failure is environmental
            logger.debug("engine_select.codex | stat failed | path={} reason={}", candidate, exc)
    return None


def _selected_kind(env: dict[str, str]) -> str:
    """Return the requested engine kind, honoring both env-var spellings."""
    raw = env.get(ENV_ENGINE) or env.get(ENV_ENGINE_ALT) or DEFAULT_ENGINE
    return raw.strip().lower()


def build_engine(
    env: dict[str, str] | None = None,
) -> tuple[EngineSession, str]:
    """Build the engine session selected by the environment.

    Args:
        env: Environment mapping to read (defaults to ``os.environ``). Injected
            in tests so the CLI-auth path can be exercised without a real CLI.

    Returns:
        A ``(session, kind)`` pair: the built ``EngineSession`` and the resolved
        engine kind (``"claude"``, ``"codex"``, or ``"scripted"``).

    Raises:
        EngineSelectionError: If the kind is unknown, or the claude/codex engine
            was requested but its CLI is genuinely absent.
    """
    resolved = dict(os.environ if env is None else env)
    kind = _selected_kind(resolved)
    logger.info("engine_select.build | requested kind={}", kind)

    if kind == "scripted":
        from .scripted_engine import ScriptedEngine

        return ScriptedEngine(), "scripted"

    if kind == "claude":
        return _build_claude_engine(resolved), "claude"

    if kind == "codex":
        return _build_codex_engine(resolved), "codex"

    logger.error("engine_select.build | unknown engine kind={}", kind)
    raise EngineSelectionError(
        f"unknown engine kind {kind!r}; expected 'claude', 'codex', or 'scripted'"
    )


def _build_claude_engine(env: dict[str, str]) -> EngineSession:
    """Build the real Claude engine, failing closed only on a missing CLI.

    Subscription auth rides in via the ``claude`` CLI subprocess, so a key is NOT
    required. When ``ANTHROPIC_BASE_URL`` is set it is honored (LiteLLM-style
    routing) and passed through ``EngineConfig.litellm_base``; no secret value is
    ever logged here.
    """
    cli_path = find_claude_cli()
    if cli_path is None:
        logger.error("engine_select.claude | {}", _CLI_MISSING_MSG)
        raise EngineSelectionError(_CLI_MISSING_MSG)
    logger.info("engine_select.claude | cli resolved (auth via subscription CLI)")

    from .claude_engine import ClaudeEngine

    cfg = EngineConfig(
        kind="claude",
        model=env.get(ENV_MODEL, DEFAULT_MODEL),
        litellm_base=env.get(ENV_BASE) or None,
        backup_model=None,
    )
    extra_env: dict[str, str] = {}
    base = env.get(ENV_BASE)
    key = env.get(ENV_KEY)
    if base:
        extra_env[ENV_BASE] = base
    if key:
        extra_env[ENV_KEY] = key
    return ClaudeEngine(cfg, env=extra_env or None)


def _resolve_codex_model(env: dict[str, str]) -> str:
    """Resolve (and sanitize) the codex model from the environment (issue #132).

    Two untrusted-input hazards are closed HERE, at the config boundary, before
    the value can ever reach a log line at :meth:`CodexEngine.start`:

    * A blank / whitespace ``E2E_MODEL`` must NOT reach the engine — an empty
      configured model silently disables model attestation (the start/resume
      check no-ops when no model is requested). It normalizes to the supported
      default.
    * A NON-blank but malformed ``E2E_MODEL`` (an oversized blob, embedded
      newline/control bytes — a log-injection shape from an untrusted env) must
      never be interpolated raw into a log/exception. It is rejected and
      normalized to the supported default; the warning names only the env var,
      never the raw value.

    The check reuses the same safe-model-id shape the engine attests against, so
    the config boundary and the runtime attestation agree on what a model id is.
    """
    from .codex_engine import is_safe_model_id

    raw = env.get(ENV_MODEL)
    requested_model = raw or ""
    if not requested_model.strip():
        if ENV_MODEL in env:
            logger.warning(
                "engine_select.codex | blank {} override; using supported default {}",
                ENV_MODEL,
                DEFAULT_CODEX_MODEL,
            )
        return DEFAULT_CODEX_MODEL
    if not is_safe_model_id(requested_model):
        # Untrusted, malformed value: fail closed to the default. Log only the env
        # var NAME and the safe default — never the raw (possibly control-bearing
        # or oversized) value.
        logger.warning(
            "engine_select.codex | {} is not a safe model id; using supported default {}",
            ENV_MODEL,
            DEFAULT_CODEX_MODEL,
        )
        return DEFAULT_CODEX_MODEL
    return requested_model


def _build_codex_engine(env: dict[str, str]) -> EngineSession:
    """Build the real Codex engine, failing closed only on a missing CLI.

    Subscription auth rides in via the authenticated ``codex`` CLI (which the
    engine spawns as ``codex app-server``), so an API key is NOT required — the
    selection mirrors the claude path. When ``OPENAI_BASE_URL`` is set it is
    honored (LiteLLM-style routing) and passed through ``EngineConfig.litellm_base``;
    no secret value is ever logged here.
    """
    cli_path = find_codex_cli()
    if cli_path is None:
        logger.error("engine_select.codex | {}", _CODEX_CLI_MISSING_MSG)
        raise EngineSelectionError(_CODEX_CLI_MISSING_MSG)
    logger.info("engine_select.codex | cli resolved (auth via subscription CLI)")

    from .codex_engine import CodexEngine

    base = env.get(ENV_CODEX_BASE)
    key = env.get(ENV_CODEX_KEY)
    requested_model = _resolve_codex_model(env)
    cfg = EngineConfig(
        kind="codex",
        model=requested_model,
        litellm_base=base or None,
        backup_model=None,
    )
    extra_env: dict[str, str] = {}
    if base:
        extra_env[ENV_CODEX_BASE] = base
    if key:
        extra_env[ENV_CODEX_KEY] = key
    return CodexEngine(cfg, env=extra_env or None)
