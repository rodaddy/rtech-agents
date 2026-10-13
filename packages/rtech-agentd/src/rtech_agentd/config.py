"""Typed agent configuration, loaded from ``agents/<name>/agent.yaml``.

Shape (from the governing spec, Notes → "Config shape")::

    agent: <name>
    home: /home/<name>/.rtech-agents
    engine: { kind: claude, model: "...", litellm_base: "http://...:4000/v1",
              backup_model: "..." }
    nats:   { url: "nats://...:4222", env: dev }
    openbrain: { base_url: "...", token_ref: vaultwarden:<item> }
    discord: { bot_token_ref: vaultwarden:<item>, ... }
    persona: { soul: SOUL.md, user: USER.md, kernel: KERNEL.md }

Credential indirection is enforced here: ``*_ref`` fields hold the NAME of a
secret (looked up at runtime), never a literal value. A literal-looking secret in
a ``*_ref`` field is rejected at load — a secret must never sit in config or a
log line. This module never resolves refs and never prints their values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

# A ref value must look like an indirection handle (e.g. "vaultwarden:item-name"
# or "env:VAR"), NOT a raw secret. We accept the "<provider>:<name>" shape and a
# bare env-style NAME, and reject anything that smells like a literal credential.
_ALLOWED_REF_PROVIDERS = frozenset({"vaultwarden", "vw", "env", "file"})

# Value prefixes that mark a raw credential regardless of any ':' it contains.
_SECRET_PREFIXES = ("sk-", "rk-", "pk-", "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "xox", "eyj")


class ConfigError(Exception):
    """Raised when agent.yaml is missing, malformed, or holds a literal secret."""


@dataclass(frozen=True)
class EngineConfig:
    """Engine selection + model routing (LiteLLM-compatible base URL)."""

    kind: str
    model: str
    litellm_base: str | None
    backup_model: str | None


@dataclass(frozen=True)
class NatsConfig:
    """NATS connection facts (url + env prefix)."""

    url: str
    env: str


@dataclass(frozen=True)
class OpenBrainConfig:
    """Open Brain endpoint + token indirection (Phase 4 uses this)."""

    base_url: str | None
    token_ref: str | None


@dataclass(frozen=True)
class DiscordConfig:
    """Discord surface config; token is a ref, never a literal (chatd uses this)."""

    bot_token_ref: str | None
    home_channel: str | None
    free_response_channels: tuple[str, ...]
    allow_bots: str


@dataclass(frozen=True)
class PersonaPaths:
    """Persona source filenames relative to the agent home."""

    soul: str
    user: str
    kernel: str


@dataclass(frozen=True)
class AgentConfig:
    """The whole per-agent config, fully typed and validated."""

    agent: str
    home: Path
    engine: EngineConfig
    nats: NatsConfig
    openbrain: OpenBrainConfig
    discord: DiscordConfig
    persona: PersonaPaths


def _require(mapping: dict[str, Any], key: str, ctx: str) -> Any:
    """Return a required key or raise ``ConfigError`` naming the context."""
    if key not in mapping:
        raise ConfigError(f"agent.yaml: missing required {ctx}.{key}")
    return mapping[key]


def _looks_like_literal_secret(value: str) -> bool:
    """Heuristic: does ``value`` look like a raw credential, not a ref handle?

    A ref is ``<provider>:<name>`` with a known provider, or a bare env-style
    NAME. Anything long, containing whitespace, or carrying an unknown
    ``provider:`` prefix is treated as a literal secret and rejected — refusing
    to let a token slip into config/logs is the safe default.
    """
    stripped = value.strip()
    if not stripped:
        return False
    # A known secret PREFIX (sk-/ghp_/xox.-/eyJ...) is a literal token even when
    # it happens to contain a ':' — reject before the provider check.
    lowered = stripped.lower()
    for prefix in _SECRET_PREFIXES:
        if lowered.startswith(prefix):
            return True
    if ":" in stripped:
        provider = stripped.split(":", 1)[0].strip().lower()
        return provider not in _ALLOWED_REF_PROVIDERS
    # No provider prefix: only accept a short, secret-name-shaped token.
    if len(stripped) > 64 or any(ch.isspace() for ch in stripped):
        return True
    return False


def _load_ref(mapping: dict[str, Any], key: str, ctx: str) -> str | None:
    """Load an optional ``*_ref`` field, rejecting literal secrets.

    Returns the ref handle (a NAME), or ``None`` when absent. Never returns or
    logs a secret value.
    """
    raw = mapping.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"agent.yaml: {ctx}.{key} must be a non-empty ref string")
    if _looks_like_literal_secret(raw):
        # Do NOT echo the value — it may be the secret we're refusing.
        raise ConfigError(
            f"agent.yaml: {ctx}.{key} looks like a literal secret; use a "
            f"'<provider>:<name>' ref ({sorted(_ALLOWED_REF_PROVIDERS)}) instead"
        )
    return raw.strip()


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    """Coerce a YAML list/scalar to a tuple of strings (empty when absent)."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(v) for v in value)
    raise ConfigError(f"agent.yaml: expected a list of strings, got {type(value).__name__}")


def load_agent_config(agent_yaml_path: Path) -> AgentConfig:
    """Load and validate an agent's config from ``agent.yaml``.

    Args:
        agent_yaml_path: Path to the agent's ``agent.yaml``.

    Returns:
        A fully-typed, validated :class:`AgentConfig`.

    Raises:
        ConfigError: If the file is missing, unparsable, missing required keys,
            or a ``*_ref`` field carries a literal-looking secret.
    """
    path = agent_yaml_path.resolve()
    logger.debug("config.load | path={}", path)
    if not path.is_file():
        logger.error("config.load | agent.yaml not found | path={}", path)
        raise ConfigError(f"agent.yaml not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.error("config.load | unparsable agent.yaml | path={} reason={}", path, exc)
        raise ConfigError(f"could not parse agent.yaml {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"agent.yaml must be a mapping, got {type(raw).__name__}")

    agent = str(_require(raw, "agent", "root")).strip()
    if not agent:
        raise ConfigError("agent.yaml: root.agent must be non-empty")

    home = Path(str(_require(raw, "home", "root"))).expanduser()

    engine_raw = _require(raw, "engine", "root")
    if not isinstance(engine_raw, dict):
        raise ConfigError("agent.yaml: engine must be a mapping")
    engine = EngineConfig(
        kind=str(_require(engine_raw, "kind", "engine")).strip(),
        model=str(_require(engine_raw, "model", "engine")).strip(),
        litellm_base=(
            str(engine_raw["litellm_base"]).strip() if engine_raw.get("litellm_base") else None
        ),
        backup_model=(
            str(engine_raw["backup_model"]).strip() if engine_raw.get("backup_model") else None
        ),
    )

    nats_raw = _require(raw, "nats", "root")
    if not isinstance(nats_raw, dict):
        raise ConfigError("agent.yaml: nats must be a mapping")
    nats = NatsConfig(
        url=str(_require(nats_raw, "url", "nats")).strip(),
        env=str(nats_raw.get("env", "dev")).strip(),
    )

    ob_raw = raw.get("openbrain") or {}
    if not isinstance(ob_raw, dict):
        raise ConfigError("agent.yaml: openbrain must be a mapping")
    openbrain = OpenBrainConfig(
        base_url=(str(ob_raw["base_url"]).strip() if ob_raw.get("base_url") else None),
        token_ref=_load_ref(ob_raw, "token_ref", "openbrain"),
    )

    dc_raw = raw.get("discord") or {}
    if not isinstance(dc_raw, dict):
        raise ConfigError("agent.yaml: discord must be a mapping")
    discord = DiscordConfig(
        bot_token_ref=_load_ref(dc_raw, "bot_token_ref", "discord"),
        home_channel=(str(dc_raw["home_channel"]).strip() if dc_raw.get("home_channel") else None),
        free_response_channels=_as_str_tuple(dc_raw.get("free_response_channels")),
        allow_bots=str(dc_raw.get("allow_bots", "mentions")).strip(),
    )

    p_raw = raw.get("persona") or {}
    if not isinstance(p_raw, dict):
        raise ConfigError("agent.yaml: persona must be a mapping")
    persona = PersonaPaths(
        soul=str(p_raw.get("soul", "SOUL.md")).strip(),
        user=str(p_raw.get("user", "USER.md")).strip(),
        kernel=str(p_raw.get("kernel", "KERNEL.md")).strip(),
    )

    logger.info(
        "config.load | ok | agent={} engine.kind={} nats.env={} home={}",
        agent,
        engine.kind,
        nats.env,
        home,
    )
    return AgentConfig(
        agent=agent,
        home=home,
        engine=engine,
        nats=nats,
        openbrain=openbrain,
        discord=discord,
        persona=persona,
    )
