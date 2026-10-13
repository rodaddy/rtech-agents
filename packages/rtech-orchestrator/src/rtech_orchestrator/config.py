"""Frozen configuration + target resolution for the orchestrator.

Minimal POC config: the single agentd agent id that ``scope=single`` tasks route
to. The target id is NEVER a hardcoded literal — a bare ``"agentd"`` default is
exactly what let two agentds collide on the shared bus (issue #49). The id comes
from ``RTECH_TARGET_AGENT_ID`` (or the back-compat ``RTECH_SINGLE_AGENT_ID``
alias), or a JSON registry file (``RTECH_AGENT_REGISTRY`` → ``{"single": "<id>"}``).
If neither is set, resolution FAILS LOUDLY at startup rather than silently
falling back to a shared literal.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

# Primary + back-compat env vars naming the single-scope target agent id.
ENV_TARGET_AGENT_ID = "RTECH_TARGET_AGENT_ID"
ENV_SINGLE_AGENT_ID = "RTECH_SINGLE_AGENT_ID"
# Env var naming this orchestrator's stack id for the ingress guard (issue #75).
ENV_STACK_ID = "RTECH_STACK_ID"
# Optional comma-separated fleet ids fronted by a hermes bridge (PR #73):
# dispatches to these participants complete via the typed fleet Done the
# orchestrator consumes directly, never via edge-authored turn terminality.
ENV_BRIDGED_AGENTS = "RTECH_BRIDGED_AGENTS"
# Optional JSON registry mapping a scope to its target id, e.g. {"single": "…"}.
ENV_AGENT_REGISTRY = "RTECH_AGENT_REGISTRY"
_REGISTRY_SINGLE_KEY = "single"


class TargetResolutionError(RuntimeError):
    """Raised when no target agent id can be resolved (unset and no registry)."""


class StackResolutionError(RuntimeError):
    """Raised when no stack id can be resolved (ingress guard config missing)."""


@dataclass(frozen=True)
class OrchestratorConfig:
    """Configuration for single-scope routing and ingress stack scoping.

    Attributes:
        single_agent_id: The agentd agent id that ``scope=single`` tasks route
            to. Required and non-empty.
        stack_id: This orchestrator's stack id for the ingress guard
            (issue #75). Only ``TaskFrame``s whose ``stack`` matches are
            consumed; a missing id fails loudly at startup — the guard is
            security-floor config and never defaults to a shared literal.
        bridged_agents: Fleet ids fronted by a hermes bridge (PR #73 redesign).
            A dispatch to one of these records the participant as bridged on
            the task graph; its terminal evidence is the typed fleet ``Done``
            the orchestrator consumes directly. Empty by default — no bridge,
            no fleet-side consumption.
    """

    single_agent_id: str
    stack_id: str
    bridged_agents: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Validate every configured id is non-empty.

        Raises:
            ValueError: If ``single_agent_id``, ``stack_id``, or any bridged
                agent id is empty or whitespace-only.
        """
        if not self.single_agent_id or not self.single_agent_id.strip():
            logger.error("rejected empty single_agent_id")
            raise ValueError("single_agent_id must be a non-empty string")
        if not self.stack_id or not self.stack_id.strip():
            logger.error("rejected empty stack_id")
            raise ValueError("stack_id must be a non-empty string")
        if any(not agent or not agent.strip() for agent in self.bridged_agents):
            logger.error("rejected blank bridged agent id")
            raise ValueError("bridged_agents must contain only non-empty ids")


def _registry_target(registry_path: str) -> str | None:
    """Return the single-scope target id from a JSON registry file, or ``None``.

    The registry is ``{"single": "<agent-id>", ...}``. A missing file, unparsable
    content, or absent/empty ``single`` key yields ``None`` (the caller decides
    whether that is fatal). Never raises for a bad registry — it degrades to the
    env path or the loud unset error.
    """
    path = Path(registry_path).expanduser()
    if not path.is_file():
        logger.warning("config.registry | not a file, ignoring | path={}", path)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("config.registry | unreadable, ignoring | path={} reason={}", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("config.registry | non-object ignored | path={}", path)
        return None
    raw = data.get(_REGISTRY_SINGLE_KEY)
    if not isinstance(raw, str) or not raw.strip():
        logger.warning(
            "config.registry | no non-empty '{}' entry | path={}", _REGISTRY_SINGLE_KEY, path
        )
        return None
    return raw.strip()


def resolve_target_agent_id(env: dict[str, str] | None = None) -> str:
    """Resolve the single-scope target agent id, failing loudly when unset.

    Order: ``RTECH_TARGET_AGENT_ID`` → ``RTECH_SINGLE_AGENT_ID`` (back-compat) →
    the ``single`` entry of the ``RTECH_AGENT_REGISTRY`` JSON file. No hardcoded
    literal fallback — an unresolved target raises :class:`TargetResolutionError`
    so a misconfigured orchestrator never silently routes to a shared ``agentd``.

    Args:
        env: The environment mapping to read (defaults to ``os.environ``).

    Returns:
        The resolved non-empty target agent id.

    Raises:
        TargetResolutionError: If no env var and no registry entry supply an id.
    """
    source = os.environ if env is None else env
    for var in (ENV_TARGET_AGENT_ID, ENV_SINGLE_AGENT_ID):
        value = source.get(var, "").strip()
        if value:
            logger.info("config.target | resolved | agent_id={} source={}", value, var)
            return value
    registry_path = source.get(ENV_AGENT_REGISTRY, "").strip()
    if registry_path:
        target = _registry_target(registry_path)
        if target:
            logger.info("config.target | resolved | agent_id={} source=registry", target)
            return target
    logger.error(
        "config.target | UNSET | set {} (or {}) or provide {} with a 'single' entry",
        ENV_TARGET_AGENT_ID,
        ENV_SINGLE_AGENT_ID,
        ENV_AGENT_REGISTRY,
    )
    raise TargetResolutionError(
        f"no target agent id: set {ENV_TARGET_AGENT_ID} (or {ENV_SINGLE_AGENT_ID}) "
        f"or provide {ENV_AGENT_REGISTRY} pointing at a JSON file with a "
        f"'{_REGISTRY_SINGLE_KEY}' entry"
    )


def resolve_stack_id(env: dict[str, str] | None = None) -> str:
    """Resolve this orchestrator's stack id, failing loudly when unset.

    The stack id scopes task ingress (issue #75): the orchestrator only
    consumes ``TaskFrame``s whose ``stack`` field matches. There is NO default
    — a defaulted guard is exactly how two stacks on the shared bus
    cross-consume, so a missing id fails closed at startup.

    Args:
        env: The environment mapping to read (defaults to ``os.environ``).

    Returns:
        The resolved non-empty stack id.

    Raises:
        StackResolutionError: If ``RTECH_STACK_ID`` is unset or empty.
    """
    source = os.environ if env is None else env
    value = source.get(ENV_STACK_ID, "").strip()
    if value:
        logger.info("config.stack | resolved | stack_id={} source={}", value, ENV_STACK_ID)
        return value
    logger.error("config.stack | UNSET | set {} to this stack's id", ENV_STACK_ID)
    raise StackResolutionError(
        f"no stack id: set {ENV_STACK_ID} to this stack's id (ingress guard, issue #75)"
    )


def resolve_bridged_agents(env: dict[str, str] | None = None) -> frozenset[str]:
    """Resolve the bridge-fronted fleet ids, empty when none are configured.

    Optional by design (PR #73): with no bridge deployed there is nothing to
    consume on the fleet side. When set, ``RTECH_BRIDGED_AGENTS`` is a
    comma-separated id list; blank fragments are rejected loudly rather than
    silently skipped so a typo never half-configures the bridge boundary.

    Args:
        env: The environment mapping to read (defaults to ``os.environ``).

    Returns:
        The resolved (possibly empty) bridged agent id set.

    Raises:
        ValueError: If the variable is set but contains a blank fragment.
    """
    source = os.environ if env is None else env
    raw = source.get(ENV_BRIDGED_AGENTS, "").strip()
    if not raw:
        logger.debug("config.bridged | none configured")
        return frozenset()
    ids = [fragment.strip() for fragment in raw.split(",")]
    if any(not agent for agent in ids):
        logger.error("config.bridged | blank id fragment in {}", ENV_BRIDGED_AGENTS)
        raise ValueError(f"{ENV_BRIDGED_AGENTS} must not contain blank id fragments")
    resolved = frozenset(ids)
    logger.info("config.bridged | resolved | bridged_agents={}", sorted(resolved))
    return resolved
