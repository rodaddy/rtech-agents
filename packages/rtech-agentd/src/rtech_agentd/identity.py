"""Resolve this agentd's unique bus identity — never a bare shared literal.

Two agentds on the same fleet bus that both default to ``agent_id="agentd"`` both
subscribe ``rtech.agent.agentd.task`` and both consume the same task.
The fix is a per-host default: absent an explicit ``RTECH_AGENT_ID``, the id is
``agentd-<hostname>`` so two hosts never collide by accident. An operator may
still pin any explicit id via the env var.

The identity is a frozen dataclass validated non-empty at construction. This
module reads ``RTECH_AGENT_ID`` and the host name only; it never touches secrets.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass

from loguru import logger

ENV_AGENT_ID = "RTECH_AGENT_ID"

# The per-host default PREFIX. The bare word is never a valid final id — the
# hostname suffix is what makes it collision-safe across the shared bus.
_DEFAULT_PREFIX = "agentd"


@dataclass(frozen=True)
class AgentIdentity:
    """This agentd's unique id on the shared fleet bus.

    Attributes:
        agent_id: The bus identity; non-empty, never the bare shared literal.
    """

    agent_id: str

    def __post_init__(self) -> None:
        """Reject an empty id.

        Raises:
            ValueError: If ``agent_id`` is empty or whitespace-only.
        """
        if not self.agent_id or not self.agent_id.strip():
            logger.error("identity | rejected empty agent_id")
            raise ValueError("agent_id must be a non-empty string")


def _hostname() -> str:
    """Return the short host name, falling back to ``unknown`` if unavailable."""
    try:
        host = socket.gethostname().strip()
    except OSError as exc:
        logger.warning("identity.hostname | lookup failed | reason={}", exc)
        return "unknown"
    # Short name only; a fully-qualified name would bloat the subject string.
    short = host.split(".", 1)[0]
    return short or "unknown"


def default_agent_id() -> str:
    """Return the collision-safe per-host default id ``agentd-<hostname>``."""
    return f"{_DEFAULT_PREFIX}-{_hostname()}"


def resolve_agent_id(env: dict[str, str] | None = None) -> AgentIdentity:
    """Resolve this agentd's identity from the environment.

    An explicit ``RTECH_AGENT_ID`` wins; otherwise the per-host default
    ``agentd-<hostname>`` is used so two hosts never share ``agentd`` by accident.

    Args:
        env: The environment mapping to read (defaults to ``os.environ``).

    Returns:
        The resolved, validated :class:`AgentIdentity`.
    """
    source = os.environ if env is None else env
    raw = source.get(ENV_AGENT_ID, "").strip()
    agent_id = raw or default_agent_id()
    logger.info(
        "identity.resolve | agent_id={} source={}",
        agent_id,
        "env" if raw else "host-default",
    )
    return AgentIdentity(agent_id=agent_id)
