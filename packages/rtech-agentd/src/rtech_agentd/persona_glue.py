"""Compose the engine's system prompt from the agent's persona.

Thin glue over ``rtech_persona``: load SOUL/USER/KERNEL from the agent home
(scan + cap), then compose the ordered slots into one system-prompt string.
``memory_block`` is ``None`` for now — Open Brain recall lands in Phase 4 and
will be threaded through the same slot.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from rtech_persona import compose, load_persona


def build_system_prompt(
    agent_home: Path,
    *,
    model: str,
    platform_hint: str | None = None,
    memory_block: str | None = None,
    now: datetime | None = None,
    context_budget: int | None = None,
) -> str:
    """Load the persona and compose the full system prompt.

    Args:
        agent_home: Directory holding SOUL.md / USER.md / KERNEL.md.
        model: Engine model id (stamped into the timestamp/model line).
        platform_hint: Optional platform behavioral hint slot.
        memory_block: OB recall block; ``None`` until Phase 4.
        now: Reference time (defaults to UTC now).
        context_budget: Model context window in tokens (scales persona cap).

    Returns:
        The composed system-prompt string.

    Raises:
        rtech_persona.MissingSoulError: If SOUL.md is absent/empty.
        rtech_persona.PersonaScanError: If a persona file fails the content scan.
    """
    reference = now or datetime.now(UTC)
    logger.debug("persona_glue.build | home={} model={}", agent_home, model)
    persona = load_persona(agent_home, context_budget=context_budget)
    prompt = compose(
        persona,
        memory_block=memory_block,
        platform_hint=platform_hint,
        now=reference,
        model=model,
    )
    logger.info("persona_glue.build | ok | prompt_len={}", len(prompt))
    return prompt
