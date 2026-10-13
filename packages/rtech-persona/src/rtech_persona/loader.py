"""Load persona files (SOUL/USER/KERNEL) from an agent home.

Replicates the Hermes ``load_soul_md`` behavior: read the file, run a content
scan (fail closed on findings — the content would otherwise enter the system
prompt verbatim), and cap the length to a context-budget-scaled limit with
truncation logged.

SOUL.md is required (fail fast). USER.md and KERNEL.md are optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ._scan import scan_persona_content
from .config import (
    DEFAULT_CAP,
    KERNEL_FILENAME,
    SOUL_FILENAME,
    USER_FILENAME,
    CapConfig,
)


class PersonaError(Exception):
    """Base error for persona loading failures."""


class MissingSoulError(PersonaError):
    """Raised when the required SOUL.md is absent or empty."""


class PersonaScanError(PersonaError):
    """Raised when a persona file fails the content scan."""


@dataclass(frozen=True)
class Persona:
    """A loaded persona.

    Attributes:
        soul: SOUL.md content (identity). Always present — loading fails
            otherwise.
        user: USER.md content (profile), or None when absent.
        kernel: KERNEL.md content (rules), or None when absent.
    """

    soul: str
    user: str | None
    kernel: str | None


def _truncate(content: str, filename: str, context_budget: int | None, cap: CapConfig) -> str:
    """Cap ``content`` to the scaled max, keeping a head+tail slice with a marker.

    Args:
        content: The scanned file content.
        filename: Source filename (for the truncation marker + logs).
        context_budget: Model context window in tokens (scales the cap).
        cap: Cap parameters.

    Returns:
        The content unchanged, or a head+marker+tail truncation.
    """
    max_chars = cap.max_chars(context_budget)
    if len(content) <= max_chars:
        return content

    head_chars = int(max_chars * cap.truncate_head_ratio)
    tail_chars = int(max_chars * cap.truncate_tail_ratio)
    logger.warning(
        "persona.truncate | file={} original_len={} cap={} kept={}+{}",
        filename,
        len(content),
        max_chars,
        head_chars,
        tail_chars,
    )
    head = content[:head_chars]
    tail = content[-tail_chars:]
    marker = (
        f"\n\n[...truncated {filename}: kept {head_chars}+{tail_chars} of "
        f"{len(content)} chars. The middle is omitted.]\n\n"
    )
    return head + marker + tail


def _load_one(
    home: Path,
    filename: str,
    *,
    required: bool,
    context_budget: int | None,
    cap: CapConfig,
) -> str | None:
    """Read, scan, and cap a single persona file.

    Args:
        home: Agent home directory.
        filename: File to load (e.g. ``SOUL.md``).
        required: If True, absence/emptiness raises ``MissingSoulError``.
        context_budget: Model context window in tokens (scales the cap).
        cap: Cap parameters.

    Returns:
        The scanned, capped content, or None when the file is optional and
        absent/empty.

    Raises:
        MissingSoulError: If a required file is missing or empty.
        PersonaScanError: If the content fails the scan.
    """
    path = (home / filename).resolve()
    logger.debug("persona.load | file={} required={}", filename, required)

    if not path.is_file():
        if required:
            logger.error("persona.load | required file missing | file={} home={}", filename, home)
            raise MissingSoulError(f"Required persona file not found: {path}")
        logger.info("persona.load | optional file absent, skipping | file={}", filename)
        return None

    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.error("persona.load | read failed | file={} reason={}", filename, exc)
        if required:
            raise MissingSoulError(f"Could not read required persona file {path}: {exc}") from exc
        return None
    finally:
        logger.debug("persona.load | read complete | file={}", filename)

    if not content:
        if required:
            logger.error("persona.load | required file empty | file={}", filename)
            raise MissingSoulError(f"Required persona file is empty: {path}")
        logger.info("persona.load | optional file empty, skipping | file={}", filename)
        return None

    findings = scan_persona_content(content)
    if findings:
        logger.warning(
            "persona.scan | rejected | file={} findings={}", filename, ", ".join(findings)
        )
        raise PersonaScanError(
            f"Persona file {filename} failed content scan: {', '.join(findings)}"
        )

    capped = _truncate(content, filename, context_budget, cap)
    logger.info("persona.load | ok | file={} len={}", filename, len(capped))
    return capped


def load_persona(
    agent_home: Path,
    *,
    context_budget: int | None = None,
    cap: CapConfig = DEFAULT_CAP,
) -> Persona:
    """Load a persona from ``agent_home``.

    SOUL.md is required; USER.md and KERNEL.md are optional. Every file is
    scanned and length-capped. Loading fails fast on a missing SOUL.md or a
    scan finding.

    Args:
        agent_home: Directory containing the persona files.
        context_budget: Model context window in tokens; scales the length cap.
            None uses the flat floor.
        cap: Cap parameters. Defaults to the module default.

    Returns:
        A populated :class:`Persona`.

    Raises:
        MissingSoulError: If SOUL.md is missing or empty.
        PersonaScanError: If any persona file fails the content scan.
    """
    home = agent_home.resolve()
    logger.debug("persona.load_persona | home={} context_budget={}", home, context_budget)

    if not home.is_dir():
        logger.error("persona.load_persona | agent home not a directory | home={}", home)
        raise MissingSoulError(f"Agent home is not a directory: {home}")

    soul = _load_one(home, SOUL_FILENAME, required=True, context_budget=context_budget, cap=cap)
    # _load_one with required=True never returns None (it raises); assert for mypy.
    assert soul is not None
    user = _load_one(home, USER_FILENAME, required=False, context_budget=context_budget, cap=cap)
    kernel = _load_one(
        home, KERNEL_FILENAME, required=False, context_budget=context_budget, cap=cap
    )

    logger.info(
        "persona.load_persona | ok | soul_len={} user={} kernel={}",
        len(soul),
        "present" if user else "absent",
        "present" if kernel else "absent",
    )
    return Persona(soul=soul, user=user, kernel=kernel)
