"""Frozen schedule configuration for the cron scheduler.

A schedule table is owner-authored JSON loaded from ``RTECH_SCHEDULE_FILE``. Each
entry is a frozen, self-validating :class:`ScheduleEntry`; the whole table plus
the stack-scope stamp is a frozen :class:`SchedulerConfig`. Loading fails
**closed** — a missing file, unparsable JSON, an unknown field, a bad cron
expression, an unknown timezone, or a scope/participant mismatch raises
:class:`ScheduleConfigError` at startup rather than minting broken frames later.

The stack id is REQUIRED with no default: a defaulted stack guard is
exactly how two stacks on a shared bus cross-consume, so an unset id fails loud.

Security floor: the loader logs sanitized summaries only — schedule ids, cron
shapes, and counts — never the task text, requester, or origin identifiers, and
never the raw file contents.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from rtech_frames import validate_participants

from .cron import CronParseError, CronSchedule, parse_cron

# Env var naming the JSON schedule table. No default: unset fails closed.
ENV_SCHEDULE_FILE = "RTECH_SCHEDULE_FILE"
# Env var naming this stack's id, stamped on every minted frame.
ENV_STACK_ID = "RTECH_STACK_ID"

_ALLOWED_SCOPES: frozenset[str] = frozenset({"single", "duo", "trio"})
_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "schedule_id",
        "cron",
        "timezone",
        "enabled",
        "text",
        "scope",
        "participants",
        "requester",
        "origin_edge",
    }
)
# Bound the id so a schedule id becomes a safe, non-degenerate task-id segment.
_MAX_SCHEDULE_ID_LEN = 128
# The ASCII alphabet permitted in a schedule id. ``str.isalnum`` is deliberately
# NOT used: it accepts Unicode letters/digits, which are not subject-safe.
_ID_ALPHABET: frozenset[str] = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
)


class ScheduleConfigError(RuntimeError):
    """Raised when the schedule table cannot be loaded or validated (fail closed)."""


@dataclass(frozen=True)
class ScheduleEntry:
    """One validated cron schedule that mints a TaskFrame per occurrence.

    Attributes:
        schedule_id: Stable identity for this schedule. Drives the deterministic
            per-occurrence task id, so it must be unique across the table and a
            safe task-id segment (``[A-Za-z0-9_-]+``).
        cron: The parsed cron schedule (already timezone-bound).
        enabled: When ``False`` the schedule is loaded and validated but never
            armed — a disabled schedule mints nothing.
        text: The task text stamped on the minted ``TaskFrame``.
        scope: The task scope literal (``single``/``duo``/``trio``).
        participants: The named agent ids for a duo/trio task; empty for single.
        requester: The ``requester`` stamped on the minted frame (who asked).
        origin_edge: The ``origin_edge`` provenance stamped on the minted frame.
    """

    schedule_id: str
    cron: CronSchedule
    enabled: bool
    text: str
    scope: Literal["single", "duo", "trio"]
    participants: tuple[str, ...]
    requester: str
    origin_edge: str

    def __post_init__(self) -> None:
        """Reject a schedule that could mint an invalid or un-routable frame.

        Every check mirrors a ``TaskFrame`` invariant so a schedule that would
        fail at mint time fails at CONFIG LOAD instead: the id must be a safe
        task-id segment, text/requester/origin_edge non-blank, scope known, and
        participants consistent with scope (the one shared wire-safety check).

        Raises:
            ScheduleConfigError: On any invariant violation.
        """
        sid = self.schedule_id
        if not sid or not sid.strip():
            raise ScheduleConfigError("schedule_id must be non-empty")
        if len(sid) > _MAX_SCHEDULE_ID_LEN:
            raise ScheduleConfigError(f"schedule_id {sid!r} exceeds {_MAX_SCHEDULE_ID_LEN} chars")
        # ASCII-only: ``str.isalnum`` accepts Unicode letters/digits (e.g. "é",
        # "٢"), so it is NOT sufficient. The id becomes a subject-safe task-id
        # segment, so restrict to ASCII ``[A-Za-z0-9_-]+`` explicitly.
        if not all(c in _ID_ALPHABET for c in sid):
            raise ScheduleConfigError(
                f"schedule_id {sid!r} must be an ASCII-safe id segment ([A-Za-z0-9_-]+)"
            )
        if not self.text or not self.text.strip():
            raise ScheduleConfigError(f"schedule {sid!r}: text must be non-empty")
        if not self.requester or not self.requester.strip():
            raise ScheduleConfigError(f"schedule {sid!r}: requester must be non-empty")
        if not self.origin_edge or not self.origin_edge.strip():
            raise ScheduleConfigError(f"schedule {sid!r}: origin_edge must be non-empty")
        if self.scope not in _ALLOWED_SCOPES:
            raise ScheduleConfigError(f"schedule {sid!r}: unknown scope {self.scope!r}")
        try:
            validate_participants(list(self.participants), self.scope)
        except ValueError as exc:
            raise ScheduleConfigError(
                f"schedule {sid!r}: participants/scope invalid: {exc}"
            ) from exc


@dataclass(frozen=True)
class SchedulerConfig:
    """The frozen schedule table plus the stack-scope stamp for minted frames.

    Attributes:
        stack_id: This stack's id, stamped on every minted frame so a
            stack-guarded orchestrator consumes it. Non-empty.
        schedules: The validated schedule entries. Schedule ids are unique.
    """

    stack_id: str
    schedules: tuple[ScheduleEntry, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Reject a blank stack id or duplicate schedule ids.

        Raises:
            ScheduleConfigError: On a blank stack id or a repeated schedule_id.
        """
        if not self.stack_id or not self.stack_id.strip():
            raise ScheduleConfigError("stack_id must be non-empty; there is no default stack id")
        seen: set[str] = set()
        for entry in self.schedules:
            if entry.schedule_id in seen:
                raise ScheduleConfigError(
                    f"duplicate schedule_id {entry.schedule_id!r}; ids must be unique"
                )
            seen.add(entry.schedule_id)


def _entry_from_raw(raw: Any, index: int) -> ScheduleEntry:
    """Build one validated :class:`ScheduleEntry` from a raw JSON object.

    Args:
        raw: The decoded JSON object for one schedule.
        index: The entry index, for error context.

    Returns:
        The validated schedule entry.

    Raises:
        ScheduleConfigError: On a non-object, unknown key, missing required
            field, wrong type, or bad cron/timezone.
    """
    if not isinstance(raw, dict):
        raise ScheduleConfigError(f"schedule #{index} must be a JSON object")
    unknown = set(raw) - _ALLOWED_KEYS
    if unknown:
        raise ScheduleConfigError(
            f"schedule #{index}: unknown keys {sorted(unknown)}; allowed {sorted(_ALLOWED_KEYS)}"
        )
    schedule_id = raw.get("schedule_id")
    cron_expr = raw.get("cron")
    if not isinstance(schedule_id, str):
        raise ScheduleConfigError(f"schedule #{index}: schedule_id must be a string")
    if not isinstance(cron_expr, str):
        raise ScheduleConfigError(f"schedule {schedule_id!r}: cron must be a string")
    timezone = raw.get("timezone", "UTC")
    if not isinstance(timezone, str):
        raise ScheduleConfigError(f"schedule {schedule_id!r}: timezone must be a string")
    text = raw.get("text")
    if not isinstance(text, str):
        raise ScheduleConfigError(f"schedule {schedule_id!r}: text must be a string")
    scope = raw.get("scope", "single")
    if not isinstance(scope, str):
        raise ScheduleConfigError(f"schedule {schedule_id!r}: scope must be a string")
    requester = raw.get("requester", "scheduler")
    origin_edge = raw.get("origin_edge", "scheduler")
    if not isinstance(requester, str) or not isinstance(origin_edge, str):
        raise ScheduleConfigError(
            f"schedule {schedule_id!r}: requester/origin_edge must be strings"
        )
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ScheduleConfigError(f"schedule {schedule_id!r}: enabled must be a boolean")
    raw_participants = raw.get("participants", [])
    if not isinstance(raw_participants, list) or any(
        not isinstance(p, str) for p in raw_participants
    ):
        raise ScheduleConfigError(
            f"schedule {schedule_id!r}: participants must be a list of strings"
        )
    try:
        cron = parse_cron(cron_expr, timezone)
    except CronParseError as exc:
        raise ScheduleConfigError(f"schedule {schedule_id!r}: {exc}") from exc
    return ScheduleEntry(
        schedule_id=schedule_id,
        cron=cron,
        enabled=enabled,
        text=text,
        scope=scope,  # type: ignore[arg-type]  # narrowed by ScheduleEntry.__post_init__
        participants=tuple(raw_participants),
        requester=requester,
        origin_edge=origin_edge,
    )


def load_schedules(raw_table: Any, stack_id: str) -> SchedulerConfig:
    """Build a validated :class:`SchedulerConfig` from a decoded JSON table.

    Args:
        raw_table: The decoded JSON — a list of schedule objects.
        stack_id: This stack's id, stamped on minted frames.

    Returns:
        The frozen, fully validated scheduler config.

    Raises:
        ScheduleConfigError: On a non-list table or any invalid entry.
    """
    if not isinstance(raw_table, list):
        raise ScheduleConfigError("schedule table must be a JSON list of schedule objects")
    entries = tuple(_entry_from_raw(raw, i) for i, raw in enumerate(raw_table))
    config = SchedulerConfig(stack_id=stack_id, schedules=entries)
    enabled_count = sum(1 for entry in entries if entry.enabled)
    logger.info(
        "scheduler.config | loaded | stack_id={} schedules={} enabled={}",
        stack_id,
        len(entries),
        enabled_count,
    )
    return config


def load_scheduler_config(env: dict[str, str] | None = None) -> SchedulerConfig:
    """Resolve env, read the schedule file, and build a validated config.

    Reads ``RTECH_SCHEDULE_FILE`` (required) and ``RTECH_STACK_ID`` (required),
    resolves the file path, decodes its JSON, and validates the whole table.
    Every failure — unset env, missing file, unreadable/unparsable JSON, or an
    invalid entry — raises :class:`ScheduleConfigError` so startup fails closed.

    Args:
        env: Environment mapping (defaults to ``os.environ``).

    Returns:
        The validated scheduler config.

    Raises:
        ScheduleConfigError: On any resolution, read, or validation failure.
    """
    source = os.environ if env is None else env
    file_ref = source.get(ENV_SCHEDULE_FILE, "").strip()
    if not file_ref:
        logger.error("scheduler.config | {} unset; aborting", ENV_SCHEDULE_FILE)
        raise ScheduleConfigError(f"{ENV_SCHEDULE_FILE} must name the JSON schedule table")
    stack_id = source.get(ENV_STACK_ID, "").strip()
    if not stack_id:
        logger.error("scheduler.config | {} unset; aborting", ENV_STACK_ID)
        raise ScheduleConfigError(f"{ENV_STACK_ID} must be set (ingress guard)")
    path = Path(file_ref).expanduser().resolve()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        # Neither the resolved path nor the OSError text is logged: an absolute
        # schedule-file path can carry a home dir / user name, and the OSError
        # message echoes that same path. Log the exception TYPE only; the raised
        # error keeps context for the caller, which itself logs only the type.
        logger.error(
            "scheduler.config | schedule file unreadable | error_type={}",
            type(exc).__name__,
        )
        raise ScheduleConfigError(f"schedule file {path} is unreadable: {exc}") from exc
    try:
        raw_table = json.loads(text)
    except json.JSONDecodeError as exc:
        # The path is not logged here either (same rationale), and the decode
        # error is never logged raw — a JSON error message can echo file content.
        logger.error(
            "scheduler.config | schedule file invalid JSON | error_type={}",
            type(exc).__name__,
        )
        raise ScheduleConfigError(f"schedule file {path} is not valid JSON: {exc}") from exc
    return load_schedules(raw_table, stack_id)


__all__ = [
    "ENV_SCHEDULE_FILE",
    "ENV_STACK_ID",
    "ScheduleConfigError",
    "ScheduleEntry",
    "SchedulerConfig",
    "load_scheduler_config",
    "load_schedules",
]
