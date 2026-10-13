"""Parse a standard 5-field cron expression and compute next occurrences.

Occurrence parsing and calculation are delegated to :mod:`croniter` (a
maintained, widely-used cron library). This module is a thin, fully-owned typed
wrapper: it keeps a small stable API (:class:`CronSchedule`, :func:`parse_cron`,
:class:`CronParseError`) so callers, config, and tests never touch croniter
directly, and it wraps croniter's raw next-occurrence result with a small
*generic* DST guard. It does NOT re-implement field parsing or occurrence
scanning — croniter owns the standard five-field grammar (``*``, comma lists,
ranges, steps such as ``*/15``; the standard DOM/DOW OR rule; Sunday as both 0
and 7; ``*/1`` wildcard-equivalents in non-day fields — see the ``*/1`` day-field
normalization below for the one exception).

One owned normalization runs *before* croniter, because croniter diverges from
the standard OR rule for a wildcard-equivalent day field:

* **``*/1`` day field → ``*``.** ``*/1`` means "every unit", so under the
  standard DOM/DOW OR rule a ``*/1`` day field is *unrestricted*. croniter
  instead treats a raw ``*/1`` in a day field as restricted, collapsing OR
  expressions such as ``0 0 13 * */1`` (13th OR every weekday) and
  ``0 0 */1 * 5`` (every day OR Friday) to daily. Only an *exact* ``*/1`` day
  field is rewritten; every other step/range/list keeps its restricted semantics.

Two owned validations run *before* croniter, because croniter is more permissive
than the standard five-field contract this daemon accepts:

* **Field count must be exactly five.** croniter also accepts 6-/7-field forms
  (seconds/year) and ``@daily``-style macros; those are rejected here so the
  schedule table only ever holds standard five-field expressions.
* **Ranges must be ascending (``a-b`` with ``a <= b``).** croniter silently
  treats an inverted ``5-1`` as a wrap-around set; the standard contract rejects
  it.

The DST guard is generic and parser-free. Occurrences are iterated in croniter's
*naive* (timezone-free) field space, then localized here at ``fold=0`` and
checked with the standard library:

* **Spring-forward (nonexistent wall time):** a naive occurrence whose localized
  wall clock does not survive a full-date-and-time UTC round-trip fell in the DST
  gap and is skipped — never fired at a phantom instant. The round-trip compares
  the whole local date and time, so it catches a skipped hour AND a skipped
  calendar date (e.g. Pacific/Apia's all-of-2011-12-30 gap), not just an
  hour/minute mismatch. (croniter, given a tz-aware base, would instead substitute
  a shifted real time; iterating in naive space and guarding here avoids that.)
* **Fall-back (ambiguous wall time):** naive field arithmetic yields each wall
  clock once, and it is pinned to ``fold=0`` (the earlier real instant), so an
  ambiguous hour fires at most once and never returns an instant ``<= after``,
  including when queried during the second fold.

Malformed expressions (bad field count, out-of-range value, bad step, inverted
range, non-numeric token, unknown timezone) raise :class:`CronParseError` so a
schedule fails closed at config load, never at fire time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import CroniterError, croniter
from loguru import logger

# The standard five-field contract: minute hour day-of-month month day-of-week.
_FIELD_COUNT = 5
# Zero-based positions of the two day fields (DOM and DOW) in a five-field
# expression. The standard DOM/DOW OR rule keys off whether each is *restricted*.
_DAY_FIELD_INDICES = (2, 4)
# A day field written as literally ``*/1`` is wildcard-equivalent (every unit),
# so under the standard OR rule it must NOT count as restricted. croniter,
# however, treats a raw ``*/1`` in a day field as restricted, which silently
# collapses OR expressions such as ``0 0 13 * */1`` to daily. Normalizing the
# exact ``*/1`` day token to plain ``*`` restores standard OR semantics.
_WILDCARD_STEP_ONE = "*/1"
# Hard cap on the naive-space guard loop. A DST gap skips at most one day's worth
# of minute occurrences; a handful of iterations always suffices for a real
# schedule, so this bounds a pathologically empty result rather than looping
# forever. Well above any real gap so it never trips on a valid schedule.
_MAX_GUARD_ITERATIONS = 1440 * 4


class CronParseError(ValueError):
    """Signal an invalid cron expression (malformed field, out-of-range value)."""


def _reject_non_ascending_ranges(expression: str) -> None:
    """Raise if any ``a-b`` term has ``a > b`` (croniter accepts wrap-around).

    croniter silently treats an inverted range like ``5-1`` as a wrap-around
    set; the standard five-field contract this daemon accepts rejects it. This is
    a small syntactic guard, not an occurrence parser: it only inspects the
    numeric endpoints of explicit ranges and leaves all field semantics
    (bounds, steps, DOM/DOW, wildcards) to croniter.

    Raises:
        CronParseError: On an inverted numeric range in any field.
    """
    for field in expression.split():
        for term in field.split(","):
            # Strip an optional step so ``1-5/2`` inspects the ``1-5`` range only.
            base = term.split("/", 1)[0]
            if "-" not in base:
                continue
            start_text, _, end_text = base.partition("-")
            try:
                start, end = int(start_text), int(end_text)
            except ValueError:
                # Non-numeric endpoints are croniter's to reject (e.g. names it
                # does not know); leave them to :func:`croniter.is_valid`.
                continue
            if start > end:
                raise CronParseError(f"inverted range {base!r} in {expression!r}")


def _normalize_wildcard_step_one_day_fields(expression: str) -> str:
    """Rewrite an exact ``*/1`` day field (DOM or DOW) to plain ``*``.

    ``*/1`` means "every unit" and is wildcard-equivalent, so under the standard
    DOM/DOW OR rule a ``*/1`` day field must be treated as *unrestricted* (the
    same as ``*``). croniter instead treats a raw ``*/1`` in a day field as a
    restricted set, which collapses OR expressions like ``0 0 13 * */1`` (13th OR
    every weekday) to daily and ``0 0 */1 * 5`` (every day OR Friday) to daily.

    This is a purely syntactic normalization at the wrapper boundary: only a day
    field that is *exactly* ``*/1`` is rewritten. Every other step, range, or
    list — including a non-one step like ``*/2`` or a ``*/1`` in a non-day field
    — is left untouched so its restricted semantics are preserved. Applied once
    in :func:`parse_cron`, it flows to both validation and occurrence
    calculation without exposing croniter internals to callers.
    """
    fields = expression.split()
    if len(fields) != _FIELD_COUNT:
        # Field-count validation is :func:`parse_cron`'s; leave a malformed count
        # untouched so it fails closed there rather than being silently reshaped.
        return expression
    changed = False
    for index in _DAY_FIELD_INDICES:
        if fields[index] == _WILDCARD_STEP_ONE:
            fields[index] = "*"
            changed = True
    if not changed:
        return expression
    normalized = " ".join(fields)
    logger.debug("cron.normalize | */1 day field | {!r} -> {!r}", expression, normalized)
    return normalized


@dataclass(frozen=True)
class CronSchedule:
    """A validated 5-field cron expression bound to a timezone.

    Wraps :mod:`croniter` behind a small owned API. Field parsing and occurrence
    calculation are croniter's; this type adds the timezone binding, the
    strictly-after / DST guard in :meth:`next_after`, and a stable surface for
    callers, config, and tests.

    Attributes:
        expression: The validated five-field cron string.
        tz: The timezone occurrences are evaluated in.
    """

    expression: str
    tz: ZoneInfo

    def _wall_time_exists(self, moment: datetime) -> bool:
        """Return whether ``moment``'s wall clock is a real instant in :attr:`tz`.

        During a spring-forward the clock skips forward, so a wall time in the gap
        does not exist. ``zoneinfo`` renders such a gap time with the
        pre-transition offset, so converting to UTC and back yields a DIFFERENT
        wall clock. A real wall time survives the round-trip unchanged; a
        nonexistent one does not.

        The comparison spans the WHOLE local date and time, not just hour+minute.
        Some zones skip an entire calendar date rather than an hour — e.g.
        Pacific/Apia jumped 2011-12-29 24:00 straight to 2011-12-31 00:00, so all
        of 2011-12-30 never existed. A Dec-30 wall time round-trips to the SAME
        hour/minute on Dec-31, so an hour/minute-only check would wrongly accept
        the phantom date. Comparing year/month/day/hour/minute rejects both a
        skipped hour and a skipped date.
        """
        round_tripped = moment.astimezone(UTC).astimezone(self.tz)
        return (
            round_tripped.year,
            round_tripped.month,
            round_tripped.day,
            round_tripped.hour,
            round_tripped.minute,
        ) == (moment.year, moment.month, moment.day, moment.hour, moment.minute)

    def next_after(self, after: datetime) -> datetime:
        """Return the first matching occurrence strictly after ``after``.

        Occurrences are iterated in croniter's naive field space (so croniter
        owns all field semantics), localized here at ``fold=0``, and returned as a
        tz-aware datetime in :attr:`tz`. The result is always strictly after
        ``after`` in real UTC.

        DST correctness (see the module docstring for the full rationale):

        * **Spring-forward:** a naive occurrence whose localized wall clock does
          not survive a UTC round-trip fell in the gap and is skipped — never a
          phantom instant.
        * **Fall-back:** each wall clock is produced once by naive arithmetic and
          pinned to ``fold=0`` (the earlier real instant), so an ambiguous hour
          fires at most once and never returns an instant ``<= after``, including
          when queried during the second fold.

        Args:
            after: The exclusive lower bound (any tz-aware datetime).

        Returns:
            The next tz-aware occurrence strictly after ``after``.

        Raises:
            CronParseError: If no occurrence is found within the guard horizon
                (an effectively unsatisfiable schedule).
        """
        after_utc = after.astimezone(UTC)
        # croniter iterates in naive field space; seed it from ``after``'s wall
        # clock in this schedule's timezone so field arithmetic matches the tz.
        naive_base = after.astimezone(self.tz).replace(tzinfo=None)
        iterator = croniter(self.expression, naive_base)
        for _ in range(_MAX_GUARD_ITERATIONS):
            naive = iterator.get_next(datetime)
            local = naive.replace(tzinfo=self.tz, fold=0)
            if not self._wall_time_exists(local):
                # Spring-forward gap: this wall time never occurs; skip it rather
                # than fire at a phantom (or croniter-substituted) instant.
                continue
            if local.astimezone(UTC) <= after_utc:
                # Fall-back second fold or a same-instant edge: not strictly after.
                continue
            return local
        logger.error(
            "cron.next | no occurrence within horizon | tz={} expr={!r}",
            self.tz.key,
            self.expression,
        )
        raise CronParseError(
            "cron expression has no occurrence within the search horizon (unsatisfiable?)"
        )


def parse_cron(expression: str, timezone: str = "UTC") -> CronSchedule:
    """Parse a 5-field cron expression in ``timezone`` into a :class:`CronSchedule`.

    Field parsing/validation is croniter's; two owned checks run first to hold the
    standard five-field contract (exactly five fields; ascending ranges only).

    Args:
        expression: A whitespace-separated 5-field cron string.
        timezone: An IANA timezone name (default ``UTC``).

    Returns:
        The validated, timezone-bound schedule.

    Raises:
        CronParseError: On the wrong field count, an inverted range, any field
            croniter rejects, or an unknown timezone. A schedule that cannot be
            parsed fails closed at config load rather than at fire time.
    """
    fields = expression.split()
    if len(fields) != _FIELD_COUNT:
        raise CronParseError(
            f"cron expression must have exactly {_FIELD_COUNT} fields, got {len(fields)}: "
            f"{expression!r}"
        )
    # Normalize once, up front, so validation and occurrence calculation both see
    # the OR-correct form. Only an exact ``*/1`` day field is rewritten to ``*``;
    # every other step/range/list keeps its restricted semantics.
    expression = _normalize_wildcard_step_one_day_fields(expression)
    _reject_non_ascending_ranges(expression)
    if not croniter.is_valid(expression):
        raise CronParseError(f"malformed cron expression {expression!r}")
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError, KeyError) as exc:
        raise CronParseError(f"unknown timezone {timezone!r}: {exc}") from exc
    # Construct once to surface any residual croniter parse error at load time.
    try:
        croniter(expression)
    except (CroniterError, ValueError) as exc:
        raise CronParseError(f"malformed cron expression {expression!r}: {exc}") from exc
    schedule = CronSchedule(expression=expression, tz=tz)
    logger.debug("cron.parse | parsed | expr={!r} tz={}", expression, timezone)
    return schedule


__all__ = ["CronParseError", "CronSchedule", "parse_cron"]
