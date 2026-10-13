"""Regression tests for the cron wrapper's next-occurrence contract.

Occurrence calculation is delegated to croniter behind the owned
:class:`~rtech_schedulerd.cron.CronSchedule` wrapper; these tests pin the
behavior the wrapper GUARANTEES regardless of the underlying library — standard
five-field syntax, the DOM/DOW OR rule, Sunday as 0 and 7, strictly-after
semantics, timezone-aware results, spring-forward gap skipping, fall-back
at-most-once (including a query during the second fold), and fail-closed parsing.
The DST cases in particular are the guard around the library, so they are
asserted independently of croniter's own (looser) DST behavior.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest
from rtech_schedulerd.cron import CronParseError, parse_cron

_NY = ZoneInfo("America/New_York")


def test_import_entrypoint() -> None:
    """The package entrypoint loads and exposes the public surface."""
    import rtech_schedulerd

    assert "parse_cron" in rtech_schedulerd.__all__
    assert "CronScheduler" in rtech_schedulerd.__all__


def test_every_minute_next_occurrence() -> None:
    """``* * * * *`` fires the very next whole minute after ``after``."""
    cron = parse_cron("* * * * *")
    after = datetime(2026, 1, 1, 8, 30, 45, tzinfo=UTC)
    assert cron.next_after(after) == datetime(2026, 1, 1, 8, 31, tzinfo=UTC)


def test_next_occurrence_is_strictly_after() -> None:
    """An ``after`` landing exactly on an occurrence yields the NEXT one."""
    cron = parse_cron("0 8 * * *")
    exact = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
    assert cron.next_after(exact) == datetime(2026, 1, 2, 8, 0, tzinfo=UTC)


def test_daily_at_eight_next_occurrence() -> None:
    """``0 8 * * *`` fires at 08:00 the same day when queried before it."""
    cron = parse_cron("0 8 * * *")
    after = datetime(2026, 1, 1, 7, 59, tzinfo=UTC)
    assert cron.next_after(after) == datetime(2026, 1, 1, 8, 0, tzinfo=UTC)


def test_daily_rolls_to_next_day() -> None:
    """A daily schedule queried after its time rolls to the next day."""
    cron = parse_cron("0 8 * * *")
    after = datetime(2026, 1, 1, 8, 30, tzinfo=UTC)
    assert cron.next_after(after) == datetime(2026, 1, 2, 8, 0, tzinfo=UTC)


def test_step_and_range_fields() -> None:
    """Steps and ranges parse and constrain occurrences."""
    cron = parse_cron("*/15 9-17 * * *")
    after = datetime(2026, 1, 1, 9, 5, tzinfo=UTC)
    assert cron.next_after(after) == datetime(2026, 1, 1, 9, 15, tzinfo=UTC)


def test_list_field() -> None:
    """A comma list constrains the minute field."""
    cron = parse_cron("0,30 * * * *")
    after = datetime(2026, 1, 1, 9, 5, tzinfo=UTC)
    assert cron.next_after(after) == datetime(2026, 1, 1, 9, 30, tzinfo=UTC)


def test_wildcard_step_one_is_every_unit() -> None:
    """``*/1`` is wildcard-equivalent in every field it appears in.

    ``*/1 * * * */1`` fires every minute on every day-of-week — identical to
    ``* * * * *`` — so the next occurrence is simply the next whole minute.
    """
    cron = parse_cron("*/1 * * * */1")
    after = datetime(2026, 1, 1, 8, 30, 10, tzinfo=UTC)
    assert cron.next_after(after) == datetime(2026, 1, 1, 8, 31, tzinfo=UTC)


def test_day_of_week_restricts() -> None:
    """``0 9 * * 1`` fires only on Monday (cron dow: Mon=1)."""
    cron = parse_cron("0 9 * * 1")
    # 2026-01-01 is a Thursday; next Monday is 2026-01-05.
    after = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
    assert cron.next_after(after) == datetime(2026, 1, 5, 9, 0, tzinfo=UTC)


def test_day_of_week_seven_is_sunday() -> None:
    """``* * * * 7`` accepts 7 as Sunday, matching ``0``."""
    seven = parse_cron("0 9 * * 7")
    zero = parse_cron("0 9 * * 0")
    # 2026-01-01 is a Thursday; next Sunday is 2026-01-04.
    after = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
    assert seven.next_after(after) == datetime(2026, 1, 4, 9, 0, tzinfo=UTC)
    assert seven.next_after(after) == zero.next_after(after)


def test_day_of_week_seven_in_list_range_step() -> None:
    """Sunday matches whether written as 7 in a list, a range, or a step.

    Each expression includes Sunday via a 7 (in a list, a range, or a stepped
    range). Queried from the Sunday itself (2026-01-04 is a Sunday), each must
    fire that Sunday at 00:00 — proving 7 is accepted as Sunday in all three
    forms. (A range like ``5-7`` also matches Fri/Sat, so this queries from the
    Sunday directly to isolate the Sunday-via-7 behavior.)
    """
    saturday = datetime(2026, 1, 3, 12, 0, tzinfo=UTC)  # after Sat noon; next is Sunday
    for expr in ("0 0 * * 1,7", "0 0 * * 5-7", "0 0 * * 1-7/3"):
        occ = parse_cron(expr).next_after(saturday)
        assert occ == datetime(2026, 1, 4, 0, 0, tzinfo=UTC), expr


def test_both_day_fields_restricted_match_or() -> None:
    """When BOTH day fields are restricted, either matching fires (standard OR).

    ``0 0 13 * 5`` (13th OR any Friday). 2026-02-01 is a Sunday, so 2026-02-06 is
    a Friday and must match even though it is not the 13th — the standard cron OR
    rule.
    """
    cron = parse_cron("0 0 13 * 5")
    after = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)
    assert cron.next_after(after) == datetime(2026, 2, 6, 0, 0, tzinfo=UTC)


def test_single_restricted_day_field_unchanged() -> None:
    """A single restricted day field still constrains alone (no OR partner)."""
    dom_only = parse_cron("0 0 13 * *")
    assert dom_only.next_after(datetime(2026, 2, 1, tzinfo=UTC)) == datetime(
        2026, 2, 13, tzinfo=UTC
    )
    dow_only = parse_cron("0 0 * * 1")  # Mondays
    assert dow_only.next_after(datetime(2026, 2, 1, tzinfo=UTC)) == datetime(2026, 2, 2, tzinfo=UTC)


def test_wildcard_step_one_dow_with_restricted_dom_is_dom_only() -> None:
    """``0 0 13 * */1`` is DOM-only: ``*/1`` DOW is unrestricted (OR rule).

    ``*/1`` in the day-of-week field means "every weekday" — wildcard-equivalent,
    so it must NOT count as a restricted OR partner. The expression is therefore
    equivalent to ``0 0 13 * *`` (the 13th only), NOT a daily schedule. (croniter
    treats a raw ``*/1`` day field as restricted, which would fire daily; the
    wrapper normalizes it back to ``*``.)
    """
    normalized = parse_cron("0 0 13 * */1")
    dom_only = parse_cron("0 0 13 * *")
    after = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)
    # 2026-02-13, not 2026-02-02 (which is what a daily/collapsed schedule fires).
    assert normalized.next_after(after) == datetime(2026, 2, 13, 0, 0, tzinfo=UTC)
    assert normalized.next_after(after) == dom_only.next_after(after)


def test_wildcard_step_one_dom_with_restricted_dow_is_dow_only() -> None:
    """``0 0 */1 * 5`` is DOW-only: ``*/1`` DOM is unrestricted (OR rule).

    ``*/1`` in the day-of-month field is wildcard-equivalent, so it must not be a
    restricted OR partner. The expression is equivalent to ``0 0 * * 5`` (Fridays
    only), NOT a daily schedule.
    """
    normalized = parse_cron("0 0 */1 * 5")
    dow_only = parse_cron("0 0 * * 5")
    # 2026-02-01 is a Sunday; the next Friday is 2026-02-06.
    after = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)
    assert normalized.next_after(after) == datetime(2026, 2, 6, 0, 0, tzinfo=UTC)
    assert normalized.next_after(after) == dow_only.next_after(after)


def test_plain_wildcard_day_fields_unaffected_by_normalization() -> None:
    """Plain ``*`` day fields (the control) keep OR-only-partner behavior.

    ``0 0 13 * *`` fires the 13th and ``0 0 * * 5`` fires Fridays — the same
    single-restricted-field results the ``*/1`` forms must match, proving the
    normalization only touches ``*/1`` and leaves plain ``*`` alone.
    """
    after = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)
    assert parse_cron("0 0 13 * *").next_after(after) == datetime(2026, 2, 13, 0, 0, tzinfo=UTC)
    assert parse_cron("0 0 * * 5").next_after(after) == datetime(2026, 2, 6, 0, 0, tzinfo=UTC)


def test_non_one_step_day_fields_stay_restricted() -> None:
    """A non-one step like ``*/2`` in a day field stays restricted (control).

    Only ``*/1`` is wildcard-equivalent. ``0 0 */2 * *`` (odd days of the month)
    must remain a restricted DOM schedule: queried from 2026-02-01 (a matching
    odd day at 00:00 is not strictly after), the next occurrence is 2026-02-03,
    NOT 2026-02-02 — so the step is preserved, not normalized away.
    """
    cron = parse_cron("0 0 */2 * *")
    after = datetime(2026, 2, 1, 0, 0, tzinfo=UTC)
    assert cron.next_after(after) == datetime(2026, 2, 3, 0, 0, tzinfo=UTC)
    # And a ``*/2`` DOW stays a restricted OR partner (Sun/Tue/Thu/Sat).
    dow_step = parse_cron("0 0 13 * */2")
    # 2026-02-01 Sun; ``*/2`` DOW = 0,2,4,6 → next matching weekday is Tue 2026-02-03.
    assert dow_step.next_after(after) == datetime(2026, 2, 3, 0, 0, tzinfo=UTC)


def test_dst_spring_forward_skips_nonexistent_wall_time() -> None:
    """A daily occurrence in the spring-forward gap fires the next real day.

    US DST 2026-03-08: the clock jumps 02:00→03:00, so 02:30 does not exist. A
    ``30 2 * * *`` schedule skips the phantom 2026-03-08 02:30 and fires the next
    real occurrence, 2026-03-09 02:30 local — never a substituted 03:00.
    """
    cron = parse_cron("30 2 * * *", "America/New_York")
    after = datetime(2026, 3, 7, 3, 0, tzinfo=_NY)
    occurrence = cron.next_after(after)
    assert (occurrence.year, occurrence.month, occurrence.day) == (2026, 3, 9)
    assert (occurrence.hour, occurrence.minute) == (2, 30)


def test_dst_spring_forward_every_minute_jumps_gap() -> None:
    """An every-minute schedule crosses the gap from 01:59 straight to 03:00."""
    cron = parse_cron("* * * * *", "America/New_York")
    after = datetime(2026, 3, 8, 1, 59, tzinfo=_NY)
    occurrence = cron.next_after(after)
    assert (occurrence.hour, occurrence.minute) == (3, 0)


def test_dst_fall_back_fires_once_on_first_fold() -> None:
    """A fall-back-hour occurrence fires exactly once, on the earlier instant.

    US DST ends 2026-11-01: the clock falls 02:00→01:00, so 01:30 occurs twice. A
    ``30 1 * * *`` schedule fires it once, at the FIRST (fold=0, EDT) instant —
    the deterministic fall-back rule — never twice.
    """
    cron = parse_cron("30 1 * * *", "America/New_York")
    after = datetime(2026, 11, 1, 0, 0, tzinfo=_NY)
    first = cron.next_after(after)
    assert (first.hour, first.minute) == (1, 30)
    # First instant is fold=0 (EDT, UTC-4): 01:30 EDT == 05:30 UTC, not 06:30.
    assert first.astimezone(UTC) == datetime(2026, 11, 1, 5, 30, tzinfo=UTC)
    # The next occurrence is the following day, never the duplicate fold=1 hour.
    following = cron.next_after(first)
    assert (following.year, following.month, following.day) == (2026, 11, 2)


def test_dst_fall_back_query_during_second_fold_never_returns_earlier() -> None:
    """Queried DURING the ambiguous hour's second fold, the next is strictly later.

    The 01:30 EDT/EST pair is ambiguous. A query pinned to the SECOND fold
    (fold=1, EST, 06:30 UTC) must not return an instant ``<= after`` — the
    duplicate 01:30 is never re-fired; the next occurrence is the following day.
    """
    cron = parse_cron("30 1 * * *", "America/New_York")
    second_fold = datetime(2026, 11, 1, 1, 30, tzinfo=_NY, fold=1)
    occurrence = cron.next_after(second_fold)
    assert occurrence.astimezone(UTC) > second_fold.astimezone(UTC)
    assert (occurrence.year, occurrence.month, occurrence.day) == (2026, 11, 2)


def test_dst_whole_date_gap_apia_skips_nonexistent_date() -> None:
    """A schedule on a SKIPPED CALENDAR DATE fires the next real date, not a phantom.

    Pacific/Apia crossed the international date line at the end of 2011: the clock
    jumped 2011-12-29 24:00 straight to 2011-12-31 00:00, so ALL of 2011-12-30
    never existed. A Dec-30 wall time round-trips to the SAME hour/minute on
    Dec-31, so an hour/minute-only existence check would wrongly fire the phantom
    date. The whole-date round-trip skips 2011-12-30 entirely and fires the next
    real occurrence (2012-12-30, the following year the 30th exists in December).
    """
    apia = ZoneInfo("Pacific/Apia")
    # ``0 9 30 12 *`` — 09:00 on 30 December. 2011-12-30 does not exist in Apia.
    cron = parse_cron("0 9 30 12 *", "Pacific/Apia")
    after = datetime(2011, 12, 29, 0, 0, tzinfo=apia)
    occurrence = cron.next_after(after)
    assert (occurrence.year, occurrence.month, occurrence.day) == (2012, 12, 30)
    assert (occurrence.hour, occurrence.minute) == (9, 0)
    # And the fired instant is a real, round-trip-stable moment in the zone.
    assert occurrence.astimezone(UTC).astimezone(apia) == occurrence


def test_dst_every_minute_jumps_whole_skipped_date() -> None:
    """An every-minute schedule crosses the Apia date-line gap without a phantom.

    The last real minute before the gap is 2011-12-29 23:59; the next real minute
    is 2011-12-31 00:00. No 2011-12-30 wall time is ever returned.
    """
    apia = ZoneInfo("Pacific/Apia")
    cron = parse_cron("* * * * *", "Pacific/Apia")
    after = datetime(2011, 12, 29, 23, 59, tzinfo=apia)
    occurrence = cron.next_after(after)
    assert (occurrence.year, occurrence.month, occurrence.day) == (2011, 12, 31)
    assert (occurrence.hour, occurrence.minute) == (0, 0)


def test_timezone_boundary_new_york() -> None:
    """A schedule in a named tz fires at local wall-clock, mapped to real UTC.

    ``0 8 * * *`` in America/New_York during EST (UTC-5) fires at 13:00 UTC.
    """
    cron = parse_cron("0 8 * * *", "America/New_York")
    after = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    occurrence = cron.next_after(after)
    assert occurrence.tzinfo == _NY
    assert occurrence.astimezone(UTC) == datetime(2026, 1, 1, 13, 0, tzinfo=UTC)


def test_timezone_dst_boundary() -> None:
    """Across a DST change the same wall-clock time maps to a different UTC hour.

    US DST begins 2026-03-08; on 2026-03-09 EDT is UTC-4, so 08:00 local is
    12:00 UTC (vs 13:00 in winter). The wall-clock field is stable; the UTC
    offset moves — the zoneinfo-backed conversion proves it.
    """
    cron = parse_cron("0 8 * * *", "America/New_York")
    after = datetime(2026, 3, 9, 0, 0, tzinfo=UTC)
    occurrence = cron.next_after(after)
    assert occurrence.astimezone(UTC) == datetime(2026, 3, 9, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "expr",
    [
        "* * * *",  # too few fields
        "* * * * * *",  # too many fields (6-field/seconds form rejected)
        "60 * * * *",  # minute out of range
        "* 24 * * *",  # hour out of range
        "* * 0 * *",  # day-of-month out of range
        "* * * 13 *",  # month out of range
        "* * * * 8",  # day-of-week out of range (0-7 allowed; 8 is not)
        "*/0 * * * *",  # zero step
        "5-1 * * * *",  # inverted range (not a wrap-around under the strict contract)
        "x * * * *",  # non-numeric
        "@daily",  # macro form rejected (standard five-field only)
    ],
)
def test_malformed_expressions_raise(expr: str) -> None:
    """Every malformed expression fails closed with ``CronParseError``."""
    with pytest.raises(CronParseError):
        parse_cron(expr)


def test_unknown_timezone_raises() -> None:
    """An unknown timezone fails closed at parse time."""
    with pytest.raises(CronParseError):
        parse_cron("0 8 * * *", "Mars/Olympus_Mons")
