"""Test frozen schedule config validation and the fail-closed JSON loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rtech_schedulerd.config import (
    ENV_SCHEDULE_FILE,
    ENV_STACK_ID,
    ScheduleConfigError,
    SchedulerConfig,
    load_scheduler_config,
    load_schedules,
)


def _single(**over: object) -> dict[str, object]:
    """Return a minimal valid single-scope schedule dict, with overrides."""
    base: dict[str, object] = {
        "schedule_id": "morning-summary",
        "cron": "0 8 * * *",
        "text": "Summarize overnight activity",
    }
    base.update(over)
    return base


def test_load_valid_single_schedule() -> None:
    """A minimal valid table loads with defaults applied."""
    config = load_schedules([_single()], stack_id="dev")
    assert isinstance(config, SchedulerConfig)
    assert len(config.schedules) == 1
    entry = config.schedules[0]
    assert entry.schedule_id == "morning-summary"
    assert entry.enabled is True  # default
    assert entry.scope == "single"
    assert entry.requester == "scheduler"  # default
    assert entry.origin_edge == "scheduler"  # default


def test_disabled_schedule_loads_but_flagged() -> None:
    """An ``enabled: false`` schedule validates and carries the flag."""
    config = load_schedules([_single(enabled=False)], stack_id="dev")
    assert config.schedules[0].enabled is False


def test_duo_scope_requires_two_participants() -> None:
    """A duo schedule with the right participant count validates."""
    config = load_schedules(
        [_single(schedule_id="debate", scope="duo", participants=["agentd-a", "agentd-b"])],
        stack_id="dev",
    )
    assert config.schedules[0].scope == "duo"
    assert config.schedules[0].participants == ("agentd-a", "agentd-b")


def test_scope_participant_mismatch_fails_closed() -> None:
    """A duo scope with one participant fails at config load, not mint time."""
    with pytest.raises(ScheduleConfigError):
        load_schedules(
            [_single(scope="duo", participants=["agentd-only"])],
            stack_id="dev",
        )


def test_bad_cron_fails_closed() -> None:
    """A malformed cron expression fails at config load."""
    with pytest.raises(ScheduleConfigError):
        load_schedules([_single(cron="not a cron")], stack_id="dev")


def test_unknown_timezone_fails_closed() -> None:
    """An unknown timezone fails at config load."""
    with pytest.raises(ScheduleConfigError):
        load_schedules([_single(timezone="Nowhere/Land")], stack_id="dev")


def test_unknown_key_rejected() -> None:
    """An unexpected key is rejected (typo protection, fail closed)."""
    with pytest.raises(ScheduleConfigError):
        load_schedules([_single(taget="agentd-a")], stack_id="dev")  # typo


def test_blank_text_rejected() -> None:
    """Blank task text fails config load."""
    with pytest.raises(ScheduleConfigError):
        load_schedules([_single(text="  ")], stack_id="dev")


def test_unsafe_schedule_id_rejected() -> None:
    """A schedule id with unsafe characters fails config load."""
    with pytest.raises(ScheduleConfigError):
        load_schedules([_single(schedule_id="bad id!")], stack_id="dev")


def test_duplicate_schedule_ids_rejected() -> None:
    """Two schedules sharing an id fail config load (idempotency depends on it)."""
    with pytest.raises(ScheduleConfigError):
        load_schedules([_single(), _single()], stack_id="dev")


def test_blank_stack_id_rejected() -> None:
    """A blank stack id fails closed (issue #75 ingress guard)."""
    with pytest.raises(ScheduleConfigError):
        load_schedules([_single()], stack_id="  ")


def test_non_list_table_rejected() -> None:
    """A non-list schedule table fails closed."""
    with pytest.raises(ScheduleConfigError):
        load_schedules({"schedule_id": "x"}, stack_id="dev")


def test_enabled_non_bool_rejected() -> None:
    """A non-boolean ``enabled`` value fails closed."""
    with pytest.raises(ScheduleConfigError):
        load_schedules([_single(enabled="yes")], stack_id="dev")


def test_load_from_file_missing_env_fails_closed() -> None:
    """An unset schedule-file env fails closed."""
    with pytest.raises(ScheduleConfigError):
        load_scheduler_config(env={ENV_STACK_ID: "dev"})


def test_load_from_file_missing_stack_fails_closed(tmp_path: Path) -> None:
    """An unset stack id fails closed even with a valid file."""
    path = tmp_path / "sched.json"
    path.write_text(json.dumps([_single()]), encoding="utf-8")
    with pytest.raises(ScheduleConfigError):
        load_scheduler_config(env={ENV_SCHEDULE_FILE: str(path)})


def test_load_from_file_roundtrip(tmp_path: Path) -> None:
    """A valid file + env loads through the full resolver."""
    path = tmp_path / "sched.json"
    path.write_text(json.dumps([_single(), _single(schedule_id="second")]), encoding="utf-8")
    config = load_scheduler_config(env={ENV_SCHEDULE_FILE: str(path), ENV_STACK_ID: "dev-stack"})
    assert config.stack_id == "dev-stack"
    assert len(config.schedules) == 2


def test_load_from_missing_file_fails_closed(tmp_path: Path) -> None:
    """A schedule-file path that does not exist fails closed."""
    with pytest.raises(ScheduleConfigError):
        load_scheduler_config(
            env={ENV_SCHEDULE_FILE: str(tmp_path / "nope.json"), ENV_STACK_ID: "dev"}
        )


def test_load_from_invalid_json_fails_closed(tmp_path: Path) -> None:
    """A file with invalid JSON fails closed."""
    path = tmp_path / "sched.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ScheduleConfigError):
        load_scheduler_config(env={ENV_SCHEDULE_FILE: str(path), ENV_STACK_ID: "dev"})


def test_unreadable_file_log_omits_path_and_raw_exception(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The read-failure log carries the exception TYPE only — no raw path or reason.

    A resolved schedule-file path can carry a home dir / user name, and an
    ``OSError`` message echoes that same path. The unreadable-file log line must
    surface neither. Pointing the loader at a DIRECTORY makes ``read_text`` raise
    ``IsADirectoryError`` (an ``OSError``) on a path that exists but cannot be read.
    """
    import logging

    from loguru import logger

    secret_dir = tmp_path / "SECRET_user_home_marker"
    secret_dir.mkdir()

    handler_id = logger.add(caplog.handler, format="{message}", level="DEBUG")
    try:
        with caplog.at_level(logging.DEBUG), pytest.raises(ScheduleConfigError):
            load_scheduler_config(env={ENV_SCHEDULE_FILE: str(secret_dir), ENV_STACK_ID: "dev"})
    finally:
        logger.remove(handler_id)

    logged = caplog.text
    assert "schedule file unreadable" in logged
    # No raw resolved path (its unique marker segment) and no raw OSError text.
    assert "SECRET_user_home_marker" not in logged
    assert "IsADirectoryError" in logged  # the exception TYPE is logged, not the message
    assert str(secret_dir) not in logged


def test_invalid_json_log_omits_path(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The invalid-JSON log line does not echo the resolved schedule-file path."""
    import logging

    from loguru import logger

    path = tmp_path / "SECRET_json_marker.json"
    path.write_text("{not json", encoding="utf-8")

    handler_id = logger.add(caplog.handler, format="{message}", level="DEBUG")
    try:
        with caplog.at_level(logging.DEBUG), pytest.raises(ScheduleConfigError):
            load_scheduler_config(env={ENV_SCHEDULE_FILE: str(path), ENV_STACK_ID: "dev"})
    finally:
        logger.remove(handler_id)

    logged = caplog.text
    assert "invalid JSON" in logged
    assert "SECRET_json_marker" not in logged
    assert str(path) not in logged
