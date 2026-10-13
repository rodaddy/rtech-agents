"""Cover the TaskFrame scope/participant wire invariant."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from rtech_frames.frames import TaskFrame, validate_participants


def _kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "task-1",
        "origin_edge": "discord",
        "requester": "rico",
        "text": "Route this task.",
        "scope": "single",
        "participants": [],
        "stack": "dev",
    }
    base.update(overrides)
    return base


def test_valid_shapes_accepted() -> None:
    TaskFrame(**_kwargs())  # single, default routing
    TaskFrame(**_kwargs(participants=["agentd-bilby"]))
    TaskFrame(**_kwargs(scope="duo", participants=["codex-a", "opus-b"]))
    TaskFrame(**_kwargs(scope="trio", participants=["codex-a", "opus-b", "opus_c"]))


def test_duplicate_participants_rejected() -> None:
    # ['a', 'a'] would derive duo, publish twice to one lane, and collapse to a
    # one-member dispatch set that can never satisfy duo completion.
    with pytest.raises(ValidationError, match="unique"):
        TaskFrame(**_kwargs(scope="duo", participants=["a", "a"]))


@pytest.mark.parametrize("bad_id", ["", " ", "a b", "a.b", "a*", "a>", "a\n"])
def test_subject_unsafe_participant_rejected(bad_id: str) -> None:
    with pytest.raises(ValidationError, match="NATS subject token"):
        TaskFrame(**_kwargs(participants=[bad_id]))


@pytest.mark.parametrize(
    ("scope", "participants"),
    [
        ("single", ["a", "b"]),
        ("duo", ["a"]),
        ("duo", ["a", "b", "c"]),
        ("trio", ["a", "b"]),
    ],
)
def test_scope_count_mismatch_rejected(scope: str, participants: list[str]) -> None:
    with pytest.raises(ValidationError, match="requires exactly"):
        TaskFrame(**_kwargs(scope=scope, participants=participants))


@pytest.mark.parametrize("scope", ["duo", "trio"])
def test_identityless_multi_scope_rejected(scope: str) -> None:
    # A duo/trio with no named participants can never derive done.
    with pytest.raises(ValidationError, match="named participants"):
        TaskFrame(**_kwargs(scope=scope, participants=[]))


def test_validate_participants_rejects_unknown_scope() -> None:
    with pytest.raises(ValueError, match="unknown scope"):
        validate_participants(["a"], "quartet")
