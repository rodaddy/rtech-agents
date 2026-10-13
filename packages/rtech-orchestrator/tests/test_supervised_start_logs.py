"""Assert supervised-Start logs never interpolate a raw request_id (#146).

Terminal-audit finding: logs are a possibly-leaked surface, so the caller-supplied
``request_id`` must never ride a log line raw. Every log site instead emits a
NON-REVERSIBLE deterministic ``request_token`` derived purely from the bounded id
(:func:`rtech_orchestrator.supervised_start._request_token`). These tests drive the
normal, replay, and publish-failure log paths with a distinctive PRIVATE MARKER as
the request_id and assert the marker NEVER appears in any captured log record,
while the bounded token AND the exception type/reason remain for triage.

The shared doubles come from the ``supervised_start_kit`` fixture in ``conftest``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from loguru import logger
from rtech_orchestrator.supervised_start import _request_token

from .conftest import SupervisedStartKit

# A distinctive, in-bounds (<= MAX_REQUEST_ID_LEN) request_id that would be trivial
# to spot if it ever leaked into a log line. It stands in for any caller-controlled
# idempotency key that must never ride the log surface raw.
_PRIVATE_MARKER = "PRIVATE-MARKER-request-id-do-not-leak-0001"


@pytest.fixture
def captured_logs() -> Iterator[list[str]]:
    """Capture every supervised-start log record's fully-formatted text.

    Formats each record with ``{message}`` exactly as the app sink would, so a
    raw request_id interpolated into a ``{}`` placeholder would appear here if it
    leaked.
    """
    records: list[str] = []
    sink_id = logger.add(
        lambda message: records.append(str(message)),
        level="DEBUG",
        format="{message}",
    )
    try:
        yield records
    finally:
        logger.remove(sink_id)


def _assert_marker_absent(records: list[str]) -> None:
    """The raw private-marker request_id must appear in NO captured log line."""
    for line in records:
        assert _PRIVATE_MARKER not in line, f"raw request_id leaked into log: {line!r}"


@pytest.mark.asyncio
async def test_online_log_uses_token_never_raw_request_id(
    supervised_start_kit: SupervisedStartKit,
    captured_logs: list[str],
) -> None:
    """The happy-path Online log emits the bounded token, never the raw id."""
    kit = supervised_start_kit
    supervisor, _starter, _publisher = kit.supervisor(
        authorizer=kit.Authorizer(), guard=kit.Guard()
    )

    result = await supervisor.start_agent(kit.request(request_id=_PRIVATE_MARKER))

    assert result.state == "Online"
    _assert_marker_absent(captured_logs)
    # The non-reversible correlation token IS present for triage.
    token = _request_token(_PRIVATE_MARKER)
    online_logs = [line for line in captured_logs if "supervised_start.start | online" in line]
    assert online_logs
    assert all(token in line for line in online_logs)


@pytest.mark.asyncio
async def test_idempotent_replay_log_uses_token_never_raw_request_id(
    supervised_start_kit: SupervisedStartKit,
    captured_logs: list[str],
) -> None:
    """A sequential redelivery logs the replay with the token, never the raw id."""
    kit = supervised_start_kit
    supervisor, _starter, _publisher = kit.supervisor(
        authorizer=kit.Authorizer(), guard=kit.Guard()
    )

    first = await supervisor.start_agent(kit.request(request_id=_PRIVATE_MARKER))
    replay = await supervisor.start_agent(kit.request(request_id=_PRIVATE_MARKER))

    assert first == replay
    _assert_marker_absent(captured_logs)
    token = _request_token(_PRIVATE_MARKER)
    replay_logs = [line for line in captured_logs if "supervised_start.idempotent | replay" in line]
    assert replay_logs
    assert all(token in line for line in replay_logs)


@pytest.mark.asyncio
async def test_publish_failure_log_keeps_token_and_error_type_never_raw_id(
    supervised_start_kit: SupervisedStartKit,
    captured_logs: list[str],
) -> None:
    """A publish-failure path keeps the token AND the exception type — never the raw id.

    The Starting publish is forced to fail, driving the ``starting failed`` log path
    (raw-id log site under audit). Triage must still see the bounded token and the
    exception TYPE (``error_type=RuntimeError``); the caller-supplied id must not.
    """
    kit = supervised_start_kit
    publisher = kit.Publisher(fail_states=("Starting",))
    supervisor, _starter, _publisher = kit.supervisor(
        authorizer=kit.Authorizer(), guard=kit.Guard(), publisher=publisher
    )

    result = await supervisor.start_agent(kit.request(request_id=_PRIVATE_MARKER))

    assert result.state == "Failed"
    assert result.failure == "state_publish_failed"
    _assert_marker_absent(captured_logs)
    token = _request_token(_PRIVATE_MARKER)
    failure_logs = [
        line for line in captured_logs if "supervised_start.publish | starting failed" in line
    ]
    assert failure_logs
    # Both the correlation token AND the exception type survive for triage.
    assert all(token in line for line in failure_logs)
    assert all("RuntimeError" in line for line in failure_logs)


def test_request_token_is_deterministic_short_and_non_reversible() -> None:
    """The token is a stable, short, digest-only derivation of the bounded id.

    Deterministic (same id -> same token for cross-line correlation), short
    (bounded width), and derived ONLY from the id via SHA-256 (non-reversible,
    never containing the caller text itself).
    """
    token = _request_token(_PRIVATE_MARKER)

    assert token == _request_token(_PRIVATE_MARKER)
    assert _PRIVATE_MARKER not in token
    assert len(token) < len(_PRIVATE_MARKER)
    assert all(char in "0123456789abcdef" for char in token)
    # A different id yields a different token (correlation is per-id).
    assert token != _request_token(_PRIVATE_MARKER + "x")
