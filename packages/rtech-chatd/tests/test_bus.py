"""Log-hygiene tests for the NatsFrameBus chatd edge boundary.

NATS typed frames are truth; this edge is a dumb translator. Every failure path
on NatsFrameBus (connect, close, publish, subscribe, and the subscribe handler)
must log only the exception *type*, never the rendered exception, so a credential
embedded in a NATS URL or upstream error can never leak into logs. These tests
drive credential-bearing errors through each path and assert the secret never
reaches a log sink while the exception is still re-raised (or contained, for the
handler) exactly as before.

Two complementary batteries live here:

* Type-only + cancellation guards: assert the exact exception object is
  re-raised, that ``error_type=<Name>`` renders, and that CancelledError bypasses
  ``except Exception`` untouched on both connect and close.
* Credentialed-URL leak guards: pin each failure log to a sanitized
  shape, asserting none of the credential substrings (token/url/requester) survive
  in the fully rendered line and the original exception still propagates.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from loguru import logger
from rtech_chatd.bus import NatsFrameBus

CREDENTIAL = "super-secret-token"
LEAKY_MESSAGE = f"nats://user:{CREDENTIAL}@nats.internal:4222 refused"


def _new_bus() -> NatsFrameBus:
    """Build a NatsFrameBus without touching the network (no connect())."""

    return NatsFrameBus("nats://ignored")


class _CapturedLogs:
    """Attach a loguru sink and expose the joined rendered text."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._sink_id = logger.add(lambda message: self._lines.append(str(message)), level="DEBUG")

    def remove(self) -> None:
        logger.remove(self._sink_id)

    @property
    def text(self) -> str:
        return "".join(self._lines)


async def test_connect_failure_logs_type_only_and_reraises() -> None:
    """A connect error carrying a credential must not leak it and must re-raise."""

    bus = _new_bus()
    connect_exc = RuntimeError(LEAKY_MESSAGE)

    async def _raise() -> None:
        raise connect_exc

    bus._bus.connect = _raise  # type: ignore[method-assign]  # noqa: SLF001

    captured = _CapturedLogs()
    try:
        with pytest.raises(RuntimeError) as raised:
            await bus.connect()
    finally:
        captured.remove()

    assert raised.value is connect_exc
    assert CREDENTIAL not in captured.text
    assert "error_type=RuntimeError" in captured.text


async def test_close_failure_logs_type_only_and_reraises() -> None:
    """close() must type-log before any caller can sanitize, and still re-raise."""

    bus = _new_bus()
    close_exc = RuntimeError(LEAKY_MESSAGE)

    async def _raise() -> None:
        raise close_exc

    bus._bus.close = _raise  # type: ignore[method-assign]  # noqa: SLF001

    captured = _CapturedLogs()
    try:
        with pytest.raises(RuntimeError) as raised:
            await bus.close()
    finally:
        captured.remove()

    assert raised.value is close_exc
    assert CREDENTIAL not in captured.text
    assert "error_type=RuntimeError" in captured.text


async def test_publish_failure_logs_type_only_and_reraises() -> None:
    """A publish error carrying a credential must not leak it and must re-raise."""

    bus = _new_bus()
    publish_exc = RuntimeError(LEAKY_MESSAGE)

    def _guard(_subject: str, _data: bytes) -> None:
        raise publish_exc

    bus._bus._guard_payload_size = _guard  # type: ignore[method-assign]  # noqa: SLF001

    captured = _CapturedLogs()
    try:
        with pytest.raises(RuntimeError) as raised:
            await bus.publish("chatd.subject", b"payload")
    finally:
        captured.remove()

    assert raised.value is publish_exc
    assert CREDENTIAL not in captured.text
    assert "error_type=RuntimeError" in captured.text


async def test_subscribe_failure_logs_type_only_and_reraises() -> None:
    """A subscribe error carrying a credential must not leak it and must re-raise."""

    bus = _new_bus()
    subscribe_exc = RuntimeError(LEAKY_MESSAGE)

    async def _raise(_subject: str, _cb: Any) -> None:
        raise subscribe_exc

    bus._bus.raw_subscribe = _raise  # type: ignore[method-assign]  # noqa: SLF001

    async def _handler(_data: bytes) -> None:
        return None

    captured = _CapturedLogs()
    try:
        with pytest.raises(RuntimeError) as raised:
            await bus.subscribe("chatd.subject", _handler)
    finally:
        captured.remove()

    assert raised.value is subscribe_exc
    assert CREDENTIAL not in captured.text
    assert "error_type=RuntimeError" in captured.text


async def test_subscribe_handler_failure_is_contained_and_type_only() -> None:
    """A raising handler must be swallowed (contained) and logged type-only."""

    bus = _new_bus()
    handler_exc = RuntimeError(LEAKY_MESSAGE)
    captured_cb: dict[str, Any] = {}

    async def _capture(subject: str, cb: Any) -> object:
        captured_cb["cb"] = cb
        return object()

    bus._bus.raw_subscribe = _capture  # type: ignore[method-assign]  # noqa: SLF001

    async def _handler(_data: bytes) -> None:
        raise handler_exc

    captured = _CapturedLogs()
    try:
        await bus.subscribe("chatd.subject", _handler)

        class _Msg:
            data = b"payload"

        # Handler failure is contained: driving a message must not raise.
        await captured_cb["cb"](_Msg())
    finally:
        captured.remove()

    assert CREDENTIAL not in captured.text
    assert "handler failed" in captured.text
    assert "error_type=RuntimeError" in captured.text


async def test_connect_cancelled_then_close_error_reraises_cancellation() -> None:
    """P3: connect raises one CancelledError; close raises a leaky error.

    NatsFrameBus.connect catches only Exception, so the CancelledError propagates
    untouched — the exact object is re-raised and connect never logs it. A caller
    that then invokes close() sees close's own error re-raised via close()'s
    type-only path, and the close error text never reaches a log sink.
    """

    bus = _new_bus()
    connect_cancel = asyncio.CancelledError()
    close_exc = RuntimeError(LEAKY_MESSAGE)

    async def _cancel() -> None:
        raise connect_cancel

    async def _close_raise() -> None:
        raise close_exc

    bus._bus.connect = _cancel  # type: ignore[method-assign]  # noqa: SLF001
    bus._bus.close = _close_raise  # type: ignore[method-assign]  # noqa: SLF001

    captured = _CapturedLogs()
    try:
        with pytest.raises(asyncio.CancelledError) as connect_raised:
            await bus.connect()
        with pytest.raises(RuntimeError) as close_raised:
            await bus.close()
    finally:
        captured.remove()

    # The original cancellation object is re-raised untouched.
    assert connect_raised.value is connect_cancel
    assert close_raised.value is close_exc
    # Neither the cancellation nor the leaky close text reaches the logs.
    assert CREDENTIAL not in captured.text
    assert "error_type=RuntimeError" in captured.text


async def test_connect_cancelled_and_close_also_cancelled_reraise_both() -> None:
    """P3 variant: connect and close both raise distinct CancelledError objects.

    connect's CancelledError propagates untouched (not caught by except Exception);
    close's CancelledError likewise propagates untouched and is never logged. Each
    call re-raises its own exact object with no leaked text.
    """

    bus = _new_bus()
    connect_cancel = asyncio.CancelledError()
    close_cancel = asyncio.CancelledError("private close cancellation")

    async def _connect_cancel() -> None:
        raise connect_cancel

    async def _close_cancel() -> None:
        raise close_cancel

    bus._bus.connect = _connect_cancel  # type: ignore[method-assign]  # noqa: SLF001
    bus._bus.close = _close_cancel  # type: ignore[method-assign]  # noqa: SLF001

    captured = _CapturedLogs()
    try:
        with pytest.raises(asyncio.CancelledError) as connect_raised:
            await bus.connect()
        with pytest.raises(asyncio.CancelledError) as close_raised:
            await bus.close()
    finally:
        captured.remove()

    assert connect_raised.value is connect_cancel
    assert close_raised.value is close_cancel
    # CancelledError bypasses except Exception, so close text is never logged.
    assert "private close cancellation" not in captured.text


# --------------------------------------------------------------------------
# Credentialed-URL leak guards. These drive a sentinel URL bearing
# token/requester substrings through each failure path and assert the FULLY
# RENDERED log line never carries any credential substring, while the original
# exception still propagates (or, for the inbound handler, is swallowed).
# --------------------------------------------------------------------------


class _BoomNats:
    """Stand in for the underlying nats client and raise on publish."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def publish(self, subject: str, data: bytes) -> None:
        raise self._exc


def _bus_that_booms(exc: Exception) -> NatsFrameBus:
    """Build a NatsFrameBus whose publish path raises `exc`.

    The adapter builds a real FleetBus without connecting; we replace its
    private client and no-op the size guard so the raise comes from publish.
    """
    bus = NatsFrameBus("nats://placeholder")
    bus._bus._nc = _BoomNats(exc)  # type: ignore[attr-defined]  # noqa: SLF001
    bus._bus._guard_payload_size = lambda subject, data: None  # type: ignore[attr-defined]  # noqa: SLF001
    return bus


def _capture_rendered() -> tuple[list[str], int]:
    """Attach a sink capturing fully RENDERED log lines; return (lines, id).

    `record["message"]` is the un-substituted template, so it would hide a
    leak that lives in the rendered arguments. `str(message)` is the formatted
    line with `{}` already filled in, which is what a real log destination sees.
    """
    lines: list[str] = []
    sink_id = logger.add(lambda m: lines.append(str(m)), level="ERROR")
    return lines, sink_id


def test_publish_failure_log_omits_credentialed_url_and_still_raises() -> None:
    # A callback publish failure can carry the credentialed NATS URL in its
    # message text. The adapter must log only the exception TYPE plus safe
    # subject context, never the URL/token/requester, and must still propagate.
    secret = "nats://host?token=SUPER_SECRET requester=123"
    exc = RuntimeError(secret)
    bus = _bus_that_booms(exc)

    lines, sink_id = _capture_rendered()
    try:
        with pytest.raises(RuntimeError):
            asyncio.run(bus.publish("rtech.dev.task.abc", b"frame-bytes"))
    finally:
        logger.remove(sink_id)

    blob = "\n".join(lines)
    # The failure was logged at least once on the publish path.
    assert blob, "expected an ERROR log line from the publish failure"
    # None of the credential-bearing substrings may appear anywhere.
    assert "SUPER_SECRET" not in blob
    assert "token=" not in blob
    assert "nats://host" not in blob
    assert "requester=123" not in blob
    # The safe shape survives: exception type name and the subject context.
    assert "RuntimeError" in blob
    assert "rtech.dev.task.abc" in blob


def test_publish_failure_reraises_original_exception() -> None:
    # Sanitizing the log must not swallow or wrap the error: the original
    # exception instance propagates so callers still fail closed.
    exc = RuntimeError("nats://host?token=SUPER_SECRET requester=123")
    bus = _bus_that_booms(exc)

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(bus.publish("rtech.dev.task.abc", b"frame-bytes"))

    assert caught.value is exc


# A single sentinel whose substrings must never appear in any rendered log line.
_SECRET_URL = "nats://host?token=SUPER_SECRET requester=123"


def _assert_no_leak(lines: list[str]) -> str:
    """Assert a log line was emitted and no credential substring survived."""
    blob = "\n".join(lines)
    assert blob, "expected an ERROR log line from the failure"
    assert "SUPER_SECRET" not in blob
    assert "token=" not in blob
    assert "nats://host" not in blob
    assert "requester=123" not in blob
    return blob


def test_connect_failure_log_omits_credentialed_url_and_still_raises() -> None:
    # A NATS connect failure can carry the credentialed URL in its message text.
    exc = RuntimeError(_SECRET_URL)
    bus = NatsFrameBus("nats://placeholder")

    async def _boom() -> None:
        raise exc

    bus._bus.connect = _boom  # type: ignore[attr-defined,method-assign]  # noqa: SLF001

    lines, sink_id = _capture_rendered()
    try:
        with pytest.raises(RuntimeError) as caught:
            asyncio.run(bus.connect())
    finally:
        logger.remove(sink_id)

    blob = _assert_no_leak(lines)
    assert "RuntimeError" in blob
    assert caught.value is exc


def test_close_failure_log_omits_credentialed_url_and_still_raises() -> None:
    exc = RuntimeError(_SECRET_URL)
    bus = NatsFrameBus("nats://placeholder")

    async def _boom() -> None:
        raise exc

    bus._bus.close = _boom  # type: ignore[attr-defined,method-assign]  # noqa: SLF001

    lines, sink_id = _capture_rendered()
    try:
        with pytest.raises(RuntimeError) as caught:
            asyncio.run(bus.close())
    finally:
        logger.remove(sink_id)

    blob = _assert_no_leak(lines)
    assert "RuntimeError" in blob
    assert caught.value is exc


def test_subscribe_failure_log_omits_credentialed_url_and_still_raises() -> None:
    # A raw_subscribe failure can surface the credentialed URL; only the type and
    # the safe subject may render.
    exc = RuntimeError(_SECRET_URL)
    bus = NatsFrameBus("nats://placeholder")

    async def _boom(subject: str, cb: object) -> None:
        raise exc

    bus._bus.raw_subscribe = _boom  # type: ignore[attr-defined,method-assign]  # noqa: SLF001

    async def _handler(_: bytes) -> None:
        return None

    lines, sink_id = _capture_rendered()
    try:
        with pytest.raises(RuntimeError) as caught:
            asyncio.run(bus.subscribe("rtech.dev.task.abc", _handler))
    finally:
        logger.remove(sink_id)

    blob = _assert_no_leak(lines)
    assert "RuntimeError" in blob
    assert "rtech.dev.task.abc" in blob
    assert caught.value is exc


def test_inbound_handler_failure_log_omits_credentialed_url_and_swallows() -> None:
    # An inbound callback (handler) failure is caught and logged, never re-raised
    # (a bad frame must not tear the subscription down). Its message text can carry
    # the credentialed URL, so only the type and safe subject may render.
    captured_cb: list[object] = []

    class _FakeMsg:
        def __init__(self, data: bytes) -> None:
            self.data = data

    async def _capture_subscribe(subject: str, cb: object) -> object:
        captured_cb.append(cb)
        return object()

    bus = NatsFrameBus("nats://placeholder")
    bus._bus.raw_subscribe = _capture_subscribe  # type: ignore[attr-defined,method-assign]  # noqa: SLF001

    async def _handler(_: bytes) -> None:
        raise RuntimeError(_SECRET_URL)

    lines, sink_id = _capture_rendered()
    try:

        async def _drive() -> None:
            await bus.subscribe("rtech.dev.task.abc", _handler)
            # Invoke the wrapped inbound callback exactly as NATS would.
            on_message = captured_cb[0]
            await on_message(_FakeMsg(b"frame-bytes"))  # type: ignore[operator]

        # The handler failure is swallowed (no raise escapes the callback).
        asyncio.run(_drive())
    finally:
        logger.remove(sink_id)

    blob = _assert_no_leak(lines)
    assert "RuntimeError" in blob
    assert "rtech.dev.task.abc" in blob
