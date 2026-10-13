"""Cover the shared TypedFrameBus typed transport errors + confirmation seam."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import TYPE_CHECKING, cast

import pytest
from loguru import logger
from rtech_frames import TaskFrame
from rtech_frames.codec import encode_frame
from rtech_orchestrator.transport import (
    FrameConfirmError,
    FramePublishError,
    FrameSubscribeError,
)
from rtech_orchestrator.typedbus import TypedFrameBus, _idempotent_unsubscribe

if TYPE_CHECKING:
    from fleet_nats import FleetBus
    from nats.aio.msg import Msg

# A credential-bearing token planted in every failing transport/decode/handler
# exception message: it must never reach a log line or a raised typed-error
# string, only the exception type name may.
_SECRET = "nats://user:hunter2-SECRET@nats.internal:4222"


class _ExplodingNc:
    """Raw NATS client stub whose publish/subscribe/flush always fail."""

    async def publish(self, subject: str, data: bytes) -> None:
        raise ConnectionError("nats connection lost")

    async def subscribe(self, subject: str, cb: object) -> None:
        raise ConnectionError("nats connection lost")

    async def flush(self, timeout: int = 10) -> None:
        raise TimeoutError("flush timed out")


class _BufferingNc:
    """Raw NATS client stub modeling core-NATS enqueue-only publish.

    ``publish`` always succeeds (the client buffers, e.g. mid-reconnect);
    ``flush`` outcome is configurable, and the call log records ordering.
    """

    def __init__(self, *, flush_ok: bool) -> None:
        self.flush_ok = flush_ok
        self.calls: list[str] = []
        self.flush_timeouts: list[int] = []

    async def publish(self, subject: str, data: bytes) -> None:
        self.calls.append(f"publish:{subject}")

    async def flush(self, timeout: int = 10) -> None:
        self.calls.append("flush")
        self.flush_timeouts.append(timeout)
        if not self.flush_ok:
            raise TimeoutError("flush timed out")


class _StubFleetBus:
    """FleetBus stub exposing the private seams TypedFrameBus touches."""

    def __init__(self, nc: object) -> None:
        self._nc = nc

    def _guard_payload_size(self, subject: str, data: bytes) -> None:
        return None


def _bus(nc: object) -> TypedFrameBus:
    return TypedFrameBus(cast("FleetBus", _StubFleetBus(nc)))


def _task() -> TaskFrame:
    return TaskFrame(
        id="t1",
        origin_edge="web",
        requester="rico",
        text="do the thing",
        scope="single",
        stack="dev",
    )


async def test_publish_failure_raises_typed_error() -> None:
    # A swallowed publish failure is how a never-dispatched agent got recorded
    # as dispatched (issue #75 round 2): the failure must surface typed.
    bus = _bus(_ExplodingNc())
    with pytest.raises(FramePublishError, match="rtech.agent.a1.task"):
        await bus.publish_frame("rtech.agent.a1.task", _task())


async def test_subscribe_failure_raises_typed_error() -> None:
    async def handler(frame: object) -> None:
        return None

    bus = _bus(_ExplodingNc())
    with pytest.raises(FrameSubscribeError, match="rtech.task.t1.turn"):
        await bus.subscribe_frame("rtech.task.t1.turn", handler)


async def test_enqueue_success_then_flush_timeout_raises_confirm_error() -> None:
    # Round-3 H1: core NATS publish only enqueues; a publish that "succeeds"
    # while the client is reconnect-buffering is NOT a hand-off. The flush
    # round-trip is the confirmation boundary and its timeout must surface as
    # the typed confirm error.
    nc = _BufferingNc(flush_ok=False)
    bus = _bus(nc)
    await bus.publish_frame("rtech.agent.a1.task", _task())
    with pytest.raises(FrameConfirmError, match="flush"):
        await bus.confirm_publishes()
    assert nc.calls == ["publish:rtech.agent.a1.task", "flush"]


async def test_confirm_publishes_flushes_with_timeout() -> None:
    nc = _BufferingNc(flush_ok=True)
    bus = TypedFrameBus(cast("FleetBus", _StubFleetBus(nc)), confirm_timeout_s=3)
    await bus.publish_frame("rtech.agent.a1.task", _task())
    await bus.confirm_publishes()
    assert nc.calls == ["publish:rtech.agent.a1.task", "flush"]
    assert nc.flush_timeouts == [3]


async def test_idempotent_unsubscribe_serializes_concurrent_callers() -> None:
    """Two callers share one raw unsubscribe; both return successfully."""

    class _RejectSecondUnsubscribe:
        def __init__(self) -> None:
            self.calls = 0

        async def unsubscribe(self) -> None:
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("raw subscription already retired")
            await asyncio.sleep(0)

    raw = _RejectSecondUnsubscribe()
    unsubscribe = _idempotent_unsubscribe("rtech.task.t1.turn", raw)
    await asyncio.gather(unsubscribe(), unsubscribe())
    assert raw.calls == 1


async def test_idempotent_unsubscribe_retries_after_raw_failure() -> None:
    """A failed first raw retire leaves the handle available for a later retry."""

    class _FailOnceUnsubscribe:
        def __init__(self) -> None:
            self.calls = 0

        async def unsubscribe(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient raw unsubscribe failure")

    raw = _FailOnceUnsubscribe()
    unsubscribe = _idempotent_unsubscribe("rtech.task.t1.turn", raw)
    with pytest.raises(RuntimeError, match="transient"):
        await unsubscribe()
    await unsubscribe()
    assert raw.calls == 2


@pytest.fixture
def captured_logs() -> Iterator[list[str]]:
    """Capture every typedbus log record's fully-formatted text.

    Formats the record the same way the app sink would so an interpolated
    secret in ``{}`` placeholders would show up here if it leaked.
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


class _SecretLeakingNc:
    """Raw NATS client whose publish/subscribe/flush raise secret-bearing errors."""

    async def publish(self, subject: str, data: bytes) -> None:
        raise ConnectionError(f"connect to {_SECRET} refused")

    async def subscribe(self, subject: str, cb: object) -> None:
        raise ConnectionError(f"connect to {_SECRET} refused")

    async def flush(self, timeout: int = 10) -> None:
        raise TimeoutError(f"flush to {_SECRET} timed out")


class _DeliveringNc:
    """Raw NATS client that captures the subscribe callback and delivers to it."""

    def __init__(self) -> None:
        self.cb: object = None

    async def subscribe(self, subject: str, cb: object) -> object:
        self.cb = cb
        return object()


def _assert_secret_absent(logs: list[str]) -> None:
    for line in logs:
        assert _SECRET not in line, f"secret leaked into log line: {line!r}"
        assert "hunter2" not in line, f"secret fragment leaked into log line: {line!r}"


async def test_publish_failure_scrubs_secret_from_log_and_error(
    captured_logs: list[str],
) -> None:
    bus = _bus(_SecretLeakingNc())
    with pytest.raises(FramePublishError) as excinfo:
        await bus.publish_frame("rtech.agent.a1.task", _task())
    assert _SECRET not in str(excinfo.value)
    assert "hunter2" not in str(excinfo.value)
    assert "ConnectionError" in str(excinfo.value)
    # Chained cause preserved for operator diagnosis.
    assert isinstance(excinfo.value.__cause__, ConnectionError)
    assert _SECRET in str(excinfo.value.__cause__)
    _assert_secret_absent(captured_logs)


async def test_publish_checked_failure_scrubs_secret_from_log_and_error(
    captured_logs: list[str],
) -> None:
    bus = _bus(_SecretLeakingNc())
    with pytest.raises(FramePublishError) as excinfo:
        await bus.publish_frame_checked("rtech.agent.a1.task", _task())
    assert _SECRET not in str(excinfo.value)
    assert "hunter2" not in str(excinfo.value)
    assert "ConnectionError" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ConnectionError)
    _assert_secret_absent(captured_logs)


async def test_confirm_failure_scrubs_secret_from_log_and_error(
    captured_logs: list[str],
) -> None:
    bus = _bus(_SecretLeakingNc())
    with pytest.raises(FrameConfirmError) as excinfo:
        await bus.confirm_publishes()
    assert _SECRET not in str(excinfo.value)
    assert "hunter2" not in str(excinfo.value)
    assert "TimeoutError" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, TimeoutError)
    assert _SECRET in str(excinfo.value.__cause__)
    _assert_secret_absent(captured_logs)


async def test_subscribe_failure_scrubs_secret_from_log_and_error(
    captured_logs: list[str],
) -> None:
    async def handler(frame: object) -> None:
        return None

    bus = _bus(_SecretLeakingNc())
    with pytest.raises(FrameSubscribeError) as excinfo:
        await bus.subscribe_frame("rtech.task.t1.turn", handler)
    assert _SECRET not in str(excinfo.value)
    assert "hunter2" not in str(excinfo.value)
    assert "ConnectionError" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ConnectionError)
    assert _SECRET in str(excinfo.value.__cause__)
    _assert_secret_absent(captured_logs)


async def test_decode_failure_is_contained_and_scrubs_secret_from_log(
    captured_logs: list[str],
) -> None:
    # A bad frame must not kill the subscription; the decode failure is logged
    # and swallowed, and a secret in the decode exception must not reach the log.
    async def handler(frame: object) -> None:
        raise AssertionError("handler must not run for an undecodable frame")

    nc = _DeliveringNc()
    bus = _bus(nc)
    await bus.subscribe_frame("rtech.task.t1.turn", handler)
    assert nc.cb is not None
    on_message = cast("object", nc.cb)

    class _BadMsg:
        # Non-decodable bytes; decode_frame will raise. The secret rides in via
        # the msg content, but the log records only the exception type.
        data = _SECRET.encode()

    # Must not raise: decode failure is contained inside the delivery callback.
    await on_message(cast("Msg", _BadMsg()))  # type: ignore[operator]
    assert any("typedbus.decode" in line for line in captured_logs)
    _assert_secret_absent(captured_logs)


async def test_handler_failure_is_contained_and_scrubs_secret_from_log(
    captured_logs: list[str],
) -> None:
    # A handler exception must not kill the subscription; it is logged and
    # swallowed, and a secret in the handler exception must not reach the log.
    async def handler(frame: object) -> None:
        raise ConnectionError(f"handler failed talking to {_SECRET}")

    nc = _DeliveringNc()
    bus = _bus(nc)
    await bus.subscribe_frame("rtech.task.t1.turn", handler)
    assert nc.cb is not None
    on_message = cast("object", nc.cb)

    class _GoodMsg:
        data = encode_frame(_task())

    # Must not raise: handler failure is contained inside the delivery callback.
    await on_message(cast("Msg", _GoodMsg()))  # type: ignore[operator]
    assert any("typedbus.handler" in line for line in captured_logs)
    _assert_secret_absent(captured_logs)
