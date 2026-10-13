"""Test the atomic all-or-nothing subscription-set primitive."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest
import rtech_frames
from loguru import logger
from rtech_frames.subs import (
    FrameSubscribeError,
    Subscription,
    SubscriptionSet,
    subscribe_all,
)


@dataclass
class FakeSub:
    """Fake bus subscription tracking unsubscribe calls.

    ``unsub_raises`` makes unsubscribe fail permanently. ``unsub_raise_times``
    makes it fail transiently: it raises while ``unsub_calls <= unsub_raise_times``
    and succeeds thereafter, so a retried retire can observe recovery.
    """

    subject: str
    unsub_calls: int = 0
    unsub_raises: bool = False
    unsub_raise_times: int = 0

    async def unsubscribe(self) -> None:
        self.unsub_calls += 1
        if self.unsub_raises or self.unsub_calls <= self.unsub_raise_times:
            raise RuntimeError(f"unsubscribe boom: {self.subject}")


@dataclass
class FakeBus:
    """Fake bus whose subscribe fails on a designated subject."""

    fail_subject: str | None = None
    unsub_raises_subjects: frozenset[str] = frozenset()
    unsub_transient_subjects: frozenset[str] = frozenset()
    subscribe_calls: list[str] = field(default_factory=list)
    made: list[FakeSub] = field(default_factory=list)

    async def subscribe(self, subject: str, handler: object) -> Subscription:
        self.subscribe_calls.append(subject)
        if subject == self.fail_subject:
            raise RuntimeError(f"subscribe boom: {subject}")
        sub = FakeSub(
            subject=subject,
            unsub_raises=subject in self.unsub_raises_subjects,
            unsub_raise_times=1 if subject in self.unsub_transient_subjects else 0,
        )
        self.made.append(sub)
        return sub


@dataclass
class BlockingSub:
    """Block unsubscribe until released, optionally failing every attempt."""

    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    unsub_calls: int = 0
    fail: bool = False

    async def unsubscribe(self) -> None:
        self.unsub_calls += 1
        self.started.set()
        await self.release.wait()
        if self.fail:
            raise RuntimeError("unsubscribe boom")


def _entries() -> list[tuple[str, object]]:
    return [("a", object()), ("b", object()), ("c", object())]


async def test_subscribe_all_success_subscribes_all_in_order() -> None:
    bus = FakeBus()
    result = await subscribe_all(bus.subscribe, _entries())

    assert isinstance(result, SubscriptionSet)
    assert len(result) == 3
    assert bus.subscribe_calls == ["a", "b", "c"]
    assert all(sub.unsub_calls == 0 for sub in bus.made)
    assert result.retired is False


async def test_second_of_three_fails_rolls_back_first_only() -> None:
    bus = FakeBus(fail_subject="b")

    with pytest.raises(FrameSubscribeError) as excinfo:
        await subscribe_all(bus.subscribe, _entries())

    assert "b" in str(excinfo.value)
    # only "a" was successfully subscribed, then rolled back; "c" never attempted
    assert bus.subscribe_calls == ["a", "b"]
    assert len(bus.made) == 1
    assert bus.made[0].subject == "a"
    assert bus.made[0].unsub_calls == 1


async def test_third_of_three_fails_rolls_back_first_two() -> None:
    bus = FakeBus(fail_subject="c")

    with pytest.raises(FrameSubscribeError) as excinfo:
        await subscribe_all(bus.subscribe, _entries())

    assert "c" in str(excinfo.value)
    assert bus.subscribe_calls == ["a", "b", "c"]
    assert [s.subject for s in bus.made] == ["a", "b"]
    assert all(s.unsub_calls == 1 for s in bus.made)


async def test_rollback_continues_past_failing_unsubscribe() -> None:
    # Second subscribe fails; rolling back "a" fails TRANSIENTLY. The still-live
    # "a" subscription is not lost: it rides on FrameSubscribeError.leaked.
    bus = FakeBus(fail_subject="b", unsub_transient_subjects=frozenset({"a"}))

    with pytest.raises(FrameSubscribeError) as excinfo:
        await subscribe_all(bus.subscribe, _entries())

    assert len(bus.made) == 1
    assert bus.made[0].subject == "a"
    assert bus.made[0].unsub_calls == 1  # attempted despite raising

    leaked = excinfo.value.leaked
    assert len(leaked) == 1
    assert leaked.subjects == ("a",)
    assert leaked.retired is False

    # The fake now stops raising: retiring the leaked handle succeeds on retry.
    assert await leaked.retire() is True
    assert leaked.retired is True
    assert len(leaked) == 0
    assert bus.made[0].unsub_calls == 2  # retried


async def test_retire_unsubscribes_all_and_flips_state() -> None:
    bus = FakeBus()
    result = await subscribe_all(bus.subscribe, _entries())

    await result.retire()

    assert result.retired is True
    assert len(result) == 0
    assert [s.subject for s in bus.made] == ["a", "b", "c"]
    assert all(s.unsub_calls == 1 for s in bus.made)


async def test_double_retire_is_noop() -> None:
    bus = FakeBus()
    result = await subscribe_all(bus.subscribe, _entries())

    await result.retire()
    await result.retire()

    assert all(s.unsub_calls == 1 for s in bus.made)
    assert result.retired is True
    assert len(result) == 0


async def test_retire_survives_unsubscribe_failure_mid_sweep() -> None:
    # Middle unsubscribe fails transiently. retire() sweeps the rest, never
    # raises, RETAINS "b" (returns False), and a second retire recovers it.
    bus = FakeBus(unsub_transient_subjects=frozenset({"b"}))
    result = await subscribe_all(bus.subscribe, _entries())

    assert await result.retire() is False  # must not raise
    # "a" and "c" unsubscribed; "b" retained for retry.
    assert all(s.unsub_calls == 1 for s in bus.made)
    assert len(result) == 1
    assert result.subjects == ("b",)
    assert result.retired is False

    # Second retire: "b" now succeeds.
    assert await result.retire() is True
    assert result.retired is True
    assert len(result) == 0
    by_subject = {s.subject: s for s in bus.made}
    assert by_subject["b"].unsub_calls == 2  # retried
    assert by_subject["a"].unsub_calls == 1  # not re-swept
    assert by_subject["c"].unsub_calls == 1


async def test_clean_rollback_error_carries_empty_leaked() -> None:
    # Rollback of "a" succeeds fully -> leaked is empty and born retired.
    bus = FakeBus(fail_subject="b")

    with pytest.raises(FrameSubscribeError) as excinfo:
        await subscribe_all(bus.subscribe, _entries())

    leaked = excinfo.value.leaked
    assert len(leaked) == 0
    assert leaked.retired is True
    assert leaked.subjects == ()


async def test_retire_retries_permanently_failing_handle_each_call() -> None:
    # A permanently failing unsubscribe: retire returns False both times, the
    # handle stays retained, and unsubscribe is attempted on each call.
    bus = FakeBus(unsub_raises_subjects=frozenset({"a", "b", "c"}))
    result = await subscribe_all(bus.subscribe, _entries())

    assert await result.retire() is False
    assert len(result) == 3
    assert result.retired is False
    assert all(s.unsub_calls == 1 for s in bus.made)

    assert await result.retire() is False
    assert len(result) == 3
    assert all(s.unsub_calls == 2 for s in bus.made)  # retried each call


async def test_concurrent_retire_serializes_one_successful_attempt() -> None:
    subscription = BlockingSub()
    result = SubscriptionSet([("a", subscription)])

    first = asyncio.create_task(result.retire())
    await subscription.started.wait()
    second = asyncio.create_task(result.retire())
    await asyncio.sleep(0)

    assert result.retired is False
    assert result.subjects == ("a",)
    assert subscription.unsub_calls == 1

    subscription.release.set()
    assert await asyncio.gather(first, second) == [True, True]
    assert result.retired is True
    assert result.subjects == ()
    assert subscription.unsub_calls == 1


async def test_concurrent_retire_serializes_failed_attempts_and_retains_handle() -> None:
    subscription = BlockingSub(fail=True)
    result = SubscriptionSet([("a", subscription)])

    first = asyncio.create_task(result.retire())
    await subscription.started.wait()
    second = asyncio.create_task(result.retire())
    await asyncio.sleep(0)

    assert result.retired is False
    assert result.subjects == ("a",)
    assert subscription.unsub_calls == 1

    subscription.release.set()
    assert await asyncio.gather(first, second) == [False, False]
    assert result.retired is False
    assert result.subjects == ("a",)
    assert subscription.unsub_calls == 2


def test_subscription_set_exported() -> None:
    assert "SubscriptionSet" in rtech_frames.__all__
    assert "subscribe_all" in rtech_frames.__all__
    assert "FrameSubscribeError" in rtech_frames.__all__
    assert "Subscription" in rtech_frames.__all__


# A subscribe exception whose message carries a credential-shaped secret: neither
# the rollback log nor the raised FrameSubscribeError text may render this string.
_SUBSCRIBE_SECRET = "nats://user:s3cr3t-t0ken-DEADBEEF@bus.internal:4222"  # noqa: S105


@dataclass
class SecretFailBus(FakeBus):
    """A bus whose subscribe raises with a credential-bearing message."""

    async def subscribe(self, subject: str, handler: object) -> Subscription:
        self.subscribe_calls.append(subject)
        if subject == self.fail_subject:
            raise RuntimeError(f"connection refused for {_SUBSCRIBE_SECRET}")
        sub = FakeSub(subject=subject)
        self.made.append(sub)
        return sub


@dataclass
class SecretUnsubscribeSub(FakeSub):
    """Fail one unsubscribe with a credential-bearing broker error."""

    async def unsubscribe(self) -> None:
        self.unsub_calls += 1
        if self.unsub_calls == 1:
            raise RuntimeError(f"authorization failed for {_SUBSCRIBE_SECRET}")


@dataclass
class SecretRollbackBus(FakeBus):
    """Create one live subscription, then fail the next subscribe."""

    async def subscribe(self, subject: str, handler: object) -> Subscription:
        self.subscribe_calls.append(subject)
        if subject == self.fail_subject:
            raise RuntimeError("second subscribe failed")
        sub = SecretUnsubscribeSub(subject=subject)
        self.made.append(sub)
        return sub


async def test_subscribe_all_omits_raw_exception_from_log_and_error() -> None:
    # Regression (finding #2): subscribe_all must render the exception TYPE, never
    # the raw caught exception, in BOTH the rollback log and the raised
    # FrameSubscribeError message — a broker failure can carry a credential.
    # Safe subject context and rollback semantics are preserved.
    captured_logs: list[str] = []
    sink_id = logger.add(
        lambda m: captured_logs.append(m.record["message"]),
        level="ERROR",
    )
    try:
        bus = SecretFailBus(fail_subject="b")
        with pytest.raises(FrameSubscribeError) as excinfo:
            await subscribe_all(bus.subscribe, _entries())
    finally:
        logger.remove(sink_id)

    # Rollback semantics preserved: "a" was subscribed then rolled back, "c"
    # never attempted; the safe subject still rides on the error message.
    assert bus.subscribe_calls == ["a", "b"]
    assert [s.subject for s in bus.made] == ["a"]
    assert bus.made[0].unsub_calls == 1

    error_text = str(excinfo.value)
    assert "subscribe to b failed" in error_text
    assert "error_type=RuntimeError" in "\n".join(captured_logs)
    assert "RuntimeError" in error_text

    # The secret appears in neither the log nor the raised error text.
    joined = error_text + "\n" + "\n".join(captured_logs)
    assert _SUBSCRIBE_SECRET not in joined
    assert "s3cr3t" not in joined
    # The chained cause still carries the original for the caller's own handling.
    assert isinstance(excinfo.value.__cause__, RuntimeError)


async def test_rollback_omits_raw_unsubscribe_exception_and_retains_handle() -> None:
    """Keep rollback credentials out of output without losing retry authority."""

    captured_logs: list[str] = []
    sink_id = logger.add(
        lambda message: captured_logs.append(message.record["message"]),
        level="ERROR",
    )
    try:
        bus = SecretRollbackBus(fail_subject="b")
        with pytest.raises(FrameSubscribeError) as excinfo:
            await subscribe_all(bus.subscribe, _entries())
    finally:
        logger.remove(sink_id)

    assert bus.subscribe_calls == ["a", "b"]
    assert [sub.subject for sub in bus.made] == ["a"]
    assert bus.made[0].unsub_calls == 1

    leaked = excinfo.value.leaked
    assert leaked.subjects == ("a",)
    assert leaked.retired is False
    assert len(leaked) == 1

    output = str(excinfo.value) + "\n" + "\n".join(captured_logs)
    assert "subscribe to b failed: RuntimeError" in output
    assert "unsubscribe failed subject=a error_type=RuntimeError" in output
    assert _SUBSCRIBE_SECRET not in output
    assert "s3cr3t" not in output

    assert await leaked.retire() is True
    assert leaked.retired is True
    assert len(leaked) == 0
    assert bus.made[0].unsub_calls == 2
