"""Functional tests for the transport-neutral channel dataclasses."""

from __future__ import annotations

from rtech_chatd.channel import InboundMessage, SendTarget


def test_send_target_thread_wins_over_channel() -> None:
    assert SendTarget(channel_id="c1", thread_id="t1").resolve() == "t1"


def test_send_target_channel_when_no_thread() -> None:
    assert SendTarget(channel_id="c1").resolve() == "c1"


def test_inbound_message_defaults() -> None:
    m = InboundMessage(channel_id="c1", thread_id=None, author="u1", text="hi")
    assert m.is_bot is False
    assert m.is_dm is False
    assert m.mentions_me is False
    assert m.is_reply is False


def test_inbound_message_is_frozen() -> None:
    m = InboundMessage(channel_id="c1", thread_id=None, author="u1", text="hi")
    try:
        m.text = "changed"  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("InboundMessage should be frozen")
