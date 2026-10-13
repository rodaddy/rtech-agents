"""Steering: a `steer` envelope must land INSIDE an active turn.

The proof is an event-order assertion: after a chat turn starts (an
AssistantMessage delta has been emitted, so the turn is active), a `steer`
envelope's text is sent to the SAME session while the turn is still open — the
fake records that the second send happened `during_active_turn=True`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from fleet_nats import Envelope
from rtech_agentd.engine import TextDelta, ToolsPolicy, TurnDone
from rtech_agentd.nats_ingress import NatsIngress
from rtech_agentd.supervisor import Supervisor, SupervisorConfig
from rtech_frames import TurnFrame

from .fakes import FakeEngine, RecordingPublisher


def _env(kind: str, *, text: str | None = None, channel: str | None = "chan-1") -> Envelope:
    payload = {"text": text} if text is not None else {}
    return Envelope.new(
        msg_id=uuid4().hex,
        ts=datetime.now(UTC).isoformat(),
        sender="rico",
        kind=kind,
        channel=channel,
        payload=payload,
    )


def _make(tmp_path: Path) -> tuple[FakeEngine, Supervisor, NatsIngress, RecordingPublisher]:
    engine = FakeEngine()
    pub = RecordingPublisher()
    sup = Supervisor(
        session=engine,
        config=SupervisorConfig(agent="tester", env="dev"),
        home=tmp_path,
        publisher=pub,
        system_prompt="SYS",
        tools_policy=ToolsPolicy(),
    )
    ingress = NatsIngress(agent="tester", env="dev", supervisor=sup, publisher=pub)
    return engine, sup, ingress, pub


@pytest.mark.asyncio
async def test_steer_lands_inside_active_turn(tmp_path: Path) -> None:
    engine, _sup, ingress, _pub = _make(tmp_path)

    # Inbound chat turn -> first send (no active turn yet).
    await ingress.handle_envelope(_env("chat", text="do the thing"))
    assert len(engine.sends) == 1
    assert engine.sends[0].during_active_turn is False

    # The turn is now streaming (an assistant delta arrived): mark active.
    await engine.push(TextDelta(text="working..."))
    engine.mark_turn_active()

    # A steer envelope arrives mid-turn -> second send DURING the active turn.
    await ingress.handle_envelope(_env("steer", text="actually, do it differently"))
    assert len(engine.sends) == 2
    assert engine.sends[1].during_active_turn is True
    assert engine.sends[1].text == "actually, do it differently"


@pytest.mark.asyncio
async def test_interrupt_envelope_calls_interrupt(tmp_path: Path) -> None:
    engine, _sup, ingress, _pub = _make(tmp_path)
    await ingress.handle_envelope(_env("chat", text="start"))
    engine.mark_turn_active()
    await ingress.handle_envelope(_env("interrupt"))
    assert engine.interrupts == 1


@pytest.mark.asyncio
async def test_steer_with_empty_text_is_ignored(tmp_path: Path) -> None:
    engine, _sup, ingress, _pub = _make(tmp_path)
    await ingress.handle_envelope(_env("chat", text="start"))
    before = len(engine.sends)
    await ingress.handle_envelope(_env("steer", text="   "))
    assert len(engine.sends) == before  # empty steer sends nothing


@pytest.mark.asyncio
async def test_turn_done_produces_chat_reply(tmp_path: Path) -> None:
    engine, sup, ingress, pub = _make(tmp_path)
    # Drive the pump so the sink fires on TurnDone.
    await ingress.handle_envelope(_env("chat", text="hello"))
    # Simulate the engine finishing the turn; feed the sink directly (the sink is
    # what the supervisor pump would call).
    await ingress._on_engine_event(TurnDone(text="hi there", session_ref="s1"))

    replies = pub.by_kind("chat_reply")
    assert len(replies) == 1
    subject, env = replies[0]
    assert subject == "dev.direct.chatd"
    assert TurnFrame.model_validate(env.payload).visible_text == "hi there"
    assert env.correlation_id is not None
