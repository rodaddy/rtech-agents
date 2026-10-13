"""Failed turns render a human one-liner, never the raw engine payload (#123).

Observed live (AI Test Group, #109 smoke): an engine terminal error produced a
TurnFrame whose ``visible_text`` was the vendor's raw error JSON, which the edge
posted verbatim to chat. Chat surfaces are treated as leaked, so the summary is
built by :func:`rtech_agentd.engine.failure_visible_text` and applied at BOTH
publish sites (frame_turn and nats_ingress); the raw payload goes to logs only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fleet_nats import Envelope
from rtech_agentd.engine import ToolsPolicy, TurnDone, failure_visible_text
from rtech_agentd.frame_turn import FrameTurnRunner
from rtech_agentd.nats_ingress import NatsIngress
from rtech_agentd.supervisor import Supervisor, SupervisorConfig
from rtech_frames import Stance, TaskFrame, TurnFrame
from rtech_orchestrator import subjects

from .fakes import FakeEngine, FakeFrameBus, RecordingPublisher

_AGENT = "tester"

# The shape observed live: a vendor error body with a URL-bearing message.
_RAW_ENGINE_ERROR = (
    '{"type":"error","status":400,"error":{"type":"invalid_request_error",'
    '"message":"The gpt-5-codex model is not supported, see '
    'https://vendor.example/docs?key=cfg-fragment"}}'
)


# --------------------------------------------------------------- helper unit


def test_failure_text_extracts_type_and_status_from_json() -> None:
    line = failure_visible_text(_RAW_ENGINE_ERROR)
    assert line == "engine error: invalid_request_error (status 400) — details in agentd logs"


def test_failure_text_never_echoes_payload_fragments() -> None:
    line = failure_visible_text(_RAW_ENGINE_ERROR)
    assert "{" not in line
    assert "vendor.example" not in line
    assert "cfg-fragment" not in line


def test_failure_text_uses_status_code_key_and_flat_shape() -> None:
    line = failure_visible_text('{"type":"rate_limit_error","statusCode":429}')
    assert line == "engine error: rate_limit_error (status 429) — details in agentd logs"


def test_failure_text_keeps_short_plain_message_as_detail() -> None:
    assert (
        failure_visible_text("codex turn error")
        == "engine error: codex turn error — details in agentd logs"
    )


def test_failure_text_generic_for_urls_braces_multiline_or_long() -> None:
    generic = "engine error — details in agentd logs"
    assert failure_visible_text("see https://vendor.example/trace/123") == generic
    assert failure_visible_text("oops {partial json") == generic
    assert failure_visible_text("line one\nline two") == generic
    assert failure_visible_text("x" * 500) == generic
    assert failure_visible_text("not-json-but-{brace") == generic
    assert failure_visible_text("") == generic


def test_failure_text_malformed_json_is_generic() -> None:
    assert failure_visible_text('{"type": broken') == "engine error — details in agentd logs"


def test_failure_text_json_extracted_detail_gets_the_same_payload_guards() -> None:
    # Review finding (PR #125): the URL/brace/newline guards must apply to
    # JSON-extracted type/code values too, not only to the plain branch — a
    # vendor payload can smuggle payload material inside those fields.
    generic = "engine error — details in agentd logs"
    url = '{"error":{"type":"https://vendor.example/secret?cfg=x"},"status":400}'
    assert failure_visible_text(url) == "engine error (status 400) — details in agentd logs"
    newline = '{"error":{"code":"first line\\nFORGED log entry cfg=secret"}}'
    assert failure_visible_text(newline) == generic
    braces = '{"error":{"type":"{config-fragment}"}}'
    assert failure_visible_text(braces) == generic
    long_type = '{"error":{"type":"' + "x" * 500 + '"}}'
    assert failure_visible_text(long_type) == generic


def test_failure_text_detail_is_allowlisted_not_blocklisted() -> None:
    # Verifier finding (PR #125): a blocklist missed Unicode line separators,
    # control chars, bidi overrides, scheme-less URLs, and token-shaped
    # payloads. The detail must FULLMATCH a plain-ASCII sentence shape with no
    # unbroken run longer than a real word or vendor error code.
    generic = "engine error — details in agentd logs"
    assert failure_visible_text("split detail") == generic
    assert failure_visible_text("bad\x00detail") == generic
    assert failure_visible_text("\x1b[31mansi styled\x1b[0m") == generic
    assert failure_visible_text("tab\tdetail") == generic
    assert failure_visible_text("‮evil bidi") == generic
    assert failure_visible_text("vendor.example/private?token=abc") == generic
    # A bare provider key fits the charset but is one long unbroken run —
    # and the outbound guard only knows prefixed shapes, so it MUST die here.
    # (Key assembled at runtime so no secret-shaped literal lands in source.)
    fake_key = "sk_" + "live_" + "a1B2" * 6
    assert failure_visible_text('{"error":{"type":"' + fake_key + '"}}') == generic
    # Legit shapes still render.
    assert (
        failure_visible_text('{"error":{"type":"temporarily_unavailable"}}')
        == "engine error: temporarily_unavailable — details in agentd logs"
    )
    assert (
        failure_visible_text("rate limited, try later")
        == "engine error: rate limited, try later — details in agentd logs"
    )


def test_failure_text_boolean_status_is_not_a_status() -> None:
    # Review finding (PR #125): bool is a subclass of int; {"status": true}
    # must not render "(status True)".
    assert (
        failure_visible_text('{"status":true,"type":"rate_limit"}')
        == "engine error: rate_limit — details in agentd logs"
    )


# --------------------------------------------------------------- frame_turn


def _runner(tmp_path: Path) -> tuple[FakeEngine, FrameTurnRunner, FakeFrameBus]:
    engine = FakeEngine()
    sup = Supervisor(
        session=engine,
        config=SupervisorConfig(agent=_AGENT, env="dev"),
        home=tmp_path,
        publisher=RecordingPublisher(),
        system_prompt="SYS",
        tools_policy=ToolsPolicy(),
    )
    bus = FakeFrameBus()
    return engine, FrameTurnRunner(agent_id=_AGENT, supervisor=sup, bus=bus), bus


def _task(task_id: str = "T-fail") -> TaskFrame:
    return TaskFrame(
        id=task_id,
        origin_edge="web",
        requester="rico",
        text="run it",
        scope="single",
        created_at=datetime.now(UTC),
    )


async def test_frame_turn_failed_turn_publishes_summary_not_payload(tmp_path: Path) -> None:
    _engine, runner, bus = _runner(tmp_path)
    await runner.start()
    task = _task()
    await bus.deliver(subjects.agent_task(_AGENT), task)

    await runner._on_engine_event(
        TurnDone(
            text=_RAW_ENGINE_ERROR,
            is_error=True,
            stance=Stance(position="failed", because="engine reported a terminal error"),
        )
    )
    turns = [f for f in bus.frames_on(subjects.task_turn(task.id)) if isinstance(f, TurnFrame)]
    assert len(turns) == 1
    visible = turns[0].visible_text
    assert visible == "engine error: invalid_request_error (status 400) — details in agentd logs"
    assert "vendor.example" not in visible
    assert turns[0].stance.position == "failed"


async def test_frame_turn_non_error_turn_text_is_untouched(tmp_path: Path) -> None:
    _engine, runner, bus = _runner(tmp_path)
    await runner.start()
    task = _task("T-ok")
    await bus.deliver(subjects.agent_task(_AGENT), task)

    await runner._on_engine_event(
        TurnDone(text="a normal reply", stance=Stance(position="answered", because="did it"))
    )
    turns = [f for f in bus.frames_on(subjects.task_turn(task.id)) if isinstance(f, TurnFrame)]
    assert turns[0].visible_text == "a normal reply"


# --------------------------------------------------------------- nats_ingress


def _ingress(tmp_path: Path) -> tuple[FakeEngine, NatsIngress, RecordingPublisher]:
    engine = FakeEngine()
    pub = RecordingPublisher()
    sup = Supervisor(
        session=engine,
        config=SupervisorConfig(agent=_AGENT, env="dev"),
        home=tmp_path,
        publisher=pub,
        system_prompt="SYS",
        tools_policy=ToolsPolicy(),
    )
    return engine, NatsIngress(agent=_AGENT, env="dev", supervisor=sup, publisher=pub), pub


async def test_ingress_failed_turn_replies_with_summary_not_payload(tmp_path: Path) -> None:
    _engine, ingress, pub = _ingress(tmp_path)
    inbound = Envelope.new(
        msg_id=uuid4().hex,
        ts=datetime.now(UTC).isoformat(),
        sender="mgr",
        kind="chat",
        channel="general",
        payload={"text": "yo"},
    )
    await ingress.handle_envelope(inbound)

    await ingress._on_engine_event(
        TurnDone(
            text=_RAW_ENGINE_ERROR,
            is_error=True,
            stance=Stance(position="failed", because="engine reported a terminal error"),
        )
    )
    _subj, reply = pub.by_kind("chat_reply")[0]
    frame = TurnFrame.model_validate(reply.payload)
    assert (
        frame.visible_text
        == "engine error: invalid_request_error (status 400) — details in agentd logs"
    )
    assert "vendor.example" not in frame.visible_text
