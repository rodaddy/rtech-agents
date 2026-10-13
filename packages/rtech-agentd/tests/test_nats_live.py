"""Live NATS round-trip: an inbound chat envelope drives the ingress + a reply.

Proves the real wire path against the fleet NATS server (fleet-hub,
nats://127.0.0.1:4222 by default, FLEET_ENV=dev). Uses a FakeEngine so the
test needs no Claude SDK — it exercises subscribe -> handle_envelope -> publish
over a real bus. Auto-skips when the server is unreachable (fleet-bus pattern).
"""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from fleet_nats import BusConfig, Envelope, FleetBus, subjects
from rtech_agentd.engine import ToolsPolicy, TurnDone
from rtech_agentd.nats_ingress import NatsIngress
from rtech_agentd.supervisor import Supervisor, SupervisorConfig

from .fakes import FakeEngine

_URL = BusConfig.from_env("pytest").url


def _server_reachable() -> bool:
    parts = urlsplit(_URL)
    host, port = parts.hostname or "", parts.port or 4222
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _server_reachable(), reason="NATS server at the configured URL not reachable"
)


@pytest.mark.asyncio
async def test_live_chat_roundtrip(tmp_path: Path) -> None:
    agent = f"agentd-live-{uuid4().hex[:8]}"
    engine = FakeEngine()

    # A subscriber for the chatd reply lane so we can observe the published reply.
    replies: list[Envelope] = []
    got_reply = asyncio.Event()

    async def on_reply(env: Envelope) -> None:
        replies.append(env)
        got_reply.set()

    reply_subject = subjects.direct("dev", "chatd")

    async with (
        FleetBus(BusConfig(agent_id=agent, url=_URL, env="dev")) as agent_bus,
        FleetBus(BusConfig(agent_id="chatd-probe", url=_URL, env="dev")) as probe_bus,
    ):
        await probe_bus.subscribe(reply_subject, on_reply)

        sup = Supervisor(
            session=engine,
            config=SupervisorConfig(agent=agent, env="dev"),
            home=tmp_path,
            publisher=agent_bus.publish,
            system_prompt="SYS",
            tools_policy=ToolsPolicy(),
        )
        ingress = NatsIngress(agent=agent, env="dev", supervisor=sup, publisher=agent_bus.publish)
        for subject in ingress.subscription_subjects():
            await agent_bus.subscribe(subject, ingress.handle_envelope)

        # Publish an inbound chat envelope onto this agent's direct lane.
        inbound_id = uuid4().hex
        await agent_bus.publish(
            subjects.direct("dev", agent),
            Envelope.new(
                msg_id=inbound_id,
                ts="t",
                sender="rico",
                kind="chat",
                channel="live-chan",
                payload={"text": "identify yourself"},
            ),
        )
        # Let the subscription deliver + the ingress send the turn to the engine.
        await asyncio.sleep(0.3)
        assert engine.sends, "inbound chat did not reach the engine"

        # Engine finishes the turn -> ingress publishes the chat_reply.
        await ingress._on_engine_event(TurnDone(text="I am the live test agent", session_ref="s1"))
        await asyncio.wait_for(got_reply.wait(), timeout=5)

    assert any(r.correlation_id == inbound_id for r in replies)
    assert any("live test agent" in r.payload.get("visible_text", "") for r in replies)
