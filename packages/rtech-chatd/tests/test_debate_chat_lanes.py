"""Chat-scoped persistent debate lanes: per-chat agent->topic selection order.

These tests pin the additive chat-scoped lane surface (rtech-chatd intake/config
parsing plus edge topic selection). The invariant under test is the selection
ORDER in :meth:`DiscordEdge._debate_thread_for`:

    exact chat+agent  ->  legacy agent-only  ->  per-turn thread creation

An exact chat+agent lane wins first, the legacy single-group agent-only lane is
the byte-for-byte fallback, and an unlisted chat/agent falls through to the
existing lazy per-turn thread create so full text is NEVER dropped. The live trio
mappings (three groups, same three agents, different topic ids per group) are
exercised end-to-end through the config env parsing + edge normalization.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from loguru import logger
from rtech_chatd.channel import InboundMessage, SendTarget
from rtech_chatd.config import IntakeConfig
from rtech_chatd.edge import (
    DiscordEdge,
    debate_chat_lanes_from_triples,
    debate_lanes_from_pairs,
)

# The three live groups and their per-chat topic ids for the same three agents.
TRIO_CHAT = "-1001111111111"
RTECH_CHAT = "-1002222222222"
THIRD_CHAT = "-1003333333333"

LIVE_CHAT_LANES: tuple[tuple[str, str, str], ...] = (
    (TRIO_CHAT, "sprocket", "93"),
    (TRIO_CHAT, "nimbus", "94"),
    (TRIO_CHAT, "birk", "95"),
    (RTECH_CHAT, "sprocket", "5"),
    (RTECH_CHAT, "nimbus", "6"),
    (RTECH_CHAT, "birk", "7"),
    (THIRD_CHAT, "sprocket", "15"),
    (THIRD_CHAT, "nimbus", "16"),
    (THIRD_CHAT, "birk", "17"),
)


@dataclass
class _FakeAdapter:
    """Records create_thread calls; hands back a deterministic per-turn thread id."""

    thread_calls: list[tuple[str, str, str]] = field(default_factory=list)

    async def events(self) -> AsyncIterator[InboundMessage]:
        return
        yield  # pragma: no cover — never reached; makes this an async generator.

    async def send(self, target: SendTarget, text: str, files: list[Path]) -> str | None:
        return "sent-1"

    async def typing(self, target: SendTarget, on: bool) -> None:
        return None

    async def create_thread(self, channel_id: str, message_id: str, name: str) -> str | None:
        self.thread_calls.append((channel_id, message_id, name))
        return f"per-turn-{channel_id}-{message_id}"


class _FakeBus:
    async def publish(self, subject: str, data: bytes) -> None:  # pragma: no cover
        return None

    async def subscribe(self, subject: str, handler: object) -> object:  # pragma: no cover
        return object()


def _edge(
    *,
    debate_lanes: dict[str, str] | None = None,
    debate_chat_lanes: dict[tuple[str, str], str] | None = None,
    adapter: _FakeAdapter | None = None,
) -> DiscordEdge:
    return DiscordEdge(
        adapter or _FakeAdapter(),
        _FakeBus(),
        stack="dev",
        id_factory=lambda: "task-1",
        intake_enabled=True,
        debate_lanes=debate_lanes or {},
        debate_chat_lanes=debate_chat_lanes or {},
    )


def _thread_for(edge: DiscordEdge, *, chat_id: str, agent_id: str) -> str | None:
    target = SendTarget(channel_id=chat_id, thread_id=None)
    # A landed TLDR anchor so the per-turn fallback path can actually create.
    edge._debate_tldr_msgs[("task-1", agent_id)] = "anchor-1"
    return asyncio.run(edge._debate_thread_for("task-1", agent_id, target))


# --------------------------------------------------------------- config parsing


def test_env_parses_live_chat_lane_mappings(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = ",".join(f"{chat}:{name}={topic}" for chat, name, topic in LIVE_CHAT_LANES)
    monkeypatch.setenv("RTECH_CHATD_DEBATE_CHAT_LANES", raw)
    cfg = IntakeConfig.from_env()
    assert cfg.debate_chat_lanes == LIVE_CHAT_LANES


def test_env_canonicalizes_signed_chat_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "RTECH_CHATD_DEBATE_CHAT_LANES",
        "+123:sprocket=93,-0100:nimbus=94",
    )
    cfg = IntakeConfig.from_env()
    assert cfg.debate_chat_lanes == (("123", "sprocket", "93"), ("-100", "nimbus", "94"))


def test_chat_lane_normalizer_keys_on_chat_and_agent() -> None:
    lanes = debate_chat_lanes_from_triples(LIVE_CHAT_LANES)
    assert lanes[(TRIO_CHAT, "sprocket")] == "93"
    assert lanes[(RTECH_CHAT, "sprocket")] == "5"
    assert lanes[(THIRD_CHAT, "sprocket")] == "15"
    # Same agent, three chats, three distinct topic ids.
    assert len({lanes[(chat, "sprocket")] for chat in (TRIO_CHAT, RTECH_CHAT, THIRD_CHAT)}) == 3


def test_chat_lane_normalizer_normalizes_agentd_convention_ids() -> None:
    # The agent name is normalized exactly like the legacy map (agentd- prefix +
    # host suffix stripped), so a lane keyed by bare name matches an agentd id.
    lanes = debate_chat_lanes_from_triples(((TRIO_CHAT, "agentd-nimbus-dev", "94"),))
    assert lanes == {(TRIO_CHAT, "nimbus"): "94"}


def test_chat_lane_normalizer_preserves_hyphenated_identities_and_rejects_collisions() -> None:
    lanes = debate_chat_lanes_from_triples(
        ((TRIO_CHAT, "agentd-dash-alpha", "93"), (TRIO_CHAT, "agentd-dash-beta", "94"))
    )
    assert lanes == {(TRIO_CHAT, "dash-alpha"): "93", (TRIO_CHAT, "dash-beta"): "94"}
    assert debate_lanes_from_pairs((("agentd-nimbus-dev", "95"),)) == {"nimbus": "95"}
    with pytest.raises(ValueError, match="normalizes multiple source keys"):
        debate_lanes_from_pairs((("agentd-dash-alpha-dev", "93"), ("dash-alpha-prod", "94")))
    with pytest.raises(ValueError, match="normalizes multiple source keys"):
        debate_chat_lanes_from_triples(
            ((TRIO_CHAT, "agentd-dash-alpha-dev", "93"), (TRIO_CHAT, "dash-alpha-prod", "94"))
        )


def test_invalid_chat_lane_entries_warn_safely_and_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []
    sink_id = logger.add(
        lambda message: captured.append(message.record["message"]), level="WARNING"
    )
    monkeypatch.setenv(
        "RTECH_CHATD_DEBATE_CHAT_LANES",
        (
            f"sensitive-garbage,{TRIO_CHAT}:sprocket,{TRIO_CHAT}:=93,:nimbus=94,"
            f"not-a-chat:sprocket=93,0:sprocket=93,{TRIO_CHAT}:nimbus=not-a-topic,"
            f"{TRIO_CHAT}:birk=0,{TRIO_CHAT}:birk=-1,{RTECH_CHAT}:birk=7"
        ),
    )
    try:
        cfg = IntakeConfig.from_env()
    finally:
        logger.remove(sink_id)

    assert cfg.debate_chat_lanes == ((RTECH_CHAT, "birk", "7"),)
    assert len(captured) == 9
    assert all("skipping invalid chat-scoped debate lane entry" in message for message in captured)
    assert all("index=" in message and "reason=" in message for message in captured)
    assert "sensitive-garbage" not in "\n".join(captured)
    assert "not-a-chat" not in "\n".join(captured)
    assert "not-a-topic" not in "\n".join(captured)


# ------------------------------------------------------------- selection order


def test_exact_chat_agent_selected_for_all_three_groups() -> None:
    edge = _edge(debate_chat_lanes=debate_chat_lanes_from_triples(LIVE_CHAT_LANES))
    # Trio group topics.
    assert _thread_for(edge, chat_id=TRIO_CHAT, agent_id="agentd-sprocket") == "93"
    assert _thread_for(edge, chat_id=TRIO_CHAT, agent_id="agentd-nimbus") == "94"
    assert _thread_for(edge, chat_id=TRIO_CHAT, agent_id="agentd-birk") == "95"
    # rTech Group topics.
    assert _thread_for(edge, chat_id=RTECH_CHAT, agent_id="agentd-sprocket") == "5"
    assert _thread_for(edge, chat_id=RTECH_CHAT, agent_id="agentd-nimbus") == "6"
    assert _thread_for(edge, chat_id=RTECH_CHAT, agent_id="agentd-birk") == "7"
    # Third group topics.
    assert _thread_for(edge, chat_id=THIRD_CHAT, agent_id="agentd-sprocket") == "15"
    assert _thread_for(edge, chat_id=THIRD_CHAT, agent_id="agentd-nimbus") == "16"
    assert _thread_for(edge, chat_id=THIRD_CHAT, agent_id="agentd-birk") == "17"


def test_same_agent_different_topic_by_chat() -> None:
    edge = _edge(debate_chat_lanes=debate_chat_lanes_from_triples(LIVE_CHAT_LANES))
    sprocket = {
        chat: _thread_for(edge, chat_id=chat, agent_id="agentd-sprocket")
        for chat in (TRIO_CHAT, RTECH_CHAT, THIRD_CHAT)
    }
    assert sprocket == {TRIO_CHAT: "93", RTECH_CHAT: "5", THIRD_CHAT: "15"}


def test_legacy_agent_only_lane_unchanged_as_fallback() -> None:
    # With NO chat-scoped map, the legacy single-group agent->topic lane routes
    # every chat's debate for that agent to the same stable lane (byte-for-byte
    # the prior persistent-lane behavior), and no per-turn thread is minted.
    adapter = _FakeAdapter()
    edge = _edge(
        debate_lanes=debate_lanes_from_pairs((("nimbus", "legacy-topic"),)),
        adapter=adapter,
    )
    assert _thread_for(edge, chat_id=TRIO_CHAT, agent_id="agentd-nimbus") == "legacy-topic"
    assert _thread_for(edge, chat_id=THIRD_CHAT, agent_id="agentd-nimbus") == "legacy-topic"
    assert adapter.thread_calls == []  # lane pre-exists; nothing created


def test_chat_lane_wins_over_legacy_lane() -> None:
    # An exact chat+agent lane is selected FIRST, even when a legacy agent-only
    # lane also exists for the same agent.
    edge = _edge(
        debate_lanes=debate_lanes_from_pairs((("sprocket", "legacy-topic"),)),
        debate_chat_lanes=debate_chat_lanes_from_triples(((TRIO_CHAT, "sprocket", "93"),)),
    )
    # In the listed chat: the chat-scoped lane wins.
    assert _thread_for(edge, chat_id=TRIO_CHAT, agent_id="agentd-sprocket") == "93"
    # In an UNLISTED chat: falls back to the legacy agent-only lane.
    assert _thread_for(edge, chat_id=THIRD_CHAT, agent_id="agentd-sprocket") == "legacy-topic"


def test_unlisted_chat_agent_falls_through_to_per_turn_thread() -> None:
    # Neither a chat-scoped nor a legacy lane lists this chat/agent: the edge
    # falls through to the existing lazy per-turn thread create — full text is
    # never dropped.
    adapter = _FakeAdapter()
    edge = _edge(
        debate_chat_lanes=debate_chat_lanes_from_triples(((TRIO_CHAT, "sprocket", "93"),)),
        adapter=adapter,
    )
    # birk is unlisted in both maps for this chat -> per-turn create fires.
    thread = _thread_for(edge, chat_id=THIRD_CHAT, agent_id="agentd-birk")
    assert thread == f"per-turn-{THIRD_CHAT}-anchor-1"
    assert adapter.thread_calls == [(THIRD_CHAT, "anchor-1", "birk — full thought")]


def test_no_lanes_configured_is_pure_per_turn_behavior() -> None:
    # A deploy with neither map is byte-identical to the pre-lane per-turn path.
    adapter = _FakeAdapter()
    edge = _edge(adapter=adapter)
    thread = _thread_for(edge, chat_id=TRIO_CHAT, agent_id="agentd-nimbus")
    assert thread == f"per-turn-{TRIO_CHAT}-anchor-1"
    assert adapter.thread_calls == [(TRIO_CHAT, "anchor-1", "nimbus — full thought")]
