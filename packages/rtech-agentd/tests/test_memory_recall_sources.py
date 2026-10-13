"""Verify typed, bounded source-aware Open Brain recall."""

from __future__ import annotations

from typing import Any

from rtech_agentd.memory import AgentMemoryLane, MemoryConfig, MemoryRecallResult


class _RecallClient:
    """Return canned local recall results without contacting Open Brain."""

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results = results

    def search_all(self, **arguments: Any) -> dict[str, Any]:
        return {"results": self._results}


def _lane(results: list[dict[str, Any]], **config: Any) -> AgentMemoryLane:
    defaults: dict[str, Any] = {
        "base_url": "https://ob.test",
        "token": "test-token",
        "namespace": "test",
        "agent": "tester",
    }
    defaults.update(config)
    return AgentMemoryLane(MemoryConfig(**defaults), client=_RecallClient(results))


def test_recall_with_sources_assigns_stable_labels_and_safe_metadata() -> None:
    lane = _lane(
        [
            {
                "text": "first fact",
                "source": "session_event",
                "kind": "fact",
                "id": "fact-1",
                "path": "docs/source.md",
                "tier": "durable",
                "token": "must-not-escape",
                "metadata": {"private": "must-not-escape"},
            },
            {"text": "second fact", "source": "decision", "kind": "decision"},
        ]
    )

    result = lane.recall_with_sources("query")

    assert result.block == "- [OB1] first fact\n- [OB2] second fact"
    assert [source.label for source in result.sources] == ["OB1", "OB2"]
    assert result.sources[0].source == "session_event"
    assert result.sources[0].kind == "fact"
    assert result.sources[0].metadata == (
        ("id", "fact-1"),
        ("path", "docs/source.md"),
        ("tier", "durable"),
    )
    assert "must-not-escape" not in repr(result.sources)


def test_legacy_recall_preserves_original_prompt_format() -> None:
    lane = _lane([{"text": "remember this", "source": "brain", "kind": "fact"}])

    typed = lane.recall_with_sources("query")

    assert isinstance(typed, MemoryRecallResult)
    assert typed.block == "- [OB1] remember this"
    assert lane.recall("query") == "- [fact/brain] remember this"


def test_source_aware_recall_preserves_caps_and_redaction() -> None:
    marker = "Authorization: Bearer " + "x" * 40
    lane = _lane(
        [
            {"text": f"private {marker}", "source": marker, "kind": "fact"},
            {"text": "second", "source": "brain", "kind": "fact"},
        ],
        max_items=1,
        max_chars=48,
        max_item_chars=40,
    )

    result = lane.recall_with_sources("query")

    assert len(result.sources) == 1
    assert len(result.block) <= 48
    assert marker not in result.block
    assert marker not in repr(result.sources)


def test_source_aware_recall_failure_returns_empty_result() -> None:
    class _FailingClient(_RecallClient):
        def search_all(self, **arguments: Any) -> dict[str, Any]:
            raise RuntimeError("offline")

    lane = AgentMemoryLane(
        MemoryConfig(
            base_url="https://ob.test",
            token="test-token",
            namespace="test",
            agent="tester",
        ),
        client=_FailingClient([]),
    )

    result = lane.recall_with_sources("query")

    assert result == MemoryRecallResult(block="", status="failed")
    assert lane.recall("query") == ""
