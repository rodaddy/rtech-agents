"""persona_glue: composes a system prompt from an agent home via rtech_persona."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from rtech_agentd.persona_glue import build_system_prompt
from rtech_persona import MissingSoulError


def test_build_system_prompt_includes_soul(tmp_path: Path) -> None:
    (tmp_path / "SOUL.md").write_text("I am Nova, a test agent.", encoding="utf-8")
    (tmp_path / "KERNEL.md").write_text("Rule: be precise.", encoding="utf-8")
    prompt = build_system_prompt(
        tmp_path,
        model="claude-sonnet-5",
        now=datetime(2026, 7, 7, tzinfo=UTC),
    )
    assert "I am Nova" in prompt
    assert "Rule: be precise." in prompt
    assert "claude-sonnet-5" in prompt  # timestamp/model line
    # memory_block is None for now (Phase 4) — no memory slot present.


def test_missing_soul_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(MissingSoulError):
        build_system_prompt(tmp_path, model="m")
