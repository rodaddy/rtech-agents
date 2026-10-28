"""Per-agent config resolution from RTECH_AGENT_ID + committed agent.yaml.

Covers the entrypoint seam that resolves ``agents/<id>/agent.yaml`` from the
agent id (found -> AgentConfig used; missing -> None fallback + loud log) and a
guard that every committed trio agent.yaml parses via ``load_agent_config`` with
no literal secrets.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from loguru import logger
from rtech_agentd.__main__ import (
    DEFAULT_AGENTS_ROOT,
    ENV_AGENTS_ROOT,
    _engine_env_from_config,
    _resolve_agent_config,
    _resolve_system_prompt,
)
from rtech_agentd.config import AgentConfig, load_agent_config
from rtech_agentd.engine_select import ENV_ENGINE, ENV_ENGINE_ALT, ENV_MODEL, _selected_kind

# Repo root: this file is packages/rtech-agentd/tests/<f> -> up 3 = repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_AGENTS_ROOT = _REPO_ROOT / "agents"
_TRIO = ("agentd-birk", "agentd-sprocket", "agentd-nimbus")


def _capture_logs() -> tuple[list[str], int]:
    """Route loguru to an in-memory sink; return (buffer, sink_id)."""
    buf: list[str] = []
    sink_id = logger.add(lambda m: buf.append(str(m)), level="WARNING")
    return buf, sink_id


def _write_manifest(dir_path: Path, agent: str, *, kind: str = "claude") -> None:
    """Write a minimal valid agent.yaml at ``dir_path`` for ``agent``."""
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "agent.yaml").write_text(
        f"agent: {agent}\n"
        "home: /home/x/.rtech-agents\n"
        f'engine:\n  kind: {kind}\n  model: "claude-sonnet-5"\n'
        'nats:\n  url: "nats://127.0.0.1:4222"\n  env: dev\n',
        encoding="utf-8",
    )


def test_resolve_found_config_is_used(tmp_path: Path, monkeypatch) -> None:
    """A present agents/<id>/agent.yaml resolves to an AgentConfig."""
    _write_manifest(tmp_path / "agents" / "agentd-nova", "agentd-nova")
    monkeypatch.setenv(ENV_AGENTS_ROOT, str(tmp_path / "agents"))
    config = _resolve_agent_config("agentd-nova")
    assert config is not None
    assert config.agent == "agentd-nova"
    assert config.engine.model == "claude-sonnet-5"


def test_resolve_by_slug_strips_agentd_prefix(tmp_path: Path, monkeypatch) -> None:
    """RTECH_AGENT_ID 'agentd-sprocket' resolves the slug dir agents/sprocket.

    Dirs are slug-named but the systemd instance id carries the 'agentd-' prefix;
    the exact-id dir does NOT exist, so resolution must fall through to the slug.
    """
    _write_manifest(tmp_path / "agents" / "sprocket", "agentd-sprocket")
    monkeypatch.setenv(ENV_AGENTS_ROOT, str(tmp_path / "agents"))
    config = _resolve_agent_config("agentd-sprocket")
    assert config is not None
    assert config.agent == "agentd-sprocket"


def test_resolve_exact_id_dir_wins_over_slug(tmp_path: Path, monkeypatch) -> None:
    """When BOTH an exact-id and a slug dir exist, the exact-id dir wins."""
    root = tmp_path / "agents"
    # slug dir carries model claude-sonnet-5; exact-id dir carries a distinct home.
    _write_manifest(root / "sprocket", "agentd-sprocket")
    exact = root / "agentd-sprocket"
    exact.mkdir(parents=True)
    (exact / "agent.yaml").write_text(
        "agent: agentd-sprocket\n"
        "home: /home/exact/.rtech-agents\n"
        'engine:\n  kind: claude\n  model: "claude-opus-5"\n'
        'nats:\n  url: "nats://127.0.0.1:4222"\n  env: dev\n',
        encoding="utf-8",
    )
    monkeypatch.setenv(ENV_AGENTS_ROOT, str(root))
    config = _resolve_agent_config("agentd-sprocket")
    assert config is not None
    # The exact-id dir's manifest (opus) resolved, not the slug's (sonnet).
    assert config.engine.model == "claude-opus-5"


def test_resolve_missing_config_falls_back_with_loud_log(tmp_path: Path, monkeypatch) -> None:
    """A missing agent.yaml returns None (env-only fallback) and logs loudly."""
    monkeypatch.setenv(ENV_AGENTS_ROOT, str(tmp_path / "agents"))
    buf, sink_id = _capture_logs()
    try:
        config = _resolve_agent_config("agentd-nope")
    finally:
        logger.remove(sink_id)
    assert config is None
    joined = "".join(buf)
    assert "FALLBACK" in joined
    assert "agentd-nope" in joined


def test_default_agents_root_is_relative(monkeypatch) -> None:
    """Absent the env var, the root default is the './agents' convention."""
    monkeypatch.delenv(ENV_AGENTS_ROOT, raising=False)
    assert DEFAULT_AGENTS_ROOT == "agents"


_NO_REAL_AGENTS = "real agent configs are deployment config; only agents/_skeleton ships"


@pytest.mark.skip(reason=_NO_REAL_AGENTS)
def test_engine_env_overlays_config_kind_and_model(monkeypatch) -> None:
    """config.engine threads into build_engine's env (kind + model keys)."""
    monkeypatch.delenv(ENV_ENGINE, raising=False)
    monkeypatch.delenv(ENV_ENGINE_ALT, raising=False)
    monkeypatch.delenv(ENV_MODEL, raising=False)
    cfg = load_agent_config(_AGENTS_ROOT / "birk" / "agent.yaml")
    env = _engine_env_from_config(cfg)
    # BOTH spellings are set so the manifest kind is authoritative.
    assert env[ENV_ENGINE] == cfg.engine.kind
    assert env[ENV_ENGINE_ALT] == cfg.engine.kind
    assert env[ENV_MODEL] == cfg.engine.model


def test_manifest_kind_wins_over_stray_e2e_engine(tmp_path: Path, monkeypatch) -> None:
    """A stray E2E_ENGINE must NOT override the manifest engine kind.

    _selected_kind reads E2E_ENGINE before DEV_STACK_ENGINE, so the overlay has
    to pin the primary spelling too. Verified through _selected_kind (the exact
    reader build_engine uses) rather than just asserting the dict key.
    """
    monkeypatch.setenv(ENV_ENGINE, "scripted")  # stray process env
    _write_manifest(tmp_path / "agents" / "nova", "agentd-nova", kind="claude")
    monkeypatch.setenv(ENV_AGENTS_ROOT, str(tmp_path / "agents"))
    cfg = _resolve_agent_config("agentd-nova")
    assert cfg is not None
    env = _engine_env_from_config(cfg)
    assert _selected_kind(env) == "claude"


def test_missing_config_system_prompt_is_hardcoded_fallback() -> None:
    """With no config, the system prompt is the hardcoded default (unchanged)."""
    from rtech_agentd.__main__ import SYSTEM_PROMPT

    assert _resolve_system_prompt(None) == SYSTEM_PROMPT


@pytest.mark.skip(reason=_NO_REAL_AGENTS)
def test_committed_trio_configs_parse_without_literal_secrets() -> None:
    """Every committed trio agent.yaml loads and carries only *_ref indirection."""
    for agent_id in _TRIO:
        slug = agent_id.removeprefix("agentd-")
        cfg: AgentConfig = load_agent_config(_AGENTS_ROOT / slug / "agent.yaml")
        assert cfg.agent == agent_id
        # Credential fields are refs (or None), never a literal token.
        for ref in (cfg.openbrain.token_ref, cfg.discord.bot_token_ref):
            assert ref is None or ":" in ref
        # SOUL.md exists next to the config (persona home is source-owned).
        assert (_AGENTS_ROOT / slug / "SOUL.md").is_file()


def test_grep_gate_passes_no_live_hermes_import() -> None:
    """The grep-gate (which now bans live rtech_hermes imports) passes.

    CI runs scripts/grep-gate.sh in its own job on every push/PR; this test pins
    the same script locally so the import ban is exercised in the suite too.
    Skips cleanly when zsh/rg (the script's deps) are not on PATH.
    """
    script = _REPO_ROOT / "scripts" / "grep-gate.sh"
    if shutil.which("zsh") is None or shutil.which("rg") is None:
        pytest.skip("grep-gate.sh needs zsh + rg on PATH")
    result = subprocess.run(  # noqa: S603
        ["zsh", str(script)],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"grep-gate failed:\n{result.stdout}\n{result.stderr}"
