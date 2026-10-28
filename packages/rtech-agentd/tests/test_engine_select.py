"""Engine-selection unit tests: scripted, claude CLI-auth, and fail-closed paths.

The claude path deliberately requires only an installed CLI (subscription auth
rides in via that subprocess), so these tests prove ``build_engine`` selects the
real ``ClaudeEngine`` with an EMPTY env (no ``ANTHROPIC_API_KEY``) once the CLI
resolves, and fails closed with an actionable message when it does not.
"""

from __future__ import annotations

import pytest
from rtech_agentd import engine_select
from rtech_agentd.claude_engine import ClaudeEngine
from rtech_agentd.engine import EngineSession
from rtech_agentd.engine_select import EngineSelectionError, build_engine


def test_build_engine_scripted_selected() -> None:
    session, kind = build_engine({"DEV_STACK_ENGINE": "scripted"})
    assert kind == "scripted"
    assert isinstance(session, EngineSession)


def test_build_engine_claude_cli_auth_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_select, "find_claude_cli", lambda: "/fake/bin/claude")
    # Empty env: no ANTHROPIC_API_KEY. The CLI-auth path must still build.
    session, kind = build_engine({})
    assert kind == "claude"
    assert isinstance(session, ClaudeEngine)


def test_build_engine_claude_missing_cli_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_select, "find_claude_cli", lambda: None)
    with pytest.raises(EngineSelectionError) as excinfo:
        build_engine({})
    assert "CLI" in str(excinfo.value)


def test_build_engine_unknown_kind_fails() -> None:
    with pytest.raises(EngineSelectionError):
        build_engine({"E2E_ENGINE": "bogus"})


def test_build_engine_honors_base_url_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_select, "find_claude_cli", lambda: "/fake/bin/claude")
    session, kind = build_engine({"ANTHROPIC_BASE_URL": "http://x:4000/v1"})
    assert kind == "claude"
    assert isinstance(session, ClaudeEngine)


def test_build_engine_codex_cli_auth_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from rtech_agentd.codex_engine import CodexEngine

    monkeypatch.setattr(engine_select, "find_codex_cli", lambda: "/fake/bin/codex")
    # Empty env: no OPENAI_API_KEY. The CLI-auth path must still build.
    session, kind = build_engine({"E2E_ENGINE": "codex"})
    assert kind == "codex"
    assert isinstance(session, CodexEngine)
    assert isinstance(session, EngineSession)


def test_default_codex_model_is_not_the_retired_tier() -> None:
    """The safe default must be a supported tier, never retired gpt-5-codex.

    gpt-5-codex is not supported on a ChatGPT-account app-server and 400s every
    turn, so a fallback landing on it is itself broken. The default is the
    GPT-5.6 Sol tier.
    """
    assert engine_select.DEFAULT_CODEX_MODEL == "gpt-5.6-sol"
    assert engine_select.DEFAULT_CODEX_MODEL != "gpt-5-codex"


def test_build_engine_codex_uses_supported_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no E2E_MODEL override, the codex engine config carries the safe default."""
    monkeypatch.setattr(engine_select, "find_codex_cli", lambda: "/fake/bin/codex")
    session, kind = build_engine({"E2E_ENGINE": "codex"})
    assert kind == "codex"
    # The engine's config model is the supported default, not the retired tier.
    assert session._config.model == "gpt-5.6-sol"  # type: ignore[attr-defined]


def test_build_engine_codex_honors_e2e_model_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit E2E_MODEL still governs the codex config (spec-driven routing)."""
    monkeypatch.setattr(engine_select, "find_codex_cli", lambda: "/fake/bin/codex")
    session, _ = build_engine({"E2E_ENGINE": "codex", "E2E_MODEL": "gpt-5.6-sol"})
    assert session._config.model == "gpt-5.6-sol"  # type: ignore[attr-defined]


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_build_engine_codex_blank_model_resolves_to_default(
    monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    """A blank/whitespace E2E_MODEL must NOT reach the engine as an empty model.

    An empty configured model silently disables model attestation (the
    start/resume check no-ops when no model is requested), so a blank override is
    normalized back to the supported default and enforcement always governs.
    """
    monkeypatch.setattr(engine_select, "find_codex_cli", lambda: "/fake/bin/codex")
    session, _ = build_engine({"E2E_ENGINE": "codex", "E2E_MODEL": blank})
    assert session._config.model == engine_select.DEFAULT_CODEX_MODEL  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "bad_model",
    [
        "gpt\n5.6-sol",  # embedded newline (log-injection shape)
        "gpt\x00sol",  # control byte
        "has spaces in it",  # whitespace is not a model slug
        "a" * 100,  # oversized (over the 64-char cap)
        " gpt-5.6-sol",  # leading whitespace — raw padded value, not stripped
        "gpt-5.6-sol ",  # trailing whitespace
        "gpt-5.6-sol\n",  # trailing newline (would strip to a valid slug)
        "\tgpt-5.6-sol",  # leading tab
    ],
)
def test_build_engine_codex_malformed_model_resolves_to_default(
    monkeypatch: pytest.MonkeyPatch, bad_model: str
) -> None:
    """An untrusted, malformed E2E_MODEL is sanitized at the config boundary.

    A non-blank but malformed override (oversized / newline / control bytes / any
    leading or trailing whitespace) must never reach the engine (and thus a log
    line at CodexEngine.start). It fails closed to the supported default rather
    than flowing through raw or being silently trimmed into an accepted slug.
    """
    monkeypatch.setattr(engine_select, "find_codex_cli", lambda: "/fake/bin/codex")
    session, _ = build_engine({"E2E_ENGINE": "codex", "E2E_MODEL": bad_model})
    assert session._config.model == engine_select.DEFAULT_CODEX_MODEL  # type: ignore[attr-defined]
    # The raw untrusted value never becomes the configured model.
    assert session._config.model != bad_model  # type: ignore[attr-defined]


def test_build_engine_codex_missing_cli_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_select, "find_codex_cli", lambda: None)
    with pytest.raises(EngineSelectionError) as excinfo:
        build_engine({"E2E_ENGINE": "codex"})
    assert "codex" in str(excinfo.value).lower()


def test_attach_wake_on_non_claude_engine_fails_actionably(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_attach_wake with a wake service + a non-Claude engine fails loud.

    Wake is Claude-engine-only in this slice; a triage deploy that set the
    allow-list on a codex engine must fail closed with an actionable message
    (unset RTECH_WAKE_ALLOWED_AGENTS, or select the claude engine) rather than
    silently drop the capability.
    """
    from rtech_agentd.__main__ import _attach_wake
    from rtech_agentd.codex_engine import CodexEngine
    from rtech_agentd.wake_service import WakeConfig, WakeService

    monkeypatch.setattr(engine_select, "find_codex_cli", lambda: "/fake/bin/codex")
    codex_engine, _kind = build_engine({"E2E_ENGINE": "codex"})
    assert isinstance(codex_engine, CodexEngine)

    wake_service = WakeService(
        config=WakeConfig(
            triage_agent="agentd-nimbus",
            allowed_agents=frozenset({"agentd-birk"}),
            stack="dev-stack",
        ),
        bus=object(),  # type: ignore[arg-type]
        origin_provider=lambda: None,
    )

    with pytest.raises(EngineSelectionError) as excinfo:
        _attach_wake(codex_engine, wake_service)
    message = str(excinfo.value)
    assert "RTECH_WAKE_ALLOWED_AGENTS" in message
    assert "claude" in message.lower()
