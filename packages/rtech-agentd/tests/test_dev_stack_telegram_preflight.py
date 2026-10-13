"""Functional preflight tests for Telegram-backed local dev stack startup."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_DEV_STACK_PATH = Path(__file__).parents[3] / "scripts" / "dev_stack.py"


def _load_dev_stack() -> ModuleType:
    """Load the script as a module so its startup boundary is directly testable."""

    module_name = "rtech_agents_dev_stack_test"
    spec = importlib.util.spec_from_file_location(module_name, _DEV_STACK_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _set_telegram_debate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure a minimal Telegram debate edge with non-secret token refs."""

    monkeypatch.setenv("DEV_STACK_EDGE", "tg")
    monkeypatch.setenv("RTECH_CHATD_TOKEN_REF", "env:TG_EDGE_TOKEN")
    monkeypatch.setenv("RTECH_CHATD_INTAKE_ENABLED", "1")
    monkeypatch.setenv("RTECH_CHATD_TRIAGE_AGENT", "agentd-nagatha")
    monkeypatch.setenv("RTECH_CHATD_DEBATE_PARTICIPANTS", "agentd-nagatha,agentd-bilby")
    monkeypatch.setenv(
        "RTECH_TG_IDENTITY_MAP",
        "agentd-nagatha=env:TG_NAGATHA,agentd-bilby=env:TG_BILBY",
    )


def test_telegram_preflight_rejects_unmapped_debate_participant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every debate voice needs a Telegram identity before runtime construction."""

    _set_telegram_debate_env(monkeypatch)
    monkeypatch.setenv("RTECH_TG_IDENTITY_MAP", "agentd-nagatha=env:TG_NAGATHA")
    dev_stack = _load_dev_stack()

    with pytest.raises(ValueError, match="RTECH_TG_IDENTITY_MAP"):
        dev_stack.TelegramEdgeConfig.from_env()  # type: ignore[attr-defined]


def test_invalid_telegram_preflight_has_no_nats_or_agent_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bad Telegram config exits before reachability, bus, or supervisor startup."""

    _set_telegram_debate_env(monkeypatch)
    monkeypatch.setenv("RTECH_TG_IDENTITY_MAP", "agentd-nagatha=env:TG_NAGATHA")
    dev_stack = _load_dev_stack()
    calls: list[str] = []

    monkeypatch.setattr(
        dev_stack,
        "_nats_reachable",
        lambda _url: calls.append("nats-reachability") or True,
    )

    async def _unexpected_start(*_args: object) -> object:
        calls.append("runtime-start")
        raise AssertionError("invalid Telegram config must not start runtime roles")

    monkeypatch.setattr(dev_stack, "_connect_fleet_bus", _unexpected_start)
    monkeypatch.setattr(dev_stack, "_boot_agent", _unexpected_start)

    assert asyncio.run(dev_stack._run()) == 2  # type: ignore[attr-defined]
    assert calls == []


def test_invalid_telegram_transport_config_has_no_nats_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Telegram transport validation also runs before the first NATS operation."""

    _set_telegram_debate_env(monkeypatch)
    monkeypatch.setenv("RTECH_TG_POLL_TIMEOUT_S", "0")
    dev_stack = _load_dev_stack()
    calls: list[str] = []

    monkeypatch.setattr(
        dev_stack,
        "_nats_reachable",
        lambda _url: calls.append("nats-reachability") or True,
    )

    assert asyncio.run(dev_stack._run()) == 2  # type: ignore[attr-defined]
    assert calls == []


def test_valid_telegram_preflight_preserves_identity_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete identity map passes the same pure startup boundary unchanged."""

    _set_telegram_debate_env(monkeypatch)
    dev_stack = _load_dev_stack()

    config = dev_stack.TelegramEdgeConfig.from_env()  # type: ignore[attr-defined]

    assert config.intake_config.debate_participants == ("agentd-nagatha", "agentd-bilby")
    assert config.identity_refs == {
        "agentd-nagatha": "env:TG_NAGATHA",
        "agentd-bilby": "env:TG_BILBY",
    }


# --- DEV_STACK_AGENTS roster: subject-safety (F1) + state-home isolation (F4) ---


def test_agent_specs_default_uses_legacy_state_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env unset keeps ONE agent on the pre-roster ``.dev-stack/home`` (finding #4)."""

    monkeypatch.delenv("DEV_STACK_AGENTS", raising=False)
    dev_stack = _load_dev_stack()

    specs = dev_stack._agent_specs()  # type: ignore[attr-defined]

    assert len(specs) == 1
    # The legacy single agent persists to the pre-roster home so an existing
    # checkout's durable completed-task dedup is not orphaned.
    assert specs[0].home == dev_stack.LEGACY_HOME_DIR  # type: ignore[attr-defined]
    assert specs[0].engine_kind is None


def test_agent_specs_explicit_roster_isolates_per_agent_homes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit roster isolates every agent under ``.dev-stack/<id>`` (finding #4)."""

    monkeypatch.setenv("DEV_STACK_AGENTS", "agentd-nagatha-dev=claude,agentd-bilby-dev=codex")
    dev_stack = _load_dev_stack()

    specs = dev_stack._agent_specs()  # type: ignore[attr-defined]

    assert [s.agent_id for s in specs] == ["agentd-nagatha-dev", "agentd-bilby-dev"]
    home_root = dev_stack.HOME_ROOT  # type: ignore[attr-defined]
    assert specs[0].home == home_root / "agentd-nagatha-dev"
    assert specs[1].home == home_root / "agentd-bilby-dev"
    # Isolated homes never collide with the legacy single-agent home.
    legacy = dev_stack.LEGACY_HOME_DIR  # type: ignore[attr-defined]
    assert all(s.home != legacy for s in specs)


@pytest.mark.parametrize("bad_id", ["agentd.nagatha", "agentd-*", "agentd>x", "agentd nagatha"])
def test_agent_specs_rejects_unsafe_subject_ids(
    monkeypatch: pytest.MonkeyPatch,
    bad_id: str,
) -> None:
    """A roster id that is not a safe NATS subject token fails closed at parse (finding #1)."""

    monkeypatch.setenv("DEV_STACK_AGENTS", f"{bad_id}=claude")
    dev_stack = _load_dev_stack()

    with pytest.raises(ValueError, match="not a safe NATS subject token"):
        dev_stack._agent_specs()  # type: ignore[attr-defined]


def test_unsafe_roster_id_has_no_nats_or_engine_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsafe roster id exits 2 before engine build, reachability, or bus (finding #1)."""

    monkeypatch.setenv("DEV_STACK_AGENTS", "agentd-*=claude")
    monkeypatch.delenv("DEV_STACK_EDGE", raising=False)
    dev_stack = _load_dev_stack()
    calls: list[str] = []

    monkeypatch.setattr(
        dev_stack,
        "_build_agent_engine",
        lambda _spec: calls.append("engine-build"),
    )
    monkeypatch.setattr(
        dev_stack,
        "_nats_reachable",
        lambda _url: calls.append("nats-reachability") or True,
    )

    async def _unexpected(*_args: object) -> object:
        calls.append("runtime-start")
        raise AssertionError("an unsafe roster id must not start runtime roles")

    monkeypatch.setattr(dev_stack, "_connect_fleet_bus", _unexpected)
    monkeypatch.setattr(dev_stack, "_boot_agent", _unexpected)

    assert asyncio.run(dev_stack._run()) == 2  # type: ignore[attr-defined]
    assert calls == []
