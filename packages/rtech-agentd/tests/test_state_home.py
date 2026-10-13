"""Cover agentd state-home resolution: env override + CWD-relative default.

The rootless deploy must never write session state to a root-only
``/opt`` path. The default resolves under the unit's ``WorkingDirectory``
(``%h/rtech-agents``), and an explicit ``RTECH_STATE_HOME`` overrides it.
"""

from __future__ import annotations

import os
from pathlib import Path

from rtech_agentd.__main__ import (
    DEFAULT_STATE_HOME,
    ENV_STATE_HOME,
    _resolve_state_home,
)


def test_default_is_cwd_relative(tmp_path: Path, monkeypatch) -> None:
    """Absent the env var, the default binds to CWD/state/agentd-home."""
    monkeypatch.delenv(ENV_STATE_HOME, raising=False)
    monkeypatch.chdir(tmp_path)
    resolved = _resolve_state_home()
    assert resolved == (tmp_path / DEFAULT_STATE_HOME).resolve()


def test_default_is_never_opt(tmp_path: Path, monkeypatch) -> None:
    """The default must not resolve to the old root-only /opt tree."""
    monkeypatch.delenv(ENV_STATE_HOME, raising=False)
    monkeypatch.chdir(tmp_path)
    assert not str(_resolve_state_home()).startswith("/opt/")


def test_env_override_wins(tmp_path: Path, monkeypatch) -> None:
    """An explicit RTECH_STATE_HOME is used and resolved to absolute."""
    target = tmp_path / "custom" / "home"
    monkeypatch.setenv(ENV_STATE_HOME, str(target))
    assert _resolve_state_home() == target.resolve()


def test_relative_override_binds_to_cwd(tmp_path: Path, monkeypatch) -> None:
    """A relative override resolves against the working directory."""
    monkeypatch.setenv(ENV_STATE_HOME, "state/x")
    monkeypatch.chdir(tmp_path)
    assert _resolve_state_home() == (tmp_path / "state" / "x").resolve()


def test_resolve_returns_absolute(monkeypatch) -> None:
    """The resolved path is always absolute regardless of input."""
    monkeypatch.setenv(ENV_STATE_HOME, "relative/dir")
    assert os.path.isabs(str(_resolve_state_home()))
