"""Pin the __main__ exit-code mapping for a circuit-open death (#84 slice A).

When the supervisor's restart budget is spent, ``Supervisor.run`` raises
``CircuitOpenError``; the entrypoint must map that fatal marker to a NON-ZERO
exit so systemd's ``Restart=on-failure`` relaunches the unit. A clean operator
stop (``run`` returns) must stay ``0``.

The entrypoint is coupled to the live bus, so the collaborators (engine
selection, bus, runner, supervisor) are stubbed in the ``__main__`` module
namespace and only the exit-code decision is exercised at the seam.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import rtech_agentd.__main__ as main_mod
from rtech_agentd.supervisor import CircuitOpenError


class _FakeBus:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def publish(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _FakeRunner:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def start(self) -> None:
        return None

    def current_origin(self) -> None:
        # The wake wiring (#93) reads the runner's active origin; this stub agent
        # is not a waker, so it always reports "no origin".
        return None


class _FakeSupervisor:
    """A supervisor stub whose ``run`` outcome is controlled per test."""

    def __init__(self, *, circuit: bool, **_kwargs: Any) -> None:
        self._circuit = circuit
        self.stopped = False

    async def start(self) -> None:
        return None

    async def run(self) -> None:
        # Yield so the entrypoint's asyncio.wait sees a running task first.
        await asyncio.sleep(0)
        if self._circuit:
            raise CircuitOpenError("budget spent")

    async def stop(self, **_kwargs: Any) -> None:
        self.stopped = True


def _patch_common(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, circuit: bool) -> None:
    monkeypatch.setenv("RTECH_STATE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("RTECH_AGENT_ID", "tester")  # canonical NATS identity token
    monkeypatch.setattr(main_mod, "FleetBus", _FakeBus)
    monkeypatch.setattr(main_mod, "FrameTurnRunner", _FakeRunner)
    monkeypatch.setattr(main_mod, "build_engine", lambda _env=None: (object(), "scripted"))
    monkeypatch.setattr(
        main_mod,
        "Supervisor",
        lambda **kwargs: _FakeSupervisor(circuit=circuit, **kwargs),
    )


@pytest.mark.asyncio
async def test_circuit_open_run_maps_to_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run() raising CircuitOpenError -> non-zero exit for the systemd outer loop."""
    _patch_common(monkeypatch, tmp_path, circuit=True)
    exit_code = await main_mod._run()
    assert exit_code == 1


@pytest.mark.asyncio
async def test_clean_run_return_maps_to_zero_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run() returning cleanly (operator stop) -> exit 0, not restarted."""
    _patch_common(monkeypatch, tmp_path, circuit=False)
    exit_code = await main_mod._run()
    assert exit_code == 0
