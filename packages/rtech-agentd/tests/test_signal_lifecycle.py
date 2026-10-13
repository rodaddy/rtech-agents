"""Cover the agentd signal -> shutdown-lifecycle initiator seam (#84).

The shutdown lifecycle frame names its initiator ``signal:<name>``. The signal
handler must stamp the triggering signal onto the shared ``_StopSignal`` so the
teardown path can pass that initiator to ``Supervisor.stop``.
"""

from __future__ import annotations

import asyncio
import signal

import pytest
from rtech_agentd.__main__ import _install_signal_handlers, _StopSignal


@pytest.mark.asyncio
async def test_signal_handler_records_signal_name() -> None:
    """A delivered signal stamps its name and sets the stop event."""
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    stop_signal = _StopSignal()

    captured: list[object] = []

    def _fake_add(sig: object, cb: object, *args: object) -> None:
        # Mirror loop.add_signal_handler(sig, cb, sig): record the SIGTERM wiring.
        if sig == signal.SIGTERM:
            captured.append((cb, args))

    # Drive the wiring without registering real OS handlers.
    loop.add_signal_handler = _fake_add  # type: ignore[method-assign]
    _install_signal_handlers(loop, stop_event, stop_signal)

    assert captured, "SIGTERM handler was not wired"
    cb, args = captured[0]  # type: ignore[misc]
    cb(*args)  # type: ignore[operator]

    assert stop_signal.name == "SIGTERM"
    assert stop_event.is_set()


@pytest.mark.asyncio
async def test_default_initiator_when_no_signal() -> None:
    """With no signal delivered, the holder stays empty (initiator=operator)."""
    stop_signal = _StopSignal()
    assert stop_signal.name is None
