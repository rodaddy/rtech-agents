"""Confirm the package entrypoint loads without error."""

from __future__ import annotations


def test_package_imports() -> None:
    import rtech_hermes_bridge

    assert rtech_hermes_bridge.__version__


def test_main_entrypoint_importable() -> None:
    from rtech_hermes_bridge.__main__ import main

    assert callable(main)


def test_public_modules_import() -> None:
    from rtech_hermes_bridge.config import BridgeConfig
    from rtech_hermes_bridge.heartbeat import BridgeHeartbeat
    from rtech_hermes_bridge.runtime import HermesBridge, build_buses
    from rtech_hermes_bridge.translate import build_content_turn, build_inbound_envelope

    assert BridgeConfig
    assert BridgeHeartbeat
    assert HermesBridge
    assert build_buses
    assert build_content_turn
    assert build_inbound_envelope
