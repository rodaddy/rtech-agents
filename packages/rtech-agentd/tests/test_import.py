"""Import smoke test — the package entrypoint loads without error."""

from __future__ import annotations

import rtech_agentd


def test_version_present() -> None:
    assert rtech_agentd.__version__
