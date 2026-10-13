"""Import smoke test — the package entrypoint loads without error."""

from __future__ import annotations

import rtech_persona


def test_public_api_present() -> None:
    assert rtech_persona.__version__
    assert callable(rtech_persona.load_persona)
    assert callable(rtech_persona.compose)
