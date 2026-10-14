"""Pure seams of the session generator: banner rendering and JSON persistence.

The interactive Telethon login (:func:`rtech_chatd.mint_session.mint_session`) is
NOT tested — it needs a real phone/code/2FA. Only the pure, network-free seams
are covered: the security banner text and the ``--write`` merge that preserves
api_id/api_hash while rewriting session_string.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rtech_chatd.mint_session import persist_session_string, render_session_banner


def test_banner_never_contains_the_session_string() -> None:
    # The banner is written to the loguru STDERR sink, which terminal/system log
    # capture may record — so it must NEVER embed the raw session. Only a
    # redacted fingerprint (length + short tail) is allowed. The raw value goes
    # to STDOUT only. Regression guard for the gauntlet CRITICAL.
    secret = "1AsELg==this-is-the-full-secret-session-value=="
    banner = render_session_banner(secret)
    assert secret not in banner
    # A generous middle slice must not leak either (guards against a partial echo).
    assert secret[6:30] not in banner
    # The redacted fingerprint and the security note still surface.
    assert "len=" in banner
    assert "ONE IP address at a time" in banner
    assert "gitignored secrets file" in banner


def test_banner_handles_empty_session_without_leaking() -> None:
    banner = render_session_banner("")
    assert "EMPTY" in banner


def test_persist_merges_session_and_preserves_credentials(tmp_path: Path) -> None:
    path = tmp_path / "telegram_user.json"
    path.write_text(
        json.dumps({"api_id": "42", "api_hash": "deadbeef", "session_string": ""}),
        encoding="utf-8",
    )
    persist_session_string(path, "1MINTED==")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["session_string"] == "1MINTED=="
    assert data["api_id"] == "42"  # preserved untouched (still a string)
    assert data["api_hash"] == "deadbeef"


def test_persist_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ACTION REQUIRED"):
        persist_session_string(tmp_path / "nope.json", "x")
