"""Regression tests for the shipped repo personas.

Iterate every agent directory under the repo ``agents/`` tree, assert each
persona loads cleanly through :func:`load_persona`, has a non-empty SOUL, and
carries no secret-shaped content. Forbidden patterns are assembled from
fragments so a plain grep of this test file stays clean.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from rtech_persona.loader import Persona, load_persona

# Repo root: .../packages/rtech-persona/tests/ -> up three -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_AGENTS_DIR = _REPO_ROOT / "agents"

# Expected shipped agent homes (each must load).
_EXPECTED_AGENTS = ("_skeleton",)  # real agents are deployment config, not shipped

# Persona filenames scanned for secret-shaped content.
_PERSONA_FILES = ("SOUL.md", "USER.md", "KERNEL.md")

# Discord snowflakes are 17-19 digit runs. Assembled to keep greps clean.
_SNOWFLAKE = re.compile(r"[0-9]{" + "17" + "," + "19" + "}")

# Webhook URL infix, assembled from fragments.
_WEBHOOK = "discord" + ".com/" + "api/" + "webhooks"

# Credential-shaped prefixes, assembled via concatenation. Substring-matched
# against lowercased content, so each fragment is specific enough to avoid
# colliding with ordinary prose (e.g. bare "sk-" would hit "task-id").
_CRED_PREFIXES: tuple[str, ...] = (
    "gh" + "p_",
    "github" + "_pat_",
    "xo" + "xb-",
    "xo" + "xp-",
    "sk" + "_live_",
    "sk" + "_test_",
)

# OpenAI-style secret keys: "sk-" plus a long token body. A dedicated pattern
# (not a bare "sk-" substring) so ordinary hyphenated words do not match.
_OPENAI_KEY = re.compile(r"sk-[A-Za-z0-9_-]{" + "20" + ",}")

# PII/infra shapes the redaction pass removed — pinned so they cannot
# regress while CI stays green:
# email addresses, US phone shapes, RFC1918/internal IPs, and long unbroken
# base64/hex blobs (key or hash material; no persona word is that shape).
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"\+?1?[-. (]{0,2}[0-9]{3}[-. )]{1,2}[0-9]{3}[-. ]{1}[0-9]{4}")
_INTERNAL_IP = re.compile(
    r"\b(?:10|127|192\.168|172\.(?:1[6-9]|2[0-9]|3[01]))(?:\.[0-9]{1,3}){2,3}\b"
)
_BLOB = re.compile(r"[A-Za-z0-9+/=]{" + "40" + ",}")


def _agent_homes() -> list[Path]:
    """Return the sorted list of agent home directories under ``agents/``."""
    return sorted(p for p in _AGENTS_DIR.iterdir() if p.is_dir())


def test_agents_dir_exists() -> None:
    assert _AGENTS_DIR.is_dir(), f"agents dir missing: {_AGENTS_DIR}"


def test_expected_agents_present() -> None:
    names = {p.name for p in _agent_homes()}
    for expected in _EXPECTED_AGENTS:
        assert expected in names, f"expected agent home missing: {expected}"


@pytest.mark.parametrize("home", _agent_homes(), ids=lambda p: p.name)
def test_shipped_persona_loads(home: Path) -> None:
    persona = load_persona(home)
    assert isinstance(persona, Persona)
    assert persona.soul.strip(), f"SOUL is empty for {home.name}"


@pytest.mark.parametrize("home", _agent_homes(), ids=lambda p: p.name)
def test_shipped_persona_has_no_secrets(home: Path) -> None:
    for filename in _PERSONA_FILES:
        path = home / filename
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")

        snowflake = _SNOWFLAKE.search(content)
        assert snowflake is None, f"{home.name}/{filename}: snowflake-shaped digit run present"

        assert _WEBHOOK not in content, f"{home.name}/{filename}: webhook URL present"

        lowered = content.lower()
        for prefix in _CRED_PREFIXES:
            assert prefix not in lowered, (
                f"{home.name}/{filename}: credential-shaped prefix present"
            )

        assert _OPENAI_KEY.search(content) is None, (
            f"{home.name}/{filename}: OpenAI-shaped secret key present"
        )

        assert _EMAIL.search(content) is None, f"{home.name}/{filename}: email address present"
        assert _PHONE.search(content) is None, f"{home.name}/{filename}: phone-shaped run present"
        assert _INTERNAL_IP.search(content) is None, (
            f"{home.name}/{filename}: internal IP address present"
        )
        assert _BLOB.search(content) is None, (
            f"{home.name}/{filename}: base64/hex blob-shaped run present"
        )
