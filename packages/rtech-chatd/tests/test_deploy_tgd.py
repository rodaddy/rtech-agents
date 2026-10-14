"""Deploy-artifact consistency tests for the standalone Telegram edge.

These assert the OFFLINE deploy artifacts agree with each other and with the
code, and that no literal secret leaks into a committed file:

* ``rtech-tgd.service`` is one explicit (non-templated) user unit that runs the
  ``rtech-tgd`` console script, with restart + start-limit backoff, an
  EnvironmentFile, and the network-online ordering + default.target install the
  other units use.
* ``install.sh`` COPIES the unit but never ``enable --now``s it (operator opt-in,
  like ``rtech-chatd.service``).
* The env template names only ``*_ref`` indirections and public chat ids — never
  a literal token/account/member id or auth JSON — and every var it sets is one
  the code actually reads.
* The console entrypoint declared in ``pyproject.toml`` resolves to a real
  callable.

No network, no host mutation: every check reads repo files or imports a module.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
KBOARD = REPO_ROOT / "deploy" / "kboard"
TGD_UNIT = KBOARD / "rtech-tgd.service"
CHATD_UNIT = KBOARD / "rtech-chatd.service"
INSTALL_SH = KBOARD / "install.sh"
README = KBOARD / "README.md"
CANARY = KBOARD / "watcher-canary.md"
TGD_ENV = KBOARD / "env" / "watcher-tgd.env.example"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --- the unit exists and is a well-formed opt-in edge ------------------------


def test_tgd_unit_exists() -> None:
    assert TGD_UNIT.is_file(), f"missing unit file {TGD_UNIT}"


def test_tgd_unit_is_not_a_template() -> None:
    # A %i-template would collapse the multi-identity edge into N racing
    # single-identity processes. It must be ONE explicit unit.
    assert "@" not in TGD_UNIT.name
    assert "%i" not in _read(TGD_UNIT)


def test_tgd_unit_runs_the_tgd_console_script() -> None:
    text = _read(TGD_UNIT)
    exec_lines = [ln for ln in text.splitlines() if ln.startswith("ExecStart=")]
    assert len(exec_lines) == 1
    assert exec_lines[0].strip().endswith("rtech-tgd"), exec_lines[0]
    assert "uv run --frozen rtech-tgd" in exec_lines[0]


def test_tgd_unit_has_restart_and_backoff() -> None:
    text = _read(TGD_UNIT)
    assert "Restart=on-failure" in text
    assert "RestartSec=" in text
    # Start-limit backoff bounds a fail-closed crash loop.
    assert "StartLimitIntervalSec=" in text
    assert "StartLimitBurst=" in text


def test_tgd_unit_has_environmentfile_and_ordering() -> None:
    text = _read(TGD_UNIT)
    assert "EnvironmentFile=%h/rtech-agents/.env" in text
    assert "After=network-online.target" in text
    assert "Wants=network-online.target" in text
    assert "WantedBy=default.target" in text


# --- install.sh: copied, never auto-enabled ---------------------------------


def test_install_copies_the_tgd_unit() -> None:
    text = _read(INSTALL_SH)
    # It appears in the install loop's unit list.
    assert "rtech-tgd.service" in text
    install_loop = re.search(r"for unit in ([^\n]*rtech-tgd\.service[^\n]*)", text)
    assert install_loop is not None, "rtech-tgd.service not in the install unit loop"


def test_install_does_not_enable_the_tgd_unit() -> None:
    text = _read(INSTALL_SH)
    # No enable/start of tgd anywhere — it is an operator opt-in like chatd.
    assert not re.search(r"enable\s+--now\s+rtech-tgd", text)
    assert not re.search(r"systemctl[^\n]*start[^\n]*rtech-tgd", text)


def test_tgd_opt_in_matches_chatd_convention() -> None:
    # Both edges are copied-not-enabled; the chatd unit is the precedent.
    text = _read(INSTALL_SH)
    assert "rtech-chatd.service" in text
    assert not re.search(r"enable\s+--now\s+rtech-chatd", text)


# --- env template: refs + public ids only, no literal secrets ---------------

# A bot token is `<digits>:<35-char base64ish>` — the shape we must never see as a
# literal in a committed file. account/member ids are opaque; auth JSON has a
# telltale key set.
_BOT_TOKEN_LITERAL = re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{30,}\b")
_AUTH_JSON_HINT = re.compile(r'"(client_secret|private_key|refresh_token|api_key)"\s*:')


def _uncommented_assignments(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def test_tgd_env_template_exists() -> None:
    assert TGD_ENV.is_file(), f"missing env template {TGD_ENV}"


def test_tgd_env_token_refs_are_indirections_only() -> None:
    env = _uncommented_assignments(_read(TGD_ENV))
    # Every *_TOKEN_REF assignment set (uncommented) must be an env: indirection,
    # never a literal or an unsupported resolver scheme.
    for key, value in env.items():
        if key.endswith("_TOKEN_REF"):
            assert value.startswith("env:"), f"{key}={value!r} is not an env ref"
    # The identity map's values are all refs too.
    identity = env.get("RTECH_TG_IDENTITY_MAP", "")
    for pair in (p for p in identity.split(",") if p.strip()):
        _, _, ref = pair.partition("=")
        assert ref.strip().startswith("env:"), f"identity ref not indirection: {pair!r}"


def test_tgd_env_has_no_literal_secret() -> None:
    text = _read(TGD_ENV)
    assert _BOT_TOKEN_LITERAL.search(text) is None, (
        "a literal bot token shape leaked into the env template"
    )
    assert _AUTH_JSON_HINT.search(text) is None, "auth JSON leaked into the env template"
    # Any actual secret env var (the ref target) must be left blank/commented.
    env = _uncommented_assignments(text)
    for key in ("RTECH_CHATD_BOT_TOKEN",):
        assert key not in env, f"{key} must be left commented/blank in a committed template"


def test_tgd_env_requires_group_allowlist() -> None:
    env = _uncommented_assignments(_read(TGD_ENV))
    # The allowlist is required by the code; the template must set it (public id).
    assert "RTECH_TG_ALLOWED_GROUPS" in env
    assert env["RTECH_TG_ALLOWED_GROUPS"], "group allowlist must be non-empty in the template"


def test_tgd_env_vars_are_all_read_by_code() -> None:
    # Every var the template sets (uncommented) must be one the tgd/config code
    # actually reads — no orphan knobs that silently do nothing.
    from rtech_chatd import config as cfg
    from rtech_chatd import telegram_service as tgd

    sources = _read(Path(cfg.__file__)) + _read(Path(tgd.__file__))
    env = _uncommented_assignments(_read(TGD_ENV))
    # The refs' target vars (RTECH_CHATD_BOT_TOKEN, RTECH_TG_TOKEN_*) are resolved
    # dynamically by name via resolve_token, so exempt the token VALUE vars.
    exempt = {k for k in env if k.startswith("RTECH_TG_TOKEN_") or k == "RTECH_CHATD_BOT_TOKEN"}
    for key in env:
        if key in exempt:
            continue
        assert key in sources, f"{key} is set in the template but read nowhere in the code"


# --- console entrypoint declared in pyproject resolves -----------------------


def test_pyproject_declares_rtech_tgd_entrypoint() -> None:
    pyproject = REPO_ROOT / "packages" / "rtech-chatd" / "pyproject.toml"
    data = tomllib.loads(_read(pyproject))
    scripts = data["project"]["scripts"]
    assert scripts.get("rtech-tgd") == "rtech_chatd.tgd_main:main"


def test_rtech_tgd_entrypoint_is_callable() -> None:
    from rtech_chatd.tgd_main import main

    assert callable(main)


# --- docs mention the unit and the no-ingress fact ---------------------------


def test_readme_documents_the_tgd_edge() -> None:
    text = _read(README)
    assert "rtech-tgd.service" in text
    # Long polling means no inbound port/webhook/domain — the fact.
    assert re.search(r"no inbound port", text, re.IGNORECASE)


def test_canary_has_telegram_diagnosis_table() -> None:
    text = _read(CANARY)
    assert "rtech-tgd" in text
    assert "What deviations mean (Telegram)" in text


@pytest.mark.parametrize("doc", [README, CANARY, TGD_ENV])
def test_docs_have_no_literal_secret(doc: Path) -> None:
    text = _read(doc)
    assert _BOT_TOKEN_LITERAL.search(text) is None, f"literal token shape in {doc.name}"
    assert _AUTH_JSON_HINT.search(text) is None, f"auth JSON in {doc.name}"
