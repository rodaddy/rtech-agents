"""TelegramUserConfig loader, the .example template, and the secrets gitignore.

Covers the fail-loud secrets loader: a valid round-trip with the
string→int api_id coercion, every ACTION REQUIRED failure path (missing file,
placeholder/blank api_id or api_hash), the session_string default, plus two
rot-guards — the committed ``.example`` stays parseable with recognisable
placeholders, and ``.gitignore`` excludes ``secrets/`` while allowlisting the
template. The interactive Telethon login is NOT tested here (a real phone is
required); only the pure banner + persist seams are covered in
``test_mint_session.py``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from rtech_chatd.user_session.config import (
    EXAMPLE_TEMPLATE_PATH,
    PLACEHOLDER_API_HASH,
    PLACEHOLDER_API_ID,
    JsonFileSource,
    TelegramUserConfig,
    load_telegram_user_config,
    secrets_path_for_env,
)

# Repo root: tests/ -> rtech-chatd/ -> packages/ -> <root>.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_secrets(path: Path, **overrides: object) -> Path:
    """Write a telegram_user.json at ``path`` with sensible valid defaults."""
    data: dict[str, object] = {
        "api_id": "1234567",
        "api_hash": "0123456789abcdef0123456789abcdef",
        "session_string": "",
    }
    data.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --------------------------------------------------------------- happy path


def test_from_secrets_file_loads_and_coerces_api_id(tmp_path: Path) -> None:
    path = _write_secrets(tmp_path / "telegram_user.json", api_id="987654")
    cfg = TelegramUserConfig.from_secrets_file(path)
    assert cfg.api_id == 987654  # coerced from the JSON string
    assert isinstance(cfg.api_id, int)
    assert cfg.api_hash == "0123456789abcdef0123456789abcdef"
    assert cfg.session_string == ""


def test_from_secrets_file_roundtrips_session_string(tmp_path: Path) -> None:
    path = _write_secrets(tmp_path / "telegram_user.json", session_string="1AsELg==")
    assert TelegramUserConfig.from_secrets_file(path).session_string == "1AsELg=="


def test_session_string_defaults_to_empty_when_absent(tmp_path: Path) -> None:
    path = tmp_path / "telegram_user.json"
    valid_hash = "0123456789abcdef0123456789abcdef"  # 32 hex chars
    path.write_text(json.dumps({"api_id": "42", "api_hash": valid_hash}), encoding="utf-8")
    assert TelegramUserConfig.from_secrets_file(path).session_string == ""


# --------------------------------------------------------------- fail-loud


def test_missing_file_raises_file_not_found_with_action_required(tmp_path: Path) -> None:
    missing = tmp_path / "nope" / "telegram_user.json"
    with pytest.raises(FileNotFoundError, match="ACTION REQUIRED"):
        TelegramUserConfig.from_secrets_file(missing)


def test_placeholder_api_id_raises(tmp_path: Path) -> None:
    path = _write_secrets(tmp_path / "telegram_user.json", api_id=PLACEHOLDER_API_ID)
    with pytest.raises(ValueError, match="api_id"):
        TelegramUserConfig.from_secrets_file(path)


def test_blank_api_id_raises(tmp_path: Path) -> None:
    path = _write_secrets(tmp_path / "telegram_user.json", api_id="")
    with pytest.raises(ValueError, match="api_id"):
        TelegramUserConfig.from_secrets_file(path)


def test_placeholder_api_hash_raises(tmp_path: Path) -> None:
    path = _write_secrets(tmp_path / "telegram_user.json", api_hash=PLACEHOLDER_API_HASH)
    with pytest.raises(ValueError, match="api_hash"):
        TelegramUserConfig.from_secrets_file(path)


def test_blank_api_hash_raises(tmp_path: Path) -> None:
    path = _write_secrets(tmp_path / "telegram_user.json", api_hash="  ")
    with pytest.raises(ValueError, match="api_hash"):
        TelegramUserConfig.from_secrets_file(path)


@pytest.mark.parametrize(
    "placeholder",
    [
        "PASTE_YOUR_API_HASH_HERE",  # a DIFFERENT wording than the sentinel
        "YOUR_TELEGRAM_HASH",
        "<your-api-hash>",
        "paste-hash-here",
    ],
)
def test_placeholder_wording_variants_raise(tmp_path: Path, placeholder: str) -> None:
    # Any un-filled placeholder must be rejected, not just the exact sentinel —
    # else it sails through to Telegram and fails cryptically at connect time.
    path = _write_secrets(tmp_path / "telegram_user.json", api_hash=placeholder)
    with pytest.raises(ValueError, match="api_hash"):
        TelegramUserConfig.from_secrets_file(path)


@pytest.mark.parametrize(
    "bad_hash",
    [
        "deadbeef",  # 8 chars — too short
        "0123456789abcdef0123456789abcde",  # 31 chars — off by one
        "0123456789abcdef0123456789abcdef0",  # 33 chars — too long
        "0123456789abcdef0123456789abcdeZ",  # 32 chars but non-hex (Z)
    ],
)
def test_non_32_hex_api_hash_raises(tmp_path: Path, bad_hash: str) -> None:
    # A real Telegram api_hash is EXACTLY 32 hex chars; anything else is not a
    # credential and must fail loud at load, not at connect time.
    path = _write_secrets(tmp_path / "telegram_user.json", api_hash=bad_hash)
    with pytest.raises(ValueError, match="api_hash"):
        TelegramUserConfig.from_secrets_file(path)


def test_non_integer_api_id_raises(tmp_path: Path) -> None:
    path = _write_secrets(tmp_path / "telegram_user.json", api_id="not-a-number")
    with pytest.raises(ValueError, match="api_id"):
        TelegramUserConfig.from_secrets_file(path)


def test_post_init_rejects_nonpositive_api_id() -> None:
    with pytest.raises(ValueError, match="api_id"):
        TelegramUserConfig(api_id=0, api_hash="deadbeef")


# --------------------------------------------------------------- seam / loader


def test_secrets_path_for_env_uses_local_tree(tmp_path: Path) -> None:
    path = secrets_path_for_env("prod", root=tmp_path)
    assert path == tmp_path / "secrets" / "local" / "prod" / "telegram_user.json"


@pytest.mark.parametrize(
    "evil_env",
    [
        "../examples",  # escape into the git-trackable examples tree
        "..",
        "dev/../prod",
        "a/b",  # any path separator
        "",  # blank
        "dev ",  # trailing space is not a bare slug
        "a" * 65,  # over the length cap
    ],
)
def test_secrets_path_for_env_rejects_traversal(tmp_path: Path, evil_env: str) -> None:
    # An env slug must never escape secrets/local/<env>/ — else --env ../examples
    # --write persists a real credential into the committed examples tree.
    with pytest.raises(ValueError, match="env slug"):
        secrets_path_for_env(evil_env, root=tmp_path)


@pytest.mark.parametrize("bad_json", ["[]", "42", '"just-a-string"', "null"])
def test_non_object_json_raises_action_required(tmp_path: Path, bad_json: str) -> None:
    # A syntactically valid but non-object JSON must fail loud with ACTION
    # REQUIRED, not a cryptic AttributeError on ``.get``.
    path = tmp_path / "telegram_user.json"
    path.write_text(bad_json, encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        TelegramUserConfig.from_secrets_file(path)


def test_load_via_json_file_source(tmp_path: Path) -> None:
    _write_secrets(secrets_path_for_env("dev", root=tmp_path), api_id="777")
    cfg = load_telegram_user_config("dev", source=JsonFileSource(root=tmp_path))
    assert cfg.api_id == 777


# --------------------------------------------------------------- template rot-guard


@pytest.mark.skip(reason="the secrets/ tree is excluded from the public export")
def test_example_template_exists_and_parses() -> None:
    template = _REPO_ROOT / EXAMPLE_TEMPLATE_PATH
    assert template.is_file(), f"missing committed template: {template}"
    data = json.loads(template.read_text(encoding="utf-8"))
    # Placeholders must be the recognisable sentinels (so the loader rejects an
    # un-filled copy) and session_string must ship blank.
    assert data["api_id"] == PLACEHOLDER_API_ID
    assert data["api_hash"] == PLACEHOLDER_API_HASH
    assert data["session_string"] == ""
    # A pointer to my.telegram.org/apps must survive in the template.
    blob = template.read_text(encoding="utf-8")
    assert "my.telegram.org/apps" in blob


# --------------------------------------------------------------- gitignore guard


def test_gitignore_excludes_secrets_but_allowlists_examples() -> None:
    gitignore = (_REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "secrets/*" in gitignore
    assert "!secrets/examples/" in gitignore
    assert "*.session" in gitignore


def test_git_check_ignore_confirms_local_secret_ignored() -> None:
    # A real per-env secret path must be ignored; the template must NOT be.
    ignored = subprocess.run(
        ["git", "check-ignore", "secrets/local/dev/telegram_user.json"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert ignored.returncode == 0, "secrets/local/<env>/ must be gitignored"

    tracked = subprocess.run(
        ["git", "check-ignore", str(EXAMPLE_TEMPLATE_PATH)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert tracked.returncode == 1, "secrets/examples/ template must NOT be ignored"
