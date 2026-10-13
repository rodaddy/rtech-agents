"""Load the Telethon user-session credential from a gitignored per-env file.

The credential (``api_id`` / ``api_hash`` / ``session_string``) lives in a
gitignored JSON file, one per environment, under ``secrets/local/<env>/
telegram_user.json``. This module holds the frozen :class:`TelegramUserConfig`
that carries it and the :class:`SecretsSource` seam that reads it — the ONE place
a future Vaultwarden/GSM backend slots in without touching callers.

Fail-loud discipline (mirrors the repo's ``fail intelligently`` rule): a missing
file, or a file still holding the ``*.example`` placeholders, raises with an
ACTION REQUIRED remediation rather than returning a half-built config that only
explodes later at Telegram connect time.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from loguru import logger

# An env slug must be a bare name — no path separators, no ``..`` — so it can
# never escape secrets/local/<env>/ into a git-trackable tree (see
# ``secrets_path_for_env``). Allowlist beats blocklist for untrusted input.
_SAFE_ENV_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")

# The env-partitioned on-disk layout. The generator writes here and the launcher
# reads here; the tree is gitignored (see .gitignore) with only the committed
# ``secrets/examples/`` template surviving version control.
SECRETS_LOCAL_ROOT = Path("secrets") / "local"
SECRETS_FILENAME = "telegram_user.json"

# The committed template a fresh checkout copies into place. Its api_id/api_hash
# are the sentinel placeholders below, so a config that still holds them is an
# un-filled template, not a real credential.
EXAMPLE_TEMPLATE_PATH = Path("secrets") / "examples" / "telegram_user.json.example"

# Placeholder sentinels shipped in the .example. A value equal to one of these
# (or blank) means the operator copied the template but never filled it in.
PLACEHOLDER_API_ID = "YOUR_TELEGRAM_API_ID"
PLACEHOLDER_API_HASH = "YOUR_API_HASH"

# Where api_id/api_hash are obtained. Named in every ACTION REQUIRED message so
# the remediation is copy-pasteable, not a scavenger hunt.
TELEGRAM_APPS_URL = "https://my.telegram.org/apps"


def secrets_path_for_env(env: str, root: Path | None = None) -> Path:
    """Return the ``telegram_user.json`` path for ``env``.

    Args:
        env: Environment slug (e.g. ``dev``, ``prod``). Selects the per-env
            subdirectory under ``secrets/local/``.
        root: Optional override of the repo root the ``secrets/`` tree hangs off
            (tests point this at a tmp dir). Defaults to the current working
            directory, matching how the CLI is invoked from the repo root.

    Returns:
        The resolved path to the environment's secrets JSON file.

    Raises:
        ValueError: If ``env`` is not a bare slug (e.g. contains a path
            separator, ``..``, or is blank). This is the ONE chokepoint both the
            loader and the ``--write`` persister route through, so validating
            here stops an ``--env ../examples`` traversal from writing a real
            credential into the git-trackable ``secrets/examples/`` tree.
    """
    if not _SAFE_ENV_RE.fullmatch(env):
        raise ValueError(
            f"CRITICAL: env slug {env!r} is not a bare name. ACTION REQUIRED: "
            "use a simple slug like 'dev' or 'prod' (letters, digits, '-', '_' "
            "only) — no path separators or '..'."
        )
    base = (root or Path.cwd()) / SECRETS_LOCAL_ROOT
    return base / env / SECRETS_FILENAME


@dataclass(frozen=True)
class TelegramUserConfig:
    """Immutable Telethon user-session credential for one environment.

    ``api_id`` is stored in the JSON as a STRING (mirroring the b1x
    ``TestAccount`` convention) and int-coerced on load, so the on-disk shape is
    editor-friendly while the in-memory type is the ``int`` Telethon wants.

    Attributes:
        api_id: Telegram application id (from https://my.telegram.org/apps).
        api_hash: Telegram application hash (same source). Never logged.
        session_string: The minted Telethon ``StringSession`` value, or ``""``
            before a session has been minted for this env.

    Raises:
        ValueError: If ``api_id`` is non-positive or ``api_hash`` is blank —
            construction fails loud rather than yielding a config that only
            breaks at Telegram connect time.
    """

    api_id: int
    api_hash: str
    session_string: str = ""

    def __post_init__(self) -> None:
        if self.api_id <= 0:
            raise ValueError(
                "TelegramUserConfig.api_id must be a positive integer "
                f"(got {self.api_id!r}). ACTION REQUIRED: set api_id from "
                f"{TELEGRAM_APPS_URL}."
            )
        if not self.api_hash.strip():
            raise ValueError(
                "TelegramUserConfig.api_hash must be non-empty. ACTION REQUIRED: "
                f"set api_hash from {TELEGRAM_APPS_URL}."
            )

    @classmethod
    def from_secrets_file(cls, path: Path) -> TelegramUserConfig:
        """Load a config from a ``telegram_user.json`` file, failing loud.

        The JSON shape is ``{"api_id": "<string>", "api_hash": "...",
        "session_string": ""}``. ``api_id`` is int-coerced; ``session_string``
        defaults to ``""`` when absent. Placeholder or blank credentials are
        rejected with an ACTION REQUIRED remediation — a template copied but
        never filled must fail here, not silently at connect time.

        Args:
            path: Path to the secrets JSON file.

        Returns:
            A validated :class:`TelegramUserConfig`.

        Raises:
            FileNotFoundError: If ``path`` does not exist (names the copy-from
                template and the credential source).
            ValueError: If ``api_id`` / ``api_hash`` are missing, blank, still the
                ``*.example`` placeholders, or ``api_id`` is not an integer.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"CRITICAL: telegram_user secrets not found at {path}. "
                f"ACTION REQUIRED: copy {EXAMPLE_TEMPLATE_PATH} to {path} and "
                f"fill api_id/api_hash from {TELEGRAM_APPS_URL}."
            )

        logger.debug("TG_SESSION: loading user config from {}", path)
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)

        # A syntactically valid but non-object JSON (``[]``, ``42``, ``"x"``)
        # must fail loud here, not with a cryptic AttributeError on ``.get``.
        if not isinstance(data, dict):
            raise ValueError(
                f"CRITICAL: {path} must contain a JSON object, got "
                f"{type(data).__name__}. ACTION REQUIRED: restore the "
                f"{{'api_id': ..., 'api_hash': ..., 'session_string': ...}} shape "
                f"from {EXAMPLE_TEMPLATE_PATH}."
            )

        raw_api_id = str(data.get("api_id", "")).strip()
        api_hash = str(data.get("api_hash", "")).strip()

        if not raw_api_id or raw_api_id == PLACEHOLDER_API_ID:
            raise ValueError(
                f"CRITICAL: api_id is missing or still the placeholder in {path}. "
                f"ACTION REQUIRED: replace {PLACEHOLDER_API_ID!r} with your real "
                f"api_id from {TELEGRAM_APPS_URL}."
            )
        # Catch ANY unfilled placeholder, not just the exact template string —
        # a different placeholder wording (PASTE_..., YOUR_..., <...>) would
        # otherwise sail through and fail cryptically at Telegram
        # ("api_id/api_hash combination is invalid"). A real Telegram api_hash
        # is exactly 32 hex characters, so anything else is not a credential.
        looks_like_placeholder = (
            not api_hash
            or api_hash == PLACEHOLDER_API_HASH
            or api_hash.upper().startswith(("YOUR_", "PASTE"))
            or api_hash.startswith("<")
        )
        is_32_hex = len(api_hash) == 32 and all(c in "0123456789abcdefABCDEF" for c in api_hash)
        if looks_like_placeholder or not is_32_hex:
            raise ValueError(
                f"CRITICAL: api_hash in {path} is not a real credential (expected "
                f"32 hex characters, got {len(api_hash)}). ACTION REQUIRED: replace "
                f"it with your real api_hash from {TELEGRAM_APPS_URL}."
            )

        try:
            api_id = int(raw_api_id)
        except ValueError:
            raise ValueError(
                f"CRITICAL: api_id must be an integer string in {path} (got "
                f"{raw_api_id!r}). ACTION REQUIRED: set api_id from "
                f"{TELEGRAM_APPS_URL}."
            ) from None

        session_string = str(data.get("session_string", "") or "")
        return cls(api_id=api_id, api_hash=api_hash, session_string=session_string)


class SecretsSource(Protocol):
    """The storage seam: WHERE the user-session credential lives.

    One method, ``load(env)``. The whole point is that swapping the credential
    home (JSON file today; Vaultwarden/GSM later) is a new impl of THIS protocol,
    not a change at any call site. See :func:`load_telegram_user_config`.
    """

    def load(self, env: str) -> TelegramUserConfig:
        """Return the :class:`TelegramUserConfig` for ``env``."""
        ...


@dataclass(frozen=True)
class JsonFileSource:
    """Read the credential from a gitignored per-env JSON file (Option 1).

    This is the ONLY :class:`SecretsSource` today — a known-working dev pattern
    that avoids fighting a live secret store during development. The credential
    file is gitignored; only the ``secrets/examples/`` template is committed.

    Attributes:
        root: Optional repo-root override for the ``secrets/`` tree (tests point
            this at a tmp dir). ``None`` = current working directory.
    """

    root: Path | None = None

    def load(self, env: str) -> TelegramUserConfig:
        """Load the config for ``env`` from its ``telegram_user.json`` file."""
        path = secrets_path_for_env(env, root=self.root)
        return TelegramUserConfig.from_secrets_file(path)


def load_telegram_user_config(
    env: str = "dev", *, source: SecretsSource | None = None
) -> TelegramUserConfig:
    """Load the Telethon user-session config for ``env`` via the storage seam.

    This is the single entry point callers use; they never read the file
    directly. The active :class:`SecretsSource` is resolved here, so a future
    backend swap is one edit at this seam.

    Args:
        env: Environment slug (default ``dev``).
        source: Optional explicit source (dependency injection for tests /
            future backends). Defaults to :class:`JsonFileSource`.

    Returns:
        A validated :class:`TelegramUserConfig`.
    """
    # TODO(#110-secrets-backend): VaultwardenSource/GsmSource once VW proven fast.
    # Select the source here (e.g. by an env flag) — no caller changes needed:
    # every caller already goes through this one function and the SecretsSource
    # protocol, so a new impl slots in behind it. JSON file is the proven dev
    # default (Option 1); do NOT reach for VW/GSM until it is proven fast.
    active = source or JsonFileSource()
    return active.load(env)


__all__ = [
    "EXAMPLE_TEMPLATE_PATH",
    "PLACEHOLDER_API_HASH",
    "PLACEHOLDER_API_ID",
    "SECRETS_FILENAME",
    "SECRETS_LOCAL_ROOT",
    "TELEGRAM_APPS_URL",
    "JsonFileSource",
    "SecretsSource",
    "TelegramUserConfig",
    "load_telegram_user_config",
    "secrets_path_for_env",
]
