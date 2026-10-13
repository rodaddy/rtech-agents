"""User-session (Telethon StringSession) minting for the Telegram edge.

The chatd Telegram edge polls the Bot API with a bot token (see
:mod:`rtech_chatd.telegram_adapter`). Some surfaces — reading a user's own
dialogs, acting as a real Telegram *user* rather than a bot — need a Telethon
*user session* instead: an authenticated ``StringSession`` minted once via an
interactive phone/code/2FA login and then replayed on every connect. This
subpackage owns minting and loading that credential.

Architecture:
    The credential is split into two concerns so the interactive login and the
    at-rest storage evolve independently:

    * :mod:`rtech_chatd.user_session.config` holds the storage seam. A frozen
      :class:`TelegramUserConfig` carries ``api_id`` / ``api_hash`` /
      ``session_string``; :func:`load_telegram_user_config` reads it from a
      gitignored per-env JSON file via the ``SecretsSource`` seam. Today the only
      source is :class:`JsonFileSource` (a working, no-fight-with-VW dev pattern);
      a Vaultwarden/GSM source can slot behind the SAME seam later without
      touching callers.
    * :mod:`rtech_chatd.mint_session` is the interactive generator (the
      ``python -m rtech_chatd.mint_session`` / ``rtech-tg-session`` entry point).
      It reads ``api_id`` / ``api_hash`` from the config, runs the Telethon login
      (phone/code/2FA callbacks), prints the resulting ``StringSession``, and —
      with ``--write`` — persists it back into the same secrets JSON.

Key Components:
    - TelegramUserConfig: frozen dataclass; ``api_id`` (int, coerced from the
      string the JSON stores), ``api_hash``, ``session_string`` (``""`` until
      minted). Fails loud on a missing file or placeholder credentials.
    - SecretsSource: the one-method storage seam (a ``Protocol``). Swap the impl
      to change WHERE the credential lives without changing WHAT loads it.
    - JsonFileSource: the only impl today — a gitignored ``secrets/local/<env>/
      telegram_user.json`` file (Option 1: known-working dev pattern).
    - load_telegram_user_config: the module-level entry callers use; resolves the
      per-env path and delegates to the active source.

Pattern/Convention:
    Callers NEVER read the file themselves:

        >>> from rtech_chatd.user_session import load_telegram_user_config
        >>> cfg = load_telegram_user_config("dev")  # doctest: +SKIP
        >>> client = TelegramClient(StringSession(cfg.session_string),
        ...                         cfg.api_id, cfg.api_hash)  # doctest: +SKIP

    To move the secret to Vaultwarden/GSM later, implement a new
    ``SecretsSource`` and select it in :func:`load_telegram_user_config` — see the
    ``TODO(#110-secrets-backend)`` marker in
    :mod:`rtech_chatd.user_session.config`. No caller changes.

Example:
    Mint a session for the ``dev`` env and persist it back to the secrets file::

        uv run python -m rtech_chatd.mint_session --env dev --write

See Also:
    - rtech_chatd.mint_session: the interactive generator + CLI entry point.
    - rtech_chatd.telegram_adapter: the bot-token inbound edge (issue #108).
    - rtech_chatd.config.resolve_token: the parallel *bot*-token indirection seam.
"""

from __future__ import annotations

from .admin import (
    CreatedGroup,
    DialogInfo,
    connected_user_client,
    create_supergroup,
    ensure_forum,
    list_dialogs,
    promote_admins,
)
from .backfill import (
    DEFAULT_BACKFILL_LIMIT,
    MAX_BACKFILL_LIMIT,
    MAX_ENTITIES,
    MAX_REPLY_EDGES,
    BackfillResult,
    BackfillScopeError,
    EntityRecord,
    MessageRecord,
    OnEntity,
    OnMessage,
    OnReply,
    ReplyRecord,
    backfill_history,
    backfill_recent,
)
from .config import (
    JsonFileSource,
    SecretsSource,
    TelegramUserConfig,
    load_telegram_user_config,
    secrets_path_for_env,
)
from .provisioning import (
    DEFAULT_BOT_HANDLES,
    GroupAudit,
    InviteOutcome,
    ProvisioningError,
    ProvisionResult,
    audit_group,
    export_invite_link,
    find_supergroup_by_title,
    invite_members,
    promote_members,
    provision_debate_group,
    resolve_bot_handles,
    send_message,
)

__all__ = [
    "DEFAULT_BACKFILL_LIMIT",
    "DEFAULT_BOT_HANDLES",
    "MAX_BACKFILL_LIMIT",
    "MAX_ENTITIES",
    "MAX_REPLY_EDGES",
    "BackfillResult",
    "BackfillScopeError",
    "CreatedGroup",
    "DialogInfo",
    "EntityRecord",
    "GroupAudit",
    "InviteOutcome",
    "JsonFileSource",
    "MessageRecord",
    "OnEntity",
    "OnMessage",
    "OnReply",
    "ProvisionResult",
    "ProvisioningError",
    "ReplyRecord",
    "SecretsSource",
    "TelegramUserConfig",
    "audit_group",
    "backfill_history",
    "backfill_recent",
    "connected_user_client",
    "create_supergroup",
    "ensure_forum",
    "export_invite_link",
    "find_supergroup_by_title",
    "invite_members",
    "list_dialogs",
    "load_telegram_user_config",
    "promote_admins",
    "promote_members",
    "provision_debate_group",
    "resolve_bot_handles",
    "secrets_path_for_env",
    "send_message",
]
