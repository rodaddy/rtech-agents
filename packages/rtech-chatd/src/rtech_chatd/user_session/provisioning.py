"""Provision the trio debate supergroup idempotently over MTProto.

The recipe discovers or creates the group, enables Topics, seats and promotes
the fleet bots, and audits the result. It returns canonical ``-100…`` ids and
sent message ids where applicable. Only verified fleet-bot identities are granted
admin; arbitrary accounts are never promoted. No durable invite link is minted by
default — bots are added directly, not via a link — and any link the caller does
request is time-boxed and use-capped (bounded/revocable). Fatal auth and
rate-limit errors are classified without logging exception text or credentials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from loguru import logger
from telethon import TelegramClient, functions, utils
from telethon.errors import (
    AuthKeyUnregisteredError,
    FloodWaitError,
    PeerFloodError,
    UserAlreadyParticipantError,
    UserNotMutualContactError,
    UserPrivacyRestrictedError,
)
from telethon.tl import types
from telethon.tl.types import Channel, Chat, User

from .admin import (
    BOT_ADMIN_RIGHTS,
    _canonical_id,
    _coerce_chat_ref,
    create_supergroup,
    ensure_forum,
    promote_admins,
)

# The bot-token edge addresses the trio by fleet slug (``sprocket``/``nimbus``/
# ``birk`` — the same slugs deploy/kboard/install.sh uses). A Telethon invite
# needs a Telegram handle, which is not derivable from the slug, so this table is
# the sole configured identity source. Caller-supplied maps and arbitrary handles
# are intentionally not accepted at this trust boundary.
DEFAULT_BOT_HANDLES: dict[str, str] = {
    "sprocket": "@Sprocket_example_bot",
    "nimbus": "@Nimbus_example_bot",
    "birk": "@Birk_example_bot",
}

# An invite link is a bearer credential: anyone holding it can join. A durable,
# unbounded, unlimited-use link that lands on stdout/a log is a standing leak, so
# every link this module mints is time-boxed AND use-capped. The operator re-runs
# to get a fresh bounded link; a stale one simply stops working.
INVITE_LINK_TTL = timedelta(hours=1)
INVITE_LINK_USAGE_LIMIT = 10


class ProvisioningError(RuntimeError):
    """A provisioning RPC failed in a way the operator must act on.

    Carries a machine-readable ``kind`` (see :func:`_classify_rpc_error`) so a
    caller can branch — ``auth_revoked`` means re-mint, ``flood_wait`` means
    back off and retry later — without string-matching. The ``str`` is a
    remediation message that NEVER embeds the credential (only the error class
    name and, for a FloodWait, the server-supplied wait seconds).
    """

    def __init__(self, kind: str, message: str, *, retry_after_s: int | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.retry_after_s = retry_after_s


def _classify_rpc_error(exc: BaseException) -> ProvisioningError | None:
    """Map a Telethon RPC error to a fail-loud :class:`ProvisioningError`, or None.

    Only the classes a batch provisioning run actually hits are mapped; anything
    else returns ``None`` so the caller re-raises it unchanged (we do not swallow
    unknown failures). No branch interpolates the exception's ``str`` — a Telethon
    error can stringify to a request that embeds credentials, so only the class
    name and, for a FloodWait, the numeric ``seconds`` reach the message.

    * ``AuthKeyUnregisteredError`` → ``auth_revoked``: the stored session is dead
      (revoked or logged out). Remediation points at the minter, matching
      :func:`connected_user_client`.
    * ``FloodWaitError`` → ``flood_wait``: Telegram is rate-limiting; the wait
      seconds are surfaced on ``retry_after_s`` so a caller can back off.
    * ``PeerFloodError`` → ``peer_flood``: too many add/invite actions in a
      window; the account must cool down (no server-supplied duration).
    """
    if isinstance(exc, AuthKeyUnregisteredError):
        return ProvisioningError(
            "auth_revoked",
            "CRITICAL: Telegram session is not authorized (revoked or logged out). "
            "ACTION REQUIRED: re-mint with `python -m rtech_chatd.mint_session "
            "--env <env> --write`.",
        )
    if isinstance(exc, FloodWaitError):
        seconds = int(getattr(exc, "seconds", 0) or 0)
        return ProvisioningError(
            "flood_wait",
            f"Telegram FloodWait: rate-limited for {seconds}s. ACTION REQUIRED: "
            "back off and retry the provisioning run after the wait window.",
            retry_after_s=seconds,
        )
    if isinstance(exc, PeerFloodError):
        return ProvisioningError(
            "peer_flood",
            "Telegram PeerFlood: too many add/invite actions from this account. "
            "ACTION REQUIRED: pause provisioning and let the account cool down "
            "before retrying.",
        )
    return None


def _raise_if_fatal(exc: BaseException) -> None:
    """Re-raise ``exc`` as a :class:`ProvisioningError` when it is a fatal class.

    Used at the top of a batch op's per-member ``except`` so a session-revoked or
    rate-limit error aborts the WHOLE run loudly (retrying the next member would
    just hammer the same wall), while an ordinary per-member failure (a bad handle,
    a privacy block handled elsewhere) is left for the caller's local handling.
    """
    classified = _classify_rpc_error(exc)
    if classified is not None:
        raise classified from exc


def resolve_bot_handles(bots: list[str]) -> list[str]:
    """Resolve only configured fleet slugs or their canonical handles.

    An arbitrary caller-supplied handle or numeric id is not an identity proof and
    is refused. Exact configured handles are accepted as a convenience, then
    normalized to the configured spelling. The resolved Telegram entity is still
    checked for ``bot=True`` before every invitation and promotion.

    Args:
        bots: Requested fleet slugs or exact configured Telegram handles.

    Returns:
        The resolved identifiers in input order, de-duplicated (first wins).

    Raises:
        ValueError: If a bare slug is not in the map, or ``bots`` is empty.
    """
    if not bots:
        raise ValueError(
            "CRITICAL: no bots requested. ACTION REQUIRED: pass at least one bot "
            "(e.g. --bots sprocket,nimbus,birk)."
        )
    configured_handles = {handle.lower(): handle for handle in DEFAULT_BOT_HANDLES.values()}
    resolved: list[str] = []
    seen: set[str] = set()
    for raw in bots:
        ident = raw.strip()
        if not ident:
            continue
        if ident.startswith("@"):
            handle = configured_handles.get(ident.lower())
            if handle is None:
                raise ValueError(
                    f"CRITICAL: {ident!r} is not a configured fleet bot handle. ACTION "
                    f"REQUIRED: use one of {sorted(DEFAULT_BOT_HANDLES)}."
                )
        else:
            key = ident.lower()
            if key not in DEFAULT_BOT_HANDLES:
                raise ValueError(
                    f"CRITICAL: unknown bot slug {ident!r}. ACTION REQUIRED: use one "
                    f"of {sorted(DEFAULT_BOT_HANDLES)} or its configured @handle."
                )
            handle = DEFAULT_BOT_HANDLES[key]
        if handle not in seen:
            seen.add(handle)
            resolved.append(handle)
    if not resolved:
        raise ValueError(
            "CRITICAL: bots resolved to an empty list. ACTION REQUIRED: pass at "
            "least one non-blank bot identifier."
        )
    return resolved


async def _verified_bot_input(client: TelegramClient, handle: str) -> object:
    """Resolve one configured handle and prove it names a Telegram bot."""
    try:
        entity = await client.get_entity(handle)
    except Exception as exc:  # noqa: BLE001 -- classify fatal errors, else fail closed.
        _raise_if_fatal(exc)
        raise ValueError(
            "CRITICAL: configured bot identity could not be resolved. ACTION REQUIRED: "
            "verify the configured fleet bot handle."
        ) from exc
    if not isinstance(entity, User) or not bool(getattr(entity, "bot", False)):
        raise ValueError(
            "CRITICAL: configured identity did not resolve to a Telegram bot. ACTION "
            "REQUIRED: correct the configured fleet bot handle before provisioning."
        )
    try:
        return await client.get_input_entity(entity)
    except Exception as exc:  # noqa: BLE001 -- input peer is required for the mutation.
        _raise_if_fatal(exc)
        raise ValueError(
            "CRITICAL: verified bot identity has no usable input peer. ACTION REQUIRED: "
            "refresh the Telegram session and retry."
        ) from exc


async def find_supergroup_by_title(client: TelegramClient, title: str) -> str | None:
    """Return the canonical ``-100…`` id of an OWNED megagroup titled ``title``.

    The idempotency probe: before creating a group, look for one this account
    already owns/created with the exact title, so a re-run reuses it instead of
    spawning a duplicate. Only *creator* megagroups match — we must not adopt a
    same-named group someone else made and merely invited us to (we could not
    admin it). Matching is exact and case-sensitive to avoid quietly reusing a
    near-name.

    Args:
        client: A connected, authorized client.
        title: The exact group title to find.

    Returns:
        The marked ``-100…`` id of the first matching creator megagroup, or
        ``None`` when none exists.
    """
    try:
        async for dialog in client.iter_dialogs():
            entity = dialog.entity
            if not isinstance(entity, Channel):
                continue
            if entity.broadcast or not entity.megagroup:
                continue
            if not getattr(entity, "creator", False):
                continue
            if str(dialog.title) != title:
                continue
            marked = _canonical_id(entity)
            logger.info("TG_PROVISION: found existing owned supergroup {} ({})", title, marked)
            return marked
    except Exception as exc:  # noqa: BLE001 — classify auth/flood, else re-raise.
        _raise_if_fatal(exc)
        raise
    logger.info("TG_PROVISION: no existing owned supergroup titled {!r}", title)
    return None


@dataclass(frozen=True)
class InviteOutcome:
    """The result of inviting members, split by how each landed.

    Attributes:
        added: Identifiers seated directly (or already present) — the success set.
        privacy_blocked: Identifiers a privacy restriction blocked from a direct
            add. For a BOT this is a hard failure: a Bot API account cannot join
            via an invite link — only an admin can add it — so there is no truthful
            link fallback. Each carries actionable guidance in :attr:`failed`.
        failed: Identifier → reason string for adds that did not seat the member
            (a privacy block, a bad handle, etc.). ``privacy_blocked`` entries
            appear here too, with a bot-specific remediation.
    """

    added: list[str] = field(default_factory=list)
    privacy_blocked: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


async def export_invite_link(
    client: TelegramClient, chat: str | int, *, title: str | None = None
) -> str:
    """Mint a BOUNDED, revocable invite link for ``chat``.

    The link is time-boxed (:data:`INVITE_LINK_TTL`) AND use-capped
    (:data:`INVITE_LINK_USAGE_LIMIT`) — never a durable, unlimited-use link,
    because the URL is a bearer credential that ends up on stdout and in logs.
    A stale link simply stops working; the operator re-runs for a fresh one.

    Args:
        client: A connected client with invite rights.
        chat: Target chat id or handle.
        title: Optional label shown in Telegram.

    Returns:
        The bounded invite link URL.

    Raises:
        ProvisioningError: On fatal auth or rate-limit errors.
        ValueError: If Telegram returns no link.
    """
    entity = await client.get_entity(_coerce_chat_ref(chat))
    # A fresh expiry is computed on every mint so a re-run never re-emits an
    # already-issued long-lived link; the window is short and bounded.
    expire_date = datetime.now(UTC) + INVITE_LINK_TTL
    try:
        result = await client(
            functions.messages.ExportChatInviteRequest(
                peer=entity,
                title=title or "",
                expire_date=expire_date,
                usage_limit=INVITE_LINK_USAGE_LIMIT,
            )
        )
    except Exception as exc:  # noqa: BLE001 — classify auth/flood, else re-raise.
        _raise_if_fatal(exc)
        raise
    link = getattr(result, "link", None)
    if not link:
        raise ValueError(
            "CRITICAL: ExportChatInviteRequest returned no link. ACTION REQUIRED: "
            "confirm the account still has invite rights on the group."
        )
    logger.info(
        "TG_PROVISION: bounded invite link ready for {} (expires in {}, max {} uses)",
        chat,
        INVITE_LINK_TTL,
        INVITE_LINK_USAGE_LIMIT,
    )
    return str(link)


# The remediation a privacy-blocked BOT needs. A Bot API account cannot accept an
# invite link — only a group admin can add it — so the honest guidance is to add
# the bot manually (or ensure the provisioning account has add-member rights),
# NOT to hand out a link the bot can never use.
_BOT_PRIVACY_REMEDIATION = (
    "privacy-blocked: a bot cannot join via an invite link — an admin must add it. "
    "ACTION REQUIRED: add this bot to the group manually (or confirm the "
    "provisioning account has add-member rights and the bot allows being added)."
)


async def invite_members(
    client: TelegramClient, chat: str | int, members: list[str]
) -> InviteOutcome:
    """Invite ``members`` to ``chat``, seating each directly (bots only).

    Each member is added independently via ``InviteToChannelRequest``. An
    already-participant member is idempotent success. When a member's privacy
    settings block a direct add (``UserPrivacyRestricted`` / not-a-mutual-contact),
    it is recorded under ``privacy_blocked`` AND ``failed`` with bot-truthful
    guidance — a Bot API account CANNOT join through an invite link, so there is no
    link fallback to offer; the operator must add the bot manually. A non-privacy
    failure (bad handle, etc.) goes to ``failed`` and never aborts the rest. A
    revoked session or a flood/peer-flood aborts the whole batch loudly (retrying
    the next member would just hit the same wall).

    Args:
        client: A connected, authorized client with invite rights on ``chat``.
        chat: The target chat.
        members: Fleet slugs or exact configured bot handles.

    Returns:
        An :class:`InviteOutcome`.

    Raises:
        ProvisioningError: On a revoked session / flood / peer-flood.
    """
    resolved = resolve_bot_handles(members)
    channel = await client.get_entity(_coerce_chat_ref(chat))
    added: list[str] = []
    privacy_blocked: list[str] = []
    failed: dict[str, str] = {}
    for member in resolved:
        try:
            user = await _verified_bot_input(client, member)
            await client(functions.channels.InviteToChannelRequest(channel=channel, users=[user]))
            added.append(str(member))
            logger.info("TG_PROVISION: invited {} directly", member)
        except UserAlreadyParticipantError:
            # A re-run over a group that already seats this bot is a SUCCESS, not a
            # failure — the desired end state (bot present) already holds. Record it
            # as added so idempotent provisioning does not report a spurious failure.
            added.append(str(member))
            logger.info("TG_PROVISION: {} already a participant; treating as seated", member)
        except (UserPrivacyRestrictedError, UserNotMutualContactError):
            # No invite-link fallback: bots cannot join via a link. Fail truthfully
            # with actionable guidance instead of emitting a link the bot can't use.
            privacy_blocked.append(str(member))
            failed[str(member)] = _BOT_PRIVACY_REMEDIATION
            logger.warning(
                "TG_PROVISION: {} is privacy-blocked; a bot cannot join via a link — "
                "an admin must add it manually",
                member,
            )
        except Exception as exc:  # noqa: BLE001 — fatal aborts; else record + continue.
            _raise_if_fatal(exc)
            failed[str(member)] = type(exc).__name__
            logger.warning("TG_PROVISION: could not invite {}: {}", member, type(exc).__name__)

    return InviteOutcome(added=added, privacy_blocked=privacy_blocked, failed=failed)


async def promote_members(
    client: TelegramClient, chat: str | int, members: list[str], *, rank: str = "bot"
) -> dict[str, str]:
    """Promote ``members`` to bot-admin (``manage_topics`` + least privilege).

    Thin wrapper over the primitive
    :func:`rtech_chatd.user_session.admin.promote_admins` so the provisioning
    recipe reuses the SAME right-set (:data:`BOT_ADMIN_RIGHTS`) rather than
    re-deriving it, but classifies a session-revoked / flood failure into a
    :class:`ProvisioningError` — the reference op reports every failure as a plain
    string, and the provisioning caller must be able to abort loudly on a fatal
    class.

    Args:
        client: A connected, authorized client with add-admins rights.
        chat: The target supergroup.
        members: Fleet slugs or exact configured bot handles.
        rank: Admin-rank label (default ``"bot"``).

    Returns:
        member → ``"ok"`` or the failure class name.

    Raises:
        ProvisioningError: If any promotion hits a fatal (auth/flood) class.
    """
    resolved = resolve_bot_handles(members)
    for member in resolved:
        await _verified_bot_input(client, member)
    results = await promote_admins(client, chat, resolved, rank=rank)
    # promote_admins swallows every per-member error into a class-name string; if
    # any of those names is a fatal class we must surface it as a ProvisioningError
    # so the run aborts rather than reporting a soft "failed" for a dead session.
    fatal_promotions = {
        "AuthKeyUnregisteredError": (
            "auth_revoked",
            "re-mint the session",
        ),
        "FloodWaitError": (
            "flood_wait",
            "back off and retry after the rate-limit window",
        ),
        "PeerFloodError": (
            "peer_flood",
            "pause promotions and let the account cool down before retrying",
        ),
    }
    for name in results.values():
        classified = fatal_promotions.get(name)
        if classified is None:
            continue
        kind, remediation = classified
        raise ProvisioningError(
            kind,
            f"CRITICAL: promotion hit {name}. ACTION REQUIRED: {remediation}.",
        )
    return results


@dataclass(frozen=True)
class GroupAudit:
    """A point-in-time audit of a provisioned group's forum/admin/member state.

    Attributes:
        chat_id: The canonical ``-100…`` id.
        title: The group title.
        is_forum: Whether Topics are enabled.
        admins: Marked/user ids of the group's admins.
        member_count: Total participant count (best-effort; ``-1`` if unknown).
    """

    chat_id: str
    title: str
    is_forum: bool
    admins: list[str]
    member_count: int


async def audit_group(client: TelegramClient, chat: str | int) -> GroupAudit:
    """Read the forum, admin, and member state of ``chat``.

    Args:
        client: A connected client.
        chat: Target chat id or handle.

    Returns:
        The current :class:`GroupAudit`.
    """
    try:
        entity = await client.get_entity(_coerce_chat_ref(chat))
        admins_raw = await client.get_participants(entity, filter=types.ChannelParticipantsAdmins())
        # Total count is a separate read; if it is unavailable (e.g. a basic Chat)
        # fall back to the admin-list length rather than failing the whole audit.
        try:
            all_members = await client.get_participants(entity, limit=0)
            member_count = getattr(all_members, "total", len(all_members))
        except Exception as exc:  # noqa: BLE001 — count is best-effort.
            _raise_if_fatal(exc)
            member_count = -1
    except Exception as exc:  # noqa: BLE001 — classify auth/flood, else re-raise.
        _raise_if_fatal(exc)
        raise

    admins = [str(utils.get_peer_id(a)) for a in admins_raw]
    is_forum = bool(getattr(entity, "forum", False))
    title = str(getattr(entity, "title", ""))
    chat_id = _canonical_id(entity) if isinstance(entity, (Channel, Chat)) else str(chat)
    logger.info(
        "TG_PROVISION: audit {} — forum={} admins={} members={}",
        chat_id,
        is_forum,
        len(admins),
        member_count,
    )
    return GroupAudit(
        chat_id=chat_id,
        title=title,
        is_forum=is_forum,
        admins=admins,
        member_count=member_count,
    )


async def send_message(
    client: TelegramClient, chat: str | int, text: str, *, reply_to: int | None = None
) -> int:
    """Send text to ``chat`` and return the message id.

    Args:
        client: A connected client with post rights.
        chat: Target chat id or handle.
        text: Message body.
        reply_to: Optional topic root or message id.

    Returns:
        The sent message id.
    """
    try:
        sent = await client.send_message(_coerce_chat_ref(chat), text, reply_to=reply_to)
    except Exception as exc:  # noqa: BLE001 — classify auth/flood, else re-raise.
        _raise_if_fatal(exc)
        raise
    message_id = int(sent.id)
    logger.info("TG_PROVISION: sent message {} to {}", message_id, chat)
    return message_id


@dataclass(frozen=True)
class ProvisionResult:
    """Describe the complete idempotent provisioning outcome.

    ``chat_id`` is the canonical ``-100…`` id. Seating is split among ``invited``,
    ``invite_privacy_blocked`` (bots a privacy setting blocked — no link fallback),
    and ``invite_failed``; ``promotions`` maps each seated verified bot to ``"ok"``
    or an error class. ``audit`` records the final group state.

    ``invite_link`` is ``None`` by default: the recipe seats *bots*, which cannot
    use an invite link, so no durable link is minted or returned. A link is only
    ever present when the caller explicitly opts into one (``mint_invite_link``),
    and even then it is bounded/revocable (see :func:`export_invite_link`).
    """

    chat_id: str
    title: str
    invite_link: str | None
    created: bool
    forum_enabled_now: bool
    invited: list[str]
    invite_privacy_blocked: list[str]
    invite_failed: dict[str, str]
    promotions: dict[str, str]
    audit: GroupAudit


async def provision_debate_group(
    client: TelegramClient,
    title: str,
    bots: list[str],
    *,
    about: str = "",
    mint_invite_link: bool = False,
) -> ProvisionResult:
    """Discover or create, configure, seat, promote, and audit the group.

    Args:
        client: A connected client.
        title: Exact group title and idempotency key.
        bots: Fleet slugs or exact configured bot handles.
        about: Description used only on creation.
        mint_invite_link: When True, mint a BOUNDED, revocable link (for a human
            operator to join). Default False — the recipe seats *bots*, which
            cannot use a link, so no durable link is minted or returned by default.

    Returns:
        The complete :class:`ProvisionResult`.

    Raises:
        ValueError: If bot identifiers are invalid.
        ProvisioningError: On fatal auth or rate-limit errors.
    """
    resolved = resolve_bot_handles(bots)
    logger.info("TG_PROVISION: provisioning {!r} for bots {}", title, resolved)

    existing = await find_supergroup_by_title(client, title)
    if existing is not None:
        chat_id = existing
        created = False
    else:
        # create_supergroup already creates the forum-enabled megagroup AND does a
        # first-pass direct invite; we do NOT pass members here so the privacy-aware
        # invite_members path below is the single place invites (and the link
        # fallback) are decided — keeping one invite policy, not two.
        made = await create_supergroup(client, title, members=None, about=about, forum=True)
        chat_id = made.chat_id
        created = True

    forum_enabled_now = await ensure_forum(client, chat_id)

    invite = await invite_members(client, chat_id, resolved)

    # Promotion is restricted to the bots the direct invite path actually seated.
    # ``promote_members`` resolves and verifies ``bot=True`` immediately before the
    # admin RPC, so an identity can never inherit trust from caller input alone.
    promotions: dict[str, str] = {}
    if invite.added:
        promotions.update(await promote_members(client, chat_id, invite.added))

    # No durable link by default: bots are seated by direct add, not by link, so a
    # link would be a standing bearer-credential leak for no operational gain. Only
    # mint one — bounded and revocable — when the caller explicitly asks for a
    # human-join link.
    invite_link: str | None = None
    if mint_invite_link:
        invite_link = await export_invite_link(client, chat_id, title=title)

    audit = await audit_group(client, chat_id)

    return ProvisionResult(
        chat_id=chat_id,
        title=title,
        invite_link=invite_link,
        created=created,
        forum_enabled_now=forum_enabled_now,
        invited=invite.added,
        invite_privacy_blocked=invite.privacy_blocked,
        invite_failed=invite.failed,
        promotions=promotions,
        audit=audit,
    )


__all__ = [
    "BOT_ADMIN_RIGHTS",
    "DEFAULT_BOT_HANDLES",
    "GroupAudit",
    "InviteOutcome",
    "ProvisionResult",
    "ProvisioningError",
    "audit_group",
    "export_invite_link",
    "find_supergroup_by_title",
    "invite_members",
    "promote_members",
    "provision_debate_group",
    "resolve_bot_handles",
    "send_message",
]
