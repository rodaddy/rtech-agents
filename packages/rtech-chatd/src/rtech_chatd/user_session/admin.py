"""User-session admin ops for the Telegram edge: dialog discovery + forum toggle.

The MTProto automation tier (issue #110). A Telegram *bot* cannot list the
account's chats or turn a supergroup into a Topics/forum group — those are
user-account privileges — so these ops run on the Telethon *user* session minted
by :mod:`rtech_chatd.mint_session`, loaded through the same fail-loud storage
seam (:func:`rtech_chatd.user_session.config.load_telegram_user_config`).

Three capabilities, deliberately minimal:

* :func:`list_dialogs` — enumerate the account's groups/channels (id, title,
  type, is_forum) so an operator can find the canonical ``-100…`` id of a chat
  by name. This is the "how do I get the chat id" helper the reference apps
  carry; it is READ-ONLY.
* :func:`create_supergroup` — create a fresh forum-enabled megagroup and invite
  members (e.g. the trio bots). Mirrors the proven b1x ``create_new_channel``
  flow: ``CreateChannelRequest(megagroup=True)`` then ``InviteToChannelRequest``.
  Creating the container as a supergroup up front is why the reference apps never
  migrate/toggle — Topics is available immediately.
* :func:`ensure_forum` — turn Topics ON for an EXISTING supergroup (idempotent).
  Kept for the case where a supergroup already exists without Topics.

The ops take an already-connected ``TelegramClient`` so the connect/disconnect
lifecycle (and its error handling) stays owned by one place —
:func:`connected_user_client` — rather than duplicated per op.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass

from loguru import logger
from telethon import TelegramClient, functions, utils
from telethon.errors import FloodWaitError, UserAlreadyParticipantError
from telethon.sessions import StringSession
from telethon.tl import types
from telethon.tl.types import Channel, Chat

from .config import TelegramUserConfig, load_telegram_user_config

# Bounded reconnect schedule (seconds), lifted from the b1x
# ``telethon_client``/``bot_telethon_bridge`` backoff. A transient connect
# failure (network blip, a Telegram edge dropping the socket) is retried on this
# fixed, FINITE ladder — never an unbounded loop — so a genuinely dead endpoint
# fails loud after ~31s of total wait rather than hanging the provisioning run.
_CONNECT_BACKOFF_S: tuple[int, ...] = (1, 2, 4, 8, 16)

# Cap on how long a single FloodWait is honored during connect. Telegram can
# hand back an arbitrarily large wait; past this ceiling we stop waiting and fail
# loud so a multi-hour FloodWait never silently parks a provisioning run.
_CONNECT_FLOOD_WAIT_CEILING_S = 60

BOT_ADMIN_RIGHTS = types.ChatAdminRights(
    change_info=False,
    post_messages=True,
    edit_messages=True,
    delete_messages=True,
    ban_users=False,
    invite_users=True,
    pin_messages=True,
    add_admins=False,
    manage_call=False,
    anonymous=False,
    manage_topics=True,
)


@dataclass(frozen=True)
class DialogInfo:
    """A single account dialog, reduced to the fields an operator needs.

    Attributes:
        chat_id: The canonical id. For supergroups/channels this is the
            ``-100…`` form the edge's adapter emits as ``channel_id``; for a
            basic group it is the smaller signed id.
        title: The human chat title (for eyeballing which one is which).
        kind: One of ``broadcast``, ``megagroup``, ``supergroup``, or ``group``.
        is_forum: True when Topics are already enabled on this chat.
    """

    chat_id: str
    title: str
    kind: str
    is_forum: bool


def _coerce_chat_ref(chat: str | int) -> str | int:
    """Normalize a chat reference so Telethon's ``get_entity`` can resolve it.

    A marked ``-100…`` id passed as a STRING fails to resolve on a fresh session
    (``Cannot find any entity corresponding to "-100…"``), whereas the same value
    as an INT resolves fine. So coerce a signed-integer string to ``int`` while
    leaving an ``@handle`` (or anything non-numeric) untouched.
    """
    if isinstance(chat, str):
        stripped = chat.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
        return stripped
    return chat


def _canonical_id(entity: Channel | Chat) -> str:
    """Return the Bot-API-marked chat-id string for a channel/chat entity.

    Delegates to Telethon's :func:`telethon.utils.get_peer_id`, the library's own
    id-marking: a :class:`Channel`/supergroup becomes ``-100<id>`` while a basic
    :class:`Chat` becomes ``-<id>`` — exactly the form the edge adapter's
    ``canonical_chat_id`` emits (Bot API convention), so discovery output is
    directly pasteable into ``RTECH_TG_ALLOWED_GROUPS``. Using the library
    function (not a hand-rolled f-string) keeps the marking correct across every
    entity flavor — broadcast channels, megagroups, gigagroups, basic chats —
    which the reference admin code relies on rather than re-deriving.
    """
    return str(utils.get_peer_id(entity))


def _classify(entity: Channel | Chat) -> str:
    """Classify an entity as broadcast/megagroup/supergroup/group."""
    if isinstance(entity, Channel):
        if entity.broadcast:
            return "broadcast"
        return "megagroup" if entity.megagroup else "supergroup"
    return "group"


async def _connect_with_backoff(client: TelegramClient) -> None:
    """Connect ``client`` with a bounded, FloodWait-aware retry ladder.

    Retries a transient connect failure on the finite :data:`_CONNECT_BACKOFF_S`
    schedule (``[1,2,4,8,16]`` s), sleeping between attempts. A ``FloodWaitError``
    at connect is honored for its server-supplied duration *only up to*
    :data:`_CONNECT_FLOOD_WAIT_CEILING_S` — a longer wait fails loud immediately
    rather than parking the run for hours. After the ladder is exhausted the last
    exception is re-raised so the caller (and the operator) sees the real cause;
    the loop is never unbounded. NO credential or exception text is logged — only
    the attempt index, the class name, and any FloodWait seconds.

    Args:
        client: A constructed (not yet connected) :class:`TelegramClient`.

    Raises:
        FloodWaitError: If a connect FloodWait exceeds the ceiling.
        Exception: The last connect error after every backoff step is spent.
    """
    attempts = len(_CONNECT_BACKOFF_S) + 1
    for index in range(attempts):
        last_attempt = index >= len(_CONNECT_BACKOFF_S)
        try:
            await client.connect()
            if index:
                logger.info("TG_ADMIN: connected after {} retry attempt(s)", index)
            return
        except FloodWaitError as exc:
            seconds = int(getattr(exc, "seconds", 0) or 0)
            if seconds > _CONNECT_FLOOD_WAIT_CEILING_S:
                logger.error(
                    "TG_ADMIN: connect FloodWait {}s exceeds {}s ceiling; failing loud",
                    seconds,
                    _CONNECT_FLOOD_WAIT_CEILING_S,
                )
                raise
            # A FloodWait on the LAST attempt must fail loud too — honoring it and
            # then falling out of the loop would return an unconnected client as if
            # connect had succeeded. Re-raise the real FloodWait instead.
            if last_attempt:
                logger.error(
                    "TG_ADMIN: connect FloodWait {}s on final attempt; failing loud",
                    seconds,
                )
                raise
            logger.warning("TG_ADMIN: connect FloodWait {}s; honoring then retrying", seconds)
            await asyncio.sleep(seconds)
        except Exception as exc:  # noqa: BLE001 — retry transient connect failures.
            if last_attempt:
                logger.error(
                    "TG_ADMIN: connect failed after {} attempt(s): {}",
                    attempts,
                    type(exc).__name__,
                )
                raise
            delay = _CONNECT_BACKOFF_S[index]
            logger.warning(
                "TG_ADMIN: connect attempt {} failed ({}); retrying in {}s",
                index + 1,
                type(exc).__name__,
                delay,
            )
            await asyncio.sleep(delay)
    # Unreachable: every branch above either returns or raises on the last attempt.
    # A guard here keeps the function from ever returning an unconnected client if
    # the loop bounds are later changed.
    raise RuntimeError(  # pragma: no cover
        "CRITICAL: connect retry loop exited without connecting. ACTION REQUIRED: "
        "check network reachability to Telegram and re-run."
    )


@contextlib.asynccontextmanager
async def connected_user_client(
    config: TelegramUserConfig | None = None, *, env: str = "dev"
) -> AsyncIterator[TelegramClient]:
    """Yield a connected, authorized Telethon client from the stored session.

    Loads the per-env credential (fail-loud) when ``config`` is not supplied,
    connects using the persisted ``session_string`` via the bounded, FloodWait-aware
    :func:`_connect_with_backoff` ladder (``[1,2,4,8,16]`` s — a transient blip
    retries, a dead endpoint fails loud, never an unbounded loop), and asserts the
    session is authorized before yielding — an unauthorized/empty session fails
    loud here with an ACTION REQUIRED pointing at the minter, rather than each op
    erroring cryptically. The client is always disconnected on exit.

    Args:
        config: Optional pre-loaded credential; loaded from ``env`` when None.
        env: Environment slug used when ``config`` is None.

    Yields:
        A connected, authorized :class:`TelegramClient`.

    Raises:
        ValueError: If the loaded session_string is empty or unauthorized.
    """
    cfg = config or load_telegram_user_config(env)
    if not cfg.session_string:
        raise ValueError(
            "CRITICAL: no session_string for this env. ACTION REQUIRED: mint one "
            "with `python -m rtech_chatd.mint_session --env <env> --write` first."
        )
    client = TelegramClient(StringSession(cfg.session_string), cfg.api_id, cfg.api_hash)
    # The connect AND the authorization check live INSIDE the try so a failure (or
    # a cancellation) in either — including a partial connect — still routes to the
    # finally and disconnects. Otherwise a raised connect/auth error would leak the
    # socket that _connect_with_backoff may have partially opened.
    try:
        await _connect_with_backoff(client)
        if not await client.is_user_authorized():
            raise ValueError(
                "CRITICAL: stored session is not authorized (revoked or stale). "
                "ACTION REQUIRED: re-mint with `python -m rtech_chatd.mint_session "
                "--env <env> --write`."
            )
        logger.info("TG_ADMIN: connected on stored user session (env={})", env)
        yield client
    finally:
        # disconnect() is safe on a never-/partially-connected client, but guard it
        # so a cleanup error never masks the original failure being propagated.
        try:
            await client.disconnect()
            logger.debug("TG_ADMIN: disconnected")
        except Exception as exc:  # noqa: BLE001 — cleanup must not mask the real error.
            logger.warning("TG_ADMIN: disconnect during cleanup failed: {}", type(exc).__name__)


async def list_dialogs(client: TelegramClient) -> list[DialogInfo]:
    """Enumerate the account's group/channel dialogs (READ-ONLY).

    Skips users/bots (1:1 chats); returns only groups and channels, which is
    what an operator is looking for when hunting a chat id by title. Ordering
    follows Telethon's dialog order (most-recent first).

    Args:
        client: A connected, authorized client (see :func:`connected_user_client`).

    Returns:
        The list of :class:`DialogInfo` for every group/channel dialog.
    """
    dialogs: list[DialogInfo] = []
    async for dialog in client.iter_dialogs():
        entity = dialog.entity
        if not isinstance(entity, (Channel, Chat)):
            continue  # skip 1:1 user/bot dialogs
        dialogs.append(
            DialogInfo(
                chat_id=_canonical_id(entity),
                title=str(dialog.title),
                kind=_classify(entity),
                is_forum=bool(getattr(entity, "forum", False)),
            )
        )
    logger.info("TG_ADMIN: listed {} group/channel dialogs", len(dialogs))
    return dialogs


async def ensure_forum(client: TelegramClient, chat: str | int) -> bool:
    """Ensure Topics/forum mode is ON for ``chat`` (idempotent).

    A no-op when the chat is already a forum. Turning Topics on requires the
    chat to be a supergroup (megagroup); Telegram rejects the toggle on a basic
    group or broadcast channel — that surfaces as a Telethon RPC error, which we
    let propagate with context rather than swallow.

    Args:
        client: A connected, authorized client with admin rights on ``chat``.
        chat: The target chat — a ``-100…`` id, a numeric id, or an ``@handle``.

    Returns:
        True if this call flipped Topics on; False if it was already a forum.

    Raises:
        Exception: Any Telethon RPC error (e.g. not-a-supergroup, missing admin
            rights) propagates so the operator sees the real cause.
    """
    entity = await client.get_entity(_coerce_chat_ref(chat))
    already = bool(getattr(entity, "forum", False))
    if already:
        logger.info("TG_ADMIN: {} is already a forum; nothing to do", chat)
        return False

    logger.info("TG_ADMIN: enabling Topics on {}", chat)
    # tabs=False selects the classic topic-list layout (not the newer tabs view);
    # this Telethon schema requires the arg explicitly.
    await client(functions.channels.ToggleForumRequest(channel=entity, enabled=True, tabs=False))
    logger.info("TG_ADMIN: Topics enabled on {}", chat)
    return True


@dataclass(frozen=True)
class CreatedGroup:
    """The result of creating a fresh trio supergroup.

    Attributes:
        chat_id: The new supergroup's marked ``-100…`` id (paste into
            ``RTECH_TG_ALLOWED_GROUPS``).
        title: The group title.
        added: The member identifiers successfully invited.
        failed: Member identifiers that could not be added (identifier -> reason).
    """

    chat_id: str
    title: str
    added: list[str]
    failed: dict[str, str]


async def create_supergroup(
    client: TelegramClient,
    title: str,
    members: list[str] | None = None,
    *,
    about: str = "",
    forum: bool = True,
) -> CreatedGroup:
    """Create a fresh forum-enabled megagroup and invite ``members``.

    Mirrors the proven b1x ``create_new_channel`` flow: a single
    :class:`~telethon.tl.functions.channels.CreateChannelRequest` with
    ``megagroup=True`` (a supergroup, NOT a broadcast channel), then
    :class:`~telethon.tl.functions.messages.InviteToChannelRequest` per member.
    Creating the container as a supergroup up front is why the reference apps
    never migrate/toggle — Topics is available immediately, so we just enable it.

    Args:
        client: A connected, authorized client (the creator becomes owner).
        title: The new group's title.
        members: Optional list of user identifiers (``@handle`` or id) to invite,
            e.g. the trio bots. A member that fails to add is recorded in the
            result's ``failed`` map rather than aborting the whole creation.
        about: Optional group description.
        forum: Enable Topics on creation (default True). When True, the new
            supergroup comes up with Topics already on — no separate toggle.

    Returns:
        A :class:`CreatedGroup` with the new marked id, title, and add results.

    Raises:
        ValueError: If creation returns no supergroup channel.
        Exception: Any Telethon RPC error on creation propagates.
    """
    logger.info("TG_ADMIN: creating supergroup {!r} (forum={})", title, forum)
    result = await client(
        functions.channels.CreateChannelRequest(
            title=title, about=about, megagroup=True, forum=forum
        )
    )
    new_channel = next(
        (c for c in getattr(result, "chats", []) if isinstance(c, Channel) and not c.broadcast),
        None,
    )
    if new_channel is None:
        raise ValueError(
            "CRITICAL: channel creation returned no supergroup. ACTION REQUIRED: "
            "re-list dialogs to check whether the group was partially created."
        )
    marked = str(utils.get_peer_id(new_channel))
    logger.info("TG_ADMIN: created supergroup {} ({})", title, marked)

    added: list[str] = []
    failed: dict[str, str] = {}
    for member in members or []:
        try:
            user = await client.get_input_entity(member)
            await client(
                functions.channels.InviteToChannelRequest(channel=new_channel, users=[user])
            )
            added.append(str(member))
            logger.info("TG_ADMIN: invited {} to {}", member, marked)
        except UserAlreadyParticipantError:
            # Already seated is the desired end state — record as added, not failed,
            # so a re-run over an existing group stays idempotent.
            added.append(str(member))
            logger.info(
                "TG_ADMIN: {} already a participant in {}; treating as seated", member, marked
            )
        except Exception as exc:  # noqa: BLE001 — one bad member must not abort the rest.
            failed[str(member)] = type(exc).__name__
            logger.warning("TG_ADMIN: could not invite {}: {}", member, type(exc).__name__)

    return CreatedGroup(chat_id=marked, title=title, added=added, failed=failed)


async def promote_admins(
    client: TelegramClient, chat: str | int, members: list[str], *, rank: str = "bot"
) -> dict[str, str]:
    """Grant ``members`` the admin rights a bot needs to run threaded debate.

    Mirrors the b1x ``promote_user_to_admin`` flow (EditAdminRequest +
    ChatAdminRights) but with a LEAST-PRIVILEGE right set fit for a posting bot:
    it can manage topics, post/edit/delete/pin messages, and invite — but NOT
    add other admins or change group info. ``manage_topics`` is the load-bearing
    right: without it a bot cannot create or post into forum Topics.

    Each member is promoted independently; a failure is recorded and does not
    abort the rest.

    Args:
        client: A connected, authorized client with add-admins rights (the owner).
        chat: The supergroup — a ``-100…`` id, numeric id, or ``@handle``.
        members: User identifiers to promote (``@handle`` or id).
        rank: The admin-rank label shown in Telegram (default ``"bot"``).

    Returns:
        A map of member identifier -> ``"ok"`` or the failure reason
        (an empty map means every promotion succeeded).
    """
    channel = await client.get_entity(_coerce_chat_ref(chat))
    # Warm the participant cache so a member referenced by a BARE numeric id
    # resolves on the first EditAdminRequest. On a fresh session Telethon has no
    # access_hash for such a user until it has seen them in the channel, so
    # promoting by bare id otherwise raises ValueError on the first attempt.
    try:
        await client.get_participants(channel, limit=200)
    except Exception as exc:  # noqa: BLE001 — cache-warm is best-effort.
        logger.debug("TG_ADMIN: participant cache warm skipped: {}", type(exc).__name__)

    results: dict[str, str] = {}
    for member in members:
        try:
            user = await client.get_input_entity(member)
            await client(
                functions.channels.EditAdminRequest(
                    channel=channel,
                    user_id=user,
                    admin_rights=BOT_ADMIN_RIGHTS,
                    rank=rank,
                )
            )
            results[str(member)] = "ok"
            logger.info("TG_ADMIN: promoted {} to admin (manage_topics) in {}", member, chat)
        except Exception as exc:  # noqa: BLE001 — one failure must not abort the rest.
            results[str(member)] = type(exc).__name__
            logger.warning(
                "TG_ADMIN: could not promote {} in {}: {}",
                member,
                chat,
                type(exc).__name__,
            )
    return results
