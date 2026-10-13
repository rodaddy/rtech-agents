"""Backfill bounded Telegram history into content-free records.

Backfill reads a *bounded* slice of a chat's history and emits content-free
records (id graph + shape flags — never message text or media). Every public
entry requires an explicit approved-chat capability: the caller must name which
canonical ``-100…`` chat ids backfill is authorized to touch. A read of any chat
NOT in that allowlist fails loud — a history read is a privileged, capability-
gated action, never something that runs against an arbitrary chat by default.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from loguru import logger
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.types import Channel, Chat, User

from .admin import _CONNECT_FLOOD_WAIT_CEILING_S, _coerce_chat_ref
from .provisioning import ProvisioningError, _raise_if_fatal

# A provisioning run is not a history importer. Keep every history read bounded.
DEFAULT_BACKFILL_LIMIT = 50
MAX_BACKFILL_LIMIT = 500

# Hard TOTAL caps on downstream work. These are real ceilings — not batch sizes:
# once the cap is hit, extra referenced ids/edges are dropped (and the drop is
# logged), so entity resolution and reply-edge emission remain bounded even when
# a dense history window contains more references than messages.
MAX_ENTITIES = 200
MAX_REPLY_EDGES = 100

# A finite retry budget prevents a persistently rate-limited batch from looping.
_BACKFILL_MAX_FLOOD_RETRIES = 3


class BackfillScopeError(RuntimeError):
    """A backfill was requested against a chat outside the approved capability.

    Raised fail-closed when the target chat is not in the caller-supplied
    ``allowed_chats`` set (or no capability was supplied at all). Carries no
    credential — only the offending chat ref and the approved set.
    """


def _require_scope(chat: str | int, allowed_chats: frozenset[str]) -> str | int:
    """Fail loud unless ``chat`` is inside the approved-chat capability.

    ``allowed_chats`` is REQUIRED: ``None`` or an empty set means the caller never
    granted a scope, so the read is refused (fail-closed). The chat is normalized
    with :func:`_coerce_chat_ref` and its string form must be a member of the
    allowlist. Returns the coerced ref so callers reuse it for the RPC.

    Args:
        chat: The requested chat, which must normalize to an approved canonical id.
        allowed_chats: The approved canonical chat ids this backfill may touch.

    Raises:
        BackfillScopeError: If no capability was supplied, or ``chat`` is not in it.
    """
    if not allowed_chats:
        raise BackfillScopeError(
            "CRITICAL: backfill requires an explicit approved-chat scope, but none "
            "was supplied. ACTION REQUIRED: pass allowed_chats={<canonical -100 id>} "
            "naming the chat(s) this backfill is authorized to read."
        )
    coerced = _coerce_chat_ref(chat)
    if str(coerced) not in allowed_chats:
        raise BackfillScopeError(
            f"CRITICAL: backfill of {coerced!r} is outside the approved scope "
            f"{sorted(allowed_chats)}. ACTION REQUIRED: add the chat to allowed_chats "
            "only if this read is intended."
        )
    return coerced


@dataclass(frozen=True)
class MessageRecord:
    """Describe one backfilled message without retaining private content.

    Attributes:
        message_id: The message's id within the chat.
        sender_id: The marked author id, or ``None`` for a service/anonymous post.
        reply_to_msg_id: The parent message id when this is a reply, else ``None``.
        date_epoch: The message's UTC timestamp as an int epoch (``0`` if absent).
        has_text: True when the message carried any text body.
        has_media: True when the message carried media.
    """

    message_id: int
    sender_id: int | None
    reply_to_msg_id: int | None
    date_epoch: int
    has_text: bool
    has_media: bool


@dataclass(frozen=True)
class EntityRecord:
    """Describe a resolved entity referenced by backfilled messages.

    Attributes:
        entity_id: The marked id referenced by one or more messages.
        kind: One of ``user``, ``channel``, ``chat``, or ``unknown``.
        resolved: True when ``get_entity`` returned an entity for this id.
    """

    entity_id: int
    kind: str
    resolved: bool


@dataclass(frozen=True)
class ReplyRecord:
    """Describe a child-to-parent reply edge between backfilled messages.

    Attributes:
        child_id: The replying message's id.
        parent_id: The parent (replied-to) message's id.
        parent_present: True when ``parent_id`` was among the phase-1 records.
    """

    child_id: int
    parent_id: int
    parent_present: bool


@dataclass(frozen=True)
class BackfillResult:
    """Describe the complete three-phase backfill outcome.

    Attributes:
        messages: Phase-1 bounded message records (Telethon newest-first order).
        entities: Phase-2 resolved user/entity references.
        replies: Phase-3 resolved reply edges.
    """

    messages: list[MessageRecord]
    entities: list[EntityRecord]
    replies: list[ReplyRecord]


# Each optional callback is awaited once per record as that record is produced.
OnMessage = Callable[[MessageRecord], Awaitable[None]]
OnEntity = Callable[[EntityRecord], Awaitable[None]]
OnReply = Callable[[ReplyRecord], Awaitable[None]]


async def _run_batch_rpc(op: Callable[[], Awaitable[object]], *, label: str) -> object:
    """Run one backfill RPC with bounded FloodWait retries."""
    for attempt in range(_BACKFILL_MAX_FLOOD_RETRIES + 1):
        try:
            return await op()
        except FloodWaitError as exc:
            seconds = int(getattr(exc, "seconds", 0) or 0)
            if seconds > _CONNECT_FLOOD_WAIT_CEILING_S or attempt >= _BACKFILL_MAX_FLOOD_RETRIES:
                raise ProvisioningError(
                    "flood_wait",
                    f"Telegram FloodWait: rate-limited for {seconds}s during {label} "
                    "backfill. ACTION REQUIRED: back off and retry after the wait window.",
                    retry_after_s=seconds,
                ) from exc
            logger.warning(
                "TG_PROVISION: {} backfill FloodWait {}s; honoring then retrying", label, seconds
            )
            await asyncio.sleep(seconds)
        except Exception as exc:  # noqa: BLE001 -- classify fatal errors, else re-raise.
            _raise_if_fatal(exc)
            raise
    raise ProvisioningError(  # pragma: no cover
        "flood_wait",
        f"Telegram FloodWait: {label} backfill exhausted retries. ACTION REQUIRED: "
        "back off and retry after the wait window.",
    )


async def backfill_recent(
    client: TelegramClient,
    chat: str | int,
    *,
    allowed_chats: frozenset[str],
    limit: int = DEFAULT_BACKFILL_LIMIT,
) -> list[int]:
    """Read bounded recent message ids from an explicitly approved chat."""
    approved_chat = _require_scope(chat, allowed_chats)
    capped = max(0, min(int(limit), MAX_BACKFILL_LIMIT))
    if capped == 0:
        return []
    for attempt in range(_BACKFILL_MAX_FLOOD_RETRIES + 1):
        ids: list[int] = []
        try:
            async for message in client.iter_messages(approved_chat, limit=capped):
                ids.append(int(message.id))
        except FloodWaitError as exc:
            seconds = int(getattr(exc, "seconds", 0) or 0)
            if seconds > _CONNECT_FLOOD_WAIT_CEILING_S or attempt >= _BACKFILL_MAX_FLOOD_RETRIES:
                raise ProvisioningError(
                    "flood_wait",
                    "Telegram FloodWait while reading recent backfill ids. ACTION REQUIRED: "
                    "back off and retry after the wait window.",
                    retry_after_s=seconds,
                ) from exc
            logger.warning(
                "TG_PROVISION: recent-id backfill FloodWait {}s; retrying before return",
                seconds,
            )
            await asyncio.sleep(seconds)
        except Exception as exc:  # noqa: BLE001 -- classify fatal errors, else re-raise.
            _raise_if_fatal(exc)
            raise
        else:
            logger.info("TG_PROVISION: backfill read {} message id(s) from {}", len(ids), chat)
            return ids
    raise AssertionError("unreachable recent-id backfill retry exhaustion")  # pragma: no cover


def _kind_of(entity: object) -> str:
    """Coarse-classify a resolved entity without retaining profile data."""
    if isinstance(entity, User):
        return "user"
    if isinstance(entity, Channel):
        return "channel"
    if isinstance(entity, Chat):
        return "chat"
    return "unknown"


def _date_epoch(message: object) -> int:
    """Return a message timestamp as an integer epoch, or zero when absent."""
    date = getattr(message, "date", None)
    if date is None:
        return 0
    if isinstance(date, int):
        return date
    timestamp = getattr(date, "timestamp", None)
    if callable(timestamp):
        return int(timestamp())
    return 0


async def _collect_message_records(
    client: TelegramClient,
    chat: str | int,
    *,
    limit: int,
) -> list[MessageRecord]:
    """Collect phase-1 records before emitting callbacks, retrying safely on FloodWait.

    An iterator can fail only after yielding some messages. Records are deliberately
    buffered and callbacks delayed until a complete attempt succeeds, so restarting
    after a bounded FloodWait never replays an already-emitted callback.
    """
    for attempt in range(_BACKFILL_MAX_FLOOD_RETRIES + 1):
        records: list[MessageRecord] = []
        try:
            async for message in client.iter_messages(chat, limit=limit):
                reply = getattr(message, "reply_to", None)
                sender_id = getattr(message, "sender_id", None)
                records.append(
                    MessageRecord(
                        message_id=int(message.id),
                        sender_id=int(sender_id) if sender_id is not None else None,
                        reply_to_msg_id=(
                            int(reply.reply_to_msg_id)
                            if reply is not None
                            and getattr(reply, "reply_to_msg_id", None) is not None
                            else None
                        ),
                        date_epoch=_date_epoch(message),
                        has_text=bool(getattr(message, "message", None)),
                        has_media=getattr(message, "media", None) is not None,
                    )
                )
        except FloodWaitError as exc:
            seconds = int(getattr(exc, "seconds", 0) or 0)
            if seconds > _CONNECT_FLOOD_WAIT_CEILING_S or attempt >= _BACKFILL_MAX_FLOOD_RETRIES:
                raise ProvisioningError(
                    "flood_wait",
                    "Telegram FloodWait during phase-1 backfill. ACTION REQUIRED: back "
                    "off and retry after the wait window.",
                    retry_after_s=seconds,
                ) from exc
            logger.warning(
                "TG_PROVISION: phase-1 backfill FloodWait {}s; retrying before callbacks",
                seconds,
            )
            await asyncio.sleep(seconds)
        except Exception as exc:  # noqa: BLE001 -- classify fatal errors, else re-raise.
            _raise_if_fatal(exc)
            raise
        else:
            logger.info(
                "TG_PROVISION: backfill phase-1 collected {} message record(s)", len(records)
            )
            return records
    raise AssertionError("unreachable phase-1 backfill retry exhaustion")  # pragma: no cover


async def _resolve_entities(
    client: TelegramClient,
    records: list[MessageRecord],
    *,
    on_entity: OnEntity | None,
) -> list[EntityRecord]:
    """Resolve phase-2 entity references up to the documented total cap."""
    referenced = sorted({record.sender_id for record in records if record.sender_id is not None})
    if len(referenced) > MAX_ENTITIES:
        logger.warning(
            "TG_PROVISION: dropping {} entity reference(s) over total cap {}",
            len(referenced) - MAX_ENTITIES,
            MAX_ENTITIES,
        )
        referenced = referenced[:MAX_ENTITIES]
    entities: list[EntityRecord] = []
    for entity_id in referenced:

        async def _resolve(eid: int = entity_id) -> object:
            resolved: object = await client.get_entity(eid)
            return resolved

        try:
            entity = await _run_batch_rpc(_resolve, label="entity")
            record = EntityRecord(entity_id=entity_id, kind=_kind_of(entity), resolved=True)
        except ProvisioningError:
            raise
        except Exception as exc:  # noqa: BLE001 -- one bad id must not abort the phase.
            logger.warning(
                "TG_PROVISION: backfill could not resolve entity: {}", type(exc).__name__
            )
            record = EntityRecord(entity_id=entity_id, kind="unknown", resolved=False)
        entities.append(record)
        if on_entity is not None:
            await on_entity(record)
    logger.info(
        "TG_PROVISION: backfill phase-2 resolved {} of {} referenced entity id(s)",
        sum(1 for entity in entities if entity.resolved),
        len(entities),
    )
    return entities


async def _resolve_replies(
    records: list[MessageRecord],
    *,
    on_reply: OnReply | None,
) -> list[ReplyRecord]:
    """Resolve phase-3 reply relationships up to the documented total cap."""
    present_ids = {record.message_id for record in records}
    edges = [
        (record.message_id, record.reply_to_msg_id)
        for record in records
        if record.reply_to_msg_id is not None
    ]
    if len(edges) > MAX_REPLY_EDGES:
        logger.warning(
            "TG_PROVISION: dropping {} reply edge(s) over total cap {}",
            len(edges) - MAX_REPLY_EDGES,
            MAX_REPLY_EDGES,
        )
        edges = edges[:MAX_REPLY_EDGES]
    replies: list[ReplyRecord] = []
    for child_id, parent_id in edges:
        record = ReplyRecord(
            child_id=child_id,
            parent_id=parent_id,
            parent_present=parent_id in present_ids,
        )
        replies.append(record)
        if on_reply is not None:
            await on_reply(record)
    logger.info("TG_PROVISION: backfill phase-3 resolved {} reply edge(s)", len(replies))
    return replies


async def backfill_history(
    client: TelegramClient,
    chat: str | int,
    *,
    allowed_chats: frozenset[str],
    limit: int = DEFAULT_BACKFILL_LIMIT,
    on_message: OnMessage | None = None,
    on_entity: OnEntity | None = None,
    on_reply: OnReply | None = None,
) -> BackfillResult:
    """Read a bounded chat history in three ordered phases.

    The explicit ``allowed_chats`` capability must contain the requested canonical
    ``-100…`` chat id before any history/entity/reply access. Each optional callback
    is awaited once per record. Fatal auth errors,
    over-ceiling or exhausted FloodWaits, and peer-floods abort the whole
    backfill as a credential-free :class:`ProvisioningError`.
    """
    approved_chat = _require_scope(chat, allowed_chats)
    capped = max(0, min(int(limit), MAX_BACKFILL_LIMIT))
    if capped == 0:
        logger.info("TG_PROVISION: backfill limit is 0; skipping all phases for {}", chat)
        return BackfillResult(messages=[], entities=[], replies=[])

    messages = await _collect_message_records(client, approved_chat, limit=capped)
    if not messages:
        logger.info("TG_PROVISION: backfill found empty history for {}", chat)
        return BackfillResult(messages=[], entities=[], replies=[])

    if on_message is not None:
        for record in messages:
            await on_message(record)

    entities = await _resolve_entities(client, messages, on_entity=on_entity)
    replies = await _resolve_replies(messages, on_reply=on_reply)
    logger.info(
        "TG_PROVISION: backfill done — {} message(s), {} entity(ies), {} reply edge(s) from {}",
        len(messages),
        len(entities),
        len(replies),
        chat,
    )
    return BackfillResult(messages=messages, entities=entities, replies=replies)


__all__ = [
    "DEFAULT_BACKFILL_LIMIT",
    "MAX_BACKFILL_LIMIT",
    "MAX_ENTITIES",
    "MAX_REPLY_EDGES",
    "BackfillScopeError",
    "BackfillResult",
    "EntityRecord",
    "MessageRecord",
    "OnEntity",
    "OnMessage",
    "OnReply",
    "ReplyRecord",
    "backfill_history",
    "backfill_recent",
]
