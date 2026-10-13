"""Pure decision functions for the Discord behavioral contract.

NO discord.py imports here — everything operates on :class:`InboundMessage` and
the typed config so every branch is unit-testable without a gateway. This module
is the deliverable: it reproduces the Hermes gating + auto-thread contract
exactly.

Two public entry points:

* :func:`should_respond` — may the agent answer this message at all?
* :func:`thread_decision` — should a fresh auto-thread be created, and named
  what?

The Hermes source these port from (v0_17_0 discord adapter):
``on_message`` :988-1085 (self/bot/allowlist/multi-agent gates),
``_handle_message`` :5148-5260 (channel lists, mention gate, auto-thread),
``_auto_create_thread`` :4561-4600 (thread-name derivation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from loguru import logger

from .channel import InboundMessage
from .config import (
    WILDCARD,
    AllowBots,
    GateConfig,
    IntakeConfig,
    IntakeMode,
    ThreadConfig,
)


@dataclass(frozen=True)
class ThreadDecision:
    """Outcome of :func:`thread_decision`.

    Attributes:
        create: True when a new auto-thread should be created.
        name: The derived thread name (only meaningful when ``create`` is
            True), markup-stripped and capped.
        reason: A short label for logs/tests explaining the decision.
    """

    create: bool
    name: str
    reason: str


def _channel_ids(msg: InboundMessage) -> frozenset[str]:
    """Return the id set that channel allow/ignore/free lists match against.

    A threaded message matches on BOTH its thread id and its parent channel id,
    so a per-parent policy still applies inside auto-threads (Hermes adds the
    parent id to ``channel_ids``).
    """
    ids = {msg.channel_id}
    if msg.thread_id:
        ids.add(msg.thread_id)
    return frozenset(ids)


def _wildcard_or_member(candidate: frozenset[str], ids: frozenset[str]) -> bool:
    """True when ``candidate`` contains ``*`` or intersects ``ids``."""
    return WILDCARD in candidate or bool(candidate & ids)


def _author_allowed(msg: InboundMessage, cfg: GateConfig, *, role_authorized: bool) -> bool:
    """Return whether a NON-bot author passes the user/role allowlists.

    Empty ``allowed_users`` means everyone is allowed (the Hermes default). A
    role match short-circuits the user check (``role_authorized`` is computed by
    the adapter, which alone can see the author's guild roles).
    """
    if not cfg.allowed_users and not cfg.allowed_roles:
        return True
    if role_authorized:
        return True
    if not cfg.allowed_users:
        # Roles configured but no user list, and no role match → not allowed.
        return False
    return msg.author in cfg.allowed_users


def should_respond(
    msg: InboundMessage,
    cfg: GateConfig,
    *,
    is_self: bool = False,
    role_authorized: bool = False,
) -> bool:
    """Decide whether the agent may respond to ``msg``.

    Ordering mirrors the Hermes contract precisely:

    1. Never answer our own message.
    2. Bot-author filter via ``allow_bots`` (none|mentions|all) — this runs
       BEFORE the human allowlist so an allowed bot is not rejected for being
       absent from ``allowed_users``.
    3. Non-bot authors must pass the user/role allowlist (empty = everyone).
    4. DMs fail closed unless ``allow_dms`` is explicitly enabled (security
       floor: a DM is a free-response surface outside the designated inbox).
       When allowed, DMs are never gated further — an allowed author in a DM
       always gets a response (mention/channel/thread checks are skipped).
    5. Channel whitelist / blacklist (guild only).
    6. Multi-agent filter: if the message @mentions another bot (or a human)
       but not us, stay silent — unless the channel is free-response.
    7. Mention requirement in guilds, waived only by a free-response channel.

    Args:
        msg: The normalised inbound message.
        cfg: The gate configuration.
        is_self: True when the author is this agent (adapter-supplied).
        role_authorized: True when the author holds an allowed guild role.

    Returns:
        True to respond, False to stay silent.
    """
    logger.debug(
        "gates.should_respond | dm={} bot={} mentions_me={} thread={} reply={}",
        msg.is_dm,
        msg.is_bot,
        msg.mentions_me,
        msg.thread_id,
        msg.is_reply,
    )

    # 1. Never answer ourselves.
    if is_self:
        logger.debug("gates.should_respond | drop=self")
        return False

    # 2. Bot-author filter (BEFORE the human allowlist — Hermes #4466).
    if msg.is_bot:
        if cfg.allow_bots is AllowBots.NONE:
            logger.debug("gates.should_respond | drop=bot allow_bots=none")
            return False
        if cfg.allow_bots is AllowBots.MENTIONS and not msg.mentions_me:
            logger.debug("gates.should_respond | drop=bot allow_bots=mentions no-mention")
            return False
        # allow_bots=all, or =mentions with a mention → bot permitted; the human
        # allowlist below is skipped (bots aren't in it).
    else:
        # 3. Non-bot: enforce user/role allowlist.
        if not _author_allowed(msg, cfg, role_authorized=role_authorized):
            logger.debug("gates.should_respond | drop=user-not-allowed")
            return False

    # 4. DMs fail closed unless explicitly allowed (security floor: a DM is a
    #    free-response surface outside the designated inbox). When allowed,
    #    DMs are never gated further.
    if msg.is_dm:
        if not cfg.allow_dms:
            logger.debug("gates.should_respond | drop=dm-not-allowed")
            return False
        logger.debug("gates.should_respond | allow=dm")
        return True

    ids = _channel_ids(msg)

    # 5. Channel whitelist / blacklist.
    if cfg.allowed_channels and not _wildcard_or_member(cfg.allowed_channels, ids):
        logger.debug("gates.should_respond | drop=not-in-allowed-channels ids={}", set(ids))
        return False
    if _wildcard_or_member(cfg.ignored_channels, ids):
        logger.debug("gates.should_respond | drop=ignored-channel ids={}", set(ids))
        return False

    is_free = _wildcard_or_member(cfg.free_response_channels, ids)

    # 6. Multi-agent filter: message addressed to another bot/human, not us.
    #    Free-response channels answer regardless of who was mentioned.
    if not is_free:
        if msg.mentions_other_bot and not msg.mentions_me:
            logger.debug("gates.should_respond | drop=other-bot-mentioned")
            return False
        if msg.mentions_human and not msg.mentions_me:
            logger.debug("gates.should_respond | drop=human-mentioned")
            return False

    # 7. Mention requirement, waived by free-response only.
    if cfg.require_mention and not is_free:
        if not msg.mentions_me:
            logger.debug("gates.should_respond | drop=mention-required")
            return False

    logger.debug("gates.should_respond | allow free={}", is_free)
    return True


def _intake_channel_in_scope(
    msg: InboundMessage, gate_cfg: GateConfig, intake_cfg: IntakeConfig
) -> bool:
    """Return whether ``msg``'s channel is in intake scope.

    ALL_CHANNELS widens intake to every readable guild channel EXCEPT ones
    explicitly ignored (``ignored_channels`` is still honored — a blacklisted
    channel is never an intake source). LISTED preserves today's scoping: intake
    only fires where the existing gate config already responds
    (``allowed_channels`` / ``free_response_channels``); an empty allow list in
    LISTED mode means no widening, so no channel is in intake scope.
    """
    ids = _channel_ids(msg)

    # Ignored channels are never an intake source, in either mode.
    if _wildcard_or_member(gate_cfg.ignored_channels, ids):
        return False

    if intake_cfg.mode is IntakeMode.ALL_CHANNELS:
        return True

    # LISTED: only channels the existing config already scopes to.
    scoped = gate_cfg.allowed_channels | gate_cfg.free_response_channels
    return _wildcard_or_member(scoped, ids)


def _intake_content_hit(msg: InboundMessage, intake_cfg: IntakeConfig) -> bool:
    """Return whether the message content/mentions match a configured trigger.

    A HIT iff ANY of:

    * the message @mentions a configured trio ``member_id`` (id-set match), OR
    * the stripped message text starts with a configured ``trigger_prefix`` as a
      whole token — the prefix must be followed by end-of-string or whitespace,
      so ``!ask`` fires on ``!ask foo`` and ``!ask`` but NOT ``!asking about
      lunch`` (case-insensitive), OR
    * the message text contains a configured ``name_trigger`` as a whole word
      (case-insensitive).

    Purely deterministic — no LLM classification of chatter.
    """
    if intake_cfg.member_ids & msg.mentioned_ids:
        return True

    lowered = msg.text.strip().lower()

    for prefix in intake_cfg.trigger_prefixes:
        token = prefix.strip().lower()
        if not token or not lowered.startswith(token):
            continue
        # Word-boundary guard: the prefix must be the whole leading token, i.e.
        # followed by end-of-string or whitespace, so ``!ask`` does not fire on
        # ``!asking``.
        rest = lowered[len(token) :]
        if not rest or rest[0].isspace():
            return True

    if intake_cfg.name_triggers:
        words = set(re.findall(r"[a-z0-9_]+", lowered))
        for trigger in intake_cfg.name_triggers:
            token = trigger.strip().lower()
            if token and token in words:
                return True

    return False


def intake_hit(
    msg: InboundMessage,
    gate_cfg: GateConfig,
    intake_cfg: IntakeConfig,
    *,
    is_self: bool = False,
    role_authorized: bool = False,
) -> bool:
    """Decide whether ``msg`` is an intake HIT to dispatch to the triage agent.

    Intake is the always-on listener of issue #93. It is DETERMINISTIC: no LLM
    classification of raw channel chatter. Unlike :func:`should_respond`, a hit
    does NOT require an @mention of THIS bot — it fires on a mention of any
    configured trio member or a configured trigger — and it can widen beyond the
    responded-in channels via :class:`IntakeMode`.

    The humans-only floor is unchanged: intake reuses the same self / bot /
    author-allowlist gates as :func:`should_respond`, so under the default
    ``AllowBots.NONE`` no bot message is ever an intake hit. DMs are excluded —
    intake is a channel listener; a DM already flows through the normal responder
    (and is itself fail-closed behind ``allow_dms``).

    Ordering:

    1. Intake disabled → never a hit.
    2. Never our own message.
    3. Bot-author policy (same ``allow_bots`` semantics as the responder).
    4. Non-bot authors must pass the user/role allowlist.
    5. DMs are not an intake source.
    6. Channel must be in intake scope (mode-dependent; ignored channels out).
    7. Content/mention trigger match.

    Args:
        msg: The normalised inbound message.
        gate_cfg: Baseline gate config, reused for the humans-only + ignored-
            channel floor so intake never relaxes ``allow_bots`` or
            ``ignored_channels``.
        intake_cfg: The intake hit-gate configuration.
        is_self: True when the author is this agent (adapter-supplied).
        role_authorized: True when the author holds an allowed guild role.

    Returns:
        True when the message should be minted into a triage TaskFrame.
    """
    if not intake_cfg.enabled:
        return False

    if is_self:
        logger.debug("gates.intake_hit | drop=self")
        return False

    # Humans-only floor — identical bot policy to the responder (#93 keeps the
    # DISCORD_ALLOW_BOTS baseline). Intake never relaxes it.
    if msg.is_bot:
        if gate_cfg.allow_bots is AllowBots.NONE:
            logger.debug("gates.intake_hit | drop=bot allow_bots=none")
            return False
        if gate_cfg.allow_bots is AllowBots.MENTIONS and not msg.mentions_me:
            logger.debug("gates.intake_hit | drop=bot allow_bots=mentions no-mention")
            return False
    else:
        if not _author_allowed(msg, gate_cfg, role_authorized=role_authorized):
            logger.debug("gates.intake_hit | drop=user-not-allowed")
            return False

    # Intake is a channel listener; DMs go through the normal responder.
    if msg.is_dm:
        logger.debug("gates.intake_hit | drop=dm")
        return False

    if not _intake_channel_in_scope(msg, gate_cfg, intake_cfg):
        logger.debug("gates.intake_hit | drop=channel-out-of-scope")
        return False

    if not _intake_content_hit(msg, intake_cfg):
        logger.debug("gates.intake_hit | drop=no-trigger")
        return False

    logger.info("gates.intake_hit | hit channel={} thread={}", msg.channel_id, msg.thread_id)
    return True


# <@123>, <@!123>, <@&123> (user/role mentions) and <#123> (channel mentions).
_USER_ROLE_MENTION = re.compile(r"<@[!&]?\d+>")
_CHANNEL_MENTION = re.compile(r"<#\d+>")
_WHITESPACE = re.compile(r"\s+")


def _derive_thread_name(text: str, cfg: ThreadConfig) -> str:
    """Derive a human-readable thread name from ``text``.

    Strips Discord mention markup (so titles never show raw ``<@id>``),
    collapses whitespace, and caps to ``thread_name_max`` with an ellipsis when
    truncated. Falls back to ``default_thread_name`` on an empty result.
    """
    stripped = _USER_ROLE_MENTION.sub("", text)
    stripped = _CHANNEL_MENTION.sub("", stripped)
    stripped = _WHITESPACE.sub(" ", stripped).strip()

    if not stripped:
        return cfg.default_thread_name

    if len(stripped) <= cfg.thread_name_max:
        return stripped
    # Reserve 3 chars for the ellipsis so the total stays within the cap.
    keep = max(1, cfg.thread_name_max - 3)
    return stripped[:keep] + "..."


def thread_decision(msg: InboundMessage, cfg: ThreadConfig) -> ThreadDecision:
    """Decide whether to auto-thread ``msg`` and what to name the thread.

    Auto-threading is SKIPPED (answer inline) when ANY of these hold — these are
    the three no-thread couplings the contract turns on:

    * ``auto_thread`` is disabled, OR
    * the message is already inside a thread, OR a DM (nothing to thread), OR
    * the channel is a **free-response** channel (free-response ⇒ no-thread), OR
    * the channel is in **no_thread_channels**, OR
    * the message is a **reply** (reply ⇒ no-thread).

    Args:
        msg: The normalised inbound message.
        cfg: Thread configuration.

    Returns:
        A :class:`ThreadDecision`. When ``create`` is False the ``name`` is the
        fallback and ``reason`` states which coupling fired.
    """
    logger.debug(
        "gates.thread_decision | auto_thread={} thread={} dm={} reply={} channel={}",
        cfg.auto_thread,
        msg.thread_id,
        msg.is_dm,
        msg.is_reply,
        msg.channel_id,
    )

    if not cfg.auto_thread:
        return ThreadDecision(False, cfg.default_thread_name, "auto-thread-disabled")

    # Already inside a thread, or a DM — there is nothing to thread from.
    if msg.thread_id is not None:
        return ThreadDecision(False, cfg.default_thread_name, "already-in-thread")
    if msg.is_dm:
        return ThreadDecision(False, cfg.default_thread_name, "dm")

    ids = _channel_ids(msg)

    # free-response ⇒ no-thread.
    if _wildcard_or_member(cfg.free_response_channels, ids):
        return ThreadDecision(False, cfg.default_thread_name, "free-response-channel")

    # explicit no-thread channel.
    if _wildcard_or_member(cfg.no_thread_channels, ids):
        return ThreadDecision(False, cfg.default_thread_name, "no-thread-channel")

    # reply ⇒ no-thread.
    if msg.is_reply:
        return ThreadDecision(False, cfg.default_thread_name, "reply")

    name = _derive_thread_name(msg.text, cfg)
    logger.debug("gates.thread_decision | create name={!r}", name)
    return ThreadDecision(True, name, "auto-thread")


__all__ = [
    "ThreadDecision",
    "intake_hit",
    "should_respond",
    "thread_decision",
]
