"""Typed, immutable configuration for the chat gate + thread decisions.

Every knob here mirrors a Hermes ``discord.*`` / ``DISCORD_*`` setting whose
behavior the pure decision functions in :mod:`rtech_chatd.gates` reproduce. The
adapter reads live config (config ref / agent.yaml) and builds one of these; the
gates never read the environment themselves, which keeps them pure and testable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from loguru import logger
from rtech_frames.frames import validate_participants

from .channel import SendTarget

# Debate participant count -> duo/trio scope. A debate is a MULTI-participant
# exchange, so a single participant is not a debate (use single-scope intake);
# more than three exceeds the ratified ``TaskFrame`` scope vocabulary.
_DEBATE_SCOPE_BY_COUNT: dict[int, Literal["duo", "trio"]] = {2: "duo", 3: "trio"}

# Deployed agent ids follow the ``agentd-<name>`` convention (deploy/kboard/
# install.sh AGENT_INSTANCES, e.g. ``agentd-bilby``). A debate participant that is
# merely subject-safe (``nagatha``) passes ``validate_participants`` yet addresses
# ``rtech.agent.nagatha.task`` — a subject no deployed agentd subscribes — so the
# fanout would silently vanish. This is the SAME trap fixed for the wake allow-list
# in ``rtech_agentd.wake_service`` (``_ALLOWED_AGENT_CONVENTION``); mirror the exact
# regex so a mis-set debate roster fails loud at config build, not mid-debate.
_AGENTD_CONVENTION = re.compile(r"^agentd-[a-z0-9][a-z0-9-]*$")

# Wildcard entry: an allow/free/ignore set containing "*" matches every channel.
WILDCARD = "*"

DEFAULT_NATS_URL = "nats://127.0.0.1:4222"
DEFAULT_ORIGIN_EDGE = "discord"

# Chat-scoped debate-lane token validation. A chat id is a signed integer
# (``-100…`` supergroup or a smaller signed id); a topic id is a positive
# integer (a forum topic id is always >= 1). Both are matched with ``fullmatch``
# so a partial/garbage token is rejected, not silently coerced.
_SIGNED_INTEGER = re.compile(r"[+-]?\d+")
_POSITIVE_INTEGER = re.compile(r"[1-9]\d*")


class AllowBots(StrEnum):
    """Bot-author policy, matching Hermes ``DISCORD_ALLOW_BOTS``.

    NONE: ignore every other bot (default — "treat Discord as leaked").
    MENTIONS: accept a bot's message only when it @mentions us.
    ALL: accept every bot message.
    """

    NONE = "none"
    MENTIONS = "mentions"
    ALL = "all"

    @classmethod
    def parse(cls, raw: str | None) -> AllowBots:
        """Parse a raw config value; unknown/empty falls back to NONE."""
        if not raw:
            return cls.NONE
        try:
            return cls(raw.strip().lower())
        except ValueError:
            return cls.NONE


@dataclass(frozen=True)
class GateConfig:
    """Inputs to :func:`rtech_chatd.gates.should_respond`.

    Attributes:
        require_mention: In guild channels, require an @mention unless the
            channel is free-response.
        allow_bots: Bot-author policy (see :class:`AllowBots`).
        allowed_channels: Whitelist; if non-empty (and no ``*``), respond ONLY
            in these channels. Empty = no channel whitelist.
        ignored_channels: Never respond in these channels, even when mentioned.
        free_response_channels: Channels where no mention is required.
        allowed_users: Author allowlist; empty = everyone allowed.
        allowed_roles: Role-id allowlist; empty = no role gate. A message from
            an author holding any listed role passes even if not in
            ``allowed_users``.
        allow_dms: Explicit opt-in for direct messages. Defaults to False
            (fail closed): a DM is a free-response surface outside the
            designated inbox channel, which the security floor forbids unless
            an operator deliberately enables it.
    """

    require_mention: bool = True
    allow_bots: AllowBots = AllowBots.NONE
    allow_dms: bool = False
    allowed_channels: frozenset[str] = field(default_factory=frozenset)
    ignored_channels: frozenset[str] = field(default_factory=frozenset)
    free_response_channels: frozenset[str] = field(default_factory=frozenset)
    allowed_users: frozenset[str] = field(default_factory=frozenset)
    allowed_roles: frozenset[str] = field(default_factory=frozenset)


class IntakeMode(StrEnum):
    """Channel scope for the always-on intake listener.

    LISTED: intake hits only fire in channels the existing gate config already
        scopes to (``allowed_channels`` / ``free_response_channels``). This is
        the default so existing single-channel deploys are unchanged.
    ALL_CHANNELS: intake hits fire in ANY guild channel the bot can read (the
        trio deploy opts into this). The mention/trigger hit-gate still decides
        each message; this only widens WHERE it is evaluated. ``ignored_channels``
        is still honored in both modes.
    """

    LISTED = "listed"
    ALL_CHANNELS = "all-channels"

    @classmethod
    def parse(cls, raw: str | None) -> IntakeMode:
        """Parse a raw config value; unknown/empty falls back to LISTED."""
        if not raw:
            return cls.LISTED
        try:
            return cls(raw.strip().lower())
        except ValueError:
            return cls.LISTED


_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _env_flag(name: str) -> bool:
    """Return True when env var ``name`` is a recognised truthy token."""
    return os.getenv(name, "").strip().lower() in _TRUTHY


def _env_token_set(name: str) -> frozenset[str]:
    """Parse a comma-separated env var into a frozenset of non-blank tokens."""
    raw = os.getenv(name, "")
    return frozenset(tok.strip() for tok in raw.split(",") if tok.strip())


def _env_token_tuple(name: str) -> tuple[str, ...]:
    """Parse a comma-separated env var into an ORDERED tuple of non-blank tokens.

    Unlike :func:`_env_token_set`, order is preserved: debate participant order is
    the typed identity a duo/trio ``TaskFrame`` carries and the orchestrator fans
    out in that order.
    """
    raw = os.getenv(name, "")
    return tuple(tok.strip() for tok in raw.split(",") if tok.strip())


def _env_lane_tuple(name: str) -> tuple[tuple[str, str], ...]:
    """Parse a ``name=topic,name=topic`` env var into ordered (name, topic) pairs.

    Blank or ``=``-less entries are skipped (edge-side parsing normalizes and
    warns on the survivors), so a malformed lane never fails config construction —
    the agent simply falls back to a per-turn thread.

    Topic ids are validated as POSITIVE INTEGERS exactly like the chat-scoped
    twin :func:`_env_chat_lane_tuple` (a Telegram forum topic id is always >= 1):
    a non-integer topic is a lane that could never route, so it is skipped with a
    sanitized warning (the rejected value is never logged) and the agent degrades
    to a per-turn thread rather than the map silently carrying an unusable id.
    """
    raw = os.getenv(name, "")
    pairs: list[tuple[str, str]] = []
    for index, token in enumerate((tok.strip() for tok in raw.split(",") if tok.strip()), start=1):
        name_part, sep, topic = token.partition("=")
        name_part = name_part.strip()
        topic = topic.strip()
        if not (sep and name_part and topic):
            logger.warning(
                "chatd.config | skipping invalid debate lane entry index={} reason={}",
                index,
                "expected name=topic with no empty fields",
            )
            continue
        if _POSITIVE_INTEGER.fullmatch(topic) is None:
            logger.warning(
                "chatd.config | skipping invalid debate lane entry index={} reason={}",
                index,
                "topic id must be a positive integer",
            )
            continue
        pairs.append((name_part, topic))
    return tuple(pairs)


def _env_chat_lane_tuple(name: str) -> tuple[tuple[str, str, str], ...]:
    """Parse a ``chat:name=topic,chat:name=topic`` env var into ordered triples.

    This is the CHAT-SCOPED extension of :func:`_env_lane_tuple`: each entry
    binds an agent's persistent lane to ONE specific chat id, so the SAME agent
    can route to a DIFFERENT topic id per chat (the trio runs the same three
    voices in multiple groups, each group with its own forum topics). Shape per
    entry: ``<chat_id>:<name>=<topic>`` — the chat id is a signed-integer
    Telegram chat id, ``:`` separates it from the agent lane pair, and ``=``
    separates the agent name from its topic id.

    Each nonblank malformed entry emits a sanitized warning and is skipped. Chat
    ids must be signed integers and topic ids must be positive integers; rejected
    values are never included in the warning. A misconfigured lane therefore
    degrades to the legacy lane or per-turn thread without silently accepting an
    unusable mapping. Returns ordered ``(chat_id, name, topic)`` triples (order is
    irrelevant to the resulting map but preserved for stable last-write-wins
    semantics).
    """
    raw = os.getenv(name, "")
    triples: list[tuple[str, str, str]] = []
    tokens = (tok.strip() for tok in raw.split(",") if tok.strip())
    for index, token in enumerate(tokens, start=1):
        chat_part, chat_sep, lane = token.partition(":")
        chat_part = chat_part.strip()
        name_part, eq_sep, topic = lane.partition("=")
        name_part = name_part.strip()
        topic = topic.strip()
        reason: str | None = None
        if not (chat_sep and eq_sep and chat_part and name_part and topic):
            reason = "expected chat:name=topic with no empty fields"
        elif _SIGNED_INTEGER.fullmatch(chat_part) is None:
            reason = "chat id must be a signed integer"
        elif int(chat_part) == 0:
            reason = "chat id must be non-zero"
        elif _POSITIVE_INTEGER.fullmatch(topic) is None:
            reason = "topic id must be a positive integer"
        if reason is not None:
            logger.warning(
                "chatd.config | skipping invalid chat-scoped debate lane entry index={} reason={}",
                index,
                reason,
            )
            continue
        triples.append((str(int(chat_part)), name_part, topic))
    return tuple(triples)


@dataclass(frozen=True)
class IntakeConfig:
    """Deterministic intake hit-gate configuration.

    Intake is OFF unless ``enabled`` is true, so an existing deploy that builds a
    default :class:`IntakeConfig` sees no behavior change. When on, a message is a
    HIT iff it @mentions a configured trio member (by id) OR contains a
    configured name trigger as a whole word OR begins with a configured trigger
    prefix. There is NO LLM classification of channel chatter — the gate is pure
    string/id matching.

    The ``DISCORD_ALLOW_BOTS`` baseline is untouched: intake reuses the
    responder's bot policy, so under the default ``AllowBots.NONE`` no bot
    message is ever an intake hit (humans only).

    Double-activation is deliberate: a message that both @mentions the triage
    agent AND matches an intake trigger fires the responder AND intake for the
    same message id. This slice does NOT dedup that — the agentd-side sink is
    responsible for dedup keyed by message_id.

    Attributes:
        enabled: Master switch. False (default) = no intake behavior.
        mode: Channel scope (see :class:`IntakeMode`); default LISTED preserves
            today's channel scoping.
        triage_agent: The agent id a hit is dispatched to. There is NO default:
            a hit publishes to ``rtech.agent.<triage_agent>.task``, and a
            deployed agentd subscribes under its FULL id (``RTECH_AGENT_ID`` =
            ``agentd-nagatha``, per ``rtech_agentd.identity`` +
            ``deploy/kboard/install.sh``). A bare-name default (``nagatha``)
            would silently publish into the void, so when intake is enabled this
            MUST be set explicitly to the deployed ``agentd-<name>`` id (mirrors
            the orchestrator's fail-loud ``resolve_target_agent_id``). The
            disabled-path placeholder below is inert and never reaches a publish.
        member_ids: Trio member Discord user/bot ids; an @mention of any is a
            hit (matched against the message's mentioned-id set).
        name_triggers: Case-insensitive name tokens (e.g. ``nagatha``); a
            whole-word occurrence in the message text is a hit.
        trigger_prefixes: Literal prefixes (e.g. ``!ask``); a message whose
            stripped text starts with any (case-insensitive) is a hit.
        debate_participants: The ordered agentd ids a hit fans out to as a REAL
            multi-participant sibling debate (duo/trio). EMPTY/UNSET (default)
            preserves single-scope intake EXACTLY as today — one triage TaskFrame
            to ``triage_agent``. When set, a hit mints ONE duo/trio TaskFrame with
            these participants and the orchestrator fans it verbatim to each
            participant's task lane (mirroring ``rtech_orchestrator`` ``_route_multi``).
            Every id must be a NATS-subject-safe ``agentd-<name>`` token (same
            fail-loud convention as ``triage_agent``), the count must be 2 or 3
            (duo/trio — ``TaskFrame`` scope semantics), and ``triage_agent`` MUST be
            one of the participants (the triage agent is a debate voice, not a
            bystander).
    """

    enabled: bool = False
    mode: IntakeMode = IntakeMode.LISTED
    # Placeholder only for the disabled path (existing deploys build a default
    # IntakeConfig and never dispatch). A blank value NEVER reaches a publish:
    # __post_init__ fails closed the moment intake is enabled without a real id.
    triage_agent: str = ""
    member_ids: frozenset[str] = field(default_factory=frozenset)
    name_triggers: frozenset[str] = field(default_factory=frozenset)
    trigger_prefixes: frozenset[str] = field(default_factory=frozenset)
    # Ordered (a list, not a set): participant order is the typed identity a
    # duo/trio TaskFrame carries, and the orchestrator fans out in that order.
    debate_participants: tuple[str, ...] = ()
    # Persistent per-agent verbose lanes: a ``name=topic,name=topic`` map of an
    # agent's normalized name to its stable surface thread id. When set, each
    # agent's debate full text routes to its own durable lane every turn instead
    # of minting a fresh per-turn thread. Empty (default) keeps the lazy per-turn
    # thread behavior, so a surface with no lanes configured is unchanged. This is
    # the LEGACY, single-group agent->topic map — its behavior is preserved
    # byte-for-byte as the fallback under the chat-scoped map below.
    debate_lanes: tuple[tuple[str, str], ...] = ()
    # Chat-scoped persistent lanes: ordered ``(chat_id, name, topic)`` triples
    # binding an agent's lane to ONE specific chat id, so the SAME agent routes to
    # a DIFFERENT topic id per chat (the trio runs the same voices in several
    # groups, each with its own forum topics). Empty (default) keeps ONLY the
    # legacy agent-only ``debate_lanes`` behavior. Selection order at the edge is
    # exact-chat+agent FIRST, then the legacy agent-only lane, then per-turn thread
    # creation — this map is purely ADDITIVE and never drops an unlisted
    # chat/agent's full text.
    debate_chat_lanes: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        # When intake is on, a blank triage agent id would build the subject
        # ``rtech.agent..task`` and never wake any agent, so the config fails
        # closed rather than mint a dead dispatch subject. There is NO bare-name
        # default: a deployed agentd subscribes under its FULL id
        # (``agentd-<name>``), so the operator must name that id explicitly.
        # Disabled config is left untouched (existing deploys see no change).
        if self.enabled and _AGENTD_CONVENTION.fullmatch(self.triage_agent) is None:
            raise ValueError(
                "RTECH_CHATD_TRIAGE_AGENT must match the full agentd-<name> "
                "convention when intake is enabled (e.g. 'agentd-nagatha'); "
                "the value becomes rtech.agent.<id>.task and must equal the "
                "deployed agentd's RTECH_AGENT_ID -- a blank or bare name like "
                "'nagatha' publishes to a subject no deployed agentd subscribes to"
            )
        # Debate fanout (trio intake): validate the participant set fail-loud, the
        # SAME way ``triage_agent`` and the wire (``validate_participants``) do, so
        # a misconfigured debate is caught at construction, not at the first live
        # hit. Empty is the untouched single-scope path (validated above only for
        # the triage id) — validation engages ONLY when participants are configured.
        if self.debate_participants:
            scope = _DEBATE_SCOPE_BY_COUNT.get(len(self.debate_participants))
            if scope is None:
                raise ValueError(
                    "RTECH_CHATD_DEBATE_PARTICIPANTS must name 2 (duo) or 3 (trio) "
                    f"agentd ids, got {len(self.debate_participants)}: a debate is a "
                    "multi-participant sibling exchange; use single-scope intake "
                    "(unset this) for one triage agent"
                )
            # Reuse the wire invariant: subject-safe, unique, count matches scope.
            validate_participants(list(self.debate_participants), scope)
            # Subject-safe is necessary but not sufficient: a deployed agentd only
            # subscribes ``rtech.agent.<agentd-name>.task``, so a bare id like
            # ``nagatha`` addresses a subject nobody listens on and the fanout
            # silently vanishes. Enforce the ``agentd-<name>`` convention (same
            # fail-loud check as the wake allow-list) so a mis-set roster fails at
            # config build, not mid-debate. Applied whenever participants are
            # configured, regardless of ``enabled``, so a misconfigured roster is
            # never accepted.
            for participant in self.debate_participants:
                if _AGENTD_CONVENTION.match(participant) is None:
                    raise ValueError(
                        f"RTECH_CHATD_DEBATE_PARTICIPANTS id {participant!r} does not "
                        "follow the required agentd-<name> convention: every id must "
                        r"match '^agentd-[a-z0-9][a-z0-9-]*$' (e.g. 'agentd-nagatha'); "
                        "a bare name addresses rtech.agent.<name>.task, a subject no "
                        "deployed agentd subscribes, so the fanout would silently vanish"
                    )
            if self.enabled and self.triage_agent not in self.debate_participants:
                raise ValueError(
                    "RTECH_CHATD_TRIAGE_AGENT must be one of "
                    "RTECH_CHATD_DEBATE_PARTICIPANTS: the triage agent is a voice in "
                    f"the debate, not a bystander (triage={self.triage_agent!r}, "
                    f"participants={list(self.debate_participants)!r})"
                )

    @property
    def debate_scope(self) -> Literal["duo", "trio"] | None:
        """Return the duo/trio scope of the configured debate, or None when unset.

        None is the single-scope path (no debate participants). A configured set
        is already count-validated in ``__post_init__``, so the lookup is total.
        """
        if not self.debate_participants:
            return None
        return _DEBATE_SCOPE_BY_COUNT[len(self.debate_participants)]

    @classmethod
    def from_env(cls) -> IntakeConfig:
        """Build intake config from the environment.

        Intake is OFF unless ``RTECH_CHATD_INTAKE_ENABLED`` is truthy, so an
        existing deploy that sets none of these vars builds a disabled config and
        sees no behavior change. Env names follow the existing ``RTECH_CHATD_*``
        convention in this module; every id/trigger value is a plain
        comma-separated token list (Discord ids are opaque public ids, not
        secrets — the ONLY credential ref remains the bot token in
        :class:`ChatdConfig`).

        Vars:
            RTECH_CHATD_INTAKE_ENABLED: ``1``/``true``/``yes``/``on`` enables it.
            RTECH_CHATD_INTAKE_MODE: ``listed`` (default) or ``all-channels``.
            RTECH_CHATD_TRIAGE_AGENT: triage agent id. REQUIRED (no default)
                when intake is enabled; must be the deployed ``agentd-<name>``
                id (e.g. ``agentd-nagatha``). A blank/unset value fails closed in
                ``__post_init__`` rather than defaulting to a bare name that
                would publish into the void.
            RTECH_CHATD_INTAKE_MEMBER_IDS: comma-separated trio member ids.
            RTECH_CHATD_INTAKE_NAME_TRIGGERS: comma-separated name tokens.
            RTECH_CHATD_INTAKE_TRIGGER_PREFIXES: comma-separated literal prefixes.
            RTECH_CHATD_DEBATE_PARTICIPANTS: comma-separated agentd ids the hit
                fans out to as a duo/trio debate. Unset/empty (default) keeps
                single-scope intake. Order is preserved (the typed participant
                identity), so this parses to an ordered tuple, NOT a set.
            RTECH_CHATD_DEBATE_LANES: ``name=topic,name=topic`` LEGACY
                single-group persistent per-agent verbose lanes; each agent's
                debate full text routes to its own stable thread every turn
                instead of a per-turn thread. Preserved byte-for-byte as the
                fallback under the chat-scoped map.
            RTECH_CHATD_DEBATE_CHAT_LANES: ``chat:name=topic,chat:name=topic``
                CHAT-SCOPED persistent lanes — the same agent name may map to a
                DIFFERENT topic id per chat id. Additive: an exact chat+agent
                match wins, else the legacy agent-only lane, else a per-turn
                thread. Unlisted chats/agents never lose full text.
        """
        return cls(
            enabled=_env_flag("RTECH_CHATD_INTAKE_ENABLED"),
            mode=IntakeMode.parse(os.getenv("RTECH_CHATD_INTAKE_MODE")),
            triage_agent=os.getenv("RTECH_CHATD_TRIAGE_AGENT", "").strip(),
            member_ids=_env_token_set("RTECH_CHATD_INTAKE_MEMBER_IDS"),
            name_triggers=_env_token_set("RTECH_CHATD_INTAKE_NAME_TRIGGERS"),
            trigger_prefixes=_env_token_set("RTECH_CHATD_INTAKE_TRIGGER_PREFIXES"),
            debate_participants=_env_token_tuple("RTECH_CHATD_DEBATE_PARTICIPANTS"),
            debate_lanes=_env_lane_tuple("RTECH_CHATD_DEBATE_LANES"),
            debate_chat_lanes=_env_chat_lane_tuple("RTECH_CHATD_DEBATE_CHAT_LANES"),
        )


@dataclass(frozen=True)
class ThreadConfig:
    """Inputs to :func:`rtech_chatd.gates.thread_decision`.

    Attributes:
        auto_thread: Master switch for auto-thread on a trigger message.
        no_thread_channels: Channels where the bot answers inline (no thread).
        free_response_channels: Free-response channels also imply no-thread
            (the free-response⇒no-thread coupling).
        thread_name_max: Char budget for a derived thread name (~80).
        default_thread_name: Name used when the stripped body is empty.
    """

    auto_thread: bool = True
    no_thread_channels: frozenset[str] = field(default_factory=frozenset)
    free_response_channels: frozenset[str] = field(default_factory=frozenset)
    thread_name_max: int = 80
    default_thread_name: str = "chat"


# Telegram long-poll update kinds this edge subscribes to. ``message`` and
# ``edited_message`` are the inbound surface (edits are logged+dropped, never
# crash — v1 policy). ``message_reaction`` is opt-in on the Bot API:
# reactions are NEVER auto-included in getUpdates ``allowed_updates`` — a bot that
# does not list them explicitly is silently blind to every reaction. We list it
# now so the outbound + reactions slice
# inherits a subscription already receiving the events; the inbound half here
# does not yet act on a reaction, it just no longer drops it at the transport.
# ``callback_query`` is the inline-keyboard tap surface — like
# ``message_reaction`` it is NEVER auto-included, so a bot that omits it is
# silently blind to every button tap. It is listed so the interactive layer's
# taps actually reach the poll loop.
DEFAULT_TELEGRAM_ALLOWED_UPDATES: tuple[str, ...] = (
    "message",
    "edited_message",
    "message_reaction",
    "callback_query",
)

# Long-poll timeout (seconds) handed to getUpdates. aiogram retries the poll
# internally on transport errors; this is the server-side long-poll hold, not a
# client retry budget.
DEFAULT_TELEGRAM_POLL_TIMEOUT_SECONDS = 30

# A Telegram chat id is a signed integer string (a ``-100…`` supergroup id, or a
# smaller signed id for a private/basic-group chat). The allowlist tokens must
# match this shape; a token that does not is a config error, not a chat we
# silently never match.
_TELEGRAM_CHAT_ID_RE = re.compile(r"^-?\d+$")


def _parse_allowed_groups(raw: str | None) -> frozenset[str]:
    """Parse ``RTECH_TG_ALLOWED_GROUPS`` fail-closed on a set-but-invalid value.

    UNSET (``None``) or all-whitespace-only-blanks with the var never present is
    the documented no-allowlist case — but that case is signalled by the var being
    ABSENT. If the var is PRESENT (any non-None string, including ``""`` or
    ``",,"``) yet yields no valid chat-id token, that is a configured-but-
    meaningless guard: it must fail closed rather than collapse to an empty
    allowlist that silently accepts every group.

    Args:
        raw: The raw ``RTECH_TG_ALLOWED_GROUPS`` value, or ``None`` when unset.

    Returns:
        The frozenset of validated chat-id strings. Empty ONLY when the var is
        unset (``None``).

    Raises:
        ValueError: When the var is set but every token is blank or not a signed
            integer (naming the var and the required shape).
    """
    if raw is None:
        # Unset: documented no-allowlist (Discord parity; the pure gates decide).
        return frozenset()
    tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
    if not tokens:
        raise ValueError(
            "RTECH_TG_ALLOWED_GROUPS is set but contains no usable group id: a "
            "configured-but-empty allowlist would silently accept every group. "
            "Unset the var entirely for the no-allowlist default, or list at least "
            "one canonical chat id."
        )
    invalid = [tok for tok in tokens if _TELEGRAM_CHAT_ID_RE.match(tok) is None]
    if invalid:
        raise ValueError(
            "RTECH_TG_ALLOWED_GROUPS entries must be signed-integer chat ids "
            r"(matching '^-?\d+$', e.g. '-1001234567890'); rejected: "
            f"{invalid!r}"
        )
    return frozenset(tokens)


@dataclass(frozen=True)
class TelegramConfig:
    """Typed, immutable Telegram long-poll edge settings (inbound).

    Mirrors :class:`ChatdConfig`'s credential discipline: the bot token is NEVER
    a literal here. It arrives as a VALUE resolved elsewhere (deploy wiring /
    :func:`resolve_token` against a ``*_ref``) and is handed to the adapter via a
    provider callable — this config holds only the non-secret knobs, so nothing
    on this dataclass is ever unsafe to log.

    Attributes:
        allowed_groups: Optional supergroup allowlist of canonical ``-100…`` id
            STRINGS (the normalized chat id the adapter emits as ``channel_id``).
            Empty (default) = no group allowlist at the transport layer; the
            existing pure gates (:mod:`rtech_chatd.gates`) still decide every
            message. When non-empty, an inbound message whose canonical chat id
            is absent is dropped before normalization (a cheap edge-level scope
            cut that never reaches the gate pipeline).
        allowed_updates: The getUpdates ``allowed_updates`` subscription. MUST
            explicitly include ``message_reaction`` — the Bot API never
            auto-includes it, so omitting it makes the bot silently blind to
            reactions — and ``callback_query``, which is likewise
            never auto-included, so omitting it silently disables every
            interactive steer button. Defaults to
            :data:`DEFAULT_TELEGRAM_ALLOWED_UPDATES`.
        poll_timeout_s: Long-poll hold (seconds) handed to getUpdates. MUST be
            >= 1 — a 0 hold returns immediately and hot-loops the poll. The
            adapter passes ``request_timeout = poll_timeout_s + buffer`` so the
            server-side hold never exceeds the aiogram HTTP timeout.
    """

    allowed_groups: frozenset[str] = field(default_factory=frozenset)
    allowed_updates: tuple[str, ...] = DEFAULT_TELEGRAM_ALLOWED_UPDATES
    poll_timeout_s: int = DEFAULT_TELEGRAM_POLL_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        # >= 1, not >= 0: getUpdates with timeout=0 returns IMMEDIATELY (no
        # server-side long-poll hold), so a 0 hold turns the poll loop into a hot
        # spin hammering the API. A positive hold is required.
        if self.poll_timeout_s < 1:
            raise ValueError("poll_timeout_s must be >= 1 (timeout=0 hot-loops the poll)")
        # A subscription that omits message_reaction silently drops every reaction
        # (Bot API never auto-includes it). This edge always subscribes to it so
        # inherits a live reaction feed; guard against a caller
        # accidentally handing an allowed_updates tuple that would blind us.
        if "message_reaction" not in self.allowed_updates:
            raise ValueError(
                "allowed_updates must include 'message_reaction': the Telegram Bot "
                "API never auto-includes reaction updates in getUpdates, so omitting "
                "it makes the bot silently blind to every reaction"
            )
        if "message" not in self.allowed_updates:
            raise ValueError("allowed_updates must include 'message' (the inbound surface)")
        # A subscription that omits callback_query silently drops every inline-keyboard
        # tap (Bot API never auto-includes it, exactly like message_reaction). The
        # interactive steer layer rides those taps, so omitting it would
        # silently disable every steer button — guard against a caller handing an
        # allowed_updates tuple that would blind us.
        if "callback_query" not in self.allowed_updates:
            raise ValueError(
                "allowed_updates must include 'callback_query': the Telegram Bot API "
                "never auto-includes inline-keyboard tap updates in getUpdates, so "
                "omitting it silently disables every interactive steer button"
            )

    @classmethod
    def from_env(cls) -> TelegramConfig:
        """Build Telegram edge config from the environment (non-secret knobs only).

        The bot token is NOT read here — it is a credential resolved via the
        ``*_ref`` indirection in :class:`ChatdConfig` / :func:`resolve_token`, so
        this config never holds or logs a token. Group ids are opaque public ids
        (canonical ``-100…`` strings), not secrets.

        Vars:
            RTECH_TG_ALLOWED_GROUPS: comma-separated canonical ``-100…`` group
                ids the edge accepts. UNSET (default) = no transport allowlist
                (Discord parity — the pure gates still decide every message). SET
                but collapsing to nothing usable (all-blank, or a token that is
                not a signed integer) RAISES: a configured-but-meaningless guard
                must fail closed, never silently accept every group.
            RTECH_TG_POLL_TIMEOUT_S: getUpdates long-poll hold in seconds.
                Malformed (non-integer) or out-of-range (< 1) RAISES, naming the
                var — a bad poll hold silently defaulting would mask a real deploy
                mistake (0 hot-loops the poll). Any hold >= 1 is safe: the adapter
                derives its HTTP request_timeout as the hold + a buffer, so a long
                hold can never outlive the HTTP timeout.

        Raises:
            ValueError: On a malformed/out-of-range ``RTECH_TG_POLL_TIMEOUT_S`` or
                a set-but-invalid ``RTECH_TG_ALLOWED_GROUPS`` (both name the var).
        """
        raw_timeout = os.getenv("RTECH_TG_POLL_TIMEOUT_S", "").strip()
        if not raw_timeout:
            poll_timeout_s = DEFAULT_TELEGRAM_POLL_TIMEOUT_SECONDS
        else:
            try:
                poll_timeout_s = int(raw_timeout)
            except ValueError:
                raise ValueError(
                    "RTECH_TG_POLL_TIMEOUT_S must be a positive integer (getUpdates "
                    f"long-poll hold in seconds); got {raw_timeout!r}. A malformed value "
                    "is rejected rather than silently defaulting so a deploy mistake "
                    "fails loud."
                ) from None
            if poll_timeout_s < 1:
                raise ValueError(
                    "RTECH_TG_POLL_TIMEOUT_S must be >= 1 (getUpdates timeout=0 returns "
                    f"immediately and hot-loops the poll); got {poll_timeout_s}."
                )
        return cls(
            allowed_groups=_parse_allowed_groups(os.getenv("RTECH_TG_ALLOWED_GROUPS")),
            poll_timeout_s=poll_timeout_s,
        )


@dataclass(frozen=True)
class HomeChannel:
    """A per-agent home channel (Hermes semantics: an ID, never a ``*_ref``).

    The home channel is where cron job results and cross-platform messages get
    delivered. Only the bot token is an indirection ref; the home channel is a
    bare Discord channel id string, mirroring Hermes ``gateway/config.py``.

    Attributes:
        chat_id: Bare Discord channel id string (non-empty).
        name: Human-readable display name.
        thread_id: Optional thread id within the channel.
    """

    chat_id: str
    name: str = "Home"
    thread_id: str | None = None

    def __post_init__(self) -> None:
        if not self.chat_id:
            raise ValueError("home_channel chat_id must be non-empty")


# The advisory wording (Hermes intent). Its ONLY Discord-visible route is a
# genuine failed delivery attempt to the home channel — never inbound onboarding.
HOME_ADVISORY = (
    "📬 A home channel is where cron job results and cross-platform messages get "
    "delivered. None is set for this agent; set DISCORD_HOME_CHANNEL to enable it."
)


@dataclass(frozen=True)
class ChatdConfig:
    """Hold chatd edge runtime settings resolved via indirection only.

    ``token_ref`` is ALWAYS an ``env:VAR`` reference — never a literal token.
    The literal is resolved at runtime by :func:`resolve_token`, and only where
    the secret actually lives.

    Attributes:
        nats_url: NATS URL (must be a ``nats://`` URL).
        token_ref: Indirection ref naming WHERE the bot token lives.
        inbox_channel: Designated free-response inbox channel id, or None.
        origin_edge: Edge label stamped onto minted task frames.
        home_channel: Per-agent home channel (id-based), or None when unset.
        dash_base_url: Base URL of the Dash live-page. A presence
            notification composes its live-page link as
            ``f"{dash_base_url.rstrip('/')}/agents/<agent_id>"``. Empty (default)
            means render the notification WITHOUT a link rather than a broken URL —
            still notify, fail safe. This is a public base URL, NOT a secret: it is a
            plain value, not a ``*_ref`` (mirroring ``home_channel``'s bare id idiom;
            only the bot token uses indirection).
    """

    nats_url: str = DEFAULT_NATS_URL
    token_ref: str = "env:RTECH_CHATD_BOT_TOKEN"
    inbox_channel: str | None = None
    origin_edge: str = DEFAULT_ORIGIN_EDGE
    home_channel: HomeChannel | None = None
    dash_base_url: str = ""

    def __post_init__(self) -> None:
        if not self.nats_url.startswith("nats://"):
            raise ValueError(f"nats_url must be a nats:// URL, got {self.nats_url!r}")
        validate_token_ref(self.token_ref)
        if not self.origin_edge:
            raise ValueError("origin_edge must be non-empty")

    @classmethod
    def from_env(cls) -> ChatdConfig:
        """Build config from environment variables (indirection refs only)."""

        home_id = os.getenv("DISCORD_HOME_CHANNEL")
        home_channel = (
            HomeChannel(
                chat_id=home_id,
                name=os.getenv("DISCORD_HOME_CHANNEL_NAME", "Home"),
                thread_id=os.getenv("DISCORD_HOME_CHANNEL_THREAD_ID") or None,
            )
            if home_id
            else None
        )
        return cls(
            nats_url=os.getenv("RTECH_CHATD_NATS_URL", DEFAULT_NATS_URL),
            token_ref=os.getenv("RTECH_CHATD_TOKEN_REF", "env:RTECH_CHATD_BOT_TOKEN"),
            inbox_channel=os.getenv("RTECH_CHATD_INBOX_CHANNEL") or None,
            home_channel=home_channel,
            dash_base_url=(os.getenv("RTECH_CHATD_DASH_BASE_URL", "") or "").strip(),
        )

    def home_send_target(self) -> SendTarget | None:
        """Map the home channel to a :class:`SendTarget`, or None when unset."""

        if self.home_channel is None:
            return None
        return SendTarget(
            channel_id=self.home_channel.chat_id,
            thread_id=self.home_channel.thread_id,
        )


# Per-identity bot token refs (multi-identity sender). Comma list of
# ``agent=token_ref`` pairs; ``=`` separates because refs contain ``:``.
ENV_TG_IDENTITY_MAP = "RTECH_TG_IDENTITY_MAP"


def parse_identity_map(raw: str | None) -> dict[str, str]:
    """Parse ``RTECH_TG_IDENTITY_MAP`` into ``agent_id -> token_ref`` (refs only).

    Shape: ``agentd-nagatha=env:TG_TOKEN_NAGATHA,agentd-bilby=env:TG_TOKEN_BILBY``.
    Every agent id must follow the deployed ``agentd-<name>`` convention (the
    SAME fail-loud rule as the debate roster — a bare name would silently never
    match a TurnFrame's ``agent_id`` and every send would fail as unknown), and
    every value must be a ``*_ref`` indirection (:func:`validate_token_ref`
    rejects a literal token WITHOUT echoing it).

    UNSET (``None``) is the no-identity-pool default (empty map). SET but
    yielding no usable pair fails closed, mirroring ``RTECH_TG_ALLOWED_GROUPS``:
    a configured-but-meaningless identity map must never silently collapse.

    Raises:
        ValueError: Set-but-empty, a malformed pair, a non-convention agent id,
            a duplicate agent id, or a non-indirection token ref.
    """
    if raw is None:
        return {}
    tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
    if not tokens:
        raise ValueError(
            f"{ENV_TG_IDENTITY_MAP} is set but contains no usable pair; unset it "
            "entirely for the no-identity-pool default, or list at least one "
            "'agentd-<name>=<token_ref>' pair"
        )
    identities: dict[str, str] = {}
    for position, token in enumerate(tokens, start=1):
        agent, sep, ref = token.partition("=")
        agent = agent.strip()
        ref = ref.strip()
        # NEVER echo either side of a malformed/invalid entry: a literal token
        # accidentally pasted on EITHER side of the '=' (or with no '=' at
        # all) must not land in the raised error or in journald. Entries are
        # identified by position only.
        if not sep or not agent or not ref:
            raise ValueError(
                f"{ENV_TG_IDENTITY_MAP} entry #{position} is malformed: each entry "
                "must be 'agentd-<name>=<token_ref>' (values withheld from this error)"
            )
        if _AGENTD_CONVENTION.match(agent) is None:
            raise ValueError(
                f"{ENV_TG_IDENTITY_MAP} entry #{position}: the agent id does not follow "
                r"the required agentd-<name> convention ('^agentd-[a-z0-9][a-z0-9-]*$'); "
                "a bare name never matches a TurnFrame agent_id, so every send under "
                "it would fail as an unknown identity (value withheld from this error)"
            )
        # The agent id matched the convention regex (lowercase agentd-<name>),
        # so it cannot be a credential — safe to echo in the duplicate error.
        if agent in identities:
            raise ValueError(f"{ENV_TG_IDENTITY_MAP} lists agent id {agent!r} twice")
        # Rejects a literal token without echoing it (the ref value is withheld
        # from the raised error and from logs).
        validate_token_ref(ref)
        identities[agent] = ref
    return identities


# The env suffix must be a valid environment-variable name. Anything else is
# treated as a possible literal credential and rejected without echoing it.
_ENVIRONMENT_VARIABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_token_ref(token_ref: str) -> None:
    """Fail closed on a non-indirection ``token_ref`` WITHOUT echoing the value.

    Only an ``env:VAR`` reference with a valid environment-variable name is
    accepted. The raw value never appears in the raised error or in any log: if
    an operator mistakenly configures a literal token where a ref belongs, the
    fail-closed path must not leak that credential to journald.

    Raises:
        ValueError: If the value is empty or not a supported ref shape.
    """
    scheme, sep, name = token_ref.partition(":") if token_ref else ("", "", "")
    if (
        not sep
        or scheme.strip().lower() != "env"
        or _ENVIRONMENT_VARIABLE_NAME.fullmatch(name.strip()) is None
    ):
        raise ValueError(
            "token_ref must be an indirection ref of shape 'env:VALID_ENV_VAR'; "
            "the configured value is withheld from this "
            "error and from logs"
        )


def resolve_token(token_ref: str) -> str | None:
    """Resolve a ``*_ref`` to the token value, or None when unprovisioned.

    Only indirection is supported: the ref names WHERE the secret lives, never
    the secret itself. Returns None (not an error) when the source is empty so
    callers can SKIP cleanly. NEVER accepts or returns a literal token path.
    """
    if not token_ref:
        return None
    scheme, _, name = token_ref.partition(":")
    scheme = scheme.strip().lower()
    if scheme == "env":
        value = os.environ.get(name.strip(), "")
        return value or None
    # NEVER echo the scheme (or any part of the value): with no ':' present the
    # "scheme" is the whole configured string, which could be a literal token.
    logger.warning("config.resolve_token | unsupported token-ref scheme; value withheld")
    return None


__all__ = [
    "DEFAULT_NATS_URL",
    "DEFAULT_ORIGIN_EDGE",
    "DEFAULT_TELEGRAM_ALLOWED_UPDATES",
    "DEFAULT_TELEGRAM_POLL_TIMEOUT_SECONDS",
    "ENV_TG_IDENTITY_MAP",
    "HOME_ADVISORY",
    "WILDCARD",
    "AllowBots",
    "ChatdConfig",
    "GateConfig",
    "HomeChannel",
    "IntakeConfig",
    "IntakeMode",
    "TelegramConfig",
    "ThreadConfig",
    "parse_identity_map",
    "resolve_token",
    "validate_token_ref",
]
