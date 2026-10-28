"""Wire the ChannelAdapter to the typed bus as a dumb Discord edge.

:class:`DiscordEdge` translates at the boundary only: it mints a ``TaskFrame``
from each gated inbound message, publishes it on :data:`subjects.TASK_INGRESS`,
subscribes that task's turn/status subjects, and renders replies back into the
ORIGINATING chat surface. It carries NO control signal — no stance, no done, no
control inference happens here. Done is orchestrator-derived from the frame graph.

Outbound render always runs the whole rendered turn text through the egress guard
before send; there is no delta path, so streaming can never bypass the guard.

Per-task reply subscriptions (turn/status/result) are installed as ONE atomic
set before the ingress publish and retired through one handle, via the typed-bus
:func:`subscribe_all` primitive: a partial-subscribe failure rolls everything
back so no subscription is ever live-but-untracked. Retirement is honest -- a
subscription set whose unsubscribe fails is never silently discarded: it is held
in ``_leaked_subscriptions`` and retried at :meth:`DiscordEdge.close`. Terminal
render is
serialized per task by a per-task lock so the turn lane and the result lane can
never both pass the rendered-key check, double-render the terminal answer, or
race the done marker against an in-flight turn.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol, runtime_checkable

from loguru import logger
from rtech_frames.codec import decode_frame, encode_frame
from rtech_frames.frames import (
    SCOPE_PARTICIPANT_COUNT,
    StatusFrame,
    TurnFrame,
    validate_participants,
)
from rtech_frames.subs import FrameSubscribeError, SubscriptionSet, subscribe_all

from . import subjects
from .bridge import (
    OutboundFilter,
    intake_reply_target,
    status_frame_to_send,
    turn_frame_to_send,
    turn_frame_to_tldr,
)
from .bridge import inbound_to_task_frame as _inbound_to_task_frame
from .bus import FrameBus, Subscription
from .channel import ChannelAdapter, ChunkProgress, InboundMessage, SendTarget, chunk_progress_from
from .config import HOME_ADVISORY
from .outbound_guard import sanitize_outbound_text
from .presence_notify import PresenceNotifyConsumer

DEFAULT_ORIGIN_EDGE = "discord"

# Origin-edge labels that map to a surface the presence-notify renderer knows.
# ``telegram`` gets markdown->safe-HTML at the sender; anything else (Discord and
# any future markdown-native surface) gets the bare-URL Discord body. Keyed off
# the edge's own ``origin_edge`` so the ONE shared bridge renders per surface
# without the connector carrying presence logic.
_TELEGRAM_ORIGIN_EDGE = "telegram"


@runtime_checkable
class TurnInteraction(Protocol):
    """A narrow, surface-owned hook for attaching interactive markup to a turn.

    The generic edge stays a dumb boundary: it carries no surface-specific
    interaction knowledge (Telegram inline keyboards, callback capabilities).
    When a debate TLDR message lands, the edge offers the SURFACE this one typed
    opportunity — the landed anchor message id plus the typed identities the
    interaction must bind to (the task, the answering agent, and the ORIGINAL
    requester actor) — and the surface decides whether and how to attach an
    interaction. A single-identity surface (Discord v1) injects nothing and the
    edge is byte-identical; only the Telegram service injects a concrete
    implementation that mints actor-bound callback capabilities (after the id is
    known) and attaches a display-only keyboard on its polled default bot.

    The hook is DISPLAY plumbing only. It never carries a control signal: a tap
    becomes a typed :class:`~rtech_frames.frames.SteerFrame` on the bus through
    the surface's own callback sink, never through this edge. A hook failure is
    contained by the edge caller — an interaction that fails to attach must never
    crash or block the render of the answer it decorates.
    """

    async def offer_debate_tldr(
        self,
        *,
        target: SendTarget,
        anchor_message_id: str,
        task_id: str,
        agent_id: str,
        asker: str | None,
    ) -> None:
        """Attach an interaction to a just-landed debate TLDR, if the surface can.

        ``target`` carries the chat the TLDR landed in; ``anchor_message_id`` is
        that TLDR message's id (known only AFTER dispatch, which is why the mint
        happens here, not at render construction). ``asker`` is the ORIGINAL
        requester actor the interaction must be bound to (``None`` when the origin
        carried no asker — the surface then declines rather than mint an
        unbindable capability).
        """
        ...


# Bound on the external-turn dedup seen-set (reply render). Each entry
# keys an already-processed external turn to the SET of render pieces that landed
# for it, so redelivery resumes only the MISSING piece (R4/R5 partial
# failure): a debate turn that posted its TLDR but whose in-thread full send
# failed retries only the full on redelivery, never re-posting the TLDR.
#
# KEY SHAPE is (task_id, agent_id, digest) for BOTH render modes. ``digest`` is a
# content-safe SHA-256 over the FULL immutable typed TurnFrame identity
# (:func:`_turn_identity_digest`), so the key is the turn's stable typed identity
# — not just its author:
#
# * A BYTE-IDENTICAL redelivery of the SAME terminal turn hashes identically, so
#   it shares the one progress/chunk-cursor record: an exact duplicate collapses
#   to one render, and a partial (chunk/piece) failure resumes only the missing
#   work on exact replay.
# * A CHANGED terminal turn for the same (task, agent) — a distinct later
#   substantive answer, not a redelivery — hashes differently, so it renders as a
#   DISTINCT turn instead of being silently suppressed by an author-only key.
#
# ``task_id`` stays key[0] so eviction cleanup keys off it unchanged; ``agent_id``
# keeps distinct authors distinct even in the (astronomically unlikely) event of
# a cross-author hash tie. Only the digest — never the private ``visible_text`` —
# is retained in state/logs. The cap is the memory backstop for a redelivery
# storm, evicting oldest-first (an already-posted turn re-arriving after eviction
# re-posts — bounded-best-effort under a flood, exact within).
DEFAULT_EXTERNAL_SEEN_MAX = 4096


def _turn_identity_digest(frame: TurnFrame) -> str:
    """Return a content-safe digest of a TurnFrame's full typed identity.

    Hashes the canonical wire encoding (:func:`encode_frame` is deterministic —
    ``sort_keys``, fixed separators) of the FULL immutable frame, so two turns
    hash the same IFF every typed field (visible_text, stance, refs, origin, …)
    matches. This is the stable typed identity the external-render dedup keys on:
    a byte-identical redelivery collapses (shares progress), while any changed
    terminal frame becomes a distinct key and renders as its own turn. Only the
    hex digest is ever stored/logged — never the private frame content.
    """

    return hashlib.sha256(encode_frame(frame)).hexdigest()


# Render-progress piece markers stored in the seen-set value (R4/R5).
# A non-debate turn is one atomic ``full`` send; a debate turn is a ``tldr`` to
# the parent channel plus a ``full`` in-thread, each marked independently so a
# partial failure retries only the missing piece.
_PIECE_TLDR = "tldr"
_PIECE_FULL = "full"


@dataclass
class _ExternalRenderProgress:
    """Completed-piece marks plus bounded chunk checkpoints for one turn key."""

    pieces: set[str] = field(default_factory=set)
    chunk_cursors: dict[str, ChunkProgress] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Return whether this key has neither landed nor checkpointed work."""
        return not self.pieces and not self.chunk_cursors


# Per-task distinct-agent TLDR cap (R hardening / D8). A TurnFrame
# carries no participant roster, so the edge renders a TLDR for ANY novel
# agent_id on a debate origin. This bounds how many DISTINCT agents may post a
# TLDR per debate task: trio (3) + one closing author = 4. A novel agent past the
# cap is dropped with a warning (never a silent wall). Not participant-list
# plumbing — that is future work; this is the minimal fan-in backstop.
DEBATE_DISTINCT_AGENT_TLDR_CAP = 4

IdFactory = Callable[[], str]

# Rendered ONCE per task when the orchestrator publishes the derived-done result
# (the result lane republishes the terminal turn; its text already rendered on
# the turn lane, so the edge renders only this typed completion marker).
DONE_MARKER = "✅ done"

# Explicit closed mapping of participant count -> scope literal. Built from the
# shared SCOPE_PARTICIPANT_COUNT source but typed as the exact Literal so the
# scope stays a ``Literal["single", "duo", "trio"]`` end-to-end (mypy-clean, no
# str widening). An assertion pins the two sources in sync at import.
_SCOPE_BY_COUNT: dict[int, Literal["single", "duo", "trio"]] = {
    SCOPE_PARTICIPANT_COUNT["single"]: "single",
    SCOPE_PARTICIPANT_COUNT["duo"]: "duo",
    SCOPE_PARTICIPANT_COUNT["trio"]: "trio",
}
assert {count: scope for scope, count in SCOPE_PARTICIPANT_COUNT.items()} == _SCOPE_BY_COUNT


def _new_task_id() -> str:
    """Return a fresh opaque task id (uuid4 hex)."""

    return uuid.uuid4().hex


# Discord thread names cap at 100 chars; keep a conservative budget with room for
# the anchor prefix so a derived name never rejects the create.
_DEBATE_THREAD_NAME_MAX = 90


def _debate_thread_name(agent_id: str) -> str:
    """Derive a bounded thread name from the posting agent's id (UX v2).

    The thread hangs off that agent's own TLDR message, so it is named for the
    agent (the ``agentd-`` convention prefix stripped for display). The name is
    capped under Discord's limit so the create never fails on an over-long name.
    """

    display = agent_id.removeprefix("agentd-") or agent_id
    name = f"{display} — full thought"
    if len(name) > _DEBATE_THREAD_NAME_MAX:
        name = name[:_DEBATE_THREAD_NAME_MAX]
    return name


def _lane_key(agent_id: str) -> str:
    """Normalize an agent id to its persistent-lane registry key.

    Strips the ``agentd-`` convention prefix and recognized deployment suffixes
    (``-dev`` and ``-prod``) so ``agentd-nimbus`` and ``agentd-nimbus-dev`` both
    key the same lane. Internal hyphens remain identity-bearing: ``agentd-dash-alpha``
    and ``agentd-dash-beta`` must never collide.
    """

    display = agent_id.removeprefix("agentd-") or agent_id
    for suffix in ("-dev", "-prod"):
        display = display.removesuffix(suffix)
    return display.strip().lower()


def debate_lanes_from_pairs(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Normalize configured ``(name, topic)`` lane pairs into an agent->lane map.

    Keys are normalized via :func:`_lane_key`; blank/malformed entries are
    skipped (a lane misconfig must not wall the debate — the agent simply falls
    back to a lazily-created per-turn thread). Values are the surface-native
    thread ids (strings), passed to the adapter verbatim. A later pair for the
    same normalized name wins (last-write), so an override can follow a default.
    """

    lanes: dict[str, str] = {}
    sources: dict[str, str] = {}
    for name, topic in pairs:
        key = _lane_key(name.strip())
        topic = topic.strip()
        if not key or not topic:
            logger.warning(
                "chatd.edge | skipping malformed debate lane entry name={!r} topic={!r}",
                name,
                topic,
            )
            continue
        prior_source = sources.get(key)
        if prior_source is not None and prior_source != name:
            raise ValueError(
                "debate lane configuration normalizes multiple source keys to "
                f"{key!r}: {prior_source!r} and {name!r}"
            )
        lanes[key] = topic
        sources[key] = name
    return lanes


def debate_chat_lanes_from_triples(
    triples: Iterable[tuple[str, str, str]],
) -> dict[tuple[str, str], str]:
    """Normalize ``(chat_id, name, topic)`` triples into a (chat, agent)->lane map.

    The chat-scoped twin of :func:`debate_lanes_from_pairs`: the key is the
    COMPOSITE ``(chat_id, lane_key(name))`` so the SAME agent name maps to a
    DIFFERENT topic id per chat. The chat id is used verbatim (it is the canonical
    surface chat id the edge sees as ``target.channel_id`` — normalized once at the
    Telegram adapter). Agent names are normalized via :func:`_lane_key` exactly as
    the legacy map, so a chat-scoped lane and its legacy fallback key on the same
    agent identity. Blank/malformed entries are skipped with a warning (degrade to
    the legacy lane or a per-turn thread, never wall the debate); last-write-wins
    for a repeated ``(chat, agent)`` so an override can follow a default.
    """

    lanes: dict[tuple[str, str], str] = {}
    sources: dict[tuple[str, str], str] = {}
    for chat_id, name, topic in triples:
        chat = chat_id.strip()
        key = _lane_key(name.strip())
        topic = topic.strip()
        if not chat or not key or not topic:
            logger.warning(
                "chatd.edge | skipping malformed chat-scoped debate lane entry "
                "chat={!r} name={!r} topic={!r}",
                chat_id,
                name,
                topic,
            )
            continue
        lane_key = (chat, key)
        prior_source = sources.get(lane_key)
        if prior_source is not None and prior_source != name:
            raise ValueError(
                "chat-scoped debate lane configuration normalizes multiple source "
                f"keys to {lane_key!r}: {prior_source!r} and {name!r}"
            )
        lanes[lane_key] = topic
        sources[lane_key] = name
    return lanes


def _scope_for_participants(participants: list[str]) -> Literal["single", "duo", "trio"]:
    """Derive the scope literal from the configured participant count.

    An empty participant set is a valid ``single`` task (the orchestrator routes
    to its configured default target). Any other count must map exactly onto a
    known scope; the edge fails closed instead of minting an unroutable frame.

    Raises:
        ValueError: If the count maps to no scope, or an id is blank/duplicate/
            subject-unsafe (delegated to :func:`validate_participants`).
    """

    scope: Literal["single", "duo", "trio"] | None = _SCOPE_BY_COUNT.get(
        len(participants), "single" if not participants else None
    )
    if scope is None:
        raise ValueError(
            f"participant count {len(participants)} maps to no scope "
            f"(known: {sorted(_SCOPE_BY_COUNT)})"
        )
    validate_participants(participants, scope)
    return scope


def _guard_filter(text: str) -> str:
    """Adapt the ported egress guard to the ``OutboundFilter`` shape (str->str)."""

    result = sanitize_outbound_text(text, platform=DEFAULT_ORIGIN_EDGE)
    if isinstance(result, str):
        return result
    return result.text


class DiscordEdge:
    """Tie a :class:`ChannelAdapter` to a :class:`FrameBus` as a dumb edge.

    Dependencies are injected for testability: any adapter satisfying the
    :class:`ChannelAdapter` protocol, any :class:`FrameBus`, an ``id_factory``
    for minting task ids, and an ``outbound_filter`` (the egress guard). The edge
    remembers the originating :class:`SendTarget` per task id so an outbound turn
    renders back into the SAME thread the human message came from.
    """

    def __init__(
        self,
        adapter: ChannelAdapter,
        bus: FrameBus,
        *,
        stack: str,
        id_factory: IdFactory | None = None,
        outbound_filter: OutboundFilter | None = None,
        origin_edge: str = DEFAULT_ORIGIN_EDGE,
        home_target: SendTarget | None = None,
        agent_label: str = "chatd",
        participants: list[str] | None = None,
        attributed: bool = False,
        intake_enabled: bool = False,
        identity_send: bool = False,
        debate_lanes: dict[str, str] | None = None,
        debate_chat_lanes: dict[tuple[str, str], str] | None = None,
        turn_interaction: TurnInteraction | None = None,
        external_seen_max: int = DEFAULT_EXTERNAL_SEEN_MAX,
        presence_notify_target: SendTarget | None = None,
        dash_base_url: str = "",
    ) -> None:
        """Store injected dependencies without opening any connection.

        ``stack`` is REQUIRED addressing config: every minted
        TaskFrame is stamped with it, and a blank id fails closed at
        construction — an independently launched edge never defaults to a
        shared literal.

        ``home_target`` is resolved eagerly at construction (startup). When it is
        missing/invalid we log LOUDLY (logs only) — the 📬 advisory is NEVER
        posted to Discord as onboarding; it only surfaces on a genuine failed
        :meth:`deliver_to_home` call.

        ``participants`` is the configured typed participant set stamped onto
        every minted TaskFrame (identities, never scraped from text); the scope
        literal is derived from its count and both fail closed at construction
        on an invalid set. ``attributed`` turns on per-agent render-back: each
        participant's TurnFrame renders as its OWN reply prefixed with the typed
        ``agent_id``.

        ``identity_send`` opts the DEBATE render into per-agent identity
        routing: each debate piece's :class:`SendTarget` is
        stamped ``as_agent=frame.agent_id`` so a multi-identity surface
        (Telegram) posts it under that agent's OWN bot, the TLDR drops its
        now-redundant name prefix, and the agent's typing indicator is stopped
        when its terminal turn renders. False (default) keeps every existing
        deploy byte-identical (single-bot voice, named TLDRs).

        ``intake_enabled`` is the SAME opt-in that makes this deploy a triage
        edge. Only when it is true does the edge subscribe the shared
        turn wildcard and render EXTERNAL task turns (answers for tasks it did not
        mint — an intake/wake answer) back into the asking channel. When false the
        edge behaves EXACTLY as before: no wildcard subscription, byte-identical.

        ``debate_lanes`` is the LEGACY single-group agent->stable-lane map: when
        an agent has a lane its debate full text routes there EVERY turn instead
        of minting a per-turn thread. ``debate_chat_lanes`` is the ADDITIVE
        chat-scoped map keyed ``(chat_id, agent) -> lane`` so the SAME agent
        routes to a DIFFERENT topic id per chat. Selection order in
        :meth:`_debate_thread_for` is exact-chat+agent FIRST, then the legacy
        agent-only lane, then per-turn thread creation — so an unlisted
        chat/agent never loses full text, and a deploy with neither map is
        byte-identical to the pre-lane per-turn behavior.

        ``presence_notify_target`` opts this edge into rendering operator
        presence notifications: when set, the edge subscribes the
        shared :data:`subjects.PRESENCE_TRANSITION` subject and renders the typed
        ``PresenceTransitionFrame`` the web-chat deriver publishes into THIS
        surface's send path (markdown body, so the surface applies its own safe
        HTML/formatting), addressed at the given target. ``None`` (default) keeps
        every existing deploy byte-identical — no presence subscription, no send.
        ``dash_base_url`` is the public Dash live-page base for the rendered link;
        empty renders the notification without a link (fail safe, still notify).
        The renderer never emits an internal agent id, subject, or payload as
        visible text — presence stays a typed frame, the edge stays a dumb edge.

        Raises:
            ValueError: If ``stack`` is blank, or ``participants`` is invalid
                (blank/duplicate/subject-unsafe id, or unmappable count).
        """

        if not stack.strip():
            logger.error("chatd.edge | blank stack id rejected at construction")
            raise ValueError("stack must be non-empty; there is no default stack id")
        self._adapter = adapter
        self._bus = bus
        self._id_factory = id_factory or _new_task_id
        self._outbound_filter = outbound_filter or _guard_filter
        self._origin_edge = origin_edge
        self._stack = stack
        self._home_target = home_target
        self._agent_label = agent_label
        self._participants = list(participants or [])
        self._scope: Literal["single", "duo", "trio"] = _scope_for_participants(self._participants)
        self._attributed = attributed
        self._targets: dict[str, SendTarget] = {}
        # ONE handle per task covering its whole reply subscription set (turn +
        # status + result), installed atomically via ``subscribe_all`` and
        # retired through the single ``SubscriptionSet.retire()`` handle.
        self._task_subscriptions: dict[str, SubscriptionSet] = {}
        # Honest-retirement escalation store: subscription sets whose unsubscribe
        # failed. These are NEVER silently discarded -- they are held here and
        # retried at close, so a live subscription can never escape untracked.
        self._leaked_subscriptions: dict[str, SubscriptionSet] = {}
        # Per-task render serialization: the turn lane and the result lane are
        # separate NATS callbacks with no ordering guarantee. This lock makes the
        # terminal render/done/drop path mutually exclusive so neither lane can
        # double-render the terminal answer or race the done marker.
        self._task_locks: dict[str, asyncio.Lock] = {}
        self._done_tasks: set[str] = set()
        # Turn identities already rendered on the turn lane, per task. The
        # result lane republishes the terminal TurnFrame; turn and result are
        # separate subjects with NO cross-subject ordering guarantee, so the
        # edge must know whether the terminal text still needs rendering.
        self._rendered_turns: dict[str, set[tuple[str, str]]] = {}
        # External-turn reply render: only a triage edge (intake
        # enabled) subscribes the shared turn wildcard to answer tasks it did not
        # mint. The subscription handle is retired at close; the seen-set dedups
        # a redelivered external turn so a single answer never double-posts.
        self._intake_enabled = intake_enabled
        self._identity_send = identity_send
        # Persistent per-agent verbose lanes (UX v3): a normalized ``agent -> stable
        # topic id`` registry. When an agent has a lane, its debate full text routes
        # there EVERY turn instead of minting a fresh per-turn thread — the noisy
        # topic-per-message behavior. Empty (default) keeps the lazy per-turn thread
        # path byte-identical, so a surface with no lanes configured is unchanged.
        # This is the LEGACY single-group map, consulted as the fallback under the
        # chat-scoped map below.
        self._debate_lanes = dict(debate_lanes or {})
        # Chat-scoped persistent lanes: ``(chat_id, agent) -> stable topic id`` so
        # the SAME agent routes to a DIFFERENT topic id per chat. Consulted FIRST
        # in _debate_thread_for (exact-chat+agent), before the legacy agent-only
        # lane and before per-turn thread creation. Empty (default) leaves the
        # legacy path unchanged; purely additive, never drops an unlisted
        # chat/agent's full text.
        self._debate_chat_lanes = dict(debate_chat_lanes or {})
        # Narrow surface-owned interaction hook. When set, the edge
        # offers each landed debate TLDR to the surface so it may attach a
        # display-only, actor-bound interaction (a Telegram inline keyboard whose
        # taps become typed steers on the bus). None (default) keeps every deploy
        # byte-identical — the edge attaches no markup and owns no surface state.
        self._turn_interaction = turn_interaction
        self._external_seen_max = external_seen_max
        self._external_subscription: Subscription | None = None
        # Presence-transition notifications. Only wired when a notify
        # target is configured, so an edge without one is byte-identical. The
        # consumer decodes the typed frame and renders THIS surface's markdown body
        # (Telegram vs the bare-URL Discord body, keyed off ``origin_edge``) through
        # the SAME guard+adapter dispatch every other outbound uses, so the surface
        # applies its own safe rendering and the connector carries no presence logic.
        # A None target leaves the consumer and subscription absent entirely.
        self._presence_notify_target = presence_notify_target
        self._dash_base_url = dash_base_url
        self._presence_subscription: Subscription | None = None
        self._presence_consumer: PresenceNotifyConsumer | None = None
        if presence_notify_target is not None:
            is_telegram = origin_edge == _TELEGRAM_ORIGIN_EDGE
            self._presence_consumer = PresenceNotifyConsumer(
                dash_base_url=dash_base_url,
                send_telegram=self._send_presence if is_telegram else None,
                send_discord=None if is_telegram else self._send_presence,
            )
        # External-turn render-progress marks. Key shape is
        # (task_id, agent_id, digest) for BOTH render modes (see
        # DEFAULT_EXTERNAL_SEEN_MAX): the digest is a content-safe hash of the FULL
        # typed TurnFrame identity, so a byte-identical redelivery shares this
        # record (exact-duplicate collapse + partial-retry on replay) while a
        # CHANGED terminal turn keys distinctly and renders as its own turn. The
        # VALUE is the set of render pieces that already landed
        # ({_PIECE_TLDR, _PIECE_FULL}), so a partial failure retries only the
        # missing piece on redelivery (R4/R5). Bounded oldest-first by
        # ``_external_seen_max``. Each mark also retains at most one
        # content-free chunk checkpoint per unfinished piece, so a Telegram
        # FloodWait resumes at its first unsent chunk on redelivery.
        self._external_seen: OrderedDict[tuple[str, ...], _ExternalRenderProgress] = OrderedDict()
        # Per debate (TASK, AGENT): the lazily-created thread id that agent's full
        # text lands in (UX v2 — one thread per agent turn, anchored to the
        # agent's own TLDR message, so each channel stance opens into its own
        # reasoning). Cached here and retired with the seen-set. ``None`` marks a
        # key whose thread creation failed so the full text degrades to the
        # parent channel without retrying the create per redelivery.
        self._debate_threads: OrderedDict[tuple[str, str], str | None] = OrderedDict()
        # Per debate (TASK, AGENT): the channel message id of the agent's landed
        # TLDR — the anchor its thread is created on. Recorded at TLDR dispatch,
        # bounded/retired with the seen-set.
        self._debate_tldr_msgs: OrderedDict[tuple[str, str], str] = OrderedDict()
        # Keys whose TLDR dispatch is IN FLIGHT right now. A concurrent duplicate
        # delivery that reaches the FULL piece while its agent's TLDR is still
        # mid-send must NOT proceed: the anchor is not recorded yet, so it would
        # degrade the full text to the channel and (worse) poison the thread
        # cache. It defers instead — the in-flight delivery renders FULL itself
        # after its TLDR lands.
        self._debate_tldr_inflight: set[tuple[str, str]] = set()
        # Per debate TASK: the set of DISTINCT agent_ids that have already rendered
        # a TLDR, for the distinct-agent TLDR cap (D8). Bounded by the cap
        # itself (at most DEBATE_DISTINCT_AGENT_TLDR_CAP ids per task) and retired
        # with the seen-set.
        self._debate_tldr_agents: OrderedDict[str, set[str]] = OrderedDict()
        # Serializes lazy debate-thread creation: without it two concurrent
        # first-renders for one (task, agent) both miss the cache and create TWO
        # Discord threads, splitting that agent's full text (check-then-await
        # race).
        self._debate_thread_lock = asyncio.Lock()
        if home_target is None:
            logger.warning(
                "chatd.home | no home channel configured for agent={} "
                "(set DISCORD_HOME_CHANNEL); cron/cross-platform delivery will fail "
                "until set",
                agent_label,
            )
        else:
            logger.info(
                "chatd.home | home channel resolved for agent={} target={}",
                agent_label,
                home_target.resolve(),
            )

    async def run_inbound(self) -> None:
        """Drain the adapter's gated events onto the bus until it closes."""

        logger.info("edge.run_inbound | entry origin_edge={}", self._origin_edge)
        await self._subscribe_external_turns()
        await self._subscribe_presence_transitions()
        try:
            async for msg in self._adapter.events():
                await self._ingest(msg)
        except Exception as exc:
            logger.error("edge.run_inbound | failed error_type={}", type(exc).__name__)
            raise
        finally:
            logger.info("edge.run_inbound | exit tasks={}", len(self._targets))

    async def _ingest(self, msg: InboundMessage) -> None:
        """Mint + publish a task frame and subscribe its reply subjects."""

        task_id = self._id_factory()
        target = SendTarget(channel_id=msg.channel_id, thread_id=msg.thread_id)
        self._targets[task_id] = target
        self._task_locks[task_id] = asyncio.Lock()
        frame = _inbound_to_task_frame(
            msg,
            task_id=task_id,
            origin_edge=self._origin_edge,
            scope=self._scope,
            participants=list(self._participants),
            stack=self._stack,
        )
        # Member IDs never reach logs (security floor: treat Discord as leaked);
        # the author travels only inside the typed frame.
        logger.debug(
            "edge.ingest | task_id={} channel={} thread={}",
            task_id,
            msg.channel_id,
            msg.thread_id,
        )
        # Reply subscriptions are installed BEFORE the publish (same
        # pattern, mirroring rtech_orchestrator): a fast orchestrator/agent can
        # emit the first turn/status/result the moment the task lands, and a
        # subscription installed after the publish misses it forever. The whole
        # set is installed atomically (all-or-nothing via the typed-bus
        # ``subscribe_all`` primitive), so a partial-subscribe failure never
        # leaves a live-but-untracked subscription. Both a subscription-set
        # failure and a publish failure roll the task fully back (drop the
        # target, lock and set) before re-raising; nothing survives before a
        # confirmed publish.
        try:
            await self._subscribe_replies(task_id)
            await self._bus.publish(subjects.TASK_INGRESS, encode_frame(frame))
        except FrameSubscribeError as exc:
            # Rollback may have left live subscriptions on exc.leaked. Never
            # discard them: stash for retry-at-close and fail the task closed.
            if len(exc.leaked) > 0:
                self._leaked_subscriptions[task_id] = exc.leaked
                logger.critical(
                    "edge.ingest | task_id={} refused; {} live subscription(s) held "
                    "for retry subjects={}",
                    task_id,
                    len(exc.leaked),
                    exc.leaked.subjects,
                )
            logger.error(
                "edge.ingest | task_id={} setup failed error_type={}",
                task_id,
                type(exc).__name__,
            )
            await self._drop_task(task_id)
            raise
        except Exception as exc:
            logger.error(
                "edge.ingest | task_id={} setup failed error_type={}",
                task_id,
                type(exc).__name__,
            )
            await self._drop_task(task_id)
            raise

    async def _subscribe_replies(self, task_id: str) -> None:
        """Install the per-task turn + status + result subjects as ONE atomic set.

        The three ``(subject, handler)`` entries are handed to the typed-bus
        :func:`subscribe_all` primitive, which subscribes them in order and rolls
        the whole set back on ANY failure (raising ``FrameSubscribeError``). The
        returned :class:`SubscriptionSet` is the single retire handle stored for
        this task — there is no manual per-subscription bookkeeping to leak.
        """

        async def _on_result(data: bytes) -> None:
            await self._render_result(task_id, data)

        entries = [
            (subjects.task_turn(task_id), self._render_frame),
            (subjects.task_status(task_id), self._render_frame),
            (subjects.task_result(task_id), _on_result),
        ]
        self._task_subscriptions[task_id] = await subscribe_all(self._bus.subscribe, entries)
        logger.debug("edge.subscribe_replies | task_id={} turn+status+result subscribed", task_id)

    async def _render_frame(self, data: bytes) -> None:
        """Decode a reply frame and render it to the originating chat surface."""

        try:
            frame = decode_frame(data)
        except Exception as exc:
            logger.error("edge.render_frame | decode failed error_type={}", type(exc).__name__)
            return
        if isinstance(frame, TurnFrame):
            await self._render_turn(frame)
            return
        if isinstance(frame, StatusFrame):
            await self._render_status(frame)
            return
        logger.debug("edge.render_frame | ignored frame type={}", type(frame).__name__)

    async def _render_turn(self, frame: TurnFrame) -> None:
        """Guard the whole rendered turn text, then send to the origin target.

        Serialized against the result lane by the per-task lock: the target
        lookup, dispatch and rendered-key record all happen while holding the
        lock, so an in-flight turn can never race the result lane's done marker
        or drop. A missing lock means the task was never tracked or already
        dropped; the turn is discarded (logged) rather than rendered late.
        """

        lock = self._task_locks.get(frame.task_id)
        if lock is None:
            logger.warning("edge.render_turn | no lock for task_id={}", frame.task_id)
            return
        async with lock:
            # Re-check inside the lock: the task can drop while this call waits.
            target = self._targets.get(frame.task_id)
            if target is None:
                logger.warning("edge.render_turn | no target for task_id={}", frame.task_id)
                return
            send = turn_frame_to_send(
                frame,
                target=target,
                outbound_filter=self._outbound_filter,
                attributed=self._attributed,
            )
            await self._dispatch(send.target, send.text, suppressed=send.suppressed)
            self._rendered_turns.setdefault(frame.task_id, set()).add(
                (frame.agent_id, frame.visible_text)
            )

    async def _render_status(self, frame: StatusFrame) -> None:
        """Render a minimal typed status indicator to the origin target."""

        target = self._targets.get(frame.task_id)
        if target is None:
            logger.warning("edge.render_status | no target for task_id={}", frame.task_id)
            return
        send = status_frame_to_send(
            frame,
            target=target,
            outbound_filter=self._outbound_filter,
            attributed=self._attributed,
        )
        await self._dispatch(send.target, send.text, suppressed=send.suppressed)

    async def _subscribe_external_turns(self) -> None:
        """Subscribe the shared turn wildcard for external answers.

        A no-op unless this deploy is a triage edge (``intake_enabled``): only
        then does the edge render answers for tasks it did not mint itself. A
        subscribe failure is logged and contained — the edge still runs its own
        minted-task reply path — so an external-render setup failure never crashes
        the edge or blocks inbound.
        """

        if not self._intake_enabled:
            logger.debug("edge.external | intake disabled; no turn-wildcard subscription")
            return
        try:
            self._external_subscription = await self._bus.subscribe(
                subjects.TASK_TURN_WILDCARD, self._on_external_turn
            )
        except Exception as exc:
            logger.error(
                "edge.external | wildcard subscribe failed error_type={}", type(exc).__name__
            )
            return
        logger.info(
            "edge.external | subscribed turn wildcard subject={}", subjects.TASK_TURN_WILDCARD
        )

    async def _on_external_turn(self, data: bytes) -> None:
        """Render a terminal EXTERNAL task turn back into the asking channel.

        The shared turn wildcard also carries this edge's OWN minted-task turns;
        those keep the existing per-task path and are IGNORED here (they carry a
        ``_task_locks`` entry) so nothing double-renders. An external turn is
        rendered only when it is TERMINAL (``stance.needs_from`` empty — no
        streaming for external tasks in this slice: non-terminal intermediate
        debate rounds do NOT render, trickle pacing is a future layer) AND carries
        an ``origin`` the answer can be routed back to.

        PER-KEY DEDUP with split render-progress marks (R3/R4/R5). The
        key is ``(task_id, agent_id, digest)`` for BOTH render modes, where
        ``digest`` is a content-safe hash of the FULL typed TurnFrame identity
        (:func:`_turn_identity_digest`). A byte-identical redelivery hashes the
        same and shares the record, so an exact duplicate collapses to one render
        and a partial failure resumes only the missing piece on exact replay; a
        CHANGED terminal turn for the same (task, agent) hashes differently and
        renders as its OWN distinct turn instead of being suppressed. Each key's
        VALUE is the set of render pieces that landed; a partial failure leaves
        the missing piece unmarked so redelivery resumes ONLY that piece — a TLDR
        that landed is never re-posted when the in-thread full send later succeeds
        on retry.

        RENDER SHAPE is a typed decision on the frame's ``origin.debate`` marker,
        never inferred from surface geometry. A single-scope answer (``debate``
        False, today's intake path) renders the FULL text back in-channel EXACTLY
        as before. A debate answer (``debate`` True — a duo/trio task) renders a
        short stance-derived TLDR as a PLAIN message in the PARENT channel plus
        the full turn text in a thread anchored on that TLDR message (one thread
        per task+agent, UX v2); when the thread cannot be created the full text
        degrades to the parent channel (loud, never dropped) alongside the TLDR.
        A render failure is contained (never crashes the edge); its send-failure
        log is owned by ``_dispatch`` (R5).
        """

        try:
            frame = decode_frame(data)
        except Exception as exc:
            logger.error("edge.external | decode failed error_type={}", type(exc).__name__)
            return
        if not isinstance(frame, TurnFrame):
            logger.debug("edge.external | ignored non-turn type={}", type(frame).__name__)
            return
        # Own minted task: the per-task turn subscription already renders it.
        if frame.task_id in self._task_locks:
            logger.debug(
                "edge.external | own task turn, not double-rendered task_id={}", frame.task_id
            )
            return
        # No streaming for external tasks: render only the terminal answer.
        if frame.stance.needs_from:
            logger.debug(
                "edge.external | non-terminal external turn ignored task_id={}", frame.task_id
            )
            return
        target = intake_reply_target(frame)
        if target is None:
            logger.debug(
                "edge.external | external turn carries no origin task_id={}", frame.task_id
            )
            return
        debate = frame.origin is not None and frame.origin.debate
        # Content-digest key over the FULL typed TurnFrame identity for BOTH render
        # modes: a byte-identical redelivery shares this key (so an exact duplicate
        # collapses to one render and a partial chunk/piece failure resumes only the
        # missing work on exact replay), while a CHANGED terminal turn for the same
        # (task, agent) — a distinct later substantive answer — hashes differently
        # and renders as its OWN turn instead of being suppressed. ``task_id`` stays
        # key[0] for eviction cleanup; ``agent_id`` keeps authors distinct. Only the
        # digest, never the private ``visible_text``, enters state/logs.
        key: tuple[str, ...] = (frame.task_id, frame.agent_id, _turn_identity_digest(frame))
        # Reserve the key's progress record up front so concurrent deliveries share
        # ONE mark set and cannot double-post (the R3 concurrency guard). An
        # already-COMPLETE key (all its pieces landed) is a pure repeat and skipped;
        # a partially-complete key resumes only the missing pieces.
        progress = self._external_seen.get(key)
        if progress is None:
            progress = _ExternalRenderProgress()
            self._external_seen[key] = progress
            self._evict_external_seen()
        self._external_seen.move_to_end(key)
        try:
            await self._render_external(frame, target, debate=debate, progress=progress)
        except Exception:
            # A piece send raised. Whatever landed is already marked in
            # ``progress.pieces`` (retained); an attached chunk checkpoint is
            # retained too, so redelivery resumes the missing piece at its first
            # unsent chunk. An ordinary no-progress failure keeps prior behavior:
            # remove the empty record so it is never mistaken for complete.
            if progress.is_empty():
                self._external_seen.pop(key, None)
            # The contextual send-failure log is owned by _dispatch (R5).
            return
        if progress.is_empty():
            self._external_seen.pop(key, None)
            logger.debug(
                "edge.external | suppressed external answer released task_id={} agent={}",
                frame.task_id,
                frame.agent_id,
            )
            return
        logger.info(
            "edge.external | posted external answer task_id={} channel={} debate={}",
            frame.task_id,
            target.channel_id,
            debate,
        )

    async def _subscribe_presence_transitions(self) -> None:
        """Subscribe the shared presence-transition subject.

        A no-op unless a notify target is configured (``_presence_consumer`` set):
        only then does this edge render operator Offline/Recovered notifications
        from the typed ``PresenceTransitionFrame`` the web-chat deriver publishes.
        A subscribe failure is logged sanitized and contained — the edge still runs
        its inbound/reply path — so a presence-notify setup failure never crashes
        the edge. The subject MUST match the producer's flat constant byte-for-byte.
        """

        if self._presence_consumer is None:
            logger.debug("edge.presence | no notify target; no presence subscription")
            return
        try:
            self._presence_subscription = await self._bus.subscribe(
                subjects.PRESENCE_TRANSITION, self._presence_consumer.handle
            )
        except Exception as exc:
            logger.error("edge.presence | subscribe failed error_type={}", type(exc).__name__)
            return
        logger.info(
            "edge.presence | subscribed presence transitions subject={} origin_edge={}",
            subjects.PRESENCE_TRANSITION,
            self._origin_edge,
        )

    async def _send_presence(self, text: str) -> None:
        """Send one rendered presence-notify body through the guard + adapter.

        The consumer hands an already-rendered SURFACE markdown body; this runs it
        through the SAME egress guard and adapter dispatch every other outbound
        uses, addressed at the configured notify target, so the surface applies its
        own safe rendering (Telegram sender -> allowlisted HTML, Discord -> native).
        A guard-suppressed body is dropped (never an empty send); a send failure
        propagates to the consumer's per-surface containment.
        """

        target = self._presence_notify_target
        if target is None:  # pragma: no cover - consumer is only built with a target
            logger.warning("edge.presence | send skipped: no notify target")
            return
        guarded = self._outbound_filter(text)
        await self._dispatch(target, guarded, suppressed=not guarded.strip())

    async def _render_external(
        self,
        frame: TurnFrame,
        target: SendTarget,
        *,
        debate: bool,
        progress: _ExternalRenderProgress,
    ) -> None:
        """Render one external terminal turn per its typed ``origin.debate`` shape.

        ``progress`` is the shared render-progress state for this turn's dedup key
        (R4/R5). Each piece is RESERVED in its mark set BEFORE dispatch,
        so a concurrent redelivery sharing the same set finds the piece claimed and
        does not double-post (the R3 concurrency guard); on a dispatch FAILURE the
        piece is removed so redelivery resumes ONLY the missing piece — a landed
        piece stays reserved and is never re-posted, a failed piece retries.

        Single-scope (``debate`` False): one atomic ``full`` dispatch of the
        guard-sanitized text to the reply target, EXACTLY as the pre-debate intake
        path. Debate (``debate`` True): a stance-derived TLDR posted PLAIN to the
        PARENT channel (never a reply), then the full turn text in a thread
        anchored on that TLDR message — one thread per (task, agent). A piece send
        that raises propagates to the caller's resume guard; the failed piece is
        un-reserved first so it is retried, landed pieces stay reserved.
        """
        origin = frame.origin
        if not debate or origin is None:
            if _PIECE_FULL in progress.pieces:
                return
            send = turn_frame_to_send(
                frame,
                target=target,
                outbound_filter=self._outbound_filter,
                attributed=self._attributed,
            )
            # Reserve before dispatch (concurrency claim); un-reserve on failure so
            # a redelivery retries this single piece.
            progress.pieces.add(_PIECE_FULL)
            try:
                await self._dispatch_piece(
                    send.target,
                    send.text,
                    suppressed=send.suppressed,
                    piece=_PIECE_FULL,
                    progress=progress,
                )
                if send.suppressed:
                    progress.pieces.discard(_PIECE_FULL)
                progress.chunk_cursors.pop(_PIECE_FULL, None)
            except Exception as exc:
                progress.pieces.discard(_PIECE_FULL)
                self._record_chunk_progress(progress, _PIECE_FULL, exc)
                raise
            return

        # Debate shape (UX v2): each agent's TLDR is a PLAIN channel message (no
        # reply reference — Rico: "don't do replies, just messages in the
        # channel"), and that TLDR message anchors the agent's OWN thread holding
        # its full text — a thread per (task, agent), never one shared debate
        # thread. Each piece is reserved-before-dispatch and un-reserved on
        # failure so a partial failure retries only the missing piece (R4/R5)
        # while a redelivery cannot double-post a landed piece (R3).
        # Identity routing: with ``identity_send`` every debate
        # piece is voiced as its author — the target carries the typed agent_id
        # for the surface's per-identity pool, and the TLDR drops its name
        # prefix (the posting bot IS the attribution).
        voice = frame.agent_id if self._identity_send else None
        channel_target = SendTarget(
            channel_id=target.channel_id, thread_id=None, reply_to=None, as_agent=voice
        )
        # This agent's terminal turn has arrived: stop its typing indicator
        # (started at debate mint by the intake sink's typing hook). The stop
        # target carries the ORIGIN thread id so it keys to the same indicator
        # the sink started (typing is (chat, thread, agent)-keyed). Contained
        # best-effort — a typing-stop failure must never affect the render.
        if self._identity_send:
            typing_target = SendTarget(
                channel_id=target.channel_id, thread_id=target.thread_id, as_agent=voice
            )
            try:
                await self._adapter.typing(typing_target, False)
            except Exception as exc:  # noqa: BLE001 — liveness signal, never blocks render.
                logger.warning(
                    "edge.external | typing stop failed task_id={} agent={} error_type={}",
                    frame.task_id,
                    frame.agent_id,
                    type(exc).__name__,
                )
        key = (frame.task_id, frame.agent_id)
        if _PIECE_TLDR not in progress.pieces:
            progress.pieces.add(_PIECE_TLDR)
            # Distinct-agent TLDR cap (D8): a TurnFrame carries no participant
            # roster, so bound how many distinct agents may post a TLDR per task.
            # A novel agent past the cap is dropped (warned), but its FULL text
            # still routes below so debate content is never lost — the TLDR piece
            # stays reserved so the drop is not retried as a fresh post.
            if self._debate_tldr_agent_allowed(frame.task_id, frame.agent_id):
                tldr = self._outbound_filter(
                    turn_frame_to_tldr(frame, named=not self._identity_send)
                )
                self._debate_tldr_inflight.add(key)
                try:
                    tldr_msg_id = await self._dispatch_piece(
                        channel_target,
                        tldr,
                        suppressed=not tldr.strip(),
                        piece=_PIECE_TLDR,
                        progress=progress,
                    )
                    if not tldr.strip():
                        progress.pieces.discard(_PIECE_TLDR)
                    progress.chunk_cursors.pop(_PIECE_TLDR, None)
                except Exception as exc:
                    progress.pieces.discard(_PIECE_TLDR)
                    self._record_chunk_progress(progress, _PIECE_TLDR, exc)
                    raise
                finally:
                    # Discard + (below) anchor record run in one synchronous block
                    # after the awaited dispatch, so no other task can observe
                    # "not in flight, no anchor" for a TLDR that DID land.
                    self._debate_tldr_inflight.discard(key)
                if tldr_msg_id is not None:
                    self._debate_tldr_msgs[(frame.task_id, frame.agent_id)] = tldr_msg_id
                    self._evict_debate_threads()
                    # The TLDR message id is now known — the ONLY point a callback
                    # capability can be minted and bound to it. Offer
                    # the landed TLDR to the surface interaction hook so it may
                    # attach a display-only, actor-bound keyboard on its polled
                    # default bot. Contained: a hook failure never crashes the
                    # render of the answer it decorates, and the TLDR piece stays
                    # reserved so a hook failure does not trigger a TLDR re-post.
                    await self._offer_turn_interaction(frame, target, tldr_msg_id)

        if key in self._debate_tldr_inflight:
            # A concurrent delivery of this same (task, agent) turn is mid-TLDR.
            # Its anchor is not recorded yet; claiming FULL here would degrade the
            # full text to the channel and skip the thread. Defer — that delivery
            # renders FULL itself right after its TLDR lands (or, if it fails, it
            # un-reserves and a redelivery resumes both pieces).
            return

        if _PIECE_FULL not in progress.pieces:
            progress.pieces.add(_PIECE_FULL)
            try:
                thread_id = await self._debate_thread_for(frame.task_id, frame.agent_id, target)
                if thread_id is not None:
                    thread_target = SendTarget(
                        channel_id=target.channel_id, thread_id=thread_id, as_agent=voice
                    )
                else:
                    logger.warning(
                        "edge.external | no debate thread available; full text degraded to "
                        "parent channel (not dropped) task_id={} agent={}",
                        frame.task_id,
                        frame.agent_id,
                    )
                    thread_target = channel_target
                # The full thought carries the SAME verification-aware render as a
                # non-debate answer (visible_text + typed Sources/Limitations
                # suffix), built ONCE by the shared bridge renderer. The TLDR stays
                # the concise stance line; only the in-thread full text surfaces the
                # typed evidence block. It ships through the chunk-progress dispatch
                # (piece-reserved, chunk-checkpointed) exactly like the non-debate
                # full piece so a partial send resumes rather than re-posting.
                full = turn_frame_to_send(
                    frame,
                    target=thread_target,
                    outbound_filter=self._outbound_filter,
                    attributed=self._attributed,
                )
                await self._dispatch_piece(
                    full.target,
                    full.text,
                    suppressed=full.suppressed,
                    piece=_PIECE_FULL,
                    progress=progress,
                )
                if full.suppressed:
                    progress.pieces.discard(_PIECE_FULL)
                progress.chunk_cursors.pop(_PIECE_FULL, None)
            except Exception as exc:
                progress.pieces.discard(_PIECE_FULL)
                self._record_chunk_progress(progress, _PIECE_FULL, exc)
                raise

    async def _offer_turn_interaction(
        self, frame: TurnFrame, target: SendTarget, anchor_message_id: str
    ) -> None:
        """Offer a just-landed debate TLDR to the surface interaction hook.

        Contained best-effort: the interaction is display plumbing, so a hook
        failure is logged and swallowed — it must never crash the render path or
        cause the reserved TLDR piece to re-post. The ORIGINAL requester actor
        (``origin.asker``) is passed through so the surface binds the capability
        to the human who asked, never to whoever taps.
        """
        if self._turn_interaction is None:
            return
        asker = frame.origin.asker if frame.origin is not None else None
        try:
            await self._turn_interaction.offer_debate_tldr(
                target=target,
                anchor_message_id=anchor_message_id,
                task_id=frame.task_id,
                agent_id=frame.agent_id,
                asker=asker,
            )
        except Exception as exc:  # noqa: BLE001 — interaction is display-only; never blocks render.
            logger.warning(
                "edge.external | turn interaction hook failed task_id={} agent={} error_type={}",
                frame.task_id,
                frame.agent_id,
                type(exc).__name__,
            )

    def _debate_tldr_agent_allowed(self, task_id: str, agent_id: str) -> bool:
        """Return whether ``agent_id`` may post a TLDR for ``task_id`` (D8 cap).

        Tracks the distinct agent ids that have posted a TLDR per debate task and
        caps them at :data:`DEBATE_DISTINCT_AGENT_TLDR_CAP` (trio + closing). An
        already-counted agent always passes (idempotent for its own resume); a
        novel agent past the cap is refused with a warning so a runaway fan-in of
        novel authors cannot wall the channel.
        """
        agents = self._debate_tldr_agents.get(task_id)
        if agents is None:
            agents = set()
            self._debate_tldr_agents[task_id] = agents
        if agent_id in agents:
            return True
        if len(agents) >= DEBATE_DISTINCT_AGENT_TLDR_CAP:
            logger.warning(
                "edge.external | debate distinct-agent TLDR cap reached; novel agent "
                "TLDR dropped task_id={} agent={} cap={}",
                task_id,
                agent_id,
                DEBATE_DISTINCT_AGENT_TLDR_CAP,
            )
            return False
        agents.add(agent_id)
        return True

    async def _debate_thread_for(
        self, task_id: str, agent_id: str, target: SendTarget
    ) -> str | None:
        """Return the lazily-created thread id for this agent's turn (UX v2).

        ONE thread per (task, agent), anchored to the agent's OWN landed TLDR
        message, created at that agent's first full-text render and cached
        (retired with the seen-set). A cached ``None`` marks a key whose create
        ATTEMPT already failed (or a surface with no thread primitive), so the
        create is not retried per redelivery and the full text degrades to the
        parent channel. A merely-MISSING anchor (cap-dropped agent,
        guard-suppressed TLDR, id-less surface) degrades WITHOUT caching, so an
        anchor that lands later is never blocked by a poisoned entry. Creation is
        delegated to the adapter's ``create_thread`` primitive, which fails soft
        (returns ``None``) rather than raising.

        Persistent-lane selection runs FIRST and in a fixed order (UX v3):

        1. **Exact chat+agent** — a lane configured for THIS chat id
           (``target.channel_id``) and this agent. The same agent maps to a
           different topic id per chat, so a group with its own forum topics gets
           its own lane. Keyed on ``(chat_id, agent)`` ALONE (not task), so every
           debate in that chat reuses the same topic.
        2. **Legacy agent-only** — a single-group ``agent -> topic`` lane,
           preserved byte-for-byte as the fallback when no chat-scoped lane
           matches. Keyed on ``agent`` alone.
        3. **Per-turn thread** — the existing lazy create below, unchanged, when
           neither map lists this chat/agent (so unlisted chats/agents NEVER lose
           full text).
        """
        agent_lane_key = _lane_key(agent_id)
        # 1. Exact chat+agent: this agent's lane FOR THIS CHAT wins first. No
        #    per-turn thread is minted and no TLDR anchor is needed (the lane
        #    pre-exists). Keyed on (chat_id, agent) so every debate in that chat
        #    reuses the same topic.
        chat_lane = self._debate_chat_lanes.get((target.channel_id, agent_lane_key))
        if chat_lane:
            return chat_lane
        # 2. Legacy agent-only lane (single-group fallback), byte-for-byte the
        #    prior persistent-lane behavior. Keyed on agent_id ALONE (not task) so
        #    every debate reuses the same lane.
        lane = self._debate_lanes.get(agent_lane_key)
        if lane:
            return lane

        # 3. Per-turn thread (existing lazy create): mint one thread per
        #    (task, agent) anchored to the agent's TLDR when no lane is configured.
        key = (task_id, agent_id)
        async with self._debate_thread_lock:
            if key in self._debate_threads:
                return self._debate_threads[key]
            message_id = self._debate_tldr_msgs.get(key)
            create = getattr(self._adapter, "create_thread", None)
            if create is None:
                # The surface offers no thread primitive: a stable property —
                # degrade and cache the decision so it is not re-tried.
                self._debate_threads[key] = None
                self._evict_debate_threads()
                return None
            if message_id is None:
                # No landed TLDR to anchor on (cap-dropped agent, guard-suppressed
                # TLDR, or an id-less surface). Degrade WITHOUT caching: an anchor
                # that lands later (e.g. a retried TLDR piece) must not find a
                # poisoned cache entry blocking its thread forever.
                return None
            name = _debate_thread_name(agent_id)
            thread_id: str | None
            try:
                thread_id = await create(target.channel_id, message_id, name)
            except Exception as exc:
                # The primitive is documented to fail soft; a raising impl is
                # still contained here so a thread failure never crashes the
                # render path.
                logger.warning(
                    "edge.external | debate thread create raised; degrading to "
                    "parent channel task_id={} agent={} error_type={}",
                    task_id,
                    agent_id,
                    type(exc).__name__,
                )
                thread_id = None
            self._debate_threads[key] = thread_id
            self._evict_debate_threads()
            return thread_id

    def _evict_external_seen(self) -> None:
        """Bound the external-turn seen-set, evicting oldest-first over the cap.

        Retire the per-task debate thread cache and distinct-agent TLDR set for a
        task whose LAST seen-set key is evicted, so those caches never outlive the
        turn state they shadow (bounded with the seen-set, as documented).
        """

        while len(self._external_seen) > self._external_seen_max:
            evicted, _ = self._external_seen.popitem(last=False)
            evicted_task = evicted[0]
            if not any(k[0] == evicted_task for k in self._external_seen):
                for key in [k for k in self._debate_threads if k[0] == evicted_task]:
                    self._debate_threads.pop(key, None)
                for key in [k for k in self._debate_tldr_msgs if k[0] == evicted_task]:
                    self._debate_tldr_msgs.pop(key, None)
                self._debate_tldr_agents.pop(evicted_task, None)

    def _evict_debate_threads(self) -> None:
        """Bound the per-(task, agent) debate caches alongside the seen-set cap."""

        while len(self._debate_threads) > self._external_seen_max:
            self._debate_threads.popitem(last=False)
        while len(self._debate_tldr_msgs) > self._external_seen_max:
            self._debate_tldr_msgs.popitem(last=False)
        while len(self._debate_tldr_agents) > self._external_seen_max:
            self._debate_tldr_agents.popitem(last=False)

    async def _render_result(self, task_id: str, data: bytes) -> None:
        """Render the orchestrator-derived done marker ONCE, then drop the task.

        The result lane republishes the TERMINAL ``TurnFrame``. When the turn
        lane already delivered that turn, this edge renders only a single typed
        completion marker; when the result arrives FIRST (no cross-subject
        ordering guarantee), the terminal turn is rendered here before the
        marker so the final answer is never suppressed. Done is marked only
        after the marker send succeeds, exactly once, driven entirely by an
        orchestrator-derived done arriving on the per-task result subject. The
        edge never derives done: it translates a bus-published terminal signal
        into one closing reply.

        Serialized against the turn lane by the per-task lock: the done check,
        rendered-key check/record, both dispatches, the done-mark and the drop
        all happen while holding the lock, so an in-flight turn cannot slip a
        second terminal render past the done marker or drop.
        """

        lock = self._task_locks.get(task_id)
        if lock is None:
            logger.warning("edge.render_result | no lock for task_id={}", task_id)
            return
        async with lock:
            if task_id in self._done_tasks:
                logger.debug("edge.render_result | task_id={} already done; skipping", task_id)
                return
            # Re-check inside the lock: the task can drop while this call waits.
            target = self._targets.get(task_id)
            if target is None:
                logger.warning("edge.render_result | no target for task_id={}", task_id)
                return
            try:
                frame = decode_frame(data)
            except Exception as exc:
                logger.error(
                    "edge.render_result | task_id={} decode failed error_type={}",
                    task_id,
                    type(exc).__name__,
                )
                return
            if not isinstance(frame, TurnFrame):
                logger.warning(
                    "edge.render_result | task_id={} ignored non-TurnFrame type={}",
                    task_id,
                    type(frame).__name__,
                )
                return
            # Turn and result are separate subjects with NO cross-subject
            # delivery ordering guarantee: the republished terminal turn can
            # land here before the turn lane delivers it. Render it first when
            # it hasn't rendered yet, so dropping the task never suppresses the
            # final answer.
            key = (frame.agent_id, frame.visible_text)
            if key not in self._rendered_turns.get(task_id, set()):
                send = turn_frame_to_send(
                    frame,
                    target=target,
                    outbound_filter=self._outbound_filter,
                    attributed=self._attributed,
                )
                await self._dispatch(send.target, send.text, suppressed=send.suppressed)
                self._rendered_turns.setdefault(task_id, set()).add(key)
            text = f"{DONE_MARKER} — {frame.agent_id}" if self._attributed else DONE_MARKER
            guarded = self._outbound_filter(text)
            # Done is recorded only AFTER the marker send succeeds: a transient
            # send failure propagates, the task stays live with its
            # subscriptions, and a redelivered result can retry instead of
            # losing the marker.
            await self._dispatch(target, guarded, suppressed=not guarded.strip())
            self._done_tasks.add(task_id)
            await self._drop_task(task_id)

    async def _drop_task(self, task_id: str) -> None:
        """Retire a completed task's reply subscription set and forget its state.

        Constraint: this must NOT acquire ``self._task_locks[task_id]``. The
        result path calls it while already holding that lock, so acquiring it
        here would deadlock. It pops the lock instead, under the caller's lock.
        """

        subscriptions = self._task_subscriptions.pop(task_id, None)
        if subscriptions is not None and not await subscriptions.retire():
            # An unsubscribe failed; the set retains the live handle(s). Never
            # silently discard it -- hold it for retry at close.
            self._leaked_subscriptions[task_id] = subscriptions
            logger.critical(
                "edge.drop_task | task_id={} retire incomplete; live subscription(s) "
                "held for retry subjects={}",
                task_id,
                subscriptions.subjects,
            )
        self._targets.pop(task_id, None)
        self._rendered_turns.pop(task_id, None)
        self._task_locks.pop(task_id, None)
        logger.info("edge.drop_task | task_id={} dropped", task_id)

    def _record_chunk_progress(
        self, progress: _ExternalRenderProgress, piece: str, exc: Exception
    ) -> None:
        """Retain a monotonic, content-free checkpoint exposed by an adapter."""

        checkpoint = chunk_progress_from(exc)
        if checkpoint is None:
            return
        prior = progress.chunk_cursors.get(piece)
        if (
            prior is not None
            and prior.plan_fingerprint == checkpoint.plan_fingerprint
            and checkpoint.chunks_landed < prior.chunks_landed
        ):
            logger.warning(
                "edge.external | ignored regressing chunk checkpoint piece={} "
                "prior_landed={} reported_landed={} total={}",
                piece,
                prior.chunks_landed,
                checkpoint.chunks_landed,
                checkpoint.chunk_count,
            )
            return
        progress.chunk_cursors[piece] = checkpoint
        logger.warning(
            "edge.external | chunk checkpoint retained piece={} landed={} total={}",
            piece,
            checkpoint.chunks_landed,
            checkpoint.chunk_count,
        )

    async def _dispatch_piece(
        self,
        target: SendTarget,
        text: str,
        *,
        suppressed: bool,
        piece: str,
        progress: _ExternalRenderProgress,
    ) -> str | None:
        """Dispatch one external render piece with its prior chunk checkpoint."""

        return await self._dispatch(
            replace(target, chunk_resume=progress.chunk_cursors.get(piece)),
            text,
            suppressed=suppressed,
        )

    async def _dispatch(self, target: SendTarget, text: str, *, suppressed: bool) -> str | None:
        """Send guarded text to the chat surface, skipping suppressed output.

        Returns the sent message id when the adapter exposes one (``None`` for a
        suppressed send or an id-less surface) so a debate TLDR can anchor its
        own thread.
        """

        if suppressed:
            logger.info("edge.dispatch | suppressed by guard channel={}", target.channel_id)
            return None
        try:
            message_id = await self._adapter.send(target, text, [])
        except Exception as exc:
            logger.error(
                "edge.dispatch | send failed channel={} error_type={}",
                target.channel_id,
                type(exc).__name__,
            )
            raise
        logger.debug("edge.dispatch | sent channel={} len={}", target.channel_id, len(text))
        return message_id

    async def close(self) -> None:
        """Retire every remaining reply subscription set, retrying leaked ones once.

        A per-task set whose retire fails is moved into ``_leaked_subscriptions``
        rather than discarded; every leaked set is then retried ONCE. Sets that
        fully retire are cleared; any still live after the final retry are logged
        at critical level (never silently dropped).
        """

        logger.debug("edge.close | task_subscriptions={}", len(self._task_subscriptions))
        # Retire the shared external-turn wildcard before the per-task
        # sets. A failure is logged and contained so teardown still proceeds.
        if self._external_subscription is not None:
            try:
                await self._external_subscription.unsubscribe()
            except Exception as exc:
                logger.error(
                    "edge.close | external subscription unsubscribe failed error_type={}",
                    type(exc).__name__,
                )
            else:
                logger.debug("edge.close | external turn subscription retired")
            self._external_subscription = None
        # Retire the shared presence-transition subscription. Contained
        # exactly like the external-turn one so a failure never aborts teardown.
        if self._presence_subscription is not None:
            try:
                await self._presence_subscription.unsubscribe()
            except Exception as exc:
                logger.error(
                    "edge.close | presence subscription unsubscribe failed error_type={}",
                    type(exc).__name__,
                )
            else:
                logger.debug("edge.close | presence transition subscription retired")
            self._presence_subscription = None
        for task_id, subscriptions in list(self._task_subscriptions.items()):
            if not await subscriptions.retire():
                self._leaked_subscriptions[task_id] = subscriptions
                logger.critical(
                    "edge.close | task_id={} retire incomplete subjects={}",
                    task_id,
                    subscriptions.subjects,
                )
            else:
                logger.debug("edge.close | retired task_id={}", task_id)
        self._task_subscriptions.clear()
        # Final retry pass over every leaked set; clear the ones that fully
        # retire, escalate any that remain live.
        for task_id, subscriptions in list(self._leaked_subscriptions.items()):
            if await subscriptions.retire():
                del self._leaked_subscriptions[task_id]
                logger.debug("edge.close | leaked set retired task_id={}", task_id)
            else:
                logger.critical(
                    "edge.close | task_id={} still live after final retry subjects={}",
                    task_id,
                    subscriptions.subjects,
                )
        logger.info("edge.close | closed leaked_remaining={}", len(self._leaked_subscriptions))

    async def deliver_to_home(self, text: str) -> bool:
        """Deliver ``text`` to the home channel through the egress guard.

        This is the ONLY path allowed to surface the missing-home advisory into
        Discord — and only on a genuine failed delivery attempt, never inbound
        onboarding. Returns True on a successful send, False when no home target
        is configured (the caller may surface :data:`HOME_ADVISORY`).
        """

        if self._home_target is None:
            logger.error(
                "chatd.home | deliver_to_home failed agent={} reason=no-home-channel; "
                "advisory available to caller only",
                self._agent_label,
            )
            return False
        guarded = self._outbound_filter(text)
        if not guarded.strip():
            logger.info(
                "chatd.home | deliver_to_home suppressed by guard agent={}", self._agent_label
            )
            return False
        try:
            await self._adapter.send(self._home_target, guarded, [])
        except Exception as exc:
            logger.error(
                "chatd.home | deliver_to_home send failed error_type={}", type(exc).__name__
            )
            raise
        logger.info("chatd.home | delivered to home target={}", self._home_target.resolve())
        return True

    @property
    def home_advisory(self) -> str:
        """Return the missing-home advisory string (caller-only; never auto-posted)."""

        return HOME_ADVISORY
