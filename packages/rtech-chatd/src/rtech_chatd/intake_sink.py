"""The intake SINK: mint + publish a triage TaskFrame from an intake hit.

The pure hit-gate (:func:`rtech_chatd.gates.intake_hit`) decides WHETHER a
channel message is an intake hit; the adapter fires the injected sink on a hit
(:meth:`DiscordChannelAdapter._maybe_intake`). This module is that sink's body —
the composition-layer piece that turns a hit into a bus publish toward the
configured triage agent (the deployed ``agentd-<name>``).

It owns three concerns the two prior reviews deferred here:

1. **Dedup by the composite ``channel_id:message_id`` key.** Discord (and a
   gateway reconnect) can redeliver the SAME message; a member-config @mention
   that ALSO fires the responder is a documented double-activation. The sink mints
   a given message at most ONCE within a bounded in-process retention window, so a
   redelivered or double-fired message never mints twice. The dedup key is built
   here at the call site as ``channel_id:message_id``: Discord message ids are
   globally-unique snowflakes so the channel prefix is a no-op there, but Telegram
   message ids are PER-CHAT, so the composite is required to keep the same id in
   two different chats from colliding into one mint. The store is a small
   count+time-bounded
   LRU — this is an in-process, daily-volume Discord edge, not a durable
   cross-restart authority (contrast :class:`rtech_agentd.completed_store`, which
   is on-disk because agentd redelivery survives a restart; the intake edge's
   redelivery window is a single live gateway session, so in-process is enough).

2. **Responder-yield (suppress the double-activation at the source).** Whenever
   the responder path would itself answer a message, intake would mint a
   redundant second wake for it. The sink yields on the responder's OWN verdict —
   the same injected ``should_respond`` against the same ``GateConfig`` the
   responder path evaluates — so exactly-one-handler holds: a direct @mention, a
   free-response inbox name-trigger (deploy makes the inbox free-response, which
   the responder answers with NO mention), and an allowed DM all dispatch once
   via the responder, while a hit the responder DECLINES (e.g. a member-config
   mention of a DIFFERENT trio member, or an ignored-for-the-responder channel)
   still mints an intake frame. Yielding on ``mentions_me`` alone would miss the
   free-response/DM cases and let both the responder and intake fire.

3. **Publish errors logged ONCE, then contained.** A publish/encode failure is
   logged EXACTLY once here — with full context (subject, triage agent, channel /
   message / task ids) — and then CONTAINED (the sink returns, it does not
   re-raise). This is the authoritative intake-publish error log. The adapter's
   intake error boundary (:meth:`DiscordChannelAdapter._maybe_intake`) stays only
   as a backstop for OTHER sink implementations, so a real publish failure is not
   double-logged. Note: :mod:`rtech_chatd.bus`'s own transport error log is
   pre-existing shared-bus behavior and out of scope for this sink.

The mint itself is the pure :func:`rtech_chatd.bridge.inbound_to_intake_task_frame`
(config-driven triage agent id, required stack, channel origin, ``created_at``
stamp). The subject is built ONLY via :func:`rtech_chatd.subjects.agent_task`.
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable

from loguru import logger
from rtech_frames.codec import encode_frame

from . import subjects
from .bridge import inbound_to_intake_task_frame
from .bus import FrameBus
from .channel import InboundMessage, SendTarget
from .config import GateConfig, IntakeConfig
from .gates import should_respond

# Sink defaults for the in-process dedup store. These bound an in-process,
# daily-volume Discord edge — a large LRU by count, aged out by a time horizon.
# Each entry is an id string + a float, so 65536 entries is trivial memory: the
# cap sits far above any normal channel's per-horizon message volume so capacity
# eviction never trims an id still inside the horizon under normal load. If an
# id-flood ever DOES force a not-yet-expired eviction, the store logs a loud
# warning (dedup window degraded) — at-most-once is bounded-best-effort under a
# flood, exact within normal volume.
DEFAULT_DEDUP_MAX_ENTRIES = 65536
DEFAULT_DEDUP_HORIZON_SECONDS = 3600.0

# Monotonic clock injected for testability (dedup time-eviction is monotonic —
# it never needs to survive a restart, unlike the agentd wall-clock store).
Clock = Callable[[], float]
IdFactory = Callable[[], str]
ShouldRespond = Callable[[InboundMessage, GateConfig], bool]
# Instant-ack sender (UX v2): fire-and-forget surface send wired to the adapter
# by the service layer. Returns the sent message id (unused here) or None.
AckSend = Callable[[SendTarget, str], Awaitable[str | None]]
# Typing starter (three-signal contract, signal 3): wired to the
# adapter's typing-on by the composition layer. Invoked once per debate
# participant with a target stamped ``as_agent`` so each drafting agent's OWN
# identity shows a live typing indicator; the edge stops it when that agent's
# terminal turn renders.
TypingStart = Callable[[SendTarget], Awaitable[None]]


class _MessageIdDedup:
    """A count+time-bounded set of already-minted message ids (in-process).

    Keyed by Discord ``message_id``. An id is remembered when first minted and
    suppressed on any repeat within the retention window. Bounded two ways so an
    adversarial or high-volume channel cannot grow it without limit:

    * COUNT: an :class:`~collections.OrderedDict` used as an LRU — inserting past
      ``max_entries`` evicts the oldest-inserted id (expired-first, then oldest).
    * TIME: an id older than ``horizon_s`` (monotonic) is treated as absent, so a
      message id can mint again only after the window has fully elapsed.

    At-most-once is EXACT within normal volume (the default ``max_entries`` sits
    far above any channel's per-horizon message count, so capacity eviction only
    ever removes already-expired entries) and BOUNDED-BEST-EFFORT under an
    id-flood: if capacity eviction is ever forced to remove a not-yet-expired
    entry, that is logged as a loud warning (the dedup window has degraded) so
    overload is never silent, and a redelivered id could then re-mint inside the
    window. This is intentional — the cap is the memory backstop, the warning is
    the loud signal, and the horizon is the normal-case correctness bound.

    Monotonic time is deliberate: this store never survives a restart (a gateway
    reconnect starts a fresh session), so there is no wall-clock durability need —
    unlike :class:`rtech_agentd.completed_store`, whose horizon must survive one.
    """

    def __init__(self, *, max_entries: int, horizon_s: float, clock: Clock) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if horizon_s <= 0:
            raise ValueError("horizon_s must be > 0")
        self._max_entries = max_entries
        self._horizon_s = horizon_s
        self._clock = clock
        # message_id -> monotonic insert time. OrderedDict order = insertion order
        # (LRU by age of first mint).
        self._seen: OrderedDict[str, float] = OrderedDict()

    def check_and_remember(self, message_id: str) -> bool:
        """Return True when ``message_id`` is FRESH (mint it), False when a repeat.

        A fresh id is recorded and returns True. A repeat still inside the horizon
        returns False (suppress). A repeat whose prior record has aged past the
        horizon is treated as fresh: it is re-recorded and returns True.
        """
        now = self._clock()
        self._evict_expired(now)
        prior = self._seen.get(message_id)
        if prior is not None and (now - prior) <= self._horizon_s:
            # Known and still within the window — suppress. Refresh recency so a
            # steadily-repeated id does not get LRU-evicted out from under the
            # suppression (its window keeps sliding while repeats keep arriving).
            self._seen.move_to_end(message_id)
            return False
        # Fresh (never seen, or the prior record aged out): record and admit.
        self._seen[message_id] = now
        self._seen.move_to_end(message_id)
        self._evict_over_capacity(now)
        return True

    def _evict_expired(self, now: float) -> None:
        """Drop every id whose record has aged past the horizon (oldest first)."""
        while self._seen:
            oldest_id, oldest_at = next(iter(self._seen.items()))
            if (now - oldest_at) <= self._horizon_s:
                break
            self._seen.popitem(last=False)

    def _evict_over_capacity(self, now: float) -> None:
        """Evict oldest-inserted ids until the store is within ``max_entries``.

        Eviction order stays expired-first then oldest: :meth:`_evict_expired`
        already ran, so any remaining over-capacity eviction removes the OLDEST
        surviving entry. If that oldest survivor is still INSIDE the horizon, the
        dedup window has degraded (an id-flood outran the cap) — log a loud
        warning naming the cap + horizon so overload is never silent. Without it,
        a redelivered id whose record was capacity-evicted would re-mint inside
        the window: at-most-once becomes bounded-best-effort, so we make it loud.
        """
        while len(self._seen) > self._max_entries:
            oldest_id, oldest_at = next(iter(self._seen.items()))
            if (now - oldest_at) <= self._horizon_s:
                logger.warning(
                    "chatd.intake_sink | dedup window degraded | capacity eviction "
                    "removed a not-yet-expired id (cap={} horizon_s={}); a "
                    "redelivered id may re-mint inside the window",
                    self._max_entries,
                    self._horizon_s,
                )
            self._seen.popitem(last=False)

    def __len__(self) -> int:
        return len(self._seen)


class IntakeSink:
    """Mint + publish a triage TaskFrame from a deterministic intake hit.

    Wired at chatd's composition layer against the same :class:`FrameBus` the
    edge uses. On an intake hit the adapter invokes :meth:`__call__`; the sink
    dedups by ``message_id``, yields to the responder whenever the responder
    would itself answer (double-activation suppression), mints via
    :func:`inbound_to_intake_task_frame`, and publishes the encoded frame to the
    triage agent's task subject.

    Args:
        bus: The connected frame bus (reused from the entrypoint's publisher).
        intake_config: Intake config (triage agent id lives here).
        gate_config: The responder gate config, reused so the responder-yield
            decision is IDENTICAL to what the responder path itself computes.
        stack: Required stack-scope stamp; a blank stack fails closed
            in the mint (never a default shared literal).
        id_factory: Task-id minter (uuid4 hex by default).
        should_respond_fn: The responder decision (injectable for tests); the
            default is the real :func:`rtech_chatd.gates.should_respond`.
        clock: Monotonic clock for dedup time-eviction (injectable for tests).
        dedup_max_entries / dedup_horizon_s: Dedup store bounds.
        ack_send: Optional surface send for the instant debate-mint ack (UX v2);
            when None (default, and for single-scope mints) no ack is posted.
        typing_start: Optional per-participant typing starter;
            when set, a debate mint starts a typing indicator for EVERY
            participant identity in the origin chat. None (default) preserves
            existing behavior exactly.
        voiced: True (default): the ack text carries the ``**arbiter** —``
            name prefix (single-bot surfaces, where the prefix IS the
            attribution). False: the ack is sent under the triage
            agent's OWN bot identity (``as_agent`` stamped on the target), so
            the prefix is dropped as redundant ventriloquism.
    """

    def __init__(
        self,
        *,
        bus: FrameBus,
        intake_config: IntakeConfig,
        gate_config: GateConfig,
        stack: str,
        id_factory: IdFactory | None = None,
        should_respond_fn: ShouldRespond | None = None,
        clock: Clock | None = None,
        dedup_max_entries: int = DEFAULT_DEDUP_MAX_ENTRIES,
        dedup_horizon_s: float = DEFAULT_DEDUP_HORIZON_SECONDS,
        ack_send: AckSend | None = None,
        typing_start: TypingStart | None = None,
        voiced: bool = True,
    ) -> None:
        if not stack or not stack.strip():
            # Fail closed at construction, mirroring DiscordEdge: an intake sink
            # that would mint blank-stack frames every orchestrator drops is a
            # misconfiguration, caught before the first live hit.
            raise ValueError("stack must be non-empty; there is no default stack id")
        self._bus = bus
        self._intake_config = intake_config
        self._gate_config = gate_config
        self._stack = stack
        self._id_factory = id_factory or _new_task_id
        self._should_respond = should_respond_fn or should_respond
        self._dedup = _MessageIdDedup(
            max_entries=dedup_max_entries,
            horizon_s=dedup_horizon_s,
            clock=clock or time.monotonic,
        )
        self._ack_send = ack_send
        self._typing_start = typing_start
        self._voiced = voiced

    @property
    def triage_agent(self) -> str:
        """Return the configured triage agent id a hit dispatches to."""
        return self._intake_config.triage_agent

    async def __call__(self, msg: InboundMessage) -> None:
        """Mint + publish a triage TaskFrame for ``msg`` (an intake hit).

        The adapter has already confirmed the hit-gate; this owns dedup,
        responder-yield, mint, and publish. A publish failure is logged ONCE
        with full context and CONTAINED (the sink returns; it does not re-raise),
        so a real publish failure is not double-logged at the adapter boundary.
        """
        # 1. Responder-yield: if the responder path would itself answer this
        #    message (its own should_respond verdict — mention, free-response
        #    inbox, or allowed DM), it owns the message; do not mint a redundant
        #    second wake.
        if self._responder_handles(msg):
            logger.info(
                "chatd.intake_sink | yield=responder | triage={} channel={} message_id={}",
                self.triage_agent,
                msg.channel_id,
                msg.message_id,
            )
            return

        # 2. Dedup by the COMPOSITE (channel_id, message_id) key: the same message
        #    never mints twice within the retention window (redelivery /
        #    double-activation). Discord message ids are globally-unique snowflakes,
        #    so prefixing the channel id is a harmless no-op there; Telegram message
        #    ids are PER-CHAT (every chat restarts its own id sequence), so the
        #    composite is REQUIRED — without it, message "1" in group A would
        #    suppress an unrelated message "1" in group B. The store itself stays
        #    keyed on an opaque string; the composition happens here at the edge.
        key = f"{msg.channel_id}:{msg.message_id}" if msg.message_id is not None else None
        if key is not None and not self._dedup.check_and_remember(key):
            logger.info(
                "chatd.intake_sink | dedup=repeat | triage={} channel={} message_id={}",
                self.triage_agent,
                msg.channel_id,
                msg.message_id,
            )
            return

        # 3. Mint the triage TaskFrame (pure field copy; channel origin + stamp).
        #    Single-scope (no debate participants): one triage turn, published
        #    DIRECT to the triage agent's task lane, EXACTLY as before. Debate
        #    (duo/trio participants configured): one multi-participant frame
        #    published to TASK_INGRESS so the ORCHESTRATOR fans it verbatim to each
        #    participant's lane — mirroring ``rtech_orchestrator`` ``_route_multi``/
        #    ``_dispatch_batch`` (one ingress frame -> orchestrator publishes to
        #    every ``agent_task(participant)`` lane), NOT a fanout reinvented here.
        task_id = self._id_factory()
        scope = self._intake_config.debate_scope
        if scope is None:
            frame = inbound_to_intake_task_frame(
                msg,
                task_id=task_id,
                stack=self._stack,
                triage_agent=self.triage_agent,
            )
            subject = subjects.agent_task(self.triage_agent)
        else:
            frame = inbound_to_intake_task_frame(
                msg,
                task_id=task_id,
                stack=self._stack,
                triage_agent=self.triage_agent,
                scope=scope,
                participants=list(self._intake_config.debate_participants),
                debate=True,
                arbiter=self.triage_agent,
            )
            subject = subjects.TASK_INGRESS

        # 4. Publish (subject built ONLY via subjects / the ratified ingress token).
        try:
            await self._bus.publish(subject, encode_frame(frame))
        except Exception as exc:
            # Log ONCE here with full context, then CONTAIN the failure (return,
            # do not re-raise). This is the single authoritative intake-publish
            # error log — re-raising would double-log at the adapter boundary
            # (:meth:`DiscordChannelAdapter._maybe_intake`), which stays only as a
            # backstop for OTHER sink implementations that fail before this point.
            # (bus.py logs its own transport error too; that is pre-existing
            # shared-bus behavior, out of scope for this sink.)
            # Log the exception TYPE only, never its message: a bus/encode
            # failure can carry connection/credential detail. The safe subject +
            # ids keep this authoritative log actionable. Containment is
            # unchanged (return, do not re-raise).
            logger.error(
                "chatd.intake_sink | publish failed | subject={} triage={} channel={} "
                "message_id={} task_id={} error_type={}",
                subject,
                self.triage_agent,
                msg.channel_id,
                msg.message_id,
                task_id,
                type(exc).__name__,
            )
            return
        logger.info(
            "chatd.intake_sink | published | subject={} triage={} channel={} thread={} "
            "message_id={} task_id={}",
            subject,
            self.triage_agent,
            msg.channel_id,
            msg.thread_id,
            msg.message_id,
            task_id,
        )

        # 5. Instant ack (UX v2): a debate mint immediately posts a deterministic
        #    one-liner voiced as the triage agent so the human sees life within a
        #    second, long before the first engine turn lands. Plain channel
        #    message (never a reply). Best-effort: an ack failure is logged and
        #    contained — the task is already published and must not be affected.
        if scope is not None and self._ack_send is not None:
            # Unvoiced: the ack goes out under the triage agent's
            # OWN bot, so the target carries the identity and the text drops
            # the name prefix. Voiced (default) is the existing single-bot ack.
            ack_target = SendTarget(
                channel_id=msg.channel_id,
                thread_id=msg.thread_id,
                as_agent=None if self._voiced else self.triage_agent,
            )
            try:
                await self._ack_send(ack_target, self._debate_ack_text())
            except Exception as exc:
                logger.warning(
                    "chatd.intake_sink | ack failed | channel={} task_id={} error_type={}",
                    msg.channel_id,
                    task_id,
                    type(exc).__name__,
                )

        # 6. Typing start (signal 3): every debate participant's
        #    identity starts typing in the origin chat so the gap between ack
        #    and first TLDR never looks dead. Best-effort per participant and
        #    contained — the task is already published; a typing failure (or an
        #    unknown identity) logs and never affects the mint. The edge stops
        #    each agent's indicator when its terminal turn renders.
        if scope is not None and self._typing_start is not None:
            for participant in self._intake_config.debate_participants:
                typing_target = SendTarget(
                    channel_id=msg.channel_id,
                    thread_id=msg.thread_id,
                    as_agent=participant,
                )
                try:
                    await self._typing_start(typing_target)
                except Exception as exc:
                    logger.warning(
                        "chatd.intake_sink | typing start failed | channel={} agent={} "
                        "task_id={} error_type={}",
                        msg.channel_id,
                        participant,
                        task_id,
                        type(exc).__name__,
                    )

    def _debate_ack_text(self) -> str:
        """Compose the deterministic debate-mint ack, voiced as the triage agent.

        Pure template over configured ids (no model output, nothing user-echoed):
        the triage agent announces pickup and names the other participants as
        drafting. Ids render with the ``agentd-`` convention prefix stripped.
        With ``voiced=False`` the arbiter's name prefix is dropped
        — the ack is posted under the arbiter's own bot identity, which IS the
        attribution.

        NOTE (cross-bot amplification): the ack names participants, and those
        names may be another edge's intake content triggers. Under the default
        ``allow_bots`` gating a bot-authored ack can never hit intake; if a
        deploy ever relaxes bot gating in a multi-edge channel, this text
        becomes a mint->ack->mint amplifier — keep acks bot-gated or strip
        names before relaxing.
        """
        arbiter = self.triage_agent.removeprefix("agentd-")
        others = [
            p.removeprefix("agentd-")
            for p in self._intake_config.debate_participants
            if p != self.triage_agent
        ]
        if others:
            drafting = " and ".join(others) if len(others) <= 2 else ", ".join(others)
            body = f"on it. {drafting} are drafting their takes; I'll arbitrate."
        else:
            body = "on it."
        if self._voiced:
            return f"**{arbiter}** — {body}"
        # str.capitalize() would lowercase the rest of the line; only the first
        # character needs raising.
        return body[0].upper() + body[1:]

    def _responder_handles(self, msg: InboundMessage) -> bool:
        """Return whether the responder path already answers ``msg`` itself.

        This yields on the responder's OWN verdict — the exact same injected
        :func:`should_respond` against the exact same :class:`GateConfig` the
        responder path evaluates. If the responder will answer, it owns the
        message and intake must NOT mint a second wake for it; if the responder
        declines, intake mints (the builder's intended case).

        It intentionally does NOT also require ``mentions_me``: the responder
        answers WITHOUT a mention in free-response channels (deploy always makes
        the inbox free-response — ``service.py`` / ``gates.should_respond`` step
        7) and in allowed DMs (``gates.should_respond`` step 4). A name-trigger
        message in the free-response inbox is both an intake hit AND a responder
        hit with no mention; gating only on ``mentions_me`` would let both the
        responder and intake fire on it. Yielding purely on ``should_respond``
        closes that free-response/DM double-fire while still minting when the
        responder declines.
        """
        return self._should_respond(msg, self._gate_config)


def _new_task_id() -> str:
    """Return a fresh opaque task id (uuid4 hex — the repo idiom)."""
    return uuid.uuid4().hex


__all__ = [
    "DEFAULT_DEDUP_HORIZON_SECONDS",
    "DEFAULT_DEDUP_MAX_ENTRIES",
    "IntakeSink",
]
