"""Render operator presence notifications from typed transition frames.

chatd is a DUMB EDGE: it ONLY formats the typed
:class:`~rtech_frames.frames.PresenceTransitionFrame` the web-chat deriver publishes
on ``rtech.presence.transition``. It never infers a transition from prose, and it
renders ONLY the safe fields — ``display_name``, a human observed time, and a Dash
live-page link. It NEVER emits ``agent_id`` as visible identity text, a NATS subject,
a raw payload, exception text, or any secret.

The Dash live-page URL is ``f"{validated_base}/agents/<agent_id>"`` — the base is
validated/sanitized at the construction boundary (:func:`validate_dash_base_url`,
Finding 4: absolute http/https only, no userinfo/query/fragment/control chars) and
the ``agent_id`` appears ONLY as an opaque URL PATH segment (the stable locator, the
``EvidenceSource.locator`` precedent), never as rendered identity. An empty/invalid
``dash_base_url`` renders the notification WITHOUT a link rather than a broken/unsafe
URL (fail safe, still notify); the raw URL is never logged.

Render is SURFACE-NEUTRAL MARKDOWN that flows through the EXISTING per-surface send
path, so each surface applies its OWN safe rendering — chatd never hand-rolls a
second HTML pipeline:

- **Telegram** — the notification body is markdown; the send path
  (:class:`~rtech_chatd.telegram_sender.TelegramSender` via
  :func:`rtech_chatd.telegram_html.telegram_chunks`) escapes it once into the
  allowlisted-tag HTML the Bot API accepts, so ``**bold**`` and a
  ``[label](url)`` link become safe ``<b>``/``<a>`` — a pre-rendered HTML body would
  be double-escaped by that same path. The Dash link rides as a markdown
  ``[label](url)`` so the safe scheme-restricted anchor renders it.
- **Discord** — Discord does NOT render a ``[label](url)`` link in an ordinary
  message (only in embeds), so the Discord body inlines the BARE Dash URL instead.
  ``**bold**`` renders natively.

Malformed/unknown/other-kind frame bytes fail CLOSED: the consumer logs sanitized,
drops, and never crashes the subscription or renders prose.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import urlsplit

from loguru import logger
from rtech_frames.codec import FrameDecodeError, decode_frame
from rtech_frames.frames import PresenceTransitionFrame

# Only a subject-safe persona slug may enter the Dash URL path. The
# frame validator (Finding 1) already constrains ``agent_id`` to this exact
# ``[A-Za-z0-9_-]+`` shape — the ``EvidenceSource.locator`` precedent for a safe
# path segment — but this is the belt-and-suspenders guard AT the URL boundary: a
# future producer that weakened the validator can never inject a path-traversal or
# scheme-breaking segment, because a non-matching slug drops the link entirely
# rather than rendering a broken/unsafe URL.
_SAFE_URL_SLUG = re.compile(r"^[A-Za-z0-9_-]+$")

# The ONLY schemes a Dash live-page base URL may use (Finding 4). A
# rendered Dash link rides into a possibly-leaked operator surface, so the wire
# admits only absolute http/https — never ``javascript:``, ``file:``, ``data:``,
# or a scheme-relative/relative base that could resolve unexpectedly.
_ALLOWED_DASH_SCHEMES = frozenset({"http", "https"})


def validate_dash_base_url(dash_base_url: str) -> str:
    """Validate/sanitize a configured Dash base URL, or return ``""`` (fail closed).

    Called at the CONSTRUCTION boundary (:class:`PresenceNotifyConsumer`) so a
    malformed base can NEVER reach link rendering (Finding 4). Admits ONLY
    an absolute ``http``/``https`` URL with a host and NO userinfo (``user:pass@`` —
    a credential-exposure/mention-injection vector), NO control or newline characters,
    NO query, and NO fragment. The safe trailing-slash form is normalized (a single
    trailing ``/`` on the path is stripped) so the rendered link is
    ``f"{base}/agents/<slug>"`` with exactly one separator. Anything else FAILS CLOSED
    to ``""`` — the consumer then renders notifications WITHOUT a link rather than a
    broken/unsafe URL (still notify). The raw URL is NEVER logged.

    Args:
        dash_base_url: The configured base URL (may be empty/whitespace/malformed).

    Returns:
        The normalized safe base URL, or ``""`` when unset or invalid.
    """

    base = dash_base_url.strip()
    if not base:
        return ""
    # Reject any control/newline char anywhere (defends a header/log-injection or a
    # mention smuggled past a naive check). ``strip`` above only trims the ends.
    if any(ch in base for ch in "\r\n\t") or any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in base):
        logger.error("chatd.presence_notify | rejected Dash base URL: control characters")
        return ""
    try:
        parts = urlsplit(base)
    except ValueError:
        logger.error("chatd.presence_notify | rejected Dash base URL: unparseable")
        return ""
    if parts.scheme not in _ALLOWED_DASH_SCHEMES:
        logger.error("chatd.presence_notify | rejected Dash base URL: unsupported scheme")
        return ""
    if not parts.hostname:
        logger.error("chatd.presence_notify | rejected Dash base URL: missing host")
        return ""
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        # Userinfo (``user:pass@host``) is a credential-exposure + mention-injection
        # vector; never render it. Fail closed without logging the raw URL.
        logger.error("chatd.presence_notify | rejected Dash base URL: userinfo present")
        return ""
    if parts.query or parts.fragment:
        logger.error("chatd.presence_notify | rejected Dash base URL: query/fragment present")
        return ""
    # Normalize the safe trailing-slash form: one trailing '/' on the path is stripped
    # so the rendered link has exactly one separator before ``/agents``. A path that is
    # ONLY slashes normalizes to empty.
    path = parts.path.rstrip("/")
    return f"{parts.scheme}://{parts.netloc}{path}"


# Human-facing headline per typed transition (never scraped — a fixed vocabulary
# keyed by the typed ``transition`` literal). Rendered ``**bold**`` in markdown so
# each surface's own send path styles it safely.
_HEADLINE: dict[str, str] = {
    "offline": "🔴 {name} went offline",
    "recovered": "🟢 {name} is back online",
}

# The two surfaces this bridge renders for. Kept as a typed literal so a caller
# cannot ask for an unknown surface rendering.
Surface = Literal["telegram", "discord"]

# Sends one already-rendered markdown notification body to an operator surface.
# Injected so the renderer/consumer stay pure and testable without a live adapter;
# the service wires it to the EXISTING chatd send target, never a new channel.
NotifySend = Callable[[str], Awaitable[None]]


def _observed_line(observed_at: datetime) -> str:
    """Format the observed transition time as a safe human UTC string.

    Uses a fixed ``%Y-%m-%d %H:%M:%S UTC`` shape from the frame's tz-aware datetime —
    no locale, no free text, nothing that could carry private content. The instant is
    normalized to UTC before formatting so the literal ``UTC`` suffix is HONEST even
    if a producer weakened the frame validator: a ``+05:00`` instant renders the true
    UTC wall clock, never a local time under a false UTC label. A naive datetime (the
    frame validator already rejects it) is treated as UTC rather than crashing the
    render.
    """

    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=UTC)
    return observed_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def dash_link_url(dash_base_url: str, agent_id: str) -> str | None:
    """Compose the Dash live-page URL from a VALIDATED base, or None when unavailable.

    DESIGN: the Dash link is a FIXED validated base plus a safe-slug
    path — ``f"{validated_base}/agents/<agent_id>"``. The base is run through
    :func:`validate_dash_base_url` so ONLY an absolute http/https URL with no
    userinfo/query/fragment/control chars can render a link — anything else fails
    closed to None (Finding 4). ``agent_id`` rides ONLY as an opaque URL PATH segment
    (the ``EvidenceSource.locator`` precedent), never as visible identity text. The
    frame validator already constrains it to ``[A-Za-z0-9_-]+``; this guard re-checks
    the slug AT the URL boundary as belt-and-suspenders, so a future producer that
    weakened the validator can never inject a path-traversal (``../``) or
    scheme-breaking segment — a non-safe slug (or an invalid/empty base URL) returns
    None and the caller renders WITHOUT a link rather than a broken/unsafe URL (fail
    safe, still notify). The raw base URL and raw agent_id are NEVER logged.
    """

    base = validate_dash_base_url(dash_base_url)
    if not base:
        return None
    if _SAFE_URL_SLUG.fullmatch(agent_id) is None:
        # Defensive: only a subject-safe slug may enter the URL path. Drop the link
        # rather than emit an unsafe/broken URL (the agent_id is NEVER logged raw).
        logger.warning("chatd.presence_notify | dropped Dash link: unsafe agent_id slug")
        return None
    return f"{base}/agents/{agent_id}"


# Discord mention triggers a connector must neutralize even if a producer weakened
# the frame validator (renderer defense). ``@`` is broken with a
# zero-width space so ``@everyone``/``@here``/``<@id>`` can never ping; Markdown
# control chars are backslash-escaped so a crafted name cannot inject formatting or
# a ``[label](url)`` link. Belt-and-suspenders atop the frame validator, which
# already forbids these characters in ``display_name``.
_ZWSP = "​"
# Markdown/formatting control characters neutralized for the Discord surface.
_DISCORD_MD_SPECIALS = "*_`~|[]()<>#-"


def _neutralize_discord(name: str) -> str:
    """Neutralize mentions and Markdown control chars in a Discord display name.

    Breaks every ``@`` with a zero-width space (so ``@everyone``/``@here``/``<@id>``
    cannot ping) and backslash-escapes Markdown/formatting control chars (so a name
    cannot inject bold/italic/code or a ``[label](url)`` link). Belt-and-suspenders:
    the frame validator already forbids these chars in ``display_name``, so a
    well-formed name passes through unchanged; this only defends a future producer
    that weakened the validator.
    """

    out = name.replace("@", f"@{_ZWSP}")
    for ch in _DISCORD_MD_SPECIALS:
        out = out.replace(ch, f"\\{ch}")
    return out


def render_notification(
    frame: PresenceTransitionFrame, dash_base_url: str, *, surface: Surface
) -> str:
    """Render one SURFACE-NEUTRAL markdown notification body for ``surface``.

    Renders ONLY ``display_name`` + human observed time + Dash link; NEVER the raw
    ``agent_id`` as visible text, a NATS subject, or a raw payload. The body is
    markdown: the per-surface send path applies the surface's OWN safe rendering
    (Telegram escapes it into allowlisted-tag HTML; Discord renders the markdown
    natively). The ONE surface difference is the Dash link — Telegram renders a
    markdown ``[label](url)``; Discord (which does not render markdown links in an
    ordinary message) inlines the bare URL. An empty base URL renders no link.

    RENDERER DEFENSE (belt-and-suspenders atop the frame validator): the frame
    validator already constrains ``display_name`` to a safe vocabulary. For TELEGRAM
    the existing per-surface send path already escapes the whole markdown body once
    into allowlisted-tag HTML (``telegram_html.render_telegram_html`` /
    ``telegram_chunks``), so a ``<b>``/``<a>`` tag in the name can never ride through
    as raw markup — the name is passed through UN-pre-escaped so the single send-path
    escape is not double-applied. For DISCORD there is no such escaping send path, so
    the renderer neutralizes the name here: ``@`` is zero-width-space-broken and
    Markdown control chars are escaped so ``@everyone``/``<@id>``/``[x](url)`` are
    inert even if a future producer weakened the validator. A well-formed name passes
    through visually unchanged on both surfaces.
    """

    logger.debug(
        "chatd.presence_notify | render surface={} transition={}", surface, frame.transition
    )
    # Telegram: the send path escapes the body once; do NOT pre-escape (double-escape).
    # Discord: no escaping send path -> neutralize mentions/markdown defensively here.
    safe_name = (
        frame.display_name if surface == "telegram" else _neutralize_discord(frame.display_name)
    )
    headline = _HEADLINE[frame.transition].format(name=safe_name)
    lines = [f"**{headline}**", f"Observed: {_observed_line(frame.observed_at)}"]
    url = dash_link_url(dash_base_url, frame.agent_id)
    if url is not None:
        if surface == "telegram":
            lines.append(f"[View {safe_name} on Dash]({url})")
        else:
            lines.append(f"Dash live page: {url}")
    return "\n".join(lines)


# Bounded seen-transition-id cache size (delivery idempotency). A small
# deterministic FIFO ring is enough: the at-least-once bus redelivers the SAME edge
# only in a tight retry window, so a modest number of remembered ids covers every
# realistic in-flight duplicate while keeping the cache memory hard-bounded. It is
# delivery idempotency ONLY — the NATS typed frame stays truth — never a durable log.
MAX_SEEN_TRANSITION_IDS = 512


class _SeenTransitionCache:
    """Bounded deterministic FIFO cache of already-dispatched ``transition_id`` values.

    Delivery-idempotency only: an ambiguous at-least-once redelivery of
    the SAME edge carries the SAME typed ``transition_id`` (minted once by the deriver,
    retained on retry). :meth:`is_seen` PEEKS (no commit) and :meth:`commit` records the
    id only AFTER a successful connector delivery, so a failed send never suppresses a
    valid redelivery; the SAME edge is rendered/sent exactly once once it has delivered.
    Two DISTINCT edges carry distinct ids and both dispatch, in order. Eviction is
    deterministic oldest-first (insertion-
    ordered ``OrderedDict``), never time- or random-based, and the size is hard-capped
    so a churn of edges can never grow the cache without bound. This is NOT a database,
    Redis, or a durable receipt store — the NATS typed frame remains truth; this only
    de-dups delivery.
    """

    def __init__(self, *, capacity: int = MAX_SEEN_TRANSITION_IDS) -> None:
        """Create an empty cache bounded to ``capacity`` remembered ids."""

        self._capacity = capacity
        # Insertion-ordered so eviction is deterministic oldest-first. Value is unused
        # (a set with FIFO ordering); None keeps the key-only intent explicit.
        self._seen: OrderedDict[str, None] = OrderedDict()

    def is_seen(self, transition_id: str) -> bool:
        """Return True if ``transition_id`` was ALREADY committed (a duplicate).

        A pure PEEK — it does NOT commit. Committing is deferred to :meth:`commit`
        AFTER a successful connector delivery, so a failed send never suppresses a
        valid redelivery (Finding 3). A seen id is an ambiguous-ack
        redelivery of an edge already dispatched successfully; the caller drops it.
        """

        return transition_id in self._seen

    def commit(self, transition_id: str) -> None:
        """Record ``transition_id`` as dispatched, AFTER a successful delivery.

        Called ONLY once a connector send succeeded, so seen-status commits on
        delivery outcome — a failed send leaves the id uncommitted and eligible for
        retry. Re-committing an already-seen id is a no-op (it keeps its original FIFO
        position). Evicts the oldest id when at capacity: deterministic oldest-first,
        never random or time-based. Bounded — never a durable store.
        """

        if transition_id in self._seen:
            return
        self._seen[transition_id] = None
        if len(self._seen) > self._capacity:
            # Deterministic oldest-first eviction (FIFO), never random or time-based.
            self._seen.popitem(last=False)


class PresenceNotifyConsumer:
    """Decode presence-transition bytes and render+send an operator notification.

    A dumb-edge bus consumer: it decodes via :func:`rtech_frames.codec.decode_frame`
    and acts ONLY on a :class:`PresenceTransitionFrame`. Malformed bytes, an unknown
    frame kind, or a decoded frame of any OTHER type fail CLOSED — logged sanitized,
    dropped, never raised out of the subscription and never rendered as prose. It never
    logs the agent_id at human level, a subject, or a raw payload.

    Rendering is surface-neutral markdown handed to the injected per-surface send
    callbacks; each callback is the EXISTING chatd send path, so the surface applies
    its own safe HTML/formatting. Either send may be None on a single-surface deploy;
    a None surface is skipped, never a crash.

    DELIVERY IDEMPOTENCY: the deriver's at-least-once retry can
    re-publish the SAME edge under an ambiguous transport ack. Each edge carries a
    stable typed ``transition_id``; this consumer keys a bounded deterministic
    :class:`_SeenTransitionCache` on it so the SAME edge renders/sends exactly once
    while two DISTINCT edges (distinct ids) both dispatch, in order. The seen-id is
    committed ONLY after a successful connector delivery (Finding 3): a failed send
    leaves the id UNcommitted so a redelivery of that edge is retried, not suppressed —
    while a redelivery of an edge that DID deliver is still dropped. Each edge (a
    Telegram-origin and a Discord-origin edge subscribe with their OWN consumer
    instance, each with a single surface send wired) owns its own cache — surface
    coverage is complete because both service paths run this same consumer. The cache
    is delivery idempotency ONLY; the NATS typed frame stays truth.
    """

    def __init__(
        self,
        *,
        dash_base_url: str,
        send_telegram: NotifySend | None = None,
        send_discord: NotifySend | None = None,
    ) -> None:
        """Create the consumer with the Dash base URL and optional per-surface sends.

        The Dash base URL is validated/sanitized AT this construction boundary
        (:func:`validate_dash_base_url`, Finding 4): only an absolute http/https URL
        with no userinfo/query/fragment/control chars is stored (normalized trailing
        slash); anything else fails closed to ``""`` so link rendering is skipped
        entirely (still notify). The raw URL is never logged.
        """

        self._dash_base_url = validate_dash_base_url(dash_base_url)
        self._send_telegram = send_telegram
        self._send_discord = send_discord
        # Bounded deterministic seen-transition-id cache: same edge (same
        # transition_id) dispatches once even under at-least-once redelivery.
        self._seen = _SeenTransitionCache()

    async def handle(self, data: bytes) -> None:
        """Decode one frame's bytes and dispatch to the configured operator surfaces.

        Fails closed on malformed/unknown/other-type bytes (sanitized log + drop). A
        per-surface send failure is contained so one surface's error never blocks the
        other or crashes the subscription.
        """

        logger.debug("chatd.presence_notify.consume | entry byte_count={}", len(data))
        try:
            frame = decode_frame(data)
        except FrameDecodeError:
            # Malformed bytes / unknown frame_type: fail closed, drop, never crash.
            logger.warning("chatd.presence_notify.consume | dropped undecodable bytes")
            return
        if not isinstance(frame, PresenceTransitionFrame):
            # A well-formed frame of a DIFFERENT kind on this subject: not ours, drop.
            logger.warning(
                "chatd.presence_notify.consume | dropped non-presence frame type={}",
                type(frame).__name__,
            )
            return
        if self._seen.is_seen(frame.transition_id):
            # Ambiguous at-least-once redelivery of the SAME edge (same transition_id)
            # that ALREADY delivered successfully: drop the duplicate. Distinct edges
            # have distinct ids and are never suppressed. The transition_id is a
            # delivery key, never rendered or logged as identity.
            logger.debug(
                "chatd.presence_notify.consume | dropped duplicate transition (already sent)"
            )
            return
        logger.info(
            "chatd.presence_notify.consume | rendering transition={} name={}",
            frame.transition,
            frame.display_name,
        )
        # Commit the seen-id ONLY after a successful connector delivery (Finding 3): a
        # failed send leaves the id uncommitted so a redelivery of the SAME edge is
        # retried, not suppressed. A LATER duplicate that DID deliver is still dropped.
        if await self._dispatch(frame):
            self._seen.commit(frame.transition_id)
        else:
            logger.warning(
                "chatd.presence_notify.consume | delivery failed transition={} "
                "(retained for redelivery)",
                frame.transition,
            )

    async def _dispatch(self, frame: PresenceTransitionFrame) -> bool:
        """Render and send to each configured surface; return True iff a send succeeded.

        Per-surface send errors are contained (one surface's failure never blocks the
        other or crashes the subscription). The aggregate outcome is EXPLICIT: True
        when at least one configured surface delivered, False when NO configured
        surface delivered (so the caller withholds the seen-id commit and the edge
        stays eligible for at-least-once redelivery). A consumer wires exactly one
        surface, so its single-surface outcome is that surface's delivery result.
        """

        delivered = False
        if self._send_telegram is not None:
            delivered = (
                await self._send_safe(
                    "telegram",
                    self._send_telegram,
                    render_notification(frame, self._dash_base_url, surface="telegram"),
                )
                or delivered
            )
        if self._send_discord is not None:
            delivered = (
                await self._send_safe(
                    "discord",
                    self._send_discord,
                    render_notification(frame, self._dash_base_url, surface="discord"),
                )
                or delivered
            )
        return delivered

    async def _send_safe(self, surface: str, send: NotifySend, text: str) -> bool:
        """Send one rendered body; return True on success, False on contained error.

        A send failure is contained (logged sanitized, returns False, never raised) so
        one surface's error never blocks the other or crashes the subscription — and
        the explicit False lets the caller withhold the seen-id commit for retry.
        """

        try:
            await send(text)
        except Exception as exc:
            logger.error(
                "chatd.presence_notify.consume | send failed surface={} error_type={}",
                surface,
                type(exc).__name__,
            )
            return False
        return True


__all__ = [
    "MAX_SEEN_TRANSITION_IDS",
    "NotifySend",
    "PresenceNotifyConsumer",
    "Surface",
    "dash_link_url",
    "render_notification",
    "validate_dash_base_url",
]
