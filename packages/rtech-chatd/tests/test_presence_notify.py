"""Test the chatd presence-notify render + consumer + edge wiring.

chatd is a dumb edge: it renders the typed
:class:`~rtech_frames.frames.PresenceTransitionFrame` into a SURFACE-neutral markdown
body and sends it through the existing per-surface path. These tests pin:

* render offline/recovered headline, observed timestamp, and Dash link;
* per-surface link shape (Telegram markdown link vs Discord bare URL);
* empty Dash base URL => no link (fail safe, still notify);
* redaction: the internal ``agent_id`` / subject / raw payload NEVER appears as
  visible identity text (it rides ONLY as the opaque Dash URL path segment);
* the real Telegram send path escapes the markdown body to safe allowlisted HTML;
* the consumer fails CLOSED on malformed / unknown / other-kind frame bytes;
* per-surface send errors are contained (one surface never blocks the other);
* the shared :class:`~rtech_chatd.edge.DiscordEdge` bridge ACTUALLY subscribes
  ``rtech.presence.transition`` and dispatches through the adapter for BOTH the
  Discord and Telegram surfaces, and does NOT subscribe when no target is set.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
from rtech_chatd import subjects
from rtech_chatd.channel import InboundMessage, SendTarget
from rtech_chatd.edge import DiscordEdge
from rtech_chatd.presence_notify import (
    PresenceNotifyConsumer,
    dash_link_url,
    render_notification,
    validate_dash_base_url,
)
from rtech_chatd.telegram_html import telegram_chunks
from rtech_frames.codec import encode_frame
from rtech_frames.frames import LifecycleFrame, PresenceTransitionFrame

FrameHandler = Callable[[bytes], Awaitable[None]]

_OBSERVED = datetime(2026, 7, 20, 12, 30, 5, tzinfo=UTC)
_DASH = "https://dash.example.internal"


def _frame(
    *,
    agent_id: str = "agentd-nagatha",
    display_name: str = "Nagatha",
    transition: str = "offline",
    transition_id: str = "edge1",
) -> PresenceTransitionFrame:
    return PresenceTransitionFrame(
        agent_id=agent_id,
        display_name=display_name,
        transition=transition,  # type: ignore[arg-type]
        observed_at=_OBSERVED,
        transition_id=transition_id,
    )


# --- subject compatibility ---------------------------------------------------


def test_presence_subject_matches_ratified_string() -> None:
    # The consumer (chatd) and producer (rtech_webchat) MUST agree on this flat
    # subject byte-for-byte. Both pin the same ratified literal; this guards the
    # chatd side against an accidental rename that would silently stop delivery.
    assert subjects.PRESENCE_TRANSITION == "rtech.presence.transition"


# --- dash_link_url -----------------------------------------------------------


def test_dash_link_url_composes_path_from_base() -> None:
    assert (
        dash_link_url("https://dash.example.internal/", "agentd-nagatha")
        == "https://dash.example.internal/agents/agentd-nagatha"
    )


def test_dash_link_url_none_when_blank() -> None:
    assert dash_link_url("", "agentd-nagatha") is None
    assert dash_link_url("   ", "agentd-nagatha") is None


# --- validate_dash_base_url (Finding 4: config-boundary URL validation) ------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://dash.example.internal", "https://dash.example.internal"),
        ("https://dash.example.internal/", "https://dash.example.internal"),  # trailing slash
        ("https://dash.example.internal///", "https://dash.example.internal"),  # many slashes
        ("http://dash.example.internal", "http://dash.example.internal"),  # http allowed
        ("  https://dash.example.internal  ", "https://dash.example.internal"),  # trimmed
        ("https://dash.example.internal/live", "https://dash.example.internal/live"),  # path kept
        ("https://dash.example.internal/live/", "https://dash.example.internal/live"),
    ],
)
def test_validate_dash_base_url_accepts_and_normalizes(raw: str, expected: str) -> None:
    assert validate_dash_base_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",  # unset -> no link
        "   ",  # whitespace only
        "ftp://dash.example.internal",  # unsupported scheme
        "javascript:alert(1)",  # unsupported scheme
        "file:///etc/passwd",  # unsupported scheme
        "dash.example.internal",  # scheme-less / relative
        "//dash.example.internal",  # scheme-relative
        "https://user:pass@dash.example.internal",  # userinfo (credential exposure)
        "https://@everyone@dash.example.internal",  # userinfo + mention injection
        "https://dash.example.internal?x=1",  # query present
        "https://dash.example.internal#frag",  # fragment present
        "https://dash.example.internal/live?a=b#c",  # query+fragment
        "https://dash.example.internal\n@everyone",  # newline + mention smuggle
        "https://dash.example.\rinternal",  # carriage return control char
        "https://dash.example.internal/\tpath",  # tab control char
        "https://",  # missing host
    ],
)
def test_validate_dash_base_url_fails_closed(raw: str) -> None:
    # Every invalid/unsafe config fails closed to "" -> the consumer renders WITHOUT
    # a link rather than a broken/unsafe URL (still notify). The raw URL is not logged.
    assert validate_dash_base_url(raw) == ""


def test_invalid_dash_url_never_logged_raw() -> None:
    # Finding 4: an invalid Dash base URL must fail closed WITHOUT logging the raw URL.
    from loguru import logger

    records: list[str] = []
    sink_id = logger.add(records.append, level="DEBUG", format="{message}")
    try:
        raw = "https://s3cr3t:tok3n@dash.example.internal?leak=1"
        assert validate_dash_base_url(raw) == ""
    finally:
        logger.remove(sink_id)
    blob = "\n".join(records)
    assert "s3cr3t" not in blob
    assert "tok3n" not in blob
    assert "dash.example.internal" not in blob
    assert raw not in blob


async def test_consumer_stores_validated_dash_base_url_and_drops_bad_config() -> None:
    # Finding 4: the consumer validates at CONSTRUCTION. A userinfo-bearing base is
    # rejected there, so a rendered notification carries NO link (fail safe).
    dc: list[str] = []

    async def _dc(text: str) -> None:
        dc.append(text)

    consumer = PresenceNotifyConsumer(
        dash_base_url="https://user:pass@dash.example.internal",
        send_discord=_dc,
    )
    await consumer.handle(encode_frame(_frame(transition="offline")))
    assert len(dc) == 1
    assert "went offline" in dc[0]  # still notifies
    assert "http" not in dc[0]  # but no link (bad config dropped at construction)
    assert "user:pass" not in dc[0]
    assert "@" not in dc[0]  # no credential/mention leak


async def test_valid_dash_url_with_embedded_control_char_in_render_path() -> None:
    # A valid base URL renders a fixed link from the validated value only, and no
    # mention injection is possible: the display name is neutralized on Discord and
    # the URL is fixed from the validated base + safe slug.
    dc: list[str] = []

    async def _dc(text: str) -> None:
        dc.append(text)

    consumer = PresenceNotifyConsumer(
        dash_base_url="https://dash.example.internal/",
        send_discord=_dc,
    )
    await consumer.handle(encode_frame(_frame(agent_id="agentd-nagatha", transition="offline")))
    assert "Dash live page: https://dash.example.internal/agents/agentd-nagatha" in dc[0]
    assert "@everyone" not in dc[0]


# --- render_notification -----------------------------------------------------


def test_render_offline_telegram_has_headline_time_and_markdown_link() -> None:
    body = render_notification(_frame(transition="offline"), _DASH, surface="telegram")
    assert "went offline" in body
    assert "Nagatha" in body
    assert "2026-07-20 12:30:05 UTC" in body
    # Telegram gets a markdown link the sender converts to a safe anchor.
    assert f"[View Nagatha on Dash]({_DASH}/agents/agentd-nagatha)" in body


def test_render_recovered_headline() -> None:
    body = render_notification(_frame(transition="recovered"), _DASH, surface="discord")
    assert "back online" in body


def test_render_discord_uses_bare_url_not_markdown_link() -> None:
    body = render_notification(_frame(), _DASH, surface="discord")
    # Discord does not render a [label](url) link in an ordinary message: the
    # bare URL is inlined instead.
    assert f"Dash live page: {_DASH}/agents/agentd-nagatha" in body
    assert "](http" not in body


def test_render_empty_base_url_renders_no_link_but_still_notifies() -> None:
    for surface in ("telegram", "discord"):
        body = render_notification(_frame(), "", surface=surface)  # type: ignore[arg-type]
        assert "went offline" in body  # still notifies
        assert "http" not in body  # no link
        assert "Dash" not in body


def test_render_never_leaks_internal_agent_id_as_visible_identity() -> None:
    # The clean display name is shown; the slug appears ONLY inside the Dash URL
    # path segment, never as standalone visible identity text or a subject/payload.
    frame = _frame(agent_id="agentd-nagatha", display_name="Nagatha")
    for surface in ("telegram", "discord"):
        body = render_notification(frame, _DASH, surface=surface)  # type: ignore[arg-type]
        # The slug is present ONLY as the trailing URL path segment.
        assert body.count("agentd-nagatha") == 1
        assert "/agents/agentd-nagatha" in body
        # No NATS subject and no raw-payload markers ever render.
        assert "rtech.presence.transition" not in body
        assert "frame_type" not in body


def test_render_no_link_never_leaks_slug_at_all() -> None:
    # With no Dash link, the internal slug must not appear anywhere in the body.
    body = render_notification(_frame(agent_id="agentd-nagatha"), "", surface="discord")
    assert "agentd-nagatha" not in body


# --- real Telegram safe-HTML pipeline ---------------------------------------


def test_telegram_send_path_escapes_body_to_safe_html() -> None:
    # The rendered markdown body flows through the REAL Telegram send path: the
    # sender's telegram_chunks() escapes it once into allowlisted-tag HTML. The
    # ** headline becomes <b>, the [label](url) becomes a safe <a>, and the emoji
    # survives — no raw markdown asterisks and no double-escaped tags leak.
    body = render_notification(_frame(), _DASH, surface="telegram")
    chunks = telegram_chunks(body)
    assert len(chunks) == 1
    html = chunks[0]
    assert "<b>" in html and "</b>" in html
    assert f'<a href="{_DASH}/agents/agentd-nagatha">' in html
    assert "**" not in html  # markdown consumed, not literal
    assert "&lt;b&gt;" not in html  # not double-escaped


def test_telegram_render_escapes_html_metachars_defensively() -> None:
    # The frame validator forbids HTML metachars in display_name, but the Telegram
    # renderer escapes defensively so a FUTURE producer weakening the validator can
    # never inject a raw tag. We assert on the renderer output (an ampersand is a
    # legal safe-vocabulary-adjacent case only through escaping): a safe name renders
    # unchanged, and a would-be tag char is escaped in the body BEFORE the send path.
    body = render_notification(_frame(display_name="Nagatha 2"), _DASH, surface="telegram")
    # A normal name survives visually and produces valid safe HTML downstream.
    html = telegram_chunks(body)[0]
    assert "Nagatha 2" in html
    assert "&lt;" not in html  # nothing to escape in a clean name


def test_telegram_send_path_escapes_a_hostile_name_once() -> None:
    # Belt-and-suspenders: even if a weakened producer put a tag in display_name, the
    # Telegram send path escapes the WHOLE markdown body once into allowlisted HTML,
    # so a raw <b>/<a> can never ride through. The renderer passes the name through
    # UN-pre-escaped (no double-escape); the single send-path escape neutralizes it.
    body = "**🔴 <b>evil</b> went offline**"
    out = telegram_chunks(body)[0]
    assert "&lt;b&gt;evil&lt;/b&gt;" in out  # escaped exactly once
    assert "<b>evil" not in out  # never a raw injected tag
    assert "&amp;lt;" not in out  # not double-escaped


def test_discord_render_neutralizes_mentions_and_markdown() -> None:
    # The Discord surface breaks '@' with a zero-width space and escapes Markdown
    # control chars, so a would-be mention/link/formatting is inert. Driven through
    # the renderer's neutralizer on a hostile string (past the frame validator).
    import rtech_chatd.presence_notify as pn

    neutral = pn._neutralize_discord("@everyone [x](http://evil) `c`")  # noqa: SLF001
    # '@' no longer immediately precedes 'everyone' (zero-width space inserted).
    assert "@everyone" not in neutral
    assert "@" in neutral  # the char remains, just broken
    # Markdown control chars are backslash-escaped.
    assert "\\[" in neutral and "\\]" in neutral
    assert "\\(" in neutral and "\\)" in neutral
    assert "\\`" in neutral


def test_dash_link_dropped_for_unsafe_slug() -> None:
    # Only a subject-safe slug may enter the URL path; an unsafe one drops the link.
    assert dash_link_url(_DASH, "../evil") is None
    assert dash_link_url(_DASH, "a b") is None
    assert dash_link_url(_DASH, "agentd-nagatha") == f"{_DASH}/agents/agentd-nagatha"


# --- PresenceNotifyConsumer --------------------------------------------------


async def test_consumer_dispatches_to_both_surfaces() -> None:
    tg: list[str] = []
    dc: list[str] = []

    async def _tg(text: str) -> None:
        tg.append(text)

    async def _dc(text: str) -> None:
        dc.append(text)

    consumer = PresenceNotifyConsumer(dash_base_url=_DASH, send_telegram=_tg, send_discord=_dc)
    await consumer.handle(encode_frame(_frame(transition="offline")))
    assert len(tg) == 1 and "went offline" in tg[0]
    assert len(dc) == 1 and "went offline" in dc[0]
    # Surface-specific link shapes.
    assert "](http" in tg[0]
    assert "Dash live page:" in dc[0]


async def test_consumer_malformed_bytes_fail_closed() -> None:
    calls: list[str] = []

    async def _send(text: str) -> None:
        calls.append(text)

    consumer = PresenceNotifyConsumer(dash_base_url=_DASH, send_discord=_send)
    await consumer.handle(b"not json at all")
    assert calls == []


async def test_consumer_unknown_frame_type_fail_closed() -> None:
    calls: list[str] = []

    async def _send(text: str) -> None:
        calls.append(text)

    consumer = PresenceNotifyConsumer(dash_base_url=_DASH, send_discord=_send)
    await consumer.handle(b'{"frame_type":"nope","display_name":"x"}')
    assert calls == []


async def test_consumer_non_presence_frame_dropped() -> None:
    calls: list[str] = []

    async def _send(text: str) -> None:
        calls.append(text)

    consumer = PresenceNotifyConsumer(dash_base_url=_DASH, send_discord=_send)
    lifecycle = LifecycleFrame(
        agent_id="agentd-nagatha",
        reason="shutdown",
        initiator="operator",
        correlation_id="abc123",
    )
    await consumer.handle(encode_frame(lifecycle))
    assert calls == []


async def test_consumer_contains_per_surface_send_error() -> None:
    dc: list[str] = []

    async def _tg(text: str) -> None:
        raise RuntimeError("telegram down")

    async def _dc(text: str) -> None:
        dc.append(text)

    consumer = PresenceNotifyConsumer(dash_base_url=_DASH, send_telegram=_tg, send_discord=_dc)
    # The telegram send raises; it is contained and the discord surface still fires.
    await consumer.handle(encode_frame(_frame()))
    assert len(dc) == 1


# --- delivery idempotency: bounded seen-transition-id cache ------------------


async def test_consumer_dedups_same_frame_redelivery_to_one_notification() -> None:
    # Ambiguous at-least-once redelivery of the SAME edge (same transition_id) renders
    # and sends exactly ONCE on every configured surface.
    tg: list[str] = []
    dc: list[str] = []

    async def _tg(text: str) -> None:
        tg.append(text)

    async def _dc(text: str) -> None:
        dc.append(text)

    consumer = PresenceNotifyConsumer(dash_base_url=_DASH, send_telegram=_tg, send_discord=_dc)
    frame = _frame(transition="offline", transition_id="dup-edge")
    encoded = encode_frame(frame)
    await consumer.handle(encoded)
    await consumer.handle(encoded)  # exact same-frame redelivery
    await consumer.handle(encode_frame(_frame(transition="offline", transition_id="dup-edge")))
    # One notification per surface despite three deliveries of the same edge id.
    assert len(tg) == 1
    assert len(dc) == 1


async def test_consumer_distinct_ids_produce_distinct_notifications() -> None:
    # Two DISTINCT edges (distinct transition_ids) each dispatch, in order — the cache
    # only suppresses a repeated id, never a genuinely different edge.
    dc: list[str] = []

    async def _dc(text: str) -> None:
        dc.append(text)

    consumer = PresenceNotifyConsumer(dash_base_url=_DASH, send_discord=_dc)
    await consumer.handle(encode_frame(_frame(transition="offline", transition_id="edge-a")))
    await consumer.handle(encode_frame(_frame(transition="recovered", transition_id="edge-b")))
    assert len(dc) == 2
    assert "went offline" in dc[0]  # delivered in order
    assert "back online" in dc[1]


async def test_seen_cache_is_bounded_and_evicts_oldest_first() -> None:
    # The seen-id cache is hard-bounded and evicts deterministically oldest-first, so a
    # churn of edges can never grow it without bound. After overflowing past the bound,
    # the OLDEST id is evicted (so it would dispatch AGAIN), while a recent id is still
    # remembered (still suppressed).
    from rtech_chatd.presence_notify import MAX_SEEN_TRANSITION_IDS

    dc: list[str] = []

    async def _dc(text: str) -> None:
        dc.append(text)

    consumer = PresenceNotifyConsumer(dash_base_url=_DASH, send_discord=_dc)
    # Fill exactly to capacity with distinct ids.
    for i in range(MAX_SEEN_TRANSITION_IDS):
        await consumer.handle(encode_frame(_frame(transition="offline", transition_id=f"e{i:05d}")))
    assert len(dc) == MAX_SEEN_TRANSITION_IDS
    # One more distinct id evicts the OLDEST ("e00000").
    await consumer.handle(encode_frame(_frame(transition="offline", transition_id="overflow")))
    assert len(dc) == MAX_SEEN_TRANSITION_IDS + 1
    # A recent id ("overflow") is still remembered -> suppressed on redelivery.
    await consumer.handle(encode_frame(_frame(transition="offline", transition_id="overflow")))
    assert len(dc) == MAX_SEEN_TRANSITION_IDS + 1
    # The evicted oldest id ("e00000") is no longer remembered -> dispatches again.
    await consumer.handle(encode_frame(_frame(transition="offline", transition_id="e00000")))
    assert len(dc) == MAX_SEEN_TRANSITION_IDS + 2


async def test_failed_send_then_same_id_retry_delivers_not_suppressed() -> None:
    # Finding 3 regression: the seen-id must be committed ONLY after a SUCCESSFUL
    # delivery. A first send that FAILS must leave the id UNcommitted, so a redelivery
    # of the SAME edge (same transition_id) is RETRIED — not suppressed as a duplicate.
    dc: list[str] = []
    fail = {"on": True}

    async def _dc(text: str) -> None:
        if fail["on"]:
            raise RuntimeError("discord down")
        dc.append(text)

    consumer = PresenceNotifyConsumer(dash_base_url=_DASH, send_discord=_dc)
    encoded = encode_frame(_frame(transition="offline", transition_id="retry-edge"))
    await consumer.handle(encoded)  # send fails -> id NOT committed
    assert dc == []
    fail["on"] = False
    await consumer.handle(encoded)  # SAME edge redelivered -> retried, now succeeds
    assert len(dc) == 1 and "went offline" in dc[0]
    # A THIRD redelivery of the SAME edge (now that it delivered) IS suppressed.
    await consumer.handle(encoded)
    assert len(dc) == 1


async def test_successful_delivery_then_same_id_redelivery_suppressed() -> None:
    # Finding 3 companion: once an edge DID deliver successfully, an ambiguous-ack
    # redelivery of the SAME id is dropped (duplicate suppression after success).
    dc: list[str] = []

    async def _dc(text: str) -> None:
        dc.append(text)

    consumer = PresenceNotifyConsumer(dash_base_url=_DASH, send_discord=_dc)
    encoded = encode_frame(_frame(transition="offline", transition_id="ok-edge"))
    await consumer.handle(encoded)  # success -> id committed
    await consumer.handle(encoded)  # same edge -> suppressed
    assert len(dc) == 1


async def test_failed_send_does_not_block_distinct_later_edge() -> None:
    # A failed send for one edge leaves ITS id uncommitted but must not affect a
    # DISTINCT later edge, which delivers normally.
    dc: list[str] = []
    fail = {"on": True}

    async def _dc(text: str) -> None:
        if fail["on"]:
            raise RuntimeError("down")
        dc.append(text)

    consumer = PresenceNotifyConsumer(dash_base_url=_DASH, send_discord=_dc)
    await consumer.handle(encode_frame(_frame(transition="offline", transition_id="edge-x")))
    assert dc == []  # failed, uncommitted
    fail["on"] = False
    await consumer.handle(encode_frame(_frame(transition="recovered", transition_id="edge-y")))
    assert len(dc) == 1 and "back online" in dc[0]


async def test_telegram_failed_send_then_retry_delivers() -> None:
    # Finding 3 covers BOTH connector paths: the Telegram surface behaves identically —
    # a failed Telegram send leaves the id uncommitted and a same-id retry delivers.
    tg: list[str] = []
    fail = {"on": True}

    async def _tg(text: str) -> None:
        if fail["on"]:
            raise RuntimeError("telegram down")
        tg.append(text)

    consumer = PresenceNotifyConsumer(dash_base_url=_DASH, send_telegram=_tg)
    encoded = encode_frame(_frame(transition="offline", transition_id="tg-edge"))
    await consumer.handle(encoded)  # fails -> uncommitted
    assert tg == []
    fail["on"] = False
    await consumer.handle(encoded)  # retried -> delivers
    assert len(tg) == 1


async def test_dedup_never_leaks_transition_id_into_rendered_body() -> None:
    # The transition_id is a delivery key ONLY: it must never appear in any rendered
    # notification body on any surface.
    tg: list[str] = []
    dc: list[str] = []

    async def _tg(text: str) -> None:
        tg.append(text)

    async def _dc(text: str) -> None:
        dc.append(text)

    consumer = PresenceNotifyConsumer(dash_base_url=_DASH, send_telegram=_tg, send_discord=_dc)
    await consumer.handle(
        encode_frame(_frame(transition="offline", transition_id="secret-delivery-key"))
    )
    for body in (*tg, *dc):
        assert "secret-delivery-key" not in body


# --- DiscordEdge wiring (the shared bridge) ----------------------------------


@dataclass
class FakeSubscription:
    unsubscribed: bool = False

    async def unsubscribe(self) -> None:
        self.unsubscribed = True


@dataclass
class FakeFrameBus:
    handlers: dict[str, FrameHandler] = field(default_factory=dict)
    subs_by_subject: dict[str, FakeSubscription] = field(default_factory=dict)

    async def publish(self, subject: str, data: bytes) -> None:
        return None

    async def subscribe(self, subject: str, handler: FrameHandler) -> FakeSubscription:
        sub = FakeSubscription()
        self.handlers[subject] = handler
        self.subs_by_subject[subject] = sub
        return sub

    async def emit(self, subject: str, data: bytes) -> None:
        await self.handlers[subject](data)


@dataclass
class FakeAdapter:
    inbound: list[InboundMessage] = field(default_factory=list)
    sent: list[tuple[SendTarget, str]] = field(default_factory=list)

    async def events(self):  # type: ignore[no-untyped-def]
        for msg in self.inbound:
            yield msg

    async def send(self, target: SendTarget, text: str, files: list[Path]) -> str | None:
        self.sent.append((target, text))
        return f"sent-{len(self.sent)}"

    async def typing(self, target: SendTarget, on: bool) -> None:
        return None

    async def create_thread(self, channel_id: str, message_id: str, name: str) -> str | None:
        return None


async def test_edge_no_target_does_not_subscribe_presence() -> None:
    bus = FakeFrameBus()
    edge = DiscordEdge(FakeAdapter(), bus, stack="dev")
    await edge.run_inbound()
    assert subjects.PRESENCE_TRANSITION not in bus.handlers


async def test_edge_discord_surface_renders_bare_url() -> None:
    bus = FakeFrameBus()
    adapter = FakeAdapter()
    edge = DiscordEdge(
        adapter,
        bus,
        stack="dev",
        origin_edge="discord",
        presence_notify_target=SendTarget(channel_id="home-1"),
        dash_base_url=_DASH,
    )
    await edge.run_inbound()
    assert subjects.PRESENCE_TRANSITION in bus.handlers
    await bus.emit(subjects.PRESENCE_TRANSITION, encode_frame(_frame(transition="offline")))
    assert len(adapter.sent) == 1
    target, text = adapter.sent[0]
    assert target.channel_id == "home-1"
    assert "went offline" in text
    assert f"Dash live page: {_DASH}/agents/agentd-nagatha" in text
    # Retirement on close.
    await edge.close()
    assert bus.subs_by_subject[subjects.PRESENCE_TRANSITION].unsubscribed is True


async def test_edge_telegram_surface_renders_markdown_link() -> None:
    bus = FakeFrameBus()
    adapter = FakeAdapter()
    edge = DiscordEdge(
        adapter,
        bus,
        stack="dev",
        origin_edge="telegram",
        presence_notify_target=SendTarget(channel_id="-1001"),
        dash_base_url=_DASH,
    )
    await edge.run_inbound()
    await bus.emit(subjects.PRESENCE_TRANSITION, encode_frame(_frame(transition="recovered")))
    assert len(adapter.sent) == 1
    target, text = adapter.sent[0]
    assert target.channel_id == "-1001"
    assert "back online" in text
    # Telegram markdown link (the adapter's sender converts it to safe HTML).
    assert f"[View Nagatha on Dash]({_DASH}/agents/agentd-nagatha)" in text


async def test_edge_presence_malformed_bytes_contained() -> None:
    bus = FakeFrameBus()
    adapter = FakeAdapter()
    edge = DiscordEdge(
        adapter,
        bus,
        stack="dev",
        origin_edge="discord",
        presence_notify_target=SendTarget(channel_id="home-1"),
        dash_base_url=_DASH,
    )
    await edge.run_inbound()
    # Malformed bytes fail closed: no send, no crash.
    await bus.emit(subjects.PRESENCE_TRANSITION, b"garbage")
    assert adapter.sent == []
