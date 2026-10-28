"""Unit tests for the codex ``EngineSession`` adapter against a fake transport.

``FakeCodexTransport`` is a scripted JSON-RPC/stdio double: it records every
``request``/``notify`` the adapter makes, answers ``thread/start``/``turn/start``/
``turn/steer`` with canned results, and lets a test *push* ``ServerNotification``
messages into the adapter's handler to drive the event stream. No real ``codex``
binary runs — the suite is deterministic and offline.

The tests prove the seven verbs, handshake + threadId ref capture, resume,
send-idle vs send-mid-turn routing (``turn/steer`` with the correct
``expectedTurnId``), interrupt, notification→event mapping (including TurnDone
with a stance parsed from the ``outputSchema`` JSON, transient-error suppression,
and a terminal EngineError), and close-safety when never started.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from rtech_agentd.codex_engine import CodexEngine
from rtech_agentd.codex_protocol import CodexTransportError
from rtech_agentd.config import EngineConfig
from rtech_agentd.engine import (
    EngineError,
    EngineSession,
    TextDelta,
    ToolsPolicy,
    ToolUse,
    TurnDone,
)


class FakeCodexTransport:
    """A scripted stand-in for ``CodexAppServerTransport``.

    Mirrors the transport's public surface (``start``/``close``/``request``/
    ``notify``/``set_notification_handler``) so a ``CodexEngine`` can be driven
    with no real child process. Requests are recorded and answered from
    :attr:`responses` (keyed by method); tests push notifications via
    :meth:`emit`.
    """

    def __init__(self, *, model: str | None = None) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.notifies: list[tuple[str, dict[str, Any]]] = []
        self.started = False
        self.closed = False
        self._handler: Any = None
        self._stream_end_handler: Any = None
        self._turn_seq = 0
        # The model the app-server echoes at the response TOP LEVEL on
        # thread/start|resume (attestation). Default matches the
        # config the tests build so the happy path attests instead of failing
        # closed; a test that wants a mismatch/missing model overrides via
        # ``responses`` or passes ``model=None``.
        self.model = model
        # Per-method scripted results; None means "derive a default".
        self.responses: dict[str, Any] = {}
        # Methods that should raise CodexTransportError when called.
        self.fail_methods: set[str] = set()
        # When True, close() raises to simulate an owned-child reap failure; the
        # count of close() calls is tracked so a test can prove a retry ran.
        self.fail_close = False
        self.close_calls = 0

    def set_notification_handler(self, handler: Any) -> None:
        self._handler = handler

    def set_stream_end_handler(self, handler: Any) -> None:
        self._stream_end_handler = handler

    async def end_stream(self) -> None:
        """Simulate a child death / stream end: invoke the adapter's handler."""
        assert self._stream_end_handler is not None, "stream-end handler not registered"
        await self._stream_end_handler()

    async def start(self) -> dict[str, Any]:
        self.started = True
        return {"userAgent": "fake", "platformOs": "test"}

    async def close(self) -> None:
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError("scripted close failure")
        self.closed = True

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((method, params))
        if method in self.fail_methods:
            raise CodexTransportError(f"scripted failure for {method}")
        if method in self.responses:
            return self.responses[method]
        if method == "thread/start":
            result: dict[str, Any] = {"thread": {"id": "thread-1", "sessionId": "sess-1"}}
            if self.model is not None:
                result["model"] = self.model
            return result
        if method == "thread/resume":
            result = {"thread": {"id": params.get("threadId"), "sessionId": "sess-r"}}
            if self.model is not None:
                result["model"] = self.model
            return result
        if method == "turn/start":
            self._turn_seq += 1
            return {"turn": {"id": f"turn-{self._turn_seq}", "status": "inProgress"}}
        if method == "turn/steer":
            return {"turnId": params.get("expectedTurnId")}
        if method == "turn/interrupt":
            return {}
        return {}

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self.notifies.append((method, params))

    async def emit(self, method: str, params: dict[str, Any]) -> None:
        """Deliver a scripted ServerNotification to the adapter's handler."""
        assert self._handler is not None, "handler not registered yet"
        await self._handler({"method": method, "params": params})


# The model the default fixtures agree on: ``_cfg()`` requests it and a bare
# ``FakeCodexTransport()`` echoes it, so behavior tests that do not care about
# attestation take the happy (attested) path instead of failing closed.
_CFG_MODEL = "gpt-5.6-sol"


def _cfg() -> EngineConfig:
    return EngineConfig(kind="codex", model=_CFG_MODEL, litellm_base=None, backup_model=None)


def _fake() -> FakeCodexTransport:
    """A fake transport that echoes the default config model (attests by default)."""
    return FakeCodexTransport(model=_CFG_MODEL)


async def _drain_one(engine: CodexEngine) -> Any:
    """Return the next single event from the engine stream (with a timeout)."""
    agen = engine.events()
    try:
        return await asyncio.wait_for(agen.__anext__(), timeout=1.0)
    finally:
        await agen.aclose()


def test_codex_engine_satisfies_protocol() -> None:
    engine = CodexEngine(_cfg(), transport=FakeCodexTransport())
    assert isinstance(engine, EngineSession)


@pytest.mark.parametrize(
    "value",
    [
        "gpt-5.6-sol",
        "claude-sonnet-5",
        "gpt-5.5",
        "provider/model_name.v2",
    ],
)
def test_is_safe_model_id_accepts_clean_slugs(value: str) -> None:
    from rtech_agentd.codex_engine import is_safe_model_id

    assert is_safe_model_id(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "",  # empty
        "   ",  # whitespace only
        123,  # non-string
        {"nested": "obj"},  # non-string
        "a" * 100,  # oversized
        "gpt\n5.6-sol",  # embedded newline
        "gpt\x00sol",  # control byte
        "has spaces",  # internal whitespace
        " gpt-5.6-sol",  # leading whitespace — NOT stripped, fails closed
        "gpt-5.6-sol ",  # trailing whitespace
        "gpt-5.6-sol\n",  # trailing newline (would strip to a valid slug)
        "\tgpt-5.6-sol",  # leading tab
        "\ngpt-5.6-sol\n",  # surrounding newlines
    ],
)
def test_is_safe_model_id_rejects_padded_and_malformed(value: Any) -> None:
    # A raw model identifier with ANY leading/trailing whitespace or newline is
    # malformed and fails closed — the guard matches the RAW value exactly and
    # never strips-then-accepts (a padded value must not govern or log).
    from rtech_agentd.codex_engine import is_safe_model_id

    assert is_safe_model_id(value) is False


def test_engine_docs_agree_on_authoritative_codex_cli_version() -> None:
    # All engine documentation must agree with the authoritative codex-cli
    # version pinned in the protocol module. No stale 0.144.6 mention may
    # survive in the engine source docstrings.
    import inspect

    from rtech_agentd import codex_engine
    from rtech_agentd.codex_protocol import TESTED_CODEX_CLI_VERSION

    assert TESTED_CODEX_CLI_VERSION == "codex-cli 0.144.5"
    source = inspect.getsource(codex_engine)
    assert "0.144.6" not in source
    # And the authoritative version is actually referenced in the engine docs.
    assert "0.144.5" in source


async def test_start_handshake_and_ref_capture() -> None:
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("you are birk", ToolsPolicy())
    assert fake.started is True
    # thread/start was called with the system prompt + stance directive.
    methods = [m for m, _ in fake.requests]
    assert methods == ["thread/start"]
    _, params = fake.requests[0]
    assert params["baseInstructions"].startswith("you are birk")
    assert "output schema" in params["baseInstructions"].lower()
    health = await engine.health()
    assert health.session_ref == "thread-1"
    assert health.connected is True


async def test_start_sets_model_launch_params() -> None:
    # the configured model must ride on thread/start with provider fallback
    # forbidden, so a supported model actually governs turns instead of silently
    # falling to a retired default. ``allowProviderModelFallback`` is a real
    # thread/start field on the installed codex-cli 0.144.5 app-server binary.
    fake = FakeCodexTransport(model="gpt-5.6-sol")
    cfg = EngineConfig(kind="codex", model="gpt-5.6-sol", litellm_base=None, backup_model=None)
    engine = CodexEngine(cfg, transport=fake)
    await engine.start("sys", ToolsPolicy())
    _, params = fake.requests[0]
    assert params["model"] == "gpt-5.6-sol"
    assert params["allowProviderModelFallback"] is False


async def test_start_no_model_omits_launch_params() -> None:
    # An empty configured model leaves the launch params unset (deterministic:
    # no invented model, no fallback flag) rather than forcing a value.
    fake = FakeCodexTransport()
    cfg = EngineConfig(kind="codex", model="", litellm_base=None, backup_model=None)
    engine = CodexEngine(cfg, transport=fake)
    await engine.start("sys", ToolsPolicy())
    _, params = fake.requests[0]
    assert "model" not in params
    assert "allowProviderModelFallback" not in params


async def test_start_observed_model_match_is_accepted() -> None:
    # The app-server echoes the resolved model; when it matches the request the
    # session proceeds normally (thread ref captured).
    fake = FakeCodexTransport()
    fake.responses["thread/start"] = {
        "thread": {"id": "thread-1"},
        "model": "gpt-5.6-sol",
    }
    cfg = EngineConfig(kind="codex", model="gpt-5.6-sol", litellm_base=None, backup_model=None)
    engine = CodexEngine(cfg, transport=fake)
    await engine.start("sys", ToolsPolicy())
    health = await engine.health()
    assert health.session_ref == "thread-1"


async def test_start_observed_model_mismatch_fails_loud() -> None:
    # the server silently selected the retired tier. We FAIL CLOSED by
    # RAISING — and, unlike the old design, we do NOT enqueue an EngineError. The
    # attestation teardown swaps in a fresh queue so no stale terminal can later
    # be drained as a phantom event on a recovered fresh session.
    from rtech_agentd.codex_engine import CodexModelMismatchError

    fake = FakeCodexTransport()
    fake.responses["thread/start"] = {
        "thread": {"id": "thread-1"},
        "model": "gpt-5-codex",  # not what we asked for
    }
    cfg = EngineConfig(kind="codex", model="gpt-5.6-sol", litellm_base=None, backup_model=None)
    engine = CodexEngine(cfg, transport=fake)
    with pytest.raises(CodexModelMismatchError) as excinfo:
        await engine.start("sys", ToolsPolicy())
    # The message names both models (evidence) but carries no secret material.
    msg = str(excinfo.value)
    assert "gpt-5.6-sol" in msg and "gpt-5-codex" in msg


async def test_start_observed_model_nested_thread_fallback_is_attested() -> None:
    # Forward-compat: if a future app-server carries the resolved model on the
    # nested ``thread`` object instead of the top level, the nested value still
    # attests. (The verified 0.144.5 shape echoes it at the TOP level; this only
    # guards a defensive fallback and never regresses the top-level path.)
    fake = FakeCodexTransport()
    fake.responses["thread/start"] = {
        "thread": {"id": "thread-1", "model": "gpt-5.6-sol"},
        # no top-level "model"
    }
    cfg = EngineConfig(kind="codex", model="gpt-5.6-sol", litellm_base=None, backup_model=None)
    engine = CodexEngine(cfg, transport=fake)
    await engine.start("sys", ToolsPolicy())  # must not raise
    health = await engine.health()
    assert health.session_ref == "thread-1"


async def test_start_top_level_model_wins_over_nested_thread() -> None:
    # When BOTH a top-level and a nested thread.model are present, the verified
    # top-level field is authoritative — a nested drift never overrides it.
    fake = FakeCodexTransport()
    fake.responses["thread/start"] = {
        "thread": {"id": "thread-1", "model": "gpt-5-codex"},
        "model": "gpt-5.6-sol",
    }
    cfg = EngineConfig(kind="codex", model="gpt-5.6-sol", litellm_base=None, backup_model=None)
    engine = CodexEngine(cfg, transport=fake)
    await engine.start("sys", ToolsPolicy())  # top-level matches → must not raise
    health = await engine.health()
    assert health.session_ref == "thread-1"


async def test_start_missing_observed_model_fails_closed() -> None:
    # hardening: a NON-EMPTY requested model with NO echoed model can no
    # longer be confirmed, so we FAIL CLOSED (raise) rather than let a possibly
    # swapped session run. The default fake carries no top-level model here.
    from rtech_agentd.codex_engine import CodexModelMismatchError

    fake = FakeCodexTransport()  # model=None → no echoed model
    cfg = EngineConfig(kind="codex", model="gpt-5.6-sol", litellm_base=None, backup_model=None)
    engine = CodexEngine(cfg, transport=fake)
    with pytest.raises(CodexModelMismatchError) as excinfo:
        await engine.start("sys", ToolsPolicy())
    msg = str(excinfo.value)
    # The requested model is named; the type of the missing value is named, but
    # no raw response text is interpolated.
    assert "gpt-5.6-sol" in msg
    assert "NoneType" in msg


@pytest.mark.parametrize(
    "bad_model",
    [
        "",  # blank
        "   ",  # whitespace only
        123,  # non-string
        {"nested": "obj"},  # non-string
        "a" * 100,  # oversized (over the 64-char cap)
        "gpt\n5.6-sol",  # embedded newline (log-injection shape)
        "gpt\x00sol",  # embedded control byte
        " gpt-5.6-sol",  # leading whitespace — raw padded value must fail closed
        "gpt-5.6-sol ",  # trailing whitespace
        "gpt-5.6-sol\n",  # trailing newline (would strip to a valid slug)
        "\tgpt-5.6-sol",  # leading tab
    ],
)
async def test_start_unsafe_observed_model_fails_closed(bad_model: Any) -> None:
    # A missing/blank/non-string/oversized/newline/control observation is treated
    # as unattested and FAILS CLOSED — none of these may govern a session, and
    # none may be interpolated raw into a log/error.
    from rtech_agentd.codex_engine import CodexModelMismatchError

    fake = FakeCodexTransport()
    fake.responses["thread/start"] = {"thread": {"id": "thread-1"}, "model": bad_model}
    cfg = EngineConfig(kind="codex", model="gpt-5.6-sol", litellm_base=None, backup_model=None)
    engine = CodexEngine(cfg, transport=fake)
    with pytest.raises(CodexModelMismatchError) as excinfo:
        await engine.start("sys", ToolsPolicy())
    msg = str(excinfo.value)
    # The malformed/oversized/control-bearing raw value NEVER appears in the
    # message — only the requested model and the observed value's TYPE name do.
    assert "gpt-5.6-sol" in msg
    if isinstance(bad_model, str) and bad_model:
        assert bad_model not in msg


async def test_resume_observed_model_mismatch_fails_loud() -> None:
    # Resume echoes the resolved model too; a swap on resume must fail loud.
    from rtech_agentd.codex_engine import CodexModelMismatchError

    fake = FakeCodexTransport()
    fake.responses["thread/resume"] = {
        "thread": {"id": "thread-9"},
        "model": "gpt-5-codex",
    }
    cfg = EngineConfig(kind="codex", model="gpt-5.6-sol", litellm_base=None, backup_model=None)
    engine = CodexEngine(cfg, transport=fake)
    with pytest.raises(CodexModelMismatchError):
        await engine.resume("thread-9")


async def test_attestation_failure_tears_down_owned_transport() -> None:
    # An OWNED transport whose start() fails attestation MUST be closed so no
    # child process leaks. The engine owns the transport when none is injected —
    # here we assert the owned transport is torn down (its close() ran) and the
    # engine reports down, leaving no live child.
    from rtech_agentd.codex_engine import CodexEngine as _CE
    from rtech_agentd.codex_engine import CodexModelMismatchError

    owned = FakeCodexTransport()  # no echoed model → fails closed
    cfg = EngineConfig(kind="codex", model="gpt-5.6-sol", litellm_base=None, backup_model=None)
    engine = _CE(cfg, transport=owned)
    # Force ownership of the injected fake so we can observe teardown of an owned
    # transport (the real path builds its own; the fake mirrors that surface).
    engine._owns_transport = True
    with pytest.raises(CodexModelMismatchError):
        await engine.start("sys", ToolsPolicy())
    assert owned.closed is True  # owned child was reaped, not leaked
    health = await engine.health()
    assert health.connected is False
    assert health.detail == "closed"


async def test_owned_close_failure_retains_handle_and_retries() -> None:
    # teardown durability: if the OWNED transport's close() RAISES, the child
    # was not confirmed reaped. The engine must NOT drop the handle and must NOT
    # latch closed — a subsequent close() has to RETRY the reap. Here the first
    # close fails, the handle is retained, and a second close (now succeeding)
    # actually reaps the child.
    fake = _fake()
    fake.fail_close = True
    engine = CodexEngine(_cfg(), transport=fake)
    engine._owns_transport = True
    await engine.start("sys", ToolsPolicy())

    await engine.close()  # transport.close() raises → retryable, must not raise
    assert fake.close_calls == 1
    assert fake.closed is False  # child NOT confirmed reaped
    # Handle retained + not latched closed, so a retry is possible.
    assert engine._transport is fake
    assert engine._closed is False
    # Reader/pump ended and health reports down even though the reap is pending.
    health = await engine.health()
    assert health.connected is False

    # The reap now succeeds on retry — the child is finally reaped and dropped.
    fake.fail_close = False
    await engine.close()
    assert fake.close_calls == 2
    assert fake.closed is True
    assert engine._transport is None
    assert engine._closed is True


async def test_attestation_failure_survives_owned_close_error() -> None:
    # The original CodexModelMismatchError must propagate even when the owned
    # transport's close() raises during teardown — the close error is swallowed
    # (logged, retryable) and never masks the attestation failure. The handle is
    # retained so the child can still be reaped on a later close().
    from rtech_agentd.codex_engine import CodexModelMismatchError

    owned = FakeCodexTransport()  # no echoed model → attestation fails closed
    owned.fail_close = True  # and the teardown reap also fails
    cfg = EngineConfig(kind="codex", model="gpt-5.6-sol", litellm_base=None, backup_model=None)
    engine = CodexEngine(cfg, transport=owned)
    engine._owns_transport = True
    with pytest.raises(CodexModelMismatchError):
        await engine.start("sys", ToolsPolicy())
    # The reap was attempted, failed, and the handle is retained for a retry.
    assert owned.close_calls == 1
    assert owned.closed is False
    assert engine._transport is owned
    assert engine._closed is False


async def test_injected_close_resets_engine_local_state_without_closing() -> None:
    # An INJECTED (caller-owned) transport is NEVER closed by the engine, but a
    # close() must still reset the engine-local lifecycle, queue sentinel,
    # reader/pump, and health so a re-start on the same engine object begins
    # clean. The caller-owned transport's close() must not run.
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)  # owns_transport False (injected)
    await engine.start("sys", ToolsPolicy())
    await engine.send("go")
    await fake.emit("turn/started", {"threadId": "thread-1", "turn": {"id": "turn-1"}})

    collected: list[Any] = []

    async def consume() -> None:
        async for ev in engine.events():
            collected.append(ev)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    await engine.close()
    # Reader/pump ended (the sentinel terminated events()).
    await asyncio.wait_for(task, timeout=1.0)
    assert collected == []
    # The caller-owned transport was NEVER closed by the engine.
    assert fake.closed is False
    assert fake.close_calls == 0
    # Engine-local lifecycle reset: turn state cleared, health down, fresh queue.
    assert engine._active_turn_id is None
    assert engine._interrupting_turn_id is None
    health = await engine.health()
    assert health.connected is False
    assert health.detail == "closed"

    # A re-start on the same injected transport begins clean: the fresh queue has
    # no stale sentinel, so the rebuilt session's first event is a real delta and
    # events() does not end prematurely.
    await engine.start("sys", ToolsPolicy())
    await engine.send("again")
    await fake.emit(
        "item/agentMessage/delta",
        {"threadId": "thread-1", "turnId": "turn-2", "itemId": "i1", "delta": "fresh"},
    )
    event = await _drain_one(engine)
    assert isinstance(event, TextDelta)
    assert event.text == "fresh"


async def test_close_start_clears_partial_text_buffer() -> None:
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    await fake.emit(
        "item/agentMessage/delta",
        {"threadId": "thread-1", "turnId": "turn-old", "delta": "stale partial"},
    )

    await engine.close()
    await engine.start("sys", ToolsPolicy())
    await fake.emit(
        "turn/completed",
        {"threadId": "thread-1", "turn": {"id": "turn-new", "status": "completed"}},
    )

    event = await _drain_one(engine)
    assert isinstance(event, TurnDone)
    assert event.text == ""


async def test_close_start_allows_reused_turn_id() -> None:
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    completed = {
        "threadId": "thread-1",
        "turn": {
            "id": "turn-1",
            "status": "completed",
            "items": [{"type": "agentMessage", "id": "m1", "text": "first"}],
        },
    }
    await fake.emit("turn/completed", completed)
    first = await _drain_one(engine)
    assert isinstance(first, TurnDone)

    await engine.close()
    await engine.start("sys", ToolsPolicy())
    completed["turn"]["items"][0]["text"] = "reused"
    await fake.emit("turn/completed", completed)

    reused = await _drain_one(engine)
    assert isinstance(reused, TurnDone)
    assert reused.text == "reused"


async def test_resume_mismatch_then_fresh_start_has_clean_queue() -> None:
    # The supervisor reuses ONE engine object for resume→fresh-start fallback
    # (_resume_or_start): a failed resume is followed by a fresh start on the same
    # engine. After an attestation teardown the queue is swapped fresh, so the
    # rebuilt session's events() sees NO stale EngineError and NO stale
    # reader-stopped sentinel from the retired child — its first event is a real
    # turn event, and the stream does not end prematurely.
    from rtech_agentd.codex_engine import CodexEngine as _CE
    from rtech_agentd.codex_engine import CodexModelMismatchError

    fake = FakeCodexTransport(model="gpt-5.6-sol")
    # Resume returns a MISMATCHED model → fails closed + tears down.
    fake.responses["thread/resume"] = {"thread": {"id": "thread-9"}, "model": "gpt-5-codex"}
    cfg = EngineConfig(kind="codex", model="gpt-5.6-sol", litellm_base=None, backup_model=None)
    engine = _CE(cfg, transport=fake)
    engine._owns_transport = True  # owned so teardown drops + rebuilds

    with pytest.raises(CodexModelMismatchError):
        await engine.resume("thread-9")

    # The owned transport was dropped: a fresh start() rebuilds. Point the engine
    # at a NEW attesting fake so the rebuilt handshake attests cleanly.
    fresh_fake = FakeCodexTransport(model="gpt-5.6-sol")
    engine._transport = fresh_fake
    engine._owns_transport = False  # injected replacement, single-use
    await engine.start("sys", ToolsPolicy())  # must not raise

    # Drive one real turn on the rebuilt session and assert events() yields it —
    # proving the queue is fresh (no stale sentinel ended the stream, no phantom
    # EngineError from the retired child).
    await engine.send("go")
    await fresh_fake.emit(
        "item/agentMessage/delta",
        {"threadId": "thread-1", "turnId": "turn-1", "itemId": "i1", "delta": "alive"},
    )
    event = await _drain_one(engine)
    assert isinstance(event, TextDelta)
    assert event.text == "alive"


async def test_resume_replays_session_ref() -> None:
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.resume("thread-99")
    assert fake.requests[0] == ("thread/resume", {"threadId": "thread-99"})
    health = await engine.health()
    assert health.session_ref == "thread-99"


async def test_send_idle_routes_to_turn_start() -> None:
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    await engine.send("hello")
    turn_calls = [m for m, _ in fake.requests if m in ("turn/start", "turn/steer")]
    assert turn_calls == ["turn/start"]
    _, params = next(p for p in fake.requests if p[0] == "turn/start")
    assert params["input"] == [{"type": "text", "text": "hello"}]
    # outputSchema is attached for typed stance.
    assert params["outputSchema"]["properties"]["position"]["type"] == "string"


async def test_send_mid_turn_routes_to_steer_with_expected_turn_id() -> None:
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    await engine.send("start task")  # opens turn-1 (recorded eagerly)
    # A turn/started notification also confirms the active turn id.
    await fake.emit("turn/started", {"threadId": "thread-1", "turn": {"id": "turn-1"}})
    await engine.send("actually, do it differently")  # mid-turn → steer
    steer = next(p for p in fake.requests if p[0] == "turn/steer")
    assert steer[1]["expectedTurnId"] == "turn-1"
    assert steer[1]["input"] == [{"type": "text", "text": "actually, do it differently"}]


async def test_steer_precondition_failure_falls_through_to_turn_start() -> None:
    fake = _fake()
    fake.fail_methods.add("turn/steer")
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    await engine.send("start task")
    await fake.emit("turn/started", {"threadId": "thread-1", "turn": {"id": "turn-1"}})
    await engine.send("steer me")  # steer fails → fresh turn/start
    methods = [m for m, _ in fake.requests]
    assert "turn/steer" in methods
    # A second turn/start followed the failed steer.
    assert methods.count("turn/start") == 2


async def test_interrupt_no_op_when_idle() -> None:
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    await engine.interrupt()
    assert not any(m == "turn/interrupt" for m, _ in fake.requests)


async def test_interrupt_targets_active_turn() -> None:
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    await engine.send("go")
    await fake.emit("turn/started", {"threadId": "thread-1", "turn": {"id": "turn-1"}})
    await engine.interrupt()
    interrupt = next(p for p in fake.requests if p[0] == "turn/interrupt")
    assert interrupt[1] == {"threadId": "thread-1", "turnId": "turn-1"}


async def test_delta_maps_to_text_delta() -> None:
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    await fake.emit(
        "item/agentMessage/delta",
        {"threadId": "thread-1", "turnId": "turn-1", "itemId": "i1", "delta": "hi"},
    )
    event = await _drain_one(engine)
    assert isinstance(event, TextDelta)
    assert event.text == "hi"


async def test_reasoning_delta_is_not_surfaced() -> None:
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    # Reasoning first (dropped), then a real delta so the stream has one event.
    await fake.emit(
        "item/reasoning/textDelta",
        {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "itemId": "r1",
            "delta": "secret cot",
            "contentIndex": 0,
        },
    )
    await fake.emit(
        "item/agentMessage/delta",
        {"threadId": "thread-1", "turnId": "turn-1", "itemId": "i1", "delta": "visible"},
    )
    event = await _drain_one(engine)
    assert isinstance(event, TextDelta)
    assert event.text == "visible"


async def test_tool_started_maps_to_tool_use() -> None:
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    await fake.emit(
        "item/started",
        {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "startedAtMs": 0,
            "item": {"type": "commandExecution", "id": "cmd-1", "command": "ls"},
        },
    )
    event = await _drain_one(engine)
    assert isinstance(event, ToolUse)
    assert event.name == "commandExecution"
    assert event.tool_id == "cmd-1"
    assert event.tool_input == {"command": "ls"}


@pytest.mark.parametrize(
    "tool",
    [
        "mcp__rtech_stance__stance",
        "mcp__rtech_wake__wake_agent",
        "arbitrary_mcp_tool",
        "read_file",
        "web_fetch",
    ],
)
async def test_mcp_tool_call_uses_real_tool_name(tool: str) -> None:
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    await fake.emit(
        "item/started",
        {
            "item": {
                "type": "mcpToolCall",
                "id": "mcp-1",
                "tool": tool,
                "arguments": {"path": "packages/frames.py"},
            }
        },
    )

    event = await _drain_one(engine)
    assert isinstance(event, ToolUse)
    assert event.name == tool


async def test_mcp_tool_call_without_tool_name_fails_closed() -> None:
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    await fake.emit("item/started", {"item": {"type": "mcpToolCall", "id": "mcp-1"}})

    event = await _drain_one(engine)
    assert isinstance(event, ToolUse)
    assert event.name == "unknown_mcp_tool"


async def test_turn_completed_maps_to_turn_done_with_typed_stance() -> None:
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    final = json.dumps(
        {
            "position": "agree",
            "because": "the plan is sound",
            "needs_from": ["nimbus"],
            "verification": {
                "status": "grounded",
                "claim_kind": "factual",
                "ob_checked": True,
                "sources": [
                    {
                        "source_id": "OB1",
                        "kind": "open_brain",
                        "label": "task context",
                        "locator": "entity-1",
                        "authority": "memory",
                    }
                ],
                "limitations": [],
                "methods": ["open-brain"],
            },
            "reply": "Shipping it.",
        }
    )
    await fake.emit(
        "turn/completed",
        {
            "threadId": "thread-1",
            "turn": {
                "id": "turn-1",
                "status": "completed",
                "items": [{"type": "agentMessage", "id": "m1", "text": final}],
            },
        },
    )
    event = await _drain_one(engine)
    assert isinstance(event, TurnDone)
    assert event.is_error is False
    assert event.text == "Shipping it."
    assert event.session_ref == "thread-1"
    assert event.stance.position == "agree"
    assert event.stance.because == "the plan is sound"
    assert event.stance.needs_from == ["nimbus"]
    assert event.verification.status == "grounded"
    assert event.verification.sources[0].source_id == "OB1"
    # Active turn is cleared after completion.
    health = await engine.health()
    assert health.active_turn is False


async def test_final_text_from_item_completed_when_turn_items_empty() -> None:
    # codex-cli 0.144.5: turn/completed sends items=[] / itemsView="notLoaded",
    # so the final agentMessage text arrives ONLY via item/completed. The engine
    # must capture it from there (keyed by turnId) and use it for the TurnDone —
    # this is the empty-content regression.
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    final = json.dumps(
        {
            "position": "agree",
            "because": "the plan is sound",
            "needs_from": [],
            "reply": "Shipping it.",
        }
    )
    # 1. Final text lands on item/completed for turn-1.
    await fake.emit(
        "item/completed",
        {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "item": {"type": "agentMessage", "id": "m1", "text": final},
        },
    )
    # 2. turn/completed carries NO inline items (the 0.144.5 shape).
    await fake.emit(
        "turn/completed",
        {
            "threadId": "thread-1",
            "turn": {"id": "turn-1", "status": "completed", "items": [], "itemsView": "notLoaded"},
        },
    )
    event = await _drain_one(engine)
    assert isinstance(event, TurnDone)
    assert event.is_error is False
    assert event.text == "Shipping it."  # NOT empty — captured from item/completed
    assert event.stance.position == "agree"


async def test_item_completed_ignored_for_non_agent_message() -> None:
    # A non-agentMessage item/completed (e.g. a tool item) must NOT be captured as
    # the turn's final text; the turn then falls back cleanly (no false content).
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    await fake.emit(
        "item/completed",
        {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "item": {"type": "commandExecution", "id": "c1", "command": "ls"},
        },
    )
    await fake.emit(
        "turn/completed",
        {"threadId": "thread-1", "turn": {"id": "turn-1", "status": "completed", "items": []}},
    )
    event = await _drain_one(engine)
    assert isinstance(event, TurnDone)
    # No agentMessage text anywhere -> empty reply (neutral stance), never a
    # tool item masquerading as the answer.
    assert event.text == ""


async def test_turn_completed_malformed_stance_falls_back_neutral() -> None:
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    await fake.emit(
        "turn/completed",
        {
            "threadId": "thread-1",
            "turn": {
                "id": "turn-1",
                "status": "completed",
                "items": [{"type": "agentMessage", "id": "m1", "text": "not json at all"}],
            },
        },
    )
    event = await _drain_one(engine)
    assert isinstance(event, TurnDone)
    # Never scraped from prose: neutral default, original text as reply.
    assert event.stance.position == "responded"
    assert event.text == "not json at all"


async def test_turn_completed_malformed_needs_from_fails_safe_as_typed_error() -> None:
    # A malformed needs_from (non-string element, or — observed live 2026-07-17 —
    # full sentences) is a malformed STANCE-SHAPED response: it can neither be
    # coerced into collaboration needs (that would invent a false terminality
    # claim) nor kept as a substantive terminal turn with its position while
    # silently dropping the field. It fails SAFE through the typed engine-error
    # path (``is_error=True``) instead of publishing a normal substantive turn —
    # and the raw stance JSON NEVER becomes the trusted reply. The failed stance
    # is terminal (empty needs_from) so a single-scope task does not hang; the
    # frame-turn layer renders the human one-liner (raw payload stays in logs).
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    final = json.dumps(
        {
            "position": "agree",
            "because": "sound",
            "needs_from": [{"not": "a string"}],
            "reply": "ok",
        }
    )
    await fake.emit(
        "turn/completed",
        {
            "threadId": "thread-1",
            "turn": {
                "id": "turn-1",
                "status": "completed",
                "items": [{"type": "agentMessage", "id": "m1", "text": final}],
            },
        },
    )
    event = await _drain_one(engine)
    assert isinstance(event, TurnDone)
    # Typed error, not a normal substantive terminal turn.
    assert event.is_error is True
    # The unverified model position is NOT carried through as a substantive turn;
    # the stance is the terminal failed stance (empty needs_from → never hangs).
    assert event.stance.position == "failed"
    assert event.stance.needs_from == []


async def test_turn_completed_truncated_stance_blob_fails_safe_as_typed_error() -> None:
    # A stance-SHAPED blob that fails to JSON-parse (truncation) must never dump
    # the raw fragment into chat: it fails safe as a typed error turn. The blocked
    # payload is carried on the (error) TurnDone for the log-only failure path, so
    # the raw JSON is never the trusted reply and a distinct-frame render still
    # keys off a valid typed identity downstream.
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    truncated = '{"position":"x","because":"y","needs_from":[],"reply":"hi"'  # no closing brace
    await fake.emit(
        "turn/completed",
        {
            "threadId": "thread-1",
            "turn": {
                "id": "turn-1",
                "status": "completed",
                "items": [{"type": "agentMessage", "id": "m1", "text": truncated}],
            },
        },
    )
    event = await _drain_one(engine)
    assert isinstance(event, TurnDone)
    assert event.is_error is True
    assert event.stance.position == "failed"


async def test_turn_completed_stance_missing_reply_fails_safe_as_typed_error() -> None:
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    malformed = json.dumps(
        {
            "position": "ship",
            "because": "the boundary is sound",
            "needs_from": [],
        }
    )
    await fake.emit(
        "turn/completed",
        {
            "threadId": "thread-1",
            "turn": {
                "id": "turn-1",
                "status": "completed",
                "items": [{"type": "agentMessage", "id": "m1", "text": malformed}],
            },
        },
    )
    event = await _drain_one(engine)
    assert isinstance(event, TurnDone)
    assert event.is_error is True
    assert event.stance.position == "failed"
    assert event.stance.needs_from == []


async def test_turn_completed_plain_prose_ships_as_normal_turn() -> None:
    # The plain-prose fallback is preserved: a non-stance-shaped final message is a
    # normal (non-error) turn shipping the prose verbatim with the neutral stance.
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    await fake.emit(
        "turn/completed",
        {
            "threadId": "thread-1",
            "turn": {
                "id": "turn-1",
                "status": "completed",
                "items": [{"type": "agentMessage", "id": "m1", "text": "Just answering directly."}],
            },
        },
    )
    event = await _drain_one(engine)
    assert isinstance(event, TurnDone)
    assert event.is_error is False
    assert event.text == "Just answering directly."
    assert event.stance.position == "responded"


async def test_failed_turn_sets_is_error() -> None:
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    await fake.emit(
        "turn/completed",
        {"threadId": "thread-1", "turn": {"id": "turn-1", "status": "failed", "items": []}},
    )
    event = await _drain_one(engine)
    assert isinstance(event, TurnDone)
    assert event.is_error is True


async def test_transient_error_is_suppressed() -> None:
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    # willRetry=True must NOT terminate the turn: no EngineError, no TurnDone.
    await fake.emit(
        "error",
        {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "willRetry": True,
            "error": {"message": "rate limited, retrying"},
        },
    )
    # Follow with a real delta so we can assert the stream's FIRST event is it,
    # proving the transient produced nothing before it.
    await fake.emit(
        "item/agentMessage/delta",
        {"threadId": "thread-1", "turnId": "turn-1", "itemId": "i1", "delta": "ok"},
    )
    event = await _drain_one(engine)
    assert isinstance(event, TextDelta)
    assert event.text == "ok"


async def test_terminal_turn_error_maps_to_single_turn_done() -> None:
    # H2: a terminal error tied to a turn yields exactly ONE TurnDone(is_error),
    # FIFO-completing — never a bare EngineError that strands the task. A late
    # turn/completed for the same turn emits nothing (dedup).
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    await engine.send("go")
    await fake.emit("turn/started", {"threadId": "thread-1", "turn": {"id": "turn-1"}})
    await fake.emit(
        "error",
        {
            "threadId": "thread-1",
            "turnId": "turn-1",
            "willRetry": False,
            "error": {"message": "model unavailable"},
        },
    )
    # A late completion for the already-terminated turn must produce no event.
    await fake.emit(
        "turn/completed",
        {"threadId": "thread-1", "turn": {"id": "turn-1", "status": "completed", "items": []}},
    )
    collected: list[Any] = []

    async def consume() -> None:
        async for ev in engine.events():
            collected.append(ev)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    await engine.close()
    await asyncio.wait_for(task, timeout=1.0)
    terminals = [e for e in collected if isinstance(e, (TurnDone, EngineError))]
    assert len(terminals) == 1
    done = terminals[0]
    assert isinstance(done, TurnDone)
    assert done.is_error is True
    assert done.text == "model unavailable"
    assert done.session_ref == "thread-1"


async def test_turnless_error_maps_to_engine_error() -> None:
    # A transport error with no turn stays an EngineError (turn-less failure).
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    await fake.emit(
        "error",
        {"threadId": "thread-1", "willRetry": False, "error": {"message": "transport lost"}},
    )
    event = await _drain_one(engine)
    assert isinstance(event, EngineError)
    assert event.message == "transport lost"


async def test_stream_end_ends_events_with_no_pending_request() -> None:
    # H1: a child death mid-turn with NO pending RPC must end events() (so the
    # supervisor pump returns and recovery fires), exactly like ClaudeEngine.
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    await engine.send("go")
    await fake.emit("turn/started", {"threadId": "thread-1", "turn": {"id": "turn-1"}})
    collected: list[Any] = []

    async def consume() -> None:
        async for ev in engine.events():
            collected.append(ev)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    # Child dies with the turn still open and no pending request.
    await fake.end_stream()
    # events() must terminate (would hang forever before the fix).
    await asyncio.wait_for(task, timeout=1.0)


async def test_double_interrupt_sends_single_rpc() -> None:
    # M5: interrupt() is idempotent — a repeat for the same turn is a no-op.
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    await engine.send("go")
    await fake.emit("turn/started", {"threadId": "thread-1", "turn": {"id": "turn-1"}})
    await engine.interrupt()
    await engine.interrupt()
    interrupts = [m for m, _ in fake.requests if m == "turn/interrupt"]
    assert interrupts == ["turn/interrupt"]


async def test_rpc_failure_on_send_raises_and_emits_engine_error() -> None:
    fake = _fake()
    fake.fail_methods.add("turn/start")
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    with pytest.raises(CodexTransportError):
        await engine.send("boom")
    event = await _drain_one(engine)
    assert isinstance(event, EngineError)
    assert "send failed" in event.message


async def test_close_safe_when_never_started() -> None:
    # No start(): close must be a safe no-op teardown (transport is owned+lazy).
    engine = CodexEngine(_cfg())
    await engine.close()  # must not raise
    health = await engine.health()
    assert health.connected is False


async def test_close_ends_event_stream_and_closes_owned_transport() -> None:
    fake = _fake()
    engine = CodexEngine(_cfg(), transport=fake)
    await engine.start("sys", ToolsPolicy())
    collected: list[Any] = []

    async def consume() -> None:
        async for ev in engine.events():
            collected.append(ev)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    await engine.close()
    await asyncio.wait_for(task, timeout=1.0)
    # Injected (not owned) transports are NOT closed by the engine... but this
    # one is passed in, so ownership is False and we assert the engine still ends
    # the stream cleanly.
    assert collected == []
    assert fake.closed is False  # engine does not own an injected transport
