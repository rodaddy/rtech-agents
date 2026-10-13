"""``EngineSession`` over the ``codex app-server`` JSON-RPC protocol.

One long-lived ``codex app-server`` child per session (see
:mod:`rtech_agentd.codex_protocol` for the transport). This adapter layers the
managed-session state machine on top:

* :meth:`start` runs ``thread/start`` and captures the returned ``threadId`` as
  the ``session_ref``; :meth:`resume` runs ``thread/resume`` with a prior ref.
  The configured model is set on ``thread/start`` (the app-server honors it and
  every turn inherits it) with ``allowProviderModelFallback: false`` — a real
  thread/start field on the installed codex-cli 0.144.5 app-server binary — so a
  swap to a provider default fails loud rather than silently. Both responses echo
  the resolved model at the response top level (``ThreadStartResponse.model`` /
  ``ThreadResumeResponse.model``), which :meth:`_verify_observed_model` attests
  against the requested one, FAILING CLOSED with :class:`CodexModelMismatchError`
  on a missing / blank / malformed / mismatched observation (issue #132: a silent
  fall to the retired ``gpt-5-codex`` made every ChatGPT-account turn 400).
  :meth:`_attest_or_teardown` reaps the owned child on attestation failure so no
  orphan process leaks. Only VALIDATED model NAME strings are logged; no
  auth/session/token data.
* :meth:`send` decides start-vs-steer from the live turn: when a turn is active
  it calls ``turn/steer`` with the active ``expectedTurnId`` (native mid-turn
  steering); otherwise ``turn/start``. A steer precondition failure (the turn
  raced to completion) falls through to a fresh ``turn/start``.
* :meth:`interrupt` calls ``turn/interrupt`` with the active turn id (no-op-safe
  when idle).
* A single notification handler on the transport translates the
  ``ServerNotification`` stream into typed :class:`~rtech_agentd.engine.EngineEvent`s
  onto an internal queue that :meth:`events` yields from — mirroring the
  Claude engine's background-reader shape.

Typed stance rides on a per-turn ``outputSchema`` (see
:mod:`rtech_agentd.codex_stance`): the final ``agentMessage`` is strict JSON the
adapter parses into a :class:`~rtech_frames.Stance` — never scraped from prose.

WAKE (issue #93) is deliberately NOT wired here. The triage answer-or-wake
capability is exposed to the model through an in-process SDK MCP tool
(:mod:`rtech_agentd.wake_tool`), which is a Claude-Agent-SDK construct; the
codex app-server has no equivalent in this adapter yet. This is a conservative,
additive gap: a codex-backed triage deploy that sets ``RTECH_WAKE_ALLOWED_AGENTS``
FAILS LOUD at wiring time (``__main__._attach_wake`` raises
:class:`~rtech_agentd.engine_select.EngineSelectionError`) rather than silently
dropping the wake tool. Codex wake plumbing (an ``outputSchema``-style typed wake
call, or a codex tool) is future work.

Every transport call is wrapped; a failure becomes an
:class:`~rtech_agentd.engine.EngineError` event and is never swallowed. The codex
CLI version this adapter was verified against is logged at :meth:`start`
(:data:`~rtech_agentd.codex_protocol.TESTED_CODEX_CLI_VERSION`); the app-server
schema is experimental and can drift between releases.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from loguru import logger
from rtech_frames import Stance

from .codex_protocol import (
    TESTED_CODEX_CLI_VERSION,
    CodexAppServerTransport,
    CodexTransportError,
)
from .codex_stance import STANCE_DIRECTIVE, STANCE_OUTPUT_SCHEMA, parse_stance_result
from .engine import (
    EngineError,
    EngineEvent,
    EngineHealth,
    TextDelta,
    ToolsPolicy,
    ToolUse,
    TurnDone,
)

if TYPE_CHECKING:
    from .config import EngineConfig

# Item types (ThreadItem.type) surfaced as ToolUse observability events.
_TOOL_ITEM_TYPES = frozenset(
    {"commandExecution", "mcpToolCall", "dynamicToolCall", "webSearch", "fileChange"}
)

_UNKNOWN_MCP_TOOL_NAME = "unknown_mcp_tool"


# Sentinel pushed onto the queue when the session stops — ends events().
_READER_STOPPED: object = object()

# A model identifier is a short, printable slug: it MUST start with a letter or
# digit and then carry only letters/digits and ``.-_/`` — no whitespace, no
# control bytes. Examples: ``gpt-5.6-sol``, ``claude-sonnet-5``. We NEVER let
# arbitrary response text (or an untrusted configured value) ride into a
# log/exception as a model id: a drifted/hostile field could otherwise smuggle a
# huge blob, a newline (log injection), or control characters into our logs. A
# value that is not this shape is treated as malformed and fails closed on a
# nonempty request (issue #132).
#
# The match is EXACT on the RAW value — we do NOT strip first. A raw identifier
# with ANY leading/trailing whitespace or newline (``" gpt-5.6-sol"``,
# ``"gpt-5.6-sol\n"``) is malformed and fails closed: stripping-then-validating
# would silently accept a padded/log-injecting value and let it govern (or be
# logged), which is exactly the untrusted shape this guard exists to reject.
# Anchored with ``\A``/``\Z`` (absolute string bounds), NOT ``^``/``$``: in
# Python ``$`` also matches just BEFORE a trailing ``\n``, so ``"gpt-5.6-sol\n"``
# would slip through a ``$``-anchored pattern and defeat the fail-closed guard.
# ``\Z`` matches only the very end of the string, so a trailing newline fails.
_MODEL_ID_MAX_LEN = 64
_MODEL_ID_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._/\-]*\Z")


def is_safe_model_id(value: object) -> bool:
    """True when ``value`` is a short, safe model-identifier string.

    Guards what may be interpolated into a log/error as a model id — both an
    observed model from the app-server and the configured model at the selection
    boundary (:func:`rtech_agentd.engine_select._resolve_codex_model`). Requires a
    non-empty string within the length cap that matches the model-slug charset
    EXACTLY, with no leading/trailing whitespace: no whitespace, newlines,
    control bytes, or multi-KB blobs ever reach a log line, and a padded raw
    value (``" gpt-5.6-sol"``, ``"gpt-5.6-sol\\n"``) fails closed rather than
    being silently trimmed into an accepted slug.
    """
    if not isinstance(value, str):
        return False
    if not value or len(value) > _MODEL_ID_MAX_LEN:
        return False
    return _MODEL_ID_RE.match(value) is not None


def _observed_model(result: dict[str, Any]) -> Any:
    """Extract the app-server's resolved model from a thread/start|resume result.

    Attests against the response shape of the installed ``codex-cli 0.144.5``
    app-server (``ThreadStartResponse`` / ``ThreadResumeResponse``, verified via
    the ``test_start_observed_model_*`` engine tests): the resolved model is
    echoed at the response TOP LEVEL (``result["model"]``), not inside the nested
    ``thread`` object (whose model-facing field is ``modelProvider``). We read the
    top-level field first and fall back to a nested ``thread.model`` only as a
    defensive, forward-compatible variant (absent in the verified shape, so it is
    a no-op there and cannot regress attestation). The RAW value is returned
    unvalidated; the caller decides match/malformed via :func:`is_safe_model_id`.
    """
    top = result.get("model")
    if top is not None:
        return top
    thread = result.get("thread")
    if isinstance(thread, dict):
        return thread.get("model")
    return None


class CodexModelMismatchError(Exception):
    """The app-server selected a model different from the one we requested.

    ``thread/start``/``thread/resume`` echo the model the app-server actually
    resolved (``ThreadStartResponse.model``). When a configured, supported model
    is silently swapped for a provider default (the issue #132 failure mode:
    ``gpt-5.6-sol`` requested, retired ``gpt-5-codex`` selected → every turn
    400s), we FAIL LOUD here instead of letting broken turns run. Only model
    NAME strings ride in the message — never auth/session/token data.
    """


class CodexEngine:
    """A managed ``codex app-server`` session implementing ``EngineSession``.

    Args:
        engine_config: Engine kind/model/litellm_base/backup from the agent
            config (only model/base/backup are modeled, matching ClaudeEngine).
        env: Extra environment for the codex child process. Never logged.
        transport: Injectable transport (the fake double in tests). When absent a
            real :class:`CodexAppServerTransport` is built at :meth:`start`.
    """

    def __init__(
        self,
        engine_config: EngineConfig,
        *,
        env: dict[str, str] | None = None,
        transport: CodexAppServerTransport | None = None,
    ) -> None:
        self._config = engine_config
        self._extra_env = dict(env or {})
        self._transport = transport
        self._owns_transport = transport is None
        self._queue: asyncio.Queue[EngineEvent | object] = asyncio.Queue()
        self._thread_id: str | None = None
        # The live turn id, guarded by a lock so the start-vs-steer decision and
        # the interrupt precondition read a consistent value.
        self._active_turn_id: str | None = None
        self._turn_lock = asyncio.Lock()
        # Per-turn buffer of assistant-visible agentMessage text deltas (in order).
        self._text_buf: list[str] = []
        # Final agentMessage text captured from ``item/completed`` events, keyed by
        # turn id. As of codex-cli 0.144.5 the ``turn/completed`` payload no longer
        # inlines items (``items: []``, ``itemsView: "notLoaded"``), so the final
        # text lives ONLY in the per-item ``item/completed`` notification. This is
        # the authoritative source and is race-proof across the steer->fresh-turn
        # fallback (unlike the session-wide delta buffer, which desyncs when a turn
        # restarts). Bounded the same way as ``_terminated_turns``.
        self._final_item_text: dict[str, str] = {}
        # Turn ids that already emitted a terminal event (TurnDone). A late
        # duplicate terminal (e.g. turn/completed after an error) is logged +
        # dropped so a consumer never sees two terminals for one turn.
        self._terminated_turns: set[str] = set()
        # The turn id an interrupt RPC is in flight / completed for, so a repeat
        # interrupt() is a no-op until the turn actually terminates (idempotent).
        self._interrupting_turn_id: str | None = None
        self._started = False
        self._closed = False

    # --- transport construction ----------------------------------------------

    def _build_transport(self) -> CodexAppServerTransport:
        """Build the real app-server transport, honoring litellm_base via env."""
        env = dict(self._extra_env)
        # A LiteLLM-compatible base is passed to the child; codex reads
        # OPENAI_BASE_URL for OpenAI-compatible routing. Only set when absent.
        if self._config.litellm_base and "OPENAI_BASE_URL" not in env:
            env["OPENAI_BASE_URL"] = self._config.litellm_base
        return CodexAppServerTransport(env=env or None)

    # --- lifecycle ------------------------------------------------------------

    async def start(self, system_prompt: str, tools_policy: ToolsPolicy) -> None:
        """Spawn/handshake the child and open a fresh thread (``thread/start``)."""
        logger.info(
            "codex_engine.start | model={} tested_cli={} allow={} deny={}",
            self._config.model,
            TESTED_CODEX_CLI_VERSION,
            list(tools_policy.allow),
            list(tools_policy.deny),
        )
        await self._ensure_started()
        params: dict[str, Any] = {
            "baseInstructions": system_prompt + STANCE_DIRECTIVE,
        }
        if self._config.model:
            # Set the model on thread/start (the app-server honors it and every
            # turn inherits it — verified against codex-cli 0.144.5). Forbid the
            # provider from silently substituting a default for an unavailable
            # requested model: a swap must FAIL LOUD (issue #132), not 400 later.
            params["model"] = self._config.model
            params["allowProviderModelFallback"] = False
        try:
            result = await self._require_transport().request("thread/start", params)
        except CodexTransportError as exc:
            logger.error("codex_engine.start | thread/start failed: {}", exc)
            await self._queue.put(EngineError(message=f"start failed: {exc}"))
            raise
        await self._attest_or_teardown(result, phase="start")
        self._capture_thread(result.get("thread"))

    async def resume(self, session_ref: str) -> None:
        """Reopen a prior thread by ``session_ref`` (``thread/resume``)."""
        if not session_ref:
            raise ValueError("codex_engine.resume: session_ref must be non-empty")
        logger.info("codex_engine.resume | session_ref={}", session_ref)
        await self._ensure_started()
        try:
            result = await self._require_transport().request(
                "thread/resume", {"threadId": session_ref}
            )
        except CodexTransportError as exc:
            logger.error("codex_engine.resume | thread/resume failed: {}", exc)
            await self._queue.put(EngineError(message=f"resume failed: {exc}"))
            raise
        await self._attest_or_teardown(result, phase="resume")
        self._thread_id = session_ref
        self._capture_thread(result.get("thread"))

    async def _ensure_started(self) -> None:
        """Build (if owned) and handshake the transport, wiring the handler."""
        if self._started:
            return
        if self._transport is None:
            self._transport = self._build_transport()
        self._transport.set_notification_handler(self._on_notification)
        # Stream-end (child death / stream error) must end events() even with no
        # pending RPC — mirror ClaudeEngine's reader-stopped guarantee (H1).
        self._transport.set_stream_end_handler(self._on_stream_end)
        try:
            await self._transport.start()
        except CodexTransportError as exc:
            logger.error("codex_engine.start | transport handshake failed: {}", exc)
            await self._queue.put(EngineError(message=f"handshake failed: {exc}"))
            raise
        self._started = True
        self._closed = False

    def _verify_observed_model(self, result: dict[str, Any], *, phase: str) -> None:
        """Fail closed unless the app-server confirms the model we requested.

        ``thread/start``/``thread/resume`` echo the resolved model at the response
        top level (``result["model"]`` — the ``ThreadStartResponse`` /
        ``ThreadResumeResponse`` shape of the installed codex-cli 0.144.5
        app-server; see the ``test_start_observed_model_*`` engine tests). When a
        NON-EMPTY model was requested, the observation must be a safe model
        identifier that matches it; a missing, blank, malformed, or mismatched
        observation FAILS CLOSED with :class:`CodexModelMismatchError` rather than
        letting a possibly-swapped session run (issue #132: a silent fall to the
        retired ``gpt-5-codex`` 400s every ChatGPT-account turn).

        No consumer drains ``events()`` before ``start``/``resume`` returns (the
        supervisor pump starts only after), so the failure signal is the RAISED
        exception ONLY — never a queued ``EngineError`` (a stale terminal left on
        the queue would later be drained as a phantom event and wrongly mark a
        recovered fresh session active). Deterministic teardown of the owned child
        is the caller's job (:meth:`start`/:meth:`resume`).

        Logs only VALIDATED model NAME strings — never raw response text.
        """
        requested = self._config.model
        if not requested:
            # Empty configured model: enforcement is disabled by construction. The
            # config/selection boundary rejects an empty codex model before we get
            # here, so this only guards non-codex reuse — keep it a no-op.
            return
        observed = _observed_model(result)
        if not is_safe_model_id(observed):
            # Missing / blank / malformed / whitespace-padded (or unsafe-to-log):
            # fail closed. Do NOT interpolate the raw value — only its type is
            # safe to name.
            message = (
                f"codex app-server did not echo a valid model at thread/{phase} "
                f"(got {type(observed).__name__}); {requested!r} was requested and "
                f"cannot be confirmed (fail closed, issue #132)"
            )
            logger.error("codex_engine.{} | model unattested | {}", phase, message)
            raise CodexModelMismatchError(message)
        # ``observed`` is Any (from _observed_model) and is_safe_model_id() has
        # confirmed it is a nonempty, whitespace-free str; compare it EXACTLY
        # (no strip — a padded observation already failed closed above).
        observed_id = observed
        if observed_id == requested:
            logger.info(
                "codex_engine.{} | model confirmed | requested={} observed={}",
                phase,
                requested,
                observed_id,
            )
            return
        message = (
            f"codex app-server selected model {observed_id!r} but {requested!r} was "
            f"requested at thread/{phase}; the configured model is not governing "
            f"this session (fail closed, issue #132)"
        )
        logger.error("codex_engine.{} | model mismatch | {}", phase, message)
        raise CodexModelMismatchError(message)

    async def _attest_or_teardown(self, result: dict[str, Any], *, phase: str) -> None:
        """Attest the observed model; tear the owned child down on failure.

        A start/resume that reaches attestation has already spawned the child +
        reader task + transport (via :meth:`_ensure_started`). If attestation
        fails we own that child and MUST NOT leak it: close the transport
        deterministically (child process, reader task, pending futures) before
        re-raising, so a fresh rotation start that fails attestation leaves no
        orphan process. :meth:`close` is idempotent and only closes an owned
        transport — an injected (test) transport is left to its owner.
        """
        try:
            self._verify_observed_model(result, phase=phase)
        except CodexModelMismatchError:
            logger.error("codex_engine.{} | attestation failed; tearing down owned child", phase)
            await self.close()
            raise

    def _capture_thread(self, thread: Any) -> None:
        """Capture ``threadId`` from a thread/start|resume result as session_ref."""
        if isinstance(thread, dict):
            tid = thread.get("id")
            if isinstance(tid, str) and tid:
                self._thread_id = tid
                logger.debug("codex_engine | captured session_ref={}", tid)

    def _reset_engine_local_state(self) -> None:
        """Reset engine-local lifecycle/turn/queue state after a teardown.

        Ends the reader/pump for the retired session and marks health down
        WITHOUT touching the transport handle (ownership decides that). Resets
        the live turn state tied to the retired child (the durable thread ref is
        left for health()/resume(); a rebuilt start() overwrites it) and swaps in
        a FRESH queue so the sentinel (and any stale events) from the retired
        child never bleed into a rebuilt session's events() — a rebuilt start()
        must not have its first events() read end the stream. Runs for BOTH the
        owned and the injected (caller-owned) path: an injected transport is
        never closed here, but the engine-local lifecycle, queue sentinel,
        reader/pump, and health all reset the same way so a re-``start`` on the
        same engine object begins clean.
        """
        self._started = False
        self._active_turn_id = None
        self._interrupting_turn_id = None
        self._text_buf.clear()
        self._terminated_turns.clear()
        self._queue = asyncio.Queue()

    async def close(self) -> None:
        """Tear down the transport (if owned) and end ``events()``. Idempotent.

        Safe to call when never started — the transport guards its own child
        handle, so ``close`` before ``start`` is a no-op teardown.

        An OWNED transport is dropped after a SUCCESSFUL close so a subsequent
        ``start`` / ``resume`` on the same engine rebuilds a fresh child (the
        supervisor reuses one engine object for the resume→fresh-start fallback,
        and an attestation teardown must not leave the engine permanently
        unusable). If the owned transport's ``close`` RAISES, the child was not
        confirmed reaped: we RETAIN the handle and do NOT latch ``_closed``, so a
        subsequent ``close`` retries the reap. The teardown does not itself raise
        (an attestation caller's original :class:`CodexModelMismatchError`
        propagates independently and is never masked by a close error). An
        INJECTED (caller-owned) transport is NEVER closed by the engine — its
        owner reaps it — but the engine-local lifecycle, queue sentinel,
        reader/pump, and health still reset so a re-``start`` begins clean.
        """
        if self._closed:
            return
        logger.info("codex_engine.close | tearing down session_ref={}", self._thread_id)
        # End the current events() stream: any consumer blocked on the live queue
        # gets the reader-stopped sentinel.
        await self._queue.put(_READER_STOPPED)
        if self._transport is not None and self._owns_transport:
            try:
                await self._transport.close()
            except Exception as exc:  # noqa: BLE001 - close must not raise
                # The owned child was NOT confirmed reaped. Retain the handle and
                # leave ``_closed`` unlatched so a subsequent close() retries the
                # reap; still end the reader/pump and reset engine-local state so
                # health reports down and no stale turn state survives. The caller
                # that triggered an attestation teardown re-raises its own
                # CodexModelMismatchError, which this swallow must not mask.
                logger.warning(
                    "codex_engine.close | owned transport close error (retryable): {}", exc
                )
                self._reset_engine_local_state()
                return
            # Drop the reaped owned child so the next start()/resume() rebuilds a
            # fresh handshake instead of reusing it.
            self._transport = None
        # Owned-and-reaped, or injected (never closed here): reset engine-local
        # lifecycle/queue/turn state and latch closed.
        self._reset_engine_local_state()
        self._closed = True

    async def _on_stream_end(self) -> None:
        """Transport reader stopped (child death): end ``events()`` like Claude.

        With no pending RPC a turn-less death would otherwise leave ``events()``
        blocked on the queue forever (supervisor pump never returns, recovery
        never fires). Enqueue the reader-stopped sentinel so the stream
        terminates exactly like :class:`ClaudeEngine`. Idempotent with
        :meth:`close` — the sentinel simply ends the stream on first receipt.
        """
        if self._closed:
            return
        logger.warning(
            "codex_engine.stream_end | transport stream ended, ending events() session_ref={}",
            self._thread_id,
        )
        await self._queue.put(_READER_STOPPED)

    # --- turn I/O -------------------------------------------------------------

    async def send(self, text: str) -> None:
        """Send a user turn — or steer inline if a turn is already active.

        Start-vs-steer is decided under the turn lock: an active turn id means
        ``turn/steer`` with that ``expectedTurnId`` (native mid-turn steer);
        otherwise ``turn/start``. A steer precondition failure (turn raced to
        completion) falls through to a fresh ``turn/start``.

        Concurrency: callers serialize ``send`` per session (frame_turn drives
        one send per FIFO); this method does not itself serialize concurrent
        sends — same property as :class:`ClaudeEngine`, out of the engine
        contract.
        """
        transport = self._require_transport()
        async with self._turn_lock:
            active = self._active_turn_id
        user_input = [{"type": "text", "text": text}]
        base_params: dict[str, Any] = {
            "threadId": self._thread_id,
            "input": user_input,
            "outputSchema": STANCE_OUTPUT_SCHEMA,
        }
        if active is not None:
            logger.info(
                "codex_engine.send | mode=steer expectedTurnId={} len={}", active, len(text)
            )
            try:
                await transport.request(
                    "turn/steer",
                    {
                        "threadId": self._thread_id,
                        "expectedTurnId": active,
                        "input": user_input,
                    },
                )
                return
            except CodexTransportError as exc:
                # Precondition race: the turn completed between our read and the
                # call. Fall through to a fresh turn/start (memo gap 2).
                logger.warning("codex_engine.send | steer failed ({}); starting fresh turn", exc)
        logger.info("codex_engine.send | mode=turn len={}", len(text))
        try:
            result = await transport.request("turn/start", base_params)
        except CodexTransportError as exc:
            logger.error("codex_engine.send | turn/start failed: {}", exc)
            await self._queue.put(EngineError(message=f"send failed: {exc}"))
            raise
        # Record the started turn id eagerly so a fast follow-up send steers it,
        # even before the turn/started notification lands.
        turn = result.get("turn")
        if isinstance(turn, dict):
            tid = turn.get("id")
            if isinstance(tid, str) and tid:
                async with self._turn_lock:
                    self._active_turn_id = tid

    async def interrupt(self) -> None:
        """Abort the active turn via ``turn/interrupt`` (no-op-safe when idle)."""
        transport = self._require_transport()
        async with self._turn_lock:
            active = self._active_turn_id
            if active is None:
                logger.info("codex_engine.interrupt | no active turn; no-op")
                return
            # Idempotent: a repeat interrupt for the same turn must not re-send
            # the RPC. The flag clears only when the turn actually terminates.
            if self._interrupting_turn_id == active:
                logger.info(
                    "codex_engine.interrupt | already interrupting turnId={}; no-op", active
                )
                return
            self._interrupting_turn_id = active
        logger.info("codex_engine.interrupt | turnId={}", active)
        try:
            await transport.request(
                "turn/interrupt", {"threadId": self._thread_id, "turnId": active}
            )
        except CodexTransportError as exc:
            logger.error("codex_engine.interrupt | failed: {}", exc)
            # Failed RPC: allow a retry rather than latching the no-op flag.
            async with self._turn_lock:
                if self._interrupting_turn_id == active:
                    self._interrupting_turn_id = None
            await self._queue.put(EngineError(message=f"interrupt failed: {exc}"))
            raise

    async def events(self) -> AsyncIterator[EngineEvent]:
        """Yield typed engine events until the session closes."""
        while True:
            item = await self._queue.get()
            if item is _READER_STOPPED:
                logger.debug("codex_engine.events | reader stopped, ending stream")
                return
            assert isinstance(item, (TextDelta, ToolUse, TurnDone, EngineError))
            yield item

    async def health(self) -> EngineHealth:
        """A current health snapshot."""
        connected = self._started and not self._closed
        async with self._turn_lock:
            active = self._active_turn_id is not None
        return EngineHealth(
            connected=connected,
            session_ref=self._thread_id,
            active_turn=active,
            detail="closed" if self._closed else ("connected" if connected else "down"),
        )

    # --- notification → event translation -------------------------------------

    async def _on_notification(self, message: dict[str, Any]) -> None:
        """Translate one ``ServerNotification`` into typed events on the queue."""
        method = message.get("method")
        params = message.get("params")
        params = params if isinstance(params, dict) else {}
        try:
            await self._dispatch_notification(method, params)
        except Exception as exc:  # noqa: BLE001 - never let a bad message kill the pump
            logger.error("codex_engine.notify | handling error method={}: {}", method, exc)
            await self._queue.put(EngineError(message=f"notification error: {exc}"))

    async def _dispatch_notification(self, method: Any, params: dict[str, Any]) -> None:
        """Map a single notification method to queue events / internal state."""
        if method == "thread/started":
            self._capture_thread(params.get("thread"))
            return
        if method == "turn/started":
            turn = params.get("turn")
            if isinstance(turn, dict):
                tid = turn.get("id")
                if isinstance(tid, str) and tid:
                    async with self._turn_lock:
                        self._active_turn_id = tid
            return
        if method == "item/agentMessage/delta":
            delta = params.get("delta")
            if isinstance(delta, str) and delta:
                self._text_buf.append(delta)
                await self._queue.put(TextDelta(text=delta))
            return
        if method in ("item/reasoning/textDelta", "item/reasoning/summaryTextDelta"):
            # Reasoning is not assistant-visible — never feed it as TextDelta
            # (memo gap 6: keep chain-of-thought off the outbound path).
            return
        if method == "item/started":
            await self._maybe_tool_use(params.get("item"))
            return
        if method == "item/completed":
            self._capture_final_item(params)
            return
        if method == "turn/completed":
            await self._on_turn_completed(params.get("turn"))
            return
        if method == "error":
            await self._on_error(params)
            return
        # All other notifications (token usage, plan updates, etc.) are not
        # surfaced as agentd events in this slice.

    async def _maybe_tool_use(self, item: Any) -> None:
        """Emit a ToolUse observability event for a tool-shaped started item."""
        if not isinstance(item, dict):
            return
        item_type = item.get("type")
        if item_type not in _TOOL_ITEM_TYPES:
            return
        tool_id = item.get("id")
        tool_input: dict[str, Any] = {}
        for key in ("arguments", "command", "changes", "query"):
            value = item.get(key)
            if value is not None:
                tool_input[key] = value
        name = self._tool_use_name(item)
        await self._queue.put(
            ToolUse(
                tool_id=str(tool_id) if tool_id is not None else "",
                name=name,
                tool_input=tool_input,
            )
        )

    @staticmethod
    def _tool_use_name(item: dict[str, Any]) -> str:
        """Return the underlying MCP tool name instead of its wrapper type."""
        item_type = item.get("type")
        if item_type != "mcpToolCall":
            return str(item_type)
        tool = item.get("tool")
        return tool if isinstance(tool, str) and tool else _UNKNOWN_MCP_TOOL_NAME

    async def _mark_turn_terminated(self, turn_id: str | None) -> bool:
        """Record that ``turn_id`` reached a terminal event; clear live turn state.

        Returns ``True`` if this is the FIRST terminal for the turn (the caller
        should emit the terminal event) and ``False`` if the turn already
        terminated (a duplicate the caller must log + drop). A ``None`` turn id
        (turn-less transport failure) is never deduped — it always returns
        ``True``. Clears the active-turn and interrupt-in-flight state.
        """
        async with self._turn_lock:
            self._active_turn_id = None
            self._interrupting_turn_id = None
            if turn_id is None:
                return True
            if turn_id in self._terminated_turns:
                return False
            self._terminated_turns.add(turn_id)
            # Bound the set: only the most recent ids matter for late-duplicate
            # detection; keep it from growing unbounded over a long session.
            if len(self._terminated_turns) > 256:
                self._terminated_turns = set(list(self._terminated_turns)[-128:])
            return True

    async def _on_turn_completed(self, turn: Any) -> None:
        """Emit a terminal TurnDone from a completed turn (stance from schema)."""
        turn_id = turn.get("id") if isinstance(turn, dict) else None
        turn_id = turn_id if isinstance(turn_id, str) and turn_id else None
        if not await self._mark_turn_terminated(turn_id):
            logger.warning(
                "codex_engine.turn_done | duplicate terminal for turnId={}; dropped", turn_id
            )
            return
        status = turn.get("status") if isinstance(turn, dict) else None
        is_error = status == "failed"
        # Prefer the final agentMessage text from the turn's items (authoritative,
        # in order); fall back to the delta buffer if the turn carries no items.
        final_text = self._final_message_text(turn)
        if not final_text:
            final_text = "".join(self._text_buf)
        self._text_buf.clear()
        parsed = parse_stance_result(final_text)
        # A malformed STANCE-SHAPED response (issue: parse-validity boundary) can
        # neither be trusted as a substantive terminal turn nor leaked as raw JSON.
        # It fails SAFE through the SAME typed failure path a ``status=="failed"``
        # turn takes: ``TurnDone(is_error=True)`` carries the failed stance, and the
        # frame-turn layer renders the human one-liner (issue #123) and logs the
        # raw payload once at that publish boundary. This never publishes a false normal substantive
        # terminal TurnFrame, and the failed stance is terminal so a single-scope
        # task does not silently hang. Valid stances and plain prose are unchanged.
        if parsed.is_blocked and not is_error:
            logger.error(
                "codex_engine.turn_done | blocked malformed stance-shaped turn; failing safe | "
                "reason={} turnId={} session_ref={}",
                parsed.reason,
                turn_id,
                self._thread_id,
            )
            await self._queue.put(
                TurnDone(
                    text=final_text,
                    session_ref=self._thread_id,
                    is_error=True,
                    stance=parsed.stance,
                    verification=parsed.verification,
                )
            )
            return
        stance = parsed.stance
        verification = parsed.verification
        reply = parsed.reply
        if is_error:
            # A failed turn keeps its neutral/failed framing; do not trust the
            # partial text as a stanced reply.
            reply = reply or final_text
        logger.info(
            "codex_engine.turn_done | len={} is_error={} session_ref={}",
            len(reply),
            is_error,
            self._thread_id,
        )
        await self._queue.put(
            TurnDone(
                text=reply,
                session_ref=self._thread_id,
                is_error=is_error,
                stance=stance,
                verification=verification,
            )
        )

    def _capture_final_item(self, params: dict[str, Any]) -> None:
        """Record the final agentMessage text from an ``item/completed`` event.

        As of codex-cli 0.144.5 this is the authoritative carrier of a turn's
        final assistant text (``turn/completed`` sends ``items: []`` /
        ``itemsView: "notLoaded"``). Keyed by turn id so ``_on_turn_completed``
        can read the right turn's text even when a steer->fresh-turn fallback
        interleaves delta streams. A later agentMessage for the same turn wins
        (a turn emits one final message; overwrite is a harmless no-op).
        """
        item = params.get("item")
        if not isinstance(item, dict) or item.get("type") != "agentMessage":
            return
        text = item.get("text")
        if not isinstance(text, str) or not text:
            return
        turn_id = params.get("turnId")
        if not isinstance(turn_id, str) or not turn_id:
            return
        self._final_item_text[turn_id] = text
        # Bound the map so a long session does not accumulate every turn's text.
        if len(self._final_item_text) > 256:
            for stale in list(self._final_item_text)[:128]:
                self._final_item_text.pop(stale, None)

    def _final_message_text(self, turn: Any) -> str:
        """Return a completed turn's final agentMessage text.

        Preference order, most authoritative first:
        1. The ``item/completed`` capture for this turn id (0.144.5+ carrier).
        2. Any agentMessage items inlined on the turn (older CLIs / if present).
        The session-wide delta buffer is the last resort, applied by the caller.
        """
        if not isinstance(turn, dict):
            return ""
        turn_id = turn.get("id")
        if isinstance(turn_id, str) and turn_id:
            captured = self._final_item_text.pop(turn_id, None)
            if captured:
                return captured
        items = turn.get("items")
        if not isinstance(items, list):
            return ""
        parts: list[str] = []
        for item in items:
            if isinstance(item, dict) and item.get("type") == "agentMessage":
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "".join(parts)

    async def _on_error(self, params: dict[str, Any]) -> None:
        """Surface a turn error as a single terminal event unless it is transient.

        ``willRetry == true`` marks a transient the engine will recover from — it
        must NOT terminate the turn (memo gap 5), so it is logged and suppressed.

        A terminal error tied to a turn (``turnId`` present) yields exactly ONE
        terminal ``TurnDone(is_error=True)`` — the same shape ClaudeEngine emits —
        so the supervisor FIFO completes the in-flight task and correlation does
        not slip onto the next TurnDone. The turn is marked terminated, so a late
        ``turn/completed`` for it is dropped (no duplicate terminal). A turn-less
        transport error keeps ``EngineError`` (the stream still ends via the
        transport stream-end path).
        """
        will_retry = bool(params.get("willRetry"))
        error = params.get("error")
        message = error.get("message") if isinstance(error, dict) else str(error)
        message = message if isinstance(message, str) and message else "codex turn error"
        if will_retry:
            logger.warning("codex_engine.error | transient (willRetry): {}", message)
            return
        turn_id = params.get("turnId")
        turn_id = turn_id if isinstance(turn_id, str) and turn_id else None
        if turn_id is None:
            async with self._turn_lock:
                turn_id = self._active_turn_id
        if turn_id is not None:
            if not await self._mark_turn_terminated(turn_id):
                logger.warning(
                    "codex_engine.error | duplicate terminal for turnId={}; dropped", turn_id
                )
                return
            logger.error("codex_engine.error | terminal turn turnId={}: {}", turn_id, message)
            self._text_buf.clear()
            await self._queue.put(
                TurnDone(
                    text=message,
                    session_ref=self._thread_id,
                    is_error=True,
                    stance=Stance(position="failed", because="engine reported a terminal error"),
                )
            )
            return
        # Turn-less transport failure: no FIFO entry to complete; surface raw.
        logger.error("codex_engine.error | terminal (turn-less): {}", message)
        await self._queue.put(EngineError(message=message))

    # --- helpers --------------------------------------------------------------

    def _require_transport(self) -> CodexAppServerTransport:
        """Return the live transport or raise if the session is not started."""
        if self._transport is None or not self._started:
            raise RuntimeError("codex_engine: session not started")
        return self._transport
