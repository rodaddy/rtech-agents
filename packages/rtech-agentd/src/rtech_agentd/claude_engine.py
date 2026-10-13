"""``EngineSession`` over the Claude Agent SDK ``ClaudeSDKClient``.

Streaming-input mode: one persistent :class:`ClaudeSDKClient`, kept connected for
the whole session. A single long-lived background reader drains
``client.receive_messages()`` and translates SDK messages into typed
:class:`~rtech_agentd.engine.EngineEvent`s onto an internal queue that
:meth:`events` yields from. Because the reader runs independently of sending,
:meth:`send` can be called MID-TURN — the SDK queues the new user message and the
running turn sees it (inline steering). :meth:`interrupt` aborts the active turn.

Session id is captured from the ``init`` system message (early) and refreshed
from each ``ResultMessage`` so :meth:`resume` can reopen the same conversation.

All SDK I/O is wrapped in try/except with a clear log and surfaced as an
:class:`EngineError` event — failures are never swallowed silently.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from loguru import logger
from rtech_frames import Stance, VerificationReport

from .engine import (
    EngineError,
    EngineEvent,
    EngineHealth,
    TextDelta,
    ToolsPolicy,
    ToolUse,
    TurnDone,
    unverified_verification,
)
from .stance_tool import (
    STANCE_DIRECTIVE,
    STANCE_QUALIFIED_TOOL,
    STANCE_SERVER_NAME,
    build_stance_server,
    stance_from_tool_input,
)
from .wake_tool import (
    WAKE_DIRECTIVE,
    WAKE_QUALIFIED_TOOL,
    WAKE_SERVER_NAME,
    build_wake_server,
)

if TYPE_CHECKING:
    from .config import EngineConfig
    from .wake_service import WakeService

# Import the SDK lazily-friendly at module top; it is a hard dependency of the
# claude engine specifically (the protocol in engine.py stays SDK-free).
from claude_agent_sdk import (  # noqa: E402
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
)
from claude_agent_sdk.types import McpServerConfig  # noqa: E402

# Sentinel pushed onto the queue when the reader stops — ends events().
_READER_STOPPED: object = object()


class ClaudeEngine:
    """A managed Claude Agent SDK session implementing ``EngineSession``.

    Args:
        engine_config: Engine kind/model/litellm_base from the agent config.
        env: Extra environment for the SDK subprocess (e.g. ANTHROPIC_BASE_URL,
            ANTHROPIC_API_KEY). Never logged.
        wake_service: The wake capability. When present, the in-process
            ``wake_agent`` MCP tool is registered so the triage model can wake an
            allow-listed teammate; when ``None`` (a non-waker agent) the tool is
            NOT exposed at all — its mere presence is the allow-list opt-in.
    """

    def __init__(
        self,
        engine_config: EngineConfig,
        *,
        env: dict[str, str] | None = None,
        wake_service: WakeService | None = None,
    ) -> None:
        self._config = engine_config
        self._extra_env = dict(env or {})
        self._wake_service = wake_service
        self._client: ClaudeSDKClient | None = None
        self._reader: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[EngineEvent | object] = asyncio.Queue()
        self._session_ref: str | None = None
        self._active_turn = False
        self._closed = False
        # A captured typed stance for the in-flight turn (from the `stance` tool).
        self._pending_stance: Stance | None = None
        # Captured with stance from the SAME tool call; never inferred from text.
        self._pending_verification: VerificationReport | None = None

    def set_wake_service(self, wake_service: WakeService) -> None:
        """Attach the wake capability BEFORE the session starts.

        The wake ``WakeService`` needs the frame bus and the runner's active-origin
        provider, which are wired after the engine is built — so it is injected
        here rather than at construction. Registering the ``wake_agent`` MCP tool
        happens in :meth:`_build_options`, called at :meth:`start`; attaching after
        a session is already connected would not add the tool to the live session,
        so this MUST be called before ``start``.
        """
        if self._client is not None:
            raise RuntimeError(
                "claude_engine.set_wake_service: must be called before start(); the wake "
                "tool is registered at session start and cannot be added to a live session"
            )
        self._wake_service = wake_service

    # --- construction of SDK options -----------------------------------------

    def _tool_registration(
        self, extra_allow: tuple[str, ...] = ()
    ) -> tuple[dict[str, McpServerConfig], list[str], str]:
        """Assemble the in-process MCP servers, allowed tools, and directive suffix.

        Shared by :meth:`_build_options` (fresh start) AND :meth:`resume` so a
        resumed session keeps the SAME in-process tools as a fresh one. The
        ``stance`` tool is always registered; the ``wake`` tool is
        registered ONLY when a :class:`WakeService` is attached (the waker opt-in),
        so a non-waker never gets it and cannot wake-chain.

        Args:
            extra_allow: Extra allow-list tool names to prepend (the tools policy's
                ``allow`` on a fresh start; empty on resume).

        Returns:
            A ``(mcp_servers, allowed_tools, directive)`` triple.
        """
        stance_server = build_stance_server()
        mcp_servers: dict[str, McpServerConfig] = {STANCE_SERVER_NAME: stance_server}
        allowed_tools = [*extra_allow, STANCE_QUALIFIED_TOOL]
        directive = STANCE_DIRECTIVE
        if self._wake_service is not None:
            mcp_servers[WAKE_SERVER_NAME] = build_wake_server(self._wake_service)
            allowed_tools.append(WAKE_QUALIFIED_TOOL)
            directive += WAKE_DIRECTIVE
            logger.info("claude_engine.options | wake tool exposed (agent is a waker)")
        return mcp_servers, allowed_tools, directive

    def _build_options(self, system_prompt: str, tools_policy: ToolsPolicy) -> ClaudeAgentOptions:
        """Translate agent config + tools policy into ``ClaudeAgentOptions``."""
        env = dict(self._extra_env)
        if self._config.litellm_base and "ANTHROPIC_BASE_URL" not in env:
            # LiteLLM-compatible endpoint is passed to the SDK subprocess via the
            # standard Anthropic base-url env var.
            env["ANTHROPIC_BASE_URL"] = self._config.litellm_base
        # The in-process `stance` (+ optional `wake`) tools are registered here so
        # the model fills TurnFrame.stance / wakes a teammate via a typed tool call
        # — never scraped prose. The same registration is reused by resume().
        mcp_servers, allowed_tools, directive = self._tool_registration(tools_policy.allow)
        options = ClaudeAgentOptions(
            system_prompt=system_prompt + directive,
            model=self._config.model or None,
            fallback_model=self._config.backup_model or None,
            allowed_tools=allowed_tools,
            disallowed_tools=list(tools_policy.deny),
            mcp_servers=mcp_servers,
            # Managed daemon: no interactive permission prompts. Deny-list governs
            # what the engine may touch; anything not denied is auto-approved.
            permission_mode="bypassPermissions",
            include_partial_messages=True,
            env=env,
        )
        return options

    # --- lifecycle ------------------------------------------------------------

    async def start(self, system_prompt: str, tools_policy: ToolsPolicy) -> None:
        """Connect a fresh session and start the background reader."""
        logger.info(
            "claude_engine.start | model={} allow={} deny={}",
            self._config.model,
            list(tools_policy.allow),
            list(tools_policy.deny),
        )
        await self._connect(self._build_options(system_prompt, tools_policy))

    async def resume(self, session_ref: str) -> None:
        """Reopen a prior session by ``session_ref`` (fresh reader, empty prompt).

        The in-process ``stance`` (and, for a waker, ``wake``) MCP servers are NOT
        persisted on the resumed conversation, so they are re-registered here via
        the SAME :meth:`_tool_registration` a fresh start uses. Without this a
        resumed triage session would silently lose its ``wake_agent`` tool (and
        stance capture) — a correctness gap, since the supervisor resumes on
        restart and on dead-session recovery.
        """
        if not session_ref:
            raise ValueError("claude_engine.resume: session_ref must be non-empty")
        logger.info("claude_engine.resume | session_ref={}", session_ref)
        mcp_servers, allowed_tools, _directive = self._tool_registration()
        options = ClaudeAgentOptions(
            resume=session_ref,
            model=self._config.model or None,
            allowed_tools=allowed_tools,
            mcp_servers=mcp_servers,
            permission_mode="bypassPermissions",
            include_partial_messages=True,
            env=self._resume_env(),
        )
        self._session_ref = session_ref
        await self._connect(options)

    def _resume_env(self) -> dict[str, str]:
        env = dict(self._extra_env)
        if self._config.litellm_base and "ANTHROPIC_BASE_URL" not in env:
            env["ANTHROPIC_BASE_URL"] = self._config.litellm_base
        return env

    async def _connect(self, options: ClaudeAgentOptions) -> None:
        """Open the SDK client and launch the reader. Fails fast + loud."""
        if self._client is not None:
            raise RuntimeError("claude_engine: already started")
        self._closed = False
        try:
            client = ClaudeSDKClient(options=options)
            await client.connect()
        except Exception as exc:
            logger.error("claude_engine.connect | failed reason={}", exc)
            raise
        self._client = client
        self._reader = asyncio.create_task(self._read_loop(), name="claude-engine-reader")
        logger.debug("claude_engine.connect | connected, reader started")

    async def close(self) -> None:
        """Cancel the reader and disconnect the SDK client. Idempotent."""
        if self._closed:
            return
        self._closed = True
        logger.info("claude_engine.close | tearing down session_ref={}", self._session_ref)
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning("claude_engine.close | reader error on cancel: {}", exc)
            self._reader = None
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception as exc:
                logger.warning("claude_engine.close | disconnect error: {}", exc)
            self._client = None
        # Unblock any events() consumer.
        await self._queue.put(_READER_STOPPED)

    # --- turn I/O -------------------------------------------------------------

    async def send(self, text: str) -> None:
        """Send a user turn — or steer inline if a turn is already active.

        Both are the same SDK call (``client.query``); the SDK queues the message
        and the running turn consumes it. The reader is untouched, so a mid-turn
        send does not block the event stream.
        """
        if self._client is None:
            raise RuntimeError("claude_engine.send: session not started")
        mode = "steer" if self._active_turn else "turn"
        logger.info("claude_engine.send | mode={} len={}", mode, len(text))
        try:
            await self._client.query(text)
        except Exception as exc:
            logger.error("claude_engine.send | mode={} failed: {}", mode, exc)
            await self._queue.put(EngineError(message=f"send failed: {exc}"))
            raise

    async def interrupt(self) -> None:
        """Abort the active turn (best-effort; no-op-safe when idle)."""
        if self._client is None:
            raise RuntimeError("claude_engine.interrupt: session not started")
        logger.info("claude_engine.interrupt | active_turn={}", self._active_turn)
        try:
            await self._client.interrupt()
        except Exception as exc:
            logger.error("claude_engine.interrupt | failed: {}", exc)
            await self._queue.put(EngineError(message=f"interrupt failed: {exc}"))
            raise

    async def events(self) -> AsyncIterator[EngineEvent]:
        """Yield typed engine events until the session closes."""
        while True:
            item = await self._queue.get()
            if item is _READER_STOPPED:
                logger.debug("claude_engine.events | reader stopped, ending stream")
                return
            # Everything the reader enqueues besides the stop sentinel is an
            # EngineEvent; narrow for the type checker.
            assert isinstance(item, (TextDelta, ToolUse, TurnDone, EngineError))
            yield item

    async def health(self) -> EngineHealth:
        """A current health snapshot."""
        connected = self._client is not None and not self._closed
        return EngineHealth(
            connected=connected,
            session_ref=self._session_ref,
            active_turn=self._active_turn,
            detail="closed" if self._closed else ("connected" if connected else "down"),
        )

    # --- background reader ----------------------------------------------------

    async def _read_loop(self) -> None:
        """Drain SDK messages → typed events onto the queue for the session life.

        Runs until cancelled (:meth:`close`) or the SDK stream ends. Each message
        is translated defensively; a translation error becomes an EngineError
        event rather than killing the reader.
        """
        assert self._client is not None
        text_buf: list[str] = []
        try:
            async for message in self._client.receive_messages():
                try:
                    await self._handle_message(message, text_buf)
                except Exception as exc:
                    logger.error("claude_engine.read | message handling error: {}", exc)
                    await self._queue.put(EngineError(message=f"read error: {exc}"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("claude_engine.read | stream error: {}", exc)
            await self._queue.put(EngineError(message=f"stream error: {exc}"))
        finally:
            logger.debug("claude_engine.read | loop exiting")
            await self._queue.put(_READER_STOPPED)

    async def _handle_message(self, message: Any, text_buf: list[str]) -> None:
        """Translate one SDK message to typed events, buffering turn text."""
        if isinstance(message, SystemMessage):
            if message.subtype == "init":
                sid = None
                if isinstance(message.data, dict):
                    sid = message.data.get("session_id")
                if isinstance(sid, str) and sid:
                    self._session_ref = sid
                    logger.debug("claude_engine.read | captured session_ref={}", sid)
            return

        if isinstance(message, AssistantMessage):
            self._active_turn = True
            for block in message.content:
                if isinstance(block, TextBlock) and block.text:
                    text_buf.append(block.text)
                    await self._queue.put(TextDelta(text=block.text))
                elif isinstance(block, ToolUseBlock):
                    tool_input = block.input if isinstance(block.input, dict) else {}
                    # Capture the typed stance the model passed to the `stance`
                    # tool — verbatim, never scraped. Observability is unchanged:
                    # the ToolUse event is still emitted below as before.
                    if block.name == STANCE_QUALIFIED_TOOL:
                        captured = stance_from_tool_input(tool_input)
                        if captured is not None:
                            self._pending_stance, self._pending_verification = captured
                        # Stance/verification can name private sources. Keep the
                        # observability event but never carry its raw input into
                        # downstream logs or metrics.
                        tool_input = {}
                    await self._queue.put(
                        ToolUse(tool_id=block.id, name=block.name, tool_input=tool_input)
                    )
            return

        if isinstance(message, ResultMessage):
            if isinstance(message.session_id, str) and message.session_id:
                self._session_ref = message.session_id
            full_text = "".join(text_buf)
            text_buf.clear()
            self._active_turn = False
            is_error = bool(getattr(message, "is_error", False))
            logger.info(
                "claude_engine.read | turn_done len={} is_error={} session_ref={}",
                len(full_text),
                is_error,
                self._session_ref,
            )
            # Stance is a typed field the engine fills — NEVER scraped from
            # full_text. Prefer the stance the model recorded via the `stance`
            # tool this turn; otherwise fall back to a fixed position keyed on the
            # SDK's own is_error result.
            stance = self._pending_stance or Stance(
                position="failed" if is_error else "responded",
                because="engine reported an error result"
                if is_error
                else "engine completed the turn",
            )
            verification = self._pending_verification or unverified_verification()
            self._pending_stance = None
            self._pending_verification = None
            await self._queue.put(
                TurnDone(
                    text=full_text,
                    session_ref=self._session_ref,
                    is_error=is_error,
                    stance=stance,
                    verification=verification,
                )
            )
            return
        # UserMessage (tool results) / StreamEvent / RateLimitEvent: not surfaced
        # as agentd events in this slice. Buffered turn text comes from
        # AssistantMessage blocks; deltas are already emitted above.
