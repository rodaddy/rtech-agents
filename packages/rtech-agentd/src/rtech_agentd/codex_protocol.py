"""JSON-RPC framing over stdio for the ``codex app-server`` child process.

The codex CLI exposes a long-lived ``codex app-server`` mode that speaks
newline-delimited JSON-RPC (draft-07) over stdin/stdout: an ``initialize``
request, an ``initialized`` notification, then request/response pairs correlated
by integer ``id`` interleaved with server ``ServerNotification`` messages (no
``id``). This module owns ONLY that transport — spawning the child, framing
lines, correlating responses to their requests, and pumping notifications to a
handler. It has no knowledge of the ``EngineSession`` contract; the adapter in
:mod:`rtech_agentd.codex_engine` layers the state machine on top.

The handshake mirrors OpenAI's own reference client (``codex-companion``'s
``app-server.mjs``) exactly: ``initialize`` with ``clientInfo`` + ``capabilities``
then an ``initialized`` notification. Unlike the companion we deliberately do NOT
opt out of the ``item/agentMessage/delta`` stream — the adapter needs those
deltas to surface ``TextDelta`` events — so ``optOutNotificationMethods`` is
empty.

All transport I/O is wrapped and surfaced as :class:`CodexTransportError`; a
dead child or unparsable line never silently strands a pending request.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

# The codex CLI version this protocol shim was written and tested against. The
# app-server surface is flagged experimental and its 500+ definition schema can
# drift between releases, so the adapter logs this at start and it is the single
# place to bump when re-verifying against a newer CLI.
TESTED_CODEX_CLI_VERSION = "codex-cli 0.144.5"

# JSON-RPC id used for a request that expects a correlated response.
_JsonRpcId = int

# A parsed server notification is a raw JSON object with ``method`` + ``params``.
NotificationHandler = Callable[[dict[str, Any]], Awaitable[None]]

# Called once when the reader stops (child exit / stream error): lets the adapter
# end its event stream so a turn-less death does not hang ``events()`` forever.
StreamEndHandler = Callable[[], Awaitable[None]]

# Env var names whose VALUES must never appear in a log/exception (credentials
# passed to the child). Their literal values are scrubbed from any stderr tail.
_CREDENTIAL_ENV_NAMES = frozenset(
    {
        "OPENAI_API_KEY",
        "OPENAI_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "LITELLM_API_KEY",
        "LITELLM_MASTER_KEY",
    }
)

# Generic provider-token shapes (OpenAI ``sk-``/``sk-proj-``, Anthropic
# ``sk-ant-``) scrubbed even when we did not pass the value ourselves.
_TOKEN_PATTERN = re.compile(r"sk-(?:ant-|proj-)?[A-Za-z0-9_\-]{8,}")

# Hard cap on any diagnostic tail so a runaway child cannot bloat a log line.
_STDERR_TAIL_CAP = 500

_REDACTED = "***"


def _redact(text: str, env: dict[str, str] | None) -> str:
    """Scrub credential values + generic tokens from ``text`` and cap its length.

    Removes the literal values of known credential env vars passed to the child
    (so a stderr echo of a key can never surface) and any generic ``sk-``-style
    token, then truncates. Used before any child stderr is placed in an
    exception message or log — never emit raw child stderr.
    """
    if not text:
        return text
    if env:
        for name in _CREDENTIAL_ENV_NAMES:
            value = env.get(name)
            if value:
                text = text.replace(value, _REDACTED)
    text = _TOKEN_PATTERN.sub(_REDACTED, text)
    if len(text) > _STDERR_TAIL_CAP:
        text = text[:_STDERR_TAIL_CAP] + "…"
    return text


class CodexTransportError(Exception):
    """A failure in the codex app-server transport (spawn/IO/parse/RPC).

    Raised for RPC-level errors (the server returned a JSON-RPC ``error``), a
    dead child, or a stream that closed with requests still pending. Never
    swallowed — the adapter maps it to an ``EngineError`` event.
    """


class CodexAppServerTransport:
    """A spawned ``codex app-server`` child speaking JSON-RPC over stdio.

    Lifecycle: :meth:`start` spawns the child, launches the stdout reader, and
    performs the ``initialize``/``initialized`` handshake. :meth:`request`
    correlates a response by id; server notifications are pumped to the handler
    set via :meth:`set_notification_handler`. :meth:`close` ends stdin and
    terminates the child. Safe to :meth:`close` when never started.

    Args:
        command: The argv for the child (default ``["codex", "app-server"]``).
        cwd: Working directory for the child, or ``None`` for the caller's cwd.
        env: Environment for the child, or ``None`` to inherit. Never logged.
        client_info: JSON-RPC ``clientInfo`` sent at ``initialize``.
    """

    def __init__(
        self,
        *,
        command: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        client_info: dict[str, str] | None = None,
    ) -> None:
        self._command = command or ["codex", "app-server"]
        self._cwd = cwd
        self._env = dict(env) if env is not None else None
        self._client_info = client_info or {
            "name": "rtech-agentd",
            "title": "rtech-agentd codex engine",
            "version": "0.0.0",
        }
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._next_id: _JsonRpcId = 1
        self._pending: dict[_JsonRpcId, asyncio.Future[dict[str, Any]]] = {}
        self._handler: NotificationHandler | None = None
        self._stream_end_handler: StreamEndHandler | None = None
        self._closed = False
        # Set when the reader stops so a late request fails fast instead of hanging.
        self._reader_done = asyncio.Event()
        # Guards a single stream-end notification to the adapter (reader exit +
        # close() must not both fire it).
        self._stream_end_signalled = False

    def set_notification_handler(self, handler: NotificationHandler) -> None:
        """Register the coroutine that receives each server notification dict."""
        self._handler = handler

    def set_stream_end_handler(self, handler: StreamEndHandler) -> None:
        """Register the coroutine invoked once when the reader stops.

        Fires on child exit / stream error (not on an orderly :meth:`close`,
        which the adapter already drives). Lets the adapter end ``events()`` so a
        death with no pending RPC does not strand its consumer forever.
        """
        self._stream_end_handler = handler

    # --- lifecycle ------------------------------------------------------------

    async def start(self) -> dict[str, Any]:
        """Spawn the child, start the reader, and run the initialize handshake.

        Returns the ``initialize`` response result. Fails loud on spawn error.
        """
        if self._proc is not None:
            raise CodexTransportError("codex transport already started")
        logger.info(
            "codex_protocol.start | command={} tested={}",
            self._command,
            TESTED_CODEX_CLI_VERSION,
        )
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._cwd,
                env=self._env,
            )
        except (OSError, ValueError) as exc:
            logger.error("codex_protocol.start | spawn failed: {}", exc)
            raise CodexTransportError(f"codex app-server spawn failed: {exc}") from exc
        self._closed = False
        self._stream_end_signalled = False
        self._reader_done.clear()
        self._reader = asyncio.create_task(self._read_loop(), name="codex-transport-reader")
        # Mirror the reference handshake exactly: initialize then initialized. A
        # handshake failure must not leave the spawned child + reader alive —
        # close() reaps both before we re-raise.
        try:
            result = await self.request(
                "initialize",
                {
                    "clientInfo": self._client_info,
                    "capabilities": {
                        "experimentalApi": True,
                        "requestAttestation": False,
                        # Empty: we WANT the agentMessage delta stream (unlike the
                        # companion, which opts out of deltas for its UX).
                        "optOutNotificationMethods": [],
                    },
                },
            )
            self.notify("initialized", {})
        except CodexTransportError:
            logger.error("codex_protocol.start | handshake failed; reaping child")
            await self.close()
            raise
        logger.debug("codex_protocol.start | handshake complete")
        return result

    async def close(self) -> None:
        """End stdin, terminate the child, and cancel the reader. Idempotent.

        Safe when never started (no child handle).
        """
        if self._closed:
            return
        self._closed = True
        logger.info("codex_protocol.close | tearing down transport")
        proc = self._proc
        if proc is not None and proc.stdin is not None and not proc.stdin.is_closing():
            try:
                proc.stdin.close()
            except (BrokenPipeError, ConnectionResetError, RuntimeError) as exc:
                logger.debug("codex_protocol.close | stdin close error: {}", exc)
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except (ProcessLookupError, PermissionError) as exc:
                logger.debug("codex_protocol.close | terminate error: {}", exc)
            except TimeoutError:
                logger.warning("codex_protocol.close | child did not exit; killing")
                try:
                    proc.kill()
                    # Await the kill so close() proves the child is reaped (no
                    # zombie / leaked pid escaping teardown).
                    await proc.wait()
                except ProcessLookupError:
                    pass
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001 - defensive: never let close raise
                logger.warning("codex_protocol.close | reader error on cancel: {}", exc)
            self._reader = None
        self._fail_pending(CodexTransportError("codex transport closed"))
        self._proc = None

    # --- request / notify -----------------------------------------------------

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and await its correlated response result.

        Raises:
            CodexTransportError: On a JSON-RPC error response, a closed transport,
                or the reader stopping before the response arrives.
        """
        if self._closed or self._proc is None:
            raise CodexTransportError(f"codex transport not started (method={method})")
        # The reader has stopped (stream ended): no response can ever arrive, so
        # fail fast instead of registering a future that waits forever.
        if self._reader_done.is_set():
            raise CodexTransportError(f"codex transport stream ended (method={method})")
        req_id = self._next_id
        self._next_id += 1
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = future
        try:
            self._write({"id": req_id, "method": method, "params": params})
        except CodexTransportError:
            self._pending.pop(req_id, None)
            raise
        return await future

    def notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a fire-and-forget JSON-RPC notification (no id, no response)."""
        if self._closed or self._proc is None:
            raise CodexTransportError(f"codex transport not started (notify={method})")
        self._write({"method": method, "params": params})

    def _write(self, message: dict[str, Any]) -> None:
        """Serialize + write one newline-framed JSON message to the child stdin."""
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise CodexTransportError("codex transport stdin unavailable")
        line = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            proc.stdin.write(line)
        except (BrokenPipeError, ConnectionResetError, RuntimeError) as exc:
            logger.error("codex_protocol.write | failed method={}: {}", message.get("method"), exc)
            raise CodexTransportError(f"codex transport write failed: {exc}") from exc

    # --- reader ---------------------------------------------------------------

    async def _read_loop(self) -> None:
        """Drain child stdout: resolve responses by id, pump notifications.

        Runs until the stream closes (child exit) or the task is cancelled. Each
        line is parsed defensively; a bad line is logged and skipped rather than
        killing the reader. On stream end, all still-pending requests fail loud.
        """
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    logger.debug("codex_protocol.read | stdout EOF")
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                await self._dispatch_line(line)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface as pending-request failure
            logger.error("codex_protocol.read | loop error: {}", exc)
            self._fail_pending(CodexTransportError(f"codex transport read error: {exc}"))
        finally:
            self._reader_done.set()
            # Stream ended: strand nothing. Include a REDACTED stderr tail for
            # diagnosis (raw child stderr may echo credential material).
            stderr_tail = await self._drain_stderr()
            self._fail_pending(CodexTransportError(f"codex app-server stream ended{stderr_tail}"))
            # Signal stream-end to the adapter so a turn-less death ends events()
            # (a cancelled reader is an orderly close(); it does not signal here).
            await self._signal_stream_end()

    async def _signal_stream_end(self) -> None:
        """Invoke the stream-end handler exactly once (never lets close() double-fire)."""
        if self._stream_end_signalled:
            return
        self._stream_end_signalled = True
        handler = self._stream_end_handler
        if handler is None:
            return
        try:
            await handler()
        except Exception as exc:  # noqa: BLE001 - handler error must not kill teardown
            logger.error("codex_protocol.read | stream-end handler error: {}", exc)

    async def _dispatch_line(self, line: str) -> None:
        """Route one parsed JSON line to a pending future or the handler."""
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.error("codex_protocol.read | unparsable line: {}", exc)
            return
        if not isinstance(message, dict):
            logger.warning("codex_protocol.read | non-object message dropped")
            return
        msg_id = message.get("id")
        # A response (has id, no method). Server requests (id + method) are not
        # supported here — the adapter needs no approval callbacks in this slice.
        if msg_id is not None and "method" not in message:
            future = self._pending.pop(msg_id, None)
            if future is None or future.done():
                return
            error = message.get("error")
            if error is not None:
                msg = error.get("message") if isinstance(error, dict) else str(error)
                future.set_exception(CodexTransportError(f"codex RPC error: {msg}"))
            else:
                result = message.get("result")
                future.set_result(result if isinstance(result, dict) else {})
            return
        if msg_id is not None and "method" in message:
            # Server-initiated request (e.g. approval): reply "unsupported" so the
            # server does not block, matching the reference client's behavior.
            method = message.get("method")
            logger.debug("codex_protocol.read | unsupported server request method={}", method)
            with_id = {
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"unsupported server request: {method}",
                },
            }
            try:
                self._write(with_id)
            except CodexTransportError as exc:
                logger.debug("codex_protocol.read | could not reply to server request: {}", exc)
            return
        # A notification (method, no id).
        if "method" in message and self._handler is not None:
            try:
                await self._handler(message)
            except Exception as exc:  # noqa: BLE001 - handler error must not kill reader
                logger.error("codex_protocol.read | notification handler error: {}", exc)

    async def _drain_stderr(self) -> str:
        """Best-effort read of the child stderr tail for error context."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return ""
        try:
            data = await asyncio.wait_for(proc.stderr.read(), timeout=0.5)
        except (TimeoutError, Exception):  # noqa: BLE001 - diagnostics only
            return ""
        text = data.decode("utf-8", errors="replace").strip()
        # Never surface raw child stderr: scrub credential values + tokens, cap.
        text = _redact(text, self._env)
        return f": {text}" if text else ""

    def _fail_pending(self, error: CodexTransportError) -> None:
        """Resolve every still-pending request future with ``error``."""
        pending = self._pending
        self._pending = {}
        for future in pending.values():
            if not future.done():
                future.set_exception(error)
