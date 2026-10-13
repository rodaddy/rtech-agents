"""Unit tests for the JSON-RPC stdio transport against a fake child process.

A tiny in-memory duplex stands in for the ``codex app-server`` child: the test
feeds newline-framed JSON responses/notifications into the reader and captures
what the transport writes to stdin. This proves id correlation, the initialize
handshake framing, notification pumping, RPC-error surfacing, and pending-request
failure when the stream ends — with no real ``codex`` binary.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from rtech_agentd.codex_protocol import CodexAppServerTransport, CodexTransportError


class _FakeStreamWriter:
    """Captures newline-framed lines the transport writes to child stdin."""

    def __init__(self) -> None:
        self.lines: list[dict[str, Any]] = []
        self._closing = False

    def write(self, data: bytes) -> None:
        for chunk in data.decode("utf-8").splitlines():
            if chunk.strip():
                self.lines.append(json.loads(chunk))

    def is_closing(self) -> bool:
        return self._closing

    def close(self) -> None:
        self._closing = True


class _FakeStreamReader:
    """Yields bytes lines the test queues, then EOF (empty bytes)."""

    def __init__(self) -> None:
        self._q: asyncio.Queue[bytes] = asyncio.Queue()

    def feed(self, message: dict[str, Any]) -> None:
        self._q.put_nowait((json.dumps(message) + "\n").encode("utf-8"))

    def feed_eof(self) -> None:
        self._q.put_nowait(b"")

    async def readline(self) -> bytes:
        return await self._q.get()

    async def read(self) -> bytes:
        return b""


class _StderrStreamReader:
    """A stderr stand-in that yields a fixed blob once, then empties."""

    def __init__(self, blob: bytes) -> None:
        self._blob = blob

    async def read(self) -> bytes:
        blob, self._blob = self._blob, b""
        return blob


class _FakeProc:
    """A stand-in ``asyncio.subprocess.Process`` for the transport."""

    def __init__(self, stdout: _FakeStreamReader) -> None:
        self.stdin = _FakeStreamWriter()
        self.stdout = stdout
        self.stderr = _FakeStreamReader()
        self.returncode: int | None = None

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        return 0


async def _wire_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[CodexAppServerTransport, _FakeProc, _FakeStreamReader]:
    """Build a transport whose spawn is patched to return a fake child."""
    stdout = _FakeStreamReader()
    proc = _FakeProc(stdout)

    async def fake_exec(*_args: Any, **_kwargs: Any) -> _FakeProc:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    transport = CodexAppServerTransport(command=["codex", "app-server"])
    return transport, proc, stdout


async def test_initialize_handshake_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, proc, stdout = await _wire_transport(monkeypatch)
    # Answer the initialize request as soon as it is written.
    stdout.feed({"id": 1, "result": {"userAgent": "codex", "platformOs": "test"}})
    result = await asyncio.wait_for(transport.start(), timeout=1.0)
    assert result["platformOs"] == "test"
    # First line is the initialize request; then an initialized notification.
    assert proc.stdin.lines[0]["method"] == "initialize"
    assert proc.stdin.lines[0]["params"]["capabilities"]["optOutNotificationMethods"] == []
    assert proc.stdin.lines[1] == {"method": "initialized", "params": {}}
    await transport.close()


async def test_request_response_correlation(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, proc, stdout = await _wire_transport(monkeypatch)
    stdout.feed({"id": 1, "result": {}})
    await transport.start()
    # A later request gets id 2; feed a matching response.
    stdout.feed({"id": 2, "result": {"thread": {"id": "t-1"}}})
    result = await asyncio.wait_for(transport.request("thread/start", {}), timeout=1.0)
    assert result["thread"]["id"] == "t-1"
    await transport.close()


async def test_rpc_error_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, proc, stdout = await _wire_transport(monkeypatch)
    stdout.feed({"id": 1, "result": {}})
    await transport.start()
    stdout.feed({"id": 2, "error": {"code": -32000, "message": "bad request"}})
    with pytest.raises(CodexTransportError) as exc:
        await asyncio.wait_for(transport.request("turn/start", {}), timeout=1.0)
    assert "bad request" in str(exc.value)
    await transport.close()


async def test_notifications_are_pumped(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, proc, stdout = await _wire_transport(monkeypatch)
    received: list[dict[str, Any]] = []

    async def handler(msg: dict[str, Any]) -> None:
        received.append(msg)

    transport.set_notification_handler(handler)
    stdout.feed({"id": 1, "result": {}})
    await transport.start()
    stdout.feed({"method": "turn/started", "params": {"turn": {"id": "x"}}})
    await asyncio.sleep(0.05)
    assert received and received[0]["method"] == "turn/started"
    await transport.close()


async def test_stream_end_fails_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    transport, proc, stdout = await _wire_transport(monkeypatch)
    stdout.feed({"id": 1, "result": {}})
    await transport.start()
    # Issue a request that will never be answered, then EOF the stream.
    pending = asyncio.create_task(transport.request("turn/start", {}))
    await asyncio.sleep(0)
    stdout.feed_eof()
    with pytest.raises(CodexTransportError):
        await asyncio.wait_for(pending, timeout=1.0)
    await transport.close()


async def test_stream_end_redacts_stderr_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    # H3: child stderr may echo credential material. The stream-ended exception
    # must NOT contain a raw sk- token or a passed credential env value.
    stdout = _FakeStreamReader()
    proc = _FakeProc(stdout)
    proc.stderr = _StderrStreamReader(
        b"boom: key=sk-proj-ABCDEFGH1234567890 SECRETVALUE leaked"  # gitleaks:allow
    )

    async def fake_exec(*_args: Any, **_kwargs: Any) -> _FakeProc:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    transport = CodexAppServerTransport(env={"OPENAI_API_KEY": "SECRETVALUE"})
    stdout.feed({"id": 1, "result": {}})
    await transport.start()
    pending = asyncio.create_task(transport.request("turn/start", {}))
    await asyncio.sleep(0)
    stdout.feed_eof()
    with pytest.raises(CodexTransportError) as exc:
        await asyncio.wait_for(pending, timeout=1.0)
    text = str(exc.value)
    assert "sk-proj-ABCDEFGH1234567890" not in text
    assert "SECRETVALUE" not in text
    assert "***" in text
    await transport.close()


async def test_request_after_stream_end_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    # H1: once the reader is done, a new request must fail fast — not register a
    # future that waits forever.
    transport, proc, stdout = await _wire_transport(monkeypatch)
    stdout.feed({"id": 1, "result": {}})
    await transport.start()
    stdout.feed_eof()
    await asyncio.sleep(0.05)  # let the reader observe EOF and set _reader_done
    with pytest.raises(CodexTransportError):
        await asyncio.wait_for(transport.request("turn/start", {}), timeout=1.0)
    await transport.close()


async def test_close_safe_when_never_started() -> None:
    transport = CodexAppServerTransport()
    await transport.close()  # no child spawned; must not raise
