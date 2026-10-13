"""Persist resume refs and guard one live supervisor per agent."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class DuplicateSessionError(RuntimeError):
    """Raised when a second live supervisor starts for the same agent."""


@dataclass(frozen=True)
class SessionRefRecord:
    """The persisted resume pointer for one agent."""

    agent_id: str
    session_ref: str
    updated: str = field(default_factory=_now_iso)


class SessionRegistry:
    """Store the latest engine session refs keyed by agent id."""

    def __init__(self, home: Path, filename: str, agent_id: str = "default") -> None:
        self._path = (home / filename).resolve()
        self._agent_id = agent_id

    @property
    def path(self) -> Path:
        """Return the resolved JSON registry path."""
        return self._path

    def load(self) -> str | None:
        """Return the stored session ref for this agent, if present."""
        data = self._load_all()
        raw = data.get(self._agent_id)
        if not isinstance(raw, dict):
            return None
        ref = raw.get("session_ref")
        return ref if isinstance(ref, str) and ref else None

    def save(self, session_ref: str) -> None:
        """Atomically persist this agent's latest session ref."""
        if not session_ref:
            logger.warning("registry.save | empty session_ref rejected | agent={}", self._agent_id)
            raise ValueError("session_ref must be non-empty")
        data = self._load_all()
        record = SessionRefRecord(agent_id=self._agent_id, session_ref=session_ref)
        data[self._agent_id] = {
            "agent_id": record.agent_id,
            "session_ref": record.session_ref,
            "updated": record.updated,
        }
        self._save_all(data)

    def _load_all(self) -> dict[str, object]:
        if not self._path.is_file():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "registry.load | unreadable, ignoring | path={} reason={}", self._path, exc
            )
            return {}
        if not isinstance(data, dict):
            logger.warning("registry.load | non-object ignored | path={}", self._path)
            return {}
        return data

    def _save_all(self, data: dict[str, object]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, separators=(",", ":"), sort_keys=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self._path.parent), prefix=".session-registry-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self._path)
            logger.debug(
                "registry.save | session_ref persisted | agent={} path={}",
                self._agent_id,
                self._path,
            )
        except OSError as exc:
            logger.error("registry.save | failed | path={} reason={}", self._path, exc)
            try:
                os.unlink(tmp_name)
            except OSError as cleanup_exc:
                logger.warning(
                    "registry.save | temp cleanup failed | path={} reason={}",
                    tmp_name,
                    cleanup_exc,
                )
            raise


class ActiveSessionRegistry:
    """Track live supervisor claims by agent id inside this process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owners: dict[str, object] = {}

    def claim(self, agent_id: str, owner: object) -> None:
        """Claim ``agent_id`` for ``owner`` or reject a duplicate live session."""
        if not agent_id:
            raise ValueError("agent_id must be non-empty")
        with self._lock:
            current = self._owners.get(agent_id)
            if current is not None and current is not owner:
                logger.warning("registry.active | duplicate rejected | agent={}", agent_id)
                raise DuplicateSessionError(f"agent already has a live supervisor: {agent_id}")
            self._owners[agent_id] = owner
        logger.debug("registry.active | claimed | agent={}", agent_id)

    def release(self, agent_id: str, owner: object) -> None:
        """Release ``agent_id`` if ``owner`` still holds the claim."""
        with self._lock:
            if self._owners.get(agent_id) is owner:
                del self._owners[agent_id]
                logger.debug("registry.active | released | agent={}", agent_id)


ACTIVE_SESSIONS = ActiveSessionRegistry()
