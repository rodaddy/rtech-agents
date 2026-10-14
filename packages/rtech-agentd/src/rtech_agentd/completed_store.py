"""Persist completed-task dedup state across agentd restarts.

The frame-turn runner keeps an in-process ``_completed_tasks`` cache so a
redelivered already-completed frame is never re-dispatched. That cache is
MEMORY-ONLY: an agentd restart within the replay horizon starts empty, so a
redelivered completed frame with a fresh-enough timestamp passes the stale
guard and gets RE-DISPATCHED. This store is the on-disk half — it persists the
completion WALL time per (task_id, agent_id) so the dedup survives a restart.

The horizon check here is WALL-CLOCK on purpose: monotonic time does not
survive a restart. The runner keeps monotonic timing for in-process eviction of
``_completed_tasks``; this persisted store is the durable, wall-clock-bounded
authority a restart reloads. Entries are evicted by horizon on every load and
every save so the file never grows past the safety window.

Missing vs. corrupt are DISTINCT integrity states, handled differently:

* MISSING file = a legitimate first boot. The store starts EMPTY with a loud
  INFO log; nothing is degraded. There is a narrow residual redelivery risk if
  the file is LOST (not corrupt) between a completion and a restart within the
  horizon — the belt-and-suspenders closure for that is JetStream ``Msg-Id``
  dedup at the transport, which drops a duplicate before it
  ever reaches this store.
* CORRUPT / unparseable file = PROVABLE integrity loss: the durable dedup
  authority cannot be trusted for the entries it should have held. Rather than
  silently start empty and fail OPEN, the store enters a DISTRUST WINDOW of
  ``horizon_s`` seconds (wall clock) from load. While inside that window the
  runner REFUSES any untracked-but-within-horizon frame (fail closed, loud
  warning per refusal). After the window every redeliverable frame has aged
  past the horizon anyway, so normal operation resumes and the guarantee is
  restored honestly. The corrupt file is PRESERVED as ``<name>.corrupt-<ts>``
  for forensics rather than overwritten on the next save.

Persisted shape is a NESTED map ``{task_id: {agent_id: iso_wall_ts}}`` so an
arbitrary (hostile/odd, e.g. NUL-bearing) task id or agent id round-trips as a
native JSON object key — no composite string key, no separator to mangle.

Single-writer discipline is structural: the filename is PER-AGENT
(``completed-tasks-<agent_id>.json``). ``ActiveSessionRegistry`` already
enforces one live runner per agent id, so a per-agent file makes every
whole-map save single-writer by construction — two runners never race the same
file. ``agent_id`` is kept inside the nested map anyway (defense in depth,
negligible cost).

Disk idiom mirrors :class:`~rtech_agentd.registry.SessionRegistry`: atomic
write (tempfile + fsync + ``os.replace``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from loguru import logger

# Filesystem-safe agent-id characters for the per-agent filename. Anything else
# is sanitized, and a short stable hash of the RAW id is ALWAYS appended, so
# two distinct raw ids can never collide onto one filename (a conditional
# hash would let a crafted clean id impersonate a dirty id's stem+hash).
_SAFE_AGENT_ID = re.compile(r"[^A-Za-z0-9._-]")
_AGENT_HASH_LEN = 12


def _agent_filename(agent_id: str) -> str:
    """Return the per-agent dedup filename, collision-safe by construction.

    ``agent_id`` is sanitized for filesystem use (no path separators, no odd
    bytes) and a short stable hash of the RAW id is always appended: distinct
    raw ids yield distinct names unconditionally, because the hash is over the
    raw id and no unhashed name exists to collide with. Same raw id always
    yields the same name.
    """
    safe = _SAFE_AGENT_ID.sub("-", agent_id)
    digest = hashlib.sha256(agent_id.encode("utf-8")).hexdigest()[:_AGENT_HASH_LEN]
    return f"completed-tasks-{safe}-{digest}.json"


class CompletedTaskStore:
    """Durable, wall-clock-bounded record of completed (task_id, agent_id) keys.

    The file is per-agent (single-writer by construction, see module docstring)
    and holds a nested ``{task_id: {agent_id: iso_wall_ts}}`` map.

    Args:
        home: The agent state home directory the JSON file lives under.
        agent_id: The agent id whose completions this store persists; it derives
            the per-agent filename.
    """

    def __init__(self, home: Path, agent_id: str) -> None:
        self._agent_id = agent_id
        self._path = (home / _agent_filename(agent_id)).resolve()
        # Set on load when the file was CORRUPT: the wall-clock instant past
        # which the distrust window closes and normal operation resumes.
        self._distrust_until: datetime | None = None

    @property
    def path(self) -> Path:
        """Return the resolved JSON dedup-state path."""
        return self._path

    @property
    def distrust_until(self) -> datetime | None:
        """Return the distrust-window expiry set on a corrupt load, else ``None``."""
        return self._distrust_until

    def in_distrust_window(self, now: datetime | None = None) -> bool:
        """Return whether the corrupt-load distrust window is still open.

        Inside the window the runner must fail closed on untracked-but-within-
        horizon frames: the durable dedup authority was provably lost, so an
        untracked frame cannot be proven a first delivery. After the window
        every redeliverable frame has aged past the horizon, so this returns
        ``False`` and normal operation resumes.
        """
        if self._distrust_until is None:
            return False
        current = now if now is not None else datetime.now(UTC)
        return current < self._distrust_until

    def load(self, horizon_s: float) -> dict[tuple[str, str], datetime]:
        """Return the persisted completions still within ``horizon_s`` (wall clock).

        A MISSING file yields an empty map with a loud INFO log (legitimate
        first boot). A CORRUPT file yields an empty map AND opens a distrust
        window of ``horizon_s`` seconds (see :meth:`in_distrust_window`),
        preserving the bad file for forensics. Entries older than the horizon
        are dropped on load.
        """
        raw = self._read_raw(horizon_s)
        now = datetime.now(UTC)
        live: dict[tuple[str, str], datetime] = {}
        damaged = False
        for task_id, agents in raw.items():
            for agent_id, iso in agents.items():
                completed = self._parse_iso(iso)
                if completed is None:
                    # Value-level damage is still damage: the dropped record's
                    # redelivery would dispatch as fresh, so it opens the same
                    # distrust window structural corruption does. Parseable
                    # entries are kept — they refuse strictly more.
                    damaged = True
                    logger.warning(
                        "completed_store.load | dropping unparseable entry | "
                        "path={} task_id={!r} agent_id={!r}",
                        self._path,
                        task_id,
                        agent_id,
                    )
                    continue
                if (now - completed).total_seconds() > horizon_s:
                    continue
                live[(task_id, agent_id)] = completed
        if damaged:
            self._enter_distrust(horizon_s, reason="unparseable completion timestamp(s)")
        return live

    def save(self, completions: dict[tuple[str, str], datetime], horizon_s: float) -> None:
        """Atomically persist ``completions`` as a nested map, evicting past-horizon.

        Write-on-completion is the conservative choice: every
        completion re-persists the horizon-pruned map, so a restart at any point
        reloads a correct, bounded dedup set. FORK NOTE: if write frequency
        becomes a concern this could batch or debounce, but a lost write would
        risk a double-dispatch, so per-completion write stands.
        """
        now = datetime.now(UTC)
        payload: dict[str, dict[str, str]] = {}
        for (task_id, agent_id), completed in completions.items():
            if (now - completed).total_seconds() > horizon_s:
                continue
            payload.setdefault(task_id, {})[agent_id] = completed.isoformat()
        self._write_atomic(payload)

    def _read_raw(self, horizon_s: float) -> dict[str, dict[str, str]]:
        if not self._path.is_file():
            logger.info(
                "completed_store.load | no dedup file yet, starting empty (first boot) | "
                "path={} agent={}",
                self._path,
                self._agent_id,
            )
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._enter_distrust(horizon_s, reason=str(exc))
            return {}
        if not isinstance(data, dict):
            self._enter_distrust(horizon_s, reason="top-level JSON is not an object")
            return {}
        nested: dict[str, dict[str, str]] = {}
        for task_id, agents in data.items():
            if not isinstance(agents, dict):
                self._enter_distrust(horizon_s, reason=f"task entry {task_id!r} is not an object")
                return {}
            nested[str(task_id)] = {str(a): str(ts) for a, ts in agents.items()}
        return nested

    def _enter_distrust(self, horizon_s: float, *, reason: str) -> None:
        """Open the distrust window and preserve the corrupt file for forensics.

        A CORRUPT file is provable integrity loss: the durable dedup authority
        cannot be trusted, so refuse untracked-but-within-horizon frames for
        ``horizon_s`` seconds (fail closed). The bad file is renamed to
        ``<name>.corrupt-<ts>`` so it is preserved for inspection and the next
        save writes a clean file rather than overwriting the evidence.
        """
        now = datetime.now(UTC)
        self._distrust_until = now + timedelta(seconds=horizon_s)
        logger.warning(
            "completed_store.load | CORRUPT dedup file; entering distrust window until {} "
            "({}s), refusing untracked within-horizon frames until then | path={} reason={}",
            self._distrust_until.isoformat(),
            horizon_s,
            self._path,
            reason,
        )
        self._preserve_corrupt(now)

    def _preserve_corrupt(self, now: datetime) -> None:
        stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
        preserved = self._path.with_name(f"{self._path.name}.corrupt-{stamp}")
        # Uniquify rather than overwrite: a same-stamp collision would silently
        # discard the earlier forensic artifact.
        counter = 1
        while preserved.exists():
            preserved = self._path.with_name(f"{self._path.name}.corrupt-{stamp}-{counter}")
            counter += 1
        try:
            os.replace(self._path, preserved)
            logger.warning(
                "completed_store.load | corrupt dedup file preserved for forensics | from={} to={}",
                self._path,
                preserved,
            )
        except OSError as exc:
            logger.error(
                "completed_store.load | could not preserve corrupt dedup file | path={} reason={}",
                self._path,
                exc,
            )

    @staticmethod
    def _parse_iso(iso: str) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(iso)
        except ValueError:
            return None
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed

    def _write_atomic(self, payload: dict[str, dict[str, str]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        blob = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self._path.parent), prefix=".completed-tasks-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(blob)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self._path)
            logger.debug(
                "completed_store.save | dedup state persisted | path={} tasks={}",
                self._path,
                len(payload),
            )
        except OSError as exc:
            logger.error("completed_store.save | failed | path={} reason={}", self._path, exc)
            try:
                os.unlink(tmp_name)
            except OSError as cleanup_exc:
                logger.warning(
                    "completed_store.save | temp cleanup failed | path={} reason={}",
                    tmp_name,
                    cleanup_exc,
                )
            raise
