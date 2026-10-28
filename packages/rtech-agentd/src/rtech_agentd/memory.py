"""Open Brain memory lane for a managed agentd session.

Thin wrapper over the reusable ``openbrain-memory`` client (MCP client +
``AgentMemory`` facade + spool + contract validation) — REUSED AS-IS, zero
server changes. This module owns only the agentd-side posture the plan (Phase 4)
requires on top of that facade:

* A single lane keyed per task/conversation (``session_start`` → lane).
* ``recall()`` returning a bounded prompt block for the persona memory slot.
* ``append_session_event`` / ``log_decision`` / ``session_wrap`` on idle, each
  with idempotency keys (the facade stamps those).
* A secret-redaction guard applied on EVERY write — content is redacted with the
  package's ported redactor before it leaves this process, and a high-confidence
  token in a write payload FAILS CLOSED (raises) rather than being silently
  shipped.
* ``get_contract()`` validation that fails closed on version/tool mismatch,
  reading the contract version from ONE pinned source
  (``openbrain_memory.CURRENT_CONTRACT_VERSION``) — never a local literal.
* "Spooled != saved" surfaced: a failed write that the facade spools is returned
  as a typed :class:`MemoryWriteResult` with ``spooled=True`` and logged at
  WARN, so the caller can drive a ``StatusFrame(state="blocked")``. It is never
  silently dropped.

The base URL defaults to the canonical ``https://openbrain.example.internal``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from loguru import logger
from openbrain_memory import (
    CURRENT_CONTRACT_VERSION,
    AgentMemory,
    MemoryContext,
    MemoryPolicy,
    OpenBrainClient,
    validate_required_memory_contract,
)
from openbrain_memory.policy import SECRET_PATTERNS, SENSITIVE_KEY_RE, redact_text

if TYPE_CHECKING:
    from openbrain_memory import MemoryClient, MemorySpool

# Canonical Open Brain endpoint (the plan pins this).
CANONICAL_BASE_URL = "https://openbrain.example.internal"

# Process-global OB env vars (single-agent / dev fallback). A per-agent suffixed
# variable (``ENV_OPENBRAIN_TOKEN`` + ``_<SUFFIX>``) ALWAYS wins over these when
# set, so one process hosting a token-bound trio never collapses onto a single
# process-global namespace. Values are secrets — never logged (value or length).
ENV_OPENBRAIN_TOKEN = "OPENBRAIN_TOKEN"
ENV_OPENBRAIN_NAMESPACE = "OPENBRAIN_NAMESPACE"
ENV_OPENBRAIN_BASE_URL = "OPENBRAIN_BASE_URL"

# The contract version this client requires is read from ONE pinned source: the
# reused package's own constant. Never re-declare it as a local literal.
REQUIRED_CONTRACT_VERSION = CURRENT_CONTRACT_VERSION

_RECALL_SOURCE_METADATA_KEYS = ("id", "path", "tier")
_RECALL_SOURCE_VALUE_MAX_CHARS = 256


class MemoryError(RuntimeError):
    """Base class for agentd memory failures."""


class MemoryContractError(MemoryError):
    """Raised when the live OB contract fails validation — fail closed.

    Carries the validation reasons (safe, no secret material) so the caller can
    log why the lane refused to open.
    """

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("Open Brain contract validation failed: " + "; ".join(reasons))


class MemoryRedactionError(MemoryError):
    """Raised when a write payload contains high-confidence secret material.

    The guard fails CLOSED: a token-shaped string never reaches Open Brain, even
    redacted — the write is refused so the leak surfaces instead of being
    silently rewritten.
    """


@dataclass(frozen=True)
class MemoryConfig:
    """Typed, immutable memory-lane configuration.

    Attributes:
        base_url: Open Brain endpoint (defaults to the canonical URL).
        token: Resolved bearer token (a secret; never logged).
        namespace: OB namespace to scope this agent's writes/reads.
        agent: Agent id (lane owner + event source).
        project: Optional project label attached to lane events.
        max_items: Recall item cap (bounds the prompt block).
        max_chars: Recall total-character cap (bounds the prompt block).
        max_item_chars: Per-item character cap.
        timeout_s: HTTP timeout for OB calls.
    """

    base_url: str
    token: str
    namespace: str
    agent: str
    project: str | None = None
    max_items: int = 8
    max_chars: int = 4000
    max_item_chars: int = 800
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("MemoryConfig.base_url must be non-empty")
        if not self.token:
            # Fail closed on missing auth — no anonymous OB writes.
            raise ValueError("MemoryConfig.token must be non-empty (fail closed on auth)")
        if not self.namespace:
            raise ValueError("MemoryConfig.namespace must be non-empty")
        if not self.agent:
            raise ValueError("MemoryConfig.agent must be non-empty")

    def policy(self) -> MemoryPolicy:
        """Return the recall bounding policy derived from this config."""
        return MemoryPolicy(
            max_items=self.max_items,
            max_chars=self.max_chars,
            max_item_chars=self.max_item_chars,
        )


@dataclass(frozen=True)
class MemoryWriteResult:
    """Outcome of one memory write.

    ``saved`` and ``spooled`` are mutually exclusive: a write either reached Open
    Brain (``saved``) or failed and was spooled for later replay (``spooled``).
    A caller surfaces ``spooled`` as a ``StatusFrame(state="blocked")`` — it is
    never silently dropped.
    """

    operation: str
    saved: bool
    spooled: bool
    detail: str | None = None


@dataclass(frozen=True)
class MemoryRecallSource:
    """Describe one safe OB source without recalled text or raw payload data."""

    label: str
    source: str | None
    kind: str | None
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class MemoryRecallResult:
    """Carry a bounded prompt block and its matching safe source descriptors."""

    block: str
    sources: tuple[MemoryRecallSource, ...] = ()
    status: Literal["success", "failed"] = "success"
    _legacy_block: str = field(default="", repr=False, compare=False)


def _bounded_safe_source_value(value: object) -> str | None:
    """Return one bounded, redacted scalar source value, or ``None``."""
    if not isinstance(value, (str, int, float, bool)):
        return None
    safe = redact_text(str(value)).strip()
    if not safe:
        return None
    if len(safe) <= _RECALL_SOURCE_VALUE_MAX_CHARS:
        return safe
    return safe[: _RECALL_SOURCE_VALUE_MAX_CHARS - 3] + "..."


def _recall_source(item: object, label: str) -> MemoryRecallSource:
    """Build one safe source descriptor from an ``openbrain-memory`` item."""
    source = _bounded_safe_source_value(getattr(item, "source", None))
    kind = _bounded_safe_source_value(getattr(item, "kind", None))
    raw_metadata = getattr(item, "metadata", {})
    metadata: list[tuple[str, str]] = []
    if isinstance(raw_metadata, Mapping):
        for key in _RECALL_SOURCE_METADATA_KEYS:
            safe_value = _bounded_safe_source_value(raw_metadata.get(key))
            if safe_value is not None:
                metadata.append((key, safe_value))
    return MemoryRecallSource(
        label=label,
        source=source,
        kind=kind,
        metadata=tuple(metadata),
    )


def _guard_write_content(operation: str, *values: str | None) -> None:
    """Fail closed if any write value carries high-confidence secret material.

    Uses the reused package's high-confidence ``SECRET_PATTERNS`` (the same set
    that drives its fail-closed receipt reject gate) — NOT the aggressive
    heuristic redactors, which would false-trip on benign paths. A match raises
    :class:`MemoryRedactionError` so a token never ships, even redacted.

    Args:
        operation: The write operation name (for the log/guard-trigger line).
        values: Free-text values about to be written to Open Brain.

    Raises:
        MemoryRedactionError: If any value matches a high-confidence secret
            pattern.
    """
    for value in values:
        if not value:
            continue
        if any(pattern.search(value) for pattern in SECRET_PATTERNS):
            logger.warning(
                "memory.guard | secret-shaped material refused | op={} (fail closed)",
                operation,
            )
            raise MemoryRedactionError(
                f"write '{operation}' refused: content contains secret-like material"
            )


def _guard_metadata(operation: str, metadata: Mapping[str, object]) -> None:
    """Fail closed if metadata keys/values carry secret material.

    Args:
        operation: The write operation name (for the guard-trigger line).
        metadata: The metadata mapping about to be written.

    Raises:
        MemoryRedactionError: On a sensitive key name or secret-shaped value.
    """
    for key, value in metadata.items():
        if SENSITIVE_KEY_RE.search(str(key)):
            logger.warning(
                "memory.guard | sensitive metadata key refused | op={} key={} (fail closed)",
                operation,
                key,
            )
            raise MemoryRedactionError(
                f"write '{operation}' refused: metadata key '{key}' is sensitive"
            )
        if isinstance(value, str):
            _guard_write_content(operation, value)


class AgentMemoryLane:
    """A single Open Brain memory lane for one managed agent session.

    Wraps :class:`openbrain_memory.AgentMemory`. Every write passes the
    redaction guard first; recall returns a bounded prompt block; the contract is
    validated fail-closed at lane open.
    """

    def __init__(
        self,
        config: MemoryConfig,
        *,
        client: MemoryClient | None = None,
        spool: MemorySpool | None = None,
    ) -> None:
        """Build a lane.

        Args:
            config: Typed memory-lane configuration.
            client: Injected memory client (defaults to a live
                :class:`OpenBrainClient`). Injected under test.
            spool: Optional durable spool for failed writes; when present, a
                failed write is spooled and surfaced (never silently lost).
        """
        self._config = config
        self._client: MemoryClient = client or OpenBrainClient(
            base_url=config.base_url,
            token=config.token,
            namespace=config.namespace,
            agent_id=config.agent,
            timeout=config.timeout_s,
        )
        self._memory = AgentMemory(
            client=self._client,
            agent=config.agent,
            project=config.project,
            policy=config.policy(),
            spool=spool,
        )
        self._spool = spool
        logger.debug(
            "memory.lane | built | agent={} namespace={} base_url={}",
            config.agent,
            config.namespace,
            config.base_url,
        )

    @property
    def conversation_key(self) -> str | None:
        """Return the active lane key, or ``None`` before ``session_start``."""
        return self._memory.conversation_key

    @property
    def max_items(self) -> int:
        """Return the configured recall item cap (bounds the prompt block)."""
        return self._config.max_items

    def bound_http_timeout(self, seconds: float) -> None:
        """Lower the underlying OB HTTP timeout to a per-turn recall deadline.

        The per-turn recall runs the blocking OB request on a worker thread under
        an ``asyncio.wait_for`` deadline; ``wait_for`` cancels the AWAIT but not
        the worker thread, so the thread stays occupied until the client's own
        (much larger) HTTP timeout elapses. Clamping the client timeout DOWN to
        the recall deadline lets the worker thread unwind at (or near) the same
        deadline instead of outliving it by the full configured timeout.

        This is a best-effort hardening tweak, never a turn-failing operation: it
        no-ops on a non-positive ``seconds``, on a client that does not expose a
        numeric ``timeout``, or when the requested value would RAISE (never
        lower) the current timeout. Any failure to set it is swallowed and logged
        so a turn is never failed on this tweak. No secrets are logged.

        Args:
            seconds: The hard recall deadline to clamp the HTTP timeout to. A
                value ``<= 0`` is a no-op.
        """
        if seconds <= 0:
            return
        current = getattr(self._client, "timeout", None)
        if not isinstance(current, (int, float)) or isinstance(current, bool):
            return
        if seconds >= current:
            return
        try:
            self._client.timeout = seconds  # type: ignore[attr-defined]
        except Exception as exc:  # best-effort hardening; never fail a turn on it
            logger.warning(
                "memory.bound_http_timeout | failed | agent={} reason={}",
                self._config.agent,
                type(exc).__name__,
            )
            return
        logger.info(
            "memory.bound_http_timeout | lowered | agent={} timeout_s={}",
            self._config.agent,
            seconds,
        )

    def session_start(
        self,
        conversation_key: str,
        *,
        channel_id: str | None = None,
        thread_id: str | None = None,
        topic: str | None = None,
        validate_contract: bool = True,
    ) -> None:
        """Open the lane keyed per task/conversation, contract-checked first.

        Validates the live Open Brain contract (version + required tools) and
        FAILS CLOSED on mismatch before any write. The contract version is read
        from the single pinned source, never a local literal.

        Args:
            conversation_key: Stable per-task/conversation lane key.
            channel_id: Optional originating channel id.
            thread_id: Optional originating thread id.
            topic: Optional lane topic label.
            validate_contract: When true (default), validate the contract before
                opening the lane. Disable ONLY in tests with a client that has no
                ``get_contract``.

        Raises:
            MemoryContractError: If the live contract fails validation.
        """
        logger.debug("memory.session_start | key={}", conversation_key)
        if validate_contract:
            self._validate_contract()
        metadata: dict[str, str] = {}
        if channel_id is not None:
            metadata["channel_id"] = channel_id
        if thread_id is not None:
            metadata["thread_id"] = thread_id
        if topic is not None:
            metadata["topic"] = topic
        self._memory.start_session(conversation_key, **metadata)
        logger.info(
            "memory.session_start | ok | agent={} key={}",
            self._config.agent,
            conversation_key,
        )

    def _validate_contract(self) -> None:
        """Fetch + validate the live OB contract; fail closed on mismatch.

        Raises:
            MemoryContractError: On a missing contract, a version/tool mismatch,
                or a transport failure (fail closed — no lane without a verified
                contract).
        """
        get_contract = getattr(self._client, "get_contract", None)
        if get_contract is None:
            logger.error("memory.contract | client has no get_contract (fail closed)")
            raise MemoryContractError(("client does not expose get_contract",))
        try:
            manifest = get_contract()
        except Exception as exc:  # fail closed on any transport/protocol failure
            logger.error("memory.contract | fetch failed | reason={} (fail closed)", exc)
            raise MemoryContractError((f"get_contract failed: {type(exc).__name__}",)) from exc
        result = validate_required_memory_contract(manifest)
        if not result.ok:
            logger.error(
                "memory.contract | mismatch | required={} reasons={} (fail closed)",
                REQUIRED_CONTRACT_VERSION,
                result.reasons,
            )
            raise MemoryContractError(result.reasons)
        logger.info("memory.contract | ok | version={}", REQUIRED_CONTRACT_VERSION)

    def recall(self, query: str, *, limit: int | None = None) -> str:
        """Return a bounded recall block for the persona memory slot.

        The block is bounded by the configured :class:`MemoryPolicy` (item count,
        total chars, per-item chars) AND run through the redactor, so no
        secret-shaped material surfaces in the prompt.

        Args:
            query: Recall query text.
            limit: Optional item cap (clamped to the policy max).

        Returns:
            A bounded, redacted prompt-text block (empty string when nothing is
            recalled or on a recall failure — recall is best-effort and never
            fails the turn).
        """
        return self.recall_with_sources(query, limit=limit)._legacy_block

    def recall_with_sources(self, query: str, *, limit: int | None = None) -> MemoryRecallResult:
        """Return bounded labeled recall plus safe ``OB<n>`` source descriptors."""
        try:
            context: MemoryContext = self._memory.recall(query, limit=limit)
        except Exception as exc:  # recall is best-effort; never fail the turn on it
            logger.warning("memory.recall | failed | reason={} (empty result)", type(exc).__name__)
            return MemoryRecallResult(block="", status="failed")

        lines: list[str] = []
        sources: list[MemoryRecallSource] = []
        used_chars = 0
        for index, item in enumerate(context.items, start=1):
            label = f"OB{index}"
            prefix = f"- [{label}] "
            separator_chars = 1 if lines else 0
            remaining = self._config.max_chars - used_chars - separator_chars - len(prefix)
            if remaining <= 0:
                break
            safe_text = redact_text(item.text)
            if len(safe_text) > remaining:
                if remaining <= 3:
                    safe_text = safe_text[:remaining]
                else:
                    safe_text = safe_text[: remaining - 3] + "..."
            line = prefix + safe_text
            lines.append(line)
            sources.append(_recall_source(item, label))
            used_chars += separator_chars + len(line)

        block = "\n".join(lines)
        logger.info(
            "memory.recall | ok | items={} block_len={}",
            len(sources),
            len(block),
        )
        return MemoryRecallResult(
            block=block,
            sources=tuple(sources),
            _legacy_block=redact_text(context.as_prompt_text()),
        )

    def append_session_event(
        self,
        content: str,
        *,
        event_type: str = "fact",
        importance: str | None = None,
        **metadata: str,
    ) -> MemoryWriteResult:
        """Append a durable lane event with an idempotency key.

        Args:
            content: Event content (guarded + redacted before write).
            event_type: One of the facade's supported event types.
            importance: Optional importance tier.
            metadata: Extra event metadata (guarded for secret keys/values).

        Returns:
            A :class:`MemoryWriteResult`; ``spooled=True`` when the write failed
            and was spooled (surfaced, never silent).

        Raises:
            MemoryRedactionError: If content or metadata carries secret material.
        """
        _guard_write_content("append_session_event", content)
        _guard_metadata("append_session_event", metadata)
        safe_content = redact_text(content)
        extra: dict[str, str] = {"event_type": event_type, **metadata}
        if importance is not None:
            extra["importance"] = importance
        return self._write(
            "append_session_event",
            lambda: self._memory.append_event(self._config.agent, safe_content, **extra),
        )

    def log_decision(
        self,
        text: str,
        *,
        alternatives: list[str] | None = None,
        context: str | None = None,
    ) -> MemoryWriteResult:
        """Log a durable decision with an idempotency key.

        Args:
            text: Decision rationale (guarded + redacted before write).
            alternatives: Optional alternatives considered.
            context: Optional decision context.

        Returns:
            A :class:`MemoryWriteResult`; ``spooled=True`` on a spooled failure.

        Raises:
            MemoryRedactionError: If any field carries secret material.
        """
        _guard_write_content("log_decision", text, context, *(alternatives or ()))
        safe_text = redact_text(text)
        extra: dict[str, object] = {}
        if alternatives is not None:
            extra["alternatives"] = [redact_text(item) for item in alternatives]
        if context is not None:
            extra["context"] = redact_text(context)
        return self._write(
            "log_decision",
            lambda: self._memory.remember_decision(safe_text, **extra),
        )

    def session_wrap(
        self,
        summary: str,
        *,
        key_decisions: list[str] | None = None,
        next_steps: list[str] | None = None,
    ) -> MemoryWriteResult:
        """Checkpoint the lane on idle with a durable summary.

        Args:
            summary: Session summary (guarded + redacted before write).
            key_decisions: Optional key decisions list.
            next_steps: Optional next steps list.

        Returns:
            A :class:`MemoryWriteResult`; ``spooled=True`` on a spooled failure.

        Raises:
            MemoryRedactionError: If any field carries secret material.
        """
        _guard_write_content(
            "session_wrap",
            summary,
            *(key_decisions or ()),
            *(next_steps or ()),
        )
        safe_summary = redact_text(summary)
        extra: dict[str, object] = {}
        if key_decisions is not None:
            extra["key_decisions"] = [redact_text(item) for item in key_decisions]
        if next_steps is not None:
            extra["next_steps"] = [redact_text(item) for item in next_steps]
        return self._write(
            "session_wrap",
            lambda: self._memory.wrap_session(safe_summary, **extra),
        )

    def _write(self, operation: str, call: Callable[[], object]) -> MemoryWriteResult:
        """Run a guarded write, surfacing a spooled failure instead of hiding it.

        The facade spools on failure and re-raises; we catch, decide whether the
        payload was spooled (spool present), log at WARN, and return a typed
        result. A leaked write is never silent.

        Args:
            operation: Write operation name (for logs/result).
            call: Zero-arg callable performing the facade write.

        Returns:
            A :class:`MemoryWriteResult` describing the outcome.
        """
        try:
            call()
        except Exception as exc:
            spooled = self._spool is not None
            logger.warning(
                "memory.write | {} | op={} reason={} (spooled != saved surfaced)",
                "spooled" if spooled else "lost",
                operation,
                type(exc).__name__,
            )
            return MemoryWriteResult(
                operation=operation,
                saved=False,
                spooled=spooled,
                detail=type(exc).__name__,
            )
        logger.info("memory.write | saved | op={}", operation)
        return MemoryWriteResult(operation=operation, saved=True, spooled=False)


# Agentd deploy convention (mirrors `frame_turn._short_name`): a bus id is
# `agentd-<name>` optionally suffixed by the stack lane (`-dev`/`-prod`).
# The per-agent env suffix is derived from that <name>, so `agentd-nimbus` and
# `agentd-nimbus-dev` both resolve to the SAME `NIMBUS` suffix and read the
# same operator-provisioned OPENBRAIN_*_NIMBUS vars regardless of stack lane.
_AGENT_ID_PREFIX = "agentd-"
_AGENT_ID_SUFFIXES = ("-dev", "-prod")


def agent_env_suffix(agent_id: str) -> str:
    """Return the deterministic per-agent env-var suffix for `agent_id`.

    Strips the `agentd-` convention prefix and a trailing `-dev`/`-prod`
    stack lane, then normalizes the remainder to an env-var-safe token: uppercased
    with every run of non-alphanumeric characters collapsed to a single `_` and
    leading/trailing `_` trimmed. So `agentd-nimbus` and `agentd-nimbus-dev`
    both yield `NIMBUS`; `agentd-my.agent` yields `MY_AGENT`.

    Args:
        agent_id: The agent's bus id.

    Returns:
        The uppercase env-var suffix (may be empty for a degenerate id, in which
        case :func:`build_agent_memory_lane` uses only the process-global vars).
    """
    name = agent_id.removeprefix(_AGENT_ID_PREFIX)
    for suffix in _AGENT_ID_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    normalized = "".join(char if char.isalnum() else "_" for char in name.upper())
    return "_".join(part for part in normalized.split("_") if part)


def _resolve_per_agent(
    env: Mapping[str, str], base_key: str, suffix: str
) -> tuple[str, str | None]:
    """Resolve one OB setting with per-agent precedence over the global var.

    Precedence: the suffixed per-agent var (`<base_key>_<SUFFIX>`) wins when set
    (non-blank); otherwise the process-global `<base_key>` is the single-agent /
    dev fallback. Values are NEVER logged — only WHICH source won (the var name).

    Args:
        env: The environment mapping to read.
        base_key: The un-suffixed OB env var name (e.g. `OPENBRAIN_TOKEN`).
        suffix: The per-agent suffix from :func:`agent_env_suffix` (may be empty).

    Returns:
        A `(value, source_key)` pair where `value` is the resolved value
        (empty string when neither source is set) and `source_key` is the env
        var name it came from, or `None` when unset.
    """
    if suffix:
        per_agent_key = f"{base_key}_{suffix}"
        per_agent = env.get(per_agent_key, "").strip()
        if per_agent:
            return per_agent, per_agent_key
    glob = env.get(base_key, "").strip()
    if glob:
        return glob, base_key
    return "", None


def build_agent_memory_lane(
    agent_id: str,
    *,
    env: Mapping[str, str] | None = None,
    client: MemoryClient | None = None,
    spool: MemorySpool | None = None,
) -> AgentMemoryLane | None:
    """Build one per-agent OB recall lane from the environment, or `None`.

    The single owning factory for per-turn recall wiring (Phase 4): both the
    standalone `__main__` entrypoint and the in-process dev stack call THIS so a
    process hosting a token-bound trio derives one token-scoped lane per agent
    instead of collapsing onto a single process-global `OPENBRAIN_TOKEN`.

    Resolution, per setting, with the per-agent suffixed var winning over the
    process-global var (see :func:`_resolve_per_agent`):

    * token: `OPENBRAIN_TOKEN_<SUFFIX>` then `OPENBRAIN_TOKEN`.
    * namespace: `OPENBRAIN_NAMESPACE_<SUFFIX>` then `OPENBRAIN_NAMESPACE`,
      defaulting to a per-agent `rtech.agentd.<agent_id>` scope.
    * base url: `OPENBRAIN_BASE_URL_<SUFFIX>` then `OPENBRAIN_BASE_URL`,
      defaulting to :data:`CANONICAL_BASE_URL`.

    Recall is OPT-IN on a resolved token: a missing token (neither per-agent nor
    global) DISABLES recall for THIS agent only — it warns once, loudly, and
    returns `None` so a co-hosted agent WITH a token still gets its lane. The
    token is a secret and is NEVER logged (value or length); only the resolving
    var name and the non-secret namespace/base url are logged.

    Args:
        agent_id: This agent's id (lane owner + namespace default + suffix source).
        env: The environment mapping to read (defaults to `os.environ`).
        client: Injected memory client (defaults to a live client; injected under
            test to stay offline).
        spool: Optional durable spool for failed writes.

    Returns:
        A wired :class:`AgentMemoryLane`, or `None` when recall is disabled for
        this agent (no token) or the lane could not be built.
    """
    source = os.environ if env is None else env
    suffix = agent_env_suffix(agent_id)
    token, token_key = _resolve_per_agent(source, ENV_OPENBRAIN_TOKEN, suffix)
    if not token:
        logger.warning(
            "agentd.memory | per-turn OB recall DISABLED: no {} (nor {}_{}) set | agent={}",
            ENV_OPENBRAIN_TOKEN,
            ENV_OPENBRAIN_TOKEN,
            suffix or "<none>",
            agent_id,
        )
        return None
    namespace, _ = _resolve_per_agent(source, ENV_OPENBRAIN_NAMESPACE, suffix)
    namespace = namespace or f"rtech.agentd.{agent_id}"
    base_url, _ = _resolve_per_agent(source, ENV_OPENBRAIN_BASE_URL, suffix)
    base_url = base_url or CANONICAL_BASE_URL
    try:
        config = MemoryConfig(
            base_url=base_url,
            token=token,
            namespace=namespace,
            agent=agent_id,
        )
        lane = AgentMemoryLane(config, client=client, spool=spool)
    except Exception as exc:
        logger.error(
            "agentd.memory | lane build failed; per-turn OB recall DISABLED | "
            "agent={} namespace={} base_url={} reason={}",
            agent_id,
            namespace,
            base_url,
            type(exc).__name__,
        )
        return None
    logger.info(
        "agentd.memory | per-turn OB recall ENABLED | agent={} token_src={} namespace={} "
        "base_url={}",
        agent_id,
        token_key,
        namespace,
        base_url,
    )
    return lane


__all__ = [
    "CANONICAL_BASE_URL",
    "ENV_OPENBRAIN_BASE_URL",
    "ENV_OPENBRAIN_NAMESPACE",
    "ENV_OPENBRAIN_TOKEN",
    "REQUIRED_CONTRACT_VERSION",
    "AgentMemoryLane",
    "MemoryConfig",
    "MemoryContractError",
    "MemoryError",
    "MemoryRedactionError",
    "MemoryRecallResult",
    "MemoryRecallSource",
    "MemoryWriteResult",
    "agent_env_suffix",
    "build_agent_memory_lane",
]
