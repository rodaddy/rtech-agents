"""Define typed NATS wire frames.

The rtech wire frames are the additive product layer and their serialized bytes
are wire-locked (live deployments speak the current format). They deliberately do
NOT subclass ``fleet_core.spec`` models: fleet-core's ``Stance`` serializes by
alias (``stance``/``reason`` plus a ``fact_pull`` key) and constrains ``position``
to the ``StancePosition`` enum, so rebasing rtech's frames onto it would rewrite
every TurnFrame's bytes. Interop with fleet-core is provided compositionally by
``stance_to_fleet`` without changing the rtech wire.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Literal

from fleet_core.spec import Stance as FleetStance
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

SCHEMA_VERSION = "9"

# Scope literal -> required participant count. Scope names a COUNT, not
# identities; this is the one shared source of that mapping (done derivation and
# edges both consume it).
SCOPE_PARTICIPANT_COUNT: dict[str, int] = {
    "single": 1,
    "duo": 2,
    "trio": 3,
}

# NATS subject tokens must not contain whitespace, ``.``, ``*``, or ``>``.
# Participant ids become subject tokens (``rtech.agent.<id>.task``), so the
# wire enforces a conservative safe alphabet.
_AGENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

# Bounded cap for a presence-transition ``agent_id`` (issue #147). It is a persona
# slug that also rides as a Dash live-page URL path segment; bound it with the same
# 64-char cap convention the evidence source ids use (``MAX_SOURCE_ID_LEN``) so an
# over-long slug can neither blow the frame nor the URL. Its own named constant so
# the bound is documented and not a re-used magic literal.
MAX_PRESENCE_AGENT_ID_LEN = 64

# Bounded SAFE human display vocabulary for a rendered presence ``display_name``
# (issue #147): letters, digits, spaces, dot, hyphen, underscore ONLY. It rejects
# ``@`` (mentions), brackets/parentheses, slashes, backticks, any markup char, and
# control characters. Leading/trailing whitespace is rejected separately in the
# model validator. A connector renders this string into a possibly-leaked surface,
# so the wire constrains it to a vocabulary that cannot carry a Discord mention, a
# Markdown link, or a Telegram HTML tag.
_DISPLAY_NAME_PATTERN = re.compile(r"^[A-Za-z0-9 ._-]+$")
# Max rendered display name length (same 64-char human cap as the slug).
MAX_PRESENCE_DISPLAY_NAME_LEN = 64

# Bounded cap + safe alphabet for a presence-transition ``transition_id`` (issue
# #147 delivery idempotency). The deriver mints ONE stable id per edge when the
# pending frame is created (the repo's ``uuid4().hex`` idiom -> 32 hex chars) and
# retains it on retry, so an ambiguous at-least-once redelivery of the SAME edge
# carries the SAME id and a consumer renders it exactly once. It is NOT rendered to
# any operator surface — a delivery-idempotency key only — but it rides the wire and
# enters a consumer's bounded seen-id cache, so the wire bounds it to a short
# subject-safe token (``[A-Za-z0-9_-]+``, <= 64) so an oversized/unsafe id can
# neither blow the frame nor poison the cache key space. The typed frame stays truth.
_TRANSITION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
MAX_PRESENCE_TRANSITION_ID_LEN = 64


def is_valid_presence_agent_id(agent_id: str) -> bool:
    """Return True iff ``agent_id`` is a valid presence-transition authoritative id.

    The ONE shared identity predicate for a presence ``agent_id`` (issue #147): a
    subject-safe ``[A-Za-z0-9_-]+`` token within :data:`MAX_PRESENCE_AGENT_ID_LEN`.
    :meth:`PresenceTransitionFrame._check_identity` enforces exactly these
    constraints at construction/decode; the deriver calls it BEFORE frame
    construction so ONE malformed/unsupported id is skipped, not a whole tick.
    """

    return (
        _AGENT_ID_PATTERN.fullmatch(agent_id) is not None
        and len(agent_id) <= MAX_PRESENCE_AGENT_ID_LEN
    )


def validate_participants(participants: list[str], scope: str) -> None:
    """Enforce the scope/participant invariant shared by frames and edges.

    Every participant id must be non-blank, unique, and NATS-subject-safe
    (``[A-Za-z0-9_-]+`` — ids become subject tokens). A non-empty participant
    list must match the scope's required count exactly; ``duo``/``trio`` REQUIRE
    identities (an identity-less multi-agent task can never derive done). An
    empty list is valid only for ``single``, where the orchestrator routes to
    its configured default target.

    Args:
        participants: The participant agent ids to validate.
        scope: The task scope literal the ids must be consistent with.

    Raises:
        ValueError: On a blank/unsafe/duplicate id or a scope/count mismatch.
    """
    for participant in participants:
        if not _AGENT_ID_PATTERN.fullmatch(participant):
            raise ValueError(
                f"participant id {participant!r} is not a safe NATS subject token ([A-Za-z0-9_-]+)"
            )
    if len(set(participants)) != len(participants):
        raise ValueError(f"participants must be unique, got {participants!r}")
    required = SCOPE_PARTICIPANT_COUNT.get(scope)
    if required is None:
        raise ValueError(f"unknown scope {scope!r}")
    if not participants:
        if scope != "single":
            raise ValueError(f"scope {scope!r} requires {required} named participants, got none")
        return
    if len(participants) != required:
        raise ValueError(
            f"scope {scope!r} requires exactly {required} participants, got {len(participants)}"
        )


# The one broadcast wire token for ``SteerFrame.target_agent`` (issue #74/#76):
# edges default to it when the human names no agent, and consumers honor it as
# "any agent with the steer's task active". Owned here — the producer/consumer
# boundary imports this constant, never re-declares the literal.
BROADCAST_TARGET = "all"

# Evidence-gated verification bounds (evidence-gated responses). A verification
# report rides the answering TurnFrame, so the model cannot flood the wire (or a
# rendered "Sources checked" block) with unbounded evidence. These caps are the
# ONE owned source of the verification field bounds; the producer that builds a
# report and any consumer that renders it import them, never re-declare the
# literals. Individual string fields are additionally length-bounded at the
# model so a single label/locator/limitation cannot blow the frame size.
MAX_EVIDENCE_SOURCES = 12
MAX_VERIFICATION_LIMITATIONS = 8
MAX_VERIFICATION_METHODS = 12
MAX_SOURCE_ID_LEN = 64
MAX_SOURCE_LABEL_LEN = 200
MAX_SOURCE_LOCATOR_LEN = 512
_MAX_LIMITATION_LEN = 300
_MAX_METHOD_LEN = 80

# Evidence source kinds (where a checked fact came from) and authority tiers
# (how much a source outranks others in a conflict). Both are typed literals so
# a connector renders evidence provenance from a fixed vocabulary, never scraped
# prose. Authority order lives with the runtime that consumes it; the wire only
# constrains the value set.
EvidenceKind = Literal["open_brain", "repo", "file", "url", "live", "user_context"]
EvidenceAuthority = Literal["live", "source", "memory", "user", "unknown"]
VerificationStatus = Literal["grounded", "partial", "unverified", "not_applicable"]
ClaimKind = Literal["factual", "mixed", "opinion", "social", "action"]


class WireFrame(BaseModel):
    """Provide strict shared frame validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION, min_length=1)


class ChannelOrigin(BaseModel):
    """Carry the chat-surface identity a task came from (issue #93 intake).

    Rides a :class:`TaskFrame` so a downstream triage agent knows both the Open
    Brain channel lane (guild + channel) and the exact reply destination
    (channel / thread / the originating message to reply to). All ids are
    opaque, stringified surface ids; ``asker`` is the opaque author id (never a
    display name).

    Additive-optional on :class:`TaskFrame` (default ``None``): frames minted
    before intake omit it and decode unchanged. The intake edge always sets it so
    the reply path can answer in the SAME channel the question arrived in.

    Deploy ordering (wire compat): the ``origin`` key rides an ``extra="forbid"``
    frame, so EVERY consumer must upgrade rtech-frames (to the SCHEMA_VERSION "4"
    build that knows the field) BEFORE any producer starts stamping ``origin``.
    Upgrade consumers first, then producers — a producer that stamps ``origin``
    against an un-upgraded consumer is rejected at decode.

    The additive ``debate`` render marker (trio intake) bumped SCHEMA_VERSION to
    "6". Like ``origin`` it is dropped-on-default (see :meth:`_serialize`), so a
    non-debate origin is byte-identical to the "5" wire and an old
    ``extra="forbid"`` decoder without the field still accepts it; the bump
    records the additive field, not a breaking change. The later additive
    ``TaskFrame.arbiter`` debate-role field bumped SCHEMA_VERSION to "7" on the
    same dropped-on-None additive-safe pattern. The later additive
    ``TurnFrame.verification`` / ``StatusFrame.phase`` evidence-gated fields
    bumped SCHEMA_VERSION to "8" on that same dropped-on-None additive-safe
    pattern, consumers-before-producers.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    channel_id: str = Field(min_length=1)
    guild_id: str | None = None
    thread_id: str | None = None
    message_id: str | None = None
    asker: str | None = None
    # Render-intent marker (issue #93 trio intake): True only when this origin
    # rides a MULTI-participant debate task (duo/trio). A triage edge renders a
    # debate answer as a short stance-derived TLDR in the parent channel plus the
    # full turn text in-thread, instead of the full text in-channel used for a
    # single-scope answer. It is a typed render DECISION carried on the frame —
    # never inferred from surface geometry (a single-scope ask made inside a
    # thread must NOT be mis-rendered as a debate), which is why it is its own
    # field rather than an overload of ``thread_id``. Additive-optional and
    # dropped-on-default on the wire (see :meth:`_serialize`) so a legacy origin is
    # byte-identical and an old ``extra="forbid"`` decoder without the field still
    # accepts it — same consumers-before-producers deploy ordering as ``origin``.
    debate: bool = False

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Omit the ``debate`` key from the wire when it is the ``False`` default.

        Mirrors :meth:`TaskFrame._serialize`/:meth:`TurnFrame._serialize`: the
        additive field is the ONLY key dropped-on-default, so a non-debate origin
        (the single-scope intake path and every pre-debate frame) serializes
        byte-identically to the pre-debate wire and an old ``extra="forbid"``
        decoder without the field still accepts it. When ``debate`` is True it
        rides normally and round-trips.
        """
        data: dict[str, Any] = handler(self)
        if not self.debate:
            data.pop("debate", None)
        return data


class TaskFrame(WireFrame):
    """Carry a user task from an edge onto the bus.

    ``participants`` names the specific agent ids a duo/trio task fans out to
    (typed field, never scraped from ``text``). An empty list means the
    orchestrator routes a single-scope task to its configured default target.
    ``scope`` names the required COUNT; ``participants`` names the identities —
    the edge derives scope from the participant count and validates consistency.

    ``stack`` is the typed stack-scope guard (issue #75): the edge stamps the
    stack id of the deployment it fronts, and an orchestrator only consumes
    tasks whose ``stack`` matches its own configured id — so dev and deployed
    stacks sharing one bus cannot cross-consume. Additive with a default of
    ``""`` (unscoped legacy wire) so pre-#75 bytes still decode; stack-guarded
    orchestrators fail closed on any mismatch, including ``""``.

    ``origin`` carries the originating chat-surface identity (issue #93 intake):
    an always-on channel listener mints a task the triage agent must be able to
    answer back in-channel, so the guild/channel/thread/message ids and asker
    ride the frame. Additive-optional (default ``None``); non-intake frames omit
    it and decode unchanged.

    ``arbiter`` names the debate ROLE authority: on a multi-participant (duo/trio)
    debate task it is the participant that MODERATES and writes the final TLDR
    ruling instead of arguing the substance itself; the other participants each
    argue their own position and engage sibling turns. It is a typed role DECISION
    carried on the frame (the intake edge stamps it with the triage agent id),
    never inferred from ``text`` or from participant ordering. Additive-optional
    (default ``None``); when set it MUST be one of ``participants`` (a
    non-participant cannot moderate a debate it is not in — enforced even when
    ``participants`` is empty, so a stray arbiter on a single-scope frame fails
    closed). It bumped SCHEMA_VERSION to "7"; like ``origin`` it is
    dropped-on-None on the wire (see :meth:`_serialize`), so a non-arbiter frame
    carries no ``arbiter`` key and differs from the "6" wire ONLY in the
    ``schema_version`` stamp — an old ``extra="forbid"`` decoder without the
    field (and without a version pin) still accepts it. The bump records the
    additive field, not a breaking change. Same consumers-before-producers
    deploy ordering as ``origin``.
    """

    id: str = Field(min_length=1)
    origin_edge: str = Field(min_length=1)
    requester: str = Field(min_length=1)
    text: str = Field(min_length=1)
    scope: Literal["single", "duo", "trio"]
    participants: list[str] = Field(default_factory=list)
    stack: str = ""
    created_at: datetime | None = None
    origin: ChannelOrigin | None = None
    arbiter: str | None = None

    @model_validator(mode="after")
    def _check_participants(self) -> TaskFrame:
        """Reject blank/duplicate/subject-unsafe ids and scope/count mismatches.

        The wire enforces the invariant (issue #75 round 2): a duplicate like
        ``['a', 'a']`` would derive ``duo``, publish twice to one agent lane,
        and collapse to a one-member dispatch set that can never satisfy duo
        completion — so it never validates in the first place.

        ``arbiter``, when set on a task with named participants, MUST be one of
        those participants: a debate is moderated by one of its OWN debaters, so a
        non-participant arbiter is a mint bug and fails closed at decode.
        """
        validate_participants(self.participants, self.scope)
        if self.arbiter is not None and self.arbiter not in self.participants:
            raise ValueError(
                f"arbiter {self.arbiter!r} must be one of participants {self.participants!r}"
            )
        return self

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Omit the ``origin``/``arbiter`` keys from the wire when they are None.

        A legacy-shaped frame (no ``origin``, no ``arbiter``) MUST serialize
        byte-identically to the pre-#93 wire so an old ``extra="forbid"`` decoder
        without the fields still accepts it: ``origin`` and ``arbiter`` are the
        ONLY keys dropped-on-None. Every other Optional field's wire behavior is
        left exactly as the base serializer emits it (e.g. ``created_at`` still
        rides as ``null`` — it exists on the legacy decoder). When set, each rides
        normally and round-trips.
        """
        data: dict[str, Any] = handler(self)
        if self.origin is None:
            data.pop("origin", None)
        if self.arbiter is None:
            data.pop("arbiter", None)
        return data


class Stance(BaseModel):
    """Capture an agent's stated position and collaboration needs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    position: str = Field(min_length=1)
    because: str = Field(min_length=1)
    needs_from: list[str] = Field(default_factory=list)


def stance_to_fleet(stance: Stance) -> FleetStance:
    """Map an rtech ``Stance`` onto the fleet-core ``Stance`` base shape.

    The two models share their semantic fields (``position``/``because``/
    ``needs_from``); fleet-core's ``Stance`` uses ``populate_by_name`` so its
    Python field names accept rtech's values directly. This bridge lets a
    fleet-core client consume rtech stance data through the shared vocabulary
    without rtech emitting the fleet alias wire (``stance``/``reason``/
    ``fact_pull``). ``position`` must be a fleet ``StancePosition`` value
    (``agree``/``dissent``/``neutral``); other rtech positions are additive and
    have no fleet-core base equivalent.

    Args:
        stance: The rtech stance to project onto the fleet-core base model.

    Returns:
        A validated fleet-core ``Stance`` carrying the same position, reason,
        and dependency identities.

    Raises:
        pydantic.ValidationError: If ``position`` is outside the fleet-core
            ``StancePosition`` vocabulary.
    """

    return FleetStance.model_validate(
        {
            "position": stance.position,
            "because": stance.because,
            "needs_from": list(stance.needs_from),
        }
    )


class EvidenceSource(BaseModel):
    """Name one authoritative source a factual claim was checked against.

    A bounded, safe descriptor of evidence — never the recalled content itself.
    It rides a :class:`VerificationReport` so a connector can render a compact
    "Sources checked" block from typed fields, and so the runtime can validate
    that the model only cited sources it was actually given. Fields are
    deliberately minimal and length-bounded:

    - ``source_id`` is the stable prompt label the grounding packet assigned
      (e.g. ``"OB1"``, ``"repo:frames"``); the runtime validates a reported id
      against the exact recall/source set for that task and drops unknown ids, so
      a fabricated citation cannot ride the wire.
    - ``kind`` is the typed provenance (:data:`EvidenceKind`).
    - ``label`` is a short human display string (safe, redacted upstream — never
      a token, namespace credential, member id, or raw recalled payload).
    - ``locator`` is an OPTIONAL safe pointer (a repo path, a public url, an OB
      entity id) — never a secret and never private recalled content.
    - ``authority`` is the typed conflict-precedence tier
      (:data:`EvidenceAuthority`); ``live``/``source`` outrank ``memory``.

    Security floor: this model carries only what a connector may render into a
    "possibly-leaked" channel. Redaction and the no-private-content rule are
    enforced by the producing runtime BEFORE construction; the wire only bounds
    the shape and length so a malformed report cannot blow frame size.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1, max_length=MAX_SOURCE_ID_LEN)
    kind: EvidenceKind
    label: str = Field(min_length=1, max_length=MAX_SOURCE_LABEL_LEN)
    locator: str | None = Field(default=None, max_length=MAX_SOURCE_LOCATOR_LEN)
    authority: EvidenceAuthority = "unknown"


class VerificationReport(BaseModel):
    """Carry the typed evidence state of an agent's answer (evidence-gated turns).

    Rides the answering :class:`TurnFrame` so grounding is a typed frame field,
    never scraped from the reply prose. The engine submits it structurally
    (Codex strict ``outputSchema`` / Claude typed tool); the runtime validates
    and bounds it before it rides the wire, and a connector renders it purely.

    - ``status`` (:data:`VerificationStatus`) is the evidence verdict.
      ``not_applicable`` is reserved for ``social``/``opinion``/action-only
      turns; factual/mixed turns require at least ``partial``.
    - ``claim_kind`` (:data:`ClaimKind`) classifies what was answered, which
      gates the egress policy (a factual turn cannot ship ``not_applicable``).
    - ``ob_checked`` records whether Open Brain recall ran for this turn —
      recorded even when ``status`` is ``not_applicable``, so "did we check
      memory" is always answerable and an OB timeout/failure is legible.
    - ``sources`` is the bounded, validated evidence set
      (``<= MAX_EVIDENCE_SOURCES``).
    - ``limitations`` is a bounded list of plain-text caveats
      (``<= MAX_VERIFICATION_LIMITATIONS``) — e.g. "OB recall timed out",
      "conflict between live state and memory" — surfaced to the user, never
      silently dropped.
    - ``methods`` records the observed verification method / tool names the
      runtime saw the turn use (``<= MAX_VERIFICATION_METHODS``); tool NAMES
      only, never tool inputs or private content.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: VerificationStatus
    claim_kind: ClaimKind
    ob_checked: bool
    sources: list[EvidenceSource] = Field(default_factory=list, max_length=MAX_EVIDENCE_SOURCES)
    limitations: list[str] = Field(default_factory=list, max_length=MAX_VERIFICATION_LIMITATIONS)
    methods: list[str] = Field(default_factory=list, max_length=MAX_VERIFICATION_METHODS)

    @model_validator(mode="after")
    def _bound_text_fields(self) -> VerificationReport:
        """Reject blank/oversize limitation and method strings.

        List membership is bounded by ``max_length`` on the field; this bounds
        each ELEMENT so a single 100 KB "limitation" cannot ride the wire. Blank
        entries are rejected outright — an empty caveat carries no signal and is
        a producer bug.
        """
        for limitation in self.limitations:
            if not limitation.strip():
                raise ValueError("limitation entries must be non-blank")
            if len(limitation) > _MAX_LIMITATION_LEN:
                raise ValueError(
                    f"limitation exceeds {_MAX_LIMITATION_LEN} chars ({len(limitation)})"
                )
        for method in self.methods:
            if not method.strip():
                raise ValueError("method entries must be non-blank")
            if len(method) > _MAX_METHOD_LEN:
                raise ValueError(f"method exceeds {_MAX_METHOD_LEN} chars ({len(method)})")
        return self


class TurnFrame(WireFrame):
    """Carry visible agent output plus typed stance data.

    ``origin`` carries the originating chat-surface identity (issue #93 reply
    render), copied from the triggering :class:`TaskFrame` when the answered task
    arrived from an intake/wake path. It lets a triage edge render an EXTERNAL
    agent's answer back into the SAME channel the question came from, for a task
    the edge did not mint itself. Additive-optional (default ``None``), mirroring
    :class:`TaskFrame`'s precedent EXACTLY: omitted from the wire when None so a
    legacy-shaped turn is byte-identical to the pre-#93 wire and an old
    ``extra="forbid"`` decoder without the field still accepts it.

    Deploy ordering (wire compat): the ``origin`` key rides an ``extra="forbid"``
    frame, so EVERY consumer must upgrade rtech-frames (to the SCHEMA_VERSION "5"
    build that knows the field) BEFORE any producer starts stamping ``origin`` on
    a TurnFrame. Upgrade consumers first, then producers.

    ``verification`` carries the typed evidence state of this answer
    (:class:`VerificationReport`, evidence-gated responses). The engine submits
    it structurally and the runtime validates and bounds it BEFORE it rides the
    wire — grounding is a typed frame field, never scraped from ``visible_text``.
    Additive-optional (default ``None``), mirroring ``origin`` EXACTLY: dropped
    from the wire when None (see :meth:`_serialize`), so a legacy-shaped turn has
    the pre-evidence-gated key set (apart from the version stamp) and an old
    ``extra="forbid"`` decoder without the field still accepts it. It bumped
    SCHEMA_VERSION to "8"
    on the same dropped-on-None additive-safe pattern; same
    consumers-before-producers deploy ordering as ``origin`` — every consumer
    must know the field before any producer stamps it.
    """

    agent_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    visible_text: str = Field(min_length=1)
    stance: Stance
    refs: list[str] = Field(default_factory=list)
    origin: ChannelOrigin | None = None
    verification: VerificationReport | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Omit the ``origin``/``verification`` keys from the wire when None.

        Mirrors :meth:`TaskFrame._serialize` exactly: ``origin`` and
        ``verification`` are the ONLY keys dropped-on-None, so a legacy-shaped
        turn (neither set) preserves the pre-evidence-gated key set apart from the
        version stamp, and an old ``extra="forbid"`` decoder without the fields
        still accepts it. When either is set it rides normally and round-trips.
        """
        data: dict[str, Any] = handler(self)
        if self.origin is None:
            data.pop("origin", None)
        if self.verification is None:
            data.pop("verification", None)
        return data


class SteerFrame(WireFrame):
    """Carry human steering input to a target agent.

    ``interrupt`` is the typed control signal (never scraped from ``text``): when
    true the target aborts its active turn before injecting ``text``. It stays a
    frame field so the steer wire carries no prose command channel.
    """

    task_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    target_agent: str = Field(min_length=1)
    interrupt: bool = False


class StatusFrame(WireFrame):
    """Carry typed in-progress or blocked task status.

    ``agent_id`` is the status author (frame contract amendment, PR #73
    round 5): per-participant staleness classification in done derivation
    needs the participant on the frame, mirroring ``TurnFrame`` authorship.
    Required with no default — a v1 status without authorship fails closed at
    decode (schema_version 2 wire change; deployed stacks re-sync on merge).

    ``phase`` is the additive cold-acknowledgement marker (evidence-gated
    responses): ``"verifying"`` marks a cold/first-contact turn emitting the
    "verifying first" acknowledgement, ``"responding"`` marks the warm fast path.
    It does NOT invent connector-owned state — ``state="working"`` is unchanged
    and ``phase`` only distinguishes the cold ack; done remains
    orchestrator-derived from the frame graph and ``phase`` is NOT a control
    signal. Additive-optional (default ``None``); dropped-on-None on the wire
    (see :meth:`_serialize`), so a legacy status preserves the pre-evidence-gated
    key set apart from the version stamp, and an old ``extra="forbid"`` decoder
    without the field still accepts it. It bumped SCHEMA_VERSION to "8"; same
    consumers-before-producers deploy ordering as the other additive fields.
    """

    task_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    state: Literal["working", "input_required", "blocked"]
    phase: Literal["responding", "verifying"] | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Omit the ``phase`` key from the wire when it is None.

        Mirrors :meth:`TurnFrame._serialize`: ``phase`` is the ONLY key
        dropped-on-None, so a legacy-shaped status (no ``phase``) preserves the
        pre-evidence-gated key set apart from the version stamp, and an old
        ``extra="forbid"`` decoder without the field still accepts it. When
        ``phase`` is set it rides normally and round-trips.
        """
        data: dict[str, Any] = handler(self)
        if self.phase is None:
            data.pop("phase", None)
        return data


class LifecycleFrame(WireFrame):
    """Carry a daemon lifecycle event so shutdown/recovery is legible (issue #84).

    Daemon stop/restart was opaque: Discord users saw a shutdown with no reason
    or initiator, and supervisor recovery was silent to the bus. This frame makes
    each lifecycle transition an explicit, typed bus event.

    ``reason`` names the transition kind; ``initiator`` names what triggered it
    (``"signal:SIGTERM"``, ``"supervisor"``, ``"operator"``). ``correlation_id``
    is a per-lifecycle-event id (the repo's ``uuid4().hex`` idiom) so a restart's
    death and its recovery carry the SAME id and correlate as one pair.
    ``detail`` is optional free context (e.g. ``"stream-ended"`` vs
    ``"stream-error"``) and MUST never carry secrets. ``ts`` mirrors the frame
    timestamp idiom (``TaskFrame.created_at``): an optional ``datetime`` the
    producer stamps, defaulting to ``None`` for additive-safe legacy decode.
    """

    agent_id: str = Field(min_length=1)
    reason: Literal["requested_restart", "crash", "supervisor_recovery", "shutdown"]
    initiator: str = Field(min_length=1)
    correlation_id: str = Field(min_length=1)
    detail: str = ""
    ts: datetime | None = None


class PresenceTransitionFrame(WireFrame):
    """Carry ONE typed agent presence-edge event (issue #147).

    Rendered operator notifications (Telegram/Discord) are dumb edges: they ONLY
    format this typed frame, never infer a transition from prose. Presence/liveness
    truth is owned upstream by the web-chat ``PresenceRegistry`` (issue #145); a
    deriver observes that registry's alive-state deltas and emits exactly one frame
    per genuine edge (``offline`` on alive->dead, ``recovered`` on dead->alive),
    with steady-state repeats deduped to nothing.

    Every field is TYPED and safe-to-render, EXCEPT that a connector renders ONLY
    ``display_name`` / ``observed_at`` / a Dash link — never ``agent_id`` as visible
    identity text:

    - ``agent_id`` is the persona slug (``[A-Za-z0-9_-]+``) carried for the deriver's
      per-agent dedup keying and orchestrator correlation, and as the stable opaque
      locator inside the Dash live-page URL PATH ONLY (a safe path segment, the
      ``EvidenceSource.locator`` precedent). It is NEVER emitted as visible rendered
      identity.
    - ``display_name`` is the clean, safe human name the deriver stamps (title-cased
      from the slug). This is the ONLY identity a connector shows.
    - ``transition`` is the typed edge kind (``offline``/``recovered``) — never a
      scraped prose signal.
    - ``observed_at`` is the time the transition was observed. It MUST be tz-aware
      (a naive datetime is rejected at construction and at decode) and any aware
      offset is NORMALIZED to UTC, so the stored value is always the true UTC instant
      and every renderer's literal ``UTC`` suffix is honest.

    ``agent_id`` and ``display_name`` are additionally bounded/validated: ``agent_id``
    must be a NATS-subject-safe ``[A-Za-z0-9_-]+`` token within
    :data:`MAX_PRESENCE_AGENT_ID_LEN` (it rides as a Dash URL path segment);
    ``display_name`` must be within the bounded safe human vocabulary
    (:data:`_DISPLAY_NAME_PATTERN`, ``<= MAX_PRESENCE_DISPLAY_NAME_LEN``, no
    leading/trailing whitespace) so a connector can never render a mention, a
    Markdown link, or a Telegram HTML tag from it.

    ``transition_id`` is a STABLE typed delivery-idempotency key (issue #147),
    minted ONCE by the deriver when the pending frame is created and RETAINED on
    retry, so an at-least-once redelivery of the SAME edge (an ambiguous transport
    ack) carries the SAME id and a consumer renders/sends it exactly once while two
    DISTINCT edges (distinct ids) still deliver in order. It is a delivery-
    idempotency key ONLY — the NATS typed frame remains truth — and is NEVER rendered
    to an operator surface. Required (this frame kind is new in #147, so there is no
    legacy id-less wire to stay compatible with); bounded to a short subject-safe
    token (:data:`MAX_PRESENCE_TRANSITION_ID_LEN`, ``[A-Za-z0-9_-]+``) so an
    oversized/unsafe id can neither blow the frame nor poison a consumer's cache key
    space.

    Additive NEW frame kind (its own ``frame_type`` key ``presence_transition``);
    existing frames are unchanged and the wire is byte-compatible for them.
    """

    agent_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    transition: Literal["offline", "recovered"]
    observed_at: datetime
    transition_id: str = Field(min_length=1)

    @field_validator("observed_at")
    @classmethod
    def _normalize_observed_at(cls, value: datetime) -> datetime:
        """Reject a naive ``observed_at`` and normalize any aware offset to UTC.

        A naive datetime (no tzinfo) carries no honest instant — a renderer that
        stamps a literal ``UTC`` suffix would then lie — so it is rejected at
        construction AND at decode. An aware datetime at any offset is normalized
        to UTC (``astimezone(timezone.utc)``) so the stored value is the true UTC
        instant and every renderer formats it honestly.

        Raises:
            ValueError: If ``observed_at`` is naive (no tzinfo).
        """
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("observed_at must be timezone-aware (naive datetimes are rejected)")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _check_identity(self) -> PresenceTransitionFrame:
        """Reject a subject-unsafe/over-cap ``agent_id`` or unsafe ``display_name``.

        ``agent_id`` must fullmatch the shared ``_AGENT_ID_PATTERN``
        (``[A-Za-z0-9_-]+`` — the same NATS-subject-safe token
        :func:`validate_participants` enforces) and stay within
        :data:`MAX_PRESENCE_AGENT_ID_LEN`, because it rides as a Dash live-page URL
        path segment. ``display_name`` must fullmatch the bounded safe human
        vocabulary (:data:`_DISPLAY_NAME_PATTERN`), carry no leading/trailing
        whitespace, and stay within :data:`MAX_PRESENCE_DISPLAY_NAME_LEN`, because a
        connector renders it into a possibly-leaked surface — it must not be able to
        carry a Discord mention, a Markdown link, or a Telegram HTML tag.

        ``transition_id`` must fullmatch the safe token vocabulary
        (:data:`_TRANSITION_ID_PATTERN`, ``[A-Za-z0-9_-]+``) and stay within
        :data:`MAX_PRESENCE_TRANSITION_ID_LEN`: it rides the wire and enters a
        consumer's bounded seen-id cache as a key, so an oversized/unsafe id is
        rejected at construction and at decode before it can blow the frame or poison
        the cache key space.

        Raises:
            ValueError: On a subject-unsafe/over-cap ``agent_id``, an unsafe/
                over-cap/whitespace-padded ``display_name``, or an unsafe/over-cap
                ``transition_id``.
        """
        if _AGENT_ID_PATTERN.fullmatch(self.agent_id) is None:
            raise ValueError(
                f"agent_id {self.agent_id!r} is not a safe NATS subject token ([A-Za-z0-9_-]+)"
            )
        if len(self.agent_id) > MAX_PRESENCE_AGENT_ID_LEN:
            raise ValueError(
                f"agent_id exceeds {MAX_PRESENCE_AGENT_ID_LEN} chars ({len(self.agent_id)})"
            )
        if len(self.display_name) > MAX_PRESENCE_DISPLAY_NAME_LEN:
            raise ValueError(
                f"display_name exceeds {MAX_PRESENCE_DISPLAY_NAME_LEN} chars "
                f"({len(self.display_name)})"
            )
        if self.display_name != self.display_name.strip():
            raise ValueError("display_name must not have leading/trailing whitespace")
        if not _DISPLAY_NAME_PATTERN.fullmatch(self.display_name):
            raise ValueError(
                f"display_name {self.display_name!r} contains an unsafe character "
                "(allowed: letters, digits, spaces, '.', '-', '_')"
            )
        if len(self.transition_id) > MAX_PRESENCE_TRANSITION_ID_LEN:
            raise ValueError(
                f"transition_id exceeds {MAX_PRESENCE_TRANSITION_ID_LEN} chars "
                f"({len(self.transition_id)})"
            )
        if not _TRANSITION_ID_PATTERN.fullmatch(self.transition_id):
            raise ValueError(
                f"transition_id {self.transition_id!r} is not a safe token ([A-Za-z0-9_-]+)"
            )
        return self


# The supervised-Start frame family lives in the sibling ``supervised`` module.
# It is RE-EXPORTED here (never re-defined, same model identities) so the historical
# ``from rtech_frames.frames import StartAgentFrame`` surface keeps working. Placed
# at end-of-file: ``supervised`` imports ``WireFrame`` from this module, so the
# one-way ``supervised -> frames`` dependency has no import cycle.
from .supervised import MAX_REQUEST_ID_LEN as MAX_REQUEST_ID_LEN  # noqa: E402
from .supervised import StartAgentFrame as StartAgentFrame  # noqa: E402
from .supervised import StartAllAgentsFrame as StartAllAgentsFrame  # noqa: E402
from .supervised import StartFailure as StartFailure  # noqa: E402
from .supervised import SupervisedAgentStateFrame as SupervisedAgentStateFrame  # noqa: E402
