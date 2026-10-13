"""Define the supervised-Start frame family.

The external-supervisor Start control plane is a cohesive, recently added frame
family: a request to start one agent (:class:`StartAgentFrame`) or every agent
(:class:`StartAllAgentsFrame`), and the typed per-request Start receipt
(:class:`SupervisedAgentStateFrame`) with its :data:`StartFailure` outcome
vocabulary. They share the :data:`MAX_REQUEST_ID_LEN` correlation-key bound,
which nothing outside this family consumes.

The family is split into this sibling module to keep :mod:`rtech_frames.frames`
under the repo's file-line cap without changing any wire schema or public import.
It depends only on :class:`~rtech_frames.frames.WireFrame` (one-way: ``frames``
never imports ``supervised``, so there is no import cycle). Every symbol is
re-exported from :mod:`rtech_frames.frames` and the package root, so
``from rtech_frames.frames import StartAgentFrame`` and
``from rtech_frames import SupervisedAgentStateFrame`` keep working unchanged.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .frames import WireFrame

# One shared conservative maximum length for a supervised-Start ``request_id``
# (terminal-audit finding). ``request_id`` is a caller-supplied
# correlation/idempotency key that flows straight into the supervisor's in-flight
# and completed-receipt keys and rides every terminal receipt on the wire — so an
# oversized id must be rejected at DECODE, before it can enter those maps or blow a
# receipt's frame size. 128 chars comfortably fits a UUID (36) and any normal
# client idempotency key while staying far below a wire-size concern. It is the ONE
# owned source of this bound: every Start model that carries a ``request_id``
# imports it, never re-declares the literal. Minimum-nonempty validation
# (``min_length=1``) is preserved alongside it.
#
# Compatibility (consumers-before-producers): this is a NARROWING of an
# already-required field, not a new/removed wire key — a ``request_id`` of 1..128
# chars is byte-identical on the wire and decodes unchanged, so no SCHEMA_VERSION
# bump is warranted. The only behavior change is that a >128-char id, which was
# previously accepted, is now REJECTED at decode. Because a stricter consumer
# rejects what a laxer producer might still emit, every CONSUMER of the Start lanes
# must upgrade to this bound BEFORE any producer that could mint an over-128-char
# id is deployed against it; in practice UUID/idempotency-key producers never
# approach 128 chars, so no live producer emits an id this bound would newly reject.
MAX_REQUEST_ID_LEN = 128


StartFailure = Literal[
    "auth_unavailable",
    "unauthorized",
    "guard_unavailable",
    "guard_rejected",
    "unknown_agent",
    "start_failed",
    "liveness_unconfirmed",
    "state_publish_failed",
]


class StartAgentFrame(WireFrame):
    """Request that the external supervisor start one configured agent.

    ``requester_id`` is a SELF-ASSERTED, UNTRUSTED hint carried in the frame
    body. It is NOT an authenticated identity and MUST NOT be treated as a
    cryptographic requester principal: any publisher on the lane can set it to
    any value. The trusted authority for who may invoke the external supervisor
    is the NATS publisher-boundary (Tailscale/WireGuard-only, trusted-publisher
    lane) and the server-side authorizer, which authorizes on a server-resolved
    principal — never on this field. That separation is STRUCTURAL: the
    supervisor's authorizer and principal-resolver trust seams are never handed
    this frame at all (finding P1 trust), so an implementation cannot
    authorize from ``requester_id`` even by mistake. The frame carries neither
    configuration nor a process command; the external supervisor resolves the
    named agent from its server-side registry and authorizes the request before it
    invokes its own starter.

    Deploy ordering (wire compat): the supervised Start frames
    (``start_agent`` / ``start_all_agents`` / ``supervised_agent_state``) are
    NEW frame types, not new fields on an existing frame, and they bumped
    SCHEMA_VERSION to "9". An old ``extra="forbid"`` decoder rejects an unknown
    frame type at the codec ``frame_type`` lookup, so every consumer of these
    lanes must upgrade rtech-frames BEFORE any producer publishes them; existing
    frame types are byte-unchanged and unaffected by the bump.
    """

    request_id: str = Field(min_length=1, max_length=MAX_REQUEST_ID_LEN)
    agent_id: str = Field(min_length=1)
    requester_id: str = Field(min_length=1)


class StartAllAgentsFrame(WireFrame):
    """Request that the external supervisor start every configured agent.

    The known-agent set remains server-side. A caller cannot smuggle an agent
    configuration or an arbitrary process target into this bulk primitive.
    """

    request_id: str = Field(min_length=1, max_length=MAX_REQUEST_ID_LEN)
    requester_id: str = Field(min_length=1)


class SupervisedAgentStateFrame(WireFrame):
    """Report one external-supervisor Start transition with typed outcomes.

    This is deliberately separate from :class:`~rtech_frames.frames.LifecycleFrame`:
    that frame is emitted by a running daemon, while this frame is the external
    owner's Start-only control-plane receipt. Completion is not represented here
    and remains orchestrator-derived from task frames.

    ``request_id`` correlates the receipt to ONE originating request. A single
    shared per-agent process start may be awaited by many distinct requests
    (including a bulk request racing a single one); each distinct request gets its
    OWN terminal receipt carrying its OWN ``request_id`` — one requester's receipt
    is never reused for another (finding P2). ``Starting`` is the
    per-request breadcrumb; ``Online``/``Failed`` are the per-request terminal.
    ``request_id`` is bounded to :data:`MAX_REQUEST_ID_LEN` on all three Start
    models so an oversized id is rejected at decode before it can enter an in-flight
    or completed-receipt key or ride a terminal receipt (terminal-audit finding).

    Failure vocabulary (typed :data:`StartFailure`, v9 contract):

    - ``liveness_unconfirmed`` is a terminal Failed outcome meaning the external
      starter's start ACTION returned but the required authoritative post-start
      liveness probe did NOT affirmatively confirm the process is alive (missing,
      failing, or DOWN probe). A start action returning is not proof of Online, so
      the supervisor fails closed here rather than publish a false ``Online`` for a
      process that did not become ready or has already exited (finding
      P1 liveness). Safe retry/reconciliation is preserved.
    - ``state_publish_failed`` is a terminal Failed outcome that means the process
      was started AND confirmed alive but the confirmed ``Online`` receipt could
      NOT be published; internal ownership already reflects the confirmed start, so
      a retry re-confirms liveness and re-publishes the ``Online`` receipt without
      starting the agent again.
    """

    request_id: str = Field(min_length=1, max_length=MAX_REQUEST_ID_LEN)
    agent_id: str = Field(min_length=1)
    state: Literal["Starting", "Online", "Failed"]
    failure: StartFailure | None = None

    @model_validator(mode="after")
    def _check_failure(self) -> SupervisedAgentStateFrame:
        """Require failure codes exactly for failed state transitions."""
        if self.state == "Failed" and self.failure is None:
            raise ValueError("failed supervised state requires a typed failure code")
        if self.state != "Failed" and self.failure is not None:
            raise ValueError("only failed supervised state may carry a failure code")
        return self
