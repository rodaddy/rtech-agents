"""Pure frame <-> fleet-envelope translation for the Hermes bridge.

The bridge is a **dumb edge**: it translates at the boundary and carries no
control signal of its own. This module holds the pure, bus-free translation so it
is unit-tested directly; :mod:`runtime` wires it to two live buses.

Direction 1 — rtech ``TaskFrame`` -> Hermes inbound ``Envelope``:
    A ``TaskFrame`` arriving on ``rtech.agent.<fleet_id>.task`` becomes a fleet
    ``Envelope`` (kind ``dispatch``) on ``{env}.agent.dispatch.<fleet_id>`` — the
    bare dispatch lane the gateway's ``fleet_subscriber`` adapter subscribes. The
    envelope is built with :func:`fleet_nats.Envelope.new` (the sanctioned
    encoder — the envelope is NOT hand-rolled). ``payload.text`` = the task text;
    ``task_id`` = the rtech task id (so the gateway keys its reply back on the
    SAME id); ``channel`` = the configured default channel (the gateway skips any
    turn with no resolvable channel).

Direction 2 — Hermes reply ``Envelope`` -> CONTENT-ONLY translation:
    The gateway's ``fleet_subscriber.send()`` publishes on the task's status
    subject, in order: zero or more ``kind="status"`` envelopes (stance-only
    structural signals and/or visible reply text), then exactly one
    ``kind="done"`` envelope carrying the typed ``fleet_core.spec.Done`` signal.
    This module *classifies* each envelope into a typed :class:`ReplyText`
    (reply content the runtime publishes immediately as a CONTENT ``TurnFrame``)
    or a :class:`ReplyCompletion` (the typed completion — consumed by the
    runtime for LIFECYCLE ONLY: unsubscribe, drop state, advance the queue; it
    never becomes a frame here). Terminal semantics for bridged participants
    live in the ORCHESTRATOR, which consumes the same typed ``Done`` on its own
    fleet-side subscription and records it as in-graph terminal evidence (see
    docs/bridged-participant-lifecycle.md). Every signal is read from
    NAMED typed fields (``kind`` / ``sender`` / ``task_id`` / ``typed_signal`` /
    ``payload["text"]``) — never scraped from prose. Identity is fail-closed:
    an envelope whose ``sender`` or ``task_id`` does not match the dispatched
    task is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

from fleet_core.spec import Done as FleetDone
from fleet_core.spec import Stance as FleetStance
from fleet_nats import Envelope, subjects
from loguru import logger
from rtech_frames import Stance, TaskFrame, TurnFrame

# The stance the bridge states when a Hermes reply carried no typed stance of its
# own: the agent responded, attributed to the bridge. When the reply DID carry a
# typed fleet-core Stance, its position/because ride through instead — the real
# typed signal is never discarded in favor of this fallback.
_STANCE_POSITION = "responded"
_STANCE_BECAUSE = "hermes-bridge"

# The inbound Hermes dispatch kind. The gateway's mapping treats dispatch/relay/
# wake/inbox identically (any kind with text+channel forges a turn); dispatch is
# the semantically correct "here is work" kind and the lane we publish on.
_DISPATCH_KIND = "dispatch"

# The two envelope kinds the gateway's send() path emits on the status subject.
_REPLY_KIND = "status"
_DONE_KIND = "done"
_PAYLOAD_TEXT_KEY = "text"


@dataclass(frozen=True)
class InboundEnvelope:
    """A built Hermes-side dispatch: the subject and the encoded-ready envelope."""

    subject: str
    envelope: Envelope


@dataclass(frozen=True)
class ReplyText:
    """Visible reply content lifted from one ``kind="status"`` envelope.

    The runtime publishes it immediately as a CONTENT ``TurnFrame``; nothing
    here is terminal. The typed fleet-core stance (when the envelope carried
    one) rides along verbatim so the gateway's real typed signal is never
    discarded or replaced.
    """

    text: str
    stance: FleetStance | None


@dataclass(frozen=True)
class ReplyCompletion:
    """The typed completion signal — bridge LIFECYCLE input only, never a frame.

    The orchestrator consumes the same typed ``Done`` on its own fleet-side
    subscription; the bridge only uses this to end the task's local lifecycle.
    """

    outcome: str


def task_to_dispatch_subject(fleet_env: str, fleet_id: str) -> str:
    """Return the bare dispatch subject the gateway subscribes for ``fleet_id``.

    The gateway subscribes BOTH the bare ``{env}.agent.dispatch.<id>`` and the
    per-channel wildcard ``{env}.agent.*.dispatch.<id>``; publishing on the bare
    lane is delivered to the bare subscription (one fewer token, no channel
    needed in the subject — the channel rides the envelope field).
    """
    return subjects.agent_dispatch(fleet_env, fleet_id)


def reply_status_subject(fleet_env: str, task_id: str) -> str:
    """Return the Hermes status subject a reply for ``task_id`` arrives on."""
    return subjects.agent_status(fleet_env, task_id)


def build_inbound_envelope(
    *,
    task: TaskFrame,
    fleet_id: str,
    fleet_env: str,
    default_channel: str | None,
    msg_id: str,
    ts: str,
) -> InboundEnvelope:
    """Translate a rtech ``TaskFrame`` into a Hermes inbound dispatch envelope.

    Args:
        task: The inbound rtech task frame.
        fleet_id: The fleet agent id this task is dispatched to.
        fleet_env: The fleet subject env prefix (``dev``/``prod``).
        default_channel: The channel stamped on the envelope (the gateway skips a
            turn with no resolvable channel). ``None`` leaves it unset — only safe
            when the target agent has a ``home_channel`` configured.
        msg_id: Caller-supplied unique message id (deterministic under test).
        ts: Caller-supplied ISO-8601 timestamp (deterministic under test).

    Returns:
        The :class:`InboundEnvelope` (subject + envelope) ready to publish.
    """
    subject = task_to_dispatch_subject(fleet_env, fleet_id)
    envelope = Envelope.new(
        msg_id=msg_id,
        ts=ts,
        sender=fleet_id,
        kind=_DISPATCH_KIND,
        task_id=task.id,
        channel=default_channel,
        payload={_PAYLOAD_TEXT_KEY: task.text},
    )
    logger.info(
        "hermes_bridge.translate.inbound | task_id={} fleet_id={} subject={} has_channel={}",
        task.id,
        fleet_id,
        subject,
        default_channel is not None,
    )
    return InboundEnvelope(subject=subject, envelope=envelope)


def classify_reply(
    *, envelope: Envelope, fleet_id: str, task_id: str
) -> ReplyText | ReplyCompletion | None:
    """Classify one status-subject envelope by its typed fields, fail-closed.

    Identity binding: the envelope's own ``sender`` and ``task_id`` fields must
    match the dispatched ``fleet_id``/``task_id``. A mismatched or forged
    envelope on the task's subject is dropped loudly — callback-captured values
    are never trusted over the wire's typed identity fields.

    Completion is recognised ONLY as ``kind="done"`` carrying a typed
    ``fleet_core.spec.Done`` whose own ``task_id`` matches. A ``done`` without
    the typed signal (or with a mismatched typed task id) fails closed.

    Args:
        envelope: The envelope observed on the task's status subject.
        fleet_id: The fleet agent id the task was dispatched to.
        task_id: The rtech task id the subscription was keyed on.

    Returns:
        A :class:`ReplyText` (buffer candidate), a :class:`ReplyCompletion`
        (release the buffered reply), or ``None`` (drop this envelope).
    """
    if envelope.sender != fleet_id:
        logger.warning(
            "hermes_bridge.translate.reply | sender mismatch — dropped | "
            "task_id={} expected={} got={}",
            task_id,
            fleet_id,
            envelope.sender,
        )
        return None
    if envelope.task_id != task_id:
        logger.warning(
            "hermes_bridge.translate.reply | task_id mismatch — dropped | "
            "subscribed={} envelope={} fleet_id={}",
            task_id,
            envelope.task_id,
            fleet_id,
        )
        return None

    if envelope.kind == _DONE_KIND:
        signal = envelope.typed_signal
        if not isinstance(signal, FleetDone):
            logger.warning(
                "hermes_bridge.translate.reply | done without typed Done signal — "
                "dropped (fail closed) | task_id={} fleet_id={}",
                task_id,
                fleet_id,
            )
            return None
        if signal.task_id != task_id:
            logger.warning(
                "hermes_bridge.translate.reply | typed Done task mismatch — dropped | "
                "subscribed={} signal={}",
                task_id,
                signal.task_id,
            )
            return None
        logger.info(
            "hermes_bridge.translate.reply | typed completion | task_id={} outcome={}",
            task_id,
            signal.outcome,
        )
        return ReplyCompletion(outcome=signal.outcome)

    if envelope.kind != _REPLY_KIND:
        logger.warning(
            "hermes_bridge.translate.reply | unexpected kind={} on status lane | task_id={}",
            envelope.kind,
            task_id,
        )
        return None

    stance = envelope.typed_signal if isinstance(envelope.typed_signal, FleetStance) else None
    raw_text = envelope.payload.get(_PAYLOAD_TEXT_KEY)
    if not isinstance(raw_text, str) or not raw_text.strip():
        logger.warning(
            "hermes_bridge.translate.reply | status has no text | task_id={} has_stance={}",
            task_id,
            stance is not None,
        )
        return None
    logger.info(
        "hermes_bridge.translate.reply | reply text buffered | task_id={} len={} has_stance={}",
        task_id,
        len(raw_text),
        stance is not None,
    )
    return ReplyText(text=raw_text, stance=stance)


def build_content_turn(
    *,
    fleet_id: str,
    task_id: str,
    text: str,
    stance: FleetStance | None,
) -> TurnFrame:
    """Build a CONTENT ``TurnFrame`` from one fleet reply — never terminal semantics.

    Pure translation of the gateway's typed fields: ``visible_text`` is the
    named ``payload["text"]`` verbatim, and the stance fields are a verbatim
    copy of the gateway's typed fleet-core stance (``position`` / ``because`` /
    ``needs_from``) when one rode the envelope. Nothing is synthesized as a
    control signal: for a BRIDGED participant the orchestrator ignores turn
    terminality entirely — its terminal evidence is exclusively the typed fleet
    ``Done`` the orchestrator consumes itself — so these
    stance fields are display/content data, not completion state.

    When the envelope carried no typed stance, the fixed content attribution
    (``responded`` / ``hermes-bridge``) satisfies the frame schema's required
    non-empty fields; ``needs_from`` stays an empty passthrough. By the same
    contract as above it carries no terminal meaning.

    Args:
        fleet_id: The fleet agent id the turn is attributed to.
        task_id: The rtech task id this turn belongs to.
        text: The reply text (typed ``payload["text"]``), verbatim.
        stance: The typed fleet-core stance the reply carried, if any.

    Returns:
        The content :class:`TurnFrame`.
    """
    if stance is not None:
        position = str(stance.position)
        because = stance.because if stance.because else _STANCE_BECAUSE
        needs_from = list(stance.needs_from)
    else:
        position = _STANCE_POSITION
        because = _STANCE_BECAUSE
        needs_from = []
    frame = TurnFrame(
        agent_id=fleet_id,
        task_id=task_id,
        visible_text=text,
        stance=Stance(position=position, because=because, needs_from=needs_from),
        refs=[task_id],
    )
    logger.info(
        "hermes_bridge.translate.turn | content turn built | task_id={} fleet_id={} "
        "position={} visible_len={}",
        task_id,
        fleet_id,
        position,
        len(text),
    )
    return frame
