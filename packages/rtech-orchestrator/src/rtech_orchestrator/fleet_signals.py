"""Consume typed fleet-side completion signals for bridged participants.

Bridged participants (docs/bridged-participant-lifecycle.md)
are existing Hermes gateway agents fronted on the rtech bus by a dumb-edge
bridge. Their completion signal — the typed ``fleet_core.spec.Done`` — arrives
on the FLEET namespace (``{env}.agent.status.<task_id>``), and the orchestrator
consumes it DIRECTLY through the seam in this module: the typed signal is read
as a typed input and recorded as in-graph :class:`~rtech_frames.TerminalEvidence`.
It is never re-authored as a ``TurnFrame`` — terminal semantics for bridged
participants originate here and nowhere else.

``FleetSignalSource`` is the transport seam (tests inject a fake that
round-trips ``Envelope.to_bytes``/``from_bytes``); ``FleetBusDoneSource`` is the
real fleet-bus implementation; :func:`evidence_from_envelope` is the pure,
fail-closed validation that turns one observed envelope into evidence or
nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from fleet_core.spec import Done as FleetDone
from fleet_nats import subjects as fleet_subjects
from loguru import logger
from rtech_frames import TerminalEvidence

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fleet_nats import Envelope, FleetBus

# The one envelope kind that may carry the typed completion signal.
_DONE_KIND = "done"
# Evidence source tag: where the orchestrator observed the typed signal.
EVIDENCE_SOURCE_FLEET_DONE = "fleet-done"


class FleetSignalSource(Protocol):
    """Subscription seam for a task's fleet-side completion lane."""

    async def subscribe_done(
        self, task_id: str, handler: Callable[[Envelope], Awaitable[None]]
    ) -> Callable[[], Awaitable[None]]:
        """Subscribe ``handler`` to the task's fleet done lane.

        Args:
            task_id: The rtech task id (also the fleet-side correlation id).
            handler: Awaited with each decoded fleet envelope on the lane.

        Returns:
            An awaitable unsubscribe callable for lifecycle cleanup.

        Raises:
            Exception: Implementations surface subscription failure so the
                caller can fail the dispatch closed instead of waiting forever.
        """
        ...


class FleetBusDoneSource:
    """Real ``FleetSignalSource`` over a connected ``fleet_nats.FleetBus``."""

    def __init__(self, *, bus: FleetBus, env: str) -> None:
        """Wire the source to an already-connected fleet bus.

        Args:
            bus: The fleet-namespace bus connection.
            env: The fleet subject env prefix (``dev``/``prod``) — must match
                the gateway's ``FLEET_ENV``.
        """
        self._bus = bus
        self._env = env

    async def subscribe_done(
        self, task_id: str, handler: Callable[[Envelope], Awaitable[None]]
    ) -> Callable[[], Awaitable[None]]:
        """Subscribe the task's ``{env}.agent.status.<task_id>`` lane."""
        subject = fleet_subjects.agent_status(self._env, task_id)
        subscription = await self._bus.subscribe(subject, handler)
        logger.info("fleet_signals.subscribe | task_id={} subject={}", task_id, subject)

        async def _unsubscribe() -> None:
            try:
                await subscription.unsubscribe()
            except Exception as exc:
                logger.error(
                    "fleet_signals.unsubscribe failed | task_id={} reason={}", task_id, exc
                )

        return _unsubscribe


def evidence_from_envelope(
    *, envelope: Envelope, task_id: str, bridged: frozenset[str]
) -> TerminalEvidence | None:
    """Validate one fleet envelope into terminal evidence, fail-closed.

    Every check reads a NAMED typed field; any mismatch drops the envelope
    loudly and returns ``None``:

    - ``kind`` must be ``done`` (stance-only statuses are the bridge's content
      path, not completion);
    - ``sender`` must be one of the task's dispatched BRIDGED participants
      (spoofed/foreign senders never terminate anyone);
    - ``typed_signal`` must be a ``fleet_core.spec.Done``;
    - the envelope's ``task_id`` and the typed ``Done.task_id`` must both match
      the subscribed task.

    Args:
        envelope: The envelope observed on the task's fleet done lane.
        task_id: The rtech task id the subscription was keyed on.
        bridged: The task's dispatched bridged participant set.

    Returns:
        The typed :class:`~rtech_frames.TerminalEvidence`, or ``None``.
    """
    if envelope.kind != _DONE_KIND:
        logger.debug(
            "fleet_signals.evidence | non-done kind ignored | task_id={} kind={}",
            task_id,
            envelope.kind,
        )
        return None
    if envelope.sender not in bridged:
        logger.warning(
            "fleet_signals.evidence | sender not a bridged participant — dropped | "
            "task_id={} sender={} bridged={}",
            task_id,
            envelope.sender,
            sorted(bridged),
        )
        return None
    signal = envelope.typed_signal
    if not isinstance(signal, FleetDone):
        logger.warning(
            "fleet_signals.evidence | done without typed Done signal — dropped | "
            "task_id={} sender={}",
            task_id,
            envelope.sender,
        )
        return None
    if envelope.task_id != task_id or signal.task_id != task_id:
        logger.warning(
            "fleet_signals.evidence | task id mismatch — dropped | "
            "subscribed={} envelope={} signal={}",
            task_id,
            envelope.task_id,
            signal.task_id,
        )
        return None
    logger.info(
        "fleet_signals.evidence | typed Done consumed | task_id={} agent_id={} outcome={}",
        task_id,
        envelope.sender,
        signal.outcome,
    )
    return TerminalEvidence(
        agent_id=envelope.sender,
        task_id=task_id,
        source=EVIDENCE_SOURCE_FLEET_DONE,
        outcome=signal.outcome,
    )
