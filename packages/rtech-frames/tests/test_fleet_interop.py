"""Prove fleet-core base-model interop over the rtech wire, wire unchanged.

These tests pin the shared surface between rtech frames and ``fleet_core.spec``:
a fleet-core client can parse an rtech TurnFrame's stance and carry an rtech
frame as opaque ``Envelope`` payload, all without rtech changing its serialized
bytes. If rtech's ``Stance`` field names or wire ever drift from fleet-core's
base vocabulary, these tests fail closed as a regression guard.
"""

from __future__ import annotations

import json

from fleet_core.spec import Envelope, StancePosition
from fleet_core.spec import Stance as FleetStance
from rtech_frames.codec import decode_frame, encode_frame
from rtech_frames.frames import Stance as RtechStance
from rtech_frames.frames import TurnFrame, stance_to_fleet


def _turn() -> TurnFrame:
    return TurnFrame(
        agent_id="codex-a",
        task_id="task-1",
        visible_text="I will inspect the package.",
        stance=RtechStance(
            position="agree", because="The scope is bounded.", needs_from=["opus-b"]
        ),
    )


def test_fleet_stance_parses_rtech_stance_dict_by_name() -> None:
    """A fleet-core client parses rtech's stance wire dict via populate_by_name.

    fleet-core ``Stance`` shares rtech's Python field names, so its
    ``model_validate`` consumes the rtech stance payload directly without an
    alias remap.
    """
    turn = _turn()
    stance_wire = json.loads(encode_frame(turn).decode("utf-8"))["stance"]

    fleet_stance = FleetStance.model_validate(stance_wire)

    assert fleet_stance.position is StancePosition.AGREE
    assert fleet_stance.because == "The scope is bounded."
    assert fleet_stance.needs_from == ("opus-b",)


def test_stance_to_fleet_bridge_projects_shared_fields() -> None:
    """The compositional bridge maps rtech Stance onto the fleet-core base model."""
    fleet_stance = stance_to_fleet(
        RtechStance(position="dissent", because="Wrong scope.", needs_from=["a", "b"])
    )

    assert fleet_stance.position is StancePosition.DISSENT
    assert fleet_stance.because == "Wrong scope."
    assert fleet_stance.needs_from == ("a", "b")


def test_fleet_envelope_carries_rtech_frame_as_opaque_payload() -> None:
    """A fleet-core Envelope round-trips an rtech-encoded frame unchanged.

    The rtech frame rides inside the fleet ``Envelope`` payload; fleet-core's
    ``extra="allow"`` and stable ``to_bytes`` preserve the rtech bytes exactly.
    """
    encoded = encode_frame(_turn())
    frame_dict = json.loads(encoded.decode("utf-8"))

    envelope = Envelope.new(
        msg_id="m1",
        ts="2026-07-12T00:00:00Z",
        sender="codex-a",
        kind="turn",
        payload={"frame": frame_dict},
    )
    round_tripped = Envelope.from_bytes(envelope.to_bytes())

    carried = round_tripped.payload["frame"]
    assert carried == frame_dict
    # The rtech frame re-encodes to the identical wire after the fleet round-trip.
    assert decode_frame(encoded) == decode_frame(
        json.dumps(carried, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )


def test_rtech_turn_wire_is_unchanged_by_fleet_adoption() -> None:
    """The rtech TurnFrame wire bytes stay field-name stable (no fleet aliases).

    Regression guard for the wire-lock: rtech must NOT emit fleet-core's
    ``stance``/``reason``/``fact_pull`` alias keys. It carries its own
    ``position``/``because``/``needs_from`` shape.
    """
    stance_wire = json.loads(encode_frame(_turn()).decode("utf-8"))["stance"]

    assert set(stance_wire) == {"position", "because", "needs_from"}
    assert "stance" not in stance_wire
    assert "reason" not in stance_wire
    assert "fact_pull" not in stance_wire
