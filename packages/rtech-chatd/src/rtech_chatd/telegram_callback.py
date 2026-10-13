"""Authorize bounded Telegram callback data and map it to typed steers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger
from rtech_frames.frames import SteerFrame

CALLBACK_DATA_MAX_BYTES = 64
CALLBACK_CONTRACT_VERSION = "v1"
DEFAULT_CALLBACK_TTL_S = 900.0
DEFAULT_CALLBACK_MAX_ENTRIES = 4096
_DELIMITER = "|"
_NONCE_BYTES = 12
_SIGNATURE_BYTES = 12


@dataclass(frozen=True)
class CallbackAction:
    """Describe one closed callback action and its typed steer semantics."""

    token: str
    steer_text: str
    interrupt: bool


_ACTIONS: dict[str, CallbackAction] = {
    action.token: action
    for action in (
        CallbackAction("approve", "The human approved this direction; continue.", False),
        CallbackAction("revise", "The human asked you to reconsider your current approach.", True),
        CallbackAction("stop", "The human asked you to stop the current line of work.", True),
    )
}
SUPPORTED_ACTIONS: frozenset[str] = frozenset(_ACTIONS)

# The human-facing button label for each supported action. The button is DISPLAY
# ONLY: its typed steer semantics live in :data:`_ACTIONS`, so the label carries
# no control signal — a tap becomes a typed SteerFrame via the callback sink, the
# label just names the affordance. Ordered so the least-disruptive action reads
# first (approve → revise → stop).
_ACTION_LABELS: dict[str, str] = {
    "approve": "✅ Approve",
    "revise": "✏️ Revise",
    "stop": "🛑 Stop",
}


class CallbackContractError(ValueError):
    """Report a malformed callback contract at its minting boundary."""


@dataclass(frozen=True)
class CallbackCommand:
    """Represent an authorized action against one immutable task/agent target."""

    action: str
    task_id: str
    target_agent: str

    def to_steer(self) -> SteerFrame:
        """Return the only callback-derived control-plane payload."""
        spec = _ACTIONS[self.action]
        return SteerFrame(
            task_id=self.task_id,
            text=spec.steer_text,
            target_agent=self.target_agent,
            interrupt=spec.interrupt,
        )


@dataclass(frozen=True)
class _Reservation:
    """A two-phase handle over one reserved capability (commit or rollback)."""

    nonce: str
    command: CallbackCommand


@dataclass(frozen=True)
class _IssuedCallback:
    """Keep authorization state only; task completion is never stored here.

    ``actor`` binds the capability to the ONE Telegram user it was minted for, so
    a copied valid button tapped by a DIFFERENT user in the same message fails
    closed. ``reserved`` is a two-phase gate: a consume RESERVES the capability
    (so a concurrent replay is rejected) but only a downstream commit marks it
    ``consumed``; a rollback clears the reservation so a failed publish leaves the
    tap retryable rather than stranding it as spent.
    """

    action: str
    task_id: str
    target_agent: str
    channel_id: str
    message_id: str
    actor: str
    expires_at: float
    reserved: bool = False
    consumed: bool = False


Clock = Callable[[], float]


def _token(byte_count: int) -> str:
    """Return a URL-safe compact token without padding."""
    return base64.urlsafe_b64encode(secrets.token_bytes(byte_count)).rstrip(b"=").decode("ascii")


class TelegramCallbackRegistry:
    """Mint and consume single-use, chat/message-bound callback capabilities.

    The signed payload contains no task or agent identity.  Those identities stay
    in the locally issued record, so changing callback data cannot retarget a
    steer.  A copied valid capability is rejected unless it arrives from the
    exact chat and message it was issued for AND is tapped by the exact actor it
    was minted for; a two-phase reserve/commit makes replays fail closed while a
    rolled-back reservation (failed publish) stays retryable.  This is
    interaction authorization state, not connector-owned task or done state.
    """

    def __init__(
        self,
        *,
        secret: bytes | None = None,
        ttl_s: float = DEFAULT_CALLBACK_TTL_S,
        max_entries: int = DEFAULT_CALLBACK_MAX_ENTRIES,
        clock: Clock,
    ) -> None:
        if ttl_s <= 0:
            raise ValueError("ttl_s must be > 0")
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._secret = secret or secrets.token_bytes(32)
        self._ttl_s = ttl_s
        self._max_entries = max_entries
        self._clock = clock
        self._issued: OrderedDict[str, _IssuedCallback] = OrderedDict()

    def issue(
        self,
        *,
        action: str,
        task_id: str,
        target_agent: str,
        channel_id: str,
        message_id: str,
        actor: str,
    ) -> str:
        """Mint a single-use callback capability bound to one rendered button.

        ``actor`` is the ONE Telegram user id the button is meant for; only a tap
        whose ``callback.author`` equals it can consume the capability, so a
        copied valid button tapped by another chat member fails closed.
        """
        if action not in _ACTIONS:
            raise CallbackContractError(f"unsupported callback action {action!r}")
        if not all((task_id, target_agent, channel_id, message_id, actor)):
            raise CallbackContractError(
                "task, agent, chat, message, and actor ids must be non-empty"
            )
        nonce = _token(_NONCE_BYTES)
        data = self._encode(action, nonce)
        self._prune()
        self._issued[nonce] = _IssuedCallback(
            action=action,
            task_id=task_id,
            target_agent=target_agent,
            channel_id=channel_id,
            message_id=message_id,
            actor=actor,
            expires_at=self._clock() + self._ttl_s,
        )
        self._issued.move_to_end(nonce)
        if len(self._issued) > self._max_entries:
            self._issued.popitem(last=False)
            logger.warning("chatd.tg.callback | authorization capacity evicted oldest entry")
        return data

    def reserve(
        self,
        data: str | None,
        *,
        channel_id: str | None,
        message_id: str | None,
        actor: str | None,
    ) -> _Reservation | None:
        """Validate a capability and RESERVE it, returning a two-phase handle.

        The returned :class:`_Reservation` carries the nonce plus the typed
        :class:`CallbackCommand`; the caller MUST then :meth:`commit` it once the
        steer is published, or :meth:`rollback` it if publication fails so the tap
        stays retryable. A reserved capability rejects a concurrent replay, and an
        actor mismatch fails closed: only the user the button was minted for can
        act on it.
        """
        parsed = self._decode(data)
        if parsed is None or channel_id is None or message_id is None or not actor:
            return None
        action, nonce = parsed
        entry = self._issued.get(nonce)
        if entry is None:
            logger.info("chatd.tg.callback | reject=unknown-or-stale-capability")
            return None
        if entry.expires_at <= self._clock():
            self._issued.pop(nonce, None)
            logger.info("chatd.tg.callback | reject=expired-capability")
            return None
        if entry.consumed or entry.reserved:
            logger.info("chatd.tg.callback | reject=replayed-capability")
            return None
        if (
            entry.action != action
            or entry.channel_id != channel_id
            or entry.message_id != message_id
        ):
            logger.info("chatd.tg.callback | reject=mismatched-capability")
            return None
        if entry.actor != actor:
            logger.info("chatd.tg.callback | reject=actor-mismatch")
            return None
        self._issued[nonce] = _replace_flags(entry, reserved=True)
        return _Reservation(nonce, CallbackCommand(action, entry.task_id, entry.target_agent))

    def commit(self, reservation: _Reservation) -> None:
        """Mark a reserved capability spent so it can never be replayed."""
        entry = self._issued.get(reservation.nonce)
        if entry is None:
            return
        self._issued[reservation.nonce] = _replace_flags(entry, reserved=False, consumed=True)

    def rollback(self, reservation: _Reservation) -> None:
        """Release a reservation so a failed publish leaves the tap retryable."""
        entry = self._issued.get(reservation.nonce)
        if entry is None or entry.consumed:
            return
        self._issued[reservation.nonce] = _replace_flags(entry, reserved=False)

    def _encode(self, action: str, nonce: str) -> str:
        body = _DELIMITER.join((CALLBACK_CONTRACT_VERSION, action, nonce))
        signature = _token_signature(self._secret, body)
        data = _DELIMITER.join((body, signature))
        if len(data.encode("utf-8")) > CALLBACK_DATA_MAX_BYTES:
            raise CallbackContractError("callback_data exceeds Telegram's 64-byte limit")
        return data

    def _decode(self, data: str | None) -> tuple[str, str] | None:
        if not data or len(data.encode("utf-8")) > CALLBACK_DATA_MAX_BYTES:
            return None
        parts = data.split(_DELIMITER)
        if len(parts) != 4:
            return None
        version, action, nonce, signature = parts
        if version != CALLBACK_CONTRACT_VERSION or action not in _ACTIONS:
            return None
        if not nonce or not signature:
            return None
        body = _DELIMITER.join((version, action, nonce))
        if not hmac.compare_digest(signature, _token_signature(self._secret, body)):
            logger.info("chatd.tg.callback | reject=invalid-signature")
            return None
        return action, nonce

    def _prune(self) -> None:
        now = self._clock()
        while self._issued:
            nonce, entry = next(iter(self._issued.items()))
            if entry.expires_at > now:
                return
            self._issued.pop(nonce)


def _replace_flags(
    entry: _IssuedCallback, *, reserved: bool = False, consumed: bool = False
) -> _IssuedCallback:
    """Return a copy of one issued record with only its lifecycle flags changed."""
    return _IssuedCallback(
        action=entry.action,
        task_id=entry.task_id,
        target_agent=entry.target_agent,
        channel_id=entry.channel_id,
        message_id=entry.message_id,
        actor=entry.actor,
        expires_at=entry.expires_at,
        reserved=reserved,
        consumed=consumed,
    )


def _token_signature(secret: bytes, body: str) -> str:
    """Return a compact fixed-length HMAC tag for one callback body."""
    digest = hmac.new(secret, body.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest[:_SIGNATURE_BYTES]).rstrip(b"=").decode("ascii")


__all__ = [
    "CALLBACK_CONTRACT_VERSION",
    "CALLBACK_DATA_MAX_BYTES",
    "DEFAULT_CALLBACK_MAX_ENTRIES",
    "DEFAULT_CALLBACK_TTL_S",
    "SUPPORTED_ACTIONS",
    "CallbackAction",
    "CallbackCommand",
    "CallbackContractError",
    "TelegramCallbackRegistry",
]
