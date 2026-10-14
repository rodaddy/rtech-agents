"""Publish authorized Telegram callback actions as typed steering frames."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from loguru import logger
from rtech_frames.codec import encode_frame

from . import subjects
from .bus import FrameBus
from .channel import SendTarget
from .telegram_callback import _ACTION_LABELS, SUPPORTED_ACTIONS, TelegramCallbackRegistry
from .telegram_sender import TelegramSender

if TYPE_CHECKING:
    from .telegram_adapter import InboundCallback

# The steer actions offered on a debate-TLDR keyboard, in display order (least
# disruptive first). Every entry is a supported callback action; the assertion
# pins this list to the action registry so a renamed/removed action fails loud at
# import instead of minting a button no capability can back.
_TLDR_ACTIONS: tuple[str, ...] = ("approve", "revise", "stop")
assert set(_TLDR_ACTIONS) <= SUPPORTED_ACTIONS


class TelegramCallbackSink:
    """Authenticate one tap, publish its typed steer, and always acknowledge it."""

    def __init__(
        self,
        *,
        bus: FrameBus,
        sender: TelegramSender,
        registry: TelegramCallbackRegistry | None = None,
    ) -> None:
        self._bus = bus
        self._sender = sender
        self._registry = registry or TelegramCallbackRegistry(clock=time.monotonic)

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
        """Mint one message-bound, actor-bound callback capability for a button."""
        return self._registry.issue(
            action=action,
            task_id=task_id,
            target_agent=target_agent,
            channel_id=channel_id,
            message_id=message_id,
            actor=actor,
        )

    async def __call__(self, callback: InboundCallback) -> None:
        """Fail closed on any invalid tap; publish only an authorized typed steer.

        The capability is RESERVED first (an actor-mismatched or replayed tap
        never reserves), the steer is published, and only a successful publish
        COMMITS the reservation. A failed publish ROLLS BACK the reservation so
        the human's retry is not rejected as replayed — the tap stays retryable.

        On a COMMITTED tap the message's inline keyboard is CLEARED (best-effort)
        so the spent buttons cannot invite a confusing replay tap — the typed
        steer is already bus truth. Clearing runs only after commit: a rolled-back
        (retryable) tap keeps its keyboard so the human can retry.
        """
        reservation = self._registry.reserve(
            callback.data,
            channel_id=callback.channel_id,
            message_id=callback.message_id,
            actor=callback.author,
        )
        committed = False
        try:
            if reservation is not None:
                steer = reservation.command.to_steer()
                await self._bus.publish(subjects.task_steer(steer.task_id), encode_frame(steer))
                self._registry.commit(reservation)
                committed = True
                logger.info(
                    "chatd.tg.callback | steer published | task_id={} target={}",
                    steer.task_id,
                    steer.target_agent,
                )
        except Exception as exc:  # noqa: BLE001 -- bus failure cannot strand the UI.
            if reservation is not None:
                # Publish failed AFTER reserving: release so the tap stays
                # retryable instead of stranding as spent.
                self._registry.rollback(reservation)
            logger.error("chatd.tg.callback | publish failed | error_type={}", type(exc).__name__)
        finally:
            await self._sender.answer_callback(
                callback.callback_query_id,
                channel_id=callback.channel_id,
            )
        if committed and callback.channel_id is not None and callback.message_id is not None:
            # The tap succeeded: disable the now-spent keyboard so a stale button
            # never invites a confusing second tap. The markup rides the POLLED
            # default bot (as_agent=None) — the only identity that received the
            # tap — so the edit is issued under it too. Best-effort: a failed clear
            # never un-does the committed steer (the registry already rejects a
            # replay), it only leaves a cosmetic stale keyboard.
            await self._sender.edit_reply_markup(
                SendTarget(channel_id=callback.channel_id),
                callback.message_id,
                reply_markup=None,
            )


# The default-bot acknowledgement line that carries the steer keyboard. It is a
# plain, redaction-safe control affordance posted under the POLLED default bot
# (never a per-identity bot), anchored to the agent's own TLDR. The buttons let
# the ORIGINAL requester steer; the full thought stays under the agent's identity
# bot, unchanged.
_ACK_TEXT = "Steer this direction:"


class TelegramTurnInteraction:
    """Attach an actor-bound steer keyboard to a landed debate TLDR.

    This is the concrete Telegram implementation of the edge's narrow
    :class:`~rtech_chatd.edge.TurnInteraction` hook. It keeps interactive markup
    on a DEFAULT-BOT message that is actually polled — a per-agent identity bot
    runs no getUpdates loop, so a button under it could never deliver its tap.
    The agent's full thought still rides its own identity bot untouched; only this
    small acknowledgement carries the buttons.

    The mint-after-id contract: the callback data is minted ONLY after the
    default-bot acknowledgement message's id is known (send first, then mint bound
    to that id, then attach markup). Every capability is bound to the ORIGINAL
    requester actor (``asker``) plus that exact chat + message, so a copied button
    tapped by another chat member — or the wrong message — fails closed at the
    sink. Attaching the keyboard is best-effort: a failure leaves the plain ack in
    place (no dead buttons), never a half-wired interaction.
    """

    def __init__(self, *, sink: TelegramCallbackSink, sender: TelegramSender) -> None:
        self._sink = sink
        self._sender = sender

    async def offer_debate_tldr(
        self,
        *,
        target: SendTarget,
        anchor_message_id: str,
        task_id: str,
        agent_id: str,
        asker: str | None,
    ) -> None:
        """Post a default-bot ack under the TLDR and attach an actor-bound keyboard.

        Declines (does nothing) when the origin carried no ``asker`` — an
        unbindable capability is never minted. The ack is posted UNSTAMPED
        (``as_agent=None``) so it rides the polled default bot, anchored as a reply
        to the agent's TLDR (``anchor_message_id``). Only once the ack's own
        message id is known is each button's callback data minted, bound to that
        message + the requester, and the keyboard attached via ``edit_reply_markup``.
        """
        if not asker:
            logger.info(
                "chatd.tg.interaction | no asker on origin; no steer keyboard offered "
                "task_id={} agent={}",
                task_id,
                agent_id,
            )
            return
        ack_target = SendTarget(channel_id=target.channel_id, reply_to=anchor_message_id)
        ack_message_id = await self._sender.send(ack_target, _ACK_TEXT, [])
        if ack_message_id is None:
            logger.warning(
                "chatd.tg.interaction | ack send returned no message id; no keyboard "
                "task_id={} agent={}",
                task_id,
                agent_id,
            )
            return
        # Mint AFTER the id is known: each button's callback data is bound to this
        # exact chat + message + the original requester actor.
        buttons = [
            InlineKeyboardButton(
                text=_ACTION_LABELS[action],
                callback_data=self._sink.issue(
                    action=action,
                    task_id=task_id,
                    target_agent=agent_id,
                    channel_id=target.channel_id,
                    message_id=ack_message_id,
                    actor=asker,
                ),
            )
            for action in _TLDR_ACTIONS
        ]
        markup = InlineKeyboardMarkup(inline_keyboard=[buttons])
        edit_target = SendTarget(channel_id=target.channel_id)
        attached = await self._sender.edit_reply_markup(
            edit_target, ack_message_id, reply_markup=markup
        )
        if not attached:
            logger.warning(
                "chatd.tg.interaction | keyboard attach failed; plain ack left in place "
                "task_id={} agent={} message_id={}",
                task_id,
                agent_id,
                ack_message_id,
            )
            return
        logger.info(
            "chatd.tg.interaction | steer keyboard attached task_id={} agent={} message_id={}",
            task_id,
            agent_id,
            ack_message_id,
        )


__all__ = ["TelegramCallbackSink", "TelegramTurnInteraction"]
