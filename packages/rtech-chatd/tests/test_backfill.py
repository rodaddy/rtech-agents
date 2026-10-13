"""Verify bounded three-phase Telegram history backfill behavior."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from rtech_chatd.user_session.backfill import (
    MAX_BACKFILL_LIMIT,
    MAX_ENTITIES,
    MAX_REPLY_EDGES,
    BackfillResult,
    BackfillScopeError,
    EntityRecord,
    MessageRecord,
    ReplyRecord,
    backfill_history,
    backfill_recent,
)
from rtech_chatd.user_session.provisioning import ProvisioningError
from telethon.errors import AuthKeyUnregisteredError, FloodWaitError, PeerFloodError
from telethon.tl.types import User

_ALLOWED_CHATS = frozenset({"-1009"})


@dataclass
class _FakeMessage:
    id: int


@dataclass
class _FakeReplyHeader:
    reply_to_msg_id: int | None


class _FakeDate:
    def __init__(self, epoch: int) -> None:
        self._epoch = epoch

    def timestamp(self) -> float:
        return float(self._epoch)


@dataclass
class _FakeHistoryMessage:
    id: int
    sender_id: int | None = None
    reply_to_msg_id: int | None = None
    epoch: int = 0
    message: str | None = None
    media: object | None = None

    @property
    def reply_to(self) -> _FakeReplyHeader | None:
        if self.reply_to_msg_id is None:
            return None
        return _FakeReplyHeader(reply_to_msg_id=self.reply_to_msg_id)

    @property
    def date(self) -> _FakeDate:
        return _FakeDate(self.epoch)


def _user(uid: int) -> User:
    return User(id=uid)


class _RecentClient:
    def __init__(self, *, history_ids: list[int]) -> None:
        self._history_ids = history_ids

    async def iter_messages(self, chat: object, *, limit: int | None = None) -> object:
        for mid in self._history_ids[: (limit or len(self._history_ids))]:
            yield _FakeMessage(id=mid)


# ------------------------------------------------------------- backfill (bounded)


async def test_backfill_recent_returns_ids_and_clamps() -> None:
    client = _RecentClient(history_ids=list(range(10)))
    ids = await backfill_recent(client, "-1009", allowed_chats=_ALLOWED_CHATS, limit=5)  # type: ignore[arg-type]
    assert ids == [0, 1, 2, 3, 4]


async def test_backfill_recent_zero_limit_is_noop() -> None:
    client = _RecentClient(history_ids=[1, 2, 3])
    assert await backfill_recent(client, "-1009", allowed_chats=_ALLOWED_CHATS, limit=0) == []  # type: ignore[arg-type]


async def test_backfill_recent_clamps_to_max() -> None:
    client = _RecentClient(history_ids=list(range(1000)))
    ids = await backfill_recent(client, "-1009", allowed_chats=_ALLOWED_CHATS, limit=99999)  # type: ignore[arg-type]
    assert len(ids) == MAX_BACKFILL_LIMIT


# ------------------------------------------------------------- three-phase backfill


class _BackfillClient:
    """A Telethon stand-in for the three-phase :func:`backfill_history` path.

    Scripts the two RPCs the backfill issues — ``iter_messages`` (phase 1) and
    per-id ``get_entity`` (phase 2) — plus optional per-call error injection to
    prove bounded FloodWait/retry and fatal-abort behavior. Phase 3 issues no RPC,
    so nothing here scripts it. Every observable is recorded for assertions.
    """

    def __init__(
        self,
        *,
        history: list[_FakeHistoryMessage] | None = None,
        iter_error: BaseException | None = None,
        iter_errors: list[BaseException] | None = None,
        partial_error_after: int | None = None,
        entity_errors: dict[int, list[BaseException]] | None = None,
        unresolved: set[int] | None = None,
    ) -> None:
        self._history = history or []
        self._iter_errors = list(iter_errors or ([] if iter_error is None else [iter_error]))
        self._partial_error_after = partial_error_after
        # id -> queue of exceptions to raise on successive get_entity calls; when
        # the queue drains the id resolves normally (models a transient FloodWait).
        self._entity_errors = {k: list(v) for k, v in (entity_errors or {}).items()}
        self._unresolved = unresolved or set()
        # Observability.
        self.iter_limit: int | None = None
        self.iter_attempts = 0
        self.entity_calls: list[int] = []
        self.sleeps: list[float] = []

    async def iter_messages(self, chat: object, *, limit: int | None = None) -> object:
        self.iter_attempts += 1
        self.iter_limit = limit
        error = self._iter_errors.pop(0) if self._iter_errors else None
        if error is not None and self._partial_error_after is None:
            raise error
        for index, message in enumerate(self._history[: (limit or len(self._history))], start=1):
            yield message
            if error is not None and index == self._partial_error_after:
                raise error

    async def get_entity(self, ident: object) -> object:
        entity_id = int(ident)  # type: ignore[arg-type]
        self.entity_calls.append(entity_id)
        queue = self._entity_errors.get(entity_id)
        if queue:
            raise queue.pop(0)
        if entity_id in self._unresolved:
            raise RuntimeError(f"cannot resolve {entity_id}")
        return _user(entity_id)


# -- phase ordering + caps


async def test_backfill_three_phase_ordering_and_shape() -> None:
    # m3 replies to m2; m2 replies to m1 (all in-window). m1 has media, no text.
    history = [
        _FakeHistoryMessage(id=3, sender_id=100, reply_to_msg_id=2, epoch=30, message="c"),
        _FakeHistoryMessage(id=2, sender_id=200, reply_to_msg_id=1, epoch=20, message="b"),
        _FakeHistoryMessage(id=1, sender_id=100, epoch=10, media=object()),
    ]
    client = _BackfillClient(history=history)

    order: list[str] = []
    seen_messages: list[MessageRecord] = []
    seen_entities: list[EntityRecord] = []
    seen_replies: list[ReplyRecord] = []

    async def on_message(record: MessageRecord) -> None:
        order.append("m")
        seen_messages.append(record)

    async def on_entity(record: EntityRecord) -> None:
        order.append("e")
        seen_entities.append(record)

    async def on_reply(record: ReplyRecord) -> None:
        order.append("r")
        seen_replies.append(record)

    result = await backfill_history(
        client,  # type: ignore[arg-type]
        "-1009",
        allowed_chats=_ALLOWED_CHATS,
        limit=50,
        on_message=on_message,
        on_entity=on_entity,
        on_reply=on_reply,
    )

    assert isinstance(result, BackfillResult)
    # Strict phase ordering: every message callback precedes every entity callback,
    # which precedes every reply callback.
    assert order == ["m", "m", "m", "e", "e", "r", "r"]
    # Phase 1: content-free records, id-graph + shape flags only.
    assert [m.message_id for m in result.messages] == [3, 2, 1]
    assert result.messages[2].has_media is True
    assert result.messages[2].has_text is False
    assert result.messages[0].has_text is True
    assert result.messages[0].sender_id == 100
    assert result.messages[0].date_epoch == 30
    # Phase 2: distinct sender ids resolved (100, 200), de-duplicated + sorted.
    assert [e.entity_id for e in result.entities] == [100, 200]
    assert all(e.resolved and e.kind == "user" for e in result.entities)
    # Phase 3: reply edges with parent-presence.
    assert [(r.child_id, r.parent_id) for r in result.replies] == [(3, 2), (2, 1)]
    assert all(r.parent_present for r in result.replies)


async def test_backfill_entity_total_cap_drops_excess_references() -> None:
    # The 450-message input exceeds the documented total entity cap.
    history = [_FakeHistoryMessage(id=i, sender_id=1000 + i, epoch=i) for i in range(1, 451)]
    client = _BackfillClient(history=history)
    result = await backfill_history(
        client, "-1009", allowed_chats=_ALLOWED_CHATS, limit=MAX_BACKFILL_LIMIT
    )  # type: ignore[arg-type]

    assert MAX_ENTITIES == 200
    # limit clamps to MAX_BACKFILL_LIMIT (500); the 450-message history is smaller,
    # so all 450 survive and each unique sender is resolved once.
    assert client.iter_limit == MAX_BACKFILL_LIMIT
    assert len(result.messages) == 450
    assert len(result.entities) == MAX_ENTITIES
    assert client.entity_calls == sorted(1000 + i for i in range(1, MAX_ENTITIES + 1))


async def test_backfill_reply_total_cap_drops_excess_edges() -> None:
    # The reply linkage cap is a total cap, not a batch size.
    history = [_FakeHistoryMessage(id=1, sender_id=1, epoch=1)]
    history += [
        _FakeHistoryMessage(id=i, sender_id=1, reply_to_msg_id=1, epoch=i) for i in range(2, 152)
    ]
    client = _BackfillClient(history=history)
    result = await backfill_history(
        client, "-1009", allowed_chats=_ALLOWED_CHATS, limit=MAX_BACKFILL_LIMIT
    )  # type: ignore[arg-type]

    assert MAX_REPLY_EDGES == 100
    assert len(result.replies) == MAX_REPLY_EDGES
    assert all(r.parent_id == 1 and r.parent_present for r in result.replies)


# -- reply linkage: out-of-window parent


async def test_backfill_reply_to_out_of_window_parent_marked_absent() -> None:
    # m5 replies to m1, but only m5..m3 are in the window (limit=3). The parent is
    # absent, so the edge is still emitted with parent_present=False.
    history = [
        _FakeHistoryMessage(id=5, sender_id=1, reply_to_msg_id=1, epoch=50),
        _FakeHistoryMessage(id=4, sender_id=1, epoch=40),
        _FakeHistoryMessage(id=3, sender_id=1, epoch=30),
    ]
    client = _BackfillClient(history=history)
    result = await backfill_history(client, "-1009", allowed_chats=_ALLOWED_CHATS, limit=3)  # type: ignore[arg-type]

    assert [(r.child_id, r.parent_id, r.parent_present) for r in result.replies] == [(5, 1, False)]


# -- empty history


async def test_backfill_empty_history_yields_empty_result() -> None:
    client = _BackfillClient(history=[])
    result = await backfill_history(client, "-1009", allowed_chats=_ALLOWED_CHATS, limit=50)  # type: ignore[arg-type]
    assert result == BackfillResult(messages=[], entities=[], replies=[])
    # An empty window must not fan out into phase-2 entity resolution.
    assert client.entity_calls == []


async def test_backfill_zero_limit_skips_all_phases() -> None:
    client = _BackfillClient(history=[_FakeHistoryMessage(id=1, sender_id=1, epoch=1)])
    result = await backfill_history(client, "-1009", allowed_chats=_ALLOWED_CHATS, limit=0)  # type: ignore[arg-type]

    assert result == BackfillResult(messages=[], entities=[], replies=[])
    # Zero limit short-circuits before any RPC (iter_messages never called).
    assert client.iter_limit is None
    assert client.entity_calls == []


# -- bounded FloodWait / retries in phase 2


async def test_backfill_entity_flood_wait_bounded_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("rtech_chatd.user_session.backfill.asyncio.sleep", fake_sleep)

    # Entity 100 FloodWaits once (2s, within the 60s ceiling) then resolves; the
    # backfill must honor the bounded wait and retry, not abort.
    history = [_FakeHistoryMessage(id=1, sender_id=100, epoch=1)]
    client = _BackfillClient(
        history=history,
        entity_errors={100: [FloodWaitError(request=None, capture=2)]},
    )
    result = await backfill_history(client, "-1009", allowed_chats=_ALLOWED_CHATS, limit=50)  # type: ignore[arg-type]

    assert slept == [2.0]  # honored exactly one bounded wait
    assert [e.entity_id for e in result.entities] == [100]
    assert result.entities[0].resolved is True
    # get_entity called twice for id 100: the FloodWait attempt + the retry.
    assert client.entity_calls == [100, 100]


async def test_backfill_entity_flood_wait_over_ceiling_fails_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(seconds: float) -> None:  # pragma: no cover - must not run
        raise AssertionError("over-ceiling FloodWait must not sleep")

    monkeypatch.setattr("rtech_chatd.user_session.backfill.asyncio.sleep", fake_sleep)

    history = [_FakeHistoryMessage(id=1, sender_id=100, epoch=1)]
    client = _BackfillClient(
        history=history,
        entity_errors={100: [FloodWaitError(request=None, capture=99999)]},
    )
    with pytest.raises(ProvisioningError) as exc:
        await backfill_history(client, "-1009", allowed_chats=_ALLOWED_CHATS, limit=50)  # type: ignore[arg-type]

    assert exc.value.kind == "flood_wait"
    assert exc.value.retry_after_s == 99999


async def test_backfill_entity_flood_wait_exhausts_retries_fails_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("rtech_chatd.user_session.backfill.asyncio.sleep", fake_sleep)

    # Four bounded FloodWaits in a row: 3 retries are honored, the 4th exhausts the
    # finite retry budget and fails loud (never an infinite loop).
    history = [_FakeHistoryMessage(id=1, sender_id=100, epoch=1)]
    client = _BackfillClient(
        history=history,
        entity_errors={100: [FloodWaitError(request=None, capture=1) for _ in range(4)]},
    )
    with pytest.raises(ProvisioningError) as exc:
        await backfill_history(client, "-1009", allowed_chats=_ALLOWED_CHATS, limit=50)  # type: ignore[arg-type]

    assert exc.value.kind == "flood_wait"
    assert slept == [1.0, 1.0, 1.0]  # exactly the retry budget, then loud failure


# -- fatal aborts propagate from each phase


async def test_backfill_phase1_auth_revoked_aborts_loud() -> None:
    client = _BackfillClient(iter_error=AuthKeyUnregisteredError(request=None))
    with pytest.raises(ProvisioningError) as exc:
        await backfill_history(client, "-1009", allowed_chats=_ALLOWED_CHATS, limit=50)  # type: ignore[arg-type]
    assert exc.value.kind == "auth_revoked"


async def test_backfill_phase2_peer_flood_aborts_loud() -> None:
    history = [_FakeHistoryMessage(id=1, sender_id=100, epoch=1)]
    client = _BackfillClient(
        history=history,
        entity_errors={100: [PeerFloodError(request=None)]},
    )
    with pytest.raises(ProvisioningError) as exc:
        await backfill_history(client, "-1009", allowed_chats=_ALLOWED_CHATS, limit=50)  # type: ignore[arg-type]
    assert exc.value.kind == "peer_flood"


# -- non-fatal per-id failure surfaces an unresolved record, does not abort


async def test_backfill_entity_non_fatal_failure_marks_unresolved() -> None:
    history = [
        _FakeHistoryMessage(id=2, sender_id=200, epoch=20),
        _FakeHistoryMessage(id=1, sender_id=100, epoch=10),
    ]
    client = _BackfillClient(history=history, unresolved={200})
    result = await backfill_history(client, "-1009", allowed_chats=_ALLOWED_CHATS, limit=50)  # type: ignore[arg-type]
    by_id = {entity.entity_id: entity for entity in result.entities}
    assert by_id[100].resolved is True
    assert by_id[200].resolved is False
    assert by_id[200].kind == "unknown"
    # A single bad id must not drop the good one or abort the phase.
    assert set(by_id) == {100, 200}


async def test_backfill_scope_rejects_unapproved_chat_before_any_read() -> None:
    client = _BackfillClient(history=[_FakeHistoryMessage(id=1, sender_id=1)])
    with pytest.raises(BackfillScopeError):
        await backfill_history(client, "-1009", allowed_chats=frozenset({"-1008"}))  # type: ignore[arg-type]
    assert client.iter_attempts == 0
    assert client.entity_calls == []


async def test_backfill_recent_rejects_unapproved_chat_before_any_read() -> None:
    client = _RecentClient(history_ids=[1, 2])
    with pytest.raises(BackfillScopeError):
        await backfill_recent(client, "-1009", allowed_chats=frozenset({"-1008"}))  # type: ignore[arg-type]


async def test_backfill_phase1_flood_wait_retries_before_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []
    callbacks: list[int] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    async def on_message(record: MessageRecord) -> None:
        callbacks.append(record.message_id)

    monkeypatch.setattr("rtech_chatd.user_session.backfill.asyncio.sleep", fake_sleep)
    client = _BackfillClient(
        history=[_FakeHistoryMessage(id=2, sender_id=1), _FakeHistoryMessage(id=1, sender_id=2)],
        iter_errors=[FloodWaitError(request=None, capture=1)],
        partial_error_after=1,
    )
    result = await backfill_history(
        client,
        "-1009",
        allowed_chats=_ALLOWED_CHATS,
        on_message=on_message,
    )  # type: ignore[arg-type]

    assert slept == [1.0]
    assert client.iter_attempts == 2
    assert callbacks == [2, 1]
    assert [record.message_id for record in result.messages] == [2, 1]
