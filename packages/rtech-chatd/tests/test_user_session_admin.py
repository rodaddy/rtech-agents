"""Admin-op logic: id canonicalization, classification, and the fake-client seams.

The real Telethon network calls (``iter_dialogs``, ``ToggleForumRequest``) are
NOT exercised — they need a live account. Instead the pure helpers
(``_canonical_id``, ``_classify``) are tested directly, and ``list_dialogs`` /
``ensure_forum`` are driven through a minimal fake client so the branching
(skip-non-group, already-a-forum no-op, flip-when-off) is covered without a
socket.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import rtech_chatd.user_session.admin as admin_mod
from loguru import logger
from rtech_chatd.user_session.admin import (
    _CONNECT_BACKOFF_S,
    _CONNECT_FLOOD_WAIT_CEILING_S,
    DialogInfo,
    _canonical_id,
    _classify,
    _connect_with_backoff,
    create_supergroup,
    ensure_forum,
    list_dialogs,
    promote_admins,
)
from telethon.errors import FloodWaitError
from telethon.tl.types import Channel, Chat, User

# Real Telethon entities need several positional args; these builders supply the
# minimum so the admin helpers (which use telethon.utils.get_peer_id and real
# isinstance checks) run against genuine types — not stand-ins that get_peer_id
# would reject.

_EPOCH = 0  # ``date`` fields; value is irrelevant to id/kind logic.


def _channel(
    cid: int, *, broadcast: bool = False, megagroup: bool = False, forum: bool = False
) -> Channel:
    return Channel(
        id=cid,
        title="c",
        photo=None,
        date=_EPOCH,
        broadcast=broadcast,
        megagroup=megagroup,
        forum=forum,
    )


def _chat(cid: int) -> Chat:
    return Chat(
        id=cid,
        title="g",
        photo=None,
        participants_count=1,
        date=_EPOCH,
        version=1,
    )


@dataclass
class _FakeDialog:
    title: str
    entity: object


class _FakeClient:
    """Minimal async client exposing only what the admin ops call.

    Routes __call__ by request type: ToggleForum records the flip; CreateChannel
    returns ``create_result``; InviteToChannel records the invite (or raises for
    members in ``invite_fail``).
    """

    def __init__(
        self,
        dialogs: list[_FakeDialog],
        entity: object | None = None,
        create_result: object = None,
        invite_fail: set[str] | None = None,
    ) -> None:
        self._dialogs = dialogs
        self._entity = entity
        self.create_result = create_result
        self.invite_fail = invite_fail or set()
        self.toggled: list[tuple[object, object]] = []
        self.invited: list[object] = []
        self.promoted: list[object] = []

    async def iter_dialogs(self) -> object:
        for d in self._dialogs:
            yield d

    async def get_entity(self, chat: object) -> object:
        return self._entity

    async def get_input_entity(self, ident: object) -> object:
        # The admin op passes whatever get_input_entity returns straight into the
        # invite; echoing the identifier keeps the fake simple and assertable.
        if str(ident) in self.invite_fail:
            raise RuntimeError(f"cannot resolve {ident}")
        return ident

    async def __call__(self, request: object) -> object:
        name = type(request).__name__
        if name == "ToggleForumRequest":
            self.toggled.append(
                (getattr(request, "channel", None), getattr(request, "enabled", None))
            )
            return None
        if name == "CreateChannelRequest":
            return self.create_result
        if name == "InviteToChannelRequest":
            users = getattr(request, "users", [])
            for u in users:
                if str(u) in self.invite_fail:
                    raise RuntimeError(f"invite failed for {u}")
                self.invited.append(u)
            return None
        if name == "EditAdminRequest":
            uid = getattr(request, "user_id", None)
            if str(uid) in self.invite_fail:
                raise RuntimeError(f"promote failed for {uid}")
            self.promoted.append(uid)
            return None
        return None


# --------------------------------------------------------------- pure helpers


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("-1001111111111", -1001111111111),  # marked id string -> int
        ("5000000001", 5000000001),
        (" -100 ", -100),  # trimmed
        ("@Bilby_example_bot", "@Bilby_example_bot"),  # handle untouched
        (-1001111111111, -1001111111111),  # already int
    ],
)
def test_coerce_chat_ref(raw: object, expected: object) -> None:
    from rtech_chatd.user_session.admin import _coerce_chat_ref

    assert _coerce_chat_ref(raw) == expected  # type: ignore[arg-type]


def test_canonical_id_channel_uses_minus100_form() -> None:
    # get_peer_id marks a Channel/supergroup into the -100<id> space. For a
    # real-width supergroup id the result is the familiar -100XXXXXXXXXX; for a
    # small id it is zero-padded across the full channel-id width. Both are what
    # the Bot API / edge adapter expect — which is exactly why we defer to the
    # library rather than a hand-rolled f"-100{id}" (that mis-marks small ids).
    assert _canonical_id(_channel(1234567890)) == "-1001234567890"
    assert _canonical_id(_channel(2)) == "-1000000000002"


def test_canonical_id_basic_chat_uses_plain_negative() -> None:
    assert _canonical_id(_chat(42)) == "-42"


@pytest.mark.parametrize(
    "entity,expected",
    [
        (_channel(1, broadcast=True), "broadcast"),
        (_channel(1, megagroup=True), "megagroup"),
        (_channel(1), "supergroup"),
        (_chat(1), "group"),
    ],
)
def test_classify(entity: object, expected: str) -> None:
    assert _classify(entity) == expected  # type: ignore[arg-type]


# --------------------------------------------------------------- list_dialogs


async def test_list_dialogs_skips_users_and_reports_forum_flag() -> None:
    dialogs = [
        _FakeDialog("Broadcast", _channel(1, broadcast=True)),
        _FakeDialog("A Forum Group", _channel(2, megagroup=True, forum=True)),
        _FakeDialog("Plain Super", _channel(3)),
        _FakeDialog("Basic Group", _chat(4)),
        _FakeDialog("Some Person", User(id=5)),  # must be skipped
    ]
    result = await list_dialogs(_FakeClient(dialogs))  # type: ignore[arg-type]
    assert [d.title for d in result] == [
        "Broadcast",
        "A Forum Group",
        "Plain Super",
        "Basic Group",
    ]
    forum_group = next(d for d in result if d.title == "A Forum Group")
    assert forum_group.is_forum is True
    assert forum_group.chat_id == "-1000000000002"  # get_peer_id marking
    plain = next(d for d in result if d.title == "Plain Super")
    assert plain.is_forum is False
    assert all(isinstance(d, DialogInfo) for d in result)


# --------------------------------------------------------------- ensure_forum


async def test_ensure_forum_noop_when_already_forum() -> None:
    client = _FakeClient([], entity=_channel(9, megagroup=True, forum=True))
    flipped = await ensure_forum(client, "-1009")  # type: ignore[arg-type]
    assert flipped is False
    assert client.toggled == []  # no mutating request sent


async def test_ensure_forum_flips_when_off() -> None:
    entity = _channel(9, megagroup=True, forum=False)
    client = _FakeClient([], entity=entity)
    flipped = await ensure_forum(client, "-1009")  # type: ignore[arg-type]
    assert flipped is True
    assert len(client.toggled) == 1
    channel, enabled = client.toggled[0]
    assert channel is entity
    assert enabled is True


# --------------------------------------------------------------- create_supergroup


@dataclass
class _CreateUpdates:
    """Stands in for the updates object CreateChannelRequest returns."""

    chats: list[object]


async def test_create_supergroup_returns_marked_id_and_adds_members() -> None:
    new_super = _channel(424242, megagroup=True, forum=True)
    client = _FakeClient([], create_result=_CreateUpdates(chats=[new_super]))
    result = await create_supergroup(
        client,  # type: ignore[arg-type]
        "Trio Debate",
        members=["@Bilby_example_bot", "@Skippy_example_bot", "@Nagatha_example_bot"],
    )
    assert result.chat_id == "-1000000424242"  # get_peer_id marking
    assert result.title == "Trio Debate"
    assert result.added == [
        "@Bilby_example_bot",
        "@Skippy_example_bot",
        "@Nagatha_example_bot",
    ]
    assert result.failed == {}
    assert len(client.invited) == 3


async def test_create_supergroup_records_failed_members_without_aborting() -> None:
    new_super = _channel(5, megagroup=True)
    client = _FakeClient(
        [],
        create_result=_CreateUpdates(chats=[new_super]),
        invite_fail={"@ghost_bot"},
    )
    result = await create_supergroup(
        client,  # type: ignore[arg-type]
        "Trio Debate",
        members=["@Bilby_example_bot", "@ghost_bot"],
    )
    assert result.added == ["@Bilby_example_bot"]
    assert "@ghost_bot" in result.failed
    # The good member still got added despite the bad one failing.
    assert len(client.invited) == 1


async def test_create_supergroup_raises_when_no_channel_in_result() -> None:
    client = _FakeClient([], create_result=_CreateUpdates(chats=[]))
    with pytest.raises(ValueError, match="no supergroup"):
        await create_supergroup(client, "Trio Debate")  # type: ignore[arg-type]


async def test_create_supergroup_no_members_is_fine() -> None:
    new_super = _channel(7, megagroup=True)
    client = _FakeClient([], create_result=_CreateUpdates(chats=[new_super]))
    result = await create_supergroup(client, "Empty")  # type: ignore[arg-type]
    assert result.added == []
    assert result.failed == {}


# --------------------------------------------------------------- promote_admins


async def test_promote_admins_all_ok() -> None:
    client = _FakeClient([], entity=_channel(9, megagroup=True))
    results = await promote_admins(
        client,  # type: ignore[arg-type]
        "-1009",
        ["@Bilby_example_bot", "@Skippy_example_bot", "5000000001"],
    )
    assert results == {
        "@Bilby_example_bot": "ok",
        "@Skippy_example_bot": "ok",
        "5000000001": "ok",
    }
    assert len(client.promoted) == 3


async def test_promote_admins_records_failures_without_aborting() -> None:
    client = _FakeClient([], entity=_channel(9, megagroup=True), invite_fail={"@ghost_bot"})
    results = await promote_admins(
        client,  # type: ignore[arg-type]
        "-1009",
        ["@Bilby_example_bot", "@ghost_bot"],
    )
    assert results["@Bilby_example_bot"] == "ok"
    assert results["@ghost_bot"] != "ok"
    # The good member was still promoted despite the bad one failing.
    assert len(client.promoted) == 1


# --------------------------------------------------- _connect_with_backoff (bounded)


class _ScriptedConnectClient:
    """A fake client whose ``connect`` follows a scripted per-attempt outcome.

    ``outcomes`` is one entry per attempt: ``None`` = succeed, an exception
    instance = raise it. Each connect pops the next outcome, so a test scripts
    "fail twice then succeed" as ``[err, err, None]``. Attempts are counted for
    assertions.
    """

    def __init__(self, outcomes: list[BaseException | None]) -> None:
        self._outcomes = list(outcomes)
        self.attempts = 0

    async def connect(self) -> None:
        self.attempts += 1
        outcome = self._outcomes.pop(0)
        if outcome is not None:
            raise outcome


@pytest.fixture
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Patch ``asyncio.sleep`` in the admin module to record (not perform) waits."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(admin_mod.asyncio, "sleep", fake_sleep)
    return slept


async def test_connect_backoff_succeeds_first_try_no_sleep(_no_sleep: list[float]) -> None:
    client = _ScriptedConnectClient([None])
    await _connect_with_backoff(client)  # type: ignore[arg-type]
    assert client.attempts == 1
    assert _no_sleep == []  # no retry, no backoff sleep


async def test_connect_backoff_retries_transient_then_succeeds(_no_sleep: list[float]) -> None:
    client = _ScriptedConnectClient([ConnectionError("blip"), ConnectionError("blip"), None])
    await _connect_with_backoff(client)  # type: ignore[arg-type]
    assert client.attempts == 3
    # Slept on the fixed ladder for the two failed attempts.
    assert _no_sleep == [_CONNECT_BACKOFF_S[0], _CONNECT_BACKOFF_S[1]]


async def test_connect_backoff_exhausts_ladder_then_raises(_no_sleep: list[float]) -> None:
    # One more failure than the ladder has steps -> bounded, then re-raise.
    attempts = len(_CONNECT_BACKOFF_S) + 1
    client = _ScriptedConnectClient([ConnectionError("dead")] * attempts)
    with pytest.raises(ConnectionError):
        await _connect_with_backoff(client)  # type: ignore[arg-type]
    assert client.attempts == attempts
    # Slept exactly once per ladder step, never unbounded.
    assert _no_sleep == list(_CONNECT_BACKOFF_S)


async def test_connect_backoff_honors_floodwait_under_ceiling(_no_sleep: list[float]) -> None:
    wait = _CONNECT_FLOOD_WAIT_CEILING_S - 1
    client = _ScriptedConnectClient([FloodWaitError(request=None, capture=wait), None])
    await _connect_with_backoff(client)  # type: ignore[arg-type]
    assert client.attempts == 2
    assert _no_sleep == [wait]  # honored the server wait, then reconnected


async def test_connect_backoff_floodwait_over_ceiling_fails_loud(_no_sleep: list[float]) -> None:
    wait = _CONNECT_FLOOD_WAIT_CEILING_S + 1
    client = _ScriptedConnectClient([FloodWaitError(request=None, capture=wait)])
    with pytest.raises(FloodWaitError):
        await _connect_with_backoff(client)  # type: ignore[arg-type]
    assert client.attempts == 1
    assert _no_sleep == []  # never parked for an over-ceiling wait


async def test_connect_backoff_does_not_log_exception_text(_no_sleep: list[float]) -> None:
    client = _ScriptedConnectClient([RuntimeError("session_string=do-not-log"), None])
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")
    try:
        await _connect_with_backoff(client)  # type: ignore[arg-type]
    finally:
        logger.remove(sink_id)
    rendered = "".join(messages)
    assert "RuntimeError" in rendered  # class name is fine
    assert "do-not-log" not in rendered
    assert "session_string" not in rendered
