"""Provisioning orchestration: idempotent rerun, canonical id, forum, promote,
privacy fallback, and revoked-session failure — all with mocked Telethon RPCs.

No live account is touched. A single :class:`_FakeClient` routes every RPC the
provisioning boundary issues (``CreateChannelRequest``, ``ToggleForumRequest``,
``InviteToChannelRequest``, ``EditAdminRequest``, ``ExportChatInviteRequest``,
``get_participants``, ``send_message``, and ``iter_dialogs``) so
the recipe's branching is proven end-to-end without a socket (issue #110):

* idempotent rerun — an existing owned megagroup by title is REUSED, not recreated;
* canonical ``-100…`` id is what the result exposes;
* Topics enable (and its no-op when already a forum);
* promotion carries ``manage_topics`` + the least-privilege right-set;
* privacy-restricted add falls back to an invite link;
* a revoked session (``AuthKeyUnregisteredError``) fails loud with ``auth_revoked``;
* FloodWait / PeerFlood classify and abort loudly, credential-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from loguru import logger
from rtech_chatd.user_session.admin import promote_admins as promote_admins_primitive
from rtech_chatd.user_session.provisioning import (
    BOT_ADMIN_RIGHTS,
    DEFAULT_BOT_HANDLES,
    ProvisioningError,
    audit_group,
    export_invite_link,
    find_supergroup_by_title,
    invite_members,
    promote_members,
    provision_debate_group,
    resolve_bot_handles,
    send_message,
)
from telethon.errors import (
    AuthKeyUnregisteredError,
    FloodWaitError,
    PeerFloodError,
    UserPrivacyRestrictedError,
)
from telethon.tl.types import Channel, User

_EPOCH = 0


def _channel(
    cid: int,
    *,
    title: str = "c",
    broadcast: bool = False,
    megagroup: bool = True,
    forum: bool = False,
    creator: bool = True,
) -> Channel:
    return Channel(
        id=cid,
        title=title,
        photo=None,
        date=_EPOCH,
        broadcast=broadcast,
        megagroup=megagroup,
        forum=forum,
        creator=creator,
    )


def _bot(handle: str, *, bot: bool = True) -> User:
    """Build a resolved Telegram identity for the configured bot handle."""
    return User(id=abs(hash(handle)) % 10_000 + 1, bot=bot, username=handle.removeprefix("@"))


@dataclass
class _FakeDialog:
    title: str
    entity: object


@dataclass
class _CreateUpdates:
    chats: list[object]


@dataclass
class _ExportedInvite:
    link: str


@dataclass
class _SentMessage:
    id: int


class _FakeParticipants(list):  # type: ignore[type-arg]
    """A list subclass carrying a ``total`` like Telethon's participants result."""

    total: int = 0


class _FakeClient:
    """Routes every provisioning RPC by request/method, scripted per-scenario.

    Rather than a knob per behavior, callers set the small state this fake reads:
    ``dialogs`` (for discovery), ``entity`` (what get_entity returns),
    ``create_result`` (CreateChannel updates), ``privacy_fail`` (members whose
    direct invite raises UserPrivacyRestricted), ``raise_on`` (a request-type name
    → exception, to inject FloodWait/PeerFlood/AuthKey on a specific RPC), and
    ``admins`` (the participants-admins audit result).
    """

    def __init__(
        self,
        *,
        dialogs: list[_FakeDialog] | None = None,
        entity: object | None = None,
        create_result: object = None,
        privacy_fail: set[str] | None = None,
        invite_fail: set[str] | None = None,
        raise_on: dict[str, BaseException] | None = None,
        admins: list[object] | None = None,
        member_total: int = 3,
        non_bot_handles: set[str] | None = None,
    ) -> None:
        self._dialogs = dialogs or []
        self._entity = entity
        self.create_result = create_result
        self.privacy_fail = privacy_fail or set()
        self.invite_fail = invite_fail or set()
        self.raise_on = raise_on or {}
        self._admins = admins or []
        self._member_total = member_total
        self._non_bot_handles = non_bot_handles or set()
        # Observability for assertions.
        self.toggled: list[tuple[object, object]] = []
        self.invited: list[object] = []
        self.promoted: list[tuple[object, object]] = []
        self.exported = 0
        self.sent: list[tuple[object, str]] = []
        self.created_titles: list[str] = []

    async def iter_dialogs(self) -> object:
        for d in self._dialogs:
            yield d

    async def get_entity(self, chat: object) -> object:
        if isinstance(chat, str) and chat in DEFAULT_BOT_HANDLES.values():
            return _bot(chat, bot=chat not in self._non_bot_handles)
        return self._entity

    async def get_input_entity(self, ident: object) -> object:
        if isinstance(ident, User):
            username = getattr(ident, "username", None)
            if username is None:
                raise RuntimeError("verified bot has no username")
            ident = f"@{username}"
        if str(ident) in self.invite_fail:
            raise RuntimeError(f"cannot resolve {ident}")
        return ident

    async def get_participants(
        self, entity: object, *, filter: object = None, limit: int | None = None
    ) -> object:
        if "get_participants" in self.raise_on:
            raise self.raise_on["get_participants"]
        if filter is not None:
            return list(self._admins)
        result = _FakeParticipants(self._admins)
        result.total = self._member_total
        return result

    async def send_message(self, chat: object, text: str, *, reply_to: object = None) -> object:
        if "send_message" in self.raise_on:
            raise self.raise_on["send_message"]
        self.sent.append((chat, text))
        return _SentMessage(id=777)

    async def __call__(self, request: object) -> object:
        name = type(request).__name__
        if name in self.raise_on:
            raise self.raise_on[name]
        if name == "ToggleForumRequest":
            self.toggled.append(
                (getattr(request, "channel", None), getattr(request, "enabled", None))
            )
            return None
        if name == "CreateChannelRequest":
            self.created_titles.append(getattr(request, "title", ""))
            return self.create_result
        if name == "InviteToChannelRequest":
            for u in getattr(request, "users", []):
                if str(u) in self.privacy_fail:
                    raise UserPrivacyRestrictedError(request=None)
                if str(u) in self.invite_fail:
                    raise RuntimeError(f"invite failed for {u}")
                self.invited.append(u)
            return None
        if name == "EditAdminRequest":
            uid = getattr(request, "user_id", None)
            if str(uid) in self.invite_fail:
                raise RuntimeError(f"promote failed for {uid}")
            self.promoted.append((uid, getattr(request, "admin_rights", None)))
            return None
        if name == "ExportChatInviteRequest":
            self.exported += 1
            return _ExportedInvite(link="https://t.me/+FAKEINVITE")
        return None


# ------------------------------------------------------------- resolve_bot_handles


def test_resolve_bot_handles_maps_slugs() -> None:
    assert resolve_bot_handles(["skippy", "nagatha", "bilby"]) == [
        DEFAULT_BOT_HANDLES["skippy"],
        DEFAULT_BOT_HANDLES["nagatha"],
        DEFAULT_BOT_HANDLES["bilby"],
    ]


def test_resolve_bot_handles_accepts_only_configured_handle() -> None:
    assert resolve_bot_handles(["@skippy_example_bot"]) == [DEFAULT_BOT_HANDLES["skippy"]]


def test_resolve_bot_handles_rejects_caller_supplied_handle() -> None:
    with pytest.raises(ValueError, match="configured fleet bot"):
        resolve_bot_handles(["@custom_bot"])


def test_resolve_bot_handles_rejects_caller_supplied_id() -> None:
    with pytest.raises(ValueError, match="unknown bot slug"):
        resolve_bot_handles(["12345"])


def test_resolve_bot_handles_dedups_preserving_order() -> None:
    assert resolve_bot_handles(["skippy", "skippy", "bilby"]) == [
        DEFAULT_BOT_HANDLES["skippy"],
        DEFAULT_BOT_HANDLES["bilby"],
    ]


def test_resolve_bot_handles_unknown_slug_fails_loud() -> None:
    with pytest.raises(ValueError, match="unknown bot slug"):
        resolve_bot_handles(["ghost"])


def test_resolve_bot_handles_empty_fails_loud() -> None:
    with pytest.raises(ValueError, match="no bots requested"):
        resolve_bot_handles([])


# ------------------------------------------------------------- find_supergroup_by_title


async def test_find_supergroup_returns_marked_id_for_owned_match() -> None:
    client = _FakeClient(
        dialogs=[_FakeDialog("Trio Debate", _channel(424242, title="Trio Debate"))]
    )
    found = await find_supergroup_by_title(client, "Trio Debate")  # type: ignore[arg-type]
    assert found == "-1000000424242"


async def test_find_supergroup_ignores_non_creator_and_broadcast_and_mismatch() -> None:
    client = _FakeClient(
        dialogs=[
            _FakeDialog("Trio Debate", _channel(1, title="Trio Debate", creator=False)),
            _FakeDialog("Trio Debate", _channel(2, title="Trio Debate", broadcast=True)),
            _FakeDialog("Other Group", _channel(3, title="Other Group")),
        ]
    )
    assert await find_supergroup_by_title(client, "Trio Debate") is None  # type: ignore[arg-type]


# ------------------------------------------------------------- invite_members (privacy)


async def test_invite_members_privacy_block_is_truthful_and_never_mints_link() -> None:
    entity = _channel(9, megagroup=True)
    client = _FakeClient(entity=entity, privacy_fail={"@Nagatha_example_bot"})
    outcome = await invite_members(
        client,  # type: ignore[arg-type]
        "-1009",
        ["@Skippy_example_bot", "@Nagatha_example_bot"],
    )
    assert outcome.added == ["@Skippy_example_bot"]
    assert outcome.privacy_blocked == ["@Nagatha_example_bot"]
    assert "@Nagatha_example_bot" in outcome.failed
    assert "cannot join via an invite link" in outcome.failed["@Nagatha_example_bot"]
    assert client.exported == 0


async def test_invite_members_no_privacy_no_link() -> None:
    client = _FakeClient(entity=_channel(9))
    outcome = await invite_members(
        client,  # type: ignore[arg-type]
        "-1009",
        ["@Skippy_example_bot"],
    )
    assert outcome.privacy_blocked == []
    assert outcome.failed == {}
    assert client.exported == 0


async def test_invite_members_refuses_unconfigured_handle_before_inviting() -> None:
    client = _FakeClient(entity=_channel(9), invite_fail={"@ghost_bot"})
    with pytest.raises(ValueError, match="configured fleet bot"):
        await invite_members(
            client,  # type: ignore[arg-type]
            "-1009",
            ["@Skippy_example_bot", "@ghost_bot"],
        )
    assert client.invited == []


async def test_invite_members_refuses_configured_identity_that_is_not_a_bot() -> None:
    client = _FakeClient(entity=_channel(9), non_bot_handles={DEFAULT_BOT_HANDLES["skippy"]})
    outcome = await invite_members(
        client,  # type: ignore[arg-type]
        "-1009",
        ["@Skippy_example_bot"],
    )
    assert outcome.added == []
    assert outcome.failed == {"@Skippy_example_bot": "ValueError"}
    assert client.invited == []


# ------------------------------------------------------------- promote_members rights


async def test_promote_members_grants_manage_topics_least_privilege() -> None:
    client = _FakeClient(entity=_channel(9))
    results = await promote_members(
        client,  # type: ignore[arg-type]
        "-1009",
        ["@Skippy_example_bot"],
    )
    assert results == {"@Skippy_example_bot": "ok"}
    _uid, rights = client.promoted[0]
    assert rights.manage_topics is True
    assert rights.add_admins is False  # least privilege
    assert rights.change_info is False
    assert rights.post_messages is True
    assert rights is BOT_ADMIN_RIGHTS


async def test_promote_members_refuses_configured_identity_that_is_not_a_bot() -> None:
    client = _FakeClient(entity=_channel(9), non_bot_handles={DEFAULT_BOT_HANDLES["skippy"]})
    with pytest.raises(ValueError, match="did not resolve to a Telegram bot"):
        await promote_members(client, "-1009", ["@Skippy_example_bot"])  # type: ignore[arg-type]
    assert client.promoted == []


async def test_promote_admins_does_not_log_exception_text() -> None:
    class _SensitiveFailureClient(_FakeClient):
        async def __call__(self, request: object) -> object:
            if type(request).__name__ == "EditAdminRequest":
                raise RuntimeError("session_string=do-not-log")
            return await super().__call__(request)

    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")
    try:
        results = await promote_admins_primitive(
            _SensitiveFailureClient(entity=_channel(9)),  # type: ignore[arg-type]
            "-1009",
            ["@Skippy_example_bot"],
        )
    finally:
        logger.remove(sink_id)

    assert results == {"@Skippy_example_bot": "RuntimeError"}
    rendered = "".join(messages)
    assert "RuntimeError" in rendered
    assert "do-not-log" not in rendered
    assert "session_string" not in rendered


# ------------------------------------------------------------- export_invite_link


async def test_export_invite_link_returns_url() -> None:
    client = _FakeClient(entity=_channel(9))
    link = await export_invite_link(client, "-1009")  # type: ignore[arg-type]
    assert link == "https://t.me/+FAKEINVITE"


async def test_export_invite_link_raises_when_no_link() -> None:
    class _NoLinkClient(_FakeClient):
        async def __call__(self, request: object) -> object:
            if type(request).__name__ == "ExportChatInviteRequest":
                return _ExportedInvite(link="")
            return await super().__call__(request)

    client = _NoLinkClient(entity=_channel(9))
    with pytest.raises(ValueError, match="no link"):
        await export_invite_link(client, "-1009")  # type: ignore[arg-type]


# ------------------------------------------------------------- audit_group


async def test_audit_group_reports_forum_admins_members() -> None:
    entity = _channel(424242, title="Trio Debate", forum=True)
    admins = [_channel(1001, megagroup=False, broadcast=False)]  # any peer-id-able entity
    client = _FakeClient(entity=entity, admins=admins, member_total=5)
    audit = await audit_group(client, "-1000000424242")  # type: ignore[arg-type]
    assert audit.chat_id == "-1000000424242"
    assert audit.title == "Trio Debate"
    assert audit.is_forum is True
    assert audit.member_count == 5
    assert len(audit.admins) == 1


# ------------------------------------------------------------- send_message returns id


async def test_send_message_returns_message_id() -> None:
    client = _FakeClient(entity=_channel(9))
    mid = await send_message(client, "-1009", "hello topic")  # type: ignore[arg-type]
    assert mid == 777
    # _coerce_chat_ref normalizes the numeric-string chat to an int before the send.
    assert client.sent == [(-1009, "hello topic")]


# ------------------------------------------------------------- fail-loud classes


async def test_auth_revoked_raises_provisioning_error() -> None:
    client = _FakeClient(
        entity=_channel(9),
        raise_on={"InviteToChannelRequest": AuthKeyUnregisteredError(request=None)},
    )
    with pytest.raises(ProvisioningError) as exc:
        await invite_members(client, "-1009", ["@Skippy_example_bot"])  # type: ignore[arg-type]
    assert exc.value.kind == "auth_revoked"
    # Credential-free remediation.
    assert "re-mint" in str(exc.value)


async def test_flood_wait_surfaces_retry_after() -> None:
    flood = FloodWaitError(request=None, capture=42)
    client = _FakeClient(
        entity=_channel(9),
        raise_on={"InviteToChannelRequest": flood},
    )
    with pytest.raises(ProvisioningError) as exc:
        await invite_members(client, "-1009", ["@Skippy_example_bot"])  # type: ignore[arg-type]
    assert exc.value.kind == "flood_wait"
    assert exc.value.retry_after_s == 42


async def test_peer_flood_aborts_loud() -> None:
    client = _FakeClient(
        entity=_channel(9),
        raise_on={"InviteToChannelRequest": PeerFloodError(request=None)},
    )
    with pytest.raises(ProvisioningError) as exc:
        await invite_members(client, "-1009", ["@Skippy_example_bot"])  # type: ignore[arg-type]
    assert exc.value.kind == "peer_flood"


@pytest.mark.parametrize(
    ("rpc_error", "expected_kind"),
    [
        (AuthKeyUnregisteredError(request=None), "auth_revoked"),
        (FloodWaitError(request=None, capture=42), "flood_wait"),
        (PeerFloodError(request=None), "peer_flood"),
    ],
)
async def test_promote_members_classifies_fatal_errors_distinctly(
    rpc_error: BaseException, expected_kind: str
) -> None:
    client = _FakeClient(
        entity=_channel(9),
        raise_on={"EditAdminRequest": rpc_error},
    )
    with pytest.raises(ProvisioningError) as exc:
        await promote_members(client, "-1009", ["@Skippy_example_bot"])  # type: ignore[arg-type]
    assert exc.value.kind == expected_kind


# ------------------------------------------------------------- full recipe: idempotent


async def test_provision_creates_when_absent() -> None:
    new_super = _channel(424242, title="Trio Debate", megagroup=True, forum=True)
    client = _FakeClient(
        dialogs=[],  # nothing to discover -> create
        entity=new_super,
        create_result=_CreateUpdates(chats=[new_super]),
        admins=[new_super],
        member_total=4,
    )
    result = await provision_debate_group(
        client,  # type: ignore[arg-type]
        "Trio Debate",
        ["skippy", "nagatha", "bilby"],
    )
    assert result.created is True
    assert result.chat_id == "-1000000424242"
    assert result.invite_link is None
    assert client.exported == 0
    assert client.created_titles == ["Trio Debate"]  # created exactly once
    # All three promoted with manage_topics.
    assert len(client.promoted) == 3
    assert result.audit.is_forum is True


async def test_provision_reuses_when_present_idempotent() -> None:
    existing = _channel(424242, title="Trio Debate", megagroup=True, forum=True)
    client = _FakeClient(
        dialogs=[_FakeDialog("Trio Debate", existing)],  # discovered
        entity=existing,
        admins=[existing],
    )
    result = await provision_debate_group(
        client,  # type: ignore[arg-type]
        "Trio Debate",
        ["skippy", "nagatha", "bilby"],
    )
    assert result.created is False  # REUSED, not recreated
    assert result.chat_id == "-1000000424242"
    assert client.created_titles == []  # never called CreateChannelRequest
    # Already a forum -> no toggle sent.
    assert client.toggled == []
    assert result.forum_enabled_now is False


async def test_provision_privacy_bot_stays_unpromoted_without_link() -> None:
    existing = _channel(5, title="Trio Debate", megagroup=True, forum=True)
    client = _FakeClient(
        dialogs=[_FakeDialog("Trio Debate", existing)],
        entity=existing,
        admins=[existing],
        privacy_fail={DEFAULT_BOT_HANDLES["nagatha"]},
    )
    result = await provision_debate_group(
        client,  # type: ignore[arg-type]
        "Trio Debate",
        ["skippy", "nagatha", "bilby"],
    )
    assert DEFAULT_BOT_HANDLES["nagatha"] in result.invite_privacy_blocked
    assert result.invite_link is None
    assert DEFAULT_BOT_HANDLES["nagatha"] not in result.promotions
    assert client.exported == 0


async def test_provision_mints_a_bounded_link_only_when_explicitly_requested() -> None:
    existing = _channel(5, title="Trio Debate", megagroup=True, forum=True)
    client = _FakeClient(
        dialogs=[_FakeDialog("Trio Debate", existing)], entity=existing, admins=[existing]
    )
    result = await provision_debate_group(
        client,  # type: ignore[arg-type]
        "Trio Debate",
        ["skippy"],
        mint_invite_link=True,
    )
    assert result.invite_link == "https://t.me/+FAKEINVITE"
    assert client.exported == 1


async def test_provision_revoked_session_fails_loud() -> None:
    existing = _channel(5, title="Trio Debate", megagroup=True, forum=True)
    client = _FakeClient(
        dialogs=[_FakeDialog("Trio Debate", existing)],
        entity=existing,
        raise_on={"InviteToChannelRequest": AuthKeyUnregisteredError(request=None)},
    )
    with pytest.raises(ProvisioningError) as exc:
        await provision_debate_group(
            client,  # type: ignore[arg-type]
            "Trio Debate",
            ["skippy"],
        )
    assert exc.value.kind == "auth_revoked"
