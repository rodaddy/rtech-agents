"""Focused grounding evidence normalization proofs."""

from __future__ import annotations

from rtech_agentd.grounding import (
    GroundingState,
    classify_turn_eligibility,
    is_verification_method,
    merge_verification,
    needs_guarded_fallback,
    recalled_evidence,
)
from rtech_agentd.memory import MemoryRecallResult, MemoryRecallSource
from rtech_frames import (
    MAX_SOURCE_ID_LEN,
    MAX_SOURCE_LABEL_LEN,
    MAX_SOURCE_LOCATOR_LEN,
    EvidenceSource,
    VerificationReport,
)


def test_recalled_evidence_clamps_long_source_fields() -> None:
    source = "s" * 201
    kind = "k" * 256
    locator = "l" * 256
    result = MemoryRecallResult(
        block="- recalled fact",
        sources=(
            MemoryRecallSource(
                label="i" * 256,
                source=source,
                kind=kind,
                metadata=(("path", locator),),
            ),
        ),
    )

    evidence = recalled_evidence(result)

    assert len(evidence) == 1
    assert evidence[0].source_id == "i" * MAX_SOURCE_ID_LEN
    assert evidence[0].label == source[:MAX_SOURCE_LABEL_LEN]
    assert evidence[0].locator == locator[:MAX_SOURCE_LOCATOR_LEN]


def _verification(
    *,
    claim_kind: str,
    status: str,
    sources: list[EvidenceSource] | None = None,
) -> VerificationReport:
    """Build a VerificationReport mirroring the frame-turn test helpers."""
    return VerificationReport.model_validate(
        {
            "claim_kind": claim_kind,
            "status": status,
            "ob_checked": False,
            "sources": sources or [],
        }
    )


# --- Finding 1: mislabeled-nonfactual bypass is gated -----------------------


def test_needs_guarded_fallback_gates_factual_answer_wearing_nonfactual_tag() -> None:
    # opinion tag but a FACTUAL verdict (unverified) -> gated.
    opinion = _verification(claim_kind="opinion", status="unverified")
    assert needs_guarded_fallback(opinion, cold=True) is True
    assert needs_guarded_fallback(opinion, cold=False) is True

    # social tag, partial verdict, no sources, cold -> gated (factual path).
    social = _verification(claim_kind="social", status="partial")
    assert needs_guarded_fallback(social, cold=True) is True

    # action tag, grounded verdict WITH a source, warm -> factual path, sources
    # present, not cold -> NOT gated.
    with_source = _verification(
        claim_kind="action",
        status="grounded",
        sources=[
            EvidenceSource(
                source_id="repo-ok",
                kind="repo",
                label="frames",
                locator="packages/frames.py",
            )
        ],
    )
    assert needs_guarded_fallback(with_source, cold=False) is False
    # Same grounded verdict but cold and NO sources -> gated.
    no_source = _verification(claim_kind="action", status="grounded")
    assert needs_guarded_fallback(no_source, cold=True) is True


def test_needs_guarded_fallback_honors_genuine_nonfactual_bypass() -> None:
    # A runtime-classified social prompt whose reply is self-consistently
    # nonfactual + not_applicable bypasses; likewise an opinion prompt.
    social = _verification(claim_kind="social", status="not_applicable")
    assert needs_guarded_fallback(social, cold=True, eligibility="social") is False
    assert needs_guarded_fallback(social, cold=False, eligibility="social") is False

    opinion = _verification(claim_kind="opinion", status="not_applicable")
    assert needs_guarded_fallback(opinion, cold=True, eligibility="opinion") is False
    assert needs_guarded_fallback(opinion, cold=False, eligibility="opinion") is False

    # An ``action`` claim_kind is a consistent nonfactual self-report too, so a
    # social/opinion-eligible prompt reporting it still bypasses.
    action = _verification(claim_kind="action", status="not_applicable")
    assert needs_guarded_fallback(action, cold=True, eligibility="social") is False


def test_runtime_eligibility_owns_nonfactual_bypass() -> None:
    # Gap A: the model self-tags a factual prompt as opinion + not_applicable to
    # skip the guard, but the runtime classified the prompt factual (default), so
    # the bypass is DENIED — the model cannot self-exempt.
    mislabel = _verification(claim_kind="opinion", status="not_applicable")
    assert needs_guarded_fallback(mislabel, cold=True) is True
    assert needs_guarded_fallback(mislabel, cold=False, eligibility="factual") is True


def test_factual_or_mixed_opinion_wording_cannot_bypass_evidence() -> None:
    report = _verification(claim_kind="opinion", status="not_applicable")
    for prompt in (
        "What do you think 2 + 2 equals?",
        "What do you think is in frame_turn.py?",
    ):
        assert classify_turn_eligibility(prompt) == "factual"
        assert needs_guarded_fallback(
            report, cold=True, eligibility=classify_turn_eligibility(prompt)
        )


def test_classify_turn_eligibility_social_opinion_and_factual() -> None:
    for social in ("hi there!", "thanks", "good morning"):
        assert classify_turn_eligibility(social) == "social"
    for opinion in ("what do you think?",):
        assert classify_turn_eligibility(opinion) == "opinion"
    for factual in (
        "what is the current version?",
        "list the open PRs",
        "",
        "thanks, now what is the status?",
        "What do you think 2 + 2 equals?",
        "What do you think is in frame_turn.py?",
        "what is your opinion on frame_turn.py?",
    ):
        assert classify_turn_eligibility(factual) == "factual"


# --- Finding 2: stance/wake tools never count as verification ----------------


def test_is_verification_method_allows_only_known_real_tools() -> None:
    for name in (
        "mcp__rtech_stance__stance",
        "mcp__rtech_wake__wake_agent",
        "arbitrary_mcp_tool",
        "mcpToolCall",
        "unknown_mcp_tool",
    ):
        assert is_verification_method(name) is False
    for name in ("repo_search", "read_file", "web_fetch", "web_search"):
        assert is_verification_method(name) is True


def test_observe_tool_drops_control_tools_and_records_real_verification() -> None:
    for name in (
        "mcp__rtech_stance__stance",
        "stance",
        "mcp__rtech_wake__wake_agent",
        "wake_agent",
    ):
        state = GroundingState(cold=True)
        state.observe_tool(name)
        assert state.observed_methods == []

    state = GroundingState(cold=True)
    state.observe_tool("repo_search")
    assert state.observed_methods == ["repo_search"]

    unrelated = GroundingState(cold=True)
    unrelated.observe_tool("unrelated_tool", {"path": "packages/x.py"})
    assert unrelated.observed_methods == []
    assert unrelated.observed_locators == set()


def test_merge_verification_drops_repo_source_when_only_stance_observed() -> None:
    repo_source = EvidenceSource(
        source_id="repo-x",
        kind="repo",
        label="x",
        locator="packages/x.py",
    )
    report = _verification(claim_kind="factual", status="partial", sources=[repo_source])

    stance_only = GroundingState(cold=True)
    stance_only.observe_tool("mcp__rtech_stance__stance")
    merged = merge_verification(report, stance_only)
    assert merged.sources == []

    accepting = GroundingState(cold=True)
    accepting.observe_tool("mcp__rtech_stance__stance")
    accepting.observe_tool("repo_search", {"path": "packages/x.py"})
    merged_ok = merge_verification(report, accepting)
    assert [source.source_id for source in merged_ok.sources] == ["repo-x"]


# --- Gap B: bounded runtime registry of safe observed locators ---------------


def test_observe_tool_registers_whitelisted_locator_and_drops_raw_input() -> None:
    state = GroundingState(cold=True)
    state.observe_tool(
        "read_file",
        {"arguments": {"path": "packages/frames.py", "secret": "nope"}},
    )
    assert ("repo", "packages/frames.py") in state.observed_locators
    assert "nope" not in str(state.observed_locators)
    assert "nope" not in str(state.__dict__)


def test_observe_tool_registers_url_locator() -> None:
    state = GroundingState(cold=True)
    state.observe_tool("web_fetch", {"url": "https://example.test/docs"})
    assert ("url", "https://example.test/docs") in state.observed_locators


def test_url_locator_registration_and_publication_strip_private_components() -> None:
    state = GroundingState(cold=True)
    raw_locator = "https://rico:password@example.test/docs?token=private#fragment"
    state.observe_tool(
        "web_fetch",
        {"arguments": {"url": raw_locator, "authorization": "private"}},
    )

    safe_locator = "https://example.test/docs"
    assert state.observed_locators == {("url", safe_locator)}
    assert "password" not in str(state.__dict__)
    assert "private" not in str(state.__dict__)

    report = _verification(
        claim_kind="factual",
        status="partial",
        sources=[EvidenceSource(source_id="url", kind="url", label="docs", locator=raw_locator)],
    )
    merged = merge_verification(report, state)
    assert merged.sources[0].locator == safe_locator
    assert "password" not in str(merged)
    assert "private" not in str(merged)


def test_observe_tool_drops_unsafe_locators() -> None:
    repo_state = GroundingState(cold=True)
    repo_state.observe_tool("repo_search", {"path": "../secret"})
    assert repo_state.observed_locators == set()

    url_state = GroundingState(cold=True)
    url_state.observe_tool("web_fetch", {"url": "http://x"})
    assert url_state.observed_locators == set()


def test_merge_verification_requires_reported_source_locator_in_registry() -> None:
    observed = EvidenceSource(
        source_id="repo-ok", kind="repo", label="frames", locator="packages/frames.py"
    )
    unobserved = EvidenceSource(
        source_id="repo-other", kind="repo", label="other", locator="packages/other.py"
    )
    live = EvidenceSource(source_id="live", kind="live", label="live", locator="tail -f")
    report = _verification(
        claim_kind="factual", status="partial", sources=[observed, unobserved, live]
    )

    state = GroundingState(cold=True)
    state.observe_tool("repo_search", {"path": "packages/frames.py"})
    merged = merge_verification(report, state)
    assert [source.source_id for source in merged.sources] == ["repo-ok"]


def test_unrelated_tool_cannot_authorize_a_reported_source() -> None:
    report = _verification(
        claim_kind="factual",
        status="partial",
        sources=[
            EvidenceSource(
                source_id="repo", kind="repo", label="frames", locator="packages/frames.py"
            )
        ],
    )
    state = GroundingState(cold=True)
    state.observe_tool("unrelated_tool", {"path": "packages/frames.py"})

    merged = merge_verification(report, state)
    assert merged.sources == []
    assert merged.methods == []
