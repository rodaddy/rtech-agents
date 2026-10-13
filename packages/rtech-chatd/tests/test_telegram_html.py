"""Markdown -> Telegram-HTML conversion + 4096-safe chunking tests.

Two surfaces: :func:`render_telegram_html` (per-chunk conversion, allowlist-only
tags, code extracted-then-restored so markdown inside code stays literal, and a
scheme-restricted link filter) and :func:`telegram_chunks` (chunk-then-convert,
every emitted payload within Telegram's 4096 ceiling, ``(n/m)`` indicators, and
per-chunk balanced fences even for a code block that spans chunks).
"""

from __future__ import annotations

from rtech_chatd.telegram_html import (
    TELEGRAM_MAX_LENGTH,
    render_telegram_html,
    telegram_chunks,
)

# --------------------------------------------------------------- render: prose


def test_html_escapes_ampersand_lt_gt_in_prose() -> None:
    out = render_telegram_html("a & b < c > d")
    assert "&amp;" in out
    assert "&lt;" in out
    assert "&gt;" in out
    assert "<c" not in out  # the literal < was escaped, not left as a tag start


def test_bold_becomes_b_tag() -> None:
    assert render_telegram_html("say **hello** now") == "say <b>hello</b> now"


def test_star_italic_becomes_i_tag() -> None:
    assert render_telegram_html("say *hello* now") == "say <i>hello</i> now"


def test_underscore_italic_becomes_i_but_snake_case_untouched() -> None:
    assert render_telegram_html("an _emphasised_ word") == "an <i>emphasised</i> word"
    # snake_case identifier in prose must NOT italicize (word-boundary anchored).
    out = render_telegram_html("call snake_case_identifier here")
    assert "<i>" not in out
    assert "snake_case_identifier" in out


# --------------------------------------------------------------- render: code


def test_inline_code_escapes_and_does_not_style_markdown_inside() -> None:
    out = render_telegram_html("use `a < b && **bold**` here")
    assert "<code>" in out and "</code>" in out
    assert "&lt;" in out and "&amp;" in out  # content escaped
    assert "<b>" not in out  # markdown inside inline code is NOT styled


def test_fence_with_language_emits_pre_code_class() -> None:
    out = render_telegram_html("```py\nx = 1 < 2 & 3\n```")
    assert '<pre><code class="language-py">' in out
    assert "</code></pre>" in out
    assert "&lt;" in out and "&amp;" in out  # body escaped


def test_fence_without_language_emits_bare_pre() -> None:
    out = render_telegram_html("```\nplain code < & >\n```")
    assert "<pre>" in out and "</pre>" in out
    assert "language-" not in out
    assert "&lt;" in out


def test_markdown_inside_fence_stays_literal() -> None:
    out = render_telegram_html("```\n**not bold** *not italic* [x](https://y)\n```")
    assert "<b>" not in out
    assert "<i>" not in out
    assert "<a href" not in out
    # The markdown metachars survive as literal text inside the pre block.
    assert "**not bold**" in out


# --------------------------------------------------------------- render: links


def test_http_link_becomes_anchor() -> None:
    out = render_telegram_html("see [docs](https://example.com/x?a=1&b=2)")
    assert '<a href="https://example.com/x?a=1&amp;b=2">docs</a>' in out


def test_javascript_scheme_link_not_converted_to_anchor() -> None:
    out = render_telegram_html("[click](javascript:alert(1))")
    assert "<a href" not in out  # scheme-restricted: never an anchor
    assert "javascript" in out  # left as escaped literal text


# -------------------------------------------------------------- render: blocks


def test_realistic_markdown_renders_clean_telegram_html() -> None:
    source = """## Release checklist

Ship the `telegram_chunks` change before review.

- Prepare release
  - Run focused tests
1. Open the PR
2) Request review

> Keep **all** output readable.
> See [the runbook](https://example.com/runbook?a=1&b=2).

```python
# not a heading
- not a bullet
print(\"*literal* <unsafe>\")
```

<script>alert(\"no\")</script>"""

    assert (
        render_telegram_html(source)
        == """<b>Release checklist</b>

Ship the <code>telegram_chunks</code> change before review.

• Prepare release
  • Run focused tests
1. Open the PR
2. Request review

<blockquote>Keep <b>all</b> output readable.
See <a href=\"https://example.com/runbook?a=1&amp;b=2\">the runbook</a>.</blockquote>

<pre><code class=\"language-python\"># not a heading
- not a bullet
print(&quot;*literal* &lt;unsafe&gt;&quot;)
</code></pre>

&lt;script&gt;alert(&quot;no&quot;)&lt;/script&gt;"""
    )


def test_block_markers_inside_fence_stay_literal_and_malformed_quote_degrades() -> None:
    fenced = "```text\n# heading\n- bullet\n1. step\n> quote\n```"
    out = render_telegram_html(fenced)
    assert out == (
        '<pre><code class="language-text"># heading\n- bullet\n1. step\n&gt; quote\n</code></pre>'
    )

    # The blockquote may contain ordinary inline markup, but crossing tags
    # still fail the shared validator and are sent as safe plain text.
    chunks = telegram_chunks("> **a *b** c*\n> still quoted")
    assert len(chunks) == 1
    assert "<blockquote>" not in chunks[0]
    assert "<b>" not in chunks[0]
    assert "<i>" not in chunks[0]


# --------------------------------------------------------------- chunks


def test_short_text_single_chunk_within_ceiling() -> None:
    chunks = telegram_chunks("just a short message")
    assert len(chunks) == 1
    assert len(chunks[0]) <= TELEGRAM_MAX_LENGTH
    assert chunks[0] == "just a short message"


def test_long_text_all_chunks_within_ceiling_with_indicators() -> None:
    text = "\n".join(f"line {i} with some filler content here" for i in range(400))
    assert len(text) > 10000
    chunks = telegram_chunks(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= TELEGRAM_MAX_LENGTH
        assert chunk.strip() != ""  # no empty chunk
    # (n/m) continuation indicators present on the multi-chunk output.
    joined = "\n".join(chunks)
    assert f"(1/{len(chunks)})" in joined
    assert f"({len(chunks)}/{len(chunks)})" in joined


def test_long_fenced_block_keeps_balanced_pre_per_chunk() -> None:
    # A single fenced code block far larger than one chunk: the markdown chunker
    # closes/reopens the fence per chunk, so each converted chunk has matching
    # <pre> and </pre> counts (balanced tags by construction).
    body = "\n".join(f"code_line_{i} = value_{i}" for i in range(400))
    text = f"```python\n{body}\n```"
    chunks = telegram_chunks(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= TELEGRAM_MAX_LENGTH
        # Every opening <pre (bare or as <pre><code class=...>) begins with the
        # literal "<pre>", so counting that is the open count; each has a </pre>.
        opens = chunk.count("<pre>")
        closes = chunk.count("</pre>")
        assert opens == closes
        assert opens >= 1  # each chunk of a fenced block carries a fence


def test_pathological_escape_density_all_chunks_within_ceiling() -> None:
    # 8k of '&' — each escapes to '&amp;' (5x), the worst expansion case. The
    # post-conversion oversize guard must still land every chunk <= 4096.
    text = "&" * 8000
    chunks = telegram_chunks(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= TELEGRAM_MAX_LENGTH
        assert chunk != ""


# ------------------------------------------- gauntlet fix regressions


def test_nul_bytes_are_stripped_and_cannot_forge_stash_sentinels() -> None:
    # A crafted NUL sentinel in model output must not collide with the code
    # stash placeholders (which would restore an arbitrary stash entry).
    out = render_telegram_html("`real code` and a forged \x000\x00 sentinel")
    assert "\x00" not in out
    assert out.count("<code>") == 1  # only the real span rendered


def test_emphasis_cannot_wrap_a_stashed_code_block() -> None:
    # Bold spanning a fence would restore <pre> INSIDE <b> — a block-in-inline
    # nesting the Bot API rejects. The emphasis regexes exclude the stash
    # sentinel, so the stars render literally and the fence stays block-level.
    out = render_telegram_html("**before\n```\ncode\n```\nafter**")
    assert "<b>" not in out
    assert "<pre>" in out


def test_crossing_inline_tags_degrade_chunk_to_plain() -> None:
    # Mis-nested emphasis converts to crossing tags (<b>a <i>b</b> c</i>) — a
    # guaranteed Bot API parse-400. The well-formedness gate degrades the
    # chunk to escaped plain text pre-emptively: one unstyled send, not a
    # rejected send plus a literal-tag resend.
    chunks = telegram_chunks("**a *b** c*")
    assert len(chunks) == 1
    assert "<b>" not in chunks[0]
    assert "<i>" not in chunks[0]
    assert "a" in chunks[0]  # content still lands


def test_resplit_of_crossing_bold_body_stays_parser_safe() -> None:
    # F6 adjudication (follow-up): the re-split path emits no mis-nested
    # piece even for a huge, escape-dense, emphasis-heavy body. Taking a
    # contiguous sub-piece of a well-formed chunk can only UNMATCH an emphasis
    # span (a lone ``**``/``*``/``_`` renders as a literal metachar, never a
    # tag), never create a new crossing interleave, and chunk_message is
    # fence-aware — so a converted re-split piece can never fail the Bot API
    # parser. Every emitted piece is therefore within the ceiling and tag-safe.
    seg = "alpha **bravo _charlie_ delta** echo & < > "
    text = seg * 400  # far over one chunk; escaping (& < >) inflates each piece
    chunks = telegram_chunks(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= TELEGRAM_MAX_LENGTH
        assert chunk.count("<b>") == chunk.count("</b>")
        assert chunk.count("<i>") == chunk.count("</i>")


def test_emphasis_wrapping_inline_code_still_styles() -> None:
    # Fix-delta V2 regression: the first sentinel fix excluded BOTH stash
    # sentinels from emphasis, so ``**`func`**`` shipped literal asterisks.
    # Inline code inside emphasis is Bot-API-legal and must keep styling.
    assert (
        render_telegram_html("**`get_service`** does X") == "<b><code>get_service</code></b> does X"
    )
    assert render_telegram_html("*prefer `uv run`*") == "<i>prefer <code>uv run</code></i>"
    out = render_telegram_html("[the `send` docs](https://example.com/api)")
    assert out == '<a href="https://example.com/api">the <code>send</code> docs</a>'


def test_forged_inline_sentinel_is_stripped_too() -> None:
    # Both sentinel control chars are stripped from input, so neither can be
    # forged to restore an arbitrary stash entry.
    out = render_telegram_html("`real` forged \x010\x01 and \x000\x00 markers")
    assert "\x00" not in out
    assert "\x01" not in out
    assert out.count("<code>") == 1


def test_plain_pieces_never_split_an_html_entity() -> None:
    # the plain-degrade hard cut must not sever an
    # escaped entity (a piece ending "&am" 400s the HTML attempt and renders
    # literally on the plain resend). Dense '&' input puts an entity on every
    # boundary candidate.
    from rtech_chatd.telegram_html import _plain_pieces

    pieces = _plain_pieces("&" * 5000)
    assert len(pieces) > 1
    assert "".join(pieces) == "&amp;" * 5000  # nothing lost or reordered
    for piece in pieces:
        assert len(piece) <= TELEGRAM_MAX_LENGTH
        assert piece.count("&") == piece.count(";")  # every entity intact
