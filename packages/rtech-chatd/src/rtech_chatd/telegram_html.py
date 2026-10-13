"""Markdown -> Telegram-HTML rendering + 4096-safe chunking (issue #109).

Telegram's Bot API offers two formatting modes; this edge uses ``HTML``, NOT
MarkdownV2 — MarkdownV2 requires escaping ~18 metacharacters in all prose and
400s on any miss (the one prior bot that formatted used legacy Markdown and
died on special chars). HTML inverts the failure mode: prose is escaped once,
and only a small ALLOWLIST of tags is ever emitted (``b``/``i``/``code``/
``pre``/``a``/``blockquote``), so arbitrary model output can never smuggle
markup.

The pipeline is chunk-then-convert, deliberately in that order:

1. :func:`rtech_chatd.chunking.chunk_message` splits the MARKDOWN at a
   conservative budget (it already refuses to split inside a code fence or an
   inline code span, and carries an open fence's language across the split);
2. each chunk is converted to HTML INDEPENDENTLY, so every emitted message has
   balanced tags by construction — no HTML-aware splitter exists or is needed.

Escaping expands text (``&`` -> ``&amp;``), so the markdown budget sits well
under Telegram's 4096 ceiling; a chunk whose converted form still exceeds the
ceiling (pathological escape density) is re-split at half budget, bottoming
out at a hard plain-text cut — content lands, loudly, never a parse 400.
"""

from __future__ import annotations

import html
import re

from loguru import logger

from .chunking import chunk_message

# Telegram's hard per-message ceiling (visible characters; we conservatively
# budget the WHOLE HTML payload under it, tags included).
TELEGRAM_MAX_LENGTH = 4096

# Markdown budget per chunk. Escaping + tag overhead expand a chunk, so the
# markdown is split at 3/4 of the ceiling; the post-conversion guard re-splits
# the rare chunk that still lands over.
TELEGRAM_CHUNK_BUDGET = 3072

# Smallest budget the oversize re-split will try before hard-cutting. The
# chunker itself needs headroom for indicators/fence reopens; below this a
# pathological chunk is escaped plain and hard-cut instead.
_MIN_RESPLIT_BUDGET = 256

# Fenced code block: ```lang\n ... ``` (language tag optional). DOTALL so the
# body spans lines; non-greedy so adjacent fences never merge.
_FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)\n?(.*?)```", re.DOTALL)
# Inline code span (single backticks, no newline inside).
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
# Bold (**...**), matched before single-star italic consumes the stars. The
# character classes EXCLUDE the BLOCK sentinel (\x00, a stashed fence) so an
# emphasis span can never straddle one — that would restore a <pre> INSIDE
# <b>/<i>, a block-in-inline nesting the Bot API rejects (PR #122 gauntlet).
# The INLINE sentinel (\x01, a stashed `code` span) is deliberately allowed:
# ``**`func`**`` -> ``<b><code>func</code></b>`` is Bot-API-legal and the
# single most common formatting shape in technical chat (PR #122 fix-delta V2
# caught the first fix breaking it by excluding BOTH sentinels).
_BOLD_RE = re.compile(r"\*\*([^\x00]+?)\*\*", re.DOTALL)
# Italic: *...* not adjacent to another star, or _..._ on word boundaries so
# snake_case identifiers in prose never italicize.
_STAR_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n\x00]+)\*(?!\*)")
_UNDERSCORE_ITALIC_RE = re.compile(r"(?<![\w_])_([^_\n\x00]+)_(?![\w_])")
# Markdown link [text](http(s)://url) — scheme-restricted so a crafted
# ``[x](javascript:...)`` never becomes an anchor; the label excludes the
# BLOCK sentinel so an anchor can never wrap a stashed fence.
_LINK_RE = re.compile(r"\[([^\]\x00]+)\]\((https?://[^\s)]+)\)")

# Block markdown is deliberately rendered after code has been stashed and the
# remaining prose escaped. That means neither raw HTML nor fence contents can
# participate in block recognition.
_ATX_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$")
_UNORDERED_LIST_RE = re.compile(r"^([ \t]*)[-*+][ \t]+(.*)$")
_ORDERED_LIST_RE = re.compile(r"^([ \t]*)(\d+)[.)][ \t]+(.*)$")
_QUOTE_RE = re.compile(r"^[ \t]*&gt;[ \t]?(.*)$")

# Placeholder sentinels for extracted code segments: \x00 wraps a stashed
# FENCE (block-level <pre>, must never end up inside an inline entity), \x01
# wraps a stashed INLINE span (<code>, legal inside emphasis). Neither control
# char occurs in chat text (Telegram rejects them, and the renderer strips
# both from input up front), so the sentinels cannot collide with content.
_BLOCK_SENTINEL = "\x00{}\x00"
_INLINE_SENTINEL = "\x01{}\x01"
_SENTINEL_RE = re.compile("[\x00\x01](\\d+)[\x00\x01]")

# Allowlist tag tokenizer for the well-formedness check. Only these tags can
# appear in rendered output; anything else in the text is already escaped.
_TAG_RE = re.compile(r"<(/?)(b|i|code|pre|a|blockquote)(?:\s[^>]*)?>")
# Tags whose CONTENT must be plain (no nested entities), per Bot API rules.
_OPAQUE_TAGS = frozenset({"code", "pre"})
# Block-level tag: must never sit inside an open inline entity.
_BLOCK_TAGS = frozenset({"pre", "blockquote"})


def _well_formed(chunk: str) -> bool:
    """Return whether ``chunk``'s tags nest the way the Bot API accepts.

    Sequential regex substitution can emit CROSSING tags from mis-nested
    markdown (``**a *b** c*`` -> ``<b>a <i>b</b> c</i>``) — the Bot API
    rejects those with a parse-400, costing a doubled send and all styling
    (PR #122 gauntlet). This is a cheap LIFO check: proper open/close pairing,
    nothing nested inside ``code``/``pre``, and no ``pre`` inside an open
    inline entity. A chunk that fails is re-rendered PLAIN by the caller.
    """
    stack: list[str] = []
    for match in _TAG_RE.finditer(chunk):
        closing, tag = match.group(1), match.group(2)
        if not closing:
            if stack and stack[-1] in _OPAQUE_TAGS:
                # The ONE legal nesting inside an opaque tag is the renderer's
                # own language form: <pre><code class="language-…">.
                if not (tag == "code" and stack[-1] == "pre"):
                    return False
            if tag in _BLOCK_TAGS and stack:
                return False
            stack.append(tag)
        else:
            if not stack or stack[-1] != tag:
                return False
            stack.pop()
    return not stack


def _render_blocks(escaped_text: str) -> str:
    """Render a conservative Markdown block subset in already-escaped prose.

    This intentionally recognizes only line-oriented forms. It leaves normal
    prose and blank lines untouched, keeps list indentation as supplied, and
    groups adjacent quote lines into one Telegram ``blockquote``. Fenced code
    is a block sentinel by this point, so markdown-looking code stays literal.
    """
    lines = escaped_text.split("\n")
    rendered: list[str] = []
    index = 0

    while index < len(lines):
        quote = _QUOTE_RE.match(lines[index])
        if quote is not None:
            quote_lines: list[str] = []
            while index < len(lines):
                quote = _QUOTE_RE.match(lines[index])
                if quote is None:
                    break
                quote_lines.append(quote.group(1))
                index += 1
            rendered.append(f"<blockquote>{'\n'.join(quote_lines)}</blockquote>")
            continue

        line = lines[index]
        heading = _ATX_HEADING_RE.match(line)
        if heading is not None:
            rendered.append(f"<b>{heading.group(1)}</b>")
        else:
            unordered = _UNORDERED_LIST_RE.match(line)
            if unordered is not None:
                rendered.append(f"{unordered.group(1)}• {unordered.group(2)}")
            else:
                ordered = _ORDERED_LIST_RE.match(line)
                if ordered is not None:
                    rendered.append(f"{ordered.group(1)}{ordered.group(2)}. {ordered.group(3)}")
                else:
                    rendered.append(line)
        index += 1

    return "\n".join(rendered)


def render_telegram_html(text: str) -> str:
    """Convert one markdown chunk to allowlist-only Telegram HTML.

    Code segments (fences, then inline spans) are extracted FIRST and restored
    LAST, so markdown inside code is never styled and code content is escaped
    exactly once. Everything outside code is HTML-escaped, then the inline
    styles (links, bold, italic) are applied to the escaped text — the only
    tags that can appear in the output are the allowlist.

    Sentinel control chars are stripped up front: Telegram rejects them in
    message text anyway, and a crafted ``\\x00N\\x00``/``\\x01N\\x01`` in model
    output would otherwise collide with the code-stash sentinels (PR #122
    gauntlet).
    """
    text = text.replace("\x00", "").replace("\x01", "")
    stash: list[str] = []

    def _stash(rendered: str, sentinel: str) -> str:
        stash.append(rendered)
        return sentinel.format(len(stash) - 1)

    def _fence(match: re.Match[str]) -> str:
        lang = match.group(1)
        body = html.escape(match.group(2))
        if lang:
            rendered = f'<pre><code class="language-{lang}">{body}</code></pre>'
            return _stash(rendered, _BLOCK_SENTINEL)
        return _stash(f"<pre>{body}</pre>", _BLOCK_SENTINEL)

    def _inline(match: re.Match[str]) -> str:
        return _stash(f"<code>{html.escape(match.group(1))}</code>", _INLINE_SENTINEL)

    working = _FENCE_RE.sub(_fence, text)
    working = _INLINE_CODE_RE.sub(_inline, working)
    working = html.escape(working)
    working = _render_blocks(working)

    def _link(match: re.Match[str]) -> str:
        # The surrounding text is already escaped, so the label is safe as-is;
        # the URL re-escapes its quotes for the attribute position. The
        # original ``&`` in the URL is already ``&amp;`` here, which is the
        # correct HTML attribute encoding.
        url = match.group(2).replace('"', "&quot;")
        return f'<a href="{url}">{match.group(1)}</a>'

    working = _LINK_RE.sub(_link, working)
    working = _BOLD_RE.sub(r"<b>\1</b>", working)
    working = _STAR_ITALIC_RE.sub(r"<i>\1</i>", working)
    working = _UNDERSCORE_ITALIC_RE.sub(r"<i>\1</i>", working)

    return _SENTINEL_RE.sub(lambda m: stash[int(m.group(1))], working)


def telegram_chunks(text: str, *, budget: int = TELEGRAM_CHUNK_BUDGET) -> list[str]:
    """Split markdown ``text`` into ready-to-send Telegram HTML payloads.

    Chunk-then-convert (module docstring): the markdown is split at ``budget``
    (fence-aware, ``(n/m)`` indicators appended when it spans chunks), each
    chunk converts independently, and any converted chunk still over the 4096
    ceiling is re-split at half budget. The floor case escapes plain and
    hard-cuts — loud, never a parse error, never dropped.
    """
    rendered: list[str] = []
    for chunk in chunk_message(text, max_length=budget):
        converted = render_telegram_html(chunk)
        if not _well_formed(converted):
            # Mis-nested markdown emphasis produced crossing tags — a
            # guaranteed Bot API parse-400. Degrade THIS chunk to plain
            # escaped text pre-emptively: content lands unstyled in ONE send
            # instead of a rejected send plus a literal-tag resend.
            logger.warning(
                "chatd.tg.html | mis-nested inline markup; chunk degraded to plain | len={}",
                len(chunk),
            )
            rendered.extend(_plain_pieces(chunk))
            continue
        if len(converted) <= TELEGRAM_MAX_LENGTH:
            rendered.append(converted)
            continue
        rendered.extend(_resplit_oversize(chunk, budget // 2))
    return rendered


def _plain_pieces(chunk: str) -> list[str]:
    """Escape ``chunk`` with no styling and hard-cut it under the ceiling.

    Cuts never split an HTML entity (``&amp;`` etc., at most 6 chars): a piece
    ending mid-entity would trip the Bot API parser on the HTML attempt AND
    render the severed fragment literally on the plain resend (PR #122
    terminal audit). A cut landing inside an entity backs up to just before
    its ``&``.
    """
    escaped = html.escape(chunk)
    pieces: list[str] = []
    start = 0
    while start < len(escaped):
        end = min(start + TELEGRAM_MAX_LENGTH, len(escaped))
        if end < len(escaped):
            amp = escaped.rfind("&", max(start, end - 6), end)
            if amp > start and ";" not in escaped[amp:end]:
                end = amp
        pieces.append(escaped[start:end])
        start = end
    return pieces


def _resplit_oversize(chunk: str, budget: int) -> list[str]:
    """Re-split one over-ceiling chunk at a smaller budget, recursing to a floor.

    At the floor the chunk is escaped PLAIN (no styling — markdown metachars
    render literally) and hard-cut to the ceiling per piece: a pathological
    input degrades to unstyled text rather than a Bot API parse 400 or a
    dropped message, and the degrade is logged loudly.
    """
    if budget >= _MIN_RESPLIT_BUDGET:
        logger.warning(
            "chatd.tg.html | converted chunk over ceiling; re-splitting | budget={}", budget
        )
        out: list[str] = []
        for piece in chunk_message(chunk, max_length=budget):
            converted = render_telegram_html(piece)
            if len(converted) <= TELEGRAM_MAX_LENGTH:
                out.append(converted)
            else:
                out.extend(_resplit_oversize(piece, budget // 2))
        return out
    logger.error(
        "chatd.tg.html | pathological chunk; degrading to plain escaped hard-cut | len={}",
        len(chunk),
    )
    return _plain_pieces(chunk)


__all__ = [
    "TELEGRAM_CHUNK_BUDGET",
    "TELEGRAM_MAX_LENGTH",
    "render_telegram_html",
    "telegram_chunks",
]
