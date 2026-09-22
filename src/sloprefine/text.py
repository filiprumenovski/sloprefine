"""Tokenization with byte offsets, so every hit can point at the source."""

from __future__ import annotations

import re
from dataclasses import dataclass

WORD = re.compile(r"[A-Za-z][A-Za-z'\u2019-]*")

_SENT_END = re.compile(r"[.!?]+(?=[\"')\]]*(?:\s|$))")
_ABBREV = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|vs|etc|al|Fig|Eq|approx|cf|e\.g|i\.e)\.$",
    re.IGNORECASE,
)
_INITIAL = re.compile(r"\b[A-Z]\.$")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str

    @property
    def words(self) -> list[str]:
        return WORD.findall(self.text)

    def __len__(self) -> int:
        return len(self.words)


def _strip_span(text: str, start: int, end: int) -> Span | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end <= start:
        return None
    return Span(start, end, text[start:end])


def sentence_spans(text: str) -> list[Span]:
    """Split on terminal punctuation and blank lines, keeping offsets.

    Abbreviations, decimals and single-letter initials do not end a sentence.
    """
    boundaries: list[int] = []
    for m in _SENT_END.finditer(text):
        # Only the tail matters: the abbreviation and initial patterns are
        # anchored at the end. Slicing the whole prefix made this O(n^2),
        # which cost 11 of the 12 seconds spent analysing a 20k-word file.
        head = text[max(0, m.end() - 24):m.end()]
        if _ABBREV.search(head) or _INITIAL.search(head):
            continue
        boundaries.append(m.end())
    for m in re.finditer(r"\n\s*\n", text):
        boundaries.append(m.start())
    boundaries.append(len(text))

    spans: list[Span] = []
    cursor = 0
    for b in sorted(set(boundaries)):
        if b <= cursor:
            continue
        span = _strip_span(text, cursor, b)
        if span is not None:
            spans.append(span)
        cursor = b
    return spans


def paragraph_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    cursor = 0
    for m in re.finditer(r"\n\s*\n", text):
        span = _strip_span(text, cursor, m.start())
        if span is not None:
            spans.append(span)
        cursor = m.end()
    span = _strip_span(text, cursor, len(text))
    if span is not None:
        spans.append(span)
    return spans


# Both spellings are accepted. The marker is written into user documents,
# so renaming the tool must not silently un-suppress a region somebody
# marked a year ago.
_OFF = re.compile(r"slop(?:check|refine):\s*off\b", re.IGNORECASE)
_ON = re.compile(r"slop(?:check|refine):\s*on\b", re.IGNORECASE)
_FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)


def suppressed_ranges(text: str, skip_code: bool = False) -> list[tuple[int, int]]:
    """Regions exempt from checking.

    A line containing ``sloprefine: off`` suppresses everything until a line
    containing ``sloprefine: on`` (or end of file). In Markdown both go inside
    HTML comments. With ``skip_code``, fenced code blocks are exempt too: a
    linter that flags the examples in its own docs gets switched off entirely.
    """
    ranges: list[tuple[int, int]] = []
    off: int | None = None
    for m in re.finditer(r"^.*$", text, re.MULTILINE):
        if off is None and _OFF.search(m.group(0)):
            off = m.start()
        elif off is not None and _ON.search(m.group(0)):
            ranges.append((off, m.end()))
            off = None
    if off is not None:
        ranges.append((off, len(text)))
    if skip_code:
        ranges += [(m.start(), m.end()) for m in _FENCE.finditer(text)]
    return ranges


_HEADING = re.compile(r"^#{1,6} .*$", re.MULTILINE)
# List markers are not sentences. "1." at the start of a line was being
# split off as a zero-word sentence and then refused by the floor rule.
_LIST_MARKER = re.compile(r"^[ \t]*(?:[-*+]|\d{1,3}[.)])[ \t]+", re.MULTILINE)
# A figure on its own line is not prose either. "![](assets/refinery.svg)"
# was read as a three-word sentence and refused by the floor rule, which
# makes illustrating a document cost hits it has not earned. Inline images,
# which sit inside a real sentence, are left alone.
_FIGURE = re.compile(r"^[ \t]*!\[[^\]]*\]\([^)]*\)[ \t]*$", re.MULTILINE)


def markdown_furniture(text: str) -> list[tuple[int, int]]:
    """Headings are not prose. They must not be read as sentences, or a
    heading after two short lines invents a fragment stack that isn't there."""
    return ([(m.start(), m.end()) for m in _HEADING.finditer(text)]
            + [(m.start(), m.end()) for m in _LIST_MARKER.finditer(text)]
            + [(m.start(), m.end()) for m in _FIGURE.finditer(text)])


def mask(text: str, ranges: list[tuple[int, int]]) -> str:
    """Blank out ranges, preserving length and newlines so every offset and
    line number still points at the original file."""
    if not ranges:
        return text
    buf = list(text)
    for start, end in ranges:
        for i in range(max(0, start), min(len(buf), end)):
            if buf[i] != "\n":
                buf[i] = " "
    return "".join(buf)


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def word_count(text: str) -> int:
    return len(WORD.findall(text))


class Document:
    """A parsed document. Tokenization is computed once and reused by checks."""

    def __init__(self, text: str, path: str = "<stdin>"):
        self.text = text
        self.path = path
        self.sentences = sentence_spans(text)
        self.paragraphs = paragraph_spans(text)
        self.words = WORD.findall(text)

    @property
    def word_count(self) -> int:
        return len(self.words)

    def line(self, offset: int) -> int:
        return line_of(self.text, offset)
