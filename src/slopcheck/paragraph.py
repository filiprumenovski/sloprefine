"""Paragraph-level structure: where the punch lands, and whether there is
anything specific in the paragraph at all.

Two checks, of two different kinds.

Positional: the closer budget
-----------------------------
Machine prose puts its punch at the end of the paragraph, because that is
where the punchline goes in the register it learned. `rules.fragments` catches
the staccato run and `parallel` catches the repeated shape, but both are
pattern matchers and both can be satisfied by rephrasing. A positional rule
cannot: it does not care what the closing sentence says, only that a short one
keeps arriving in the same slot. To satisfy it a generator has to stop putting
punches there, which is the behaviour change actually wanted.

Requirement: the concreteness floor
-----------------------------------
Every other rule in this package is a prohibition, and every prohibition is
satisfiable by writing less. A requirement is not: a paragraph with nothing
specific in it fails no matter how it is phrased. Generic prose is the failure
mode that matters, and specifics are what generic prose lacks. "Several
studies" fails; "Kobak 2025" passes.

The obvious abuse is invented precision, and this check cannot detect it. A
model told to produce a number per paragraph will produce a number, and a
wrong number is worse than a vague phrase because it is checkable and false.
This check pairs with a fact-checking step it does not provide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .cadence import has_finite_verb
from .text import Document, Span

CLOSER_MAX_WORDS = 8
CLOSER_BUDGET_RATIO = 0.25   # share of paragraphs allowed to end on a punch
MIN_PARAGRAPH_WORDS = 40     # below this, a paragraph owes no specifics

_NUMERAL = re.compile(r"\b\d")
_YEAR = re.compile(r"\b(?:1[6-9]|20|21)\d{2}\b")
_UNIT = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent|fold|x|kb|mb|gb|ms|s|kg|g|mg|nm|um|mm|cm|km|bp|kDa)\b", re.IGNORECASE)
# Spoken prose spells its numbers. "four times out of five" is as specific as
# "4/5" and the digit test misses it entirely, which made a talk score as
# having no specifics anywhere in it.
_NUMBER_WORD = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
    r"million|billion|half|quarter|third|twice|dozen)\b", re.IGNORECASE)

_QUOTED = re.compile(r"[\"\u201c][^\"\u201d]{2,40}[\"\u201d]")
_CODE = re.compile(r"`[^`]{2,60}`")


@dataclass(frozen=True)
class Closer:
    span: Span
    words: int
    verbless: bool


def closers(doc: Document) -> list[Closer]:
    """Paragraphs whose final sentence is short."""
    out: list[Closer] = []
    sentences, i = doc.sentences, 0
    for para in doc.paragraphs:
        while i < len(sentences) and sentences[i].start < para.start:
            i += 1
        j = i
        while j < len(sentences) and sentences[j].end <= para.end:
            j += 1
        inside = sentences[i:j]
        i = j
        if len(inside) < 2:
            # A one-sentence paragraph is a heading, a caption or a beat by
            # construction. Charging it as a closer would flag every list.
            continue
        last = inside[-1]
        if len(last) <= CLOSER_MAX_WORDS:
            out.append(Closer(last, len(last), not has_finite_verb(last)))
    return out


def closer_ratio(doc: Document) -> float:
    paragraphs = [p for p in doc.paragraphs if len(p) >= 15]
    if not paragraphs:
        return 0.0
    return round(len(closers(doc)) / len(paragraphs), 3)


def specifics(text: str) -> list[str]:
    """Concrete anchors: numbers, years, units, proper nouns, quoted or code
    terms. Sentence-initial capitals are excluded, since every sentence has
    one and it says nothing about specificity."""
    found: list[str] = []
    found += _YEAR.findall(text)
    found += [m.group(0) for m in _UNIT.finditer(text)]
    found += [m.group(0) for m in _QUOTED.finditer(text)]
    found += [m.group(0) for m in _CODE.finditer(text)]
    found += [m.group(0) for m in _NUMBER_WORD.finditer(text)]
    if not found:
        found += [m.group(0) for m in _NUMERAL.finditer(text)]

    # proper nouns: capitalised words that are not opening a sentence
    for m in re.finditer(r"(?<=[a-z,;:)\"'\s])\b([A-Z][a-zA-Z'\u2019-]{1,})", text):
        before = text[: m.start()].rstrip()
        if before and before[-1] in ".!?":
            continue
        if not before:
            continue
        found.append(m.group(1))
    return found


def vague_paragraphs(doc: Document, min_words: int = MIN_PARAGRAPH_WORDS
                     ) -> list[Span]:
    """Paragraphs long enough to owe a specific, and carrying none."""
    return [p for p in doc.paragraphs
            if len(p) >= min_words and not specifics(p.text)]


def specificity_per_1k(doc: Document) -> float:
    words = doc.word_count or 1
    return round(len(specifics(doc.text)) / words * 1000, 2)
