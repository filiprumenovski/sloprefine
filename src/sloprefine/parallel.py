"""Parallelism detection: the general case the tricolon rule only sampled.

Why the tricolon rule is not enough
-----------------------------------
`rules.tricolon` bans two surface forms: "a, b, and c" of short items, and the
same word opening three units. Both are one paraphrase from silent:

    No order. No gradient. No structure.          caught
    No order. No gradient. No structure. No spacing.   four items, silent
    No sense of order. No gradient to speak of. No structure at all.
                                               items too long, silent
    We trained on human. We tested on rice. We ran it backwards.
                                               clause-level, silent

Three is not the problem. The problem is a *repeated syntactic skeleton*, and
the count is incidental. Banning arity three pushes a generator to arity four,
which reads identically and scores clean. So this module detects runs of >= 3
sibling units that share a shape, at any arity, at three levels:

    in-sentence   comma or semicolon separated segments
    cross-sentence   consecutive sentences
    paragraph-opening   consecutive paragraphs opening the same way

and reports the run length, so a tetracolon is a worse hit than a tricolon
rather than an escape from it.

Shape, not words
----------------
Two units are siblings when their skeletons match. A skeleton is the first two
token classes plus a length bucket: a function word contributes itself, a
content word contributes a placeholder. So "for the data" and "for the model"
share a skeleton, and so do "We trained on human" and "We tested on rice",
while "The buffer was cold" and "Nobody had checked the timer" do not.

Length bucketing is deliberately coarse (3 words wide) so that padding one
unit does not break the run. That is the specific evasion this replaces: with
exact-length matching, adding one word to the third item defeats the check.

Budget, not ban
---------------
Parallelism is a real device and good writers use it. The output is a density,
runs per 1000 words, and the default budget is 1. One deliberate parallel
construction in a piece is a choice. Six is a cadence.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

from .stylometry import STOPWORDS
from .text import WORD, Document

MIN_RUN = 3
BUCKET = 3          # words per length bucket; coarse on purpose
MAX_UNIT_WORDS = 14  # beyond this, shared shape is coincidence
DEFAULT_BUDGET_PER_1K = 1.0

_SPLIT = re.compile(r"\s*[,;:]\s+")
# Without an Oxford comma the final coordinator carries two items: "fast,
# cheap and reliable" segments as two, not three, and the run is missed.
_FINAL_COORD = re.compile(r"\s+(?:and|or|nor)\s+", re.IGNORECASE)
_LEAD = re.compile(r"^(?:and|or|but|nor|then|so)\s+", re.IGNORECASE)


def _segment(text: str) -> list[str]:
    """Split on commas and semicolons that are NOT inside brackets or quotes.

    Without this, "Binoculars (Hans et al., ICML 2024), whose perplexity..."
    segments at the comma inside the citation and reads as a parallel list.
    """
    parts, buf, depth, quoted = [], [], 0, False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch in '"\u201c\u201d':
            quoted = not quoted
        if ch in ",;:" and depth == 0 and not quoted and i + 1 < len(text) \
                and text[i + 1].isspace():
            parts.append("".join(buf))
            buf = []
            i += 1
            while i < len(text) and text[i].isspace():
                i += 1
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return [p for p in parts if p.strip()]


def _class(token: str) -> str:
    """Function words contribute themselves; content words are a placeholder.
    This is what makes the match structural rather than lexical."""
    t = token.lower()
    return t if t in STOPWORDS else "#"


def _skeleton(text: str) -> tuple | None:
    words = WORD.findall(_LEAD.sub("", text.strip()))
    if not words or len(words) > MAX_UNIT_WORDS:
        return None
    first = _class(words[0])
    second = _class(words[1]) if len(words) > 1 else ""
    return (first, second, len(words) // BUCKET)


def _normalize_carriers(
    segments: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    """Trim the sentence stem off the first segment and the sentence tail off
    the last, so a list matches on its items rather than its frame.

    "It was fast, cheap and reliable" has the stem on item one; "nitrate,
    sulfate and chloride as the analytes" has the tail on item three. Both
    read as three parallel items and neither matches on a full-span skeleton.

    This replaces an earlier approach that grew runs outward from a single
    matching unit, which manufactured three-unit runs out of one real match.
    """
    if len(segments) < 3:
        return segments
    middle = [len(WORD.findall(s[2])) for s in segments[1:-1]]
    if not middle:
        return segments
    n = sorted(middle)[len(middle) // 2]
    if n < 1:
        return segments
    out = list(segments)
    first_words = WORD.findall(_LEAD.sub("", out[0][2]))
    if len(first_words) > n:
        out[0] = (out[0][0], out[0][1], " ".join(first_words[-n:]))
    last_words = WORD.findall(_LEAD.sub("", out[-1][2]))
    if len(last_words) > n:
        out[-1] = (out[-1][0], out[-1][1], " ".join(last_words[:n]))
    return out


def _all_proper(units: list[str]) -> bool:
    """A list of proper nouns is a fact about the world. "Qwen, Gemma and
    Llama" is not a cadence choice, and neither is a citation."""
    firsts = [WORD.findall(u)[:1] for u in units]
    return all(f and f[0][:1].isupper() for f in firsts)


# Doublets. MIN_RUN = 3 was the same rule-of-three assumption this module
# exists to reject: the tell is BALANCE, and balance starts at two. The
# antithetical pair is arguably the most characteristic machine construction
# there is ("that isn't what's there, and it isn't a close call"), and every
# instance of it sits one unit under the run threshold.
#
# Doublets are also ordinary English, so the criteria are tight: two adjacent
# units of near-equal length that share a content word or a negator, with a
# polarity flip or a repeated opening counting extra. A shared pronoun does
# not count, or "I ran it again, and I got the same nothing" would fire.
DOUBLET_MIN_WORDS = 2
DOUBLET_MAX_WORDS = 14
DOUBLET_MAX_DIFF = 3

_NEGATOR = re.compile(
    r"\b(?:not|no|never|nothing|none|nor|isn'?t|aren'?t|wasn'?t|doesn'?t|"
    r"don'?t|didn'?t|hasn'?t|haven'?t|won'?t|can'?t|cannot|couldn'?t)\b", re.IGNORECASE)

# Only a real polarity contrast counts for the same-opening path. Treating
# quantifiers as negation fired on "I ran it again with fresh reagent, and I
# got the same nothing back", which is ordinary prose.
_STRONG_NEGATOR = re.compile(
    r"\b(?:not|never|cannot|isn'?t|aren'?t|wasn'?t|doesn'?t|don'?t|didn'?t|"
    r"hasn'?t|haven'?t|won'?t|can'?t|couldn'?t)\b", re.IGNORECASE)

_PRONOUN = frozenset(
    ["i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them", "this", "that", "these", "those", "there"])


def _shared_anchor(a: list[str], b: list[str]) -> str | None:
    """A token both units share that is not a bare pronoun."""
    sa, sb = {w.lower() for w in a}, {w.lower() for w in b}
    for w in sorted(sa & sb):
        if w in _PRONOUN:
            continue
        if w not in STOPWORDS or _NEGATOR.fullmatch(w):
            return w
    return None


def _is_doublet(left: str, right: str) -> str | None:
    """Return a reason string when two units form a balanced pair."""
    a = WORD.findall(_LEAD.sub("", left.strip()))
    b = WORD.findall(_LEAD.sub("", right.strip()))
    if not (DOUBLET_MIN_WORDS <= len(a) <= DOUBLET_MAX_WORDS):
        return None
    if not (DOUBLET_MIN_WORDS <= len(b) <= DOUBLET_MAX_WORDS):
        return None
    if abs(len(a) - len(b)) > DOUBLET_MAX_DIFF:
        return None

    anchor = _shared_anchor(a, b)
    neg_a, neg_b = bool(_NEGATOR.search(left)), bool(_NEGATOR.search(right))
    flip = neg_a != neg_b
    same_open = a[0].lower() == b[0].lower()

    # Signals are checked independently. Gating them all behind a shared
    # content anchor missed "Same catchments. Same horizon types.", where the
    # repeated word is a stopword and IS the whole construction.
    if same_open and a[0].lower() not in _PRONOUN:
        return f"anaphora on '{a[0].lower()}'"
    if len(a) > 1 and len(b) > 1 and [w.lower() for w in a[:2]] == [w.lower() for w in b[:2]]:
        return f"repeated opening '{a[0]} {a[1]}'"
    strong_flip = (bool(_STRONG_NEGATOR.search(left))
                   != bool(_STRONG_NEGATOR.search(right)))
    if strong_flip and same_open and abs(len(a) - len(b)) <= 2:
        return "antithesis, same opening"
    if flip and anchor is not None:
        return f"antithesis on '{anchor}'"
    if neg_a and neg_b and anchor is not None and _NEGATOR.fullmatch(anchor):
        return f"negated pair on '{anchor}'"
    return None


@dataclass(frozen=True)
class Run:
    start: int
    end: int
    level: str        # "in-sentence" | "cross-sentence" | "paragraph"
    arity: int
    preview: str

    def describe(self) -> str:
        return f"{self.arity} parallel units ({self.level})"


def _runs_over(units: list[tuple[int, int, str]]) -> list[tuple[int, int, int]]:
    """Index runs of >= MIN_RUN consecutive units sharing a skeleton."""
    out = []
    i = 0
    while i < len(units):
        skeleton = _skeleton(units[i][2])
        if skeleton is None:
            i += 1
            continue
        j = i + 1
        while j < len(units) and _skeleton(units[j][2]) == skeleton:
            j += 1
        if j - i >= MIN_RUN:
            out.append((i, j - 1, j - i))
        i = max(j, i + 1)
    return out


def _anaphora_runs(units: list[tuple[int, int, str]]) -> list[tuple[int, int, int]]:
    """Same opening word repeated. Caught separately because anaphora survives
    wild length variation, which the skeleton bucket would not tolerate."""
    out = []
    i = 0
    while i < len(units):
        words = WORD.findall(_LEAD.sub("", units[i][2]))
        if not words:
            i += 1
            continue
        lead = words[0].lower()
        j = i + 1
        while j < len(units):
            nxt = WORD.findall(_LEAD.sub("", units[j][2]))
            if not nxt or nxt[0].lower() != lead:
                break
            j += 1
        if j - i >= MIN_RUN:
            out.append((i, j - 1, j - i))
        i = max(j, i + 1)
    return out


def doublets(doc: Document) -> list[Run]:
    """Balanced pairs, inside a sentence and across adjacent sentences."""
    out: list[Run] = []
    for span in doc.sentences:
        segs = _segment(span.text)
        cursor = span.start
        located = []
        for piece in segs:
            idx = doc.text.find(piece, cursor)
            if idx < 0:
                continue
            located.append((idx, idx + len(piece), piece))
            cursor = idx + len(piece)
        for (s1, _, left), (_, e2, right) in itertools.pairwise(located):
            reason = _is_doublet(left, right)
            if reason:
                out.append(Run(s1, e2, f"doublet, {reason}", 2,
                               f"{left.strip()} | {right.strip()}"[:70]))
    sents = [(s.start, s.end, s.text) for s in doc.sentences]
    for (s1, _, left), (_, e2, right) in itertools.pairwise(sents):
        reason = _is_doublet(left, right)
        if reason:
            out.append(Run(s1, e2, f"doublet, {reason}", 2,
                           f"{left.strip()} | {right.strip()}"[:70]))
    return _dedupe(out)


def find(doc: Document) -> list[Run]:
    runs: list[Run] = []

    # in-sentence: comma/semicolon separated segments
    for span in doc.sentences:
        segments: list[tuple[int, int, str]] = []
        cursor = span.start
        pieces = _segment(span.text)
        if len(pieces) >= 2 and _FINAL_COORD.search(pieces[-1]):
            head, _, tail = _FINAL_COORD.split(pieces[-1], maxsplit=1)[0], None, \
                _FINAL_COORD.split(pieces[-1], maxsplit=1)[-1]
            pieces = pieces[:-1] + [head, tail]
        for piece in pieces:
            idx = doc.text.find(piece, cursor)
            if idx < 0:
                continue
            segments.append((idx, idx + len(piece), piece))
            cursor = idx + len(piece)
        if len(segments) < MIN_RUN:
            continue
        # Carrier trimming helps the skeleton path and destroys the anaphora
        # path: trimming the stem off item one removes the repeated opening
        # word that anaphora is defined by. Each path gets its own view.
        normalized = _normalize_carriers(segments)
        for a, b, arity in (_runs_over(normalized) + _anaphora_runs(segments)):
            texts = [segments[k][2] for k in range(a, b + 1)]
            if _all_proper(texts):
                continue
            runs.append(Run(segments[a][0], segments[b][1], "in-sentence",
                            arity, " | ".join(texts)[:70]))

    # cross-sentence and paragraph-opening
    sentence_units = [(s.start, s.end, s.text) for s in doc.sentences]
    for a, b, arity in _runs_over(sentence_units) + _anaphora_runs(sentence_units):
        runs.append(Run(sentence_units[a][0], sentence_units[b][1],
                        "cross-sentence", arity,
                        " | ".join(u[2] for u in sentence_units[a:b + 1])[:70]))

    para_units = [(p.start, p.end, p.text) for p in doc.paragraphs]
    for a, b, arity in _anaphora_runs(para_units):
        runs.append(Run(para_units[a][0], para_units[b][1], "paragraph",
                        arity, " | ".join(u[2][:22] for u in para_units[a:b + 1])))

    return _dedupe(runs)


def _dedupe(runs: list[Run]) -> list[Run]:
    kept: list[Run] = []
    for run in sorted(runs, key=lambda r: (r.start, -(r.end - r.start), -r.arity)):
        if any(run.start >= k.start and run.end <= k.end for k in kept):
            continue
        kept.append(run)
    return kept


def density(doc: Document, runs: list[Run] | None = None) -> float:
    runs = find(doc) if runs is None else runs
    words = doc.word_count or 1
    return round(len(runs) / words * 1000, 3)
