"""Paragraph structure in expository and scientific prose.

The rule table catches words and the cadence layer catches what a listener
hears. Neither sees what a de-slopped draft keeps after both have been
satisfied: every paragraph given the same weight, the same length, the same
caveat and the same pointer to a supplementary table. Banning the markers
does not remove that shape. It is partly produced by the bans, because each
suppressed habit leaves a replacement behind. Forbid the confident causal
verb and "accompanied" arrives in its place. Require every limitation to be
stated and every paragraph closes on one.

Six measurements live here. All are positional, a budget, or a repetition,
so none can be satisfied by swapping one word for its synonym.

  rhythm    sentence lengths inside a paragraph sit in one narrow band
  tail      a paragraph closes on a caveat or on a pointer to elsewhere
  pointer   a sentence or clause whose only content is where a result lives
  caveat    limitation sentences scattered through many paragraphs rather
            than gathered into a few
  relverb   one relation-gesture verb ("accompanied", "mirrored") carrying
            the same load over and over
  gloss     the same term defined twice, in the body and again in a box,
            glossary or later section

How these were measured
-----------------------
Two piles. On one side is the main text of one machine-assisted scientific
manuscript (3,945 words, drafted with an LLM under a no-tells instruction and
then edited). On the other is the "learned" section of the Brown corpus: 80
samples of scientific and academic prose published in 1961, 159,000 words,
with its 75 samples of belles lettres as a second human register. The
manuscript is ONE document, so its column is an illustration and not a
rate. Brown is human and pre-LLM by construction, which makes it a fair test
of false positives and a poor test of anything else: papers in 1961 had no
supplementary tables to point at, so `pointer` is quiet there for reasons of
era as much as of authorship.

    measure                          manuscript  Brown learned  Brown belles
    paragraph ends on caveat/pointer   25%         0.4%           0.1%
    paragraph carries a caveat         33%         0.7%           0.3%
    flat paragraph (CV<0.20, <30w)     17%         4.6%           5.5%
    terms defined twice                  2          0 of 80 docs   0 of 75
    "accompany" per 1000 words         1.01        0.08           0.09

With default budgets, documents in which each rule fires at least once:

    rule      manuscript  Brown learned  Brown belles
    tail        8 hits      0 of 80        0 of 75
    pointer     7 hits      1 of 80        0 of 75
    caveat      6 hits      0 of 80        0 of 75
    relverb     3 hits      1 of 80        3 of 75
    gloss       2 hits      0 of 80        0 of 75
    rhythm      2 hits      6 of 80        9 of 75   (at a 10% budget)

`rhythm` is the weak one. Flat paragraphs are about 3.5 times as common in
the manuscript, but at the setting that catches any of them it also fires on
one human document in ten, so it ships off. Turn it on with
`rhythm_budget_ratio` when the reference register is known to be varied.

What was tried and dropped
--------------------------
The first version of `tail` was a sequence rule: classify each sentence as a
claim, a number, a caveat or a pointer, collapse repeats, and flag three
consecutive paragraphs sharing one sequence. The manuscript it was built
for produced one run with more than one role in it, two paragraphs long, and
its sequence was claim, number, claim. The template is real but it is not
exact: paragraphs vary in how many numbers they carry, so their sequences
rarely match token for token. What does repeat is the slot at the end. The
positional reading is what ships.

The first `rhythm` computed the coefficient of variation over every
sentence. Adding one five-word fragment to a flat paragraph raised that CV
past the threshold and cleared the flag, which rewards the TED cadence that
`cadence.py` exists to remove. The CV is now taken over sentences longer
than the fragment ceiling only, so a fragment changes nothing and the fix
has to be a longer sentence.
"""

from __future__ import annotations

import bisect
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass

from .stylometry import STOPWORDS
from .text import Document, Span

# ------------------------------------------------------------- paragraphing

MIN_PARAGRAPH_WORDS = 15  # same floor as paragraph._rateable


def sentences_by_paragraph(doc: Document) -> list[tuple[Span, list[Span]]]:
    """Each paragraph with the sentences wholly inside it."""
    starts = [s.start for s in doc.sentences]
    out = []
    for para in doc.paragraphs:
        i = bisect.bisect_left(starts, para.start)
        inside = []
        while i < len(doc.sentences) and doc.sentences[i].end <= para.end:
            inside.append(doc.sentences[i])
            i += 1
        out.append((para, inside))
    return out


def rateable(doc: Document) -> list[tuple[Span, list[Span]]]:
    """Paragraphs long enough to have a shape: two sentences or more and 15
    words or more. A one-sentence paragraph is a heading, a caption or a
    list item, and has no end distinct from its beginning."""
    return [(p, s) for p, s in sentences_by_paragraph(doc)
            if len(p) >= MIN_PARAGRAPH_WORDS and len(s) >= 2]


# ------------------------------------------------------------------- rhythm

FRAGMENT_CEILING = 7      # sentences at or under this are not counted
RHYTHM_MIN_SENTENCES = 4  # of the counted kind, before a paragraph is rated
RHYTHM_CV = 0.20
RHYTHM_LONG = 30          # one sentence this long and the paragraph has range


@dataclass(frozen=True)
class Flat:
    span: Span
    cv: float
    lengths: tuple[int, ...]


def flat_paragraphs(doc: Document) -> tuple[list[Flat], int]:
    """Paragraphs whose non-fragment sentences all sit in one narrow band.

    Returns the flat paragraphs and the number that were eligible. The
    fragment exclusion is the whole design: see the module docstring."""
    flats, eligible = [], 0
    for para, sents in sentences_by_paragraph(doc):
        lengths = [len(s) for s in sents if len(s) > FRAGMENT_CEILING]
        if len(lengths) < RHYTHM_MIN_SENTENCES:
            continue
        eligible += 1
        mean = statistics.mean(lengths)
        cv = statistics.pstdev(lengths) / mean
        if cv < RHYTHM_CV and max(lengths) < RHYTHM_LONG:
            flats.append(Flat(para, round(cv, 3), tuple(lengths)))
    return flats, eligible


# ----------------------------------------------------------- pointer clauses

_REF_HEAD = (r"(?:Supplementary\s+|Extended\s+Data\s+)?"
             r"(?:Tables?|Figs?\.|Figures?|Box(?:es)?|Methods|Notes?|"
             r"Information|Appendix|Appendices|Section)")
_REF_ID = r"(?:\s*S?\d+[a-z]?(?:\s*[,–-]\s*S?\d*[a-z]?)*)"
REF_SPAN = re.compile(
    _REF_HEAD + _REF_ID + r"?(?:\s*(?:,|and|to)\s*(?:" + _REF_HEAD + r")?"
    + _REF_ID + r")*")
_HAS_REF = re.compile(
    r"\b(?:Supplementary\s+\w+|(?:Tables?|Figs?\.|Figures?|Box)\s*S?\d"
    r"|Methods\b|Appendix)")

_POINT_PASSIVE = re.compile(
    r"\b(?:is|are|was|were)\s+(?:\w+ly\s+)?(?:reported|given|provided|shown|"
    r"listed|described|detailed|summari[sz]ed|presented|tabulated|available|"
    r"collected|documented)\b", re.IGNORECASE)
_POINT_ACTIVE = re.compile(
    r"\b(?:Tables?|Figs?\.|Figures?|Methods|Box|Notes?|Information)"
    r"(?:\s+S?\d+[a-z]?)?\s+(?:reports?|gives?|lists?|shows?|describes?|"
    r"details?|provides?|summari[sz]es?|contains?|collects?|"
    r"specif(?:y|ies)|documents?|explains?)\b(?!\s+(?:that|how|whether)\b)")
# What a pointer points at. A sentence that says the instrument is shown in
# Figure 1 tells the reader what the instrument is; one that says the
# comparisons are given in Table S22 tells them only where to go.
_META = re.compile(
    r"\b(?:comparisons?|results?|data|details?|values?|estimates?|intervals?|"
    r"denominators?|components?|populations?|definitions?|analys[ie]s|"
    r"coverage|statistics|numbers|counts|ratios|methods|procedures|criteria|"
    r"parameters|breakdowns?|summar(?:y|ies)|tests|windows|transformations|"
    r"sources|derivations?|specifications?|rules|thresholds|record|"
    r"uncertaint(?:y|ies)|sensitivity|outcomes?)\b", re.IGNORECASE)


def strip_refs(text: str) -> str:
    """The clause with its references removed: parenthetical ones whole,
    inline ones by their head and identifiers."""
    text = re.sub(r"\([^()]*" + _HAS_REF.pattern + r"[^()]*\)", " ", text)
    return REF_SPAN.sub(" ", text)


def clauses(span: Span) -> list[Span]:
    """A sentence split on semicolons, offsets kept."""
    out, cursor = [], 0
    for m in re.finditer(r";", span.text):
        out.append(_trim(span, cursor, m.start()))
        cursor = m.end()
    out.append(_trim(span, cursor, len(span.text)))
    return [c for c in out if c is not None]


def _trim(span: Span, a: int, b: int) -> Span | None:
    piece = span.text[a:b]
    lead = len(piece) - len(piece.lstrip())
    piece = piece.strip()
    if not piece:
        return None
    start = span.start + a + lead
    return Span(start, start + len(piece), piece)


def is_pointer(text: str) -> bool:
    """Only content is a location. A reference, a pointing verb, a meta-noun
    as its subject or object, and no number left once the references are
    gone, since a clause that carries a figure is reporting a result."""
    if not _HAS_REF.search(text):
        return False
    passive = _POINT_PASSIVE.search(text)
    active = _POINT_ACTIVE.search(text)
    if passive:
        target = text[:passive.start()]
    elif active:
        target = text[active.end():]
    else:
        return False
    if not _META.search(target):
        return False
    return not re.search(r"\d", strip_refs(text))


def pointer_clauses(doc: Document) -> list[Span]:
    return [c for s in doc.sentences for c in clauses(s) if is_pointer(c.text)]


# ------------------------------------------------------------------ caveats

_EPISTEMIC = (r"(?:establish|demonstrate|prove|determine|imply|exclude|"
              r"distinguish|resolve|confirm|verify|guarantee|attribute|"
              r"show\s+(?:that|whether|how|which))")
CAVEAT = re.compile(
    r"\b(?:cannot|can\s+not|could\s+not|does\s+not|do\s+not|did\s+not|"
    r"will\s+not|may\s+not)\s+(?:by\s+(?:itself|themselves)\s+)?(?:be\s+)?"
    r"(?:wholly\s+|fully\s+|independently\s+)?" + _EPISTEMIC + r"\w*"
    r"|\b(?:is|are|was|were)\s+not\s+(?:independently\s+)?(?:measured|"
    r"validated|established|determined|verified|estimated)\b"
    r"|\b(?:alone|by\s+itself|by\s+themselves)\s+(?:cannot|does\s+not|do\s+not)\b"
    r"|\bincomplete\s+(?:view|evidence|picture|record)\b"
    r"|\blower\s+bounds?\b"
    r"|\b(?:requires?|warrants?)\s+caution\b"
    r"|\bshould\s+be\s+(?:read|interpreted)\s+(?:with\s+caution|cautiously|alongside)\b"
    r"|\b(?:is|are|remains?)\s+unrecorded\b"
    r"|\bcannot\s+be\s+(?:excluded|ruled\s+out)\b",
    re.IGNORECASE)


def is_caveat(text: str) -> bool:
    return bool(CAVEAT.search(text))


# -------------------------------------------------------------------- roles

def role(span: Span) -> str:
    """claim | number | caveat | pointer. Pointer wins when the sentence ends
    on a pointing clause, caveat when any clause limits a claim."""
    parts = clauses(span)
    if parts and is_pointer(parts[-1].text):
        return "pointer"
    if is_caveat(span.text):
        return "caveat"
    if re.search(r"\d", strip_refs(span.text)):
        return "number"
    return "claim"


def tails(doc: Document) -> tuple[list[tuple[Span, str]], int]:
    """Final sentences that are a caveat or a pointer, and the number of
    paragraphs rated."""
    paras = rateable(doc)
    out = []
    for _, sents in paras:
        r = role(sents[-1])
        if r in ("caveat", "pointer"):
            out.append((sents[-1], r))
    return out, len(paras)


def caveat_paragraphs(doc: Document) -> tuple[list[tuple[int, float, list[Span]]], int]:
    """For each rateable paragraph carrying a caveat: how many caveat
    sentences it holds in all, what share of its sentences they are, and
    those outside its final slot. The final
    slot belongs to `tail`, so a closing caveat counts toward how dense the
    paragraph is but is reported once, by the rule whose fix is about
    position."""
    paras = rateable(doc)
    found = []
    for _, sents in paras:
        flagged = [s for s in sents if is_caveat(s.text)]
        if flagged:
            inside = [s for s in flagged if s is not sents[-1]]
            found.append((len(flagged), len(flagged) / len(sents), inside))
    return found, len(paras)


# ---------------------------------------------------------- stand-in verbs

# Verbs that gesture at a relationship without stating it. Each has a plain
# literal sense that is rare in expository prose, which is why "contribute"
# and "track" are not here: the first is literal in any paper about who
# contributed what, the second in any paper about software. "Correspond" was
# here and fired on 10 uses in one 1961 mathematics paper, where
# "corresponding to" is how a mapping is stated. "Underlying" as an
# adjective is ordinary; only the verb forms count.
STANDIN_VERBS: dict[str, str] = {
    "accompany": r"\baccompan(?:y|ies|ied|ying)\b",
    "coincide": r"\bcoincid(?:e|es|ed|ing)\b",
    "mirror": r"\bmirror(?:s|ed|ing)\b",
    "parallel": r"\bparallel(?:s|ed|ing|led|ling)\b",
    "reflect": r"\breflect(?:s|ed|ing)?\b",
    "align": r"\balign(?:s|ed)?\s+with\b",
    "help explain": r"\bhelp(?:s|ed|ing)?\s+(?:to\s+)?explain\b",
    "encompass": r"\bencompass(?:es|ed|ing)?\b",
    "underlie": r"\bunderl(?:ie|ies|ay)\b",
}


def standin_uses(doc: Document) -> dict[str, list[tuple[int, int, str]]]:
    out: dict[str, list[tuple[int, int, str]]] = {}
    for family, pattern in STANDIN_VERBS.items():
        found = [(m.start(), m.end(), m.group(0))
                 for m in re.finditer(pattern, doc.text, re.IGNORECASE)]
        if found:
            out[family] = found
    return out


# ------------------------------------------------------------------- glosses

_DEF_VERBS = (r"(?:means?|denotes?|refers?\s+to|measures?|express(?:es)?|"
              r"accounts?\s+for|average|averages|links?|stores?|index(?:es)?|"
              r"represents?|combines?|captures?|quantif(?:y|ies)|"
              r"summari[sz]es?|fragments?)")
# A sentence that opens on a bare term and defines it. Narrower verb list:
# "Paper links cover 75%" would otherwise read "links" as the verb.
_OPEN_VERBS = (r"(?:means?|denotes?|refers?\s+to|measures?|express(?:es)?|"
               r"accounts?\s+for|average|averages|quantif(?:y|ies)|summari[sz]es?)")
_WHICH = re.compile(
    r"\b((?:[A-Za-z][\w-]*\s+){0,2}[A-Za-z][\w-]*(?:\s+\([A-Z][\w]*\))?)"
    r"(?:\s+files?)?,\s+which\s+" + _DEF_VERBS + r"\b")
_OPEN = re.compile(r"^((?:[A-Z][a-z][\w-]*)(?:\s+[a-z][\w-]*){0,2})\s+"
                   + _OPEN_VERBS + r"\b")
_PAREN = re.compile(
    r"\b((?:[a-z][\w-]*\s+)?[a-z][\w-]*)\s+\((?!(?:or|and|e\.g|i\.e|"
    r"especially|probably|approximately|see|including|about|mostly|mainly|"
    r"a|an|the|not|only|usually|often)\b)([a-z][a-z -]{3,60})\)")
# A glossary or box entry: "Effective number. Hill numbers express ..."
_LABEL = re.compile(r"^([A-Z][a-z]+(?:[ -][a-z]+){0,3})\.\s")
_NOT_HEADS = frozenset(["it", "this", "that", "these", "those", "there",
                        "they", "we", "which", "what", "here", "each", "one",
                        "both", "all", "most", "some", "such"])


@dataclass(frozen=True)
class Definition:
    term: str
    start: int
    end: int
    kind: str  # label | open | which | paren
    text: str


def _term_key(term: str) -> str | None:
    acronym = re.findall(r"\b[A-Z]{2,}[a-z]?\b", term)
    if acronym:
        return acronym[-1]
    words = [w.lower() for w in re.findall(r"[A-Za-z][\w-]*", term)]
    words = [w for w in words if w not in {"the", "a", "an", "file", "files"}]
    if not words or words[0] in _NOT_HEADS:
        return None

    def singular(w: str) -> str:
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            return w[:-1]
        return w
    return " ".join(singular(w) for w in words[-2:])


def definitions(doc: Document) -> list[Definition]:
    found: list[Definition] = []
    for para, sents in sentences_by_paragraph(doc):
        m = _LABEL.match(para.text)
        if not m or len(m.group(1)) < 5 or not sents:
            continue
        body = para.text[m.end():]
        first = re.split(r"(?<=[.!?])\s", body, maxsplit=1)[0]
        if re.search(r"\b" + _DEF_VERBS + r"\b", first):
            key = _term_key(m.group(1))
            if key:
                found.append(Definition(key, para.start, para.start + m.end(),
                                        "label", m.group(0).strip()))
    for s in doc.sentences:
        m = _OPEN.match(s.text)
        bare = re.sub(r"\([^()]*\)", " ", s.text)
        if (m and len(bare.split()) <= 25
                and m.group(1).split()[0].lower() not in STOPWORDS):
            key = _term_key(m.group(1))
            if key:
                found.append(Definition(key, s.start, s.end, "open", s.text))
        for pattern, kind in ((_WHICH, "which"), (_PAREN, "paren")):
            for m in pattern.finditer(s.text):
                key = _term_key(m.group(1))
                if key:
                    found.append(Definition(key, s.start + m.start(),
                                            s.start + m.end(), kind, m.group(0)))
    return sorted(found, key=lambda d: d.start)


def repeated_glosses(doc: Document) -> list[tuple[Definition, Definition]]:
    """(duplicate, kept) pairs. When a term has a box or glossary entry, that
    entry is the one kept and every inline gloss of it is the duplicate,
    because the box is where a reader goes to look a term up. Otherwise the
    first definition is kept."""
    by_term: dict[str, list[Definition]] = defaultdict(list)
    for d in definitions(doc):
        by_term[d.term].append(d)
    out = []
    for defs in by_term.values():
        if len(defs) < 2:
            continue
        labels = [d for d in defs if d.kind == "label"]
        kept = labels[0] if labels else defs[0]
        out += [(d, kept) for d in defs if d is not kept]
    return sorted(out, key=lambda pair: pair[0].start)


# ------------------------------------------------------------------ summary

def summary(doc: Document) -> dict:
    """Shares for the report. A share needs a denominator, so each comes
    with the count it was taken over."""
    flats, eligible = flat_paragraphs(doc)
    tail, rated = tails(doc)
    with_caveat = sum(
        1 for _, sents in rateable(doc) if any(is_caveat(s.text) for s in sents))
    return {
        "rated_paragraphs": rated,
        "tail_ratio": round(len(tail) / rated, 3) if rated else 0.0,
        "caveat_paragraph_share": round(with_caveat / rated, 3) if rated else 0.0,
        "flat_paragraph_share": round(len(flats) / eligible, 3) if eligible else 0.0,
        "pointer_clauses": len(pointer_clauses(doc)),
        "repeated_glosses": len(repeated_glosses(doc)),
    }
