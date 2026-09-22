"""Checks. Each yields Hit objects carrying offsets back into the source."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .cadence import has_finite_verb
from .paragraph import closers, vague_paragraphs
from .parallel import doublets as find_doublets
from .parallel import find as find_parallel
from .rules import LEXICON, RULES
from .stylometry import STOPWORDS
from .text import WORD, Document

FRAGMENT_MAX_WORDS = 7
FRAGMENT_RUN = 3
MIN_SENTENCE_WORDS = 0  # 0 disables the floor; 5 is the usual setting


@dataclass(frozen=True)
class Hit:
    rule_id: str
    start: int
    end: int
    text: str
    note: str = ""

    @property
    def rule(self):
        return RULES[self.rule_id]


def _regex_check(doc: Document, rule_id: str) -> list[Hit]:
    hits = []
    for pattern in RULES[rule_id].patterns:
        for m in re.finditer(pattern, doc.text, re.IGNORECASE):
            hits.append(Hit(rule_id, m.start(), m.end(), m.group(0).strip()))
    return hits


_QUOTED_SPAN = re.compile(r"[\"\u201c][^\"\u201d\n]{1,80}[\"\u201d]|`[^`\n]{1,80}`")


def check_vocab(doc: Document, allow: frozenset[str] = frozenset()) -> list[Hit]:
    """Quoting a marker is not using one. A sentence that discusses "delves"
    is talking about the word, and flagging it makes the tool unusable for
    writing about the tool."""
    quoted = [(m.start(), m.end()) for m in _QUOTED_SPAN.finditer(doc.text)]
    hits = []
    for m in WORD.finditer(doc.text):
        token = m.group(0).lower().replace("\u2019", "'")
        if token not in LEXICON or token in allow:
            continue
        if any(s <= m.start() < e for s, e in quoted):
            continue
        hits.append(Hit("vocab", m.start(), m.end(), m.group(0)))
    return hits


def check_fragments(doc: Document) -> list[Hit]:
    """Runs of very short sentences. Spans paragraph breaks on purpose: a
    listener has no paragraphs, so the cadence carries across them."""
    hits = []
    run: list = []
    for span in doc.sentences:
        if len(span) <= FRAGMENT_MAX_WORDS:
            run.append(span)
        else:
            hits += _emit_run(run)
            run = []
    hits += _emit_run(run)
    return hits


def _emit_run(run: list) -> list[Hit]:
    if len(run) < FRAGMENT_RUN:
        return []
    return [Hit("fragments", run[0].start, run[-1].end,
                " / ".join(s.text for s in run), f"{len(run)} in a row")]


def check_colon(doc: Document) -> list[Hit]:
    hits = []
    # [ \t]+ not \s+: a colon at the end of a line introduces a block, not a
    # reveal. (?<!\d) so a decimal point does not end the clause.
    for m in re.finditer(r"[a-z]{3,}:[ \t]+[A-Za-z][^.\n]{0,70}(?<!\d)\.", doc.text):
        hits.append(Hit("colon", m.start(), m.end(), m.group(0)))
    return hits


def check_question(doc: Document) -> list[Hit]:
    hits = []
    for para in doc.paragraphs:
        first = next((s for s in doc.sentences if s.start >= para.start), None)
        if first is not None and first.start < para.end and first.text.endswith("?"):
            hits.append(Hit("question", first.start, first.end, first.text))
    return hits


def check_emoji(doc: Document) -> list[Hit]:
    pattern = re.compile(
        "[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]"
    )
    return [Hit("emoji", m.start(), m.end(), m.group(0))
            for m in pattern.finditer(doc.text)]


def _dedupe(hits: list[Hit]) -> list[Hit]:
    seen, out = set(), []
    for h in sorted(hits, key=lambda x: (x.start, -x.end)):
        if any(h.start >= s and h.end <= e for s, e in seen):
            continue
        seen.add((h.start, h.end))
        out.append(h)
    return out


CHECKS = {
    "vocab": check_vocab,
    "emdash": lambda d: _regex_check(d, "emdash"),
    "negation": lambda d: _regex_check(d, "negation"),
    "fragments": check_fragments,
    "transitions": lambda d: _regex_check(d, "transitions"),
    "narrator": lambda d: _regex_check(d, "narrator"),
    "participial": lambda d: _regex_check(d, "participial"),
    "colon": check_colon,
    "question": check_question,
    "recap": lambda d: _regex_check(d, "recap"),
    "hedge": lambda d: _regex_check(d, "hedge"),
    "emoji": check_emoji,
}


def run_checks(
    doc: Document,
    disabled=(),
    allow=frozenset(),
    floor: int = MIN_SENTENCE_WORDS,
    runt_mode: str = "verbless",
    allow_runts: frozenset[str] = frozenset(),
    parallel_budget_per_1k: float = 1.0,
    closer_budget_ratio: float = 0.25,
    doublet_budget_per_1k: float = 2.0,
    vague_min_words: int = 40,
) -> list[Hit]:
    hits: list[Hit] = []
    for rule_id, fn in CHECKS.items():
        if rule_id in disabled:
            continue
        if rule_id == "vocab":
            hits += check_vocab(doc, allow)
        elif rule_id == "runt":
            hits += check_runt(doc, floor, runt_mode, allow_runts)
        elif rule_id == "parallel":
            hits += check_parallel(doc, parallel_budget_per_1k)
        elif rule_id == "doublet":
            hits += check_doublet(doc, doublet_budget_per_1k)
        elif rule_id == "closer":
            hits += check_closer(doc, closer_budget_ratio)
        elif rule_id == "vague":
            hits += check_vague(doc, vague_min_words)
        else:
            hits += fn(doc)
    return sorted(hits, key=lambda h: h.start)


# ---------------------------------------------------------------- v0.2 checks

OPENER_RUN = 3
TEMPLATE_N = 4
TEMPLATE_MIN_REPEATS = 3
# A four-word frame needs three uses to read as filler, but a six-word
# verbatim repeat is a tell at two: nobody writes the same six words twice
# by accident in a thousand words.
TEMPLATE_LONG_N = 6
TEMPLATE_LONG_REPEATS = 2


def check_opener(doc: Document) -> list[Hit]:
    """Consecutive sentences beginning with the same word. Common openers
    ('the', 'it', 'and') are excluded: repeating those is English, not slop."""
    boring = {"the", "it", "and", "but", "a", "i", "we", "this", "that", "so"}
    hits: list[Hit] = []
    run: list = []
    for span in doc.sentences:
        word = span.words[0].lower() if span.words else ""
        if run and word == run[0].words[0].lower():
            run.append(span)
            continue
        if len(run) >= OPENER_RUN and run[0].words[0].lower() not in boring:
            hits.append(_opener_hit(run))
        run = [span] if word else []
    if len(run) >= OPENER_RUN and run[0].words[0].lower() not in boring:
        hits.append(_opener_hit(run))
    return hits


def _opener_hit(run: list) -> Hit:
    preview = run[0].text[:34] + ("..." if len(run[0].text) > 34 else "")
    return Hit("opener", run[0].start, run[-1].end,
               f"{len(run)}x '{run[0].words[0]}' from \"{preview}\"")


def check_template(doc: Document) -> list[Hit]:
    """A 4-word frame reused three or more times. Frames made entirely of
    function words are skipped; 'one of the most' is a phrase, 'in the case
    of the' is grammar."""
    tokens = [(m.group(0).lower(), m.start(), m.end())
              for m in WORD.finditer(doc.text)]
    if len(tokens) < TEMPLATE_N * TEMPLATE_MIN_REPEATS:
        return []
    hits = []
    for width, repeats in ((TEMPLATE_N, TEMPLATE_MIN_REPEATS),
                           (TEMPLATE_LONG_N, TEMPLATE_LONG_REPEATS)):
        grams: dict[tuple[str, ...], list[tuple[int, int]]] = {}
        for i in range(len(tokens) - width + 1):
            window = tokens[i:i + width]
            key = tuple(w for w, _, _ in window)
            if all(w in STOPWORDS for w in key):
                continue
            grams.setdefault(key, []).append((window[0][1], window[-1][2]))
        for key, spans in grams.items():
            if len(spans) >= repeats:
                start, end = spans[0]
                hits.append(Hit("template", start, end, " ".join(key),
                                f"{len(spans)}x, {width}-gram"))
    return _dedupe(hits)


def check_runt(
    doc: Document,
    floor: int = MIN_SENTENCE_WORDS,
    mode: str = "verbless",
    allow: frozenset[str] = frozenset(),
) -> list[Hit]:
    """Sentences below the floor.

    mode="verbless" (default) refuses only those with no finite verb, because
    the floor is a proxy for the thing that actually reads as a beat. "One
    gene." and "Matched null." are fragments; "It worked." is a sentence that
    happens to be two words long, and refusing it costs a real device.

    mode="all" is the absolute floor: nothing under it survives.
    """
    if floor < 2:
        return []
    hits = []
    for span in doc.sentences:
        if len(span) >= floor or span.text in allow:
            continue
        if mode == "verbless" and has_finite_verb(span):
            continue
        hits.append(Hit("runt", span.start, span.end, span.text,
                        f"{len(span)}w, floor {floor}"))
    return hits


def check_closer(doc: Document, budget_ratio: float = 0.25) -> list[Hit]:
    """Paragraphs ending on a short sentence, beyond the allowance.

    The allowance covers the mildest cases, so the shortest and most
    fragment-like closers are the ones reported.
    """
    found = closers(doc)
    if not found:
        return []
    paragraphs = [p for p in doc.paragraphs if len(p) >= 15]
    allowance = int(len(paragraphs) * budget_ratio)
    ranked = sorted(found, key=lambda c: (c.verbless, -c.words))
    over = ranked[allowance:]
    return [
        Hit("closer", c.span.start, c.span.end, c.span.text,
            f"{c.words}w closer{', verbless' if c.verbless else ''}, "
            f"budget {budget_ratio:.0%} of paragraphs")
        for c in sorted(over, key=lambda c: c.span.start)
    ]


def check_vague(doc: Document, min_words: int = 40) -> list[Hit]:
    return [
        Hit("vague", p.start, p.end, p.text[:70],
            f"{len(p)}w paragraph, no number, year, unit or proper noun")
        for p in vague_paragraphs(doc, min_words)
    ]


def check_parallel(doc: Document, budget_per_1k: float = 1.0) -> list[Hit]:
    """Runs of repeated syntactic skeleton beyond the document's allowance.

    Budget, not ban. Parallelism is a real device and good writers use it;
    what marks machine prose is using it constantly. The allowance is spent
    on the widest runs first, so a tetracolon costs more than a tricolon and
    the fix instruction lands on the worst offender rather than the earliest.
    """
    runs = find_parallel(doc)
    if not runs:
        return []
    # The allowance covers the MILDEST runs, so the widest ones are what gets
    # reported. Spending it on the widest first was the original ordering and
    # it was backwards: it gave the tetracolon a free pass and flagged the
    # tricolons underneath it.
    allowance = int(doc.word_count / 1000 * budget_per_1k)
    over = sorted(runs, key=lambda r: (r.arity, r.start))[allowance:]
    return [
        Hit("parallel", r.start, r.end, r.preview,
            f"{r.arity} units, {r.level}, budget {budget_per_1k}/1k")
        for r in sorted(over, key=lambda r: r.start)
    ]


CHECKS["parallel"] = check_parallel
def check_doublet(doc: Document, budget_per_1k: float = 2.0) -> list[Hit]:
    found = find_doublets(doc)
    if not found:
        return []
    allowance = int(doc.word_count / 1000 * budget_per_1k)
    over = sorted(found, key=lambda r: r.start)[allowance:]
    return [
        Hit("doublet", r.start, r.end, r.preview,
            f"{r.level}, budget {budget_per_1k}/1k")
        for r in over
    ]


CHECKS["doublet"] = check_doublet
CHECKS["closer"] = check_closer
CHECKS["vague"] = check_vague
CHECKS["runt"] = check_runt
CHECKS["fromto"] = lambda d: _regex_check(d, "fromto")
CHECKS["opener"] = check_opener
CHECKS["template"] = check_template
