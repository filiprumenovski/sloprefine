"""Checks. Each yields Hit objects carrying offsets back into the source."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .rules import LEXICON, RULES
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


def check_vocab(doc: Document, allow: frozenset[str] = frozenset()) -> list[Hit]:
    hits = []
    for m in WORD.finditer(doc.text):
        token = m.group(0).lower().replace("\u2019", "'")
        if token in LEXICON and token not in allow:
            hits.append(Hit("vocab", m.start(), m.end(), m.group(0)))
    return hits


def check_tricolon(doc: Document) -> list[Hit]:
    """Two shapes: 'a, b, and c' of bare items, and anaphora repeated 3x."""
    hits = []
    for m in re.finditer(
        r"\b([A-Za-z][\w-]*), ([A-Za-z][\w-]*),? and ([A-Za-z][\w-]*)\b", doc.text
    ):
        items = (m.group(1), m.group(2), m.group(3))
        if len(WORD.findall(m.group(0))) > 6:
            continue
        # A list of proper nouns is a fact about the world, not a cadence
        # choice: "Shan, Lee and Hao" and "Qwen, Gemma and Llama" are not
        # tricolons. Sentence-initial capitals are excluded from the test.
        mid_sentence = m.start() > 0 and doc.text[m.start() - 1] not in ".!?\n"
        if mid_sentence and all(w[:1].isupper() for w in items):
            continue
        hits.append(Hit("tricolon", m.start(), m.end(), m.group(0), "list of three"))
    for m in re.finditer(
        r"\b(\w+)\s+[\w\s-]{1,28}[,.;]\s+\1\s+[\w\s-]{1,28}[,.;]\s+\1\s+[\w\s-]{1,28}[.,;]",
        doc.text,
        re.IGNORECASE,
    ):
        hits.append(
            Hit("tricolon", m.start(), m.end(),
                re.sub(r"\s+", " ", m.group(0)), "anaphora x3")
        )
    return _dedupe(hits)


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
    "tricolon": check_tricolon,
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
        else:
            hits += fn(doc)
    return _drop_subsumed_tricolons(sorted(hits, key=lambda h: h.start))


def _drop_subsumed_tricolons(hits: list[Hit]) -> list[Hit]:
    """`tricolon` is the narrow cited case and `parallel` is the general one.
    Where they overlap the same construction is one problem, not two, so the
    narrow hit is dropped rather than counted twice."""
    wide = [h for h in hits if h.rule_id == "parallel"]
    return [
        h for h in hits
        if h.rule_id != "tricolon"
        or not any(w.start <= h.start and h.end <= w.end for w in wide)
    ]


# ---------------------------------------------------------------- v0.2 checks

OPENER_RUN = 3
TEMPLATE_N = 4
TEMPLATE_MIN_REPEATS = 3


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
    from .stylometry import STOPWORDS

    grams: dict[tuple[str, ...], list[tuple[int, int]]] = {}
    for i in range(len(tokens) - TEMPLATE_N + 1):
        window = tokens[i:i + TEMPLATE_N]
        key = tuple(w for w, _, _ in window)
        if all(w in STOPWORDS for w in key):
            continue
        grams.setdefault(key, []).append((window[0][1], window[-1][2]))

    hits = []
    for key, spans in grams.items():
        if len(spans) >= TEMPLATE_MIN_REPEATS:
            start, end = spans[0]
            hits.append(Hit("template", start, end, " ".join(key),
                            f"{len(spans)}x"))
    return hits


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
    from .cadence import _has_finite_verb

    if floor < 2:
        return []
    hits = []
    for span in doc.sentences:
        if len(span) >= floor or span.text in allow:
            continue
        if mode == "verbless" and _has_finite_verb(span):
            continue
        hits.append(Hit("runt", span.start, span.end, span.text,
                        f"{len(span)}w, floor {floor}"))
    return hits


def check_parallel(doc: Document, budget_per_1k: float = 1.0) -> list[Hit]:
    """Runs of repeated syntactic skeleton beyond the document's allowance.

    Budget, not ban. Parallelism is a real device and good writers use it;
    what marks machine prose is using it constantly. The allowance is spent
    on the widest runs first, so a tetracolon costs more than a tricolon and
    the fix instruction lands on the worst offender rather than the earliest.
    """
    from .parallel import find

    runs = find(doc)
    if not runs:
        return []
    allowance = int(doc.word_count / 1000 * budget_per_1k)
    over = sorted(runs, key=lambda r: (-r.arity, r.start))[allowance:]
    return [
        Hit("parallel", r.start, r.end, r.preview,
            f"{r.arity} units, {r.level}, budget {budget_per_1k}/1k")
        for r in sorted(over, key=lambda r: r.start)
    ]


CHECKS["parallel"] = check_parallel
CHECKS["runt"] = check_runt
CHECKS["fromto"] = lambda d: _regex_check(d, "fromto")
CHECKS["opener"] = check_opener
CHECKS["template"] = check_template
