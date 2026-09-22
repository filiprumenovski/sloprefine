"""Weighting and ranking: which problems matter, and where they are.

Two things the flat hit count cannot express.

**Not all hits are equal.** `total` counted a stock transition and a paragraph
with nothing specific in it as one each. They are not one each. Severity
weights turn the count into a score, so a document with three high-severity
hits ranks worse than one with five low-severity ones, which is the ordering a
reviser wants.

**A document is not uniformly bad.** Hits cluster. A revision loop that is
handed nineteen scattered instructions does worse than one told which
paragraph to rewrite, because rewriting a paragraph fixes its hits together
and usually fixes them better than patching each in place. `worst_paragraphs`
returns that ordering.

The weights are a judgement, not a measurement. They encode that cadence and
structure outrank vocabulary, which is what the one human reaction available
on this repository's origin document actually reacted to. `slopcheck audit`
against a real corpus pair is how they get replaced with evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from .checks import Hit
from .rules import RULES
from .text import Document

WEIGHTS = {"high": 3.0, "medium": 2.0, "low": 1.0}


def weight(hit: Hit) -> float:
    return WEIGHTS.get(RULES[hit.rule_id].severity, 1.0)


def score(hits: list[Hit], words: int) -> float:
    """Severity-weighted hits per 1000 words."""
    if not words:
        return 0.0
    return round(sum(weight(h) for h in hits) / words * 1000, 2)


@dataclass
class ParagraphScore:
    index: int
    line: int
    words: int
    hits: int
    weighted: float
    rules: tuple[str, ...]
    preview: str

    def render(self) -> str:
        return (f"P{self.index} (L{self.line}, {self.words}w): "
                f"{self.hits} hits, weight {self.weighted:.0f} "
                f"[{', '.join(self.rules)}]")


def worst_paragraphs(doc: Document, hits: list[Hit], limit: int = 3
                     ) -> list[ParagraphScore]:
    """Paragraphs ranked by weighted hit density.

    Density, not count: a 200-word paragraph with four hits is in better shape
    than a 40-word paragraph with three, and ranking by raw count would send a
    reviser to the long one.
    """
    ordered = sorted(hits, key=lambda h: h.start)
    scored: list[ParagraphScore] = []
    cursor = 0
    for i, para in enumerate(doc.paragraphs, start=1):
        while cursor < len(ordered) and ordered[cursor].start < para.start:
            cursor += 1
        end = cursor
        while end < len(ordered) and ordered[end].start < para.end:
            end += 1
        inside = ordered[cursor:end]
        cursor = end
        if not inside:
            continue
        words = len(para) or 1
        scored.append(ParagraphScore(
            index=i,
            line=doc.line(para.start),
            words=words,
            hits=len(inside),
            weighted=round(sum(weight(h) for h in inside) / words * 1000, 1),
            rules=tuple(dict.fromkeys(h.rule_id for h in inside)),
            preview=para.text[:60].replace("\n", " "),
        ))
    return sorted(scored, key=lambda p: -p.weighted)[:limit]
