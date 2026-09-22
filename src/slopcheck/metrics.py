"""Statistical metrics. These are scored, not counted as hits."""

from __future__ import annotations

import itertools
import re
import statistics
from dataclasses import asdict, dataclass

from .rules import FOOTPRINT_PATTERNS
from .text import Document

# Reference bands. Human nonfiction prose typically sits above 0.55; text
# generated in one pass clusters low because sentences land in the same
# 15-25 word range. [PALV][GK][eLearningIndustry]
CV_HUMAN_FLOOR = 0.55
CV_FLAT_CEILING = 0.45
FLAT_RUN_LIMIT = 6  # consecutive sentences within 4 words of each other


@dataclass
class Metrics:
    sentences: int
    words: int
    mean_len: float
    stdev_len: float
    cv: float
    pct_short: float
    pct_long: float
    longest_flat_run: int
    footprint: int
    footprint_per_1k: float

    def warnings(self) -> list[str]:
        out = []
        if self.cv < CV_FLAT_CEILING:
            out.append(
                f"sentence length is flat (CV={self.cv:.2f}, human prose "
                f">={CV_HUMAN_FLOOR}); mix in short and long sentences"
            )
        if self.longest_flat_run > FLAT_RUN_LIMIT:
            out.append(
                f"{self.longest_flat_run} consecutive sentences of near-identical "
                "length; break the metronome"
            )
        if self.words > 400 and self.footprint == 0:
            out.append(
                "no first-person epistemic markers: nothing the author noticed, "
                "expected, doubted or got wrong"
            )
        return out

    def as_dict(self) -> dict:
        return asdict(self)


def compute(doc: Document) -> Metrics:
    lens = [len(s) for s in doc.sentences] or [0]
    mean = statistics.mean(lens)
    sd = statistics.pstdev(lens) if len(lens) > 1 else 0.0
    cv = sd / mean if mean else 0.0

    worst = run = 1
    for a, b in itertools.pairwise(lens):
        run = run + 1 if abs(a - b) <= 4 else 1
        worst = max(worst, run)

    footprint = sum(
        len(re.findall(p, doc.text, re.IGNORECASE)) for p in FOOTPRINT_PATTERNS
    )
    words = doc.word_count
    return Metrics(
        sentences=len(doc.sentences),
        words=words,
        mean_len=round(mean, 1),
        stdev_len=round(sd, 1),
        cv=round(cv, 3),
        pct_short=round(sum(1 for n in lens if n <= 7) / len(lens), 3),
        pct_long=round(sum(1 for n in lens if n >= 25) / len(lens), 3),
        longest_flat_run=worst,
        footprint=footprint,
        footprint_per_1k=round(footprint / words * 1000, 2) if words else 0.0,
    )
