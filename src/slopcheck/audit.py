"""Contamination audit: does this heuristic still discriminate?

The problem this solves
-----------------------
Every marker in this repository is dated. The rule of three was good advice
for a century, got prescribed everywhere, landed in the training data, and is
now a detection marker. The same process is running right now on everything
currently being written about "how to sound human": short sentences, admitting
uncertainty, concrete detail, the em-dash ban. Publish a fix widely enough and
models absorb it, at which point the fix becomes the next tell.

A tool built on a fixed marker list therefore decays, and it decays silently.
The only defense is to make the decay measurable. Point this at a corpus of
machine text and a corpus of human text and it reports, per rule, how much
more often the pattern appears in the machine half. That number is the rule's
remaining discriminative power.

  enrichment >> 1   the marker still works
  enrichment ~ 1    the marker is dead; the pattern is now equally common
                    in both, either because models stopped or because humans
                    started
  enrichment < 1    inverted: the pattern is now MORE common in human text,
                    which is what happens after a fix gets widely adopted by
                    models and abandoned by writers

Run it against your own corpora, on your own models, and rerun it when the
models change. An enrichment table with a date on it is worth more than a
wordlist without one.

Stylometric and reader features get the same treatment via Cohen's d, which is
how [SLH26] and [MGF25] report their own effects, so the numbers are directly
comparable to the published ones.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from . import reader as reader_mod
from . import stylometry
from .report import Config, analyze
from .rules import RULES
from .text import Document

TEXT_SUFFIXES = (".txt", ".md", ".markdown", ".rst", ".text")
MIN_CORPUS = 5
DEAD_BAND = (0.8, 1.25)  # enrichment inside this band: no discrimination left


@dataclass
class RuleAudit:
    rule: str
    ai_per_1k: float
    human_per_1k: float
    enrichment: float
    verdict: str

    def render(self) -> str:
        return (f"{self.rule:<13} ai {self.ai_per_1k:6.2f}  human "
                f"{self.human_per_1k:6.2f}  x{self.enrichment:5.2f}  "
                f"{self.verdict}")


@dataclass
class FeatureAudit:
    feature: str
    ai_mean: float
    human_mean: float
    cohens_d: float

    def render(self) -> str:
        arrow = "higher" if self.cohens_d > 0 else "lower"
        return (f"{self.feature:<22} ai {self.ai_mean:8.3f}  human "
                f"{self.human_mean:8.3f}  d={self.cohens_d:+6.2f} "
                f"(machine {arrow})")


@dataclass
class Audit:
    n_ai: int
    n_human: int
    rules: list[RuleAudit] = field(default_factory=list)
    features: list[FeatureAudit] = field(default_factory=list)

    def dead(self) -> list[RuleAudit]:
        return [r for r in self.rules if r.verdict in {"dead", "inverted"}]

    def as_dict(self) -> dict:
        return {
            "n_ai": self.n_ai,
            "n_human": self.n_human,
            "rules": [vars(r) for r in self.rules],
            "features": [vars(f) for f in self.features],
        }

    def render(self) -> str:
        lines = [
            f"corpora: {self.n_ai} machine, {self.n_human} human",
            "",
            "rules (enrichment = machine rate / human rate)",
        ]
        lines += ["  " + r.render()
                  for r in sorted(self.rules, key=lambda r: -r.enrichment)]
        if self.features:
            lines += ["", "features (Cohen's d, machine vs human)"]
            lines += ["  " + f.render()
                      for f in sorted(self.features, key=lambda f: -abs(f.cohens_d))]
        dead = self.dead()
        if dead:
            lines += ["", "no longer discriminating on these corpora:"]
            lines += [f"  {r.rule} (x{r.enrichment:.2f})" for r in dead]
        return "\n".join(lines)


def load_corpus(path: str | Path) -> list[tuple[str, str]]:
    p = Path(path)
    files = (
        sorted(f for f in p.rglob("*")
               if f.is_file() and f.suffix.lower() in TEXT_SUFFIXES)
        if p.is_dir() else [p]
    )
    return [(str(f), f.read_text(encoding="utf-8", errors="replace"))
            for f in files]


def _rates(corpus: list[tuple[str, str]], config: Config) -> dict[str, list[float]]:
    per_doc: dict[str, list[float]] = {rid: [] for rid in RULES}
    for path, text in corpus:
        result = analyze(path, text, config)
        words = max(1, result.metrics.words if result.metrics else 1)
        counts = result.counts()
        for rid in RULES:
            per_doc[rid].append(counts[rid] / words * 1000)
    return per_doc


def _features(corpus: list[tuple[str, str]], audience: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for path, text in corpus:
        doc = Document(text, path)
        values = stylometry.compute(doc).as_dict()
        values.update(reader_mod.compute(doc, audience).as_dict())
        for key, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out.setdefault(key, []).append(float(value))
    return out


def _cohens_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va, vb = statistics.pvariance(a), statistics.pvariance(b)
    pooled = math.sqrt((va + vb) / 2)
    if pooled < 1e-12:
        return 0.0
    return (statistics.mean(a) - statistics.mean(b)) / pooled


def audit(
    ai_corpus: list[tuple[str, str]],
    human_corpus: list[tuple[str, str]],
    config: Config | None = None,
    audience: str = "expert",
) -> Audit:
    if len(ai_corpus) < MIN_CORPUS or len(human_corpus) < MIN_CORPUS:
        raise ValueError(
            f"need at least {MIN_CORPUS} documents per corpus; got "
            f"{len(ai_corpus)} machine and {len(human_corpus)} human. "
            "An enrichment ratio from fewer is noise with a decimal point."
        )
    config = config or Config()
    ai_rates, human_rates = _rates(ai_corpus, config), _rates(human_corpus, config)

    rule_audits = []
    for rid in RULES:
        ai_mean = statistics.mean(ai_rates[rid])
        human_mean = statistics.mean(human_rates[rid])
        if ai_mean == 0 and human_mean == 0:
            enrichment, verdict = 1.0, "unobserved"
        elif human_mean == 0:
            enrichment, verdict = float("inf"), "machine-only"
        else:
            enrichment = ai_mean / human_mean
            if enrichment < DEAD_BAND[0]:
                verdict = "inverted"
            elif enrichment <= DEAD_BAND[1]:
                verdict = "dead"
            else:
                verdict = "live"
        rule_audits.append(RuleAudit(rid, round(ai_mean, 3), round(human_mean, 3),
                                     round(enrichment, 3), verdict))

    ai_feats, human_feats = _features(ai_corpus, audience), _features(human_corpus, audience)
    feature_audits = [
        FeatureAudit(
            feature=key,
            ai_mean=round(statistics.mean(ai_feats[key]), 4),
            human_mean=round(statistics.mean(human_feats[key]), 4),
            cohens_d=round(_cohens_d(ai_feats[key], human_feats[key]), 3),
        )
        for key in sorted(set(ai_feats) & set(human_feats))
    ]
    return Audit(len(ai_corpus), len(human_corpus), rule_audits, feature_audits)
