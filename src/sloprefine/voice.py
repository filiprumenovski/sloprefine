"""Voiceprints: calibrate against the author's own writing, not a population.

Why this exists. [LZM23] showed that detectors with fixed population
thresholds systematically misclassify non-native English writing as machine
output, because the features they key on (simple word choice, low perplexity)
track fluency rather than authorship. Any tool that ships a global "this looks
AI" cutoff inherits that bias, and inherits it hardest for exactly the writers
who can least afford it.

The fix used here is to make the baseline personal. Feed sloprefine a folder of
things you wrote before, get a voiceprint: per-feature mean and standard
deviation across those documents. A new draft is then scored in z-units
against your own distribution. "Your lexical density is 2.4 sigma below your
own baseline" is a claim about a change in your writing. "Your lexical density
is 0.48" is a claim about you, and it is the kind of claim that gets people
accused of things.

Robust statistics are used throughout (median, MAD) because a voiceprint built
from a handful of documents should not be wrecked by one outlier.

  [LZM23] Liang et al., Patterns 2023. GPT detectors are biased against
          non-native English writers.
  [SLH26] Shan, Lee & Hao, arXiv:2608.27855 (2026).
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from . import stylometry
from .text import Document

MIN_DOCS = 3
MIN_WORDS = 150
MAD_TO_SIGMA = 1.4826  # scale factor making MAD comparable to a stdev


@dataclass
class Voiceprint:
    """Per-feature robust location and scale from an author's own corpus."""

    center: dict[str, float]
    scale: dict[str, float]
    n_docs: int
    n_words: int
    sources: list[str] = field(default_factory=list)
    version: int = 1

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "n_docs": self.n_docs,
                "n_words": self.n_words,
                "sources": self.sources,
                "center": self.center,
                "scale": self.scale,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> Voiceprint:
        data = json.loads(text)
        if data.get("version") != 1:
            raise ValueError(f"unsupported voiceprint version: {data.get('version')}")
        for key in ("center", "scale", "n_docs", "n_words"):
            if key not in data:
                raise ValueError(f"voiceprint missing '{key}'")
        return cls(
            center=data["center"],
            scale=data["scale"],
            n_docs=data["n_docs"],
            n_words=data["n_words"],
            sources=data.get("sources", []),
        )

    @classmethod
    def load(cls, path: str | Path) -> Voiceprint:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def z(self, feature: str, value: float) -> float | None:
        """Deviation in robust sigma units. None when the author's own
        spread on this feature is degenerate, in which case no honest
        statement about deviation can be made."""
        scale = self.scale.get(feature, 0.0)
        if feature not in self.center or scale <= 1e-9:
            return None
        return (value - self.center[feature]) / scale

    def compare(self, style: stylometry.Stylometry) -> dict[str, float]:
        out = {}
        for feature, value in style.as_dict().items():
            if not isinstance(value, (int, float)):
                continue
            score = self.z(feature, float(value))
            if score is not None:
                out[feature] = round(score, 2)
        return out


FEATURES_TO_TRACK = (
    "lexical_diversity", "entropy_norm", "lexical_density", "word_burstiness",
    "pct_punctuation", "pct_long_words", "mean_sentence_len", "gunning_fog",
    "contraction_rate", "opener_diversity",
)


def build(documents: list[tuple[str, str]]) -> Voiceprint:
    """Build a voiceprint from (path, text) pairs.

    Short documents are dropped rather than downweighted: the length-corrected
    features still get noisy under ~150 words, and a noisy baseline is worse
    than a smaller one.
    """
    usable = []
    for path, text in documents:
        doc = Document(text, path)
        if doc.word_count < MIN_WORDS:
            continue
        usable.append((path, doc))
    if len(usable) < MIN_DOCS:
        raise ValueError(
            f"need at least {MIN_DOCS} documents of {MIN_WORDS}+ words to build "
            f"a voiceprint; got {len(usable)}. A baseline from fewer documents "
            "describes those documents, not your voice."
        )

    samples: dict[str, list[float]] = {f: [] for f in FEATURES_TO_TRACK}
    total_words = 0
    for _, doc in usable:
        style = stylometry.compute(doc).as_dict()
        total_words += doc.word_count
        for feature in FEATURES_TO_TRACK:
            samples[feature].append(float(style[feature]))

    center, scale = {}, {}
    for feature, values in samples.items():
        med = statistics.median(values)
        mad = statistics.median([abs(v - med) for v in values])
        center[feature] = round(med, 6)
        # Fall back to stdev when MAD collapses (common with few documents).
        spread = mad * MAD_TO_SIGMA
        if spread <= 1e-9 and len(values) > 1:
            spread = statistics.pstdev(values)
        scale[feature] = round(spread, 6)

    return Voiceprint(
        center=center,
        scale=scale,
        n_docs=len(usable),
        n_words=total_words,
        sources=sorted(path for path, _ in usable),
    )


def interpret(deviations: dict[str, float], threshold: float = 2.0) -> list[str]:
    """Turn z-scores into statements a writer can act on.

    The direction annotations come from [SLH26]: generation raises entropy and
    lexical diversity together; editing pulls lexical density down and entropy
    slightly down. A lone excursion means little. The joint patterns are what
    the paper found stable, so those are what get named.
    """
    notes: list[str] = []
    ld = deviations.get("lexical_density")
    div = deviations.get("lexical_diversity")
    ent = deviations.get("entropy_norm")

    if ld is not None and ld <= -threshold:
        note = (
            f"lexical density {ld:+.1f}sigma below your baseline: more function "
            "words per content word than you normally write"
        )
        if ent is not None and ent < 0:
            note += ". [SLH26] reports this pair as the AI-editing signature"
        notes.append(note)

    if (div is not None and div >= threshold) and (ent is not None and ent >= threshold):
        notes.append(
            f"lexical diversity {div:+.1f}sigma and entropy {ent:+.1f}sigma above "
            "baseline together: the joint pattern [SLH26] found stable for "
            "AI-generated rather than AI-edited text"
        )

    for feature, z in sorted(deviations.items(), key=lambda kv: -abs(kv[1])):
        if abs(z) >= threshold and feature not in {
            "lexical_density", "lexical_diversity", "entropy_norm"
        }:
            notes.append(f"{feature} {z:+.1f}sigma from your baseline")
    return notes
