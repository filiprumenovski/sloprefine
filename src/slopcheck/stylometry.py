"""Model-free stylometric features.

Sources
-------
[SLH26] Shan, Lee & Hao, arXiv:2608.27855 (2026). "AI Writers Have a
        Consistent Stylometric Footprint, but AI Editors Do Not."
        45k human/AI texts, 8 LLMs, 5 domains. 14 features, none needing a
        language model or corpus statistics. Findings used here:
          - AI GENERATION: high entropy + high lexical diversity, the only
            two features stable across generators and domains.
          - AI EDITING: does NOT reproduce that. Lexical density falls hard
            (d = -3.10), entropy falls slightly, diversity rises slightly.
          - Length is a confound for diversity/entropy; the authors had to
            stratify by length. We length-correct instead (MATTR, normalized
            entropy), which is why our numbers are not their numbers.
[KUM23] Kumarage et al., arXiv:2303.03697. Phraseology / punctuation /
        linguistic-diversity feature families; moving-average TTR.
[LZM23] Liang, Yuksekgonul, Mao, Wu & Zou, Patterns 2023. Detectors
        misclassify non-native English writing as AI. Any absolute threshold
        on these features inherits that bias, which is why this module ships
        no population thresholds. See voice.py.

Deliberate omission: no absolute "AI-like" cutoffs. [SLH26] reports its
features z-scored against its own corpora, and a threshold lifted from one
corpus and applied to another is exactly the failure mode in [LZM23]. These
values are reported raw, and judged only against a baseline the user builds
from their own writing.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass

from .text import Document

MATTR_WINDOW = 50

# Compact English function-word list for lexical density. Content words are
# alphabetic tokens that are not in this set [SLH26 feature 2].
STOPWORDS: frozenset[str] = frozenset(["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its", "itself", "let", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "should", "shouldn't", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "were", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "won't", "would", "wouldn't", "you", "your", "yours", "yourself", "yourselves", "will", "shall", "may", "might", "must", "can't", "i'm", "you're", "it's", "we're", "they're", "i've", "we've", "don", "t", "s"])

VOWEL_GROUP = re.compile(r"[aeiouy]+")


def syllables(word: str) -> int:
    """Heuristic syllable count for readability formulas. Approximate by
    design: Gunning Fog only needs the 3+ syllable cutoff."""
    w = word.lower().strip("'")
    if not w:
        return 0
    groups = VOWEL_GROUP.findall(w)
    n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1
    return max(1, n)


@dataclass
class Stylometry:
    words: int
    sentences: int
    lexical_diversity: float      # MATTR, length-corrected [KUM23]
    entropy_norm: float           # H / log2(types), length-corrected [SLH26]
    entropy_bits: float
    word_burstiness: float        # CV of the word-frequency vector [SLH26]
    lexical_density: float        # content words / tokens [SLH26]
    pct_punctuation: float
    pct_long_words: float
    mean_sentence_len: float
    gunning_fog: float
    contraction_rate: float       # per 1k words; formality proxy
    opener_diversity: float       # distinct sentence-opening words / sentences
    para_len_cv: float            # paragraph-length variation

    def as_dict(self) -> dict:
        return asdict(self)

    # Which direction each feature moves under generation vs editing [SLH26].
    DIRECTION: dict[str, str] = None  # populated below


Stylometry.DIRECTION = {
    "lexical_diversity": "generation: higher; editing: slightly higher",
    "entropy_norm": "generation: higher; editing: slightly lower",
    "lexical_density": "editing: sharply lower (d=-3.10)",
    "word_burstiness": "generation: moderate signal, not stable across models",
    "pct_punctuation": "generation: moderate signal",
    "pct_long_words": "domain-dependent",
    "mean_sentence_len": "domain-dependent",
    "gunning_fog": "domain-dependent",
    "contraction_rate": "not from [SLH26]; formality proxy",
    "opener_diversity": "not from [SLH26]; template repetition proxy",
    "para_len_cv": "not from [SLH26]; layout uniformity proxy",
}


def _mattr(tokens: list[str], window: int = MATTR_WINDOW) -> float:
    """Moving-average type-token ratio [KUM23]. Plain TTR falls with length,
    so a long document looks less diverse than a short one for no stylistic
    reason; MATTR holds the window fixed."""
    if len(tokens) < window:
        return len(set(tokens)) / len(tokens) if tokens else 0.0
    ratios = []
    counts: Counter = Counter(tokens[:window])
    ratios.append(len(counts) / window)
    for i in range(window, len(tokens)):
        out, inc = tokens[i - window], tokens[i]
        counts[out] -= 1
        if counts[out] == 0:
            del counts[out]
        counts[inc] += 1
        ratios.append(len(counts) / window)
    return statistics.mean(ratios)


def compute(doc: Document) -> Stylometry:
    tokens = [w.lower() for w in doc.words]
    n = len(tokens) or 1
    freqs = Counter(tokens)
    types = len(freqs) or 1

    probs = [c / n for c in freqs.values()]
    bits = -sum(p * math.log2(p) for p in probs)
    entropy_norm = bits / math.log2(types) if types > 1 else 0.0

    counts = list(freqs.values())
    mean_f = statistics.mean(counts)
    burst = (statistics.pstdev(counts) / mean_f) if mean_f else 0.0

    content = [t for t in tokens if t not in STOPWORDS]
    punct = sum(1 for ch in doc.text if ch in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

    sent_lens = [len(s) for s in doc.sentences] or [0]
    long_words = sum(1 for t in tokens if len(t) > 6)
    complex_words = sum(1 for t in tokens if syllables(t) >= 3)
    asl = statistics.mean(sent_lens) if sent_lens else 0.0
    fog = 0.4 * (asl + 100 * complex_words / n)

    openers = [s.words[0].lower() for s in doc.sentences if s.words]
    opener_div = len(set(openers)) / len(openers) if openers else 0.0

    para_lens = [len(p) for p in doc.paragraphs] or [0]
    para_cv = (
        statistics.pstdev(para_lens) / statistics.mean(para_lens)
        if len(para_lens) > 1 and statistics.mean(para_lens) else 0.0
    )

    contractions = len(re.findall(r"\b\w+['\u2019](?:t|s|re|ve|ll|d|m)\b", doc.text, re.IGNORECASE))

    return Stylometry(
        words=len(doc.words),
        sentences=len(doc.sentences),
        lexical_diversity=round(_mattr(tokens), 4),
        entropy_norm=round(entropy_norm, 4),
        entropy_bits=round(bits, 3),
        word_burstiness=round(burst, 4),
        lexical_density=round(len(content) / n, 4),
        pct_punctuation=round(punct / max(1, len(doc.text)), 4),
        pct_long_words=round(long_words / n, 4),
        mean_sentence_len=round(asl, 2),
        gunning_fog=round(fog, 2),
        contraction_rate=round(contractions / n * 1000, 2),
        opener_diversity=round(opener_div, 4),
        para_len_cv=round(para_cv, 4),
    )
