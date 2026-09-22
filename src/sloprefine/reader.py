"""Reader-preference features, with directions measured after 2022.

Why this module is separate from rules.py
-----------------------------------------
Every heuristic in rules.py is negative: a pattern to remove. That is only
half a writing tool, and the missing half cannot be filled from a rhetoric
textbook, because of a contamination problem.

Any device that was widely *prescribed* before 2022 is in the advice corpus,
therefore in the training data, therefore in the model's default register. The
rule of three is the clearest case. It is genuinely effective, taught
everywhere for a century, and now so heavily overproduced that [CL] and [PALV]
list it as a detection marker. Prescription made it a target; optimization
made it a tell. The same logic threatens parallelism, alliteration,
readability-formula optimization, and "vary your sentence length."

So a positive feature earns its place here only if its direction was measured
after 2022 by comparing human and machine text against actual reader
judgments. Not "good writers do this." Rather: readers preferred texts with
more of this, and machine text has less of it.

Sources
-------
[MGF25] Marco, Gonzalo & Fresno, arXiv:2506.03310 (2025). "The Reader is the
        Metric." 1,471 stories, 101 annotators (critics, students, lay
        readers), 17 reference-less features, per-reader preference models.
        Two findings drive this module:
          (a) Reader preferences cluster into two profiles that want
              different things. Lay readers weight readability, sentence
              length, syntactic depth and lexical diversity. Experts weight
              sentiment dynamics, sentence rhythm, rhetorical variety and
              thematic entropy. There is no single "good writing" target, so
              this module takes an audience.
          (b) Their Table 3 gives per-corpus feature means for human and
              machine text. Four directions replicate across both
              expert-annotated corpora (PRONVSPROMPT, TTCW):
                mean sentiment      human lower   (-0.12, +0.05) vs AI (+0.74, +0.78)
                sentiment variance  human higher  (0.91, 0.91) vs AI (0.43, 0.36)
                sentence rhythm     human higher  (11.0, 24.5) vs AI (10.3, 16.9)
                local coherence     human lower   (0.31, 0.32) vs AI (0.41, 0.48)
[CGD26] Chakrabarty, Ginsburg & Dhillon, arXiv:2510.13939. 10,920 pairwise
        judgments. In-context prompting was disfavored by MFA readers
        (quality OR=0.13) while general readers favored it (OR=1.82).
        Fine-tuning on one author's complete works reversed it: MFA
        stylistic-fidelity OR=8.16. Matching a specific author beats
        generic quality. This is the empirical case for voice.py.
[LU24]  Lu et al., arXiv:2410.04265. Creativity Index: professional authors
        score 66.2% higher than LLMs, and RLHF reduces it ~30.1%.
        Unpredictability is trained out, not absent by accident.
[HM25]  Haverals & Martin, arXiv:2510.08831. Attribution bias: the same text
        is rated differently once labeled as machine-written. Preference
        measured under labels is not preference.

Deliberate limits
-----------------
Three [MGF25] features need models this package will not require: local
coherence via sentence embeddings, thematic entropy via LDA, and rhetorical
variety via a large model. Each is approximated here with a lexical proxy, and
each proxy is labeled as one. A proxy with a known name beats a missing
feature with a good excuse, but it is not the measurement.
"""

from __future__ import annotations

import itertools
import re
import statistics
from dataclasses import asdict, dataclass

from .stylometry import STOPWORDS, syllables
from .text import Document

# Compact valence lexicon for the sentiment-dynamics proxy. [MGF25] used
# fine-tuned RoBERTa models; this is a crude stand-in and is labeled as one in
# the output. It exists because mean sentiment and sentiment variance are the
# most consistently replicated directions in their Table 3.
POSITIVE = frozenset(["good", "great", "best", "better", "wonderful", "excellent", "beautiful", "happy", "joy", "joyful", "love", "loved", "loving", "warm", "bright", "hope", "hopeful", "gentle", "kind", "calm", "peace", "peaceful", "smile", "smiled", "laugh", "laughed", "light", "delight", "delighted", "pleasure", "proud", "pride", "success", "succeed", "win", "won", "triumph", "gift", "lucky", "safe", "comfort", "comforted", "tender", "sweet", "rich", "clear", "strong", "brilliant", "perfect", "amazing", "glad", "grace", "graceful", "free", "freedom", "generous", "thrill", "thrilled", "charm", "charming", "bloom", "flourish", "gleam", "glow", "radiant"])

NEGATIVE = frozenset(["bad", "worse", "worst", "awful", "terrible", "ugly", "sad", "sorrow", "grief", "cry", "cried", "tears", "pain", "painful", "hurt", "wound", "wounded", "dark", "darkness", "fear", "afraid", "scared", "anger", "angry", "rage", "hate", "hated", "hatred", "cruel", "cold", "bitter", "harsh", "lonely", "alone", "loss", "lost", "fail", "failed", "failure", "broken", "break", "shattered", "ruin", "ruined", "empty", "hollow", "dead", "death", "dying", "kill", "killed", "violence", "sick", "illness", "ache", "aching", "dread", "despair", "numb", "regret", "shame", "ashamed", "guilt", "guilty", "betray", "betrayed", "abandon", "abandoned"])

# Rhetorical devices for the variety proxy. Detecting the PRESENCE of several
# distinct devices, not the repetition of one. [MGF25] found expert readers
# weight variety; [CL] found machine text overuses one device (the triad).
DEVICE_PATTERNS: dict[str, str] = {
    "simile": r"\b(?:like|as)\s+(?:a|an|the)\s+\w+",
    "question": r"\?",
    "direct_address": r"\byou\b",
    "imperative": r"(?:^|\.\s+)[A-Z][a-z]+(?:\s+\w+){0,3}[.!]",
    "repetition": r"\b(\w{4,})\b[^.]{0,60}\b\1\b",
    "dash_aside": r"\([^)]{3,60}\)",
    "negation_frame": r"\bnot\s+\w+,?\s+but\b",
    "conditional": r"\bif\b[^.]{5,80}\bthen\b|\bwere\s+it\s+not\b",
    "list": r"\b\w+,\s+\w+,\s+\w+\b",
    "quotation": r"[\"\u201c][^\"\u201d]{8,}[\"\u201d]",
}

AUDIENCES = ("expert", "general")

# [MGF25] Fig 3 / Sec 5.3: which features each reader cluster weights.
AUDIENCE_FEATURES: dict[str, tuple[str, ...]] = {
    "expert": ("sentence_rhythm", "sentiment_variance", "mean_sentiment",
               "device_variety", "adjacent_overlap", "topic_spread"),
    "general": ("mean_sentence_len", "subordination", "lexical_diversity",
                "smog"),
}


# Below this many matched valence tokens the lexicon proxy is measuring noise.
# Technical prose can run hundreds of words with no lexicon hit at all, and a
# variance computed from three matches is a number with no evidence under it.
MIN_VALENCE_TOKENS = 15


@dataclass
class ReaderSignals:
    audience: str
    sentence_rhythm: float       # stdev of sentence length [MGF25]
    adjacent_overlap: float      # local-coherence proxy [MGF25]
    mean_sentiment: float | None       # lexicon proxy [MGF25]; None if sparse
    sentiment_variance: float | None   # lexicon proxy [MGF25]; None if sparse
    valence_tokens: int          # evidence behind the sentiment proxy
    device_variety: int          # distinct devices present [MGF25]
    device_concentration: float  # share held by the most-used device
    topic_spread: float          # thematic-entropy proxy [MGF25]
    subordination: float         # clauses per sentence [MGF25, lay cluster]
    mean_sentence_len: float
    lexical_diversity: float
    smog: float

    def as_dict(self) -> dict:
        return asdict(self)


def _chunks(doc: Document, n: int = 8) -> list[list[str]]:
    words = [w.lower() for w in doc.words]
    if len(words) < n * 4:
        return [words] if words else []
    size = len(words) // n
    return [words[i * size:(i + 1) * size] for i in range(n)]


def _valence(chunk: list[str]) -> float:
    pos = sum(1 for w in chunk if w in POSITIVE)
    neg = sum(1 for w in chunk if w in NEGATIVE)
    total = pos + neg
    return (pos - neg) / total if total else 0.0


def compute(doc: Document, audience: str = "expert") -> ReaderSignals:
    if audience not in AUDIENCES:
        raise ValueError(f"audience must be one of {AUDIENCES}")

    lens = [len(s) for s in doc.sentences] or [0]
    rhythm = statistics.pstdev(lens) if len(lens) > 1 else 0.0

    # Local-coherence proxy: content-word overlap between adjacent sentences.
    # [MGF25] used sentence-embedding cosine; lexical overlap is the classic
    # cohesion stand-in and moves in the same direction.
    overlaps = []
    contents = [
        {w.lower() for w in s.words if w.lower() not in STOPWORDS}
        for s in doc.sentences
    ]
    for a, b in itertools.pairwise(contents):
        union = a | b
        if union:
            overlaps.append(len(a & b) / len(union))
    overlap = statistics.mean(overlaps) if overlaps else 0.0

    chunks = _chunks(doc)
    matched = sum(1 for w in (t.lower() for t in doc.words)
                  if w in POSITIVE or w in NEGATIVE)
    if matched >= MIN_VALENCE_TOKENS and len(chunks) > 1:
        valences = [_valence(c) for c in chunks]
        mean_sent: float | None = statistics.mean(valences)
        var_sent: float | None = statistics.pvariance(valences)
    else:
        mean_sent = var_sent = None

    counts = {
        name: len(re.findall(pattern, doc.text, re.IGNORECASE | re.MULTILINE))
        for name, pattern in DEVICE_PATTERNS.items()
    }
    present = {k: v for k, v in counts.items() if v}
    total_devices = sum(present.values())
    concentration = (max(present.values()) / total_devices) if total_devices else 0.0

    # Thematic-spread proxy: how evenly content words distribute over the
    # document's sections. [MGF25] used LDA topic entropy.
    spread = 0.0
    if len(chunks) > 1:
        vocab = [
            {w for w in c if w not in STOPWORDS and len(w) > 3} for c in chunks
        ]
        shared = set.intersection(*vocab) if all(vocab) else set()
        union = set().union(*vocab) if vocab else set()
        spread = 1.0 - (len(shared) / len(union)) if union else 0.0

    subordinators = len(re.findall(
        r"\b(?:which|that|because|although|though|while|whereas|since|unless|"
        r"until|whether|after|before|when|where|if)\b", doc.text, re.IGNORECASE))
    subordination = subordinators / max(1, len(doc.sentences))

    tokens = [w.lower() for w in doc.words]
    poly = sum(1 for t in tokens if syllables(t) >= 3)
    smog = 1.0430 * ((poly * 30 / max(1, len(doc.sentences))) ** 0.5) + 3.1291

    types = len(set(tokens))
    diversity = types / len(tokens) if tokens else 0.0

    return ReaderSignals(
        audience=audience,
        sentence_rhythm=round(rhythm, 2),
        adjacent_overlap=round(overlap, 4),
        mean_sentiment=None if mean_sent is None else round(mean_sent, 3),
        sentiment_variance=None if var_sent is None else round(var_sent, 4),
        valence_tokens=matched,
        device_variety=len(present),
        device_concentration=round(concentration, 3),
        topic_spread=round(spread, 4),
        subordination=round(subordination, 2),
        mean_sentence_len=round(statistics.mean(lens), 2),
        lexical_diversity=round(diversity, 4),
        smog=round(smog, 2),
    )


# Thresholds below are midpoints between the human and machine means in
# [MGF25] Table 3, averaged over the two expert-annotated corpora. They are
# corpus-specific and genre-specific. They are reported as "this is the side
# machine text sat on", never as a verdict.
_TABLE3 = {
    "mean_sentiment": {"human": -0.03, "ai": 0.76, "prefer": "lower"},
    "sentiment_variance": {"human": 0.91, "ai": 0.39, "prefer": "higher"},
    "adjacent_overlap": {"human": 0.31, "ai": 0.44, "prefer": "lower"},
}


def notes(signals: ReaderSignals) -> list[str]:
    """Reader-facing observations, in the direction the evidence supports."""
    out: list[str] = []
    if signals.audience == "expert":
        if signals.mean_sentiment is None:
            out.append(
                f"sentiment proxy unavailable: only {signals.valence_tokens} "
                "valence tokens matched. The strongest published direction "
                "[MGF25] cannot be checked on this text without a sentiment "
                "model; the rest of this section still applies"
            )
        elif signals.mean_sentiment > 0.35:
            out.append(
                f"mean sentiment {signals.mean_sentiment:+.2f} (proxy): machine "
                "text ran markedly more positive than human text in both "
                "expert-annotated corpora of [MGF25]. Positivity is the cheap "
                "register; let something be bad"
            )
        if (signals.sentiment_variance is not None
                and signals.sentiment_variance < 0.05 and signals.device_variety):
            out.append(
                f"sentiment variance {signals.sentiment_variance:.3f} (proxy): "
                "flat affect across the document. Expert readers weighted "
                "sentiment dynamics heavily and human texts varied roughly "
                "twice as much [MGF25]"
            )
        if signals.adjacent_overlap > 0.22:
            out.append(
                f"adjacent-sentence overlap {signals.adjacent_overlap:.2f} "
                "(local-coherence proxy): consecutive sentences restate each "
                "other's vocabulary. Machine text was measurably smoother here "
                "and expert-preferred human text was less so [MGF25]"
            )
        if signals.device_variety <= 2 and signals.device_concentration > 0.6:
            out.append(
                f"one device carries {signals.device_concentration:.0%} of the "
                f"rhetorical work across {signals.device_variety} device type(s). "
                "Expert readers weighted rhetorical VARIETY, not any single "
                "device [MGF25]"
            )
        if signals.topic_spread and signals.topic_spread < 0.55:
            out.append(
                f"topic spread {signals.topic_spread:.2f} (thematic-entropy "
                "proxy): the same vocabulary recurs in every section"
            )
    else:
        if signals.smog > 14:
            out.append(
                f"SMOG {signals.smog:.0f}: general readers weighted readability "
                "and preferred accessible framing [MGF25]"
            )
        if signals.lexical_diversity < 0.40:
            out.append(
                f"lexical diversity {signals.lexical_diversity:.2f}: the lay "
                "reader cluster weighted lexical richness [MGF25]"
            )
        if signals.subordination < 0.8:
            out.append(
                f"subordination {signals.subordination:.1f} clauses/sentence: "
                "the lay cluster weighted syntactic depth and preferred texts "
                "with more of it [MGF25]"
            )
    return out


def contract_lines(audience: str) -> list[str]:
    """Generation-time constraints for an audience. Directional, not numeric:
    a model handed a target number optimizes the number."""
    if audience == "expert":
        return [
            (
                "Affect: do not hold a uniform positive tone. Machine text ran "
                "markedly more positive and roughly half as variable as human "
                "text in expert-rated corpora [MGF25]. Let the register move, "
                "and let something in the piece be bad."
            ),
            (
                "Transitions: do not have each sentence restate the previous "
                "sentence's vocabulary. Expert-preferred human text scored "
                "LOWER on local coherence than machine text [MGF25]."
            ),
            (
                "Devices: use several different rhetorical moves once each "
                "rather than one move repeatedly. Experts weighted variety; "
                "one device repeated is the marker."
            ),
            (
                "Unpredictability: professional writers score 66% higher than "
                "LLMs on linguistic originality, and alignment training cuts "
                "that by about 30% [LU24]. Assume your first phrasing is the "
                "predictable one."
            ),
        ]
    return [
        (
            "Accessibility: this audience weighted readability. Keep the "
            "sentence structure navigable and the vocabulary concrete [MGF25]."
        ),
        (
            "Richness inside accessibility: the lay reader cluster valued "
            "lexical variety and syntactic depth presented in an accessible "
            "frame, not simplification [MGF25]."
        ),
    ]
