"""Choppiness: the primary cadence metric.

Why this is its own module and not just the `fragments` rule
-----------------------------------------------------------
The `fragments` rule fires on runs of three or more short sentences. That is a
local pattern detector, and it misses the global case: a document whose every
sentence is eight words long never produces a run-of-three flag against its own
neighbours, but it is the most choppy prose in the corpus. The one human
judgement available on this repository's origin document was a reaction to
exactly that global property.

Worse, the metrics that were supposed to provide independent evidence cannot.
Lexical density rises when prose is chopped into fragments, because fragments
carry almost no function words. Sentence-length burstiness also rises, because
alternating very short sentences with normal ones is high variance. So both
metrics *reward* the TED cadence by construction. A revision that reintroduced
fragments therefore scored better on three of four layers, and the fourth was
outvoted. That is not three independent confirmations; it is one artifact
counted three times.

So: choppiness is measured directly, it is reported first, and `drift` treats
density and burstiness as collinear with it rather than as independent.

Components
----------
short_share    fraction of WORDS living in sentences of <= 7 words. Word-
               weighted, not sentence-weighted, because ten fragments and one
               long sentence is a choppy document even though the sentence
               count says it is half normal prose.
verbless_share fraction of sentences with no detectable finite verb. This is
               the distinction that matters for delivery: "It worked." is a
               short sentence, "One gene." is a fragment. Heuristic, no POS
               tagger; see has_finite_verb. Two known misses: a bare past participle
               ("Matched null.") passes the verb test, and a third-person
               singular verb outside FINITE_FORMS fails it. Use
               --strict-runts when the floor must be absolute.
run_mass       fraction of words inside runs of >= 3 consecutive short
               sentences, which is what the rule layer catches.

Calibration honesty
-------------------
The threshold sits at 0.17, the midpoint between the only two labelled points
available: a conference talk a domain expert called machine-sounding (0.25)
and a revision written to fix that reaction (0.09). That is ONE label. The
first threshold tried here was 0.40, picked by intuition, and it passed the
document the expert had already rejected, which is the whole argument against
picking thresholds by intuition. Replace it with `slopcheck audit` numbers
from a corpus you trust.

Note the word-weighting matters. The rejected talk had 57% of its SENTENCES
under 8 words but only 30% of its WORDS in them. The sentence-count figure
overstates: a document can be half short sentences and still spend most of its
runtime in long ones. Word share is what a listener experiences.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass

from .text import Document, Span

SHORT_WORDS = 7
RUN_LENGTH = 3

# Weights: short_share carries the most because it is the property the one
# available human judgement reacted to. verbless_share is weighted next
# because a verbless clause is what makes a short sentence land as a beat.
WEIGHTS = {"short_share": 0.45, "verbless_share": 0.35, "run_mass": 0.20}

CHOPPY_THRESHOLD = 0.17

# Finite-verb heuristic. Auxiliaries, copulas and modals cover most clauses;
# the suffix test catches lexical verbs. It over-fires on nouns ending in -s
# and under-fires on irregular pasts, which is acceptable for a share computed
# over a whole document but would not be for a per-sentence verdict.
FINITE_FORMS = frozenset(["is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "done", "have", "has", "had", "having", "can", "could", "will", "would", "shall", "should", "may", "might", "must", "get", "gets", "got", "go", "goes", "went", "come", "comes", "came", "make", "makes", "made", "take", "takes", "took", "see", "sees", "saw", "know", "knows", "knew", "think", "thinks", "thought", "say", "says", "said", "work", "works", "worked", "run", "runs", "ran", "sit", "sits", "sat", "find", "finds", "found", "give", "gives", "gave"])

FINITE_FORMS = FINITE_FORMS | frozenset([
    "fit", "fits", "hold", "holds", "held", "stand", "stands", "stood",
    "mean", "means", "meant", "keep", "keeps", "kept", "want", "wants",
    "need", "needs", "seem", "seems", "turn", "turns", "put", "puts",
    "let", "lets", "thank", "thanks", "look", "looks", "ask", "asks",
    "tell", "tells", "show", "shows", "call", "calls", "try", "tries",
    "read", "reads", "use", "uses", "map", "maps", "drop", "drops",
])

# "doesn't" and "hasn't" carry finite verbs. Without expansion the floor rule
# refused "OGT doesn't fit." as a fragment, which it plainly is not.
_CONTRACTIONS = {
    "n't": "", "'s": "is", "'re": "are", "'ve": "have",
    "'ll": "will", "'d": "would", "'m": "am",
}


def _expand(word: str) -> list[str]:
    w = word.lower().replace("\u2019", "'")
    forms = [w]
    for suffix, expansion in _CONTRACTIONS.items():
        if w.endswith(suffix) and len(w) > len(suffix):
            forms.append(w[: -len(suffix)])
            if expansion:
                forms.append(expansion)
    return forms


# No bare -s: it read "Same proteins." and "Thousands of substrates."
# as verbed sentences, which is exactly backwards for fragment
# detection. Third-person singulars are covered by FINITE_FORMS instead.
_VERBISH = re.compile(r"\w+(?:ed|ing)$")
_PLURAL_NOUNISH = re.compile(r"\w+(?:ss|us|is|ics|ness|tions?|ments?)$")


def has_finite_verb(span: Span) -> bool:
    words = [form for w in span.words for form in _expand(w)]
    if any(w in FINITE_FORMS for w in words):
        return True
    for w in words:
        if _VERBISH.match(w) and not _PLURAL_NOUNISH.match(w) and len(w) > 4:
            return True
    return False


@dataclass
class Cadence:
    choppiness: float
    short_share: float
    verbless_share: float
    run_mass: float
    mean_sentence_len: float
    verdict: str

    def as_dict(self) -> dict:
        return asdict(self)

    def warnings(self) -> list[str]:
        if self.verdict == "choppy":
            return [
                (
                    f"choppiness {self.choppiness:.2f}: {self.short_share:.0%} "
                    f"of words sit in sentences of {SHORT_WORDS} words or "
                    f"fewer and {self.verbless_share:.0%} of sentences have no "
                    "finite verb. This is the staccato cadence, and it is the "
                    "property a reader hears first"
                )
            ]
        return []

    def render(self) -> str:
        return (f"choppiness {self.choppiness:.2f} [{self.verdict}]  "
                f"short-word-share {self.short_share:.0%}  "
                f"verbless {self.verbless_share:.0%}  "
                f"run-mass {self.run_mass:.0%}")


def compute(doc: Document) -> Cadence:
    sentences = doc.sentences
    if not sentences:
        return Cadence(0.0, 0.0, 0.0, 0.0, 0.0, "n/a")

    lengths = [len(s) for s in sentences]
    total_words = sum(lengths) or 1

    short_words = sum(n for n in lengths if n <= SHORT_WORDS)
    short_share = short_words / total_words

    verbless = sum(1 for s in sentences if not has_finite_verb(s))
    verbless_share = verbless / len(sentences)

    run_words, run = 0, []
    for span, n in zip(sentences, lengths):
        if n <= SHORT_WORDS:
            run.append(n)
        else:
            if len(run) >= RUN_LENGTH:
                run_words += sum(run)
            run = []
        _ = span
    if len(run) >= RUN_LENGTH:
        run_words += sum(run)
    run_mass = run_words / total_words

    score = (WEIGHTS["short_share"] * short_share
             + WEIGHTS["verbless_share"] * verbless_share
             + WEIGHTS["run_mass"] * run_mass)
    verdict = "choppy" if score >= CHOPPY_THRESHOLD else "ok"

    return Cadence(
        choppiness=round(score, 4),
        short_share=round(short_share, 4),
        verbless_share=round(verbless_share, 4),
        run_mass=round(run_mass, 4),
        mean_sentence_len=round(statistics.mean(lengths), 2),
        verdict=verdict,
    )


# Features that move WITH choppiness for structural reasons, so their deltas
# are not independent evidence when cadence changes. Named here so drift can
# discount them instead of counting the same artifact three times.
COLLINEAR_WITH_CHOPPINESS = ("lexical_density", "cv", "word_burstiness")


def collinearity_note(delta_choppiness: float) -> str | None:
    if abs(delta_choppiness) < 0.02:
        return None
    direction = "rose" if delta_choppiness > 0 else "fell"
    return (
        f"choppiness {direction} {abs(delta_choppiness):.2f}, so lexical "
        "density and burstiness moved with it by construction: fragments are "
        "short and carry almost no function words. Their deltas are not "
        "independent evidence about this revision"
    )
