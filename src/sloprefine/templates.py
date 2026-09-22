"""Syntactic templates: repetition of structure rather than of words.

Method
------
[SEL24] Shaib, Elazar, Li & Wallace, EMNLP 2024. "Detection and Measurement
        of Syntactic Templates in Generated Text." Templates are frequent
        part-of-speech n-grams. Three metrics: CR-POS, a gzip compression
        ratio over the POS sequence, where higher means more redundant;
        template rate, the fraction of texts containing at least one
        template; and templates-per-token. They report that 76% of templates
        in model output are traceable to pretraining data against 35% for
        human text, and that RLHF does not remove them.

This measures the shape vocabulary of a whole document. It is NOT the general
case of the shape rules, which was the first thing assumed about it here and
is measured false below.

Deriving the tagger
-------------------
[SEL24] uses a POS tagger. This package has no dependencies and is not
acquiring one for a single metric, so the tags are derived from what English
makes available for free:

  - Function words are a closed class. They ARE their own tags, so each maps
    to itself. This is most of the signal, because syntax is mostly function
    words.
  - Content words get a coarse tag from morphology: -ing, -ed, -ly, -tion /
    -ment / -ness / -ity, plural -s, capitalisation, digits. English suffixes
    carry enough category information for a sequence-level statistic.
  - Everything else is a generic content tag.

The result is an alphabet of roughly 60 symbols. It will disagree with a real
tagger on individual tokens and that is tolerable here, because the metric is
a property of the whole sequence, not of any one tag.

The shuffled control
--------------------
Compression ratio rises with document length, so CR-POS is not comparable
across documents of different sizes. [SEL24] compares within fixed-size
samples, which sidesteps it. A linter gets one document at a time.

So every measure here is taken as a ratio against a shuffled control.
Shuffling the tag sequence destroys ORDER while preserving the tag
DISTRIBUTION and the length exactly. Whatever survives shuffling came from
the vocabulary being skewed; whatever is lost was sequential structure. The
quotient is length-controlled and distribution-controlled by construction.

Why gzip is not the estimator here
-----------------------------------
CR-POS is computed and reported because [SEL24] defines it, but at the length
of a single document it is nearly useless. Measured:

    slop control   1.02      clean control  1.04
    talk (R0)      1.06      talk (R2)      1.07

A 0.05 spread with the sign inconsistent. The reason is that gzip on a
thousand tokens has almost nothing for LZ77 to match, so the ratio is
dominated by Huffman coding of the symbol distribution, which the shuffle
preserves exactly. gzip needs corpus-sized input to see what it is supposed
to see.

The direct estimator is the share of tag n-grams that occur more than once,
against the same shuffled baseline. Same quantity, no compression middleman,
and it separates:

    n=4   slop control 3.33   clean control 1.56   R0 1.90   R2 1.76

It does not subsume the local rules. Measured
----------------------------------------------
The claim this module was built on was that doublets, triads, repeated openers
and fragment stacks are all instances of "the text reuses a small set of
structures", so a document-level redundancy statistic should see them all
without being told what any of them look like. That is false, and cheap to
check. Taking the talk this repository was built around, which has eight
doublets, and breaking seven of them by hand:

    with 8 doublets     structure 1.884   repeat_4 0.0950
    7 of 8 broken       structure 1.884   repeat_4 0.0967

No change, and the second number moved the wrong way. The arithmetic is plain
once looked at: a doublet is two occurrences of a roughly four-token pattern,
eight of them are about 1% of the 4-gram mass in a 1250-token document, and
9.5% of that document's 4-grams already repeat because that is what English
function-word syntax does. The base rate swamps it.

Local constructions need local detectors. A document statistic and a span
rule measure different things, and neither subsumes the other. Keep both.

Scope, stated plainly
---------------------
This measures template reuse, which is a GENERATION signal. [SEL24] compared
model output against human reference text. It does not separate R0 from R2,
which are the same author writing the same content with different cadence,
and it is right not to: both are human-written and neither reuses syntax.

This is the same generation-versus-editing split [SLH26] found in
stylometry.py. Template redundancy catches text a model wrote. It says
nothing about text a model edited, and the rest of this package is aimed at
the latter.

The controls behind these numbers are two documents written by hand for this
repository. That is not evidence. Run `sloprefine audit` on real corpora
before trusting any threshold here.
"""

from __future__ import annotations

import random
import re
import zlib
from collections import Counter
from dataclasses import asdict, dataclass

from .stylometry import STOPWORDS
from .text import WORD, Document

TEMPLATE_N = 6
SHUFFLE_TRIALS = 25
SHUFFLE_SEED = 0

_DIGIT = re.compile(r"\d")


def tag(word: str) -> str:
    """POS proxy. Function words tag as themselves; content words by suffix."""
    w = word.lower().replace("\u2019", "'")
    if _DIGIT.search(word):
        return "9"
    if w in STOPWORDS:
        return w
    if word[:1].isupper():
        return "^"
    if w.endswith("ly"):
        return "~adv"
    if w.endswith("ing"):
        return "~ing"
    if w.endswith("ed"):
        return "~ed"
    if w.endswith(("tion", "sion", "ment", "ness", "ity", "ance", "ence")):
        return "~nom"
    if w.endswith(("ous", "ive", "able", "ible", "al", "ic", "ful", "less")):
        return "~adj"
    if w.endswith("s") and not w.endswith("ss"):
        return "~pl"
    return "~n"


def tag_sequence(doc: Document) -> list[str]:
    return [tag(w) for w in doc.words]


def _compression_ratio(tags: list[str]) -> float:
    """Original size over compressed size. Higher means more redundant."""
    if not tags:
        return 1.0
    blob = " ".join(tags).encode()
    compressed = zlib.compress(blob, 9)
    return len(blob) / len(compressed)


@dataclass
class Templates:
    tokens: int
    cr_pos: float            # [SEL24] gzip ratio; reported, not relied on
    cr_structure: float | None
    repeat_3: float          # share of tag trigrams occurring more than once
    repeat_4: float
    structure: float | None  # repeat_4 over its shuffled control; None if
                             # the baseline is too sparse to measure
    template_rate: float     # share of sentences containing a template
    templates_per_1k: float
    top_templates: list[tuple[str, int]]

    def as_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        if self.structure is None:
            return (f"structure unmeasurable ({self.tokens} tokens: the "
                    "shuffled baseline has no 4-gram repeats to compare "
                    "against)")
        return (f"structure {self.structure:.2f}x shuffled  "
                f"template-rate {self.template_rate:.0%}  "
                f"cr-pos {self.cr_pos:.2f}")


def _repeat_mass(tags: list[str], n: int) -> float:
    """Share of n-gram occurrences that are not the only one of their kind."""
    grams = Counter(tuple(tags[i:i + n]) for i in range(len(tags) - n + 1))
    total = sum(grams.values())
    return sum(c for c in grams.values() if c > 1) / total if total else 0.0


def _against_shuffled(tags: list[str], fn, trials: int = SHUFFLE_TRIALS
                      ) -> tuple[float, float | None]:
    """Observed value, and its ratio to the shuffled baseline.

    Returns None for the ratio when the baseline is zero. Returning 1.0 there
    was a silent failure: it reads as "no structure above chance" when it
    actually means "the baseline is too sparse to measure against", and on a
    short document those are opposite conclusions.
    """
    observed = fn(tags)
    rng = random.Random(SHUFFLE_SEED)
    baseline = []
    for _ in range(trials):
        shuffled = tags[:]
        rng.shuffle(shuffled)
        baseline.append(fn(shuffled))
    mean = sum(baseline) / len(baseline)
    if mean <= 0:
        return observed, None
    return observed, observed / mean


def compute(doc: Document, n: int = TEMPLATE_N, min_count: int = 2
            ) -> Templates:
    tags = tag_sequence(doc)
    if len(tags) < n * 4:
        return Templates(len(tags), 1.0, None, 0.0, 0.0, None, 0.0, 0.0, [])

    cr, cr_structure = _against_shuffled(tags, _compression_ratio)
    repeat_3, _ = _against_shuffled(tags, lambda s: _repeat_mass(s, 3))
    repeat_4, structure = _against_shuffled(tags, lambda s: _repeat_mass(s, 4))

    grams = Counter(
        tuple(tags[i:i + n]) for i in range(len(tags) - n + 1)
    )
    templates = {g: c for g, c in grams.items() if c >= min_count}

    # which sentences contain at least one template
    hit_sentences = 0
    cursor = 0
    for span in doc.sentences:
        count = len(WORD.findall(span.text))
        window = tags[cursor:cursor + count]
        cursor += count
        if any(tuple(window[i:i + n]) in templates
               for i in range(max(0, len(window) - n + 1))):
            hit_sentences += 1

    total_template_tokens = sum(templates.values())
    return Templates(
        tokens=len(tags),
        cr_pos=round(cr, 3),
        cr_structure=None if cr_structure is None else round(cr_structure, 4),
        repeat_3=round(repeat_3, 4),
        repeat_4=round(repeat_4, 4),
        structure=None if structure is None else round(structure, 3),
        template_rate=round(hit_sentences / len(doc.sentences), 3)
        if doc.sentences else 0.0,
        templates_per_1k=round(total_template_tokens / len(tags) * 1000, 2),
        top_templates=[(" ".join(g), c)
                       for g, c in Counter(templates).most_common(5)],
    )
