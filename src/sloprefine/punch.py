"""The punch index, and profiles that decide what a document is being judged as.

Why this overrides the audit
----------------------------
`calibration/fiction-2023.json` measures `fragments` at 0.37 and `runt` at
0.43: both fire MORE on human text than on machine text in that corpus. Taken
at face value, a calibrated run would weight them at zero.

That corpus is creative fiction, where a short-story writer breaks sentences
for effect on every page, and the models in it are 2023-vintage. The one human
judgement this package was built around is a domain expert reacting to a 2026
conference talk and calling the staccato cadence machine-sounding. One label,
but it is in the right register, from the right audience, about the right
text, and it outranks a stronger measurement taken somewhere else.

So the override is explicit rather than quiet. A profile PINS the weight of a
rule, and a pinned weight survives calibration. The calibration still reports
what it measured, so the disagreement stays visible instead of being resolved
by whichever file loaded last.

The punch index
---------------
The cadence signals were scattered across four modules and no single number
tracked the thing that started this project. `punch` is that number: the
staccato register, measured as one quantity.

    choppiness        short and verbless sentences, word-weighted
    closer ratio      paragraphs that end on a punch
    doublet density   balanced pairs per 1000 words
    parallel density  runs of three or more per 1000 words

Measured against real writing, and it inverts
----------------------------------------------
The first version of this index was anchored on two fixtures written by hand
for this repository, where the clean one scored 0.05. That was circular. Run
against the labelled corpus in calibration/:

    HUMAN   (New Yorker + Confederacy, n=17)   mean 0.232   53% punchy
    MACHINE (GPT-3.5/4, Claude 1.3, n=51)      mean 0.163   25% punchy

Human writing scores HIGHER. Per component:

    choppiness     human 0.108   machine 0.046   enrichment 0.43
    closer ratio   human 0.319   machine 0.153   enrichment 0.48
    doublets/1k    human 2.43    machine 3.12    enrichment 1.29
    triads+/1k     human 1.87    machine 2.42    enrichment 1.29

The two components carrying 70% of the weight are the two that invert.
Several New Yorker stories close 100% of their paragraphs on a short
sentence. On that corpus this index is a better detector of professional
fiction than of machine text.

What to make of that
--------------------
The corpus is fiction and the marker is register-relative. A staccato
paragraph close is craft in a short story and reads as machine-written in a
conference talk, because scientists do not write that way and a listener
notices the borrowed register. The construction is not the problem; the
construction appearing where the register does not support it is.

So the index is kept, the weights are kept for the talk profile, and the
contrary measurement is printed here rather than buried. For fiction this
index is worse than useless and PROFILE_THRESHOLDS reflects that. For a
scientific talk the only evidence either way is one expert reaction, which
is thin, and is the reason `talk` pins rather than measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cadence import compute as compute_cadence
from .paragraph import closer_ratio
from .parallel import doublets as find_doublets
from .parallel import find as find_parallel
from .text import Document

# Scaling denominators: the value at which a component counts as fully
# saturated. Set from the choppy control, which is deliberate slop.
SCALE = {"closers": 0.60, "doublets": 20.0, "parallel": 8.0}
COMPONENT_WEIGHTS = {"choppiness": 0.40, "closers": 0.30,
                     "doublets": 0.20, "parallel": 0.10}
PUNCHY_THRESHOLD = 0.20

# Per profile, with provenance. A single global threshold is indefensible
# once the human distribution is known to differ this much by register.
PROFILE_THRESHOLDS = {
    # one expert reaction to one talk; no distribution behind it
    "talk": 0.20,
    "essay": 0.35,
    # 53% of New Yorker stories clear 0.20, so a fiction threshold there
    # would flag half the corpus. Set above the human 90th percentile.
    "fiction": 0.55,
    "docs": 0.40,
}


@dataclass
class Punch:
    punch: float
    choppiness: float
    closer_ratio: float
    doublets_per_1k: float
    parallel_per_1k: float
    verdict: str

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    def render(self) -> str:
        return (f"punch {self.punch:.2f} [{self.verdict}]  "
                f"chop {self.choppiness:.2f}  closers {self.closer_ratio:.0%}  "
                f"doublets {self.doublets_per_1k:.1f}/1k  "
                f"triads+ {self.parallel_per_1k:.1f}/1k")

    def warnings(self) -> list[str]:
        if self.verdict != "punchy":
            return []
        return [
            (
                f"punch {self.punch:.2f}: the staccato register. "
                f"{self.choppiness:.0%} choppiness, {self.closer_ratio:.0%} of "
                f"paragraphs ending on a short sentence, "
                f"{self.doublets_per_1k:.0f} balanced pairs and "
                f"{self.parallel_per_1k:.0f} triads per 1000 words. This is "
                "the thing a reader hears first"
            )
        ]


def compute(doc: Document) -> Punch:
    words = doc.word_count or 1
    chop = compute_cadence(doc).choppiness
    closers = closer_ratio(doc)
    doub = len(find_doublets(doc)) / words * 1000
    par = len(find_parallel(doc)) / words * 1000

    parts = {
        "choppiness": min(1.0, chop / 0.6),
        "closers": min(1.0, closers / SCALE["closers"]),
        "doublets": min(1.0, doub / SCALE["doublets"]),
        "parallel": min(1.0, par / SCALE["parallel"]),
    }
    score = sum(COMPONENT_WEIGHTS[k] * v for k, v in parts.items())
    return Punch(
        punch=round(score, 3),
        choppiness=round(chop, 3),
        closer_ratio=round(closers, 3),
        doublets_per_1k=round(doub, 2),
        parallel_per_1k=round(par, 2),
        verdict="punchy" if score >= PUNCHY_THRESHOLD else "ok",
    )


# --------------------------------------------------------------- profiles

@dataclass(frozen=True)
class Profile:
    name: str
    why: str
    settings: dict = field(default_factory=dict)
    pins: dict = field(default_factory=dict)


PROFILES: dict[str, Profile] = {
    "talk": Profile(
        name="talk",
        # Opens on a clause, not a two-word fragment. This text is emitted
        # into the generation contract, where a fragment demonstrates the
        # cadence the profile exists to suppress.
        why=("The register is spoken delivery, so the cadence rules are "
             "pinned to their maximum weight regardless of what a "
             "calibration measured, because the only human label this "
             "package has is a domain expert calling exactly this register "
             "machine-sounding in a conference talk."),
        settings={
            "doublet_budget_per_1k": 0.0,
            "parallel_budget_per_1k": 0.0,
            "closer_budget_ratio": 0.10,
            "min_sentence_words": 5,
            "max_choppiness": 0.12,
            "person_budget_per_1k": 1.0,
        },
        pins={"fragments": 5.0, "doublet": 5.0, "parallel": 5.0,
              "runt": 4.0, "closer": 4.0, "opener": 3.0, "person": 3.0},
    ),
    "essay": Profile(
        name="essay",
        why="The register is written prose for a reader who can re-read, so "
            "cadence matters less than specificity and the requirement "
            "rules lead.",
        settings={
            "doublet_budget_per_1k": 2.0,
            "parallel_budget_per_1k": 1.0,
            "closer_budget_ratio": 0.25,
            "min_sentence_words": 0,
        },
        pins={"vague": 4.0, "template": 4.0},
    ),
    "docs": Profile(
        name="docs",
        why="The register is technical documentation, where enumerating "
            "things is the job, so parallelism runs on a loose budget with no sentence floor underneath it.",
        settings={
            "doublet_budget_per_1k": 3.0,
            "parallel_budget_per_1k": 3.0,
            "closer_budget_ratio": 0.35,
            "min_sentence_words": 0,
        },
        pins={},
    ),
}


def apply(profile_name: str, config):
    """Return a copy of config with the profile's settings applied."""
    from dataclasses import replace
    profile = PROFILES[profile_name]
    return replace(config, **profile.settings)
