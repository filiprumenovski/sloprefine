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

Scaled so each component is roughly 0 to 1 over the observed range, then
averaged with choppiness and closers carrying the most, because those are the
two a listener experiences directly. Measured on the documents available:

    choppy control   0.79    the talk as delivered (R0)   0.36
    the same talk revised (R2/final)   0.10    human prose control   0.03

The thresholds come from four documents, one of which carries a real human
label. Thin, and stated as thin.
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
        why=("Spoken delivery. The cadence rules are pinned to their maximum "
             "weight regardless of what a calibration measured, because the "
             "only human label this package has is a domain expert calling "
             "exactly this register machine-sounding in a conference talk."),
        settings={
            "doublet_budget_per_1k": 0.0,
            "parallel_budget_per_1k": 0.0,
            "closer_budget_ratio": 0.10,
            "min_sentence_words": 5,
            "max_choppiness": 0.12,
        },
        pins={"fragments": 5.0, "doublet": 5.0, "parallel": 5.0,
              "runt": 4.0, "closer": 4.0, "opener": 3.0},
    ),
    "essay": Profile(
        name="essay",
        why="Written prose for a reader who can re-read. Cadence matters less "
            "than specificity, so the requirement rules lead.",
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
        why="Technical documentation. Enumerating things is the job, so the "
            "parallelism budget is loose and the floor is off.",
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
