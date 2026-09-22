"""The agent loop: generate, check, revise, stop.

slopcheck's primary consumer is a model revising its own draft, not a person
reading a report. That changes three things.

**Output is imperative and cheap.** A revising model does not need the citation
or the rationale; it needs the span and the instruction. `render_agent` emits
roughly a tenth the tokens of the human report.

**Prevention beats correction.** `style_contract` emits the rules as
constraints to put in front of generation. A draft that never contains the
pattern costs nothing to fix.

**The loop has a known failure mode, and the tool detects it.** Optimizing a
draft toward zero hits does not monotonically improve it. Measured on a real
revision: rule hits fell 24 to 1 while lexical density fell 0.530 to 0.493,
moving the text toward the AI-editing footprint of [SLH26] even as it moved
away from the AI cadence. Dissolving fragment stacks into flowing sentences
adds function words by construction. `drift` compares two revisions and says
when the fix is costing more than the hit did, so the loop has something to
stop on besides zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .report import Config, Result, analyze
from .rules import RULES

# Below this, a revision is churning rather than improving.
MIN_HIT_REDUCTION = 1
# Lexical-density loss (in absolute points) that outweighs a hit fix [SLH26].
DENSITY_COST = 0.015
# Hard stop. Past this the model is editing to satisfy the linter.
MAX_ROUNDS = 3


@dataclass
class Instruction:
    rule: str
    line: int
    span: str
    fix: str

    def render(self) -> str:
        return f"L{self.line} [{self.rule}] {self.span}\n  -> {self.fix}"


@dataclass
class Review:
    """A machine-facing verdict: pass/fail plus what to change."""

    path: str
    passed: bool
    total: int
    instructions: list[Instruction] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    result: Result | None = None

    def render(self, max_spans: int = 6) -> str:
        """Grouped by rule: the fix instruction is printed once per rule, not
        once per hit. Twelve fragment stacks do not need twelve copies of the
        same sentence, and a revising model pays for every one of them."""
        if self.passed and not self.notes:
            return "PASS"
        lines = ["PASS" if self.passed else f"FAIL {self.total}"]
        grouped: dict[str, list[Instruction]] = {}
        for ins in self.instructions:
            grouped.setdefault(ins.rule, []).append(ins)
        for rule, group in grouped.items():
            lines.append(f"[{rule}] x{len(group)}: {RULES[rule].fix}")
            for ins in group[:max_spans]:
                lines.append(f"  L{ins.line} {ins.span}")
            if len(group) > max_spans:
                lines.append(f"  ... {len(group) - max_spans} more")
        lines += [f"! {n}" for n in self.notes]
        return "\n".join(lines)


def review(path: str, text: str, config: Config | None = None) -> Review:
    config = config or Config()
    result = analyze(path, text, config)
    instructions = [
        Instruction(
            rule=hit.rule_id,
            line=result.text.count("\n", 0, hit.start) + 1,
            span=hit.text if len(hit.text) <= 70 else hit.text[:67] + "...",
            fix=RULES[hit.rule_id].fix,
        )
        for hit in result.hits
    ]
    notes = list(result.metrics.warnings()) + list(result.voice_notes)
    limit = config.max_hits if config.max_hits is not None else 0
    return Review(
        path=path,
        passed=result.total <= limit,
        total=result.total,
        instructions=instructions,
        notes=notes,
        result=result,
    )


def style_contract(
    disabled: tuple[str, ...] = (),
    voice: object | None = None,
) -> str:
    """Constraints to place in front of generation.

    Written as prohibitions with the repair built in, because a model given
    only a prohibition tends to satisfy it in the most literal available way.
    """
    lines = [
        (
            "Write to these constraints. They encode measured markers of "
            "machine-generated prose."
        ),
        "",
    ]
    for rule in RULES.values():
        if rule.id in disabled:
            continue
        lines.append(f"- {rule.why} {rule.fix}")
    lines += [
        "",
        (
            "Rhythm: vary sentence length deliberately. Mix sentences under 8 "
            "words with sentences over 25 in the same paragraph. Do not let "
            "three consecutive sentences land in the same length band."
        ),
        (
            "Stance: state at least one thing you are uncertain about, "
            "expected and did not find, or got wrong. Prose with no epistemic "
            "position reads as having no author."
        ),
        (
            "Specificity: prefer the concrete case to the general claim. A "
            "number, a name, or a date beats a category."
        ),
    ]
    if voice is not None and getattr(voice, "center", None):
        c = voice.center
        lines += [
            "",
            "Match this author's measured baseline:",
            f"- mean sentence length near {c.get('mean_sentence_len', 0):.0f} words",
            (
                f"- lexical density near {c.get('lexical_density', 0):.2f} "
                "(content words over total)"
            ),
            f"- contractions near {c.get('contraction_rate', 0):.0f} per 1000 words",
        ]
    return "\n".join(lines)


@dataclass
class Drift:
    """Did the revision help, and what did it cost?"""

    hits_before: int
    hits_after: int
    density_before: float
    density_after: float
    entropy_before: float
    entropy_after: float
    verdict: str
    notes: list[str] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        return self.verdict == "improved"

    def as_dict(self) -> dict:
        return {
            "hits_before": self.hits_before,
            "hits_after": self.hits_after,
            "density_before": self.density_before,
            "density_after": self.density_after,
            "entropy_before": self.entropy_before,
            "entropy_after": self.entropy_after,
            "verdict": self.verdict,
            "notes": self.notes,
        }

    def render(self) -> str:
        head = (
            f"{self.verdict.upper()} "
            f"hits {self.hits_before}->{self.hits_after} "
            f"density {self.density_before:.3f}->{self.density_after:.3f}"
        )
        return "\n".join([head] + [f"! {n}" for n in self.notes])


def drift(before: str, after: str, config: Config | None = None) -> Drift:
    """Compare two revisions of the same text.

    Verdicts:
      improved   - fewer hits, no meaningful stylometric cost
      traded     - fewer hits, but bought with lexical density [SLH26]
      churned    - hits unchanged or worse; the edit did not land
      overfit    - at zero hits and still editing
    """
    config = config or Config()
    r_before = analyze("before", before, config)
    r_after = analyze("after", after, config)
    s_before, s_after = r_before.style, r_after.style
    assert s_before is not None and s_after is not None

    d_before, d_after = s_before.lexical_density, s_after.lexical_density
    e_before, e_after = s_before.entropy_norm, s_after.entropy_norm
    delta_hits = r_before.total - r_after.total
    delta_density = d_after - d_before

    notes: list[str] = []
    if delta_hits <= 0 and r_after.total == 0:
        verdict = "overfit"
        notes.append(
            "already at zero hits before this edit; further passes optimize "
            "for the linter, not the reader"
        )
    elif delta_hits < MIN_HIT_REDUCTION:
        verdict = "churned"
        notes.append("the edit did not reduce hits; revert or try a different fix")
    elif delta_density <= -DENSITY_COST:
        verdict = "traded"
        notes.append(
            f"lexical density fell {abs(delta_density):.3f} while fixing "
            f"{delta_hits} hit(s): the revision added function words. [SLH26] "
            "measures that drop as the AI-editing signature, so this trade "
            "moves the text toward a different marker than the one removed"
        )
        if e_after < e_before:
            notes.append("entropy also fell, which is the same direction")
    else:
        verdict = "improved"
    return Drift(
        hits_before=r_before.total,
        hits_after=r_after.total,
        density_before=d_before,
        density_after=d_after,
        entropy_before=e_before,
        entropy_after=e_after,
        verdict=verdict,
        notes=notes,
    )
