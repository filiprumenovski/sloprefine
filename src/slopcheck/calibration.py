"""Calibration: replace guessed severities with measured enrichment.

Until this module existed, every weight and threshold in the package was an
opinion. `rules.severity` was assigned by hand, and `weighting.WEIGHTS` turned
those opinions into a score. A calibration file replaces them with a number
measured on a corpus: how much more often does this rule fire on machine text
than on human text?

    weight = log2(enrichment), floored at 0

Log because enrichment is a ratio and a rule that fires 16x more often is not
16 times as informative as one that fires 2x. Floored at 0 because a rule that
fires MORE on human text carries no evidence for the thing this tool is for,
and giving it negative weight would credit a document for containing it.

A calibration is scoped to the corpus it came from. A rule that discriminates
on 2023-era models writing fiction may be dead on 2026 models writing
abstracts, which is the decay the audit exists to measure. The file records
the corpus sizes and the date so the scope travels with the numbers.

Rules that invert are not silently dropped. They are reported, because a rule
firing more on human text is a finding about the rule, and the user should see
which of their assumptions just failed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .rules import RULES

# Below this the rule is not distinguishing anything on the calibration corpus.
LIVE_THRESHOLD = 1.25
MAX_WEIGHT = 5.0


@dataclass
class Calibration:
    weights: dict[str, float]
    enrichment: dict[str, float]
    verdicts: dict[str, str]
    n_machine: int = 0
    n_human: int = 0
    corpus: str = ""
    note: str = ""
    pins: dict[str, float] = field(default_factory=dict)
    version: int = 1

    @classmethod
    def from_audit(cls, audit_json: dict, corpus: str = "", note: str = ""
                   ) -> Calibration:
        weights, enrichment, verdicts = {}, {}, {}
        for row in audit_json.get("rules", []):
            rule = row["rule"]
            if rule not in RULES:
                continue
            value = row["enrichment"]
            verdicts[rule] = row["verdict"]
            if value == float("inf") or value is None:
                # Fires only on machine text. Informative, but the rate is
                # usually tiny, so it does not earn an unbounded weight.
                enrichment[rule] = float("inf")
                weights[rule] = MAX_WEIGHT if row["ai_per_1k"] >= 0.5 else 1.0
                continue
            enrichment[rule] = value
            weights[rule] = (
                0.0 if value < LIVE_THRESHOLD
                else min(MAX_WEIGHT, round(math.log2(value), 3))
            )
        return cls(
            weights=weights, enrichment=enrichment, verdicts=verdicts,
            n_machine=audit_json.get("n_ai", 0),
            n_human=audit_json.get("n_human", 0),
            corpus=corpus, note=note,
        )

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != 1:
            raise ValueError(f"unsupported calibration version: {data.get('version')}")
        # JSON has no infinity, so to_json writes the string; convert back
        # rather than letting it reach the comparison as a str.
        data["enrichment"] = {
            k: (float("inf") if v == "inf" else float(v))
            for k, v in data.get("enrichment", {}).items()
        }
        return cls(**data)

    def to_json(self) -> str:
        payload = dict(self.__dict__)
        payload["enrichment"] = {
            k: ("inf" if v == float("inf") else v)
            for k, v in self.enrichment.items()
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def weight(self, rule_id: str, default: float = 1.0) -> float:
        """A pinned weight beats a measured one.

        Pins exist so a user's judgement about their own register can
        override a measurement taken somewhere else, without the measurement
        being deleted. `disagreements()` keeps the conflict visible."""
        if rule_id in self.pins:
            return self.pins[rule_id]
        return self.weights.get(rule_id, default)

    def disagreements(self) -> list[str]:
        """Rules where a pin overrides what the corpus measured."""
        out = []
        for rule, pinned in sorted(self.pins.items()):
            measured = self.weights.get(rule)
            if measured is not None and abs(measured - pinned) > 0.5:
                enrich = self.enrichment.get(rule, 0.0)
                out.append(
                    f"{rule}: pinned at {pinned:g}, but this corpus measured "
                    f"enrichment {enrich:.2f}"
                    + (" (fires MORE on human text)" if enrich < 1 else "")
                )
        return out

    def inverted(self) -> list[str]:
        return sorted(r for r, v in self.verdicts.items()
                      if v in {"inverted", "dead"})

    def summary(self) -> str:
        live = sorted(
            ((r, self.enrichment.get(r, 0.0)) for r, v in self.verdicts.items()
             if v in {"live", "machine-only"}),
            key=lambda kv: -(kv[1] if kv[1] != float("inf") else 1e9),
        )
        lines = [
            f"calibration: {self.n_machine} machine / {self.n_human} human"
            + (f", {self.corpus}" if self.corpus else ""),
            "  discriminating: " + ", ".join(
                f"{r} x{'inf' if e == float('inf') else f'{e:.1f}'}"
                for r, e in live) or "  discriminating: none",
        ]
        if self.inverted():
            lines.append("  NOT discriminating on this corpus: "
                         + ", ".join(self.inverted()))
        for line in self.disagreements():
            lines.append("  pinned over measurement: " + line)
        if self.note:
            lines.append("  " + self.note)
        return "\n".join(lines)


@dataclass
class CalibratedScore:
    raw: int
    weighted: float
    per_1k: float
    ignored: list[str] = field(default_factory=list)
