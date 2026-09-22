"""Config, analysis entry point, and output rendering."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import cadence as cadence_mod
from . import metrics as metrics_mod
from . import paragraph as paragraph_mod
from . import reader as reader_mod
from . import stylometry as stylometry_mod
from . import voice as voice_mod
from . import weighting as weighting_mod
from .checks import Hit, run_checks
from .rules import RULES
from .text import Document, markdown_furniture, mask, suppressed_ranges

CONFIG_NAMES = (".slopcheck.toml", "slopcheck.toml")


@dataclass
class Config:
    disabled: tuple[str, ...] = ()
    allow: frozenset[str] = frozenset()
    min_sentence_words: int = 0  # 0 disables
    runt_mode: str = "verbless"   # "verbless" | "all"
    allow_runts: frozenset[str] = frozenset()
    parallel_budget_per_1k: float = 1.0
    closer_budget_ratio: float = 0.25
    doublet_budget_per_1k: float = 2.0
    vague_min_words: int = 40
    max_choppiness: float | None = None
    max_hits: int | None = None
    max_per_1k: float | None = None
    skip_code_blocks: bool = True
    voice: voice_mod.Voiceprint | None = None
    voice_path: str | None = None
    z_threshold: float = 2.0
    audience: str | None = None  # "expert" | "general"; None disables

    @classmethod
    def load(cls, start: Path | None = None) -> Config:
        start = (start or Path.cwd()).resolve()
        for directory in (start, *start.parents):
            for name in CONFIG_NAMES:
                candidate = directory / name
                if candidate.is_file():
                    return cls.from_toml(candidate)
        return cls()

    @classmethod
    def from_toml(cls, path: Path) -> Config:
        data = tomllib.loads(path.read_text()).get("slopcheck", {})
        unknown = set(data.get("disable", [])) - set(RULES)
        if unknown:
            raise ValueError(f"{path}: unknown rule(s): {sorted(unknown)}")
        return cls(
            disabled=tuple(data.get("disable", [])),
            allow=frozenset(w.lower() for w in data.get("allow", [])),
            max_hits=data.get("max_hits"),
            max_per_1k=data.get("max_per_1k"),
            max_choppiness=data.get("max_choppiness"),
            min_sentence_words=data.get("min_sentence_words", 0),
            runt_mode=data.get("runt_mode", "verbless"),
            parallel_budget_per_1k=data.get("parallel_budget_per_1k", 1.0),
            closer_budget_ratio=data.get("closer_budget_ratio", 0.25),
            doublet_budget_per_1k=data.get("doublet_budget_per_1k", 2.0),
            vague_min_words=data.get("vague_min_words", 40),
            allow_runts=frozenset(data.get("allow_runts", [])),
            skip_code_blocks=data.get("skip_code_blocks", True),
            voice_path=data.get("voice"),
            audience=data.get("audience"),
        )


@dataclass
class Result:
    path: str
    text: str = field(repr=False, default="")
    hits: list[Hit] = field(default_factory=list)
    metrics: metrics_mod.Metrics | None = None
    cadence: cadence_mod.Cadence | None = None
    structure: dict = field(default_factory=dict)
    worst: list = field(default_factory=list)
    style: stylometry_mod.Stylometry | None = None
    reader: reader_mod.ReaderSignals | None = None
    reader_notes: list[str] = field(default_factory=list)
    deviations: dict[str, float] = field(default_factory=dict)
    voice_notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.hits)

    @property
    def score(self) -> float:
        """Severity-weighted hits per 1000 words."""
        w = self.metrics.words if self.metrics else 0
        return weighting_mod.score(self.hits, w)

    @property
    def per_1k(self) -> float:
        w = self.metrics.words if self.metrics else 0
        return round(self.total / w * 1000, 2) if w else 0.0

    def counts(self) -> dict[str, int]:
        out = {rid: 0 for rid in RULES}
        for h in self.hits:
            out[h.rule_id] += 1
        return out


def analyze(path: str, text: str, config: Config) -> Result:
    """Suppressed regions are masked before tokenization, not filtered after.
    Filtering afterwards leaks: a multi-sentence hit can start outside a
    suppressed region and reach into it."""
    is_md = path.endswith((".md", ".markdown"))
    regions = suppressed_ranges(text, skip_code=config.skip_code_blocks and is_md)
    if is_md:
        regions += markdown_furniture(text)
    doc = Document(mask(text, regions), path)
    hits = run_checks(
        doc, disabled=config.disabled, allow=config.allow,
        floor=config.min_sentence_words, runt_mode=config.runt_mode,
        allow_runts=config.allow_runts,
        parallel_budget_per_1k=config.parallel_budget_per_1k,
        closer_budget_ratio=config.closer_budget_ratio,
        doublet_budget_per_1k=config.doublet_budget_per_1k,
        vague_min_words=config.vague_min_words,
    )
    style = stylometry_mod.compute(doc)
    deviations, notes = {}, []
    if config.voice is not None:
        deviations = config.voice.compare(style)
        notes = voice_mod.interpret(deviations, threshold=config.z_threshold)
    cad = cadence_mod.compute(doc)
    structure = {
        "closer_ratio": paragraph_mod.closer_ratio(doc),
        "specifics_per_1k": paragraph_mod.specificity_per_1k(doc),
        "vague_paragraphs": len(paragraph_mod.vague_paragraphs(
            doc, config.vague_min_words)),
    }
    signals, reader_notes = None, []
    if config.audience:
        signals = reader_mod.compute(doc, config.audience)
        reader_notes = reader_mod.notes(signals)
    return Result(
        path=path, text=text, hits=hits, metrics=metrics_mod.compute(doc),
        style=style, deviations=deviations, voice_notes=notes,
        reader=signals, reader_notes=reader_notes, cadence=cad,
        structure=structure,
        worst=weighting_mod.worst_paragraphs(doc, hits),
    )


# ------------------------------------------------------------------ render

_STYLE_ROWS = [
    ("lexical diversity", "lexical_diversity"),
    ("entropy (norm)", "entropy_norm"),
    ("lexical density", "lexical_density"),
    ("word burstiness", "word_burstiness"),
    ("punctuation", "pct_punctuation"),
    ("contractions /1k", "contraction_rate"),
    ("opener diversity", "opener_diversity"),
    ("gunning fog", "gunning_fog"),
]

BOLD, DIM, RED, YELLOW, GREEN, RESET = (
    "\033[1m", "\033[2m", "\033[31m", "\033[33m", "\033[32m", "\033[0m",
)
SEV_COLOR = {"high": RED, "medium": YELLOW, "low": DIM}


def render_text(result: Result, verbose: bool = True, color: bool = True) -> str:
    def c(code: str, s: str) -> str:
        return f"{code}{s}{RESET}" if color else s

    doc_lines = result.text.splitlines()
    out = [c(BOLD, result.path) + f"  ({result.metrics.words} words)"]
    counts = result.counts()

    for rule_id, rule in RULES.items():
        n = counts[rule_id]
        mark = "ok " if n == 0 else c(SEV_COLOR[rule.severity], "HIT")
        out.append(
            f"  {mark} {rule.title:<34} {n:>3}  {c(DIM, rule.citation)}"
        )
        if n and verbose:
            shown = [h for h in result.hits if h.rule_id == rule_id][:6]
            for h in shown:
                line = result.text.count("\n", 0, h.start) + 1
                snippet = h.text if len(h.text) <= 76 else h.text[:73] + "..."
                note = f" {c(DIM, '(' + h.note + ')')}" if h.note else ""
                out.append(f"       {c(DIM, f'L{line}')}  {snippet}{note}")
            if n > len(shown):
                out.append(c(DIM, f"       ... {n - len(shown)} more"))

    m = result.metrics
    out.append("  " + "-" * 60)
    if result.cadence is not None:
        colour = RED if result.cadence.verdict == "choppy" else GREEN
        out.append("  " + c(BOLD, "cadence ") + c(colour, result.cadence.render()))
    if result.structure:
        out.append("  " + c(BOLD, "shape   ")
                   + f"closers {result.structure['closer_ratio']:.0%} of paragraphs"
                   + f"   specifics {result.structure['specifics_per_1k']}/1k")
    out.append(
        f"  sentences={m.sentences}  mean={m.mean_len}w  sd={m.stdev_len}  "
        f"CV={m.cv}  flat-run={m.longest_flat_run}"
    )
    out.append(
        f"  short<=7w {m.pct_short:.0%}   long>=25w {m.pct_long:.0%}   "
        f"footprint={m.footprint} ({m.footprint_per_1k}/1k words)"
    )
    s = result.style
    if s is not None:
        out.append("  " + "-" * 60)
        if result.deviations:
            out.append("  " + c(DIM, "stylometry (value, deviation from your voiceprint)"))
            for label, key in _STYLE_ROWS:
                value = getattr(s, key)
                z = result.deviations.get(key)
                zs = "     " if z is None else f"{z:+5.1f}s"
                flag = c(YELLOW, zs) if z is not None and abs(z) >= 2 else c(DIM, zs)
                out.append(f"    {label:<22} {value:>9}   {flag}")
        else:
            out.append("  " + c(DIM, "stylometry (no voiceprint: values only, no judgement)"))
            for label, key in _STYLE_ROWS:
                out.append(f"    {label:<22} {getattr(s, key):>9}")
    for note in result.reader_notes:
        out.append("  " + c(YELLOW, "read  ") + note)
    for note in result.voice_notes:
        out.append("  " + c(YELLOW, "voice ") + note)
    for w in (result.cadence.warnings() if result.cadence else []):
        out.append("  " + c(RED, "CHOP  ") + w)
    for w in m.warnings():
        out.append("  " + c(YELLOW, "warn ") + w)
    total_color = GREEN if result.total == 0 else RED
    out.append(
        "  " + c(BOLD, "total ") + c(total_color, str(result.total))
        + f"  ({result.per_1k}/1k, weighted {result.score}/1k)"
    )
    if result.worst:
        out.append("  " + c(BOLD, "worst  ") + result.worst[0].render())
    _ = doc_lines
    return "\n".join(out)


def render_json(results: list[Result]) -> str:
    payload = [
        {
            "path": r.path,
            "total": r.total,
            "per_1k": r.per_1k,
            "score": r.score,
            "worst_paragraphs": [vars(p) for p in r.worst],
            "counts": r.counts(),
            "cadence": r.cadence.as_dict() if r.cadence else {},
            "structure": r.structure,
            "metrics": r.metrics.as_dict(),
            "stylometry": r.style.as_dict() if r.style else {},
            "deviations": r.deviations,
            "reader": r.reader.as_dict() if r.reader else {},
            "reader_notes": r.reader_notes,
            "voice_notes": r.voice_notes,
            "warnings": r.metrics.warnings(),
            "hits": [
                {
                    "rule": h.rule_id,
                    "severity": RULES[h.rule_id].severity,
                    "citation": RULES[h.rule_id].citation,
                    "line": r.text.count("\n", 0, h.start) + 1,
                    "start": h.start,
                    "end": h.end,
                    "text": h.text,
                    "note": h.note,
                }
                for h in r.hits
            ],
        }
        for r in results
    ]
    return json.dumps(payload, indent=2)
