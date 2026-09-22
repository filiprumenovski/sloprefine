"""sloprefine command line interface."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from . import voice as voice_mod
from .report import Config, analyze, render_json, render_text
from .rules import COMMONLY_LEGITIMATE, RULES

TEXT_SUFFIXES = (".txt", ".md", ".markdown", ".rst", ".text")

EPILOG = """\
examples:
  sloprefine draft.md                        report
  sloprefine draft.md --json                 machine-readable, with offsets
  sloprefine voice build ~/writing -o me.json   baseline from your own prose
  sloprefine draft.md --voice me.json        deviation from your own baseline
  sloprefine draft.md --max-hits 0           exit 1 if anything fires

config: .sloprefine.toml, searched upward from cwd

  [sloprefine]
  disable = ["colon", "hedge"]
  allow = ["landscape", "robust"]
  voice = "me.json"
  max_hits = 0

exit codes: 0 under threshold, 1 over, 2 bad invocation
"""


def _add_check_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("paths", nargs="*", help="files to check; - for stdin")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="counts only, no matched spans")
    p.add_argument("--format", choices=("text", "json", "agent"), default="text",
                   help="agent: compact PASS/FAIL plus imperative fixes, for "
                        "a model revising its own draft")
    p.add_argument("--json", action="store_true", help="alias for --format json")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--disable", action="append", default=[], metavar="RULE")
    p.add_argument("--allow", action="append", default=[], metavar="WORD")
    p.add_argument("--allow-domain-words", action="store_true",
                   help=f"exempt: {', '.join(sorted(COMMONLY_LEGITIMATE))}")
    p.add_argument("--profile", choices=("talk", "essay", "docs"),
                   help="what this document is being judged as. talk pins the "
                        "cadence rules to maximum weight and zeroes the "
                        "parallelism budgets")
    p.add_argument("--calibration", metavar="FILE",
                   help="weight rules by measured enrichment instead of "
                        "hand-assigned severity (see sloprefine audit --save)")
    p.add_argument("--voice", metavar="FILE",
                   help="voiceprint JSON; report stylometry as deviation "
                        "from your own baseline instead of raw values")
    p.add_argument("--z-threshold", type=float, default=2.0, metavar="X")
    p.add_argument("--audience", choices=("expert", "general"),
                   help="apply reader-preference signals for this audience "
                        "(directions measured post-2022; see reader.py)")
    p.add_argument("--lm", metavar="MODEL",
                   help="optional causal LM for perplexity structure "
                        '(needs pip install "sloprefine[lm]")')
    p.add_argument("--check-code-blocks", action="store_true",
                   help="do not exempt fenced code blocks in Markdown")
    p.add_argument("--min-sentence", type=int, metavar="N",
                   help="refuse sentences under N words. Off unless set here "
                        "or in config; 5 is the usual setting")
    p.add_argument("--strict-runts", action="store_true",
                   help="refuse every sentence under the floor, not just the "
                        "verbless ones ('It worked.' gets refused too)")
    p.add_argument("--parallel-budget", type=float, metavar="X",
                   help="allowed parallelism runs per 1000 words (default 1)")
    p.add_argument("--closer-budget", type=float, metavar="X",
                   help="share of paragraphs allowed to end on a short "
                        "sentence (default 0.25)")
    p.add_argument("--max-choppiness", type=float, metavar="X",
                   help="exit 1 above this cadence score (0.17 is the "
                        "calibrated band; see cadence.py)")
    p.add_argument("--max-hits", type=int, metavar="N")
    p.add_argument("--max-per-1k", type=float, metavar="X")


COMMANDS = ("check", "voice", "rules", "prompt", "drift", "audit")


def build_parser() -> argparse.ArgumentParser:
    """Parser for `check`, which is also the default with no subcommand."""
    p = argparse.ArgumentParser(
        prog="sloprefine",
        description="Lint prose against published AI-writing tells. "
                    "Subcommands: check (default), voice, rules, "
                    "prompt, drift, audit.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_check_args(p)
    return p


def build_voice_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sloprefine voice",
        description="Build or inspect a voiceprint: a baseline of your own "
                    "stylometry, so a draft is judged against you and not "
                    "against a population threshold.",
    )
    sub = p.add_subparsers(dest="voice_command", required=True)
    build = sub.add_parser("build", help="build a baseline from your writing")
    build.add_argument("paths", nargs="+", help="files or directories")
    build.add_argument("-o", "--output", required=True, metavar="FILE")
    show = sub.add_parser("show", help="print a voiceprint")
    show.add_argument("path")
    return p


def _collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files += sorted(
                f for f in p.rglob("*")
                if f.is_file() and f.suffix.lower() in TEXT_SUFFIXES
            )
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(raw)
    return files


def _cmd_rules() -> int:
    for rule in RULES.values():
        print(f"{rule.id:<13} {rule.severity:<7} {rule.title}")
        print(f"{'':<13} {rule.citation}  {rule.why}")
    return 0


def _cmd_voice(args) -> int:
    if args.voice_command == "show":
        try:
            vp = voice_mod.Voiceprint.load(args.path)
        except (OSError, ValueError) as exc:
            print(f"sloprefine: {exc}", file=sys.stderr)
            return 2
        print(vp.to_json())
        return 0

    try:
        files = _collect(args.paths)
    except FileNotFoundError as exc:
        print(f"sloprefine: no such file or directory: {exc}", file=sys.stderr)
        return 2
    docs = [(str(f), f.read_text(encoding="utf-8", errors="replace")) for f in files]
    try:
        vp = voice_mod.build(docs)
    except ValueError as exc:
        print(f"sloprefine: {exc}", file=sys.stderr)
        return 2
    Path(args.output).write_text(vp.to_json(), encoding="utf-8")
    print(f"voiceprint written to {args.output}: {vp.n_docs} documents, "
          f"{vp.n_words} words")
    return 0


def _cmd_prompt(argv: list[str]) -> int:
    from .agent import style_contract
    p = argparse.ArgumentParser(
        prog="sloprefine prompt",
        description="Emit the rules as generation-time constraints. Put this "
                    "in front of the model instead of fixing the output.")
    p.add_argument("--disable", action="append", default=[], metavar="RULE")
    p.add_argument("--voice", metavar="FILE")
    p.add_argument("--audience", choices=("expert", "general"))
    p.add_argument("--profile", choices=("talk", "essay", "docs"),
                   help="constrain generation with the same profile that will "
                        "judge the draft")
    args = p.parse_args(argv)
    voiceprint = None
    if args.voice:
        try:
            voiceprint = voice_mod.Voiceprint.load(args.voice)
        except (OSError, ValueError) as exc:
            print(f"sloprefine: voiceprint: {exc}", file=sys.stderr)
            return 2
    print(style_contract(disabled=tuple(args.disable), voice=voiceprint,
                         audience=args.audience, profile=args.profile))
    return 0


def _cmd_drift(argv: list[str]) -> int:
    from .agent import drift
    p = argparse.ArgumentParser(
        prog="sloprefine drift",
        description="Compare two revisions. Tells you whether the edit "
                    "improved the draft or just traded one marker for another.")
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--json", action="store_true")
    p.add_argument("--profile", choices=("talk", "essay", "docs"),
                   help="judge both revisions under this profile; without it "
                        "drift and check disagree about the hit count")
    p.add_argument("--fail-on", default="churned,overfit,traded", metavar="LIST",
                   help="comma-separated verdicts that exit 1 "
                        "(default: churned,overfit,traded)")
    args = p.parse_args(argv)
    for path in (args.before, args.after):
        if not Path(path).is_file():
            print(f"sloprefine: no such file: {path}", file=sys.stderr)
            return 2
    config = Config.load()
    if args.profile:
        from .punch import apply
        config = apply(args.profile, config)
    d = drift(Path(args.before).read_text(encoding="utf-8"),
              Path(args.after).read_text(encoding="utf-8"),
              config)
    if args.json:
        import json
        print(json.dumps(d.as_dict(), indent=2))
    else:
        print(d.render())
    return 1 if d.verdict in {v.strip() for v in args.fail_on.split(",")} else 0


def _cmd_audit(argv: list[str]) -> int:
    from .audit import audit, load_corpus
    p = argparse.ArgumentParser(
        prog="sloprefine audit",
        description="Measure whether each rule still discriminates between "
                    "machine and human text on YOUR corpora. Markers decay: "
                    "a fix published widely enough gets absorbed by the next "
                    "model generation, and then it is no longer a marker.")
    p.add_argument("--ai", required=True, metavar="DIR",
                   help="directory of machine-written text")
    p.add_argument("--human", required=True, metavar="DIR",
                   help="directory of human-written text")
    p.add_argument("--audience", choices=("expert", "general"), default="expert")
    p.add_argument("--json", action="store_true")
    p.add_argument("--save", metavar="FILE",
                   help="write a calibration file: per-rule weights measured "
                        "from these corpora, for use with check --calibration")
    p.add_argument("--corpus-note", default="", metavar="TEXT")
    args = p.parse_args(argv)
    try:
        report = audit(load_corpus(args.ai), load_corpus(args.human),
                       Config.load(), args.audience)
    except (OSError, ValueError) as exc:
        print(f"sloprefine: {exc}", file=sys.stderr)
        return 2
    if args.save:
        from .calibration import Calibration
        cal = Calibration.from_audit(report.as_dict(), note=args.corpus_note)
        Path(args.save).write_text(cal.to_json(), encoding="utf-8")
        print(cal.summary(), file=sys.stderr)
    if args.json:
        import json
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(report.render())
    return 0


def _cmd_check(args) -> int:
    if not args.paths:
        print("sloprefine: no input files", file=sys.stderr)
        return 2

    calibration = None
    if args.calibration:
        from .calibration import Calibration
        try:
            calibration = Calibration.load(args.calibration)
        except (OSError, ValueError) as exc:
            print(f"sloprefine: calibration: {exc}", file=sys.stderr)
            return 2

    config = Config.load()
    unknown = set(args.disable) - set(RULES)
    if unknown:
        print(f"sloprefine: unknown rule(s): {sorted(unknown)}", file=sys.stderr)
        return 2

    voiceprint = None
    voice_path = args.voice or getattr(config, "voice_path", None)
    if voice_path:
        try:
            voiceprint = voice_mod.Voiceprint.load(voice_path)
        except (OSError, ValueError) as exc:
            print(f"sloprefine: voiceprint: {exc}", file=sys.stderr)
            return 2

    allow = set(config.allow) | {w.lower() for w in args.allow}
    if args.allow_domain_words:
        allow |= COMMONLY_LEGITIMATE
    config = Config(
        disabled=tuple(set(config.disabled) | set(args.disable)),
        allow=frozenset(allow),
        max_hits=args.max_hits if args.max_hits is not None else config.max_hits,
        max_per_1k=(args.max_per_1k if args.max_per_1k is not None
                    else config.max_per_1k),
        max_choppiness=(args.max_choppiness if args.max_choppiness is not None
                        else config.max_choppiness),
        min_sentence_words=(
            args.min_sentence if args.min_sentence is not None
            else (5 if args.strict_runts and not config.min_sentence_words
                  else config.min_sentence_words)),
        runt_mode="all" if args.strict_runts else config.runt_mode,
        closer_budget_ratio=(args.closer_budget if args.closer_budget is not None
                             else config.closer_budget_ratio),
        parallel_budget_per_1k=(args.parallel_budget
                                if args.parallel_budget is not None
                                else config.parallel_budget_per_1k),
        allow_runts=config.allow_runts,
        skip_code_blocks=not args.check_code_blocks and config.skip_code_blocks,
        voice=voiceprint,
        z_threshold=args.z_threshold,
        audience=args.audience or config.audience,
        calibration=calibration,
        doublet_budget_per_1k=config.doublet_budget_per_1k,
        person_budget_per_1k=config.person_budget_per_1k,
    )

    # Applied LAST, over the finished config. Applying it earlier meant the
    # Config rebuilt from args above silently discarded every profile setting
    # whose field was not named there.
    if args.profile:
        from .punch import PROFILES, apply
        config = apply(args.profile, config)
        pins = PROFILES[args.profile].pins
        if pins:
            from .calibration import Calibration
            calibration = calibration or Calibration(
                weights={}, enrichment={}, verdicts={})
            calibration.pins = dict(pins)
            config = replace(config, calibration=calibration)

    results = []
    for path in args.paths:
        if path == "-":
            results.append(analyze("<stdin>", sys.stdin.read(), config))
            continue
        f = Path(path)
        if not f.is_file():
            print(f"sloprefine: no such file: {path}", file=sys.stderr)
            return 2
        results.append(analyze(str(f), f.read_text(encoding="utf-8"), config))

    if args.lm:
        _report_lm(results, args.lm)

    fmt = "json" if args.json else args.format
    if fmt == "agent":
        from .agent import review
        print("\n\n".join(
            review(r.path, r.text, config).render() for r in results))
    elif fmt == "json":
        print(render_json(results))
    else:
        color = sys.stdout.isatty() and not args.no_color
        print("\n\n".join(
            render_text(r, verbose=not args.quiet, color=color) for r in results
        ))

    over = any(
        (config.max_hits is not None and r.total > config.max_hits)
        or (config.max_per_1k is not None and r.per_1k > config.max_per_1k)
        or (config.max_choppiness is not None and r.cadence is not None
            and r.cadence.choppiness > config.max_choppiness)
        for r in results
    )
    return 1 if over else 0


def _report_lm(results, model_name: str) -> None:
    from .scorer import TransformersScorer, measure
    try:
        scorer = TransformersScorer(model_name)
    except ImportError as exc:
        print(f"sloprefine: {exc}", file=sys.stderr)
        return
    for r in results:
        signal = measure(scorer, r.text)
        print(f"{r.path}: perplexity {signal.perplexity:.1f} under "
              f"{signal.scorer}, window CV {signal.window_cv:.2f} "
              f"({signal.token_count} tokens). Signal only, not a verdict.")


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except BrokenPipeError:
        # Piped into head/less. Silence the flush-on-exit traceback.
        try:
            sys.stdout.close()
        except BrokenPipeError:
            pass
        return 0
    except KeyboardInterrupt:
        return 130


def _main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0] if argv and argv[0] in COMMANDS else "check"
    if argv and argv[0] in COMMANDS:
        argv = argv[1:]

    if command == "rules":
        return _cmd_rules()
    if command == "prompt":
        return _cmd_prompt(argv)
    if command == "drift":
        return _cmd_drift(argv)
    if command == "audit":
        return _cmd_audit(argv)
    if command == "voice":
        return _cmd_voice(build_voice_parser().parse_args(argv))

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.paths:
        parser.print_usage(sys.stderr)
        print("sloprefine: no input files", file=sys.stderr)
        return 2
    return _cmd_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
