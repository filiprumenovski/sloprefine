"""slopcheck command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .report import Config, analyze, render_json, render_text
from .rules import COMMONLY_LEGITIMATE, RULES

EPILOG = """\
exit codes:
  0  under threshold
  1  over threshold (use --max-hits / --max-per-1k, or config, to gate CI)
  2  bad invocation

config: .slopcheck.toml, searched upward from cwd

  [slopcheck]
  disable = ["colon", "hedge"]
  allow = ["landscape", "robust"]   # domain words exempt from the lexicon
  max_hits = 0
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="slopcheck",
        description="Lint prose against published AI-writing tells.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("paths", nargs="*", help="files to check; - for stdin")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="counts only, no matched spans")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--disable", action="append", default=[], metavar="RULE")
    p.add_argument("--allow", action="append", default=[], metavar="WORD",
                   help="exempt a word from the lexicon check")
    p.add_argument("--allow-domain-words", action="store_true",
                   help=f"exempt the usual false positives: "
                        f"{', '.join(sorted(COMMONLY_LEGITIMATE))}")
    p.add_argument("--max-hits", type=int, metavar="N")
    p.add_argument("--max-per-1k", type=float, metavar="X")
    p.add_argument("--check-code-blocks", action="store_true",
                   help="do not exempt fenced code blocks in Markdown")
    p.add_argument("--list-rules", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_rules:
        for rule in RULES.values():
            print(f"{rule.id:<13} {rule.severity:<7} {rule.title}")
            print(f"{'':<13} {rule.citation}  {rule.why}")
        return 0

    if not args.paths:
        build_parser().print_usage(sys.stderr)
        print("slopcheck: no input files", file=sys.stderr)
        return 2

    config = Config.load()
    unknown = set(args.disable) - set(RULES)
    if unknown:
        print(f"slopcheck: unknown rule(s): {sorted(unknown)}", file=sys.stderr)
        return 2

    disabled = tuple(set(config.disabled) | set(args.disable))
    allow = set(config.allow) | {w.lower() for w in args.allow}
    if args.allow_domain_words:
        allow |= COMMONLY_LEGITIMATE
    config = Config(
        disabled=disabled,
        allow=frozenset(allow),
        max_hits=args.max_hits if args.max_hits is not None else config.max_hits,
        max_per_1k=(args.max_per_1k if args.max_per_1k is not None
                    else config.max_per_1k),
        skip_code_blocks=not args.check_code_blocks and config.skip_code_blocks,
    )

    results = []
    for path in args.paths:
        if path == "-":
            results.append(analyze("<stdin>", sys.stdin.read(), config))
            continue
        f = Path(path)
        if not f.is_file():
            print(f"slopcheck: no such file: {path}", file=sys.stderr)
            return 2
        results.append(analyze(str(f), f.read_text(encoding="utf-8"), config))

    if args.json:
        print(render_json(results))
    else:
        color = sys.stdout.isatty() and not args.no_color
        print("\n\n".join(
            render_text(r, verbose=not args.quiet, color=color) for r in results
        ))

    failed = False
    for r in results:
        if config.max_hits is not None and r.total > config.max_hits:
            failed = True
        if config.max_per_1k is not None and r.per_1k > config.max_per_1k:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
