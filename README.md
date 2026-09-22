# slopcheck

A linter for prose, in the spirit of `clippy` or `ruff`, except the lints are
the published tells of machine-generated writing.

It does not try to tell you whether an LLM wrote something. That question is
unanswerable from the text alone and the commercial detectors that claim to
answer it are wrong often enough to ruin people. `slopcheck` answers a narrower
and more useful question: **which specific spans in this draft carry the
patterns that readers have learned to associate with generated text.**

Every rule cites the study or the write-up it comes from. Every hit points at a
line and an offset. Nothing is a score out of a hundred.

```
$ slopcheck talk.txt
talk.txt  (1265 words)
  HIT LLM excess vocabulary                 2  [K25][JW25][LHF][WP]
       L17  landscape
       L61  landscape
  ok  em dash                               0  [AE][PALV]
  ok  parallel negation / binary opposition 0  [PALV][CL]
  HIT tricolon / triad rhythm               2  [CL][PALV][GK]
       L25  no serine or threonine term, no spacing, no disorder score (anaphora x3)
       L47  no function, no localization, no pathway membership (anaphora x3)
  HIT fragment stack (TED cadence)          2  [FB]
       L33  Serine, threonine, proline. / Almost no aromatics. / Acceptors... (3 in a row)
  ...
  ------------------------------------------------------------
  sentences=91  mean=14.0w  sd=8.5  CV=0.6  flat-run=4
  short<=7w 29%   long>=25w 14%   footprint=14 (11.0/1k words)
  total 8  (6.32 per 1k words)
```

## Install

```bash
pip install -e ".[dev]"     # from a clone
slopcheck draft.md
```

Python 3.11+. No dependencies.

## Usage

```bash
slopcheck draft.md                     # report
slopcheck draft.md -q                  # counts only
slopcheck *.md --json                  # machine-readable, with offsets
cat draft.md | slopcheck -             # stdin
slopcheck --list-rules                 # what it checks and why
slopcheck draft.md --max-hits 0        # exit 1 if anything fires (CI gate)
```

## What it checks

<!-- slopcheck: off (this table quotes the patterns it documents) -->

| rule | what it catches | source |
|---|---|---|
| `vocab` | delve, underscore, showcase, pivotal, crucial, intricate, realm, tapestry | Kobak et al. 2025; Juzek & Ward 2025 |
| `emdash` | the "ChatGPT dash" | Augmented Educator; palvdm |
| `negation` | "not just X, but Y", "it isn't A, it's B" | palvdm; Cherryleaf |
| `tricolon` | "fast, cheap, and reliable"; three-fold anaphora | Cherryleaf; Kao |
| `fragments` | three or more very short sentences in a row | Forbes |
| `transitions` | Moreover, Furthermore, It is important to note | Wikipedia; Kao |
| `narrator` | "Here's the thing", "That's the paradox", "Let me be clear" | palvdm; Forbes |
| `participial` | ", highlighting the need for..." | Kao; Kobak |
| `colon` | setup colon, then the payoff | palvdm |
| `question` | sections that open with a rhetorical question | palvdm |
| `recap` | "In short", "The takeaway is" | palvdm |
| `hedge` | "That said", "At the end of the day" | Wikipedia |
| `emoji` | 🚀 in prose | Wikipedia |

<!-- slopcheck: on -->

Two metrics are reported rather than counted as hits:

- **burstiness (CV)**, the standard deviation of sentence length over the mean.
  Human nonfiction usually sits above 0.55. One-pass generated text clusters
  low because the sentences all land in the same 15-25 word band. A long
  `flat-run` says the same thing about a local stretch of the draft.
- **footprint**, meaning first-person epistemic markers. Things the author
  noticed, expected, doubted, or got wrong. This one runs the other way, so
  higher is better. Text
  with zero of these across 400+ words reads as having no author.

Fenced code blocks in Markdown are exempt by default, and a region between
`<!-- slopcheck: off -->` and `<!-- slopcheck: on -->` is skipped, so docs that
quote the patterns do not fail their own lint. `--check-code-blocks` turns the
first part off.

## Configuration

`.slopcheck.toml`, searched upward from the working directory:

```toml
[slopcheck]
disable = ["colon", "hedge"]
allow = ["landscape", "robust"]   # domain words exempt from the lexicon
max_hits = 0
```

The allowlist matters more than it looks. `landscape` is a genuine term of art
in half a dozen fields, and `robust` is how statisticians say robust. A linter
that cannot be told this is a linter you stop running. `--allow-domain-words`
exempts the usual suspects in one flag.

## What this tool will not do

It will not tell you the writing is good. A draft that scores zero can be
lifeless, and a draft that fires eight times can be the better piece. Several
of these patterns are effective rhetoric used deliberately and sparingly:
fragment triads in particular work from a podium, where a listener has no
rewind button. The tool counts. You decide.

It will not survive contact with its own popularity. These are surface
patterns, and surface patterns move. The vocabulary list is already aging.
Treat the rules as a snapshot with citations attached, not as a law.

Do not use it to accuse anyone of anything.

## Contributing

New rules need a citation. See `CONTRIBUTING.md`.

## License

MIT.
