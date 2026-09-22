# slopcheck

A linter for prose, in the spirit of `clippy` or `ruff`, except the lints are
the published markers of machine-generated writing.

It will not tell you whether an LLM wrote something. That question cannot be
answered from the text alone, and the tools claiming to answer it ruin people
for a living. `slopcheck` answers a narrower question: **which specific spans
in this draft carry the patterns readers have learned to distrust, and how far
has this draft drifted from the way you normally write.**

Three layers, in increasing order of how much they can tell you and how wrong
they can be:

1. **Rules.** Sixteen lexical and structural checks, each citing the study or
   write-up it came from, each pointing at a line and an offset.
2. **Stylometry.** Model-free features from the AI-detection literature,
   reported with no thresholds attached.
3. **Voiceprints.** A baseline built from your own prior writing, against
   which a draft is scored in standard deviations. This is the layer that
   makes the stylometry mean anything.

```
$ slopcheck talk.txt --voice me.json
talk.txt  (1297 words)
  ok  LLM excess vocabulary                0  [K25][JW25][LHF][WP]
  HIT tricolon / triad rhythm              1  [CL][PALV][GK]
       L23  No order, no motif, no structure. (anaphora x3)
  ok  fragment stack (TED cadence)         0  [FB]
  ...
  ------------------------------------------------------------
  sentences=92  mean=14.1w  sd=8.0  CV=0.57  flat-run=4
  short<=7w 24%   long>=25w 13%   footprint=14 (10.79/1k words)
  ------------------------------------------------------------
  stylometry (value, deviation from your voiceprint)
    lexical diversity            0.8199    +0.4s
    entropy (norm)               0.8808    -0.6s
    lexical density              0.4927    -2.4s
  voice lexical density -2.4sigma below your baseline: more function words per
        content word than you normally write. [SLH26] reports this pair as the
        AI-editing signature
  total 1  (0.77 per 1k words)
```

## Install

```bash
pip install -e ".[dev]"
slopcheck draft.md
```

Python 3.11+. No runtime dependencies. The optional `lm` extra pulls in
`transformers` and `torch`, and nothing else needs them.

## The three layers

### Rules

<!-- slopcheck: off (this table quotes the patterns it documents) -->

| rule | catches | source |
|---|---|---|
| `vocab` | delve, underscore, showcase, pivotal, crucial, intricate, realm, tapestry | Kobak et al. 2025; Juzek & Ward 2025 |
| `emdash` | the "ChatGPT dash" | Augmented Educator; palvdm |
| `negation` | "not just X, but Y", "it isn't A, it's B" | palvdm; Cherryleaf |
| `tricolon` | "fast, cheap, and reliable"; threefold anaphora | Cherryleaf; Kao |
| `fragments` | three or more very short sentences in a row | Forbes |
| `transitions` | Moreover, Furthermore, It is important to note | Wikipedia; Kao |
| `narrator` | "Here's the thing", "That's the paradox", "Let me be clear" | palvdm; Forbes |
| `participial` | ", highlighting the need for..." | Kao; Kobak |
| `fromto` | "From bustling cities to serene coastlines" | Augmented Educator |
| `opener` | three consecutive sentences opening on the same word | Kao; Kumarage et al. |
| `template` | the same four-word frame reused three times | Kumarage et al. |
| `colon` | setup colon, then the payoff | palvdm |
| `question` | sections opening with a rhetorical question | palvdm |
| `recap` | "In short", "The takeaway is" | palvdm |
| `hedge` | "That said", "At the end of the day" | Wikipedia |
| `emoji` | rockets in prose | Wikipedia |

<!-- slopcheck: on -->

`slopcheck rules` prints all of them with citations and rationale.

### Stylometry

Eight model-free features, computed per document with no language model, no
API, and no corpus statistics. The core set comes from Shan, Lee and Hao
(arXiv:2608.27855), who tested 14 such features over 45,000 texts across 8
LLMs and 5 domains.

Their central finding inverts the folk theory. AI-generated text has **higher**
lexical diversity and **higher** entropy than human writing, not lower, and
those two features are the only ones that stay important across every
generator and domain they tried.

Their second finding matters more for anyone actually using this tool. AI
*editing* does not reproduce that footprint. Text a human wrote and an LLM
polished shows only a slight diversity rise, an entropy *drop*, and one large
effect on lexical density, which falls by d = -3.10 as the proportion of
content words to function words collapses. Their classifiers separate AI-edited from
AI-generated text at AUC 0.98, but manage only 0.80 separating AI-edited from
human. Editing is not a weak form of generation. It is a different thing with
a different signature, and editing is what most people are doing.

Two implementation departures from the paper, both deliberate:

- **Length correction.** Type-token ratio and entropy both fall with document
  length for arithmetic reasons, which is why the authors had to stratify
  their corpora by length. Rather than stratify, `slopcheck` uses a
  moving-average TTR over a fixed window and normalizes entropy by log2 of
  the type count. A regression test asserts that plain TTR collapses on a
  quadrupled document while the corrected measure holds. Our numbers are
  therefore not their numbers.
- **No thresholds.** The paper reports everything z-scored against its own
  corpora. A cutoff lifted from one corpus and applied to another is exactly
  the failure documented by Liang et al. in *Patterns*, where detectors
  systematically misread non-native English writing as machine output because
  simpler word choice reads as low perplexity. So the stylometry layer ships
  no population cutoffs at all. Without a voiceprint it prints values and says
  nothing about them.

### Voiceprints

```bash
slopcheck voice build ~/writing/published -o me.json
slopcheck draft.md --voice me.json
```

A voiceprint is the robust center and spread (median and MAD) of each feature
across documents you wrote. A draft is scored in sigma units against your own
distribution, so the claim becomes "your lexical density is 2.4 sigma below
your own baseline", which is a statement about a change in your writing.
Without it the claim would be "your lexical density is 0.49", which is a
statement about you, and that is the kind of statement that gets people
accused of things.

Requires at least 3 documents of 150+ words. It refuses below that, because a
baseline from two paragraphs describes those two paragraphs.

### Optional: perplexity structure

```bash
pip install "slopcheck[lm]"
slopcheck draft.md --lm gpt2
```

Reports perplexity and, more usefully, how much perplexity varies across
windows of the document. Uniform predictability end to end is the flat-rhythm
finding one layer down. This is a number, never a verdict. The state of the
art is Binoculars (Hans et al., ICML 2024), whose perplexity to
cross-perplexity ratio reaches over 90% TPR at 0.01% FPR; its own authors ship
a fixed threshold and still caution against unsupervised use. Use their
implementation if you want detection. This is a writing tool.

## Configuration

`.slopcheck.toml`, searched upward from the working directory:

```toml
[slopcheck]
disable = ["colon", "hedge"]
allow = ["landscape", "robust"]   # domain words exempt from the lexicon
voice = "me.json"
max_hits = 0
```

The allowlist carries more weight than it looks. `landscape` is on the
excess-vocabulary list and is also a normal word in half a dozen fields, and
`robust` is how statisticians say robust. A linter that cannot be told this
gets uninstalled within a week, and then it protects nobody.
`--allow-domain-words` exempts the usual suspects in one flag.

Fenced code blocks in Markdown are exempt by default, and anything between
`<!-- slopcheck: off -->` and `<!-- slopcheck: on -->` is skipped, so
documentation that quotes the patterns does not fail its own lint. CI asserts
that this README scores zero.

## A worked example, including the embarrassing part

The tool started as a script for cleaning the machine cadence out of a
conference talk. On the rule layer it worked, taking that talk from 24 hits
down to 1.

Then the stylometry layer went in and got pointed at both versions. Lexical
density had gone from 0.530 to 0.493 and entropy had ticked down. That is the
direction Shan et al. measure for AI editing, produced by the exact pass that
removed the AI cadence.

Both things are true. The rewrite reads less like a machine and measures
slightly more like machine-edited text, because dissolving fragment stacks
into flowing sentences adds function words, which is what lexical density
counts. If you use this tool to clean a draft, expect that trade and watch the
density number.

## What this will not do

It will not tell you the writing is good. A draft scoring zero can be
lifeless, and a draft firing eight times can be the better piece. Several of
these patterns are effective rhetoric used deliberately and sparingly, and
fragment triads in particular work from a podium where a listener has no
rewind button. The tool counts. You decide.

It will not survive its own popularity. These are surface patterns, and
surface patterns move once enough people know them. The vocabulary list is
already aging. Treat the rules as a cited snapshot, not a law.

Do not use it to accuse anyone of anything. Not students, not applicants, not
colleagues. Liang et al. is the reason, and the harm falls hardest on the
writers who can least afford it.

## Contributing

New rules need a citation. See `CONTRIBUTING.md`.

## References

- Kobak, González-Márquez, Horvát and Lause, *Science Advances* 2025. Delving into LLM-assisted writing in biomedical publications through excess vocabulary.
- Juzek and Ward, COLING 2025. Why Does ChatGPT "Delve" So Much?
- Shan, Lee and Hao, arXiv:2608.27855 (2026). AI Writers Have a Consistent Stylometric Footprint, but AI Editors Do Not.
- Liang, Yuksekgonul, Mao, Wu and Zou, *Patterns* 2023. GPT detectors are biased against non-native English writers.
- Hans et al., ICML 2024. Spotting LLMs With Binoculars.
- Bao et al., ICLR 2024. Fast-DetectGPT.
- Mitchell et al., ICML 2023. DetectGPT.
- Verma, Fleisig, Tomlin and Klein, NAACL 2024. Ghostbuster.
- Kumarage et al., arXiv:2303.03697. Stylometric Detection of AI-Generated Text in Twitter Timelines.

## License

MIT.
