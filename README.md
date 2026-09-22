# slopcheck

A linter for prose, in the spirit of `clippy` or `ruff`, built to sit inside a
model's own write-check-revise loop.

The markers of machine-generated writing are documented in a dozen papers and
a hundred blog posts. A model can be told to avoid them up front, and its
output can be checked against them afterwards. Neither step needs a human in
the middle, and neither step needs a detector making claims about authorship.

```bash
# 1. constrain generation
slopcheck prompt --voice me.json > .style-contract

# 2. check the output, get imperative fixes back
slopcheck check draft.md --format agent

# 3. decide whether the revision actually helped
slopcheck drift draft.md draft.v2.md
```

Step 3 is the one people skip, and the reason this repository exists at all.
Optimizing a draft toward zero hits does not monotonically improve it. On the
revision this tool was built for, rule hits fell from 24 to 1 while lexical
density fell 0.530 to 0.493, which is the direction Shan et al. measure for
AI-*edited* text. Dissolving fragment stacks into flowing sentences adds
function words by construction. Trading one measured marker for another is not
progress, so the loop needs something to stop on besides zero.

```
$ slopcheck check talk.txt --format agent
FAIL 22
[fragments] x9: Join at least two of the fragments into one longer sentence. Vary the lengths deliberately.
  L5 Specificity lives in the enzyme. / OGT breaks that. / One gene. / O...
  L11 We tested that. / It's wrong. / And it isn't close.
  L15 Same proteins. / Same residue types. / Matched null.
  ... 6 more
[narrator] x5: Delete the announcement and let the point land unannounced.
  L17 That's the paradox
  L25 Here is the test
[tricolon] x4: Use two items or four, or dissolve the list into a clause.
  L23 No order. No motif. No structure.
! 9 consecutive sentences of near-identical length; break the metronome

$ slopcheck drift talk.txt talk.v2.txt
TRADED hits 22->1 density 0.530->0.493
! lexical density fell 0.037 while fixing 21 hit(s): the revision added
  function words. [SLH26] measures that drop as the AI-editing signature
```

Output is grouped by rule so the fix instruction is emitted once rather than
once per hit: 1,543 characters for the agent format against 2,842 for the
human report and 7,636 for JSON, on the same document.

Three layers under the hood:

1. **Rules.** Sixteen lexical and structural checks, each citing its source,
   each carrying an imperative fix a model can act on.
2. **Stylometry.** Model-free features from the detection literature,
   reported with no thresholds attached.
3. **Voiceprints.** A baseline built from your own writing. In a generation
   loop this is a *target*, not a defense: `slopcheck prompt --voice me.json`
   puts your measured sentence length, lexical density and contraction rate
   in front of the model before it writes a word.

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
slopcheck prompt --voice me.json          # targets, before generation
slopcheck check draft.md --voice me.json  # deviations, after
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

## Wiring it into an agent

Exit codes are the interface: `check` returns 1 over threshold, `drift`
returns 1 on `churned`, `overfit` or `traded` by default.

```bash
#!/usr/bin/env bash
# revise.sh: loop with a stop condition that is not "zero hits"
cp draft.md work.md
for round in 1 2 3; do
  slopcheck check work.md --format agent > feedback.txt || true
  grep -q '^PASS$' feedback.txt && break
  your-model --instructions feedback.txt --input work.md --output next.md
  if ! slopcheck drift work.md next.md; then
    echo "round $round did not improve the draft; keeping the previous version"
    break
  fi
  mv next.md work.md
done
```

Three rounds is the hard ceiling in `agent.MAX_ROUNDS`, and the reason is the
trade above: past a couple of passes the model is writing for the linter.

As a library:

```python
import slopcheck

review = slopcheck.review("draft.md", text)
if not review.passed:
    feedback = review.render()          # grouped, imperative, compact
    revised = my_model(text, feedback)
    d = slopcheck.drift(text, revised)
    text = revised if d.improved else text
```

For a persistent agent config, put the contract where the model reads it:

```bash
slopcheck prompt --voice me.json >> AGENTS.md
```

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
