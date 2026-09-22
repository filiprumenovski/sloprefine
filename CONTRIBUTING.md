# Contributing

## Adding a rule

Every rule needs a `fix`: one imperative sentence a model can act on without
reading the rationale. `why` explains the rule to a person; `fix` is what gets
shipped into the revision loop and into `slopcheck prompt`. A test asserts
every rule has one.


Every rule needs a source. Not a vibe, a source: a study, a corpus analysis, or
a written argument someone put their name on. Add the short key and full
reference to the docstring at the top of `src/slopcheck/rules.py`, then declare
the rule with `_rule(...)` and give it a `why` that a writer can act on.

A rule also needs:

1. A positive test in `tests/test_slopcheck.py::test_rule_fires`.
2. No new hits on `corpus/clean_control.txt`. That file is a regression fence.
   `test_clean_control_is_silent` asserts zero. If your rule fires there,
   either the rule is too broad or the fence was wrong; argue for one.

## False positives are the whole game

A prose linter that cries wolf gets uninstalled in a week, and then it protects
nobody. Prefer a rule that misses half the instances over one that fires on
ordinary writing. If a word is load-bearing in some field, add it to
`COMMONLY_LEGITIMATE` rather than dropping it from the lexicon.

## Adding a stylometric feature

Different bar. Features go in `stylometry.py` and must be computable with no
language model, no API call, and no corpus-level statistics, so that a draft
can be checked offline on a laptop and the number means the same thing
tomorrow.

Length-correct anything derived from token counts. Type-token ratio and
Shannon entropy both fall with document length for arithmetic reasons, so an
uncorrected feature measures how long the document is and reports it as style.

Do not add a threshold. `stylometry.py` ships no population cutoffs, and the
reason is in `voice.py`: a cutoff transferred between corpora is the documented
failure mode of deployed detectors, and it lands hardest on non-native
speakers. If a feature is only meaningful against a threshold, it belongs in
the voiceprint layer as a deviation, not in the report as a judgement.

## Running everything

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```
