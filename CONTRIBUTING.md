# Contributing

## Adding a rule

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

## Running everything

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```
