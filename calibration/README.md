# calibration

<!-- slopcheck: off -->

Per-rule weights measured from a labelled corpus pair, rather than the
severities assigned by hand in `rules.py`.

```bash
slopcheck audit --ai out/model --human ~/writing/mine --save mine.json
slopcheck check draft.md --calibration mine.json
```

## fiction-2023.json

68 documents: 17 human, 51 machine. Human side is New Yorker short stories
(TTCW) plus the human entries from Confederacy. Machine side is GPT-3.5,
GPT-4, Claude 1.3, and the 2023 Confederacy model set. Both corpora come from
Marco, Gonzalo & Fresno's collection (github.com/grmarco/the-reader-is-the-metric).

Measured enrichment, machine rate over human rate:

| rule | enrichment | verdict |
|---|---|---|
| vague | 14.9 | live |
| template | 8.3 | live |
| vocab | 6.0 | live |
| parallel | 2.8 | live |
| opener | 2.8 | live |
| doublet | 2.1 | live |
| negation, participial, hedge | machine-only, tiny rates | live |
| recap | 1.15 | dead |
| closer | 0.81 | dead |
| transitions | 0.68 | inverted |
| runt | 0.43 | inverted |
| emdash | 0.39 | inverted |
| fragments | 0.37 | inverted |
| colon | 0.33 | inverted |
| question | 0.23 | inverted |

Half the rule set does not discriminate here, and eight rules fire MORE on the
human side.

<!-- slopcheck: on -->

## Read the scope before believing any of it

There are two confounds here and both of them are large.

**Domain.** This is creative fiction. Short-story writers use fragments,
em dashes and one-word sentences constantly and deliberately, which is most
likely why those rules invert. A scientific talk is a different register and
the numbers may not transfer at all.

**Model vintage.** These are 2023 models. The em-dash habit and the staccato
cadence that this repository was built to catch are post-2023 behaviours, so
"inverted" here may mean "these models had not started doing it yet" rather
than "this is not a marker". That is exactly the decay the audit measures,
running in the other direction.

17 human documents is also a small sample.

This file demonstrates that the mechanism works end to end, and it shows that
half the hand-assigned severities in this package were wrong about something.
It will not calibrate your own writing, and for that you need to build one
out of your own corpora.
