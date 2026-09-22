"""Local alignment for parallelism, in place of hand-written shape rules.

The problem the heuristics kept hitting
---------------------------------------
Two units read as parallel when they share a structure, but the shared
structure rarely covers the whole unit. The first item of a list carries the
sentence stem, the last carries the tail, and a middle item can be padded
without breaking what a reader hears:

    It was fast, cheap and reliable.
       ^^^^^^ stem on item one
    serine, threonine and proline as the acceptors
                                 ^^^^^^^^^^^^^^^^ tail on item three

`parallel.py` handles both with special cases: trim the stem off the first
segment, trim the tail off the last, match the middles on a bucketed skeleton.
That is three heuristics approximating one operation, and each was added after
a false negative in real text.

The operation is local alignment. Smith-Waterman finds the highest-scoring
matching subsequence of two sequences and ignores the flanks, which is exactly
"these two units share a shape in the middle and differ at the edges". Aligning
token CLASSES rather than tokens (function words as themselves, content words
as a wildcard) makes it structural rather than lexical, so "for the data" and
"for the model" align perfectly while "the buffer was cold" and "nobody had
checked the timer" do not.

This is the same algorithm as sequence alignment in a search engine, run over a
four-letter alphabet of token classes instead of twenty of residues.

Result: measured, and it loses
------------------------------
NOT WIRED INTO THE CHECKS. It is kept because a negative result nobody
recorded gets repeated.

Scored against the same battery the heuristics pass, normalized alignment
score, two encodings tried (a two-symbol class alphabet, then one that keeps
the literal token wherever it repeats across the pair):

    parallel pairs    0.25 - 0.83
    ordinary pairs    0.17 - 0.62

The distributions overlap with no usable threshold between them. "The gel ran
slowly / the room stayed cold all afternoon" scores 0.62 and is ordinary
prose; "That isn't what's there / it isn't a close call" scores 0.25 and is
the construction this whole module exists to catch.

The reason is that the signal in these pairs is lexical repetition plus
length balance, not structural similarity. Alignment is built to find shared
structure through noisy edges, and two English clauses of similar shape share
plenty of structure without being parallel in the sense a reader hears.
Smith-Waterman is the right algorithm for a different problem than this one.

What the heuristics in parallel.py do instead, and why they win here: they
key on the repeated token itself (anaphora, a shared negator, a repeated
two-word opening) and use length only as a balance check. Cruder, and it
separates the cases cleanly.

Cost, if anyone does find a use: O(mn) per pair with both capped at MAX_UNIT.
"""

from __future__ import annotations

from dataclasses import dataclass

MATCH = 2.0
MISMATCH = -1.0
GAP = -1.5
MAX_UNIT = 20

# A match on a function word is worth more than a match on a content-word
# wildcard: "for the" aligning with "for the" is structure, while two
# unrelated nouns aligning is coincidence.
WILDCARD = "#"
WILDCARD_MATCH = 1.0


@dataclass(frozen=True)
class Alignment:
    score: float
    length: int
    identity: float      # aligned length over the shorter unit
    normalized: float    # score over the best achievable score


SAME_CONTENT_SLOT = 0.5


def _pair_score(a: str, b: str) -> float:
    if a == b:
        return WILDCARD_MATCH if a == WILDCARD else MATCH
    if a == WILDCARD or b == WILDCARD:
        return MISMATCH
    return MISMATCH


def align(a: list[str], b: list[str]) -> Alignment:
    """Smith-Waterman local alignment of two token-class sequences."""
    a, b = a[:MAX_UNIT], b[:MAX_UNIT]
    if not a or not b:
        return Alignment(0.0, 0, 0.0, 0.0)

    rows, cols = len(a) + 1, len(b) + 1
    matrix = [[0.0] * cols for _ in range(rows)]
    best, best_cell = 0.0, (0, 0)

    for i in range(1, rows):
        for j in range(1, cols):
            diagonal = matrix[i - 1][j - 1] + _pair_score(a[i - 1], b[j - 1])
            value = max(0.0, diagonal, matrix[i - 1][j] + GAP,
                        matrix[i][j - 1] + GAP)
            matrix[i][j] = value
            if value > best:
                best, best_cell = value, (i, j)

    # traceback, to measure how long the aligned region is
    i, j = best_cell
    length = 0
    while i > 0 and j > 0 and matrix[i][j] > 0:
        diagonal = matrix[i - 1][j - 1] + _pair_score(a[i - 1], b[j - 1])
        if matrix[i][j] == diagonal:
            i, j, length = i - 1, j - 1, length + 1
        elif matrix[i][j] == matrix[i - 1][j] + GAP:
            i -= 1
        else:
            j -= 1

    shorter = min(len(a), len(b))
    ceiling = shorter * MATCH
    return Alignment(
        score=round(best, 2),
        length=length,
        identity=round(length / shorter, 3) if shorter else 0.0,
        normalized=round(best / ceiling, 3) if ceiling else 0.0,
    )


def parallel_score(a: list[str], b: list[str]) -> float:
    """One number for how parallel two units are, in [0, 1]."""
    return align(a, b).normalized
