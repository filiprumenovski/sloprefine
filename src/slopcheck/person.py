"""Person: who the sentence says did the work.

Two different "you"s
--------------------
    You find an O-GlcNAc site on a protein and mutate the serine.
    I would genuinely like you to try to break it.

The first is generic. It means "one", or it means "we", and it puts the
audience inside a procedure they did not run. It is the second person of a
TED transcript and of a recipe, and in a talk about work someone actually did
it hands that work away.

The second is real address. The speaker is talking to the room.

Only the first is worth flagging, and the two are not distinguishable by
counting pronouns. The heuristic here: a second-person pronoun is ADDRESS
when the same sentence carries a first-person singular pronoun (the speaker
is present, so there is a real I talking to a real you), when the sentence is
a question (a question to the room is address), or when it is an imperative
appeal. Otherwise it is generic, and the fix is to say who actually did it.

Wrong sometimes in both directions. "You'd expect a smear" has no I and is
flagged, though it is closer to "one would expect" than to a procedure. That
is why this is budgeted rather than banned, and why it is off by default.

This has no citation behind it. It is a register preference, tagged [local]
like the sentence floor, and it is on in the talk profile because a
conference talk about your own work should say we.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .text import Document, Span

# "thank you" is a fixed phrase, neither generic nor address. Without this
# the closing line of every talk is a hit.
_FIXED = re.compile(r"\bthanks?\s+you\b|\bthank\s+you\b", re.IGNORECASE)

SECOND = re.compile(r"\b(?:you|your|you'?re|you'?d|you'?ll|you'?ve)\b", re.IGNORECASE)
FIRST_SINGULAR = re.compile(r"\b(?:i|i'?m|i'?d|i'?ve|i'?ll|me|my|mine)\b", re.IGNORECASE)
FIRST_PLURAL = re.compile(r"\b(?:we|we'?re|we'?d|we'?ve|us|our|ours)\b", re.IGNORECASE)

# Openings that make a sentence a real appeal to the room rather than a
# procedure the listener is being walked through.
APPEAL = re.compile(
    r"^(?:please\b|let me\b|let's\b|i'?d\b|i would\b|i want\b|i know\b|"
    r"look\b|listen\b|imagine\b|consider\b|think about\b|remember\b)", re.IGNORECASE)


@dataclass
class PersonMix:
    words: int
    first_plural: int        # we / our / us
    first_singular: int      # I / my
    second_generic: int      # "you" standing in for "one" or "we"
    second_address: int      # "you" meaning the room
    generic_per_1k: float

    def as_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        return (f"we/our {self.first_plural}  I/my {self.first_singular}  "
                f"generic-you {self.second_generic} "
                f"({self.generic_per_1k:.1f}/1k)  "
                f"addressed-you {self.second_address}")


def _strip_fixed(text: str) -> str:
    return _FIXED.sub(" ", text)


def is_address(span: Span) -> bool:
    """True when the second person in this sentence means the room."""
    text = span.text.strip()
    if text.endswith("?"):
        return True
    if APPEAL.match(text):
        return True
    return bool(FIRST_SINGULAR.search(text))


def generic_spans(doc: Document) -> list[Span]:
    """Sentences using a generic second person."""
    return [s for s in doc.sentences
            if SECOND.search(_strip_fixed(s.text)) and not is_address(s)]


def compute(doc: Document) -> PersonMix:
    generic = address = 0
    for span in doc.sentences:
        found = len(SECOND.findall(_strip_fixed(span.text)))
        if not found:
            continue
        if is_address(span):
            address += found
        else:
            generic += found
    words = doc.word_count or 1
    return PersonMix(
        words=words,
        first_plural=len(FIRST_PLURAL.findall(doc.text)),
        first_singular=len(FIRST_SINGULAR.findall(doc.text)),
        second_generic=generic,
        second_address=address,
        generic_per_1k=round(generic / words * 1000, 2),
    )
