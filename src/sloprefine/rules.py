"""Rule definitions.

Every rule carries the source that motivated it. If a rule has no citation it
does not belong in this file; see CONTRIBUTING.md.

  [K25]  Kobak et al., Science Advances 2025. "Delving into LLM-assisted
         writing in biomedical publications through excess vocabulary."
         Excess-frequency style words in 15M PubMed abstracts.
  [JW25] Juzek & Ward, COLING 2025. "Why Does ChatGPT 'Delve' So Much?"
         21 focal words with LLM-attributable frequency spikes.
  [LHF]  arXiv:2508.01930. Word overuse traced to learning from human feedback.
  [WP]   Wikipedia:Signs of AI writing. Community catalogue of markers.
  [PALV] palvdm.com/blog/ai-writing-tells (2026). Structural cluster:
         parallel negation, tricolon, uniform length, em dashes, recaps.
  [CL]   Cherryleaf (2026). Tricolon obsession, binary opposition, flat affect.
  [GK]   George Kao, "How To Write Without Sounding Like AI." Length variance,
         triads for rhythm, smooth transitions, generic over concrete.
  [FB]   Forbes, "The Seven Tells Of AI Writing." TED-talk punchline cadence.
  [AE]   The Augmented Educator, "Ten Telltale Signs." The em dash, the
         "from X to Y" range template.
  [local] No citation: a floor set by the user for their own writing. Kept
         separate from the sourced rules so the distinction stays visible.
  [KUM23] Kumarage et al., arXiv:2303.03697. Stylometric detection: n-gram
         repetition, phraseology, punctuation families.
  [SLH26] Shan, Lee & Hao, arXiv:2608.27855 (2026). Stylometric footprint of
         AI generation vs AI editing. See stylometry.py.
  [LZM23] Liang et al., Patterns 2023. Detector bias against non-native
         English writers. Motivates voice.py and the absence of absolute
         stylometric thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    citation: str
    severity: str = "medium"  # high | medium | low
    why: str = ""
    fix: str = ""  # imperative instruction for a revising model
    patterns: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------- vocabulary

# [K25][JW25][LHF][WP]. Inflections listed explicitly; matching is exact-token
# so that e.g. "delve" does not fire on "delved" unless "delved" is present.
LEXICON: frozenset[str] = frozenset(
    ["delve", "delves", "delving", "delved", "underscore", "underscores", "underscoring", "underscored", "showcase", "showcases", "showcasing", "showcased", "pivotal", "crucial", "crucially", "realm", "realms", "intricate", "intricacies", "intricately", "meticulous", "meticulously", "nuanced", "nuance", "robust", "comprehensive", "comprehensively", "multifaceted", "holistic", "leverage", "leverages", "leveraging", "harness", "harnesses", "harnessing", "landscape", "landscapes", "tapestry", "testament", "beacon", "myriad", "plethora", "paramount", "vital", "noteworthy", "commendable", "invaluable", "groundbreaking", "transformative", "unprecedented", "profound", "profoundly", "seamless", "seamlessly", "unwavering", "cutting-edge", "state-of-the-art", "game-changer", "aligns", "aligning", "boasts", "garnered", "garner", "surpasses", "surpassing", "advancements", "underpinning", "fostering", "cultivate", "elevate", "elevates", "streamline", "streamlining", "empower", "empowering", "unlock", "unlocking", "resonate", "resonates", "reimagine", "redefine", "revolutionize", "spearhead", "ever-evolving", "fast-paced"]
)

# Words in LEXICON that are load-bearing in some fields. Not excluded by
# default; listed so a project can allowlist them with one line of config.
COMMONLY_LEGITIMATE: frozenset[str] = frozenset(
    ["robust", "landscape", "vital", "realm", "nuance", "nuanced", "comprehensive", "harness"]
)


RULES: dict[str, Rule] = {}


def _rule(**kw) -> None:
    r = Rule(**kw)
    RULES[r.id] = r


_rule(
    id="vocab",
    title="LLM excess vocabulary",
    citation="[K25][JW25][LHF][WP]",
    severity="high",
    why="Words whose frequency spiked in post-2022 text with no other cause.",
    fix="Replace with the plainest word that carries the meaning. Do not substitute another word from the same register.",
)

_rule(
    id="emdash",
    title="em dash",
    citation="[AE][PALV]",
    severity="medium",
    why="The 'ChatGPT dash'. A comma or full stop usually does the job.",
    fix="Use a comma, a full stop, or parentheses.",
    patterns=(r"[\u2014\u2013]", r"(?<=\s)-{2,}(?=\s)"),
)

_rule(
    id="negation",
    title="parallel negation / binary opposition",
    citation="[PALV][CL]",
    severity="high",
    why="'Not just X, but Y'. Effective once, formulaic on repeat.",
    fix="State the positive claim directly. Delete the negated half.",
    patterns=(
        r"\b(?:is|it'?s|this is|that'?s)\s+not\s+(?:just|only|merely|simply)\b",
        r"\b(?:isn'?t|aren'?t|wasn'?t|doesn'?t|don'?t)\s+(?:just|only|merely|simply)\b",
        r"\bnot\s+(?:just|only|merely)\s+\w+[^.]{0,40}\bbut\b",
        r"\bnot\s+(?:about|because of)\s+\w+[^.]{0,40},\s*(?:it'?s|but)\b",
        r"\bthis (?:is|was) not an?\b",
        r"\bit'?s not (?:a|an|the)\b[^.]{0,40}\bit'?s (?:a|an|the)\b",
        # "This isn't a treatment, it's a language."
        (
            r"\b(?:isn'?t|aren'?t|wasn'?t|weren'?t)\s+(?:a|an|the)\b[^.]{0,50},"
            r"\s*(?:it'?s|they'?re|that'?s|it is)\b"
        ),
    ),
)

_rule(
    id="fragments",
    title="fragment stack (TED cadence)",
    citation="[FB]",
    severity="medium",
    why="Three or more very short sentences in a row, building to a punchline.",
    fix="Join at least two of the fragments into one longer sentence. Vary the lengths deliberately.",
)

_rule(
    id="transitions",
    title="stock transitions",
    citation="[WP][GK]",
    severity="medium",
    why="Connectives that exist to look organized.",
    fix="Delete the connective. If the logical link is unclear without it, rewrite the sentence.",
    patterns=(
        r"\bmoreover\b", r"\bfurthermore\b", r"\bconsequently\b",
        r"\bin addition,", r"\badditionally\b", r"\bnotably,",
        r"\bimportantly,", r"\bindeed,", r"\bthus,", r"\bhence,",
        r"\bit is (?:important|worth) (?:to note|noting)\b",
        r"\bit'?s worth noting\b", r"\bin today'?s\b", r"\bin conclusion\b",
        r"\bultimately,", r"\bin essence\b", r"\bat its core\b",
    ),
)

_rule(
    id="narrator",
    title="self-labelled beat",
    citation="[PALV][FB]",
    severity="high",
    why="Announcing the rhetorical move instead of making it.",
    fix="Delete the announcement and let the point land unannounced.",
    patterns=(
        r"\bhere'?s the thing\b",
        r"\bhere (?:is|are) the (?:test|point|part|result|kicker)\b",
        r"\bthe point is\b",
        r"\bwhat'?s (?:remarkable|striking|fascinating|interesting) (?:is|here)\b",
        r"\bthat'?s the (?:paradox|puzzle|point|kicker|twist|rub)\b",
        r"\band (?:it'?s|that'?s) my favou?rite\b",
        r"\blet me be clear\b", r"\bi want to be clear\b", r"\bto be clear,",
        r"\bmake no mistake\b", r"\bthe (?:real|bigger|deeper) question is\b",
        r"\bwhich (?:brings|takes) (?:us|me) (?:to|back)\b",
        r"\bbut here'?s\b", r"\bthe bottom line\b",
    ),
)

_rule(
    id="participial",
    title="participial summary tail",
    citation="[GK][K25]",
    severity="medium",
    why="', highlighting the importance of...' tacked onto a finished sentence.",
    fix="Cut the trailing clause or promote it to its own sentence with a subject.",
    patterns=(
        (
            r",\s+(?:highlighting|underscoring|showcasing|demonstrating|reflecting|"
            r"ensuring|emphasizing|illustrating|revealing|marking|cementing|paving|"
            r"signaling|signalling|solidifying)\b"
        ),
    ),
)

_rule(
    id="colon",
    title="colon reveal",
    citation="[PALV]",
    severity="low",
    why="Setup colon, then the payoff. Fine once per document.",
    fix="Fold the payoff into the sentence, or split into two sentences without the colon.",
)

_rule(
    id="question",
    title="rhetorical question opener",
    citation="[PALV]",
    severity="medium",
    why="Opening a section with a question instead of stating the point.",
    fix="Open with the claim. Delete the question.",
)

_rule(
    id="recap",
    title="recap phrase",
    citation="[PALV]",
    severity="medium",
    why="Restating what was just said, in the same words.",
    fix="Delete the recap. The reader just read it.",
    patterns=(
        r"\bin (?:short|summary|other words)\b",
        r"\bto (?:recap|summarize|summarise|sum up)\b",
        r"\bwhat this means is\b", r"\bthe takeaway (?:here )?is\b",
    ),
)

_rule(
    id="hedge",
    title="hedge boilerplate",
    citation="[WP]",
    severity="low",
    why="Caveats that carry no information.",
    fix="Delete the hedge, or replace it with the specific uncertainty it is standing in for.",
    patterns=(
        r"\bit'?s important to (?:note|remember|understand)\b",
        r"\bthat said,", r"\bthat being said\b",
        r"\bat the end of the day\b", r"\bin many ways\b",
        r"\bto some extent\b", r"\bit'?s worth (?:remembering|considering)\b",
    ),
)

_rule(
    id="fromto",
    title="'from X to Y' range template",
    citation="[AE]",
    severity="medium",
    why="A stock way to gesture at breadth without naming anything.",
    fix="Name the specific cases instead of gesturing at a range.",
    patterns=(
        r"\bfrom\s+[\w-]+(?:\s+[\w-]+){0,2}\s+to\s+[\w-]+(?:\s+[\w-]+){0,2}\b(?=[,.;])",
    ),
)

_rule(
    id="doublet",
    title="balanced pair over budget",
    citation="[CL][PALV]",
    severity="high",
    why="Two units of near-equal length sharing a shape: anaphora, a "
        "repeated opening, or an antithesis. The tell is balance, and "
        "balance starts at two, not three.",
    fix="Break the symmetry. Make one half longer, drop one half, or fold "
        "the pair into a single clause.",
)

_rule(
    id="closer",
    title="paragraph ends on a punch",
    citation="[FB][PALV]",
    severity="high",
    why="A short closing sentence arriving in the same slot paragraph after "
        "paragraph. Positional, so rephrasing the punch does not satisfy it.",
    fix="End the paragraph on the substantive sentence. Move the short one "
        "earlier, or fold it into the sentence before it.",
)

_rule(
    id="vague",
    title="paragraph with no specifics",
    citation="[GK][local]",
    severity="high",
    why="A paragraph of 40+ words containing no number, year, unit, proper "
        "noun or quoted term. A requirement, not a prohibition: it cannot be "
        "satisfied by cutting words.",
    fix="Name the thing. A number, a date, a proper noun or a quoted term. "
        "Do not invent one: an unchecked figure is worse than a vague phrase.",
)

_rule(
    id="parallel",
    title="parallelism over budget",
    citation="[CL][PALV][GK]",
    severity="high",
    why="A repeated syntactic skeleton at any arity. Banning three items "
        "just moves a generator to four; this counts the shape instead, and "
        "allows a budget because parallelism is a real device.",
    fix="Break the shape. Rewrite one unit with a different structure, or "
        "collapse the run into a single sentence. Adding a fourth item does "
        "not help.",
)

_rule(
    id="person",
    title="generic second person",
    citation="[local]",
    severity="medium",
    why="'You' standing in for 'one' or 'we'. It puts the audience inside a "
        "procedure they did not run, and in a talk about your own work it "
        "hands that work away. Real address to the room is not flagged.",
    fix="Say who did it. Use we or our for work you did, or rewrite the "
        "clause impersonally.",
)

_rule(
    id="runt",
    title="sentence under the floor",
    citation="[FB][local]",
    severity="high",
    why="A sentence too short to carry a clause. Default floor is 5 words, "
        "and by default only verbless ones are refused: 'One plot.' is a "
        "fragment, 'It worked.' is a sentence.",
    fix="Fold it into the sentence beside it, or give it a subject and a "
        "finite verb.",
)

_rule(
    id="opener",
    title="repeated sentence opener",
    citation="[GK][KUM23]",
    severity="medium",
    why="Three or more consecutive sentences starting with the same word.",
    fix="Rewrite at least one sentence to begin differently. Vary the subject position.",
)

_rule(
    id="template",
    title="repeated phrase template",
    citation="[KUM23][WP]",
    severity="medium",
    why="The same 4-word frame reused; n-gram repetition is a known marker.",
    fix="Rephrase all but one occurrence. Reuse of a frame reads as filler.",
)

_rule(
    id="emoji",
    title="decorative emoji",
    citation="[WP]",
    severity="low",
    why="Section-marker emoji in prose.",
    fix="Remove.",
)


# Inverse metric [CL]: writing about work you care about leaves fingerprints.
FOOTPRINT_PATTERNS: tuple[str, ...] = (
    r"\bi (?:didn'?t|did not) (?:believe|expect|think)\b",
    r"\bi expected\b", r"\bwe assumed\b", r"\bwe thought\b",
    r"\bi was (?:wrong|surprised|skeptical|sceptical)\b",
    r"\bwent looking for\b", r"\bi haven'?t found\b",
    r"\bi don'?t (?:have|know)\b", r"\bharder than i expected\b",
    r"\bi'?d rather\b", r"\bi wouldn'?t defend\b", r"\btook us longest\b",
    r"\bi'?m not claiming\b", r"\bmostly out of curiosity\b",
    r"\bi would genuinely\b", r"\bwe lose the most ground\b",
    r"\bit turned out\b", r"\bto my surprise\b",
    r"\bi (?:assumed|suspected|guessed|worried)\b",
    r"\bi (?:don'?t|do not|didn'?t|did not) know\b",
    r"\bi still (?:don'?t|do not)\b",
    r"\bi'?m afraid (?:of|that)\b",
    r"\bnobody had a (?:reason|good answer)\b",
)
