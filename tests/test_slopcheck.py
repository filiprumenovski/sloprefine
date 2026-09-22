from pathlib import Path

import pytest

from slopcheck import Document, analyze, compute
from slopcheck.checks import CHECKS, run_checks
from slopcheck.cli import main
from slopcheck.report import Config
from slopcheck.rules import RULES

CORPUS = Path(__file__).resolve().parents[1] / "corpus"
CFG = Config()


def ids(text):
    return [h.rule_id for h in run_checks(Document(text))]


# ------------------------------------------------------------- tokenization

def test_sentence_split_respects_abbreviations():
    doc = Document("Dr. Fehl runs the lab. He is in Room 421. That is all.")
    assert len(doc.sentences) == 3


def test_sentence_split_respects_decimals():
    doc = Document("The effect was 3.5 fold. Then it fell.")
    assert len(doc.sentences) == 2


def test_offsets_point_at_source():
    text = "We delve into it."
    hit = run_checks(Document(text))[0]
    assert text[hit.start:hit.end] == "delve"


# -------------------------------------------------------------------- rules

@pytest.mark.parametrize("text,rule", [
    ("We delve into the data.", "vocab"),
    ("It was pivotal.", "vocab"),
    ("A thought \u2014 interrupted.", "emdash"),
    ("This isn't just chemistry.", "negation"),
    ("It's not about the enzyme, it's the substrate.", "negation"),
    ("Fast, cheap, and reliable.", "tricolon"),
    ("No order. No motif. No structure.", "tricolon"),
    ("One gene. One site. Ten substrates. But scale matters here somehow.", "fragments"),
    ("Moreover, the data held.", "transitions"),
    ("Here's the thing: it worked.", "narrator"),
    ("The run failed, highlighting the need for care.", "participial"),
    ("The reason: nobody checked the buffer.", "colon"),
    ("In short, we were wrong.", "recap"),
    ("That said, the trend is real.", "hedge"),
    ("Results \U0001F680 were strong.", "emoji"),
])
def test_rule_fires(text, rule):
    assert rule in ids(text), f"{rule} did not fire on {text!r}"


@pytest.mark.parametrize("text", [
    "The buffer was cold and the gel ran slowly that afternoon.",
    "She asked why we measured at thirty minutes and nobody had an answer.",
    "I ran it again with fresh reagent and got the same nothing.",
])
def test_no_false_positives_on_plain_prose(text):
    assert ids(text) == []


def test_every_rule_has_a_check_and_a_citation():
    assert set(RULES) == set(CHECKS)
    for rule in RULES.values():
        assert rule.citation.startswith("["), rule.id
        assert rule.severity in {"high", "medium", "low"}


# ------------------------------------------------------------------ corpora

def test_slop_control_lights_up():
    result = analyze("slop", (CORPUS / "slop_control.txt").read_text(), CFG)
    assert result.total >= 20
    fired = {rid for rid, n in result.counts().items() if n}
    assert {"vocab", "negation", "tricolon", "narrator", "transitions"} <= fired


def test_clean_control_is_silent():
    result = analyze("clean", (CORPUS / "clean_control.txt").read_text(), CFG)
    assert result.total == 0, [(h.rule_id, h.text) for h in result.hits]


def test_clean_control_has_footprint_and_variance():
    doc = Document((CORPUS / "clean_control.txt").read_text())
    m = compute(doc)
    assert m.footprint > 0
    assert m.cv >= 0.45


# ------------------------------------------------------------------ config

def test_allowlist_suppresses_lexicon_hit():
    cfg = Config(allow=frozenset({"landscape"}))
    assert analyze("x", "The landscape shifted.", cfg).total == 0


def test_disable_turns_a_rule_off():
    cfg = Config(disabled=("vocab",))
    assert analyze("x", "We delve into it.", cfg).total == 0


def test_config_rejects_unknown_rule(tmp_path):
    (tmp_path / ".slopcheck.toml").write_text(
        '[slopcheck]\ndisable = ["nonsense"]\n'
    )
    with pytest.raises(ValueError):
        Config.from_toml(tmp_path / ".slopcheck.toml")


# --------------------------------------------------------------------- cli

def test_cli_exit_code_gates(tmp_path, capsys):
    f = tmp_path / "a.txt"
    f.write_text("We delve into the intricate landscape.")
    assert main([str(f), "--max-hits", "0", "--no-color"]) == 1
    assert main([str(f), "--max-hits", "10", "--no-color"]) == 0


def test_cli_json_is_parseable(tmp_path, capsys):
    import json
    f = tmp_path / "a.txt"
    f.write_text("Here's the thing: we delve.")
    main([str(f), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["hits"][0]["line"] == 1
    assert "citation" in payload[0]["hits"][0]


def test_cli_missing_file_is_usage_error(tmp_path):
    assert main([str(tmp_path / "nope.txt")]) == 2


# -------------------------------------------------------------- suppression

def test_off_on_region_is_skipped():
    text = "We delve here.\n<!-- slopcheck: off -->\nWe delve here too.\n<!-- slopcheck: on -->\nAnd delve again."
    assert analyze("x.md", text, CFG).total == 2


def test_off_without_on_runs_to_end_of_file():
    text = "We delve here.\nslopcheck: off\nWe delve. We delve."
    assert analyze("x.txt", text, CFG).total == 1


def test_markdown_code_blocks_are_exempt_by_default():
    text = "Plain line.\n\n```\nWe delve into the intricate realm.\n```\n"
    assert analyze("x.md", text, CFG).total == 0
    # as .txt the fence is just prose: 3 lexicon hits plus a fragment stack
    assert analyze("x.txt", text, CFG).total == 4
    assert analyze("x.md", text, Config(skip_code_blocks=False)).total == 4


def test_readme_passes_its_own_linter():
    readme = Path(__file__).resolve().parents[1] / "README.md"
    result = analyze(str(readme), readme.read_text(), CFG)
    assert result.total == 0, [(h.rule_id, h.text) for h in result.hits]
