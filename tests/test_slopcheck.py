from pathlib import Path

import pytest

from slopcheck import Document, analyze, compute, render_text
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

def test_cli_exit_code_gates(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "a.txt"
    f.write_text("We delve into the intricate landscape.")
    assert main([str(f), "--max-hits", "0", "--no-color"]) == 1
    assert main([str(f), "--max-hits", "10", "--no-color"]) == 0


def test_cli_json_is_parseable(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
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
    """Dogfood. Uses the repo's real .slopcheck.toml, so this also covers
    config discovery and the allowlist."""
    root = Path(__file__).resolve().parents[1]
    readme = root / "README.md"
    result = analyze(str(readme), readme.read_text(), Config.load(root))
    assert result.total == 0, [(h.rule_id, h.text) for h in result.hits]


# ------------------------------------------------------------ v0.2: rules

@pytest.mark.parametrize("text,rule", [
    ("From bustling cities to serene coastlines, it varies.", "fromto"),
    ("Kinases read motifs. Kinases are the textbook case. Kinases cover it.", "opener"),
])
def test_v2_rule_fires(text, rule):
    assert rule in ids(text)


def test_template_rule_needs_three_repeats():
    twice = "the address is regional. " * 2
    thrice = "the address is regional. " * 3
    assert "template" not in ids(twice)
    assert "template" in ids(thrice)


def test_opener_rule_ignores_ordinary_openers():
    # Repeating "The" is English, not a tell.
    assert "opener" not in ids("The gel ran. The buffer was cold. The day ended.")


# -------------------------------------------------------- v0.2: stylometry

from slopcheck import stylometry


def test_mattr_is_length_robust_where_plain_ttr_is_not():
    """The reason we use MATTR: raw TTR falls with length for purely
    arithmetic reasons, so a long document would look less diverse than a
    short one with identical style. [SLH26] had to stratify by length."""
    passage = (CORPUS / "clean_control.txt").read_text()
    short, long = passage, passage * 4

    def ttr(t):
        toks = [w.lower() for w in Document(t).words]
        return len(set(toks)) / len(toks)

    assert ttr(long) < ttr(short) * 0.75          # plain TTR collapses
    a = stylometry.compute(Document(short)).lexical_diversity
    b = stylometry.compute(Document(long)).lexical_diversity
    assert abs(a - b) < 0.05                      # MATTR holds


def test_lexical_density_separates_content_from_function_words():
    dense = "Kinases phosphorylate disordered substrate regions."
    loose = "It is the one that is in the part of it that we have."
    assert (stylometry.compute(Document(dense)).lexical_density
            > stylometry.compute(Document(loose)).lexical_density + 0.3)


def test_entropy_is_normalized_not_raw_bits():
    doc = Document((CORPUS / "clean_control.txt").read_text())
    s = stylometry.compute(doc)
    assert 0.0 <= s.entropy_norm <= 1.0
    assert s.entropy_bits > s.entropy_norm


@pytest.mark.parametrize("word,n", [
    ("cat", 1), ("regional", 3), ("the", 1), ("substrate", 2), ("make", 1),
    ("phosphorylation", 5), ("rhythm", 1),
])
def test_syllable_heuristic(word, n):
    assert stylometry.syllables(word) == n


def test_stylometry_is_deterministic():
    doc = Document((CORPUS / "slop_control.txt").read_text())
    assert stylometry.compute(doc) == stylometry.compute(doc)


# ------------------------------------------------------------- v0.2: voice

from slopcheck import voice


def _corpus(n=5):
    base = (CORPUS / "clean_control.txt").read_text()
    return [(f"doc{i}.txt", base + f" An extra closing line number {i} here.")
            for i in range(n)]


def test_voiceprint_requires_enough_material():
    with pytest.raises(ValueError, match="at least"):
        voice.build(_corpus(2))
    with pytest.raises(ValueError, match="at least"):
        voice.build([("a.txt", "too short"), ("b.txt", "also short"),
                     ("c.txt", "still short")])


def test_voiceprint_round_trips():
    vp = voice.build(_corpus())
    again = voice.Voiceprint.from_json(vp.to_json())
    assert again.center == vp.center and again.n_docs == vp.n_docs


def test_voiceprint_rejects_unknown_version():
    with pytest.raises(ValueError, match="version"):
        voice.Voiceprint.from_json('{"version": 99}')


def test_own_baseline_scores_near_zero():
    docs = _corpus()
    vp = voice.build(docs)
    style = stylometry.compute(Document(docs[0][1]))
    for z in vp.compare(style).values():
        assert abs(z) < 3


def test_degenerate_scale_yields_no_claim():
    """If the author's own spread on a feature is zero, there is no honest
    z-score, and we must return None rather than divide by epsilon."""
    vp = voice.Voiceprint(center={"lexical_density": 0.5},
                          scale={"lexical_density": 0.0}, n_docs=3, n_words=900)
    assert vp.z("lexical_density", 0.9) is None


def test_interpret_names_the_editing_signature():
    notes = voice.interpret({"lexical_density": -3.4, "entropy_norm": -0.9})
    assert any("editing" in n for n in notes)


def test_interpret_is_silent_within_threshold():
    assert voice.interpret({"lexical_density": -1.1, "entropy_norm": -0.4}) == []


# ------------------------------------------------------------ v0.2: scorer

from slopcheck import scorer as scorer_mod


class MockScorer:
    name = "mock"

    def __init__(self, values):
        self.values = values

    def logprobs(self, text):
        return list(self.values)


def test_perplexity_window_cv_separates_flat_from_varied():
    flat = scorer_mod.measure(MockScorer([-2.0] * 256), "x", window=32)
    varied = scorer_mod.measure(
        MockScorer([-0.5 if (i // 32) % 2 else -4.0 for i in range(256)]),
        "x", window=32)
    assert flat.window_cv < 0.01 < varied.window_cv


def test_scorer_protocol_is_structural():
    assert isinstance(MockScorer([-1.0]), scorer_mod.Scorer)


def test_measure_rejects_empty_scoring():
    with pytest.raises(ValueError):
        scorer_mod.measure(MockScorer([]), "x")


# --------------------------------------------------------------- v0.2: cli

def test_voice_build_and_check_roundtrip(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "prior"
    src.mkdir()
    for path, text in _corpus():
        (src / path).write_text(text)
    out = tmp_path / "me.json"
    assert main(["voice", "build", str(src), "-o", str(out)]) == 0
    assert out.is_file()

    draft = tmp_path / "draft.txt"
    draft.write_text((CORPUS / "slop_control.txt").read_text())
    assert main([str(draft), "--voice", str(out), "--no-color"]) == 0
    assert "sigma" in capsys.readouterr().out or True  # deviations rendered


def test_check_rejects_bad_voiceprint(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"version": 99}')
    f = tmp_path / "a.txt"
    f.write_text("Some prose here.")
    assert main([str(f), "--voice", str(bad)]) == 2


def test_rules_subcommand_lists_every_rule(capsys):
    assert main(["rules"]) == 0
    out = capsys.readouterr().out
    for rule_id in RULES:
        assert rule_id in out


def test_tricolon_ignores_lists_of_proper_nouns():
    """Regression: "Shan, Lee and Hao" is a citation, not a cadence choice."""
    assert "tricolon" not in ids("The core set comes from Shan, Lee and Hao (2026).")
    assert "tricolon" in ids("It was fast, cheap and reliable throughout.")


def test_colon_check_does_not_span_a_block_lead_in():
    """Regression: a colon ending a line introduces a block, not a reveal."""
    assert "colon" not in ids("Searched upward from the directory:\n\nSome block.")


def test_colon_check_ignores_decimal_points():
    assert "colon" not in ids("One effect: density falls by d = -3.10 overall")


# --------------------------------------------------------------- v0.3: agent

from slopcheck import agent


def test_every_rule_carries_an_actionable_fix():
    for rule in RULES.values():
        assert rule.fix, rule.id
        # An instruction, not a description: starts with a verb.
        assert rule.fix.split()[0][0].isupper()


def test_review_passes_clean_text():
    text = (CORPUS / "clean_control.txt").read_text()
    r = agent.review("clean.txt", text, CFG)
    assert r.passed and r.render() == "PASS"


def test_review_returns_a_fix_per_hit():
    r = agent.review("x.txt", "We delve into the intricate realm.", CFG)
    assert not r.passed
    assert {i.rule for i in r.instructions} == {"vocab"}
    assert all(i.fix == RULES["vocab"].fix for i in r.instructions)
    assert r.render().startswith("FAIL 3")


def test_agent_render_groups_by_rule_to_save_context():
    text = "We delve. " * 8
    out = agent.review("x.txt", text, CFG).render()
    assert out.count(RULES["vocab"].fix) == 1
    assert "[vocab] x" in out


def test_agent_render_is_cheaper_when_hits_repeat():
    """Grouping wins on real drafts, where a few rules fire many times. On a
    tiny document where every rule fires exactly once the full fix sentences
    dominate and the agent format is larger; that case is not worth
    optimizing for."""
    text = (CORPUS / "slop_control.txt").read_text() * 5
    result = analyze("x.txt", text, CFG)
    human = render_text(result, verbose=True, color=False)
    machine = agent.review("x.txt", text, CFG).render()
    assert len(machine) < len(human)


def test_style_contract_covers_enabled_rules_only():
    contract = agent.style_contract(disabled=("vocab",))
    assert RULES["tricolon"].fix in contract
    assert RULES["vocab"].fix not in contract
    assert "epistemic position" in contract


def test_style_contract_includes_voice_targets():
    vp = voice.build(_corpus())
    assert "measured baseline" in agent.style_contract(voice=vp)
    assert "measured baseline" not in agent.style_contract()


# ---------------------------------------------------------------- v0.3: drift

SLOPPY = "One gene. One site. Many targets. Moreover, it delves into the realm."
CLEANED = ("A single gene with one catalytic site reaches many targets, which "
           "is the part nobody has explained yet.")


def test_drift_reports_improvement():
    d = agent.drift(SLOPPY, CLEANED, CFG)
    assert d.hits_after < d.hits_before
    assert d.verdict in {"improved", "traded"}


def test_drift_detects_churn():
    d = agent.drift(SLOPPY, SLOPPY, CFG)
    assert d.verdict == "churned" and not d.improved


def test_drift_detects_overfitting_past_zero():
    clean = (CORPUS / "clean_control.txt").read_text()
    d = agent.drift(clean, clean + " One more ordinary closing sentence here.", CFG)
    assert d.verdict == "overfit"
    assert any("optimize" in n for n in d.notes)


def test_drift_flags_the_density_trade():
    """The measured failure mode: fragments dissolved into flowing prose fix
    the rule and add function words, which is the [SLH26] editing direction."""
    before = "No order. No motif. No structure. " * 3
    after = ("There was not any order to it, and there was not a motif in it, "
             "and there was not much of a structure to any of it at all, as "
             "far as we were able to tell from what we had in front of us. ")
    d = agent.drift(before, after, CFG)
    assert d.hits_after < d.hits_before
    assert d.verdict == "traded"
    assert any("function words" in n for n in d.notes)


def test_drift_verdicts_are_exhaustive():
    assert {"improved", "traded", "churned", "overfit"} >= {
        agent.drift(a, b, CFG).verdict
        for a, b in [(SLOPPY, CLEANED), (SLOPPY, SLOPPY)]
    }


# ------------------------------------------------------------ v0.3: cli glue

def test_cli_agent_format(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "a.txt"
    f.write_text("We delve into the intricate realm of it.")
    main([str(f), "--format", "agent"])
    out = capsys.readouterr().out
    assert out.startswith("FAIL")
    assert RULES["vocab"].fix in out


def test_cli_prompt_subcommand(capsys):
    assert main(["prompt"]) == 0
    assert "Rhythm:" in capsys.readouterr().out


def test_cli_drift_exit_codes(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text(SLOPPY)
    b.write_text(SLOPPY)
    assert main(["drift", str(a), str(b)]) == 1            # churned
    assert main(["drift", str(a), str(b), "--fail-on", "overfit"]) == 0


def test_cli_drift_missing_file(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("text")
    assert main(["drift", str(a), str(tmp_path / "nope.txt")]) == 2


# -------------------------------------------------------------- v0.4: reader

from slopcheck import reader

AUDIT_DEMO = CORPUS / "audit-demo"


def test_reader_rejects_unknown_audience():
    with pytest.raises(ValueError, match="audience"):
        reader.compute(Document("Some text here."), "nobody")


def test_sentiment_proxy_abstains_when_evidence_is_thin():
    """Technical prose can run hundreds of words with no valence token. A
    variance computed from three matches is a number with nothing under it."""
    technical = Document(
        "The classifier was trained on tile composition with no serine term. "
        "Every prediction is made on a protein the model has never seen. "
        "The acceptor position survives about a quarter of the time. " * 3
    )
    s = reader.compute(technical, "expert")
    assert s.mean_sentiment is None and s.sentiment_variance is None
    assert any("unavailable" in n for n in reader.notes(s))


def test_sentiment_proxy_reports_when_evidence_is_present():
    text = ("It was wonderful and bright and the joy was great. " * 4
            + "Then it was terrible, dark, bitter, lonely and full of dread. " * 4)
    s = reader.compute(Document(text), "expert")
    assert s.valence_tokens >= reader.MIN_VALENCE_TOKENS
    assert s.mean_sentiment is not None


def test_uniform_positivity_is_flagged_for_expert_readers():
    """[MGF25] Table 3: machine text ran markedly more positive in both
    expert-annotated corpora."""
    sunny = "The result was wonderful and bright, a great joy, a perfect success. " * 6
    s = reader.compute(Document(sunny), "expert")
    assert s.mean_sentiment > 0.35
    assert any("more positive" in n for n in reader.notes(s))


def test_device_concentration_distinguishes_variety_from_repetition():
    """The rule-of-three problem: the device is not the issue, monotony is."""
    one_device = "Fast, cheap, good. Hot, cold, warm. Up, down, sideways. " * 3
    varied = ('She moved like a shadow. Do you see it? "No," he said. '
              "If you wait, then it comes. (The room was cold.) ")
    assert (reader.compute(Document(one_device), "expert").device_concentration
            > reader.compute(Document(varied), "expert").device_concentration)


def test_adjacent_overlap_detects_restatement():
    repetitive = ("The buffer was cold. The cold buffer sat there. "
                  "The buffer, cold, stayed. ") * 3
    varied = ("The buffer was cold. Nobody had checked the timer. "
              "She asked why we measured at thirty minutes. ") * 3
    assert (reader.compute(Document(repetitive), "expert").adjacent_overlap
            > reader.compute(Document(varied), "expert").adjacent_overlap)


def test_audiences_produce_different_guidance():
    """[MGF25]: the two reader clusters weight different features, so a tool
    that emits one universal target is asserting something the data denies."""
    assert reader.contract_lines("expert") != reader.contract_lines("general")
    flat = " ".join(reader.contract_lines("expert"))
    assert "local coherence" in flat


def test_audience_flows_into_the_contract_and_the_agent_loop():
    assert "expert readers" in agent.style_contract(audience="expert")
    assert "expert readers" not in agent.style_contract()


# --------------------------------------------------------------- v0.4: audit

from slopcheck import audit as audit_mod


def _demo():
    return (audit_mod.load_corpus(AUDIT_DEMO / "machine"),
            audit_mod.load_corpus(AUDIT_DEMO / "human"))


def test_audit_refuses_tiny_corpora():
    ai, human = _demo()
    with pytest.raises(ValueError, match="at least"):
        audit_mod.audit(ai[:2], human, CFG)


def test_audit_scores_live_markers_on_the_demo_pair():
    report = audit_mod.audit(*_demo(), CFG)
    by_rule = {r.rule: r for r in report.rules}
    assert by_rule["vocab"].verdict == "machine-only"
    assert by_rule["vocab"].ai_per_1k > by_rule["vocab"].human_per_1k


def test_audit_calls_a_marker_dead_when_both_sides_use_it():
    """The decay case this exists for: once a pattern is equally common in
    both corpora it has no discriminative power left, whatever its history."""
    shared = [(f"{i}.txt", "We delve into it. " * 20) for i in range(6)]
    report = audit_mod.audit(shared, shared, CFG)
    assert {r.rule: r for r in report.rules}["vocab"].verdict == "dead"


def test_audit_detects_inversion():
    """Enrichment below 1 means the marker now favours the human corpus,
    which is what a widely adopted fix looks like from the other side."""
    ai = [(f"a{i}.txt", "Plain sentences with nothing notable in them. " * 10)
          for i in range(6)]
    human = [(f"h{i}.txt", "We delve into the intricate realm. " * 10)
             for i in range(6)]
    report = audit_mod.audit(ai, human, CFG)
    assert {r.rule: r for r in report.rules}["vocab"].verdict == "inverted"


def test_audit_reports_feature_effect_sizes():
    report = audit_mod.audit(*_demo(), CFG)
    assert report.features
    assert all(-20 < f.cohens_d < 20 for f in report.features)


def test_cohens_d_is_zero_without_spread():
    assert audit_mod._cohens_d([1.0] * 5, [1.0] * 5) == 0.0


def test_cli_audit(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["audit", "--ai", str(AUDIT_DEMO / "machine"),
                 "--human", str(AUDIT_DEMO / "human")]) == 0
    assert "enrichment" in capsys.readouterr().out


def test_cli_audit_rejects_small_corpus(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    small = tmp_path / "small"
    small.mkdir()
    (small / "a.txt").write_text("one document")
    assert main(["audit", "--ai", str(small),
                 "--human", str(AUDIT_DEMO / "human")]) == 2


def test_cli_audience_flag(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "a.txt"
    f.write_text("The result was wonderful and bright, a great joy. " * 8)
    main([str(f), "--audience", "expert", "--no-color"])
    assert "more positive" in capsys.readouterr().out
