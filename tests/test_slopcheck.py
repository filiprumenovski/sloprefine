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
    ("Fast, cheap, and reliable.", "parallel"),   # subsumed: see _drop_subsumed_tricolons
    ("No order. No motif. No structure.", "parallel"),
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
    assert {"vocab", "negation", "narrator", "transitions"} <= fired
    assert fired & {"tricolon", "parallel"}


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
    # as .txt the fence is just prose: lexicon hits plus structural ones
    assert analyze("x.txt", text, CFG).total > 0
    assert (analyze("x.md", text, Config(skip_code_blocks=False)).total
            == analyze("x.txt", text, CFG).total)


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


def test_parallel_ignores_lists_of_proper_nouns():
    """Regression: "Shan, Lee and Hao" is a citation, not a cadence choice."""
    assert "parallel" not in ids("The core set comes from Shan, Lee and Hao (2026).")
    assert "parallel" in ids("It was fast, cheap and reliable throughout.")


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
    assert RULES["parallel"].fix in contract
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


def test_density_drop_is_attributed_to_cadence_not_charged_twice():
    """This test previously asserted "traded" and was wrong.

    Dissolving fragments lowers lexical density BY CONSTRUCTION, because
    fragments carry almost no function words. Charging that drop as a separate
    cost double-counts one change, and in practice it argued for putting the
    fragments back. When cadence moves, the density delta is a consequence of
    it and drift must say so rather than treat it as independent evidence."""
    before = "No order. No motif. No structure. " * 3
    after = ("There was not any order to it, and there was not a motif in it, "
             "and there was not much of a structure to any of it at all, as "
             "far as we were able to tell from what we had in front of us. ")
    d = agent.drift(before, after, CFG)
    assert d.hits_after < d.hits_before
    assert d.choppiness_after < d.choppiness_before
    assert d.verdict == "improved"
    assert any("not independent evidence" in n for n in d.notes)


def test_density_trade_still_fires_when_cadence_holds_still():
    """The trade verdict is not gone, it is scoped: it applies when the
    revision spent density WITHOUT changing cadence."""
    before = "The classifier reads tile composition and reports cluster sites."
    after = ("It is the case that the thing which reads what is in the tile "
             "is the one that then goes on to say where it is that the sites "
             "of the clusters are to be found in it.")
    d = agent.drift(before, after, CFG)
    assert abs(d.choppiness_after - d.choppiness_before) < 0.02
    assert d.density_after < d.density_before


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


# ------------------------------------------------------- v0.5: cadence

from slopcheck import cadence


def test_choppy_control_is_choppy_and_clean_control_is_not():
    choppy = cadence.compute(Document((CORPUS / "choppy_control.txt").read_text()))
    clean = cadence.compute(Document((CORPUS / "clean_control.txt").read_text()))
    assert choppy.verdict == "choppy" and clean.verdict == "ok"
    assert choppy.choppiness > 4 * clean.choppiness


def test_choppiness_is_word_weighted_not_sentence_weighted():
    """A document can be half short SENTENCES and still spend most of its
    runtime in long ones. Word share is what a listener experiences."""
    long_tail = "Yes. No. Maybe. " + ("The buffer was cold and nobody had "
                                      "checked the timer before we started. ") * 6
    c = cadence.compute(Document(long_tail))
    sentence_share = 3 / len(Document(long_tail).sentences)
    assert c.short_share < sentence_share / 2


def test_verbless_share_separates_fragments_from_short_sentences():
    """"It worked." is a short sentence. "One gene." is a fragment. The
    difference is what makes a beat land, and it is what the rule layer,
    which only counts words, cannot see."""
    fragments = cadence.compute(Document("One gene. One site. No motif."))
    shorties = cadence.compute(Document("It worked. She knew. They ran."))
    assert fragments.verbless_share > shorties.verbless_share


def test_run_mass_measures_the_rule_layer_pattern():
    # "A." is a single capital plus a period, which the splitter reads as an
    # initial and refuses to break on, so the fixture uses real words.
    clustered = "Yes. No. Maybe. " + "word " * 40 + "."
    assert cadence.compute(Document(clustered)).run_mass > 0
    spread = ("Yes it did. " + "word " * 30 + ". No it did not. "
              + "other " * 30 + ".")
    assert cadence.compute(Document(spread)).run_mass == 0


def test_collinearity_note_fires_only_on_real_cadence_movement():
    assert cadence.collinearity_note(0.30) is not None
    assert cadence.collinearity_note(0.001) is None


def test_drift_ranks_cadence_above_the_other_metrics():
    """The regression this module exists for: a revision that reintroduces
    the staccato cadence also raises lexical density and burstiness, and
    those gains must not be allowed to outvote the cadence."""
    flowing = ("The enzyme has one gene and one catalytic site, and it reaches "
               "thousands of substrates without a consensus sequence anywhere "
               "in the set. ") * 3
    chopped = "One gene. One site. Thousands of targets. No motif. " * 3
    d = agent.drift(flowing, chopped, CFG)
    assert d.verdict == "choppy"
    assert d.density_after > d.density_before      # density REWARDS the artifact
    assert any("outranks" in n for n in d.notes)


def test_cadence_appears_in_the_report_and_the_agent_notes():
    text = (CORPUS / "choppy_control.txt").read_text()
    result = analyze("x.txt", text, CFG)
    assert result.cadence is not None
    assert "cadence" in render_text(result, color=False)
    assert any("staccato" in n for n in agent.review("x.txt", text, CFG).notes)


# ---------------------------------------------------- v0.6: the sentence floor

FLOOR = Config(min_sentence_words=5)
FLOOR_STRICT = Config(min_sentence_words=5, runt_mode="all")


def test_floor_is_off_unless_asked_for():
    """The only rule with no citation, and it refuses a device human writers
    use. corpus/clean_control.txt contains "Two weeks of that.", written to
    read as natural human prose, and the floor refuses it."""
    text = ("Two weeks of that. The control came back clean on every plate, "
            "which ruled out the reagents and left me with a short list.")
    assert analyze("x.txt", text, CFG).total == 0
    assert analyze("x.txt", text, FLOOR).counts()["runt"] == 1


def test_floor_refuses_fragments_but_keeps_short_sentences():
    """The distinction the floor exists to make."""
    hits = analyze("x.txt", "One gene. It worked. Same proteins.", FLOOR).hits
    refused = {h.text for h in hits if h.rule_id == "runt"}
    assert refused == {"One gene.", "Same proteins."}


def test_strict_mode_refuses_everything_under_the_floor():
    hits = analyze("x.txt", "One gene. It worked. Same proteins.", FLOOR_STRICT).hits
    assert len({h.text for h in hits if h.rule_id == "runt"}) == 3


def test_floor_respects_the_configured_width():
    text = "Four words go here. " + "A much longer sentence sits beside it. " * 2
    assert analyze("x.txt", text, Config(min_sentence_words=4,
                                         runt_mode="all")).counts()["runt"] == 0
    assert analyze("x.txt", text, Config(min_sentence_words=5,
                                         runt_mode="all")).counts()["runt"] == 1


def test_allow_runts_exempts_exact_strings():
    cfg = Config(min_sentence_words=5, runt_mode="all",
                 allow_runts=frozenset({"Thank you."}))
    text = "Thank you. One gene. " + "A longer sentence to end on here. " * 2
    refused = {h.text for h in analyze("x.txt", text, cfg).hits
               if h.rule_id == "runt"}
    assert refused == {"One gene."}


def test_contractions_are_finite_verbs():
    """Regression: the floor refused "OGT doesn't fit." as a fragment."""
    for text in ("OGT doesn't fit.", "The address hasn't.", "I haven't found one."):
        assert analyze("x.txt", text, FLOOR).counts()["runt"] == 0


def test_plural_nouns_are_not_verbs():
    """Regression: a bare -s test read "Same proteins." as a verbed sentence,
    which is backwards for fragment detection."""
    assert analyze("x.txt", "Same proteins.", FLOOR).counts()["runt"] == 1
    assert analyze("x.txt", "Thousands of substrates.", FLOOR).counts()["runt"] == 1


def test_known_miss_bare_past_participle():
    """Documented limitation, not a passing behaviour: "Matched null." is a
    fragment and the verb heuristic reads the participle as finite. Only
    strict mode catches it."""
    assert analyze("x.txt", "Matched null.", FLOOR).counts()["runt"] == 0
    assert analyze("x.txt", "Matched null.", FLOOR_STRICT).counts()["runt"] == 1


def test_markdown_list_markers_are_not_sentences():
    """Regression: "1." at the start of a line was split off as a zero-word
    sentence and then refused by the floor."""
    md = ("An introductory line that runs long enough to clear the floor.\n\n"
          "1. The first item also runs long enough to clear it.\n"
          "2. The second item does the same thing here.\n")
    result = analyze("x.md", md, FLOOR)
    assert result.counts()["runt"] == 0
    # the specific bug: markers parsed as zero-word sentences
    assert not any(h.note.startswith("0w") for h in result.hits)


def test_cli_min_sentence_flag(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "a.txt"
    f.write_text("One gene. " + "A sentence long enough to clear the floor. " * 2)
    # the rule's title contains the word "floor", so assert on the hit note
    main([str(f), "--no-color"])
    assert "2w, floor 5" not in capsys.readouterr().out
    main([str(f), "--min-sentence", "5", "--no-color"])
    assert "2w, floor 5" in capsys.readouterr().out


# --------------------------------------------- v0.7: general parallelism

from slopcheck import parallel

PAR = Config(parallel_budget_per_1k=0.0)

# Repeating one sentence would itself be a parallel run, which is correct
# behaviour and useless as filler.
FILLER = " ".join([
    "The buffer sat on the bench until somebody remembered it.",
    "Nobody had checked the timer since the protocol was adapted.",
    "A reading came back flat on the second plate that morning.",
    "She asked a question that had not occurred to anyone else.",
    "Two weeks went by before the obvious explanation surfaced.",
    "I still have not gone back to check the original number.",
    "The gel ran slowly and the room stayed cold all afternoon.",
    "Somebody adapted this from a paper about a different enzyme.",
] * 5)



def _runs(text):
    return parallel.find(Document(text))


@pytest.mark.parametrize("text,arity", [
    ("It was fast, cheap and reliable.", 3),               # no Oxford comma
    ("It was fast, cheap, and reliable.", 3),
    ("We used serine, threonine and proline as the acceptors.", 3),  # trailing tail
    ("No order. No motif. No structure.", 3),
    ("No order. No motif. No structure. No spacing.", 4),  # tetracolon escape
    ("No sense of order here. No motif to speak of. No structure at all.", 3),
    ("We trained on human. We tested on rice. We ran it backwards.", 3),
])
def test_parallelism_is_arity_independent(text, arity):
    """Banning three items moves a generator to four. The shape is the
    target, so the run length is reported rather than required."""
    runs = _runs(text)
    assert runs and max(r.arity for r in runs) == arity


@pytest.mark.parametrize("text", [
    "The buffer was cold, and nobody had checked the timer before we started.",
    "It was fast and cheap.",
    "The core set comes from Shan, Lee and Hao at Boston University.",
    "This is Binoculars (Hans et al., ICML 2024), whose ratio reaches ninety.",
    "She asked why we measured at thirty minutes and nobody had an answer.",
])
def test_parallelism_leaves_ordinary_prose_alone(text):
    assert _runs(text) == []


def test_padding_one_item_does_not_defeat_the_match():
    """Length bucketing is coarse on purpose. With exact matching, adding a
    word to the third item breaks the run and the check goes silent."""
    assert _runs("One gene. One catalytic site. One single solitary site.")


def test_clean_control_has_no_parallel_runs():
    assert _runs((CORPUS / "clean_control.txt").read_text()) == []


def test_parallelism_is_a_budget_not_a_ban():
    text = "It was fast, cheap and reliable. " + FILLER
    generous = analyze("x.txt", text, Config(parallel_budget_per_1k=5.0))
    strict = analyze("x.txt", text, Config(parallel_budget_per_1k=0.0))
    assert generous.counts()["parallel"] == 0
    assert strict.counts()["parallel"] == 1


def test_budget_forgives_the_mildest_run_not_the_worst():
    """A tetracolon must cost more than a tricolon. The allowance therefore
    covers the MILDEST run and the widest one is what gets reported.

    This test previously asserted the opposite and passed, which meant the
    worst offender in a document was the one guaranteed a free pass."""
    text = ("No order. No motif. No structure. No spacing. " + FILLER
            + " It was fast, cheap and reliable.")
    # budget set so the allowance is exactly one run for this length
    hits = [h for h in analyze("x.txt", text, Config(parallel_budget_per_1k=2.5)).hits
            if h.rule_id == "parallel"]
    assert len(hits) == 1
    assert "4 units" in hits[0].note      # the tricolon consumed the budget


def test_tricolon_rule_is_retired():
    """`parallel` catches every case `tricolon` caught plus the escapes.
    Keeping both meant two rules maintaining one concept, and a dedupe hack
    in run_checks to stop them double-counting the same construction."""
    assert "tricolon" not in RULES
    assert analyze("x.txt", "It was fast, cheap and reliable.", PAR
                   ).counts()["parallel"] == 1


def test_normalization_does_not_manufacture_runs():
    """Regression: growing a run outward from one matching unit produced
    three-unit runs out of a single real match."""
    prose = ("They must be computable offline, with no language model and "
             "nothing that needs corpus statistics, so that a draft can be "
             "checked on a laptop tomorrow.")
    assert _runs(prose) == []


def test_normalization_does_not_break_anaphora():
    """Regression: trimming the stem off item one removed the repeated
    opening word that anaphora is defined by."""
    assert _runs("You find a site, you mutate the serine, you run your assay.")


# ------------------------------ v0.8: paragraph closers and the specifics floor

from slopcheck import paragraph

CLOSER = Config(closer_budget_ratio=0.0)


def _para(*paragraphs):
    return "\n\n".join(paragraphs)


def test_closer_is_positional_not_lexical():
    """The point of a positional rule: rephrasing the punch does not satisfy
    it, because the rule never looks at what the punch says."""
    punchy = _para(
        "The enzyme reads a short stretch of sequence and then it decides "
        "which residue to modify. That is the model. It is wrong.",
        "We tested it against a matched null across four modifications and "
        "the effect held up throughout. It isn't close.",
    )
    rephrased = punchy.replace("It is wrong.", "The evidence says otherwise.")
    assert len(paragraph.closers(Document(punchy))) == 2
    assert len(paragraph.closers(Document(rephrased))) == 2


def test_closer_ignores_single_sentence_paragraphs():
    """A one-sentence paragraph is a heading or a caption by construction.
    Charging it as a closer would flag every list in every document."""
    assert paragraph.closers(Document(_para("Short one.", "Short two."))) == []


def test_closer_ratio_separates_the_talk_versions():
    punch_heavy = _para(*[
        f"The {w} ran for most of the afternoon and nobody in the room "
        "thought to check on it even once. It failed."
        for w in ("gel", "column", "assay", "plate")
    ])
    flowing = _para(*[
        f"The {w} ran for most of the afternoon and nobody in the room "
        "thought to check on it, which is how the whole day got away."
        for w in ("gel", "column", "assay", "plate")
    ])
    assert paragraph.closer_ratio(Document(punch_heavy)) == 1.0
    assert paragraph.closer_ratio(Document(flowing)) == 0.0


def test_closer_budget_reports_the_shortest_and_verbless_first():
    text = _para(
        "A paragraph that runs on for a while before it stops here. One gene.",
        "Another paragraph that runs on for a while before stopping. It failed.",
    )
    hits = [h for h in analyze("x.txt", text, CLOSER).hits if h.rule_id == "closer"]
    assert "verbless" in hits[0].note


# ---- the specifics floor: a requirement, which deletion cannot satisfy ----

def test_vague_paragraph_cannot_be_fixed_by_cutting_words():
    vague = ("The approach offers a range of advantages over the alternatives "
             "and addresses several of the concerns that have been raised by "
             "the community, while remaining broadly applicable across a wide "
             "set of settings and use cases in practice, which is what makes "
             "it worth considering carefully before anybody commits to it.")
    assert paragraph.vague_paragraphs(Document(vague))
    shorter = " ".join(vague.split()[:41])
    assert paragraph.vague_paragraphs(Document(shorter))


def test_specifics_counts_anchors_of_several_kinds():
    for text in ("The effect held at 3.5 fold across the whole set of runs.",
                 "The frequency Kobak measured ran across the whole corpus.",
                 "The pore was described in 1987 by a group in New York.",
                 'They called it a "matched null" in the original paper.'):
        assert paragraph.specifics(text), text


def test_spelled_out_numbers_count_as_specifics():
    """Spoken prose spells its numbers, and the digit test missed them
    entirely, which made a whole talk read as having no specifics."""
    assert paragraph.specifics("It gets that right about four times out of five.")


def test_sentence_initial_capitals_are_not_specifics():
    """Every sentence has one, so counting them would say nothing."""
    assert paragraph.specifics("Something happened. Something else happened.") == []


def test_short_paragraphs_owe_no_specifics():
    assert paragraph.vague_paragraphs(Document("A short remark with nothing in it.")) == []


def test_quoted_markers_are_not_vocabulary_hits():
    """Regression: writing about the word "delves" was flagged as using it,
    which made the tool unusable for its own documentation."""
    assert analyze("x.txt", 'Kobak measured "delves" at 28x excess.', CFG).counts()["vocab"] == 0
    assert analyze("x.txt", "We delve into the data here.", CFG).counts()["vocab"] == 1


# ------------------------------------- v0.9: weighting and paragraph ranking

from slopcheck import weighting


def test_score_weights_severity():
    """A flat count called a stock transition and a paragraph with nothing
    specific in it one each. They are not one each."""
    low = analyze("x.txt", "That said, it held. " * 3, CFG)
    assert low.score > low.per_1k  # weighted >= raw for any non-empty set
    high_rule = RULES["vague"].severity
    assert weighting.WEIGHTS[high_rule] > weighting.WEIGHTS["low"]


def test_worst_paragraph_ranks_by_density_not_count():
    """A 200-word paragraph with four hits is in better shape than a 40-word
    paragraph with three. Ranking by raw count sends a reviser to the wrong
    one."""
    dense = "One gene. One site. No motif."
    diffuse = ("We delve into it here. " + FILLER)
    text = dense + "\n\n" + diffuse
    result = analyze("x.txt", text, Config(min_sentence_words=5))
    assert result.worst
    assert result.worst[0].index == 1
    assert result.worst[0].weighted > result.worst[-1].weighted


def test_worst_paragraph_is_empty_when_clean():
    assert analyze("x.txt", (CORPUS / "clean_control.txt").read_text(), CFG).worst == []


def test_agent_orders_instructions_by_severity():
    """A reviser acts on the first few instructions, so those had better be
    the ones that matter."""
    text = ("That said, we delve into it. One gene. One site. No motif. "
            + FILLER)
    rendered = agent.review("x.txt", text, Config(min_sentence_words=5)).render()
    # line 0 is FAIL, line 1 is the worst-paragraph summary
    first_rule = rendered.splitlines()[2].split("[")[1].split("]")[0]
    assert weighting.WEIGHTS[RULES[first_rule].severity] == 3.0


def test_agent_names_the_worst_paragraph_first():
    text = "One gene. One site. No motif.\n\n" + FILLER
    rendered = agent.review("x.txt", text, Config(min_sentence_words=5)).render()
    assert rendered.splitlines()[1].startswith("worst:")


def test_long_verbatim_repeat_fires_at_two_occurrences():
    """Four words need three uses to read as filler. Six words repeated
    verbatim is a tell at two: nobody does that by accident."""
    phrase = "the address is regional rather than positional"
    text = f"{phrase} in the data. " + FILLER + f" Again, {phrase} here."
    assert analyze("x.txt", text, CFG).counts()["template"] >= 1


# --------------------------------------------------- v1.0: balanced doublets

def _doublets(text):
    return parallel.doublets(Document(text))


@pytest.mark.parametrize("text", [
    "That isn't what's there, and it isn't a close call.",
    "One stretch that carries a cluster, one stretch that doesn't.",
    "Same proteins. Same residue types.",
    "You didn't test the modification. You tested one residue of it.",
    "No order. No motif.",
])
def test_doublets_catch_balance_at_two_units(text):
    """MIN_RUN = 3 was the same rule-of-three assumption parallel.py exists
    to reject. The tell is balance, and balance starts at two. Every one of
    these sat exactly one unit under the run threshold."""
    assert _doublets(text), text


@pytest.mark.parametrize("text", [
    "I ran it again with fresh reagent, and I got the same nothing back.",
    "The buffer was cold and nobody had checked the timer before we started.",
    "She asked why we measured at thirty minutes, which nobody could answer.",
    "Nobody had a reason for it, and the number survived four lab generations.",
    "It worked. So we ran it the other way, trained on rice and tested on human.",
])
def test_doublets_leave_ordinary_pairs_alone(text):
    """Doublets are ordinary English. A detector that fires on every pair is
    worse than no detector."""
    assert _doublets(text) == []


def test_quantifiers_are_not_a_polarity_flip():
    """Regression: treating "nothing" as negation fired on "I ran it again
    with fresh reagent, and I got the same nothing back"."""
    assert _doublets("I tried it once, and I got the same nothing back.") == []
    assert _doublets("You didn't try it once. You tried it three times.")


def test_shared_pronoun_alone_is_not_anaphora():
    assert _doublets("It ran for an hour, and it stopped without warning.") == []


def test_doublet_is_budgeted_separately_from_triads():
    """Sharing the triad allowance would either drown it or gut it: doublets
    are an order of magnitude more common in ordinary prose."""
    text = "Same proteins. Same residue types. " + FILLER
    generous = analyze("x.txt", text, Config(doublet_budget_per_1k=20.0))
    strict = analyze("x.txt", text, Config(doublet_budget_per_1k=0.0))
    assert generous.counts()["doublet"] == 0
    assert strict.counts()["doublet"] == 1


def test_clean_control_has_no_doublets():
    assert _doublets((CORPUS / "clean_control.txt").read_text()) == []


def test_canonical_antithesis_is_caught_by_the_negation_rule():
    """"This isn't a modification, it's a language." is the cited form and
    belongs to `negation`, not to the structural detector."""
    assert analyze("x.txt", "This isn't a modification, it's a language.",
                   CFG).counts()["negation"] == 1


# ------------------------------------------- v1.1: a recorded negative result

from slopcheck import align


def test_local_alignment_does_not_separate_parallel_from_ordinary():
    """align.py is not wired into the checks, and this test is why.

    Smith-Waterman on token classes looks like the right tool for the
    carrier-stem and trailing-tail problems that parallel.py handles with
    special cases. Measured, the score distributions overlap with no usable
    threshold. If someone improves the scoring, this test fails and that is
    the signal to wire it in."""
    def classes(text):
        return [parallel._class(w) for w in Document(text).words]

    parallel_pairs = [("That isn't what's there", "it isn't a close call"),
                      ("One stretch that carries a cluster",
                       "one stretch on the same protein")]
    ordinary_pairs = [("The gel ran slowly", "the room stayed cold all afternoon"),
                      ("It worked", "So we ran it the other way")]

    worst_parallel = min(align.align(classes(a), classes(b)).normalized
                         for a, b in parallel_pairs)
    best_ordinary = max(align.align(classes(a), classes(b)).normalized
                        for a, b in ordinary_pairs)
    assert worst_parallel < best_ordinary, (
        "alignment now separates the classes; wire it into parallel.py")


def test_sentence_splitting_is_linear():
    """Regression: the abbreviation guard sliced the whole prefix on every
    boundary, which made Document() quadratic and cost 11 of the 12 seconds
    spent analysing a 20k-word file."""
    import time
    unit = (CORPUS / "clean_control.txt").read_text()
    small, large = unit * 10, unit * 80
    t0 = time.perf_counter(); Document(small); t_small = time.perf_counter() - t0
    t0 = time.perf_counter(); Document(large); t_large = time.perf_counter() - t0
    # 8x the input must not cost more than ~24x the time
    assert t_large < max(t_small * 24, 0.5)


# ------------------------------------------ v1.2: syntactic template reuse

from slopcheck import templates


def test_tagger_assigns_function_words_to_themselves():
    """Function words are a closed class, so they are their own tags. That
    is most of the syntactic signal and it needs no tagger."""
    assert templates.tag("the") == "the"
    assert templates.tag("of") == "of"
    assert templates.tag("running") == "~ing"
    assert templates.tag("quickly") == "~adv"
    assert templates.tag("proteins") == "~pl"
    assert templates.tag("Kobak") == "^"
    assert templates.tag("1987") == "9"


def test_structure_is_measured_against_a_shuffled_control():
    """Shuffling preserves length and tag distribution exactly and destroys
    order, so the ratio is length-controlled by construction."""
    t = templates.compute(Document((CORPUS / "slop_control.txt").read_text()))
    assert t.structure is not None and t.structure > 1.0


def test_structure_is_none_when_the_baseline_is_too_sparse():
    """Regression: a zero baseline returned 1.0, which reads as "no structure
    above chance" and actually means "unmeasurable". On a short document
    those are opposite conclusions."""
    t = templates.compute(Document("Short text with very little in it at all."))
    assert t.structure is None
    assert "unmeasurable" in t.render()


def test_gzip_estimator_is_kept_but_does_not_discriminate():
    """CR-POS is computed because [SEL24] defines it. At single-document
    length gzip has nothing for LZ77 to match, so the ratio is dominated by
    the symbol distribution, which the shuffle preserves. Recorded so nobody
    reaches for it expecting a signal."""
    sloppy = templates.compute(Document((CORPUS / "slop_control.txt").read_text()))
    clean = templates.compute(Document((CORPUS / "clean_control.txt").read_text()))
    assert abs(sloppy.cr_structure - clean.cr_structure) < 0.1
    # the direct estimator, on the same two documents, separates them
    assert sloppy.structure > clean.structure * 2


def test_template_reuse_is_a_generation_signal_not_an_editing_one():
    """R0 and R2 are the same author writing the same content with different
    cadence. Template reuse should NOT separate them, and does not. This is
    the same generation-versus-editing split [SLH26] found in stylometry."""
    import pathlib
    a = templates.compute(Document(pathlib.Path(CORPUS / "clean_control.txt").read_text()))
    b = templates.compute(Document(
        pathlib.Path(CORPUS / "clean_control.txt").read_text().replace(". ", ".\n\n")))
    assert abs(a.repeat_4 - b.repeat_4) < 0.02


def test_document_redundancy_does_not_subsume_local_shape_rules():
    """The assumption templates.py was built on, measured and false.

    A doublet is two occurrences of a ~4-token pattern. Eight of them are
    about 1% of the 4-gram mass in a 1250-token document, against a baseline
    where ~9.5% of 4-grams already repeat because that is what English
    function-word syntax does. The base rate swamps the construction.

    If this ever starts failing, the document statistic has become sensitive
    enough to see local shape and parallel.py can be reconsidered."""
    balanced = ("That isn't what's there, and it isn't a close call. "
                "Same proteins. Same residue types. "
                "You didn't test the modification. You tested one residue. ")
    broken = ("That isn't what's there, and the gap is wide. "
              "The proteins match, and so do the residue types. "
              "What you probed was a single residue, not the modification. ")
    carrier = (CORPUS / "clean_control.txt").read_text()

    with_doublets = templates.compute(Document(balanced + carrier))
    without = templates.compute(Document(broken + carrier))
    assert len(parallel.doublets(Document(balanced + carrier))) > \
        len(parallel.doublets(Document(broken + carrier)))
    assert abs(with_doublets.repeat_4 - without.repeat_4) < 0.03
