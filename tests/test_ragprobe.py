"""Tests for the parts that guard result validity: perturbations must
actually corrupt (and be deterministic), the lexical scorer must behave as
designed (catch swaps, tolerate clean), meta-eval math must be right, and
judge parsing must survive messy model output -- all without network."""
from __future__ import annotations

import random

from ragprobe.dataset import EvalSample, split_sentences
from ragprobe.perturb import (contaminate, perturb_entity_swap,
                              perturb_negation, perturb_number)
from ragprobe.runner import run_scorer
from ragprobe.scorers import LexicalScorer
from ragprobe.scorers.judge import JudgeScorer

PASSAGE = ("The Eiffel Tower was completed in 1889 in Paris. It stands 330 "
           "meters tall. Gustave Eiffel led the construction project. The "
           "tower was the tallest structure in the world until 1930. It "
           "attracts nearly 7 million visitors every year.")


def _sample(sid: str = "s1") -> EvalSample:
    return EvalSample(
        sample_id=sid, question="How tall is the Eiffel Tower?",
        passage=PASSAGE, is_contaminated=False,
        answer_sentences=["The Eiffel Tower was completed in 1889 in Paris.",
                          "It stands 330 meters tall."],
    )


# ---------------- perturbations ----------------

def test_number_perturb_changes_value() -> None:
    out = perturb_number("It stands 330 meters tall.", random.Random(0))
    assert out is not None and out != "It stands 330 meters tall."
    assert "330" not in out


def test_negation_insert_and_remove_roundtrip() -> None:
    rng = random.Random(0)
    negated = perturb_negation("The tower was the tallest structure.", rng)
    assert negated is not None and " not " in negated
    restored = perturb_negation(negated, rng)
    assert restored is not None and " not " not in restored


def test_entity_swap_uses_donor_entity() -> None:
    donor = "Nikola Tesla worked in New York on alternating current."
    out = perturb_entity_swap(
        "Construction was led by Gustave Eiffel in Paris.", donor,
        random.Random(1))
    assert out is not None
    assert ("Tesla" in out) or ("New York" in out) or ("Nikola" in out)


def test_contaminate_is_deterministic_and_localized() -> None:
    samples = [_sample(f"s{i}") for i in range(8)]
    a = contaminate(samples, seed=42)
    b = contaminate(samples, seed=42)
    assert [x.answer_sentences for x in a] == [x.answer_sentences for x in b]
    for twin in a:
        assert twin.is_contaminated
        assert twin.contamination_type is not None
        assert twin.corrupted_sentence_idx is not None
        assert 0 <= twin.corrupted_sentence_idx < len(twin.answer_sentences)


# ---------------- lexical scorer ----------------

def test_lexical_passes_clean_and_flags_entity_swap() -> None:
    scorer = LexicalScorer()
    clean = scorer.evaluate("c", PASSAGE, _sample().answer_sentences)
    assert not clean.flagged, "grounded sentences should not be flagged"
    swapped = scorer.evaluate(
        "d", PASSAGE, ["The Sydney Opera House was completed in 1889 in Paris.",
                       "It stands 8841 meters tall."])
    assert swapped.flagged, "swapped entity + number should crater coverage"


def test_lexical_known_blind_spot_negation_is_documented_behavior() -> None:
    """Negation barely moves token overlap -- this SHOULD slip through.
    The meta-evaluation exists to expose it; the test pins the behavior."""
    scorer = LexicalScorer()
    res = scorer.evaluate(
        "e", PASSAGE, ["The Eiffel Tower was not completed in 1889 in Paris."])
    assert not res.flagged


# ---------------- meta-evaluation math ----------------

def test_run_scorer_meta_counts() -> None:
    clean = [_sample(f"c{i}") for i in range(4)]
    contam = contaminate(clean, seed=1)
    report = run_scorer(LexicalScorer(), clean + contam)
    assert report.n_clean == 4
    assert report.n_contaminated == len(contam)
    total = (report.true_positives + report.false_negatives
             + report.false_positives + report.true_negatives)
    assert total == 4 + len(contam)
    assert 0.0 <= report.false_alarm_rate <= 1.0


# ---------------- judge parsing (no network) ----------------

def test_judge_parses_strict_and_fenced_json() -> None:
    strict = '{"verdicts": [{"index": 0, "verdict": "supported"}, {"index": 1, "verdict": "unsupported"}]}'
    assert JudgeScorer.parse_verdicts(strict, 2) == [1.0, 0.0]
    fenced = f"```json\n{strict}\n```"
    assert JudgeScorer.parse_verdicts(fenced, 2) == [1.0, 0.0]


def test_sentence_splitter_handles_abbreviation_free_text() -> None:
    assert len(split_sentences(PASSAGE)) == 5
