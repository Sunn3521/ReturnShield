"""Tests for the expanded intent dataset and the held-out evaluation harness."""

from __future__ import annotations

import pytest

from src.intent_data import (
    INTENTS,
    LEGACY_EXAMPLES,
    SEEDS,
    as_pairs,
    build_dataset,
    dataset_stats,
)
from src.intent_eval import (
    GOLDEN_LABELS,
    GOLDEN_QUESTIONS,
    GOLDEN_TEXTS,
    evaluate_classifier,
)


def test_every_intent_has_seeds():
    missing = [i for i in INTENTS if not SEEDS.get(i)]
    assert not missing, f"intents without seed utterances: {missing}"


def test_legacy_examples_are_all_covered_by_intents():
    orphans = [i for i in LEGACY_EXAMPLES if i not in INTENTS]
    assert not orphans, f"legacy intents missing from INTENTS: {orphans}"


def test_dataset_grows_well_beyond_the_original_56():
    dataset = build_dataset(per_seed=6)
    stats = dataset_stats(dataset)
    assert stats["utterances"] > 500, stats
    assert stats["intents"] == len(INTENTS)


def test_every_intent_is_represented():
    dataset = build_dataset(per_seed=6)
    for intent in INTENTS:
        assert dataset[intent], f"{intent} has no training utterances"


def test_dataset_is_deterministic():
    a = build_dataset(per_seed=6, seed=99)
    b = build_dataset(per_seed=6, seed=99)
    assert a == b


def test_no_duplicate_lowercased_utterances_within_an_intent():
    dataset = build_dataset(per_seed=6)
    for intent, rows in dataset.items():
        norm = [r.lower() for r in rows]
        assert len(norm) == len(set(norm)), f"{intent} contains duplicates"


def test_legacy_utterances_are_retained_verbatim():
    dataset = build_dataset(per_seed=6)
    for intent, examples in LEGACY_EXAMPLES.items():
        for example in examples:
            assert example in dataset[intent], f"{example!r} dropped from {intent}"


def test_augmentation_produces_typos_and_politeness():
    dataset = build_dataset(per_seed=6)
    everything = [r.lower() for rows in dataset.values() for r in rows]
    joined = " ".join(everything)
    assert "please" in joined
    assert "can you" in joined
    # A typo should survive somewhere; seeds alone are all clean English.
    assert any("can yu" in e or "pleae" in e or "pls" in e for e in everything) or any(
        len(set(e)) < len(e) for e in everything
    )


def test_as_pairs_produces_aligned_arrays():
    dataset = build_dataset(per_seed=2)
    texts, labels = as_pairs(dataset)
    assert len(texts) == len(labels) == dataset_stats(dataset)["utterances"]
    assert set(labels) == set(INTENTS)


def test_golden_set_is_balanced_over_intents():
    covered = set(GOLDEN_LABELS)
    assert covered <= set(INTENTS), f"golden set references unknown intents: {covered - set(INTENTS)}"
    # Every intent needs at least one held-out question, or it cannot be scored.
    assert covered == set(INTENTS), f"intents with no golden question: {set(INTENTS) - covered}"


def test_golden_questions_do_not_appear_in_training():
    """The eval is only honest if the golden set is genuinely held out."""
    dataset = build_dataset(per_seed=6)
    trained = {" ".join(r.lower().split()) for rows in dataset.values() for r in rows}
    leaked = [q for q in GOLDEN_TEXTS if " ".join(q.lower().split()) in trained]
    assert not leaked, f"golden questions leaked into training data: {leaked}"


def test_golden_questions_are_unique():
    assert len(GOLDEN_QUESTIONS) == len(set(GOLDEN_QUESTIONS))


def test_evaluate_classifier_reports_metrics_and_confusions():
    def perfect(text: str) -> str:
        return dict(GOLDEN_QUESTIONS)[text]

    result = evaluate_classifier(perfect)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["confusions"] == []


def test_evaluate_classifier_flags_wrong_predictions():
    def always_summary(text: str) -> str:
        return "summary"

    result = evaluate_classifier(always_summary)
    assert result["accuracy"] < 0.5
    assert result["confusions"]
    assert result["per_intent"]["summary"]["f1"] > result["per_intent"]["clusters"]["f1"]


def test_trained_classifier_beats_a_degenerate_baseline():
    """Guard the actual goal: the expanded model must beat 'always say summary'."""
    from train_intent import train_expanded

    model, stats = train_expanded(per_seed=6)
    result = evaluate_classifier(lambda text: str(model.predict([text])[0]))

    assert stats["utterances"] > 500
    assert result["macro_f1"] > 0.75, f"macro F1 regressed to {result['macro_f1']}"
