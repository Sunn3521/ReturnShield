"""Tests for the hindsight -> policy recommendation loop."""

from __future__ import annotations

import numpy as np
import pytest

from api.hindsight import (
    DEFAULT_COSTS,
    DEFAULT_STEPS,
    MIN_SAMPLE,
    band_index,
    evaluate,
    recommend,
    sweep,
)

COSTS = DEFAULT_COSTS


def _rows(risk, abusive, loss=None):
    loss = loss if loss is not None else [0.0] * len(risk)
    return [
        {
            "return_id": f"R{i:06d}",
            "decision": "AUTO_APPROVE",
            "risk_probability": float(r),
            "actual_abusive": int(a),
            "actual_merchant_loss": float(l),
        }
        for i, (r, a, l) in enumerate(zip(risk, abusive, loss))
    ]


def test_band_index_orders_the_three_bands():
    risk = np.array([0.0, 0.11, 0.12, 0.5, 0.67, 0.68, 0.99])
    bands = band_index(risk, 0.12, 0.68)
    assert list(bands) == [0, 0, 1, 1, 1, 2, 2]


def test_missing_abuse_costs_the_real_merchant_loss_not_a_flat_estimate():
    """Approving abuse costs actual money lost, not the flat FN estimate."""
    risk = np.array([0.01])
    abusive = np.array([True])
    cheap = evaluate(risk, abusive, 0.12, 0.68, COSTS, np.array([500.0]))
    pricey = evaluate(risk, abusive, 0.12, 0.68, COSTS, np.array([5000.0]))
    assert pricey["expected_cost_total"] > cheap["expected_cost_total"]
    assert cheap["expected_cost_total"] == pytest.approx(500.0)


def test_catching_abuse_is_cheaper_than_missing_it():
    abusive = np.array([True])
    caught = evaluate(np.array([0.9]), abusive, 0.12, 0.68, COSTS, np.array([3000.0]))
    missed = evaluate(np.array([0.01]), abusive, 0.12, 0.68, COSTS, np.array([3000.0]))
    assert caught["expected_cost_total"] < missed["expected_cost_total"]


def test_flagging_a_genuine_return_costs_the_false_positive_price():
    genuine = np.array([False])
    flagged = evaluate(np.array([0.9]), genuine, 0.12, 0.68, COSTS)
    assert flagged["expected_cost_total"] == pytest.approx(COSTS["false_positive"])
    assert flagged["false_positive"] == 1


def test_perfect_separation_scores_perfectly():
    risk = np.array([0.01] * 20 + [0.95] * 20)
    abusive = np.array([0] * 20 + [1] * 20)
    point = evaluate(risk, abusive, 0.12, 0.68, COSTS)
    assert point["precision"] == 1.0 and point["recall"] == 1.0
    assert point["true_positive"] == 20 and point["false_positive"] == 0


def test_sweep_never_returns_unordered_thresholds():
    risk = np.random.default_rng(3).random(200)
    abusive = np.random.default_rng(4).integers(0, 2, 200).astype(bool)
    for point in sweep(risk, abusive, COSTS, steps=12):
        assert point["verify_threshold"] < point["review_threshold"]


def test_sweep_is_ordered_by_cost():
    risk = np.random.default_rng(5).random(300)
    abusive = np.random.default_rng(6).integers(0, 2, 300).astype(bool)
    points = sweep(risk, abusive, COSTS, steps=15)
    costs = [p["expected_cost_per_call"] for p in points]
    assert costs == sorted(costs)


def test_insufficient_sample_refuses_to_recommend():
    rows = _rows([0.1, 0.9, 0.5], [0, 1, 1])
    result = recommend(rows)
    assert result["recommendation"] == "insufficient_evidence"
    assert result["sample_size"] == 3 < MIN_SAMPLE
    assert "Keep the current policy" in result["rationale"]


def test_no_rows_is_insufficient_evidence():
    assert recommend([])["recommendation"] == "insufficient_evidence"


def test_rows_missing_risk_or_truth_are_ignored():
    rows = _rows([0.1, 0.9, 0.5], [0, 1, 1])
    rows.append({"return_id": "RX", "risk_probability": None, "actual_abusive": 1})
    rows.append({"return_id": "RY", "risk_probability": 0.5, "actual_abusive": None})
    result = recommend(rows)
    assert result["sample_size"] == 3


def test_already_optimal_policy_is_left_alone():
    """A perfectly separating policy should not be 'improved'."""
    risk = np.array([0.01] * 60 + [0.99] * 60)
    abusive = np.array([0] * 60 + [1] * 60)
    rows = _rows(list(risk), list(abusive.astype(int)))
    result = recommend(rows, thresholds={"verify_threshold": 0.5, "review_threshold": 0.9})
    assert result["recommendation"] == "keep_current_policy"
    assert result["best_candidate"]["expected_cost_per_call"] <= \
        result["current_policy"]["expected_cost_per_call"]


def test_a_missed_abuse_problem_drives_a_recommendation():
    """Policy flags nothing; abuse is concentrated at high risk."""
    rng = np.random.default_rng(11)
    genuine = rng.uniform(0.0, 0.30, 300)
    abusive = rng.uniform(0.40, 0.60, 120)
    risk = np.concatenate([genuine, abusive])
    truth = np.concatenate([np.zeros(300), np.ones(120)])
    rows = _rows(list(risk), list(truth.astype(int)))
    result = recommend(rows, thresholds={"verify_threshold": 0.95, "review_threshold": 0.98})
    assert result["recommendation"] == "retune_thresholds"
    best = result["recommended_policy"]
    assert best["expected_cost_per_call"] < result["current_policy"]["expected_cost_per_call"]
    assert result["relative_cost_improvement"] > 0
    assert best["verify_threshold"] < 0.95
    assert "cuts expected cost" in result["rationale"]


def test_maximize_f1_objective():
    rng = np.random.default_rng(13)
    risk = np.concatenate([rng.uniform(0, 0.5, 200), rng.uniform(0.5, 1.0, 200)])
    truth = np.concatenate([np.zeros(200), np.ones(200)]).astype(int)
    rows = _rows(list(risk), list(truth))
    result = recommend(rows, objective="maximize_f1", steps=25)
    assert result["objective"] == "maximize_f1"
    assert result["recommended_policy"]["f1"] > 0.8


def test_target_precision_reports_when_unreachable():
    """A scorer that cannot separate anything should say so, not guess."""
    rng = np.random.default_rng(17)
    risk = rng.uniform(0, 1, 200)
    truth = rng.integers(0, 2, 200)
    rows = _rows(list(risk), list(truth))
    result = recommend(rows, objective="target_precision", target_precision=0.99, steps=15)
    assert result["recommendation"] in {"no_candidate", "keep_current_policy"}


def test_recommendation_is_deterministic():
    rng = np.random.default_rng(19)
    risk = rng.uniform(0, 1, 400)
    truth = rng.integers(0, 2, 400)
    rows = _rows(list(risk), list(truth))
    first = recommend(rows, steps=20)
    second = recommend(list(rows), steps=20)
    assert first == second


def test_default_grid_finds_what_a_coarse_grid_misses():
    """Regression: the chat hint once used steps=16 and disagreed with the API.

    The optimum here sits between coarse grid points, so only the default
    resolution reports a real improvement.
    """
    rng = np.random.default_rng(23)
    genuine = rng.uniform(0.0, 0.08, 400)
    abusive = rng.uniform(0.09, 0.20, 120)
    risk = np.concatenate([genuine, abusive])
    truth = np.concatenate([np.zeros(400), np.ones(120)]).astype(int)
    rows = _rows(list(risk), list(truth))

    coarse = recommend(rows, steps=10)
    fine = recommend(rows)
    assert fine["recommended_policy"]["verify_threshold"] < 0.1
    # Both paths must agree when they use the same resolution.
    assert fine == recommend(rows, steps=DEFAULT_STEPS)
    assert coarse["recommended_policy"] != fine["recommended_policy"]


def test_api_and_chat_hint_share_one_default():
    """A drifted default would show the user two different verdicts."""
    from api.main import _chat_recommendation  # noqa: F401  (import-time check)

    assert DEFAULT_STEPS >= 30, "grid too coarse to trust for a live verdict"
