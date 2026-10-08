"""Hindsight: turn resolved decisions into an actionable policy recommendation.

``MemoryStore`` records what the agent decided and ``resolve_outcomes`` joins it
to ground truth. This module closes the loop: given those resolved calls, it
replays every candidate pair of policy thresholds over the *actual* risk scores,
prices each one with the cost assumptions already in ``models/policy.json``, and
recommends an operating point only when the evidence justifies moving.

Everything is computed from logged decisions, so the recommendation explains
itself with the same numbers it was derived from.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

#: Ascending severity. Index doubles as the band a risk score falls into.
BANDS = ("AUTO_APPROVE", "VERIFY", "MANUAL_REVIEW")

#: Used when no policy.json is available; mirrors the shipped cost assumptions.
DEFAULT_COSTS = {
    "false_positive": 250.0,
    "false_negative": 2000.0,
    "verification": 40.0,
    "manual_review": 60.0,
}
DEFAULT_THRESHOLDS = {"verify_threshold": 0.12, "review_threshold": 0.68}

#: Below this many resolved calls a recommendation would be noise.
MIN_SAMPLE = 40
#: A move must beat the current policy by this relative margin to be proposed.
MIN_RELATIVE_GAIN = 0.02
#: Threshold grid resolution. 50 -> ~1225 candidate pairs, ~150 ms at 800 rows.
#: Coarser grids genuinely miss good operating points, so the API and the chat
#: hint must both use this rather than each picking their own.
DEFAULT_STEPS = 50


def band_index(risk: np.ndarray, verify: np.ndarray, review: np.ndarray) -> np.ndarray:
    """Map risk scores to band indices under broadcastable thresholds."""
    return np.where(risk < verify, 0, np.where(risk < review, 1, 2)).astype(np.int8)


def price(
    bands: np.ndarray,
    abusive: np.ndarray,
    costs: dict[str, float],
    merchant_loss: np.ndarray | None = None,
) -> np.ndarray:
    """Cost of each call, given the band it landed in and what actually happened.

    Approving an abusive return costs the merchant the real observed loss (not a
    flat estimate), because that is the money that is gone. Flagging a genuine
    return costs the false-positive price. Catching abuse costs only the review.
    """
    fp = float(costs["false_positive"])
    verify_cost = float(costs["verification"])
    review_cost = float(costs["manual_review"])
    fn_default = float(costs["false_negative"])

    if merchant_loss is None:
        missed = np.full(abusive.shape, fn_default, dtype=np.float64)
    else:
        missed = np.where(np.isfinite(merchant_loss) & (merchant_loss > 0),
                          merchant_loss, fn_default)

    flagged = bands >= 1
    cost = np.zeros(bands.shape, dtype=np.float64)
    # Genuine return that got flagged.
    cost = np.where(flagged & ~abusive, fp, cost)
    # Abuse that got caught: only the review cost.
    cost = np.where(flagged & abusive, np.where(bands >= 2, review_cost, verify_cost), cost)
    # Abuse that slipped through.
    cost = np.where(~flagged & abusive, missed, cost)
    return cost


def evaluate(
    risk: np.ndarray,
    abusive: np.ndarray,
    verify: float,
    review: float,
    costs: dict[str, float],
    merchant_loss: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score one candidate operating point."""
    bands = band_index(risk, verify, review)
    flagged = bands >= 1
    tp = int(np.sum(flagged & abusive))
    fp = int(np.sum(flagged & ~abusive))
    fn = int(np.sum(~flagged & abusive))
    tn = int(np.sum(~flagged & ~abusive))

    total_cost = float(np.sum(price(bands, abusive, costs, merchant_loss)))
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else 0.0

    return {
        "verify_threshold": round(float(verify), 4),
        "review_threshold": round(float(review), 4),
        "expected_cost_total": round(total_cost, 2),
        "expected_cost_per_call": round(total_cost / max(1, risk.size), 2),
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4),
        "flagged": tp + fp,
        "flag_rate": round((tp + fp) / max(1, risk.size), 4),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def _grid(steps: int) -> np.ndarray:
    return np.round(np.linspace(0.0, 0.99, steps), 4)


def sweep(
    risk: np.ndarray,
    abusive: np.ndarray,
    costs: dict[str, float],
    merchant_loss: np.ndarray | None = None,
    steps: int = 50,
) -> list[dict[str, Any]]:
    """Evaluate every (verify, review) pair on a grid. Ordered by cost."""
    candidates: list[dict[str, Any]] = []
    values = _grid(steps)
    for verify in values:
        for review in values:
            if review <= verify:
                continue  # bands must be ordered AUTO_APPROVE < VERIFY < MANUAL_REVIEW
            candidates.append(evaluate(risk, abusive, verify, review, costs, merchant_loss))
    candidates.sort(key=lambda c: c["expected_cost_per_call"])
    return candidates


def _objective(candidates: Sequence[dict[str, Any]], objective: str,
               target_precision: float) -> dict[str, Any] | None:
    if not candidates:
        return None
    if objective == "maximize_f1":
        return max(candidates, key=lambda c: (c["f1"], -c["expected_cost_per_call"]))
    if objective == "target_precision":
        eligible = [c for c in candidates if c["precision"] is not None
                    and c["precision"] >= target_precision]
        if not eligible:
            return None
        # Cheapest point that still clears the precision floor.
        return min(eligible, key=lambda c: c["expected_cost_per_call"])
    return candidates[0]  # minimize_cost


def recommend(
    rows: Iterable[dict[str, Any]],
    thresholds: dict[str, float] | None = None,
    costs: dict[str, float] | None = None,
    objective: str = "minimize_cost",
    target_precision: float = 0.75,
    steps: int = DEFAULT_STEPS,
) -> dict[str, Any]:
    """Recommend a policy operating point from resolved decision rows.

    Returns a recommendation with its own evidence. When the sample is too small
    or no candidate beats the current policy by a real margin, the answer is
    "keep what you have" rather than a spurious change.
    """
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    costs = {**DEFAULT_COSTS, **(costs or {})}

    usable = [
        r for r in rows
        if r.get("risk_probability") is not None and r.get("actual_abusive") is not None
    ]
    sample = len(usable)

    if sample < MIN_SAMPLE:
        return {
            "recommendation": "insufficient_evidence",
            "sample_size": sample,
            "min_sample": MIN_SAMPLE,
            "objective": objective,
            "current_policy": {
                "verify_threshold": thresholds["verify_threshold"],
                "review_threshold": thresholds["review_threshold"],
            },
            "rationale": (
                f"Only {sample} of {MIN_SAMPLE} resolved calls needed for a threshold "
                "recommendation. Keep the current policy and log more decisions."
            ),
        }

    risk = np.array([float(r["risk_probability"]) for r in usable], dtype=np.float64)
    abusive = np.array([bool(int(r["actual_abusive"])) for r in usable])
    loss_raw = [r.get("actual_merchant_loss") for r in usable]
    merchant_loss = None
    if any(v is not None for v in loss_raw):
        merchant_loss = np.array(
            [np.nan if v is None else float(v) for v in loss_raw], dtype=np.float64)

    current_point = evaluate(risk, abusive, thresholds["verify_threshold"],
                             thresholds["review_threshold"], costs, merchant_loss)
    candidates = sweep(risk, abusive, costs, merchant_loss, steps=steps)
    best = _objective(candidates, objective, target_precision)

    if best is None:
        return {
            "recommendation": "no_candidate",
            "sample_size": sample,
            "objective": objective,
            "current_policy": current_point,
            "rationale": (
                f"No threshold pair reached the target precision of {target_precision:.0%}. "
                "The scorer cannot separate abuse at this operating point."
            ),
        }

    gain = 1.0 - (best["expected_cost_per_call"] / current_point["expected_cost_per_call"]) \
        if current_point["expected_cost_per_call"] else 0.0
    same_as_current = (
        abs(best["verify_threshold"] - current_point["verify_threshold"]) < 1e-9
        and abs(best["review_threshold"] - current_point["review_threshold"]) < 1e-9
    )

    if same_as_current or gain < MIN_RELATIVE_GAIN:
        return {
            "recommendation": "keep_current_policy",
            "sample_size": sample,
            "objective": objective,
            "current_policy": current_point,
            "best_candidate": best,
            "relative_cost_improvement": round(gain, 4),
            "rationale": (
                f"The best candidate (verify {best['verify_threshold']:.2f} / review "
                f"{best['review_threshold']:.2f}) saves only {gain:.1%}, below the "
                f"{MIN_RELATIVE_GAIN:.0%} bar required to move a live policy."
            ),
        }

    return {
        "recommendation": "retune_thresholds",
        "sample_size": sample,
        "objective": objective,
        "current_policy": current_point,
        "recommended_policy": best,
        "relative_cost_improvement": round(gain, 4),
        "expected_cost_saved_total": round(
            current_point["expected_cost_total"] - best["expected_cost_total"], 2),
        "costs": costs,
        "rationale": (
            f"Moving verify {current_point['verify_threshold']:.2f} -> "
            f"{best['verify_threshold']:.2f} and review "
            f"{current_point['review_threshold']:.2f} -> {best['review_threshold']:.2f} "
            f"cuts expected cost {current_point['expected_cost_per_call']:.2f} -> "
            f"{best['expected_cost_per_call']:.2f} per call "
            f"(precision {best['precision']}, recall {best['recall']}, "
            f"flag rate {best['flag_rate']:.1%}) over {sample} resolved calls."
        ),
    }


def from_store(store, **kwargs) -> dict[str, Any]:
    """Convenience wrapper: recommend straight from a MemoryStore."""
    return recommend(store.resolved_decision_rows(), **kwargs)
