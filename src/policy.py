from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    false_positive: float = 250.0
    false_negative: float = 2000.0
    verification: float = 40.0
    manual_review: float = 60.0


def evaluate_policy(df: pd.DataFrame, prob, verify_threshold, review_threshold, costs=Costs()):
    y = df["abusive_return"].to_numpy().astype(int)
    prob = np.asarray(prob)
    decision = np.where(prob < verify_threshold, "AUTO_APPROVE", np.where(prob < review_threshold, "VERIFY", "MANUAL_REVIEW"))

    auto = decision == "AUTO_APPROVE"
    verify = decision == "VERIFY"
    review = decision == "MANUAL_REVIEW"

    fn = int(((auto) & (y == 1)).sum())
    fp = int(((~auto) & (y == 0)).sum())
    total = float(fn * costs.false_negative + fp * costs.false_positive + verify.sum() * costs.verification + review.sum() * costs.manual_review)
    return {
        "verify_threshold": float(verify_threshold),
        "review_threshold": float(review_threshold),
        "expected_cost": total,
        "loss_per_1000": total / max(len(df), 1) * 1000,
        "false_negatives": fn,
        "false_positives": fp,
        "verification_rate": float(verify.mean()),
        "manual_review_rate": float(review.mean()),
        "auto_approve_rate": float(auto.mean()),
        "decision": decision,
    }


def optimize_policy(df: pd.DataFrame, prob, costs=Costs()):
    best = None
    # Grid chosen to cover the operating range while keeping the policy interpretable.
    for t1 in np.arange(0.02, 0.90, 0.02):
        for t2 in np.arange(t1 + 0.04, 0.99, 0.04):
            result = evaluate_policy(df, prob, t1, t2, costs)
            if result["manual_review_rate"] > 0.25:
                continue
            if best is None or result["expected_cost"] < best["expected_cost"]:
                best = result
    return best or evaluate_policy(df, prob, .12, .68, costs)
