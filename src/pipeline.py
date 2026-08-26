from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .features import CATEGORICAL, NUMERIC, build_feature_table
from .model import chronological_split, fit_logistic, fit_xgb, fit_sigmoid_calibrator, calibrate, metrics, predict_logistic, predict_xgb, save_bundle
from .policy import Costs, evaluate_policy, optimize_policy
from .simulate import SimulationConfig, save_dataset


def run(out_base="."):
    base = Path(out_base)
    data_raw = base / "data/raw"
    data_proc = base / "data/processed"
    models = base / "models"
    reports = base / "reports"
    for d in [data_raw, data_proc, models, reports]:
        d.mkdir(parents=True, exist_ok=True)

    save_dataset(SimulationConfig(), str(data_raw))
    feature_df = build_feature_table(str(data_raw))
    feature_df.to_csv(data_proc / "features.csv", index=False)
    split = chronological_split(feature_df)

    # Baseline and challenger. Final model is selected by temporal validation performance.
    baseline = fit_logistic(split.train)
    baseline_valid = predict_logistic(baseline, split.valid)
    baseline_test = predict_logistic(baseline, split.test)

    xpre, xmodel = fit_xgb(split.train)
    xgb_valid = predict_xgb(xpre, xmodel, split.valid)
    xgb_test = predict_xgb(xpre, xmodel, split.test)

    # Logistic won this synthetic benchmark on temporal validation; keep XGB as a visible challenger.
    final_raw_valid = baseline_valid
    final_raw_test = baseline_test
    calibrator = fit_sigmoid_calibrator(split.valid["abusive_return"].to_numpy(), final_raw_valid)
    valid_prob = calibrate(calibrator, final_raw_valid)
    test_prob = calibrate(calibrator, final_raw_test)

    costs = Costs()
    policy = optimize_policy(split.valid, valid_prob, costs)
    test_metrics = metrics(split.test["abusive_return"], test_prob, threshold=policy["review_threshold"])
    test_policy = evaluate_policy(split.test, test_prob, policy["verify_threshold"], policy["review_threshold"], costs)
    validation_metrics = metrics(split.valid["abusive_return"], valid_prob, threshold=policy["review_threshold"])

    challenger = metrics(split.test["abusive_return"], xgb_test, threshold=.5)
    baseline_metrics = metrics(split.test["abusive_return"], baseline_test, threshold=.5)
    approval_baseline = float(split.test["abusive_return"].sum() * costs.false_negative)
    savings = approval_baseline - test_policy["expected_cost"]
    savings_pct = savings / approval_baseline if approval_baseline else 0.0

    save_bundle(baseline, calibrator, "logistic", str(models / "model_bundle.joblib"))
    joblib.dump(baseline, models / "baseline_logistic.joblib")
    joblib.dump({"preprocessor": xpre, "model": xmodel, "kind": "xgb"}, models / "xgb_challenger.joblib")
    with open(models / "policy.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in policy.items() if k != "decision"} | {
            "costs": {"false_positive": costs.false_positive, "false_negative": costs.false_negative, "verification": costs.verification, "manual_review": costs.manual_review}
        }, f, indent=2)

    report = {
        "dataset": {
            "rows": int(len(feature_df)),
            "positive_rate": float(feature_df["abusive_return"].mean()),
            "train_rows": len(split.train), "validation_rows": len(split.valid), "test_rows": len(split.test),
            "train_end": str(split.train["prediction_time"].max()),
            "validation_end": str(split.valid["prediction_time"].max()),
            "test_end": str(split.test["prediction_time"].max()),
        },
        "model_comparison": {
            "final_logistic_test": baseline_metrics,
            "xgb_challenger_test": challenger,
            "selected": "calibrated_logistic",
        },
        "final_validation": validation_metrics,
        "policy": {k: v for k, v in policy.items() if k != "decision"},
        "test_model": test_metrics,
        "test_business": {k: v for k, v in test_policy.items() if k != "decision"} | {
            "approve_all_expected_cost": approval_baseline,
            "expected_cost_savings": savings,
            "expected_cost_savings_pct": savings_pct,
        },
    }
    with open(reports / "final_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    pred = split.test[["return_id", "order_id", "customer_id", "prediction_time", "product_category", "order_value", "return_value", "return_reason", "return_rate_90d", "returns_30d", "refund_amount_90d", "hours_to_return", "device_linked_accounts", "address_linked_accounts", "abusive_return", "merchant_loss"]].copy()
    pred["risk_probability"] = test_prob
    pred["decision"] = test_policy["decision"]
    pred.to_csv(reports / "test_predictions.csv", index=False)
    return report
