from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .features import CATEGORICAL, NUMERIC


def top_features(bundle, row: pd.DataFrame, top_n=6):
    kind = bundle["kind"]
    if kind == "logistic":
        pipe = bundle["model"]
        pre = pipe.named_steps["prep"]
        clf = pipe.named_steps["clf"]
        X = pre.transform(row[NUMERIC + CATEGORICAL])
        values = np.asarray(X.multiply(clf.coef_[0])) if hasattr(X, "multiply") else np.asarray(X) * clf.coef_[0]
        vals = values[0]
        names = pre.get_feature_names_out()
    else:
        pre = bundle["preprocessor"]
        model = bundle["model"]
        X = pre.transform(row[NUMERIC + CATEGORICAL])
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        if isinstance(values, list):
            values = values[1]
        vals = np.asarray(values)[0]
        names = pre.get_feature_names_out()

    order = np.argsort(np.abs(vals))[::-1][:top_n]
    return [{"feature": str(names[i]), "contribution": float(vals[i])} for i in order]


def concise_reasoning(df_row: pd.Series, explanation_items):
    reasons = []
    if df_row.get("return_rate_90d", 0) > 0.35:
        reasons.append(f"High historical return rate ({df_row['return_rate_90d']:.0%} over 90 days)")
    if df_row.get("returns_30d", 0) >= 3:
        reasons.append(f"High recent return velocity ({int(df_row['returns_30d'])} returns in 30 days)")
    if df_row.get("device_linked_accounts", 0) >= 3 or df_row.get("address_linked_accounts", 0) >= 3:
        reasons.append("Shared device/address with multiple accounts")
    if df_row.get("refund_amount_90d", 0) > 20000:
        reasons.append(f"High historical refund value (₹{df_row['refund_amount_90d']:,.0f})")
    if df_row.get("hours_to_return", 999) < 12:
        reasons.append(f"Very fast return request ({df_row['hours_to_return']:.1f}h after delivery)")
    for item in explanation_items:
        if len(reasons) >= 5:
            break
        cleaned = item["feature"].replace("num__", "").replace("cat__", "")
        text = f"{cleaned} increased model risk" if item["contribution"] > 0 else f"{cleaned} reduced model risk"
        if text not in reasons:
            reasons.append(text)
    return reasons[:5]
