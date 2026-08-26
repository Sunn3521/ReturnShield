from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .features import CATEGORICAL, NUMERIC, TARGET


@dataclass
class DataSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def chronological_split(df: pd.DataFrame, train_frac=.6, valid_frac=.2) -> DataSplit:
    df = df.sort_values("prediction_time").reset_index(drop=True)
    n = len(df)
    a = int(n * train_frac)
    b = int(n * (train_frac + valid_frac))
    return DataSplit(df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy())


def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])


def fit_logistic(train: pd.DataFrame):
    X = train[NUMERIC + CATEGORICAL]
    y = train[TARGET]
    model = Pipeline([
        ("prep", make_preprocessor()),
        ("clf", LogisticRegression(max_iter=2500, class_weight="balanced", C=0.5)),
    ])
    model.fit(X, y)
    return model


def fit_xgb(train: pd.DataFrame):
    pre = make_preprocessor()
    X_train = pre.fit_transform(train[NUMERIC + CATEGORICAL])
    y = train[TARGET].to_numpy()
    pos = max(int(y.sum()), 1)
    neg = max(len(y) - pos, 1)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=4,
        reg_lambda=3.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=neg / pos,
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y)
    return pre, model


def fit_sigmoid_calibrator(valid_y, valid_prob):
    from sklearn.linear_model import LogisticRegression
    clipped = np.clip(np.asarray(valid_prob), 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    cal = LogisticRegression(C=10.0, max_iter=1000)
    cal.fit(logit, np.asarray(valid_y))
    return cal


def calibrate(calibrator, prob):
    clipped = np.clip(np.asarray(prob), 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logit)[:, 1]


def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "returns_30d" in df.columns and "orders_30d" in df.columns and "return_rate_30d" not in df.columns:
        df["return_rate_30d"] = df["returns_30d"] / df["orders_30d"].clip(lower=1)
    if "returns_90d" in df.columns and "orders_90d" in df.columns and "return_rate_90d" not in df.columns:
        df["return_rate_90d"] = df["returns_90d"] / df["orders_90d"].clip(lower=1)
    if "return_value" in df.columns and "order_value" in df.columns and "return_value_ratio" not in df.columns:
        df["return_value_ratio"] = df["return_value"] / df["order_value"].clip(lower=1.0)
    if "order_value" in df.columns and "high_value_flag" not in df.columns:
        df["high_value_flag"] = (df["order_value"] >= 7500.0).astype(int)
    for col in NUMERIC:
        if col not in df.columns:
            df[col] = 0.0
    for col in CATEGORICAL:
        if col not in df.columns:
            df[col] = "unknown"
    return df


def predict_logistic(model, df):
    df_clean = ensure_features(df)
    return model.predict_proba(df_clean[NUMERIC + CATEGORICAL])[:, 1]


def predict_xgb(pre, model, df):
    df_clean = ensure_features(df)
    X = pre.transform(df_clean[NUMERIC + CATEGORICAL])
    return model.predict_proba(X)[:, 1]


def metrics(y_true, prob, threshold=.5):
    pred = (np.asarray(prob) >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return {
        "pr_auc": float(average_precision_score(y_true, prob)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "brier": float(brier_score_loss(y_true, prob)),
    }


def save_bundle(model, calibrator, kind="logistic", path="models/model_bundle.joblib"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "calibrator": calibrator, "kind": kind}, path)


def load_bundle(path="models/model_bundle.joblib"):
    return joblib.load(path)


def predict_bundle(bundle, df):
    df_clean = ensure_features(df)
    if bundle["kind"] == "logistic":
        raw = predict_logistic(bundle["model"], df_clean)
    else:
        pre = bundle["preprocessor"]
        raw = predict_xgb(pre, bundle["model"], df_clean)
    return calibrate(bundle["calibrator"], raw)
