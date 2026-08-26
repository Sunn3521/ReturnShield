from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from src.model import load_bundle, predict_bundle
from src.explain import top_features, concise_reasoning
from src.responder import generate_agent_response

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

app = FastAPI(
    title="ReturnShield AI API",
    description="High-Throughput Cost-Sensitive Return Abuse Risk Scorer & Decision Agent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bundle = None
policy = None

def get_bundle():
    global bundle, policy
    if bundle is None:
        bundle_path = MODELS / "model_bundle.joblib"
        policy_path = MODELS / "policy.json"
        if not bundle_path.exists():
            raise RuntimeError("Model bundle not found. Please run `python run_pipeline.py` first.")
        bundle = load_bundle(str(bundle_path))
        with open(policy_path, "r", encoding="utf-8") as f:
            policy = json.load(f)
    return bundle, policy


class ReturnRequestPayload(BaseModel):
    return_id: str = Field(..., example="R999001")
    order_id: str = Field(..., example="O888001")
    customer_id: str = Field(..., example="C00123")
    product_category: str = Field("electronics", example="electronics")
    payment_method: str = Field("card", example="card")
    return_reason: str = Field("damaged", example="damaged")
    order_value: float = Field(..., example=12500.0)
    product_price: float = Field(..., example=12500.0)
    discount_pct: float = Field(0.0, example=0.0)
    customer_account_age_days: int = Field(120, example=120)
    orders_7d: int = Field(1, example=1)
    orders_30d: int = Field(3, example=3)
    orders_90d: int = Field(5, example=5)
    returns_7d: int = Field(0, example=0)
    returns_30d: int = Field(2, example=2)
    returns_90d: int = Field(4, example=4)
    refund_amount_30d: float = Field(4500.0, example=4500.0)
    refund_amount_90d: float = Field(18500.0, example=18500.0)
    hours_to_return: float = Field(3.5, example=3.5)
    same_product_returns_90d: int = Field(0, example=0)
    same_category_returns_90d: int = Field(2, example=2)
    velocity_24h: int = Field(1, example=1)
    velocity_7d: int = Field(2, example=2)
    device_linked_accounts: int = Field(3, example=3)
    address_linked_accounts: int = Field(2, example=2)
    device_return_rate_90d: float = Field(0.40, example=0.40)
    address_return_rate_90d: float = Field(0.35, example=0.35)


class ScoringResponse(BaseModel):
    return_id: str
    risk_probability: float
    risk_display: str
    decision: str
    merchant_loss_estimate: float
    top_signals: List[str]
    merchant_action_protocol: str
    customer_message: str
    latency_ms: float


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "ReturnShield AI Risk Scorer",
        "docs_url": "/docs",
        "endpoints": ["/api/v1/health", "/api/v1/policy", "/api/v1/score", "/api/v1/batch_score"]
    }


@app.get("/api/v1/health")
async def health():
    b, p = get_bundle()
    return {
        "status": "healthy",
        "model_kind": b["kind"],
        "policy": {
            "verify_threshold": p["verify_threshold"],
            "review_threshold": p["review_threshold"]
        }
    }


@app.get("/api/v1/policy")
async def get_policy():
    _, p = get_bundle()
    return p


@app.post("/api/v1/score", response_model=ScoringResponse)
async def score_return(payload: ReturnRequestPayload):
    t0 = time.perf_counter()
    b, p = get_bundle()

    row_dict = payload.model_dump()
    row_dict["return_value"] = row_dict["order_value"]
    row_dict["return_value_ratio"] = 1.0
    row_dict["high_value_flag"] = int(row_dict["order_value"] >= 7500.0)
    row_dict["return_rate_30d"] = row_dict["returns_30d"] / max(row_dict["orders_30d"], 1)
    row_dict["return_rate_90d"] = row_dict["returns_90d"] / max(row_dict["orders_90d"], 1)
    row_dict["prediction_time"] = pd.Timestamp.now()

    df_row = pd.DataFrame([row_dict])
    prob = float(predict_bundle(b, df_row)[0])

    t1_verify = p["verify_threshold"]
    t2_review = p["review_threshold"]

    if prob < t1_verify:
        decision = "AUTO_APPROVE"
    elif prob < t2_review:
        decision = "VERIFY"
    else:
        decision = "MANUAL_REVIEW"

    shap_items = top_features(b, df_row, top_n=5)
    reasons = concise_reasoning(df_row.iloc[0], shap_items)

    df_row_series = df_row.iloc[0].copy()
    df_row_series["risk_probability"] = prob
    df_row_series["decision"] = decision
    agent_output = generate_agent_response(df_row_series, reasons)

    loss_est = float(np.clip(payload.order_value * prob * 0.85, 0.0, payload.order_value))
    latency = (time.perf_counter() - t0) * 1000.0

    return ScoringResponse(
        return_id=payload.return_id,
        risk_probability=round(prob, 4),
        risk_display=f"{prob:.1%}",
        decision=decision,
        merchant_loss_estimate=round(loss_est, 2),
        top_signals=reasons,
        merchant_action_protocol=agent_output["merchant_action"],
        customer_message=agent_output["customer_message"],
        latency_ms=round(latency, 2)
    )


@app.post("/api/v1/batch_score", response_model=List[ScoringResponse])
async def batch_score(payloads: List[ReturnRequestPayload]):
    results = []
    for item in payloads:
        res = await score_return(item)
        results.append(res)
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
