from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.model import load_bundle, predict_bundle

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def run_redteam_simulation() -> pd.DataFrame:
    """
    Simulates 4 common e-commerce return fraud attack vectors and evaluates
    ReturnShield's defense interception rate.
    """
    bundle = load_bundle(str(MODELS / "model_bundle.joblib"))
    with open(MODELS / "policy.json", "r", encoding="utf-8") as f:
        policy = json.load(f)
        
    t1 = policy["verify_threshold"]
    t2 = policy["review_threshold"]

    attack_scenarios = [
        {
            "attack_type": "Wardrobing Fashion Attack",
            "description": "Customer buys high-end luxury fashion item, wears for single event, submits return 10h after delivery.",
            "return_id": "ATK_WARDROBE_01",
            "customer_id": "C_ATK_001",
            "order_id": "O_ATK_001",
            "product_category": "fashion",
            "payment_method": "card",
            "return_reason": "changed_mind",
            "order_value": 24500.0, "return_value": 24500.0, "product_price": 24500.0, "discount_pct": 0.0,
            "customer_account_age_days": 25,
            "orders_7d": 3, "orders_30d": 7, "orders_90d": 12,
            "returns_7d": 2, "returns_30d": 5, "returns_90d": 9,
            "return_rate_30d": 0.71, "return_rate_90d": 0.75,
            "refund_amount_30d": 42000.0, "refund_amount_90d": 85000.0,
            "hours_to_return": 10.5, "return_value_ratio": 1.0,
            "same_product_returns_90d": 2, "same_category_returns_90d": 5,
            "velocity_24h": 2, "velocity_7d": 4,
            "device_linked_accounts": 2, "address_linked_accounts": 2,
            "device_return_rate_90d": 0.65, "address_return_rate_90d": 0.60,
            "high_value_flag": 1,
            "prediction_time": pd.Timestamp.now()
        },
        {
            "attack_type": "Sybil Abuse Ring Attack",
            "description": "Multi-account network sharing single device/address claiming damaged high-value items.",
            "return_id": "ATK_RING_02",
            "customer_id": "C_ATK_RING_09",
            "order_id": "O_ATK_RING_09",
            "product_category": "electronics",
            "payment_method": "upi",
            "return_reason": "damaged",
            "order_value": 38000.0, "return_value": 38000.0, "product_price": 38000.0, "discount_pct": 0.0,
            "customer_account_age_days": 14,
            "orders_7d": 2, "orders_30d": 3, "orders_90d": 3,
            "returns_7d": 2, "returns_30d": 3, "returns_90d": 3,
            "return_rate_30d": 1.00, "return_rate_90d": 1.00,
            "refund_amount_30d": 76000.0, "refund_amount_90d": 76000.0,
            "hours_to_return": 1.8, "return_value_ratio": 1.0,
            "same_product_returns_90d": 1, "same_category_returns_90d": 3,
            "velocity_24h": 2, "velocity_7d": 3,
            "device_linked_accounts": 6, "address_linked_accounts": 5,
            "device_return_rate_90d": 0.85, "address_return_rate_90d": 0.80,
            "high_value_flag": 1,
            "prediction_time": pd.Timestamp.now()
        },
        {
            "attack_type": "Empty Box Electronics Scam",
            "description": "Buying expensive electronics, filing return request 1.2 hours after delivery claiming wrong item.",
            "return_id": "ATK_EMPTYBOX_03",
            "customer_id": "C_ATK_003",
            "order_id": "O_ATK_003",
            "product_category": "electronics",
            "payment_method": "cod",
            "return_reason": "wrong_item",
            "order_value": 45000.0, "return_value": 45000.0, "product_price": 45000.0, "discount_pct": 0.0,
            "customer_account_age_days": 18,
            "orders_7d": 1, "orders_30d": 2, "orders_90d": 2,
            "returns_7d": 1, "returns_30d": 2, "returns_90d": 2,
            "return_rate_30d": 1.00, "return_rate_90d": 1.00,
            "refund_amount_30d": 45000.0, "refund_amount_90d": 45000.0,
            "hours_to_return": 1.2, "return_value_ratio": 1.0,
            "same_product_returns_90d": 1, "same_category_returns_90d": 2,
            "velocity_24h": 1, "velocity_7d": 2,
            "device_linked_accounts": 4, "address_linked_accounts": 3,
            "device_return_rate_90d": 0.75, "address_return_rate_90d": 0.70,
            "high_value_flag": 1,
            "prediction_time": pd.Timestamp.now()
        },
        {
            "attack_type": "Velocity Storm Scam",
            "description": "Rapid sequence of 5 return requests filed across 3 consecutive days.",
            "return_id": "ATK_VELOCITY_04",
            "customer_id": "C_ATK_004",
            "order_id": "O_ATK_004",
            "product_category": "home",
            "payment_method": "upi",
            "return_reason": "not_as_described",
            "order_value": 18000.0, "return_value": 18000.0, "product_price": 18000.0, "discount_pct": 0.0,
            "customer_account_age_days": 40,
            "orders_7d": 5, "orders_30d": 8, "orders_90d": 12,
            "returns_7d": 5, "returns_30d": 7, "returns_90d": 10,
            "return_rate_30d": 0.88, "return_rate_90d": 0.83,
            "refund_amount_30d": 54000.0, "refund_amount_90d": 92000.0,
            "hours_to_return": 4.5, "return_value_ratio": 1.0,
            "same_product_returns_90d": 2, "same_category_returns_90d": 4,
            "velocity_24h": 3, "velocity_7d": 5,
            "device_linked_accounts": 3, "address_linked_accounts": 3,
            "device_return_rate_90d": 0.60, "address_return_rate_90d": 0.55,
            "high_value_flag": 1,
            "prediction_time": pd.Timestamp.now()
        }
    ]
    
    results = []
    for sc in attack_scenarios:
        df_row = pd.DataFrame([sc])
        prob = float(predict_bundle(bundle, df_row)[0])
        decision = "AUTO_APPROVE" if prob < t1 else "VERIFY" if prob < t2 else "MANUAL_REVIEW"
        intercepted = int(decision in ("VERIFY", "MANUAL_REVIEW"))
        
        results.append({
            "Attack Vector": sc["attack_type"],
            "Description": sc["description"],
            "Order Value": f"₹{sc['order_value']:,.2f}",
            "Assessed Abuse Risk": f"{prob:.1%}",
            "Policy Action": decision,
            "Defense Intercepted?": "✅ Intercepted" if intercepted else "❌ Missed"
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    df_res = run_redteam_simulation()
    print("=" * 70)
    print("🛡️ ReturnShield Red-Team Fraud Defense Simulation Benchmark")
    print("=" * 70)
    print(df_res.to_string(index=False))
