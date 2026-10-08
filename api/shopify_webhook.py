from __future__ import annotations

import json
import time
from typing import Dict, Any
import pandas as pd

from src.model import load_bundle, predict_bundle
from src.responder import generate_agent_response

def process_shopify_return_webhook(webhook_payload: Dict[str, Any], bundle_path: str = "models/model_bundle.joblib", policy_path: str = "models/policy.json") -> Dict[str, Any]:
    """
    Parses standard Shopify/WooCommerce return webhook payloads, calculates
    ReturnShield abuse risk, and formats operational response actions.
    """
    t0 = time.perf_counter()
    
    with open(policy_path, "r", encoding="utf-8") as f:
        policy = json.load(f)
    bundle = load_bundle(bundle_path)
    
    # Parse Shopify Webhook Payload fields
    return_id = str(webhook_payload.get("id", f"R_SHOPIFY_{int(time.time())}"))
    order_id = str(webhook_payload.get("order_id", "O_SHOPIFY_1001"))
    customer = webhook_payload.get("customer", {})
    customer_id = str(customer.get("id", "C_SHOPIFY_001"))
    
    line_items = webhook_payload.get("return_line_items", [{}])
    item_val = float(line_items[0].get("total_discounted_amount", 14500.0)) if line_items else 14500.0
    category = str(line_items[0].get("category", "electronics")) if line_items else "electronics"
    
    reason = str(webhook_payload.get("reason", "damaged"))
    
    # Construct feature frame
    feature_dict = {
        "return_id": return_id,
        "order_id": order_id,
        "customer_id": customer_id,
        "product_category": category,
        "payment_method": "card",
        "return_reason": reason,
        "order_value": item_val,
        "return_value": item_val,
        "product_price": item_val,
        "discount_pct": 0.0,
        "customer_account_age_days": customer.get("account_age_days", 90),
        "orders_7d": customer.get("orders_7d", 2),
        "orders_30d": customer.get("orders_30d", 5),
        "orders_90d": customer.get("orders_90d", 8),
        "returns_7d": customer.get("returns_7d", 1),
        "returns_30d": customer.get("returns_30d", 3),
        "returns_90d": customer.get("returns_90d", 5),
        "return_rate_30d": customer.get("returns_30d", 3) / max(customer.get("orders_30d", 5), 1),
        "return_rate_90d": customer.get("returns_90d", 5) / max(customer.get("orders_90d", 8), 1),
        "refund_amount_30d": customer.get("refund_amount_30d", 12000.0),
        "refund_amount_90d": customer.get("refund_amount_90d", 32000.0),
        "hours_to_return": float(webhook_payload.get("hours_since_delivery", 4.2)),
        "return_value_ratio": 1.0,
        "same_product_returns_90d": 1,
        "same_category_returns_90d": 2,
        "velocity_24h": 1,
        "velocity_7d": 2,
        "device_linked_accounts": customer.get("linked_device_accounts", 3),
        "address_linked_accounts": customer.get("linked_address_accounts", 2),
        "device_return_rate_90d": 0.45,
        "address_return_rate_90d": 0.40,
        "high_value_flag": int(item_val >= 7500.0),
        "prediction_time": pd.Timestamp.now()
    }
    
    df_row = pd.DataFrame([feature_dict])
    prob = float(predict_bundle(bundle, df_row)[0])
    
    t1 = policy["verify_threshold"]
    t2 = policy["review_threshold"]
    
    if prob < t1:
        decision = "AUTO_APPROVE"
        shopify_status = "approved"
    elif prob < t2:
        decision = "VERIFY"
        shopify_status = "requires_evidence"
    else:
        decision = "MANUAL_REVIEW"
        shopify_status = "hold_pending_review"
        
    reasons = [
        f"Historical return rate: {feature_dict['return_rate_90d']:.0%}",
        f"Linked infrastructure: {feature_dict['device_linked_accounts']} accounts",
        f"Return delay: {feature_dict['hours_to_return']:.1f} hours after delivery"
    ]
    
    df_row_series = df_row.iloc[0].copy()
    df_row_series["risk_probability"] = prob
    df_row_series["decision"] = decision
    agent_output = generate_agent_response(df_row_series, reasons)
    
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    return {
        "webhook_processed": True,
        "shopify_return_id": return_id,
        "risk_probability": round(prob, 4),
        "policy_decision": decision,
        "shopify_action_status": shopify_status,
        "merchant_protocol": agent_output["merchant_action"],
        "customer_communication": agent_output["customer_message"],
        "processing_time_ms": round(latency_ms, 2)
    }
