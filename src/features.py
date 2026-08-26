from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORICAL = ["product_category", "payment_method", "return_reason"]
NUMERIC = [
    "order_value", "product_price", "discount_pct", "customer_account_age_days",
    "orders_7d", "orders_30d", "orders_90d", "returns_7d", "returns_30d", "returns_90d",
    "return_rate_30d", "return_rate_90d", "refund_amount_30d", "refund_amount_90d",
    "hours_to_return", "return_value_ratio", "same_product_returns_90d", "same_category_returns_90d",
    "velocity_24h", "velocity_7d", "device_linked_accounts", "address_linked_accounts",
    "device_return_rate_90d", "address_return_rate_90d", "high_value_flag",
]
TARGET = "abusive_return"


def build_feature_table(data_dir: str = "data/raw") -> pd.DataFrame:
    p = Path(data_dir)
    customers = pd.read_csv(p / "customers.csv", parse_dates=["account_created_at"])
    orders = pd.read_csv(p / "orders.csv", parse_dates=["order_created_at", "delivery_at"])
    returns = pd.read_csv(p / "returns.csv", parse_dates=["return_requested_at"])
    outcomes = pd.read_csv(p / "return_outcomes.csv")

    base = returns.merge(orders, on=["order_id", "customer_id"], how="left", validate="one_to_one")
    base = base.merge(customers[["customer_id", "account_created_at"]], on="customer_id", how="left", validate="many_to_one")
    base = base.merge(outcomes[["return_id", "merchant_loss", "abusive_return"]], on="return_id", how="left", validate="one_to_one")
    base = base.sort_values("return_requested_at").reset_index(drop=True)
    orders = orders.sort_values("order_created_at").reset_index(drop=True)

    customer_events = defaultdict(deque)
    device_events = defaultdict(deque)
    address_events = defaultdict(deque)
    customer_products = defaultdict(deque)
    customer_categories = defaultdict(deque)
    order_ptr = 0

    rows = []

    def prune(q, now):
        cutoff = now - pd.Timedelta(days=90)
        while q and q[0][0] < cutoff:
            q.popleft()

    def add_order(orow):
        ts = orow["order_created_at"]
        cid = orow["customer_id"]
        did = orow["device_id"]
        aid = orow["address_id"]
        pid = orow["product_id"]
        cat = orow["product_category"]
        event = {"event": "order", "customer_id": cid, "product_id": pid, "category": cat, "refund": 0.0}
        customer_events[cid].append((ts, event))
        device_events[did].append((ts, event))
        address_events[aid].append((ts, event))

    for _, r in base.iterrows():
        now = r["return_requested_at"]
        while order_ptr < len(orders) and orders.iloc[order_ptr]["order_created_at"] < now:
            add_order(orders.iloc[order_ptr])
            order_ptr += 1

        cid = r["customer_id"]
        did = r["device_id"]
        aid = r["address_id"]
        pid = r["product_id"]
        cat = r["product_category"]

        for q in [customer_events[cid], device_events[did], address_events[aid], customer_products[cid], customer_categories[cid]]:
            prune(q, now)

        ce = list(customer_events[cid])
        de = list(device_events[did])
        ae = list(address_events[aid])

        def count_window(events, days, event_name=None):
            cutoff = now - pd.Timedelta(days=days)
            return sum(1 for ts, obj in events if ts >= cutoff and (event_name is None or obj["event"] == event_name))

        orders_7d = count_window(ce, 7, "order")
        orders_30d = count_window(ce, 30, "order")
        orders_90d = count_window(ce, 90, "order")
        returns_7d = count_window(ce, 7, "return")
        returns_30d = count_window(ce, 30, "return")
        returns_90d = count_window(ce, 90, "return")
        refund_30d = sum(obj.get("refund", 0.0) for ts, obj in ce if ts >= now - pd.Timedelta(days=30) and obj["event"] == "return")
        refund_90d = sum(obj.get("refund", 0.0) for ts, obj in ce if ts >= now - pd.Timedelta(days=90) and obj["event"] == "return")

        device_accounts = {obj["customer_id"] for ts, obj in de if ts >= now - pd.Timedelta(days=90)}
        address_accounts = {obj["customer_id"] for ts, obj in ae if ts >= now - pd.Timedelta(days=90)}
        device_returns = sum(1 for ts, obj in de if ts >= now - pd.Timedelta(days=90) and obj["event"] == "return")
        device_orders = sum(1 for ts, obj in de if ts >= now - pd.Timedelta(days=90) and obj["event"] == "order")
        address_returns = sum(1 for ts, obj in ae if ts >= now - pd.Timedelta(days=90) and obj["event"] == "return")
        address_orders = sum(1 for ts, obj in ae if ts >= now - pd.Timedelta(days=90) and obj["event"] == "order")

        account_age = max(1, (now - r["account_created_at"]).days)
        order_value = float(r["order_value"])
        return_value = float(r["return_value"])
        previous_customer_products = list(customer_products[cid])
        previous_customer_categories = list(customer_categories[cid])

        rows.append({
            "return_id": r["return_id"], "order_id": r["order_id"], "customer_id": cid,
            "prediction_time": now,
            "product_category": r["product_category"], "payment_method": r["payment_method"], "return_reason": r["return_reason"],
            "order_value": order_value, "return_value": return_value, "product_price": float(r["product_price"]), "discount_pct": float(r["discount_pct"]),
            "customer_account_age_days": account_age,
            "orders_7d": orders_7d, "orders_30d": orders_30d, "orders_90d": orders_90d,
            "returns_7d": returns_7d, "returns_30d": returns_30d, "returns_90d": returns_90d,
            "return_rate_30d": returns_30d / max(orders_30d, 1), "return_rate_90d": returns_90d / max(orders_90d, 1),
            "refund_amount_30d": refund_30d, "refund_amount_90d": refund_90d,
            "hours_to_return": max(0.01, (now - r["delivery_at"]).total_seconds() / 3600.0),
            "return_value_ratio": return_value / max(order_value, 1.0),
            "same_product_returns_90d": sum(1 for ts, pid_prev in previous_customer_products if ts >= now - pd.Timedelta(days=90) and pid_prev == pid),
            "same_category_returns_90d": sum(1 for ts, cat_prev in previous_customer_categories if ts >= now - pd.Timedelta(days=90) and cat_prev == cat),
            "velocity_24h": count_window(ce, 1), "velocity_7d": count_window(ce, 7),
            "device_linked_accounts": len(device_accounts), "address_linked_accounts": len(address_accounts),
            "device_return_rate_90d": device_returns / max(device_orders, 1), "address_return_rate_90d": address_returns / max(address_orders, 1),
            "high_value_flag": int(order_value >= 7500.0),
            "abusive_return": int(r["abusive_return"]), "merchant_loss": float(r["merchant_loss"]),
        })

        customer_products[cid].append((now, pid))
        customer_categories[cid].append((now, cat))
        return_event = {"event": "return", "customer_id": cid, "product_id": pid, "category": cat, "refund": return_value}
        customer_events[cid].append((now, return_event))
        device_events[did].append((now, return_event))
        address_events[aid].append((now, return_event))

    return pd.DataFrame(rows).sort_values("prediction_time").reset_index(drop=True)


def save_feature_table(data_dir="data/raw", out_path="data/processed/features.csv"):
    table = build_feature_table(data_dir)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    
    # Save Parquet for high-speed columnar access if pyarrow/fastparquet is available
    try:
        parquet_path = out_path.replace(".csv", ".parquet")
        table.to_parquet(parquet_path, index=False)
    except Exception:
        pass
    return table
