from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SimulationConfig:
    seed: int = 42
    n_customers: int = 10000
    n_orders: int = 70000
    start_date: str = "2025-01-01"
    n_days: int = 365


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 30), -30)))


def simulate(cfg: SimulationConfig):
    rng = np.random.default_rng(cfg.seed)
    start = pd.Timestamp(cfg.start_date)

    # Latent customer behavior; never exposed directly as model features.
    customer_types = rng.choice(
        ["normal", "frequent_legit", "abusive", "coordinated"],
        size=cfg.n_customers,
        p=[0.90, 0.07, 0.02, 0.01],
    )
    customer_ids = np.array([f"C{i:05d}" for i in range(cfg.n_customers)])

    regions = rng.choice(["N", "S", "E", "W", "NE"], size=cfg.n_customers, p=[.28,.28,.16,.22,.06])
    account_age_days = rng.integers(15, 1800, size=cfg.n_customers)

    # Mostly unique infrastructure, with coordinated customers deliberately sharing clusters.
    device_ids = np.array([f"D{i:05d}" for i in range(cfg.n_customers)], dtype=object)
    address_ids = np.array([f"A{i:05d}" for i in range(cfg.n_customers)], dtype=object)
    payment_ids = np.array([f"P{i:05d}" for i in range(cfg.n_customers)], dtype=object)
    coord_idx = np.where(np.isin(customer_types, ["coordinated"]))[0]
    if len(coord_idx) > 0:
        for cluster_no, chunk in enumerate(np.array_split(coord_idx, max(1, len(coord_idx)//5))):
            if len(chunk) == 0:
                continue
            device = f"DCO{cluster_no:04d}"
            address = f"ACO{cluster_no:04d}"
            payment = f"PCO{cluster_no:04d}"
            device_ids[chunk] = device
            address_ids[chunk] = address
            payment_ids[chunk] = payment

    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "account_created_at": start + pd.to_timedelta((cfg.n_days - account_age_days).clip(min=0), unit="D"),
        "home_region": regions,
        "latent_type": customer_types,
        "device_id": device_ids,
        "address_id": address_ids,
        "payment_fingerprint": payment_ids,
    })

    categories = np.array(["electronics", "fashion", "home", "beauty", "sports", "grocery"])
    cat_p = np.array([.18, .28, .18, .12, .12, .12])
    base_price = {"electronics": 9000, "fashion": 1800, "home": 3200, "beauty": 1200, "sports": 2200, "grocery": 900}

    order_rows = []
    return_rows = []
    outcome_rows = []

    # We maintain event history so every feature is point-in-time.
    history = {cid: [] for cid in customer_ids}
    device_history = {}
    address_history = {}

    order_dates = start + pd.to_timedelta(rng.integers(0, cfg.n_days, size=cfg.n_orders), unit="D")
    order_dates = pd.Series(order_dates).sort_values().reset_index(drop=True)

    for i, order_date in enumerate(order_dates):
        ci = int(rng.integers(0, cfg.n_customers))
        cid = customer_ids[ci]
        ctype = customer_types[ci]
        age_at_order = max(1, (order_date - customers.loc[ci, "account_created_at"]).days)

        cat = rng.choice(categories, p=cat_p)
        price = float(np.clip(rng.lognormal(np.log(base_price[cat]), 0.65), 150, 100000))
        if cat == "electronics":
            price *= rng.uniform(0.8, 1.35)
        discount = float(np.clip(rng.beta(2.0, 7.0) * 0.6, 0.0, 0.6))
        order_value = price * (1 - discount)
        payment = rng.choice(["upi", "card", "cod", "wallet"], p=[.38,.34,.18,.10])

        # Behavior-dependent ordering intensity.
        if ctype in ("abusive", "coordinated") and rng.random() < 0.18:
            order_value *= rng.uniform(1.1, 2.5)

        delivery_delay = int(rng.integers(1, 6))
        delivery_at = order_date + pd.Timedelta(days=delivery_delay)
        order_id = f"O{i:06d}"
        device_id = customers.loc[ci, "device_id"]
        address_id = customers.loc[ci, "address_id"]
        payment_fp = customers.loc[ci, "payment_fingerprint"]

        order_rows.append({
            "order_id": order_id,
            "customer_id": cid,
            "order_created_at": order_date,
            "delivery_at": delivery_at,
            "product_id": f"SKU{rng.integers(0, 4500):05d}",
            "product_category": cat,
            "product_price": round(price, 2),
            "discount_pct": round(discount * 100, 2),
            "order_value": round(order_value, 2),
            "payment_method": payment,
            "device_id": device_id,
            "address_id": address_id,
            "payment_fingerprint": payment_fp,
            "region": customers.loc[ci, "home_region"],
            "customer_account_age_at_order": age_at_order,
        })

        # Probability of a return. Legitimate frequent returners can overlap with abusive customers.
        if ctype == "normal":
            logit = -2.45 + 0.25 * (cat == "fashion") + 0.12 * (cat == "electronics")
        elif ctype == "frequent_legit":
            logit = -1.10 + 0.30 * (cat == "fashion")
        elif ctype == "abusive":
            logit = -0.45 + 0.25 * (cat == "electronics") + 0.15 * (order_value > 10000)
        else:
            logit = -0.65 + 0.20 * (cat == "electronics") + 0.20 * (order_value > 10000)
        p_return = _sigmoid(logit)
        if rng.random() >= p_return:
            continue

        # Point-in-time history before this return request.
        customer_hist = history[cid]
        device_hist = device_history.get(device_id, [])
        address_hist = address_history.get(address_id, [])

        return_delay_hours = float(np.clip(rng.lognormal(np.log(48), 0.9), 2, 360))
        if ctype == "abusive":
            return_delay_hours = float(np.clip(rng.lognormal(np.log(18), 0.75), 0.5, 120))
        elif ctype == "coordinated":
            return_delay_hours = float(np.clip(rng.lognormal(np.log(24), 0.8), 0.5, 144))

        return_at = delivery_at + pd.to_timedelta(return_delay_hours, unit="h")
        reason = rng.choice(
            ["damaged", "wrong_item", "changed_mind", "size_fit", "not_as_described"],
            p=[.13,.16,.14,.28,.29],
        )
        return_id = f"R{len(return_rows):06d}"

        # Define outcome with latent mechanisms and stochastic noise.
        same_cat_returns = sum(1 for h in customer_hist if h["category"] == cat and h["is_return"])
        past_returns = sum(1 for h in customer_hist if h["is_return"])
        past_refunds = sum(h["refund"] for h in customer_hist if h["is_return"])
        linked_accounts = max(len({h["customer_id"] for h in device_hist + address_hist}), 1)

        return_rate_proxy = past_returns / max(sum(1 for h in customer_hist if h.get("event") == "order"), 1)
        abuse_score = (
            -4.8
            + 0.30 * past_returns
            + 1.15 * (return_delay_hours < 12)
            + 0.95 * (order_value > 12000)
            + 0.80 * (same_cat_returns >= 3)
            + 1.05 * (linked_accounts >= 3)
            + 0.95 * min(return_rate_proxy, 1.0)
            + 0.000018 * past_refunds
            + 1.15 * (ctype == "abusive")
            + 1.45 * (ctype == "coordinated")
            + rng.normal(0, 0.28)
        )
        abusive_prob = _sigmoid(abuse_score)
        abusive = int(rng.random() < abusive_prob)

        # Some legitimate returns still incur small operational loss.
        if abusive:
            loss = float(np.clip(order_value * rng.uniform(.18, .65) + rng.normal(900, 500), 750, 30000))
            rule = rng.choice(["refund_abuse", "condition_mismatch", "coordinated_abuse"], p=[.45,.30,.25])
            item_received = int(rng.random() > 0.28)
            condition_ok = int(rng.random() > 0.50)
        else:
            loss = float(np.clip(order_value * rng.uniform(.0, .06) + rng.normal(40, 25), 0, 600))
            rule = "none"
            item_received = int(rng.random() > 0.04)
            condition_ok = int(rng.random() > 0.04)

        returns_obj = {
            "return_id": return_id,
            "order_id": order_id,
            "customer_id": cid,
            "return_requested_at": return_at,
            "return_reason": reason,
            "return_value": round(order_value, 2),
        }
        outcome_obj = {
            "return_id": return_id,
            "item_received": item_received,
            "item_condition_ok": condition_ok,
            "refund_completed": 1,
            "merchant_loss": round(loss, 2),
            "abuse_rule_triggered": rule,
            "abusive_return": abusive,
        }
        return_rows.append(returns_obj)
        outcome_rows.append(outcome_obj)

        event = {
            "timestamp": return_at,
            "customer_id": cid,
            "category": cat,
            "is_return": True,
            "refund": order_value,
        }
        history[cid].append(event)
        device_history.setdefault(device_id, []).append(event)
        address_history.setdefault(address_id, []).append(event)

    orders = pd.DataFrame(order_rows)
    returns = pd.DataFrame(return_rows)
    outcomes = pd.DataFrame(outcome_rows)
    customers["latent_type"] = customers["latent_type"].astype(str)

    return customers, orders, returns, outcomes


def save_dataset(cfg: SimulationConfig, out_dir: str = "data/raw") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    customers, orders, returns, outcomes = simulate(cfg)
    paths = {
        "customers": out / "customers.csv",
        "orders": out / "orders.csv",
        "returns": out / "returns.csv",
        "outcomes": out / "return_outcomes.csv",
    }
    customers.to_csv(paths["customers"], index=False)
    orders.to_csv(paths["orders"], index=False)
    returns.to_csv(paths["returns"], index=False)
    outcomes.to_csv(paths["outcomes"], index=False)
    return {k: str(v) for k, v in paths.items()}


if __name__ == "__main__":
    save_dataset(SimulationConfig())
