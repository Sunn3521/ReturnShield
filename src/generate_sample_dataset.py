from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

def generate_1000_test_returns(out_path: str = "test_merchant_1000_returns.csv") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 1000

    categories = np.array(["electronics", "fashion", "home", "beauty", "sports", "grocery"])
    cat_p = np.array([.25, .30, .15, .10, .10, .10])
    base_price = {"electronics": 12000, "fashion": 2800, "home": 4500, "beauty": 1500, "sports": 3200, "grocery": 1100}
    reasons = np.array(["damaged", "wrong_item", "changed_mind", "size_fit", "not_as_described"])
    reason_p = np.array([.15, .20, .25, .20, .20])
    payments = np.array(["upi", "card", "cod", "wallet"])
    payment_p = np.array([.40, .35, .15, .10])

    rows = []
    for i in range(n):
        rid = f"R_CUST_{i+1:04d}"
        oid = f"O_CUST_{i+1:04d}"
        cid = f"C_USER_{rng.integers(1, 450):04d}"
        cat = rng.choice(categories, p=cat_p)
        price = float(np.clip(rng.lognormal(np.log(base_price[cat]), 0.7), 250, 85000))
        discount = float(np.clip(rng.beta(2.0, 8.0) * 0.5, 0.0, 0.5))
        order_val = price * (1.0 - discount)

        r_type = rng.choice(["normal", "frequent", "abusive", "ring"], p=[0.75, 0.15, 0.07, 0.03])

        if r_type == "normal":
            account_age = int(rng.integers(60, 1200))
            orders_90d = int(rng.integers(2, 15))
            returns_90d = int(rng.integers(0, max(1, int(orders_90d * 0.25))))
            returns_30d = int(rng.integers(0, max(1, returns_90d + 1)))
            returns_7d = int(rng.integers(0, max(1, returns_30d + 1)))
            orders_30d = int(rng.integers(returns_30d, max(returns_30d + 2, int(orders_90d * 0.6))))
            orders_7d = int(rng.integers(returns_7d, max(returns_7d + 2, int(orders_30d * 0.5))))
            refund_90d = float(returns_90d * order_val * rng.uniform(0.6, 1.0))
            refund_30d = float(returns_30d * order_val * rng.uniform(0.6, 1.0))
            hours_ret = float(np.clip(rng.lognormal(np.log(48), 0.8), 6, 240))
            dev_acc = 1
            addr_acc = 1
            dev_ret_rate = float(rng.uniform(0.05, 0.20))
            addr_ret_rate = float(rng.uniform(0.05, 0.20))

        elif r_type == "frequent":
            account_age = int(rng.integers(30, 800))
            orders_90d = int(rng.integers(8, 25))
            returns_90d = int(rng.integers(2, max(3, int(orders_90d * 0.45))))
            returns_30d = int(rng.integers(1, max(2, returns_90d + 1)))
            returns_7d = int(rng.integers(0, max(1, returns_30d + 1)))
            orders_30d = int(rng.integers(returns_30d, max(returns_30d + 2, int(orders_90d * 0.6))))
            orders_7d = int(rng.integers(returns_7d, max(returns_7d + 2, int(orders_30d * 0.5))))
            refund_90d = float(returns_90d * order_val * rng.uniform(0.7, 1.1))
            refund_30d = float(returns_30d * order_val * rng.uniform(0.7, 1.1))
            hours_ret = float(np.clip(rng.lognormal(np.log(24), 0.7), 3, 144))
            dev_acc = int(rng.integers(1, 3))
            addr_acc = int(rng.integers(1, 2))
            dev_ret_rate = float(rng.uniform(0.20, 0.40))
            addr_ret_rate = float(rng.uniform(0.20, 0.40))

        elif r_type == "abusive":
            account_age = int(rng.integers(10, 180))
            orders_90d = int(rng.integers(4, 15))
            returns_90d = int(rng.integers(3, max(4, int(orders_90d * 0.85))))
            returns_30d = int(rng.integers(2, max(3, returns_90d + 1)))
            returns_7d = int(rng.integers(1, max(2, returns_30d + 1)))
            orders_30d = int(rng.integers(returns_30d, max(returns_30d + 2, int(orders_90d * 0.7))))
            orders_7d = int(rng.integers(returns_7d, max(returns_7d + 2, int(orders_30d * 0.5))))
            refund_90d = float(returns_90d * order_val * rng.uniform(0.9, 1.5))
            refund_30d = float(returns_30d * order_val * rng.uniform(0.9, 1.5))
            hours_ret = float(np.clip(rng.lognormal(np.log(8), 0.6), 0.5, 36))
            dev_acc = int(rng.integers(2, 6))
            addr_acc = int(rng.integers(2, 5))
            dev_ret_rate = float(rng.uniform(0.50, 0.85))
            addr_ret_rate = float(rng.uniform(0.50, 0.80))

        else: # ring
            account_age = int(rng.integers(5, 60))
            orders_90d = int(rng.integers(4, 12))
            returns_90d = int(rng.integers(3, max(4, orders_90d + 1)))
            returns_30d = returns_90d
            returns_7d = int(rng.integers(2, max(3, returns_30d + 1)))
            orders_30d = orders_90d
            orders_7d = int(rng.integers(returns_7d, max(returns_7d + 2, orders_30d + 1)))
            refund_90d = float(returns_90d * order_val * rng.uniform(1.0, 1.8))
            refund_30d = refund_90d
            hours_ret = float(np.clip(rng.lognormal(np.log(4), 0.5), 0.2, 18))
            dev_acc = int(rng.integers(5, 12))
            addr_acc = int(rng.integers(4, 10))
            dev_ret_rate = float(rng.uniform(0.75, 0.95))
            addr_ret_rate = float(rng.uniform(0.70, 0.90))

        same_prod = int(rng.integers(0, 3)) if r_type in ("normal", "frequent") else int(rng.integers(1, 5))
        same_cat = int(rng.integers(0, 4)) if r_type in ("normal", "frequent") else int(rng.integers(2, 8))
        v24 = int(rng.integers(0, 3 if r_type in ("normal", "frequent") else 5))
        v7 = v24 + int(rng.integers(0, 4))

        rows.append({
            "return_id": rid,
            "order_id": oid,
            "customer_id": cid,
            "product_category": cat,
            "payment_method": rng.choice(payments, p=payment_p),
            "return_reason": rng.choice(reasons, p=reason_p),
            "order_value": round(order_val, 2),
            "return_value": round(order_val, 2),
            "product_price": round(price, 2),
            "discount_pct": round(discount * 100, 2),
            "customer_account_age_days": account_age,
            "orders_7d": orders_7d,
            "orders_30d": orders_30d,
            "orders_90d": orders_90d,
            "returns_7d": returns_7d,
            "returns_30d": returns_30d,
            "returns_90d": returns_90d,
            "return_rate_30d": round(returns_30d / max(orders_30d, 1), 3),
            "return_rate_90d": round(returns_90d / max(orders_90d, 1), 3),
            "refund_amount_30d": round(refund_30d, 2),
            "refund_amount_90d": round(refund_90d, 2),
            "hours_to_return": round(hours_ret, 1),
            "return_value_ratio": 1.0,
            "same_product_returns_90d": same_prod,
            "same_category_returns_90d": same_cat,
            "velocity_24h": v24,
            "velocity_7d": v7,
            "device_linked_accounts": dev_acc,
            "address_linked_accounts": addr_acc,
            "device_return_rate_90d": round(dev_ret_rate, 3),
            "address_return_rate_90d": round(addr_ret_rate, 3),
            "high_value_flag": int(order_val >= 7500.0)
        })

    df = pd.DataFrame(rows)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} sample merchant return transactions at {out_path}")
    return df

if __name__ == "__main__":
    generate_1000_test_returns("test_merchant_1000_returns.csv")
