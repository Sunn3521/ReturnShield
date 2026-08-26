from __future__ import annotations

import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

def generate_200mb_returns(out_path: str = "test_merchant_200mb_returns.csv", target_rows: int = 1200000):
    print("=" * 70)
    print(f"[+] Generating 200MB Enterprise Return Dataset ({target_rows:,} transactions)...")
    print("=" * 70)
    
    t0 = time.perf_counter()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.unlink()

    categories = np.array(["electronics", "fashion", "home", "beauty", "sports", "grocery"])
    cat_p = np.array([.25, .30, .15, .10, .10, .10])
    base_price = {"electronics": 12000, "fashion": 2800, "home": 4500, "beauty": 1500, "sports": 3200, "grocery": 1100}
    reasons = np.array(["damaged", "wrong_item", "changed_mind", "size_fit", "not_as_described"])
    reason_p = np.array([.15, .20, .25, .20, .20])
    payments = np.array(["upi", "card", "cod", "wallet"])
    payment_p = np.array([.40, .35, .15, .10])

    chunk_size = 100000
    total_chunks = (target_rows + chunk_size - 1) // chunk_size

    rng = np.random.default_rng(42)
    first_chunk = True

    for chunk_idx in range(total_chunks):
        rows_in_chunk = min(chunk_size, target_rows - chunk_idx * chunk_size)
        start_id = chunk_idx * chunk_size

        cats = rng.choice(categories, size=rows_in_chunk, p=cat_p)
        prices = np.zeros(rows_in_chunk)
        for cat_name in base_price:
            mask = cats == cat_name
            if np.any(mask):
                prices[mask] = np.clip(rng.lognormal(np.log(base_price[cat_name]), 0.7, size=mask.sum()), 250, 85000)

        discounts = np.clip(rng.beta(2.0, 8.0, size=rows_in_chunk) * 0.5, 0.0, 0.5)
        order_vals = np.round(prices * (1.0 - discounts), 2)

        account_ages = rng.integers(10, 1200, size=rows_in_chunk)
        orders_90d = rng.integers(1, 30, size=rows_in_chunk)
        returns_90d = (orders_90d * rng.uniform(0.05, 0.85, size=rows_in_chunk)).astype(int)
        returns_30d = np.minimum(returns_90d, rng.integers(0, 15, size=rows_in_chunk))
        returns_7d = np.minimum(returns_30d, rng.integers(0, 6, size=rows_in_chunk))
        orders_30d = np.maximum(returns_30d + 1, (orders_90d * rng.uniform(0.3, 0.7, size=rows_in_chunk)).astype(int))
        orders_7d = np.maximum(returns_7d + 1, (orders_30d * rng.uniform(0.2, 0.6, size=rows_in_chunk)).astype(int))

        refund_90d = np.round(returns_90d * order_vals * rng.uniform(0.6, 1.2, size=rows_in_chunk), 2)
        refund_30d = np.round(returns_30d * order_vals * rng.uniform(0.6, 1.2, size=rows_in_chunk), 2)
        hours_ret = np.round(np.clip(rng.lognormal(np.log(36), 0.8, size=rows_in_chunk), 0.5, 360), 1)

        dev_acc = rng.integers(1, 8, size=rows_in_chunk)
        addr_acc = rng.integers(1, 6, size=rows_in_chunk)
        dev_ret_rate = np.round(rng.uniform(0.05, 0.90, size=rows_in_chunk), 3)
        addr_ret_rate = np.round(rng.uniform(0.05, 0.85, size=rows_in_chunk), 3)

        chunk_df = pd.DataFrame({
            "return_id": [f"R_MEGA_{start_id + i + 1:07d}" for i in range(rows_in_chunk)],
            "order_id": [f"O_MEGA_{start_id + i + 1:07d}" for i in range(rows_in_chunk)],
            "customer_id": [f"C_MEGA_{rng.integers(1, 50000):06d}" for _ in range(rows_in_chunk)],
            "product_category": cats,
            "payment_method": rng.choice(payments, size=rows_in_chunk, p=payment_p),
            "return_reason": rng.choice(reasons, size=rows_in_chunk, p=reason_p),
            "order_value": order_vals,
            "return_value": order_vals,
            "product_price": np.round(prices, 2),
            "discount_pct": np.round(discounts * 100, 2),
            "customer_account_age_days": account_ages,
            "orders_7d": orders_7d,
            "orders_30d": orders_30d,
            "orders_90d": orders_90d,
            "returns_7d": returns_7d,
            "returns_30d": returns_30d,
            "returns_90d": returns_90d,
            "return_rate_30d": np.round(returns_30d / np.maximum(orders_30d, 1), 3),
            "return_rate_90d": np.round(returns_90d / np.maximum(orders_90d, 1), 3),
            "refund_amount_30d": refund_30d,
            "refund_amount_90d": refund_90d,
            "hours_to_return": hours_ret,
            "return_value_ratio": 1.0,
            "same_product_returns_90d": rng.integers(0, 4, size=rows_in_chunk),
            "same_category_returns_90d": rng.integers(0, 6, size=rows_in_chunk),
            "velocity_24h": rng.integers(0, 4, size=rows_in_chunk),
            "velocity_7d": rng.integers(0, 8, size=rows_in_chunk),
            "device_linked_accounts": dev_acc,
            "address_linked_accounts": addr_acc,
            "device_return_rate_90d": dev_ret_rate,
            "address_return_rate_90d": addr_ret_rate,
            "high_value_flag": (order_vals >= 7500.0).astype(int)
        })

        chunk_df.to_csv(out_path, mode="a", index=False, header=first_chunk)
        first_chunk = False
        print(f"   - Generated chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_df):,} rows)...")

    elapsed = time.perf_counter() - t0
    file_bytes = os.path.getsize(out_path)
    file_mb = file_bytes / (1024 * 1024)

    print("=" * 70)
    print(f"[OK] Successfully generated 200MB Dataset:")
    print(f"   - Total Rows:   {target_rows:,} transactions")
    print(f"   - File Size:    {file_mb:.2f} MB")
    print(f"   - Time Taken:   {elapsed:.2f} seconds")
    print(f"   - Output Path:  {out_path}")
    print("=" * 70)

if __name__ == "__main__":
    generate_200mb_returns("test_merchant_200mb_returns.csv")
