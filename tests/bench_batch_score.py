"""Measure the batching change: N single-row predict calls vs one predict call.

Run: python tests/bench_batch_score.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.main import _expand_row, get_bundle  # noqa: E402
from src.model import predict_bundle  # noqa: E402
from src.explain import top_features, concise_reasoning  # noqa: E402


def make_payloads(n: int) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "return_id": f"R{i:06d}",
                "order_id": f"O{i:06d}",
                "customer_id": f"C{i % 500:06d}",
                "product_category": "electronics",
                "payment_method": "card",
                "return_reason": "not_as_described",
                "order_value": 5000.0 + (i % 17) * 750,
                "product_price": 5000.0 + (i % 17) * 750,
                "discount_pct": 0.0,
                "customer_account_age_days": 200,
                "orders_7d": 2,
                "orders_30d": 6,
                "orders_90d": 12,
                "returns_7d": 1,
                "returns_30d": 5,
                "returns_90d": 9,
                "refund_amount_30d": 4000.0,
                "refund_amount_90d": 15000.0,
                "hours_to_return": 4.0,
                "same_product_returns_90d": 1,
                "same_category_returns_90d": 3,
                "velocity_24h": 2,
                "velocity_7d": 4,
                "device_linked_accounts": 3,
                "address_linked_accounts": 2,
                "device_return_rate_90d": 0.3,
                "address_return_rate_90d": 0.25,
            }
        )
    return rows


def bench(n: int, repeats: int = 3) -> None:
    bundle, _policy = get_bundle()
    payloads = make_payloads(n)

    # Old behaviour: build a frame per row and predict each one.
    t0 = time.perf_counter()
    for _ in range(repeats):
        for payload in payloads:
            frame = pd.DataFrame([_expand_row(dict(payload))])
            predict_bundle(bundle, frame)
    loop_ms = (time.perf_counter() - t0) * 1000.0 / repeats

    # New behaviour: one predict pass for the whole batch.
    t0 = time.perf_counter()
    for _ in range(repeats):
        frame = pd.DataFrame([_expand_row(dict(p)) for p in payloads])
        predict_bundle(bundle, frame)
    single_ms = (time.perf_counter() - t0) * 1000.0 / repeats

    # The explanation step is unchanged, so measure it separately.
    frame = pd.DataFrame([_expand_row(dict(p)) for p in payloads])
    t0 = time.perf_counter()
    for i in range(min(n, 50)):
        row = frame.iloc[[i]]
        concise_reasoning(row.iloc[0], top_features(bundle, row, top_n=5))
    shap_ms = (time.perf_counter() - t0) * 1000.0

    speedup = loop_ms / single_ms if single_ms else float("nan")
    print(f"n={n:>4}  per-row loop: {loop_ms:9.2f} ms   single pass: {single_ms:8.2f} ms   "
          f"speedup: {speedup:5.2f}x   (shap for {min(n, 50):>3} rows: {shap_ms:8.2f} ms)")


if __name__ == "__main__":
    for size in (1, 10, 50, 200):
        bench(size)
