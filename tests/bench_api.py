"""Latency + throughput benchmark for the ReturnShield API.

Run against a live server:  python tests/bench_api.py http://127.0.0.1:8000
Reports p50/p95/p99 and requests-per-second per endpoint, plus batch scoring
throughput (rows/second), which is the number the README quotes.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 60


def call(method: str, path: str, payload=None, opener=None) -> tuple[int, float]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        exc.read()
        status = exc.code
    return status, (time.perf_counter() - start) * 1000.0


def sample(method: str, path: str, payload=None, n: int = 100, warmup: int = 5) -> dict:
    for _ in range(warmup):
        call(method, path, payload)
    samples: list[float] = []
    statuses: set[int] = set()
    for _ in range(n):
        status, ms = call(method, path, payload)
        statuses.add(status)
        samples.append(ms)
    samples.sort()
    return {
        "n": n,
        "status": sorted(statuses),
        "p50": statistics.median(samples),
        "p95": samples[int(len(samples) * 0.95) - 1],
        "p99": samples[int(len(samples) * 0.99) - 1],
        "mean": statistics.fmean(samples),
        "rps": 1000.0 / statistics.fmean(samples),
    }


def row(i: int) -> dict:
    return {
        "return_id": f"R-BENCH-{i:05d}",
        "order_id": f"O-{i:05d}",
        "customer_id": f"C-{i % 500:05d}",
        "order_value": 1000.0 + (i % 900),
        "product_price": 1000.0 + (i % 900),
        "return_reason": "damaged" if i % 3 else "not as described",
    }


def main() -> None:
    print(f"benchmarking {BASE}")
    started = time.perf_counter()

    checks = [
        ("GET ", "/api/v1/meta", None, 150),
        ("GET ", "/api/v1/health", None, 150),
        ("GET ", "/api/v1/returns?limit=100", None, 100),
        ("GET ", "/api/v1/returns/stats", None, 100),
        ("POST", "/api/v1/score", row(1), 200),
    ]
    print(f"{'endpoint':<34}{'n':>5}{'p50':>9}{'p95':>9}{'p99':>9}{'req/s':>9}  status")
    for method, path, payload, n in checks:
        r = sample(method, path, payload, n=n)
        print(
            f"{method + ' ' + path:<34}{r['n']:>5}{r['p50']:>8.1f}ms"
            f"{r['p95']:>8.1f}ms{r['p99']:>8.1f}ms{r['rps']:>9.0f}  {r['status']}"
        )

    # Batch scoring: rows/second is the throughput number that matters.
    for batch in (100, 1000):
        payload = [row(i) for i in range(batch)]
        call("POST", "/api/v1/batch_score", payload)  # warm
        best = 0.0
        runs = 5 if batch == 100 else 3
        for _ in range(runs):
            _, ms = call("POST", "/api/v1/batch_score", payload)
            best = max(best, batch / (ms / 1000.0))
        print(f"{'POST /api/v1/batch_score (' + str(batch) + ' rows)':<34}"
              f"{runs:>5}{'':>9}{'':>9}{'':>9}{best:>9.0f}  rows/s (best)")

    # Concurrent single-score throughput (8 in-flight requests).
    payload = [row(i) for i in range(64)]
    for _ in range(2):
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i: call("POST", "/api/v1/score", row(i)), range(64)))
    started_c = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: call("POST", "/api/v1/score", row(i)), range(64)))
    elapsed = time.perf_counter() - started_c
    statuses = sorted({s for s, _ in results})
    print(f"{'POST /api/v1/score (8 concurrent)':<34}{len(results):>5}"
          f"{'':>9}{'':>9}{'':>9}{len(results) / elapsed:>9.0f}  req/s status={statuses}")

    print(f"\ntotal benchmark wall time: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
