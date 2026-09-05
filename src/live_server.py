from __future__ import annotations
from pathlib import Path

import asyncio
import random
import json
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .explain import concise_reasoning, top_features
from .model import load_bundle, predict_bundle
from .responder import generate_agent_response

ROOT = Path(__file__).resolve().parent.parent if "Path" in globals() else None

# Runtime-only in-memory event store for the fake live server.
_EVENTS = deque(maxlen=100_000)
_HISTORY_LOCK = threading.Lock()
from pathlib import Path as _Path
_HISTORY_PATH = _Path(__file__).resolve().parent.parent / "data" / "live" / "events.jsonl"
_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
_LOCK = threading.Lock()
_GENERATOR_THREAD: threading.Thread | None = None
_GENERATOR_STOP = threading.Event()
_GENERATOR_RATE = 4.0
_EVENT_COUNTER = 0
_REGIME = "baseline"
_REGIME_STARTED_AT = time.monotonic()
_STARTED = False
# Process-persistent cumulative counters. These survive Streamlit page navigation.
_CUMULATIVE_COUNTS = {"total": 0, "AUTO_APPROVE": 0, "VERIFY": 0, "MANUAL_REVIEW": 0, "HIGH_RISK": 0}
_BUNDLE = None
_POLICY = None


def _ensure_model():
    global _BUNDLE, _POLICY
    if _BUNDLE is None:
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        _BUNDLE = load_bundle(str(root / "models" / "model_bundle.joblib"))
        import json
        with open(root / "models" / "policy.json", "r", encoding="utf-8") as f:
            _POLICY = json.load(f)
    return _BUNDLE, _POLICY


def _risk_decision(prob: float, policy: dict) -> str:
    if prob < float(policy["verify_threshold"]):
        return "AUTO_APPROVE"
    if prob < float(policy["review_threshold"]):
        return "VERIFY"
    return "MANUAL_REVIEW"


def _generate_one() -> dict[str, Any]:
    global _EVENT_COUNTER, _REGIME, _REGIME_STARTED_AT
    rng = np.random.default_rng()
    _ensure_model()

    # Deliberately non-stationary live stream so the dashboard visibly changes.
    # Regime rotates every ~12 seconds and changes both risk mix and ring activity.
    _EVENT_COUNTER += 1
    now_mono = time.monotonic()
    if now_mono - _REGIME_STARTED_AT >= 12:
        regimes = ["baseline", "elevated", "fraud_spike", "ring_surge", "recovery"]
        current_idx = regimes.index(_REGIME) if _REGIME in regimes else -1
        _REGIME = regimes[(current_idx + 1) % len(regimes)]
        _REGIME_STARTED_AT = now_mono

    regime_probs = {
        "baseline":   [0.64, 0.16, 0.10, 0.10],
        "elevated":   [0.48, 0.18, 0.20, 0.14],
        "fraud_spike": [0.30, 0.14, 0.30, 0.26],
        "ring_surge": [0.24, 0.10, 0.16, 0.50],
        "recovery":   [0.74, 0.12, 0.08, 0.06],
    }[_REGIME]
    scenario = rng.choice(["normal", "legit_frequent", "suspicious", "coordinated"], p=regime_probs)
    categories = ["electronics", "fashion", "home", "beauty", "sports", "grocery"]
    category = str(rng.choice(categories))
    payment = str(rng.choice(["upi", "card", "cod", "wallet"]))
    reason = str(rng.choice(["damaged", "wrong_item", "changed_mind", "size_fit", "not_as_described"]))

    base = {
        "normal": (8, 1, 3, 0, 0.55, 0.52),
        "legit_frequent": (28, 4, 12, 15000, 0.30, 0.25),
        "suspicious": (45, 7, 18, 32000, 0.10, 0.12),
        "coordinated": (62, 9, 24, 47000, 0.08, 0.08),
    }[scenario]
    return_rate, returns_30d, returns_90d, refund90, delay_mean, delay_cv = base

    order_value = float(np.clip(rng.lognormal(np.log(2500 if category != "electronics" else 9000), 0.75), 250, 75000))
    product_price = order_value / max(1e-6, (1 - float(rng.uniform(0, 0.4))))
    discount_pct = float(np.clip((1 - order_value / product_price) * 100, 0, 60))
    orders_30d = int(max(1, round(rng.normal(max(returns_30d / max(return_rate, 1e-6), 1), 2))))
    orders_90d = int(max(orders_30d, round(rng.normal(max(returns_90d / max(return_rate, 1e-6), orders_30d), 4))))
    returns_7d = int(max(0, rng.poisson(max(returns_30d / 4, 0.1))))
    orders_7d = int(max(returns_7d, rng.poisson(max(orders_30d / 4, 0.5))))
    velocity_24h = int(max(0, rng.poisson(max(returns_30d / 12, 0.15))))
    velocity_7d = int(max(velocity_24h, rng.poisson(max(returns_30d / 3, 0.4))))
    device_links = int(np.clip(round(rng.normal(1.3 if scenario == "normal" else 4.5, 1.5)), 1, 20))
    address_links = int(np.clip(round(rng.normal(1.2 if scenario == "normal" else 3.8, 1.4)), 1, 20))
    device_return_rate = float(np.clip(rng.normal(0.08 if scenario == "normal" else 0.58, 0.16), 0, 1))
    address_return_rate = float(np.clip(rng.normal(0.07 if scenario == "normal" else 0.52, 0.17), 0, 1))
    hours_to_return = float(np.clip(rng.lognormal(np.log(48 if scenario == "normal" else 14), 0.8), 0.5, 240))

    if scenario in ("suspicious", "coordinated"):
        cluster_id = f"CLIVE-{rng.integers(0, 12):03d}"
        live_device = f"D-{cluster_id}"
        live_address = f"A-{cluster_id}"
        live_payment = f"P-{cluster_id}"
    else:
        live_device = f"D-{uuid.uuid4().hex[:8].upper()}"
        live_address = f"A-{uuid.uuid4().hex[:8].upper()}"
        live_payment = f"P-{uuid.uuid4().hex[:8].upper()}"

    payload = {
        "return_id": f"LIVE-{uuid.uuid4().hex[:10].upper()}",
        "order_id": f"LIVE-O-{uuid.uuid4().hex[:10].upper()}",
        "customer_id": f"LIVE-C-{uuid.uuid4().hex[:7].upper()}",
        "device_id": live_device,
        "address_id": live_address,
        "payment_fingerprint": live_payment,
        "product_category": category,
        "payment_method": payment,
        "return_reason": reason,
        "order_value": round(order_value, 2),
        "return_value": round(order_value, 2),
        "product_price": round(product_price, 2),
        "discount_pct": round(discount_pct, 2),
        "customer_account_age_days": int(rng.integers(20, 1800)),
        "orders_7d": orders_7d,
        "orders_30d": orders_30d,
        "orders_90d": orders_90d,
        "returns_7d": returns_7d,
        "returns_30d": returns_30d,
        "returns_90d": returns_90d,
        "return_rate_30d": returns_30d / max(orders_30d, 1),
        "return_rate_90d": returns_90d / max(orders_90d, 1),
        "refund_amount_30d": round(max(0, refund90 * rng.uniform(0.18, 0.55)), 2),
        "refund_amount_90d": round(max(0, refund90 * rng.uniform(0.82, 1.18)), 2),
        "hours_to_return": hours_to_return,
        "same_product_returns_90d": int(max(0, rng.poisson(1.0 if scenario == "normal" else 2.5))),
        "same_category_returns_90d": int(max(0, rng.poisson(2.0 if scenario == "normal" else 5.0))),
        "velocity_24h": velocity_24h,
        "velocity_7d": velocity_7d,
        "device_linked_accounts": device_links,
        "address_linked_accounts": address_links,
        "device_return_rate_90d": device_return_rate,
        "address_return_rate_90d": address_return_rate,
        "return_value_ratio": 1.0,
        "high_value_flag": int(order_value >= 7500),
        "prediction_time": pd.Timestamp.now(tz="UTC"),
        "source": "FAKE_LIVE",
        "live_scenario": scenario,
        "live_regime": _REGIME,
        "event_sequence": _EVENT_COUNTER,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    df = pd.DataFrame([payload])
    bundle, policy = _ensure_model()
    prob = float(predict_bundle(bundle, df)[0])
    decision = _risk_decision(prob, policy)
    signal_items = top_features(bundle, df, top_n=5)
    reasons = concise_reasoning(df.iloc[0], signal_items)
    # Simulated eventual outcome for the built-in demo stream. This is intentionally
    # separate from the prediction-time risk score and is used only to make the
    # live calibration/outcomes visualization meaningful.
    scenario_outcome_rate = {
        "normal": 0.01,
        "legit_frequent": 0.04,
        "suspicious": 0.16,
        "coordinated": 0.32,
    }[scenario]
    observed_abuse = int(rng.random() < scenario_outcome_rate)

    enriched = payload.copy()
    enriched.update({
        "risk_probability": round(prob, 6),
        "decision": decision,
        "merchant_loss_estimate": round(float(np.clip(order_value * prob * 0.85, 0, order_value)), 2),
        "abusive_return": observed_abuse,
        "simulated_outcome": "ABUSIVE" if observed_abuse else "LEGITIMATE",
        "outcome_observed": True,
        "top_signals": reasons,
    })
    return enriched


def _persist_event(event: dict[str, Any]):
    try:
        line = json.dumps(event, default=str, separators=(",", ":"))
        with _HISTORY_LOCK:
            with _HISTORY_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        # The live UI can continue from its in-memory buffer if disk persistence fails.
        pass


def start_generator(rate_per_second: float = 4.0):
    global _GENERATOR_THREAD, _GENERATOR_RATE, _STARTED
    with _LOCK:
        _GENERATOR_RATE = max(0.2, min(float(rate_per_second), 20.0))
        if _GENERATOR_THREAD and _GENERATOR_THREAD.is_alive():
            _STARTED = True
            return
        _GENERATOR_STOP.clear()
        _GENERATOR_THREAD = threading.Thread(target=_loop, daemon=True, name="returnshield-live-generator")
        _GENERATOR_THREAD.start()
        _STARTED = True


def stop_generator():
    global _STARTED
    _GENERATOR_STOP.set()
    _STARTED = False


def _record_event(event: dict[str, Any]) -> None:
    """Persist and add one event while updating process-persistent cumulative counters."""
    _persist_event(event)
    with _LOCK:
        _EVENTS.appendleft(event)
        _CUMULATIVE_COUNTS["total"] += 1
        decision = event.get("decision", "AUTO_APPROVE")
        if decision in ("AUTO_APPROVE", "VERIFY", "MANUAL_REVIEW"):
            _CUMULATIVE_COUNTS[decision] += 1
        try:
            if float(event.get("risk_probability", 0)) >= float(_POLICY["review_threshold"]):
                _CUMULATIVE_COUNTS["HIGH_RISK"] += 1
        except Exception:
            pass


def _loop():
    while not _GENERATOR_STOP.is_set():
        try:
            _record_event(_generate_one())
        except Exception:
            pass
        time.sleep(1.0 / max(_GENERATOR_RATE, 0.2))


def generate_now(count: int = 1) -> list[dict[str, Any]]:
    out = []
    for _ in range(max(1, min(int(count), 1000))):
        event = _generate_one()
        _record_event(event)
        out.append(event)
    return out


def get_status() -> dict[str, Any]:
    with _LOCK:
        buffered = len(_EVENTS)
        latest = _EVENTS[0]["generated_at"] if _EVENTS else None
        cumulative = dict(_CUMULATIVE_COUNTS)
    return {
        "running": _STARTED,
        "rate_per_second": _GENERATOR_RATE,
        "buffered_records": buffered,
        "cumulative_records": cumulative.get("total", 0),
        "cumulative_counts": cumulative,
        "latest_generated_at": latest,
        "regime": _REGIME,
        "event_sequence": _EVENT_COUNTER,
    }


def list_events(limit: int = 100, offset: int = 0, search: str = "", before: str | None = None, after: str | None = None) -> tuple[list[dict], int]:
    with _LOCK:
        events = list(_EVENTS)
    if search:
        s = search.lower()
        events = [e for e in events if s in str(e.get("return_id", "")).lower() or s in str(e.get("customer_id", "")).lower() or s in str(e.get("decision", "")).lower()]
    if before:
        events = [e for e in events if str(e.get("generated_at", "")) <= before]
    if after:
        events = [e for e in events if str(e.get("generated_at", "")) >= after]
    total = len(events)
    return events[offset: offset + limit], total


def export_events_csv(before: str | None = None, after: str | None = None, search: str = "") -> pd.DataFrame:
    if _HISTORY_PATH.exists():
        rows = []
        with _HISTORY_LOCK:
            with _HISTORY_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        df = pd.DataFrame(rows)
        if not df.empty:
            ts = pd.to_datetime(df.get("generated_at"), errors="coerce", utc=True)
            if after:
                df = df[ts >= pd.to_datetime(after, utc=True)]
            if before:
                df = df[ts <= pd.to_datetime(before, utc=True)]
            if search:
                s = search.lower()
                mask = (
                    df["return_id"].astype(str).str.lower().str.contains(s, na=False)
                    | df["customer_id"].astype(str).str.lower().str.contains(s, na=False)
                    | df["decision"].astype(str).str.lower().str.contains(s, na=False)
                )
                df = df[mask]
            return df.sort_values("generated_at", ascending=True).reset_index(drop=True)
    data, _ = list_events(limit=100_000, offset=0, search=search, before=before, after=after)
    return pd.DataFrame(data)
