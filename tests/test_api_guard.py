"""Tests for Stage 4: API key auth, rate limiting, pagination, infrastructure join.

The guard tests exercise ``api_guard`` directly (fast, no model load) plus the
real FastAPI middleware through ``TestClient`` with the env toggled.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import main as api_module
from api import api_guard
from api import memory as memory_module
from src.network import attach_infrastructure_ids


@pytest.fixture()
def client(tmp_path, monkeypatch):
    memory_module._store = memory_module.MemoryStore(tmp_path / "guard-memory.db")
    memory_module.clear_ground_truth_cache()
    test_client = TestClient(api_module.app)
    yield test_client
    api_module._rate_limiter.reset()
    api_module._rate_limiter.per_minute = 0
    memory_module._store = None


# ------------------------------------------------------------------- api key


def test_public_paths_never_require_a_key():
    for path in api_guard.PUBLIC_PATHS:
        assert api_guard.check_api_key(path, None, "secret"), path
        assert api_guard.check_api_key(path, None, None), path


def test_missing_or_wrong_key_is_rejected():
    assert api_guard.check_api_key("/api/v1/score", None, "secret") is False
    assert api_guard.check_api_key("/api/v1/score", "nope", "secret") is False
    assert api_guard.check_api_key("/api/v1/score", "secret", "secret") is True


def test_key_is_read_from_either_header():
    assert api_guard.extract_key({"X-ReturnShield-Key": " abc "}) == "abc"
    assert api_guard.extract_key({"Authorization": "Bearer xyz"}) == "xyz"
    assert api_guard.extract_key({"Authorization": "bearer xyz"}) == "xyz"
    assert api_guard.extract_key({"Authorization": "Basic xyz"}) is None
    assert api_guard.extract_key({}) is None


def test_trailing_slash_stays_public():
    assert api_guard.is_public_path("/api/v1/health/") is True
    assert api_guard.is_public_path("/api/v1/score") is False


def test_client_rejects_unauthenticated_request(monkeypatch, client):
    monkeypatch.setenv("RETURNSHIELD_API_KEY", "test-key")
    assert client.get("/api/v1/policy").status_code == 401
    assert client.get("/api/v1/health").status_code == 200


def test_client_accepts_the_right_key_in_either_header(monkeypatch, client):
    monkeypatch.setenv("RETURNSHIELD_API_KEY", "test-key")
    for headers in ({"X-ReturnShield-Key": "test-key"},
                    {"Authorization": "Bearer test-key"}):
        res = client.get("/api/v1/policy", headers=headers)
        assert res.status_code == 200, headers
        assert res.json()["verify_threshold"] > 0


def test_unset_key_means_open_access(client):
    res = client.get("/api/v1/policy")
    assert res.status_code == 200
    assert client.get("/api/v1/meta").json()["access_control"]["auth_enabled"] is False


def test_meta_advertises_the_guard_configuration(monkeypatch, client):
    monkeypatch.setenv("RETURNSHIELD_API_KEY", "k")
    monkeypatch.setenv("RETURNSHIELD_RATE_LIMIT", "12")
    control = client.get("/api/v1/meta").json()["access_control"]
    assert control["auth_enabled"] is True
    assert control["rate_limit_per_minute"] == 12
    assert "/api/v1/health" in control["public_paths"]


def test_bad_rate_limit_env_is_ignored(monkeypatch):
    monkeypatch.setenv("RETURNSHIELD_RATE_LIMIT", "not-a-number")
    assert api_guard.rate_limit_per_minute() == 0
    monkeypatch.setenv("RETURNSHIELD_RATE_LIMIT", "-5")
    assert api_guard.rate_limit_per_minute() == 0


# --------------------------------------------------------------- rate limiter


def test_limiter_disabled_by_default():
    limiter = api_guard.RateLimiter(0)
    assert limiter.enabled is False
    assert all(limiter.check("k")[0] for _ in range(50))


def test_limiter_blocks_past_the_limit_and_reports_retry_after():
    limiter = api_guard.RateLimiter(3)
    for _ in range(3):
        allowed, remaining, retry = limiter.check("k")
        assert allowed and retry == 0.0
    allowed, remaining, retry = limiter.check("k")
    assert allowed is False
    assert remaining == 0
    assert 0.0 < retry <= 60.0


def test_limiter_separates_clients_and_window_expiry():
    limiter = api_guard.RateLimiter(2)
    assert limiter.check("a", now=0.0)[0] and limiter.check("a", now=1.0)[0]
    assert limiter.check("a", now=2.0)[0] is False
    assert limiter.check("b", now=2.0)[0] is True
    # The window slides: the hits recorded at t=0 and t=1 have aged out by t=61.
    assert limiter.check("a", now=61.0)[0] is True
    assert limiter.check("a", now=61.5)[0] is True
    assert limiter.check("a", now=62.0)[0] is False


def test_limiter_evicts_rather_than_growing_without_bound():
    limiter = api_guard.RateLimiter(5, max_clients=4)
    for i in range(50):
        limiter.check(f"client-{i}")
    assert limiter.tracked_clients() <= 4


def test_client_returns_429_with_retry_after(monkeypatch, client):
    monkeypatch.setenv("RETURNSHIELD_RATE_LIMIT", "3")
    statuses = [client.get("/api/v1/policy").status_code for _ in range(5)]
    assert 429 in statuses
    res = client.get("/api/v1/policy")
    body = res.json()["error"]
    assert body["code"] == "rate_limited"
    assert body["detail"]["limit_per_minute"] == 3
    assert body["detail"]["retry_after_seconds"] > 0
    assert res.headers["X-RateLimit-Limit"] == "3"
    assert res.headers["X-RateLimit-Remaining"] == "0"
    assert int(res.headers["Retry-After"]) >= 1
    assert client.get("/api/v1/health").status_code in (200, 429)


def test_preflight_is_never_blocked(monkeypatch, client):
    monkeypatch.setenv("RETURNSHIELD_API_KEY", "test-key")
    monkeypatch.setenv("RETURNSHIELD_RATE_LIMIT", "1")
    for _ in range(3):
        res = client.options("/api/v1/score")
        assert res.status_code not in (401, 429), res.status_code


# ------------------------------------------------------------------ pagination


def test_page_envelope_reports_more_pages():
    page = api_module._page([1, 2, 3], total=10, limit=3, offset=0)
    assert page == {"data": [1, 2, 3], "returned": 3, "total": 10,
                    "limit": 3, "offset": 0, "has_more": True}
    last = api_module._page([1], total=10, limit=3, offset=9)
    assert last["has_more"] is False


def test_page_envelope_clamps_hostile_input():
    assert api_module._page([], total=0, limit=99999, offset=-5)["limit"] == 5000
    assert api_module._page([], total=0, limit=99999, offset=-5)["offset"] == 0
    assert api_module._page([], total=0, limit=0, offset=0)["limit"] == 1


def test_clusters_endpoint_pages(client):
    first = client.get("/api/v1/clusters", params={"limit": 5}).json()
    assert first["total"] >= first["count"]
    if first["total"] > 5:
        second = client.get("/api/v1/clusters", params={"limit": 5, "offset": 5}).json()
        assert second["offset"] == 5
        assert second["has_more"] is (5 + second["count"] < second["total"])


def test_returns_endpoint_rejects_out_of_range_paging(client):
    assert client.get("/api/v1/returns", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/returns", params={"limit": 99999}).status_code == 422
    assert client.get("/api/v1/returns", params={"offset": -1}).status_code == 422


# --------------------------------------- shared-infrastructure id reattachment


def test_attach_restores_ids_keyed_on_customer(monkeypatch):
    import src.network as network
    frame = pd.DataFrame({"customer_id": ["c1", "C2"], "return_id": ["R1", "R2"]})
    monkeypatch.setattr(network, "infrastructure_lookup", lambda data_dir="data/raw": pd.DataFrame({
        "customer_id": ["C1", "C2"],
        "device_id": ["D1", "D1"],
        "address_id": ["A1", "A2"],
        "payment_fingerprint": ["P1", "P2"],
    }))
    out = network.attach_infrastructure_ids(frame)
    # Lookup is uppercase, so "c1" must still match "C1".
    assert out["device_id"].tolist() == ["D1", "D1"]
    assert out["address_id"].tolist() == ["A1", "A2"]
    assert len(out) == len(frame)


def test_attach_leaves_complete_frames_untouched():
    import src.network as network
    frame = pd.DataFrame({"customer_id": ["C1"], "device_id": ["KEEP"],
                          "address_id": ["KEEP"], "payment_fingerprint": ["KEEP"]})
    assert network.attach_infrastructure_ids(frame) is frame


def test_attach_is_a_noop_without_customer_ids():
    import src.network as network
    assert network.attach_infrastructure_ids(pd.DataFrame({"a": [1]})).equals(pd.DataFrame({"a": [1]}))
    assert network.attach_infrastructure_ids(pd.DataFrame()).empty


def test_scored_exports_yield_clusters_after_the_join():
    """The bug this fixes: the default dataset could not answer cluster questions."""
    from pathlib import Path
    path = Path("reports/test_predictions.csv")
    if not path.exists():
        pytest.skip("scored export not present")
    raw = pd.read_csv(path)
    assert not {"device_id", "address_id"} & set(raw.columns), "export changed; re-check the gap"
    joined = attach_infrastructure_ids(raw, "data/raw")
    for col in ("device_id", "address_id", "payment_fingerprint"):
        assert col in joined.columns
    multi = [
        joined.groupby(col, dropna=True)["customer_id"].nunique()
        for col in ("device_id", "address_id", "payment_fingerprint")
    ]
    assert any((g >= 2).any() for g in multi), "no coordinated clusters to answer with"
