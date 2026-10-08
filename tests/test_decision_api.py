"""Tests for the decision-API increment.

Covers the two properties that make the write path trustworthy - event
idempotency and webhook signature - plus scoped keys, sandbox isolation, the
review queue, backfill, and the feedback loop that feeds hindsight.

The model is replaced with a stub so these run without loading the bundle; the
only thing under test is the contract, not the classifier.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from api import main as api_module
from api import api_keys
from api import decision_store as decision_store_module
from api import memory as memory_module
from api import webhooks

FAKE_POLICY = {"verify_threshold": 0.12, "review_threshold": 0.68, "costs": {"false_positive": 250.0, "false_negative": 2000.0}}


def _install_fake_model(monkeypatch, probability: float = 0.9):
    """Replace the scorer so the contract can be tested without the bundle."""
    monkeypatch.setattr(api_module, "get_bundle", lambda: ({"kind": "logistic"}, dict(FAKE_POLICY)))
    monkeypatch.setattr(api_module, "predict_bundle", lambda bundle, frame: [probability] * len(frame))
    monkeypatch.setattr(
        api_module,
        "top_features",
        lambda bundle, frame, top_n=5: [
            {"feature": "num__return_rate_90d", "contribution": 0.61},
            {"feature": "num__returns_30d", "contribution": -0.12},
        ],
    )
    monkeypatch.setattr(api_module, "concise_reasoning", lambda row, items: ["High historical return rate"])


@pytest.fixture()
def calls():
    """Recorded outbound webhook attempts, shared with the client fixture."""
    return []


@pytest.fixture()
def client(tmp_path, monkeypatch, calls):
    memory_module._store = memory_module.MemoryStore(tmp_path / "decision-memory.db")
    decision_store_module._store = decision_store_module.DecisionStore(tmp_path / "decision-api.db")
    memory_module.clear_ground_truth_cache()
    webhooks.clear_endpoints()

    def recorder(url, body, headers):
        calls.append((url, body, headers))
        return 200

    monkeypatch.setattr(webhooks, "_transport", recorder)
    test_client = TestClient(api_module.app)
    yield test_client
    api_module._rate_limiter.reset()
    api_module._rate_limiter.per_minute = 0
    webhooks.clear_endpoints()
    memory_module._store = None
    decision_store_module._store = None


def _score_payload(return_id: str = "R900001") -> dict:
    return {
        "return_id": return_id,
        "order_id": "O900001",
        "customer_id": "C00123",
        "order_value": 12500.0,
        "product_price": 12500.0,
    }


def _event(event_id: str, **extra) -> dict:
    return {"event_id": event_id, "kind": "return", "return_id": "R900001", **extra}


# ------------------------------------------------------------------ scope routing


def test_required_scope_uses_the_longest_matching_prefix():
    assert api_keys.required_scope("POST", "/api/v1/events/backfill") == "events:write"
    assert api_keys.required_scope("POST", "/api/v1/events") == "events:write"
    assert api_keys.required_scope("GET", "/api/v1/events") == "events:read"
    assert api_keys.required_scope("POST", "/api/v1/decisions/score") == "score:write"
    assert api_keys.required_scope("POST", "/api/v1/decisions") == "decisions:write"
    assert api_keys.required_scope("GET", "/api/v1/reviews") == "reviews:read"
    assert api_keys.required_scope("POST", "/api/v1/reviews/7/resolve") == "reviews:write"
    assert api_keys.required_scope("POST", "/api/v1/sandbox/test-webhook") == "admin"
    # Unmapped endpoints still require a key, but no particular scope.
    assert api_keys.required_scope("GET", "/api/v1/policy") is None
    assert api_keys.required_scope("GET", "/api/v1/reviews/") == "reviews:read"


def test_admin_scope_implies_every_scope():
    record = api_keys.KeyRecord("s", "k", frozenset({"admin"}))
    assert record.permits("events:write") and record.permits("reviews:write")
    assert api_keys.KeyRecord("s", "k", frozenset({"reviews:read"})).permits("reviews:write") is False


def test_scoped_keys_parse_from_env_and_resolve(monkeypatch):
    monkeypatch.setenv("RETURNSHIELD_API_KEYS", json.dumps({
        "prod-key": {"scopes": ["score:write"], "environment": "production", "id": "checkout"},
        "sand-key": ["admin"],
    }))
    records = api_keys.load_keys()
    assert {r.key_id for r in records} == {"checkout", "key-1"}
    assert api_keys.resolve("prod-key").environment == "production"
    assert api_keys.resolve("sand-key").permits("events:write")
    assert api_keys.resolve("wrong") is None


def test_legacy_single_key_keeps_working_and_gets_all_scopes(monkeypatch):
    monkeypatch.delenv("RETURNSHIELD_API_KEYS", raising=False)
    monkeypatch.setenv("RETURNSHIELD_API_KEY", "old-key")
    record = api_keys.resolve("old-key")
    assert record.scopes == api_keys.ALL_SCOPES
    assert record.environment == "production"
    assert record.permits("admin")


def test_malformed_scoped_key_json_falls_back_to_the_legacy_key(monkeypatch):
    monkeypatch.setenv("RETURNSHIELD_API_KEYS", "{not json")
    monkeypatch.setenv("RETURNSHIELD_API_KEY", "old-key")
    assert api_keys.resolve("old-key") is not None
    monkeypatch.setenv("RETURNSHIELD_ENVIRONMENT", "sandbox")
    assert api_keys.resolve("old-key").environment == "sandbox"


def test_scoped_key_denies_a_missing_scope(client, monkeypatch):
    monkeypatch.setenv("RETURNSHIELD_API_KEYS", json.dumps({"ro": ["reviews:read"]}))
    read_ok = client.get("/api/v1/reviews", headers={"X-ReturnShield-Key": "ro"})
    assert read_ok.status_code == 200
    denied = client.post("/api/v1/events", json={"events": [_event("e1")]}, headers={"X-ReturnShield-Key": "ro"})
    assert denied.status_code == 403
    body = denied.json()["error"]
    assert body["code"] == "insufficient_scope"
    assert body["detail"]["required_scope"] == "events:write"
    assert body["detail"]["granted_scopes"] == ["reviews:read"]
    # Unmapped endpoints need a key but no scope; liveness stays public.
    assert client.get("/api/v1/policy", headers={"X-ReturnShield-Key": "ro"}).status_code == 200
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/policy").status_code == 401


def test_sandbox_key_environment_isolates_ingested_events(client, monkeypatch):
    monkeypatch.setenv("RETURNSHIELD_API_KEYS", json.dumps({
        "prod": {"scopes": ["events:write", "events:read"]},
        "sand": {"scopes": ["events:write", "events:read"], "environment": "sandbox"},
    }))
    prod_headers = {"X-ReturnShield-Key": "prod"}
    sand_headers = {"X-ReturnShield-Key": "sand"}
    client.post("/api/v1/events", json={"events": [_event("prod-1")]}, headers=prod_headers)
    client.post("/api/v1/events", json={"events": [_event("sand-1")]}, headers=sand_headers)
    # A header cannot upgrade a production key, and a sandbox key cannot leak up.
    sand_headers_with_header = {**sand_headers, "X-ReturnShield-Environment": "production"}
    client.post("/api/v1/events", json={"events": [_event("sand-2")]}, headers=sand_headers_with_header)

    prod = client.get("/api/v1/events", headers=prod_headers).json()
    sand = client.get("/api/v1/events", headers=sand_headers).json()
    assert prod["total"] == 1 and prod["environment"] == "production"
    assert sand["total"] == 2 and sand["environment"] == "sandbox"
    assert {e["event_id"] for e in prod["data"]} == {"prod-1"}


# ------------------------------------------------------------------ idempotency


def test_event_ingest_is_idempotent(client):
    batch = {"events": [_event("evt-1"), _event("evt-2")]}
    first = client.post("/api/v1/events", json=batch)
    assert first.status_code == 202
    body = first.json()
    assert (body["accepted"], body["duplicates"], body["rejected"]) == (2, 0, 0)

    second = client.post("/api/v1/events", json=batch).json()
    assert (second["accepted"], second["duplicates"]) == (0, 2)
    assert client.get("/api/v1/events").json()["total"] == 2


def test_event_requires_an_event_id(client):
    res = client.post("/api/v1/events", json={"events": [{"kind": "return"}]})
    assert res.status_code == 422
    assert client.get("/api/v1/events").json()["total"] == 0


def test_backfill_endpoint_flags_its_rows(client):
    res = client.post("/api/v1/events/backfill", json={"events": [_event("hist-1")]})
    assert res.status_code == 202
    assert res.json()["backfill"] is True and res.json()["accepted"] == 1
    listed = client.get("/api/v1/events").json()["data"]
    assert listed[0]["backfill"] == 1


def test_ingested_event_keeps_its_extra_payload(client):
    client.post("/api/v1/events", json={"events": [_event("evt-extra", device_id="D1", refund_amount=99.5)]})
    row = client.get("/api/v1/events").json()["data"][0]
    assert row["payload"]["device_id"] == "D1"
    assert row["payload"]["refund_amount"] == 99.5


# --------------------------------------------------------------------- signing


def test_webhook_signature_roundtrip_and_tamper_detection():
    body = b'{"type":"ping"}'
    timestamp = 1_700_000_000
    signature = webhooks.sign("secret", body, timestamp)
    assert signature.startswith("sha256=")
    now = timestamp + 5
    assert webhooks.verify("secret", body, signature, timestamp, now=now) is True
    # Wrong secret, altered body, stale timestamp, and junk all fail closed.
    assert webhooks.verify("other", body, signature, timestamp, now=now) is False
    assert webhooks.verify("secret", b'{"type":"pong"}', signature, timestamp, now=now) is False
    assert webhooks.verify("secret", body, signature, timestamp, now=timestamp + 10_000) is False
    assert webhooks.verify("secret", body, None, timestamp, now=now) is False
    assert webhooks.verify("secret", body, signature, "not-a-number", now=now) is False


def test_delivery_signature_is_verifiable_by_the_receiver(client, calls):
    webhooks.register_endpoint("https://example.test/hook", "top-secret", "production")
    webhooks.deliver("decision.recorded", {"return_id": "R1"})
    assert len(calls) == 1
    url, body, headers = calls[0]
    assert url == "https://example.test/hook"
    assert headers[webhooks.EVENT_HEADER] == "decision.recorded"
    assert webhooks.verify("top-secret", body, headers[webhooks.SIGNATURE_HEADER], headers[webhooks.TIMESTAMP_HEADER]) is True


def test_delivery_retries_with_backoff_then_succeeds(client):
    webhooks.register_endpoint("https://example.test/flaky", "s", "production")
    attempts = {"n": 0}

    def flaky(url, body, headers):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TimeoutError("receiver slow")
        return 204

    results = webhooks.deliver("ping", {"x": 1}, transport=flaky, sleep=lambda _s: None)
    assert len(results) == 1
    assert results[0]["status"] == "delivered"
    assert results[0]["attempts"] == 3
    assert results[0]["last_error"] is None
    logged = decision_store_module.get_decision_store().list_deliveries()
    assert logged[0]["status"] == "delivered" and logged[0]["attempts"] == 3


def test_exhausted_delivery_is_logged_not_lost(client):
    webhooks.register_endpoint("https://example.test/down", "s", "production")

    def down(_url, _body, _headers):
        return 500

    results = webhooks.deliver("ping", {"x": 1}, transport=down, sleep=lambda _s: None)
    assert results[0]["status"] == "failed"
    assert results[0]["attempts"] == webhooks.MAX_ATTEMPTS
    assert results[0]["last_error"] == "HTTP 500"
    logged = decision_store_module.get_decision_store().list_deliveries()
    assert logged[0]["status"] == "failed" and logged[0]["last_error"] == "HTTP 500"


def test_no_configured_endpoint_means_no_network_call(client, calls):
    assert webhooks.deliver("ping", {}) == []
    assert calls == []


def test_sandbox_endpoints_are_separate_from_production(client, calls):
    webhooks.register_endpoint("https://example.test/sandbox", "s", "sandbox")
    assert webhooks.deliver("ping", {}, environment="production") == []
    assert len(webhooks.deliver("ping", {}, environment="sandbox")) == 1
    assert calls[0][0] == "https://example.test/sandbox"


# ------------------------------------------------------------------- score call


def test_decision_score_returns_a_full_decision_object_with_reasons(client, monkeypatch):
    _install_fake_model(monkeypatch, probability=0.9)
    res = client.post("/api/v1/decisions/score", json=_score_payload())
    assert res.status_code == 200
    body = res.json()
    assert body["action"] == "block"
    assert body["decision"] == "MANUAL_REVIEW"
    assert body["score"] == 0.9
    assert body["reasons"][0]["feature"] == "return_rate_90d"
    assert body["reasons"][0]["direction"] == "increases_risk"
    assert body["reasons"][1]["direction"] == "decreases_risk"
    assert body["policy_version"].startswith("T1=0.120;T2=0.680")
    assert body["model_version"]
    assert body["request_id"]
    assert body["review_id"] is not None and body["review_created"] is True
    assert body["environment"] == "production"


def test_score_actions_map_across_the_four_way_space(client, monkeypatch):
    for probability, expected in ((0.05, "allow"), (0.3, "warn"), (0.7, "review"), (0.95, "block")):
        _install_fake_model(monkeypatch, probability=probability)
        body = client.post("/api/v1/decisions/score", json=_score_payload(f"R-{expected}")).json()
        assert body["action"] == expected, probability


def test_review_is_queued_once_per_return(client, monkeypatch):
    _install_fake_model(monkeypatch, probability=0.9)
    first = client.post("/api/v1/decisions/score", json=_score_payload("R-DUP")).json()
    second = client.post("/api/v1/decisions/score", json=_score_payload("R-DUP")).json()
    assert first["review_created"] is True
    assert second["review_created"] is False
    assert first["review_id"] == second["review_id"]
    assert client.get("/api/v1/reviews").json()["total"] == 1


def test_low_risk_score_queues_nothing(client, monkeypatch):
    _install_fake_model(monkeypatch, probability=0.05)
    body = client.post("/api/v1/decisions/score", json=_score_payload("R-CLEAN")).json()
    assert body["action"] == "allow" and body["review_id"] is None
    assert client.get("/api/v1/reviews").json()["total"] == 0


def test_sandbox_score_never_touches_production_state(client, monkeypatch, calls):
    _install_fake_model(monkeypatch, probability=0.9)
    sandbox = client.post(
        "/api/v1/decisions/score",
        json=_score_payload("R-SAND"),
        headers={"X-ReturnShield-Environment": "sandbox"},
    ).json()
    assert sandbox["environment"] == "sandbox"
    assert client.get("/api/v1/reviews").json()["total"] == 0
    assert client.get("/api/v1/reviews", headers={"X-ReturnShield-Environment": "sandbox"}).json()["total"] == 1
    # Production hindsight must not have learned anything from sandbox traffic.
    assert memory_module.get_store().hindsight_summary()["total_decisions"] == 0


# ------------------------------------------------------------- review queue


def test_review_queue_ranks_by_expected_cost_and_resolves_once(client, monkeypatch):
    _install_fake_model(monkeypatch, probability=0.9)
    client.post("/api/v1/decisions/score", json=_score_payload("R-SMALL"))
    big = dict(_score_payload("R-BIG"))
    big["order_value"] = 90000.0
    big["product_price"] = 90000.0
    client.post("/api/v1/decisions/score", json=big)

    queue = client.get("/api/v1/reviews").json()
    assert [r["return_id"] for r in queue["data"]] == ["R-BIG", "R-SMALL"]
    review_id = queue["data"][0]["id"]

    resolved = client.post(
        f"/api/v1/reviews/{review_id}/resolve",
        json={"resolution": "confirmed_abuse", "actor": "analyst@example.com"},
    )
    assert resolved.status_code == 200
    assert resolved.json()["review"]["state"] == "resolved"
    assert resolved.json()["review"]["actor"] == "analyst@example.com"
    assert client.get("/api/v1/reviews").json()["total"] == 1
    assert client.get("/api/v1/reviews", params={"state": "resolved"}).json()["total"] == 1
    # Resolving twice is a 404, not a silent double-resolve.
    again = client.post(f"/api/v1/reviews/{review_id}/resolve", json={"resolution": "benign"})
    assert again.status_code == 404


def test_review_resolution_emits_a_signed_webhook(client, monkeypatch, calls):
    webhooks.register_endpoint("https://example.test/hook", "sec", "production")
    _install_fake_model(monkeypatch, probability=0.9)
    review_id = client.post("/api/v1/decisions/score", json=_score_payload("R-HOOK")).json()["review_id"]
    calls.clear()
    client.post(f"/api/v1/reviews/{review_id}/resolve", json={"resolution": "benign"})
    types = [h[webhooks.EVENT_HEADER] for _u, _b, h in calls]
    assert "review.resolved" in types


# ------------------------------------------------------------------ feedback


def test_feedback_endpoint_resolves_the_open_call_for_hindsight(client, monkeypatch):
    _install_fake_model(monkeypatch, probability=0.9)
    client.post("/api/v1/decisions/score", json=_score_payload("R-FEED"))
    assert memory_module.get_store().hindsight_summary()["pending"] == 1

    res = client.post("/api/v1/decisions", json={
        "return_id": "R-FEED",
        "outcome": "abusive",
        "actor": "analyst@example.com",
        "merchant_loss": 8000.0,
    })
    assert res.status_code == 200
    body = res.json()
    assert body["actual_abusive"] == 1
    assert body["resolved_updated"] == 1 and body["resolved_inserted"] == 0
    assert body["source"] == "feedback:analyst@example.com"

    # No offline resolve job ran: the recommendation engine already sees it.
    rows = memory_module.get_store().resolved_decision_rows()
    assert len(rows) == 1 and rows[0]["actual_abusive"] == 1
    summary = memory_module.get_store().hindsight_summary()
    assert summary["resolved"] == 1 and summary["pending"] == 0


def test_outcome_only_feedback_is_counted_but_cannot_repricing(client):
    res = client.post("/api/v1/decisions", json={"return_id": "R-NEW", "outcome": "benign"})
    assert res.json()["resolved_inserted"] == 1
    summary = memory_module.get_store().hindsight_summary()
    assert summary["resolved"] == 1 and summary["pending"] == 0
    # Nothing was ever scored, so there is no probability to re-price a threshold
    # against: the row is counted by the summary but stays out of the sample.
    assert memory_module.get_store().resolved_decision_rows() == []


def test_feedback_carrying_a_score_joins_the_recommendation_sample(client):
    client.post("/api/v1/decisions", json={
        "return_id": "R-SCORED", "outcome": "abusive", "risk_probability": 0.81,
    })
    rows = memory_module.get_store().resolved_decision_rows()
    assert len(rows) == 1
    assert rows[0]["actual_abusive"] == 1 and rows[0]["risk_probability"] == 0.81


def test_unknown_outcome_is_rejected(client):
    res = client.post("/api/v1/decisions", json={"return_id": "R1", "outcome": "maybe"})
    assert res.status_code == 400
    assert "Unknown outcome" in res.json()["detail"]


def test_chargeback_counts_as_abusive(client):
    res = client.post("/api/v1/decisions", json={"return_id": "R-CB", "outcome": "CHARGEBACK"})
    assert res.json()["actual_abusive"] == 1


# ------------------------------------------------------------------ delivery log


def test_deliveries_endpoint_lists_attempts(client):
    webhooks.register_endpoint("https://example.test/hook", "s", "production")
    webhooks.deliver("ping", {"n": 1})
    log = client.get("/api/v1/webhooks/deliveries").json()
    assert log["total"] == 1
    assert log["data"][0]["event_type"] == "ping"
    assert log["data"][0]["status"] == "delivered"
    assert log["data"][0]["signature"].startswith("sha256=")


def test_sandbox_test_webhook_reports_when_nothing_is_registered(client):
    body = client.post("/api/v1/sandbox/test-webhook").json()
    assert body["endpoints"] == 0
    assert body["deliveries"] == []
    assert body["hint"]


def test_sandbox_test_webhook_delivers_to_a_sandbox_endpoint(client):
    webhooks.register_endpoint("https://example.test/sandbox", "s", "sandbox")
    body = client.post("/api/v1/sandbox/test-webhook").json()
    assert body["endpoints"] == 1
    assert body["deliveries"][0]["status"] == "delivered"


# ------------------------------------------------------------------ discovery


def test_meta_advertises_the_decision_surface(client):
    meta = client.get("/api/v1/meta").json()
    assert "/api/v1/decisions/score" in meta["endpoints"]
    assert "abusive" in meta["decision_api"]["outcomes"]
    assert meta["access_control"]["scoped_keys"]["known_scopes"]
