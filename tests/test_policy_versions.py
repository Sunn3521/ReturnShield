"""Tests for versioned, no-deploy policy control.

The property that matters: activating a version must change what the *next*
request scores against, without a restart, and the previous operating point
must stay reactatable. Every test writes to a temporary policy file - the
repository's ``models/policy.json`` is never touched.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import main as api_module
from api import api_keys
from api import decision_store as decision_store_module
from api import webhooks

FAKE_POLICY = {
    "verify_threshold": 0.12,
    "review_threshold": 0.68,
    "costs": {"false_positive": 250.0, "false_negative": 2000.0, "verification": 40.0, "manual_review": 60.0},
    "expected_cost": 82430.0,
    "auto_approve_rate": 0.94,
}


@pytest.fixture()
def calls():
    return []


@pytest.fixture()
def live_policy(tmp_path, monkeypatch, calls):
    """A stable policy dict served in-process and written to a temp file.

    ``get_bundle`` normally caches one dict, so mutating what it returns is what
    actually moves the live thresholds; the fake keeps that identity so the test
    sees the same effect a running server would.
    """
    policy_path = tmp_path / "policy.json"
    monkeypatch.setenv("RETURNSHIELD_POLICY_PATH", str(policy_path))
    webhooks.clear_endpoints()
    decision_store_module._store = decision_store_module.DecisionStore(tmp_path / "deliveries.db")

    def recorder(url, body, headers):
        calls.append((url, body, headers))
        return 200

    monkeypatch.setattr(webhooks, "_transport", recorder)
    policy = json.loads(json.dumps(FAKE_POLICY))
    monkeypatch.setattr(api_module, "get_bundle", lambda: ({"kind": "logistic"}, policy))
    monkeypatch.setattr(api_module, "_evaluation_set", lambda: None)
    # No context manager: the app lifespan starts the built-in demo generator,
    # which these contract tests have no reason to run.
    client = TestClient(api_module.app)
    yield client, policy, policy_path, calls
    api_module._rate_limiter.reset()
    api_module._rate_limiter.per_minute = 0
    webhooks.clear_endpoints()
    decision_store_module._store = None


def test_version_log_seeds_from_the_shipped_thresholds(live_policy):
    client, policy, policy_path, _ = live_policy
    body = client.get("/api/v1/policy/versions").json()

    assert body["active_version"] == 1
    assert len(body["versions"]) == 1
    seeded = body["versions"][0]
    assert seeded["verify_threshold"] == pytest.approx(FAKE_POLICY["verify_threshold"])
    assert seeded["review_threshold"] == pytest.approx(FAKE_POLICY["review_threshold"])
    assert seeded["actor"] == "shipped"


def test_creating_a_version_does_not_change_live_thresholds(live_policy):
    client, policy, _, _ = live_policy
    response = client.post(
        "/api/v1/policy/versions",
        json={"verify_threshold": 0.02, "review_threshold": 0.69, "actor": "fraud.lead", "note": "tighten"},
    )

    assert response.status_code == 201
    created = response.json()["version"]
    assert created["version"] == 2
    assert created["state"] == "draft"
    assert created["actor"] == "fraud.lead"
    # Drafting is deliberately inert.
    assert policy["verify_threshold"] == pytest.approx(FAKE_POLICY["verify_threshold"])
    assert client.get("/api/v1/policy").json()["verify_threshold"] == pytest.approx(0.12)


def test_activate_swaps_the_live_policy_without_a_restart(live_policy):
    client, policy, policy_path, calls = live_policy
    webhooks.register_endpoint("https://example.test/hook", "top-secret", "production")
    client.post("/api/v1/policy/versions", json={"verify_threshold": 0.02, "review_threshold": 0.69})

    body = client.post("/api/v1/policy/activate", json={"version": 2, "actor": "fraud.lead"}).json()

    assert body["already_active"] is False
    assert body["policy"]["verify_threshold"] == pytest.approx(0.02)
    assert body["policy"]["active_version"] == 2
    # The next score uses it, not the one that was live when the process started.
    assert policy["verify_threshold"] == pytest.approx(0.02)
    assert client.get("/api/v1/policy").json()["verify_threshold"] == pytest.approx(0.02)
    assert json.loads(policy_path.read_text())["active_version"] == 2
    # A merchant subscribed to decisions also hears about the policy move.
    assert len(calls) == 1
    _url, body, headers = calls[0]
    assert headers[webhooks.EVENT_HEADER] == "policy.activated"
    assert json.loads(body)["data"]["version"] == 2
    assert webhooks.verify("top-secret", body, headers[webhooks.SIGNATURE_HEADER], headers[webhooks.TIMESTAMP_HEADER])


def test_activation_records_who_and_when(live_policy):
    client, _, policy_path, _ = live_policy
    client.post("/api/v1/policy/versions", json={"verify_threshold": 0.02, "review_threshold": 0.69})
    client.post("/api/v1/policy/activate", json={"version": 2, "actor": "analyst@merchant", "note": "attack wave"})

    versions = client.get("/api/v1/policy/versions").json()["versions"]
    by_version = {v["version"]: v for v in versions}
    assert by_version[2]["state"] == "active"
    assert by_version[2]["actor"] == "analyst@merchant"
    assert by_version[2]["note"] == "attack wave"
    assert by_version[2]["activated_at"] is not None
    assert by_version[1]["state"] == "superseded"
    assert json.loads(policy_path.read_text())["versions"][1]["actor"] == "analyst@merchant"


def test_activating_the_already_active_version_is_a_no_op(live_policy):
    client, policy, _, calls = live_policy
    body = client.post("/api/v1/policy/activate", json={"version": 1}).json()

    assert body["already_active"] is True
    assert body["webhooks"] == []
    assert calls == []
    assert policy["verify_threshold"] == pytest.approx(FAKE_POLICY["verify_threshold"])


def test_rollback_restores_the_previous_thresholds_exactly(live_policy):
    client, policy, _, _ = live_policy
    original = (policy["verify_threshold"], policy["review_threshold"])
    client.post("/api/v1/policy/versions", json={"verify_threshold": 0.5, "review_threshold": 0.9})
    client.post("/api/v1/policy/activate", json={"version": 2})

    body = client.post("/api/v1/policy/rollback", json={"actor": "oncall"}).json()

    assert body["rolled_back_from"] == 2
    assert body["version"]["version"] == 1
    assert (policy["verify_threshold"], policy["review_threshold"]) == original
    assert policy["active_version"] == 1


def test_rollback_without_history_is_a_409_not_a_silent_no_op(live_policy):
    client, policy, _, _ = live_policy
    response = client.post("/api/v1/policy/rollback", json={})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"
    assert policy["verify_threshold"] == pytest.approx(FAKE_POLICY["verify_threshold"])


def test_unknown_version_is_a_404(live_policy):
    client, _, _, _ = live_policy
    response = client.post("/api/v1/policy/activate", json={"version": 99})

    assert response.status_code == 404
    assert "99" in response.json()["error"]["message"]


def test_inverted_thresholds_are_rejected(live_policy):
    client, _, _, _ = live_policy
    response = client.post("/api/v1/policy/versions", json={"verify_threshold": 0.8, "review_threshold": 0.2})

    assert response.status_code == 400
    assert "below" in response.json()["error"]["message"]


def test_unknown_cost_key_is_rejected_rather_than_ignored(live_policy):
    client, _, _, _ = live_policy
    response = client.post("/api/v1/policy/versions", json={"verify_threshold": 0.02, "review_threshold": 0.69, "costs": {"vibes": 5}})

    assert response.status_code == 400
    assert "vibes" in response.json()["error"]["message"]


def test_cost_overrides_are_priced_into_the_active_version(live_policy):
    client, policy, _, _ = live_policy
    client.post(
        "/api/v1/policy/versions",
        json={"verify_threshold": 0.02, "review_threshold": 0.69, "costs": {"false_negative": 5000}},
    )
    client.post("/api/v1/policy/activate", json={"version": 2})

    assert policy["costs"]["false_negative"] == pytest.approx(5000.0)
    # Unspecified costs keep their shipped values rather than resetting.
    assert policy["costs"]["false_positive"] == pytest.approx(250.0)


def test_preview_reports_both_sides_without_writing(live_policy, monkeypatch):
    client, policy, policy_path, calls = live_policy
    frame = pd.DataFrame({"risk_probability": [0.05] * 200, "abusive_return": [0] * 200})
    monkeypatch.setattr(api_module, "_evaluation_set", lambda: (frame, frame["risk_probability"].to_numpy()))
    before = policy_path.read_text() if policy_path.exists() else ""

    body = client.post("/api/v1/policy/preview", json={"verify_threshold": 0.01, "review_threshold": 0.5}).json()

    assert body["evaluation_available"] is True
    assert body["candidate"]["verify_threshold"] == pytest.approx(0.01)
    # Every row is now warned at, which costs verification money per call.
    assert body["candidate_evaluation"]["verification_rate"] == pytest.approx(1.0)
    assert body["delta"]["review_volume"] == 0
    assert policy["verify_threshold"] == pytest.approx(FAKE_POLICY["verify_threshold"])
    assert (policy_path.read_text() if policy_path.exists() else "") == before
    assert calls == []


def test_preview_says_so_when_the_holdout_is_missing(live_policy):
    client, _, _, _ = live_policy
    body = client.post("/api/v1/policy/preview", json={"verify_threshold": 0.02, "review_threshold": 0.69}).json()

    assert body["evaluation_available"] is False
    assert body["candidate_evaluation"] is None
    assert body["candidate"]["review_threshold"] == pytest.approx(0.69)


def test_meta_advertises_policy_control(live_policy):
    client, _, _, _ = live_policy
    # Reading the log seeds it from the shipped thresholds; meta then reports it.
    client.get("/api/v1/policy/versions")
    body = client.get("/api/v1/meta").json()

    assert "/api/v1/policy/activate" in body["endpoints"]
    assert body["decision_api"]["policy_control"]["active_version"] == 1


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/api/v1/policy/versions"),
        ("POST", "/api/v1/policy/activate"),
        ("POST", "/api/v1/policy/rollback"),
        ("POST", "/api/v1/policy/preview"),
    ],
)
def test_policy_mutations_require_admin_scope(monkeypatch, method, path):
    """A checkout service must not be able to move the operating point."""
    monkeypatch.setenv("RETURNSHIELD_API_KEYS", json.dumps({"scoped-key": ["score:write", "reviews:read"]}))
    try:
        response = TestClient(api_module.app).request(
            method, path, json={}, headers={"X-ReturnShield-Key": "scoped-key"}
        )
        assert response.status_code == 403
        assert "admin" in response.json()["error"]["message"]
    finally:
        monkeypatch.delenv("RETURNSHIELD_API_KEYS", raising=False)


def test_admin_scope_reaches_the_policy_endpoints(monkeypatch, tmp_path):
    monkeypatch.setenv("RETURNSHIELD_API_KEYS", json.dumps({"scoped-key": ["admin"]}))
    monkeypatch.setenv("RETURNSHIELD_POLICY_PATH", str(tmp_path / "policy.json"))
    policy = json.loads(json.dumps(FAKE_POLICY))
    monkeypatch.setattr(api_module, "get_bundle", lambda: ({"kind": "logistic"}, policy))
    monkeypatch.setattr(api_module, "_evaluation_set", lambda: None)
    try:
        client = TestClient(api_module.app)
        auth = {"X-ReturnShield-Key": "scoped-key"}
        assert client.get("/api/v1/policy/versions", headers=auth).status_code == 200
        created = client.post(
            "/api/v1/policy/versions",
            json={"verify_threshold": 0.02, "review_threshold": 0.69},
            headers=auth,
        )
        assert created.status_code == 201
    finally:
        monkeypatch.delenv("RETURNSHIELD_API_KEYS", raising=False)
        monkeypatch.delenv("RETURNSHIELD_POLICY_PATH", raising=False)


def test_required_scope_map_covers_every_policy_write():
    assert api_keys.required_scope("POST", "/api/v1/policy/activate") == "admin"
    assert api_keys.required_scope("POST", "/api/v1/policy/versions") == "admin"
    assert api_keys.required_scope("POST", "/api/v1/policy/rollback") == "admin"
    assert api_keys.required_scope("POST", "/api/v1/policy/preview") == "admin"
    # Reading the log is metadata, not a change; auth alone is enough.
    assert api_keys.required_scope("GET", "/api/v1/policy/versions") is None
    assert api_keys.required_scope("GET", "/api/v1/policy/live") is None
    # The hot-edit path is still a write, so it stays behind admin.
    assert api_keys.required_scope("POST", "/api/v1/policy/live/set") == "admin"
    assert api_keys.required_scope("POST", "/api/v1/policy/live/revert") == "admin"


def test_hot_set_changes_the_next_score_without_a_deploy(live_policy, monkeypatch):
    """The property the live tier exists for: a set lands on the next score."""
    client, policy, _, _ = live_policy
    # The fixture's bundle is a stand-in with no real weights; this test
    # verifies policy *routing*, so pin the score and stub the SHAP explainer.
    # 0.15 sits BETWEEN the two operating points (cold T1=0.12, hot T1=0.25),
    # so the decision itself proves which operating point governed the call.
    monkeypatch.setattr(api_module, "predict_bundle", lambda bundle, frame: [0.15])
    monkeypatch.setattr(api_module, "top_features", lambda bundle, frame, top_n=5: [])
    # Fresh process starts from the cold-start file snapshot.
    status = client.get("/api/v1/policy/live").json()
    assert status["source"].endswith("(cold start)")
    assert status["verify_threshold"] == pytest.approx(FAKE_POLICY["verify_threshold"])
    # A hot set does not touch the versioned log.
    set_body = client.post(
        "/api/v1/policy/live/set",
        json={"verify_threshold": 0.25, "review_threshold": 0.8, "actor": "oncall", "note": "live test"},
    ).json()
    assert set_body["after"]["verify_threshold"] == pytest.approx(0.25)
    assert set_body["after"]["source"].startswith("live_set")
    assert set_body["before"]["verify_threshold"] == pytest.approx(FAKE_POLICY["verify_threshold"])
    # The versioned log is untouched: no new version, no webhook.
    versions = client.get("/api/v1/policy/versions").json()["versions"]
    assert len(versions) == 1
    assert versions[0]["state"] == "active"
    assert versions[0]["verify_threshold"] == pytest.approx(FAKE_POLICY["verify_threshold"])
    # The next score uses the hot point: 0.15 < hot T1=0.25 -> AUTO_APPROVE
    # (at the cold T1=0.12 the same return would be VERIFY).
    scored = client.post(
        "/api/v1/score",
        json={"return_id": "R-LIVE-1", "order_id": "O1", "customer_id": "C1",
              "product_category": "electronics", "payment_method": "card",
              "return_reason": "damaged", "order_value": 1000.0, "product_price": 1000.0,
              "discount_pct": 0.0, "customer_account_age_days": 365, "orders_7d": 1,
              "orders_30d": 1, "orders_90d": 1, "refund_amount_30d": 0.0,
              "refund_amount_90d": 0.0, "hours_to_return": 24.0, "same_product_returns_90d": 0,
              "same_category_returns_90d": 0, "velocity_24h": 1, "velocity_7d": 1,
              "device_linked_accounts": 1, "address_linked_accounts": 1,
              "device_return_rate_90d": 0.0, "address_return_rate_90d": 0.0},
    ).json()
    assert scored["risk_display"] == "15.0%"
    assert scored["decision"] == "AUTO_APPROVE"
    # Reverting drops the hot set and re-syncs from the versioned log: the
    # same 0.15 return now crosses the cold T1=0.12 and is VERIFY again.
    reverted = client.post("/api/v1/policy/live/revert", json={"actor": "oncall"}).json()
    assert reverted["verify_threshold"] == pytest.approx(FAKE_POLICY["verify_threshold"])
    assert reverted["source"].endswith("(cold start)")
    scored_after = client.post(
        "/api/v1/score",
        json={"return_id": "R-LIVE-2", "order_id": "O2", "customer_id": "C2",
              "product_category": "electronics", "payment_method": "card",
              "return_reason": "damaged", "order_value": 1000.0, "product_price": 1000.0,
              "discount_pct": 0.0, "customer_account_age_days": 365, "orders_7d": 1,
              "orders_30d": 1, "orders_90d": 1, "refund_amount_30d": 0.0,
              "refund_amount_90d": 0.0, "hours_to_return": 24.0, "same_product_returns_90d": 0,
              "same_category_returns_90d": 0, "velocity_24h": 1, "velocity_7d": 1,
              "device_linked_accounts": 1, "address_linked_accounts": 1,
              "device_return_rate_90d": 0.0, "address_return_rate_90d": 0.0},
    ).json()
    assert scored_after["risk_display"] == "15.0%"
    assert scored_after["decision"] == "VERIFY"
