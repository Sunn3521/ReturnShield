"""Tests for the Python SDK and the unified error envelope.

Two things are under test:

1. **The SDK cannot drift from the server.** Every request the client actually
   issues is matched against the OpenAPI document, so renaming a route on
   either side fails here instead of in a merchant's integration.
2. **One error shape, no matter how the failure was raised.** ``_error`` built
   the envelope for middleware failures while a raised ``HTTPException`` returned
   FastAPI's bare ``{"detail": ...}``, so the contract depended on *how* a
   request failed. Both now land on the envelope, and ``detail`` is preserved
   for anything already reading it.
"""

from __future__ import annotations

import json
import re
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from api import main as api_module
from api import decision_store as decision_store_module
from api import memory as memory_module
from api import webhooks
from sdk.returnshield import (
    ABUSIVE_OUTCOMES,
    BENIGN_OUTCOMES,
    DecisionScore,
    ReturnShieldClient,
    ReturnShieldError,
    default_payload,
)

FAKE_POLICY = {"verify_threshold": 0.12, "review_threshold": 0.68,
               "costs": {"false_positive": 250.0, "false_negative": 2000.0}}


def _install_fake_model(monkeypatch, probability: float = 0.9):
    monkeypatch.setattr(api_module, "get_bundle", lambda: ({"kind": "logistic"}, dict(FAKE_POLICY)))
    monkeypatch.setattr(api_module, "predict_bundle", lambda bundle, frame: [probability] * len(frame))
    monkeypatch.setattr(
        api_module, "top_features",
        lambda bundle, frame, top_n=5: [{"feature": "num__return_rate_90d", "contribution": 0.61}],
    )
    monkeypatch.setattr(api_module, "concise_reasoning", lambda row, items: ["High historical return rate"])


class RecordingClient:
    """Satisfies the SDK's client protocol and records what it actually sent.

    The SDK only needs ``request(method, url, headers=..., json=...,
    params=...)`` back, so this drives the app through Starlette's sync
    ``TestClient`` rather than an ASGI transport. That distinction matters:
    ``httpx.ASGITransport`` exposes only ``handle_async_request``, so a sync
    ``httpx.Client`` cannot use it.
    """

    def __init__(self, app) -> None:
        self.client = TestClient(app)
        self.seen: list[tuple[str, str, dict]] = []

    def request(self, method, url, headers=None, json=None, params=None):
        lowered = {str(k).lower(): v for k, v in (headers or {}).items()}
        path = urlsplit(url).path
        self.seen.append((str(method).upper(), path, lowered))
        return self.client.request(method, url, headers=headers, json=json, params=params)

    def close(self) -> None:
        self.client.close()


@pytest.fixture()
def transport():
    return RecordingClient(api_module.app)


@pytest.fixture()
def rs(transport, tmp_path, monkeypatch):
    memory_module._store = memory_module.MemoryStore(tmp_path / "sdk-memory.db")
    decision_store_module._store = decision_store_module.DecisionStore(tmp_path / "sdk-api.db")
    memory_module.clear_ground_truth_cache()
    webhooks.clear_endpoints()
    _install_fake_model(monkeypatch, 0.9)

    client = ReturnShieldClient(base_url="http://testserver", client=transport)
    yield client
    client.close()
    api_module._rate_limiter.reset()
    api_module._rate_limiter.per_minute = 0
    webhooks.clear_endpoints()
    memory_module._store = None
    decision_store_module._store = None


@pytest.fixture()
def controlled_policy(rs, monkeypatch, tmp_path):
    """The SDK against a policy dict that survives a write.

    ``get_bundle`` normally returns one cached dict, so activating a version is
    visible to the next call. The shared fake returns a fresh copy each time,
    which would hide exactly the behaviour under test.
    """
    policy = json.loads(json.dumps(FAKE_POLICY))
    policy.setdefault("costs", {})["verification"] = 40.0
    monkeypatch.setenv("RETURNSHIELD_POLICY_PATH", str(tmp_path / "policy.json"))
    monkeypatch.setattr(api_module, "get_bundle", lambda: ({"kind": "logistic"}, policy))
    monkeypatch.setattr(api_module, "_evaluation_set", lambda: None)
    return rs, policy


# ------------------------------------------------------- contract (drift guard)

def _openapi_patterns():
    r"""OpenAPI templates compiled to matchers.

    Escape only the literal segments. Escaping first and then swapping the
    braces leaves the backslashes the escape added, so the opening brace stays
    prefixed and the resulting escaped bracket anchors a *literal* one that
    matches nothing - reporting a perfectly valid route as drift.
    """
    paths = api_module.app.openapi().get("paths", {})
    out = {}
    for path in paths:
        literals = re.split(r"\{[^}]+\}", path)
        rx = "^" + "[^/]+".join(re.escape(chunk) for chunk in literals) + "$"
        out[path] = re.compile(rx)
    return out


def test_every_sdk_route_exists_in_the_openapi_schema(controlled_policy, transport):
    rs, _policy = controlled_policy
    rs.health()
    rs.meta()
    rs.policy()
    rs.score(default_payload("R800001"))
    decision = rs.score_decision(default_payload("R800002"))
    rs.ingest_events([{"event_id": "sdk-1", "kind": "return", "return_id": "R800002"}])
    rs.backfill([{"event_id": "sdk-2", "kind": "return", "return_id": "R800003"}])
    rs.list_events()
    rs.record_decision("R800003", "abusive", actor="sdk")
    rs.reviews()
    if decision.review_id is not None:
        rs.resolve_review(decision.review_id, "confirmed_abuse", actor="sdk")
    rs.deliveries()
    rs.test_sandbox_webhook()
    rs.policy_versions()
    rs.preview_policy(0.02, 0.69)
    rs.create_policy_version(0.02, 0.69, actor="sdk")
    rs.activate_policy(2, actor="sdk")
    rs.rollback_policy(actor="sdk")
    rs.hindsight()
    rs.recommendations()
    rs.return_history("R800001")

    assert transport.seen, "the SDK issued no requests"
    patterns = _openapi_patterns()
    unknown = [(m, p) for m, p, _h in transport.seen
               if not any(rx.match(p) for rx in patterns.values())]
    assert unknown == [], f"SDK called routes absent from the OpenAPI schema: {unknown}"


def test_openapi_document_still_builds(rs):
    assert "paths" in api_module.app.openapi()


# ------------------------------------------------------------- happy-path flow

def test_full_decision_loop_through_the_client(rs):
    decision = rs.score_decision(default_payload("R700001"))
    assert isinstance(decision, DecisionScore)
    assert decision.action in {"review", "block"}
    assert decision.reasons, "the decision object must carry ranked reasons"
    assert decision.policy_version.startswith("T1=")
    assert decision.review_id is not None

    worklist = rs.reviews(state="pending")
    assert worklist["total"] >= 1
    assert worklist["data"][0]["expected_cost"] > 0

    resolved = rs.resolve_review(decision.review_id, "confirmed_abuse", actor="qa@example.com")
    assert resolved["review"]["resolution"] == "confirmed_abuse"
    assert resolved["review"]["actor"] == "qa@example.com"

    feedback = rs.record_decision("R700001", "chargeback", actor="qa@example.com")
    assert feedback["actual_abusive"] == 1
    assert feedback["resolved_inserted"] + feedback["resolved_updated"] == 1

    rows = memory_module.get_store().resolved_decision_rows()
    assert any(r["return_id"] == "R700001" and r["actual_abusive"] == 1 for r in rows)


def test_event_ingest_is_idempotent_through_the_client(rs):
    events = [{"event_id": "dup-1", "kind": "return", "return_id": "R600001"}]
    first = rs.ingest_events(events)
    second = rs.ingest_events(events)
    assert first["accepted"] == len(events)
    assert second["duplicates"] == len(events)
    assert rs.list_events()["total"] == len(events)


# ----------------------------------------------------------- policy control

def test_policy_control_loop_through_the_client(controlled_policy):
    rs, policy = controlled_policy

    log = rs.policy_versions()
    assert log["active_version"] == 1

    created = rs.create_policy_version(0.02, 0.69, actor="fraud.lead", note="tighten")
    assert created["version"]["version"] == 2
    assert created["version"]["state"] == "draft"
    assert policy["verify_threshold"] == pytest.approx(FAKE_POLICY["verify_threshold"])

    activated = rs.activate_policy(2, actor="fraud.lead")
    assert activated["policy"]["active_version"] == 2
    assert rs.policy()["verify_threshold"] == pytest.approx(0.02)

    rolled = rs.rollback_policy(actor="oncall")
    assert rolled["rolled_back_from"] == 2
    assert rs.policy()["verify_threshold"] == pytest.approx(FAKE_POLICY["verify_threshold"])


def test_preview_prices_without_committing(controlled_policy):
    rs, policy = controlled_policy
    body = rs.preview_policy(0.01, 0.5)

    assert body["candidate"]["verify_threshold"] == pytest.approx(0.01)
    assert body["active"]["verify_threshold"] == pytest.approx(FAKE_POLICY["verify_threshold"])
    assert policy["verify_threshold"] == pytest.approx(FAKE_POLICY["verify_threshold"])


def test_hindsight_surfaces_are_reachable_from_the_client(rs):
    decision = rs.score_decision(default_payload("R700010"))
    assert decision.review_id is not None
    rs.record_decision("R700010", "abusive", actor="sdk")

    history = rs.return_history("R700010")
    assert history["return_id"] == "R700010"

    summary = rs.hindsight()
    assert "summary" in summary

    recommendation = rs.recommendations()
    assert "recommendation" in recommendation


def test_low_risk_score_queues_no_review(rs, monkeypatch):
    _install_fake_model(monkeypatch, 0.01)
    decision = rs.score_decision(default_payload("R600009"))
    assert decision.action == "allow"
    assert decision.review_id is None


# ------------------------------------------------------------ error envelope

def test_raised_http_exception_uses_the_envelope(rs):
    with pytest.raises(ReturnShieldError) as exc:
        rs.record_decision("R1", "not-a-real-outcome")
    assert exc.value.status_code == 400
    assert exc.value.code == "bad_request"
    assert "Unknown outcome" in exc.value.message
    # The old key still exists for anything that already read it.
    assert "Unknown outcome" in str(exc.value.detail)


def test_validation_error_uses_the_envelope(rs):
    with pytest.raises(ReturnShieldError) as exc:
        rs.request("POST", "/api/v1/decisions", json={"outcome": "abusive"})
    assert exc.value.status_code == 422
    assert exc.value.code == "validation_error"
    assert exc.value.message


def test_404_uses_the_envelope(rs):
    with pytest.raises(ReturnShieldError) as exc:
        rs.resolve_review(999999, "benign")
    assert exc.value.status_code == 404
    assert exc.value.code == "not_found"


def test_missing_key_raises_unauthorized(rs, monkeypatch):
    monkeypatch.setenv("RETURNSHIELD_API_KEY", "the-right-key")
    rs.api_key = "wrong-key"
    with pytest.raises(ReturnShieldError) as exc:
        rs.reviews()
    assert exc.value.status_code == 401
    assert exc.value.code == "unauthorized"


def test_missing_scope_carries_the_scope_detail(rs, monkeypatch):
    monkeypatch.setenv("RETURNSHIELD_API_KEYS", json.dumps({"ro": ["reviews:read"]}))
    rs.api_key = "ro"
    assert rs.reviews()["returned"] >= 0
    with pytest.raises(ReturnShieldError) as exc:
        rs.ingest_events([{"event_id": "nope", "kind": "return"}])
    assert exc.value.status_code == 403
    assert exc.value.code == "insufficient_scope"
    assert exc.value.detail["required_scope"] == "events:write"


def test_legacy_bare_detail_still_parses(rs):
    """A server answering with FastAPI's old bare shape still produces a message."""

    class FakeResponse:
        status_code = 400
        text = '{"detail": "legacy"}'

        def json(self):
            return {"detail": "legacy"}

    with pytest.raises(ReturnShieldError) as exc:
        rs._raise(FakeResponse())
    assert exc.value.code == "http_error"
    assert exc.value.message == "legacy"


def test_retryable_is_true_only_for_transient_failures():
    assert ReturnShieldError(429, "rate_limited", "slow down").retryable
    assert ReturnShieldError(503, "unavailable", "nope").retryable
    assert not ReturnShieldError(400, "bad_request", "your fault").retryable
    assert not ReturnShieldError(404, "not_found", "gone").retryable


# ------------------------------------------------------------------- headers

def test_api_key_header_is_sent(transport):
    client = ReturnShieldClient(base_url="http://testserver", api_key="sk-test", client=transport)
    client.reviews()
    client.close()
    _method, _path, headers = transport.seen[-1]
    assert headers.get("x-returnshield-key") == "sk-test"


def test_environment_header_selects_sandbox_without_scoped_keys(rs):
    rs.environment = "sandbox"
    body = rs.reviews()
    assert body["environment"] == "sandbox"


def test_scoped_key_environment_beats_the_header(rs, monkeypatch):
    monkeypatch.setenv("RETURNSHIELD_API_KEYS", json.dumps({"prod": ["admin"]}))
    rs.api_key = "prod"
    rs.environment = "sandbox"  # a header must not be able to move a key's scope
    body = rs.reviews()
    assert body["environment"] == "production"


def test_no_api_key_means_open_demo_access(rs):
    assert rs.api_key is None
    assert rs.health()["status"] == "healthy"


# ------------------------------------------------------------- response shape

def test_decision_score_from_dict_ignores_unknown_fields():
    score = DecisionScore.from_dict({
        "request_id": "abc", "return_id": "R1", "environment": "production",
        "action": "review", "decision": "MANUAL_REVIEW", "score": 0.9,
        "score_display": "90.0%",
        "something_new": 1,
    })
    assert score.action == "review"
    assert score.score_display == "90.0%"
    assert score.raw["something_new"] == 1


def test_decision_score_covers_every_field_the_server_returns(rs):
    """Fields dropped by ``from_dict`` are silently invisible to callers.

    ``score_display`` was missing exactly this way: the server sent it, the
    dataclass filtered it out, and the attribute error only surfaced on use.
    """
    server_fields = set(api_module.DecisionScoreResponse.model_fields)
    client_fields = {f for f in DecisionScore.__dataclass_fields__ if f != "raw"}
    assert server_fields <= client_fields, server_fields - client_fields

    score = rs.score_decision(default_payload("R-SHAPE"))
    for name in sorted(server_fields):
        assert hasattr(score, name), f"client dropped {name}"
        if name in score.raw:
            assert getattr(score, name) == score.raw[name], name
    assert score.score_display, "the human-readable score must survive the round trip"


def test_default_payload_is_accepted_by_the_api(rs):
    score = rs.score_decision(default_payload())
    assert score.return_id == "R999001"
    assert 0.0 <= score.score <= 1.0


# ------------------------------------------------------------ constant drift

def test_outcome_constants_match_the_server(rs):
    """The SDK's lists are literals; the server owns the truth.

    An earlier draft advertised ``abuse_confirmed`` and ``good_return``, which
    the API rejects with a 400 - so a merchant following the client's own
    constants would have been refused. Same failure mode as the route guard.
    """
    from api.main import ABUSIVE_OUTCOMES as server_abusive
    from api.main import BENIGN_OUTCOMES as server_benign

    assert set(ABUSIVE_OUTCOMES) == set(server_abusive)
    assert set(BENIGN_OUTCOMES) == set(server_benign)
    assert set(ABUSIVE_OUTCOMES) & set(BENIGN_OUTCOMES) == set()


def test_every_advertised_outcome_is_accepted(rs):
    for outcome in ABUSIVE_OUTCOMES:
        assert rs.record_decision(f"R-OK-{outcome}", outcome)["actual_abusive"] == 1
    for outcome in BENIGN_OUTCOMES:
        assert rs.record_decision(f"R-OK-{outcome}", outcome)["actual_abusive"] == 0
