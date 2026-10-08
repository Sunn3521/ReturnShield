"""End-to-end tests for the ReturnShield API.

These hit the real FastAPI app with the real model bundle. Memory is redirected
to a temporary database so tests never touch the developer's memory store, and
the app lifespan is intentionally not started so the live event generator does
not begin writing files during the test run.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import main as api_module
from api import memory as memory_module


@pytest.fixture()
def client(tmp_path):
    # Point the memory subsystem at an isolated database for this test.
    memory_module._store = memory_module.MemoryStore(tmp_path / "api-memory.db")
    memory_module.clear_ground_truth_cache()
    # Deliberately not used as a context manager: that would run the lifespan and
    # start the live event generator, which writes to the real event log.
    test_client = TestClient(api_module.app)
    yield test_client
    memory_module._store = None


@pytest.fixture()
def sample_return():
    return {
        "return_id": "R900001",
        "order_id": "O900001",
        "customer_id": "C900001",
        "order_value": 12000.0,
        "product_price": 12000.0,
        "returns_30d": 6,
        "orders_30d": 4,
        "returns_90d": 9,
        "orders_90d": 7,
        "device_linked_accounts": 5,
        "address_linked_accounts": 4,
        "return_reason": "not_as_described",
    }


# ------------------------------------------------------------------- basics


def test_health_reports_policy(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "healthy"
    assert "verify_threshold" in body["policy"]


def test_policy_thresholds_are_ordered(client):
    policy = client.get("/api/v1/policy").json()
    assert policy["verify_threshold"] < policy["review_threshold"]


def test_meta_lists_memory_and_hindsight_endpoints(client):
    body = client.get("/api/v1/meta").json()
    assert "/api/v1/hindsight" in body["endpoints"]
    assert "/api/v1/memory" in body["endpoints"]


# ------------------------------------------------------------------ scoring


def test_score_returns_a_decision_and_signals(client, sample_return):
    res = client.post("/api/v1/score", json=sample_return)
    assert res.status_code == 200
    body = res.json()
    assert body["return_id"] == "R900001"
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert body["decision"] in {"AUTO_APPROVE", "VERIFY", "MANUAL_REVIEW"}
    assert body["top_signals"]
    assert body["latency_ms"] >= 0


def test_score_rejects_invalid_payload_with_error_envelope(client):
    res = client.post("/api/v1/score", json={"return_id": "R1"})
    assert res.status_code == 422


def test_batch_score_matches_single_scoring(client, sample_return):
    """Batching must not change the answer - only the speed."""
    batch = [dict(sample_return), {**sample_return, "return_id": "R900002"}]
    res = client.post("/api/v1/batch_score", json=batch)
    assert res.status_code == 200
    batched = res.json()
    assert len(batched) == 2

    for item, row in zip(batch, batched):
        single = client.post("/api/v1/score", json=item).json()
        assert row["decision"] == single["decision"]
        assert row["risk_probability"] == single["risk_probability"]


def test_batch_score_enforces_a_limit(client, sample_return):
    too_many = [{**sample_return, "return_id": f"R{i}"} for i in range(1001)]
    res = client.post("/api/v1/batch_score", json=too_many)
    assert res.status_code == 413


def test_batch_score_logs_decisions_for_hindsight(client, sample_return):
    client.post("/api/v1/batch_score", json=[{**sample_return, "return_id": "R900003"}])
    store = memory_module.get_store()
    assert any(d["return_id"] == "R900003" for d in store.prior_decisions("R900003"))


def test_single_score_also_feeds_hindsight(client, sample_return):
    client.post("/api/v1/score", json=sample_return)
    priors = memory_module.get_store().prior_decisions("R900001")
    assert priors and priors[0]["source"] == "score"


def test_single_score_can_opt_out_of_logging(client, sample_return):
    client.post("/api/v1/score", json=sample_return, params={"store_decision": "false"})
    assert memory_module.get_store().prior_decisions("R900001") == []


# ------------------------------------------------------- conversation memory


def test_chat_returns_a_session_id_and_stores_the_turn(client):
    res = client.post("/api/v1/agent/chat", json={"message": "which returns are highest risk?"})
    assert res.status_code == 200
    body = res.json()
    assert body["session_id"]
    assert body["answer"]


def test_chat_remembers_across_calls_in_the_same_session(client):
    first = client.post("/api/v1/agent/chat", json={"message": "show customer C06472"}).json()
    sid = first["session_id"]

    client.post("/api/v1/agent/chat", json={"message": "and their risk?", "session_id": sid})

    history = client.get(f"/api/v1/memory/{sid}").json()
    assert history["turns"] >= 4  # two calls, user + assistant each
    assert history["history"][0]["role"] == "user"


def test_memory_survives_without_the_client_echoing_history(client):
    """The point of server-side memory: a fresh client resumes the thread."""
    first = client.post("/api/v1/agent/chat", json={"message": "hello"}).json()
    sid = first["session_id"]

    # No history field at all - only the session id.
    client.post("/api/v1/agent/chat", json={"message": "what did I ask you?", "session_id": sid})

    turns = client.get(f"/api/v1/memory/{sid}").json()["history"]
    assert any("hello" in t["content"] for t in turns)


def test_forget_clears_a_conversation(client):
    sid = client.post("/api/v1/agent/chat", json={"message": "remember this"}).json()["session_id"]
    client.delete(f"/api/v1/memory/{sid}")
    assert client.get(f"/api/v1/memory/{sid}").json()["history"] == []


def test_memory_can_be_listed(client):
    client.post("/api/v1/agent/chat", json={"message": "hi"})
    body = client.get("/api/v1/memory").json()
    assert body["sessions"]
    assert body["messages"] >= 1


def test_chat_can_opt_out_of_memory(client):
    res = client.post("/api/v1/agent/chat", json={"message": "hi", "use_memory": False})
    assert res.json()["session_id"] is None


# ------------------------------------------------------------ hindsight flow


def test_hindsight_round_trip_against_real_ground_truth(client):
    truth = memory_module.load_ground_truth()
    assert not truth.empty, "ground truth fixture is required for this test"

    abusive = truth[truth["abusive_return"] == 1]["return_id"].iloc[0]
    store = memory_module.get_store()
    store.record_decision(str(abusive), "MANUAL_REVIEW", 0.9, source="test")

    resolved = client.post("/api/v1/hindsight/resolve").json()
    assert resolved["resolution"]["resolved"] >= 1

    summary = resolved["summary"]
    assert summary["resolved"] >= 1
    assert summary["true_positive"] >= 1
    assert summary["accuracy"] is not None


def test_hindsight_endpoint_reports_shape(client):
    store = memory_module.get_store()
    store.record_decision("R_NO_OUTCOME", "VERIFY", 0.5)

    body = client.get("/api/v1/hindsight").json()
    assert "summary" in body
    assert body["summary"]["total_decisions"] == 1
    assert body["summary"]["pending"] == 1


def test_hindsight_for_a_specific_return_explains_prior_calls(client):
    truth = memory_module.load_ground_truth()
    abusive = truth[truth["abusive_return"] == 1]["return_id"].iloc[0]
    store = memory_module.get_store()
    store.record_decision(str(abusive), "VERIFY", 0.55)
    client.post("/api/v1/hindsight/resolve")

    body = client.get(f"/api/v1/hindsight/return/{abusive}").json()
    assert body["prior_decisions"]
    assert "confirmed abusive" in (body["summary"] or "")


def test_hindsight_for_unknown_return_is_empty(client):
    body = client.get("/api/v1/hindsight/return/R_DOES_NOT_EXIST").json()
    assert body["prior_decisions"] == []
    assert body["summary"] is None


def test_chat_logged_decisions_can_be_resolved(client):
    """End to end: chat surfaces a return, that decision gets scored later."""
    client.post("/api/v1/agent/chat", json={"message": "which returns are highest risk right now?"})
    store = memory_module.get_store()
    logged = store.hindsight_summary()["total_decisions"]
    assert logged >= 1

    summary = client.post("/api/v1/hindsight/resolve").json()["summary"]
    assert summary["resolved"] + summary["pending"] == logged


# ----------------------------------------------------- capability endpoints


def test_clusters_returns_coordinated_accounts(client):
    res = client.get("/api/v1/clusters")
    assert res.status_code == 200
    body = res.json()
    assert body["source"] in {"live", "evaluation"}
    assert isinstance(body["clusters"], list)


def test_clusters_rejects_a_silly_minimum(client):
    assert client.get("/api/v1/clusters", params={"min_accounts": 1}).status_code == 422


def test_customer_returns_404s_for_unknown_customer(client):
    res = client.get("/api/v1/customers/C_DOES_NOT_EXIST/returns")
    assert res.status_code == 404


def test_customer_returns_summarises_a_known_customer(client):
    df = api_module._evaluation_frame()
    if df.empty or "customer_id" not in df.columns:
        pytest.skip("held-out evaluation set unavailable")
    cid = str(df["customer_id"].iloc[0])

    body = client.get(f"/api/v1/customers/{cid}/returns").json()
    assert body["customer_id"] == cid.upper()
    assert body["total_returns"] >= 1
    assert body["returns"]


# ------------------------------------------------------------ error handling


def test_missing_asset_raises_a_structured_error(client, monkeypatch):
    """A missing model bundle must not leak a raw traceback to the client."""
    monkeypatch.setattr(api_module, "get_bundle", _raise_missing_bundle)
    # ServerErrorMiddleware re-raises after building the response so the server
    # can log it; raise_server_exceptions=False lets the test assert on the
    # response a real client would actually receive.
    lenient = TestClient(api_module.app, raise_server_exceptions=False)
    res = lenient.get("/api/v1/health")
    assert res.status_code == 500
    assert res.json()["error"]["code"] == "internal_error"


def _raise_missing_bundle():
    raise RuntimeError("Model bundle not found. Please run `python run_pipeline.py` first.")


def test_openapi_document_builds(client):
    res = client.get("/openapi.json")
    assert res.status_code == 200
    paths = res.json()["paths"]
    for expected in [
        "/api/v1/score",
        "/api/v1/batch_score",
        "/api/v1/agent/chat",
        "/api/v1/hindsight",
        "/api/v1/memory/{session_id}",
        "/api/v1/clusters",
        "/api/v1/customers/{customer_id}/returns",
    ]:
        assert expected in paths, f"{expected} missing from OpenAPI"
