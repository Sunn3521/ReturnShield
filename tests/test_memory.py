"""Tests for the persistent memory store and hindsight aggregation."""

from __future__ import annotations

import pandas as pd
import pytest

from api.memory import MemoryStore, hindsight_context, new_session_id


@pytest.fixture()
def store(tmp_path):
    return MemoryStore(tmp_path / "memory.db")


def test_conversation_history_round_trips(store):
    sid = new_session_id()
    store.append_message(sid, "user", "which returns are riskiest?")
    store.append_message(sid, "assistant", "R006626 is the highest risk return.")

    history = store.get_history(sid)
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[1]["content"].startswith("R006626")


def test_history_is_session_scoped(store):
    a, b = new_session_id(), new_session_id()
    store.append_message(a, "user", "hello from A")
    store.append_message(b, "user", "hello from B")

    assert len(store.get_history(a)) == 1
    assert store.get_history(a)[0]["content"] == "hello from A"


def test_history_respects_limit_and_returns_oldest_first(store):
    sid = new_session_id()
    for i in range(10):
        store.append_message(sid, "user", f"message {i}")

    recent = store.get_history(sid, limit=3)
    assert [m["content"] for m in recent] == ["message 7", "message 8", "message 9"]


def test_clear_session_removes_turns_and_keeps_others(store):
    a, b = new_session_id(), new_session_id()
    store.append_message(a, "user", "a1")
    store.append_message(b, "user", "b1")

    assert store.clear_session(a) == 1
    assert store.get_history(a) == []
    assert len(store.get_history(b)) == 1


def test_memory_survives_reopening_the_database(tmp_path):
    path = tmp_path / "persist.db"
    sid = new_session_id()

    first = MemoryStore(path)
    first.append_message(sid, "user", "remember me")

    second = MemoryStore(path)
    assert second.get_history(sid)[0]["content"] == "remember me"


def test_record_decision_and_resolve_against_truth(store):
    sid = new_session_id()
    store.record_decision("R1", "MANUAL_REVIEW", 0.91, session_id=sid)
    store.record_decision("R2", "AUTO_APPROVE", 0.04, session_id=sid)

    truth = pd.DataFrame(
        {
            "return_id": ["R1", "R2"],
            "abusive_return": [1, 1],  # R2 was an approval mistake
            "merchant_loss": [5000.0, 200.0],
        }
    )
    summary = store.resolve_outcomes(truth)
    assert summary["resolved"] == 2

    priors = {p["return_id"]: p for p in store.prior_decisions("R1") + store.prior_decisions("R2")}
    assert priors["R1"]["correct"] == 1
    assert priors["R2"]["correct"] == 0


def test_resolve_is_idempotent(store):
    store.record_decision("R1", "VERIFY", 0.5)
    truth = pd.DataFrame({"return_id": ["R1"], "abusive_return": [0]})

    first = store.resolve_outcomes(truth)
    second = store.resolve_outcomes(truth)

    assert first["resolved"] == 1
    assert second["resolved"] == 0  # nothing left to resolve
    assert second["already_resolved"] == 1


def test_resolve_ignores_returns_without_known_outcome(store):
    store.record_decision("R_UNSEEN", "VERIFY", 0.5)
    truth = pd.DataFrame({"return_id": ["R1"], "abusive_return": [1]})

    summary = store.resolve_outcomes(truth)
    assert summary["resolved"] == 0
    assert store.hindsight_summary()["pending"] == 1


def test_resolve_handles_empty_truth(store):
    store.record_decision("R1", "VERIFY", 0.5)
    summary = store.resolve_outcomes(pd.DataFrame())
    assert summary["resolved"] == 0


def test_hindsight_summary_computes_precision_recall_and_loss(store):
    # TP: flagged and abusive. FP: flagged but legitimate.
    # TN: approved and legitimate. FN: approved but abusive (cost).
    store.record_decision("TP", "MANUAL_REVIEW", 0.9)
    store.record_decision("FP", "VERIFY", 0.4)
    store.record_decision("TN", "AUTO_APPROVE", 0.02)
    store.record_decision("FN", "AUTO_APPROVE", 0.1)

    truth = pd.DataFrame(
        {
            "return_id": ["TP", "FP", "TN", "FN"],
            "abusive_return": [1, 0, 0, 1],
            "merchant_loss": [0.0, 0.0, 0.0, 900.0],
        }
    )
    store.resolve_outcomes(truth)
    result = store.hindsight_summary()

    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["true_negative"] == 1
    assert result["false_negative"] == 1
    assert result["precision_flagged"] == 0.5
    assert result["recall_on_abuse"] == 0.5
    assert result["accuracy"] == 0.5
    assert result["merchant_loss_from_missed_abuse"] == 900.0


def test_hindsight_summary_without_resolved_rows_explains_itself(store):
    store.record_decision("R1", "VERIFY", 0.5)
    result = store.hindsight_summary()
    assert result["resolved"] == 0
    assert result["accuracy"] is None
    assert "No decisions have been resolved" in result["note"]


def test_hindsight_context_summarises_prior_calls(store):
    store.record_decision("R42", "MANUAL_REVIEW", 0.8)
    store.resolve_outcomes(pd.DataFrame({"return_id": ["R42"], "abusive_return": [1]}))

    ctx = hindsight_context("R42", store)
    assert ctx["return_id"] == "R42"
    assert "MANUAL_REVIEW" in ctx["summary"]
    assert "confirmed abusive" in ctx["summary"]


def test_hindsight_context_is_empty_for_unknown_return(store):
    assert hindsight_context("NOPE", store) == {}
    assert hindsight_context(None, store) == {}


def test_bad_risk_probability_does_not_break_recording(store):
    store.record_decision("R1", "VERIFY", risk_probability="not-a-number")
    prior = store.prior_decisions("R1")[0]
    assert prior["risk_probability"] is None


def test_stats_reports_contents(store):
    sid = new_session_id()
    store.append_message(sid, "user", "hi")
    store.record_decision("R1", "VERIFY", 0.3)

    stats = store.stats()
    assert stats["messages"] == 1
    assert stats["decisions"] == 1
    assert stats["sessions"] == 1
