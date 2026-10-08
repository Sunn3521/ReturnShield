"""Tests for the optional local ONNX embedding model.

These must all pass with the encoder absent, disabled, or present -- the chat
agent's zero-dependency path is the contract, the encoder is the upgrade.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src import local_model

ENCODER_DIR = local_model.DEFAULT_MODEL_DIR
ENCODER_PRESENT = (ENCODER_DIR / local_model.MODEL_FILENAME).exists() and (
    ENCODER_DIR / local_model.TOKENIZER_FILENAME
).exists()

needs_encoder = pytest.mark.skipif(
    not ENCODER_PRESENT,
    reason="local ONNX encoder not downloaded (python build_prototypes.py / fetch models)",
)


def test_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv("RETURNSHIELD_LOCAL_MODEL", raising=False)
    assert local_model.local_model_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "on", "yes", "enabled"])
def test_truthy_flag_values(monkeypatch, value):
    monkeypatch.setenv("RETURNSHIELD_LOCAL_MODEL", value)
    assert local_model.local_model_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "off", "no", "", "maybe"])
def test_falsy_flag_values(monkeypatch, value):
    monkeypatch.setenv("RETURNSHIELD_LOCAL_MODEL", value)
    assert local_model.local_model_enabled() is False


def test_get_embedder_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("RETURNSHIELD_LOCAL_MODEL", raising=False)
    assert local_model.get_embedder(reload=True) is None


def test_agent_survives_a_missing_encoder_directory(monkeypatch, tmp_path):
    """A broken encoder path must not break routing -- just skip the upgrade."""
    monkeypatch.setenv("RETURNSHIELD_LOCAL_MODEL", "1")
    monkeypatch.setenv("RETURNSHIELD_LOCAL_MODEL_DIR", str(tmp_path / "nope"))
    monkeypatch.setattr(local_model, "get_embedder", local_model.get_embedder)

    from src.chat_agent import ChatContext, ReturnShieldChatAgent

    assert local_model.get_embedder(reload=True) is None

    agent = ReturnShieldChatAgent()
    intent, confidence, _slots = agent.recognize("what share of returns are abusive", ChatContext())
    assert intent
    assert 0.0 < confidence <= 1.0
    assert agent.semantic_router is None


def test_prototypes_round_trip(tmp_path):
    import numpy as np

    centroids = np.eye(3, 4, dtype="float32")
    path = local_model.save_prototypes(["a", "b", "c"], centroids, tmp_path / "p.npz")
    intents, loaded = local_model.load_prototypes(path)
    assert intents == ["a", "b", "c"]
    assert loaded.shape == (3, 4)


def test_load_prototypes_returns_none_when_absent(tmp_path):
    assert local_model.load_prototypes(tmp_path / "missing.npz") is None


def test_cosine_is_symmetric_and_bounded():
    """Un-normalised inputs must still yield a bounded, symmetric true cosine."""
    import numpy as np

    rng = np.random.default_rng(7)
    a = rng.normal(size=(4, 8)).astype("float32")
    b = rng.normal(size=(4, 8)).astype("float32")
    forward = local_model.cosine(a, b)
    backward = local_model.cosine(b, a)
    assert forward.shape == (4, 4)
    assert np.allclose(forward, backward.T, atol=1e-5)
    assert np.all(forward <= 1.0001) and np.all(forward >= -1.0001)


def test_cosine_of_a_vector_with_itself_is_one():
    import numpy as np

    v = np.array([[3.0, 4.0]], dtype="float32")
    assert abs(float(local_model.cosine(v, v)[0, 0]) - 1.0) < 1e-6


@needs_encoder
def test_embeddings_are_normalised(monkeypatch):
    monkeypatch.setenv("RETURNSHIELD_LOCAL_MODEL", "1")
    embedder = local_model.get_embedder(reload=True)
    assert embedder is not None

    vectors = embedder.embed(["what share of returns are abusive", "hi"])
    assert vectors.shape == (2, embedder.dim)
    norms = (vectors ** 2).sum(axis=1) ** 0.5
    assert all(abs(float(n) - 1.0) < 1e-3 for n in norms)


@needs_encoder
def test_single_string_and_empty_inputs(monkeypatch):
    monkeypatch.setenv("RETURNSHIELD_LOCAL_MODEL", "1")
    embedder = local_model.get_embedder(reload=True)
    assert embedder.embed("hello").shape == (1, embedder.dim)
    assert embedder.embed([]).shape == (0, embedder.dim)


@needs_encoder
def test_semantically_similar_text_is_closer_than_unrelated(monkeypatch):
    monkeypatch.setenv("RETURNSHIELD_LOCAL_MODEL", "1")
    embedder = local_model.get_embedder(reload=True)
    query, near, far = embedder.embed([
        "who are the repeat offenders driving the biggest losses",
        "which returns carry the highest abuse risk",
        "please bake the sourdough loaf this afternoon",
    ])
    near_score = float(local_model.cosine(query, near)[0, 0])
    far_score = float(local_model.cosine(query, far)[0, 0])
    assert near_score > far_score, (near_score, far_score)


@needs_encoder
def test_agent_uses_encoder_when_enabled(monkeypatch):
    monkeypatch.setenv("RETURNSHIELD_LOCAL_MODEL", "1")
    monkeypatch.delenv("RETURNSHIELD_LOCAL_MODEL_DIR", raising=False)

    from src.chat_agent import ChatContext, ReturnShieldChatAgent

    agent = ReturnShieldChatAgent()
    assert agent.semantic_router is not None
    assert agent.semantic_source in {"artifact", "rebuilt"}
    intent, confidence, _slots = agent.recognize("what can you do", ChatContext())
    assert intent and 0.0 < confidence <= 1.0


@needs_encoder
def test_semantic_fallback_rescues_a_paraphrase(monkeypatch):
    """The whole point of the encoder: wording with no keyword overlap."""
    monkeypatch.setenv("RETURNSHIELD_LOCAL_MODEL", "1")
    monkeypatch.delenv("RETURNSHIELD_LOCAL_MODEL_DIR", raising=False)

    from src.chat_agent import SEMANTIC_TRIGGER, ChatContext, ReturnShieldChatAgent

    agent = ReturnShieldChatAgent()
    assert agent.semantic_router is not None

    text = "give me the 4111111111111111 card activity"
    probs = agent.model.predict_proba([text])[0]
    if float(probs.max()) >= SEMANTIC_TRIGGER:
        pytest.skip("TF-IDF is already confident here; nothing for the fallback to rescue")

    intent, _confidence, slots = agent.recognize(text, ChatContext())
    assert intent == "customer_search", (intent, probs.max())
    assert slots.get("intent_source") in {None, "local_model"}
