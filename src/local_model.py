"""Local sentence-embedding model for ReturnShield chat.

Runs a quantisation-free ONNX MiniLM encoder (``all-MiniLM-L6-v2``) through
``onnxruntime`` with the ``tokenizers`` WordPiece encoder. No torch, no
transformers, no outbound network calls at inference time.

The encoder is optional and off by default so the app keeps working in a fresh
checkout with zero extra dependencies. Enable it with::

    RETURNSHIELD_LOCAL_MODEL=1 python dashboard/app.py

Everything degrades to ``None`` rather than raising: if the flag is off, the
model directory is missing, or onnxruntime is not installed, ``get_embedder()``
returns ``None`` and callers fall back to the scikit-learn classifier.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.paths import root as _project_root

DEFAULT_MODEL_DIR = _project_root() / "models" / "local" / "all-MiniLM-L6-v2"
MODEL_FILENAME = "model.onnx"
TOKENIZER_FILENAME = "tokenizer.json"

#: Similarity floor below which a semantic match is considered "no idea".
SEMANTIC_FLOOR = 0.34

_TRUTHY = {"1", "true", "on", "yes", "enable", "enabled"}


def local_model_enabled(env: dict[str, str] | None = None) -> bool:
    """True when ``RETURNSHIELD_LOCAL_MODEL`` opts in to the ONNX encoder."""
    raw = (env or os.environ).get("RETURNSHIELD_LOCAL_MODEL", "")
    return raw.strip().lower() in _TRUTHY


def model_dir() -> Path:
    """Resolve the encoder directory, honouring ``RETURNSHIELD_LOCAL_MODEL_DIR``."""
    override = os.environ.get("RETURNSHIELD_LOCAL_MODEL_DIR", "").strip()
    return Path(override) if override else DEFAULT_MODEL_DIR


class LocalEmbedder:
    """Mean-pooled, L2-normalised sentence embeddings from an ONNX encoder."""

    def __init__(self, directory: Path | str = DEFAULT_MODEL_DIR, max_length: int = 128):
        self.directory = Path(directory)
        self.max_length = max_length

        import onnxruntime as ort  # imported lazily: optional dependency
        from tokenizers import Tokenizer

        model_path = self.directory / MODEL_FILENAME
        tokenizer_path = self.directory / TOKENIZER_FILENAME
        if not model_path.exists() or not tokenizer_path.exists():
            raise FileNotFoundError(f"local encoder incomplete in {self.directory}")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # tokenizer.json ships its own truncation/padding config, but re-apply it
        # explicitly so the pad token survives as [PAD] rather than resetting.
        pad_id = self.tokenizer.token_to_id("[PAD]") or 0
        self.tokenizer.enable_truncation(max_length)
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.dim = int(self.session.get_outputs()[0].shape[-1])

    def embed(self, texts: Sequence[str] | str) -> np.ndarray:
        """Encode one or more texts into a ``(n, dim)`` normalised float32 array."""
        if isinstance(texts, str):
            texts = [texts]
        clean = [t if t.strip() else " " for t in texts]
        if not clean:
            return np.zeros((0, self.dim), dtype=np.float32)

        encodings = self.tokenizer.encode_batch(clean)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        feeds: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention,
        }
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        feeds = {k: v for k, v in feeds.items() if k in self.input_names}

        hidden = self.session.run(None, feeds)[0]  # (n, seq, dim)
        mask = attention[:, :, None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-9, None)).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix of two sets of row vectors -> ``(len(a), len(b))``.

    Rows are normalised defensively so the result is a true cosine even when the
    caller passes un-normalised vectors; the router's embeddings are already unit
    length, making this a no-op in the hot path.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float32))
    b = np.atleast_2d(np.asarray(b, dtype=np.float32))
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-9, None)
    return a @ b.T


PROTOTYPES_FILENAME = "intent_prototypes.npz"


def prototypes_path() -> Path:
    """Where the precomputed intent centroids live."""
    override = os.environ.get("RETURNSHIELD_PROTOTYPES", "").strip()
    return Path(override) if override else DEFAULT_MODEL_DIR.parent / PROTOTYPES_FILENAME


def fit_prototypes(embedder: "LocalEmbedder", dataset: dict[str, Iterable[str]]):
    """Embed every training utterance and average it into one centroid per intent."""
    intents, rows = [], []
    for intent in sorted(dataset):
        phrases = [p for p in dataset[intent] if p.strip()]
        if not phrases:
            continue
        centroid = embedder.embed(phrases).mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-9)
        intents.append(intent)
        rows.append(centroid)
    return intents, np.vstack(rows).astype(np.float32)


def save_prototypes(intents: Sequence[str], centroids: np.ndarray, path: Path | str | None = None) -> Path:
    """Persist intent centroids (~26 KB) so startup needs no re-embedding."""
    target = Path(path) if path else prototypes_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, intents=np.asarray(list(intents), dtype=object), centroids=centroids)
    return target


def load_prototypes(path: Path | str | None = None) -> tuple[list[str], np.ndarray] | None:
    """Load precomputed centroids, or ``None`` when they are absent/unreadable."""
    source = Path(path) if path else prototypes_path()
    if not source.exists():
        return None
    try:
        with np.load(source, allow_pickle=True) as data:
            return [str(i) for i in data["intents"]], np.asarray(data["centroids"], dtype=np.float32)
    except Exception:
        return None


_CACHED: dict[str, Any] = {"embedder": None, "tried": False, "error": None}


def get_embedder(reload: bool = False) -> LocalEmbedder | None:
    """Return the process-wide embedder, or ``None`` when disabled/unavailable."""
    if reload:
        _CACHED.update({"embedder": None, "tried": False, "error": None})
    if _CACHED["tried"]:
        return _CACHED["embedder"]
    _CACHED["tried"] = True
    if not local_model_enabled():
        return None
    try:
        _CACHED["embedder"] = LocalEmbedder(model_dir())
    except Exception as exc:  # pragma: no cover - depends on local install
        _CACHED["error"] = str(exc)
        _CACHED["embedder"] = None
    return _CACHED["embedder"]


def status() -> dict[str, Any]:
    """Introspection payload for /api/v1/meta and the dashboard."""
    directory = model_dir()
    embedder = get_embedder()
    return {
        "enabled": local_model_enabled(),
        "available": embedder is not None,
        "model": "sentence-transformers/all-MiniLM-L6-v2 (ONNX)",
        "directory": str(directory),
        "dim": embedder.dim if embedder else None,
        "max_length": embedder.max_length if embedder else None,
        "error": _CACHED.get("error"),
    }


class SemanticIntentRouter:
    """Nearest-prototype intent router over the local embedding space.

    Prototypes are the per-intent seed phrases from ``src.intent_data``; each is
    embedded once and averaged into a single centroid, so matching is a single
    matrix product at query time.
    """

    def __init__(
        self,
        prototypes: dict[str, Iterable[str]] | None = None,
        floor: float = SEMANTIC_FLOOR,
        precomputed: tuple[Sequence[str], np.ndarray] | None = None,
    ):
        self.floor = floor
        embedder = get_embedder()
        if embedder is None:
            raise RuntimeError("local embedder unavailable")
        self.embedder = embedder

        if precomputed is not None:
            intents, stacked = precomputed
            self.intents = [str(i) for i in intents]
            self.centroids = np.asarray(stacked, dtype=np.float32)
            self.weights = np.ones(len(self.intents), dtype=np.float32)
            return

        if not prototypes:
            raise ValueError("SemanticIntentRouter needs prototypes or precomputed centroids")
        self.intents, self.centroids = fit_prototypes(embedder, prototypes)
        self.weights = np.ones(len(self.intents), dtype=np.float32)

    def route(self, text: str) -> tuple[str | None, float]:
        """Best intent by cosine similarity, or ``(None, score)`` under the floor."""
        if not text.strip():
            return None, 0.0
        scores = cosine(self.embedder.embed(text), self.centroids)[0]
        best = int(np.argmax(scores))
        score = float(scores[best])
        return (self.intents[best], score) if score >= self.floor else (None, score)
