"""Persistent memory for the ReturnShield agent.

Two kinds of memory live here, both backed by SQLite so they survive a restart:

1. Conversation memory - every chat turn is stored server-side and keyed by
   ``session_id``. Previously the browser echoed the whole history back on each
   request, so a refresh lost everything the agent had learned about the user.

2. Outcome feedback ("hindsight") - every decision the agent or scorer makes is
   logged. Later, :func:`MemoryStore.resolve_outcomes` joins those decisions
   against the known abuse outcomes and records whether the call was right.
   That lets the agent answer "we flagged this return before - here is how it
   turned out" instead of reasoning from scratch every time.

The store is intentionally dependency-free (stdlib ``sqlite3``) and safe for
FastAPI's threadpool via a connection-per-call plus WAL journaling.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.paths import root as _project_root

ROOT = _project_root()
DEFAULT_DB_PATH = ROOT / "data" / "memory" / "returnshield.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    role        TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    meta        TEXT,
    created_at  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS decisions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    return_id          TEXT    NOT NULL,
    decision           TEXT    NOT NULL,
    risk_probability   REAL,
    source             TEXT    NOT NULL DEFAULT 'api',
    session_id         TEXT,
    policy_version     TEXT,
    created_at         REAL    NOT NULL,
    resolved           INTEGER NOT NULL DEFAULT 0,
    resolved_at        REAL,
    actual_abusive     INTEGER,
    actual_merchant_loss REAL,
    correct            INTEGER
);
CREATE INDEX IF NOT EXISTS idx_decisions_return ON decisions(return_id);
CREATE INDEX IF NOT EXISTS idx_decisions_resolved ON decisions(resolved);

CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    turns       INTEGER NOT NULL DEFAULT 0
);
"""


def new_session_id() -> str:
    """Return a fresh conversation identifier."""
    return uuid.uuid4().hex[:16]


class MemoryStore:
    """Thread-safe SQLite store for conversation and decision memory."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lock = threading.Lock()
        self._initialised = False

    # ------------------------------------------------------------------ setup

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_schema(self) -> None:
        if self._initialised:
            return
        with self._init_lock:
            if self._initialised:
                return
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
            self._initialised = True

    # ------------------------------------------------------- conversation memory

    def append_message(self, session_id: str, role: str, content: str, meta: dict[str, Any] | None = None) -> None:
        """Store one chat turn and bump the session counters."""
        if not session_id or content is None:
            return
        self._ensure_schema()
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, meta, created_at) VALUES (?,?,?,?,?)",
                (session_id, role, str(content), json.dumps(meta or {}), now),
            )
            conn.execute(
                "INSERT INTO sessions (session_id, created_at, updated_at, turns) VALUES (?,?,?,1) "
                "ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at, turns=turns+1",
                (session_id, now, now),
            )

    def get_history(self, session_id: str, limit: int = 40) -> list[dict[str, str]]:
        """Return the most recent turns for a session, oldest first."""
        if not session_id:
            return []
        self._ensure_schema()
        limit = max(1, min(int(limit), 500))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def clear_session(self, session_id: str) -> int:
        """Forget a conversation. Returns the number of turns removed."""
        if not session_id:
            return 0
        self._ensure_schema()
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            removed = cur.rowcount or 0
            conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        return removed

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        self._ensure_schema()
        limit = max(1, min(int(limit), 500))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, created_at, updated_at, turns FROM sessions "
                "ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ----------------------------------------------------------- decision memory

    def record_decision(
        self,
        return_id: str,
        decision: str,
        risk_probability: float | None = None,
        source: str = "api",
        session_id: str | None = None,
        policy_version: str | None = None,
    ) -> None:
        """Log a decision so it can later be checked against the truth."""
        if not return_id or not decision:
            return
        self._ensure_schema()
        try:
            prob = None if risk_probability is None else float(risk_probability)
        except (TypeError, ValueError):
            prob = None
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO decisions "
                "(return_id, decision, risk_probability, source, session_id, policy_version, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (str(return_id), str(decision), prob, source, session_id, policy_version, time.time()),
            )

    def record_outcome(
        self,
        return_id: str,
        actual_abusive: bool | int,
        merchant_loss: float | None = None,
        risk_probability: float | None = None,
        decision: str | None = None,
        source: str = "feedback",
    ) -> dict[str, Any]:
        """Record ground truth for a return and resolve the calls already made.

        This is the live half of the hindsight loop. A scored return is logged
        unresolved by :meth:`record_decision`; feeding the real outcome back here
        resolves every open call for that return in one shot, so both the
        hindsight summary and the threshold recommendation reflect it on the very
        next request with no offline job - the resolved row keeps the probability
        that was scored, which is what the recommendation prices.

        When no call was logged yet, one resolved row is inserted. If the caller
        supplies ``risk_probability`` it joins the recommendation sample like any
        other. An outcome-only report has no probability to re-price a threshold
        against, so it is counted by the summary but stays out of the sample.
        """
        result: dict[str, Any] = {"inserted": 0, "updated": 0, "actual_abusive": None}
        if not return_id:
            return result
        self._ensure_schema()
        actual = 1 if int(bool(actual_abusive)) else 0
        loss = None if merchant_loss is None else float(merchant_loss)
        prob = None if risk_probability is None else float(risk_probability)
        now = time.time()
        with self._connect() as conn:
            pending = conn.execute(
                "SELECT id, decision FROM decisions WHERE return_id=? AND resolved=0",
                (str(return_id),),
            ).fetchall()
            for row in pending:
                flagged = str(row["decision"]).upper() in {"VERIFY", "MANUAL_REVIEW"}
                correct = 1 if (flagged and actual == 1) or (not flagged and actual == 0) else 0
                conn.execute(
                    "UPDATE decisions SET resolved=1, resolved_at=?, actual_abusive=?, "
                    "actual_merchant_loss=?, correct=? WHERE id=?",
                    (now, actual, loss, correct, int(row["id"])),
                )
                result["updated"] += 1
            if not pending:
                call = str(decision or ("MANUAL_REVIEW" if actual == 1 else "AUTO_APPROVE")).upper()
                flagged = call in {"VERIFY", "MANUAL_REVIEW"}
                correct = 1 if (flagged and actual == 1) or (not flagged and actual == 0) else 0
                conn.execute(
                    "INSERT INTO decisions (return_id, decision, risk_probability, source, "
                    "session_id, policy_version, created_at, resolved, resolved_at, "
                    "actual_abusive, actual_merchant_loss, correct) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (str(return_id), call, prob, source, None, None, now, 1, now, actual, loss, correct),
                )
                result["inserted"] = 1
        result["actual_abusive"] = actual
        return result

    def prior_decisions(self, return_id: str, limit: int = 5) -> list[dict[str, Any]]:
        """What did we previously decide about this return, and was it right?"""
        if not return_id:
            return []
        self._ensure_schema()
        limit = max(1, min(int(limit), 50))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT return_id, decision, risk_probability, source, created_at, "
                "resolved, resolved_at, actual_abusive, correct "
                "FROM decisions WHERE return_id=? ORDER BY id DESC LIMIT ?",
                (str(return_id), limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # -------------------------------------------------------------- hindsight

    def resolve_outcomes(self, truth: pd.DataFrame) -> dict[str, Any]:
        """Join logged decisions against known outcomes and score them.

        ``truth`` must expose ``return_id`` and ``abusive_return`` columns, and
        optionally ``merchant_loss``. Only unresolved decisions are touched, so
        this is cheap and safe to call repeatedly.
        """
        self._ensure_schema()
        summary = {"considered": 0, "resolved": 0, "already_resolved": 0, "no_truth": 0}

        if truth is None or truth.empty or "return_id" not in truth.columns:
            return summary

        cols = ["return_id", "abusive_return"]
        if "merchant_loss" in truth.columns:
            cols.append("merchant_loss")
        frame = truth[cols].copy()
        frame["return_id"] = frame["return_id"].astype(str)
        frame = frame.drop_duplicates(subset=["return_id"], keep="last")
        frame["abusive_return"] = pd.to_numeric(frame["abusive_return"], errors="coerce")
        if "merchant_loss" in frame.columns:
            frame["merchant_loss"] = pd.to_numeric(frame["merchant_loss"], errors="coerce")

        with self._connect() as conn:
            pending = conn.execute(
                "SELECT id, return_id, decision FROM decisions WHERE resolved=0"
            ).fetchall()
            summary["considered"] = len(pending)

            if pending:
                known = {row["return_id"]: row for row in frame.to_dict("records")}
                updates = []
                for row in pending:
                    actual = known.get(str(row["return_id"]))
                    if actual is None or actual.get("abusive_return") is None:
                        continue
                    abusive = int(actual["abusive_return"] == 1)
                    # "Correct" means the call matched reality: flagging an abusive
                    # return is right, approving a genuine one is right.
                    decision = str(row["decision"]).upper()
                    flagged = decision in {"VERIFY", "MANUAL_REVIEW"}
                    correct = 1 if (flagged and abusive == 1) or (not flagged and abusive == 0) else 0
                    updates.append(
                        (time.time(), abusive, actual.get("merchant_loss"), correct, int(row["id"]))
                    )

                if updates:
                    conn.executemany(
                        "UPDATE decisions SET resolved=1, resolved_at=?, actual_abusive=?, "
                        "actual_merchant_loss=?, correct=? WHERE id=?",
                        updates,
                    )
                summary["resolved"] = len(updates)

            # Always report the running total, even when nothing was pending.
            summary["already_resolved"] = int(
                conn.execute("SELECT COUNT(*) FROM decisions WHERE resolved=1").fetchone()[0]
            )
        return summary

    def hindsight_summary(self) -> dict[str, Any]:
        """Aggregate how the agent's past calls actually turned out."""
        self._ensure_schema()
        with self._connect() as conn:
            totals = conn.execute(
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN resolved=1 THEN 1 ELSE 0 END) AS resolved, "
                "SUM(CASE WHEN decision IN ('VERIFY','MANUAL_REVIEW') THEN 1 ELSE 0 END) AS flagged "
                "FROM decisions"
            ).fetchone()
            rows = conn.execute(
                "SELECT decision, correct, actual_abusive, actual_merchant_loss, risk_probability "
                "FROM decisions WHERE resolved=1"
            ).fetchall()

        total = int(totals["total"] or 0)
        resolved = int(totals["resolved"] or 0)
        flagged = int(totals["flagged"] or 0)

        by_decision: dict[str, dict[str, int]] = {}
        tp = fp = tn = fn = 0
        missed_loss = 0.0
        correct_n = 0
        for r in rows:
            d = str(r["decision"]).upper()
            bucket = by_decision.setdefault(d, {"calls": 0, "correct": 0, "actual_abusive": 0})
            bucket["calls"] += 1
            abusive = int(r["actual_abusive"] or 0)
            bucket["actual_abusive"] += abusive
            if int(r["correct"] or 0) == 1:
                bucket["correct"] += 1
                correct_n += 1
            was_flagged = d in {"VERIFY", "MANUAL_REVIEW"}
            if was_flagged and abusive:
                tp += 1
            elif was_flagged and not abusive:
                fp += 1
            elif not was_flagged and not abusive:
                tn += 1
            else:
                fn += 1
                missed_loss += float(r["actual_merchant_loss"] or 0.0)

        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None

        return {
            "total_decisions": total,
            "resolved": resolved,
            "pending": total - resolved,
            "flagged": flagged,
            "accuracy": round(correct_n / resolved, 4) if resolved else None,
            "precision_flagged": round(precision, 4) if precision is not None else None,
            "recall_on_abuse": round(recall, 4) if recall is not None else None,
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "merchant_loss_from_missed_abuse": round(missed_loss, 2),
            "by_decision": by_decision,
            "note": (
                "No decisions have been resolved against ground truth yet. "
                "Log calls, then POST /api/v1/hindsight/resolve."
                if resolved == 0
                else "Scores reflect logged decisions joined to known abuse outcomes."
            ),
        }

    def recent_feedback(self, limit: int = 25) -> list[dict[str, Any]]:
        """Recent resolved decisions, newest first - the raw material for hindsight."""
        self._ensure_schema()
        limit = max(1, min(int(limit), 200))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT return_id, decision, risk_probability, source, resolved, resolved_at, "
                "actual_abusive, actual_merchant_loss, correct, created_at "
                "FROM decisions WHERE resolved=1 ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def resolved_decision_rows(self) -> list[dict[str, Any]]:
        """Every resolved decision, oldest first, for threshold recommendations.

        ``recent_feedback`` is capped and newest-first (a UI feed); this one is
        the full resolved sample the recommendation needs.
        """
        self._ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT return_id, decision, risk_probability, source, session_id, "
                "policy_version, created_at, actual_abusive, actual_merchant_loss, correct "
                "FROM decisions WHERE resolved=1 AND risk_probability IS NOT NULL "
                "ORDER BY id ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        self._ensure_schema()
        with self._connect() as conn:
            messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            decisions = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
            sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        return {
            "db_path": str(self.db_path),
            "messages": int(messages),
            "decisions": int(decisions),
            "sessions": int(sessions),
        }


_store: MemoryStore | None = None
_store_lock = threading.Lock()


def get_store(db_path: str | Path | None = None) -> MemoryStore:
    """Process-wide singleton memory store."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = MemoryStore(db_path)
    return _store


# --------------------------------------------------------------------- truth


_truth_cache: dict[str, pd.DataFrame] = {}
_truth_lock = threading.Lock()


def load_ground_truth() -> pd.DataFrame:
    """Load abuse outcomes keyed by ``return_id``.

    Cached because it is read on every hindsight resolution and never changes
    during a run. Returns an empty frame when the source data is unavailable so
    callers can degrade instead of raising.
    """
    path = ROOT / "data" / "raw" / "return_outcomes.csv"
    key = str(path)
    cached = _truth_cache.get(key)
    if cached is not None:
        return cached
    with _truth_lock:
        cached = _truth_cache.get(key)
        if cached is not None:
            return cached
        if not path.exists():
            frame = pd.DataFrame(columns=["return_id", "abusive_return", "merchant_loss"])
        else:
            frame = pd.read_csv(path)
            if "return_id" in frame.columns:
                frame["return_id"] = frame["return_id"].astype(str)
        _truth_cache[key] = frame
    return frame


def clear_ground_truth_cache() -> None:
    """Drop the cached outcomes (used by tests)."""
    with _truth_lock:
        _truth_cache.clear()


def hindsight_context(return_id: str | None, store: MemoryStore | None = None) -> dict[str, Any]:
    """Compact 'what we did last time' block that can be injected into a prompt."""
    store = store or get_store()
    if not return_id:
        return {}
    priors = store.prior_decisions(return_id, limit=3)
    if not priors:
        return {}
    lines = []
    for p in priors:
        outcome = "outcome unknown"
        if p.get("resolved"):
            outcome = "confirmed abusive" if p.get("actual_abusive") else "legitimate return"
            if not p.get("correct"):
                outcome += " (we called this one wrong)"
        lines.append(
            f"- {p['decision']} at risk {float(p.get('risk_probability') or 0):.1%} - {outcome}"
        )
    return {"return_id": return_id, "prior_decisions": priors, "summary": "\n".join(lines)}


def iter_messages(store: MemoryStore) -> Iterable[dict[str, str]]:
    """Convenience iterator over all stored turns (debugging helper)."""
    for s in store.list_sessions(limit=500):
        yield from store.get_history(s["session_id"], limit=500)
