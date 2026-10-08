"""Write-side store for the decision API: events, review queue, webhook log.

Separate from :mod:`src.memory` on purpose. Memory owns conversation history and
the scored-decision audit trail that the hindsight loop replays; this store owns
the *ingress* and *operational queue* state:

* ``events``     - idempotent by ``(event_id, environment)``. Re-sending an event
                   is a no-op, which is the property a retrying client depends on.
* ``reviews``    - the human worklist. ``MANUAL_REVIEW`` used to be a label with
                   nowhere to go; this is the queue it drains into.
* ``webhook_deliveries`` - every outbound attempt with its status and error, so a
                   failed callback is diagnosable rather than silent.

Everything is namespaced by ``environment`` so a sandbox event or sandbox review
can never be listed alongside - or influence - production.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.paths import root as _project_root

ROOT = _project_root()
DEFAULT_DB_PATH = ROOT / "data" / "memory" / "decision_api.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id    TEXT NOT NULL,
    environment TEXT NOT NULL DEFAULT 'production',
    kind        TEXT,
    return_id   TEXT,
    customer_id TEXT,
    payload     TEXT NOT NULL,
    received_at REAL NOT NULL,
    backfill    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (event_id, environment)
);
CREATE INDEX IF NOT EXISTS idx_events_env ON events(environment, received_at);

CREATE TABLE IF NOT EXISTS reviews (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    return_id        TEXT NOT NULL,
    environment      TEXT NOT NULL DEFAULT 'production',
    state            TEXT NOT NULL DEFAULT 'pending',
    action           TEXT,
    risk_probability REAL,
    expected_cost    REAL,
    reasons          TEXT,
    policy_version   TEXT,
    created_at       REAL NOT NULL,
    resolved_at      REAL,
    resolution       TEXT,
    actor            TEXT,
    note             TEXT
);
CREATE INDEX IF NOT EXISTS idx_reviews_queue ON reviews(environment, state, expected_cost);

CREATE TABLE IF NOT EXISTS webhook_deliveries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type   TEXT NOT NULL,
    url          TEXT NOT NULL,
    environment  TEXT NOT NULL DEFAULT 'production',
    payload      TEXT NOT NULL,
    signature    TEXT,
    status       TEXT NOT NULL,
    attempts     INTEGER NOT NULL DEFAULT 0,
    last_error   TEXT,
    created_at   REAL NOT NULL,
    delivered_at REAL
);
CREATE INDEX IF NOT EXISTS idx_deliveries_env ON webhook_deliveries(environment, id);
"""


class DecisionStore:
    """Thread-safe SQLite store for ingress events, reviews, and webhooks."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lock = threading.Lock()
        self._initialised = False

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

    # ----------------------------------------------------------------- events

    def ingest(
        self,
        events: Iterable[Dict[str, Any]],
        environment: str = "production",
        backfill: bool = False,
    ) -> Dict[str, Any]:
        """Insert events idempotently.

        Returns counts and a per-event error list. A re-sent ``event_id`` inside
        the same environment counts as a duplicate and writes nothing, so a
        client retrying after a timeout cannot double-count a return.
        """
        self._ensure_schema()
        items = list(events or [])
        summary: Dict[str, Any] = {
            "total": len(items),
            "accepted": 0,
            "duplicates": 0,
            "rejected": 0,
            "errors": [],
            "environment": environment,
            "backfill": bool(backfill),
        }
        now = time.time()
        with self._connect() as conn:
            for event in items:
                if not isinstance(event, dict):
                    summary["rejected"] += 1
                    summary["errors"].append({"event_id": None, "error": "event must be a JSON object"})
                    continue
                event_id = str(event.get("event_id") or "").strip()
                if not event_id:
                    summary["rejected"] += 1
                    summary["errors"].append({"event_id": None, "error": "event_id is required"})
                    continue
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO events "
                    "(event_id, environment, kind, return_id, customer_id, payload, received_at, backfill) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        event_id,
                        environment,
                        str(event.get("kind") or "return"),
                        _opt_str(event.get("return_id")),
                        _opt_str(event.get("customer_id")),
                        json.dumps(event, default=str),
                        now,
                        1 if backfill else 0,
                    ),
                )
                if cursor.rowcount:
                    summary["accepted"] += 1
                else:
                    summary["duplicates"] += 1
        return summary

    def count_events(self, environment: str = "production") -> int:
        self._ensure_schema()
        with self._connect() as conn:
            return int(conn.execute(
                "SELECT COUNT(*) FROM events WHERE environment=?", (environment,)
            ).fetchone()[0])

    def list_events(self, limit: int = 100, offset: int = 0, environment: str = "production") -> List[Dict[str, Any]]:
        self._ensure_schema()
        limit = max(1, min(int(limit), 5000))
        offset = max(0, int(offset))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT event_id, environment, kind, return_id, customer_id, payload, "
                "received_at, backfill FROM events WHERE environment=? "
                "ORDER BY received_at DESC, rowid DESC LIMIT ? OFFSET ?",
                (environment, limit, offset),
            ).fetchall()
        out = []
        for row in rows:
            record = dict(row)
            try:
                record["payload"] = json.loads(record["payload"])
            except (TypeError, ValueError):
                pass
            out.append(record)
        return out

    # ---------------------------------------------------------------- reviews

    def enqueue_review(
        self,
        return_id: str,
        environment: str = "production",
        action: str | None = None,
        risk_probability: float | None = None,
        expected_cost: float | None = None,
        reasons: Iterable[Any] | None = None,
        policy_version: str | None = None,
    ) -> Dict[str, Any]:
        """Add a return to the worklist. Idempotent per pending return+environment."""
        self._ensure_schema()
        return_id = str(return_id or "").strip()
        if not return_id:
            raise ValueError("return_id is required to enqueue a review")
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM reviews WHERE return_id=? AND environment=? AND state='pending'",
                (return_id, environment),
            ).fetchone()
            if existing is not None:
                return {"id": int(existing["id"]), "created": False, "return_id": return_id, "environment": environment}
            cursor = conn.execute(
                "INSERT INTO reviews (return_id, environment, state, action, risk_probability, "
                "expected_cost, reasons, policy_version, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    return_id,
                    environment,
                    "pending",
                    action,
                    None if risk_probability is None else float(risk_probability),
                    None if expected_cost is None else float(expected_cost),
                    json.dumps(list(reasons or []), default=str),
                    policy_version,
                    time.time(),
                ),
            )
            new_id = int(cursor.lastrowid)
        return {"id": new_id, "created": True, "return_id": return_id, "environment": environment}

    def list_reviews(
        self,
        state: str = "pending",
        environment: str = "production",
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        self._ensure_schema()
        limit = max(1, min(int(limit), 5000))
        offset = max(0, int(offset))
        where = ["environment=?"]
        params: List[Any] = [environment]
        if state in ("pending", "resolved"):
            where.append("state=?")
            params.append(state)
        clause = " AND ".join(where)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT id, return_id, environment, state, action, risk_probability, "
                f"expected_cost, reasons, policy_version, created_at, resolved_at, "
                f"resolution, actor, note FROM reviews WHERE {clause} "
                f"ORDER BY state='pending' DESC, expected_cost IS NULL, expected_cost DESC, id ASC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        out = []
        for row in rows:
            record = dict(row)
            try:
                record["reasons"] = json.loads(record["reasons"] or "[]")
            except (TypeError, ValueError):
                record["reasons"] = []
            out.append(record)
        return out

    def count_reviews(self, state: str = "pending", environment: str = "production") -> int:
        self._ensure_schema()
        if state in ("pending", "resolved"):
            sql = "SELECT COUNT(*) FROM reviews WHERE environment=? AND state=?"
            params: tuple = (environment, state)
        else:
            sql = "SELECT COUNT(*) FROM reviews WHERE environment=?"
            params = (environment,)
        with self._connect() as conn:
            return int(conn.execute(sql, params).fetchone()[0])

    def get_review(self, review_id: int, environment: str = "production") -> Optional[Dict[str, Any]]:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM reviews WHERE id=? AND environment=?", (int(review_id), environment)
            ).fetchone()
        if row is None:
            return None
        record = dict(row)
        try:
            record["reasons"] = json.loads(record["reasons"] or "[]")
        except (TypeError, ValueError):
            record["reasons"] = []
        return record

    def resolve_review(
        self,
        review_id: int,
        resolution: str,
        actor: str | None = None,
        note: str | None = None,
        environment: str = "production",
    ) -> Optional[Dict[str, Any]]:
        """Mark a pending review resolved. Returns the row, or ``None`` if absent."""
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM reviews WHERE id=? AND environment=? AND state='pending'",
                (int(review_id), environment),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE reviews SET state='resolved', resolved_at=?, resolution=?, actor=?, note=? "
                "WHERE id=? AND environment=?",
                (time.time(), str(resolution), actor, note, int(review_id), environment),
            )
        return self.get_review(review_id, environment)

    # --------------------------------------------------------------- webhooks

    def log_delivery(
        self,
        event_type: str,
        url: str,
        payload: str,
        status: str,
        attempts: int,
        signature: str | None = None,
        last_error: str | None = None,
        environment: str = "production",
    ) -> int:
        self._ensure_schema()
        now = time.time()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO webhook_deliveries (event_type, url, environment, payload, "
                "signature, status, attempts, last_error, created_at, delivered_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    str(event_type),
                    str(url),
                    environment,
                    str(payload),
                    signature,
                    str(status),
                    int(attempts),
                    last_error,
                    now,
                    now if status == "delivered" else None,
                ),
            )
            return int(cursor.lastrowid)

    def list_deliveries(self, limit: int = 100, offset: int = 0, environment: str = "production") -> List[Dict[str, Any]]:
        self._ensure_schema()
        limit = max(1, min(int(limit), 5000))
        offset = max(0, int(offset))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, event_type, url, environment, signature, status, attempts, "
                "last_error, created_at, delivered_at FROM webhook_deliveries "
                "WHERE environment=? ORDER BY id DESC LIMIT ? OFFSET ?",
                (environment, limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def count_deliveries(self, environment: str = "production") -> int:
        self._ensure_schema()
        with self._connect() as conn:
            return int(conn.execute(
                "SELECT COUNT(*) FROM webhook_deliveries WHERE environment=?", (environment,)
            ).fetchone()[0])

    # ------------------------------------------------------------------ misc

    def stats(self) -> Dict[str, Any]:
        self._ensure_schema()
        with self._connect() as conn:
            events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            reviews = conn.execute("SELECT COUNT(*) FROM reviews WHERE state='pending'").fetchone()[0]
            deliveries = conn.execute("SELECT COUNT(*) FROM webhook_deliveries").fetchone()[0]
        return {
            "db_path": str(self.db_path),
            "events": int(events),
            "pending_reviews": int(reviews),
            "webhook_deliveries": int(deliveries),
        }


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


_store: Optional[DecisionStore] = None
_store_lock = threading.Lock()


def get_decision_store(db_path: str | Path | None = None) -> DecisionStore:
    """Process-wide singleton decision store."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = DecisionStore(db_path)
    return _store
