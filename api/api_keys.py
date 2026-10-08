"""Scoped API keys and request environments for the ReturnShield decision API.

The Stage 4 guard shipped a single all-or-nothing key. A real integration needs
per-resource scopes (so a checkout service can score but not resolve reviews) and
an environment flag (so sandbox traffic cannot move production thresholds or
scores). Both are opt-in and both preserve the legacy single-key behaviour.

Configuration, in priority order:

``RETURNSHIELD_API_KEYS``
    JSON object mapping the secret to its grant. Two accepted shapes::

        {"k1": ["score:write", "events:write"]}
        {"k2": {"scopes": ["admin"], "environment": "sandbox", "id": "local-ci"}}

``RETURNSHIELD_API_KEY``
    The legacy single key. When no scoped keys are configured it is granted
    every scope in the environment named by ``RETURNSHIELD_ENVIRONMENT``
    (default ``production``), which is exactly its old behaviour.

With neither set, authentication is disabled and every request is treated as
production, matching the loopback demo default.
"""

from __future__ import annotations

import hmac
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

#: Every scope the API understands. ``admin`` implies all of them.
ALL_SCOPES: FrozenSet[str] = frozenset({
    "score:write",
    "events:read",
    "events:write",
    "decisions:write",
    "reviews:read",
    "reviews:write",
    "webhooks:read",
    "admin",
})

ENVIRONMENTS: Tuple[str, ...] = ("production", "sandbox")

DEFAULT_ENVIRONMENT = "production"

#: ``(method, path-prefix, scope)``. A request matches when its path equals the
#: prefix or continues past it with ``/``. The longest matching prefix wins, so
#: ``/api/v1/decisions/score`` is checked before ``/api/v1/decisions``.
SCOPE_ROUTES: Tuple[Tuple[str, str, str], ...] = (
    ("POST", "/api/v1/events/backfill", "events:write"),
    ("POST", "/api/v1/events", "events:write"),
    ("GET", "/api/v1/events", "events:read"),
    ("POST", "/api/v1/decisions/score", "score:write"),
    ("POST", "/api/v1/decisions", "decisions:write"),
    ("POST", "/api/v1/score", "score:write"),
    ("POST", "/api/v1/batch_score", "score:write"),
    ("GET", "/api/v1/reviews", "reviews:read"),
    ("POST", "/api/v1/reviews", "reviews:write"),
    ("GET", "/api/v1/webhooks/deliveries", "webhooks:read"),
    ("POST", "/api/v1/sandbox/test-webhook", "admin"),
    # Moving the operating point is the most consequential write in the API:
    # every later score uses it. Reads stay broadly available, mutations are admin.
    ("POST", "/api/v1/policy/versions", "admin"),
    ("POST", "/api/v1/policy/activate", "admin"),
    ("POST", "/api/v1/policy/rollback", "admin"),
    ("POST", "/api/v1/policy/preview", "admin"),
    ("POST", "/api/v1/policy/live/set", "admin"),
    ("GET", "/api/v1/policy/live", None),
    ("POST", "/api/v1/policy/live/revert", "admin"),
)


@dataclass(frozen=True)
class KeyRecord:
    """One resolved credential.

    ``secret`` is carried so lookups can compare in constant time; it is never
    included in :func:`status` output.
    """

    secret: str
    key_id: str
    scopes: FrozenSet[str]
    environment: str = DEFAULT_ENVIRONMENT
    source: str = "legacy"

    def permits(self, scope: Optional[str]) -> bool:
        if not scope:
            return True
        return "admin" in self.scopes or scope in self.scopes


def _clean_environment(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in ENVIRONMENTS else DEFAULT_ENVIRONMENT


def _parse_record(secret: str, grant: Any, index: int) -> Optional[KeyRecord]:
    if isinstance(grant, list):
        scopes = {str(s).strip() for s in grant if str(s).strip()}
        return KeyRecord(secret, f"key-{index}", frozenset(scopes) or frozenset({"admin"}), DEFAULT_ENVIRONMENT, "scoped")
    if isinstance(grant, dict):
        scopes = {str(s).strip() for s in (grant.get("scopes") or []) if str(s).strip()}
        environment = _clean_environment(grant.get("environment"))
        key_id = str(grant.get("id") or f"key-{index}")
        return KeyRecord(secret, key_id, frozenset(scopes) or frozenset({"admin"}), environment, "scoped")
    return None


def load_keys() -> List[KeyRecord]:
    """Parse the configured credentials. Never raises on bad input."""
    records: List[KeyRecord] = []
    raw = os.getenv("RETURNSHIELD_API_KEYS", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            for index, (secret, grant) in enumerate(parsed.items()):
                record = _parse_record(str(secret).strip(), grant, index)
                if record is not None and record.secret:
                    records.append(record)
    if records:
        return records

    legacy = os.getenv("RETURNSHIELD_API_KEY", "").strip()
    if legacy:
        environment = _clean_environment(os.getenv("RETURNSHIELD_ENVIRONMENT", ""))
        records.append(KeyRecord(legacy, "legacy", ALL_SCOPES, environment, "legacy"))
    return records


def configured() -> bool:
    """True when at least one credential must be presented."""
    return bool(load_keys())


def resolve(presented: Optional[str]) -> Optional[KeyRecord]:
    """Find the record for a presented secret, or ``None``.

    Compares against every configured secret with :func:`hmac.compare_digest` so
    a wrong key cannot be narrowed down by response timing.
    """
    if not presented:
        return None
    match: Optional[KeyRecord] = None
    for record in load_keys():
        if hmac.compare_digest(str(presented), record.secret):
            match = record
    return match


def required_scope(method: str, path: str) -> Optional[str]:
    """The scope an endpoint demands, or ``None`` when auth alone suffices."""
    normalized = path.rstrip("/") or "/"
    method = (method or "").upper()
    best: Optional[Tuple[int, str]] = None
    for route_method, prefix, scope in SCOPE_ROUTES:
        if route_method != method:
            continue
        if normalized == prefix or normalized.startswith(prefix + "/"):
            if best is None or len(prefix) > best[0]:
                best = (len(prefix), scope)
    return best[1] if best else None


def status() -> Dict[str, Any]:
    """Describe the scoped-key configuration for ``/api/v1/meta`` (no secrets)."""
    records = load_keys()
    return {
        "scoped_keys_configured": bool(records),
        "key_count": len(records),
        "keys": [
            {
                "key_id": r.key_id,
                "environment": r.environment,
                "scopes": sorted(r.scopes),
                "source": r.source,
            }
            for r in records
        ],
        "environments": list(ENVIRONMENTS),
        "default_environment": DEFAULT_ENVIRONMENT,
        "known_scopes": sorted(ALL_SCOPES),
        "required_scopes": [
            {"method": m, "path": p, "scope": s} for m, p, s in SCOPE_ROUTES
        ],
    }
