"""API access control and rate limiting for the ReturnShield API.

Both features are opt-in through environment variables so the local dashboard
and the test suite keep working untouched:

``RETURNSHIELD_API_KEY``
    When set, every endpoint except :data:`PUBLIC_PATHS` requires the key, sent
    as ``X-ReturnShield-Key`` or ``Authorization: Bearer <key>``. Unset means
    no authentication, which is the correct default for a loopback demo.

``RETURNSHIELD_RATE_LIMIT``
    Requests per minute per client key. ``0`` (the default) disables limiting.

The limiter is a fixed window with a bounded deque per client, which is
deliberately simple and in-process: it protects a single-worker demo server
from a runaway client, not a multi-tenant deployment. That is stated plainly
in the ``/api/v1/meta`` payload so nobody mistakes it for a shared store.
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

#: Reachable without a key even when authentication is enabled: liveness probes
#: and service discovery must not need a credential.
PUBLIC_PATHS = frozenset({"/", "/health", "/api/v1/health", "/api/v1/meta", "/api/v1/policy/live"})

KEY_HEADER = "X-ReturnShield-Key"


def configured_api_key() -> Optional[str]:
    key = os.getenv("RETURNSHIELD_API_KEY", "").strip()
    return key or None


def rate_limit_per_minute() -> int:
    """Requests allowed per minute per client. Negative or unparseable means off."""
    raw = os.getenv("RETURNSHIELD_RATE_LIMIT", "0").strip()
    try:
        value = int(raw)
    except ValueError:
        return 0
    return max(0, value)


def extract_key(headers) -> Optional[str]:
    """Read the presented key from either supported header."""
    header_key = (headers.get(KEY_HEADER) or "").strip()
    if header_key:
        return header_key
    authorization = (headers.get("Authorization") or "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip() or None
    return None


def is_public_path(path: str) -> bool:
    normalized = path.rstrip("/") or "/"
    return normalized in PUBLIC_PATHS


def check_api_key(path: str, presented: Optional[str], expected: Optional[str]) -> bool:
    """Constant-time comparison so the key cannot be recovered by timing."""
    if not expected:
        return True
    if is_public_path(path):
        return True
    if not presented:
        return False
    return hmac.compare_digest(presented, expected)


class RateLimiter:
    """Fixed-window per-client limiter.

    One deque of request timestamps per client key, pruned on access. Memory is
    bounded by ``max_clients``; once that many distinct keys are in flight the
    least recently used key is evicted, which keeps a spoofed-key flood from
    turning into unbounded growth.
    """

    def __init__(self, per_minute: int, max_clients: int = 2048) -> None:
        self.per_minute = max(0, int(per_minute))
        self.max_clients = max(1, int(max_clients))
        self._windows: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.per_minute > 0

    def check(self, key: str, now: Optional[float] = None) -> Tuple[bool, int, float]:
        """Return ``(allowed, remaining, retry_after_seconds)``."""
        if not self.enabled:
            return True, self.per_minute, 0.0
        now = time.monotonic() if now is None else now
        window = 60.0
        cutoff = now - window
        with self._lock:
            hits = self._windows.setdefault(key, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.per_minute:
                retry_after = max(0.0, round(hits[0] + window - now, 3))
                return False, 0, retry_after
            hits.append(now)
            if len(self._windows) > self.max_clients:
                oldest = min(self._windows, key=lambda k: self._windows[k][0] if self._windows[k] else now)
                if oldest != key:
                    self._windows.pop(oldest, None)
            return True, max(0, self.per_minute - len(hits)), 0.0

    def reset(self) -> None:
        with self._lock:
            self._windows.clear()

    def tracked_clients(self) -> int:
        with self._lock:
            return len(self._windows)


def status() -> dict:
    """Surface the guard configuration in ``/api/v1/meta``."""
    key = configured_api_key()
    limit = rate_limit_per_minute()
    return {
        "auth_enabled": key is not None,
        "auth_header": KEY_HEADER,
        "public_paths": sorted(PUBLIC_PATHS),
        "rate_limit_per_minute": limit,
        "rate_limit_scope": "per-process" if limit else "disabled",
    }


def require_scope(request: Request, scope: str) -> None:
    """Raise 403 when the presented credentials do not permit ``scope``.

    Matches the middleware's contract exactly: when no credentials are
    configured the deployment is in open demo mode and every endpoint permits
    every caller; when credentials exist, admin-gated endpoints double-check
    here so the gate survives any future middleware reordering. Public paths
    are never gated.
    """
    if is_public_path(request.url.path):
        return
    if not configured_api_key():
        return
    record = _resolve(extract_key(request.headers))
    if record is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="A valid API key is required.")
    if not record.permits(scope):
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail=f"This endpoint requires the '{scope}' scope.")


def require_key(request: Request) -> None:
    """Raise 401 when no valid credential is presented and auth is configured."""
    if not configured_api_key():
        return
    record = _resolve(extract_key(request.headers))
    if record is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="A valid API key is required.")


def _resolve(presented: Optional[str]) -> Optional["KeyRecord"]:
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


