"""Signed outbound webhooks.

Every serious fraud platform signs its callbacks so a receiver can prove a
payload came from the platform and was not replayed. The scheme here is the
widely used one: sign ``<timestamp>.<body>`` with HMAC-SHA256 and send the digest
as ``sha256=<hex>`` alongside the timestamp. A receiver recomputes it with
:func:`verify` and rejects anything outside a small clock-skew window.

Endpoints come from the environment (production and sandbox separately) or from
:func:`register_endpoint` at runtime, which is what tests use. Delivery retries
with exponential backoff and every attempt is written to the decision store, so a
failed callback is visible instead of lost.

No third-party HTTP client is used: the transport is a tiny callable over
``urllib`` that tests replace with a stub.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from api import decision_store

#: Large enough for a slow receiver, small enough that a hung endpoint cannot
#: stall the request thread indefinitely.
DEFAULT_TIMEOUT = 5.0
MAX_ATTEMPTS = 3
SIGNATURE_HEADER = "X-ReturnShield-Signature"
TIMESTAMP_HEADER = "X-ReturnShield-Timestamp"
EVENT_HEADER = "X-ReturnShield-Event"
DELIVERY_HEADER = "X-ReturnShield-Delivery"
SIGNATURE_TOLERANCE_SECONDS = 300


@dataclass(frozen=True)
class WebhookEndpoint:
    """A destination for outbound callbacks."""

    url: str
    secret: str
    environment: str = "production"
    label: str = "default"


#: Transport contract: ``transport(url, body: bytes, headers: dict) -> int``
#: returning an HTTP status. Raising is treated as a failed attempt.
Transport = Callable[[str, bytes, Dict[str, str]], int]


def _urllib_transport(url: str, body: bytes, headers: Dict[str, str]) -> int:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT) as response:  # nosec B310 - url is operator-configured
        return int(getattr(response, "status", 0) or 0)


#: Replaced by tests to avoid real sockets.
_transport: Transport = _urllib_transport

_registry: Dict[str, List[WebhookEndpoint]] = {}
_registry_lock = threading.Lock()


def register_endpoint(
    url: str,
    secret: str,
    environment: str = "production",
    label: str = "runtime",
) -> WebhookEndpoint:
    """Add a destination at runtime (used by tests and local demos)."""
    endpoint = WebhookEndpoint(url=str(url), secret=str(secret), environment=str(environment), label=str(label))
    with _registry_lock:
        _registry.setdefault(endpoint.environment, []).append(endpoint)
    return endpoint


def clear_endpoints() -> None:
    with _registry_lock:
        _registry.clear()


def _env_endpoints(environment: str) -> List[WebhookEndpoint]:
    prefix = "RETURNSHIELD_WEBHOOK" if environment == "production" else "RETURNSHIELD_SANDBOX_WEBHOOK"
    url = os.getenv(f"{prefix}_URL", "").strip()
    secret = os.getenv(f"{prefix}_SECRET", "").strip()
    if not url:
        return []
    return [WebhookEndpoint(url, secret, environment, "env")]


def endpoints(environment: str = "production") -> List[WebhookEndpoint]:
    """All destinations for an environment: runtime registrations plus env vars."""
    with _registry_lock:
        registered = list(_registry.get(environment, []))
    return registered + _env_endpoints(environment)


# --------------------------------------------------------------------- signing


def sign(secret: str, body: bytes, timestamp: int) -> str:
    """``sha256=<hex>`` over ``<timestamp>.<body>``."""
    payload = str(int(timestamp)).encode() + b"." + bytes(body)
    digest = hmac.new(str(secret).encode(), payload, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify(
    secret: str,
    body: bytes,
    signature: Optional[str],
    timestamp: Optional[Any],
    tolerance: int = SIGNATURE_TOLERANCE_SECONDS,
    now: Optional[float] = None,
) -> bool:
    """Validate a signature, including the replay window.

    Returns ``False`` for a missing/malformed signature, a stale timestamp, or a
    digest mismatch. The comparison is constant time.
    """
    if not signature or timestamp is None:
        return False
    try:
        sent_at = int(str(timestamp).strip())
    except (TypeError, ValueError):
        return False
    current = time.time() if now is None else now
    if abs(current - sent_at) > float(tolerance):
        return False
    expected = sign(secret, body, sent_at)
    return hmac.compare_digest(str(signature).strip(), expected)


# -------------------------------------------------------------------- delivery


def build_payload(event_type: str, data: Dict[str, Any], environment: str) -> Dict[str, Any]:
    return {
        "id": uuid.uuid4().hex,
        "type": str(event_type),
        "environment": environment,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": data or {},
    }


def deliver(
    event_type: str,
    data: Dict[str, Any],
    environment: str = "production",
    transport: Optional[Transport] = None,
    store: Optional[decision_store.DecisionStore] = None,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = MAX_ATTEMPTS,
) -> List[Dict[str, Any]]:
    """POST a signed payload to every endpoint for the environment.

    Returns one summary per endpoint. Nothing is attempted when no endpoint is
    configured, so the model path never pays for an unused webhook.
    """
    destinations = endpoints(environment)
    if not destinations:
        return []
    caller = transport or _transport
    log = store or decision_store.get_decision_store()
    summaries: List[Dict[str, Any]] = []

    for endpoint in destinations:
        payload = build_payload(event_type, data, environment)
        body = json.dumps(payload, default=str).encode()
        timestamp = int(time.time())
        signature = sign(endpoint.secret, body, timestamp)
        base_headers = {
            "Content-Type": "application/json",
            "User-Agent": "ReturnShield-Webhooks/1.0",
            EVENT_HEADER: str(event_type),
            TIMESTAMP_HEADER: str(timestamp),
            SIGNATURE_HEADER: signature,
        }

        status = "failed"
        attempts = 0
        last_error: Optional[str] = None
        for attempt in range(1, max(1, int(max_attempts)) + 1):
            attempts = attempt
            headers = dict(base_headers)
            headers[DELIVERY_HEADER] = f"{payload['id']}:{attempt}"
            try:
                code = int(caller(endpoint.url, body, headers))
                if 200 <= code < 300:
                    status = "delivered"
                    last_error = None
                    break
                last_error = f"HTTP {code}"
            except Exception as exc:  # a broken receiver must not break scoring
                last_error = f"{type(exc).__name__}: {exc}"[:300]
            if attempt < max_attempts:
                sleep(min(4.0, 0.5 * (2 ** (attempt - 1))))

        delivery_id = log.log_delivery(
            event_type=event_type,
            url=endpoint.url,
            payload=body.decode("utf-8", "replace"),
            status=status,
            attempts=attempts,
            signature=signature,
            last_error=last_error,
            environment=environment,
        )
        summaries.append({
            "delivery_id": delivery_id,
            "url": endpoint.url,
            "label": endpoint.label,
            "event_type": event_type,
            "status": status,
            "attempts": attempts,
            "last_error": last_error,
            "signature": signature,
        })
    return summaries
