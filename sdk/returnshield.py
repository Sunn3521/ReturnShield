"""ReturnShield Decision API - Python client.

One client, one surface. This is deliberately thin: it authenticates, routes,
parses the shared error envelope into an exception, and hands back plain
dictionaries. It does not retry, cache, or re-implement server logic, because
every one of those is a policy decision the API already owns.

The OpenAPI document at ``GET /openapi.json`` is the source of truth for the
contract. A test asserts that every path this client calls exists there, so a
rename on either side fails the suite instead of surfacing in production.

Usage::

    from sdk.returnshield import ReturnShieldClient

    with ReturnShieldClient(api_key="...") as rs:
        decision = rs.score_decision({...})
        if decision.action == "review":
            rs.resolve_review(decision.review_id, "confirmed_abuse", actor="me")
        rs.record_decision("R999001", outcome="abusive")

``httpx`` is already a project dependency; nothing new is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:  # pragma: no cover - exercised by the tests via the injected client
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

DEFAULT_BASE_URL = "http://localhost:8000"

API_VERSION = "2.1.0"

#: Scopes a key may carry. ``admin`` implies all of the others. Kept here so a
#: client can pre-flight its own credential instead of guessing.
SCOPES: tuple[str, ...] = (
    "score:write",
    "events:read",
    "events:write",
    "decisions:write",
    "reviews:read",
    "reviews:write",
    "webhooks:read",
    "admin",
)

#: Outcomes ``/api/v1/decisions`` accepts, split the way the API splits them.
#: These are literals rather than a live fetch on purpose - a client that has
#: to ask the server what is valid has already asked the wrong question - so
#: ``tests/test_sdk_contract.py`` asserts they match the server's sets exactly.
ABUSIVE_OUTCOMES: tuple[str, ...] = (
    "abuse",
    "abusive",
    "chargeback",
    "confirmed_abuse",
    "fraud",
)
BENIGN_OUTCOMES: tuple[str, ...] = (
    "approved",
    "benign",
    "legitimate",
    "no_fraud",
    "ok",
    "refunded",
)


class ReturnShieldError(RuntimeError):
    """Any 4xx/5xx the API produced, in one parseable shape.

    ``code`` comes from the envelope's ``error.code``; servers that answer with
    FastAPI's bare ``{"detail": ...}`` still produce a usable message.
    """

    def __init__(
        self,
        status_code: int,
        code: str = "http_error",
        message: str = "",
        detail: Any = None,
        body: Any = None,
    ) -> None:
        super().__init__(f"{status_code} {code}: {message}")
        self.status_code = int(status_code)
        self.code = code
        self.message = message
        self.detail = detail
        self.body = body

    @property
    def retryable(self) -> bool:
        """Transient by nature: the caller should back off and try again."""
        return self.status_code == 429 or self.status_code >= 500


@dataclass(frozen=True)
class DecisionScore:
    """The decision object returned by ``POST /api/v1/decisions/score``."""

    request_id: str
    return_id: str
    environment: str
    action: str
    decision: str
    score: float
    score_display: str = ""
    reasons: List[Dict[str, Any]] = field(default_factory=list)
    reason_summary: List[str] = field(default_factory=list)
    merchant_loss_estimate: float = 0.0
    expected_cost: float = 0.0
    policy_version: str = ""
    model_version: str = ""
    review_id: Optional[int] = None
    review_created: bool = False
    webhooks: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecisionScore":
        known = {f for f in cls.__dataclass_fields__ if f != "raw"}
        return cls(**{k: v for k, v in payload.items() if k in known}, raw=dict(payload))


def default_payload(return_id: str = "R999001", **overrides: Any) -> Dict[str, Any]:
    """A valid scoring payload with sane placeholders.

    Every field here is one the API accepts with a default, so the only values a
    merchant must supply are the identifiers and the money.
    """
    payload: Dict[str, Any] = {
        "return_id": return_id,
        "order_id": f"O{return_id[1:]}",
        "customer_id": "C00123",
        "order_value": 12500.0,
        "product_price": 12500.0,
    }
    payload.update(overrides)
    return payload


class ReturnShieldClient:
    """Blocking client for the ReturnShield Decision API.

    Parameters
    ----------
    base_url:
        Scheme, host, and port of the API process.
    api_key:
        Sent as ``X-ReturnShield-Key``. Omit it on an open loopback demo.
    environment:
        ``"sandbox"`` or ``"production"``. A scoped key's own environment always
        wins server-side, so this only matters for the unscoped demo key.
    client:
        Any object with ``request(method, url, headers=..., json=...,
        params=...)`` returning a response with ``status_code``, ``json()``, and
        ``text``. Tests inject an ``httpx`` ASGI client here; production code
        leaves it ``None`` and gets a real ``httpx.Client``.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        timeout: float = 30.0,
        client: Any = None,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.api_key = api_key
        self.environment = environment
        if client is None:
            if httpx is None:  # pragma: no cover
                raise RuntimeError("httpx is required to build a client: pip install httpx")
            client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._client = client

    # ------------------------------------------------------------- plumbing
    def __enter__(self) -> "ReturnShieldClient":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-ReturnShield-Key"] = self.api_key
        if self.environment:
            headers["X-ReturnShield-Environment"] = self.environment
        return headers

    @staticmethod
    def _raise(response: Any) -> None:
        status = int(getattr(response, "status_code", 0) or 0)
        try:
            body = response.json()
        except Exception:  # noqa: BLE001 - a non-JSON body is still an error
            body = None
        code, message, detail = "http_error", "", None
        if isinstance(body, dict):
            envelope = body.get("error")
            if isinstance(envelope, dict):
                code = str(envelope.get("code") or code)
                message = str(envelope.get("message") or "")
                detail = envelope.get("detail")
            if detail is None:
                detail = body.get("detail")
            if not message:
                message = str(detail) if detail else ""
        if not message:
            message = str(getattr(response, "text", "") or "").strip()[:500]
        raise ReturnShieldError(status, code, message, detail, body)

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        url = path if str(path).startswith(("http://", "https://")) else f"{self.base_url}{path}"
        response = self._client.request(
            method.upper(), url, headers=self.headers(), json=json, params=params
        )
        if int(getattr(response, "status_code", 0) or 0) >= 400:
            self._raise(response)
        try:
            return response.json()
        except Exception:  # noqa: BLE001 - 202 with no body is normal for ingest
            return {}

    # ----------------------------------------------------------- discovery
    def health(self) -> Dict[str, Any]:
        return self.request("GET", "/api/v1/health")

    def meta(self) -> Dict[str, Any]:
        return self.request("GET", "/api/v1/meta")

    def policy(self) -> Dict[str, Any]:
        return self.request("GET", "/api/v1/policy")

    # ------------------------------------------------------- policy control
    def policy_versions(self) -> Dict[str, Any]:
        """Every operating point this deployment has used, newest first."""
        return self.request("GET", "/api/v1/policy/versions")

    def preview_policy(
        self,
        verify_threshold: float,
        review_threshold: float,
        *,
        costs: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, Any]:
        """Price a candidate operating point against the live one, changing nothing."""
        return self.request(
            "POST",
            "/api/v1/policy/preview",
            json={
                "verify_threshold": verify_threshold,
                "review_threshold": review_threshold,
                "costs": dict(costs) if costs else None,
            },
        )

    def create_policy_version(
        self,
        verify_threshold: float,
        review_threshold: float,
        *,
        costs: Optional[Mapping[str, float]] = None,
        actor: Optional[str] = None,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a candidate operating point. Drafting changes nothing that scores."""
        return self.request(
            "POST",
            "/api/v1/policy/versions",
            json={
                "verify_threshold": verify_threshold,
                "review_threshold": review_threshold,
                "costs": dict(costs) if costs else None,
                "actor": actor,
                "note": note,
            },
        )

    def activate_policy(
        self, version: int, *, actor: Optional[str] = None, note: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make a recorded version live. Effective on the next request, no restart."""
        return self.request(
            "POST",
            "/api/v1/policy/activate",
            json={"version": int(version), "actor": actor, "note": note},
        )

    def rollback_policy(self, *, actor: Optional[str] = None, note: Optional[str] = None) -> Dict[str, Any]:
        """Return to the operating point that was live before the current one."""
        return self.request("POST", "/api/v1/policy/rollback", json={"actor": actor, "note": note})

    # ------------------------------------------------------------ hindsight
    def hindsight(self, recent: int = 25) -> Dict[str, Any]:
        """How the decisions the agent logged actually turned out."""
        return self.request("GET", "/api/v1/hindsight", params={"recent": recent})

    def recommendations(
        self,
        objective: str = "minimize_cost",
        target_precision: float = 0.75,
        steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Replay resolved decisions and get a recommended operating point."""
        params: Dict[str, Any] = {"objective": objective, "target_precision": target_precision}
        if steps is not None:
            params["steps"] = int(steps)
        return self.request("GET", "/api/v1/hindsight/recommendations", params=params)

    def return_history(self, return_id: str) -> Dict[str, Any]:
        """What was decided about one return before, and whether it was right."""
        return self.request("GET", f"/api/v1/hindsight/return/{return_id}")

    # ------------------------------------------------------------- scoring
    def score(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Legacy score endpoint: returns the raw scoring response."""
        return self.request("POST", "/api/v1/score", json=dict(payload))

    def score_decision(self, payload: Mapping[str, Any]) -> DecisionScore:
        """Score one return and get the full four-way decision object."""
        return DecisionScore.from_dict(
            self.request("POST", "/api/v1/decisions/score", json=dict(payload))
        )

    # -------------------------------------------------------------- events
    def ingest_events(
        self, events: Iterable[Mapping[str, Any]], backfill: bool = False
    ) -> Dict[str, Any]:
        """Submit lifecycle events. Idempotent on ``event_id``."""
        return self.request(
            "POST",
            "/api/v1/events",
            json={"events": [dict(e) for e in events], "backfill": bool(backfill)},
        )

    def backfill(self, events: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        """Historical ingest through the same feature path as live traffic."""
        return self.ingest_events(events, backfill=True)

    def list_events(
        self, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        return self.request(
            "GET", "/api/v1/events", params={"limit": limit, "offset": offset}
        )

    # ------------------------------------------------------------ feedback
    def record_decision(
        self,
        return_id: str,
        outcome: str,
        *,
        actor: Optional[str] = None,
        note: Optional[str] = None,
        decision: Optional[str] = None,
        risk_probability: Optional[float] = None,
        merchant_loss: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Report what actually happened - the label that closes the loop."""
        payload: Dict[str, Any] = {"return_id": return_id, "outcome": outcome}
        for key, value in (
            ("actor", actor),
            ("note", note),
            ("decision", decision),
            ("risk_probability", risk_probability),
            ("merchant_loss", merchant_loss),
            ("request_id", request_id),
        ):
            if value is not None:
                payload[key] = value
        return self.request("POST", "/api/v1/decisions", json=payload)

    # -------------------------------------------------------- review queue
    def reviews(
        self, state: str = "pending", limit: int = 50, offset: int = 0
    ) -> Dict[str, Any]:
        """The human worklist, ranked by expected cost, highest first."""
        return self.request(
            "GET",
            "/api/v1/reviews",
            params={"state": state, "limit": limit, "offset": offset},
        )

    def resolve_review(
        self,
        review_id: int,
        resolution: str,
        *,
        actor: Optional[str] = None,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.request(
            "POST",
            f"/api/v1/reviews/{int(review_id)}/resolve",
            json={"resolution": resolution, "actor": actor, "note": note},
        )

    # ------------------------------------------------------------- webhooks
    def deliveries(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        return self.request(
            "GET", "/api/v1/webhooks/deliveries", params={"limit": limit, "offset": offset}
        )

    def test_sandbox_webhook(self) -> Dict[str, Any]:
        return self.request("POST", "/api/v1/sandbox/test-webhook")
