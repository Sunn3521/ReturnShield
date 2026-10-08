# ReturnShield Decision API

The integration contract: what to send, what you get back, and how to prove a
callback came from us. Every endpoint below is implemented and covered by
`tests/test_decision_api.py` and `tests/test_sdk_contract.py`.

The OpenAPI document at `GET /openapi.json` (interactive at `/docs`) is the
source of truth for request/response schemas. A test asserts the SDK only calls
paths that exist there, so this document and the server cannot silently drift.

---

## 1. Ten-minute quickstart

```bash
pip install -r requirements.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Then, in another shell:

```bash
# 1. Confirm it is up and see what you can call.
curl -s localhost:8000/api/v1/health | python -m json.tool
curl -s localhost:8000/api/v1/meta  | python -m json.tool

# 2. Score one return and get a full decision object back.
curl -s -X POST localhost:8000/api/v1/decisions/score \
  -H 'Content-Type: application/json' \
  -d '{"return_id":"R999001","order_id":"O888001","customer_id":"C00123",
       "order_value":12500.0,"product_price":12500.0}' | python -m json.tool

# 3. Score something worth a human's time. Only `review` and `block` queue a
#    worklist row, so a moderate return comes back with review_id = null.
REVIEW_ID=$(curl -s -X POST localhost:8000/api/v1/decisions/score \
  -H 'Content-Type: application/json' \
  -d '{"return_id":"R999002","order_id":"O888002","customer_id":"C00456",
       "order_value":50000.0,"product_price":50000.0,
       "returns_30d":8,"returns_90d":14,"velocity_24h":7,"velocity_7d":12,
       "device_linked_accounts":9,"address_linked_accounts":8,
       "same_product_returns_90d":6,"refund_amount_90d":300000.0,
       "hours_to_return":0.4,"customer_account_age_days":1}' \
  | python -c 'import json,sys; print(json.load(sys.stdin)["review_id"])')
echo "queued review $REVIEW_ID"

# Work the queue, ranked by expected cost so the top row is the priciest mistake.
curl -s "localhost:8000/api/v1/reviews" | python -m json.tool
curl -s -X POST "localhost:8000/api/v1/reviews/$REVIEW_ID/resolve" \
  -H 'Content-Type: application/json' \
  -d '{"resolution":"confirmed_abuse","actor":"analyst@example.com"}'

# 4. Tell the system what actually happened. This closes the learning loop.
curl -s -X POST localhost:8000/api/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{"return_id":"R999001","outcome":"chargeback","actor":"analyst@example.com"}'
```

Or with the SDK, which is the same four calls:

```python
from sdk.returnshield import ReturnShieldClient, default_payload

with ReturnShieldClient() as rs:                      # open loopback demo
    decision = rs.score_decision(default_payload("R999001"))
    if decision.action in {"review", "block"}:
        rs.resolve_review(decision.review_id, "confirmed_abuse", actor="me")
    rs.record_decision("R999001", "chargeback", actor="me")
```

Point it at a different host or credential with
`ReturnShieldClient(base_url=..., api_key=..., environment="sandbox")`.

---

## 2. Authentication and scopes

All keys are opt-in. With no key configured the API is an open loopback demo.

| Variable | Effect |
|---|---|
| `RETURNSHIELD_API_KEY` | Legacy single key. Gets **every** scope, in the environment named by `RETURNSHIELD_ENVIRONMENT` (default `production`). |
| `RETURNSHIELD_API_KEYS` | Scoped keys. JSON mapping secret → grant, in either shape: `{"k": ["score:write"]}` or `{"k": {"scopes": ["admin"], "environment": "sandbox", "id": "ci"}}` |
| `RETURNSHIELD_ENVIRONMENT` | Environment for the legacy key. |
| `RETURNSHIELD_RATE_LIMIT` | Requests/minute per client. `0` (default) disables. |

Send the key as `X-ReturnShield-Key: <secret>` or `Authorization: Bearer <secret>`.
Never a query parameter — it would land in access logs.

| Scope | Grants |
|---|---|
| `events:write` | `POST /events`, `POST /events/backfill` |
| `events:read` | `GET /events` |
| `score:write` | `POST /score`, `POST /batch_score`, `POST /decisions/score` |
| `decisions:write` | `POST /decisions` |
| `reviews:read` | `GET /reviews` |
| `reviews:write` | `POST /reviews/{id}/resolve` |
| `webhooks:read` | `GET /webhooks/deliveries` |
| `admin` | Everything, including `POST /sandbox/test-webhook` |

A request matching no route still needs a *valid* key, just no particular
scope. `/`, `/health`, `/api/v1/health`, and `/api/v1/meta` stay public so
probes and service discovery never need a credential.

Paths are matched by longest prefix, so `POST /api/v1/decisions/score` is
checked against `score:write`, not `decisions:write`.

---

## 3. Environments

`production` and `sandbox` are isolated stores: an event, score, review, or
delivery in one never becomes visible in the other.

A scoped key's own environment always wins. An `X-ReturnShield-Environment`
header can select `sandbox` only when there is no scoped key — a header can
never move a credential's environment. This is the property that makes it safe
to point a test rig at a shared deployment.

Sandbox scores never enter the hindsight recommendation sample, so staging
traffic cannot shift a production threshold.

---

## 4. Endpoints

### `POST /api/v1/events` — ingest lifecycle events (202)

```json
{"events": [{"event_id": "evt_0001", "kind": "return",
             "return_id": "R999001", "customer_id": "C00123"}],
 "backfill": false}
```

Idempotent on `event_id` **within an environment**: resubmitting a batch
returns `duplicates`, not a second copy. Extra fields on an event are preserved
verbatim, so a merchant can store its own payload without a schema change.
Response: `{"accepted": n, "duplicates": n, "rejected": n, "errors": [...]}`.

`POST /api/v1/events/backfill` is the same path flagged as historical ingest —
it runs the identical feature code, so backfilled and live events are
indistinguishable downstream. Up to 5,000 events per request.

### `POST /api/v1/decisions/score` — the decision object

Returns the four-way action, the ranked reasons behind it, and the exact
operating point that produced it:

```json
{
  "request_id": "9f2c...", "return_id": "R999001",
  "environment": "production",
  "action": "review", "decision": "MANUAL_REVIEW",
  "score": 0.9012, "score_display": "90.1%",
  "reasons": [{"feature": "num__return_rate_90d", "contribution": 0.61}],
  "reason_summary": ["High historical return rate"],
  "merchant_loss_estimate": 9675.0, "expected_cost": 9675.0,
  "policy_version": "T1=0.120;T2=0.680",
  "model_version": "...",
  "review_id": 12, "review_created": true,
  "webhooks": [], "latency_ms": 8.4
}
```

`action` is one of `allow | warn | review | block` (`allow` below the verify
threshold, `warn` = the approve-with-warning band). A `review` or `block`
queues a worklist row and fires `decision.review_required`. The legacy
`POST /api/v1/score` still exists unchanged for existing consumers.

### `POST /api/v1/decisions` — the label that closes the loop

```json
{"return_id": "R999001", "outcome": "chargeback",
 "actor": "analyst@example.com", "risk_probability": 0.9,
 "merchant_loss": 12500, "note": "manager approved"}
```

Accepted outcomes are listed in `/api/v1/meta` under
`decision_api.outcomes`. Resolves any open call for that return immediately,
so the recommendation engine reflects it on the next request with **no offline
job**. A chargeback counts as abusive; an unknown outcome is a 400.

Outcome-only feedback is recorded but cannot be joined to a score, so it does
not move the recommendation sample — carry `risk_probability` when you have it.

### `GET /api/v1/reviews` — the worklist

`?state=pending|resolved|all`, paginated, **ranked by expected cost descending**
so the first row is the most expensive thing to get wrong. Each row carries the
top reasons and the policy version it was queued under.

`POST /api/v1/reviews/{id}/resolve` takes
`{"resolution": "...", "actor": "...", "note": "..."}` and fires
`review.resolved`. Resolving twice is a 404 by design — a row is pending once.

### `GET /api/v1/webhooks/deliveries`

Every outbound attempt for this environment: endpoint, status, attempt count,
and final state. A failed callback is visible here instead of lost.

### `POST /api/v1/sandbox/test-webhook`

Sends a signed `ping` to the sandbox destinations and reports what happened.

---

## 5. Webhooks

Configure per environment:

| Variable | Purpose |
|---|---|
| `RETURNSHIELD_WEBHOOK_URL` / `RETURNSHIELD_WEBHOOK_SECRET` | production |
| `RETURNSHIELD_SANDBOX_WEBHOOK_URL` / `RETURNSHIELD_SANDBOX_WEBHOOK_SECRET` | sandbox |

Emitted types: `decision.review_required`, `review.resolved`,
`decision.recorded`, `ping`.

**Signature.** `sha256=<hex>` over `<timestamp>.<body>` with HMAC-SHA256, sent
as `X-ReturnShield-Signature` alongside `X-ReturnShield-Timestamp`, with
`X-ReturnShield-Event` and `X-ReturnShield-Delivery`. Reject anything outside a
300-second clock-skew window, and compare in constant time.

Verify in three lines:

```python
from api import webhooks
ok = webhooks.verify(secret, raw_body, headers["X-ReturnShield-Signature"],
                     headers["X-ReturnShield-Timestamp"])
```

Delivery retries with exponential backoff (3 attempts, 5s timeout each) and
records every attempt. Verify on the **raw bytes** of the request body —
re-serialising parsed JSON changes key order and breaks the digest.

---

## 6. Errors

Every failure, however it was raised, answers with the same envelope:

```json
{"error": {"code": "bad_request",
           "message": "Unknown outcome 'maybe'. Use one of: [...]",
           "detail": "..."}}
```

Top-level `detail` is retained for compatibility with FastAPI's conventional
shape. Codes: `bad_request`, `unauthorized`, `forbidden`, `not_found`,
`validation_error`, `rate_limited`, `internal_error`.

| Status | Meaning |
|---|---|
| 401 | No/invalid key |
| 403 | Valid key, insufficient scope — `detail.required_scope` says which |
| 404 | No such pending review in this environment |
| 422 | Body failed schema validation — `detail` lists the fields |
| 429 | Rate limited — read `Retry-After` and `X-RateLimit-Remaining` |

The SDK raises `ReturnShieldError` with `.status_code`, `.code`, `.message`,
`.detail`, and a `.retryable` hint.

---

## 7. Known limits

These are honest, not aspirational:

- **Rate limiting and the limiter state are per process.** Behind multiple
  uvicorn workers each worker has its own window. The Docker image runs
  `--workers 4`; use an external limiter if the quota must be global.
- **Webhooks are dispatched inline in the request** with retries. A slow
  receiver adds latency to the scoring call; the 5s timeout bounds it, but a
  queue/worker split is the next step.
- **`register_endpoint` is in-process**, so runtime-registered destinations do
  not propagate across workers — configure via environment variables in
  production.
- No delivery signing key rotation yet; rotate by changing the secret and
  redeploying.
