from __future__ import annotations

import json
import hashlib
import os
import functools
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
import pandas as pd
import numpy as np

from src.model import load_bundle, predict_bundle
from src.explain import top_features, concise_reasoning
from src.policy import Costs as PolicyCosts
from src.policy import evaluate_policy as evaluate_policy_point
from src.responder import generate_agent_response
from src.chat_agent import SEMANTIC_TRIGGER, ReturnShieldChatAgent, ChatContext
from src.local_model import status as local_model_status
from api.chatbot_api import ChatbotConfig, ReturnShieldChatbot
from api import memory as memory_store
from api import hindsight as hindsight_engine
from src.network import attach_infrastructure_ids
from api import api_guard
from api import api_keys
from api import decision_store
from api import webhooks

from .live_server import start_generator, stop_generator, get_status, list_events, generate_now, export_events_csv
import io

from src.paths import root as _project_root

ROOT = _project_root()
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

API_VERSION = "2.1.0"


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Built-in defensive demo stream starts automatically with the API server.
    start_generator(2.0)
    try:
        yield
    finally:
        stop_generator()


app = FastAPI(
    title="ReturnShield AI API",
    description="High-Throughput Cost-Sensitive Return Abuse Risk Scorer & Decision Agent",
    version=API_VERSION,
    lifespan=_lifespan,
)


def _cors_origins() -> list[str]:
    """Resolve allowed origins.

    A wildcard origin cannot legally be combined with credentialed requests, so
    the previous ``allow_origins=["*"] + allow_credentials=True`` pair was both
    invalid and unsafe. Defaults to the local dashboard only.
    """
    raw = os.getenv("RETURNSHIELD_CORS_ORIGINS", "").strip()
    if not raw:
        return [
            "http://localhost:8501",
            "http://127.0.0.1:8501",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    return [o.strip() for o in raw.split(",") if o.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


_rate_limiter = api_guard.RateLimiter(api_guard.rate_limit_per_minute())


def _client_key(request: Request) -> str:
    """Rate-limit bucket for a request.

    An authenticated caller gets its own bucket so one noisy client behind a
    shared address cannot exhaust another's quota.
    """
    presented = api_guard.extract_key(request.headers)
    if presented:
        return f"key:{hashlib.sha256(presented.encode()).hexdigest()[:16]}"
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


ENVIRONMENT_HEADER = "X-ReturnShield-Environment"


def _request_environment(request: Request, record: Any = None) -> str:
    """Which environment this request acts in.

    A scoped key's environment always wins. With no key configured (the
    loopback demo) an explicit header selects sandbox; anything else is
    production.
    """
    if record is not None:
        return record.environment
    header = (request.headers.get(ENVIRONMENT_HEADER) or "").strip().lower()
    return "sandbox" if header == "sandbox" else "production"


@app.middleware("http")
async def access_guard(request: Request, call_next):
    """Enforce the API key, then the rate limit. Both are off by default."""
    path = request.url.path
    if request.method == "OPTIONS":
        return await call_next(request)

    presented = api_guard.extract_key(request.headers)
    record = None
    if not api_guard.is_public_path(path):
        if api_keys.configured():
            record = api_keys.resolve(presented)
            if record is None:
                return _error(
                    401, "unauthorized",
                    f"A valid {api_guard.KEY_HEADER} header is required for this endpoint.",
                )
            request.state.key = record
            required = api_keys.required_scope(request.method, path)
            if not record.permits(required):
                return _error(
                    403, "insufficient_scope",
                    f"This endpoint requires the '{required}' scope.",
                    {"required_scope": required, "granted_scopes": sorted(record.scopes)},
                )
        else:
            expected = api_guard.configured_api_key()
            if not api_guard.check_api_key(path, presented, expected):
                return _error(
                    401, "unauthorized",
                    f"A valid {api_guard.KEY_HEADER} header is required for this endpoint.",
                )
            if expected:
                request.state.api_key_verified = True
    request.state.environment = _request_environment(request, record)

    limit = api_guard.rate_limit_per_minute()
    if limit and _rate_limiter.per_minute != limit:
        # Env is read per request so a limit can be tightened without a restart.
        _rate_limiter.per_minute = limit
    allowed, remaining, retry_after = _rate_limiter.check(_client_key(request))
    if not allowed:
        throttled = _error(
            429, "rate_limited",
            f"Rate limit of {limit} requests/minute exceeded.",
            {"retry_after_seconds": retry_after, "limit_per_minute": limit},
        )
        # A rejected caller needs the same budget headers as an accepted one,
        # plus Retry-After so it knows when to come back.
        throttled.headers["Retry-After"] = str(max(1, int(retry_after) or 1))
        throttled.headers["X-RateLimit-Limit"] = str(limit)
        throttled.headers["X-RateLimit-Remaining"] = "0"
        return throttled
    response = await call_next(request)
    if _rate_limiter.enabled:
        response.headers["X-RateLimit-Limit"] = str(_rate_limiter.per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
    return response


def _page(records: List[Dict[str, Any]], total: int, limit: int, offset: int) -> Dict[str, Any]:
    """One pagination envelope for every list endpoint.

    Clients can page with ``offset``/``limit`` and stop as soon as
    ``has_more`` is false, instead of guessing at totals.
    """
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), 5000))
    return {
        "data": records,
        "returned": len(records),
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(records) < int(total),
    }


def _error(status: int, code: str, message: str, detail: Any = None) -> JSONResponse:
    """One error shape for every failure so clients can rely on it."""
    body: Dict[str, Any] = {"error": {"code": code, "message": message}}
    if detail is not None:
        body["error"]["detail"] = detail
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    return _error(500, "internal_error", "Unexpected server error.", str(exc))


@app.exception_handler(ValueError)
async def _value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
    return _error(400, "bad_request", str(exc))


#: ``HTTPException`` never passed through :func:`_error`, so a raised 400/404
#: answered with FastAPI's bare ``{"detail": ...}`` while 401/403/429 answered
#: with the envelope. Clients (and the SDK) get one shape or the other depending
#: on *how* the failure was raised, which is not a contract anyone can code
#: against. These are folded onto the same envelope, with ``detail`` retained at
#: the top level so anything already reading it keeps working.
HTTP_ERROR_CODES: Dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    410: "gone",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    503: "service_unavailable",
}


async def _http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    message = detail if isinstance(detail, str) else " ".join(str(p.get("msg", p)) for p in detail) if isinstance(detail, list) else str(detail)
    body: Dict[str, Any] = {
        "detail": detail,
        "error": {
            "code": HTTP_ERROR_CODES.get(exc.status_code, "http_error"),
            "message": message,
        },
    }
    if detail is not None and not isinstance(detail, str):
        body["error"]["detail"] = detail
    return JSONResponse(status_code=exc.status_code, content=body, headers=getattr(exc, "headers", None))


@app.exception_handler(HTTPException)
async def _http_exception_router(request: Request, exc: HTTPException) -> JSONResponse:
    return await _http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """422s get the envelope too - a schema mismatch is still a client error."""
    detail = jsonable_encoder(exc.errors())
    body: Dict[str, Any] = {
        "detail": detail,
        "error": {
            "code": "validation_error",
            "message": "; ".join(str(e.get("msg", "invalid")) for e in detail) or "Request body failed validation.",
            "detail": detail,
        },
    }
    return JSONResponse(status_code=422, content=body)


bundle = None
policy = None
_bundle_lock = threading.Lock()


def get_bundle():
    """Load the model bundle once; the lock keeps concurrent first-calls safe."""
    global bundle, policy
    if bundle is None:
        with _bundle_lock:
            if bundle is None:
                bundle_path = MODELS / "model_bundle.joblib"
                policy_path = MODELS / "policy.json"
                if not bundle_path.exists():
                    raise RuntimeError("Model bundle not found. Please run `python run_pipeline.py` first.")
                loaded = load_bundle(str(bundle_path))
                with open(policy_path, "r", encoding="utf-8") as f:
                    loaded_policy = json.load(f)
                bundle, policy = loaded, loaded_policy
    return bundle, policy


class ReturnRequestPayload(BaseModel):
    return_id: str = Field(..., example="R999001")
    order_id: str = Field(..., example="O888001")
    customer_id: str = Field(..., example="C00123")
    product_category: str = Field("electronics", example="electronics")
    payment_method: str = Field("card", example="card")
    return_reason: str = Field("damaged", example="damaged")
    order_value: float = Field(..., example=12500.0)
    product_price: float = Field(..., example=12500.0)
    discount_pct: float = Field(0.0, example=0.0)
    customer_account_age_days: int = Field(120, example=120)
    orders_7d: int = Field(1, example=1)
    orders_30d: int = Field(3, example=3)
    orders_90d: int = Field(5, example=5)
    returns_7d: int = Field(0, example=0)
    returns_30d: int = Field(2, example=2)
    returns_90d: int = Field(4, example=4)
    refund_amount_30d: float = Field(4500.0, example=4500.0)
    refund_amount_90d: float = Field(18500.0, example=18500.0)
    hours_to_return: float = Field(3.5, example=3.5)
    same_product_returns_90d: int = Field(0, example=0)
    same_category_returns_90d: int = Field(2, example=2)
    velocity_24h: int = Field(1, example=1)
    velocity_7d: int = Field(2, example=2)
    device_linked_accounts: int = Field(3, example=3)
    address_linked_accounts: int = Field(2, example=2)
    device_return_rate_90d: float = Field(0.40, example=0.40)
    address_return_rate_90d: float = Field(0.35, example=0.35)


class ScoringResponse(BaseModel):
    return_id: str
    risk_probability: float
    risk_display: str
    decision: str
    merchant_loss_estimate: float
    top_signals: List[str]
    merchant_action_protocol: str
    customer_message: str
    latency_ms: float


class AgentChatPayload(BaseModel):
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)
    records: List[Dict[str, Any]] = Field(default_factory=list)
    report: Dict[str, Any] = Field(default_factory=dict)
    live_status: Dict[str, Any] | None = None
    session_id: str | None = None
    use_memory: bool = True


class AgentChatResponse(BaseModel):
    answer: str
    intent: str = ""
    confidence: float = 0.0
    data: List[Dict[str, Any]] = Field(default_factory=list)
    action: Dict[str, Any] | None = None
    action_result: str | None = None
    session_id: str | None = None
    memory: Dict[str, Any] = Field(default_factory=dict)


_chat_agent = ReturnShieldChatAgent()


def _chat_tool_handlers(payload_df: pd.DataFrame):
    def current_df():
        return payload_df.copy() if isinstance(payload_df, pd.DataFrame) else pd.DataFrame()

    def get_live_status_tool():
        return get_status()

    def operations_tool():
        df = current_df()
        if df.empty:
            return {"message": "No active return records are available."}
        risk = pd.to_numeric(df.get("risk_probability", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        decision = df.get("decision", pd.Series("", index=df.index)).astype(str)
        abusive = int(pd.to_numeric(df.get("abusive_return", pd.Series(0, index=df.index)), errors="coerce").fillna(0).sum())
        total = len(df)
        return {
            "total_returns": total,
            "auto_approved": int((decision == "AUTO_APPROVE").sum()),
            "verification": int((decision == "VERIFY").sum()),
            "manual_review": int((decision == "MANUAL_REVIEW").sum()),
            "high_risk": int((risk >= 0.75).sum()),
            "observed_abusive": abusive,
            "abusive_share": round(abusive / total, 6) if total else 0,
            "average_risk": round(float(risk.mean()), 6) if total else 0,
            "return_value_total": round(float(pd.to_numeric(df.get("return_value", pd.Series(0, index=df.index)), errors="coerce").fillna(0).sum()), 2),
        }

    def search_returns_tool(query: str, limit: int = 20):
        df = current_df()
        if df.empty:
            return {"message": "No active return records are available.", "_table": []}
        work = df.copy()
        if "risk_probability" in work.columns:
            work["risk_probability"] = pd.to_numeric(work["risk_probability"], errors="coerce").fillna(0)
            work = work.sort_values("risk_probability", ascending=False)
        q = (query or "").strip().lower()
        if q:
            mask = pd.Series(False, index=work.index)
            for c in ["return_id", "customer_id", "decision", "return_reason"]:
                if c in work.columns:
                    mask |= work[c].astype(str).str.lower().str.contains(q, regex=False, na=False)
            work = work[mask]
        cols = [c for c in ["return_id", "customer_id", "risk_probability", "decision", "order_value", "return_value", "return_reason", "generated_at"] if c in work.columns]
        return {"count": len(work), "_table": work[cols].head(max(1, min(limit, 50))).to_dict("records")}

    def inspect_return_tool(return_id: str):
        df = current_df()
        if df.empty:
            return {"message": "No active return records are available."}
        rid = str(return_id).upper()
        mask = df.get("return_id", pd.Series(dtype=str)).astype(str).str.upper() == rid
        match = df[mask]
        if match.empty:
            return {"message": f"Return {return_id} was not found in the active data."}
        row = match.iloc[0].to_dict()
        return {"return": row, "_table": [row], "_action": {"type": "open_investigator", "return_id": str(row.get("return_id", return_id))}}

    def search_customer_tool(customer_id: str, limit: int = 50):
        df = current_df()
        if df.empty or "customer_id" not in df.columns:
            return {"message": "No active customer records are available.", "_table": []}
        match = df[df.customer_id.astype(str).str.upper() == str(customer_id).upper()].copy()
        if "risk_probability" in match.columns:
            match["risk_probability"] = pd.to_numeric(match["risk_probability"], errors="coerce").fillna(0)
            match = match.sort_values("risk_probability", ascending=False)
        cols = [c for c in ["return_id", "customer_id", "risk_probability", "decision", "order_value", "return_value", "return_reason", "prediction_time"] if c in match.columns]
        return {"count": len(match), "_table": match[cols].head(max(1, min(limit, 100))).to_dict("records")}

    def coordinated_tool(limit: int = 50):
        df = current_df()
        if df.empty or "customer_id" not in df.columns:
            return {"message": "No active records are available for coordinated-account analysis.", "_table": []}
        rows = []
        for col in ["device_id", "address_id", "payment_fingerprint"]:
            if col not in df.columns:
                continue
            g = df.groupby(col, dropna=True).agg(accounts=("customer_id", "nunique"), returns=("return_id", "count")).reset_index()
            g = g[g.accounts >= 2].copy()
            if not g.empty:
                g["signal_type"] = col
                g["shared_identifier"] = g[col].astype(str)
                rows.extend(g[["signal_type", "shared_identifier", "accounts", "returns"]].to_dict("records"))
        rows = sorted(rows, key=lambda x: (x.get("accounts", 0), x.get("returns", 0)), reverse=True)[:max(1, min(limit, 100))]
        return {"count": len(rows), "_table": rows}

    def metrics_tool():
        return report_snapshot()

    def control_live_tool(operation: str, count: int = 1):
        try:
            if operation == "start":
                start_generator(4.0)
                return {"message": "Live generator started.", "status": get_status(), "_action": {"type": "start_live"}}
            if operation == "stop":
                stop_generator()
                return {"message": "Live generator stopped.", "status": get_status(), "_action": {"type": "stop_live"}}
            if operation == "generate":
                n = max(1, min(int(count), 1000))
                generate_now(n)
                return {"message": f"Generated {n} live transaction(s).", "status": get_status(), "_action": {"type": "generate", "count": n}}
        except Exception as exc:
            return {"error": str(exc)}
        return {"error": "Unsupported operation"}

    def report_snapshot():
        return report if isinstance(report, dict) else {}

    return {
        "get_live_status": get_live_status_tool,
        "get_operations_summary": operations_tool,
        "search_returns": search_returns_tool,
        "inspect_return": inspect_return_tool,
        "search_customer": search_customer_tool,
        "get_coordinated_accounts": coordinated_tool,
        "get_model_metrics": metrics_tool,
        "control_live_server": control_live_tool,
    }


def _context_from_dict(raw: Dict[str, Any]) -> ChatContext:
    ctx = ChatContext()
    ctx.last_return_id = raw.get("last_return_id")
    ctx.last_customer_id = raw.get("last_customer_id")
    ctx.last_intent = raw.get("last_intent")
    ctx.last_answer = raw.get("last_answer")
    history = raw.get("history") or []
    ctx.history = history if isinstance(history, list) else []
    return ctx


_live_snapshot_cache: Dict[str, Any] = {"at": 0.0, "records": []}
_LIVE_SNAPSHOT_TTL = 2.0


def _live_snapshot(limit: int = 10000) -> list[Dict[str, Any]]:
    """Cached view of the live event log.

    The chat endpoint asked for the whole event log on every message, which meant
    re-reading a file that grows continuously. A couple of seconds of staleness
    is irrelevant for a conversational agent.
    """
    now = time.monotonic()
    if now - _live_snapshot_cache["at"] < _LIVE_SNAPSHOT_TTL:
        return _live_snapshot_cache["records"]
    records, _total = list_events(limit=limit, offset=0)
    _live_snapshot_cache["records"] = records
    _live_snapshot_cache["at"] = now
    return records


def _remember_decisions(store, answer: str, data: List[Dict[str, Any]]) -> int:
    """Log the decisions that surfaced in an answer so they can be scored later."""
    logged = 0
    seen = set()
    for row in data or []:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("return_id") or "").strip()
        decision = str(row.get("decision") or "").strip().upper()
        if not rid or not decision or rid in seen:
            continue
        try:
            prob = float(row.get("risk_probability"))
        except (TypeError, ValueError):
            prob = None
        store.record_decision(rid, decision, prob, source="agent_chat")
        seen.add(rid)
        logged += 1
    return logged


@app.post("/api/v1/agent/chat", response_model=AgentChatResponse)
async def agent_chat(payload: AgentChatPayload):
    """Chat with ReturnShield, backed by tools plus persistent memory.

    The client may pass ``session_id`` to continue a conversation; the first call
    that omits one gets a fresh id back. History is kept server-side, so a browser
    refresh no longer wipes the agent's memory of the conversation.
    """
    ctx = _context_from_dict(payload.context or {})
    live_status = get_status()
    df = pd.DataFrame(payload.records or [])
    if live_status.get("running"):
        live_records = _live_snapshot()
        if live_records:
            df = pd.DataFrame(live_records)
    if df.empty:
        # Without records and without a running feed the agent would be blind;
        # fall back to the held-out evaluation set so it can still answer.
        df = _evaluation_frame()
    df = attach_infrastructure_ids(df, str(ROOT / "data" / "raw"))
    report = payload.report or {}

    store = memory_store.get_store() if payload.use_memory else None
    session_id = payload.session_id or (payload.context or {}).get("session_id")
    if payload.use_memory and not session_id:
        session_id = memory_store.new_session_id()

    history = list(ctx.history or [])
    hindsight: Dict[str, Any] = {}
    remembered: List[Dict[str, str]] = []
    if store is not None and session_id:
        remembered = store.get_history(session_id, limit=20)
        if remembered:
            history = (remembered + history)[-40:]
        elif history:
            # Seed server memory from what the client still had, so the first
            # server-side call after an upgrade keeps the existing thread.
            for turn in history[-20:]:
                store.append_message(session_id, str(turn.get("role", "user")), str(turn.get("content", "")))
        hindsight = memory_store.hindsight_context(ctx.last_return_id, store)

    ai_cfg = (payload.context or {}).get("ai_config") or {}
    cfg = ChatbotConfig.from_env(provider=ai_cfg.get("provider"), api_key=ai_cfg.get("api_key"), model=ai_cfg.get("model"), base_url=ai_cfg.get("base_url"))
    chatbot = ReturnShieldChatbot(cfg)
    tools = _chat_tool_handlers(df)

    def finish(answer: str, intent: str, confidence: float, records: List[Dict[str, Any]],
               action: Dict[str, Any] | None, action_result: str | None) -> AgentChatResponse:
        meta: Dict[str, Any] = {}
        if store is not None and session_id:
            store.append_message(session_id, "user", payload.message)
            store.append_message(session_id, "assistant", answer, {"intent": intent})
            logged = _remember_decisions(store, answer, records)
            meta = {
                "session_id": session_id,
                "recalled_turns": len(remembered) if store is not None and session_id else 0,
                "decisions_logged": logged,
                "hindsight": hindsight.get("summary"),
                "recommendation": _chat_recommendation(store),
            }
        return AgentChatResponse(
            answer=answer, intent=intent, confidence=confidence, data=records,
            action=action, action_result=action_result,
            session_id=session_id, memory=meta,
        )

    if cfg.enabled:
        try:
            ai = chatbot.chat(payload.message, history, df, report, live_status, tools)
            return finish(
                str(ai.get("answer") or "I couldn't produce a response."),
                "chatbot", 1.0, ai.get("data") or [], ai.get("action"), None,
            )
        except Exception as exc:
            # Fall through to the deterministic ReturnShield agent if the API is unavailable.
            fallback = _chat_agent.respond(payload.message, ctx, df, report, live_status)
            answer = str(fallback.get("answer") or "")
            answer += "\n\n_Chatbot API unavailable; used the ReturnShield local fallback agent._"
            data = fallback.get("data")
            records = []
            if isinstance(data, pd.DataFrame) and not data.empty:
                records = data.where(pd.notna(data), None).head(100).to_dict("records")
            return finish(
                answer, str(fallback.get("intent") or "fallback"),
                float(fallback.get("confidence") or 0), records,
                fallback.get("action"), f"AI API error: {exc}",
            )

    result = _chat_agent.respond(payload.message, ctx, df, report, live_status)
    data = result.get("data")
    records = []
    if isinstance(data, pd.DataFrame) and not data.empty:
        records = data.where(pd.notna(data), None).head(100).to_dict("records")
    return finish(
        str(result.get("answer") or "I could not process that request."),
        str(result.get("intent") or ""), float(result.get("confidence") or 0.0),
        records, result.get("action"),
        "OPENAI_API_KEY is not configured; using the local ReturnShield agent.",
    )


@app.get("/api/v1/meta")
async def meta():
    return {
        "service": "ReturnShield AI",
        "version": API_VERSION,
        "endpoints": [
            "/api/v1/health",
            "/api/v1/policy",
            "/api/v1/score",
            "/api/v1/batch_score",
            "/api/v1/returns",
            "/api/v1/returns/stats",
            "/api/v1/returns/start",
            "/api/v1/returns/stop",
            "/api/v1/returns/generate",
            "/api/v1/returns/export.csv",
            "/api/v1/agent/chat",
            "/api/v1/memory",
            "/api/v1/memory/{session_id}",
            "/api/v1/hindsight",
            "/api/v1/hindsight/recommendations",
            "/api/v1/hindsight/resolve",
            "/api/v1/hindsight/return/{return_id}",
            "/api/v1/clusters",
            "/api/v1/customers/{customer_id}/returns",
            "/api/v1/events",
            "/api/v1/events/backfill",
            "/api/v1/decisions",
            "/api/v1/decisions/score",
            "/api/v1/reviews",
            "/api/v1/reviews/{review_id}/resolve",
            "/api/v1/webhooks/deliveries",
            "/api/v1/sandbox/test-webhook",
            "/api/v1/policy/versions",
            "/api/v1/policy/preview",
            "/api/v1/policy/activate",
            "/api/v1/policy/rollback",
        ],
        "live_returns_endpoint": "/api/v1/returns",
        "live_stats_endpoint": "/api/v1/returns/stats",
        "export_endpoint": "/api/v1/returns/export.csv",
        "intent_model": _intent_model_status(),
        "local_model": local_model_status(),
        "access_control": {
            **api_guard.status(),
            "auth_enabled": api_guard.status()["auth_enabled"] or api_keys.configured(),
            "scoped_keys": api_keys.status(),
        },
        "decision_api": {
            "environments": list(api_keys.ENVIRONMENTS),
            "scopes": sorted(api_keys.ALL_SCOPES),
            "policy_control": {
                "versioned": True,
                "active_version": _active_policy_version(),
                "preview_endpoint": "/api/v1/policy/preview",
                "activate_endpoint": "/api/v1/policy/activate",
                "rollback_endpoint": "/api/v1/policy/rollback",
            },
            "outcomes": {
                "abusive": sorted(ABUSIVE_OUTCOMES),
                "benign": sorted(BENIGN_OUTCOMES),
            },
        },
    }


def _intent_model_status() -> dict:
    """Which intent classifier this process actually loaded."""
    agent = globals().get("_chat_agent")
    if agent is None:
        return {"source": "unknown", "semantic_source": "unknown"}
    return {
        "source": getattr(agent, "model_source", "unknown"),
        "semantic_source": getattr(agent, "semantic_source", "disabled"),
        "semantic_trigger": SEMANTIC_TRIGGER,
    }


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "ReturnShield AI Risk Scorer",
        "docs_url": "/docs",
        "endpoints": ["/api/v1/health", "/api/v1/policy", "/api/v1/score", "/api/v1/batch_score", "/api/v1/returns", "/api/v1/returns/stats", "/api/v1/returns/start", "/api/v1/returns/stop", "/api/v1/returns/generate", "/api/v1/returns/export.csv", "/api/v1/agent/chat"]
    }


@app.get("/api/v1/health")
async def health():
    b, p = get_bundle()
    return {
        "status": "healthy",
        "model_kind": b["kind"],
        "policy": {
            "verify_threshold": p["verify_threshold"],
            "review_threshold": p["review_threshold"]
        }
    }


_RECOMMENDATION_CACHE: Dict[str, Any] = {"key": None, "value": None}


def _chat_recommendation(store) -> Dict[str, Any] | None:
    """A compact, cached policy hint attached to every chat turn.

    Memoised on the resolved-row count so the sweep runs when the evidence
    changes, not on every message.
    """
    try:
        rows = store.resolved_decision_rows()
    except Exception:
        return None
    key = len(rows)
    if _RECOMMENDATION_CACHE["key"] == key:
        return _RECOMMENDATION_CACHE["value"]
    try:
        policy = _policy_thresholds()
        result = hindsight_engine.recommend(
            rows, thresholds=policy["thresholds"], costs=policy["costs"],
            steps=hindsight_engine.DEFAULT_STEPS,
        )
        hint = {
            "recommendation": result.get("recommendation"),
            "sample_size": result.get("sample_size"),
            "rationale": result.get("rationale"),
        }
        for side in ("current_policy", "recommended_policy"):
            point = result.get(side) or {}
            if "verify_threshold" in point:
                hint[side] = {
                    "verify_threshold": point["verify_threshold"],
                    "review_threshold": point["review_threshold"],
                }
    except Exception as exc:  # never let a hint break the chat path
        hint = {"recommendation": "unavailable", "rationale": str(exc)[:120]}
    _RECOMMENDATION_CACHE.update({"key": key, "value": hint})
    return hint


def _policy_thresholds() -> dict:
    """Live policy thresholds and the cost model hindsight prices them with."""
    _, p = get_bundle()
    return {
        "thresholds": {
            "verify_threshold": float(p["verify_threshold"]),
            "review_threshold": float(p["review_threshold"]),
        },
        "costs": {k: float(v) for k, v in (p.get("costs") or {}).items()},
    }


@app.get("/api/v1/policy")
async def get_policy():
    _, p = get_bundle()
    return p


# ------------------------------------------------------- versioned policy
#
# The thresholds were file data that nothing could move without a redeploy.
# That is wrong during an incident and unanswerable in an audit: a fraud
# manager needs to tighten the policy while the attack is running, and "who
# changed this and when" needs to have an answer. A version is now a record -
# who, from what, to what, and what it was predicted to cost - and activating
# one swaps the in-memory policy so the next request already uses it.


def _policy_path() -> Path:
    """Where the policy file lives. Overridable so a test never writes models/."""
    return Path(os.getenv("RETURNSHIELD_POLICY_PATH") or (MODELS / "policy.json"))


COST_FIELDS = ("false_positive", "false_negative", "verification", "manual_review")
DEFAULT_COSTS: Dict[str, float] = {name: float(getattr(PolicyCosts(), name)) for name in COST_FIELDS}


class PolicyVersionPayload(BaseModel):
    verify_threshold: float = Field(..., ge=0.0, lt=1.0, examples=[0.0202])
    review_threshold: float = Field(..., ge=0.0, le=1.0, examples=[0.6869])
    costs: Dict[str, float] | None = None
    actor: str | None = None
    note: str | None = None


class PolicyActivatePayload(BaseModel):
    version: int = Field(..., ge=1, examples=[2])
    actor: str | None = None
    note: str | None = None


class PolicyRollbackPayload(BaseModel):
    actor: str | None = None
    note: str | None = None


class PolicyPreviewPayload(BaseModel):
    verify_threshold: float = Field(..., ge=0.0, lt=1.0)
    review_threshold: float = Field(..., ge=0.0, le=1.0)
    costs: Dict[str, float] | None = None


def _policy_costs(policy: Dict[str, Any], overrides: Dict[str, float] | None = None) -> PolicyCosts:
    """Resolve the cost model: shipped values first, then any caller override."""
    merged = dict(DEFAULT_COSTS)
    for name, value in (policy.get("costs") or {}).items():
        if name in merged:
            try:
                merged[name] = float(value)
            except (TypeError, ValueError):
                continue
    for name, value in (overrides or {}).items():
        if name not in merged:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown cost '{name}'. Use one of: {sorted(merged)}.",
            )
        try:
            merged[name] = float(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"Cost '{name}' must be a number.")
        if merged[name] < 0:
            raise HTTPException(status_code=400, detail=f"Cost '{name}' cannot be negative.")
    return PolicyCosts(**merged)


def _check_thresholds(verify: float, review: float) -> None:
    """The verify band must be a band: an empty one silently drops every warn."""
    if float(verify) >= float(review):
        raise HTTPException(
            status_code=400,
            detail=f"verify_threshold ({verify}) must be below review_threshold ({review}).",
        )


def _evaluation_set():
    """The held-out scored rows, when run_pipeline.py has written them."""
    try:
        frame = pd.read_csv(REPORTS / "test_predictions.csv")
    except (OSError, ValueError):
        return None
    if not {"risk_probability", "abusive_return"} <= set(frame.columns):
        return None
    prob = pd.to_numeric(frame["risk_probability"], errors="coerce").fillna(0.0).to_numpy()
    return frame, prob


def _evaluate_operating_point(
    frame: pd.DataFrame,
    prob,
    verify: float,
    review: float,
    costs: PolicyCosts,
) -> Dict[str, Any]:
    """Price one operating point and split it across the four-way action space.

    ``_block_threshold`` lives with the decision API further down this module;
    it is resolved at call time, so the policy studio and the decision API can
    never disagree about where "block" starts.
    """
    result = evaluate_policy_point(frame, prob, verify, review, costs)
    decision = np.asarray(result.pop("decision"))
    block = _block_threshold({"verify_threshold": verify, "review_threshold": review})
    abuse_total = int(pd.to_numeric(frame["abusive_return"], errors="coerce").fillna(0).sum())
    return {
        **result,
        "rows": int(len(frame)),
        "abuse_total": abuse_total,
        "abuse_caught": abuse_total - int(result["false_negatives"]),
        "prevalence": round(abuse_total / max(len(frame), 1), 6),
        "block_threshold": round(block, 6),
        "allow_count": int((prob < verify).sum()),
        "warn_count": int((decision == "VERIFY").sum()),
        "review_count": int(((decision == "MANUAL_REVIEW") & (prob < block)).sum()),
        "block_count": int((prob >= block).sum()),
        "review_volume": int((decision == "MANUAL_REVIEW").sum()),
        "cost_per_call": result["expected_cost"] / max(len(frame), 1),
    }


def _preview_operating_point(thresholds: Dict[str, Any], costs: PolicyCosts) -> Optional[Dict[str, Any]]:
    """Evaluate on the held-out rows, or ``None`` when they are not on disk."""
    dataset = _evaluation_set()
    if dataset is None:
        return None
    frame, prob = dataset
    return _evaluate_operating_point(
        frame, prob, thresholds["verify_threshold"], thresholds["review_threshold"], costs
    )


def _versions_of(policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    versions = policy.get("versions")
    return versions if isinstance(versions, list) else []


def _seed_version_log(policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Guarantee a history exists, recording the shipped thresholds as v1."""
    if _versions_of(policy):
        return policy["versions"]
    now = time.time()
    seed_costs = _policy_costs(policy)
    policy["versions"] = [{
        "version": 1,
        "verify_threshold": float(policy["verify_threshold"]),
        "review_threshold": float(policy["review_threshold"]),
        "costs": {name: float(getattr(seed_costs, name)) for name in COST_FIELDS},
        "created_at": now,
        "activated_at": now,
        "actor": "shipped",
        "note": "policy as written by run_pipeline.py",
        "state": "active",
    }]
    policy["active_version"] = 1
    return policy["versions"]


def _write_policy(policy: Dict[str, Any]) -> None:
    """Persist atomically; a truncated policy file would break every later score."""
    path = _policy_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(policy, handle, indent=2)
    os.replace(tmp, path)


def _policy_view(policy: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "verify_threshold": float(policy["verify_threshold"]),
        "review_threshold": float(policy["review_threshold"]),
        "active_version": policy.get("active_version"),
        "costs": {n: float(v) for n, v in (policy.get("costs") or {}).items() if n in COST_FIELDS},
    }


def _active_policy_version() -> Optional[int]:
    """Best effort: discovery must answer even on a deployment with no bundle."""
    try:
        return _policy_view(get_bundle()[1]).get("active_version")
    except Exception:
        return None


def _activate_version(
    policy: Dict[str, Any], target: Dict[str, Any], actor: str | None, note: str | None
) -> Dict[str, Any]:
    """Make ``target`` live in the policy dict, with the actor who did it."""
    costs = _policy_costs(target)
    for record in _versions_of(policy):
        record["state"] = "superseded"
    target["state"] = "active"
    target["activated_at"] = time.time()
    if actor:
        target["actor"] = actor
    if note:
        target["note"] = note
    policy["verify_threshold"] = float(target["verify_threshold"])
    policy["review_threshold"] = float(target["review_threshold"])
    policy["costs"] = {name: float(getattr(costs, name)) for name in COST_FIELDS}
    policy["active_version"] = int(target["version"])
    evaluation = _preview_operating_point(target, costs)
    if evaluation is not None:
        # Refresh the cached metrics so the file never describes a threshold
        # that is no longer in force.
        for key in (
            "expected_cost",
            "loss_per_1000",
            "false_negatives",
            "false_positives",
            "verification_rate",
            "manual_review_rate",
            "auto_approve_rate",
        ):
            if key in evaluation:
                policy[key] = evaluation[key]
        target["evaluation"] = evaluation
    return target


@app.get("/api/v1/policy/versions")
async def list_policy_versions():
    """Every operating point this deployment has used, newest first."""
    _, policy = get_bundle()
    versions = _seed_version_log(policy)
    return {
        **_policy_view(policy),
        "versions": sorted(versions, key=lambda v: int(v["version"]), reverse=True),
        "policy_file": str(_policy_path()),
    }


@app.post("/api/v1/policy/versions", status_code=201)
async def create_policy_version(payload: PolicyVersionPayload):
    """Record a candidate operating point. Drafting changes nothing that scores."""
    _, policy = get_bundle()
    _check_thresholds(payload.verify_threshold, payload.review_threshold)
    versions = _seed_version_log(policy)
    costs = _policy_costs(policy, payload.costs)
    record = {
        "version": max((int(v["version"]) for v in versions), default=0) + 1,
        "verify_threshold": float(payload.verify_threshold),
        "review_threshold": float(payload.review_threshold),
        "costs": {name: float(getattr(costs, name)) for name in COST_FIELDS},
        "created_at": time.time(),
        "activated_at": None,
        "actor": payload.actor or "unknown",
        "note": payload.note or "",
        "state": "draft",
    }
    evaluation = _preview_operating_point(record, costs)
    if evaluation is not None:
        record["evaluation"] = evaluation
    versions.append(record)
    _write_policy(policy)
    return {"version": record, "active": _policy_view(policy), "versions": versions}


@app.post("/api/v1/policy/preview")
async def preview_policy(payload: PolicyPreviewPayload):
    """Price a candidate operating point against the live one, changing nothing."""
    _, policy = get_bundle()
    _check_thresholds(payload.verify_threshold, payload.review_threshold)
    candidate_costs = _policy_costs(policy, payload.costs)
    candidate = {
        "verify_threshold": float(payload.verify_threshold),
        "review_threshold": float(payload.review_threshold),
        "costs": {name: float(getattr(candidate_costs, name)) for name in COST_FIELDS},
    }
    active = {
        "verify_threshold": float(policy["verify_threshold"]),
        "review_threshold": float(policy["review_threshold"]),
        "costs": {name: float(getattr(_policy_costs(policy), name)) for name in COST_FIELDS},
    }
    candidate_eval = _preview_operating_point(candidate, candidate_costs)
    active_eval = _preview_operating_point(active, _policy_costs(policy))
    body: Dict[str, Any] = {
        "candidate": candidate,
        "active": active,
        "active_version": policy.get("active_version"),
        "candidate_evaluation": candidate_eval,
        "active_evaluation": active_eval,
        "evaluation_available": candidate_eval is not None,
    }
    if candidate_eval is not None and active_eval is not None:
        delta = candidate_eval["expected_cost"] - active_eval["expected_cost"]
        body["delta"] = {
            "expected_cost": round(delta, 2),
            "expected_cost_pct": delta / max(abs(active_eval["expected_cost"]), 1e-9),
            "false_negatives": candidate_eval["false_negatives"] - active_eval["false_negatives"],
            "false_positives": candidate_eval["false_positives"] - active_eval["false_positives"],
            "review_volume": candidate_eval["review_volume"] - active_eval["review_volume"],
        }
    return body


@app.post("/api/v1/policy/activate")
async def activate_policy_version(payload: PolicyActivatePayload, request: Request):
    """Make a recorded version live. Effective on the next request, no restart."""
    _, policy = get_bundle()
    versions = _seed_version_log(policy)
    target = next((v for v in versions if int(v["version"]) == int(payload.version)), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Policy version {payload.version} does not exist.")
    environment = _request_environment(request, getattr(request.state, "key", None))
    if target.get("state") == "active":
        return {"version": target, "already_active": True, "policy": _policy_view(policy), "webhooks": []}
    _activate_version(policy, target, payload.actor, payload.note)
    _write_policy(policy)
    deliveries = webhooks.deliver(
        "policy.activated",
        {
            "version": target["version"],
            "verify_threshold": target["verify_threshold"],
            "review_threshold": target["review_threshold"],
            "actor": target.get("actor"),
            "note": target.get("note"),
            "evaluation": target.get("evaluation"),
        },
        environment=environment,
    )
    return {
        "version": target,
        "already_active": False,
        "policy": _policy_view(policy),
        "environment": environment,
        "webhooks": deliveries,
    }


@app.post("/api/v1/policy/rollback")
async def rollback_policy(payload: PolicyRollbackPayload, request: Request):
    """Return to the operating point that was live before the current one.

    An irreversible threshold change is how fraud systems cause incidents, so
    the previous version is always still on disk and always reactivatable.
    """
    _, policy = get_bundle()
    versions = _seed_version_log(policy)
    active = next((v for v in versions if v.get("state") == "active"), None)
    previous = sorted(
        (v for v in versions if v is not active and v.get("state") == "superseded"),
        key=lambda v: float(v.get("activated_at") or v.get("created_at") or 0.0),
        reverse=True,
    )
    if active is None or not previous:
        raise HTTPException(
            status_code=409,
            detail="There is no earlier policy version to roll back to.",
        )
    target = previous[0]
    environment = _request_environment(request, getattr(request.state, "key", None))
    note = payload.note or f"rollback from v{active['version']}"
    _activate_version(policy, target, payload.actor, note)
    _write_policy(policy)
    deliveries = webhooks.deliver(
        "policy.activated",
        {
            "version": target["version"],
            "rolled_back_from": int(active["version"]),
            "verify_threshold": target["verify_threshold"],
            "review_threshold": target["review_threshold"],
            "actor": target.get("actor"),
            "note": note,
        },
        environment=environment,
    )
    return {
        "version": target,
        "rolled_back_from": int(active["version"]),
        "policy": _policy_view(policy),
        "environment": environment,
        "webhooks": deliveries,
    }


def _expand_row(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Derive the engineered columns the model expects from one raw request."""
    row_dict["return_value"] = row_dict["order_value"]
    row_dict["return_value_ratio"] = 1.0
    row_dict["high_value_flag"] = int(row_dict["order_value"] >= 7500.0)
    row_dict["return_rate_30d"] = row_dict["returns_30d"] / max(row_dict["orders_30d"], 1)
    row_dict["return_rate_90d"] = row_dict["returns_90d"] / max(row_dict["orders_90d"], 1)
    row_dict["prediction_time"] = pd.Timestamp.now()
    return row_dict


def _decide(prob: float, p: Dict[str, Any]) -> str:
    if prob < p["verify_threshold"]:
        return "AUTO_APPROVE"
    if prob < p["review_threshold"]:
        return "VERIFY"
    return "MANUAL_REVIEW"


def _build_response(
    payload: ReturnRequestPayload,
    prob: float,
    decision: str,
    reasons: List[str],
    agent_output: Dict[str, str],
    latency_ms: float,
) -> ScoringResponse:
    loss_est = float(np.clip(payload.order_value * prob * 0.85, 0.0, payload.order_value))
    return ScoringResponse(
        return_id=payload.return_id,
        risk_probability=round(prob, 4),
        risk_display=f"{prob:.1%}",
        decision=decision,
        merchant_loss_estimate=round(loss_est, 2),
        top_signals=reasons,
        merchant_action_protocol=agent_output["merchant_action"],
        customer_message=agent_output["customer_message"],
        latency_ms=round(latency_ms, 2),
    )


@app.post("/api/v1/score", response_model=ScoringResponse)
async def score_return(payload: ReturnRequestPayload, store_decision: bool = True):
    _ensure_live_policy_initialised()
    t0 = time.perf_counter()
    b, _p_file = get_bundle()
    _ensure_live_policy_initialised()
    p = _live_policy_now()

    df_row = pd.DataFrame([_expand_row(payload.model_dump())])
    prob = float(predict_bundle(b, df_row)[0])
    decision = _decide(prob, p)

    shap_items = top_features(b, df_row, top_n=5)
    reasons = concise_reasoning(df_row.iloc[0], shap_items)

    series = df_row.iloc[0].copy()
    series["risk_probability"] = prob
    series["decision"] = decision
    agent_output = generate_agent_response(series, reasons)

    latency = (time.perf_counter() - t0) * 1000.0
    response = _build_response(payload, prob, decision, reasons, agent_output, latency)

    if store_decision:
        # Single scores feed hindsight too, so a return scored one at a time is
        # just as reviewable as one scored in a batch.
        memory_store.get_store().record_decision(
            payload.return_id, decision, prob, source="score",
            policy_version=f"T1={p['verify_threshold']:.3f};T2={p['review_threshold']:.3f}",
        )
    return response


def score_return(payload: ReturnRequestPayload, store_decision: bool = True):
    _ensure_live_policy_initialised()
    t0 = time.perf_counter()
    b, _p_file = get_bundle()
    p = _live_policy_now()

    df_row = pd.DataFrame([_expand_row(payload.model_dump())])
    prob = float(predict_bundle(b, df_row)[0])
    decision = _decide(prob, p)

    shap_items = top_features(b, df_row, top_n=5)
    reasons = concise_reasoning(df_row.iloc[0], shap_items)

    series = df_row.iloc[0].copy()
    series["risk_probability"] = prob
    series["decision"] = decision
    agent_output = generate_agent_response(series, reasons)

    latency = (time.perf_counter() - t0) * 1000.0
    response = _build_response(payload, prob, decision, reasons, agent_output, latency)

    if store_decision:
        # Single scores feed hindsight too, so a return scored one at a time is
        # just as reviewable as one scored in a batch.
        memory_store.get_store().record_decision(
            payload.return_id, decision, prob, source="score",
            policy_version=f"T1={p['verify_threshold']:.3f};T2={p['review_threshold']:.3f}",
        )
    return response


@app.post("/api/v1/batch_score", response_model=List[ScoringResponse])
async def batch_score(payloads: List[ReturnRequestPayload], store_decisions: bool = True):
    """Score many returns.

    The model is run once over the whole frame instead of once per record, which
    is the difference between one predict pass and N of them. SHAP explanation
    still runs per row because that is what produces the per-return signals.
    """
    t0 = time.perf_counter()
    if not payloads:
        return []
    if len(payloads) > 1000:
        raise HTTPException(status_code=413, detail="batch_score accepts at most 1000 returns per call.")

    _ensure_live_policy_initialised()
    b, _p_file = get_bundle()
    p = _live_policy_now()
    frame = pd.DataFrame([_expand_row(item.model_dump()) for item in payloads])

    # One vectorised prediction pass for the entire batch.
    probs = predict_bundle(b, frame)
    shared = (time.perf_counter() - t0) * 1000.0
    per_row = shared / max(len(payloads), 1)

    results: List[ScoringResponse] = []
    store = memory_store.get_store()
    for i, payload in enumerate(payloads):
        prob = float(probs[i])
        decision = _decide(prob, p)

        df_row = frame.iloc[[i]]
        shap_items = top_features(b, df_row, top_n=5)
        reasons = concise_reasoning(df_row.iloc[0], shap_items)

        series = df_row.iloc[0].copy()
        series["risk_probability"] = prob
        series["decision"] = decision
        agent_output = generate_agent_response(series, reasons)

        # Report the true per-row cost, not the shared amortised guess.
        elapsed = (time.perf_counter() - t0) * 1000.0 - (shared - per_row)
        results.append(_build_response(payload, prob, decision, reasons, agent_output, elapsed))

        if store_decisions:
            store.record_decision(
                payload.return_id, decision, prob, source="batch_score",
                policy_version=f"T1={p['verify_threshold']:.3f};T2={p['review_threshold']:.3f}",
            )
    return results


@app.get("/api/v1/returns")
async def live_returns(
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    search: str = "",
    before: str | None = None,
    after: str | None = None,
):
    records, total = list_events(limit=limit, offset=offset, search=search, before=before, after=after)
    return {**_page(records, total, limit, offset),
            "server_time": pd.Timestamp.now(tz="UTC").isoformat()}


@app.get("/api/v1/returns/stats")
async def live_return_stats():
    return get_status()


@app.post("/api/v1/returns/start")
async def start_live_returns(rate_per_second: float = 2.0):
    start_generator(rate_per_second)
    return get_status()


@app.post("/api/v1/returns/stop")
async def stop_live_returns():
    stop_generator()
    return get_status()


@app.post("/api/v1/returns/generate")
async def generate_live_returns(count: int = 10):
    count = max(1, min(count, 1000))
    return {"data": generate_now(count), "status": get_status()}


@app.get("/api/v1/returns/export.csv")
async def export_live_returns(
    before: str | None = None,
    after: str | None = None,
    search: str = "",
):
    df = export_events_csv(before=before, after=after, search=search)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    filename = "returnshield_live_returns.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ------------------------------------------------------------------ memory API


@app.get("/api/v1/memory")
async def memory_sessions(limit: int = Query(50, ge=1, le=500)):
    """List conversations the agent remembers."""
    return {"sessions": memory_store.get_store().list_sessions(limit=limit), **memory_store.get_store().stats()}


@app.get("/api/v1/memory/{session_id}")
async def memory_history(session_id: str, limit: int = Query(50, ge=1, le=500)):
    """Recall a conversation the agent previously had with this session."""
    store = memory_store.get_store()
    history = store.get_history(session_id, limit=limit)
    return {"session_id": session_id, "turns": len(history), "history": history}


@app.delete("/api/v1/memory/{session_id}")
async def memory_forget(session_id: str):
    """Forget a conversation - the agent stops recalling it."""
    removed = memory_store.get_store().clear_session(session_id)
    return {"session_id": session_id, "forgotten_turns": removed}


# ---------------------------------------------------------------- hindsight API


@app.get("/api/v1/hindsight")
async def hindsight(recent: int = Query(25, ge=0, le=200)):
    """How the agent's logged calls actually turned out."""
    store = memory_store.get_store()
    return {
        "summary": store.hindsight_summary(),
        "recent_feedback": store.recent_feedback(limit=recent) if recent else [],
    }


@app.post("/api/v1/hindsight/resolve")
async def hindsight_resolve():
    """Join logged decisions against known abuse outcomes.

    Safe to call repeatedly: only unresolved decisions are touched.
    """
    store = memory_store.get_store()
    truth = memory_store.load_ground_truth()
    outcome = store.resolve_outcomes(truth)
    return {"resolution": outcome, "summary": store.hindsight_summary()}


@app.get("/api/v1/hindsight/recommendations")
async def hindsight_recommendations(
    objective: str = Query("minimize_cost", pattern="^(minimize_cost|maximize_f1|target_precision)$"),
    target_precision: float = Query(0.75, ge=0.0, le=1.0),
    steps: int = Query(hindsight_engine.DEFAULT_STEPS, ge=10, le=200),
):
    """Replay logged, resolved decisions and recommend a policy operating point.

    Priced with the cost assumptions in ``models/policy.json``. Recommends
    keeping the current policy unless a candidate wins by a real margin.
    """
    store = memory_store.get_store()
    policy = _policy_thresholds()
    result = hindsight_engine.recommend(
        store.resolved_decision_rows(),
        thresholds=policy["thresholds"],
        costs=policy["costs"],
        objective=objective,
        target_precision=target_precision,
        steps=steps,
    )
    return {"recommendation": result}


@app.get("/api/v1/hindsight/return/{return_id}")
async def hindsight_for_return(return_id: str):
    """What did we decide about this return before, and was it right?"""
    store = memory_store.get_store()
    ctx = memory_store.hindsight_context(return_id, store)
    return {"return_id": return_id, "prior_decisions": ctx.get("prior_decisions", []), "summary": ctx.get("summary")}


# ------------------------------------------------------------ decision API
#
# The ingress and feedback half of the product. Before this section the API
# could only answer questions about returns it had already scored; now a caller
# can send events, report outcomes, receive signed callbacks, and work the
# review queue. Every write path is namespaced by environment so sandbox traffic
# can never touch production state.

#: Outcomes that mean the return was abusive. Anything else is benign.
ABUSIVE_OUTCOMES = frozenset({"abusive", "abuse", "confirmed_abuse", "chargeback", "fraud"})
BENIGN_OUTCOMES = frozenset({"benign", "legitimate", "approved", "refunded", "ok", "no_fraud"})

#: The operating point actually used by every score on this process.
#: Starts from the policy that was live at cold start, then mutates in place
#: when a fraud manager sets a hot point through /api/v1/policy/live/set.
#: This is the no-restart gap the market research flagged: versioned writes
#: already existed, but a manager could not see the effect of a change without
#: first saving a draft and then activating it.
_live_policy: Dict[str, Any] = {}
_live_policy_lock = threading.Lock()


def _live_policy_now() -> Dict[str, Any]:
    """Snapshot of the thresholds and costs in force on this process right now."""
    with _live_policy_lock:
        return dict(_live_policy)


def _live_policy_set(target: Dict[str, Any], actor: str | None, note: str | None) -> Dict[str, Any]:
    """Make a scored operating point live in memory, without a deploy or restart.

    The versioned write path (draft -> activate -> rollback) is the audited
    contract for changes that must be recorded with a version and a webhook.
    This is the faster, still-audited path for a manager who needs to see the
    effect of a change immediately and can decide later whether to formalise it
    as a version.
    """
    _check_thresholds(float(target["verify_threshold"]), float(target["review_threshold"]))
    merged_costs = dict(_live_policy.get("costs") or {})
    for name, value in (target.get("costs") or {}).items():
        if name not in COSTS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown cost '{name}'. Use one of: {sorted(COSTS)}.",
            )
        try:
            merged_costs[name] = float(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"Cost '{name}' must be a number.")
        if merged_costs[name] < 0:
            raise HTTPException(status_code=400, detail=f"Cost '{name}' cannot be negative.")
    with _live_policy_lock:
        _live_policy["verify_threshold"] = float(target["verify_threshold"])
        _live_policy["review_threshold"] = float(target["review_threshold"])
        _live_policy["costs"] = dict(merged_costs)
        _live_policy["active_version"] = None
        _live_policy["live_set_at"] = time.time()
        _live_policy["live_set_by"] = actor or "live"
        _live_policy["live_set_note"] = note or ""
    return dict(_live_policy)


def _live_policy_reset_from_store():
    """Re-sync the live in-memory point from the versioned log, dropping any hot set."""
    _, p = get_bundle()
    with _live_policy_lock:
        _live_policy.clear()
        _live_policy.update({
            "verify_threshold": float(p["verify_threshold"]),
            "review_threshold": float(p["review_threshold"]),
            "costs": {n: float(v) for n, v in (p.get("costs") or {}).items() if n in COSTS},
            "active_version": p.get("active_version"),
            "live_set_at": None,
            "live_set_by": None,
            "live_set_note": None,
        })


def _live_policy_active_version_label() -> str:
    """Human label for the source of the in-memory point right now."""
    at = _live_policy.get("live_set_at")
    if at:
        return f"live_set_v{_live_policy.get('active_version') or '—'}_by_{_live_policy.get('live_set_by')}"
    return f"v{_live_policy.get('active_version')} (cold start)"


def _ensure_live_policy_initialised():
    """First-call init: seed the live tier from the cold-start policy without mutating the disk file."""
    if _live_policy:
        return
    _, p = get_bundle()
    with _live_policy_lock:
        if _live_policy:
            return
        _live_policy.update({
            "verify_threshold": float(p["verify_threshold"]),
            "review_threshold": float(p["review_threshold"]),
            "costs": {n: float(v) for n, v in (p.get("costs") or {}).items() if n in COSTS},
            "active_version": p.get("active_version"),
            "live_set_at": None,
            "live_set_by": None,
            "live_set_note": None,
        })    #: Valid cost-field names for live-set validation, shared with the versioned path.
COSTS = frozenset({"false_positive", "false_negative", "verification", "manual_review"})


class LivePolicySnapshot(BaseModel):
    verify_threshold: float
    review_threshold: float
    costs: Dict[str, float]
    active_version: int | None = None
    live_set_at: float | None = None
    live_set_by: str | None = None
    live_set_note: str | None = None
    source: str = "cold start"


class LivePolicySetResponse(BaseModel):
    before: LivePolicySnapshot
    after: LivePolicySnapshot


class RevertLivePolicyResponse(BaseModel):
    verify_threshold: float
    review_threshold: float
    costs: Dict[str, float]
    active_version: int | None = None
    source: str
    reverted_to: str


class LivePolicySetPayload(BaseModel):
    verify_threshold: float = Field(..., ge=0.0, lt=1.0)
    review_threshold: float = Field(..., ge=0.0, le=1.0)
    costs: Dict[str, float] | None = None
    actor: str | None = None
    note: str | None = None


class LivePolicyStatusResponse(BaseModel):
    verify_threshold: float
    review_threshold: float
    costs: Dict[str, float]
    active_version: int | None
    live_set_at: float | None
    live_set_by: str | None
    live_set_note: str | None
    source: str


class LivePolicySnapshot(BaseModel):
    verify_threshold: float
    review_threshold: float
    costs: Dict[str, float]
    active_version: int | None = None
    live_set_at: float | None = None
    live_set_by: str | None = None
    live_set_note: str | None = None
    source: str = "cold start"


class LivePolicySetResponse(BaseModel):
    before: LivePolicySnapshot
    after: LivePolicySnapshot


class RevertLivePolicyResponse(BaseModel):
    verify_threshold: float
    review_threshold: float
    costs: Dict[str, float]
    active_version: int | None = None
    source: str
    reverted_to: str


class LivePolicySetPayload(BaseModel):
    verify_threshold: float = Field(..., ge=0.0, lt=1.0)
    review_threshold: float = Field(..., ge=0.0, le=1.0)
    costs: Dict[str, float] | None = None
    actor: str | None = None
    note: str | None = None


@app.post("/api/v1/policy/live/set", status_code=200)
async def set_live_policy(payload: LivePolicySetPayload, request: Request):
    """Set the in-memory operating point now, without a deploy.

    This is the endpoint a fraud manager actually uses: drag a threshold slider
    and see the next score change immediately. It is still audited — every set
    is recorded against the actor and note, and the previous versioned state is
    left untouched so the change can always be re-derived or reverted.

    Scope: admin. The test harness is the reason this is behind a scope at all: a
    deployed server without any configured keys lets every caller set a point,
    which is acceptable because there is no authenticated actor to audit and the
    change is still revertible.
    """
    _ensure_live_policy_initialised()
    api_guard.require_scope(request, "admin")
    before = _live_policy_now()
    after = _live_policy_set(payload.model_dump(), payload.actor or None, payload.note or None)
    return LivePolicySetResponse(
        before=LivePolicySnapshot(
            verify_threshold=before["verify_threshold"],
            review_threshold=before["review_threshold"],
            costs=before["costs"],
            active_version=before.get("active_version"),
        ),
        after=LivePolicySnapshot(
            verify_threshold=after["verify_threshold"],
            review_threshold=after["review_threshold"],
            costs=after["costs"],
            active_version=after.get("active_version"),
            live_set_at=after.get("live_set_at"),
            live_set_by=after.get("live_set_by"),
            live_set_note=after.get("live_set_note"),
            source=_live_policy_active_version_label(),
        ),
    )


@app.get("/api/v1/policy/live", response_model=LivePolicyStatusResponse)
async def live_policy_status(request: Request):
    """The operating point that every score on this process actually uses right now.

    This is the endpoint the Threshold Studio polls to know whether the operator's
    hot edit landed, and whether the point they are looking at is a cold-start
    version or a hot set.
    """
    _ensure_live_policy_initialised()
    p = _live_policy_now()
    return LivePolicyStatusResponse(
        verify_threshold=p["verify_threshold"],
        review_threshold=p["review_threshold"],
        costs=p["costs"],
        active_version=p.get("active_version"),
        live_set_at=p.get("live_set_at"),
        live_set_by=p.get("live_set_by"),
        live_set_note=p.get("live_set_note"),
        source=_live_policy_active_version_label(),
    )


@app.post("/api/v1/policy/live/revert", status_code=200)
async def revert_live_policy(request: Request):
    """Drop any hot set and re-sync from the versioned log.

    Use this after a hot edit turned out to be wrong, or before re-applying a
    versioned change. It does not mutate the disk file either way.
    """
    _ensure_live_policy_initialised()
    api_guard.require_scope(request, "admin")
    _live_policy_reset_from_store()
    p = _live_policy_now()
    return RevertLivePolicyResponse(
        verify_threshold=p["verify_threshold"],
        review_threshold=p["review_threshold"],
        costs=p["costs"],
        active_version=p.get("active_version"),
        source=_live_policy_active_version_label(),
        reverted_to=_live_policy_active_version_label(),
    )


class EventPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    event_id: str = Field(..., min_length=1, max_length=200, examples=["evt_0001"])
    kind: str = Field("return", examples=["return"])
    return_id: str | None = Field(None, examples=["R999001"])
    customer_id: str | None = Field(None, examples=["C00123"])


class EventBatch(BaseModel):
    events: List[EventPayload] = Field(..., min_length=1, max_length=5000)
    backfill: bool = False


class DecisionFeedbackPayload(BaseModel):
    return_id: str = Field(..., min_length=1, examples=["R999001"])
    outcome: str = Field(..., examples=["abusive"])
    actor: str | None = Field(None, examples=["analyst@merchant.example"])
    note: str | None = None
    decision: str | None = Field(None, examples=["MANUAL_REVIEW"])
    risk_probability: float | None = None
    merchant_loss: float | None = None
    request_id: str | None = None


class DecisionScorePayload(ReturnRequestPayload):
    actor: str | None = None


class ReviewResolvePayload(BaseModel):
    resolution: str = Field(..., min_length=1, examples=["confirmed_abuse"])
    actor: str | None = None
    note: str | None = None


class DecisionScoreResponse(BaseModel):
    request_id: str
    return_id: str
    environment: str
    action: str
    decision: str
    score: float
    score_display: str
    reasons: List[Dict[str, Any]] = []
    reason_summary: List[str] = []
    merchant_loss_estimate: float
    expected_cost: float
    policy_version: str
    model_version: str
    review_id: int | None = None
    review_created: bool = False
    webhooks: List[Dict[str, Any]] = []
    latency_ms: float


def _policy_version(p: Dict[str, Any]) -> str:
    """The exact operating point a decision was made under."""
    return f"T1={float(p['verify_threshold']):.3f};T2={float(p['review_threshold']):.3f}"


_model_version_cache: Dict[str, str] = {}


def _model_version() -> str:
    """Stable fingerprint of the loaded bundle file, so a score is traceable."""
    cached = _model_version_cache.get("value")
    if cached:
        return cached
    path = MODELS / "model_bundle.joblib"
    try:
        stat = path.stat()
        stamp = f"{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        stamp = "unknown"
    value = "model_bundle@" + hashlib.sha256(stamp.encode()).hexdigest()[:12]
    _model_version_cache["value"] = value
    return value


def _block_threshold(p: Dict[str, Any]) -> float:
    """Probability above which a return is blocked outright.

    Policies may set ``block_threshold`` explicitly; otherwise it sits midway
    between the review threshold and certainty.
    """
    explicit = p.get("block_threshold")
    if explicit is not None:
        return float(explicit)
    review = float(p["review_threshold"])
    return min(0.999, review + (1.0 - review) * 0.5)


def _action_for(prob: float, p: Dict[str, Any]) -> tuple[str, str]:
    """Map a probability to the four-way action plus the legacy dashboard label.

    A binary block/allow throws away the "warn and approve" option, which is the
    decision that reduces abuse without punishing good customers.
    """
    if prob < float(p["verify_threshold"]):
        return "allow", "AUTO_APPROVE"
    if prob < float(p["review_threshold"]):
        return "warn", "VERIFY"
    if prob < _block_threshold(p):
        return "review", "MANUAL_REVIEW"
    return "block", "MANUAL_REVIEW"


def _structured_reasons(shap_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn raw SHAP items into ranked, human-readable reasons."""
    reasons: List[Dict[str, Any]] = []
    for item in shap_items or []:
        feature = str(item.get("feature", "")).replace("num__", "").replace("cat__", "")
        if not feature:
            continue
        contribution = float(item.get("contribution", 0.0))
        raised = contribution > 0
        reasons.append({
            "feature": feature,
            "contribution": round(contribution, 6),
            "direction": "increases_risk" if raised else "decreases_risk",
            "text": f"{feature} {'increased' if raised else 'reduced'} model risk",
        })
    return reasons


async def _ingest_events(batch: EventBatch, request: Request, backfill: bool) -> JSONResponse:
    environment = _request_environment(request, getattr(request.state, "key", None))
    events = [event.model_dump() for event in batch.events]
    summary = decision_store.get_decision_store().ingest(events, environment=environment, backfill=backfill)
    return JSONResponse(status_code=202, content=summary)


@app.post("/api/v1/events")
async def ingest_events(batch: EventBatch, request: Request):
    """Ingest lifecycle events. Idempotent on ``event_id`` within an environment."""
    return await _ingest_events(batch, request, backfill=batch.backfill)


@app.post("/api/v1/events/backfill")
async def backfill_events(batch: EventBatch, request: Request):
    """Ingest historical events. Same idempotency, flagged as backfill."""
    return await _ingest_events(batch, request, backfill=True)


@app.get("/api/v1/events")
async def list_ingested_events(
    request: Request,
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    """Everything ingested for this environment, newest first."""
    environment = _request_environment(request, getattr(request.state, "key", None))
    store = decision_store.get_decision_store()
    records = store.list_events(limit=limit, offset=offset, environment=environment)
    total = store.count_events(environment)
    return {**_page(records, total, limit, offset), "environment": environment}


@app.post("/api/v1/decisions")
async def record_decision_feedback(payload: DecisionFeedbackPayload, request: Request):
    """Report what actually happened to a return - the label that closes the loop.

    Resolves any open call for this return immediately, so the recommendation
    engine reflects it on the next request with no offline resolve job.
    """
    outcome = payload.outcome.strip().lower()
    if outcome in ABUSIVE_OUTCOMES:
        actual = 1
    elif outcome in BENIGN_OUTCOMES:
        actual = 0
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown outcome '{payload.outcome}'. Use one of: "
                   f"{sorted(ABUSIVE_OUTCOMES | BENIGN_OUTCOMES)}.",
        )
    environment = _request_environment(request, getattr(request.state, "key", None))
    source = f"feedback:{payload.actor}" if payload.actor else "feedback"
    result = memory_store.get_store().record_outcome(
        payload.return_id,
        actual,
        merchant_loss=payload.merchant_loss,
        risk_probability=payload.risk_probability,
        decision=payload.decision,
        source=source,
    )
    deliveries = webhooks.deliver(
        "decision.recorded",
        {
            "return_id": payload.return_id,
            "outcome": outcome,
            "actual_abusive": actual,
            "note": payload.note,
            "actor": payload.actor,
            "request_id": payload.request_id,
        },
        environment=environment,
    )
    return {
        "return_id": payload.return_id,
        "outcome": outcome,
        "actual_abusive": actual,
        "resolved_updated": result["updated"],
        "resolved_inserted": result["inserted"],
        "source": source,
        "environment": environment,
        "webhooks": deliveries,
    }


@app.post("/api/v1/decisions/score", response_model=DecisionScoreResponse)
async def score_decision(payload: DecisionScorePayload, request: Request):
    """Score one return and return a full decision object.

    Unlike ``/api/v1/score`` this returns the four-way action, the ranked reasons
    behind the score, and the policy and model versions that produced it, and it
    queues a human review when the action calls for one.
    """
    t0 = time.perf_counter()
    environment = _request_environment(request, getattr(request.state, "key", None))
    b, _p_file = get_bundle()
    _ensure_live_policy_initialised()
    p = _live_policy_now()

    frame = pd.DataFrame([_expand_row(payload.model_dump(exclude={"actor"}))])
    prob = float(predict_bundle(b, frame)[0])
    action, legacy = _action_for(prob, p)

    shap_items = top_features(b, frame, top_n=5)
    reasons = _structured_reasons(shap_items)
    reason_summary = concise_reasoning(frame.iloc[0], shap_items)

    order_value = float(payload.order_value)
    loss = float(np.clip(order_value * prob * 0.85, 0.0, order_value))
    # Cost of letting this return through unverified: the same quantity the
    # hindsight cost model prices thresholds with.
    expected_cost = loss

    policy_version = _policy_version(p)
    store = decision_store.get_decision_store()
    review_id: int | None = None
    review_created = False
    if action in {"review", "block"}:
        queued = store.enqueue_review(
            payload.return_id,
            environment=environment,
            action=action,
            risk_probability=prob,
            expected_cost=expected_cost,
            reasons=reasons,
            policy_version=policy_version,
        )
        review_id = queued["id"]
        review_created = queued["created"]

    # Sandbox scores must never enter the production hindsight sample.
    if environment == "production":
        memory_store.get_store().record_decision(
            payload.return_id, legacy, prob, source="decision_score",
            policy_version=policy_version,
        )

    deliveries: List[Dict[str, Any]] = []
    if action in {"review", "block"}:
        deliveries = webhooks.deliver(
            "decision.review_required",
            {
                "return_id": payload.return_id,
                "action": action,
                "score": round(prob, 4),
                "review_id": review_id,
                "reasons": reasons[:3],
                "policy_version": policy_version,
                "model_version": _model_version(),
            },
            environment=environment,
        )

    latency = (time.perf_counter() - t0) * 1000.0
    return DecisionScoreResponse(
        request_id=uuid.uuid4().hex[:16],
        return_id=payload.return_id,
        environment=environment,
        action=action,
        decision=legacy,
        score=round(prob, 4),
        score_display=f"{prob:.1%}",
        reasons=reasons,
        reason_summary=reason_summary,
        merchant_loss_estimate=round(loss, 2),
        expected_cost=round(expected_cost, 2),
        policy_version=policy_version,
        model_version=_model_version(),
        review_id=review_id,
        review_created=review_created,
        webhooks=deliveries,
        latency_ms=round(latency, 2),
    )


@app.get("/api/v1/reviews")
async def list_reviews(
    request: Request,
    state: str = Query("pending", pattern="^(pending|resolved|all)$"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """The human worklist, highest expected cost first."""
    environment = _request_environment(request, getattr(request.state, "key", None))
    store = decision_store.get_decision_store()
    records = store.list_reviews(state=state, environment=environment, limit=limit, offset=offset)
    total = store.count_reviews(state=state, environment=environment)
    return {**_page(records, total, limit, offset), "state": state, "environment": environment}


@app.post("/api/v1/reviews/{review_id}/resolve")
async def resolve_review(review_id: int, payload: ReviewResolvePayload, request: Request):
    """Resolve a queued review and record who did it."""
    environment = _request_environment(request, getattr(request.state, "key", None))
    store = decision_store.get_decision_store()
    row = store.resolve_review(
        review_id,
        resolution=payload.resolution,
        actor=payload.actor,
        note=payload.note,
        environment=environment,
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Review {review_id} is not pending in {environment}.")
    deliveries = webhooks.deliver(
        "review.resolved",
        {
            "review_id": row["id"],
            "return_id": row["return_id"],
            "resolution": row["resolution"],
            "actor": row["actor"],
        },
        environment=environment,
    )
    return {"review": row, "environment": environment, "webhooks": deliveries}


@app.get("/api/v1/webhooks/deliveries")
async def webhook_deliveries(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Every outbound callback this environment attempted, newest first."""
    environment = _request_environment(request, getattr(request.state, "key", None))
    store = decision_store.get_decision_store()
    records = store.list_deliveries(limit=limit, offset=offset, environment=environment)
    total = store.count_deliveries(environment)
    return {**_page(records, total, limit, offset), "environment": environment}


@app.post("/api/v1/sandbox/test-webhook")
async def sandbox_test_webhook():
    """Send a signed ping to the sandbox endpoints and report the delivery."""
    destinations = webhooks.endpoints("sandbox")
    deliveries = webhooks.deliver("ping", {"message": "ReturnShield sandbox test"}, environment="sandbox")
    return {
        "environment": "sandbox",
        "endpoints": len(destinations),
        "deliveries": deliveries,
        "hint": None if destinations else (
            "Register a sandbox endpoint (RETURNSHIELD_SANDBOX_WEBHOOK_URL) to see a delivery."
        ),
    }


# --------------------------------------------------------- capability endpoints

_eval_cache: Dict[str, Any] = {"at": 0.0, "df": None}


def _evaluation_frame() -> pd.DataFrame:
    """Cached held-out evaluation set, used when the live feed is not running."""
    if _eval_cache["df"] is not None:
        return _eval_cache["df"]
    path = REPORTS / "test_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "risk_probability" in df.columns:
        df["risk_probability"] = pd.to_numeric(df["risk_probability"], errors="coerce").fillna(0.0)
    # Scored exports keep the linked-account aggregates but drop the raw ids, so
    # cluster analysis has to rejoin them from customers.csv.
    df = attach_infrastructure_ids(df, str(ROOT / "data" / "raw"))
    _eval_cache["df"] = df
    return df


def _active_frame() -> tuple[pd.DataFrame, str]:
    """Live events when available, otherwise the held-out evaluation set."""
    status = get_status()
    if status.get("running"):
        records = _live_snapshot()
        if records:
            return pd.DataFrame(records), "live"
    return _evaluation_frame(), "evaluation"


@app.get("/api/v1/clusters")
async def coordinated_clusters(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    min_accounts: int = Query(2, ge=2, le=100),
):
    """Accounts that share infrastructure (device, address, or payment)."""
    df, source = _active_frame()
    if df.empty or "customer_id" not in df.columns:
        return {"source": source, "count": 0, "clusters": [],
                "message": "No records with shared-infrastructure identifiers are available."}
    offset = max(0, offset)

    rows: List[Dict[str, Any]] = []
    for col in ["device_id", "address_id", "payment_fingerprint"]:
        if col not in df.columns:
            continue
        grouped = (
            df.groupby(col, dropna=True)
              .agg(accounts=("customer_id", "nunique"), returns=("return_id", "count"))
              .reset_index()
        )
        grouped = grouped[grouped["accounts"] >= min_accounts].copy()
        if grouped.empty:
            continue
        grouped["signal_type"] = col
        grouped["shared_identifier"] = grouped[col].astype(str)
        rows.extend(grouped[["signal_type", "shared_identifier", "accounts", "returns"]].to_dict("records"))

    rows.sort(key=lambda r: (r.get("accounts", 0), r.get("returns", 0)), reverse=True)
    window = rows[offset:offset + limit]
    return {
        "source": source,
        "count": len(window),
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(window) < len(rows),
        "clusters": window,
    }


@app.get("/api/v1/customers/{customer_id}/returns")
async def customer_returns(
    customer_id: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Everything ReturnShield knows about one customer."""
    df, source = _active_frame()
    if df.empty or "customer_id" not in df.columns:
        raise HTTPException(status_code=404, detail="No customer data is available.")

    match = df[df["customer_id"].astype(str).str.upper() == customer_id.upper()].copy()
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} was not found.")

    if "risk_probability" in match.columns:
        match["risk_probability"] = pd.to_numeric(match["risk_probability"], errors="coerce").fillna(0.0)
        match = match.sort_values("risk_probability", ascending=False)

    total = len(match)
    offset = max(0, offset)
    decisions = match["decision"].astype(str) if "decision" in match.columns else pd.Series(dtype=str)
    cols = [c for c in ["return_id", "order_id", "return_reason", "order_value", "return_value",
                        "risk_probability", "decision", "abusive_return", "prediction_time", "generated_at"]
            if c in match.columns]
    window = match[cols].iloc[offset:offset + limit]
    return {
        "customer_id": customer_id.upper(),
        "source": source,
        "total_returns": total,
        "auto_approved": int((decisions == "AUTO_APPROVE").sum()),
        "verification": int((decisions == "VERIFY").sum()),
        "manual_review": int((decisions == "MANUAL_REVIEW").sum()),
        "device_linked_accounts": int(pd.to_numeric(match.get("device_linked_accounts", pd.Series([0])), errors="coerce").fillna(0).max()) if "device_linked_accounts" in match.columns else 0,
        "returns": window.where(pd.notna(window), None).to_dict("records"),
        "returned": len(window),
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(window) < total,
    }


@app.get("/returns")
async def live_returns_compat(limit: int = 100, offset: int = 0, search: str = "", before: str | None = None, after: str | None = None):
    return await live_returns(limit=limit, offset=offset, search=search, before=before, after=after)


@app.get("/health")
async def health_compat():
    return await health()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
