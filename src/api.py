from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from src.model import load_bundle, predict_bundle
from src.explain import top_features, concise_reasoning
from src.responder import generate_agent_response
from src.chat_agent import ReturnShieldChatAgent, ChatContext
from src.chatbot_api import ChatbotConfig, ReturnShieldChatbot

from fastapi.responses import StreamingResponse
from .live_server import start_generator, stop_generator, get_status, list_events, generate_now, export_events_csv
import io

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

app = FastAPI(
    title="ReturnShield AI API",
    description="High-Throughput Cost-Sensitive Return Abuse Risk Scorer & Decision Agent",
    version="2.0.0-live"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _start_fake_live_server():
    # Built-in defensive demo stream starts automatically with the API server.
    start_generator(2.0)

bundle = None
policy = None

def get_bundle():
    global bundle, policy
    if bundle is None:
        bundle_path = MODELS / "model_bundle.joblib"
        policy_path = MODELS / "policy.json"
        if not bundle_path.exists():
            raise RuntimeError("Model bundle not found. Please run `python run_pipeline.py` first.")
        bundle = load_bundle(str(bundle_path))
        with open(policy_path, "r", encoding="utf-8") as f:
            policy = json.load(f)
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


class AgentChatResponse(BaseModel):
    answer: str
    intent: str = ""
    confidence: float = 0.0
    data: List[Dict[str, Any]] = Field(default_factory=list)
    action: Dict[str, Any] | None = None
    action_result: str | None = None


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


@app.post("/api/v1/agent/chat", response_model=AgentChatResponse)
async def agent_chat(payload: AgentChatPayload):
    """General-purpose chatbot backed by a configurable OpenAI-compatible API, with ReturnShield tools."""
    ctx = _context_from_dict(payload.context or {})
    live_status = get_status()
    df = pd.DataFrame(payload.records or [])
    if live_status.get("running"):
        live_records, _ = list_events(limit=10000, offset=0)
        if live_records:
            df = pd.DataFrame(live_records)
    report = payload.report or {}

    ai_cfg = (payload.context or {}).get("ai_config") or {}
    cfg = ChatbotConfig.from_env(provider=ai_cfg.get("provider"), api_key=ai_cfg.get("api_key"), model=ai_cfg.get("model"), base_url=ai_cfg.get("base_url"))
    chatbot = ReturnShieldChatbot(cfg)
    history = ctx.history or []
    tools = _chat_tool_handlers(df)
    if cfg.enabled:
        try:
            ai = chatbot.chat(payload.message, history, df, report, live_status, tools)
            return AgentChatResponse(
                answer=str(ai.get("answer") or "I couldn't produce a response."),
                intent="chatbot",
                confidence=1.0,
                data=ai.get("data") or [],
                action=ai.get("action"),
                action_result=None,
            )
        except Exception as exc:
            # Fall through to the deterministic ReturnShield agent if the API is unavailable.
            fallback = _chat_agent.respond(payload.message, ctx, df, report, live_status)
            fallback_answer = str(fallback.get("answer") or "")
            fallback_answer += "\n\n_Chatbot API unavailable; used the ReturnShield local fallback agent._"
            data = fallback.get("data")
            records = []
            if isinstance(data, pd.DataFrame) and not data.empty:
                records = data.where(pd.notna(data), None).head(100).to_dict("records")
            return AgentChatResponse(answer=fallback_answer, intent=str(fallback.get("intent") or "fallback"), confidence=float(fallback.get("confidence") or 0), data=records, action=fallback.get("action"), action_result=f"AI API error: {exc}")

    result = _chat_agent.respond(payload.message, ctx, df, report, live_status)
    data = result.get("data")
    records = []
    if isinstance(data, pd.DataFrame) and not data.empty:
        records = data.where(pd.notna(data), None).head(100).to_dict("records")
    return AgentChatResponse(
        answer=str(result.get("answer") or "I could not process that request."),
        intent=str(result.get("intent") or ""),
        confidence=float(result.get("confidence") or 0.0),
        data=records,
        action=result.get("action"),
        action_result="OPENAI_API_KEY is not configured; using the local ReturnShield agent.",
    )


@app.get("/api/v1/meta")
async def meta():
    return {
        "service": "ReturnShield AI",
        "version": "2.0.0-live",
        "live_returns_endpoint": "/api/v1/returns",
        "live_stats_endpoint": "/api/v1/returns/stats",
        "export_endpoint": "/api/v1/returns/export.csv",
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


@app.get("/api/v1/policy")
async def get_policy():
    _, p = get_bundle()
    return p


@app.post("/api/v1/score", response_model=ScoringResponse)
async def score_return(payload: ReturnRequestPayload):
    t0 = time.perf_counter()
    b, p = get_bundle()

    row_dict = payload.model_dump()
    row_dict["return_value"] = row_dict["order_value"]
    row_dict["return_value_ratio"] = 1.0
    row_dict["high_value_flag"] = int(row_dict["order_value"] >= 7500.0)
    row_dict["return_rate_30d"] = row_dict["returns_30d"] / max(row_dict["orders_30d"], 1)
    row_dict["return_rate_90d"] = row_dict["returns_90d"] / max(row_dict["orders_90d"], 1)
    row_dict["prediction_time"] = pd.Timestamp.now()

    df_row = pd.DataFrame([row_dict])
    prob = float(predict_bundle(b, df_row)[0])

    t1_verify = p["verify_threshold"]
    t2_review = p["review_threshold"]

    if prob < t1_verify:
        decision = "AUTO_APPROVE"
    elif prob < t2_review:
        decision = "VERIFY"
    else:
        decision = "MANUAL_REVIEW"

    shap_items = top_features(b, df_row, top_n=5)
    reasons = concise_reasoning(df_row.iloc[0], shap_items)

    df_row_series = df_row.iloc[0].copy()
    df_row_series["risk_probability"] = prob
    df_row_series["decision"] = decision
    agent_output = generate_agent_response(df_row_series, reasons)

    loss_est = float(np.clip(payload.order_value * prob * 0.85, 0.0, payload.order_value))
    latency = (time.perf_counter() - t0) * 1000.0

    return ScoringResponse(
        return_id=payload.return_id,
        risk_probability=round(prob, 4),
        risk_display=f"{prob:.1%}",
        decision=decision,
        merchant_loss_estimate=round(loss_est, 2),
        top_signals=reasons,
        merchant_action_protocol=agent_output["merchant_action"],
        customer_message=agent_output["customer_message"],
        latency_ms=round(latency, 2)
    )


@app.post("/api/v1/batch_score", response_model=List[ScoringResponse])
async def batch_score(payloads: List[ReturnRequestPayload]):
    results = []
    for item in payloads:
        res = await score_return(item)
        results.append(res)
    return results


@app.get("/api/v1/returns")
async def live_returns(
    limit: int = 100,
    offset: int = 0,
    search: str = "",
    before: str | None = None,
    after: str | None = None,
):
    limit = max(1, min(limit, 5000))
    offset = max(0, offset)
    records, total = list_events(limit=limit, offset=offset, search=search, before=before, after=after)
    return {
        "data": records,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(records) < total,
        "server_time": pd.Timestamp.now(tz="UTC").isoformat(),
    }


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


@app.get("/returns")
async def live_returns_compat(limit: int = 100, offset: int = 0, search: str = "", before: str | None = None, after: str | None = None):
    return await live_returns(limit=limit, offset=offset, search=search, before=before, after=after)


@app.get("/health")
async def health_compat():
    return await health()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
