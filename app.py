from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

from src.explain import concise_reasoning, top_features
from src.model import load_bundle, predict_bundle
from src.network import attach_infrastructure_ids, build_abuse_graph, build_active_abuse_graph, plot_cluster_graph
from src.policy import Costs, optimize_policy
from src.responder import generate_agent_response
from src.redteam import run_redteam_simulation
from src.chat_agent import AGENT_COLUMNS, ReturnShieldChatAgent, ChatContext
from src.paths import root as _project_root

# Assets (reports/, models/, data/) sit at the project root, which is NOT always
# this file's folder: the packaged delivery runs app.py from the app/ subfolder
# with the assets beside the exe. src/paths.py is the single source of truth -
# it honours RETURNSHIELD_HOME (set by the launcher) and falls back to the
# checkout layout otherwise. Resolving from __file__ made the dashboard look for
# app/reports/final_report.json and fail with "Model assets not found".
ROOT = _project_root()
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

st.set_page_config(page_title="ReturnShield AI", page_icon="RS", layout="wide")


st.markdown("""
<style>
:root {
  --rs-blue: #049FD9;
  --rs-navy: #1A2432;
  --rs-bg: #F5F7FA;
  --rs-border: #D6DBE1;
  --rs-text: #1A2432;
  --rs-muted: #546170;
  --rs-ok: #24AB4C;
  --rs-warn: #EAA200;
  --rs-crit: #D13434;
}
html, body, [class*="css"] { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body, .stApp { background: var(--rs-bg); color: var(--rs-text); }
[data-testid="stSidebar"] { background: var(--rs-navy); border-right: 1px solid #0E1620; }
[data-testid="stSidebar"] > div:first-child { padding-top: 0; }
/* Streamlit's own sidebar header reserves ~60px above the product name for its
   native collapse chevron. Our minimize toggle replaces it, so drop the dead space. */
[data-testid="stSidebarHeader"] { display: none !important; }
[data-testid="stSidebarContent"] { padding-top: 0.85rem !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] hr { border-color: #2C3849; }
.rs-brand { display:flex; align-items:center; gap:9px; margin: 0 0 12px 0; padding: 0 2px; }
.rs-mark { width:28px; height:28px; border-radius:4px; background:var(--rs-blue); color:white; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:11px; letter-spacing:.4px; flex:0 0 auto; }
.rs-brand-title { font-size:15px; font-weight:700; color:#FFFFFF; line-height:1.1; white-space:nowrap; }
.rs-brand-sub { display:none; }
.rs-nav-label { font-size:10px; letter-spacing:.12em; font-weight:800; text-transform:uppercase; color:#7A8794; margin:20px 0 28px 2px; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 2px; }
[data-testid="stSidebar"] .stButton { margin: 0; }
[data-testid="stSidebar"] .stButton button {
  width:100%; text-align:left; justify-content:flex-start; gap:8px;
  border:1px solid transparent; border-left:3px solid transparent;
  border-radius:2px; background:transparent; color:#C6CDD6; padding:6px 10px;
  font-weight:600; font-size:.86rem; transition: background .12s ease, color .12s ease;
}
[data-testid="stSidebar"] .stButton button p { font-size:.86rem; }
[data-testid="stSidebar"] .stButton button > div { justify-content:flex-start !important; }
[data-testid="stSidebar"] .stButton button { line-height:1.25; }
[data-testid="stSidebar"] .stButton button:hover { background:#243040; color:#FFFFFF; border-left-color:var(--rs-blue); }
[data-testid="stSidebar"] .stButton button[kind="primary"] {
  background:#243040; color:#FFFFFF; border-left-color:var(--rs-blue); font-weight:700;
}
.rs-section-spacer { height: 8px; }
.rs-status { border:1px solid #2C3849; background:#1F2937; border-radius:4px; padding:8px 10px; margin:10px 0 12px 0; font-size:12px; color:#C6CDD6; }
.rs-status strong { color:#FFFFFF; letter-spacing:.06em; font-size:10.5px; }
.live-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--rs-ok); margin-right:7px; }
.small-muted { color:var(--rs-muted); font-size:.85rem; }
.rs-divider { border-top:1px solid var(--rs-border); margin:14px 0; }
[data-testid="stMetric"] { background:#FFFFFF; border:1px solid var(--rs-border); border-radius:4px; padding:9px 12px; }
[data-testid="stMetricLabel"] p { font-size:10.5px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--rs-muted); }
[data-testid="stMetricValue"] { font-size:1.7rem; font-weight:700; letter-spacing:-.01em; font-variant-numeric:tabular-nums; }
div[data-testid="stDataFrame"] { border:1px solid var(--rs-border); border-radius:4px; overflow:hidden; }
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button { border-radius:4px; font-weight:600; }
/* Dense, precise headings - Cisco dashboards avoid oversized display type.
   Scoped under stAppViewContainer so these outrank Streamlit's class-based rules. */
[data-testid="stAppViewContainer"] h1 { font-size:1.65rem; font-weight:700; letter-spacing:-.015em; color:var(--rs-text); }
[data-testid="stAppViewContainer"] h2 { font-size:1.2rem; font-weight:700; color:var(--rs-text); }
[data-testid="stAppViewContainer"] h3 { font-size:1.02rem; font-weight:700; color:var(--rs-text); }
hr { border-color:var(--rs-border); margin:1rem 0; }
/* Decision badge - token-based status colour, consistent with table risk cells. */
.rs-badge { display:inline-block; padding:4px 10px; border-radius:3px; font-size:.76rem; font-weight:700; letter-spacing:.05em; }
.rs-badge.low { background:#E7F6EC; color:#17793A; border:1px solid #B7E3C6; }
.rs-badge.medium { background:#FDF3DC; color:#8A6100; border:1px solid #F0DCA6; }
.rs-badge.high { background:#FBE7E7; color:#A31F1F; border:1px solid #F0BDBD; }
/* Explicit light surfaces for charts, tables, and app embeds */
.stPlotlyChart, .stPlotlyChart > div, .stPlotlyChart iframe { background:#FFFFFF !important; }
[data-testid="stDataFrame"] > div { background:#FFFFFF !important; }
/* Live fragments update in place; suppress transient refresh chrome/animations. */
[data-testid="stStatusWidget"], [data-testid="stDecoration"] { opacity:0 !important; pointer-events:none !important; }
[data-testid="stMetricValue"], [data-testid="stMetricDelta"] { transition:none !important; animation:none !important; }
[data-testid="stMetric"] * { transition:none !important; animation:none !important; }
/* Remove Streamlit stale-element fade during live fragment reruns. */
[data-stale="true"], [data-stale="true"] * { opacity:1 !important; transition:none !important; animation:none !important; filter:none !important; }
.stElementContainer, [data-testid="stElementContainer"], [data-testid="stVerticalBlock"] { transition:none !important; animation:none !important; }
.rs-risk-low { background:#E7F6EC !important; color:#17793A !important; font-weight:700 !important; }
.rs-risk-medium { background:#FDF3DC !important; color:#8A6100 !important; font-weight:700 !important; }
.rs-risk-high { background:#FBE7E7 !important; color:#A31F1F !important; font-weight:700 !important; }


/* Keep the main content fully usable while the sidebar is expanded. */
[data-testid="stAppViewContainer"] > .main { min-width: 0 !important; }
[data-testid="stMainBlockContainer"] { max-width: none !important; width: 100% !important; padding-left: 1.5rem !important; padding-right: 1.5rem !important; }
/* Streamlit reserves 6rem above content for its fixed 60px header, leaving ~36px
   of dead space over every page title. Trim to just clear the header instead.
   The chat view overrides this with a more specific rule of its own. */
[data-testid="stMainBlockContainer"] { padding-top: 4rem !important; }

.inv-wrap { width:100%; max-width:100%; }
.toolbar { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:12px; }
.searchbox { flex:1 1 280px; min-width:220px; }
.searchbox input, .selctl { width:100%; box-sizing:border-box; padding:9px 11px; border:1px solid #D6DBE1; border-radius:4px; background:#fff; color:#1A2432; }
.selctl { width:140px; }
.navbtn { border:1px solid #B9E2F3; background:#fff; color:#049FD9; border-radius:4px; padding:9px 12px; font-weight:650; cursor:pointer; }
.request-list { overflow:auto; max-height:420px; }
#inv_table { width:100%; border-collapse:separate; border-spacing:0; background:#fff; font-size:13px; }
#inv_table th { position:sticky; top:0; background:#F5F7FA; color:#546170; text-align:left; padding:10px; border-bottom:1px solid #D6DBE1; z-index:1; }
#inv_table td { padding:10px; border-bottom:1px solid #EAEEF2; color:#1A2432; white-space:normal; word-break:break-word; }
#inv_table .inv-row { cursor:pointer; }
#inv_table .inv-row:hover, #inv_table .inv-row.active { background:#EAF6FC; }
.riskcell, .decisioncell { font-weight:700; border-radius:999px; padding:6px 9px !important; }
.riskcell.low, .decisioncell.low { background:#E7F6EC !important; color:#17793A !important; }
.riskcell.medium, .decisioncell.medium { background:#FDF3DC !important; color:#8A6100 !important; }
.riskcell.high, .decisioncell.high { background:#FBE7E7 !important; color:#A31F1F !important; }
.detail-head { display:grid; grid-template-columns:1fr auto auto; align-items:center; gap:12px; margin-bottom:16px; }
.idlabel { font-size:10px; letter-spacing:.08em; font-weight:800; color:#546170; }
.idvalue { font-size:20px; font-weight:750; color:#1A2432; }
.risk-big, .decision-big { padding:9px 14px; border-radius:4px; font-weight:800; }
.detail-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
.metric { border:1px solid #D6DBE1; border-radius:4px; padding:10px 12px; background:#fff; min-width:0; }
.metric b { display:block; color:#546170; font-size:11px; margin-bottom:4px; }
.metric span { display:block; color:#1A2432; font-weight:650; overflow-wrap:anywhere; }
.detail-two { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:14px; }
.panel { border:1px solid #D6DBE1; border-radius:4px; background:#fff; padding:14px; }
.panel h4 { margin:0 0 10px 0; color:#1A2432; }
.panel li { margin-bottom:7px; }
.customer { margin-top:12px; background:#F5F7FA; border:1px solid #E3E8EE; border-radius:8px; padding:10px; }
.customer pre { white-space:pre-wrap; font-family:inherit; margin:7px 0 0 0; color:#1A2432; }
.empty { padding:20px; color:#546170; }
@media (max-width: 1000px) { .detail-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .detail-two { grid-template-columns:1fr; } }
@media (max-width: 700px) { .detail-head { grid-template-columns:1fr; } .detail-grid { grid-template-columns:1fr; } #inv_table { min-width:720px; } .request-list { overflow-x:auto; } }

</style>
""", unsafe_allow_html=True)


def risk_band(value: float) -> str:
    try:
        v = float(value)
    except Exception:
        return "low"
    if v >= float(policy.get("review_threshold", 0.70)):
        return "high"
    if v >= float(policy.get("verify_threshold", 0.35)):
        return "medium"
    return "low"


def style_risk_table(df: pd.DataFrame, risk_col: str = "risk_probability", risk_percent_col: str | None = None):
    """Style table cells according to risk while keeping the dataframe readable."""
    if df.empty:
        return df.style
    styler = df.style
    target = risk_percent_col if risk_percent_col and risk_percent_col in df.columns else risk_col if risk_col in df.columns else None
    if target:
        def risk_style(v):
            try:
                if isinstance(v, str) and v.endswith('%'):
                    x = float(v.rstrip('%')) / 100.0
                else:
                    x = float(v)
            except Exception:
                x = 0.0
            band = risk_band(x)
            if band == "high":
                return "background-color: #FBE7E7; color: #A31F1F; font-weight: 700;"
            if band == "medium":
                return "background-color: #FDF3DC; color: #8A6100; font-weight: 700;"
            return "background-color: #E7F6EC; color: #17793A; font-weight: 700;"
        styler = styler.map(risk_style, subset=[target])
    if "decision" in df.columns:
        def decision_style(v):
            text = str(v)
            if text == "MANUAL_REVIEW":
                return "background-color: #FBE7E7; color: #A31F1F; font-weight: 700;"
            if text == "VERIFY":
                return "background-color: #FDF3DC; color: #8A6100; font-weight: 700;"
            return "background-color: #E7F6EC; color: #17793A; font-weight: 700;"
        styler = styler.map(decision_style, subset=["decision"])
    return styler


def enforce_plotly_light(fig):
    """Apply the Cisco chart theme: cool white surface, hairline grid, dense type."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#1A2432", size=12),
        title=dict(font=dict(size=13, color="#1A2432", family="Inter, Segoe UI, sans-serif")),
        legend=dict(bgcolor="rgba(255,255,255,0)", font=dict(size=11)),
        margin=dict(l=44, r=16, t=44, b=36),
        bargap=0.05,
        hoverlabel=dict(font=dict(size=12)),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="#EAEEF2", zerolinecolor="#D6DBE1",
        linecolor="#C6CDD6", ticks="outside", tickcolor="#C6CDD6",
        tickfont=dict(size=11, color="#546170"), title_font=dict(size=11, color="#546170"),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#EAEEF2", zerolinecolor="#D6DBE1",
        linecolor="#C6CDD6", ticks="outside", tickcolor="#C6CDD6",
        tickfont=dict(size=11, color="#546170"), title_font=dict(size=11, color="#546170"),
    )
    return fig


@st.cache_data
def load_assets():
    report = json.load(open(REPORTS / "final_report.json", encoding="utf-8"))
    predictions = pd.read_csv(REPORTS / "test_predictions.csv", parse_dates=["prediction_time"])
    bundle = load_bundle(str(MODELS / "model_bundle.joblib"))
    policy = json.load(open(MODELS / "policy.json", encoding="utf-8"))
    features = pd.read_csv(ROOT / "data/processed/features.csv", parse_dates=["prediction_time"])
    return report, predictions, bundle, policy, features


try:
    report, default_predictions, bundle, policy, default_features = load_assets()
except Exception as e:
    st.error(f"Model assets not found. Run `python run_pipeline.py` first.\n\n{e}")
    st.stop()


def ensure_live_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, default in {
        "prediction_time": pd.Timestamp.now(tz="UTC"),
        "source": "LIVE",
        "merchant_loss_estimate": 0.0,
        "return_reason": "unknown",
        "customer_id": "unknown",
        "device_linked_accounts": 1,
        "address_linked_accounts": 1,
        "risk_probability": 0.0,
        "decision": "AUTO_APPROVE",
    }.items():
        if col not in df.columns:
            df[col] = default
    df["prediction_time"] = pd.to_datetime(df["prediction_time"], errors="coerce", utc=True)
    return df


def empty_prediction_frame() -> pd.DataFrame:
    """Return an empty prediction table with the columns expected by every UI view."""
    return pd.DataFrame(columns=[
        "return_id", "order_id", "customer_id", "prediction_time",
        "product_category", "order_value", "return_value", "return_reason",
        "return_rate_90d", "returns_30d", "refund_amount_90d",
        "hours_to_return", "device_linked_accounts", "address_linked_accounts",
        "abusive_return", "merchant_loss", "merchant_loss_estimate",
        "risk_probability", "decision", "source"
    ])


def score_dataframe(df: pd.DataFrame, policy_obj: dict) -> pd.DataFrame:
    out = df.copy()
    probs = predict_bundle(bundle, out)
    out["risk_probability"] = probs
    out["decision"] = np.where(
        probs < policy_obj["verify_threshold"], "AUTO_APPROVE",
        np.where(probs < policy_obj["review_threshold"], "VERIFY", "MANUAL_REVIEW")
    )
    out["merchant_loss"] = np.where(
        out["decision"] == "AUTO_APPROVE", probs * 2000.0,
        np.where(out["decision"] == "VERIFY", 40.0, 60.0)
    )
    return out


def reset_to_default():
    st.session_state["active_predictions"] = default_predictions.copy()
    st.session_state["active_features"] = default_features.copy()
    st.session_state["dataset_name"] = "Default Evaluation Dataset"
    st.session_state["source_mode"] = "default"
    st.session_state["live_config"] = None
    st.session_state["live_connected"] = False
    st.session_state["live_seen_ids"] = set()
    st.session_state["live_last_poll_at"] = None
    st.session_state["live_last_error"] = None
    st.session_state["live_cumulative_counts"] = {"total": 0, "AUTO_APPROVE": 0, "VERIFY": 0, "MANUAL_REVIEW": 0, "HIGH_RISK": 0}


if "active_predictions" not in st.session_state:
    reset_to_default()

# Session state for live feed
st.session_state.setdefault("live_config", None)
st.session_state.setdefault("live_seen_ids", set())
st.session_state.setdefault("live_connected", False)
st.session_state.setdefault("live_last_poll_at", None)
st.session_state.setdefault("live_cumulative_counts", {"total": 0, "AUTO_APPROVE": 0, "VERIFY": 0, "MANUAL_REVIEW": 0, "HIGH_RISK": 0})
st.session_state.setdefault("chat_context", ChatContext())
st.session_state.setdefault("chat_agent", ReturnShieldChatAgent())
st.session_state.setdefault("chat_messages", [])
st.session_state.setdefault("ai_settings_open", False)

# Live updates are rendered with Streamlit fragments only.
# Full-app autorefresh is intentionally disabled so the sidebar/static UI does not refresh.

def merge_live_records(new_df: pd.DataFrame):
    if new_df.empty:
        return
    new_df = ensure_live_columns(new_df)
    if "return_id" in new_df.columns:
        new_df["return_id"] = new_df["return_id"].astype(str)
        new_df = new_df[~new_df["return_id"].isin(st.session_state["live_seen_ids"])]
    if new_df.empty:
        return
    st.session_state["live_seen_ids"].update(new_df["return_id"].tolist())
    # Cumulative counters drive the live KPI cards so they visibly change on every poll.
    counts = st.session_state.setdefault("live_cumulative_counts", {"total": 0, "AUTO_APPROVE": 0, "VERIFY": 0, "MANUAL_REVIEW": 0, "HIGH_RISK": 0})
    counts["total"] += int(len(new_df))
    if "decision" in new_df.columns:
        dc = new_df["decision"].value_counts()
        for action in ("AUTO_APPROVE", "VERIFY", "MANUAL_REVIEW"):
            counts[action] += int(dc.get(action, 0))
    if "risk_probability" in new_df.columns:
        counts["HIGH_RISK"] += int((pd.to_numeric(new_df["risk_probability"], errors="coerce") >= float(policy["review_threshold"])).sum())
    existing = st.session_state["active_predictions"]
    combined = pd.concat([new_df, existing], ignore_index=True, sort=False)
    if "return_id" in combined.columns:
        combined = combined.drop_duplicates("return_id", keep="first")
    if "prediction_time" in combined.columns:
        combined = combined.sort_values("prediction_time", ascending=False, na_position="last")
    # Keep the UI responsive while the server can retain the full history on disk.
    combined = combined.head(100000).reset_index(drop=True)
    st.session_state["active_predictions"] = combined
    st.session_state["active_features"] = combined.copy()
    st.session_state["live_last_poll_at"] = pd.Timestamp.now(tz="UTC")


def fetch_live(base_url: str, endpoint: str, headers: dict, limit: int = 500, search: str = "") -> tuple[pd.DataFrame, dict]:
    url = base_url.rstrip("/") + endpoint
    params = {"limit": min(max(limit, 1), 5000), "offset": 0, "search": search}
    r = requests.get(url, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    payload = r.json()
    records = payload.get("data", payload if isinstance(payload, list) else [])
    return pd.DataFrame(records), payload if isinstance(payload, dict) else {"total": len(records)}


def live_status(base_url: str, headers: dict):
    url = base_url.rstrip("/") + "/api/v1/returns/stats"
    try:
        r = requests.get(url, headers=headers, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def live_display_frame(df: pd.DataFrame, max_rows: int = 300) -> pd.DataFrame:
    """Return the rolling live window used by charts/tables while keeping cumulative KPIs separate."""
    if df.empty or st.session_state.get("source_mode") != "live":
        return df
    sort_col = "prediction_time" if "prediction_time" in df.columns else "generated_at"
    if sort_col in df.columns:
        return df.sort_values(sort_col, ascending=False, na_position="last").head(max_rows).copy()
    return df.head(max_rows).copy()


def normalized_cluster_table(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize static and live cluster schemas so the UI never crashes when columns differ."""
    if cluster_df is None or cluster_df.empty:
        return pd.DataFrame(columns=[
            "cluster_id", "customer_count", "total_returns", "abusive_returns",
            "cluster_abuse_rate", "total_refund_value"
        ])
    out = cluster_df.copy()
    defaults = {
        "cluster_id": "—",
        "customer_count": 0,
        "total_returns": np.nan,
        "abusive_returns": np.nan,
        "cluster_abuse_rate": np.nan,
        "total_refund_value": np.nan,
        "node_count": 0,
        "avg_risk": np.nan,
        "max_risk": np.nan,
        "high_risk_customers": 0,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    # Live graph records are risk-based and may not carry outcome/refund columns.
    if out["total_returns"].isna().all():
        out["total_returns"] = out["customer_count"].astype(int)
    if out["abusive_returns"].isna().all():
        out["abusive_returns"] = out["high_risk_customers"].fillna(0).astype(int)
    if out["cluster_abuse_rate"].isna().all():
        out["cluster_abuse_rate"] = (
            pd.to_numeric(out["abusive_returns"], errors="coerce").fillna(0)
            / pd.to_numeric(out["total_returns"], errors="coerce").replace(0, np.nan)
        ).fillna(0.0)
    if out["total_refund_value"].isna().all():
        out["total_refund_value"] = 0.0

    return out


NAV_SECTIONS = [
    # The front door is the operational loop a fraud team actually works:
    # work the queue, close the outcomes, move the thresholds. Model metrics
    # are diagnostics - useful, but not what anyone opens the product for.
    ("Fraud Operations", [
        ("⬒", "Case Queue"),
        ("◷", "Outcome Ledger"),
        ("⚖", "Threshold Studio"),
    ]),
    ("Investigate", [
        ("▦", "Operations Overview"),
        ("⌕", "Return Investigator"),
        ("◈", "Abuse Ring Explorer"),
        ("◆", "AI Chat"),
    ]),
    ("Data Sources", [
        ("⇄", "Data & Live Server"),
    ]),
    ("Diagnostics", [
        ("▤", "Model Evaluation & ROI"),
    ]),
]

if "page" not in st.session_state:
    st.session_state["page"] = "Operations Overview"

with st.sidebar:
    _rail = bool(st.session_state.get("sidebar_minimized", False))
    if _rail:
        # Collapsed rail: Streamlit writes an inline width, so override it explicitly.
        st.markdown(
            "<style>"
            '[data-testid="stSidebar"] { width:64px !important; min-width:64px !important; }'
            '[data-testid="stSidebar"] .rs-brand { justify-content:center; margin-bottom:10px; }'
            '[data-testid="stSidebar"] .rs-brand-title,'
            '[data-testid="stSidebar"] .rs-status,'
            '[data-testid="stSidebar"] .rs-nav-label { display:none; }'
            '[data-testid="stSidebar"] .stButton button { padding:7px 0; }'
            '[data-testid="stSidebar"] .stButton button > div { justify-content:center !important; }'
            '[data-testid="stSidebar"] .stButton button p { font-size:1rem; }'
            "</style>",
            unsafe_allow_html=True,
        )

    brand_html = (
        "<div class='rs-brand rs-brand-rail'><div class='rs-mark'>RS</div>"
        "<div class='rs-brand-title'>ReturnShield AI</div></div>"
        if _rail else
        "<div class='rs-brand'><div class='rs-mark'>RS</div>"
        "<div class='rs-brand-title'>ReturnShield AI</div></div>"
    )
    if _rail:
        # The rail is too narrow for a two-column split, so stack the mark and
        # the expand control full width instead of letting the button clip.
        st.markdown(brand_html, unsafe_allow_html=True)
        if st.button("»", key="sidebar_minimize_toggle", help="Expand sidebar", width="stretch"):
            st.session_state["sidebar_minimized"] = False
            st.rerun()
    else:
        brand_col, toggle_col = st.columns([6, 1], vertical_alignment="center")
        with brand_col:
            st.markdown(brand_html, unsafe_allow_html=True)
        with toggle_col:
            if st.button("«", key="sidebar_minimize_toggle", help="Minimize sidebar", width="stretch"):
                st.session_state["sidebar_minimized"] = True
                st.rerun()

    if not _rail:
        if st.session_state["source_mode"] == "live":
            st.markdown("<div class='rs-status'><span class='live-dot'></span><strong style='color:#24AB4C'>LIVE TRANSACTION FEED</strong></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='rs-status'><strong>DATASET</strong><br>{st.session_state['dataset_name']}</div>", unsafe_allow_html=True)

    for section, items in NAV_SECTIONS:
        if not _rail:
            st.markdown(f"<div class='rs-nav-label'>{section}</div>", unsafe_allow_html=True)
        for icon, label in items:
            active = st.session_state["page"] == label
            if st.button(
                icon if _rail else f"{icon}  {label}",
                key=f"nav_{label}",
                help=label,
                width="stretch",
                type="primary" if active else "secondary",
            ):
                st.session_state["page"] = label
                st.rerun()

    st.markdown("<div class='rs-section-spacer'></div>", unsafe_allow_html=True)
    if st.button(
        "↺" if _rail else "↺  Reset to Default Dataset",
        key="reset_default_dataset",
        help="Reset to Default Dataset",
        width="stretch",
    ):
        reset_to_default()
        st.rerun()

page = st.session_state["page"]

predictions = st.session_state["active_predictions"]
features = st.session_state["active_features"]





def _live_operations_component(base_url: str, endpoint: str, token: str, verify_threshold: float, review_threshold: float, height: int = 900):
    """Stable browser-only live Operations dashboard. Polls API every second and updates DOM in place."""
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><style>
html,body{{margin:0;padding:0;background:#F5F7FA;color:#1A2432;font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;overflow:hidden}}
*{{box-sizing:border-box}}.wrap{{padding:8px 10px 18px}}
.kpis{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin-bottom:10px}}
.kpi,.card{{background:#fff;border:1px solid #D6DBE1;border-radius:10px}}
.kpi{{padding:12px 14px;min-height:74px}}.kl{{font-size:12px;color:#546170}}.kv{{font-size:24px;font-weight:750;color:#1A2432;margin-top:4px}}
.meta{{font-size:11px;color:#546170;margin:6px 2px 10px}}
.grid{{display:grid;grid-template-columns:1.3fr .9fr;gap:10px;margin-bottom:10px}}
.card{{padding:14px;min-width:0;overflow:hidden}}.title{{font-size:14px;font-weight:750;color:#1A2432;margin-bottom:10px}}
.chart{{height:270px;width:100%}}svg{{width:100%;height:100%;display:block}}
table{{width:100%;border-collapse:collapse;table-layout:fixed;font-size:12px}}th{{text-align:left;color:#546170;font-weight:700;border-bottom:1px solid #D6DBE1;padding:8px}}td{{padding:8px;border-bottom:1px solid #EAEEF2;color:#1A2432;overflow-wrap:anywhere;word-break:break-word;vertical-align:top}}
.risk{{font-weight:700;border-radius:999px;padding:5px 8px}}.low{{background:#E7F6EC;color:#17793A}}.medium{{background:#FDF3DC;color:#8A6100}}.high{{background:#FBE7E7;color:#A31F1F}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:#546170;margin-top:6px}}.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}}
@media(max-width:1000px){{.kpis{{grid-template-columns:repeat(2,minmax(0,1fr))}}.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<div class='wrap'>
<div class='kpis'>
<div class='kpi'><div class='kl'>Returns in View</div><div class='kv' id='k_total'>0</div></div>
<div class='kpi'><div class='kl'>Auto Approved</div><div class='kv' id='k_auto'>0</div></div>
<div class='kpi'><div class='kl'>Verification</div><div class='kv' id='k_verify'>0</div></div>
<div class='kpi'><div class='kl'>Manual Review</div><div class='kv' id='k_review'>0</div></div>
<div class='kpi'><div class='kl'>High Risk</div><div class='kv' id='k_high'>0</div></div>
</div>
<div class='meta' id='meta'>Connecting to live feed...</div>
<div class='grid'>
<div class='card'><div class='title'>Risk Probability Distribution</div><div id='risk_chart' class='chart'></div></div>
<div class='card'><div class='title'>Policy Actions</div><div id='pie_chart' class='chart'></div><div class='legend'><span><span class='dot' style='background:#24AB4C'></span>Auto Approve</span><span><span class='dot' style='background:#EAA200'></span>Verify</span><span><span class='dot' style='background:#D13434'></span>Manual Review</span></div></div>
</div>
<div class='card'><div class='title'>Highest-Risk Returns</div><div id='table_wrap'></div></div>
</div>
<script>
const API_BASE={json.dumps(base_url.rstrip('/'))};
const ENDPOINT={json.dumps(endpoint)};
const TOKEN={json.dumps(token or '')};
const T1={float(verify_threshold)};
const T2={float(review_threshold)};
const state={{seen:new Map(),initialized:false,total:0,auto:0,verify:0,review:0,high:0}};
function setText(id,v){{const e=document.getElementById(id);if(e && e.textContent!==String(v))e.textContent=String(v);}}
function esc(v){{return String(v??'').replace(/[&<>\"']/g,m=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[m]));}}
function riskClass(p){{p=Number(p||0);return p>=T2?'high':p>=T1?'medium':'low';}}
function riskBg(p){{const c=riskClass(p);return c==='high'?'#FBE7E7':c==='medium'?'#FDF3DC':'#E7F6EC';}}
function riskFg(p){{const c=riskClass(p);return c==='high'?'#A31F1F':c==='medium'?'#8A6100':'#17793A';}}
function ensureRiskChart(){{const host=document.getElementById('risk_chart');if(document.getElementById('risk_svg'))return;const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.id='risk_svg';svg.setAttribute('viewBox','0 0 720 270');const base=document.createElementNS(ns,'line');base.setAttribute('x1','34');base.setAttribute('x2','690');base.setAttribute('y1','240');base.setAttribute('y2','240');base.setAttribute('stroke','#CBD5DF');svg.appendChild(base);for(let i=0;i<20;i++){{const r=document.createElementNS(ns,'rect');r.id='rb_'+i;r.setAttribute('rx','2');svg.appendChild(r);}}host.appendChild(svg);}}
function renderRisk(rows){{ensureRiskChart();const bins=20,counts=Array(bins).fill(0);for(const x of rows){{const p=Math.max(0,Math.min(.999,Number(x.risk_probability||0)));counts[Math.floor(p*bins)]++;}}const max=Math.max(1,...counts);for(let i=0;i<bins;i++){{const r=document.getElementById('rb_'+i),w=656/bins,bh=195*counts[i]/max;r.setAttribute('x',34+i*w+1);r.setAttribute('width',Math.max(2,w-2));r.setAttribute('y',240-bh);r.setAttribute('height',bh);r.setAttribute('fill',i/bins>=T2?'#D13434':i/bins>=T1?'#EAA200':'#24AB4C');}}}}
function piePath(cx,cy,r,start,end){{const a=(start-90)*Math.PI/180,b=(end-90)*Math.PI/180,x0=cx+r*Math.cos(a),y0=cy+r*Math.sin(a),x1=cx+r*Math.cos(b),y1=cy+r*Math.sin(b),large=(end-start)>180?1:0;return `M ${{cx}} ${{cy}} L ${{x0}} ${{y0}} A ${{r}} ${{r}} 0 ${{large}} 1 ${{x1}} ${{y1}} Z`;}}
function ensurePie(){{const host=document.getElementById('pie_chart');if(document.getElementById('pie_svg'))return;const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.id='pie_svg';svg.setAttribute('viewBox','0 0 420 270');for(const k of ['AUTO_APPROVE','VERIFY','MANUAL_REVIEW']){{const p=document.createElementNS(ns,'path');p.id='ps_'+k;p.setAttribute('stroke','#fff');p.setAttribute('stroke-width','2');svg.appendChild(p);}}host.appendChild(svg);}}
function renderPie(rows){{ensurePie();const c={{AUTO_APPROVE:0,VERIFY:0,MANUAL_REVIEW:0}};for(const x of rows){{const a=x.decision||'AUTO_APPROVE';if(c[a]!==undefined)c[a]++;}}const total=Math.max(1,c.AUTO_APPROVE+c.VERIFY+c.MANUAL_REVIEW);let a0=0;for(const k of ['AUTO_APPROVE','VERIFY','MANUAL_REVIEW']){{const da=c[k]/total*360,e=document.getElementById('ps_'+k);e.setAttribute('d',piePath(150,135,95,a0,a0+da));e.setAttribute('fill',k==='AUTO_APPROVE'?'#24AB4C':k==='VERIFY'?'#EAA200':'#D13434');a0+=da;}}}}
function ensureTable(){{const host=document.getElementById('table_wrap');if(document.getElementById('ops_table'))return;const t=document.createElement('table');t.id='ops_table';t.innerHTML='<thead><tr><th>Return ID</th><th>Customer</th><th>Order Value</th><th>Risk</th><th>Decision</th><th>Reason</th></tr></thead><tbody></tbody>';host.appendChild(t);for(let i=0;i<12;i++){{const tr=t.tBodies[0].insertRow();for(let j=0;j<6;j++)tr.insertCell();}}}}
function renderTable(rows){{ensureTable();const top=[...rows].sort((a,b)=>Number(b.risk_probability||0)-Number(a.risk_probability||0)).slice(0,12),tb=document.getElementById('ops_table').tBodies[0];for(let i=0;i<tb.rows.length;i++){{const tr=tb.rows[i],x=top[i];tr.style.display=x?'table-row':'none';if(!x)continue;const p=Number(x.risk_probability||0),d=x.decision||'AUTO_APPROVE';tr.cells[0].textContent=x.return_id||'—';tr.cells[1].textContent=x.customer_id||'—';tr.cells[2].textContent='₹'+Number(x.order_value||0).toLocaleString();tr.cells[3].textContent=(p*100).toFixed(1)+'%';tr.cells[3].style.background=riskBg(p);tr.cells[3].style.color=riskFg(p);tr.cells[4].textContent=d.replace('_',' ');tr.cells[4].style.background=d==='MANUAL_REVIEW'?'#FBE7E7':d==='VERIFY'?'#FDF3DC':'#E7F6EC';tr.cells[4].style.color=d==='MANUAL_REVIEW'?'#A31F1F':d==='VERIFY'?'#8A6100':'#17793A';tr.cells[5].textContent=x.return_reason||'—';}}}}
async function pull(){{try{{const r=await fetch(API_BASE+ENDPOINT+'?limit=1000&offset=0',{{headers:TOKEN?{{Authorization:TOKEN}}:{{}}}});if(!r.ok)throw new Error('HTTP '+r.status);const j=await r.json();const rows=Array.isArray(j.data)?j.data:[];if(!state.initialized){{state.seen=new Map(rows.map(x=>[String(x.return_id||''),x]));state.total=rows.length;state.auto=rows.filter(x=>(x.decision||'AUTO_APPROVE')==='AUTO_APPROVE').length;state.verify=rows.filter(x=>x.decision==='VERIFY').length;state.review=rows.filter(x=>x.decision==='MANUAL_REVIEW').length;state.high=rows.filter(x=>Number(x.risk_probability||0)>=T2).length;state.initialized=true;}}else{{for(const x of rows){{const id=String(x.return_id||'');if(id&&!state.seen.has(id)){{state.seen.set(id,x);state.total++;const d=x.decision||'AUTO_APPROVE';if(d==='AUTO_APPROVE')state.auto++;else if(d==='VERIFY')state.verify++;else if(d==='MANUAL_REVIEW')state.review++;if(Number(x.risk_probability||0)>=T2)state.high++;}}}}}}const current=[...state.seen.values()].sort((a,b)=>String(b.prediction_time||b.generated_at||'').localeCompare(String(a.prediction_time||a.generated_at||''))).slice(0,300);setText('k_total',state.total.toLocaleString());setText('k_auto',state.auto.toLocaleString());setText('k_verify',state.verify.toLocaleString());setText('k_review',state.review.toLocaleString());setText('k_high',state.high.toLocaleString());setText('meta','LIVE • '+current.length+' records in rolling window • Updated '+new Date().toLocaleTimeString());renderRisk(current);renderPie(current);renderTable(current);}}catch(e){{setText('meta','Live feed connection error: '+e.message);}}}}
pull();setInterval(pull,1000);
</script></body></html>"""
    components.html(html, height=height, scrolling=False)


def _live_browser_component(title, base_url, endpoint, token, verify_threshold, review_threshold, mode, height=820):
    # Keep non-Operations live components from being changed by the Operations fix.
    if mode == "operations":
        return _live_operations_component(base_url, endpoint, token, verify_threshold, review_threshold, height)
    # Original component implementation is used for investigator/cluster/model/calibration/status.
    return _legacy_live_browser_component(title, base_url, endpoint, token, verify_threshold, review_threshold, mode, height)

def _legacy_live_browser_component(title, base_url, endpoint, token, verify_threshold, review_threshold, mode, height=820):
    """Browser-side live UI. Polls every second and mutates existing DOM nodes in place.
    No Streamlit reruns, no repeated innerHTML for live visuals, and no chart/table blink.
    """
    t1=float(verify_threshold); t2=float(review_threshold)
    if mode == 'operations':
        body="""<div class='wrap'><div class='kpis' id='kpis'><div class='kpi'><div class='kl'>Returns in View</div><div class='kv' id='k_total'>0</div></div><div class='kpi'><div class='kl'>Auto Approved</div><div class='kv' id='k_auto'>0</div></div><div class='kpi'><div class='kl'>Verification</div><div class='kv' id='k_verify'>0</div></div><div class='kpi'><div class='kl'>Manual Review</div><div class='kv' id='k_review'>0</div></div><div class='kpi'><div class='kl'>High Risk</div><div class='kv' id='k_high'>0</div></div></div><div class='meta' id='meta'>Connecting to live feed...</div><div class='charts'><div class='card'><div class='ctitle'>Risk Probability Distribution</div><div id='hist' class='chart'></div></div><div class='card'><div class='ctitle'>Policy Actions</div><div id='pie' class='chart pie-wrap'></div></div></div><div class='card'><div class='ctitle'>Highest-Risk Returns</div><div id='table'></div></div></div>"""
        script="""const API_BASE=__BASE__,ENDPOINT=__EP__,TOKEN=__TOKEN__,T1=__T1__,T2=__T2__;let rsTip=null;function tipInit(){if(rsTip)return;rsTip=document.createElement('div');rsTip.className='rs-tip';document.body.appendChild(rsTip)}function bindTip(el,html){if(!el)return;tipInit();el.onmouseenter=e=>{rsTip.innerHTML=html;rsTip.style.display='block';rsTip.style.left=(e.clientX+14)+'px';rsTip.style.top=(e.clientY+14)+'px'};el.onmousemove=e=>{rsTip.style.left=(e.clientX+14)+'px';rsTip.style.top=(e.clientY+14)+'px'};el.onmouseleave=()=>{rsTip.style.display='none'}}let init=false,seen=new Map(),c={total:0,AUTO_APPROVE:0,VERIFY:0,MANUAL_REVIEW:0,HIGH_RISK:0};
function esc(x){return String(x??'').replace(/[&<>\"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[m]))}
function pct(x){return(Number(x||0)*100).toFixed(1)+'%'}
function bg(p){p=Number(p||0);return p>=T2?'#FBE7E7':p>=T1?'#FDF3DC':'#E7F6EC'}
function fg(p){p=Number(p||0);return p>=T2?'#A31F1F':p>=T1?'#8A6100':'#17793A'}
function abg(a){return a==='MANUAL_REVIEW'?'#FBE7E7':a==='VERIFY'?'#FDF3DC':'#E7F6EC'}
function setText(id,v){const e=document.getElementById(id);if(e&&e.textContent!==String(v))e.textContent=String(v)}
function ensureHist(){const host=document.getElementById('hist');if(document.getElementById('hist_svg'))return;const w=560,h=250,pad=28,bins=20,bw=(w-pad*2)/bins;const ns='http://www.w3.org/2000/svg';const svg=document.createElementNS(ns,'svg');svg.setAttribute('id','hist_svg');svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.setAttribute('preserveAspectRatio','none');for(let i=0;i<bins;i++){const r=document.createElementNS(ns,'rect');r.setAttribute('id','bar_'+i);r.setAttribute('x',x=pad+i*bw+1);r.setAttribute('width',Math.max(2,bw-2));r.setAttribute('rx','2');svg.appendChild(r)}const line=document.createElementNS(ns,'line');line.setAttribute('x1',pad);line.setAttribute('x2',w-pad);line.setAttribute('y1',h-30);line.setAttribute('y2',h-30);line.setAttribute('stroke','#CBD5DF');svg.appendChild(line);host.appendChild(svg)}
function renderHist(rows){ensureHist();const bins=20,counts=Array(bins).fill(0);rows.forEach(x=>{const p=Math.max(0,Math.min(.999,Number(x.risk_probability||0)));counts[Math.floor(p*bins)]++});const mx=Math.max(1,...counts),h=250,pad=28,w=560,bw=(w-pad*2)/bins;for(let i=0;i<bins;i++){const bh=(h-55)*counts[i]/mx,r=document.getElementById('bar_'+i);r.setAttribute('y',h-30-bh);r.setAttribute('height',bh);r.setAttribute('fill',i/bins>=T2?'#D13434':i/bins>=T1?'#EAA200':'#24AB4C');bindTip(r,`<b>Risk band</b><br>${(i*5)}%–${((i+1)*5)}%<br><b>Returns:</b> ${counts[i].toLocaleString()}`)}}
function ensurePie(){const host=document.getElementById('pie');if(document.getElementById('pie_svg'))return;const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.id='pie_svg';svg.setAttribute('viewBox','0 0 420 250');const cx=140,cy=125,r=88;['AUTO_APPROVE','VERIFY','MANUAL_REVIEW'].forEach(k=>{const path=document.createElementNS(ns,'path');path.id='slice_'+k;path.setAttribute('stroke','#FFFFFF');path.setAttribute('stroke-width','2');svg.appendChild(path)});['AUTO_APPROVE','VERIFY','MANUAL_REVIEW'].forEach((k,i)=>{const dot=document.createElementNS(ns,'circle');dot.setAttribute('cx','285');dot.setAttribute('cy',65+38*i);dot.setAttribute('r','6');dot.setAttribute('fill',k==='AUTO_APPROVE'?'#24AB4C':k==='VERIFY'?'#EAA200':'#D13434');svg.appendChild(dot);const tx=document.createElementNS(ns,'text');tx.id='legend_'+k;tx.setAttribute('x','300');tx.setAttribute('y',70+38*i);tx.setAttribute('font-size','12');tx.setAttribute('fill','#1A2432');svg.appendChild(tx)});host.appendChild(svg)}
function piePath(cx,cy,r,start,end){const a=(start-90)*Math.PI/180,b=(end-90)*Math.PI/180,x0=cx+r*Math.cos(a),y0=cy+r*Math.sin(a),x1=cx+r*Math.cos(b),y1=cy+r*Math.sin(b),large=end-start>180?1:0;return `M ${cx} ${cy} L ${x0} ${y0} A ${r} ${r} 0 ${large} 1 ${x1} ${y1} Z`}
function renderPie(rows){ensurePie();const cc={AUTO_APPROVE:0,VERIFY:0,MANUAL_REVIEW:0};rows.forEach(x=>{const a=x.decision||'AUTO_APPROVE';if(a in cc)cc[a]++});const total=Math.max(1,cc.AUTO_APPROVE+cc.VERIFY+cc.MANUAL_REVIEW);let angle=0;Object.keys(cc).forEach(k=>{const da=cc[k]/total*360;const el=document.getElementById('slice_'+k);el.setAttribute('d',piePath(140,125,88,angle,angle+da));el.setAttribute('fill',k==='AUTO_APPROVE'?'#24AB4C':k==='VERIFY'?'#EAA200':'#D13434');bindTip(el,`<b>${k.replace('_',' ')}</b><br>Returns: ${cc[k].toLocaleString()}<br>Share: ${(cc[k]/total*100).toFixed(1)}%`);document.getElementById('legend_'+k).textContent=k.replace('_',' ')+' '+cc[k];angle+=da})}
function ensureTable(){const host=document.getElementById('table');if(document.getElementById('ops_table'))return;const table=document.createElement('table');table.id='ops_table';table.innerHTML='<thead><tr><th>Return ID</th><th>Customer</th><th>Order Value</th><th>Risk</th><th>Decision</th><th>Reason</th></tr></thead><tbody></tbody>';host.appendChild(table);for(let i=0;i<15;i++){const tr=document.createElement('tr');tr.id='op_row_'+i;for(let j=0;j<6;j++)tr.appendChild(document.createElement('td'));table.tBodies[0].appendChild(tr)}}
function renderTable(rows){ensureTable();const top=[...rows].sort((a,b)=>Number(b.risk_probability||0)-Number(a.risk_probability||0)).slice(0,15);for(let i=0;i<15;i++){const tr=document.getElementById('op_row_'+i);const x=top[i];tr.style.visibility=x?'visible':'hidden';if(!x)continue;const p=Number(x.risk_probability||0),a=x.decision||'AUTO_APPROVE',td=tr.children;td[0].textContent=x.return_id||'—';td[1].textContent=x.customer_id||'—';td[2].textContent='₹'+Number(x.order_value||0).toLocaleString();td[3].textContent=pct(p);td[3].style.background=bg(p);td[3].style.color=fg(p);td[3].style.fontWeight='700';td[3].style.borderRadius='999px';td[4].textContent=a.replace('_',' ');td[4].style.background=abg(a);td[4].style.fontWeight='700';td[4].style.borderRadius='999px';td[5].textContent=x.return_reason||'—'}}
async function pull(){try{const r=await fetch(API_BASE+ENDPOINT+'?limit=1000&offset=0',{headers:TOKEN?{'Authorization':TOKEN}:{}});if(!r.ok)throw new Error('HTTP '+r.status);const j=await r.json(),rows=j.data||[];if(!init){seen=new Map(rows.map(x=>[String(x.return_id),x]));c={total:rows.length,AUTO_APPROVE:0,VERIFY:0,MANUAL_REVIEW:0,HIGH_RISK:0};rows.forEach(x=>{const a=x.decision||'AUTO_APPROVE';if(a in c)c[a]++;if(Number(x.risk_probability||0)>=T2)c.HIGH_RISK++});init=true}else rows.forEach(x=>{const id=String(x.return_id||'');if(id&&!seen.has(id)){seen.set(id,x);c.total++;const a=x.decision||'AUTO_APPROVE';if(a in c)c[a]++;if(Number(x.risk_probability||0)>=T2)c.HIGH_RISK++}});const cur=[...seen.values()].sort((a,b)=>String(b.prediction_time||'').localeCompare(String(a.prediction_time||''))).slice(0,300);setText('k_total',c.total.toLocaleString());setText('k_auto',c.AUTO_APPROVE.toLocaleString());setText('k_verify',c.VERIFY.toLocaleString());setText('k_review',c.MANUAL_REVIEW.toLocaleString());setText('k_high',c.HIGH_RISK.toLocaleString());setText('meta','LIVE • '+cur.length+' records in rolling window • Updated '+new Date().toLocaleTimeString());renderHist(cur);renderPie(cur);renderTable(cur)}catch(e){setText('meta','Live feed connection error: '+e.message)}}pull();setInterval(pull,1000);"""
    elif mode == 'investigator':
        body="""<div class='wrap inv-wrap'><div class='toolbar'><div class='searchbox'><input id='search' placeholder='Search Return ID or Customer ID'></div><select id='page_size' class='selctl'><option value='25'>25 requests</option><option value='50' selected>50 requests</option><option value='100'>100 requests</option><option value='250'>250 requests</option></select><button id='prev' class='navbtn'>Previous</button><button id='next' class='navbtn'>Next</button><span id='status' class='meta'>Connecting...</span></div><div class='card'><div class='ctitle'>Return Request to Inspect</div><div id='queue' class='request-list'></div></div><div class='card'><div class='ctitle'>Single Return Investigation</div><div id='detail'><div class='empty'>Select a return request.</div></div></div></div>"""
        script=r"""const API_BASE=__BASE__,ENDPOINT=__EP__,TOKEN=__TOKEN__,T1=__T1__,T2=__T2__;let rsTip;function tipInit(){if(rsTip)return;rsTip=document.createElement('div');rsTip.className='rs-tip';document.body.appendChild(rsTip)}function bindTip(el,html){if(!el)return;tipInit();el.addEventListener('mouseenter',e=>{rsTip.innerHTML=html;rsTip.style.display='block';rsTip.style.left=(e.clientX+14)+'px';rsTip.style.top=(e.clientY+14)+'px'});el.addEventListener('mousemove',e=>{if(rsTip){rsTip.style.left=(e.clientX+14)+'px';rsTip.style.top=(e.clientY+14)+'px'}});el.addEventListener('mouseleave',()=>{if(rsTip)rsTip.style.display='none'})}let records=[],selected='',page=1,pageSize=50,lastIds=[];
function esc(x){return String(x??'').replace(/[&<>\"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[m]))}
function n(x){const v=Number(x);return Number.isFinite(v)?v:0}
function pct(x){return(n(x)*100).toFixed(1)+'%'}
function band(p){p=n(p);return p>=T2?'high':p>=T1?'medium':'low'}
function reasonList(x){const a=[];if(n(x.return_rate_90d)>.35)a.push('High historical return rate ('+(n(x.return_rate_90d)*100).toFixed(0)+'% over 90 days)');if(n(x.returns_30d)>=3)a.push('High recent return velocity ('+Math.round(n(x.returns_30d))+' returns in 30 days)');if(n(x.device_linked_accounts)>=3||n(x.address_linked_accounts)>=3)a.push('Shared device/address with multiple accounts');if(n(x.refund_amount_90d)>20000)a.push('High historical refund value (₹'+n(x.refund_amount_90d).toLocaleString()+')');if(n(x.hours_to_return)<12&&x.hours_to_return!==undefined)a.push('Very fast return request ('+n(x.hours_to_return).toFixed(1)+'h after delivery)');if(!a.length)a.push('No major historical risk signal exceeded the configured thresholds.');return a.slice(0,5)}
function actionText(dec){if(dec==='AUTO_APPROVE')return {cls:'low',summary:'Low-risk return. Customer behavior is within normal operating thresholds.',merchant:'Approve refund automatically. Issue the return label and release credit according to normal policy.',customer:'Your return request has been approved. A prepaid return label is available in your account.'};if(dec==='VERIFY')return {cls:'medium',summary:'Moderate-risk return. Additional evidence is recommended before completing the refund.',merchant:'Pause automatic refund and request item-condition, serial-number, and packaging evidence before authorization.',customer:'Thank you for your return request. Please upload a quick photo showing the current condition and packaging of the item so we can complete the verification.'};return {cls:'high',summary:'High-risk return. Multiple risk indicators warrant manual review.',merchant:'Hold automatic refund and route to the senior risk team. Review linked accounts, prior high-value returns, and item-condition evidence.',customer:'Your return request is undergoing standard administrative verification. Our support team will update you after the review.'}}
function filtered(){const q=(document.getElementById('search').value||'').trim().toLowerCase();let arr=records;if(q)arr=arr.filter(x=>String(x.return_id||'').toLowerCase().includes(q)||String(x.customer_id||'').toLowerCase().includes(q));return arr}
function renderQueue(){const arr=filtered(), totalPages=Math.max(1,Math.ceil(arr.length/pageSize));if(page>totalPages)page=totalPages;const start=(page-1)*pageSize,view=arr.slice(start,start+pageSize);const host=document.getElementById('queue');let table=document.getElementById('inv_table');if(!table){table=document.createElement('table');table.id='inv_table';table.innerHTML='<thead><tr><th>Return ID</th><th>Customer</th><th>Risk</th><th>Decision</th><th>Return Reason</th><th>Request Time</th></tr></thead><tbody></tbody>';host.appendChild(table)}const tbody=table.tBodies[0];while(tbody.rows.length<pageSize){const tr=tbody.insertRow();tr.className='inv-row';for(let i=0;i<6;i++)tr.insertCell()}for(let i=0;i<tbody.rows.length;i++){const tr=tbody.rows[i],x=view[i];tr.style.display=x?'table-row':'none';if(!x)continue;const p=n(x.risk_probability),d=actionText(x.decision||'AUTO_APPROVE');tr.className='inv-row'+(String(x.return_id)===selected?' active':'');tr.cells[0].textContent=x.return_id||'—';tr.cells[1].textContent=x.customer_id||'—';tr.cells[2].textContent=pct(p);tr.cells[2].className='riskcell '+band(p);tr.cells[3].textContent=(x.decision||'—').replace('_',' ');tr.cells[3].className='decisioncell '+d.cls;tr.cells[4].textContent=x.return_reason||'—';tr.cells[5].textContent=x.prediction_time?new Date(x.prediction_time).toLocaleString():'—';tr.onclick=()=>{selected=String(x.return_id);renderQueue();renderDetail()}}if(!selected&&view[0])selected=String(view[0].return_id);document.getElementById('status').textContent='Showing '+(arr.length?start+1:0)+'–'+Math.min(start+pageSize,arr.length)+' of '+arr.length+' matching return requests • Page '+page+' / '+totalPages+' • Updated '+new Date().toLocaleTimeString()}
function renderDetail(){const x=records.find(r=>String(r.return_id)===selected),d=document.getElementById('detail');if(!x){d.innerHTML='<div class="empty">Select a return request.</div>';return}const p=n(x.risk_probability),dec=x.decision||'AUTO_APPROVE',a=actionText(dec),rs=reasonList(x);d.innerHTML=`<div class="detail-head"><div><div class="idlabel">RETURN REQUEST</div><div class="idvalue">${esc(x.return_id)}</div></div><div class="risk-big ${a.cls}">${pct(p)}</div><div class="decision-big ${a.cls}">${esc(dec.replace('_',' '))}</div></div><div class="detail-grid"><div class="metric"><b>Customer</b><span>${esc(x.customer_id||'—')}</span></div><div class="metric"><b>Order ID</b><span>${esc(x.order_id||'—')}</span></div><div class="metric"><b>Order Value</b><span>₹${n(x.order_value).toLocaleString()}</span></div><div class="metric"><b>Return Value</b><span>₹${n(x.return_value).toLocaleString()}</span></div><div class="metric"><b>Expected Merchant Loss</b><span>₹${n(x.merchant_loss??x.merchant_loss_estimate).toLocaleString()}</span></div><div class="metric"><b>Return Reason</b><span>${esc(x.return_reason||'—')}</span></div><div class="metric"><b>Return Rate (90d)</b><span>${pct(x.return_rate_90d)}</span></div><div class="metric"><b>Returns (30d)</b><span>${Math.round(n(x.returns_30d))}</span></div><div class="metric"><b>Refunds (90d)</b><span>₹${n(x.refund_amount_90d).toLocaleString()}</span></div><div class="metric"><b>Hours to Return</b><span>${n(x.hours_to_return).toFixed(1)}h</span></div><div class="metric"><b>Linked Device Accounts</b><span>${Math.round(n(x.device_linked_accounts))}</span></div><div class="metric"><b>Linked Address Accounts</b><span>${Math.round(n(x.address_linked_accounts))}</span></div></div><div class="detail-two"><div class="panel"><h4>Key Risk Signals</h4><ul>${rs.map(r=>'<li>'+esc(r)+'</li>').join('')}</ul></div><div class="panel"><h4>Decision Agent Response</h4><p><b>Operational Summary</b><br>${esc(a.summary)}</p><p><b>Merchant Protocol</b><br>${esc(a.merchant)}</p><div class="customer"><b>Customer Communication</b><pre>${esc(a.customer)}</pre></div></div></div>`}
async function pull(){try{const q=document.getElementById('search').value||'';const r=await fetch(API_BASE+ENDPOINT+'?limit=1000&search='+encodeURIComponent(q),{headers:TOKEN?{'Authorization':TOKEN}:{}});if(!r.ok)throw new Error('HTTP '+r.status);const j=await r.json();records=(j.data||[]).sort((a,b)=>String(b.prediction_time||'').localeCompare(String(a.prediction_time||'')));const ids=records.map(x=>String(x.return_id));if(selected&&!ids.includes(selected))selected=ids[0]||'';renderQueue();renderDetail()}catch(e){document.getElementById('status').textContent='Live feed error: '+e.message}}
document.getElementById('search').addEventListener('input',()=>{page=1;clearTimeout(window.searchT);window.searchT=setTimeout(pull,150)});document.getElementById('page_size').addEventListener('change',e=>{pageSize=Number(e.target.value);page=1;renderQueue();renderDetail()});document.getElementById('prev').onclick=()=>{if(page>1){page--;renderQueue();renderDetail()}};document.getElementById('next').onclick=()=>{if(page<999){page++;renderQueue();renderDetail()}};pull();setInterval(pull,1000);"""
    elif mode == 'cluster':
        body="""<div class='wrap'><div class='toolbar'><input id='search' placeholder='Find customer or return'><span id='status' class='meta'>Connecting...</span></div><div class='card'><div class='ctitle'>Live Coordinated Account Graph</div><div class='legend'><span><b class='dot customer'></b> Customer</span><span><b class='dot device'></b> Shared Device</span><span><b class='dot address'></b> Shared Address</span><span><b class='dot payment'></b> Shared Payment</span><span class='hint'>Only infrastructure shared by 2+ high-risk accounts is shown.</span></div><div id='graph' class='graph'></div></div><div class='card'><div class='ctitle'>All High-Risk Coordinated Accounts</div><div id='table'></div></div></div>"""
        script=r"""const API_BASE=__BASE__,ENDPOINT=__EP__,TOKEN=__TOKEN__,T2=__T2__;let rsTip;function tipInit(){if(rsTip)return;rsTip=document.createElement('div');rsTip.className='rs-tip';document.body.appendChild(rsTip)}function bindTip(el,html){if(!el)return;tipInit();el.addEventListener('mouseenter',e=>{rsTip.innerHTML=html;rsTip.style.display='block';rsTip.style.left=(e.clientX+14)+'px';rsTip.style.top=(e.clientY+14)+'px'});el.addEventListener('mousemove',e=>{if(rsTip){rsTip.style.left=(e.clientX+14)+'px';rsTip.style.top=(e.clientY+14)+'px'}});el.addEventListener('mouseleave',()=>{if(rsTip)rsTip.style.display='none'})}let graphSig='';function esc(x){return String(x??'').replace(/[&<>\"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[m]))}function pct(x){return(Number(x||0)*100).toFixed(1)+'%'}function ensureTable(){const host=document.getElementById('table');if(document.getElementById('cluster_table'))return;const t=document.createElement('table');t.id='cluster_table';t.innerHTML='<thead><tr><th>Customer</th><th>Return</th><th>Risk</th><th>Shared Device</th><th>Shared Address</th></tr></thead><tbody></tbody>';host.appendChild(t);for(let i=0;i<250;i++){const tr=document.createElement('tr');tr.style.visibility='hidden';for(let j=0;j<5;j++)tr.appendChild(document.createElement('td'));t.tBodies[0].appendChild(tr)}}function renderTable(hr){ensureTable();const top=hr.slice().sort((a,b)=>Number(b.risk_probability||0)-Number(a.risk_probability||0)).slice(0,250),rows=document.getElementById('cluster_table').tBodies[0].children;for(let i=0;i<250;i++){const tr=rows[i],x=top[i];tr.style.visibility=x?'visible':'hidden';if(!x)continue;tr.children[0].textContent=x.customer_id||'—';tr.children[1].textContent=x.return_id||'—';tr.children[2].textContent=pct(x.risk_probability);tr.children[2].style.color='#A31F1F';tr.children[2].style.fontWeight='700';tr.children[3].textContent=x.device_id||'—';tr.children[4].textContent=x.address_id||'—'}}function renderGraph(hr){const customerMap=new Map(),infra=new Map();hr.forEach(x=>{const c=String(x.customer_id||'');if(!c)return;if(!customerMap.has(c))customerMap.set(c,{id:c,risk:Number(x.risk_probability||0),returns:1});else customerMap.get(c).returns++;for(const [type,key] of [['device',x.device_id],['address',x.address_id],['payment',x.payment_fingerprint]]){if(!key)continue;const k=type+':'+key;if(!infra.has(k))infra.set(k,{type,key:String(key),customers:new Set()});infra.get(k).customers.add(c)}});const shared=[...infra.values()].filter(v=>v.customers.size>=2);const customerIds=[...customerMap.keys()].slice(0,24);const sharedIds=shared.slice(0,18);const sig=JSON.stringify({c:customerIds,e:sharedIds.map(v=>[v.type,v.key,[...v.customers].sort()]).sort()});if(sig===graphSig)return;graphSig=sig;const w=980,h=500,svgNS='http://www.w3.org/2000/svg',g=document.getElementById('graph');g.innerHTML=`<svg id='cluster_svg' viewBox='0 0 ${w} ${h}'></svg>`;const svg=document.getElementById('cluster_svg');const leftX=220,rightX=760;const cpos=new Map();customerIds.forEach((id,i)=>{const y=45+i*(400/Math.max(1,customerIds.length-1));cpos.set(id,[leftX,y])});const ipos=new Map();sharedIds.forEach((v,i)=>{const y=55+i*(390/Math.max(1,sharedIds.length-1));ipos.set(v.type+':'+v.key,[rightX,y])});for(const v of sharedIds){for(const c of v.customers){if(!cpos.has(c))continue;const p1=cpos.get(c),p2=ipos.get(v.type+':'+v.key),line=document.createElementNS(svgNS,'line');line.setAttribute('x1',p1[0]);line.setAttribute('y1',p1[1]);line.setAttribute('x2',p2[0]);line.setAttribute('y2',p2[1]);line.setAttribute('stroke','#D6DBE1');line.setAttribute('stroke-width','1.6');svg.appendChild(line)}}for(const id of customerIds){const [x,y]=cpos.get(id),d=customerMap.get(id),circle=document.createElementNS(svgNS,'circle');circle.setAttribute('cx',x);circle.setAttribute('cy',y);circle.setAttribute('r','13');circle.setAttribute('fill',d.risk>=T2?'#D13434':'#049FD9');circle.setAttribute('stroke','#fff');circle.setAttribute('stroke-width','2');svg.appendChild(circle);const tx=document.createElementNS(svgNS,'text');tx.setAttribute('x',x-20);tx.setAttribute('y',y-18);tx.setAttribute('font-size','11');tx.setAttribute('fill','#1A2432');tx.textContent=id;svg.appendChild(tx)}for(const v of sharedIds){const [x,y]=ipos.get(v.type+':'+v.key),fill=v.type==='device'?'#EAF6FC':v.type==='address'?'#FDF3DC':'#E7F6EC',stroke=v.type==='device'?'#049FD9':v.type==='address'?'#EAA200':'#24AB4C',shape=document.createElementNS(svgNS,'rect');shape.setAttribute('x',x-12);shape.setAttribute('y',y-12);shape.setAttribute('width','24');shape.setAttribute('height','24');shape.setAttribute('rx','5');shape.setAttribute('fill',fill);shape.setAttribute('stroke',stroke);shape.setAttribute('stroke-width','2');svg.appendChild(shape);const tx=document.createElementNS(svgNS,'text');tx.setAttribute('x',x+18);tx.setAttribute('y',y+4);tx.setAttribute('font-size','10');tx.setAttribute('fill','#1A2432');tx.textContent=v.key;svg.appendChild(tx)}if(!sharedIds.length){const tx=document.createElementNS(svgNS,'text');tx.setAttribute('x',w/2);tx.setAttribute('y',h/2);tx.setAttribute('text-anchor','middle');tx.setAttribute('font-size','16');tx.setAttribute('fill','#546170');tx.textContent='No shared infrastructure cluster is present in the current high-risk window.';svg.appendChild(tx)}}async function pull(){try{const r=await fetch(API_BASE+ENDPOINT+'?limit=1000',{headers:TOKEN?{'Authorization':TOKEN}:{}});if(!r.ok)throw new Error('HTTP '+r.status);const j=await r.json();const q=(document.getElementById('search').value||'').toLowerCase();const hr=(j.data||[]).filter(x=>Number(x.risk_probability||0)>=T2&&(!q||String(x.customer_id||'').toLowerCase().includes(q)||String(x.return_id||'').toLowerCase().includes(q)));renderGraph(hr);renderTable(hr);document.getElementById('status').textContent='Updated '+new Date().toLocaleTimeString()+' • '+hr.length+' high-risk returns'}catch(e){document.getElementById('status').textContent='Live feed error: '+e.message}}document.getElementById('search').addEventListener('input',pull);pull();setInterval(pull,1000);"""
    elif mode == 'calibration':
        body="""<div class='wrap'><div class='card'><div class='ctitle'>Risk Calibration / Outcomes</div><div class='meta' id='cal_meta'>Waiting for live outcomes...</div><div id='calibration_chart' class='calibration-chart'></div></div></div>"""
        script=r"""const API_BASE=__BASE__,ENDPOINT=__EP__,TOKEN=__TOKEN__;let rsTip;function tipInit(){if(rsTip)return;rsTip=document.createElement('div');rsTip.className='rs-tip';document.body.appendChild(rsTip)}function bindTip(el,html){if(!el)return;tipInit();el.addEventListener('mouseenter',e=>{rsTip.innerHTML=html;rsTip.style.display='block';rsTip.style.left=(e.clientX+14)+'px';rsTip.style.top=(e.clientY+14)+'px'});el.addEventListener('mousemove',e=>{if(rsTip){rsTip.style.left=(e.clientX+14)+'px';rsTip.style.top=(e.clientY+14)+'px'}});el.addEventListener('mouseleave',()=>{if(rsTip)rsTip.style.display='none'})}let lastSignature='';
function setT(id,v){const e=document.getElementById(id);if(e&&e.textContent!==String(v))e.textContent=String(v)}
function ensureChart(){const host=document.getElementById('calibration_chart');if(document.getElementById('cal_svg'))return;const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.id='cal_svg';svg.setAttribute('viewBox','0 0 760 340');svg.setAttribute('preserveAspectRatio','none');const bg=document.createElementNS(ns,'rect');bg.setAttribute('x',0);bg.setAttribute('y',0);bg.setAttribute('width',760);bg.setAttribute('height',340);bg.setAttribute('fill','#fff');svg.appendChild(bg);for(let i=0;i<=5;i++){const y=290-i*48;const gl=document.createElementNS(ns,'line');gl.setAttribute('x1',65);gl.setAttribute('x2',730);gl.setAttribute('y1',y);gl.setAttribute('y2',y);gl.setAttribute('stroke','#EAEEF2');svg.appendChild(gl);const tx=document.createElementNS(ns,'text');tx.setAttribute('x',55);tx.setAttribute('y',y+4);tx.setAttribute('text-anchor','end');tx.setAttribute('font-size','11');tx.setAttribute('fill','#546170');tx.textContent=(i/5*100).toFixed(0)+'%';svg.appendChild(tx)}for(let i=0;i<=5;i++){const x=65+i*133;const gl=document.createElementNS(ns,'line');gl.setAttribute('x1',x);gl.setAttribute('x2',x);gl.setAttribute('y1',50);gl.setAttribute('y2',290);gl.setAttribute('stroke','#F0F3F5');svg.appendChild(gl);const tx=document.createElementNS(ns,'text');tx.setAttribute('x',x);tx.setAttribute('y',310);tx.setAttribute('text-anchor','middle');tx.setAttribute('font-size','11');tx.setAttribute('fill','#546170');tx.textContent=(i/5*100).toFixed(0)+'%';svg.appendChild(tx)}const axisX=document.createElementNS(ns,'line');axisX.setAttribute('x1',65);axisX.setAttribute('x2',730);axisX.setAttribute('y1',290);axisX.setAttribute('y2',290);axisX.setAttribute('stroke','#9AA9B5');svg.appendChild(axisX);const axisY=document.createElementNS(ns,'line');axisY.setAttribute('x1',65);axisY.setAttribute('x2',65);axisY.setAttribute('y1',50);axisY.setAttribute('y2',290);axisY.setAttribute('stroke','#9AA9B5');svg.appendChild(axisY);const ideal=document.createElementNS(ns,'line');ideal.id='cal_ideal';ideal.setAttribute('x1',65);ideal.setAttribute('y1',290);ideal.setAttribute('x2',730);ideal.setAttribute('y2',50);ideal.setAttribute('stroke','#9AA9B5');ideal.setAttribute('stroke-dasharray','6 5');svg.appendChild(ideal);const path=document.createElementNS(ns,'path');path.id='cal_path';path.setAttribute('fill','none');path.setAttribute('stroke','#049FD9');path.setAttribute('stroke-width','3');svg.appendChild(path);for(let i=0;i<10;i++){const c=document.createElementNS(ns,'circle');c.id='cal_pt_'+i;c.setAttribute('r','5');c.setAttribute('fill','#049FD9');c.setAttribute('stroke','#fff');c.setAttribute('stroke-width','2');c.style.visibility='hidden';svg.appendChild(c)}host.appendChild(svg)}
function render(rows){ensureChart();const bins=10,b=Array.from({length:bins},()=>({p:0,o:0,n:0}));rows.forEach(x=>{if(x.abusive_return===undefined||x.abusive_return===null)return;const p=Math.max(0,Math.min(.999,Number(x.risk_probability||0))),i=Math.min(bins-1,Math.floor(p*bins));b[i].p+=p;b[i].o+=Number(x.abusive_return||0);b[i].n++});const pts=[];for(let i=0;i<bins;i++){if(!b[i].n){const c=document.getElementById('cal_pt_'+i);c.style.visibility='hidden';continue}const p=b[i].p/b[i].n,o=b[i].o/b[i].n;const x=65+p*665,y=290-o*240;pts.push([x,y]);const c=document.getElementById('cal_pt_'+i);c.setAttribute('cx',x);c.setAttribute('cy',y);c.style.visibility='visible';bindTip(c,`<b>Calibration bucket</b><br>Predicted: ${(p*100).toFixed(1)}%<br>Observed: ${(o*100).toFixed(1)}%<br>Samples: ${b[i].n}`)}document.getElementById('cal_path').setAttribute('d',pts.map((p,i)=>(i?'L':'M')+' '+p[0]+' '+p[1]).join(' '));const count=rows.filter(x=>x.abusive_return!==undefined&&x.abusive_return!==null).length;setT('cal_meta',count?('Simulated outcomes: '+count.toLocaleString()+' • Updated '+new Date().toLocaleTimeString()):'Waiting for simulated outcomes...')}
async function pull(){try{const r=await fetch(API_BASE+ENDPOINT+'?limit=1000',{headers:TOKEN?{'Authorization':TOKEN}:{}});if(!r.ok)throw new Error('HTTP '+r.status);const j=await r.json();render(j.data||[])}catch(e){setT('cal_meta','Live outcome feed error: '+e.message)}}pull();setInterval(pull,1000);"""
    
    elif mode == 'model':
        body="""<div class='wrap'><div class='kpis' id='kpis'><div class='kpi'><div class='kl'>Live Returns</div><div class='kv' id='m_total'>0</div></div><div class='kpi'><div class='kl'>Live Auto Approved</div><div class='kv' id='m_auto'>0</div></div><div class='kpi'><div class='kl'>Live Verification</div><div class='kv' id='m_verify'>0</div></div><div class='kpi'><div class='kl'>Live Manual Review</div><div class='kv' id='m_review'>0</div></div><div class='kpi'><div class='kl'>Live High Risk</div><div class='kv' id='m_high'>0</div></div></div><div class='meta' id='meta'>Connecting...</div></div>"""
        script="""const API_BASE=__BASE__,ENDPOINT=__EP__,TOKEN=__TOKEN__,T2=__T2__;let rsTip;function tipInit(){if(rsTip)return;rsTip=document.createElement('div');rsTip.className='rs-tip';document.body.appendChild(rsTip)}function bindTip(el,html){if(!el)return;tipInit();el.addEventListener('mouseenter',e=>{rsTip.innerHTML=html;rsTip.style.display='block';rsTip.style.left=(e.clientX+14)+'px';rsTip.style.top=(e.clientY+14)+'px'});el.addEventListener('mousemove',e=>{if(rsTip){rsTip.style.left=(e.clientX+14)+'px';rsTip.style.top=(e.clientY+14)+'px'}});el.addEventListener('mouseleave',()=>{if(rsTip)rsTip.style.display='none'})}let seen=new Set(),c={total:0,AUTO_APPROVE:0,VERIFY:0,MANUAL_REVIEW:0,HIGH_RISK:0},init=false;function setT(id,v){const e=document.getElementById(id);if(e)e.textContent=v}async function pull(){try{const r=await fetch(API_BASE+ENDPOINT+'?limit=1000',{headers:TOKEN?{'Authorization':TOKEN}:{}});if(!r.ok)throw new Error('HTTP '+r.status);const j=await r.json(),rows=j.data||[];if(!init){rows.forEach(x=>seen.add(String(x.return_id)));c.total=rows.length;rows.forEach(x=>{const a=x.decision||'AUTO_APPROVE';if(a in c)c[a]++;if(Number(x.risk_probability||0)>=T2)c.HIGH_RISK++});init=true}else rows.forEach(x=>{const id=String(x.return_id);if(!seen.has(id)){seen.add(id);c.total++;const a=x.decision||'AUTO_APPROVE';if(a in c)c[a]++;if(Number(x.risk_probability||0)>=T2)c.HIGH_RISK++}});setT('m_total',c.total.toLocaleString());setT('m_auto',c.AUTO_APPROVE.toLocaleString());setT('m_verify',c.VERIFY.toLocaleString());setT('m_review',c.MANUAL_REVIEW.toLocaleString());setT('m_high',c.HIGH_RISK.toLocaleString());setT('meta','Last live update '+new Date().toLocaleTimeString())}catch(e){setT('meta','Live feed error: '+e.message)}}pull();setInterval(pull,1000);"""
    else:
        body="""<div class='wrap'><div class='kpis four' id='kpis'><div class='kpi'><div class='kl'>Server Buffer</div><div class='kv small' id='s_buffer'>0</div></div><div class='kpi'><div class='kl'>Generation Rate</div><div class='kv small' id='s_rate'>0/sec</div></div><div class='kpi'><div class='kl'>Generator</div><div class='kv small' id='s_running'>STOPPED</div></div><div class='kpi'><div class='kl'>Event Sequence</div><div class='kv small' id='s_seq'>0</div></div></div><div class='meta' id='meta'>Connecting...</div></div>"""
        script="""const API_BASE=__BASE__,TOKEN=__TOKEN__;function st(id,v){const e=document.getElementById(id);if(e)e.textContent=v}async function pull(){try{const r=await fetch(API_BASE+'/api/v1/returns/stats',{headers:TOKEN?{'Authorization':TOKEN}:{}});if(!r.ok)throw new Error('HTTP '+r.status);const j=await r.json();st('s_buffer',(j.buffered_records||0).toLocaleString());st('s_rate',(j.rate_per_second||0).toFixed(1)+'/sec');st('s_running',j.running?'RUNNING':'STOPPED');st('s_seq',(j.event_sequence||0).toLocaleString());st('meta','Live regime: '+(j.regime||'—')+' • Updated '+new Date().toLocaleTimeString())}catch(e){st('meta','Live server error: '+e.message)}}pull();setInterval(pull,1000);"""
    css="""<style>html,body{margin:0;padding:0;background:#F5F7FA;color:#1A2432;font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}.wrap{padding:8px 10px 18px}.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:10px}.kpis.four{grid-template-columns:repeat(4,1fr)}.kpi,.card{background:#fff;border:1px solid #D6DBE1;border-radius:4px}.kpi{padding:12px 14px;min-height:58px}.kl{font-size:12px;color:#546170}.kv{font-size:24px;font-weight:750;color:#1A2432;margin-top:4px}.kv.small{font-size:19px}.meta{font-size:11px;color:#546170;margin:6px 2px 10px}.rs-tip{position:fixed;z-index:99999;display:none;pointer-events:none;background:#1A2432;color:#fff;border:1px solid #2C3849;border-radius:6px;padding:7px 9px;font-size:11px;line-height:1.35;box-shadow:0 4px 14px rgba(0,0,0,.16);max-width:260px;white-space:normal}.charts{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(300px,.85fr);gap:10px;margin-bottom:10px}.card{padding:14px;min-width:0;overflow:hidden}.ctitle{font-size:14px;font-weight:750;color:#1A2432;margin-bottom:10px}.chart{height:250px;width:100%;min-width:0}.pie-wrap{display:flex;align-items:center;justify-content:center}.chart svg{width:100%;height:100%;display:block}table{width:100%;max-width:100%;border-collapse:collapse;table-layout:fixed;font-size:12px}th{text-align:left;color:#546170;font-weight:700;border-bottom:1px solid #D6DBE1;padding:8px;overflow-wrap:anywhere}td{padding:8px;border-bottom:1px solid #EAEEF2;color:#1A2432;overflow-wrap:anywhere;word-break:break-word;vertical-align:top}td[style*='border-radius']{display:table-cell}.toolbar{display:flex;gap:10px;align-items:center;margin-bottom:10px}.toolbar input{flex:1;border:1px solid #C6CDD6;border-radius:4px;padding:9px 11px;font-size:13px;outline:none}.selectlist{display:grid;gap:4px}.rowbtn{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr) 86px 118px;gap:8px;align-items:center;text-align:left;border:1px solid #EAEEF2;background:#fff;border-radius:4px;padding:8px 10px;color:#1A2432;min-width:0}.rowbtn span{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.rowbtn.sel{background:#EAF6FC;border-color:#B9E2F3}.detailgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.detailgrid>div{background:#F5F7FA;border:1px solid #E3E8EE;border-radius:4px;padding:10px}.detailgrid b{font-size:11px;color:#546170}.detailgrid div div{margin-top:4px}.riskbig{font-weight:800;font-size:20px}.riskbig.low{color:#17793A}.riskbig.medium{color:#8A6100}.riskbig.high{color:#A31F1F}.graph{height:430px;background:#fff;border:1px solid #EAEEF2;border-radius:4px}.graph svg{width:100%;height:100%}.legend{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin:0 0 10px;font-size:11px;color:#546170}.legend .dot{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}.legend .customer{background:#049FD9;border-radius:50%}.legend .device{background:#EAF6FC;border:2px solid #049FD9}.legend .address{background:#FDF3DC;border:2px solid #EAA200}.legend .payment{background:#E7F6EC;border:2px solid #24AB4C}.legend .hint{color:#7A8793}.calibration-chart{height:300px;width:100%}.calibration-chart svg{width:100%;height:100%;display:block}.empty{height:100%;display:grid;place-items:center;color:#546170}@media(max-width:1050px){.kpis{grid-template-columns:repeat(2,1fr)}.charts{grid-template-columns:1fr}.detailgrid{grid-template-columns:1fr}}@media(max-width:680px){.kpis{grid-template-columns:1fr}.rowbtn{grid-template-columns:minmax(0,1fr) minmax(0,1fr) 72px 92px;font-size:11px}.charts{grid-template-columns:1fr}}</style>"""
    vals={'__BASE__':json.dumps(base_url.rstrip('/')),'__EP__':json.dumps(endpoint),'__TOKEN__':json.dumps(token),'__T1__':str(t1),'__T2__':str(t2)}
    for a,b in vals.items(): script=script.replace(a,b)
    html=f"<!doctype html><html><head><meta charset='utf-8'>{css}</head><body>{body}<script>{script}</script></body></html>"
    components.html(html,height=height,scrolling=False)

def _poll_live_source():
    if st.session_state.get("live_connected") and st.session_state.get("live_config"):
        cfg = st.session_state["live_config"]
        try:
            new_df, _ = fetch_live(cfg["base_url"], cfg["endpoint"], cfg["headers"], limit=cfg["poll_limit"])
            merge_live_records(new_df)
            st.session_state["live_last_error"] = None
        except Exception as exc:
            st.session_state["live_last_error"] = str(exc)


def _live_poll_and_render(render_fn):
    _poll_live_source()
    render_fn()


def _render_operations_dynamic():
    predictions = st.session_state["active_predictions"]
    live_view = live_display_frame(predictions, max_rows=300)
    if st.session_state["source_mode"] == "live" and st.session_state.get("live_connected"):
        live_counts = st.session_state.get("live_cumulative_counts", {})
        total = int(live_counts.get("total", 0))
        auto = int(live_counts.get("AUTO_APPROVE", 0))
        verify = int(live_counts.get("VERIFY", 0))
        review = int(live_counts.get("MANUAL_REVIEW", 0))
        high_risk = int(live_counts.get("HIGH_RISK", 0))
    else:
        total = len(live_view)
        auto = int((live_view["decision"] == "AUTO_APPROVE").sum()) if "decision" in live_view else 0
        verify = int((live_view["decision"] == "VERIFY").sum()) if "decision" in live_view else 0
        review = int((live_view["decision"] == "MANUAL_REVIEW").sum()) if "decision" in live_view else 0
        high_risk = int((live_view["risk_probability"] >= policy["review_threshold"]).sum()) if "risk_probability" in live_view else 0

    c = st.columns(5)
    c[0].metric("Returns in View", f"{total:,}")
    c[1].metric("Auto Approved", f"{auto:,}")
    c[2].metric("Verification", f"{verify:,}")
    c[3].metric("Manual Review", f"{review:,}")
    c[4].metric("High Risk", f"{high_risk:,}")

    if st.session_state["source_mode"] == "live":
        last_poll = st.session_state.get("live_last_poll_at")
        poll_text = last_poll.strftime("%H:%M:%S.%f")[:-3] + " UTC" if last_poll is not None else "waiting"
        st.caption(f"Last live data update: {poll_text}")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(
            live_view, x="risk_probability", color="decision", nbins=35,
            title="Risk Probability Distribution",
            color_discrete_map={"AUTO_APPROVE": "#24AB4C", "VERIFY": "#EAA200", "MANUAL_REVIEW": "#D13434"},
        )
        enforce_plotly_light(fig)
        fig.update_traces(
            hovertemplate="Risk probability: %{x:.1%}<br>Decision: %{fullData.name}<br>Count: %{y}<extra></extra>"
        )
        st.plotly_chart(fig, key="ops_risk_distribution", height=340)
    with right:
        decision_counts = live_view["decision"].value_counts().rename_axis("Action").reset_index(name="Count") if "decision" in live_view else pd.DataFrame(columns=["Action","Count"])
        fig = px.pie(
            decision_counts, values="Count", names="Action", title="Policy Actions",
            color="Action",
            color_discrete_map={"AUTO_APPROVE": "#24AB4C", "VERIFY": "#EAA200", "MANUAL_REVIEW": "#D13434"},
            hole=0.58,
        )
        enforce_plotly_light(fig)
        fig.update_traces(
            hovertemplate="Action: %{label}<br>Returns: %{value:,}<br>Share: %{percent}<extra></extra>",
            textposition="inside", textinfo="percent", insidetextorientation="horizontal",
            marker=dict(line=dict(color="#FFFFFF", width=2)),
        )
        # Keep the donut compact so it matches the histogram card instead of dominating the row.
        fig.update_layout(showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5))
        st.plotly_chart(fig, key="ops_policy_actions", height=340)

    if "risk_probability" not in live_view.columns:
        live_view = ensure_live_columns(live_view)
    view = live_view.sort_values("risk_probability", ascending=False).head(25).copy()
    view["Risk"] = (pd.to_numeric(view["risk_probability"], errors="coerce").fillna(0.0) * 100).round(1).astype(str) + "%"
    risk_cols = [c for c in ["return_id", "customer_id", "order_value", "Risk", "decision", "return_reason"] if c in view.columns]
    st.dataframe(style_risk_table(view[risk_cols], risk_col="risk_probability", risk_percent_col="Risk"), hide_index=True)


if hasattr(st, "fragment"):
    @st.fragment(run_every="1s")
    def _operations_live_fragment():
        _live_poll_and_render(_render_operations_dynamic)
else:
    def _operations_live_fragment():
        _live_poll_and_render(_render_operations_dynamic)


def _render_investigator_dynamic(search: str, page_size: int, page_num: int):
    predictions = st.session_state["active_predictions"]
    features = st.session_state["active_features"]
    queue = predictions.copy()
    if queue.empty:
        queue = empty_prediction_frame()
    if search:
        s = search.lower()
        queue = queue[
            queue["return_id"].astype(str).str.lower().str.contains(s, na=False)
            | queue["customer_id"].astype(str).str.lower().str.contains(s, na=False)
        ]
    queue = queue.sort_values("prediction_time", ascending=False, na_position="last")
    total = len(queue)
    start = (page_num - 1) * page_size
    page_df = queue.iloc[start:start + page_size].copy()
    st.info(f"Showing {start + 1 if total else 0:,}–{min(start + page_size, total):,} of {total:,} matching return requests.")

    options = page_df["return_id"].astype(str).tolist() if "return_id" in page_df.columns else []
    if not options:
        st.warning("No return requests match the current filter.")
        return
    selected = st.selectbox("Return Request to Inspect", options, key="investigator_select_live")
    rows = predictions[predictions["return_id"].astype(str) == selected]
    if rows.empty:
        return
    row = rows.iloc[0]
    st.subheader(f"Return Request: {selected}")
    a, b, c, d = st.columns(4)
    a.metric("Predicted Abuse Risk", f"{float(row['risk_probability']):.1%}")
    b.metric("Order Value", f"₹{float(row.get('order_value', 0)):,.2f}")
    c.metric("Expected Merchant Loss", f"₹{float(row.get('merchant_loss', row.get('merchant_loss_estimate', row['risk_probability'] * 2000))):,.2f}")
    band = risk_band(float(row["risk_probability"]))
    d.markdown(f"### Action\n<div class='rs-badge {band}'>{row['decision'].replace('_', ' ')}</div>", unsafe_allow_html=True)

    full_rows = features[features["return_id"].astype(str) == selected]
    if full_rows.empty:
        return
    full_row = full_rows.iloc[[0]].copy()
    shap_items = top_features(bundle, full_row, top_n=6)
    reasons = concise_reasoning(full_row.iloc[0], shap_items)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Key Risk Signals")
        for reason in reasons:
            st.write("•", reason)
        profile_cols = ["return_rate_90d", "returns_30d", "refund_amount_90d", "hours_to_return", "device_linked_accounts", "address_linked_accounts"]
        profile = pd.DataFrame({"Signal": profile_cols, "Value": [row.get(c, "—") for c in profile_cols]})
        st.dataframe(profile, hide_index=True)
    with col2:
        st.subheader("Decision Agent Response")
        agent_resp = generate_agent_response(row, reasons)
        st.info(f"**Operational Summary**\n\n{agent_resp['summary']}")
        if row["decision"] == "AUTO_APPROVE":
            st.success(f"**Merchant Protocol**\n\n{agent_resp['merchant_action']}")
        elif row["decision"] == "VERIFY":
            st.warning(f"**Merchant Protocol**\n\n{agent_resp['merchant_action']}")
        else:
            st.error(f"**Merchant Protocol**\n\n{agent_resp['merchant_action']}")
        with st.expander("Customer Communication", expanded=True):
            st.code(agent_resp["customer_message"], language="markdown")


if hasattr(st, "fragment"):
    @st.fragment(run_every="1s")
    def _investigator_live_fragment(search: str, page_size: int, page_num: int):
        _live_poll_and_render(lambda: _render_investigator_dynamic(search, page_size, page_num))
else:
    def _investigator_live_fragment(search: str, page_size: int, page_num: int):
        _live_poll_and_render(lambda: _render_investigator_dynamic(search, page_size, page_num))


def _render_cluster_dynamic(search_cluster: str):
    predictions = st.session_state["active_predictions"]
    cluster_source = live_display_frame(predictions, max_rows=500)
    if cluster_source.empty or "risk_probability" not in cluster_source.columns:
        cluster_source = ensure_live_columns(cluster_source)
    high_risk = cluster_source[cluster_source["risk_probability"] >= policy["review_threshold"]].copy()
    if st.session_state["source_mode"] == "live" and {"device_id", "address_id", "payment_fingerprint"}.issubset(predictions.columns):
        G, cluster_df = build_active_abuse_graph(predictions, risk_threshold=float(policy["review_threshold"]), min_cluster_size=2)
    else:
        try:
            G, cluster_df = build_abuse_graph(str(ROOT / "data/raw"), min_cluster_size=2)
        except Exception:
            G, cluster_df = None, pd.DataFrame()
    if not high_risk.empty:
        high_risk = high_risk.sort_values("risk_probability", ascending=False)

    st.metric("High-Risk Coordinated Accounts", f"{len(high_risk):,}")
    hr_display = high_risk.copy()
    if search_cluster:
        ss = search_cluster.lower()
        hr_display = hr_display[
            hr_display["customer_id"].astype(str).str.lower().str.contains(ss, na=False)
            | hr_display["return_id"].astype(str).str.lower().str.contains(ss, na=False)
        ]
    hr_cols = [c for c in ["return_id", "customer_id", "risk_probability", "decision", "device_linked_accounts", "address_linked_accounts"] if c in hr_display.columns]
    if "risk_probability" in hr_display.columns:
        hr_display["Risk"] = (pd.to_numeric(hr_display["risk_probability"], errors="coerce").fillna(0.0) * 100).round(1).astype(str) + "%"
        hr_cols = [c for c in ["return_id", "customer_id", "Risk", "decision", "device_linked_accounts", "address_linked_accounts"] if c in hr_display.columns]
        st.dataframe(style_risk_table(hr_display[hr_cols], risk_percent_col="Risk"), hide_index=True)
    else:
        st.dataframe(style_risk_table(hr_display[hr_cols]), hide_index=True)

    cluster_view = normalized_cluster_table(cluster_df)
    if G is not None and G.number_of_nodes() > 0:
        st.subheader("Cluster Diagram")
        if not cluster_view.empty:
            cluster_options = cluster_view["cluster_id"].astype(str).tolist()
            selected_cluster = st.selectbox("Cluster to inspect", cluster_options, key="live_cluster_selector")
            cluster_row = cluster_view[cluster_view["cluster_id"].astype(str) == str(selected_cluster)].iloc[0]
            graph_nodes = cluster_row.get("nodes", [])
            fig = plot_cluster_graph(G, graph_nodes, title=f"{selected_cluster} — {int(cluster_row.get('customer_count', 0))} customers")
            enforce_plotly_light(fig)
            st.plotly_chart(fig, key="cluster_graph_live")
    else:
        st.info("Network data is not available yet. Waiting for live transactions with shared infrastructure identifiers.")

    if not cluster_view.empty:
        st.subheader("All Detected Clusters")
        cols = [c for c in ["cluster_id", "customer_count", "total_returns", "abusive_returns", "cluster_abuse_rate", "total_refund_value"] if c in cluster_view.columns]
        st.dataframe(cluster_view[cols], hide_index=True)


if hasattr(st, "fragment"):
    @st.fragment(run_every="1s")
    def _cluster_live_fragment(search_cluster: str):
        _live_poll_and_render(lambda: _render_cluster_dynamic(search_cluster))
else:
    def _cluster_live_fragment(search_cluster: str):
        _live_poll_and_render(lambda: _render_cluster_dynamic(search_cluster))


def _render_model_live():
    live_counts = st.session_state.get("live_cumulative_counts", {})
    live_total = int(live_counts.get("total", 0))
    live_auto = int(live_counts.get("AUTO_APPROVE", 0))
    live_verify = int(live_counts.get("VERIFY", 0))
    live_review = int(live_counts.get("MANUAL_REVIEW", 0))
    live_high = int(live_counts.get("HIGH_RISK", 0))
    lc = st.columns(5)
    lc[0].metric("Live Returns", f"{live_total:,}")
    lc[1].metric("Live Auto Approved", f"{live_auto:,}")
    lc[2].metric("Live Verification", f"{live_verify:,}")
    lc[3].metric("Live Manual Review", f"{live_review:,}")
    lc[4].metric("Live High Risk", f"{live_high:,}")
    last_poll = st.session_state.get("live_last_poll_at")
    if last_poll is not None:
        st.caption(f"Last live data update: {last_poll.strftime('%H:%M:%S.%f')[:-3]} UTC")


if hasattr(st, "fragment"):
    @st.fragment(run_every="1s")
    def _model_live_fragment():
        _live_poll_and_render(_render_model_live)
else:
    def _model_live_fragment():
        _live_poll_and_render(_render_model_live)


def _render_data_live_status():
    if st.session_state.get("live_connected") and st.session_state.get("live_config"):
        cfg = st.session_state["live_config"]
        status = live_status(cfg["base_url"], cfg["headers"])
        if status:
            s = st.columns(4)
            s[0].metric("Server Buffer", f"{status.get('buffered_records',0):,}")
            s[1].metric("Generation Rate", f"{status.get('rate_per_second',0):.1f}/sec")
            s[2].metric("Generator", "RUNNING" if status.get("running") else "STOPPED")
            s[3].metric("Latest Event", str(status.get("latest_generated_at", "—"))[-12:])
            st.caption(f"Live regime: {status.get('regime', '—')} | Event #{status.get('event_sequence', '—')} | Generator: {status.get('rate_per_second', 0):.1f}/sec")
        last_poll = st.session_state.get("live_last_poll_at")
        if last_poll is not None:
            st.caption(f"Live data update interval: 1000 ms | Last successful poll: {last_poll.strftime('%H:%M:%S.%f')[:-3]} UTC")
        if st.session_state.get("live_last_error"):
            st.warning(f"Last live poll error: {st.session_state['live_last_error']}")

if hasattr(st, "fragment"):
    @st.fragment(run_every="1s")
    def _data_live_status_fragment():
        _live_poll_and_render(_render_data_live_status)
else:
    def _data_live_status_fragment():
        _live_poll_and_render(_render_data_live_status)


def _chat_current_status():
    if st.session_state.get("live_connected") and st.session_state.get("live_config"):
        cfg = st.session_state["live_config"]
        try:
            return live_status(cfg["base_url"], cfg["headers"])
        except Exception:
            return None
    return None


def _execute_chat_action(action: dict[str, Any] | None):
    if not action:
        return None
    typ = action.get("type")
    if typ in {"start_live", "stop_live", "generate"} and not st.session_state.get("live_config"):
        return "Connect to a live server first from Data & Live Server."
    if typ == "start_live":
        cfg = st.session_state["live_config"]
        r = requests.post(cfg["base_url"].rstrip("/") + "/api/v1/returns/start", headers=cfg["headers"], timeout=5)
        r.raise_for_status(); st.session_state["live_connected"] = True; return "Live generator started."
    if typ == "stop_live":
        cfg = st.session_state["live_config"]
        r = requests.post(cfg["base_url"].rstrip("/") + "/api/v1/returns/stop", headers=cfg["headers"], timeout=5)
        r.raise_for_status(); return "Live generator stopped."
    if typ == "generate":
        cfg = st.session_state["live_config"]
        count = int(action.get("count", 10))
        r = requests.post(cfg["base_url"].rstrip("/") + "/api/v1/returns/generate", headers=cfg["headers"], params={"count": count}, timeout=10)
        r.raise_for_status(); return f"Generated {count} live transaction(s)."
    if typ == "open_investigator":
        st.session_state["page"] = "Return Investigator"
        return "Opening Return Investigator."
    if typ == "export":
        cfg = st.session_state.get("live_config")
        if not cfg:
            return "Connect to a live server first from Data & Live Server."
        try:
            url = cfg["base_url"].rstrip("/") + "/api/v1/returns/export.csv"
            r = requests.get(url, headers=cfg["headers"], timeout=20); r.raise_for_status()
            st.session_state["chat_export_blob"] = r.content
            return "Live CSV is ready in the download control below."
        except Exception as exc:
            return f"Export failed: {exc}"
    return None


# Cisco-styled chat surface. Streamlit gives each widget its own element container,
# so layout is anchored on marker elements inside real Streamlit containers rather
# than on <div> wrappers that Markdown's <p> parsing would break apart.
CHAT_CSS = """
<style>
  [data-testid="stAppViewContainer"] { --rs-pane-h: calc(100vh - 20rem); }

  /* The chat view owns the viewport: no page-level scrolling. */
  html:has(.rs-chat-anchor), body:has(.rs-chat-anchor) { overflow:hidden !important; }
  .stApp:has(.rs-chat-anchor) { overflow:hidden !important; }
  [data-testid="stAppScrollToBottomContainer"]:has(.rs-chat-anchor) { overflow:hidden !important; }
  [data-testid="stMainBlockContainer"]:has(.rs-chat-anchor) {
    height:100vh !important; max-height:100vh !important; overflow:hidden !important;
    /* Clear Streamlit's fixed 60px header, which would otherwise sit on the title. */
    padding-top:4.5rem !important; padding-bottom:5rem !important;
  }

  /* Header */
  .rs-chat-title { font-size:1.6rem; font-weight:700; letter-spacing:-.015em; color:#1A2432; line-height:1.2; }
  .rs-chat-subtitle { color:#546170; font-size:.88rem; margin-top:2px; }

  /* Message pane - the single scrollable surface.
     data-test-wrap="false" marks the real bordered container; a plain :has() would
     also match every ancestor block and stretch the whole page. */
  div[data-testid="stVerticalBlock"][data-test-wrap="false"]:has(.rs-chat-anchor) {
    /* Streamlit gives this container flex:1 1 0%, so the flex basis - not height -
       governs its size. Pin the flex item before setting an explicit height. */
    flex:0 0 auto !important;
    height:var(--rs-pane-h) !important; max-height:var(--rs-pane-h) !important;
    overflow-y:auto !important; overflow-x:hidden !important;
    border:1px solid #D6DBE1 !important; border-radius:4px !important;
    background:#fff !important; padding:10px 14px !important;
  }
  div[data-testid="stVerticalBlock"][data-test-wrap="false"]:has(.rs-chat-anchor)::-webkit-scrollbar { width:8px; }
  div[data-testid="stVerticalBlock"][data-test-wrap="false"]:has(.rs-chat-anchor)::-webkit-scrollbar-thumb { background:#C6CDD6; border-radius:8px; }

  /* Message bubbles */
  [data-testid="stChatMessage"] { padding:6px 0 !important; gap:10px !important; }
  [data-testid="stChatMessage"]  [data-testid="stChatAvatar"], [data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {
    border-radius:4px !important; border:1px solid #B9E2F3;
  }
  [data-testid="stChatMessageAvatarUser"] { background:#EAF6FC !important; color:#1264A3 !important; }
  [data-testid="stChatMessageAvatarAssistant"] { background:#1264A3 !important; color:#fff !important; border-color:#1264A3 !important; }
  [data-testid="stChatMessageContent"] { font-size:.93rem; line-height:1.55; color:#1A2432; }
  [data-testid="stChatMessage"] p { margin-bottom:.35em; }

  /* Memory / hindsight hint strip under an assistant bubble */
  .rs-hint { display:flex; flex-wrap:wrap; gap:6px; margin:2px 0 4px; }
  .rs-chip {
    display:inline-block; padding:2px 8px; border-radius:3px; font-size:.68rem; font-weight:700;
    letter-spacing:.06em; background:#F5F7FA; color:#546170; border:1px solid #D6DBE1;
  }
  .rs-chip.low { background:#E7F6EC; color:#17793A; border-color:#B7E3C6; }
  .rs-chip.medium { background:#FDF3DC; color:#8A6100; border-color:#F0DCA6; }
  .rs-chip.high { background:#FBE7E7; color:#A31F1F; border-color:#F0BDBD; }
  .rs-chip-sub { font-size:.68rem; color:#546170; font-weight:600; letter-spacing:.02em; }

  /* Empty state */
  .rs-chat-empty { text-align:center; padding:22px 8px 8px; }
  .rs-chat-empty-title { font-size:1.02rem; font-weight:700; color:#1A2432; }
  .rs-chat-empty-sub { color:#546170; font-size:.86rem; margin:5px 0 14px; }
  .rs-suggest { text-align:left; border:1px solid #D6DBE1 !important; border-radius:4px !important;
    background:#F5F7FA !important; color:#1264A3 !important; font-size:.84rem !important; font-weight:600 !important; }
  .rs-suggest:hover { background:#EAF6FC !important; border-color:#049FD9 !important; color:#049FD9 !important; }

  /* Settings drawer - scoped to the real container via data-test-wrap="false" so
     ancestor blocks are never turned into fixed-position drawers. */
  div[data-testid="stVerticalBlock"][data-test-wrap="false"]:has(.rs-settings-anchor) {
    position:fixed !important; right:1rem; top:4.5rem; width:min(360px,34vw);
    max-height:calc(100vh - 7rem); overflow-y:auto; z-index:9998;
    background:#fff !important; border:1px solid #D6DBE1 !important; border-radius:4px !important;
    padding:12px 14px !important; box-shadow:0 6px 18px rgba(26,36,50,.16);
  }
  .rs-settings-title { font-size:.95rem; font-weight:700; color:#1A2432; }

  /* Input pinned to the bottom of the content column */
  [data-testid="stChatInput"] {
    position:fixed !important; left:calc(var(--sidebar-width, 21rem) + 2rem);
    right:2rem; bottom:1rem; z-index:10000;
  }
  @media (max-width:900px) {
    .rs-chat-title { font-size:1.35rem; }
    div[data-testid="stVerticalBlock"][data-test-wrap="false"]:has(.rs-settings-anchor) { left:1rem; right:1rem; width:auto; }
    [data-testid="stChatInput"] { left:1rem; right:1rem; }
  }
</style>
"""

CHAT_SUGGESTIONS = [
    "Which returns are highest risk right now?",
    "Show the abuse-ring clusters in this dataset",
    "Summarise the model's business impact",
]

#: How each recommendation verdict reads in the chat hint strip.
_HINT_TONE = {
    "retune_thresholds": "medium",
    "keep_current_policy": "low",
    "insufficient_evidence": "low",
    "no_candidate": "high",
    "unavailable": "low",
}


def _render_memory_hint(memory: dict | None):
    """Show what the agent remembered and what hindsight says about the policy."""
    if not isinstance(memory, dict) or not memory:
        return
    chips: list[str] = []

    recalled = int(memory.get("recalled_turns") or 0)
    if recalled:
        chips.append(f'<span class="rs-chip">REMEMBERED {recalled} TURN{"S" if recalled != 1 else ""}</span>')

    logged = int(memory.get("decisions_logged") or 0)
    if logged:
        chips.append(f'<span class="rs-chip">LOGGED {logged} DECISION{"S" if logged != 1 else ""}</span>')

    rec = memory.get("recommendation") or {}
    verdict = str(rec.get("recommendation") or "")
    if verdict:
        label = {
            "retune_thresholds": "RETUNE ADVISED",
            "keep_current_policy": "POLICY HELD",
            "insufficient_evidence": "HINDSIGHT WARMING UP",
            "no_candidate": "NO VIABLE THRESHOLD",
            "unavailable": "HINDSIGHT UNAVAILABLE",
        }.get(verdict, verdict.upper())
        tone = _HINT_TONE.get(verdict, "low")
        sample = rec.get("sample_size")
        suffix = f' <span class="rs-chip-sub">{sample} resolved</span>' if sample else ""
        chips.append(f'<span class="rs-chip {tone}">{label}</span>{suffix}')

    if not chips:
        return
    st.markdown(f'<div class="rs-hint">{"".join(chips)}</div>', unsafe_allow_html=True)


def _reset_chat_session_state():
    """Bring the chat session into line with the current default provider.

    ``st.session_state`` persists across reloads, so a reader who started the
    dashboard under an earlier default can be left pointing at a provider that
    has no key and no server. Forcing the defaults here is safer than leaving
    the stale state in place.
    """
    if st.session_state.get("ai_provider") != "ReturnShield (Local)":
        st.session_state["ai_provider"] = "ReturnShield (Local)"
        st.session_state["ai_provider_last"] = None
        st.session_state.setdefault("ai_model", "returnshield/local")
        st.session_state.setdefault("ai_base_url", "http://localhost:8000")
        st.session_state["ai_api_key"] = ""
    st.session_state.setdefault("ai_model", "returnshield/local")
    st.session_state.setdefault("ai_base_url", "http://localhost:8000")
    st.session_state.setdefault("ai_api_key", "")


def _render_chat():
    """Cisco-styled chat view. Only the message pane scrolls."""
    st.markdown(CHAT_CSS, unsafe_allow_html=True)

    _reset_chat_session_state()

    head_l, head_r = st.columns([12, 1], vertical_alignment="center")
    with head_l:
        st.markdown(
            '<div class="rs-chat-title">ReturnShield AI Chat</div>'
            '<div class="rs-chat-subtitle">Ask anything. The assistant can use ReturnShield data and tools when relevant.</div>',
            unsafe_allow_html=True,
        )
    with head_r:
        if st.button("⚙", key="open_ai_settings", help="AI settings", width="stretch"):
            st.session_state["ai_settings_open"] = not st.session_state.get("ai_settings_open", False)
            st.rerun()

    if st.session_state.get("ai_settings_open", False):
        with st.container(border=True):
            st.markdown('<span class="rs-settings-anchor"></span>', unsafe_allow_html=True)
            s1, s2 = st.columns([5, 1], vertical_alignment="center")
            with s1:
                st.markdown('<div class="rs-settings-title">AI Settings</div>', unsafe_allow_html=True)
            with s2:
                if st.button("×", key="close_ai_settings", help="Close AI settings", width="stretch"):
                    st.session_state["ai_settings_open"] = False
                    st.rerun()
            # Allow a reader to switch providers; the session default is the local
            # path so a fresh load never silently reaches for an external key.
            # Allow a reader to switch providers; the session default is the local
            # path so a fresh load never silently reaches for an external key.
            providers = ["OpenRouter (Free)", "Groq (Free)", "Google Gemini", "Hugging Face", "Ollama (Local)", "OpenAI", "ReturnShield (Local)"]
            current = st.session_state.get("ai_provider", "ReturnShield (Local)")
            provider = st.selectbox("AI provider", providers, index=providers.index(current) if current in providers else 0, key="ai_provider_select")
            st.session_state["ai_provider"] = provider
            defaults = {
                "OpenRouter (Free)": ("openrouter/free", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
                "Groq (Free)": ("openai/gpt-oss-120b", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
                "Google Gemini": ("gemini-2.5-flash", "https://generativelanguage.googleapis.com/v1beta", "GEMINI_API_KEY"),
                "Hugging Face": ("openai/gpt-oss-120b:groq", "https://router.huggingface.co/v1", "HF_TOKEN"),
                "Ollama (Local)": ("gemma3", "http://localhost:11434/v1", ""),
                "OpenAI": ("gpt-5.6-luna", "https://api.openai.com/v1", "OPENAI_API_KEY"),
                "ReturnShield (Local)": ("returnshield/local", "http://localhost:8000", ""),
            }
            default_model, default_base, env_key = defaults[provider]
            if provider == "Ollama (Local)":
                st.info("No API key required. Ollama must be running locally.")
                st.session_state["ai_api_key"] = ""
            elif provider == "ReturnShield (Local)":
                st.info("Fully local, offline chat. Uses the ReturnShield Agent API and the ReturnShield local fallback, so no external AI key is needed.")
                st.session_state["ai_api_key"] = ""
            else:
                key_default = st.session_state.get("ai_api_key", os.getenv(env_key, ""))
                st.session_state["ai_api_key"] = st.text_input("API key", value=key_default, type="password", key="chat_ai_api_key")
            if st.session_state.get("ai_provider_last") != provider:
                st.session_state["ai_model"] = default_model
                st.session_state["ai_base_url"] = default_base
                st.session_state["ai_provider_last"] = provider
            if provider == "ReturnShield (Local)":
                # The provider already decides the behaviour. Forcing the two inputs to the
                # local defaults means a mis-typed model or bogus base URL cannot silently
                # break the offline path into the generic "Agent API unavailable" fallback.
                st.session_state["ai_model"] = "returnshield/local"
                st.session_state["ai_base_url"] = "http://localhost:8000"
            else:
                st.session_state["ai_model"] = st.text_input("Model", value=st.session_state.get("ai_model", default_model), key="chat_ai_model")
                st.session_state["ai_base_url"] = st.text_input("API base URL", value=st.session_state.get("ai_base_url", default_base), key="chat_ai_base_url")

    # Message pane - the only scrolling surface.
    with st.container(height=520, border=True):
        st.markdown('<span class="rs-chat-anchor"></span>', unsafe_allow_html=True)
        messages = st.session_state.setdefault("chat_messages", [])
        if not messages:
            st.markdown(
                '<div class="rs-chat-empty">'
                '<div class="rs-chat-empty-title">ReturnShield AI</div>'
                '<div class="rs-chat-empty-sub">Start a conversation, or pick one of these:</div>'
                "</div>",
                unsafe_allow_html=True,
            )
            for i, suggestion in enumerate(CHAT_SUGGESTIONS):
                if st.button(suggestion, key=f"chat_suggest_{i}", width="stretch"):
                    st.session_state["chat_pending"] = suggestion
        else:
            for msg in messages:
                role = "user" if msg.get("role") == "user" else "assistant"
                with st.chat_message(role):
                    st.markdown(msg.get("content", ""))
                    _render_memory_hint(msg.get("memory"))
                    table = msg.get("table")
                    if table:
                        try:
                            tdf = pd.DataFrame(table)
                            if "risk_probability" in tdf.columns:
                                # Match the dashboard: show risk as a readable percentage.
                                tdf["Risk"] = (
                                    pd.to_numeric(tdf["risk_probability"], errors="coerce").fillna(0.0) * 100
                                ).round(1).astype(str) + "%"
                                tdf = tdf.drop(columns=["risk_probability"])
                            st.dataframe(tdf, hide_index=True)
                        except Exception:
                            pass

    typed = st.chat_input("Message ReturnShield AI…", key="returnshield_chat_input")
    prompt = (typed or st.session_state.pop("chat_pending", None) or "").strip()
    if not prompt:
        return

    ctx = st.session_state["chat_context"]
    live = _chat_current_status()
    active_df = st.session_state.get("active_predictions", pd.DataFrame())
    cfg = st.session_state.get("live_config") or {"base_url": "http://localhost:8000", "headers": {}}
    agent_url = cfg["base_url"].rstrip("/") + "/api/v1/agent/chat"
    context_payload = {
        "last_return_id": getattr(ctx, "last_return_id", None),
        "last_customer_id": getattr(ctx, "last_customer_id", None),
        "last_intent": getattr(ctx, "last_intent", None),
        "last_answer": getattr(ctx, "last_answer", None),
        "history": getattr(ctx, "history", []),
        "ai_config": {
            "provider": st.session_state.get("ai_provider", "OpenRouter (Free)"),
            "api_key": st.session_state.get("ai_api_key", ""),
            "model": st.session_state.get("ai_model", "openrouter/free"),
            "base_url": st.session_state.get("ai_base_url", "https://openrouter.ai/api/v1"),
        },
    }
    # Project onto the columns the agent actually reads. Posting the full frame
    # (48 columns x thousands of rows) cost ~33 s per turn and timed out.
    if isinstance(active_df, pd.DataFrame) and not active_df.empty:
        cols = [c for c in AGENT_COLUMNS if c in active_df.columns]
        send_df = active_df[cols].head(1500)
    else:
        send_df = pd.DataFrame()
    # to_json -> json.loads, because to_dict("records") leaves numpy Timestamps
    # that json.dumps cannot encode -- that TypeError silently sent every
    # dashboard chat turn to the local fallback instead of the API. date_format
    # matters too: the default emits epoch millis, which the API would read as
    # 1970 dates in any time series.
    records = json.loads(send_df.to_json(orient="records", date_format="iso")) if not send_df.empty else []
    payload = {
        "message": prompt,
        "context": context_payload,
        "records": records,
        "report": report or {},
        "live_status": live,
    }
    with st.spinner("Thinking…"):
        try:
            rr = requests.post(agent_url, headers=cfg.get("headers", {}), json=payload, timeout=(20, 180))
            rr.raise_for_status()
            api_res = rr.json()
            res = {"answer": api_res.get("answer"), "data": pd.DataFrame(api_res.get("data") or []), "action": api_res.get("action") or {}, "intent": api_res.get("intent"), "confidence": api_res.get("confidence", 0)}
            action_note = api_res.get("action_result")
            api_memory = api_res.get("memory")
        except Exception as exc:
            api_memory = None
            try:
                # The API reattaches shared-infrastructure ids before it answers; the
                # in-process path has to do the same or cluster questions report
                # "no identifiers" the moment the API is unreachable.
                res = st.session_state["chat_agent"].respond(
                    prompt, ctx, attach_infrastructure_ids(active_df, str(ROOT / "data" / "raw")), report, live
                )
                action_note = f"Agent API unavailable ({type(exc).__name__}: {str(exc)[:120]}); used the ReturnShield local fallback agent."
            except Exception:
                res = {"answer": f"I couldn't reach the ReturnShield Agent API: {exc}", "data": None, "action": {}}
                action_note = None

    answer = str(res.get("answer") or "I couldn't produce an answer for that request.")
    if action_note and action_note not in answer:
        answer += f"\n\n_{action_note}_"
    st.session_state["chat_messages"].append({"role": "user", "content": prompt})
    msg = {"role": "assistant", "content": answer}
    if api_memory:
        msg["memory"] = api_memory
    if isinstance(res.get("data"), pd.DataFrame) and not res["data"].empty:
        msg["table"] = res["data"].head(100).to_dict("records")
    st.session_state["chat_messages"].append(msg)
    action = res.get("action") or {}
    if action.get("type") == "open_investigator":
        st.session_state["page"] = "Return Investigator"
    st.rerun()


# --------------------------------------------------------------- decision API
#
# The operations screens act on the product, not on a dataset: they read and
# write the same Decision API a merchant's checkout service would call, through
# the published client. Everything a fraud manager does here - queueing a case,
# resolving it, recording the outcome, moving a threshold - is an audited API
# call that fires a webhook and lands in the review log, not a local widget
# that only exists in the browser.

DECISION_API_URL = os.getenv("RETURNSHIELD_API_URL", "http://localhost:8000").rstrip("/")
DECISION_API_KEY = os.getenv("RETURNSHIELD_API_KEY", "").strip()

#: Resolutions a case can be closed with. The API accepts any string; these are
#: the ones a fraud team uses, and they line up with the outcome vocabulary the
#: feedback endpoint understands.
CASE_RESOLUTIONS = {
    "confirm_abuse": "Confirm abuse - deny the refund, block the account",
    "escalate": "Escalate - hold for a second reviewer",
    "refund": "Refund allowed - the claim was legitimate",
    "reject_claim": "Claim rejected - good customer, no abuse",
}


@st.cache_resource(show_spinner=False)
def decision_client():
    from sdk.returnshield import ReturnShieldClient

    return ReturnShieldClient(base_url=DECISION_API_URL, api_key=DECISION_API_KEY or None)


def decision_call(label, fn, *args, **kwargs):
    """One API call, reported as ``(ok, value_or_message)``.

    Failures are returned rather than raised: a stopped API server should leave
    the analyst with a readable reason, not a Streamlit traceback.
    """
    from sdk.returnshield import ReturnShieldError

    try:
        return True, fn(*args, **kwargs)
    except ReturnShieldError as exc:
        return False, f"{label} failed - HTTP {exc.status_code} `{exc.code}`: {exc.message}"
    except Exception as exc:  # connection refused, timeout, malformed body
        return False, f"{label} could not reach {DECISION_API_URL} ({type(exc).__name__}: {exc})."


def require_decision_api() -> dict:
    """Every operations screen needs the API; say how to start it, then stop."""
    ok, health = decision_call("Health check", decision_client().health)
    if not ok:
        st.error(health)
        st.markdown(
            "These screens work through the Decision API. Start it in another terminal, "
            "then reload this page:"
        )
        st.code("python -m src.api", language="bash")
        st.caption(
            f"Base URL `{DECISION_API_URL}` · key "
            f"{'configured' if DECISION_API_KEY else 'not required (auth disabled)'} · "
            f"model {health.get('model_kind', 'unknown') if ok else 'unreachable'}"
        )
        st.stop()
    return health


def score_request_from_row(row) -> dict:
    """Turn a scored dataset row into a ``POST /decisions/score`` body.

    Only the fields the row actually carries are sent; the API fills the rest
    with its documented defaults. The score that comes back is therefore the
    API's own, which is the one the queue and every later decision refer to.
    """
    def number(column, cast, default):
        value = row.get(column, None)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        try:
            return cast(value)
        except (TypeError, ValueError):
            return default

    order_value = number("order_value", float, 0.0)
    payload = {
        "return_id": str(row.get("return_id", "")),
        "order_id": str(row.get("order_id", "")),
        "customer_id": str(row.get("customer_id", "")),
        "product_category": str(row.get("product_category", "electronics")),
        "return_reason": str(row.get("return_reason", "damaged")),
        "order_value": order_value,
        "product_price": number("return_value", float, order_value),
        "returns_7d": number("returns_7d", int, 0),
        "returns_30d": number("returns_30d", int, 0),
        "hours_to_return": number("hours_to_return", float, 3.5),
        "refund_amount_90d": number("refund_amount_90d", float, 0.0),
        "device_linked_accounts": number("device_linked_accounts", int, 1),
        "address_linked_accounts": number("address_linked_accounts", int, 1),
    }
    return payload


def minutes_since(timestamp) -> float:
    try:
        return max(0.0, (time.time() - float(timestamp)) / 60.0)
    except (TypeError, ValueError):
        return 0.0


def render_case_queue():
    """The worklist: cases a person has to look at, ranked by what they cost."""
    st.title("Case Queue")
    st.caption(
        "Every return the policy sent to a person, ranked by the money at risk. "
        "Resolving a case is an audited API call - it fires a signed webhook and feeds "
        "the recommendation engine."
    )
    health = require_decision_api()
    client = decision_client()

    f1, f2, f3 = st.columns([2, 2, 1])
    search = f1.text_input("Search return ID", placeholder="R006011", key="queue_search")
    state = f2.selectbox("Queue", ["pending", "resolved", "all"], key="queue_state")
    page_size = f3.selectbox("Cases per page", [25, 50, 100, 200], index=1, key="queue_limit")
    refresh = st.button("Refresh", key="queue_refresh")

    ok, page = decision_call("Listing the queue", client.reviews, state=state, limit=int(page_size))
    if not ok:
        st.error(page)
        st.stop()
    rows = page.get("data") or []
    if search.strip():
        needle = search.strip().lower()
        rows = [r for r in rows if needle in str(r.get("return_id", "")).lower()]

    pending_total = 0
    ok_p, pending_page = decision_call("Counting pending cases", client.reviews, state="pending", limit=1)
    if ok_p:
        pending_total = int(pending_page.get("total", 0))
    cost_at_risk = sum(float(r.get("expected_cost") or 0.0) for r in rows if r.get("state") == "pending")
    oldest = max((minutes_since(r.get("created_at")) for r in rows if r.get("state") == "pending"), default=0.0)

    k = st.columns(4)
    k[0].metric("Cases pending", f"{pending_total:,}")
    k[1].metric("Exposure in this view", f"₹{cost_at_risk:,.0f}")
    k[2].metric("Oldest open case", f"{oldest:,.0f} min")
    k[3].metric("Policy in force", health.get("policy", {}).get("review_threshold") and
                f"T2={health['policy']['review_threshold']:.3f}")

    st.divider()
    with st.expander("Screen returns into the queue", expanded=not rows):
        st.caption(
            "The queue only fills when returns are put through the decision path. "
            "Screening takes the riskiest rows from the active dataset and asks the API to "
            "score each one; anything that comes back as review or block is queued."
        )
        s1, s2 = st.columns([3, 1])
        screen_count = s1.slider("Returns to screen", 5, 250, 25, step=5, key="queue_screen_count")
        if s2.button("Screen now", key="queue_screen_run", width="stretch"):
            _screen_into_queue(client, int(screen_count), rows)

    if not rows:
        st.info("No cases in this view. Screen some returns above, or resolve the current queue to empty it.")
        return

    table = pd.DataFrame([
        {
            "case": r.get("id"),
            "return_id": r.get("return_id"),
            "state": r.get("state"),
            "action": r.get("action"),
            "score": float(r.get("risk_probability") or 0.0),
            "exposure": float(r.get("expected_cost") or 0.0),
            "waiting_min": round(minutes_since(r.get("created_at")), 1),
            "policy": r.get("policy_version"),
        }
        for r in rows
    ])
    st.dataframe(
        style_risk_table(table, "score").format({"score": "{:.1%}", "exposure": "₹{:,.0f}"}),
        hide_index=True,
        height=320,
        key="queue_table",
    )

    by_case = {int(r["id"]): r for r in rows if r.get("id") is not None}
    selected = st.selectbox(
        "Open a case",
        options=list(by_case),
        format_func=lambda cid: f"#{cid} · {by_case[cid]['return_id']} · "
        f"{by_case[cid].get('action')} · ₹{float(by_case[cid].get('expected_cost') or 0):,.0f}",
        key="queue_selected",
    )
    case = by_case[selected]
    render_case_file(case, client)


def _screen_into_queue(client, count, existing_rows):
    """Push the riskiest unqueued rows from the active set through the API."""
    frame = predictions if "risk_probability" in predictions.columns and not predictions.empty else default_predictions
    if frame.empty:
        st.warning("No scored rows available to screen.")
        return
    queued = {str(r.get("return_id")) for r in existing_rows if r.get("state") == "pending"}
    ordered = frame.sort_values("risk_probability", ascending=False)
    candidates = [row for _, row in ordered.iterrows() if str(row.get("return_id")) not in queued][:count]
    if not candidates:
        st.info("Every high-risk return in this dataset is already queued.")
        return
    progress = st.progress(0.0, text=f"Screening {len(candidates)} returns through {DECISION_API_URL}")
    reviewed = blocked = failed = 0
    for index, row in enumerate(candidates):
        ok, result = decision_call("Screening", client.score_decision, score_request_from_row(row))
        if not ok:
            failed += 1
        elif result.action in {"review", "block"}:
            reviewed += 1
            if result.action == "block":
                blocked += 1
        progress.progress((index + 1) / len(candidates), text=f"Screening {index + 1}/{len(candidates)}")
    progress.empty()
    st.success(
        f"Screened {len(candidates):,} returns · {reviewed:,} queued for review "
        f"({blocked:,} at block level) · {failed:,} failed."
    )
    st.rerun()


def render_case_file(case, client):
    """One case: the evidence behind it, and the two decisions a human can make."""
    reasons = case.get("reasons") or []
    header = st.columns([3, 2, 2, 2])
    header[0].metric("Risk score", f"{float(case.get('risk_probability') or 0.0):.1%}")
    header[1].metric("Action", str(case.get("action") or "unknown").upper())
    header[2].metric("Exposure", f"₹{float(case.get('expected_cost') or 0.0):,.0f}")
    header[3].metric("Policy", str(case.get("policy_version") or "unknown"))

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Why this was flagged**")
        if reasons:
            fig = go.Figure(go.Bar(
                x=[float(r.get("contribution") or 0.0) for r in reasons][::-1],
                y=[str(r.get("feature") or "").replace("num__", "").replace("cat__", "") for r in reasons][::-1],
                orientation="h",
                marker=dict(color=[
                    "#D13434" if float(r.get("contribution") or 0.0) > 0 else "#24AB4C" for r in reasons
                ][::-1]),
                hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
            ))
            fig.update_xaxes(title_text="Contribution to model risk", zeroline=True)
            enforce_plotly_light(fig)
            fig.update_layout(title="Ranked reasons", height=max(160, 34 * len(reasons) + 90), margin=dict(t=44, r=16, b=40, l=8))
            st.plotly_chart(fig, key=f"reasons_{case.get('id')}")
        else:
            st.caption("No ranked reasons were stored with this case.")

        st.markdown("**What we decided about this return before**")
        ok, history = decision_call("Loading history", client.return_history, str(case.get("return_id")))
        if ok:
            prior = history.get("prior_decisions") or []
            if prior:
                st.dataframe(pd.DataFrame(prior), hide_index=True, key=f"prior_{case.get('id')}")
            else:
                st.caption("First time this return has been through a decision.")
        else:
            st.warning(history)

    with right:
        st.markdown("**Close the case**")
        st.caption(
            "Resolving records who decided and fires a signed `review.resolved` webhook. "
            "Recording the outcome is a separate call that closes the learning loop - "
            "do both and the recommendation engine prices this case."
        )
        resolution = st.selectbox(
            "Resolution", list(CASE_RESOLUTIONS), format_func=lambda key: f"{key} - {CASE_RESOLUTIONS[key]}",
            key=f"resolution_{case.get('id')}",
        )
        actor = st.text_input("Analyst", value="analyst@merchant", key=f"actor_{case.get('id')}")
        note = st.text_input("Note (optional)", key=f"note_{case.get('id')}")

        if st.button("Resolve case", key=f"resolve_{case.get('id')}", width="stretch"):
            if not actor.strip():
                st.error("An analyst identity is required - the resolution is the audit record.")
            else:
                ok, result = decision_call("Resolving", client.resolve_review, int(case["id"]), resolution, actor=actor.strip(), note=note.strip() or None)
                if not ok:
                    st.error(result)
                else:
                    resolved = result.get("review", {})
                    delivered = result.get("webhooks") or []
                    st.success(
                        f"Case #{case['id']} resolved as `{resolved.get('resolution')}` "
                        f"by {resolved.get('actor')} · {len(delivered)} webhook(s) delivered."
                    )

        abusive = resolution in {"confirm_abuse", "escalate"}
        if st.button(
            "Record outcome (closes the loop)" if abusive else "Record outcome (legitimate)",
            key=f"outcome_{case.get('id')}", width="stretch",
        ):
            outcome = "abusive" if abusive else "legitimate"
            ok, result = decision_call("Recording outcome", client.record_decision, str(case.get("return_id")), outcome, actor=actor.strip() or None)
            if not ok:
                st.error(result)
            else:
                st.success(
                    f"Outcome `{result.get('outcome')}` recorded · "
                    f"{result.get('resolved_updated', 0)} open call(s) resolved, "
                    f"{result.get('resolved_inserted', 0)} recorded."
                )


def render_outcome_ledger():
    """The loop end to end: what was decided, what happened, what is still open."""
    st.title("Outcome Ledger")
    st.caption(
        "The right-hand half of the product: every decision that was logged, the outcome it "
        "actually got, and the policy that the evidence now recommends. This is the loop a "
        "fraud team closes - and the only place the whole story is visible at once."
    )
    require_decision_api()
    client = decision_client()

    ok_h, hindsight = decision_call("Loading hindsight", client.hindsight, 25)
    ok_r, recommendation = decision_call("Loading the recommendation", client.recommendations)
    ok_e, events = decision_call("Loading ingested events", client.list_events, limit=1)
    ok_p, pending = decision_call("Counting open cases", client.reviews, state="pending", limit=1)
    ok_d, resolved_cases = decision_call("Counting closed cases", client.reviews, state="resolved", limit=1)
    if not ok_h:
        st.error(hindsight)
        st.stop()

    summary = hindsight.get("summary") or {}
    stages = [
        ("Events ingested", int((events or {}).get("total", 0))),
        ("Decisions logged", int(summary.get("total_decisions") or 0)),
        ("Flagged for a human", int(summary.get("flagged") or 0)),
        ("Cases resolved", int((resolved_cases or {}).get("total", 0))),
        ("Outcomes fed back", int(summary.get("resolved") or 0)),
    ]
    k = st.columns(len(stages))
    for column, (label, value) in zip(k, stages):
        column.metric(label, f"{value:,}")

    st.caption(summary.get("note") or "")
    st.divider()

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**The loop, stage by stage**")
        fig = go.Figure(go.Bar(
            x=[value for _label, value in stages],
            y=[label for label, _value in stages],
            orientation="h",
            marker=dict(color="#049FD9"),
            text=[f"{value:,}" for _label, value in stages],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:,}<extra></extra>",
        ))
        enforce_plotly_light(fig)
        fig.update_layout(height=250, title="Volume by stage", margin=dict(t=44, r=40, b=32, l=8))
        fig.update_xaxes(title_text="Rows")
        st.plotly_chart(fig, key="ledger_stages")

        st.markdown("**Outcomes recorded against real returns**")
        feedback = hindsight.get("recent_feedback") or []
        if feedback:
            frame = pd.DataFrame(feedback)
            frame["resolved_at"] = pd.to_datetime(frame.get("resolved_at"), unit="s", errors="coerce")
            frame["abuse"] = frame.get("actual_abusive", pd.Series(dtype=int)).map(
                {1: "abusive", 0: "legitimate"}
            )
            st.dataframe(
                frame[["return_id", "decision", "risk_probability", "abuse", "correct", "actual_merchant_loss", "resolved_at", "source"]]
                .sort_values("resolved_at", ascending=False)
                .style.format({"risk_probability": "{:.1%}"}),
                hide_index=True, height=300, key="ledger_feedback",
            )
        else:
            st.caption("No outcomes recorded yet. Resolve a case in the Case Queue, then record its outcome.")

    with right:
        st.markdown("**How the decisions turned out**")
        outcome_k = st.columns(2)
        outcome_k[0].metric("Precision when flagged", _pct(summary.get("precision_flagged")))
        outcome_k[1].metric("Recall on abuse", _pct(summary.get("recall_on_abuse")))
        outcome_k[0].metric("Abuse we let through", f"{int(summary.get('false_negative') or 0):,}")
        outcome_k[1].metric("Loss from missed abuse", f"₹{float(summary.get('merchant_loss_from_missed_abuse') or 0.0):,.0f}")
        if summary.get("by_decision"):
            st.dataframe(
                pd.DataFrame(summary["by_decision"]).T.reset_index().rename(columns={"index": "decision"}),
                hide_index=True, key="ledger_by_decision",
            )

        st.markdown("**What the evidence recommends**")
        if ok_r:
            rec = recommendation.get("recommendation") or {}
            verdict = rec.get("recommendation", "unknown")
            colours = {
                "retune_thresholds": "warning",
                "keep_current_policy": "success",
                "insufficient_evidence": "info",
                "no_candidate": "info",
                "unavailable": "warning",
            }
            getattr(st, colours.get(verdict, "info"))(f"**{verdict.replace('_', ' ')}** · {rec.get('rationale', '')}")
            best = rec.get("recommended_policy") or rec.get("best_candidate")
            if best:
                s = st.columns(3)
                s[0].metric("Recommended T1", f"{float(best['verify_threshold']):.3f}")
                s[1].metric("Recommended T2", f"{float(best['review_threshold']):.3f}")
                s[2].metric("Cost per call", f"₹{float(best['expected_cost_per_call']):,.2f}")
                st.caption(f"Priced over {rec.get('sample_size', 0)} resolved calls.")
        else:
            st.warning(recommendation)

        st.markdown("**Outbound callbacks**")
        ok_v, deliveries = decision_call("Loading the delivery log", client.deliveries, 15)
        if ok_v and deliveries.get("data"):
            log = pd.DataFrame(deliveries["data"])
            log["created_at"] = pd.to_datetime(log.get("created_at"), unit="s", errors="coerce")
            st.dataframe(
                log[["event_type", "status", "attempts", "url", "created_at"]]
                .sort_values("created_at", ascending=False),
                hide_index=True, height=220, key="ledger_deliveries",
            )
            st.caption("Register a destination with `RETURNSHIELD_WEBHOOK_URL` / `RETURNSHIELD_WEBHOOK_SECRET`.")
        else:
            st.caption("No webhook destination configured, so nothing has been delivered yet.")


def _pct(value) -> str:
    return "n/a" if value is None else f"{float(value):.1%}"


def render_threshold_studio():
    """Move the operating point without a deploy, and see what it costs first."""
    st.title("Threshold Studio")
    st.caption(
        "Thresholds a fraud manager can change while the incident is still running. "
        "Every change is priced before it is committed, recorded with who did it, and can be "
        "rolled back to the previous operating point."
    )
    require_decision_api()
    client = decision_client()

    ok_live, live = decision_call("Loading the live policy", client.policy_versions)
    if not ok_live:
        st.error(live)
        st.stop()

    active_t1 = float(live["verify_threshold"])
    active_t2 = float(live["review_threshold"])
    costs = live.get("costs") or {}
    st.caption(
        f"Live now: verify **T1={active_t1:.3f}**, review **T2={active_t2:.3f}** "
        f"(policy version {live.get('active_version')}) · policy file `{live.get('policy_file')}`"
    )

    ok_r, recommendation = decision_call("Loading the recommendation", client.recommendations)
    recommended = ((recommendation or {}).get("recommendation") or {}).get("recommended_policy")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Merchant cost model**")
        st.caption("What a false positive and a missed abusive return actually cost you. "
                   "These assumptions are what every number on this page is priced with.")
        fp = st.slider("False positive cost (₹)", 50, 2000, int(costs.get("false_positive", 250)), 50, key="studio_fp")
        fn = st.slider("Missed abuse loss (₹)", 500, 8000, int(costs.get("false_negative", 2000)), 100, key="studio_fn")
    with c2:
        st.markdown("**Candidate operating point**")
        st.caption("Below T1 the return is allowed. Between T1 and T2 it is warned. "
                   "Above T2 it goes to a person.")
        t1 = st.slider("Verify threshold (T1)", 0.0, 0.6, active_t1, 0.005, key="studio_t1")
        t2 = st.slider("Review threshold (T2)", 0.05, 0.99, active_t2, 0.005, key="studio_t2")
        if recommended:
            if st.button(
                f"Use recommendation (T1={float(recommended['verify_threshold']):.3f}, "
                f"T2={float(recommended['review_threshold']):.3f})",
                key="studio_use_reco", width="stretch",
            ):
                st.session_state["studio_t1"] = float(recommended["verify_threshold"])
                st.session_state["studio_t2"] = float(recommended["review_threshold"])
                st.rerun()

    candidate_costs = {
        "false_positive": float(fp), "false_negative": float(fn),
        "verification": float(costs.get("verification", 40.0)),
        "manual_review": float(costs.get("manual_review", 60.0)),
    }
    ok_p, preview = decision_call("Previewing", client.preview_policy, float(t1), float(t2), costs=candidate_costs)

    if not ok_p:
        st.error(preview)
    elif not preview.get("evaluation_available"):
        st.warning(
            "The API has no held-out scored rows to price this against, so the comparison "
            "below is unavailable. The change can still be applied, but unpriced."
        )
    else:
        now_eval, next_eval = preview["active_evaluation"], preview["candidate_evaluation"]
        delta = preview.get("delta") or {}
        rows = [
            ("Expected cost", f"₹{now_eval['expected_cost']:,.0f}", f"₹{next_eval['expected_cost']:,.0f}", f"₹{delta.get('expected_cost', 0):+,.0f}"),
            ("Cost per call", f"₹{now_eval['cost_per_call']:,.2f}", f"₹{next_eval['cost_per_call']:,.2f}", ""),
            ("Abuse caught", f"{now_eval['abuse_caught']} of {now_eval['abuse_total']}", f"{next_eval['abuse_caught']} of {next_eval['abuse_total']}", ""),
            ("Abuse missed", f"{now_eval['false_negatives']}", f"{next_eval['false_negatives']}", f"{delta.get('false_negatives', 0):+,}"),
            ("Good customers held", f"{now_eval['false_positives']}", f"{next_eval['false_positives']}", f"{delta.get('false_positives', 0):+,}"),
            ("Allow / warn / review / block", f"{now_eval['allow_count']} / {now_eval['warn_count']} / {next_eval['review_count']} / {now_eval['block_count']}", f"{next_eval['allow_count']} / {next_eval['warn_count']} / {next_eval['review_count']} / {next_eval['block_count']}", ""),
            ("Cases for a person", f"{now_eval['review_volume']}", f"{next_eval['review_volume']}", f"{delta.get('review_volume', 0):+,}"),
        ]
        st.caption(
            f"Priced on {now_eval['rows']:,} held-out returns, {now_eval['prevalence']:.1%} of which are abusive."
        )
        st.dataframe(
            pd.DataFrame(rows, columns=["On the held-out set", "Live now", "Candidate", "Change"])
            .style.format({"Live now": "{}"}),
            hide_index=True, key="studio_compare",
        )
        if abs(float(delta.get("expected_cost_pct") or 0.0)) < 0.05 and float(t1) != active_t1:
            st.warning(
                "This moves the working point for less than 5% of expected cost. Moving "
                "thresholds for its own sake is how teams end up with neither the old nor the "
                "new behaviour."
            )
        if float(next_eval["review_volume"]) > float(now_eval["review_volume"]) * 2 and now_eval["review_volume"] > 0:
            st.warning(
                f"This more than doubles the manual review queue "
                f"({now_eval['review_volume']} → {next_eval['review_volume']} cases on the same sample). "
                f"Work it before you ship it."
            )

    st.divider()
    a1, a2 = st.columns([3, 2])
    with a1:
        st.markdown("**Commit the change**")
        actor = st.text_input("Analyst", value="analyst@merchant", key="studio_actor")
        note = st.text_input("Why (this is the audit note)", key="studio_note")
        if st.button("Save as a new version", key="studio_draft", width="stretch"):
            if not actor.strip():
                st.error("Record who is making the change.")
            else:
                ok, created = decision_call(
                    "Saving the version", client.create_policy_version,
                    float(t1), float(t2), costs=candidate_costs, actor=actor.strip(), note=note.strip() or None,
                )
                if not ok:
                    st.error(created)
                else:
                    st.success(f"Version {created['version']['version']} recorded. It changes nothing until it is activated.")
        if st.button("Activate the latest draft", key="studio_activate", width="stretch"):
            drafts = [v for v in live.get("versions") or [] if v.get("state") == "draft"]
            if not drafts:
                st.info("There is no draft to activate. Save one above first.")
            else:
                target = drafts[-1]
                ok, activated = decision_call("Activating", client.activate_policy, int(target["version"]), actor=actor.strip() or None)
                if not ok:
                    st.error(activated)
                else:
                    st.success(
                        f"Version {activated['policy']['active_version']} is live: verify "
                        f"{activated['policy']['verify_threshold']:.3f}, review "
                        f"{activated['policy']['review_threshold']:.3f}. The next score already uses it."
                    )
                    st.rerun()
        if st.button("Roll back to the previous operating point", key="studio_rollback", width="stretch"):
            ok, rolled = decision_call("Rolling back", client.rollback_policy, actor=actor.strip() or None)
            if not ok:
                st.error(rolled)
            else:
                st.success(
                    f"Rolled back from v{rolled['rolled_back_from']} to v{rolled['policy']['active_version']} "
                    f"(verify {rolled['policy']['verify_threshold']:.3f}, review {rolled['policy']['review_threshold']:.3f})."
                )
                st.rerun()

    with a2:
        st.markdown("**Policy history**")
        versions = live.get("versions") or []
        if versions:
            st.dataframe(
                pd.DataFrame([
                    {
                        "version": v.get("version"),
                        "state": v.get("state"),
                        "T1": float(v.get("verify_threshold") or 0.0),
                        "T2": float(v.get("review_threshold") or 0.0),
                        "by": v.get("actor"),
                        "note": v.get("note"),
                    }
                    for v in versions
                ]),
                hide_index=True, height=300, key="studio_versions",
            )
            st.caption("The shipped policy is version 1, so there is always somewhere to roll back to.")
        else:
            st.caption("No version history yet.")


# Live auto-refresh for the Decision API pages.
#
# These read the same API server as the overview graphs, but on human timescales:
# cases get resolved, outcomes get recorded, thresholds get activated. The 1s
# cadence on the returns pages would only re-fetch identical rows here, so they
# refresh every 5s - still live to the eye, without hammering the API.
_DECISION_REFRESH = "5s"


def _live_stamp():
    """The clock that tells an analyst this page is reading live, not a snapshot."""
    now = pd.Timestamp.now(tz="UTC")
    st.session_state["decision_last_refresh"] = now
    st.caption(f"Auto-refresh {_DECISION_REFRESH} · last read {now.strftime('%H:%M:%S')} UTC")


if hasattr(st, "fragment"):
    @st.fragment(run_every=_DECISION_REFRESH)
    def _case_queue_live():
        render_case_queue()
        _live_stamp()

    @st.fragment(run_every=_DECISION_REFRESH)
    def _outcome_ledger_live():
        render_outcome_ledger()
        _live_stamp()

    @st.fragment(run_every=_DECISION_REFRESH)
    def _threshold_studio_live():
        render_threshold_studio()
        _live_stamp()
else:
    _case_queue_live = render_case_queue
    _outcome_ledger_live = render_outcome_ledger
    _threshold_studio_live = render_threshold_studio


# Static page chrome is rendered once. Only live data components below are refreshed.
if page == "AI Chat":
    _render_chat()

elif page == "Case Queue":
    _case_queue_live()

elif page == "Outcome Ledger":
    _outcome_ledger_live()

elif page == "Threshold Studio":
    _threshold_studio_live()

elif page == "Operations Overview":
    st.title("ReturnShield Operations Overview")
    if st.session_state.get("live_connected") and st.session_state.get("live_config"):
        cfg = st.session_state["live_config"]
        st.caption(f"Source: `{cfg['base_url']}{cfg['endpoint']}`")
        cfg=st.session_state["live_config"]
        _live_browser_component("Operations",cfg["base_url"],cfg["endpoint"],cfg["headers"].get("Authorization",""),policy["verify_threshold"],policy["review_threshold"],"operations",height=900)
    else:
        st.caption(f"Source: **{st.session_state['dataset_name']}**")
        # No live connection: use the same rendering once without a timer.
        _render_operations_dynamic()

elif page == "Return Investigator":
    st.title("Single Return Investigation")
    st.caption("Live return data updates in place without rerunning the Streamlit page.")
    if st.session_state.get("live_connected") and st.session_state.get("live_config"):
        cfg=st.session_state["live_config"]
        _live_browser_component("Investigator",cfg["base_url"],cfg["endpoint"],cfg["headers"].get("Authorization",""),policy["verify_threshold"],policy["review_threshold"],"investigator",height=840)
    else:
        q1,q2,q3=st.columns([2,1,1])
        search=q1.text_input("Search Return Request",placeholder="Return ID or Customer ID",key="investigator_search")
        page_size=q2.selectbox("Requests per page",[50,100,250,500,1000],index=1,key="investigator_page_size")
        page_num=q3.number_input("Page",min_value=1,value=1,step=1,key="investigator_page_num")
        _render_investigator_dynamic(search,int(page_size),int(page_num))

elif page == "Abuse Ring Explorer":
    st.title("Abuse Ring & Cluster Explorer")
    if st.session_state.get("live_connected") and st.session_state.get("live_config"):
        cfg=st.session_state["live_config"]
        _live_browser_component("Cluster",cfg["base_url"],cfg["endpoint"],cfg["headers"].get("Authorization",""),policy["verify_threshold"],policy["review_threshold"],"cluster",height=980)
    else:
        search_cluster=st.text_input("Find an account in the high-risk list",placeholder="Customer ID or Return ID",key="cluster_search")
        _render_cluster_dynamic(search_cluster)

elif page == "Model Evaluation & ROI":
    st.title("Model Evaluation & Business Loss Analysis")
    eval_predictions = predictions if not predictions.empty and "decision" in predictions.columns else default_predictions
    total_n = len(eval_predictions)
    action_counts = eval_predictions["decision"].value_counts() if "decision" in eval_predictions.columns else pd.Series(dtype=int)
    m = st.columns(4)
    m[0].metric("Auto Approve Rate", f"{action_counts.get('AUTO_APPROVE',0)/max(total_n,1):.1%}")
    m[1].metric("Verify Rate", f"{action_counts.get('VERIFY',0)/max(total_n,1):.1%}")
    m[2].metric("Manual Review Rate", f"{action_counts.get('MANUAL_REVIEW',0)/max(total_n,1):.1%}")
    m[3].metric("PR-AUC", f"{report['test_model']['pr_auc']:.3f}")
    st.markdown("---")
    if st.session_state.get("live_connected") and st.session_state.get("live_config"):
        st.subheader("Live Server Window")
        cfg=st.session_state["live_config"]
        _live_browser_component("Model",cfg["base_url"],cfg["endpoint"],cfg["headers"].get("Authorization",""),policy["verify_threshold"],policy["review_threshold"],"model",height=220)
        _live_browser_component("Calibration",cfg["base_url"],cfg["endpoint"],cfg["headers"].get("Authorization",""),policy["verify_threshold"],policy["review_threshold"],"calibration",height=380)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Held-out Test Metrics")
        tm = report["test_model"]
        metric_rows = pd.DataFrame([
            ["PR-AUC", tm.get("pr_auc")],
            ["ROC-AUC", tm.get("roc_auc")],
            ["Precision", tm.get("precision")],
            ["Recall", tm.get("recall")],
            ["F1 Score", tm.get("f1")],
            ["Brier Score", tm.get("brier")],
        ], columns=["Metric", "Value"])
        st.dataframe(
            metric_rows.style.format({"Value": "{:.4f}"}),
            hide_index=True,
        )
        chart_rows = metric_rows.copy()
        fig_metrics = go.Figure(
            go.Bar(
                x=chart_rows["Metric"],
                y=chart_rows["Value"],
                text=[f"{v:.3f}" for v in chart_rows["Value"]],
                textposition="outside",
                cliponaxis=False,
                marker=dict(color="#049FD9", line=dict(width=0)),
            )
        )
        enforce_plotly_light(fig_metrics)
        fig_metrics.update_traces(
            hovertemplate="%{x}<br>Score: %{y:.4f}<extra></extra>"
        )
        fig_metrics.update_layout(
            title="Held-out Model Metrics",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            margin=dict(t=55, r=20, b=20, l=20),
        )
        st.plotly_chart(fig_metrics, key="heldout_metrics_chart")
    with col2:
        st.subheader("Business Outcome")
        business = report["test_business"]
        st.metric("Expected Cost Reduction", f"{business['expected_cost_savings_pct']:.1%}")
        st.metric("Expected Loss / 1,000", f"₹{business['expected_cost'] / max(len(predictions),1) * 1000:,.0f}")
        st.write(f"Approve-all expected cost: ₹{business['approve_all_expected_cost']:,.0f}")
        st.write(f"ReturnShield expected cost: ₹{business['expected_cost']:,.0f}")
    st.subheader("Operating Thresholds")
    st.write(f"Verify threshold (T1): **{policy['verify_threshold']:.3f}**")
    st.write(f"Manual-review threshold (T2): **{policy['review_threshold']:.3f}**")
    if "abusive_return" in eval_predictions.columns and not eval_predictions.empty:
        st.subheader("Risk Calibration / Outcomes")
        cal = eval_predictions.copy()
        cal["bucket"] = pd.qcut(cal["risk_probability"].rank(method="first"), q=min(10, len(cal)), duplicates="drop")
        cal_df = cal.groupby("bucket", observed=True).agg(predicted=("risk_probability","mean"), observed=("abusive_return","mean")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cal_df["predicted"], y=cal_df["observed"], mode="lines+markers", name="Observed",
                                 line=dict(color="#1264A3", width=2), marker=dict(color="#049FD9", size=7)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect calibration",
                                 line=dict(dash="dash", color="#C6CDD6", width=1.5)))
        enforce_plotly_light(fig)
        fig.update_traces(hovertemplate="Predicted: %{x:.1%}<br>Observed: %{y:.1%}<extra></extra>")
        fig.update_layout(xaxis_title="Predicted probability", yaxis_title="Observed abuse rate")
        st.plotly_chart(fig, key="model_calibration")

else:
    st.title("Data & Live Transaction Server")
    st.caption("Upload a merchant CSV, connect an external REST API, or start the built-in fake live server.")
    st.subheader("Policy Cost Sandbox")
    fp, fn, vc, mc = st.columns(4)
    fp_cost = fp.slider("False Positive Cost (₹)", 50, 1000, 250, 50)
    fn_cost = fn.slider("False Negative Loss (₹)", 500, 5000, 2000, 100)
    verify_cost = vc.slider("Verification Cost (₹)", 10, 200, 40, 5)
    review_cost = mc.slider("Manual Review Cost (₹)", 20, 300, 60, 10)
    custom_costs = Costs(false_positive=fp_cost, false_negative=fn_cost, verification=verify_cost, manual_review=review_cost)
    # The live feed can be temporarily empty immediately after connecting.
    # Keep the policy sandbox usable by falling back to the held-out evaluation set
    # until the first live batch arrives, without ever indexing a missing column.
    policy_frame = predictions if "risk_probability" in predictions.columns and not predictions.empty else default_predictions
    probs = pd.to_numeric(policy_frame.get("risk_probability", pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy()
    opt_result = optimize_policy(policy_frame, probs, custom_costs) if len(policy_frame) else {
        "verify_threshold": policy["verify_threshold"],
        "review_threshold": policy["review_threshold"],
        "expected_cost": 0.0,
    }
    x1, x2, x3 = st.columns(3)
    x1.metric("Optimized T1", f"{opt_result['verify_threshold']:.3f}")
    x2.metric("Optimized T2", f"{opt_result['review_threshold']:.3f}")
    x3.metric("Expected Policy Cost", f"₹{opt_result['expected_cost']:,.0f}")
    st.markdown("---")
    st.subheader("Connect to a Live Transaction REST API Server")
    with st.form("live_server_form"):
        c1, c2 = st.columns([2, 1])
        server_url = c1.text_input("API Base URL", value="http://localhost:8000")
        endpoint = c1.text_input("Returns Endpoint Path", value="/api/v1/returns")
        auth_header = c2.text_input("Authorization Header (optional)", type="password")
        poll_limit = c2.number_input("Records per Poll", 50, 5000, 1000, 50)
        poll_seconds = c2.number_input("Auto-refresh (seconds)", 1, 30, 1, 1)
        connect = st.form_submit_button("Connect / Update Connection")
    if connect:
        headers = {"Authorization": auth_header} if auth_header.strip() else {}
        st.session_state["live_config"] = {"base_url": server_url.strip(), "endpoint": endpoint.strip(), "headers": headers, "poll_limit": min(int(poll_limit),5000), "poll_seconds": 1}
        st.session_state["live_connected"] = True
        st.session_state["live_seen_ids"] = set()
        st.session_state["live_cumulative_counts"] = {"total":0,"AUTO_APPROVE":0,"VERIFY":0,"MANUAL_REVIEW":0,"HIGH_RISK":0}
        st.session_state["active_predictions"] = pd.DataFrame()
        st.session_state["active_features"] = pd.DataFrame()
        st.session_state["dataset_name"] = "Live Transaction Feed"
        st.session_state["source_mode"] = "live"
        st.session_state["live_last_poll_at"] = None
        st.rerun()
    l1, l2, l3 = st.columns(3)
    if l1.button("Start Built-in Fake Live Server"):
        try:
            requests.post("http://localhost:8000/api/v1/returns/start", params={"rate_per_second":4}, timeout=5)
            st.session_state["live_config"] = {"base_url":"http://localhost:8000","endpoint":"/api/v1/returns","headers":{},"poll_limit":1000,"poll_seconds":1}
            st.session_state["live_connected"] = True
            st.session_state["live_seen_ids"] = set()
            st.session_state["live_cumulative_counts"] = {"total":0,"AUTO_APPROVE":0,"VERIFY":0,"MANUAL_REVIEW":0,"HIGH_RISK":0}
            st.session_state["active_predictions"] = pd.DataFrame()
            st.session_state["active_features"] = pd.DataFrame()
            st.session_state["dataset_name"] = "Live Transaction Feed"
            st.session_state["source_mode"] = "live"
            st.session_state["live_last_poll_at"] = None
            st.rerun()
        except Exception as e:
            st.error(f"Start failed: {e}. Run `python -m src.api` in another terminal first.")
    if l2.button("Stop Built-in Fake Live Server"):
        try: requests.post("http://localhost:8000/api/v1/returns/stop", timeout=5)
        except Exception: pass
        st.session_state["live_connected"] = False
        st.session_state["live_config"] = None
        st.session_state["live_last_poll_at"] = None
        st.rerun()
    if l3.button("Refresh Live Snapshot"):
        st.rerun()
    if st.session_state.get("live_connected") and st.session_state.get("live_config"):
        cfg=st.session_state["live_config"]
        _live_browser_component("Data Status",cfg["base_url"],cfg["endpoint"],cfg["headers"].get("Authorization",""),policy["verify_threshold"],policy["review_threshold"],"status",height=170)
    st.markdown("---")
    st.subheader("Export Transactions Before a Specific Time")
    st.caption("Exports records currently retained by the live server, filtered by UTC timestamp.")
    before_dt = st.datetime_input("Export everything generated before", value=pd.Timestamp.now().to_pydatetime())
    after_dt = st.datetime_input("Optional: generated after", value=(pd.Timestamp.now()-pd.Timedelta(hours=1)).to_pydatetime())
    if st.button("Generate Time-Bounded Live CSV"):
        if st.session_state["live_config"]:
            cfg = st.session_state["live_config"]
            base = cfg["base_url"].rstrip("/") + "/api/v1/returns/export.csv"
            params = {"before": pd.Timestamp(before_dt, tz="UTC" if pd.Timestamp(before_dt).tzinfo is None else None).isoformat()}
            if after_dt:
                params["after"] = pd.Timestamp(after_dt, tz="UTC" if pd.Timestamp(after_dt).tzinfo is None else None).isoformat()
            try:
                r = requests.get(base, headers=cfg["headers"], params=params, timeout=20); r.raise_for_status()
                st.download_button("Download Live Return CSV", r.content, "returnshield_live_returns.csv", "text/csv", key="download_live_range")
                st.success("CSV generated from the live server history.")
            except Exception as e: st.error(f"Export failed: {e}")
        else: st.warning("Connect to a live server first.")
    st.markdown("---")
    st.subheader("Upload a Merchant Returns CSV")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df):,} rows.")
        st.dataframe(df.head(10), hide_index=True)
        if st.button("Score & Activate Uploaded Dataset"):
            scored = score_dataframe(df, policy)
            st.session_state["active_predictions"] = scored; st.session_state["active_features"] = scored.copy(); st.session_state["dataset_name"] = f"{uploaded.name} ({len(scored):,} Returns)"; st.session_state["source_mode"] = "upload"
            st.success(f"Activated {len(scored):,} scored returns.")
            st.download_button("Download Scored CSV", scored.to_csv(index=False).encode(), "scored_returns.csv", "text/csv")

st.markdown("---")
st.caption("ReturnShield is defense-only. Live/demo records are synthetic unless connected to an external merchant system.")
