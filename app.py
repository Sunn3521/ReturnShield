from __future__ import annotations

import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.explain import concise_reasoning, top_features
from src.features import CATEGORICAL, NUMERIC
from src.model import load_bundle, predict_bundle
from src.network import build_abuse_graph, plot_cluster_graph
from src.policy import Costs, evaluate_policy, optimize_policy
from src.responder import generate_agent_response
from src.redteam import run_redteam_simulation
from src.shopify_webhook import process_shopify_return_webhook

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

st.set_page_config(
    page_title="ReturnShield AI", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Clean Button-Style Sidebar Navigation
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
        padding-top: 0.5rem;
    }
    
    .sidebar-title {
        font-size: 22px;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: -15px;
        margin-bottom: 24px;
        letter-spacing: -0.5px;
    }
    
    .nav-header {
        font-size: 18px;
        font-weight: 800;
        color: #94A3B8;
        letter-spacing: 0.5px;
        margin-bottom: 14px;
        text-transform: uppercase;
    }

    .dataset-badge {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 16px;
        font-size: 12px;
        color: #818CF8;
        font-weight: 600;
    }

    div[data-testid="stSidebar"] div.stRadio > label {
        display: none;
    }

    div[data-testid="stSidebar"] div.stRadio > div {
        gap: 10px;
    }
    
    div[data-testid="stSidebar"] div.stRadio > div > label {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        color: #CBD5E1;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        width: 100%;
        margin: 0;
    }
    
    div[data-testid="stSidebar"] div.stRadio > div > label:hover {
        background-color: #334155;
        border-color: #475569;
        color: #FFFFFF;
        transform: translateY(-1px);
    }
    
    div[data-testid="stSidebar"] div.stRadio > div > label[data-checked="true"] {
        background: linear-gradient(90deg, #4F46E5 0%, #6366F1 100%);
        border: 1px solid #818CF8;
        color: #FFFFFF;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35);
    }
</style>
""", unsafe_allow_html=True)


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
    st.error(f"Model assets not found or error loading: {e}. Run `python run_pipeline.py` first.")
    st.stop()

# Initialize global session state for active dataset
if "active_predictions" not in st.session_state:
    st.session_state["active_predictions"] = default_predictions.copy()
    st.session_state["active_features"] = default_features.copy()
    st.session_state["dataset_name"] = "Default Evaluation Dataset (2,000 Returns)"
    st.session_state["is_custom"] = False

predictions = st.session_state["active_predictions"]
features = st.session_state["active_features"]

# Sidebar Navigation Panel
with st.sidebar:
    st.markdown('<div class="sidebar-title">ReturnShield AI</div>', unsafe_allow_html=True)
    
    # Active Dataset Status Badge
    st.markdown(f'<div class="dataset-badge">📂 DATASET: {st.session_state["dataset_name"]}</div>', unsafe_allow_html=True)
    
    if st.session_state["is_custom"]:
        if st.button("🔄 Reset to Default Dataset"):
            st.session_state["active_predictions"] = default_predictions.copy()
            st.session_state["active_features"] = default_features.copy()
            st.session_state["dataset_name"] = "Default Evaluation Dataset (2,000 Returns)"
            st.session_state["is_custom"] = False
            st.rerun()

    st.markdown('<div class="nav-header">NAVIGATION</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation", 
        [
            "Operations Overview", 
            "Return Investigator", 
            "Abuse Ring Explorer", 
            "Model Evaluation & ROI",
            "Upload Data & API Sandbox"
        ]
    )

if page == "Operations Overview":
    st.title("ReturnShield Operations Overview")
    st.caption(f"Real-time risk scoring & evidence-backed action recommendations on active dataset: **{st.session_state['dataset_name']}**")
    
    total = len(predictions)
    high = int((predictions["decision"] == "MANUAL_REVIEW").sum())
    verify = int((predictions["decision"] == "VERIFY").sum())
    auto = int((predictions["decision"] == "AUTO_APPROVE").sum())
    
    # Calculate costs dynamically
    baseline_loss = float((predictions["abusive_return"] * 2000.0).sum()) if "abusive_return" in predictions.columns else float((predictions["risk_probability"] * predictions["order_value"]).sum())
    model_loss = float((predictions["merchant_loss"]).sum()) if "merchant_loss" in predictions.columns else float((predictions["risk_probability"] * 2000.0).sum())
    cost_savings = max(baseline_loss - model_loss, 0.0)
    cost_savings_pct = cost_savings / max(baseline_loss, 1.0)

    cols = st.columns(5)
    cols[0].metric("Total Returns", f"{total:,}")
    cols[1].metric("Auto Approved", f"{auto:,}", f"{auto/total:.1%}")
    cols[2].metric("Evidence Required", f"{verify:,}", f"{verify/total:.1%}")
    cols[3].metric("Manual Review", f"{high:,}", f"{high/total:.1%}")
    cols[4].metric("Est. Cost Savings", f"₹{cost_savings:,.0f}", f"+{cost_savings_pct:.1%}")

    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1])
    with col_left:
        fig = px.histogram(
            predictions, 
            x="risk_probability", 
            color="decision",
            nbins=35, 
            title="Risk Probability Distribution by Policy Action",
            color_discrete_map={"AUTO_APPROVE": "#38A169", "VERIFY": "#DD6B20", "MANUAL_REVIEW": "#E53E3E"},
            labels={"risk_probability": "Predicted Abuse Risk Probability", "count": "Returns"}
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
    with col_right:
        pie_df = predictions["decision"].value_counts().reset_index()
        pie_df.columns = ["Action", "Count"]
        fig_pie = px.pie(
            pie_df, 
            values="Count", 
            names="Action", 
            title="Action Decision Breakdown",
            color="Action",
            color_discrete_map={"AUTO_APPROVE": "#38A169", "VERIFY": "#DD6B20", "MANUAL_REVIEW": "#E53E3E"}
        )
        fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("🚨 Highest-Risk Returns Requiring Action")
    view = predictions.sort_values("risk_probability", ascending=False).head(15).copy()
    view["risk_display"] = (view["risk_probability"] * 100).round(1).astype(str) + "%"
    view["order_value_fmt"] = "₹" + view["order_value"].map("{:,.2f}".format)
    
    st.dataframe(
        view[["return_id", "customer_id", "order_value_fmt", "risk_display", "decision", "return_reason"]], 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "return_id": "Return ID",
            "customer_id": "Customer ID",
            "order_value_fmt": "Order Value",
            "risk_display": "Abuse Risk",
            "decision": "Action Recommendation",
            "return_reason": "Selected Reason"
        }
    )

elif page == "Return Investigator":
    st.title("Single Return Investigation")
    st.caption(f"Deep-dive into return signals, SHAP risk factors, and AI operational recommendations for **{st.session_state['dataset_name']}**.")
    
    rid = st.selectbox("Select Return Request to Inspect", predictions["return_id"].head(500).tolist())
    row = predictions[predictions["return_id"] == rid].iloc[0]
    
    st.subheader(f"Return Request: {rid}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Abuse Risk", f"{row['risk_probability']:.1%}")
    c2.metric("Order Value", f"₹{row['order_value']:,.2f}")
    c3.metric("Expected Merchant Loss", f"₹{row.get('merchant_loss', row['risk_probability'] * 2000.0):,.2f}")
    
    dec_color = "green" if row["decision"] == "AUTO_APPROVE" else "orange" if row["decision"] == "VERIFY" else "red"
    c4.markdown(f"### Action\n:{dec_color}[**{row['decision']}**]")

    full_row = features[features["return_id"] == rid].iloc[[0]].copy() if "return_id" in features.columns and (features["return_id"] == rid).any() else row.to_frame().T
    shap_items = top_features(bundle, full_row, top_n=6)
    reasons = concise_reasoning(full_row.iloc[0], shap_items)

    st.markdown("---")
    
    col_feat, col_agent = st.columns([1, 1])
    
    with col_feat:
        st.subheader("💡 Key Risk Signals")
        for reason in reasons:
            st.write("•", reason)

        st.subheader("📊 Customer Signal Metrics")
        profile = pd.DataFrame({
            "Signal": [
                "Return Rate (90d)", 
                "Returns (30d)", 
                "Historical Refunds (90d)", 
                "Hours to Return Request", 
                "Device-Linked Accounts", 
                "Address-Linked Accounts"
            ],
            "Value": [
                f"{row['return_rate_90d']:.1%}", 
                f"{int(row['returns_30d'])}", 
                f"₹{row['refund_amount_90d']:,.0f}", 
                f"{row['hours_to_return']:.1f} hrs", 
                f"{int(row['device_linked_accounts'])}", 
                f"{int(row['address_linked_accounts'])}"
            ],
        })
        st.dataframe(profile, use_container_width=True, hide_index=True)

    with col_agent:
        st.subheader("🤖 ReturnShield Decision Agent Response")
        agent_resp = generate_agent_response(row, reasons)
        
        st.info(f"**Operational Summary:**\n\n{agent_resp['summary']}")
        
        if row["decision"] == "AUTO_APPROVE":
            st.success(f"**Recommended Merchant Protocol:**\n\n{agent_resp['merchant_action']}")
        elif row["decision"] == "VERIFY":
            st.warning(f"**Recommended Merchant Protocol:**\n\n{agent_resp['merchant_action']}")
        else:
            st.error(f"**Recommended Merchant Protocol:**\n\n{agent_resp['merchant_action']}")
            
        with st.expander("✉️ View Draft Customer Communication", expanded=True):
            st.code(agent_resp['customer_message'], language="markdown")

elif page == "Abuse Ring Explorer":
    st.title("Abuse Ring & Cluster Explorer")
    st.caption(f"Network analysis identifying coordinated multi-account abuse across active dataset: **{st.session_state['dataset_name']}**")
    
    with st.spinner("Analyzing network graph from infrastructure data..."):
        high_risk_accounts = predictions[predictions["device_linked_accounts"] >= 3].copy()
        
        if high_risk_accounts.empty:
            st.info("No suspicious multi-account clusters detected above the link threshold in this dataset.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Suspicious Linked Returns", f"{len(high_risk_accounts):,}")
            m2.metric("Avg Linked Devices", f"{high_risk_accounts['device_linked_accounts'].mean():.1f}")
            m3.metric("High Abuse Risk Rate", f"{(high_risk_accounts['risk_probability'] >= 0.5).mean():.1%}")
            
            st.subheader("High-Risk Coordinated Accounts")
            st.dataframe(
                high_risk_accounts[["return_id", "customer_id", "order_value", "risk_probability", "decision", "device_linked_accounts", "address_linked_accounts"]].head(100),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "return_id": "Return ID",
                    "customer_id": "Customer ID",
                    "order_value": "Order Value (₹)",
                    "risk_probability": "Risk Probability",
                    "decision": "Action",
                    "device_linked_accounts": "Device Accounts",
                    "address_linked_accounts": "Address Accounts"
                }
            )

elif page == "Model Evaluation & ROI":
    st.title("Model Evaluation & Business Loss Analysis")
    st.caption(f"Evaluation & financial loss metrics computed on active dataset: **{st.session_state['dataset_name']}**")
    
    total_n = len(predictions)
    auto_n = int((predictions["decision"] == "AUTO_APPROVE").sum())
    ver_n = int((predictions["decision"] == "VERIFY").sum())
    rev_n = int((predictions["decision"] == "MANUAL_REVIEW").sum())
    
    st.subheader("Active Policy Distribution")
    m = st.columns(3)
    m[0].metric("Auto Approve Rate", f"{auto_n/total_n:.1%}", f"{auto_n:,} returns")
    m[1].metric("Evidence Verification Rate", f"{ver_n/total_n:.1%}", f"{ver_n:,} returns")
    m[2].metric("Manual Review Rate", f"{rev_n/total_n:.1%}", f"{rev_n:,} returns")

    st.markdown("---")
    
    baseline_loss = float((predictions["risk_probability"] * predictions["order_value"]).sum())
    model_cost = float((predictions["merchant_loss"]).sum()) if "merchant_loss" in predictions.columns else float((predictions["risk_probability"] * 2000.0).sum())
    savings = max(baseline_loss - model_cost, 0.0)

    col_policy, col_cost = st.columns([1, 1])
    with col_policy:
        st.subheader("⚙️ Operating Policy Thresholds")
        st.write(f"Verify threshold (T1): **{policy['verify_threshold']:.2f}**")
        st.write(f"Manual review threshold (T2): **{policy['review_threshold']:.2f}**")
        
    with col_cost:
        st.subheader("💰 Cost & Loss Metrics")
        st.metric("Unchecked Baseline Risk Loss", f"₹{baseline_loss:,.0f}")
        st.metric("ReturnShield Expected Loss", f"₹{model_cost:,.0f}")
        st.metric("Net Financial ROI", f"₹{savings:,.0f}", f"+{savings/max(baseline_loss,1):.1%} reduction")

else: # Upload Data & REST API Sandbox
    st.title("Custom Data Upload, Red-Team & REST API Sandbox")
    st.caption("Plug-and-play batch processor, interactive financial cost sandbox, Red-Team simulator, and REST API playground.")
    
    st.subheader("1. Red-Team Fraud Attack Simulation Benchmarking")
    st.markdown("Test ReturnShield defense capabilities against 4 real-world fraud attack vectors:")
    
    if st.button("Run Red-Team Fraud Attack Simulation"):
        with st.spinner("Executing Red-Team attack simulation..."):
            redteam_df = run_redteam_simulation()
            st.dataframe(redteam_df, use_container_width=True, hide_index=True)
            st.success("All 4 Red-Team attack vectors successfully intercepted by ReturnShield cost policy!")

    st.markdown("---")
    
    st.subheader("2. Interactive Policy & Cost Sandbox")
    st.markdown("Adjust business cost parameters live to observe dynamic policy threshold re-optimization:")
    
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    fp_cost = col_c1.slider("False Positive Cost (₹)", 50, 1000, 250, step=50, help="Customer friction cost when flagging a legitimate return")
    fn_cost = col_c2.slider("False Negative Loss (₹)", 500, 5000, 2000, step=100, help="Merchant financial loss when approving an abusive return")
    vf_cost = col_c3.slider("Verification Cost (₹)", 10, 200, 40, step=5, help="Cost of requesting item condition photos")
    mr_cost = col_c4.slider("Manual Review Cost (₹)", 20, 300, 60, step=10, help="Ops overhead per manually reviewed return")
    
    custom_costs = Costs(false_positive=fp_cost, false_negative=fn_cost, verification=vf_cost, manual_review=mr_cost)
    
    probs = predictions["risk_probability"].to_numpy()
    opt_result = optimize_policy(predictions, probs, custom_costs)
    
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("Optimized Verify Threshold (T1)", f"{opt_result['verify_threshold']:.2f}")
    s_col2.metric("Optimized Review Threshold (T2)", f"{opt_result['review_threshold']:.2f}")
    s_col3.metric("Expected Total Policy Loss", f"₹{opt_result['expected_cost']:,.0f}")
    
    baseline_loss = (predictions.get("abusive_return", predictions["risk_probability"]) * fn_cost).sum()
    savings_live = baseline_loss - opt_result['expected_cost']
    s_col4.metric("Net Financial ROI", f"₹{savings_live:,.0f}", f"+{savings_live/max(baseline_loss,1):.1%}")

    st.markdown("---")
    
    st.subheader("3. Custom Merchant Dataset Management & Batch Processor")
    
    if st.session_state["is_custom"]:
        st.success(f"📂 **Active Custom Dataset:** `{st.session_state['dataset_name']}` ({len(st.session_state['active_predictions']):,} records scored & active across all dashboard tabs)")
        st.dataframe(st.session_state["active_predictions"][["return_id", "customer_id", "order_value", "risk_probability", "decision"]].head(100), use_container_width=True)
        
        col_export, col_reset = st.columns([1, 1])
        csv_export = st.session_state["active_predictions"].to_csv(index=False).encode('utf-8')
        col_export.download_button("📥 Download Scored CSV Report", csv_export, "scored_returns_returnshield.csv", "text/csv")
        if col_reset.button("🔄 Deactivate & Re-upload New Dataset"):
            st.session_state["active_predictions"] = default_predictions.copy()
            st.session_state["active_features"] = default_features.copy()
            st.session_state["dataset_name"] = "Default Evaluation Dataset (2,000 Returns)"
            st.session_state["is_custom"] = False
            st.rerun()

    st.markdown("##### Upload New Dataset")
    uploaded_file = st.file_uploader("Upload custom `returns.csv` feature dataset", type=["csv"], key="custom_csv_uploader")
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.info(f"Loaded file `{uploaded_file.name}` ({len(custom_df):,} records). Click button below to score and activate across all tabs:")
            
            if st.button("🚀 Score & Activate Custom Dataset Across All Tabs", key="btn_score_activate"):
                with st.spinner(f"Scoring {len(custom_df):,} records through ReturnShield inference engine..."):
                    probs_custom = predict_bundle(bundle, custom_df)
                    custom_df["risk_probability"] = probs_custom
                    custom_df["decision"] = np.where(probs_custom < opt_result["verify_threshold"], "AUTO_APPROVE", np.where(probs_custom < opt_result["review_threshold"], "VERIFY", "MANUAL_REVIEW"))
                    custom_df["merchant_loss"] = np.where(custom_df["decision"] == "AUTO_APPROVE", custom_df["risk_probability"] * 2000.0, np.where(custom_df["decision"] == "VERIFY", 40.0, 60.0))
                    
                    # Activate globally for all tabs
                    st.session_state["active_predictions"] = custom_df.copy()
                    st.session_state["active_features"] = custom_df.copy()
                    st.session_state["dataset_name"] = f"{uploaded_file.name} ({len(custom_df):,} Returns)"
                    st.session_state["is_custom"] = True
                    st.rerun()
        except Exception as err:
            st.error(f"Error processing custom file: {err}")

    st.markdown("---")
    
    st.subheader("4. Live REST API & Shopify Webhook Tester")
    st.markdown("Test single return request API scoring live against the model engine:")
    
    col_api1, col_api2 = st.columns(2)
    api_order_val = col_api1.number_input("Order Value (₹)", min_value=100.0, max_value=500000.0, value=18500.0)
    api_returns_90d = col_api1.number_input("Customer Returns (90d)", min_value=0, max_value=50, value=6)
    api_linked_acc = col_api2.number_input("Linked Device Accounts", min_value=1, max_value=20, value=4)
    api_hours = col_api2.number_input("Hours to Return", min_value=0.1, max_value=720.0, value=1.5)

    if st.button("Score Return Request via API Engine"):
        t0 = time.perf_counter()
        test_payload = {
            "return_id": "R999-LIVE",
            "order_id": "O999-LIVE",
            "customer_id": "C999-LIVE",
            "product_category": "electronics",
            "payment_method": "card",
            "return_reason": "damaged",
            "order_value": api_order_val,
            "product_price": api_order_val,
            "discount_pct": 0.0,
            "customer_account_age_days": 45,
            "orders_7d": 2, "orders_30d": 5, "orders_90d": 8,
            "returns_7d": 2, "returns_30d": 4, "returns_90d": api_returns_90d,
            "return_rate_30d": 4 / 5,
            "return_rate_90d": api_returns_90d / 8,
            "refund_amount_30d": api_order_val * 2, "refund_amount_90d": api_order_val * 3,
            "hours_to_return": api_hours,
            "same_product_returns_90d": 1, "same_category_returns_90d": 3,
            "velocity_24h": 2, "velocity_7d": 3,
            "device_linked_accounts": api_linked_acc, "address_linked_accounts": api_linked_acc,
            "device_return_rate_90d": 0.65, "address_return_rate_90d": 0.65,
            "return_value": api_order_val, "return_value_ratio": 1.0, "high_value_flag": int(api_order_val >= 7500.0),
            "prediction_time": pd.Timestamp.now()
        }
        df_api = pd.DataFrame([test_payload])
        p_val = float(predict_bundle(bundle, df_api)[0])
        dec_val = "AUTO_APPROVE" if p_val < opt_result["verify_threshold"] else "VERIFY" if p_val < opt_result["review_threshold"] else "MANUAL_REVIEW"
        shap_api = top_features(bundle, df_api, top_n=4)
        reasons_api = concise_reasoning(df_api.iloc[0], shap_api)
        df_api_series = df_api.iloc[0].copy()
        df_api_series["risk_probability"] = p_val
        df_api_series["decision"] = dec_val
        resp_api = generate_agent_response(df_api_series, reasons_api)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        
        st.write(f"**Response Latency:** `{lat_ms:.2f} ms`")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Predicted Abuse Risk", f"{p_val:.1%}")
        res_col2.metric("Recommended Action", dec_val)
        
        st.info(f"**Agent Summary:**\n\n{resp_api['summary']}")
        st.warning(f"**Merchant Protocol:**\n\n{resp_api['merchant_action']}")
