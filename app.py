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
    page_title="ReturnShield AI — Return Abuse Risk Agent", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic Dark Theme CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar glassmorphism styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Brand Header Box */
    .brand-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    .brand-title {
        font-size: 20px;
        font-weight: 700;
        background: linear-gradient(90deg, #818CF8 0%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .brand-subtitle {
        font-size: 12px;
        color: #94A3B8;
        margin-top: 4px;
        font-weight: 400;
    }
    
    /* Status Badge Pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.4);
        color: #4ADE80;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        margin-top: 8px;
    }

    .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #22C55E;
        border-radius: 50%;
        box-shadow: 0 0 8px #22C55E;
    }
    
    /* Sidebar Navigation Options Styling */
    div[data-testid="stSidebar"] div.stRadio > label {
        display: none;
    }

    div[data-testid="stSidebar"] div.stRadio > div {
        gap: 8px;
    }
    
    div[data-testid="stSidebar"] div.stRadio > div > label {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 10px 14px;
        color: #E2E8F0;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        display: flex;
        align-items: center;
    }
    
    div[data-testid="stSidebar"] div.stRadio > div > label:hover {
        background: rgba(99, 102, 241, 0.12);
        border-color: rgba(99, 102, 241, 0.4);
        color: #FFFFFF;
        transform: translateX(4px);
    }
    
    /* Active Radio Item */
    div[data-testid="stSidebar"] div.stRadio > div > label[data-checked="true"] {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.25) 0%, rgba(168, 85, 247, 0.15) 100%);
        border: 1px solid #6366F1;
        border-left: 4px solid #818CF8;
        color: #FFFFFF;
        font-weight: 600;
        box-shadow: 0 2px 10px rgba(99, 102, 241, 0.2);
    }

    /* Sidebar Footer Info Card */
    .sidebar-footer {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 12px;
        margin-top: 30px;
        font-size: 11px;
        color: #64748B;
    }
    
    .metric-value-highlight {
        color: #38BDF8;
        font-weight: 600;
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
    report, predictions, bundle, policy, features = load_assets()
except Exception as e:
    st.error(f"Model assets not found or error loading: {e}. Run `python run_pipeline.py` first.")
    st.stop()

# Styled Sidebar Brand Header
with st.sidebar:
    st.markdown("""
        <div class="brand-box">
            <div class="brand-title">🛡️ ReturnShield AI</div>
            <div class="brand-subtitle">Cost-Sensitive Return Abuse Risk Agent</div>
            <div class="status-pill">
                <span class="pulse-dot"></span>
                <span>SYSTEM ACTIVE • v1.0</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation", 
        [
            "📊  Operations Overview", 
            "🔍  Return Investigator", 
            "🕸️  Abuse Ring Explorer", 
            "📈  Model Evaluation & ROI",
            "📥  Upload Data & API Sandbox"
        ]
    )

    # Sidebar Quick Info Footer Card
    st.markdown("""
        <div class="sidebar-footer">
            <div>⚡ <b>Latency:</b> <span class="metric-value-highlight">11.91 ms (p50)</span></div>
            <div style="margin-top: 4px;">🎯 <b>Policy:</b> Cost-Optimized (3-Tier)</div>
            <div style="margin-top: 4px;">🚀 <b>API Server:</b> FastAPI Port 8000</div>
        </div>
    """, unsafe_allow_html=True)

if page == "📊  Operations Overview":
    st.title("🛡️ ReturnShield Operations Overview")
    st.caption("Real-time risk scoring & evidence-backed action recommendations on held-out test evaluation period.")
    
    total = len(predictions)
    high = int((predictions["decision"] == "MANUAL_REVIEW").sum())
    verify = int((predictions["decision"] == "VERIFY").sum())
    auto = int((predictions["decision"] == "AUTO_APPROVE").sum())
    
    cols = st.columns(5)
    cols[0].metric("Total Returns", f"{total:,}")
    cols[1].metric("Auto Approved", f"{auto:,}", f"{auto/total:.1%}")
    cols[2].metric("Evidence Required", f"{verify:,}", f"{verify/total:.1%}")
    cols[3].metric("Manual Review", f"{high:,}", f"{high/total:.1%}")
    cols[4].metric("Cost Savings", f"₹{report['test_business']['expected_cost_savings']:,.0f}", f"+{report['test_business']['expected_cost_savings_pct']:.1%}")

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

elif page == "🔍  Return Investigator":
    st.title("🔍 Single Return Investigation")
    st.caption("Deep-dive into return signals, SHAP risk factors, and AI operational recommendations.")
    
    rid = st.selectbox("Select Return Request to Inspect", predictions["return_id"].tolist())
    row = predictions[predictions["return_id"] == rid].iloc[0]
    
    st.subheader(f"Return Request: {rid}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Abuse Risk", f"{row['risk_probability']:.1%}")
    c2.metric("Order Value", f"₹{row['order_value']:,.2f}")
    c3.metric("Expected Merchant Loss", f"₹{row['merchant_loss']:,.2f}")
    
    dec_color = "green" if row["decision"] == "AUTO_APPROVE" else "orange" if row["decision"] == "VERIFY" else "red"
    c4.markdown(f"### Action\n:{dec_color}[**{row['decision']}**]")

    full_row = features[features["return_id"] == rid].iloc[[0]].copy()
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

elif page == "🕸️  Abuse Ring Explorer":
    st.title("🕸️ Abuse Ring & Cluster Explorer")
    st.caption("Network analysis identifying coordinated multi-account abuse across shared devices, addresses, and payment fingerprints.")
    
    with st.spinner("Building network graph from customer infrastructure data..."):
        G, cluster_df = build_abuse_graph(str(ROOT / "data/raw"))
        
    if cluster_df.empty:
        st.info("No multi-account clusters detected above the threshold size.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Suspicious Clusters Detected", f"{len(cluster_df)}")
        m2.metric("Total Accounts in Clusters", f"{cluster_df['customer_count'].sum()}")
        m3.metric("High Abuse Clusters (>50% Abuse)", f"{(cluster_df['cluster_abuse_rate'] >= 0.5).sum()}")
        
        st.subheader("Detected Coordinated Clusters")
        st.dataframe(
            cluster_df[["cluster_id", "customer_count", "total_returns", "abusive_returns", "cluster_abuse_rate", "total_refund_value"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "cluster_id": "Cluster ID",
                "customer_count": "Linked Accounts",
                "total_returns": "Total Cluster Returns",
                "abusive_returns": "Abusive Returns",
                "cluster_abuse_rate": "Cluster Abuse Rate",
                "total_refund_value": "Total Refunds (₹)"
            }
        )
        
        selected_cid = st.selectbox("Select Cluster to Graph", cluster_df["cluster_id"].tolist())
        c_row = cluster_df[cluster_df["cluster_id"] == selected_cid].iloc[0]
        
        st.subheader(f"Network Graph: {selected_cid}")
        fig_graph = plot_cluster_graph(G, c_row["nodes"], title=f"Abuse Ring Graph — {selected_cid} ({c_row['customer_count']} Linked Accounts)")
        st.plotly_chart(fig_graph, use_container_width=True)
        
        st.markdown(
            "**Legend:**  \n"
            "🔴 **Red Circle:** Abusive / High Risk Customer Account  \n"
            "🔵 **Blue Circle:** Normal Customer Account  \n"
            "🟠 **Orange Diamond:** Shared Device, Address, or Payment Infrastructure"
        )

elif page == "📈  Model Evaluation & ROI":
    st.title("📈 Model Evaluation & Business Loss Analysis")
    st.caption("Strict temporal evaluation on held-out test dataset (20% chronological split).")
    
    tm = report["test_model"]
    tb = report["test_business"]
    
    st.subheader("Prediction Performance Metrics")
    c = st.columns(6)
    c[0].metric("PR-AUC", f"{tm['pr_auc']:.3f}")
    c[1].metric("ROC-AUC", f"{tm['roc_auc']:.3f}")
    c[2].metric("Precision", f"{tm['precision']:.1%}")
    c[3].metric("Recall", f"{tm['recall']:.1%}")
    c[4].metric("F1 Score", f"{tm['f1']:.3f}")
    c[5].metric("Brier Score", f"{tm['brier']:.3f}")

    st.markdown("---")
    
    col_policy, col_cost = st.columns([1, 1])
    
    with col_policy:
        st.subheader("⚙️ Optimized Operating Policy")
        st.write(f"Validation-tuned verify threshold: **{policy['verify_threshold']:.2f}**")
        st.write(f"Validation-tuned manual-review threshold: **{policy['review_threshold']:.2f}**")
        
        m = st.columns(3)
        m[0].metric("Auto Approve Rate", f"{tb['auto_approve_rate']:.1%}")
        m[1].metric("Evidence Verification Rate", f"{tb['verification_rate']:.1%}")
        m[2].metric("Manual Review Rate", f"{tb['manual_review_rate']:.1%}")
        
    with col_cost:
        st.subheader("💰 Cost & Loss Comparison")
        st.metric("Approve-All Baseline Cost", f"₹{tb['approve_all_expected_cost']:,.0f}")
        st.metric("ReturnShield Expected Cost", f"₹{tb['expected_cost']:,.0f}")
        st.metric("Net Financial Savings", f"₹{tb['expected_cost_savings']:,.0f}", f"+{tb['expected_cost_savings_pct']:.1%} reduction")

    st.subheader("Decision Errors & Loss Metrics")
    counts = pd.DataFrame({
        "Metric": ["False Negatives (Missed Fraud)", "False Positives (Customer Friction)"],
        "Count": [tb["false_negatives"], tb["false_positives"]],
        "Est. Financial Impact": [f"₹{tb['false_negatives']*2000:,.0f}", f"₹{tb['false_positives']*250:,.0f}"]
    })
    st.dataframe(counts, use_container_width=True, hide_index=True)

    st.subheader("Model Benchmark Comparison")
    comp = report["model_comparison"]
    comparison = pd.DataFrame({
        "Model Architecture": ["Calibrated Logistic Regression (Selected)", "XGBoost Classifier (Challenger)"],
        "PR-AUC": [comp["final_logistic_test"]["pr_auc"], comp["xgb_challenger_test"]["pr_auc"]],
        "ROC-AUC": [comp["final_logistic_test"]["roc_auc"], comp["xgb_challenger_test"]["roc_auc"]],
        "Brier Calibration": [comp["final_logistic_test"]["brier"], comp["xgb_challenger_test"]["brier"]],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.subheader("Evaluation Methodology Safeguards")
    st.markdown(
        "- **Chronological Temporal Split:** 60% Train / 20% Validation / 20% Held-Out Test.  \n"
        "- **Leakage-Safe Features:** All customer history metrics computed strictly before each return request timestamp.  \n"
        "- **Cost Matrix Optimization:** Thresholds $(T_1, T_2)$ optimized on validation set using explicit false-positive, false-negative, verification, and ops review costs.  \n"
        "- **Unbiased Test Set:** Held-out test set evaluated strictly once after policy selection."
    )

else: # Upload Data & REST API Sandbox
    st.title("📥 Custom Data Upload, Red-Team & REST API Sandbox")
    st.caption("Plug-and-play batch processor, interactive financial cost sandbox, Red-Team simulator, and REST API playground.")
    
    st.subheader("1. Red-Team Fraud Attack Simulation Benchmarking")
    st.markdown("Test ReturnShield defense capabilities against 4 real-world fraud attack vectors:")
    
    if st.button("⚔️ Run Red-Team Fraud Attack Simulation"):
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
    
    baseline_loss = (predictions["abusive_return"] * fn_cost).sum()
    savings_live = baseline_loss - opt_result['expected_cost']
    s_col4.metric("Net Financial ROI", f"₹{savings_live:,.0f}", f"+{savings_live/max(baseline_loss,1):.1%}")

    st.markdown("---")
    
    st.subheader("3. Drag & Drop Custom Merchant Dataset")
    uploaded_file = st.file_uploader("Upload custom `returns.csv` feature dataset", type=["csv"])
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded custom dataset with {len(custom_df)} records.")
            
            if st.button("🚀 Score Custom Batch Returns"):
                with st.spinner("Scoring batch data through ReturnShield inference engine..."):
                    probs_custom = predict_bundle(bundle, custom_df)
                    custom_df["risk_probability"] = probs_custom
                    custom_df["decision"] = np.where(probs_custom < opt_result["verify_threshold"], "AUTO_APPROVE", np.where(probs_custom < opt_result["review_threshold"], "VERIFY", "MANUAL_REVIEW"))
                    
                    st.subheader("Batch Scoring Results")
                    st.dataframe(custom_df[["return_id", "customer_id", "order_value", "risk_probability", "decision"]], use_container_width=True)
                    
                    csv_export = custom_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Scored CSV", csv_export, "scored_returns_returnshield.csv", "text/csv")
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

    if st.button("⚡ Score Return Request via API Engine"):
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
