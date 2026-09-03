from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


INTENT_EXAMPLES = {
    "live_status": [
        "is the live server running",
        "show live server status",
        "what is the current live feed status",
        "are transactions coming in",
        "how fast is the live feed",
    ],
    "top_risk": [
        "show the highest risk returns",
        "which returns are most risky",
        "show high risk transactions",
        "give me the top risky returns",
        "what are the highest abuse risk returns",
    ],
    "summary": [
        "give me a summary",
        "summarize the current data",
        "how is returnshield doing",
        "what is happening in the returns data",
        "show me the current overview",
    ],
    "model_metrics": [
        "show model performance",
        "what are the held out metrics",
        "show precision recall f1",
        "how accurate is the model",
        "show the evaluation metrics",
    ],
    "inspect_return": [
        "inspect return LIVE-123",
        "show details for return LIVE-123",
        "investigate return LIVE-123",
        "what happened with return LIVE-123",
        "tell me about this return",
    ],
    "customer_search": [
        "show customer LIVE-C123",
        "find customer C123",
        "what returns did customer C123 make",
        "show transactions for this customer",
        "search for a customer",
    ],
    "clusters": [
        "show abuse clusters",
        "find coordinated accounts",
        "show fraud rings",
        "what coordinated clusters are active",
        "show the abuse network",
    ],
    "start_live": [
        "start the live server",
        "start generating transactions",
        "turn on the live feed",
        "start live mode",
    ],
    "stop_live": [
        "stop the live server",
        "stop generating transactions",
        "turn off the live feed",
        "stop live mode",
    ],
    "generate": [
        "generate 20 transactions",
        "create 100 live returns",
        "generate some test transactions",
        "make 10 more live events",
    ],
    "export": [
        "export the live data",
        "download returns before 6 pm",
        "export today's live transactions",
        "give me a csv",
        "export current returns",
    ],
    "help": [
        "what can you do",
        "help me",
        "show available commands",
        "what can I ask",
    ],
}


@dataclass
class ChatContext:
    last_return_id: str | None = None
    last_customer_id: str | None = None
    last_intent: str | None = None
    last_answer: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)


class ReturnShieldChatAgent:
    """Context-aware deterministic ReturnShield agent.

    The agent first resolves the user's intent and entities, then answers from the
    supplied/current ReturnShield data. It never invents live-data values.
    """

    def __init__(self):
        texts, labels = [], []
        for intent, examples in INTENT_EXAMPLES.items():
            texts.extend(examples)
            labels.extend([intent] * len(examples))
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced")),
        ])
        self.model.fit(texts, labels)

    @staticmethod
    def _clean_number(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    def _extract_ids(self, text: str, ctx: ChatContext):
        up = text.upper()
        live_ret = re.findall(r"\bLIVE-[A-Z0-9]{6,}\b", up)
        cust = re.findall(r"\bLIVE-C[-_]?[A-Z0-9]{4,}\b", up)
        generic_ret = re.findall(r"\bR[-_]?[A-Z0-9]{5,}\b", up)
        if live_ret:
            ctx.last_return_id = live_ret[0]
        elif generic_ret:
            ctx.last_return_id = generic_ret[0].replace("_", "-")
        if cust:
            ctx.last_customer_id = cust[0]

    def recognize(self, text: str, ctx: ChatContext) -> tuple[str, float, dict[str, Any]]:
        clean = text.strip()
        self._extract_ids(clean, ctx)
        low = re.sub(r"\s+", " ", clean.lower()).strip()

        # Context-aware follow-ups.
        if ctx.last_return_id and any(p in low for p in ["this return", "that return", "the return", "it"]):
            if any(k in low for k in ["inspect", "details", "risk", "why", "explain", "investigate", "tell me about"]):
                return "inspect_return", 0.99, {"return_id": ctx.last_return_id}
        if ctx.last_customer_id and any(p in low for p in ["this customer", "that customer", "the customer"]):
            if any(k in low for k in ["show", "find", "returns", "transactions", "history", "risk"]):
                return "customer_search", 0.99, {"customer_id": ctx.last_customer_id}

        # Fine-grained deterministic intents. Ordered from most specific to broad.
        rules = [
            ("ratio", ["return ratio", "fraud ratio", "fraud vs real", "real and fraud", "legitimate vs abusive", "legit vs abusive", "abusive ratio", "fraud percentage", "abuse percentage", "what percent", "what percentage"]),
            ("operations", ["operations overview", "brief overview of operations", "overview of operations", "operational overview", "current operations", "how are operations", "how is operations"]),
            ("trend", ["trend", "over time", "changing", "change in risk", "risk changing", "how is risk changing", "risk trend"]),
            ("top_risk", ["highest risk", "high risk returns", "most risky", "top risky", "top risk"]),
            ("model_metrics", ["model metrics", "model performance", "precision recall", "f1 score", "held-out", "held out", "brier"]),
            ("calibration", ["calibration", "calibrated", "outcomes graph", "observed abuse", "predicted vs observed"]),
            ("clusters", ["abuse cluster", "fraud ring", "coordinated account", "abuse network", "coordinated accounts", "cluster explorer"]),
            ("live_status", ["live server", "live feed", "server status", "transactions coming", "live status"]),
            ("inspect_return", ["inspect return", "investigate return", "details for return", "what happened with return", "risk of return"]),
            ("customer_search", ["show customer", "find customer", "customer history", "returns did customer", "customer returns"]),
            ("start_live", ["start live", "start the live", "turn on live", "start generating transactions"]),
            ("stop_live", ["stop live", "stop the live", "turn off live", "stop generating transactions"]),
            ("generate", ["generate ", "create ", "make "] ),
            ("export", ["export", "download csv", "give me a csv"]),
            ("help", ["what can you do", "what can i ask", "help me", "available commands"]),
            ("greeting", ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]),
            ("summary", ["give me a summary", "summarize", "current overview", "how is returnshield doing", "brief summary", "overall"]),
        ]
        for intent, phrases in rules:
            if any(phrase in low for phrase in phrases):
                slots: dict[str, Any] = {}
                if intent == "generate":
                    m = re.search(r"\b(\d{1,4})\b", low)
                    slots["count"] = max(1, min(int(m.group(1)), 1000)) if m else 10
                if ctx.last_return_id and any(w in low for w in ["return", "it"]):
                    slots.setdefault("return_id", ctx.last_return_id)
                if ctx.last_customer_id and "customer" in low:
                    slots.setdefault("customer_id", ctx.last_customer_id)
                return intent, 0.98, slots

        try:
            probs = self.model.predict_proba([clean])[0]
            intent = str(self.model.classes_[int(np.argmax(probs))])
            return intent, float(np.max(probs)), {}
        except Exception:
            return "help", 0.0, {}

    def respond(self, text: str, ctx: ChatContext, active_df: pd.DataFrame, report: dict, live_status: dict | None = None) -> dict[str, Any]:
        intent, confidence, slots = self.recognize(text, ctx)
        ctx.last_intent = intent
        result = {"intent": intent, "confidence": confidence, "answer": "", "data": None, "action": None}

        df = active_df.copy() if isinstance(active_df, pd.DataFrame) else pd.DataFrame()
        if not df.empty:
            if "risk_probability" in df.columns:
                df["risk_probability"] = pd.to_numeric(df["risk_probability"], errors="coerce").fillna(0)
            for c in ["order_value", "return_value", "merchant_loss_estimate", "returns_30d", "returns_90d", "refund_amount_90d", "hours_to_return"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        def needs_data() -> bool:
            return df.empty

        if intent == "greeting":
            result["answer"] = "Hello. I’m the ReturnShield Agent. Ask me about current return risk, operations, customers, coordinated accounts, model performance, calibration, or the live server."

        elif intent == "help":
            result["answer"] = (
                "I can answer questions from the current ReturnShield data, including operations, return risk, customers, clusters, live status, model metrics, and calibration. "
                "I can also inspect a return, start or stop the live generator, generate live transactions, and prepare a CSV export. "
                "For example: ‘What is the current fraud ratio?’, ‘Which returns are highest risk?’, or ‘Inspect LIVE-ABC12345’."
            )

        elif intent in ("summary", "operations"):
            if needs_data():
                result["answer"] = "I don’t have an active return dataset in the agent context yet. Connect the dataset or start the live server, then ask again."
            else:
                total = len(df)
                auto = int((df.get("decision", pd.Series(index=df.index, dtype=str)) == "AUTO_APPROVE").sum())
                verify = int((df.get("decision", pd.Series(index=df.index, dtype=str)) == "VERIFY").sum())
                review = int((df.get("decision", pd.Series(index=df.index, dtype=str)) == "MANUAL_REVIEW").sum())
                high = int((df.get("risk_probability", pd.Series(0, index=df.index)) >= 0.75).sum())
                avg_risk = float(df["risk_probability"].mean()) if "risk_probability" in df else 0.0
                refund = float(df["return_value"].sum()) if "return_value" in df else 0.0
                result["answer"] = (
                    f"Current operations: {total:,} return requests are in the active view. "
                    f"{auto:,} are AUTO_APPROVE, {verify:,} are VERIFY, and {review:,} are MANUAL_REVIEW. "
                    f"{high:,} are high risk (≥75%), average predicted risk is {avg_risk:.1%}, and the visible return value totals ₹{refund:,.0f}."
                )
                result["data"] = df.head(15)

        elif intent == "ratio":
            if needs_data():
                result["answer"] = "I don’t have active return records in the agent context yet. Start/connect the live data and I can calculate the current legitimate-versus-abusive ratio."
            else:
                if "abusive_return" in df.columns:
                    abusive = int(pd.to_numeric(df["abusive_return"], errors="coerce").fillna(0).sum())
                elif "simulated_outcome" in df.columns:
                    abusive = int((df["simulated_outcome"].astype(str).str.upper() == "ABUSIVE").sum())
                else:
                    abusive = int((df.get("risk_probability", pd.Series(0, index=df.index)) >= 0.75).sum())
                total = len(df)
                legit = max(total - abusive, 0)
                pct = abusive / total if total else 0
                result["answer"] = (
                    f"In the current active window, {legit:,} returns are legitimate and {abusive:,} are abusive, "
                    f"so the abusive/legitimate ratio is {abusive:,}:{legit:,} and the abusive share is {pct:.1%}. "
                    f"This is the current live/sample window, not a production fraud rate."
                )
                result["data"] = pd.DataFrame([{
                    "Category": "Legitimate", "Count": legit, "Share": 1 - pct
                }, {"Category": "Abusive", "Count": abusive, "Share": pct}])

        elif intent == "top_risk":
            if needs_data() or "risk_probability" not in df:
                result["answer"] = "No scored return records are available in the current agent context."
            else:
                cols = [c for c in ["return_id", "customer_id", "risk_probability", "decision", "order_value", "return_reason"] if c in df.columns]
                top = df.sort_values("risk_probability", ascending=False).head(15)[cols].copy()
                result["answer"] = f"Here are the top {len(top)} highest-risk return requests in the current active view."
                result["data"] = top

        elif intent == "model_metrics":
            m = report.get("test_model", report) if isinstance(report, dict) else {}
            result["answer"] = (
                f"Held-out model results: PR-AUC {float(m.get('pr_auc', 0)):.4f}, ROC-AUC {float(m.get('roc_auc', 0)):.4f}, "
                f"precision {float(m.get('precision', 0)):.3f}, recall {float(m.get('recall', 0)):.3f}, "
                f"F1 {float(m.get('f1', 0)):.3f}, Brier score {float(m.get('brier', 0)):.4f}."
            )
            result["data"] = pd.DataFrame([{"Metric": k.upper().replace("_", " "), "Value": v} for k, v in m.items() if isinstance(v, (int, float))])

        elif intent == "calibration":
            if needs_data() or "risk_probability" not in df.columns:
                result["answer"] = "There is no active scored data available for live calibration. Start or connect the live feed first."
            else:
                p = df["risk_probability"].to_numpy(dtype=float)
                if "abusive_return" in df.columns:
                    y = pd.to_numeric(df["abusive_return"], errors="coerce").fillna(0).to_numpy(dtype=float)
                    bins = np.linspace(0, 1, 11)
                    rows = []
                    for lo, hi in zip(bins[:-1], bins[1:]):
                        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
                        if mask.any():
                            rows.append({"Risk Bucket": f"{lo:.1f}–{hi:.1f}", "Predicted": p[mask].mean(), "Observed": y[mask].mean(), "Count": int(mask.sum())})
                    result["answer"] = "The live calibration view compares predicted abuse probability with the simulated observed outcome rate in the current active window."
                    result["data"] = pd.DataFrame(rows)
                else:
                    result["answer"] = "The current active data does not contain observed outcomes, so only predicted risk—not calibration—can be evaluated live."

        elif intent == "trend":
            if needs_data() or "risk_probability" not in df.columns:
                result["answer"] = "I need active scored return records to describe the live risk trend."
            else:
                work = df.copy()
                tscol = "generated_at" if "generated_at" in work.columns else ("prediction_time" if "prediction_time" in work.columns else None)
                if tscol:
                    work["_ts"] = pd.to_datetime(work[tscol], errors="coerce", utc=True)
                    work = work.dropna(subset=["_ts"]).sort_values("_ts")
                if len(work) >= 10:
                    first = float(work["risk_probability"].head(max(1, len(work)//5)).mean())
                    last = float(work["risk_probability"].tail(max(1, len(work)//5)).mean())
                    direction = "increasing" if last > first + 0.01 else "decreasing" if last < first - 0.01 else "roughly stable"
                    result["answer"] = f"The current live risk trend is {direction}: the average risk moved from {first:.1%} in the early portion of the active window to {last:.1%} in the latest portion."
                    result["data"] = work[["_ts", "risk_probability"]].tail(50).rename(columns={"_ts": "Time"})
                else:
                    result["answer"] = "There are not enough active live records yet to establish a reliable trend."

        elif intent == "inspect_return":
            rid = slots.get("return_id") or ctx.last_return_id
            if not rid:
                result["answer"] = "Tell me a return ID, for example `inspect LIVE-ABC12345`."
            elif needs_data():
                result["answer"] = f"I have the return ID `{rid}`, but no active return records are currently available to inspect."
            else:
                match = df[df.get("return_id", pd.Series(dtype=str)).astype(str).str.upper() == str(rid).upper()]
                if match.empty:
                    result["answer"] = f"I couldn't find `{rid}` in the current active data window."
                else:
                    row = match.iloc[0]
                    ctx.last_return_id = str(row.get("return_id", rid))
                    result["answer"] = (
                        f"{ctx.last_return_id} has a predicted abuse risk of {float(row.get('risk_probability', 0)):.1%} and a decision of {row.get('decision', '—')}. "
                        f"Order value is ₹{float(row.get('order_value', 0)):,.2f}; return reason: {row.get('return_reason', '—')}."
                    )
                    result["data"] = pd.DataFrame([row.to_dict()])
                    result["action"] = {"type": "open_investigator", "return_id": ctx.last_return_id}

        elif intent == "customer_search":
            cid = slots.get("customer_id") or ctx.last_customer_id
            if not cid:
                result["answer"] = "Tell me a customer ID, for example `show customer LIVE-C123456`."
            elif needs_data() or "customer_id" not in df.columns:
                result["answer"] = "I don't have active customer records available right now."
            else:
                match = df[df.customer_id.astype(str).str.upper() == str(cid).upper()].copy()
                if match.empty:
                    result["answer"] = f"I couldn't find customer `{cid}` in the current data window."
                else:
                    ctx.last_customer_id = str(match.iloc[0].get("customer_id", cid))
                    result["answer"] = f"Found {len(match):,} return request(s) for customer `{ctx.last_customer_id}`."
                    cols = [c for c in ["return_id", "risk_probability", "decision", "order_value", "return_value", "return_reason", "prediction_time"] if c in match.columns]
                    result["data"] = match.sort_values("risk_probability", ascending=False)[cols].head(50)

        elif intent == "live_status":
            if live_status:
                result["answer"] = (
                    f"The live server is {'running' if live_status.get('running') else 'stopped'} at {float(live_status.get('rate_per_second', 0)):.1f} transactions/sec. "
                    f"Buffer: {int(live_status.get('buffered_records', 0)):,}. Current regime: {live_status.get('regime', '—')}."
                )
                result["data"] = pd.DataFrame([live_status])
            else:
                result["answer"] = "No live server status is available."

        elif intent == "clusters":
            if needs_data():
                result["answer"] = "I don't have active return records for cluster analysis yet."
            elif not {"device_id", "address_id", "payment_fingerprint"}.intersection(df.columns):
                result["answer"] = "The active data does not contain infrastructure identifiers needed for coordinated-account analysis."
            else:
                grouped = []
                for col in ["device_id", "address_id", "payment_fingerprint"]:
                    if col not in df.columns or "customer_id" not in df.columns:
                        continue
                    g = df.groupby(col, dropna=True).agg(accounts=("customer_id", "nunique"), returns=("return_id", "count")).reset_index()
                    g = g[g.accounts >= 2].copy()
                    if not g.empty:
                        g["signal_type"] = col
                        grouped.append(g.rename(columns={col: "shared_identifier"}))
                data = pd.concat(grouped, ignore_index=True).sort_values(["accounts", "returns"], ascending=False) if grouped else pd.DataFrame()
                result["data"] = data.head(100)
                result["answer"] = f"Found {len(data):,} shared-infrastructure signals involving multiple accounts in the active view."

        elif intent == "start_live":
            result["action"] = {"type": "start_live"}
            result["answer"] = "Starting the live transaction generator."
        elif intent == "stop_live":
            result["action"] = {"type": "stop_live"}
            result["answer"] = "Stopping the live transaction generator."
        elif intent == "generate":
            count = int(slots.get("count", 10))
            result["action"] = {"type": "generate", "count": max(1, min(count, 1000))}
            result["answer"] = f"Generating {count} live transaction(s)."
        elif intent == "export":
            result["action"] = {"type": "export"}
            result["answer"] = "I'll prepare an export from the current live server history."
        else:
            result["answer"] = "I can answer questions about the current ReturnShield data. Try asking about operations, return/fraud ratio, risk, a return, a customer, clusters, model metrics, calibration, trends, or live status."

        ctx.last_answer = result["answer"]
        ctx.history.append({"role": "user", "content": text})
        ctx.history.append({"role": "assistant", "content": result["answer"]})
        return result
