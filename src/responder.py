from __future__ import annotations

import os
import json
import pandas as pd

def generate_agent_response(row: pd.Series, reasons: list[str]) -> dict:
    risk_pct = row.get("risk_probability", 0.0)
    if isinstance(risk_pct, str):
        try:
            risk_pct = float(risk_pct.replace("%", "")) / 100.0
        except ValueError:
            risk_pct = 0.0

    decision = str(row.get("decision", "AUTO_APPROVE"))
    order_val = float(row.get("order_value", 0.0))
    customer_id = str(row.get("customer_id", "Unknown"))
    return_id = str(row.get("return_id", "Unknown"))
    reason_txt = str(row.get("return_reason", "general_return"))

    # Try Google GenAI if key is present
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = f"""
            You are ReturnShield AI Risk Agent.
            Generate merchant guidance for Return ID {return_id} (Customer {customer_id}).
            Details:
            - Order Value: ₹{order_val:,.2f}
            - Risk Score: {risk_pct:.1%}
            - Policy Recommendation: {decision}
            - Primary Risk Reasons: {", ".join(reasons)}
            
            Return JSON format with keys:
            "summary": concise operational summary,
            "merchant_action": specific step-by-step instructions for merchant ops,
            "customer_message": polite customer notification message (never accuse of fraud)
            """
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            parsed = json.loads(response.text)
            # Normalize provider output so the UI never receives missing/empty fields.
            if not isinstance(parsed, dict):
                raise ValueError("Provider returned a non-object response")
            merchant_action = (
                parsed.get("merchant_action")
                or parsed.get("merchant_protocol")
                or parsed.get("protocol")
                or ""
            )
            customer_message = (
                parsed.get("customer_message")
                or parsed.get("customer_communication")
                or parsed.get("customer_response")
                or ""
            )
            summary = parsed.get("summary") or parsed.get("operational_summary") or ""
            if summary and merchant_action and customer_message:
                return {
                    "summary": str(summary),
                    "merchant_action": str(merchant_action),
                    "customer_message": str(customer_message),
                }
        except Exception:
            pass

    # High-quality deterministic agent response engine (Fallback)
    if decision == "AUTO_APPROVE":
        summary = f"Return {return_id} exhibits low risk ({risk_pct:.1%}). Customer order and return behavior meet standard trust thresholds."
        merchant_action = "Approve refund automatically. Issue shipping label and release credit upon item carrier scan."
        customer_message = f"Your return request for order {row.get('order_id', '')} has been approved. A pre-paid return label has been emailed to you."
    elif decision == "VERIFY":
        summary = f"Return {return_id} flagged for moderate risk ({risk_pct:.1%}) due to {reasons[0] if reasons else 'elevated return activity'}."
        merchant_action = "Pause automatic refund. Request high-resolution photo verification of the item condition, serial number, and original packaging prior to authorizing return shipment."
        customer_message = f"Thank you for submitting your return request. To process your refund quickly, please upload a quick photo showing the current condition and packaging of the item via your account dashboard."
    else: # MANUAL_REVIEW
        summary = f"Return {return_id} flagged for high abuse probability ({risk_pct:.1%}). Multi-factor risk indicators detected: {'; '.join(reasons[:3])}."
        merchant_action = "Hold refund pending senior risk manager review. Cross-check shared address/device clusters, examine previous return conditions for SKU, and inspect warehouse physical receiving log before approval."
        customer_message = f"Your return request for order {row.get('order_id', '')} is currently undergoing standard administrative verification. Our customer support team will update you within 1-2 business days."

    return {
        "summary": summary,
        "merchant_action": merchant_action,
        "customer_message": customer_message
    }
