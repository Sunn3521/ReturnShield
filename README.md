# ReturnShield AI

**Cost-sensitive return-abuse risk and response agent for merchants.**

ReturnShield helps merchants reduce losses from abusive returns while minimizing unnecessary customer friction. It evaluates return requests using point-in-time behavioral signals, applies a cost-sensitive policy to choose an operational action, and exposes explainable risk signals to support human decisions.

---

## 📊 Results at a Glance

### Held-Out Test Metrics (Synthetic Dataset)

| Metric | Result |
|---|---:|
| PR-AUC | 0.2007 |
| ROC-AUC | 0.7072 |
| Precision | 1.0000 |
| Recall | 0.0133 |
| F1 Score | 0.0263 |
| Brier Score | 0.0445 |

### Business Evaluation

The evaluation uses a cost-sensitive decision policy where:

- False Positive cost = ₹250
- False Negative cost = ₹2,000
- Verification cost = ₹40
- Manual Review cost = ₹60

The policy is designed to minimize expected merchant loss while controlling unnecessary customer friction.

### Model Interpretation

The current model is deliberately operated at a **conservative decision threshold**.

This produces very high precision but low recall, meaning the system prioritizes avoiding unnecessary customer friction and false accusations over attempting to catch every abusive return.

This is a **prototype operating point** rather than a claim of production-ready fraud detection performance.

---

## 📊 Business Results

The cost-sensitive policy was evaluated on the **strictly held-out test set of 1,490 returns**.

| Metric                        |           Result |
| ----------------------------- | ---------------: |
| Held-Out Returns              |            1,490 |
| Abusive Returns               |               75 |
| Legitimate Returns            |            1,415 |
| AUTO_APPROVE                  |   1,369 (91.88%) |
| VERIFY                        |      120 (8.05%) |
| MANUAL_REVIEW                 |        1 (0.07%) |
| Abusive Returns Caught        | 28 / 75 (37.33%) |
| False Positives               |               93 |
| False Negatives               |               47 |
| Expected Merchant Loss        |     **₹122,110** |
| Approve-All Baseline Loss     |     **₹150,000** |
| Expected Savings              |      **₹27,890** |
| Loss Reduction                |       **18.59%** |
| Policy Loss / 1,000 Returns   |      **₹81,953** |
| Baseline Loss / 1,000 Returns |     **₹100,671** |
| Savings / 1,000 Returns       |      **₹18,718** |

### Cost Assumptions

The evaluation uses:

* False Positive Cost = **₹250**
* False Negative Cost = **₹2,000**
* Verification Cost = **₹40**
* Manual Review Cost = **₹60**

The optimized policy produced an expected merchant loss of **₹122,110**, compared with **₹150,000** for an approve-all baseline. This represents an estimated **₹27,890 reduction in expected merchant loss, or 18.59%**, on the 1,490-return held-out test set.

The policy remained conservative, with **91.88% of returns automatically approved**, while higher-risk returns were routed to verification or manual review.

> **Note:** These results are from a synthetic held-out evaluation dataset and should not be interpreted as real-world fraud or merchant-loss performance.

> **Clarification:** The 37.33% figure above is the *policy-level capture rate* (the share of abusive returns caught by the policy at the chosen business thresholds). Do not label this number as the model's classifier recall — the classifier recall at the evaluated model threshold remains **1.33%**. Keeping these two numbers distinct avoids confusion between model-level metrics and policy-level outcomes.

---

## 🧠 Why ReturnShield?

Traditional return systems often treat every return independently.

ReturnShield combines:

- **Point-in-time behavioral features** — strict temporal ordering to prevent data leakage
- **Cost-sensitive risk decisions** — optimization based on merchant financial costs, not classification accuracy
- **Calibrated probability estimates** — reliable risk scores for policy thresholds
- **Explainable AI** — SHAP-based feature attribution for each prediction
- **Coordinated-account graph analysis** — NetworkX-based detection of suspicious relationships
- **Real-time transaction monitoring** — continuous live transaction processing and risk scoring
- **AI-assisted investigation** — ChatGPT-style interface over deterministic ReturnShield APIs
- **Multiple AI providers** — Gemini, Groq, OpenRouter, Hugging Face, Ollama, OpenAI
- **Human-in-the-loop review** — three-tier decision framework instead of binary approve/reject

This allows merchants to move from simple rule-based return processing toward risk-aware operational decision making.

---

## Overview

ReturnShield is an AI-powered merchant risk management system designed to reduce financial losses caused by abusive returns while minimizing unnecessary friction for legitimate customers.

The system evaluates each return request using information available at the time the return request is submitted. It estimates the probability of an abusive return, calculates the expected merchant loss, and selects a cost-sensitive operational action.

ReturnShield is designed around three operational decisions:

* **AUTO_APPROVE** — low-risk return
* **VERIFY** — moderate-risk return requiring additional evidence
* **MANUAL_REVIEW** — high-risk return requiring human investigation

The system combines machine learning, temporal feature engineering, business cost optimization, coordinated-account analysis, real-time transaction processing, explainable predictions, and an AI agent to support operator workflows.

---

## Key Features

- Point-in-time synthetic event simulation
- Leakage-safe historical feature engineering (30-day / 90-day windows)
- Logistic Regression baseline and XGBoost risk model
- Probability calibration and validation-based policy optimization
- SHAP-based per-prediction explanations
- NetworkX-based coordinated-account detection
- Live transaction REST API with continuous synthetic generator
- Real-time dashboard with KPI updates, graphs and CSV export
- ChatGPT-style AI Chat with deterministic tool layer and multi-provider support

---

## System Architecture

```
                    USER
                      |
              +-------v--------+
              |   AI Chat UI   |
              +-------+--------+
                      |
              +-------v--------+
              | Provider Layer |
              +-------+--------+
                      |
       +--------------+----------------+
       |              |                |
    Gemini         Groq /        OpenRouter /
                  other APIs       Ollama / HF
                      |
              +-------v--------+
              | ReturnShield   |
              |   Agent API    |
              +-------+--------+
                      |
        +-------------+-------------+
        |             |             |
      Returns      Customers     Clusters
        |             |             |
        +-------------+-------------+
                      |
              Risk / Policy Engine
                      |
              Operational Actions
```

The AI model is the reasoning and conversation layer; ReturnShield's deterministic APIs and tools remain responsible for retrieving data and performing supported actions. The AI model cannot override the ReturnShield risk policy.

---

## Quick Start

### Windows

Open PowerShell in the ReturnShield project folder.

Create the virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate data and train the models:

```bash
python run_pipeline.py
```

Start the integrated application:

```bash
start_agent.bat
```


### Manual Startup

Start FastAPI manually:

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Then start Streamlit in another terminal:

```bash
streamlit run app.py
```


## Main Services

- Streamlit: http://127.0.0.1:8501
- FastAPI:  http://127.0.0.1:8000
- FastAPI docs: http://127.0.0.1:8000/docs


## API Endpoints (selected)

- GET /api/v1/returns — current live transactions
- GET /api/v1/returns/stats — live operational statistics
- POST /api/v1/returns/start — start continuous transaction generation
- POST /api/v1/returns/stop — stop continuous generation
- POST /api/v1/returns/generate — generate synthetic transactions on demand
- GET /api/v1/returns/export.csv — export live transaction history
- POST /api/v1/agent/chat — AI agent chat and tool access

---

## Known Limitations & Production Considerations

- The default dataset is synthetic; real merchant integrations are not included.
- Model performance may change substantially on real merchant populations.
- Production deployment requires authentication/authorization, secrets management, observability, rate limiting, and merchant-specific calibration.
- Do not commit API keys or other secrets to Git.


---

## Acknowledgements

This project was developed as a prototype/hackathon system demonstrating cost-sensitive operational decision-making for return-abuse risk. The implementation combines open-source ML tools, explainability methods, graph analysis, and a small live-demo stack.
