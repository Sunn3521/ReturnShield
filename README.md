# ReturnShield AI

**Cost-sensitive return-abuse risk and response agent for merchants.**

ReturnShield helps merchants reduce losses from abusive returns while minimizing unnecessary customer friction. It evaluates return requests using point-in-time behavioral signals, applies a cost-sensitive decision policy, explains the decision, and provides an AI agent that can answer questions and interact with ReturnShield operations.

## What this product does

For each return request, ReturnShield estimates the probability of abusive merchant loss using only information available at the time of the request.

The calibrated risk score is passed to a deterministic policy engine that selects one of three operational actions:

* **AUTO_APPROVE** — low risk
* **VERIFY** — moderate risk; request additional evidence
* **MANUAL_REVIEW** — high risk; route to manual review

The product also provides:

* Point-in-time synthetic event simulation
* Leakage-safe 30-day and 90-day historical features
* Logistic-regression baseline and XGBoost risk model
* Probability calibration
* Validation-based cost-sensitive policy optimization
* Strictly held-out temporal test evaluation
* SHAP-based risk explanations
* Suspicious network and coordinated-account detection
* Live transaction REST API server
* Continuously generated synthetic live transactions
* Real-time operations dashboard
* Return investigation workflow
* Abuse-ring / coordinated-account explorer
* Live charts, tables, and KPI updates
* Hover-based graph and data tooltips
* CSV export of generated live transaction history
* ChatGPT-style AI Chat interface
* ReturnShield Agent API with tool calling
* Multi-provider AI support
* Right-side AI Settings panel
* General-purpose AI conversation plus ReturnShield-specific actions

## System architecture

```text
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

The AI model is the reasoning and conversation layer. ReturnShield's deterministic APIs and tools remain responsible for retrieving data and performing supported actions.

The AI model cannot override the ReturnShield risk policy.

## Core risk pipeline

```text
Return Request
      |
      v
Point-in-Time Feature Engineering
      |
      v
XGBoost Risk Model
      |
      v
Probability Calibration
      |
      v
Abuse Risk Probability
      |
      +----------------------+
      |                      |
      v                      v
Network Risk          Expected Merchant Loss
      |                      |
      +----------+-----------+
                 |
                 v
         Cost-Sensitive Policy
                 |
       +---------+---------+
       |         |         |
       v         v         v
   AUTO       VERIFY   MANUAL_REVIEW
  APPROVE
```

## Live transaction system

ReturnShield includes a live/demo REST API that continuously generates synthetic transactions.

The live system supports:

* Continuous transaction generation
* Different risk regimes
* Changing low/moderate/high-risk distributions
* Coordinated-account activity
* Live risk scoring
* Live operational metrics
* Live return investigation data
* Live cluster detection
* Time-bounded CSV export

The dashboard polls the API at approximately one-second intervals.

Live visual components update without requiring the user to change tabs.

The live UI is designed so that numbers, tables, graphs, and cluster information update independently rather than forcing a full application refresh.

## Return Investigator

The Return Investigator provides detailed inspection of individual return requests.

It includes:

* Return ID
* Customer ID
* Order ID
* Order value
* Return value
* Expected merchant loss
* Return reason
* Historical return rate
* Recent return activity
* Historical refunds
* Return timing
* Linked device accounts
* Linked address accounts
* Key risk signals
* Operational summary
* Merchant protocol
* Customer communication

The return-selection list can update from the live data source while preserving the existing investigation workflow.

## Coordinated-account detection

ReturnShield uses NetworkX to identify suspicious relationships between accounts and shared infrastructure.

Signals include:

* Shared devices
* Shared addresses
* Shared payment fingerprints
* Historical return activity
* Historical refund behavior
* Coordinated high-risk activity

The graph is intended as a **risk signal**, not proof of fraud.

## AI Chat

ReturnShield includes a ChatGPT-style AI Chat interface.

The chatbot can answer general questions as well as questions about current ReturnShield data and operations.

Examples include:

```text
What is the current return-abuse ratio?

Give me a brief overview of operations.

Which returns are highest risk?

Why was this return flagged?

Show coordinated accounts.

How is the live server performing?

What are the current model metrics?

Explain precision and recall.

What is a false positive?

Inspect LIVE-ABC12345.
```

The chatbot can also use ReturnShield tools for supported actions, such as:

* Inspecting returns
* Searching customers
* Searching returns
* Viewing operations summaries
* Viewing coordinated accounts
* Viewing model metrics
* Starting the live generator
* Stopping the live generator
* Generating demo transactions
* Preparing/exporting live data

## AI providers

The Chat system supports multiple providers:

* **Google Gemini**
* **Groq**
* **OpenRouter**
* **Hugging Face**
* **Ollama**
* **OpenAI**

Ollama can be used for local inference without a per-request cloud API charge.

OpenRouter, Groq, Gemini, and other providers may offer free tiers or free models subject to their current rate limits and account policies.

The provider can be selected from the **AI Settings** panel in the Chat tab.

## AI Settings

The Chat interface includes a settings control in the chat header.

The right-side settings panel can configure:

* AI provider
* API key
* Model
* API base URL
* Connection testing

The settings panel does not control the ReturnShield risk policy.

## Evaluation design

ReturnShield uses a chronological evaluation strategy:

* **First 60%** of return requests → training
* **Next 20%** → validation, model selection, calibration, and policy optimization
* **Final 20%** → strictly held-out test evaluation

The held-out test set is not used for model or threshold selection.

### Primary prediction metrics

* PR-AUC
* ROC-AUC
* Precision
* Recall
* F1
* Brier score

### Business metrics

* Expected merchant loss
* False-positive cost
* False-negative cost
* Verification cost
* Manual-review rate
* Expected loss per 1,000 returns
* Loss reduction relative to the baseline policy

The system is optimized for merchant economics rather than raw classification accuracy.

## Point-in-time leakage prevention

Historical features are generated strictly from information available before the return request timestamp.

Examples include:

* 30-day return count
* 90-day return rate
* historical refund amount
* device return rate
* address return rate
* linked-account statistics

Future events are not used when constructing prediction-time features.

## Decision policy

The model produces a calibrated abuse probability.

The policy engine maps the probability and business costs to an operational decision:

```text
LOW RISK
    |
    v
AUTO_APPROVE

MODERATE RISK
    |
    v
VERIFY

HIGH RISK
    |
    v
MANUAL_REVIEW
```

Thresholds are selected using validation data and explicit financial costs.

The held-out test set is reserved for final evaluation.

## Safety / defense-only scope

ReturnShield is designed strictly for loss prevention.

It does not automate fraud accusations, exploit generation, or offensive activity.

The system recommends operational friction, verification, or manual review.

Coordinated-account signals are presented as risk indicators rather than definitive proof of wrongdoing.

AI-generated explanations and customer responses are constrained by the ReturnShield decision and tool layer.

The AI model cannot override the deterministic risk policy.

## Quick start

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py
```

For the integrated live application, use:

```text
start_agent.bat
```

The launcher starts the required API and dashboard services.

## Project structure

```text
returnshield/
├── app.py
├── run_pipeline.py
├── start_agent.bat
├── requirements.txt
├── README.md
├── DEMO.md
├── RELEASE_NOTES.md
│
├── src/
│   ├── api.py
│   ├── chat_agent.py
│   ├── chatbot_api.py
│   ├── explain.py
│   ├── features.py
│   ├── live_server.py
│   ├── model.py
│   ├── network.py
│   ├── pipeline.py
│   ├── policy.py
│   ├── responder.py
│   └── simulate.py
│
├── data/
├── models/
└── reports/
```

## Main services

### Streamlit

```text
http://127.0.0.1:8501
```

### FastAPI

```text
http://127.0.0.1:8000
```

### FastAPI documentation

```text
http://127.0.0.1:8000/docs
```

### Live returns API

```text
GET /api/v1/returns
```

### Live statistics

```text
GET /api/v1/returns/stats
```

### Agent Chat API

```text
POST /api/v1/agent/chat
```

## Important demo note

All default records, live/demo transactions, outcomes, and reported model metrics are synthetic unless the system is connected to an external merchant data source.

Synthetic results must not be presented as real-world fraud rates or production performance.

For a hackathon demonstration, clearly identify evaluation results as **synthetic held-out test results**.

## Core design principle

> **Predict risk with machine learning, make the business decision with an explicit cost-sensitive policy, and use AI to understand questions, retrieve relevant ReturnShield information, and communicate the result.**

This keeps ReturnShield measurable, auditable, operationally useful, and defense-only.
