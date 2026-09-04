ReturnShield AI

Cost-sensitive return-abuse risk and response agent for merchants.

Overview

ReturnShield helps merchants reduce losses from abusive returns while minimizing unnecessary friction for legitimate customers.

For each return request, the system uses information available at the time of the request to estimate abuse risk, calculate expected merchant loss, and select an operational action:

AUTO_APPROVE — low risk

VERIFY — moderate risk; request additional evidence

MANUAL_REVIEW — high risk; route to manual review

The product combines machine learning, cost-sensitive decisioning, coordinated-account analysis, real-time monitoring, and an AI agent.

What the product includes

Point-in-time synthetic event simulation

Leakage-safe 30-day and 90-day historical features

Logistic Regression baseline

XGBoost risk model

Probability calibration

Validation-based cost-sensitive policy optimization

Held-out temporal test evaluation

Risk explanations

NetworkX-based coordinated-account detection

Live transaction REST API

Continuous synthetic live transactions with changing risk regimes

Real-time operations dashboard

Return Investigator

Coordinated Account / Abuse Ring Explorer

Live charts, tables, and KPI updates

Hover information on graph and chart elements

Time-bounded CSV export

ChatGPT-style AI Chat

ReturnShield Agent API with tool calling

Multiple AI provider options

Right-side AI Settings panel

System architecture

User
  |
  v
AI Chat
  |
  v
AI Provider Layer
  |
  +-------------------------------+
  |       |       |       |       |
  v       v       v       v       v
Gemini   Groq  OpenRouter   HF   Ollama
                  |
                  v
        ReturnShield Agent API
                  |
        +---------+---------+
        |         |         |
        v         v         v
     Returns  Customers  Clusters
        |         |         |
        +---------+---------+
                  |
                  v
          Risk / Policy Engine
                  |
        +---------+---------+
        |         |         |
        v         v         v
    APPROVE    VERIFY   MANUAL REVIEW

The AI provider is the conversation and reasoning layer. ReturnShield's deterministic API and tools retrieve data and perform supported actions.

The AI model cannot override the deterministic risk policy.

Risk pipeline

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
 AUTO_APPROVE  VERIFY  MANUAL_REVIEW

Live transaction system

ReturnShield includes a FastAPI live/demo server that continuously generates synthetic return transactions.

The live generator is intentionally non-stationary. It can move through different operating regimes, such as:

baseline

elevated risk

fraud spike

coordinated-account surge

recovery

This makes live charts and operational metrics visibly change during a demonstration.

The dashboard is designed to update live data independently from the static page structure. Live numbers, charts, tables, and cluster information can update without requiring the user to change navigation tabs.

The live transaction history can also be exported for a selected time range.

Return Investigator

Return Investigator provides the detailed workflow for inspecting individual return requests.

It includes:

Return ID

Customer ID

Order ID

Order value

Return value

Expected merchant loss

Return reason

Historical return rate

Recent return activity

Historical refund amount

Hours to return

Linked device accounts

Linked address accounts

Key risk signals

Operational summary

Merchant protocol

Customer communication

The investigation workflow is retained when live mode is enabled. The available return list can refresh from the current live source without changing the investigation workflow.

Coordinated Account Explorer

ReturnShield uses NetworkX to identify suspicious relationships between accounts and shared infrastructure.

Signals include:

shared devices

shared addresses

shared payment fingerprints

historical return activity

historical refund behavior

coordinated high-risk activity

The graph presents relationships between customers and shared infrastructure so coordinated activity is easier to understand.

These signals are risk indicators and are not treated as proof of fraud.

AI Chat

ReturnShield includes a general-purpose ChatGPT-style AI assistant.

It can answer normal questions as well as questions about current ReturnShield data and operations.

Examples:

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

The assistant can also use ReturnShield tools for supported operations, including:

inspect a return

search returns

search a customer

get an operations summary

get coordinated-account information

get model metrics

start the live generator

stop the live generator

generate demo transactions

prepare/export live data

AI providers

The Chat system can be configured to use:

Google Gemini

Groq

OpenRouter

Hugging Face

Ollama

OpenAI

Provider selection and configuration are available from the AI Settings panel.

Ollama provides local inference without per-request cloud API charges.

Cloud providers may have free tiers or free models, but their quotas, limits, and pricing can change. Check the provider's current terms before relying on a free tier.

AI Settings

The Chat header includes a settings control that opens a right-side settings panel.

The panel can configure:

AI provider

API key

model

API base URL

connection testing

The settings panel does not control the ReturnShield risk policy.

Evaluation design

ReturnShield uses a chronological evaluation strategy:

First 60% of return requests → training

Next 20% → validation, model selection, calibration, and policy optimization

Final 20% → strictly held-out test evaluation

The held-out test set is not used for model or threshold selection.

Prediction metrics

PR-AUC

ROC-AUC

Precision

Recall

F1

Brier score

Business metrics

expected merchant loss

false-positive cost

false-negative cost

verification cost

manual-review rate

expected loss per 1,000 returns

loss reduction relative to the baseline policy

The system is optimized for merchant economics rather than raw accuracy.

Point-in-time leakage prevention

Historical features are computed strictly from information available before the return request timestamp.

Examples include:

30-day return count

90-day return rate

historical refund amount

device return rate

address return rate

linked-account statistics

Future events are not used when constructing prediction-time features.

Decision policy

The model outputs a calibrated abuse probability.

The policy engine maps the probability and business costs to an operational decision:

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

The decision thresholds are optimized using validation data and explicit business costs.

The held-out test set is reserved for final evaluation.

Safety and defense-only scope

ReturnShield is designed strictly for loss prevention.

It does not automate fraud accusations, exploit generation, or offensive activity.

The system recommends operational friction, verification, or manual review.

Coordinated-account signals are presented as risk indicators rather than definitive proof of wrongdoing.

AI-generated explanations and responses are constrained by the ReturnShield decision and tool layer.

The AI model cannot override the deterministic risk policy.

Quick start

Windows

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
start_agent.bat

macOS/Linux

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py

The integrated launcher starts the FastAPI service and Streamlit dashboard.

Main services

Streamlit:

http://127.0.0.1:8501

FastAPI:

http://127.0.0.1:8000

FastAPI documentation:

http://127.0.0.1:8000/docs

Live returns:

GET /api/v1/returns

Live statistics:

GET /api/v1/returns/stats

Agent Chat:

POST /api/v1/agent/chat

Project structure

returnshield/
├── app.py
├── run_pipeline.py
├── start_agent.bat
├── requirements.txt
├── README.md
├── DEMO.md
├── RELEASE_NOTES.md
├── diagnose.ps1
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

Important demo note

Default records, live/demo transactions, outcomes, and model metrics are synthetic unless ReturnShield is connected to an external merchant data source.

Synthetic results must not be presented as production fraud rates, production savings, or real merchant performance.

For a hackathon demonstration, clearly identify evaluation results as synthetic held-out test results.

Core design principle

Predict risk with machine learning, make the business decision with an explicit cost-sensitive policy, and use AI to understand questions, retrieve relevant ReturnShield information, perform supported actions, and communicate the result.

This keeps the system measurable, auditable, operationally useful, and defense-only.
