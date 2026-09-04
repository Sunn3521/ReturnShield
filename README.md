# ReturnShield AI

**Cost-sensitive return-abuse risk and response agent for merchants.**

ReturnShield helps merchants reduce losses from abusive returns while minimizing unnecessary customer friction. It evaluates return requests using point-in-time behavioral signals, applies a cost-sensitive policy to determine the appropriate operational action, and provides an AI-assisted operations interface.

---

## Overview

ReturnShield is an AI-powered merchant risk management system designed to reduce financial losses caused by abusive returns while minimizing unnecessary friction for legitimate customers.

The system evaluates each return request using information available at the time the return request is submitted. It estimates the probability of an abusive return, calculates the expected merchant loss, and applies a cost-sensitive policy to determine the appropriate operational action.

ReturnShield is designed around three operational decisions:

* **AUTO_APPROVE** — low-risk return
* **VERIFY** — moderate-risk return requiring additional evidence
* **MANUAL_REVIEW** — high-risk return requiring human investigation

The system combines machine learning, temporal feature engineering, business cost optimization, coordinated-account analysis, real-time transaction processing, explainable predictions, and an AI agent.

---

## Problem Statement

Merchants lose money through return abuse, refund abuse, coordinated account activity, repeated high-value returns, and other forms of abusive behavior.

Traditional fraud systems often reduce the problem to:

* Fraud
* Not Fraud

This does not directly answer the operational question: **What should the merchant do with this return?**

A merchant may prefer to approve a legitimate return immediately, verify a suspicious return, or route a high-risk return for manual investigation.

The correct action depends not only on the predicted probability of abuse, but also on the relative financial costs of:

* Incorrectly flagging a legitimate customer
* Allowing an abusive return
* Requesting additional evidence
* Sending a case for manual review

ReturnShield therefore treats return abuse as a cost-sensitive decision problem rather than a simple binary classification problem.

---

## Project Objective

The primary objective of ReturnShield is:

**Predict return-abuse risk at the time of a return request and convert that prediction into the lowest-cost operational action for the merchant.**

The system aims to:

1. Detect potentially abusive return requests.
2. Reduce expected merchant financial loss.
3. Minimize unnecessary customer friction.
4. Identify coordinated multi-account abuse.
5. Provide explainable risk decisions.
6. Monitor returns continuously through a live transaction stream.
7. Give merchants an AI interface for querying and operating the platform.

---

## What This Product Does

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

The AI model is the reasoning and conversation layer. ReturnShield's deterministic APIs and tools remain responsible for retrieving data and performing supported actions.

The AI model cannot override the ReturnShield risk policy.

---

## Core Risk Pipeline

```
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

---

## Live Transaction System

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

---

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

---

## Coordinated-Account Detection

ReturnShield uses NetworkX to identify suspicious relationships between accounts and shared infrastructure.

Signals include:

* Shared devices
* Shared addresses
* Shared payment fingerprints
* Historical return activity
* Historical refund behavior
* Coordinated high-risk activity

The graph is intended as a **risk signal**, not proof of fraud.

---

## AI Chat

ReturnShield includes a ChatGPT-style AI Chat interface.

The chatbot can answer general questions as well as questions about current ReturnShield data and operations.

Examples include:

```
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

---

## AI Providers

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

---

## AI Settings

The Chat interface includes a settings control in the chat header.

The right-side settings panel can configure:

* AI provider
* API key
* Model
* API base URL
* Connection testing

The settings panel does not control the ReturnShield risk policy.

---

## Evaluation Design

ReturnShield uses a chronological evaluation strategy:

* **First 60%** of return requests → training
* **Next 20%** → validation, model selection, calibration, and policy optimization
* **Final 20%** → strictly held-out test evaluation

The held-out test set is not used for model or threshold selection.

### Primary Prediction Metrics

* PR-AUC
* ROC-AUC
* Precision
* Recall
* F1
* Brier score

### Business Metrics

* Expected merchant loss
* False-positive cost
* False-negative cost
* Verification cost
* Manual-review rate
* Expected loss per 1,000 returns
* Loss reduction relative to the baseline policy

The system is optimized for merchant economics rather than raw classification accuracy.

---

## Point-in-Time Leakage Prevention

Historical features are generated strictly from information available before the return request timestamp.

Examples include:

* 30-day return count
* 90-day return rate
* Historical refund amount
* Device return rate
* Address return rate
* Linked-account statistics

Future events are not used when constructing prediction-time features.

---

## Decision Policy

The model produces a calibrated abuse probability.

The policy engine maps the probability and business costs to an operational decision:

```
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

---

## Safety / Defense-Only Scope

ReturnShield is designed strictly for loss prevention.

It does not automate fraud accusations, exploit generation, or offensive activity.

The system recommends operational friction, verification, or manual review.

Coordinated-account signals are presented as risk indicators rather than definitive proof of wrongdoing.

AI-generated explanations and customer responses are constrained by the ReturnShield decision and tool layer.

The AI model cannot override the deterministic risk policy.

---

## Key Features

ReturnShield provides the following capabilities:

* Point-in-time synthetic event simulation
* Leakage-safe historical feature engineering
* 30-day and 90-day behavioral features
* Logistic Regression baseline
* XGBoost risk model
* Probability calibration
* Validation-set cost optimization
* Chronological train/validation/test evaluation
* PR-AUC, ROC-AUC, Precision, Recall, F1 and Brier score
* Merchant expected-loss evaluation
* False-positive and false-negative cost modeling
* SHAP-based risk explanations
* NetworkX-based coordinated-account detection
* Shared-device analysis
* Shared-address analysis
* Shared-payment fingerprint analysis
* Live transaction REST API
* Continuous live transaction generation
* Non-stationary live risk regimes
* Real-time operations dashboard
* Real-time KPI updates
* Real-time graph and table updates
* Return Investigator
* Coordinated Account Explorer
* Live calibration/outcome visualization
* Time-bounded CSV export
* ChatGPT-style AI Chat
* ReturnShield Agent API
* AI tool calling
* General-purpose AI conversation
* Multiple AI provider support
* Right-side AI Settings panel
* Local Ollama support
* Free/cloud provider support
* Responsive enterprise-style interface
* Hover-based graph and chart information

---

## ReturnShield Agent API

The ReturnShield Agent API acts as the bridge between the AI provider and the application.

The general flow is:

```
User
|
v
AI Chat
|
v
AI Model
|
v
Tool Selection
|
v
ReturnShield Agent API
|
+-------------------------+
|            |            |
v            v            v
Returns     Customers     Clusters
|            |            |
+------------+------------+
|
v
Risk / Policy Data
|
v
Tool Result
|
v
AI Final Response
```

The model can request supported tools, receive real ReturnShield data, and then generate a natural-language answer.

The tool layer remains deterministic and allow-listed.

---

## Ollama

Ollama can be used for local model inference.

This is useful when:

* No cloud API key is available
* No per-request cloud cost is desired
* Internet-independent inference is preferred

Local model quality and performance depend on the hardware available on the demonstration machine and the selected model.

---

## Cloud Providers

Cloud providers can be used when appropriate.

Depending on the provider and model, free tiers or free models may be available.

Free-tier availability, quotas, model access, and rate limits can change.

The application therefore treats provider availability as configurable rather than assuming unlimited free usage.

---

## Cost-Sensitive Decision Policy

The model does not directly decide whether a return should be approved.

Instead, the model produces a calibrated abuse probability.

The deterministic policy engine converts that probability into an operational action.

```
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

The thresholds are selected using validation data and explicit business costs.

The purpose is to balance:

* Customer friction
* Missed abusive returns
* Verification costs
* Manual-review costs
* Merchant loss

The held-out test set is not used to choose the thresholds.

---

## Synthetic Data Generation

The default dataset is synthetic.

The data simulator creates customers, orders, returns, relationships, and eventual outcomes.

Different behavioral profiles are generated, such as:

* Normal customers
* Frequent legitimate returners
* Abusive customers
* Coordinated abuse groups

The distributions intentionally overlap.

This prevents the synthetic classification task from becoming unrealistically easy.

---

## Behavioral Simulation

**Normal customers** generally exhibit:

* Lower return frequency
* Normal return timing
* Lower refund values
* Fewer linked accounts
* Ordinary purchase velocity

**Frequent legitimate returners** may exhibit:

* Higher return rates
* Repeated category returns
* Higher total returns

But they are not automatically labeled abusive.

**Abusive customers** may exhibit:

* Unusually high return frequency
* High refund amounts
* Fast return requests
* Higher-value purchases
* Abnormal transaction velocity

**Coordinated abuse groups** may exhibit:

* Multiple accounts
* Shared devices
* Shared addresses
* Shared payment fingerprints
* Similar high-risk behavior

---

## Machine Learning

ReturnShield uses a baseline/challenger model strategy.

### Baseline

Logistic Regression is used as a simple interpretable baseline.

### Challenger

XGBoost is used as the primary nonlinear model.

XGBoost is suitable for the structured behavioral features in the dataset and can capture interactions between:

* Return history
* Customer behavior
* Order value
* Transaction velocity
* Network signals

---

## Class Imbalance

Abusive returns represent only a small portion of the dataset.

This creates a major problem with standard accuracy.

For example:

* 96.5% legitimate
* 3.5% abusive

A naive model that predicts every return as legitimate could achieve:

* 96.5% accuracy

While catching:

* 0% of abusive returns

That is not useful for merchant loss prevention.

ReturnShield therefore prioritizes metrics that are meaningful for rare-event classification.

---

## Probability Calibration

A raw model probability is not automatically a reliable probability estimate.

ReturnShield therefore includes a probability calibration stage.

```
XGBoost
    |
    v
Raw Probability
    |
    v
Calibration
    |
    v
Calibrated Probability
```

This enables the policy engine to work with an interpretable risk probability rather than only a ranking score.

---

## Evaluation Split

ReturnShield uses a chronological dataset split.

```
60% earliest events
    |
    v
Training

20% following events
    |
    v
Validation

20% latest events
    |
    v
Held-out Test
```

This approximates the real-world scenario of predicting future return behavior using past data.

---

## Held-Out Test Evaluation

The final test set is used only after:

* Model selection
* Probability calibration
* Threshold selection
* Policy optimization

Have been completed.

This prevents information from the test period from influencing model development.

---

## Business Evaluation

ReturnShield reports both machine-learning and financial metrics.

**Prediction metrics:**

* PR-AUC
* ROC-AUC
* Precision
* Recall
* F1
* Brier Score

**Business metrics:**

* Expected Merchant Loss
* False-Positive Cost
* False-Negative Cost
* Verification Cost
* Manual Review Rate
* Expected Loss per 1,000 Returns
* Loss Reduction vs Baseline

---

## SHAP Explanations

ReturnShield provides feature-level explanations for individual predictions.

A high-risk return might show signals such as:

* High historical return rate
* High recent return velocity
* Shared device/address
* High historical refund value
* Very fast return request

The purpose is to make the model decision understandable to a merchant operator.

---

## Example Risk Explanation

```
Predicted Abuse Risk: 91%

Top Risk Signals:

1. High historical return rate
2. High recent return velocity
3. Multiple linked accounts
4. High historical refund value
5. Very fast return request
```

The explanation describes model evidence rather than declaring that a customer is definitively fraudulent.

---

## Live Dashboard

The Operations Overview provides a real-time view of the system.

It includes live operational information such as:

* Returns in View
* Auto Approved
* Verification
* Manual Review
* High Risk
* Risk Distribution
* Policy Actions
* Highest-Risk Returns
* Coordinated-Account Activity
* Live Server Status
* Current Operating Regime

---

## Live Refresh Behavior

The live dashboard is designed for approximately one-second updates.

Live data is separated from the static application structure.

The goal is:

```
Live API
    |
    v
New Transactions
    |
    v
Data Update
    |
    v
Charts / Numbers / Tables
    |
    v
No Navigation Change Required
```

The live visual layer is designed to avoid unnecessary full-page refreshes.

---

## Anti-Blink Design

Frequent redraws can cause:

* Flashing
* Blinking
* Fade transitions
* Eye strain
* Unnecessary visual movement

ReturnShield therefore uses a component-based update strategy for the live data layer.

Static UI elements remain stable while live data changes.

The objective is to update:

* KPI values
* Chart values
* Table rows
* Live graph data

Without repeatedly rebuilding unrelated page content.

---

## Risk-Based Visualization

ReturnShield uses different visual states for operational risk.

* **Low Risk** — Green
* **Moderate Risk** — Amber
* **High Risk** — Red

The same semantic mapping is used for relevant decision cells and risk values.

---

## Charts

The dashboard can display:

* Risk probability distribution
* Policy action distribution
* Live outcome/calibration information
* Model metrics
* Coordinated-account relationships

The policy action visualization uses a pie chart rather than a donut chart.

---

## Hover Information

Interactive graphs and charts provide contextual information when the user hovers over a visual element.

Examples include:

* Risk Probability
* Decision
* Count
* Percentage
* Customer
* Cluster
* Return Count
* Risk Score

Hover information allows the user to inspect data without opening a separate page.

---

## CSV Export

The live server maintains generated transaction history.

The system supports exporting records for a selected period.

This can be useful for:

* Offline analysis
* Auditing
* Demonstrations
* Debugging
* Model evaluation
* Saving generated live sessions

The export is time-bounded rather than limited only to the currently visible dashboard records.

---

## Technical Challenges

Several major technical obstacles were addressed during development.

### 1. Data Leakage

**Challenge:**

Historical return and refund features could accidentally include future events.

**Solution:**

A chronological event processing system was created so every feature is calculated strictly from events before the prediction timestamp.

---

### 2. Class Imbalance

**Challenge:**

Abusive returns represent a small percentage of total returns.

**Solution:**

The project prioritizes PR-AUC, precision, recall, F1 and business cost metrics instead of relying on accuracy.

---

### 3. Threshold Selection

**Challenge:**

A fixed probability threshold such as 0.5 is not necessarily optimal for merchant economics.

**Solution:**

Thresholds are optimized on validation data using explicit false-positive, false-negative, verification, and manual-review costs.

---

### 4. Coordinated Abuse

**Challenge:**

Individual accounts can remain below customer-level thresholds while coordinating with other accounts.

**Solution:**

A NetworkX graph connects shared devices, addresses, and payment fingerprints to reveal coordinated patterns.

---

### 5. Real-Time Transactions

**Challenge:**

The live dashboard needs continuously changing data.

**Solution:**

A FastAPI live server produces synthetic transactions continuously while the dashboard consumes the data through the live API.

---

### 6. Live UI Stability

**Challenge:**

Frequent data refreshes can cause visual blinking, fading, or unnecessary page movement.

**Solution:**

Live components are separated from static UI content and updated independently.

---

### 7. AI Integration

**Challenge:**

A chatbot must support general conversation while also understanding ReturnShield-specific data and operations.

**Solution:**

The system uses an AI provider layer combined with the ReturnShield Agent API and allow-listed tools.

---

### 8. Multiple AI Providers

**Challenge:**

A hackathon project should not depend on one paid AI provider.

**Solution:**

The system supports multiple providers including Gemini, Groq, OpenRouter, Hugging Face, Ollama, and OpenAI.

---

## Technology Stack

### Frontend

* Streamlit
* Plotly
* HTML
* CSS
* JavaScript

### Backend

* FastAPI
* Uvicorn
* Python

### Machine Learning

* scikit-learn
* XGBoost
* SHAP
* NumPy
* Pandas

### Graph Analysis

* NetworkX

### Data Storage

* CSV
* Parquet
* JSONL

### AI

* Google Gemini
* Groq
* OpenRouter
* Hugging Face
* Ollama
* OpenAI

---

## Important Files

### app.py

Main Streamlit application.

Responsible for:

* Navigation
* Operations dashboard
* Return investigation
* Cluster explorer
* Model evaluation
* AI Chat
* AI settings
* Live data UI

---

### src/api.py

FastAPI application providing:

* Return endpoints
* Live statistics
* Live generator controls
* Export endpoints
* Agent API integration

---

### src/live_server.py

Live/demo transaction generator.

Responsible for:

* Continuous synthetic transactions
* Risk-regime changes
* Persistence of live events
* Live statistics

---

### src/features.py

Point-in-time feature engineering.

Responsible for:

* Chronological processing
* Historical windows
* Leakage prevention
* Behavioral features

---

### src/model.py

Machine-learning model logic.

Responsible for:

* Logistic Regression baseline
* XGBoost model
* Probability generation
* Calibration

---

### src/policy.py

Cost-sensitive decision layer.

Responsible for:

* Threshold selection
* Expected-cost calculation
* AUTO_APPROVE
* VERIFY
* MANUAL_REVIEW

---

### src/network.py

Coordinated-account detection.

Responsible for:

* Graph creation
* Account relationships
* Cluster analysis
* Suspicious infrastructure signals

---

### src/explain.py

Risk explanation logic.

Responsible for:

* Feature contribution
* Human-readable risk factors
* Model explanations

---

### src/responder.py

Operational response generation.

Responsible for:

* Operational summary
* Merchant protocol
* Customer communication

---

### src/chat_agent.py

ReturnShield AI assistant logic.

Responsible for:

* Chat context
* Agent actions
* Tool routing
* ReturnShield-specific reasoning

---

### src/chatbot_api.py

AI-provider integration.

Responsible for:

* Provider configuration
* Cloud/local model calls
* Tool calling
* Response handling

---

### run_pipeline.py

Runs the model/data pipeline.

Typical workflow:

```
Generate Data
    |
    v
Feature Engineering
    |
    v
Train Models
    |
    v
Calibrate Probabilities
    |
    v
Optimize Policy
    |
    v
Evaluate Held-Out Test
    |
    v
Generate Reports
```

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

---

### Manual Startup

If the integrated launcher is not used, start FastAPI manually:

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Then start Streamlit in another terminal:

```bash
streamlit run app.py
```

---

## Main Services

### Streamlit

```
http://127.0.0.1:8501
```

### FastAPI

```
http://127.0.0.1:8000
```

### FastAPI Documentation

```
http://127.0.0.1:8000/docs
```

### Live Returns API

```
GET /api/v1/returns
```

### Live Statistics API

```
GET /api/v1/returns/stats
```

### Agent Chat API

```
POST /api/v1/agent/chat
```

---

## API Endpoints

### Live Returns

```
GET /api/v1/returns
```

Returns the current live transaction records.

### Live Statistics

```
GET /api/v1/returns/stats
```

Returns live operational statistics.

### Start Live Generator

```
POST /api/v1/returns/start
```

Starts continuous transaction generation.

### Stop Live Generator

```
POST /api/v1/returns/stop
```

Stops continuous transaction generation.

### Generate Transactions

```
POST /api/v1/returns/generate
```

Generates synthetic transactions on demand.

### Export Live Data

```
GET /api/v1/returns/export.csv
```

Exports live transaction history.

### Agent Chat

```
POST /api/v1/agent/chat
```

Provides AI chat and ReturnShield tool access.

---

## Project Structure

```
returnshield/
│
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
│   ├── raw/
│   ├── processed/
│   └── live/
│
├── models/
│
└── reports/
```

---

## Important Demo Note

All default records, live/demo transactions, outcomes, and reported model metrics are synthetic unless the system is connected to an external merchant data source.

Synthetic results must not be presented as real-world fraud rates or production performance.

For a hackathon demonstration, clearly identify evaluation results as **synthetic held-out test results**.

---

## Production Considerations

Before production deployment, the synthetic environment would need to be replaced or supplemented with real merchant data.

Productionization would require:

* Real transaction ingestion
* Merchant-specific cost configuration
* Model retraining
* Monitoring
* Drift detection
* Access control
* Audit logging
* Secure secret management
* Privacy controls
* Provider-specific security configuration
* Human review workflows
* Threshold governance

The current implementation is primarily a hackathon/prototype system.

---

## Limitations

The default model is trained and evaluated on synthetic data.

Synthetic results cannot establish real-world fraud prevalence.

Business-cost assumptions are illustrative and should be replaced with merchant-specific values.

The coordinated-account graph identifies suspicious relationships but does not prove fraud.

Free AI provider tiers are subject to provider-specific quotas and policies.

Local AI models depend on the hardware available on the demonstration machine.

Real merchant deployment would require additional security, governance, monitoring, and validation.

---

## Security and Secrets

Do not commit API keys or other secrets to Git.

Use environment variables or a local secret-management mechanism.

The repository should contain:

```
.env.example
```

But should not contain:

```
.env
```

Actual provider credentials should never be included in the repository.

---

## Environment Configuration

Provider credentials can be configured through the Chat AI Settings panel or supported environment variables.

Example:

```
OPENAI_API_KEY=...
OPENAI_MODEL=...
OPENAI_BASE_URL=...

GEMINI_API_KEY=...

GROQ_API_KEY=...

OPENROUTER_API_KEY=...

HF_API_KEY=...
```

Use only the variables required by the selected provider.

---

## Troubleshooting

### Port 8000 Already in Use

Check:

```powershell
netstat -ano | findstr :8000
```

Find the PID and stop the stale process if necessary:

```powershell
taskkill /PID <PID> /F
```

---

### Port 8501 Already in Use

Check:

```powershell
netstat -ano | findstr :8501
```

Then stop the stale Streamlit process if necessary.

---

### FastAPI Does Not Start

Check:

```
http://127.0.0.1:8000/docs
```

Then inspect:

```
logs/api.log
```

Also verify the project is being started from the correct folder.

---

### API Route Returns 404

Verify that the current API exposes:

```
GET /api/v1/returns
```

Using:

```
http://127.0.0.1:8000/docs
```

If the route is missing, an older ReturnShield copy may be running.

---

### Chat Provider Unavailable

Check:

* Provider selection
* API key
* Model name
* API base URL
* Provider quota
* Local Ollama status if Ollama is selected

The ReturnShield Agent API should remain the application tool layer.

---

## Future Improvements

Potential future development areas include:

* Real merchant data connectors
* Production database integration
* Stronger graph analytics
* Graph-based machine learning
* Online model monitoring
* Drift detection
* Automated threshold revalidation
* Model versioning
* Audit trails
* Role-based access
* Enterprise authentication
* Merchant-specific policy configuration
* Additional AI tool integrations
* Provider health monitoring
* Structured agent observability

---

## Core Design Principle

> **Predict risk with machine learning, make the business decision with an explicit cost-sensitive policy, and use AI to understand questions, retrieve relevant ReturnShield information, and communicate the result.**

This keeps ReturnShield measurable, auditable, operationally useful, and defense-only.

---

## Final Product Summary

ReturnShield is more than a fraud classifier.

It is a merchant decision-support system that combines:

* Machine Learning
* Cost Optimization
* Explainability
* Graph Analysis
* Real-Time Monitoring
* REST APIs
* AI Agent

The system turns a return request into an actionable merchant decision:

```
Return Request
    |
    v
Risk Assessment
    |
    v
Expected Loss
    |
    v
Cost-Sensitive Decision
    |
    +-------------------------+
    |            |            |
    v            v            v
AUTO_APPROVE  VERIFY  MANUAL_REVIEW
```

At the same time, the AI assistant gives operators a natural-language interface to the ReturnShield platform.

The result is a complete prototype for cost-sensitive merchant return-abuse prevention with real-time monitoring, explainability, coordinated-account analysis, and AI-assisted operations.
