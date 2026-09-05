# ReturnShield AI

Cost-sensitive return-abuse risk and response agent for merchants.

## Product overview

ReturnShield evaluates return requests using only information available at request time, produces a calibrated abuse-risk score, and maps that risk to an operational decision:

- **AUTO_APPROVE** — low risk
- **VERIFY** — moderate risk; request additional evidence
- **MANUAL_REVIEW** — highest risk; route to manual review

The system separates **prediction** from **policy**: the ML model estimates risk, while the deterministic policy engine selects the operational action using validation-set business costs.

## Core capabilities

- Point-in-time synthetic event simulation
- Leakage-safe 30-day/90-day historical features
- Logistic-regression baseline and XGBoost model
- Probability calibration
- Validation-based cost optimization
- Strict chronological held-out evaluation
- SHAP-based risk explanations
- Suspicious coordinated-account / abuse-ring signals
- Live transaction REST API and continuously generated synthetic events
- Real-time Operations Overview
- Full Return Investigator workflow with live request refresh
- Cluster Explorer with coordinated-account graph and high-risk account table
- CSV export of generated live history by time range
- Hover tooltips for graph/chart data
- ChatGPT-style AI Chat interface
- Right-side AI Settings drawer
- Multiple AI provider options, including local Ollama
- ReturnShield Agent API tool/action integration

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

For normal hackathon use on Windows, the recommended entry point is:

```text
start_agent.bat
```

The launcher starts the FastAPI server and Streamlit dashboard from the current project folder, verifies the expected API route, and records startup logs under `logs/`.

## Project structure

```text
ReturnShield/
├── app.py
├── run_pipeline.py
├── requirements.txt
├── README.md
├── DEMO.md
├── .env.example
├── start_agent.bat
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── chatbot_api.py
│   ├── chat_agent.py
│   ├── simulate.py
│   ├── live_server.py
│   ├── features.py
│   ├── model.py
│   ├── policy.py
│   ├── explain.py
│   ├── network.py
│   ├── responder.py
│   └── pipeline.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── live/
├── models/
└── reports/
```

## Evaluation design

ReturnShield uses a chronological split:

- **60% earliest return requests** → training
- **next 20%** → validation, calibration, model selection, and policy optimization
- **final 20%** → strictly held-out test evaluation

The held-out test set is not used to tune the model or operating thresholds.

### Prediction metrics

- PR-AUC
- ROC-AUC
- Precision
- Recall
- F1
- Brier score

### Business metrics

- Expected merchant loss
- False-positive cost
- False-negative cost
- Verification cost
- Manual-review rate
- Expected loss per 1,000 returns

### Current synthetic test artifacts

The included report records the current synthetic held-out run. Treat these values as **demo/evaluation results**, not production performance or real merchant fraud rates.

## Risk and policy logic

The model produces a calibrated probability of abusive return risk. A validation-set policy then maps the probability to one of the operational actions.

```text
Risk probability
      |
      +---- low --------> AUTO_APPROVE
      |
      +---- moderate ---> VERIFY
      |
      +---- high -------> MANUAL_REVIEW
```

The threshold values are selected on validation data using explicit merchant costs rather than assuming a default 0.5 classification threshold.

## Live transaction REST server

The project includes a built-in synthetic live transaction server for demos and integration testing.

Start it manually with:

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Or use `start_agent.bat` to start the full stack.

### Main API endpoints

```text
GET  /api/v1/meta
GET  /api/v1/health
GET  /api/v1/returns
GET  /api/v1/returns/stats
POST /api/v1/returns/start
POST /api/v1/returns/stop
POST /api/v1/returns/generate
GET  /api/v1/returns/export.csv
POST /api/v1/agent/chat
```

`GET /api/v1/returns` supports pagination and filtering, including `limit`, `offset`, `search`, `before`, and `after` where applicable.

The live store keeps a bounded in-memory window for fast dashboard rendering and an append-only JSONL history at:

```text
data/live/events.jsonl
```

This allows older generated events to remain available for time-bounded CSV export even after the live UI window has moved on.

## Real-time dashboard behavior

When connected to the live server, the dashboard updates the active live data without requiring navigation changes.

The live layer targets approximately **1-second polling** and updates:

- Operations KPIs
- Risk distribution
- Policy-action distribution
- Highest-risk return table
- Return Investigator request list
- Cluster Explorer data
- Coordinated-account graph
- Live server status

The live visual layer is designed to avoid full-page Streamlit refreshes. The browser-side live components update their displayed data in place, and transient stale-element opacity transitions are disabled to reduce flicker.

Charts use a rolling live window for responsiveness, while cumulative live KPI counters can continue increasing during the current live session.

## Operations Overview

The Operations Overview is designed for an always-on merchant view. The page includes live KPI cards, risk/action visualizations, and current high-risk records.

The action visualization is a **pie chart**, not a donut chart.

Chart elements provide hover information where supported. Hovering a bar, pie slice, data point, or network node can show the data associated with that element.

## Return Investigator

Return Investigator remains a full investigation workflow rather than a simplified live table.

It supports:

- Return Request to Inspect selection
- Search by return/customer identifier
- Pagination / request count controls
- Risk probability and decision
- Order and return values
- Expected merchant loss
- Return reason
- Historical return rate and refund totals
- Recent return velocity
- Hours-to-return signal
- Linked device/address account signals
- Key risk signals
- Operational summary
- Merchant protocol
- Customer communication draft

In live mode, the available return-request selection list is refreshed from the live source when the selector is reopened or refreshed. The selected investigation workflow itself is not replaced by the live refresh.

## Cluster Explorer

Cluster Explorer highlights suspicious coordinated-account relationships using shared infrastructure signals.

The live relationship graph emphasizes high-risk customers and shared device/address/payment relationships so the graph remains interpretable. The accompanying table exposes the detected coordinated accounts and their risk-related aggregate information.

Shared infrastructure is treated as a **risk signal**, not proof of wrongdoing.

## Hover information

Charts and graphs use contextual hover information where the rendering technology supports it.

Examples include:

- Risk probability and count for risk bars
- Action, count, and percentage for pie slices
- Exact metric values for model-performance charts
- Predicted versus observed values for calibration plots
- Customer, cluster, and relationship information for network nodes

## AI Chat

The AI Chat tab is a general-purpose conversational interface for ReturnShield.

It is not limited to a fixed list of canned questions. The selected AI model can handle general questions and can also use ReturnShield tools when the question requires current application data or an action.

Examples:

```text
What is precision versus recall?

Give me a brief overview of current operations.

What is the current real versus abusive return ratio?

Show the highest-risk returns.

Inspect LIVE-ABC12345.

Why was this return flagged?

How is the live server doing?

Show coordinated accounts.

Explain the current model metrics.
```

### AI settings

The Chat header contains a settings control. Clicking the settings control opens a **right-side AI Settings drawer** without moving the conversation layout.

Suggested-query buttons are intentionally not shown.

Available provider options include:

- **OpenRouter (Free)**
- **Groq (Free)**
- **Google Gemini**
- **Hugging Face**
- **Ollama (Local)** — no API key required
- **OpenAI** — optional

The selected AI provider is the language/reasoning layer. The ReturnShield Agent API remains the application/tool layer.

```text
User
  |
  v
AI Chat
  |
  v
Selected LLM provider
  |
  v
ReturnShield Agent API / tools
  |
  +--> live returns
  +--> operations summary
  +--> return/customer investigation
  +--> coordinated accounts
  +--> model metrics
  +--> live-server actions
  |
  v
Tool result
  |
  v
Natural-language response
```

The model cannot arbitrarily change the deterministic risk decision. Allow-listed application actions remain controlled by the ReturnShield tool layer.

## AI provider configuration

Copy `.env.example` to `.env` and configure the provider you want to use.

For a zero-cost local setup, use Ollama:

```text
AI_PROVIDER=Ollama (Local)
AI_MODEL=gemma3
AI_BASE_URL=http://localhost:11434/v1
```

For cloud providers, enter only the corresponding API key and select a compatible model.

## Chat UI behavior

The Chat page is intentionally structured as:

- Fixed chat header
- Fixed AI Settings control
- Scrollable conversation region
- Fixed message composer at the bottom

The page itself is not intended to become a second scrolling surface. Conversation history scrolls inside the conversation region so the composer remains accessible.

## Safety / defense-only scope

ReturnShield is designed for merchant loss prevention.

It does not automate fraud accusations, account bans, exploit generation, or offensive behavior. High-risk cases are routed to verification or manual review.

Shared infrastructure is treated as a signal and does not prove wrongdoing.

The AI/LLM layer is not the source of truth for the risk score or policy decision.

## Synthetic-data disclosure

The included live generator, training data, outcomes, and demo metrics are synthetic unless ReturnShield is connected to a real merchant system.

Synthetic benchmark values must not be presented as real-world fraud prevalence, production precision/recall, or guaranteed merchant savings.

## Troubleshooting

### API import error

Run the following from the project root:

```bash
python -c "import src.api; print('API import OK')"
```

### Verify live API

Open:

```text
http://127.0.0.1:8000/docs
```

and confirm `/api/v1/returns` is present.

### Port already in use

The supplied `start_agent.bat` attempts to close stale listeners on ports 8000 and 8501 before starting the current project.

If a service is still holding a port:

```powershell
netstat -ano | findstr :8000
netstat -ano | findstr :8501
```

### Startup logs

The launcher writes:

```text
logs/api.log
logs/streamlit.log
```

These logs should be checked before troubleshooting the application code.
