# ReturnShield AI

Cost-sensitive return-abuse **decision API**, plus the operations dashboard used
to test it.

**Version:** 2.1.0 (service) / 2.1.0.0 (Windows EXE file version) — reported by
`GET /api/v1/meta` and by the shipped launcher's PE version resource.

The repository is split by application, so each half reads on its own:

```text
api/          FastAPI service: scoring, decisions, reviews, live feed, agent tools
dashboard/    Streamlit operations console (starts empty, loads its own data)
src/          shared engine: model, policy, features, pipeline, explanations
sdk/          Python client for the decision API
tests/        pytest suite + the benchmarks quoted below
```

---

# The API

`api/main.py` is a single FastAPI service. It exposes the deterministic risk
model as an HTTP decision engine, keeps the review/decision history, and drives
the synthetic live returns feed used by demos and integration tests.

## Run it

```bash
pip install -r requirements.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
# equivalent: python -m api.main
```

- Interactive docs: <http://127.0.0.1:8000/docs>
- Machine-readable routes: `GET /openapi.json`
- Service discovery: `GET /api/v1/meta` (name, version, endpoint list,
  rate-limit and auth state)
- Liveness: `GET /api/v1/health`

The first start loads the model bundle and takes **~37 s** on a laptop; every
request after that is served from memory.

## Endpoints

```text
GET  /api/v1/meta                     service identity, config flags
GET  /api/v1/health                   liveness
GET  /api/v1/policy                   active thresholds + policy versions
POST /api/v1/score                    score one return → decision object
POST /api/v1/batch_score              score up to 1,000 returns in one call
GET  /api/v1/returns                  live window (limit/offset/search/before/after)
GET  /api/v1/returns/stats            live counters + event sequence
POST /api/v1/returns/start|stop|generate   control the synthetic feed
GET  /api/v1/returns/export.csv       time-bounded export
POST /api/v1/agent/chat               tool-using agent over live data
```

Decision write path (documented in [DECISION_API.md](DECISION_API.md)):

```text
POST /api/v1/events                   idempotent lifecycle ingest
POST /api/v1/events/backfill          historical ingest, same feature path
POST /api/v1/decisions/score          decision: action, reasons, versions
POST /api/v1/decisions                outcome feedback — closes the learning loop
GET  /api/v1/reviews                  human worklist, ranked by expected cost
POST /api/v1/reviews/{id}/resolve     resolve a review, actor recorded
GET  /api/v1/webhooks/deliveries      every outbound callback attempt
POST /api/v1/sandbox/test-webhook     signed ping to sandbox endpoints
```

## Auth, scopes and environments

Auth is **opt-in**, so a loopback demo works with no configuration:

| Variable | Effect |
| --- | --- |
| `RETURNSHIELD_API_KEY` | Legacy single key. When set, every route except the public probes needs `X-ReturnShield-Key` or `Authorization: Bearer <key>`. |
| `RETURNSHIELD_API_KEYS` | Scoped keys: `{"secret": ["score:write", "events:write"]}`, or a richer object with `scopes`, `environment`, `id`. |
| `RETURNSHIELD_ENVIRONMENT` | `production` (default) or `sandbox`. Sandbox traffic cannot move production thresholds or scores. |
| `RETURNSHIELD_RATE_LIMIT` | Requests/minute/client key. `0` (default) disables the limiter. |

Scopes: `score:write`, `events:read`, `events:write`, `decisions:write`,
`reviews:read`, `reviews:write`, `webhooks:read`, `admin`. Routes are matched
longest-prefix-first, so `/api/v1/decisions/score` checks `score:write` before
`/api/v1/decisions` checks `decisions:write`.

The limiter is a fixed in-process window: it protects a single-worker demo
server from a runaway client, not a multi-tenant deployment. `/api/v1/meta`
states that plainly.

Failures use one envelope — HTTP status, machine-readable `code`, human
`message`, and a `request_id` — described fully in
[DECISION_API.md](DECISION_API.md).

## Performance

Measured on this repository, laptop CPU, **one uvicorn worker**, model bundle
already loaded. Reproduce with the two scripts below; the numbers are from a
run on 8 October 2026.

### HTTP latency

```bash
python -m uvicorn api.main:app --port 8000 &
python tests/bench_api.py http://127.0.0.1:8000
```

| Endpoint | n | p50 | p95 | p99 | req/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `GET /api/v1/meta` | 150 | 4.5 ms | 9.5 ms | 26.5 ms | 173 |
| `GET /api/v1/health` | 150 | 3.5 ms | 14.9 ms | 21.2 ms | 211 |
| `GET /api/v1/returns/stats` | 100 | 3.5 ms | 5.7 ms | 26.5 ms | 227 |
| `GET /api/v1/returns?limit=100` | 100 | 46.2 ms | 77.5 ms | 92.8 ms | 19 |
| `POST /api/v1/score` (persisted) | 200 | 73.8 ms | 288.3 ms | 460.5 ms | 9 |
| `POST /api/v1/score` (persist on) | 100 | 66.1 ms | 170.9 ms | — | 10.9 |
| `POST /api/v1/score` (persist off) | 100 | 50.6 ms | 84.9 ms | — | 18.7 |

### Throughput

| Workload | Throughput |
| --- | ---: |
| Raw model inference, one batched call, 200 rows | **~8,000 rows/s** (24.9 ms) |
| Raw model inference vs row-at-a-time loop | **156× faster** batched |
| Scoring **with** SHAP explanations | **~50–60 rows/s** |
| `POST /api/v1/batch_score`, 500 rows, persist off | **43 rows/s** |
| `POST /api/v1/batch_score`, 500 rows, persist on | **22 rows/s** |
| 8 concurrent `POST /api/v1/score` clients | scales with the above |

```bash
python tests/bench_batch_score.py   # inference vs explanations, in-process
```

**Where the time goes.** The model itself is cheap — 200 rows score in ~25 ms.
SHAP explanations dominate: ~840 ms for 50 rows, which is why the explained
scoring path settles around 50–60 rows/s. Persisting the decision (audit write)
roughly doubles single-score latency. Practical guidance:

- Use `batch_score` instead of a loop of `score` — 156× on inference alone.
- Pass `store_decision=false` / `store_decisions=false` for backfills or
  benchmarks where the audit trail is not wanted.
- Explanations are only needed when a human reads them; drop them for bulk
  scoring.

### Model quality (held-out, synthetic)

Reported from `reports/final_report.json` — **demo/evaluation results on
synthetic data**, not production fraud rates.

| Metric | Value |
| --- | ---: |
| PR-AUC | 0.201 |
| ROC-AUC | 0.707 |
| Brier score | 0.0445 |
| Verify threshold (T1) | 0.120 |
| Manual-review threshold (T2) | 0.680 |
| Auto-approve rate | 91.9 % |
| Verification rate | 8.1 % |
| Expected cost vs approve-all | **18.6 % lower** (₹122,110 vs ₹150,000) |

Evaluation is chronological: earliest 60 % train → next 20 % validation,
calibration and threshold optimisation → final 20 % strictly held out. The
held-out set never tunes the model or the thresholds.

## Python client

```python
from sdk.returnshield import ReturnShieldClient, default_payload

with ReturnShieldClient(api_key="...") as rs:
    decision = rs.score_decision(default_payload("R999001"))
```

A test asserts every route the client calls exists in `GET /openapi.json`, so
the SDK cannot silently drift from the server.

---

# The dashboard (how to test it)

`dashboard/app.py` is a Streamlit console over the API: operations overview,
return investigator, cluster explorer, threshold studio, outcome ledger and an
AI chat.

## Start it

```bash
# dashboard only
streamlit run dashboard/app.py

# whole stack (API + dashboard + browser) — Windows delivery or checkout
ReturnShield.exe            # or: start_agent.bat
```

## It starts empty on purpose

Every session begins with **no data loaded**: the working set is an empty frame
and the sidebar reads *No dataset loaded*. On startup a dialog offers the three
ways in:

1. **Upload a dataset** — a returns CSV, scored by the active model and made the
   working set.
2. **Connect to a server** — base URL + returns endpoint (+ optional auth), or
   one click to start the built-in synthetic feed and attach to it.
3. **Bundled demo data** — the 1,490-row evaluation set that ships with the
   model report.

*Skip for now* keeps the dashboard empty. **Load Dataset…** in the sidebar
reopens the dialog at any time, and **Reset to Empty Dashboard** clears the
working set and reopens it.

## Test it

```bash
# 1. self-test against a running stack (API identity, feed, score, dashboard)
"Verify ReturnShield.bat"          # → ALL CHECKS PASSED

# 2. pytest suite (API, guard, decision API, memory, SDK contract, chat)
python -m pytest tests/ -q
```

What to check in the browser at <http://127.0.0.1:8501>:

- The startup dialog appears; after choosing a source the KPI row shows real
  counts (the demo set reports 1,490 returns, 1,369 auto-approved).
- **Operations Overview** — KPI cards, risk histogram, policy-action pie,
  highest-risk table; live mode updates in place at ~1 s.
- **Return Investigator** — search an ID, open a case, read the SHAP evidence
  and the merchant/customer response drafts.
- **Data & Live Server** — upload, connect, start/stop the synthetic feed,
  export a time-bounded CSV.
- **Threshold Studio** — moving T1/T2 re-optimises against the active window.

Stopping: close the launcher window, press Ctrl+C, or run
`ReturnShield.exe --stop`.

---

# Repository layout

```text
api/                  FastAPI service
  main.py             the app (uvicorn api.main:app)
  api_guard.py        opt-in auth + rate limiting
  api_keys.py         scoped keys, environments
  decision_store.py   decisions, reviews, outcomes
  webhooks.py         signed outbound callbacks
  memory.py           chat/session memory
  hindsight.py        recommendation layer
  chatbot_api.py      agent/tool configuration
  live_server.py      synthetic returns generator
  Dockerfile, docker-compose.yml
dashboard/
  app.py              Streamlit console
src/                  shared engine
  model.py policy.py features.py pipeline.py explain.py network.py
  responder.py chat_agent.py redteam.py local_model.py paths.py ...
sdk/returnshield.py   typed client
tests/                pytest + benchmarks (bench_api.py, bench_batch_score.py)
launcher/             PyInstaller spec, supervisor, version_info.txt
build_delivery.py     assembles dist/ReturnShield/ + the zip
dist/                 gitignored Windows delivery (regenerated, not committed)
data/ models/ reports/  assets: raw+processed data, model bundle, held-out report
```

`src/paths.py` is the only place that resolves the project root: it honours
`RETURNSHIELD_HOME` (set by the delivery launcher), then a working directory
that looks like the whole project, then this checkout. Source lives under
`api/` and `dashboard/` in the delivery while assets sit beside the exe, so no
module should locate assets relative to its own file.

## Windows delivery

```bash
python build_delivery.py --zip-runtime     # folder + exe + self-contained zip
```

`dist/ReturnShield/` is the self-contained delivery (bundled `runtime/`, so no
Python install is needed on the target) and is gitignored; see
[.gitignore](.gitignore).

# Safety / scope

ReturnShield is defense-only: it does not automate fraud accusations, account
bans or offensive behaviour. High-risk returns go to verification or manual
review, shared infrastructure is a **signal** and not proof of wrongdoing, and
the LLM layer never overrides the deterministic risk decision.

The live generator, training data, outcomes and the metrics above are
**synthetic** unless connected to a real merchant system. Do not present them as
real-world fraud prevalence, production precision/recall, or guaranteed savings.

# Troubleshooting

```bash
python -c "import api.main; print('API import OK')"    # import check
python -m py_compile dashboard/app.py                    # dashboard syntax check
```

- **Ports**: the launcher steps past a busy 8000/8501 and prints the port it
  chose; `ReturnShield.exe --stop` clears both.
- **Logs**: `logs/api.log`, `logs/streamlit.log`.
- **Model assets missing**: `python run_pipeline.py` (one-time, minutes).
