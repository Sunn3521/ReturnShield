# ReturnShield AI

Cost-sensitive return-abuse risk and response agent for merchants.

## What this product does

ReturnShield predicts the probability that a return request will eventually cause abusive merchant loss, using only information available at the time of the request. It then applies a validation-set-optimized policy to choose one of:

- **AUTO_APPROVE** — low risk
- **VERIFY** — request additional evidence
- **MANUAL_REVIEW** — highest risk

The product includes:

- point-in-time synthetic event simulation
- leakage-safe historical features
- logistic-regression baseline and XGBoost model
- probability calibration
- validation-based cost optimization
- held-out temporal test evaluation
- SHAP explanations
- suspicious network/abuse-ring signals
- Streamlit operations + investigation + model-performance UI
- optional LLM explanation hook (never controls the decision)

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py
```

`run_pipeline.py` generates data, trains the model, calibrates probabilities, optimizes the policy on validation data, and evaluates once on the held-out test period.

The synthetic dataset is intentionally generated from latent behavioral processes rather than assigning the target randomly. The test split is chronological, and all historical features are computed from events strictly prior to each return request.

## Files

```text
returnshield/
├── app.py
├── run_pipeline.py
├── requirements.txt
├── README.md
├── src/
│   ├── simulate.py
│   ├── features.py
│   ├── model.py
│   ├── policy.py
│   ├── explain.py
│   └── pipeline.py
├── data/
├── models/
└── reports/
```

## Evaluation design

The pipeline uses a chronological split:

- 60% earliest return requests: training
- next 20%: validation/model and policy selection
- final 20%: held-out test

Primary prediction metrics:

- PR-AUC
- ROC-AUC
- Precision
- Recall
- F1
- Brier score

Business metrics:

- expected merchant loss
- false-positive cost
- false-negative cost
- verification cost
- manual-review rate
- expected loss per 1,000 returns

## Safety / defense-only scope

This is designed strictly for loss prevention. It does not automate accusations, account bans, or exploit generation. It recommends operational friction and manual review. The LLM layer, if connected, is an explanation/response generator only; it cannot override the deterministic policy decision.

## Important demo note

All records and metrics are synthetic. Do not present placeholder or synthetic results as production fraud rates. Use the generated held-out test report from your run.
