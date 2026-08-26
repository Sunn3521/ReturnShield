# 3-minute judge demo

## 0:00–0:25 — Problem

“ReturnShield decides how much friction to apply to a return request. It predicts return-abuse risk using only information known at request time, then chooses approve, verify, or manual review based on merchant cost.”

## 0:25–1:10 — Live investigation

1. Open **Investigate**.
2. Select a high-risk return from the dropdown.
3. Show the risk score, expected loss, decision, and evidence.
4. Emphasize that the decision is deterministic; the explanation layer cannot override it.

## 1:10–1:45 — Network signal

Use the linked-account counts in the investigation view to explain that device/address sharing is treated as a risk signal, not proof of wrongdoing.

## 1:45–2:30 — Metrics

Open **Model Performance** and show:

- PR-AUC
- ROC-AUC
- Brier score
- policy thresholds
- verification/manual-review rates
- expected loss per 1,000 returns
- expected cost savings vs. approve-all

Current synthetic run artifacts are in `reports/final_report.json`.

## 2:30–3:00 — Close

“The important metric is not accuracy. It is merchant loss under a controlled operating policy. On this synthetic held-out period, ReturnShield reduced expected cost by 18.6% versus approving every return, while retaining high ranking power on a rare positive class.”

## Do not claim

- real-world fraud prevalence
- production accuracy
- that shared infrastructure proves fraud
- that the LLM detects fraud
- that synthetic performance will transfer directly to a production merchant
