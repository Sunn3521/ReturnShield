"""Held-out evaluation for the ReturnShield intent classifier.

These questions are deliberately **not** in :mod:`src.intent_data` - they were
written to look like real users typing, including misspellings, filler words and
casual phrasing. Evaluating on the training set would prove nothing, so this file
is the only honest measure of intent accuracy.

``evaluate_pipeline`` scores the whole ``recognize()`` path, not just the
sklearn model, because the keyword rules short-circuit ahead of the classifier.
A change is only an improvement if the end-to-end number moves.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from sklearn.metrics import classification_report, f1_score

# (question, expected intent) - held out from training.
GOLDEN_QUESTIONS: List[Tuple[str, str]] = [
    # ratio
    ("what proportion of returns are abusive", "ratio"),
    ("give me the fraud percentage", "ratio"),
    ("how does fraud break down against legit", "ratio"),
    ("whats the abuse to legitimate ratio", "ratio"),
    # operations
    ("how are operations looking right now", "operations"),
    ("i want the operational overview", "operations"),
    ("brief ops picture please", "operations"),
    # trend
    ("is risk going up over time", "trend"),
    ("show me the risk movement lately", "trend"),
    ("has abuse been trending", "trend"),
    # top_risk
    ("which returns should i worry about", "top_risk"),
    ("give me the worst offenders by risk", "top_risk"),
    ("riskiest returns right now", "top_risk"),
    ("top risky transactions please", "top_risk"),
    ("show me returns with the highest abuse probability", "top_risk"),
    # model_metrics
    ("what is the pr auc", "model_metrics"),
    ("how well is the model performing", "model_metrics"),
    ("show me f1 and precision", "model_metrics"),
    ("whats the brier score", "model_metrics"),
    ("model quality numbers", "model_metrics"),
    # calibration
    ("is the risk score calibrated", "calibration"),
    ("show predicted vs observed", "calibration"),
    ("calibration curve please", "calibration"),
    # clusters
    ("any fraud rings active", "clusters"),
    ("show coordinated accounts", "clusters"),
    ("which accounts are linked together", "clusters"),
    ("abuse ring explorer", "clusters"),
    # live_status
    ("is the feed still alive", "live_status"),
    ("live server health check", "live_status"),
    ("are transactions still flowing", "live_status"),
    ("whats the live feed doing", "live_status"),
    # inspect_return
    ("inspect return R001234", "inspect_return"),
    ("drill into return R001234", "inspect_return"),
    ("why did you flag return R001234", "inspect_return"),
    ("give me everything on return R001234", "inspect_return"),
    # customer_search
    ("look up customer C012345", "customer_search"),
    ("what has C012345 returned", "customer_search"),
    ("show me C012345 activity", "customer_search"),
    ("customer history for C012345", "customer_search"),
    # start_live
    ("can you start the feed", "start_live"),
    ("switch on live transactions", "start_live"),
    ("begin streaming events", "start_live"),
    # stop_live
    ("can you stop the feed", "stop_live"),
    ("switch off live transactions", "stop_live"),
    ("kill the live stream", "stop_live"),
    # generate
    ("generate 40 more records", "generate"),
    ("make me 12 transactions", "generate"),
    ("i need 75 new events", "generate"),
    ("create some more returns please", "generate"),
    # export
    ("i need this as a csv", "export"),
    ("can you export the data", "export"),
    ("download the returns file", "export"),
    ("save everything to csv", "export"),
    # help
    ("what am i able to ask you", "help"),
    ("list your commands", "help"),
    ("how do i use this thing", "help"),
    # greeting
    ("hey you", "greeting"),
    ("greetings agent", "greeting"),
    ("good afternoon", "greeting"),
    ("hey how are you", "greeting"),
    # summary
    ("give me the overall picture", "summary"),
    ("summarise how things are", "summary"),
    ("whats the current situation", "summary"),
    ("brief me on returnshield", "summary"),
]

GOLDEN_TEXTS = [q for q, _ in GOLDEN_QUESTIONS]
GOLDEN_LABELS = [label for _, label in GOLDEN_QUESTIONS]


def evaluate_classifier(predict_fn) -> Dict[str, Any]:
    """Score a ``text -> intent`` callable against the golden set.

    Returns accuracy, macro F1 and the per-class breakdown plus the worst
    confusions, so a regression points at a specific pair of intents.
    """
    predicted = [str(predict_fn(text)) for text in GOLDEN_TEXTS]
    macro_f1 = float(f1_score(GOLDEN_LABELS, predicted, average="macro", zero_division=0))
    micro_f1 = float(f1_score(GOLDEN_LABELS, predicted, average="micro", zero_division=0))
    accuracy = sum(1 for p, g in zip(predicted, GOLDEN_LABELS) if p == g) / len(GOLDEN_LABELS)

    report = classification_report(
        GOLDEN_LABELS, predicted, zero_division=0, output_dict=True
    )

    confusions: List[Dict[str, Any]] = []
    for text, got, want in zip(GOLDEN_TEXTS, predicted, GOLDEN_LABELS):
        if got != want:
            confusions.append({"question": text, "expected": want, "got": got})

    per_intent = {
        label: {
            "f1": round(float(report.get(label, {}).get("f1-score", 0.0)), 4),
            "support": int(report.get(label, {}).get("support", 0)),
        }
        for label in sorted(set(GOLDEN_LABELS))
    }
    worst = sorted(per_intent.items(), key=lambda kv: kv[1]["f1"])[:5]

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "n": len(GOLDEN_LABELS),
        "per_intent": per_intent,
        "worst_intents": [{"intent": k, **v} for k, v in worst],
        "confusions": confusions,
    }


def evaluate_pipeline(agent) -> Dict[str, Any]:
    """Score the agent's full ``recognize()`` path, rules included."""
    from src.chat_agent import ChatContext

    def predict(text: str) -> str:
        intent, _confidence, _slots = agent.recognize(text, ChatContext())
        return intent

    return evaluate_classifier(predict)
