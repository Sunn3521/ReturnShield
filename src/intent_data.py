"""Training data for the ReturnShield intent classifier.

The original classifier was trained on 56 hand-written utterances covering only
12 intents, while ``ReturnShieldChatAgent.recognize`` can route 17. Anything the
keyword rules missed fell through to a model that had never seen five of those
intents at all.

This module keeps the original utterances (so nothing regresses) and adds seeds
for the missing intents, then expands both with cheap, deterministic
augmentation - politeness wrappers, phrasing variants and light typos - so the
linear model sees realistic variation instead of five near-identical strings.

Nothing here depends on anything outside the standard library, so this stage
needs no new packages and no downloaded weights.
"""

from __future__ import annotations

import random
from typing import Dict, List

# The intents the agent actually routes to. Keep this list in sync with the
# keyword rules in chat_agent.recognize.
INTENTS: List[str] = [
    "ratio",
    "operations",
    "trend",
    "top_risk",
    "model_metrics",
    "calibration",
    "clusters",
    "live_status",
    "inspect_return",
    "customer_search",
    "start_live",
    "stop_live",
    "generate",
    "export",
    "help",
    "greeting",
    "summary",
]

# Seed utterances per intent. These are the "clean" phrasings; augmentation below
# produces the noisy variants the classifier has to cope with.
SEEDS: Dict[str, List[str]] = {
    "ratio": [
        "what is the fraud ratio",
        "show the abuse ratio",
        "what percent of returns are abusive",
        "fraud versus legitimate ratio",
        "what percentage is fraud",
        "ratio of abusive to legitimate returns",
        "how many returns are fraud",
        "what share of returns are abusive",
        "show abusive percentage",
        "legit versus abusive split",
    ],
    "operations": [
        "show the operations overview",
        "brief overview of operations",
        "how are operations looking",
        "current operations status",
        "operational overview please",
        "give me an operations summary",
        "how is the ops dashboard looking",
        "show operations picture",
        "what does operations look like now",
    ],
    "trend": [
        "show the risk trend",
        "how is risk changing over time",
        "is risk trending up",
        "show trend over time",
        "plot the change in risk",
        "are returns getting riskier",
        "risk trend line please",
        "show me how risk is moving",
        "trend of abuse risk",
    ],
    "top_risk": [
        "show the highest risk returns",
        "which returns are most risky",
        "show high risk transactions",
        "give me the top risky returns",
        "what are the highest abuse risk returns",
        "list the riskiest returns",
        "top 10 riskiest returns",
        "show returns most likely to be abuse",
        "which transactions are highest risk",
        "worst returns by risk",
    ],
    "model_metrics": [
        "show model performance",
        "what are the held out metrics",
        "show precision recall f1",
        "how accurate is the model",
        "show the evaluation metrics",
        "what is the model pr auc",
        "how good is the classifier",
        "show roc auc and brier",
        "model accuracy numbers",
        "what are the test metrics",
    ],
    "calibration": [
        "show the calibration curve",
        "is the model calibrated",
        "predicted versus observed abuse",
        "show the outcomes graph",
        "calibration plot please",
        "are predictions calibrated",
        "show predicted vs actual",
        "calibration of the risk scores",
    ],
    "clusters": [
        "show abuse clusters",
        "find coordinated accounts",
        "show fraud rings",
        "what coordinated clusters are active",
        "show the abuse network",
        "any fraud rings detected",
        "show the cluster explorer",
        "which accounts are coordinated",
        "find abuse networks",
        "show linked accounts",
    ],
    "live_status": [
        "is the live server running",
        "show live server status",
        "what is the current live feed status",
        "are transactions coming in",
        "how fast is the live feed",
        "live server health",
        "is the feed still on",
        "show feed status",
        "how many events have we generated",
        "live transaction status please",
    ],
    "inspect_return": [
        "inspect return R006626",
        "show details for return R006626",
        "investigate return R006626",
        "what happened with return R006626",
        "tell me about return R006626",
        "why is return R006626 risky",
        "open return R006626",
        "explain return R006626",
        "drill into return R006626",
        "give me the full detail on return R006626",
    ],
    "customer_search": [
        "show customer C06472",
        "find customer C06472",
        "what returns did customer C06472 make",
        "show transactions for this customer",
        "search for a customer",
        "lookup customer C06472",
        "history for customer C06472",
        "all returns for customer C06472",
        "what has this customer returned",
        "customer C06472 activity",
    ],
    "start_live": [
        "start the live server",
        "turn on the live feed",
        "start generating transactions",
        "begin the live stream",
        "start live",
        "kick off the live feed",
        "run the live generator",
        "please start transactions",
    ],
    "stop_live": [
        "stop the live server",
        "turn off the live feed",
        "stop generating transactions",
        "halt the live stream",
        "stop live",
        "shut down transactions",
        "pause the live feed",
        "stop the generator please",
    ],
    "generate": [
        "generate 25 transactions",
        "create 50 more returns",
        "make 10 new records",
        "produce 100 events",
        "generate more data",
        "add 75 transactions",
        "make some fake returns",
        "generate 5 records please",
    ],
    "export": [
        "export the live data",
        "download the csv",
        "give me a csv of returns",
        "export transactions to csv",
        "download a report",
        "save the returns as csv",
        "i need a csv export",
        "export everything to a file",
    ],
    "help": [
        "what can you do",
        "help me",
        "what can i ask you",
        "show available commands",
        "how do i use this",
        "what are my options",
        "give me the command list",
        "explain the features",
    ],
    "greeting": [
        "hello",
        "hi there",
        "hey",
        "good morning",
        "good evening",
        "howdy",
        "hi",
        "hello there",
    ],
    "summary": [
        "give me a summary",
        "summarize the current data",
        "how is returnshield doing",
        "what is happening in the returns data",
        "show me the current overview",
        "brief summary please",
        "overall picture",
        "give me an overview",
        "summarise everything",
        "how are we doing overall",
    ],
}

# Original utterances kept verbatim so behaviour never regresses.
LEGACY_EXAMPLES: Dict[str, List[str]] = {
    "live_status": [
        "is the live server running", "show live server status",
        "what is the current live feed status", "are transactions coming in",
        "how fast is the live feed",
    ],
    "top_risk": [
        "show the highest risk returns", "which returns are most risky",
        "show high risk transactions", "give me the top risky returns",
        "what are the highest abuse risk returns",
    ],
    "summary": [
        "give me a summary", "summarize the current data", "how is returnshield doing",
        "what is happening in the returns data", "show me the current overview",
    ],
    "model_metrics": [
        "show model performance", "what are the held out metrics",
        "show precision recall f1", "how accurate is the model",
        "show the evaluation metrics",
    ],
    "inspect_return": [
        "inspect return LIVE-123", "show details for return LIVE-123",
        "investigate return LIVE-123", "what happened with return LIVE-123",
        "tell me about this return",
    ],
    "customer_search": [
        "show customer LIVE-C123", "find customer C123",
        "what returns did customer C123 make",
        "show transactions for this customer", "search for a customer",
    ],
    "clusters": [
        "show abuse clusters", "find coordinated accounts", "show fraud rings",
        "what coordinated clusters are active", "show the abuse network",
    ],
    "start_live": [
        "start the live server", "turn on the live feed",
        "start generating transactions", "start live",
    ],
    "stop_live": [
        "stop the live server", "turn off the live feed",
        "stop generating transactions", "stop live",
    ],
    "generate": [
        "generate transactions", "create more returns", "make new records",
    ],
    "export": [
        "export the live data", "download csv", "export data",
    ],
    "help": [
        "what can you do", "help", "what can i ask",
    ],
}

_PREFIXES = [
    "", "", "",  # unprefixed stays the most common
    "can you ", "could you ", "please ", "hey ", "ok ", "so ",
    "i want to ", "i need to ", "let me ", "help me ",
    "quick question - ", "quick one - ",
]

_SUFFIXES = [
    "", "", "",  # unsuffixed stays the most common
    " please", " now", " right now", " thanks", " for me",
    " if you can", " real quick",
]


def _typo(text: str, rng: random.Random) -> str:
    """Introduce one light typo so the model sees realistic user input."""
    if len(text) < 6:
        return text
    idx = rng.randrange(1, len(text) - 2)
    mode = rng.random()
    if mode < 0.4:                      # transpose
        return text[:idx] + text[idx + 1] + text[idx] + text[idx + 2:]
    if mode < 0.7:                      # drop
        return text[:idx] + text[idx + 1:]
    return text[:idx] + text[idx] + text[idx:]  # duplicate


def _augment(seed: str, rng: random.Random, per_seed: int) -> List[str]:
    """Produce deterministic variants of one seed utterance."""
    out: List[str] = []
    for _ in range(per_seed):
        variant = seed
        roll = rng.random()
        if roll < 0.15:
            variant = _typo(variant, rng)
        prefix = rng.choice(_PREFIXES)
        suffix = rng.choice(_SUFFIXES)
        candidate = f"{prefix}{variant}{suffix}".strip()
        candidate = " ".join(candidate.split())
        if candidate:
            out.append(candidate)
    return out


def build_dataset(
    per_seed: int = 6,
    seed: int = 20240917,
    include_legacy: bool = True,
) -> Dict[str, List[str]]:
    """Build the augmented intent dataset.

    Deterministic for a given ``seed`` so training and evaluation are
    reproducible. Seeds are always kept verbatim alongside their variants.
    """
    rng = random.Random(seed)
    dataset: Dict[str, List[str]] = {}

    for intent in INTENTS:
        seen = set()
        rows: List[str] = []

        if include_legacy:
            for text in LEGACY_EXAMPLES.get(intent, []):
                norm = " ".join(text.lower().split())
                if norm not in seen:
                    seen.add(norm)
                    rows.append(text)

        for text in SEEDS.get(intent, []):
            norm = " ".join(text.lower().split())
            if norm not in seen:
                seen.add(norm)
                rows.append(text)
            for variant in _augment(text, rng, per_seed):
                vnorm = variant.lower()
                if vnorm not in seen:
                    seen.add(vnorm)
                    rows.append(variant)

        dataset[intent] = rows
    return dataset


def as_pairs(dataset: Dict[str, List[str]]) -> tuple[List[str], List[str]]:
    """Flatten the dataset into (texts, labels) for sklearn."""
    texts: List[str] = []
    labels: List[str] = []
    for intent, rows in dataset.items():
        for text in rows:
            texts.append(text)
            labels.append(intent)
    return texts, labels


def dataset_stats(dataset: Dict[str, List[str]]) -> dict:
    sizes = {k: len(v) for k, v in dataset.items()}
    return {
        "intents": len(sizes),
        "utterances": sum(sizes.values()),
        "min_per_intent": min(sizes.values()) if sizes else 0,
        "max_per_intent": max(sizes.values()) if sizes else 0,
        "per_intent": sizes,
    }
