"""Contract tests for the dashboard -> API chat payload.

The Streamlit dashboard posts a projection of its frame to the agent. Two
silent-failure bugs lived in that seam:

* ``to_dict("records")`` left numpy Timestamps that json.dumps cannot encode, so
  every dashboard chat turn raised TypeError and fell back to the local agent;
* the projection could omit a column the agent reads, changing its answer.

These tests pin both.
"""

from __future__ import annotations

import ast
import json

import pandas as pd
import pytest

from src.chat_agent import AGENT_COLUMNS

#: Names in chat_agent that hold DataFrames (or a row Series) rather than results.
_RECEIVERS = {"df", "match", "work", "row", "data", "subset"}
#: Dict keys on the result envelope and columns the agent creates itself.
_NOT_COLUMNS = {"answer", "data", "action", "intent", "confidence", "type", "count", "_ts"}


def _columns_referenced_by_agent() -> set[str]:
    """Every string literal the agent reads off a DataFrame, via the AST.

    Deliberately narrow: subscript reads (``df["col"]``), ``df.get("col")`` and
    ``"col" in df.columns``. Keyword-phrase lists are not DataFrame access and
    are not collected.
    """
    import src.chat_agent as chat_agent

    tree = ast.parse(open(chat_agent.__file__, encoding="utf-8").read())
    found: set[str] = set()

    def add(node) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value)

    def receiver(node) -> str | None:
        base = node
        while isinstance(base, (ast.Attribute, ast.Subscript)):
            base = base.value
        return base.id if isinstance(base, ast.Name) else None

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and receiver(node.value) in _RECEIVERS:
            add(node.slice)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "get" and receiver(node.func) in _RECEIVERS and node.args:
            add(node.args[0])
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Attribute) and comparator.attr == "columns" \
                        and receiver(comparator) in _RECEIVERS:
                    add(node.left)
    return found


def test_agent_columns_covers_everything_the_agent_reads():
    referenced = {c for c in _columns_referenced_by_agent() - _NOT_COLUMNS if c.islower()}
    missing = sorted(referenced - set(AGENT_COLUMNS))
    assert not missing, (
        "chat_agent reads columns AGENT_COLUMNS omits, so the dashboard would "
        f"silently stop sending them: {missing}"
    )


def test_agent_columns_are_unique():
    duplicates = [c for c in set(AGENT_COLUMNS) if AGENT_COLUMNS.count(c) > 1]
    assert not duplicates, f"duplicate columns in AGENT_COLUMNS: {duplicates}"
    assert {"return_id", "customer_id", "risk_probability", "decision"} <= set(AGENT_COLUMNS)


def test_projected_frame_is_json_serialisable():
    """Regression: to_dict("records") raised TypeError on prediction_time."""
    predictions = pd.read_csv("reports/test_predictions.csv", parse_dates=["prediction_time"])
    cols = [c for c in AGENT_COLUMNS if c in predictions.columns]
    projected = predictions[cols].head(200)

    with pytest.raises(TypeError):
        json.dumps(projected.where(pd.notna(projected), None).to_dict("records"))

    records = json.loads(projected.to_json(orient="records", date_format="iso"))
    assert len(records) == len(projected)
    assert json.dumps({"records": records})  # round-trips as a real request body

    # Must survive as a real timestamp, not epoch millis (which read as 1970).
    sent = records[0]["prediction_time"]
    assert isinstance(sent, str), sent
    parsed = pd.to_datetime(sent)
    assert parsed.year >= 2020, f"timestamp did not round-trip: {sent} -> {parsed}"
