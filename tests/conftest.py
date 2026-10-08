"""Suite-wide guards.

Policy control writes ``models/policy.json``. A test that exercises it without
redirecting that path would silently rewrite the thresholds every other test -
and the running demo - load, which surfaces much later as an unexplained
"why did the policy change?". Redirecting the path for *every* test makes that
impossible rather than merely discouraged; individual tests may still point
somewhere else when they need to.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _sandbox_policy_writes(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("policy-writes") / "policy.json"
    monkeypatch.setenv("RETURNSHIELD_POLICY_PATH", str(sandbox))
    return sandbox