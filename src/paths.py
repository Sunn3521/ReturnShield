"""One shared root for every filesystem path the application resolves.

Import order makes this subtle: modules resolve ``ROOT`` at *import time*, and
the API imports all of them. The rule is therefore simple -

    root() is the ONLY way any module may locate the project root.

Resolution order:

1. ``RETURNSHIELD_HOME``  - set by the delivery launcher; wins outright so a
   packaged app (source in ``app/``, assets beside the exe) behaves exactly
   like a source checkout.
2. CWD, when it looks like the *whole* project (``src/api.py`` AND the model
   bundle) - covers running ``python -m uvicorn src.api:app`` from a checkout.
   Requiring the assets matters: the packaged layout runs with cwd inside
   ``app/``, which has the source but not the models, and a source-only cwd
   check would resolve assets to a folder that does not exist.
3. This file's parent's parent - the checkout layout, unchanged behaviour.
"""

from __future__ import annotations

import os
from pathlib import Path


def _looks_like_full_project(candidate: Path) -> bool:
    return (candidate / "src" / "api.py").exists() and (
        candidate / "models" / "model_bundle.joblib"
    ).exists()


def root() -> Path:
    """The project root: holds src/, models/, data/, reports/."""
    explicit = os.getenv("RETURNSHIELD_HOME", "").strip()
    if explicit:
        return Path(explicit)
    cwd = Path.cwd()
    if _looks_like_full_project(cwd):
        return cwd
    checkout = Path(__file__).resolve().parent.parent
    # Packaged layout in disguise: source under app/, assets one level above it.
    # Walking up from this file (never from cwd) keeps the search bounded to the
    # application's own folder, so an unrelated models/ up the tree is ignored.
    for candidate in (checkout, *checkout.parents):
        if (candidate / "models" / "model_bundle.joblib").exists():
            return candidate
    return checkout
