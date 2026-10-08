"""Provision a relocatable Python runtime inside the delivery folder.

Turns ``dist/ReturnShield`` from "needs Python 3.12 + pip install on the target
machine" into a genuinely self-contained folder: the launcher prefers
``runtime/python.exe`` when it exists and never touches the system Python.

Why the embeddable distribution rather than a venv: a venv records absolute
paths and breaks the moment the folder is moved or copied to another machine.
The embeddable zip is designed to be relocated - it only needs a ``._pth`` file
whose entries are relative, which is exactly what this script writes.

Steps (each is skipped when already done, so re-running is cheap):

    1. Download python-<VERSION>-embed-amd64.zip into build/cache/ (cached).
    2. Extract it into dist/ReturnShield/runtime/.
    3. Rewrite python312._pth to expose Lib\\site-packages and run ``import site``.
    4. pip install the delivery requirements into runtime/Lib/site-packages
       using the machine's Python (``--target``, wheels only, no bytecode).
    5. Write runtime/.returnshield-runtime with a version marker.

Usage:
    python build_runtime.py              # provision runtime/ in the stage
    python build_runtime.py --force      # rebuild even if the marker matches
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
STAGE = DIST / "ReturnShield"
RUNTIME = STAGE / "runtime"

#: Embeddable CPython to embed. 3.12 keeps wheels compatible with the 3.12 the
#: delivery is validated against.
PY_VERSION = "3.12.8"
EMBED_NAME = f"python-{PY_VERSION}-embed-amd64.zip"
EMBED_URL = f"https://www.python.org/ftp/python/{PY_VERSION}/{EMBED_NAME}"
CACHE_DIR = ROOT / "build" / "cache"
ARCHIVE = CACHE_DIR / EMBED_NAME

#: Pins every transitive dependency to the version the delivery was validated
#: against. ``requirements.txt`` uses ``>=``, so without this pip installs
#: whatever is newest (pandas 3.x, pydantic 2.13) and the bundled runtime ends
#: up as a stack the application has never actually run on.
CONSTRAINTS = ROOT / "constraints-runtime.txt"

MARKER_NAME = ".returnshield-runtime"


def log(message: str) -> None:
    print(f"[runtime] {message}", flush=True)


def marker_text() -> str:
    return f"ReturnShield embedded runtime\npython={PY_VERSION}\n"


def already_provisioned() -> bool:
    marker = RUNTIME / MARKER_NAME
    if not marker.exists():
        return False
    if marker.read_text(encoding="utf-8") != marker_text():
        return False
    return (RUNTIME / "python.exe").exists()


def download() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists() and zipfile.is_zipfile(ARCHIVE):
        log(f"Using cached {ARCHIVE.name} ({ARCHIVE.stat().st_size / 1e6:.1f} MB)")
        return ARCHIVE
    log(f"Downloading {EMBED_URL} ...")
    tmp = ARCHIVE.with_suffix(".part")
    with urllib.request.urlopen(EMBED_URL, timeout=120) as response:
        with open(tmp, "wb") as handle:
            shutil.copyfileobj(response, handle)
    if not zipfile.is_zipfile(tmp):
        raise RuntimeError("downloaded archive is not a valid zip")
    tmp.replace(ARCHIVE)
    log(f"Downloaded ({ARCHIVE.stat().st_size / 1e6:.1f} MB)")
    return ARCHIVE


def extract() -> None:
    log(f"Extracting into {RUNTIME} ...")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as zf:
        zf.extractall(RUNTIME)
    log("Extracted interpreter")


def enable_site_packages() -> None:
    """Rewrite the ``._pth`` so site-packages and the app source are importable.

    A ``._pth`` file switches CPython into isolated mode, and that matters:
    isolated mode ignores ``PYTHONPATH``, which is exactly how the launcher
    normally exposes the application source. So ``Lib\\site-packages`` is listed
    for the third-party packages, ``..\\app`` is listed so ``import api.main``
    still resolves no matter what the working directory is, and the bare
    ``import site`` line re-enables the site processing the embeddable build
    suppresses. All three are required; dropping any one breaks a different part.
    """
    candidates = sorted(RUNTIME.glob("python*._pth"))
    if not candidates:
        raise RuntimeError("no python*._pth found in the embedded runtime")
    pth = candidates[0]
    # Forward slashes on purpose: a backslash before a letter is an escape
    # sequence in plenty of contexts, and ``..\app`` silently becomes "..<bell>pp".
    # CPython accepts either separator on Windows.
    lines = [
        "python312.zip",
        ".",
        "Lib/site-packages",
        "../app",
        "import site",
    ]
    pth.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Wrote {pth.name} (site-packages + app source enabled)")


def install_dependencies() -> None:
    site = RUNTIME / "Lib" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    requirements = STAGE / "requirements.txt"
    if not requirements.exists():
        raise RuntimeError(f"{requirements} not found; run build_delivery.py first")
    command = [
        sys.executable, "-m", "pip", "install",
        "--target", str(site),
        "--requirement", str(requirements),
        "--only-binary", ":all:",   # never build from source: slow and fragile
        "--no-warn-script-location",
        "--no-compile",             # skip .pyc: smaller, and OneDrive-friendly
        "--upgrade",
    ]
    if CONSTRAINTS.exists():
        command += ["--constraint", str(CONSTRAINTS)]
        pins = sum(1 for line in CONSTRAINTS.read_text(encoding="utf-8").splitlines() if line.strip())
        log(f"Pinning to validated versions from {CONSTRAINTS.name} ({pins} constraints)")
    else:
        log("WARNING: constraints-runtime.txt missing; dependency versions will float.")
    log("Installing dependencies into the embedded runtime (this is the slow step)...")
    result = subprocess.run(command)
    if result.returncode != 0:
        raise RuntimeError(f"pip install failed with exit code {result.returncode}")
    count = sum(1 for _ in site.rglob("*") if _.is_file())
    size_mb = sum(_.stat().st_size for _ in site.rglob("*") if _.is_file()) / 1e6
    log(f"Dependencies installed: {count} files, {size_mb:.0f} MB")


def verify_runtime() -> None:
    """Prove the embedded interpreter can import the heavy stack on its own."""
    python = RUNTIME / "python.exe"
    probe = (
        "import streamlit, fastapi, uvicorn, pandas, numpy, sklearn, "
        "xgboost, shap, joblib, plotly, pyarrow, httpx; "
        "print('runtime imports OK')"
    )
    log("Verifying the embedded interpreter resolves every dependency...")
    result = subprocess.run(
        [str(python), "-c", probe],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "embedded runtime could not import the stack:\n"
            + (result.stdout or "") + (result.stderr or "")
        )
    log(result.stdout.strip() or "runtime imports OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="rebuild even when the marker already matches")
    args = parser.parse_args()

    if not STAGE.exists():
        log(f"ERROR: {STAGE} does not exist. Run build_delivery.py first.")
        return 1

    if already_provisioned() and not args.force:
        log("Runtime already provisioned; nothing to do (use --force to rebuild).")
        return 0

    if args.force and RUNTIME.exists():
        log(f"--force: removing {RUNTIME} for a clean rebuild")
        shutil.rmtree(RUNTIME)

    try:
        download()
        extract()
        enable_site_packages()
        install_dependencies()
        verify_runtime()
        (RUNTIME / MARKER_NAME).write_text(marker_text(), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - surface a readable build failure
        log(f"ERROR: {exc}")
        return 1

    total_mb = sum(f.stat().st_size for f in RUNTIME.rglob("*") if f.is_file()) / 1e6
    log(f"Self-contained runtime ready: {RUNTIME} ({total_mb:.0f} MB)")
    log("The launcher will now prefer this interpreter over any system Python.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
