"""Assemble the ReturnShield delivery folder.

Builds two artifacts next to the repository:

    dist/ReturnShield/ReturnShield.exe   supervisor launcher (PyInstaller)
    dist/ReturnShield-v<version>.zip     the whole folder, zipped

The delivery folder layout:

    ReturnShield/
      ReturnShield.exe          supervisor (boots API + UI, opens browser)
      Setup ReturnShield.bat    first-run setup: runtime, packages, shortcut
      runtime/                  bundled Python (only in the self-contained build)
      Start ReturnShield.bat    double-click fallback when .exe is blocked
      Verify ReturnShield.bat   end-to-end self-test against a running stack
      requirements.txt          Python dependencies for the target machine
      README.txt                delivery-specific quickstart
      app/                      source (api/, dashboard/, src/, sdk/, ...)
      models/                   trained model bundle + policy + intent encoder
      data/raw, data/processed  reference data the app reads
      reports/                  held-out predictions + final report
      DECISION_API.md, DEMO.md  product docs

Deliberately excluded: data/live (runtime event log), data/memory (session
state), logs/, __pycache__, tests, .git, .freebuff, OneDrive cruft.

Usage:
    python build_delivery.py                # folder + exe + zip
    python build_delivery.py --no-zip       # skip the zip step
    python build_delivery.py --no-exe       # skip PyInstaller (source launcher only)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
VERSION = "2.1.0"
STAGE_NAME = "ReturnShield"
STAGE = DIST / STAGE_NAME

#: Repository items copied into app/ (source the machine's Python will run).
APP_ITEMS = [
    "api", "dashboard", "src", "sdk", "run_pipeline.py", "train_intent.py",
    "eval_pipeline.py", "eval_local_model.py", "requirements.txt",
    "DECISION_API.md", "DEMO.md", "README.md",
]

#: Top-level asset folders copied as-is.
ASSET_DIRS = ["models", "reports"]
ASSET_DATA_DIRS = ["data/raw", "data/processed"]

#: Never copied (runtime state, caches, VCS).
EXCLUDE_NAMES = {
    "__pycache__", ".pytest_cache", ".freebuff", ".git", "dist", "build",
    "logs", "launcher", "processed", "raw", "data", "node_modules",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp", ".bat.tmp"}


def log(message: str) -> None:
    print(f"[build] {message}", flush=True)


def ignore_factory():
    def ignore(directory: str, names: list[str]) -> set[str]:
        skipped = set()
        for name in names:
            if name in EXCLUDE_NAMES:
                skipped.add(name)
            elif Path(name).suffix.lower() in EXCLUDE_SUFFIXES:
                skipped.add(name)
        return skipped
    return ignore


def copytree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=ignore_factory(), dirs_exist_ok=True)


def reset_stage() -> None:
    """Clear the staged content, leaving the expensive bundled runtime alone.

    Provisioning runtime/ downloads and installs hundreds of wheels, so a
    rebuild to pick up a launcher tweak must not throw that away. The runtime is
    left exactly where it is rather than parked beside the stage and moved back:
    moving a 700 MB tree in and out of the stage trips OneDrive/AV file locks
    (``WinError 5`` on the rename back) and buys nothing. Only build_runtime.py
    ever writes runtime/, so clearing everything else is sufficient.
    """
    STAGE.mkdir(parents=True, exist_ok=True)
    for entry in sorted(STAGE.iterdir()):
        if entry.name == "runtime":
            log("Keeping the bundled runtime in place (build_runtime.py owns it)")
            continue
        log(f"  clearing {entry.name}")
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
    for sub in ("app", "logs"):
        (STAGE / sub).mkdir(exist_ok=True)


def stage_application() -> None:
    log("Staging application source...")
    for item in APP_ITEMS:
        src = ROOT / item
        if not src.exists():
            log(f"  WARNING: {item} does not exist; skipping.")
            continue
        dst = STAGE / "app" / item
        if src.is_dir():
            copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    for rel in ASSET_DIRS:
        src = ROOT / rel
        if src.exists():
            log(f"Staging {rel}/ ...")
            copytree(src, STAGE / rel)
    for rel in ASSET_DATA_DIRS:
        src = ROOT / rel
        if src.exists():
            log(f"Staging {rel}/ ...")
            copytree(src, STAGE / rel.replace("/", os.sep))
    # Top-level requirements.txt: what the target machine must pip install.
    shutil.copy2(ROOT / "requirements.txt", STAGE / "requirements.txt")
    # Empty runtime state dirs the app expects to exist.
    (STAGE / "data" / "live").mkdir(parents=True, exist_ok=True)
    (STAGE / "data" / "memory").mkdir(parents=True, exist_ok=True)


def write_readme() -> None:
    text = f"""ReturnShield AI {VERSION} - delivery folder
================================================

WHAT THIS IS
  A self-contained deployment of ReturnShield AI: the FastAPI Decision API,
  the Streamlit operations dashboard, the trained model bundle, and the
  locally-trained intent encoder. No cloud services are required.

QUICKSTART (Windows)
  1. Double-click  Setup ReturnShield.bat     (first run only)
     It picks the bundled runtime (or a system Python), verifies the packages
     and model assets, reports the ports, and puts a shortcut on your Desktop.
     It is safe to run again at any time.
  2. Double-click  ReturnShield.exe  (or the Desktop shortcut)
     (or  Start ReturnShield.bat  if SmartScreen blocked the exe)
  3. Wait for "READY" in the console window. First start is the slowest -
     models load once, typically 1-3 minutes on a laptop.
  4. The dashboard opens in your browser automatically.
       Dashboard:  http://127.0.0.1:8501
       API:        http://127.0.0.1:8000
       API docs:   http://127.0.0.1:8000/docs

REQUIREMENTS
  - Windows 10/11, 64-bit.
  - Python is NOT required when this folder contains  runtime/  The bundled
    interpreter is used automatically and nothing is installed on the machine.
  - Without  runtime/  the app runs on a system Python 3.11/3.12 plus the
    packages listed in requirements.txt:
      ReturnShield.exe --install   installs them for you (one time)

CUSTOM PORTS / AUTOMATION
    set RS_API_PORT=8010 & set RS_UI_PORT=8510 & ReturnShield.exe --no-browser

SELF-TEST
  Double-click  Verify ReturnShield.bat  while the app is running.
  It checks API identity, the returns feed, and a real score call.

STOPPING
  Close the console window, press Ctrl+C inside it, or run
      ReturnShield.exe --stop
  Stop ReturnShield.bat does the same from Explorer. It is the fallback for the
  one case the launcher cannot cover: the launcher itself killed outright, so
  nothing is left alive to shut its children down.

TROUBLESHOOTING
  - Ports: if 8000/8501 are already taken the launcher steps to the next free
    port and prints the one it chose - it will not fail to start over a stray
    listener. Force a specific pair with RS_API_PORT / RS_UI_PORT as above.
  - Missing packages: run  ReturnShield.exe --install
  - API log: logs/api.log   Dashboard log: logs/streamlit.log
"""
    (STAGE / "README.txt").write_text(text, encoding="utf-8")
    log("Wrote README.txt")


def build_exe() -> bool:
    log("Building ReturnShield.exe with PyInstaller...")
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm",
         "--distpath", str(STAGE), "--workpath", str(ROOT / "build" / "launcher"),
         str(ROOT / "launcher" / "returnshield.spec")],
        cwd=str(ROOT / "launcher"),
    )
    if result.returncode != 0:
        log("ERROR: PyInstaller failed; delivery will include the .py launcher only.")
        return False
    exe = STAGE / "ReturnShield.exe"
    log(f"Built {exe} ({exe.stat().st_size / 1e6:.1f} MB)")
    return True


START_BAT = r"""@echo off
REM Fallback launcher: same supervisor, run from source.
REM Use when SmartScreen or policy blocks the .exe.
setlocal
cd /d "%~dp0"
if exist "runtime\python.exe" (
  "runtime\python.exe" app\app_launcher.py %*
  pause
  exit /b 0
)
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Run "Setup ReturnShield.bat" first.
  pause & exit /b 1
)
REM The launcher script is staged beside the app source, in app\ - not here.
python app\app_launcher.py %*
pause
"""

STOP_BAT = r"""@echo off
REM Stop every ReturnShield component on this machine.
REM The launcher normally cleans up on exit; this is the manual backstop.
setlocal EnableDelayedExpansion
set "API_PORT=%RS_API_PORT%"
if "%API_PORT%"=="" set "API_PORT=8000"
set "UI_PORT=%RS_UI_PORT%"
if "%UI_PORT%"=="" set "UI_PORT=8501"
echo Stopping ReturnShield on ports %API_PORT% (API) and %UI_PORT% (dashboard)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "foreach($p in @(%API_PORT%,%UI_PORT%)){$c=Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue; foreach($x in $c){try{Stop-Process -Id $x.OwningProcess -Force -ErrorAction SilentlyContinue; Write-Host ('  stopped PID '+$x.OwningProcess+' on port '+$p)}catch{}}}" 
echo Done.
pause
"""

SETUP_BAT = r"""@echo off
REM First-run setup: make sure this folder can start on this machine.
REM Safe to run more than once - every step is skipped when already satisfied.
setlocal EnableDelayedExpansion
cd /d "%~dp0"
echo ============================================================
echo  ReturnShield AI setup
echo ============================================================

set "RS_PY="
echo.
echo [1/5] Looking for a bundled runtime...
if exist "runtime\python.exe" (
  echo       OK - self-contained runtime found. No Python install needed.
  set "RS_PY=%~dp0runtime\python.exe"
  goto :assets
)
echo       This copy has no bundled runtime.

echo.
echo [2/5] Looking for a system Python...
where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo       ERROR: no Python on PATH and no bundled runtime in this folder.
  echo       Fix it either way:
  echo         * install Python 3.11/3.12 from https://www.python.org/downloads/
  echo           ^(tick "Add python.exe to PATH"^), then re-run this setup, or
  echo         * ask for the fully self-contained build of ReturnShield.
  echo.
  pause ^& exit /b 1
)
set "RS_PY=python"
for /f "delims=" %%v in ('python --version 2^>^&1') do echo       OK - %%v

echo.
echo [3/5] Checking required packages...
"%RS_PY%" -c "import streamlit, fastapi, uvicorn, pandas, numpy, sklearn, xgboost, shap, joblib, plotly, pyarrow, httpx" 2>nul
if errorlevel 1 (
  echo       Missing packages - installing from requirements.txt.
  echo       This is a one-time download and can take several minutes...
  "%RS_PY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo       ERROR: package installation failed. See the messages above.
    pause ^& exit /b 1
  )
) else (
  echo       OK - all required packages are present.
)

:assets
echo.
echo [4/5] Checking model assets...
set "MISSING="
if not exist "models\model_bundle.joblib" set "MISSING=!MISSING! models\model_bundle.joblib"
if not exist "models\policy.json" set "MISSING=!MISSING! models\policy.json"
if not "!MISSING!"=="" (
  echo       WARNING: missing!MISSING!
  echo       The app rebuilds these on first start, which takes several minutes.
) else (
  echo       OK - model bundle and policy are present.
)

echo.
echo [5/5] Checking ports...
set "API_PORT=%RS_API_PORT%"
if "%API_PORT%"=="" set "API_PORT=8000"
set "UI_PORT=%RS_UI_PORT%"
if "%UI_PORT%"=="" set "UI_PORT=8501"
powershell -NoProfile -ExecutionPolicy Bypass -Command "foreach($p in @(%API_PORT%,%UI_PORT%)){$c=Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue; if($c){Write-Host ('      port '+$p+' is IN USE (pid '+(($c.OwningProcess|Select-Object -First 1)) + ') - ReturnShield will pick the next free port')}else{Write-Host ('      port '+$p+' is free')}}"

echo.
echo Creating a desktop shortcut...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ws=New-Object -ComObject WScript.Shell; $lnk=$ws.CreateShortcut([Environment]::GetFolderPath('Desktop')+'\ReturnShield.lnk'); $lnk.TargetPath='%~dp0ReturnShield.exe'; $lnk.WorkingDirectory='%~dp0'; $lnk.Description='ReturnShield AI - fraud decision dashboard'; $lnk.Save(); Write-Host '      shortcut created on your Desktop'" 2>nul

echo.
echo ============================================================
echo  SETUP COMPLETE
if not "%RS_PY%"=="%~dp0runtime\python.exe" echo  Note: this copy needs the system Python found above.
echo  Start the app with  ReturnShield.exe  or the Desktop shortcut.
echo ============================================================
pause
exit /b 0
"""

VERIFY_BAT = r"""@echo off
REM End-to-end self-test against a running ReturnShield stack.
setlocal EnableDelayedExpansion
REM Same interpreter rule as Start: the bundled runtime first, so the self-test
REM still works on a machine that has no Python installed at all.
set "RS_PY="
if exist "runtime\python.exe" set "RS_PY=%~dp0runtime\python.exe"
if "%RS_PY%"=="" (
  where python >nul 2>nul
  if errorlevel 1 (
    echo ERROR: neither the bundled runtime nor a system Python was found.
    echo        Run "Setup ReturnShield.bat" first.
    pause ^& exit /b 1
  )
  set "RS_PY=python"
)
set "API=http://127.0.0.1:%RS_API_PORT%"
if "%RS_API_PORT%"=="" set "API=http://127.0.0.1:8000"
set "UI=http://127.0.0.1:%RS_UI_PORT%"
if "%RS_UI_PORT%"=="" set "UI=http://127.0.0.1:8501"
echo ============================================================
echo  ReturnShield self-test against %API%
echo ============================================================
"%RS_PY%" -c "import urllib.request,json; m=json.load(urllib.request.urlopen('%API%/api/v1/meta',timeout=10)); assert m.get('service')=='ReturnShield AI'; print('[OK] API identity:',m['service'],m['version'])" || goto :fail
REM NB: no %-formatting in the embedded Python below. cmd.exe expands %s / %VAR%,
REM which silently corrupts the one-liner (it used to fail with a SyntaxError).
"%RS_PY%" -c "import urllib.request,json; s=json.load(urllib.request.urlopen('%API%/api/v1/returns/stats',timeout=10)); print('[OK] returns feed: running=',s.get('running'),'buffer=',s.get('buffered_records'),'seq=',s.get('event_sequence'))" || goto :fail
"%RS_PY%" -c "import urllib.request,json,hmac,hashlib; body=json.dumps({'return_id':'R-VERIFY-1','order_id':'O-VERIFY','customer_id':'C-VERIFY','order_value':1000.0,'product_price':1000.0,'return_reason':'damaged'}).encode(); req=urllib.request.Request('%API%/api/v1/score',data=body,headers={'Content-Type':'application/json'}); r=json.load(urllib.request.urlopen(req,timeout=60)); assert 0.0<=r['risk_probability']<=1.0; print('[OK] score call:',r['return_id'],'risk=',round(r['risk_probability']*100,1),'decision=',r['decision'])" || goto :fail
"%RS_PY%" -c "import urllib.request; urllib.request.urlopen('%UI%',timeout=15); print('[OK] dashboard responds')" || goto :fail
echo ============================================================
echo  ALL CHECKS PASSED
echo ============================================================
pause & exit /b 0
:fail
echo ============================================================
echo  SELF-TEST FAILED - see the message above / logs\api.log
echo ============================================================
pause & exit /b 1
"""


def write_bats() -> None:
    # The .bat fallback runs the launcher *source* from app/ via the launcher copy.
    (STAGE / "Setup ReturnShield.bat").write_text(SETUP_BAT, encoding="utf-8")
    (STAGE / "Start ReturnShield.bat").write_text(START_BAT, encoding="utf-8")
    (STAGE / "Stop ReturnShield.bat").write_text(STOP_BAT, encoding="utf-8")
    (STAGE / "Verify ReturnShield.bat").write_text(VERIFY_BAT, encoding="utf-8")
    if not (STAGE / "ReturnShield.exe").exists():
        # No exe: the bat needs the launcher script beside the app.
        shutil.copy2(ROOT / "launcher" / "returnshield_launcher.py",
                     STAGE / "app" / "app_launcher.py")
    else:
        # Keep a source copy too, so the bat fallback always works.
        shutil.copy2(ROOT / "launcher" / "returnshield_launcher.py",
                     STAGE / "app" / "app_launcher.py")
    log("Wrote Setup / Start / Stop / Verify ReturnShield.bat")


def verify_stage() -> bool:
    """Fail the build rather than ship an incomplete folder."""
    required = [
        "app/api/main.py", "app/dashboard/app.py", "app/run_pipeline.py",
        "models/model_bundle.joblib", "models/policy.json",
        "models/local/all-MiniLM-L6-v2/model.onnx",
        "data/raw", "data/processed", "reports/test_predictions.csv",
        "requirements.txt", "README.txt", "Setup ReturnShield.bat",
    ]
    problems = [r for r in required if not (STAGE / r.replace("/", os.sep)).exists()]
    if problems:
        log("ERROR: staged folder is incomplete. Missing:")
        for item in problems:
            log(f"  - {item}")
        return False
    size_mb = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file()) / 1e6
    log(f"Staged folder complete: {STAGE} ({size_mb:.0f} MB)")
    if (STAGE / "runtime" / "python.exe").exists():
        log("Self-contained: bundled runtime present (no Python needed on target)")
    else:
        log("Not self-contained: no runtime/ - target must run Setup ReturnShield.bat")
    return True


def make_zip(include_runtime: bool = False) -> Path | None:
    """Zip the delivery folder.

    The bundled runtime is hundreds of megabytes of third-party wheels that
    barely compress. Excluding it keeps the archive shippable and the recipient
    runs ``Setup ReturnShield.bat`` to provision a runtime themselves. Pass
    ``--zip-runtime`` to emit the fully self-contained archive instead, and
    expect it to be very large and slow to produce.
    """
    zip_path = DIST / f"ReturnShield-v{VERSION}.zip"
    if zip_path.exists():
        zip_path.unlink()
    skipped = 0
    log(f"Zipping -> {zip_path.name} (this takes a minute)...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(STAGE.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(STAGE)
            if not include_runtime and rel.parts and rel.parts[0] == "runtime":
                skipped += 1
                continue
            zf.write(path, Path(STAGE_NAME) / rel)
    if skipped:
        log(f"  excluded {skipped} bundled-runtime files (use --zip-runtime to include)")
    log(f"Zip written: {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB)")
    return zip_path


def main() -> int:
    reset_stage()
    stage_application()
    exe_ok = "--no-exe" not in sys.argv
    if exe_ok and not build_exe():
        exe_ok = False
    write_readme()
    write_bats()
    if not verify_stage():
        return 1
    if "--no-zip" not in sys.argv:
        make_zip(include_runtime="--zip-runtime" in sys.argv)
    log("=" * 56)
    log(f"DONE. Deliverable folder: {STAGE}")
    log(f"  exe built: {exe_ok}")
    log("=" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
