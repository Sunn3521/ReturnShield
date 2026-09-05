@echo off
setlocal EnableExtensions EnableDelayedExpansion
TITLE ReturnShield AI - Full Stack
cd /d "%~dp0"
set "PYTHONPATH=%cd%"
set "API_PORT=8000"
set "APP_PORT=8501"

echo ============================================================
echo              RETURNSHIELD AI - STARTUP
echo ============================================================
echo Project: %cd%
echo.

echo [1/4] Closing stale ReturnShield services on ports %API_PORT% and %APP_PORT%...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ports=@(%API_PORT%,%APP_PORT%); foreach($p in $ports){$c=Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue; foreach($x in $c){try{Stop-Process -Id $x.OwningProcess -Force -ErrorAction SilentlyContinue}catch{}}}" >nul 2>&1

echo [2/4] Checking model assets...
if not exist "%cd%\models\model_bundle.joblib" (
  echo Model bundle missing. Running pipeline...
  python run_pipeline.py
  if errorlevel 1 (
    echo [ERROR] Pipeline failed.
    pause
    exit /b 1
  )
)

echo [3/4] Starting FastAPI from THIS project folder...
start "ReturnShield REST API" /d "%cd%" cmd /k "set PYTHONPATH=%cd%&& python -m uvicorn src.api:app --host 127.0.0.1 --port %API_PORT%"
timeout /t 3 /nobreak >nul

python -c "import urllib.request,sys,json; u='http://127.0.0.1:%API_PORT%/api/v1/meta'; d=json.load(urllib.request.urlopen(u,timeout=5)); assert d.get('live_returns_endpoint')=='/api/v1/returns'; print('[OK] Live API route verified:',d['live_returns_endpoint'])"
if errorlevel 1 (
  echo [ERROR] Live API self-check failed. The running API is not the expected ReturnShield build.
  echo Open http://127.0.0.1:%API_PORT%/docs to inspect it.
  pause
  exit /b 1
)

echo [4/4] Starting Streamlit dashboard...
start "ReturnShield Operations Dashboard" /d "%cd%" cmd /k "set PYTHONPATH=%cd%&& python -m streamlit run app.py --server.address 127.0.0.1 --server.port %APP_PORT% --server.maxUploadSize 1000"

echo.
echo ============================================================
echo [READY]
echo Dashboard: http://127.0.0.1:%APP_PORT%
echo API:       http://127.0.0.1:%API_PORT%
echo API Docs:  http://127.0.0.1:%API_PORT%/docs
echo Live API: http://127.0.0.1:%API_PORT%/api/v1/returns
echo ============================================================
endlocal
