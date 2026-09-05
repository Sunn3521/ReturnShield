@echo off
setlocal EnableExtensions EnableDelayedExpansion
TITLE ReturnShield AI - Full Stack
cd /d "%~dp0"
set "PYTHONPATH=%cd%"
set "API_PORT=8000"
set "APP_PORT=8501"
set "API_LOG=%cd%\api_startup.log"
set "APP_LOG=%cd%\streamlit_startup.log"

if not exist "%cd%\logs" mkdir "%cd%\logs" >nul 2>&1
set "API_LOG=%cd%\logs\api.log"
set "APP_LOG=%cd%\logs\streamlit.log"

color F0
cls
echo ============================================================
echo              RETURNSHIELD AI - STARTUP
echo ============================================================
echo Project: %cd%
echo.

echo [1/5] Closing stale ReturnShield services on ports %API_PORT% and %APP_PORT%...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ports=@(%API_PORT%,%APP_PORT%); foreach($p in $ports){$c=Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue; foreach($x in $c){try{Stop-Process -Id $x.OwningProcess -Force -ErrorAction SilentlyContinue}catch{}}}" >nul 2>&1

timeout /t 1 /nobreak >nul

echo [2/5] Checking Python and ReturnShield API import...
python -c "import sys; print('Python:',sys.executable); import src.api; print('API_IMPORT_OK'); assert any(getattr(r,'path','')=='/api/v1/returns' for r in src.api.app.routes); print('LIVE_ROUTE_OK')" 
if errorlevel 1 (
  echo.
  echo [ERROR] ReturnShield API could not be imported from this project folder.
  echo Check that Python and the required packages are installed.
  pause
  exit /b 1
)

echo [3/5] Checking model assets...
if not exist "%cd%\models\model_bundle.joblib" (
  echo Model bundle missing. Running pipeline...
  python run_pipeline.py
  if errorlevel 1 (
    echo [ERROR] Pipeline failed. See output above.
    pause
    exit /b 1
  )
)
if not exist "%cd%\models\policy.json" (
  echo [ERROR] models\policy.json is missing.
  pause
  exit /b 1
)

echo [4/5] Starting FastAPI from THIS project folder...
if exist "%API_LOG%" del /q "%API_LOG%" >nul 2>&1
start "ReturnShield REST API" /d "%cd%" cmd /c "color F0&& set PYTHONPATH=%cd%&& cd /d %cd%&& python -m uvicorn src.api:app --host 127.0.0.1 --port %API_PORT% --log-level info > "%API_LOG%" 2>&1"

echo Waiting for FastAPI to become ready...
set "API_READY=0"
for /L %%N in (1,1,60) do (
  if "!API_READY!"=="0" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:%API_PORT%/api/v1/meta' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
    if not errorlevel 1 set "API_READY=1"
    if "!API_READY!"=="0" timeout /t 1 /nobreak >nul
  )
)

if "!API_READY!"=="0" (
  echo.
  echo [ERROR] FastAPI did not become ready within 60 seconds.
  echo.
  echo ---------------- API LOG ----------------
  if exist "%API_LOG%" (type "%API_LOG%") else (echo API log was not created.)
  echo -------------- END API LOG --------------
  echo.
  echo Try manually:
  echo   python -m uvicorn src.api:app --host 127.0.0.1 --port %API_PORT%
  echo.
  pause
  exit /b 1
)

python -c "import urllib.request,sys,json; u='http://127.0.0.1:%API_PORT%/api/v1/meta'; d=json.load(urllib.request.urlopen(u,timeout=5)); assert d.get('live_returns_endpoint')=='/api/v1/returns'; print('[OK] ReturnShield API verified:',d.get('version'),d['live_returns_endpoint'])"
if errorlevel 1 (
  echo [ERROR] FastAPI is responding, but the expected ReturnShield API metadata was not found.
  echo Check http://127.0.0.1:%API_PORT%/docs
  pause
  exit /b 1
)

echo [5/5] Starting Streamlit dashboard...
if exist "%APP_LOG%" del /q "%APP_LOG%" >nul 2>&1
start "ReturnShield Operations Dashboard" /d "%cd%" cmd /c "color F0&& set PYTHONPATH=%cd%&& cd /d %cd%&& python -m streamlit run app.py --server.address 127.0.0.1 --server.port %APP_PORT% --server.maxUploadSize 1000 > "%APP_LOG%" 2>&1"

echo.
echo ============================================================
echo [READY]
echo Dashboard: http://127.0.0.1:%APP_PORT%
echo API:       http://127.0.0.1:%API_PORT%
echo API Docs:  http://127.0.0.1:%API_PORT%/docs
echo Live API:  http://127.0.0.1:%API_PORT%/api/v1/returns
echo API log:   %API_LOG%
echo UI log:    %APP_LOG%
echo ============================================================
endlocal
