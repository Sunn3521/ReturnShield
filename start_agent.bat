@echo off
TITLE ReturnShield AI Agent Launcher
echo ============================================================
echo           RETURN SHIELD AI - RETURN ABUSE RISK AGENT
echo ============================================================
echo [1/3] Verifying Python environment...

cd /d "%~dp0.."

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)

echo [2/3] Executing ReturnShield Pipeline ^& Model Training...
python run_pipeline.py
if %errorlevel% neq 0 (
    echo [WARNING] Pipeline run encountered an issue, proceeding with existing assets...
)

echo [3/3] Launching FastAPI REST API ^& Streamlit Dashboard...
start "ReturnShield REST API" cmd /k "python -m uvicorn src.api:app --host 0.0.0.0 --port 8000"
timeout /t 2 >nul
start "ReturnShield Operations Dashboard" cmd /k "python -m streamlit run app.py --server.port 8501"

echo.
echo ============================================================
echo [SUCCESS] ReturnShield AI Agent services are running!
echo   - Streamlit Dashboard: http://localhost:8501
echo   - FastAPI REST Server: http://localhost:8000
echo   - API Documentation:   http://localhost:8000/docs
echo ============================================================
echo Close the popup windows to stop services.
pause
