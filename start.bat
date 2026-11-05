@echo off
echo ====================================================
echo   Smart Farming AI Agent — IBM Granite / watsonx.ai
echo ====================================================

REM Check if .env exists
IF NOT EXIST ".env" (
    echo.
    echo [SETUP] .env file not found.
    echo [SETUP] Copying env.example to .env ...
    copy env.example .env >nul
    echo [SETUP] Done. Please edit .env and add your IBM Cloud API keys.
    echo [SETUP] Then run this script again.
    echo.
    pause
    exit /b 1
)

REM Check Python
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM Install dependencies if venv missing
IF NOT EXIST "venv\" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
    echo [SETUP] Installing dependencies (this may take a few minutes)...
    call venv\Scripts\activate.bat
    pip install --upgrade pip >nul
    pip install -r requirements.txt
) ELSE (
    call venv\Scripts\activate.bat
)

REM Ingest knowledge base
echo.
echo [INFO] Ingesting knowledge base...
python ingest.py

REM Start server
echo.
echo [INFO] Starting Smart Farming AI Agent on http://localhost:8000
echo [INFO] Press Ctrl+C to stop.
echo.
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
