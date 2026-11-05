#!/bin/bash
set -e

echo "===================================================="
echo "  Smart Farming AI Agent — IBM Granite / watsonx.ai"
echo "===================================================="

# Check .env
if [ ! -f ".env" ]; then
    echo ""
    echo "[SETUP] .env not found. Copying env.example to .env ..."
    cp env.example .env
    echo "[SETUP] Please edit .env and add your IBM Cloud API keys, then run again."
    exit 1
fi

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 not found. Install Python 3.9+ first."
    exit 1
fi

# Create venv if missing
if [ ! -d "venv" ]; then
    echo "[SETUP] Creating virtual environment..."
    python3 -m venv venv
    echo "[SETUP] Installing dependencies (this may take a few minutes)..."
    source venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
else
    source venv/bin/activate
fi

# Ingest knowledge base
echo ""
echo "[INFO] Ingesting knowledge base..."
python ingest.py

# Start server
echo ""
echo "[INFO] Starting Smart Farming AI Agent on http://localhost:8000"
echo "[INFO] Press Ctrl+C to stop."
echo ""
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
