#!/usr/bin/env bash
# KrishiAlert v2 — Quick Start
# ─────────────────────────────
echo "🌾 KrishiAlert v2 Setup"
echo "========================"

# 1. Install dependencies
echo "[1/3] Installing Python packages..."
pip install flask flask-cors numpy pandas scikit-learn xgboost requests shap gtts --quiet

# 2. (Optional) set your API keys
# export AGMARKNET_API_KEY="your_key_here"
# export TWILIO_SID="your_sid"
# export TWILIO_TOKEN="your_token"
# export TWILIO_FROM="+1XXXXXXXXXX"

# 3. Run the server
echo "[2/3] Starting KrishiAlert server..."
echo "       → Dashboard: http://localhost:5000"
echo "       → Health:    http://localhost:5000/health"
echo "       → API:       POST http://localhost:5000/forecast"
echo ""
echo "[3/3] Training models on startup (takes ~30 seconds)..."
echo "========================"
python krishialert_v2.py
