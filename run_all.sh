#!/usr/bin/env bash
# Rebuild every result from the raw files (about 3 minutes)
set -e
PY=$(command -v python3 || command -v python)
$PY src/01_prepare_market_data.py
$PY src/02_market_forecast.py
$PY src/03_prepare_orders.py
$PY src/04_order_analytics.py
$PY src/05_final_forecasts.py
echo "Done. Start the dashboard with: streamlit run app.py"
