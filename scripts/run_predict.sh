#!/bin/bash
# Run all three scanner modes and save markdown reports.
# Called by cron at 8am and 8pm daily.

set -euo pipefail

REPO="/Users/joachimchuah/Desktop/saas-ideas/r-c-model/regression-classification-model"
PYTHON="$REPO/.venv/bin/python"
LOG="$REPO/logs/predict.log"

cd "$REPO"

# Load env vars (FRED_API_KEY etc.)
source ~/.zshrc 2>/dev/null || true

echo "========================================" >> "$LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S')  predict run" >> "$LOG"
echo "========================================" >> "$LOG"

for MODE in leaps swing daily; do
    echo "--- $MODE ---" >> "$LOG"
    "$PYTHON" predict.py --mode "$MODE" --report >> "$LOG" 2>&1 || \
        echo "ERROR: $MODE scan failed" >> "$LOG"
done

echo "" >> "$LOG"
