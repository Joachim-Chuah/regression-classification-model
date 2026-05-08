#!/bin/bash
# Retrain all three models (leaps, swing, daily).
# Called by cron at 8pm every Sunday.

set -euo pipefail

REPO="/Users/joachimchuah/Desktop/saas-ideas/r-c-model/regression-classification-model"
PYTHON="$REPO/.venv/bin/python"
LOG="$REPO/logs/retrain.log"

cd "$REPO"

source ~/.zshrc 2>/dev/null || true

echo "========================================" >> "$LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S')  retrain run" >> "$LOG"
echo "========================================" >> "$LOG"

for MODEL in leaps swing daily; do
    echo "--- tuning $MODEL ---" >> "$LOG"
    "$PYTHON" -m src.tune --model "$MODEL" --trials 50 >> "$LOG" 2>&1 || \
        echo "ERROR: $MODEL tuning failed" >> "$LOG"
done

echo "" >> "$LOG"
