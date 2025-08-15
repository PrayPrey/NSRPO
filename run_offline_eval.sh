#!/bin/bash
# Offline evaluation script for NSPO

echo "=== NSPO Offline Evaluation ==="
echo "This script runs evaluation without downloading models"
echo ""

# Use dummy model for offline testing
python evaluate.py \
    --model_path "dummy" \
    --fast \
    --cpu_only \
    --limit_eval_samples 50 \
    --max_batches 5 \
    --batch_size 2 \
    --output_path "offline_eval_results.json"

echo ""
echo "Evaluation complete. Results saved to offline_eval_results.json"