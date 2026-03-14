#!/bin/bash
# Run MemRoPE inference with Self-Forcing base model (2 GPUs, 24GB+ each)
# Usage: bash scripts/run_selfforcing_2gpu.sh [config] [start_idx] [end_idx]

CONFIG=${1:-configs/selfforcing/memrope_60s.yaml}
START=${2:-0}
END=${3:-1}

echo "Config: $CONFIG"
echo "Prompts: $START to $END"

python inference_2gpu.py \
    --config_path "$CONFIG" \
    --start_idx "$START" \
    --end_idx "$END"
