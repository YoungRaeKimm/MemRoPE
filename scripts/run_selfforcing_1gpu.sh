#!/bin/bash
# Run MemRoPE inference with Self-Forcing base model (1 GPU, 40GB+ VRAM)
# Usage: bash scripts/run_selfforcing_1gpu.sh [config] [start_idx] [end_idx]

CONFIG=${1:-configs/selfforcing/memrope_60s.yaml}
START=${2:-0}
END=${3:-1}

echo "Config: $CONFIG"
echo "Prompts: $START to $END"

python inference.py \
    --config_path "$CONFIG" \
    --start_idx "$START" \
    --end_idx "$END"
