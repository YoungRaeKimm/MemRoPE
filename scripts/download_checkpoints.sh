#!/bin/bash
# Download model checkpoints for MemRoPE inference.
# Requires: huggingface_hub (installed via requirements.txt)
#
# Models:
#   1. Wan2.1-T2V-1.3B (base model)
#   2. LongLive base + LoRA checkpoint

set -e

CKPT_DIR="checkpoints"
mkdir -p "$CKPT_DIR"

echo "=== Downloading Wan2.1-T2V-1.3B base model ==="
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', local_dir='$CKPT_DIR/Wan2.1-T2V-1.3B')
"

echo ""
echo "=== Downloading LongLive checkpoints ==="
# LongLive base model + LoRA adapter
python -c "
from huggingface_hub import hf_hub_download
for filename in ['models/longlive_base.pt', 'models/lora.pt']:
    hf_hub_download(
        repo_id='Efficient-Large-Model/LongLive-1.3B',
        filename=filename,
        local_dir='$CKPT_DIR',
        local_dir_use_symlinks=False
    )
"
mv "$CKPT_DIR/models/longlive_base.pt" "$CKPT_DIR/"
mv "$CKPT_DIR/models/lora.pt" "$CKPT_DIR/"
rmdir "$CKPT_DIR/models" 2>/dev/null || true

echo ""
echo "=== All checkpoints downloaded to $CKPT_DIR/ ==="
echo ""
