#!/bin/bash
# Vast.ai setup script for Inflect Phase 3 training
# Run this once after renting an instance:
#   bash vastai_setup.sh

set -e

echo "=== Installing dependencies ==="
pip install -q kokoro>=0.9.2 soundfile torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

echo "=== Verifying GPU ==="
python3 -c "import torch; print('GPU:', torch.cuda.get_device_name(0)); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"

echo "=== Done. Now upload finetune_dataset.pt and run: ==="
echo "  python train_phase3_paralinguistic.py --fp16 --epochs 80 --save-every 5"
