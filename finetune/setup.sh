#!/usr/bin/env bash
# Nexalith Foreman — cloud instance setup script.
# Run this ONCE after spinning up the V100 instance, before train.py.
# Installs all dependencies and validates CUDA/GPU are visible.
# Estimated runtime: 5-8 minutes.

set -euo pipefail

echo "=== Nexalith Foreman Fine-tune Setup ==="
echo ""

# 1. GPU check — fail fast if no CUDA visible
echo "[1/5] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# 2. Python version check
echo "[2/5] Python version:"
python3 --version
echo ""

# 3. Install Unsloth + training dependencies
echo "[3/5] Installing dependencies..."
pip install --upgrade pip -q
pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git" -q
pip install trl datasets transformers accelerate bitsandbytes -q
echo "Dependencies installed."
echo ""

# 4. Validate torch + CUDA
echo "[4/5] Validating torch + CUDA:"
python3 -c "
import torch
print(f'  torch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    # V100 check
    cap = torch.cuda.get_device_capability(0)
    print(f'  Compute capability: {cap[0]}.{cap[1]}')
    if cap[0] >= 8:
        print('  Ada Lovelace / Ampere detected — bfloat16 supported.')
    else:
        print('  Older GPU detected — bfloat16 may not be supported, check train.py DTYPE.')
"
echo ""

# 5. Check dataset file present
echo "[5/5] Checking dataset..."
if [ -f "finetune_examples.jsonl" ]; then
    COUNT=$(wc -l < finetune_examples.jsonl)
    echo "  finetune_examples.jsonl found: $COUNT examples"
else
    echo "  ERROR: finetune_examples.jsonl not found."
    echo "  Upload it to this directory before running train.py."
    exit 1
fi

echo ""
echo "=== Setup complete. Run: python3 train.py ==="
