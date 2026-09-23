#!/usr/bin/env bash
# ==============================================================================
# Amazon ML Challenge 2025: Milestone 8 Turnkey GPU Pipeline
#
# Multimodal Tower Architecture + High-Capacity GBDT Calibration Stack:
# 1. Alibaba-NLP/gte-Qwen2-1.5B-instruct (1536-d) or Qwen2.5-3B frozen text embeddings
# 2. SigLIP text embeddings (768-d)
# 3. Dual Vision embeddings: SigLIP (768-d) + DINOv2 (768-d) -> 1536-d
# 4. Modality-Specific Tower Adapter (Dedicated 4-Tower MLP + Deep Fusion Regressor)
# 5. LightGBM (Custom Analytical SMAPE Objective + 30k TF-IDF + Empirical Bayes Target Encoding)
# 6. CatBoost GPU (MAE Loss + 48-dim Vision SVD + Target Encoding)
# 7. Nelder-Mead Multi-Model Convex Blending
# 8. Continuous Log-Affine Power-Law Decile Calibration (a * x^b)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Activate Python Environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "==========================================================================="
echo "  Amazon ML Challenge 2025: Milestone 8 (Multimodal Tower + GBDT Stack)"
echo "==========================================================================="

# 2. Check and ensure all dependencies
echo "=== Checking and Ensuring Environment Dependencies ==="
python -c "
import sys, subprocess

deps = [
    ('torch', 'torch'),
    ('transformers', 'transformers'),
    ('lightgbm', 'lightgbm'),
    ('catboost', 'catboost'),
    ('scipy', 'scipy'),
    ('sklearn', 'scikit-learn'),
    ('num2words', 'num2words'),
    ('sentencepiece', 'sentencepiece'),
    ('google.protobuf', 'protobuf'),
    ('faiss', 'faiss-cpu'),
]

for import_name, pip_name in deps:
    try:
        __import__(import_name)
        print(f'  [OK] {import_name}')
    except ImportError:
        print(f'  [INSTALLING] {pip_name} (needed by {import_name})...')
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', pip_name, '--no-warn-script-location'], check=True)
            print(f'  [INSTALLED] {pip_name}')
        except Exception as e:
            print(f'  [WARNING] Failed to auto-install {pip_name}: {e}')
"

# 3. Check / Extract Foundation Embeddings
SUBSET_ARG=""
SUBSET_EXTRACT_ARG=""
if [ -n "$1" ]; then
    SUBSET_ARG="--subset $1 --skip_visual_metadata"
    SUBSET_EXTRACT_ARG="--subset $1"
    echo ""
    echo ">>> Fast Benchmark Mode Active: Slicing to $1 samples <<<"
fi

TEXT_MODEL="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
TEXT_TAG="gte_qwen2_1_5b_instruct"

EXPECTED_ROWS=${1:-75000}

check_cache_rows() {
    local file="$1"
    local min_rows="$2"
    python -c "
import os, numpy as np, sys
path = '$file'
if not os.path.exists(path):
    sys.exit(1)
try:
    arr = np.load(path, mmap_mode='r')
    if len(arr) >= int('$min_rows'):
        sys.exit(0)
    sys.exit(1)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

# Check if Qwen2.5-3B exists instead
if check_cache_rows "data/embeddings/train_text_qwen2_5_3b.npy" "$EXPECTED_ROWS"; then
    TEXT_MODEL="Qwen/Qwen2.5-3B"
    TEXT_TAG="qwen2_5_3b"
    echo "Found existing Qwen2.5-3B text embeddings tag: $TEXT_TAG ($EXPECTED_ROWS+ rows)"
elif check_cache_rows "data/embeddings/train_text_${TEXT_TAG}.npy" "$EXPECTED_ROWS"; then
    echo "Found existing GTE-Qwen text embeddings tag: $TEXT_TAG ($EXPECTED_ROWS+ rows)"
else
    echo ""
    echo "=== Extracting Qwen Text Foundation Embeddings ($TEXT_MODEL) for $EXPECTED_ROWS samples ==="
    python -u -m src.extract_embeddings \
        --model "$TEXT_MODEL" \
        --batch_size 64 \
        --max_length 256 \
        $SUBSET_EXTRACT_ARG
fi

# Check / Extract SigLIP Text Embeddings
if check_cache_rows "data/embeddings/train_text_siglip.npy" "$EXPECTED_ROWS"; then
    echo "Found existing SigLIP text embeddings at data/embeddings/train_text_siglip.npy ($EXPECTED_ROWS+ rows)"
else
    echo ""
    echo "=== Extracting SigLIP Text Embeddings (google/siglip-base-patch16-224) for $EXPECTED_ROWS samples ==="
    python -u -m src.extract_embeddings \
        --siglip_text \
        --batch_size 128 \
        $SUBSET_EXTRACT_ARG
fi

# 4. Launch Milestone 8 GPU Training Pipeline
echo ""
echo "=== Launching Milestone 8 End-to-End Training Pipeline ==="
LOG_FILE="run_milestone8_$(date +%Y%m%d_%H%M%S).log"

python -u -m src.gpu_train_foundation \
    --model_tag "$TEXT_TAG" \
    --vision_tag dual \
    --image_dir images \
    --embeddings_dir data/embeddings \
    --use_modal_tower \
    --modal_tower_epochs 30 \
    --modal_tower_lr 3e-4 \
    --modal_tower_loss huber \
    --modal_tower_target log \
    --train_mae_adapter \
    --iqr_trim_multiplier 3.5 \
    $SUBSET_ARG \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=== Milestone 8 Training Run Complete ==="
echo "Logs saved to: $LOG_FILE"
if [ -f "dataset/test_out.csv" ]; then
    echo "Submission verified at: dataset/test_out.csv"
    python -c "
import pandas as pd
df = pd.read_csv('dataset/test_out.csv')
print(f'Rows: {len(df)}, Nulls: {df.isnull().sum().sum()}, Min: \${df[\"price\"].min():.2f}, Median: \${df[\"price\"].median():.2f}, Max: \${df[\"price\"].max():.2f}')
"
fi
