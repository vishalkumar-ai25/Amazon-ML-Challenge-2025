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

# 2. Check for optional dependencies
echo "=== Checking Environment Dependencies ==="
python -c "
for pkg in ['torch', 'transformers', 'lightgbm', 'catboost', 'faiss', 'scipy', 'sklearn', 'num2words', 'sentencepiece']:
    try:
        __import__(pkg)
        print(f'  [OK] {pkg}')
    except ImportError:
        print(f'  [MISSING] {pkg}')
"

# Optional auto-install for num2words and sentencepiece if missing
python -c "import num2words" 2>/dev/null || {
    echo "Installing num2words for sub-word tokenization..."
    pip install num2words --no-warn-script-location || true
}
python -c "import sentencepiece" 2>/dev/null || {
    echo "Installing sentencepiece for SigLIP tokenizer..."
    pip install sentencepiece --no-warn-script-location || true
}

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

# Check if Qwen2.5-3B exists instead
if [ -f "data/embeddings/train_text_qwen2_5_3b.npy" ]; then
    TEXT_MODEL="Qwen/Qwen2.5-3B"
    TEXT_TAG="qwen2_5_3b"
    echo "Found existing Qwen2.5-3B text embeddings tag: $TEXT_TAG"
elif [ -f "data/embeddings/train_text_${TEXT_TAG}.npy" ]; then
    echo "Found existing GTE-Qwen text embeddings tag: $TEXT_TAG"
else
    echo ""
    echo "=== Extracting Qwen Text Foundation Embeddings ($TEXT_MODEL) ==="
    python -u -m src.extract_embeddings \
        --model "$TEXT_MODEL" \
        --batch_size 64 \
        --max_length 256 \
        $SUBSET_EXTRACT_ARG
fi

# Check / Extract SigLIP Text Embeddings
if [ ! -f "data/embeddings/train_text_siglip.npy" ]; then
    echo ""
    echo "=== Extracting SigLIP Text Embeddings (google/siglip-base-patch16-224) ==="
    python -u -m src.extract_embeddings \
        --siglip_text \
        --batch_size 128 \
        $SUBSET_EXTRACT_ARG
else
    echo "Found existing SigLIP text embeddings at data/embeddings/train_text_siglip.npy"
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
    --modal_tower_loss mse \
    --modal_tower_target log1p \
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
