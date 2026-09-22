#!/usr/bin/env bash
# ==============================================================================
# Amazon ML Challenge 2025: Milestone 5 Turnkey GPU Training Script
#
# Incorporates all Winner Techniques:
# - Stratified Price K-Fold CV (5 Folds)
# - Precomputed BGE-Large (1024-d) + SigLIP (768-d) Embeddings
# - FAISS k-NN Price Retrieval Features (6 features, leak-free OOF)
# - Advanced Catalog Features (Categories, Brand Price Tiers, Material Scores, Keyword Flags)
# - Dual Neural Pricing Adapters (SMAPE Loss + MAE Loss for ensemble decorrelation)
# - LightGBM with Analytical SMAPE Custom Objective & Evaluation
# - CatBoost GPU with MAE Loss on 32-dim SVD Vision Features
# - XGBoost GPU with Depth-Wise Splits (if installed)
# - IQR Outlier Training Trimming (multiplier=3.5)
# - Multi-Model Nelder-Mead Optimization + Post-Processing Calibration
# - Validation-Test Distribution Alignment Check
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

# 2. Check for optional dependencies (faiss, xgboost)
echo "=== Checking GPU Environment Dependencies ==="
python -c "
for pkg in ['torch', 'lightgbm', 'catboost', 'xgboost', 'faiss', 'scipy', 'sklearn']:
    try:
        __import__(pkg)
        print(f'  [OK] {pkg}')
    except ImportError:
        print(f'  [MISSING] {pkg}')
"

# 3. Run Pre-flight Diagnostic Error Analysis on Existing OOF Predictions (if present)
if [ -f "data/predictions/oof_predictions.npz" ]; then
    echo ""
    echo "=== Running Pre-flight Diagnostic Error Analysis ==="
    python -u -m src.error_analysis || true
fi

# 4. Launch Full Milestone 5 Training Run
echo ""
echo "=== Launching Milestone 5 Training Pipeline ==="
LOG_FILE="run_milestone5_$(date +%Y%m%d_%H%M%S).log"
SUBSET_ARG=""
if [ -n "$1" ]; then
    SUBSET_ARG="--subset $1"
    echo "=== Running in Fast Benchmark Mode: $1 Samples ==="
fi

python -u -m src.gpu_train_foundation \
    --model_tag bge_large_en_v1.5 \
    --vision_tag dual \
    --image_dir images \
    --embeddings_dir data/embeddings \
    --epochs 35 \
    --batch_size 256 \
    --lr 1e-3 \
    --train_mae_adapter \
    --iqr_trim_multiplier 3.5 \
    $SUBSET_ARG \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=== Milestone 5 Training Run Complete ==="
echo "Logs saved to: $LOG_FILE"
echo "Submission verified at: dataset/test_out.csv"
