#!/bin/bash
# ==============================================================================
# Amazon ML Challenge 2025: Foundation Model + Neural Adapter GPU Pipeline
# Run this on your GPU server: 24je093024je0930@172.16.203.23
# ==============================================================================
set -e

echo "=== System GPU Information ==="
nvidia-smi

# Check Python environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo -e "\n=== Step 1: Ensuring Deep Learning Dependencies are Present ==="
pip install -q transformers sentence-transformers accelerate || true

echo -e "\n=== Step 2: Extracting Pretrained Foundation Embeddings ==="
# Extracts 1024-dim dense text representations using frozen BGE-large in ~10-15 mins
# (Change --model to 'Qwen/Qwen2.5-3B-Instruct' if you want a 3B LLM representation)
python -m src.extract_embeddings \
    --model "BAAI/bge-large-en-v1.5" \
    --batch_size 128 \
    --output_dir "data/embeddings"

echo -e "\n=== Step 3: Running Foundation Adapter + GBDT 5-Fold Training ==="
python -m src.gpu_train_foundation \
    --model_tag "bge_large_en_v1.5" \
    --embeddings_dir "data/embeddings" \
    --epochs 35 \
    --batch_size 256 \
    --lr 0.001

echo -e "\n=== Training Complete! Submission File Verified ==="
ls -lh dataset/test_out.csv
head -n 5 dataset/test_out.csv
