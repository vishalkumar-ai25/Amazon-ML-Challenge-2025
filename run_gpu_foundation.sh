#!/bin/bash
# ==============================================================================
# Amazon ML Challenge 2025: Foundation Model + Neural Adapter GPU Pipeline
# Target machine: NVIDIA GPU Server (24je093024je0930@172.16.203.23)
# ==============================================================================
set -e

# Detect Operating System & GPU
OS="$(uname -s)"
echo "=== Host Environment: $OS ($(uname -m)) ==="

BATCH_SIZE=128
if command -v nvidia-smi &> /dev/null; then
    echo "=== NVIDIA GPU Detected ==="
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    BATCH_SIZE=128
else
    echo "========================================================================"
    echo "⚠️  WARNING: 'nvidia-smi' not found on this machine!"
    if [ "$OS" = "Darwin" ]; then
        echo "You are currently running on your local MacBook Air (macOS)."
        echo "To use your NVIDIA RTX A4000 GPU, please connect to your remote server:"
        echo "    ssh -L 11434:localhost:11434 24je093024je0930@172.16.203.23"
        echo "    cd Amazon-Ml-Prep"
        echo "    ./run_gpu_foundation.sh"
        echo "========================================================================"
        echo "Falling back to local CPU/MPS mode with batch size 32..."
        BATCH_SIZE=32
    else
        echo "No NVIDIA driver detected. Continuing with CPU fallback..."
        BATCH_SIZE=32
    fi
fi

# Check Python virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo -e "\n=== Step 1: Ensuring Deep Learning Dependencies are Present ==="
pip install -q transformers sentence-transformers accelerate torch torchvision pillow || true

echo -e "\n=== Step 2: Downloading Product Images (Concurrent & Resilient) ==="
# Resumes automatically if images are already downloaded; verifies integrity
python -m src.download_images \
    --split both \
    --workers 32

echo -e "\n=== Step 3: Extracting Pretrained Text Embeddings (BGE-Large) ==="
python -m src.extract_embeddings \
    --model "BAAI/bge-large-en-v1.5" \
    --batch_size $BATCH_SIZE \
    --output_dir "data/embeddings" \
    --text_only

echo -e "\n=== Step 4: Extracting Pretrained Vision Embeddings (SigLIP Base) ==="
# Extracts 768-dim normalized representations using Google SigLIP
python -m src.extract_embeddings \
    --vision_model "google/siglip-base-patch16-224" \
    --batch_size $BATCH_SIZE \
    --output_dir "data/embeddings" \
    --vision_only

export PYTHONUNBUFFERED=1

echo -e "\n=== Step 5: Running Multimodal Adapter + Dual GBDTs 5-Fold Training ==="
python -u -m src.gpu_train_foundation \
    --model_tag "bge_large_en_v1.5" \
    --vision_tag "siglip_base_patch16_224" \
    --image_dir "images" \
    --embeddings_dir "data/embeddings" \
    --epochs 35 \
    --batch_size 256 \
    --lr 0.001

echo -e "\n=== Training Complete! Submission File Verified ==="
ls -lh dataset/test_out.csv
head -n 5 dataset/test_out.csv

