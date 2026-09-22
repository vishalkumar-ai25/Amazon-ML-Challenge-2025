"""Foundation Embedding Extraction Pipeline for Amazon ML Challenge 2025.

Performs forward-only batch inference (torch.no_grad) with frozen foundation models
(e.g., Qwen2.5, BGE-large, SigLIP-2, DINOv2) and caches dense representations to disk.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features import build_llm_prompt, extract_structured_features


def get_optimal_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on available hardware."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def extract_text_embeddings_hf(
    texts: Sequence[str],
    model_name: str = "BAAI/bge-large-en-v1.5",
    batch_size: int = 64,
    device: Optional[str] = None,
    max_length: int = 256,
) -> np.ndarray:
    """Extract dense text representations using HuggingFace / SentenceTransformers.

    Args:
        texts: List or Series of prompt strings.
        model_name: HuggingFace model repo ID.
        batch_size: Batch size for GPU inference.
        device: 'cuda', 'mps', or 'cpu'. Auto-detects if None.
        max_length: Maximum sequence length.

    Returns:
        (N, hidden_dim) float32 numpy array of normalized embeddings.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = get_optimal_device()

    print(f"Loading text encoder: {model_name} on device: {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Use bfloat16/float16 on CUDA for 2x speedup and 50% memory savings
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device == "cuda" else torch.float32)
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    embeddings_list = []
    n_samples = len(texts)

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Extracting text embeddings"):
            batch_texts = list(texts[i : i + batch_size])
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**encoded)
            # Use mean pooling with attention mask
            token_embeddings = outputs[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask

            # L2 normalize embeddings
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings_list.append(normalized.cpu().to(torch.float32).numpy())

    return np.vstack(embeddings_list)


def extract_vision_embeddings_hf(
    image_paths: Sequence[str],
    model_name: str = "google/siglip-base-patch16-224",
    batch_size: int = 64,
    device: Optional[str] = None,
) -> np.ndarray:
    """Extract dense vision representations using SigLIP, DINOv2, or other ViT models.

    Args:
        image_paths: List of local filepaths to downloaded product images.
        model_name: HuggingFace vision model ID.
        batch_size: Batch size for GPU inference.
        device: 'cuda', 'mps', or 'cpu'.

    Returns:
        (N, hidden_dim) float32 numpy array. Missing images are replaced with zero vectors.
    """
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    if device is None:
        device = get_optimal_device()

    print(f"Loading vision encoder: {model_name} on device: {device}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device == "cuda" else torch.float32)
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    n_samples = len(image_paths)
    all_embeddings: Optional[np.ndarray] = None
    hidden_dim: Optional[int] = None
    valid_count = 0
    missing_count = 0

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc=f"Vision emb ({model_name.split('/')[-1]})"):
            batch_paths = image_paths[i : i + batch_size]
            valid_images = []
            valid_indices = []

            for rel_idx, p in enumerate(batch_paths):
                if p and os.path.exists(p) and os.path.getsize(p) > 1024:
                    try:
                        img = Image.open(p).convert("RGB")
                        valid_images.append(img)
                        valid_indices.append(i + rel_idx)
                        valid_count += 1
                    except Exception:
                        missing_count += 1
                else:
                    missing_count += 1

            if valid_images:
                inputs = processor(images=valid_images, return_tensors="pt").to(device, dtype=dtype)

                # Check model architecture specifics (SigLIP / CLIP vs standard ViT / DINOv2)
                if hasattr(model, "get_image_features"):
                    emb = model.get_image_features(**inputs)
                else:
                    outputs = model(**inputs)
                    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                        emb = outputs.pooler_output
                    elif hasattr(outputs, "last_hidden_state"):
                        # If CLS token exists (e.g. DINOv2, ViT) and seq_len > 1
                        if outputs.last_hidden_state.shape[1] > 1:
                            emb = outputs.last_hidden_state[:, 0]
                        else:
                            emb = outputs.last_hidden_state.mean(dim=1)
                    else:
                        emb = outputs[0][:, 0] if outputs[0].ndim == 3 else outputs[0]

                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
                emb_np = emb.cpu().to(torch.float32).numpy()

                if all_embeddings is None:
                    hidden_dim = emb_np.shape[1]
                    all_embeddings = np.zeros((n_samples, hidden_dim), dtype=np.float32)

                all_embeddings[valid_indices] = emb_np

    if all_embeddings is None:
        hidden_dim = getattr(getattr(model, "config", None), "hidden_size", 768)
        all_embeddings = np.zeros((n_samples, hidden_dim), dtype=np.float32)

    print(f"Extracted vision embeddings: {valid_count} valid images, {missing_count} missing/corrupt. Shape: {all_embeddings.shape}")
    return all_embeddings


def clean_model_tag(model_name: str) -> str:
    """Normalize model repo string to a file-safe tag."""
    return model_name.split("/")[-1].lower().replace("-", "_").replace(".", "_")


def main():
    parser = argparse.ArgumentParser(description="Extract Frozen Foundation Model Embeddings")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5", help="Text model repo ID")
    parser.add_argument("--batch_size", type=int, default=64, help="GPU batch size")
    parser.add_argument("--output_dir", default="data/embeddings", help="Folder to save cached embeddings")
    parser.add_argument("--vision_model", default=None, help="Optional vision model ID (e.g. google/siglip-base-patch16-224 or facebook/dinov2-base)")
    parser.add_argument("--image_dir", default="images", help="Folder containing downloaded images")
    parser.add_argument("--vision_only", action="store_true", help="Skip text extraction and only extract vision embeddings")
    parser.add_argument("--text_only", action="store_true", help="Skip vision extraction and only extract text embeddings")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "dataset", "train.csv")
    test_path = os.path.join(base_dir, "dataset", "test.csv")
    out_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 1. Text embeddings
    if not args.vision_only:
        print("Extracting structured prompt fields...")
        train_struct = extract_structured_features(train_df)
        test_struct = extract_structured_features(test_df)

        tag = clean_model_tag(args.model)
        train_out_text = os.path.join(out_dir, f"train_text_{tag}.npy")
        test_out_text = os.path.join(out_dir, f"test_text_{tag}.npy")

        if not os.path.exists(train_out_text) or not os.path.exists(test_out_text):
            print(f"\n[Text] Extracting text embeddings using {args.model}...")
            train_emb = extract_text_embeddings_hf(train_struct["llm_prompt"], model_name=args.model, batch_size=args.batch_size)
            test_emb = extract_text_embeddings_hf(test_struct["llm_prompt"], model_name=args.model, batch_size=args.batch_size)

            np.save(train_out_text, train_emb)
            np.save(test_out_text, test_emb)
            print(f"Saved: {train_out_text} shape: {train_emb.shape}")
            print(f"Saved: {test_out_text} shape: {test_emb.shape}")
        else:
            print(f"Found existing cached text embeddings in {out_dir}")

    # 2. Vision embeddings (optional, supports single or comma-separated list like "google/siglip-base-patch16-224,facebook/dinov2-base")
    if args.vision_model and not args.text_only:
        train_img_paths = [os.path.join(base_dir, args.image_dir, "train", f"{sid}.jpg") for sid in train_df["sample_id"]]
        test_img_paths = [os.path.join(base_dir, args.image_dir, "test", f"{sid}.jpg") for sid in test_df["sample_id"]]

        vision_models = [m.strip() for m in args.vision_model.split(",") if m.strip()]
        for v_model in vision_models:
            v_tag = clean_model_tag(v_model)
            train_out_v = os.path.join(out_dir, f"train_vision_{v_tag}.npy")
            test_out_v = os.path.join(out_dir, f"test_vision_{v_tag}.npy")

            if not os.path.exists(train_out_v):
                print(f"\n[Vision Train] Extracting train vision embeddings using {v_model} (tag: {v_tag})...")
                train_v_emb = extract_vision_embeddings_hf(train_img_paths, model_name=v_model, batch_size=args.batch_size)
                np.save(train_out_v, train_v_emb)
                print(f"Saved: {train_out_v} shape: {train_v_emb.shape}")
            else:
                print(f"Found existing cached train vision embeddings for {v_tag} in {out_dir}")

            if not os.path.exists(test_out_v):
                print(f"\n[Vision Test] Extracting test vision embeddings using {v_model} (tag: {v_tag})...")
                test_v_emb = extract_vision_embeddings_hf(test_img_paths, model_name=v_model, batch_size=args.batch_size)
                np.save(test_out_v, test_v_emb)
                print(f"Saved: {test_out_v} shape: {test_v_emb.shape}")
            else:
                print(f"Found existing cached test vision embeddings for {v_tag} in {out_dir}")

    print("\nEmbedding extraction completed successfully!")


if __name__ == "__main__":
    main()
