"""Error Analysis Script for Amazon ML Challenge 2025.

Stratified SMAPE analysis on OOF predictions to identify where the 6.2% gap
to the 39.19% winner score is concentrated. Run on the remote GPU server after
training completes.

Usage:
    python -u -m src.error_analysis
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from src.features import extract_structured_features
from src.metrics import smape


def smape_per_sample(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute per-sample SMAPE values (0-200 scale)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-8
    return 100.0 * num / den


def analyze_by_price_decile(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Stratify SMAPE by price decile to identify where errors concentrate."""
    per_sample = smape_per_sample(y_true, y_pred)
    df = pd.DataFrame({
        "price": y_true,
        "pred": y_pred,
        "smape": per_sample,
    })
    df["price_decile"] = pd.qcut(df["price"], q=10, labels=False, duplicates="drop")
    
    summary = df.groupby("price_decile").agg(
        count=("price", "count"),
        price_min=("price", "min"),
        price_max=("price", "max"),
        price_median=("price", "median"),
        pred_median=("pred", "median"),
        mean_smape=("smape", "mean"),
        median_smape=("smape", "median"),
        p90_smape=("smape", lambda x: np.percentile(x, 90)),
        pct_over_100=("smape", lambda x: 100.0 * (x > 100).mean()),
    ).round(2)
    
    return summary


def analyze_by_category_keywords(
    catalog_content: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Estimate product category from keywords and compute SMAPE per category."""
    per_sample = smape_per_sample(y_true, y_pred)
    
    categories = {
        "Electronics": ["electronic", "battery", "charger", "cable", "usb", "bluetooth", "speaker", "headphone", "adapter", "hdmi"],
        "Beauty": ["beauty", "cosmetic", "makeup", "skin care", "moisturizer", "serum", "lipstick", "foundation", "mascara"],
        "Grocery": ["grocery", "food", "snack", "cereal", "coffee", "tea", "spice", "sauce", "pasta", "rice", "flour"],
        "Health": ["health", "vitamin", "supplement", "medicine", "first aid", "bandage", "thermometer"],
        "Home": ["home", "kitchen", "furniture", "bedding", "towel", "curtain", "pillow", "mat", "storage"],
        "Toys": ["toy", "game", "puzzle", "doll", "action figure", "lego", "board game"],
        "Sports": ["sport", "fitness", "exercise", "yoga", "bicycle", "camping", "outdoor"],
        "Clothing": ["shirt", "pants", "dress", "shoes", "socks", "jacket", "hat", "gloves"],
        "Pet": ["pet", "dog", "cat", "fish", "bird", "treat", "leash", "collar"],
        "Baby": ["baby", "infant", "toddler", "diaper", "pacifier", "stroller", "crib"],
    }
    
    text_lower = catalog_content.str.lower().fillna("")
    cat_labels = pd.Series(["Other"] * len(text_lower), index=text_lower.index)
    
    for cat_name, keywords in categories.items():
        pattern = "|".join(keywords)
        mask = text_lower.str.contains(pattern, regex=True, na=False)
        cat_labels[mask & (cat_labels == "Other")] = cat_name
    
    df = pd.DataFrame({
        "category": cat_labels,
        "price": y_true,
        "smape": per_sample,
    })
    
    summary = df.groupby("category").agg(
        count=("price", "count"),
        price_median=("price", "median"),
        mean_smape=("smape", "mean"),
        median_smape=("smape", "median"),
    ).sort_values("mean_smape", ascending=False).round(2)
    
    return summary


def analyze_by_image_presence(
    image_paths_exist: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Compare SMAPE for samples with vs without downloaded images."""
    per_sample = smape_per_sample(y_true, y_pred)
    df = pd.DataFrame({
        "has_image": image_paths_exist,
        "price": y_true,
        "smape": per_sample,
    })
    
    summary = df.groupby("has_image").agg(
        count=("price", "count"),
        price_median=("price", "median"),
        mean_smape=("smape", "mean"),
        median_smape=("smape", "median"),
    ).round(2)
    
    return summary


def analyze_prediction_bias(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Analyze systematic prediction bias (over/under-prediction patterns)."""
    log_ratio = np.log(y_pred + 1e-8) - np.log(y_true + 1e-8)
    
    return {
        "mean_log_ratio": float(np.mean(log_ratio)),
        "median_log_ratio": float(np.median(log_ratio)),
        "pct_overpredicted": float(100.0 * (y_pred > y_true).mean()),
        "pct_underpredicted": float(100.0 * (y_pred < y_true).mean()),
        "mean_overpred_magnitude": float(np.mean(log_ratio[log_ratio > 0])) if (log_ratio > 0).any() else 0.0,
        "mean_underpred_magnitude": float(np.mean(np.abs(log_ratio[log_ratio < 0]))) if (log_ratio < 0).any() else 0.0,
    }


def analyze_worst_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    catalog_content: pd.Series,
    n: int = 20,
) -> pd.DataFrame:
    """Find the N worst-predicted samples to identify systematic failure patterns."""
    per_sample = smape_per_sample(y_true, y_pred)
    df = pd.DataFrame({
        "price": y_true,
        "pred": y_pred,
        "smape": per_sample,
        "catalog_snippet": catalog_content.str[:120],
    })
    return df.nlargest(n, "smape").round(2)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pred_dir = os.path.join(base_dir, "data", "predictions")
    oof_path = os.path.join(pred_dir, "oof_predictions.npz")
    
    if not os.path.exists(oof_path):
        print(f"ERROR: OOF predictions not found at {oof_path}")
        print("Run the training pipeline first to generate OOF predictions.")
        sys.exit(1)
    
    print("=" * 70)
    print("  ERROR ANALYSIS: Where is the 6.2% SMAPE Gap Concentrated?")
    print("=" * 70)
    
    # Load data
    train_path = os.path.join(base_dir, "dataset", "train.csv")
    train_df = pd.read_csv(train_path)
    
    data = np.load(oof_path)
    if "y_true" in data:
        y_true = data["y_true"]
    else:
        y_true = train_df["price"].values

    if len(train_df) != len(y_true):
        train_df = train_df.iloc[:len(y_true)].reset_index(drop=True)

    available_keys = list(data.keys())
    print(f"\nAvailable OOF prediction keys: {available_keys}")
    
    # Use the best individual model and blended predictions
    models_to_analyze = {}
    for key in available_keys:
        if key == "y_true":
            continue
        models_to_analyze[key] = data[key]
    
    # If we have individual model OOFs, also create a simple blend
    if "oof_lgbm" in data and "oof_adapter" in data:
        blend = 0.434 * data["oof_lgbm"] + 0.566 * data["oof_adapter"]
        models_to_analyze["blend_56_43"] = blend
    
    for model_name, y_pred in models_to_analyze.items():
        if len(y_pred) != len(y_true):
            print(f"\nSkipping {model_name}: shape mismatch ({len(y_pred)} vs {len(y_true)})")
            continue
            
        overall_smape = smape(y_true, y_pred)
        print(f"\n{'='*70}")
        print(f"  Model: {model_name}  |  Overall OOF SMAPE: {overall_smape:.2f}%")
        print(f"{'='*70}")
        
        # 1. Price Decile Analysis
        print(f"\n--- SMAPE by Price Decile ---")
        decile_df = analyze_by_price_decile(y_true, y_pred)
        print(decile_df.to_string())
        
        # 2. Prediction Bias
        print(f"\n--- Prediction Bias Analysis ---")
        bias = analyze_prediction_bias(y_true, y_pred)
        for k, v in bias.items():
            print(f"  {k}: {v:.4f}")
    
    # Category analysis (use blend or best model)
    best_key = min(models_to_analyze, key=lambda k: smape(y_true, models_to_analyze[k]) if len(models_to_analyze[k]) == len(y_true) else 999)
    y_best = models_to_analyze[best_key]
    
    print(f"\n{'='*70}")
    print(f"  Category Analysis (using {best_key})")
    print(f"{'='*70}")
    cat_df = analyze_by_category_keywords(train_df["catalog_content"], y_true, y_best)
    print(cat_df.to_string())
    
    # Image presence analysis
    img_dir = os.path.join(base_dir, "images", "train")
    if os.path.exists(img_dir):
        print(f"\n--- SMAPE by Image Presence ---")
        has_image = np.array([
            os.path.exists(os.path.join(img_dir, f"{sid}.jpg"))
            for sid in train_df["sample_id"]
        ])
        img_df = analyze_by_image_presence(has_image, y_true, y_best)
        print(img_df.to_string())
    else:
        print(f"\nImage directory not found at {img_dir}, skipping image presence analysis.")
    
    # Worst samples
    print(f"\n--- Top 20 Worst Predictions ---")
    worst_df = analyze_worst_samples(y_true, y_best, train_df["catalog_content"])
    print(worst_df.to_string())
    
    print(f"\n{'='*70}")
    print("  Error Analysis Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
