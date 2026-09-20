"""Training pipeline for the Amazon ML Challenge 2025.

Config-driven training with K-Fold cross-validation, SMAPE evaluation,
and model persistence. Supports Ridge (baseline) and LightGBM.
"""
from __future__ import annotations

import os
import pickle
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from src.data import load_test, load_train, validate_submission
from src.features import (
    NUMERIC_COLS,
    build_feature_matrix,
    build_numeric_matrix,
    extract_structured_features,
    extract_text_features,
)
from src.metrics import smape


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> tuple[csr_matrix, csr_matrix, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Extract and combine all features for train and test.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        config: Feature configuration dict.

    Returns:
        Tuple of (X_train, X_test, y_train, train_structured, test_structured).
    """
    feat_cfg = config.get("features", {})

    # Extract structured features
    print("Extracting structured features...")
    train_struct = extract_structured_features(train_df)
    test_struct = extract_structured_features(test_df)

    # Unit frequency encoding (fit on train, apply to both)
    unit_freq = train_struct["unit_normalized"].value_counts().to_dict()
    train_struct["unit_freq"] = train_struct["unit_normalized"].map(unit_freq).fillna(0)
    test_struct["unit_freq"] = test_struct["unit_normalized"].map(unit_freq).fillna(0)

    # TF-IDF features
    print("Vectorizing text with TF-IDF...")
    X_text_train, vectorizer = extract_text_features(
        train_struct["catalog_content"],
        max_features=feat_cfg.get("tfidf_max_features", 15_000),
        ngram_range=tuple(feat_cfg.get("tfidf_ngram_range", [1, 2])),
        min_df=feat_cfg.get("tfidf_min_df", 3),
    )
    X_text_test, _ = extract_text_features(
        test_struct["catalog_content"],
        vectorizer=vectorizer,
    )

    # Numeric features (extend NUMERIC_COLS with unit_freq)
    num_cols = NUMERIC_COLS + ["unit_freq"]
    X_num_train = build_numeric_matrix(train_struct, columns=num_cols)
    X_num_test = build_numeric_matrix(test_struct, columns=num_cols)

    # Combine
    X_train = build_feature_matrix(X_text_train, X_num_train)
    X_test = build_feature_matrix(X_text_test, X_num_test)

    y_train = train_df["price"].values

    print(f"Feature matrix shape: {X_train.shape}")
    return X_train, X_test, y_train, train_struct, test_struct


def train_ridge_cv(
    X_train: csr_matrix,
    y_train: np.ndarray,
    X_test: csr_matrix,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train Ridge regression with K-Fold CV on log(price).

    Args:
        X_train: Training feature matrix.
        y_train: Training prices.
        X_test: Test feature matrix.
        config: Training configuration.

    Returns:
        Tuple of (oof_predictions, test_predictions, fold_smape_scores).
    """
    n_folds = config.get("n_folds", 5)
    seed = config.get("seed", 42)
    clip_min = config.get("target", {}).get("clip_min", 0.01)

    y_log = np.log(np.maximum(y_train, clip_min))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(y_train))
    test_preds = np.zeros(X_test.shape[0])
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, y_tr = X_train[train_idx], y_log[train_idx]
        X_va = X_train[val_idx]
        y_va_true = y_train[val_idx]

        model = Ridge(alpha=1.5, random_state=seed)
        model.fit(X_tr, y_tr)

        val_pred = np.exp(model.predict(X_va))
        val_pred = np.maximum(val_pred, clip_min)
        oof_preds[val_idx] = val_pred

        fold_smape = smape(y_va_true, val_pred)
        fold_scores.append(fold_smape)
        print(f"  Fold {fold + 1}/{n_folds} Ridge SMAPE: {fold_smape:.2f}%")

        test_pred = np.exp(model.predict(X_test))
        test_preds += np.maximum(test_pred, clip_min) / n_folds

    return oof_preds, test_preds, fold_scores


def train_lgbm_cv(
    X_train: csr_matrix,
    y_train: np.ndarray,
    X_test: csr_matrix,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train LightGBM with K-Fold CV on log(price).

    Args:
        X_train: Training feature matrix.
        y_train: Training prices.
        X_test: Test feature matrix.
        config: Training configuration with 'lgbm' section.

    Returns:
        Tuple of (oof_predictions, test_predictions, fold_smape_scores).
    """
    import lightgbm as lgb

    n_folds = config.get("n_folds", 5)
    seed = config.get("seed", 42)
    clip_min = config.get("target", {}).get("clip_min", 0.01)
    lgbm_params = dict(config.get("lgbm", {}))

    y_log = np.log(np.maximum(y_train, clip_min))
    early_stopping = lgbm_params.pop("early_stopping_rounds", 100)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(y_train))
    test_preds = np.zeros(X_test.shape[0])
    fold_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, y_tr = X_train[train_idx], y_log[train_idx]
        X_va, y_va_log = X_train[val_idx], y_log[val_idx]
        y_va_true = y_train[val_idx]

        model = lgb.LGBMRegressor(**lgbm_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va_log)],
            callbacks=[
                lgb.early_stopping(early_stopping, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        val_pred = np.exp(model.predict(X_va))
        val_pred = np.maximum(val_pred, clip_min)
        oof_preds[val_idx] = val_pred

        fold_smape_val = smape(y_va_true, val_pred)
        fold_scores.append(fold_smape_val)
        print(
            f"  Fold {fold + 1}/{n_folds} LightGBM SMAPE: {fold_smape_val:.2f}% "
            f"(best iter: {model.best_iteration_})"
        )

        test_pred = np.exp(model.predict(X_test))
        test_preds += np.maximum(test_pred, clip_min) / n_folds
        models.append(model)

    return oof_preds, test_preds, fold_scores


def save_submission(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    output_path: str,
) -> pd.DataFrame:
    """Generate and save submission CSV.

    Args:
        test_df: Test DataFrame with sample_id column.
        predictions: Predicted prices (must be positive floats).
        output_path: Path to save the output CSV.

    Returns:
        Submission DataFrame.
    """
    submission = pd.DataFrame({
        "sample_id": test_df["sample_id"],
        "price": np.maximum(predictions, 1e-5),
    })
    # Validate before saving
    validate_submission(submission, expected_count=len(test_df))
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(submission)} rows)")
    return submission


def main(config_path: str = "configs/default.yaml") -> None:
    """Run the full training pipeline.

    Args:
        config_path: Path to YAML configuration file.
    """
    config = load_config(config_path)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, config["paths"]["dataset_dir"])
    model_dir = os.path.join(base_dir, config["paths"]["model_dir"])
    output_path = os.path.join(base_dir, config["paths"]["output_file"])
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    print("Loading datasets...")
    train_df = load_train(dataset_dir)
    test_df = load_test(dataset_dir)
    print(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")

    # Prepare features
    X_train, X_test, y_train, _, _ = prepare_features(train_df, test_df, config)

    # Train Ridge baseline
    print("\n=== Ridge Baseline ===")
    start = time.time()
    ridge_oof, ridge_test, ridge_scores = train_ridge_cv(
        X_train, y_train, X_test, config
    )
    ridge_smape = smape(y_train, ridge_oof)
    print(f"Ridge OOF SMAPE: {ridge_smape:.2f}% ({time.time() - start:.1f}s)")

    # Train LightGBM
    try:
        print("\n=== LightGBM ===")
        start = time.time()
        lgbm_oof, lgbm_test, lgbm_scores = train_lgbm_cv(
            X_train, y_train, X_test, config
        )
        lgbm_smape = smape(y_train, lgbm_oof)
        print(f"LightGBM OOF SMAPE: {lgbm_smape:.2f}% ({time.time() - start:.1f}s)")

        # Use better model for submission
        if lgbm_smape < ridge_smape:
            print(f"\nLightGBM wins ({lgbm_smape:.2f}% < {ridge_smape:.2f}%)")
            best_test = lgbm_test
            best_name = "LightGBM"
        else:
            print(f"\nRidge wins ({ridge_smape:.2f}% < {lgbm_smape:.2f}%)")
            best_test = ridge_test
            best_name = "Ridge"
    except ImportError:
        print("\nLightGBM not available, using Ridge baseline")
        best_test = ridge_test
        best_name = "Ridge"

    # Save submission
    print(f"\n=== Generating submission with {best_name} ===")
    save_submission(test_df, best_test, output_path)

    # Summary
    print("\n==========================================")
    print(f"Best model: {best_name}")
    print(f"Ridge OOF SMAPE: {ridge_smape:.2f}%")
    if "lgbm_smape" in dir():
        print(f"LightGBM OOF SMAPE: {lgbm_smape:.2f}%")
    print("==========================================")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ML Challenge models")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(args.config)
