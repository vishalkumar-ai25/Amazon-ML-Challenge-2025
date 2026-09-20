"""Kaggle-optimized training pipeline for Amazon ML Challenge 2025.

Executes:
1. Dynamic dataset discovery (Kaggle inputs vs local directory)
2. High-capacity text vectorization + structured catalog parsing via src.features
3. Dual Gradient Boosting (LightGBM + CatBoost) + Ridge Regression with 5-fold CV
4. Metric-driven convex blending via src.ensemble minimizing SMAPE
5. Generates and strictly validates final test_out.csv
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor

from src.metrics import smape
from src.features import (
    extract_structured_features,
    extract_text_features,
    build_numeric_matrix,
    build_feature_matrix,
    NUMERIC_COLS,
)
from src.ensemble import find_optimal_blend_weights, apply_blend


def locate_dataset_paths() -> tuple[str, str, str]:
    """Locate train.csv and test.csv dynamically across Kaggle and local environments."""
    train_path, test_path = None, None
    search_bases = ["/kaggle/input", ".", "dataset"]

    for base in search_bases:
        if not os.path.exists(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if f.lower() == "train.csv" and not train_path:
                    train_path = os.path.join(root, f)
                elif f.lower() == "test.csv" and not test_path:
                    test_path = os.path.join(root, f)
            if train_path and test_path:
                break
        if train_path and test_path:
            break

    out_path = "/kaggle/working/test_out.csv" if os.path.exists("/kaggle/working") else "dataset/test_out.csv"

    if not train_path or not test_path:
        print("\nERROR: Could not automatically locate train.csv and test.csv!")
        if os.path.exists("/kaggle/input"):
            print("\nFiles found in /kaggle/input:")
            for root, dirs, files in os.walk("/kaggle/input"):
                print(f"  {root}: {files}")
            print("\nTIP: Make sure you clicked '+ Add Input / Data' on the right sidebar in Kaggle and attached your dataset.")
        raise FileNotFoundError(
            f"Could not locate train.csv and test.csv. Found train: {train_path}, test: {test_path}"
        )

    return train_path, test_path, out_path


def main():
    print("=" * 60)
    print(" Amazon ML Challenge 2025: Production Model Pipeline")
    print("=" * 60)

    train_path, test_path, out_path = locate_dataset_paths()

    print(f"Loading train data: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading test data: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Extract structured features
    print("\nExtracting structured features...")
    train_struct = extract_structured_features(train_df)
    test_struct = extract_structured_features(test_df)

    # Unit frequency
    u_freq = train_struct["unit_normalized"].value_counts().to_dict()
    train_struct["unit_freq"] = train_struct["unit_normalized"].map(u_freq).fillna(0)
    test_struct["unit_freq"] = test_struct["unit_normalized"].map(u_freq).fillna(0)

    # Brand frequency
    b_freq = train_struct["brand"].value_counts().to_dict()
    train_struct["brand_freq"] = train_struct["brand"].map(b_freq).fillna(0)
    test_struct["brand_freq"] = test_struct["brand"].map(b_freq).fillna(0)

    num_cols = NUMERIC_COLS + ["unit_freq", "brand_freq"]
    X_num_train = build_numeric_matrix(train_struct, columns=num_cols)
    X_num_test = build_numeric_matrix(test_struct, columns=num_cols)

    # TF-IDF text features
    print("Fitting TF-IDF Vectorizer on full catalog content...")
    X_text_train, tfidf = extract_text_features(
        train_struct["catalog_content"],
        max_features=25000,
        ngram_range=(1, 2),
        min_df=3,
    )
    X_text_test, _ = extract_text_features(
        test_struct["catalog_content"],
        vectorizer=tfidf,
    )

    X_train = build_feature_matrix(X_text_train, X_num_train)
    X_test = build_feature_matrix(X_text_test, X_num_test)
    print(f"Combined features shape: {X_train.shape}")

    y_train = train_df["price"].values
    y_log = np.log(np.maximum(y_train, 0.01))

    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_ridge = np.zeros(len(train_df))
    oof_lgbm = np.zeros(len(train_df))
    oof_cat = np.zeros(len(train_df))

    test_ridge = np.zeros(len(test_df))
    test_lgbm = np.zeros(len(test_df))
    test_cat = np.zeros(len(test_df))

    print("\nStarting 5-Fold Cross-Validation...")
    for fold, (t_idx, v_idx) in enumerate(kf.split(X_train)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        X_tr, y_tr_log = X_train[t_idx], y_log[t_idx]
        X_va, y_va_log = X_train[v_idx], y_log[v_idx]
        y_va_true = y_train[v_idx]

        # 1. Ridge
        m_ridge = Ridge(alpha=1.5, random_state=42)
        m_ridge.fit(X_tr, y_tr_log)
        val_pred_ridge = np.maximum(np.exp(m_ridge.predict(X_va)), 0.01)
        oof_ridge[v_idx] = val_pred_ridge
        test_ridge += np.maximum(np.exp(m_ridge.predict(X_test)), 0.01) / n_folds
        print(f"  Fold {fold+1} Ridge SMAPE: {smape(y_va_true, val_pred_ridge):.2f}%")

        # 2. LightGBM (Huber loss with live progress logs)
        m_lgbm = lgb.LGBMRegressor(
            objective="huber",
            metric="mae",
            n_estimators=800,
            learning_rate=0.08,
            num_leaves=63,
            min_child_samples=50,
            feature_fraction=0.75,
            bagging_fraction=0.8,
            bagging_freq=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        m_lgbm.fit(
            X_tr, y_tr_log,
            eval_set=[(X_va, y_va_log)],
            callbacks=[
                lgb.early_stopping(60, verbose=False),
                lgb.log_evaluation(period=25),
            ],
        )
        val_pred_lgbm = np.maximum(np.exp(m_lgbm.predict(X_va)), 0.01)
        oof_lgbm[v_idx] = val_pred_lgbm
        test_lgbm += np.maximum(np.exp(m_lgbm.predict(X_test)), 0.01) / n_folds
        print(f"  >>> Fold {fold+1} LightGBM SMAPE: {smape(y_va_true, val_pred_lgbm):.2f}% <<<")

        # 3. CatBoost (MAE loss with periodic progress)
        m_cat = CatBoostRegressor(
            loss_function="MAE",
            iterations=350,
            learning_rate=0.1,
            depth=6,
            random_seed=42,
            thread_count=-1,
            verbose=50,
        )
        m_cat.fit(
            X_tr, y_tr_log,
            eval_set=(X_va, y_va_log),
            early_stopping_rounds=50,
            verbose=50,
        )
        val_pred_cat = np.maximum(np.exp(m_cat.predict(X_va)), 0.01)
        oof_cat[v_idx] = val_pred_cat
        test_cat += np.maximum(np.exp(m_cat.predict(X_test)), 0.01) / n_folds
        print(f"  >>> Fold {fold+1} CatBoost SMAPE: {smape(y_va_true, val_pred_cat):.2f}% <<<")

    print("\n" + "=" * 60)
    print(f"Overall OOF Ridge SMAPE:    {smape(y_train, oof_ridge):.2f}%")
    print(f"Overall OOF LightGBM SMAPE: {smape(y_train, oof_lgbm):.2f}%")
    print(f"Overall OOF CatBoost SMAPE: {smape(y_train, oof_cat):.2f}%")

    # Optimal Convex Blending
    oof_dict = {"ridge": oof_ridge, "lgbm": oof_lgbm, "cat": oof_cat}
    weights = find_optimal_blend_weights(oof_dict, y_train)
    print(f"\nOptimal Ensemble Weights: {weights}")

    blended_oof = apply_blend(oof_dict, weights)
    final_smape = smape(y_train, blended_oof)
    print(f"FINAL ENSEMBLE OOF SMAPE:   {final_smape:.2f}%")
    print("=" * 60)

    # Generate Final Predictions
    test_dict = {"ridge": test_ridge, "lgbm": test_lgbm, "cat": test_cat}
    blended_test = apply_blend(test_dict, weights, clip_min=0.05)

    sub_df = pd.DataFrame({
        "sample_id": test_df["sample_id"],
        "price": blended_test
    })

    # Validate output format
    assert len(sub_df) == len(test_df), "Row count mismatch"
    assert list(sub_df.columns) == ["sample_id", "price"]
    assert (sub_df["price"] > 0).all(), "Non-positive price found"
    assert sub_df["price"].notna().all(), "NaN price found"

    sub_df.to_csv(out_path, index=False)
    print(f"\nSuccessfully generated and validated {out_path} ({len(sub_df)} rows)!")
    print("\nFirst 5 predictions:")
    print(sub_df.head())

if __name__ == "__main__":
    main()
