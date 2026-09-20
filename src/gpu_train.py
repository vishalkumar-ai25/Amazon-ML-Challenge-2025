"""High-performance GPU + Multi-core training pipeline for Amazon ML Challenge 2025.

Optimized for:
- 96 Xeon CPU cores + 187 GB RAM + NVIDIA RTX A4000 (16 GB VRAM)
- High-capacity 35,000-feature TF-IDF + regex catalog parsing
- CatBoost GPU acceleration (task_type='GPU', loss_function='MAE')
- LightGBM multi-threaded Huber loss (n_jobs=32, num_leaves=127)
- Ridge regression L2 linear regularizer
- Out-of-fold Nelder-Mead convex weight optimization targeting SMAPE directly
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


def main():
    print("=" * 70)
    print("  Amazon ML Challenge 2025: RTX A4000 GPU & Multi-Core Pipeline")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "dataset", "train.csv")
    test_path = os.path.join(base_dir, "dataset", "test.csv")
    out_path = os.path.join(base_dir, "dataset", "test_out.csv")

    print(f"Loading train data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Extract structured features
    print("\n[1/5] Extracting structured features & regex fields...")
    t0 = time.time()
    train_struct = extract_structured_features(train_df)
    test_struct = extract_structured_features(test_df)
    print(f"Structured extraction completed in {time.time() - t0:.1f}s")

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

    # TF-IDF high-capacity representations
    print("\n[2/5] Building high-capacity 35,000-term TF-IDF matrix...")
    t0 = time.time()
    X_text_train, tfidf = extract_text_features(
        train_struct["catalog_content"],
        max_features=35000,
        ngram_range=(1, 2),
        min_df=3,
    )
    X_text_test, _ = extract_text_features(
        test_struct["catalog_content"],
        vectorizer=tfidf,
    )

    X_train = build_feature_matrix(X_text_train, X_num_train)
    X_test = build_feature_matrix(X_text_test, X_num_test)
    print(f"TF-IDF completed in {time.time() - t0:.1f}s. Feature matrix shape: {X_train.shape}")

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

    # Check GPU availability for CatBoost
    catboost_task_type = "GPU"
    try:
        test_cb = CatBoostRegressor(iterations=1, task_type="GPU", verbose=0)
        test_cb.fit(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 2.0]))
        print("\n[3/5] CatBoost GPU acceleration verified and active on RTX A4000!")
    except Exception as e:
        print(f"\n[3/5] CatBoost GPU fallback to CPU: {e}")
        catboost_task_type = "CPU"

    print("\n[4/5] Starting 5-Fold Cross-Validation...")
    total_start = time.time()

    for fold, (t_idx, v_idx) in enumerate(kf.split(X_train)):
        f_start = time.time()
        print(f"\n{'='*20} Fold {fold + 1}/{n_folds} {'='*20}")
        X_tr, y_tr_log = X_train[t_idx], y_log[t_idx]
        X_va, y_va_log = X_train[v_idx], y_log[v_idx]
        y_va_true = y_train[v_idx]

        # 1. Ridge Regression (fast linear anchor)
        t_ridge = time.time()
        m_ridge = Ridge(alpha=1.5, random_state=42)
        m_ridge.fit(X_tr, y_tr_log)
        val_pred_ridge = np.maximum(np.exp(m_ridge.predict(X_va)), 0.01)
        oof_ridge[v_idx] = val_pred_ridge
        test_ridge += np.maximum(np.exp(m_ridge.predict(X_test)), 0.01) / n_folds
        print(f"  [Ridge]    Fold {fold+1} SMAPE: {smape(y_va_true, val_pred_ridge):.2f}% ({time.time() - t_ridge:.1f}s)")

        # 2. LightGBM (32 threads, Huber Loss)
        t_lgb = time.time()
        m_lgbm = lgb.LGBMRegressor(
            objective="huber",
            metric="mae",
            n_estimators=1800,
            learning_rate=0.06,
            num_leaves=127,
            min_child_samples=40,
            feature_fraction=0.75,
            bagging_fraction=0.8,
            bagging_freq=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=32,
            verbose=-1,
        )
        m_lgbm.fit(
            X_tr, y_tr_log,
            eval_set=[(X_va, y_va_log)],
            callbacks=[
                lgb.early_stopping(80, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )
        val_pred_lgbm = np.maximum(np.exp(m_lgbm.predict(X_va)), 0.01)
        oof_lgbm[v_idx] = val_pred_lgbm
        test_lgbm += np.maximum(np.exp(m_lgbm.predict(X_test)), 0.01) / n_folds
        print(f"  [LightGBM] Fold {fold+1} SMAPE: {smape(y_va_true, val_pred_lgbm):.2f}% ({time.time() - t_lgb:.1f}s, best iter: {m_lgbm.best_iteration_})")

        # 3. CatBoost (RTX A4000 GPU, MAE Loss)
        t_cat = time.time()
        cb_kwargs = {
            "loss_function": "MAE",
            "iterations": 1500,
            "learning_rate": 0.08,
            "depth": 6,
            "random_seed": 42,
            "verbose": 200,
            "early_stopping_rounds": 80,
            "task_type": catboost_task_type,
        }
        if catboost_task_type == "CPU":
            cb_kwargs["thread_count"] = 32

        m_cat = CatBoostRegressor(**cb_kwargs)
        m_cat.fit(X_tr, y_tr_log, eval_set=(X_va, y_va_log), verbose=200)
        val_pred_cat = np.maximum(np.exp(m_cat.predict(X_va)), 0.01)
        oof_cat[v_idx] = val_pred_cat
        test_cat += np.maximum(np.exp(m_cat.predict(X_test)), 0.01) / n_folds
        print(f"  [CatBoost] Fold {fold+1} SMAPE: {smape(y_va_true, val_pred_cat):.2f}% ({time.time() - t_cat:.1f}s)")

        print(f"Fold {fold+1} completed in {time.time() - f_start:.1f}s")

    print("\n" + "=" * 70)
    print(f"Total CV Execution Time: {time.time() - total_start:.1f}s")
    print(f"Overall OOF Ridge SMAPE:    {smape(y_train, oof_ridge):.2f}%")
    print(f"Overall OOF LightGBM SMAPE: {smape(y_train, oof_lgbm):.2f}%")
    print(f"Overall OOF CatBoost SMAPE: {smape(y_train, oof_cat):.2f}%")

    # Optimal Convex Blending
    print("\n[5/5] Optimizing ensemble weights via Nelder-Mead on SMAPE...")
    oof_dict = {"ridge": oof_ridge, "lgbm": oof_lgbm, "cat": oof_cat}
    weights = find_optimal_blend_weights(oof_dict, y_train)
    print(f"Optimal Ensemble Weights: {weights}")

    blended_oof = apply_blend(oof_dict, weights)
    final_smape = smape(y_train, blended_oof)
    print(f"\n=======================================================")
    print(f"  >>> FINAL BLENDED ENSEMBLE OOF SMAPE: {final_smape:.2f}% <<<")
    print(f"=======================================================")

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
    print("\nSample predictions:")
    print(sub_df.head(10))

if __name__ == "__main__":
    main()
