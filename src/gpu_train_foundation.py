"""End-to-End GPU Training Pipeline with Foundation Model Embeddings & Neural Adapter.

Optimized for:
- NVIDIA RTX A4000 GPU (16 GB VRAM) + Multi-Core CPU
- Precomputed Frozen Foundation Embeddings (Qwen2.5 / BGE-large)
- Multimodal Neural Pricing Adapter with direct Differentiable SMAPE Loss
- Physical Unit Standardization (Total Grams / Milliliters / Pieces)
- Dual GBDTs (LightGBM + CatBoost GPU)
- Out-of-Fold Nelder-Mead Convex Blending targeting SMAPE directly
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import lightgbm as lgb
from catboost import CatBoostRegressor

from src.metrics import smape
from src.features import (
    extract_structured_features,
    extract_text_features,
    build_numeric_matrix,
    build_feature_matrix,
    extract_visual_metadata_features,
    build_vision_svd_features,
    NUMERIC_COLS,
    VISUAL_METADATA_COLS,
)
from src.ensemble import (
    find_optimal_blend_weights,
    apply_blend,
    create_price_stratified_folds,
)
from src.adapter import train_adapter_cv, HAS_TORCH
from src.postprocess import (
    optimize_global_multiplier,
    optimize_clip_floor,
    apply_postprocessing,
)


def main():
    parser = argparse.ArgumentParser(description="GPU Training with Foundation Embeddings & Neural Adapter")
    parser.add_argument("--model_tag", default="bge_large_en_v1.5", help="Text embedding tag (e.g. bge_large_en_v1.5 or qwen2.5_3b)")
    parser.add_argument("--vision_tag", default=None, help="Vision embedding tag (e.g. siglip_base or dinov2_base)")
    parser.add_argument("--image_dir", default="images", help="Folder containing downloaded images")
    parser.add_argument("--embeddings_dir", default="data/embeddings", help="Directory containing pre-extracted embeddings")
    parser.add_argument("--epochs", type=int, default=35, help="Epochs for neural adapter per fold")
    parser.add_argument("--batch_size", type=int, default=256, help="Adapter mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adapter learning rate")
    args = parser.parse_args()

    print("=" * 75)
    print("  Amazon ML Challenge 2025: Multimodal Foundation Model + GBDT Pipeline")
    print("=" * 75)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "dataset", "train.csv")
    test_path = os.path.join(base_dir, "dataset", "test.csv")
    out_path = os.path.join(base_dir, "dataset", "test_out.csv")
    emb_dir = os.path.join(base_dir, args.embeddings_dir)

    print(f"Loading train data: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading test data: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # 1. Physical & Structured Feature Extraction
    print("\n[1/5] Extracting physical unit scales & catalog features...")
    t0 = time.time()
    train_struct = extract_structured_features(train_df)
    test_struct = extract_structured_features(test_df)
    print(f"Structured extraction completed in {time.time() - t0:.1f}s")

    # Unit & Brand frequency encoding
    u_freq = train_struct["unit_normalized"].value_counts().to_dict()
    train_struct["unit_freq"] = train_struct["unit_normalized"].map(u_freq).fillna(0)
    test_struct["unit_freq"] = test_struct["unit_normalized"].map(u_freq).fillna(0)

    b_freq = train_struct["brand"].value_counts().to_dict()
    train_struct["brand_freq"] = train_struct["brand"].map(b_freq).fillna(0)
    test_struct["brand_freq"] = test_struct["brand"].map(b_freq).fillna(0)

    # Check for visual metadata features
    train_img_dir = os.path.join(base_dir, args.image_dir, "train")
    test_img_dir = os.path.join(base_dir, args.image_dir, "test")
    has_images = os.path.exists(train_img_dir) and os.path.exists(test_img_dir)

    if has_images:
        print("Extracting visual metadata properties from downloaded images...")
        t_v0 = time.time()
        train_vmeta = extract_visual_metadata_features(train_df, train_img_dir)
        test_vmeta = extract_visual_metadata_features(test_df, test_img_dir)
        for col in VISUAL_METADATA_COLS:
            train_struct[col] = train_vmeta[col]
            test_struct[col] = test_vmeta[col]
        num_cols = NUMERIC_COLS + ["unit_freq", "brand_freq"] + VISUAL_METADATA_COLS
        print(f"Visual metadata extraction completed in {time.time() - t_v0:.1f}s")
    else:
        num_cols = NUMERIC_COLS + ["unit_freq", "brand_freq"]

    X_num_train = build_numeric_matrix(train_struct, columns=num_cols)
    X_num_test = build_numeric_matrix(test_struct, columns=num_cols)

    # 2. Check Foundation Embeddings (Text & Vision)
    train_emb_file = os.path.join(emb_dir, f"train_text_{args.model_tag}.npy")
    test_emb_file = os.path.join(emb_dir, f"test_text_{args.model_tag}.npy")
    has_embeddings = os.path.exists(train_emb_file) and os.path.exists(test_emb_file)

    if has_embeddings:
        print(f"\n[2/5] Loading precomputed text foundation embeddings: {train_emb_file}...")
        train_text_emb = np.load(train_emb_file)
        test_text_emb = np.load(test_emb_file)
        print(f"Loaded text embeddings shape: {train_text_emb.shape}")
    else:
        print(f"\n[2/5] Text foundation embeddings not found at {train_emb_file}!")
        print("Falling back to TruncatedSVD on TF-IDF for adapter input...")
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), min_df=3)
        X_tfidf_tr = tfidf.fit_transform(train_struct["catalog_content"])
        X_tfidf_te = tfidf.transform(test_struct["catalog_content"])
        svd = TruncatedSVD(n_components=256, random_state=42)
        train_text_emb = svd.fit_transform(X_tfidf_tr).astype(np.float32)
        test_text_emb = svd.transform(X_tfidf_te).astype(np.float32)

    # Check for vision embeddings
    v_tag = args.vision_tag
    if v_tag is None:
        for candidate in ["siglip_base", "dinov2_base", "siglip_base_patch16_224", "siglip"]:
            if os.path.exists(os.path.join(emb_dir, f"train_vision_{candidate}.npy")):
                v_tag = candidate
                break

    train_vision_emb = None
    test_vision_emb = None
    train_v_svd = None
    test_v_svd = None

    if v_tag:
        tr_v_path = os.path.join(emb_dir, f"train_vision_{v_tag}.npy")
        te_v_path = os.path.join(emb_dir, f"test_vision_{v_tag}.npy")
        if os.path.exists(tr_v_path) and os.path.exists(te_v_path):
            print(f"Loading vision embeddings ({v_tag}): {tr_v_path}...")
            train_vision_emb = np.load(tr_v_path)
            test_vision_emb = np.load(te_v_path)
            print(f"Vision embeddings shape: {train_vision_emb.shape}")

            print("Extracting 32-dim SVD vision features for GBDT models...")
            train_v_svd, test_v_svd = build_vision_svd_features(train_vision_emb, test_vision_emb, n_components=32)

    # TF-IDF high-capacity representations for GBDT
    print("Building high-capacity TF-IDF matrix for GBDT models...")
    X_text_train, tfidf = extract_text_features(
        train_struct["catalog_content"],
        max_features=30000,
        ngram_range=(1, 2),
        min_df=3,
    )
    X_text_test, _ = extract_text_features(test_struct["catalog_content"], vectorizer=tfidf)

    # LightGBM: Strictly sparse CSR (TF-IDF + physical numerics) for 3x faster histogram splits
    X_train_lgbm = build_feature_matrix(X_text_train, X_num_train, vision_svd_features=None)
    X_test_lgbm = build_feature_matrix(X_text_test, X_num_test, vision_svd_features=None)
    print(f"LightGBM Sparse Feature Matrix Shape: {X_train_lgbm.shape}", flush=True)

    # CatBoost GPU: Ingests dense SVD vision components directly in GPU memory
    X_train_cat = build_feature_matrix(X_text_train, X_num_train, vision_svd_features=train_v_svd)
    X_test_cat = build_feature_matrix(X_text_test, X_num_test, vision_svd_features=test_v_svd)
    print(f"CatBoost GPU Feature Matrix Shape:    {X_train_cat.shape}", flush=True)

    y_train = train_df["price"].values
    y_log = np.log(np.maximum(y_train, 0.01))

    # 3. Stratified K-Fold Cross Validation
    print("\n[3/5] Setting up Stratified K-Fold based on price quantiles...", flush=True)
    n_folds = 5
    cv_splits = create_price_stratified_folds(y_train, n_folds=n_folds, seed=42)

    # 4. Train Multimodal Neural Pricing Adapter
    print("\n[4/5] Training Multimodal Pricing Adapter with Differentiable SMAPE Loss...", flush=True)
    oof_adapter, test_adapter, adapter_scores = train_adapter_cv(
        train_text_emb=train_text_emb,
        y_train=y_train,
        test_text_emb=test_text_emb,
        train_vision_emb=train_vision_emb,
        test_vision_emb=test_vision_emb,
        train_tabular=X_num_train.toarray(),
        test_tabular=X_num_test.toarray(),
        cv_splits=cv_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(f"\nOverall OOF Neural Adapter SMAPE: {smape(y_train, oof_adapter):.2f}%", flush=True)

    # 5. Train LightGBM & CatBoost on GBDT Features
    oof_ridge = np.zeros(len(train_df))
    oof_lgbm = np.zeros(len(train_df))
    oof_cat = np.zeros(len(train_df))

    test_ridge = np.zeros(len(test_df))
    test_lgbm = np.zeros(len(test_df))
    test_cat = np.zeros(len(test_df))

    catboost_task_type = "GPU"
    try:
        test_cb = CatBoostRegressor(iterations=1, task_type="GPU", verbose=0)
        test_cb.fit(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 2.0]))
        print("CatBoost GPU acceleration confirmed active!", flush=True)
    except Exception:
        print("CatBoost GPU fallback to multi-threaded CPU.", flush=True)
        catboost_task_type = "CPU"

    print("\nTraining GBDT Models (LightGBM & CatBoost) across folds...", flush=True)
    for fold, (t_idx, v_idx) in enumerate(cv_splits):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---", flush=True)
        X_tr_lgbm, y_tr_log = X_train_lgbm[t_idx], y_log[t_idx]
        X_va_lgbm, y_va_log = X_train_lgbm[v_idx], y_log[v_idx]
        X_tr_cat = X_train_cat[t_idx]
        X_va_cat = X_train_cat[v_idx]
        y_va_true = y_train[v_idx]

        # Ridge linear anchor (on sparse features)
        m_ridge = Ridge(alpha=1.5, random_state=42)
        m_ridge.fit(X_tr_lgbm, y_tr_log)
        val_pred_ridge = np.maximum(np.exp(m_ridge.predict(X_va_lgbm)), 0.01)
        oof_ridge[v_idx] = val_pred_ridge
        test_ridge += np.maximum(np.exp(m_ridge.predict(X_test_lgbm)), 0.01) / n_folds

        # LightGBM (Huber Loss, on purely sparse features for 3x speedup)
        m_lgbm = lgb.LGBMRegressor(
            objective="huber",
            metric="mae",
            n_estimators=1500,
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
            X_tr_lgbm, y_tr_log,
            eval_set=[(X_va_lgbm, y_va_log)],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(period=200)],
        )
        val_pred_lgbm = np.maximum(np.exp(m_lgbm.predict(X_va_lgbm)), 0.01)
        oof_lgbm[v_idx] = val_pred_lgbm
        test_lgbm += np.maximum(np.exp(m_lgbm.predict(X_test_lgbm)), 0.01) / n_folds
        print(f"  [LightGBM] Fold {fold+1} SMAPE: {smape(y_va_true, val_pred_lgbm):.2f}%", flush=True)

        # CatBoost (MAE Loss, on dense SVD features via GPU)
        cb_kwargs = {
            "loss_function": "MAE",
            "iterations": 1200,
            "learning_rate": 0.08,
            "depth": 6,
            "random_seed": 42,
            "verbose": 200,
            "early_stopping_rounds": 80,
            "task_type": catboost_task_type,
        }
        if catboost_task_type == "CPU":
            cb_kwargs["thread_count"] = -1

        m_cat = CatBoostRegressor(**cb_kwargs)
        m_cat.fit(X_tr_cat, y_tr_log, eval_set=(X_va_cat, y_va_log), verbose=200)
        val_pred_cat = np.maximum(np.exp(m_cat.predict(X_va_cat)), 0.01)
        oof_cat[v_idx] = val_pred_cat
        test_cat += np.maximum(np.exp(m_cat.predict(X_test_cat)), 0.01) / n_folds
        print(f"  [CatBoost] Fold {fold+1} SMAPE: {smape(y_va_true, val_pred_cat):.2f}%", flush=True)

    print("\n" + "=" * 75, flush=True)
    print(f"Overall OOF Ridge SMAPE:         {smape(y_train, oof_ridge):.2f}%", flush=True)
    print(f"Overall OOF LightGBM SMAPE:      {smape(y_train, oof_lgbm):.2f}%", flush=True)
    print(f"Overall OOF CatBoost SMAPE:      {smape(y_train, oof_cat):.2f}%", flush=True)
    print(f"Overall OOF Neural Adapter SMAPE:{smape(y_train, oof_adapter):.2f}%", flush=True)

    # Save OOF and test predictions for reproducible post-processing & stacking
    pred_dir = os.path.join(base_dir, "data", "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(pred_dir, "oof_predictions.npz"),
        adapter=oof_adapter,
        lgbm=oof_lgbm,
        cat=oof_cat,
        ridge=oof_ridge,
        y_true=y_train,
    )
    np.savez_compressed(
        os.path.join(pred_dir, "test_predictions.npz"),
        adapter=test_adapter,
        lgbm=test_lgbm,
        cat=test_cat,
        ridge=test_ridge,
    )
    print(f"Saved OOF and test prediction matrices to: {pred_dir}", flush=True)

    # 6. Out-of-Fold Nelder-Mead Blending
    print("\n[5/5] Optimizing 4-Way Ensemble Weights via Nelder-Mead directly on SMAPE...", flush=True)
    oof_dict = {
        "adapter": oof_adapter,
        "lgbm": oof_lgbm,
        "cat": oof_cat,
        "ridge": oof_ridge,
    }
    weights = find_optimal_blend_weights(oof_dict, y_train)
    print(f"Optimal Ensemble Weights: {weights}", flush=True)

    blended_oof = apply_blend(oof_dict, weights)
    final_smape = smape(y_train, blended_oof)
    print(f"\n=======================================================", flush=True)
    print(f"  >>> 4-WAY BLENDED ENSEMBLE OOF SMAPE: {final_smape:.2f}% <<<", flush=True)
    print(f"=======================================================", flush=True)

    # 7. Post-Processing Multiplier & Floor Calibration
    print("\n[6/6] Optimizing Post-Processing Multiplier & Floor directly on SMAPE...", flush=True)
    opt_alpha, base_sm, mult_sm = optimize_global_multiplier(y_train, blended_oof)
    opt_floor, _, cal_smape = optimize_clip_floor(y_train, blended_oof * opt_alpha)
    print(f"Optimal Global Multiplier: alpha={opt_alpha:.4f} (SMAPE: {base_sm:.2f}% -> {mult_sm:.2f}%)", flush=True)
    print(f"Optimal Lower Floor:       clip_min={opt_floor:.4f} (SMAPE: {mult_sm:.2f}% -> {cal_smape:.2f}%)", flush=True)
    print(f"\n=======================================================", flush=True)
    print(f"  >>> FINAL CALIBRATED ENSEMBLE OOF SMAPE: {cal_smape:.2f}% <<<", flush=True)
    print(f"=======================================================", flush=True)

    # Generate, calibrate, and validate test predictions
    test_dict = {
        "adapter": test_adapter,
        "lgbm": test_lgbm,
        "cat": test_cat,
        "ridge": test_ridge,
    }
    blended_test = apply_blend(test_dict, weights, clip_min=opt_floor)
    calibrated_test = apply_postprocessing(blended_test, multiplier=opt_alpha, clip_min=opt_floor)

    sub_df = pd.DataFrame({
        "sample_id": test_df["sample_id"],
        "price": calibrated_test,
    })

    assert len(sub_df) == len(test_df), "Row count mismatch"
    assert list(sub_df.columns) == ["sample_id", "price"]
    assert (sub_df["price"] > 0).all(), "Non-positive price found"
    assert sub_df["price"].notna().all(), "NaN price found"

    sub_df.to_csv(out_path, index=False)
    print(f"\nSuccessfully generated and validated {out_path} ({len(sub_df)} rows)!", flush=True)
    print("\nTop 10 Predictions:", flush=True)
    print(sub_df.head(10), flush=True)



if __name__ == "__main__":
    main()
