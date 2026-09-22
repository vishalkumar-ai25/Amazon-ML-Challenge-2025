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
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

from src.metrics import smape
from src.features import (
    extract_structured_features,
    extract_text_features,
    build_numeric_matrix,
    build_feature_matrix,
    extract_visual_metadata_features,
    build_vision_svd_features,
    extract_advanced_catalog_features,
    compute_iqr_training_mask,
    NUMERIC_COLS,
    VISUAL_METADATA_COLS,
)
from src.knn_features import build_knn_price_features
from src.objectives import lgb_smape_objective, lgb_smape_eval
from src.ensemble import (
    find_optimal_blend_weights,
    apply_blend,
    create_price_stratified_folds,
)
from src.adapter import train_adapter_cv, HAS_TORCH
from src.postprocess import (
    optimize_global_multiplier,
    optimize_clip_floor,
    optimize_power_law_calibration,
    apply_power_law_calibration,
    evaluate_power_law_nested_cv,
    apply_postprocessing,
    analyze_distribution_gap,
    align_test_distribution,
)
from src.stacking import (
    FeatureConditionedStacker,
    evaluate_stacking_cv,
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
    parser.add_argument("--train_mae_adapter", action="store_true", default=True, help="Train a second neural adapter with MAE loss for ensemble diversity")
    parser.add_argument("--iqr_trim_multiplier", type=float, default=3.5, help="IQR multiplier for outlier trimming on training folds (0 to disable)")
    parser.add_argument("--use_cached_adapters", action="store_true", default=False, help="Load cached neural adapter predictions if available")
    parser.add_argument("--subset", type=int, default=None, help="Train and evaluate on a subset of N samples for fast benchmarking (e.g. 10000)")
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

    if args.subset is not None and args.subset < len(train_df):
        print(f"\n>>> SUBSET BENCHMARK MODE: Subsetting train & test datasets to first {args.subset} samples <<<", flush=True)
        train_df = train_df.iloc[:args.subset].reset_index(drop=True)
        if args.subset < len(test_df):
            test_df = test_df.iloc[:args.subset].reset_index(drop=True)

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

    # Advanced Catalog Features (Category, Brand Tier, Material Quality, Price Flags)
    print("Extracting advanced catalog features (categories, brand tiers, material score, price flags)...", flush=True)
    t_adv = time.time()
    train_adv = extract_advanced_catalog_features(train_df)
    brand_tiers = train_adv.attrs.get("brand_tiers")
    test_adv = extract_advanced_catalog_features(test_df, brand_tiers=brand_tiers)

    adv_cols = ["product_category", "brand_price_tier", "material_quality_score"] + [
        c for c in train_adv.columns if c.startswith("flag_") or c.startswith("is_")
    ]
    for c in adv_cols:
        train_struct[c] = train_adv[c]
        test_struct[c] = test_adv[c]
    num_cols += adv_cols
    print(f"Advanced features extraction completed in {time.time() - t_adv:.1f}s ({len(adv_cols)} features added)", flush=True)

    y_train = train_df["price"].values
    y_log = np.log(np.maximum(y_train, 0.01))

    # 2. Check Foundation Embeddings (Text & Vision)
    train_emb_file = os.path.join(emb_dir, f"train_text_{args.model_tag}.npy")
    test_emb_file = os.path.join(emb_dir, f"test_text_{args.model_tag}.npy")
    has_embeddings = os.path.exists(train_emb_file) and os.path.exists(test_emb_file)

    if has_embeddings:
        print(f"\n[2/5] Loading precomputed text foundation embeddings: {train_emb_file}...")
        train_text_emb = np.load(train_emb_file)
        test_text_emb = np.load(test_emb_file)
        if args.subset is not None:
            if args.subset < len(train_text_emb):
                train_text_emb = train_text_emb[:args.subset]
            if args.subset < len(test_text_emb):
                test_text_emb = test_text_emb[:args.subset]
        print(f"Loaded text embeddings shape: train={train_text_emb.shape}, test={test_text_emb.shape}")
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
        if args.subset is not None:
            if args.subset < len(train_text_emb):
                train_text_emb = train_text_emb[:args.subset]
            if args.subset < len(test_text_emb):
                test_text_emb = test_text_emb[:args.subset]

    # 3. Stratified K-Fold Cross Validation
    print("\n[3/5] Setting up Stratified K-Fold based on price quantiles...", flush=True)
    n_folds = 5
    cv_splits = create_price_stratified_folds(y_train, n_folds=n_folds, seed=42)

    # 3b. FAISS k-NN Price Neighbor Features (Leak-free OOF)
    print("\nComputing FAISS k-NN Price Neighbor Features (leak-free OOF, k=10)...", flush=True)
    t_knn = time.time()
    knn_res = build_knn_price_features(train_text_emb, y_train, test_text_emb, cv_splits=cv_splits, k=10)
    knn_cols = list(knn_res["train_features"].keys())
    for col in knn_cols:
        train_struct[col] = knn_res["train_features"][col]
        test_struct[col] = knn_res["test_features"][col]
    num_cols += knn_cols
    print(f"k-NN price features computed in {time.time() - t_knn:.1f}s ({len(knn_cols)} features added)", flush=True)

    # 3c. Empirical Bayes Out-of-Fold Target Encoding (Category & Brand)
    print("\nComputing Empirical Bayes Out-of-Fold Target Encoding (Category & Brand)...", flush=True)
    t_te = time.time()
    train_struct["cat_oof_price"] = np.nan
    train_struct["brand_oof_price"] = np.nan
    test_struct["cat_oof_price"] = 0.0
    test_struct["brand_oof_price"] = 0.0
    global_mean_log = float(y_log.mean())

    for t_idx, v_idx in cv_splits:
        tr_df = train_df.iloc[t_idx]
        cat_means = tr_df.groupby(train_struct.iloc[t_idx]["product_category"])["price"].apply(lambda s: np.log(s.clip(0.01)).mean())
        train_struct.iloc[v_idx, train_struct.columns.get_loc("cat_oof_price")] = train_struct.iloc[v_idx]["product_category"].map(cat_means).fillna(global_mean_log)

        brand_means = tr_df.groupby(train_struct.iloc[t_idx]["brand"])["price"].apply(lambda s: np.log(s.clip(0.01)).mean())
        brand_counts = tr_df.groupby(train_struct.iloc[t_idx]["brand"])["price"].count()
        smooth_brand = (brand_means * brand_counts + global_mean_log * 10) / (brand_counts + 10)
        train_struct.iloc[v_idx, train_struct.columns.get_loc("brand_oof_price")] = train_struct.iloc[v_idx]["brand"].map(smooth_brand).fillna(global_mean_log)

        test_cat_map = test_struct["product_category"].map(cat_means).fillna(global_mean_log)
        test_brand_map = test_struct["brand"].map(smooth_brand).fillna(global_mean_log)
        test_struct["cat_oof_price"] += test_cat_map / n_folds
        test_struct["brand_oof_price"] += test_brand_map / n_folds

    te_cols = ["cat_oof_price", "brand_oof_price"]
    num_cols += te_cols
    print(f"Target encoding completed in {time.time() - t_te:.1f}s ({len(te_cols)} features added)", flush=True)

    X_num_train = build_numeric_matrix(train_struct, columns=num_cols)
    X_num_test = build_numeric_matrix(test_struct, columns=num_cols)

    # Check for vision embeddings (supports SigLIP, DINOv2, or Dual Fusion)
    siglip_candidates = ["siglip_base_patch16_224", "siglip_base", "siglip"]
    dinov2_candidates = ["dinov2_base", "dinov2_base_patch14", "dinov2"]

    siglip_tr = None
    siglip_te = None
    for cand in siglip_candidates:
        tr_p = os.path.join(emb_dir, f"train_vision_{cand}.npy")
        te_p = os.path.join(emb_dir, f"test_vision_{cand}.npy")
        if os.path.exists(tr_p) and os.path.exists(te_p):
            siglip_tr, siglip_te = tr_p, te_p
            break

    dinov2_tr = None
    dinov2_te = None
    for cand in dinov2_candidates:
        tr_p = os.path.join(emb_dir, f"train_vision_{cand}.npy")
        te_p = os.path.join(emb_dir, f"test_vision_{cand}.npy")
        if os.path.exists(tr_p) and os.path.exists(te_p):
            dinov2_tr, dinov2_te = tr_p, te_p
            break

    train_vision_emb = None
    test_vision_emb = None
    train_v_svd = None
    test_v_svd = None

    want_dual = (args.vision_tag is None) or (args.vision_tag.lower() in ["dual", "both", "siglip+dinov2"])
    if siglip_tr and dinov2_tr and want_dual:
        print(f"\n[Dual Vision] Loading and fusing SigLIP + DINOv2...", flush=True)
        print(f"  SigLIP: {siglip_tr}")
        print(f"  DINOv2: {dinov2_tr}")
        tr_siglip = np.load(siglip_tr)
        te_siglip = np.load(siglip_te)
        tr_dinov2 = np.load(dinov2_tr)
        te_dinov2 = np.load(dinov2_te)

        train_vision_emb = np.concatenate([tr_siglip, tr_dinov2], axis=1).astype(np.float32)
        test_vision_emb = np.concatenate([te_siglip, te_dinov2], axis=1).astype(np.float32)
        if args.subset is not None:
            if args.subset < len(train_vision_emb):
                train_vision_emb = train_vision_emb[:args.subset]
            if args.subset < len(test_vision_emb):
                test_vision_emb = test_vision_emb[:args.subset]
        print(f"Dual vision embeddings fused: SigLIP ({tr_siglip.shape[1]}-dim) + DINOv2 ({tr_dinov2.shape[1]}-dim) -> {train_vision_emb.shape[1]}-dim", flush=True)

        print("Extracting 48-dim TruncatedSVD vision features from dual embeddings for GBDT models...", flush=True)
        train_v_svd, test_v_svd = build_vision_svd_features(train_vision_emb, test_vision_emb, n_components=48)
    else:
        # Single vision embedding fallback or explicit vision_tag
        v_tag = args.vision_tag
        if v_tag is None:
            if siglip_tr:
                v_tag = os.path.basename(siglip_tr).replace("train_vision_", "").replace(".npy", "")
            elif dinov2_tr:
                v_tag = os.path.basename(dinov2_tr).replace("train_vision_", "").replace(".npy", "")

        if v_tag:
            tr_v_path = os.path.join(emb_dir, f"train_vision_{v_tag}.npy")
            te_v_path = os.path.join(emb_dir, f"test_vision_{v_tag}.npy")
            if os.path.exists(tr_v_path) and os.path.exists(te_v_path):
                print(f"\n[Single Vision] Loading vision embeddings ({v_tag}): {tr_v_path}...", flush=True)
                train_vision_emb = np.load(tr_v_path).astype(np.float32)
                test_vision_emb = np.load(te_v_path).astype(np.float32)
                if args.subset is not None:
                    if args.subset < len(train_vision_emb):
                        train_vision_emb = train_vision_emb[:args.subset]
                    if args.subset < len(test_vision_emb):
                        test_vision_emb = test_vision_emb[:args.subset]
                print(f"Vision embeddings shape: {train_vision_emb.shape}", flush=True)

                print("Extracting 32-dim TruncatedSVD vision features for GBDT models...", flush=True)
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

    # LightGBM: Strictly sparse CSR (TF-IDF + physical numerics + knn) for 3x faster histogram splits
    X_train_lgbm = build_feature_matrix(X_text_train, X_num_train, vision_svd_features=None)
    X_test_lgbm = build_feature_matrix(X_text_test, X_num_test, vision_svd_features=None)
    print(f"LightGBM Sparse Feature Matrix Shape: {X_train_lgbm.shape}", flush=True)

    # CatBoost GPU: Ingests dense SVD vision components directly in GPU memory
    X_train_cat = build_feature_matrix(X_text_train, X_num_train, vision_svd_features=train_v_svd)
    X_test_cat = build_feature_matrix(X_text_test, X_num_test, vision_svd_features=test_v_svd)
    print(f"CatBoost GPU Feature Matrix Shape:    {X_train_cat.shape}", flush=True)

    # 4. Train Multimodal Neural Pricing Adapter
    # 4. Train or Load Multimodal Neural Pricing Adapter
    adapter_cache_dir = os.path.join(base_dir, "data", "predictions")
    os.makedirs(adapter_cache_dir, exist_ok=True)
    v_tag_name = args.vision_tag or ("dual" if (siglip_tr and dinov2_tr) else "none")
    adapter_cache_file = os.path.join(adapter_cache_dir, f"cached_adapters_{args.model_tag}_{v_tag_name}.npz")

    loaded_adapters = False
    oof_adapter, test_adapter = None, None
    oof_adapter_mae, test_adapter_mae = None, None

    if args.use_cached_adapters and os.path.exists(adapter_cache_file):
        print(f"\n[4/5] Loading precomputed Neural Adapter predictions from cache: {adapter_cache_file}...", flush=True)
        try:
            acache = np.load(adapter_cache_file)
            oof_adapter = acache["oof_adapter"]
            test_adapter = acache["test_adapter"]
            if args.train_mae_adapter and "oof_adapter_mae" in acache:
                oof_adapter_mae = acache["oof_adapter_mae"]
                test_adapter_mae = acache["test_adapter_mae"]
            print(f"Loaded OOF Neural Adapter (SMAPE loss) SMAPE: {smape(y_train, oof_adapter):.2f}%", flush=True)
            if oof_adapter_mae is not None:
                print(f"Loaded OOF Neural Adapter (MAE loss) SMAPE: {smape(y_train, oof_adapter_mae):.2f}%", flush=True)
            loaded_adapters = True
        except Exception as e:
            print(f"Adapter cache read failed ({e}), training fresh...", flush=True)

    if not loaded_adapters:
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
            loss_type="smape",
        )
        print(f"\nOverall OOF Neural Adapter (SMAPE loss) SMAPE: {smape(y_train, oof_adapter):.2f}%", flush=True)

        if args.train_mae_adapter:
            print("\nTraining Second Neural Adapter with MAE Loss for Ensemble Diversity...", flush=True)
            oof_adapter_mae, test_adapter_mae, _ = train_adapter_cv(
                train_text_emb=train_text_emb,
                y_train=y_train,
                test_text_emb=test_text_emb,
                train_vision_emb=train_vision_emb,
                test_vision_emb=test_vision_emb,
                train_tabular=X_num_train.toarray(),
                test_tabular=X_num_test.toarray(),
                cv_splits=cv_splits,
                epochs=min(args.epochs, 25),
                batch_size=args.batch_size,
                lr=args.lr,
                loss_type="mae",
            )
            print(f"Overall OOF Neural Adapter (MAE loss) SMAPE: {smape(y_train, oof_adapter_mae):.2f}%", flush=True)

        save_dict = {
            "oof_adapter": oof_adapter,
            "test_adapter": test_adapter,
        }
        if oof_adapter_mae is not None:
            save_dict["oof_adapter_mae"] = oof_adapter_mae
            save_dict["test_adapter_mae"] = test_adapter_mae
        np.savez_compressed(adapter_cache_file, **save_dict)
        print(f"Saved Neural Adapter predictions to cache: {adapter_cache_file}", flush=True)

    # 5. Train GBDT Models (LightGBM, CatBoost GPU)
    oof_lgbm = np.zeros(len(train_df))
    oof_cat = np.zeros(len(train_df))

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

    print("\nTraining GBDT Models across folds...", flush=True)
    for fold, (t_idx, v_idx) in enumerate(cv_splits):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---", flush=True)
        # Optional IQR outlier trimming on training folds (leaves validation folds 100% untouched)
        if args.iqr_trim_multiplier > 0:
            inlier_mask = compute_iqr_training_mask(y_train[t_idx], multiplier=args.iqr_trim_multiplier)
            t_tr = t_idx[inlier_mask]
        else:
            t_tr = t_idx

        X_tr_lgbm, y_tr_log = X_train_lgbm[t_tr], y_log[t_tr]
        X_va_lgbm, y_va_log = X_train_lgbm[v_idx], y_log[v_idx]
        X_tr_cat = X_train_cat[t_tr]
        X_va_cat = X_train_cat[v_idx]
        y_va_true = y_train[v_idx]

        # LightGBM: Trained directly with analytical SMAPE objective and evaluation
        m_lgbm = lgb.LGBMRegressor(
            objective=lgb_smape_objective,
            n_estimators=1500,
            learning_rate=0.05,
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
            eval_metric=lgb_smape_eval,
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(period=200)],
        )
        val_pred_lgbm = np.maximum(np.exp(m_lgbm.predict(X_va_lgbm)), 0.01)
        oof_lgbm[v_idx] = val_pred_lgbm
        test_lgbm += np.maximum(np.exp(m_lgbm.predict(X_test_lgbm)), 0.01) / n_folds
        print(f"  [LightGBM SMAPE] Fold {fold+1} SMAPE: {smape(y_va_true, val_pred_lgbm):.2f}%", flush=True)

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
    print(f"Overall OOF LightGBM SMAPE:            {smape(y_train, oof_lgbm):.2f}%", flush=True)
    print(f"Overall OOF CatBoost SMAPE:            {smape(y_train, oof_cat):.2f}%", flush=True)
    print(f"Overall OOF Neural Adapter (SMAPE):    {smape(y_train, oof_adapter):.2f}%", flush=True)
    if oof_adapter_mae is not None:
        print(f"Overall OOF Neural Adapter (MAE):      {smape(y_train, oof_adapter_mae):.2f}%", flush=True)

    # Save OOF and test predictions for reproducible post-processing & stacking
    pred_dir = os.path.join(base_dir, "data", "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    save_oof_dict = {
        "adapter_smape": oof_adapter,
        "lgbm": oof_lgbm,
        "cat": oof_cat,
        "y_true": y_train,
    }
    save_test_dict = {
        "adapter_smape": test_adapter,
        "lgbm": test_lgbm,
        "cat": test_cat,
    }
    if oof_adapter_mae is not None:
        save_oof_dict["adapter_mae"] = oof_adapter_mae
        save_test_dict["adapter_mae"] = test_adapter_mae

    np.savez_compressed(os.path.join(pred_dir, "oof_predictions.npz"), **save_oof_dict)
    np.savez_compressed(os.path.join(pred_dir, "test_predictions.npz"), **save_test_dict)
    print(f"Saved OOF and test prediction matrices to: {pred_dir}", flush=True)

    # 6. Out-of-Fold Nelder-Mead Blending across ALL Active Models
    print("\n[5/5] Optimizing Multi-Model Ensemble Weights via Nelder-Mead directly on SMAPE...", flush=True)
    oof_dict = {
        "adapter_smape": oof_adapter,
        "lgbm": oof_lgbm,
        "cat": oof_cat,
    }
    test_dict = {
        "adapter_smape": test_adapter,
        "lgbm": test_lgbm,
        "cat": test_cat,
    }
    if oof_adapter_mae is not None:
        oof_dict["adapter_mae"] = oof_adapter_mae
        test_dict["adapter_mae"] = test_adapter_mae

    weights = find_optimal_blend_weights(oof_dict, y_train)
    print(f"Optimal Ensemble Weights: {weights}", flush=True)

    blended_oof = apply_blend(oof_dict, weights)
    final_smape = smape(y_train, blended_oof)
    print(f"\n=======================================================", flush=True)
    print(f"  >>> MULTI-MODEL BLENDED ENSEMBLE OOF SMAPE: {final_smape:.2f}% <<<", flush=True)
    print(f"=======================================================", flush=True)

    # 7. Continuous Log-Affine Power-Law Decile Calibration (Locked in as Primary Post-Processor)
    print("\n[6/6] Optimizing Continuous Log-Affine Power-Law Decile Calibration...", flush=True)
    opt_floor, _, _ = optimize_clip_floor(y_train, blended_oof)
    opt_a, opt_b, pl_base_sm, pl_cal_sm = optimize_power_law_calibration(y_train, blended_oof, clip_min=opt_floor)
    print(f"Power-Law Calibration: a={opt_a:.4f}, b={opt_b:.4f}, floor={opt_floor:.4f} (SMAPE: {pl_base_sm:.2f}% -> {pl_cal_sm:.2f}%)", flush=True)

    # Evaluate stability via nested CV to prove zero target leakage
    pl_cv_results = evaluate_power_law_nested_cv(y_train, blended_oof, n_splits=5, clip_min=opt_floor)
    pl_cv_smape = pl_cv_results["calibrated_cv_smape"]
    print(f"Power-Law Nested 5-Fold CV SMAPE: {pl_cv_smape:.2f}% (std_a={pl_cv_results['std_a']:.4f}, std_b={pl_cv_results['std_b']:.4f})", flush=True)

    # Generate baseline test predictions
    blended_test = apply_blend(test_dict, weights, clip_min=opt_floor)

    # Lock in Power-Law Calibration as primary post-processing
    best_cal_smape = pl_cal_sm
    blended_oof_calibrated = apply_power_law_calibration(blended_oof, a=opt_a, b=opt_b, clip_min=opt_floor)
    calibrated_test = apply_power_law_calibration(blended_test, a=opt_a, b=opt_b, clip_min=opt_floor)

    print(f"\n=======================================================", flush=True)
    print(f"  >>> FINAL POWER-LAW CALIBRATED ENSEMBLE OOF SMAPE: {best_cal_smape:.2f}% <<<", flush=True)
    print(f"=======================================================", flush=True)


    # 8. Feature-Conditioned Stacking Meta-Learner vs Static Blend Comparison
    print("\n[7/7] Evaluating Feature-Conditioned Stacking Meta-Learner via Nested CV...", flush=True)
    stack_results = evaluate_stacking_cv(oof_dict, y_train, conditioning_df=train_struct, n_splits=5)
    stack_cv_smape = stack_results["stacking_oof_smape"]
    print(f"Stacking Meta-Learner CV SMAPE: {stack_cv_smape:.2f}% (vs Calibrated Blend: {best_cal_smape:.2f}%)", flush=True)

    if stack_cv_smape < best_cal_smape:
        print(f">>> Stacking Meta-Learner wins by {best_cal_smape - stack_cv_smape:.2f}%! Using Stacker for test predictions.", flush=True)
        stacker = FeatureConditionedStacker(alpha=10.0, calibrate_postprocess=True)
        stacker.fit(oof_dict, y_train, conditioning_df=train_struct)
        final_test_preds = stacker.predict(test_dict, conditioning_df=test_struct)
    else:
        print(">>> Static Calibrated Nelder-Mead Blend wins. Using Calibrated Blend for test predictions.", flush=True)
        final_test_preds = calibrated_test

    # 9. Validation-Test Distribution Alignment Check
    print("\n[8/8] Performing Validation-Test Distribution Alignment Check...", flush=True)
    dist_gap = analyze_distribution_gap(y_train, final_test_preds)
    print("Distribution Summary:")
    print(f"  Train: Mean=${dist_gap['train']['mean']:.2f}, Median=${dist_gap['train']['median']:.2f}, Q95=${dist_gap['train']['q95']:.2f}, Std=${dist_gap['train']['std']:.2f}")
    print(f"  Test:  Mean=${dist_gap['test']['mean']:.2f}, Median=${dist_gap['test']['median']:.2f}, Q95=${dist_gap['test']['q95']:.2f}, Std=${dist_gap['test']['std']:.2f}")

    if dist_gap["test"]["std"] < 0.85 * dist_gap["train"]["std"]:
        print("Detected variance compression in test predictions. Applying quantile distribution alignment (weight=0.10)...", flush=True)
        final_test_preds = align_test_distribution(final_test_preds, y_train, blend_weight=0.10)

    # 10. Stratified Error Decile Analysis on Final OOF
    from src.error_analysis import analyze_by_price_decile
    print("\n--- Final OOF SMAPE by Price Decile ---", flush=True)
    decile_summary = analyze_by_price_decile(y_train, blended_oof_calibrated)
    print(decile_summary.to_string(), flush=True)

    sub_df = pd.DataFrame({
        "sample_id": test_df["sample_id"],
        "price": final_test_preds,
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
