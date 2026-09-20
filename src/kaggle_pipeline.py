"""Self-contained Kaggle Pipeline for Amazon ML Challenge 2025.

Optimized for Kaggle Kernels (30 GB CPU RAM / T4/P100 GPU).
Executes:
1. Resilient dataset discovery (Kaggle inputs vs local directory)
2. High-capacity text vectorization (25,000 n-grams) + structured catalog parsing
3. Dual Gradient Boosting (LightGBM + CatBoost) + Ridge Regression with 5-fold CV
4. Optimal convex blending directly minimizing competition SMAPE
5. Generates and strictly validates final test_out.csv
"""
import os
import re
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor

# ---------------------------------------------------------------------------
# Evaluation Metric (Competition Exact SMAPE)
# ---------------------------------------------------------------------------

def smape(y_true, y_pred, clip_min=1e-5):
    y_true = np.clip(np.asarray(y_true, dtype=np.float64), clip_min, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), clip_min, None)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(numerator / denominator) * 100.0)

# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

_UNIT_MAP = {
    "fl oz": "fl_oz", "fl. oz": "fl_oz", "fluid ounce": "fl_oz", "fl_oz": "fl_oz",
    "ounce": "oz", "ounces": "oz", "oz": "oz",
    "pound": "lb", "pounds": "lb", "lb": "lb", "lbs": "lb",
    "gram": "g", "grams": "g", "g": "g",
    "kilogram": "kg", "kg": "kg",
    "milliliter": "ml", "ml": "ml", "liter": "l", "l": "l",
    "count": "count", "each": "count", "piece": "count", "unit": "count",
}

_UNIT_CATEGORIES = {
    "fl_oz": "volume", "ml": "volume", "l": "volume",
    "oz": "weight", "lb": "weight", "g": "weight", "kg": "weight",
    "count": "count",
}

_ARTICLES = {"la", "le", "el", "the", "de", "del", "san", "santa", "st", "st.", "dr", "dr.", "mr", "mr.", "mrs", "mrs."}
_PACK_PATTERNS = [
    re.compile(r"(?:pack|case|box|set)\s+of\s+(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)\s*(?:pack|pk|count|ct|per\s+case|boxes?)", re.IGNORECASE),
]

def extract_field(text, field):
    if not isinstance(text, str):
        return ""
    m = re.search(rf"^{re.escape(field)}:\s*(.*)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""

def normalize_unit(raw):
    lower = str(raw).strip().lower()
    return _UNIT_MAP.get(lower, lower)

def get_unit_category(unit):
    return _UNIT_CATEGORIES.get(str(unit).lower().strip(), "other")

def extract_brand(item_name):
    if not isinstance(item_name, str) or not item_name.strip():
        return ""
    clean = re.sub(r"^[^\w\s]+", "", item_name.strip())
    first_chunk = re.split(r"\s+[-–—|/]\s+|,", clean)[0].strip()
    tokens = first_chunk.split()
    if not tokens:
        return ""
    if len(tokens) >= 2 and tokens[0].lower() in _ARTICLES:
        return f"{tokens[0]} {tokens[1]}"
    if len(tokens) >= 2 and (tokens[1].startswith("&") or tokens[1].startswith("+") or "+" in tokens[1] or "&" in tokens[1]):
        if len(tokens) >= 3 and tokens[1] in ["&", "+"]:
            return f"{tokens[0]} {tokens[1]} {tokens[2]}"
        return f"{tokens[0]} {tokens[1]}"
    return tokens[0]

def extract_pack_quantity(text):
    text_str = str(text) if not isinstance(text, str) else text
    for p in _PACK_PATTERNS:
        m = p.search(text_str)
        if m:
            return float(m.group(1))
    return 1.0

def count_bullets(text):
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"^Bullet Point \d+:", text, re.MULTILINE))

def extract_all_structured(df):
    res = df.copy()
    res["item_name"] = res["catalog_content"].apply(lambda x: extract_field(x, "Item Name"))
    res["value_raw"] = res["catalog_content"].apply(lambda x: extract_field(x, "Value"))
    res["unit_raw"] = res["catalog_content"].apply(lambda x: extract_field(x, "Unit"))
    res["desc_raw"] = res["catalog_content"].apply(lambda x: extract_field(x, "Product Description"))

    res["unit_norm"] = res["unit_raw"].apply(normalize_unit)
    res["unit_cat"] = res["unit_norm"].apply(get_unit_category)
    res["brand"] = res["item_name"].apply(extract_brand)

    res["val_num"] = pd.to_numeric(res["value_raw"], errors="coerce").fillna(1.0).clip(0.01, 10000.0)
    res["pack_qty"] = res["catalog_content"].apply(extract_pack_quantity)

    res["log_val"] = np.log1p(res["val_num"])
    res["log_pack"] = np.log1p(res["pack_qty"])
    res["unit_size"] = res["val_num"] / np.maximum(res["pack_qty"], 1.0)
    res["log_unit_size"] = np.log1p(res["unit_size"])
    res["total_qty"] = res["val_num"] * res["pack_qty"]
    res["log_total_qty"] = np.log1p(res["total_qty"])

    res["num_bullets"] = res["catalog_content"].apply(count_bullets)
    res["name_len"] = res["item_name"].str.len().fillna(0)
    res["content_len"] = res["catalog_content"].str.len().fillna(0)
    res["desc_len"] = res["desc_raw"].str.len().fillna(0)

    lower = res["catalog_content"].str.lower()
    res["is_multipack"] = lower.str.contains(r"\b(?:pack|set|case|box|bundle)\b", regex=True).astype(float)
    res["is_premium"] = lower.str.contains(r"\b(?:organic|gourmet|pro|premium|luxury|collection)\b", regex=True).astype(float)
    res["is_value_size"] = lower.str.contains(r"\b(?:refill|travel|mini|sample)\b", regex=True).astype(float)

    return res

# ---------------------------------------------------------------------------
# Ensembling & Blending
# ---------------------------------------------------------------------------

def optimize_blend_weights(oof_dict, y_true):
    names = list(oof_dict.keys())
    if len(names) == 1:
        return {names[0]: 1.0}
    P = np.column_stack([oof_dict[n] for n in names])

    def obj(w):
        w_norm = np.maximum(w, 0)
        s = np.sum(w_norm)
        w_norm = w_norm / s if s > 0 else np.ones(len(names)) / len(names)
        return smape(y_true, np.dot(P, w_norm))

    res = minimize(obj, np.ones(len(names)) / len(names), method="Nelder-Mead", bounds=[(0.0, 1.0)] * len(names))
    w_opt = np.maximum(res.x, 0)
    w_opt = w_opt / np.sum(w_opt) if np.sum(w_opt) > 0 else np.ones(len(names)) / len(names)
    return {name: float(w_opt[i]) for i, name in enumerate(names)}

# ---------------------------------------------------------------------------
# Main Kaggle Pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(" Amazon ML Challenge 2025: Production Model Pipeline")
    print("=" * 60)

    # Locate datasets
    possible_paths = [
        ("dataset/train.csv", "dataset/test.csv", "dataset/test_out.csv"),
        ("/kaggle/input/amazon-ml/train.csv", "/kaggle/input/amazon-ml/test.csv", "/kaggle/working/test_out.csv"),
        ("/kaggle/input/amazon-ml-challenge-2025/train.csv", "/kaggle/input/amazon-ml-challenge-2025/test.csv", "/kaggle/working/test_out.csv"),
    ]

    train_path, test_path, out_path = None, None, None
    for tr, te, out in possible_paths:
        if os.path.exists(tr) and os.path.exists(te):
            train_path, test_path, out_path = tr, te, out
            break

    if not train_path:
        raise FileNotFoundError("Could not locate train.csv and test.csv in standard locations!")

    print(f"Loading train data: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Loading test data: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Extract structured features
    print("\nExtracting structured features...")
    train_struct = extract_all_structured(train_df)
    test_struct = extract_all_structured(test_df)

    # Unit frequency
    u_freq = train_struct["unit_norm"].value_counts().to_dict()
    train_struct["unit_freq"] = train_struct["unit_norm"].map(u_freq).fillna(0)
    test_struct["unit_freq"] = test_struct["unit_norm"].map(u_freq).fillna(0)

    # Brand frequency
    b_freq = train_struct["brand"].value_counts().to_dict()
    train_struct["brand_freq"] = train_struct["brand"].map(b_freq).fillna(0)
    test_struct["brand_freq"] = test_struct["brand"].map(b_freq).fillna(0)

    num_cols = [
        "log_val", "log_pack", "log_unit_size", "log_total_qty",
        "name_len", "content_len", "desc_len", "num_bullets",
        "is_multipack", "is_premium", "is_value_size",
        "unit_freq", "brand_freq"
    ]

    X_num_train = csr_matrix(train_struct[num_cols].fillna(0).values.astype(np.float32))
    X_num_test = csr_matrix(test_struct[num_cols].fillna(0).values.astype(np.float32))

    # TF-IDF text features
    print("Fitting TF-IDF Vectorizer on full catalog content...")
    tfidf = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), min_df=3, stop_words="english", dtype=np.float32)
    X_text_train = tfidf.fit_transform(train_struct["catalog_content"])
    X_text_test = tfidf.transform(test_struct["catalog_content"])

    X_train = hstack([X_text_train, X_num_train]).tocsr()
    X_test = hstack([X_text_test, X_num_test]).tocsr()
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

        # 2. LightGBM (Huber loss)
        m_lgbm = lgb.LGBMRegressor(
            objective="huber",
            metric="mae",
            n_estimators=1500,
            learning_rate=0.06,
            num_leaves=127,
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
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        val_pred_lgbm = np.maximum(np.exp(m_lgbm.predict(X_va)), 0.01)
        oof_lgbm[v_idx] = val_pred_lgbm
        test_lgbm += np.maximum(np.exp(m_lgbm.predict(X_test)), 0.01) / n_folds
        print(f"  Fold {fold+1} LightGBM SMAPE: {smape(y_va_true, val_pred_lgbm):.2f}%")

        # 3. CatBoost (MAE loss)
        m_cat = CatBoostRegressor(
            loss_function="MAE",
            iterations=1200,
            learning_rate=0.08,
            depth=6,
            random_seed=42,
            thread_count=-1,
            verbose=0,
        )
        m_cat.fit(
            X_tr, y_tr_log,
            eval_set=(X_va, y_va_log),
            early_stopping_rounds=80,
            verbose=False,
        )
        val_pred_cat = np.maximum(np.exp(m_cat.predict(X_va)), 0.01)
        oof_cat[v_idx] = val_pred_cat
        test_cat += np.maximum(np.exp(m_cat.predict(X_test)), 0.01) / n_folds
        print(f"  Fold {fold+1} CatBoost SMAPE: {smape(y_va_true, val_pred_cat):.2f}%")

    print("\n" + "=" * 60)
    print(f"Overall OOF Ridge SMAPE:    {smape(y_train, oof_ridge):.2f}%")
    print(f"Overall OOF LightGBM SMAPE: {smape(y_train, oof_lgbm):.2f}%")
    print(f"Overall OOF CatBoost SMAPE: {smape(y_train, oof_cat):.2f}%")

    # Optimal Convex Blending
    oof_dict = {"ridge": oof_ridge, "lgbm": oof_lgbm, "cat": oof_cat}
    weights = optimize_blend_weights(oof_dict, y_train)
    print(f"\nOptimal Ensemble Weights: {weights}")

    blended_oof = weights["ridge"] * oof_ridge + weights["lgbm"] * oof_lgbm + weights["cat"] * oof_cat
    final_smape = smape(y_train, blended_oof)
    print(f"FINAL ENSEMBLE OOF SMAPE:   {final_smape:.2f}%")
    print("=" * 60)

    # Generate Final Predictions
    blended_test = weights["ridge"] * test_ridge + weights["lgbm"] * test_lgbm + weights["cat"] * test_cat
    blended_test = np.maximum(blended_test, 0.05)

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
