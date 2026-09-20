import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack, csr_matrix
import time

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    y_pred = np.clip(y_pred, 1e-5, None)
    y_true = np.clip(y_true, 1e-5, None)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.abs(y_pred - y_true) / denom) * 100.0

def extract_field(text, field):
    m = re.search(rf'^{field}:\s*(.*)$', str(text), re.MULTILINE)
    return m.group(1).strip() if m else ''

def extract_pack_quantity(text):
    # Search for pack of X, X pack, X per case, etc.
    m = re.search(r'(?:pack of|pk-?|case of|set of|box of)\s*(\d+)', str(text), re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r'(\d+)\s*(?:pack|pk|count|ct|per case|boxes)', str(text), re.IGNORECASE)
    if m:
        return float(m.group(1))
    return 1.0

def main():
    print("Loading datasets...")
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train = pd.read_csv(os.path.join(base_dir, 'dataset', 'train.csv'))
    test = pd.read_csv(os.path.join(base_dir, 'dataset', 'test.csv'))

    print(f"Train samples: {len(train)}, Test samples: {len(test)}")

    print("Extracting structured fields...")
    for df in [train, test]:
        df['item_name'] = df['catalog_content'].apply(lambda x: extract_field(x, 'Item Name'))
        df['value_raw'] = df['catalog_content'].apply(lambda x: extract_field(x, 'Value'))
        df['unit_raw'] = df['catalog_content'].apply(lambda x: extract_field(x, 'Unit').lower())
        
        # numeric value
        df['value_num'] = pd.to_numeric(df['value_raw'], errors='coerce').fillna(1.0)
        df['value_num'] = np.clip(df['value_num'], 0.01, 10000.0)
        df['log_value'] = np.log1p(df['value_num'])
        
        # pack quantity
        df['pack_qty'] = df['catalog_content'].apply(extract_pack_quantity)
        df['log_pack_qty'] = np.log1p(df['pack_qty'])
        
        # unit size
        df['unit_size'] = df['value_num'] / np.maximum(df['pack_qty'], 1.0)
        df['log_unit_size'] = np.log1p(df['unit_size'])

        # text length stats
        df['name_len'] = df['item_name'].str.len()
        df['content_len'] = df['catalog_content'].str.len()

    print("Vectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=3, stop_words='english')
    X_text_train = tfidf.fit_transform(train['catalog_content'])
    X_text_test = tfidf.transform(test['catalog_content'])

    # Unit frequency encoding
    unit_freq = train['unit_raw'].value_counts().to_dict()
    for df in [train, test]:
        df['unit_freq'] = df['unit_raw'].map(unit_freq).fillna(0)

    num_cols = ['log_value', 'log_pack_qty', 'log_unit_size', 'name_len', 'content_len', 'unit_freq']
    X_num_train = csr_matrix(train[num_cols].fillna(0).values)
    X_num_test = csr_matrix(test[num_cols].fillna(0).values)

    X_train = hstack([X_text_train, X_num_train]).tocsr()
    X_test = hstack([X_text_test, X_num_test]).tocsr()

    y_train = train['price'].values
    y_log = np.log(np.maximum(y_train, 0.01))

    print(f"Features shape: {X_train.shape}")

    # 5-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    print("\nStarting 5-Fold Cross Validation with Ridge on log(price)...")
    fold_scores = []
    
    start_time = time.time()
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, y_tr = X_train[train_idx], y_log[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]

        model = Ridge(alpha=1.5, random_state=42)
        model.fit(X_tr, y_tr)

        # Predict in log-space, then transform back to price
        val_pred_log = model.predict(X_va)
        val_pred = np.exp(val_pred_log)
        val_pred = np.maximum(val_pred, 0.1)

        oof_preds[val_idx] = val_pred
        fold_smape = smape(y_va, val_pred)
        fold_scores.append(fold_smape)
        print(f"Fold {fold+1} SMAPE: {fold_smape:.2f}%")

        # Accumulate test predictions
        test_pred_log = model.predict(X_test)
        test_preds += np.exp(test_pred_log) / 5.0

    total_oof_smape = smape(y_train, oof_preds)
    elapsed = time.time() - start_time
    print(f"\n==========================================")
    print(f"OVERALL OUT-OF-FOLD (OOF) SMAPE: {total_oof_smape:.2f}%")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"==========================================")

    # Save initial test_out.csv
    test['price'] = np.maximum(test_preds, 0.1)
    out_df = test[['sample_id', 'price']]
    out_path = os.path.join(base_dir, 'dataset', 'test_out.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Saved baseline test_out.csv to {out_path}! Shape:", out_df.shape)
    print("Sample test predictions:")
    print(out_df.head(10))

if __name__ == "__main__":
    main()
