import numpy as np
import pytest
from sklearn.model_selection import KFold
from src.knn_features import build_knn_price_features

def test_knn_features_shapes():
    N_train = 50
    N_test = 20
    D = 8
    k = 3
    
    train_emb = np.random.randn(N_train, D).astype(np.float32)
    train_prices = np.random.uniform(10, 100, size=N_train)
    test_emb = np.random.randn(N_test, D).astype(np.float32)
    
    kf = KFold(n_splits=3)
    cv_splits = list(kf.split(train_emb))
    
    res = build_knn_price_features(train_emb, train_prices, test_emb, cv_splits, k=k)
    
    train_feats = res['train_features']
    test_feats = res['test_features']
    
    assert train_feats is not None
    assert test_feats is not None
    
    for key in ['knn_mean_log_price', 'knn_median_log_price', 'knn_std_log_price', 'knn_min_log_price', 'knn_max_log_price', 'knn_mean_distance']:
        assert train_feats[key].shape == (N_train,)
        assert test_feats[key].shape == (N_test,)

def test_knn_oof_leakage():
    # Test that OOF means it doesn't just return its own price as the nearest neighbor
    N_train = 20
    D = 4
    k = 1
    
    train_emb = np.random.randn(N_train, D).astype(np.float32)
    train_prices = np.arange(1, N_train + 1) * 10.0
    
    kf = KFold(n_splits=5)
    cv_splits = list(kf.split(train_emb))
    
    res = build_knn_price_features(train_emb, train_prices, None, cv_splits, k=k)
    train_feats = res['train_features']
    
    log_prices = np.log1p(train_prices)
    
    # Since k=1 and it's OOF, the nearest neighbor should NOT be exactly the sample itself
    # unless there is a duplicate, which we haven't added. Thus, the feature shouldn't equal its own log price
    # perfectly for all samples.
    exact_matches = np.isclose(train_feats['knn_mean_log_price'], log_prices)
    assert not exact_matches.all(), "Data leakage detected: OOF predictions exactly match true labels"

def test_knn_k_greater_than_fold_size():
    N_train = 6
    D = 2
    k = 10  # greater than fold size
    
    train_emb = np.random.randn(N_train, D).astype(np.float32)
    train_prices = np.random.uniform(10, 100, size=N_train)
    
    kf = KFold(n_splits=3) # fold sizes are 2
    cv_splits = list(kf.split(train_emb))
    
    res = build_knn_price_features(train_emb, train_prices, cv_splits=cv_splits, k=k)
    train_feats = res['train_features']
    
    assert train_feats['knn_mean_log_price'].shape == (N_train,)
    assert not np.isnan(train_feats['knn_mean_log_price']).any()
