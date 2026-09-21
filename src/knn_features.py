import numpy as np
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold

def _search_knn(X_train, X_query, k_eff):
    """Search k nearest neighbors using FAISS or sklearn fallback."""
    if HAS_FAISS:
        d = X_train.shape[1]
        index = faiss.IndexFlatIP(d)
        X_tr = X_train.copy()
        X_qu = X_query.copy()
        faiss.normalize_L2(X_tr)
        faiss.normalize_L2(X_qu)
        index.add(X_tr)
        distances, indices = index.search(X_qu, k_eff)
        return distances, indices
    else:
        # Fallback using sklearn with cosine metric
        nn = NearestNeighbors(n_neighbors=k_eff, metric='cosine', algorithm='brute')
        nn.fit(X_train)
        distances, indices = nn.kneighbors(X_query)
        # Convert cosine distance to cosine similarity (1 - distance) to match IndexFlatIP behavior
        similarities = 1.0 - distances
        return similarities, indices

def build_knn_price_features(train_embeddings, train_prices, test_embeddings=None, cv_splits=None, k=5):
    """
    Build k-NN features based on embeddings using OUT-OF-FOLD indexing to prevent data leakage.
    
    Args:
        train_embeddings (np.ndarray): (N_train, D) array of embeddings for training data.
        train_prices (np.ndarray): (N_train,) array of prices.
        test_embeddings (np.ndarray, optional): (N_test, D) array of embeddings for test data.
        cv_splits (list of tuples, optional): List of (train_idx, val_idx) tuples for OOF generation.
        k (int): Number of nearest neighbors.
        
    Returns:
        dict: A dictionary containing the features for both train and test.
    """
    train_n = train_embeddings.shape[0]
    train_features = {
        'knn_mean_log_price': np.zeros(train_n),
        'knn_median_log_price': np.zeros(train_n),
        'knn_std_log_price': np.zeros(train_n),
        'knn_min_log_price': np.zeros(train_n),
        'knn_max_log_price': np.zeros(train_n),
        'knn_mean_distance': np.zeros(train_n),
    }
    
    log_prices = np.log1p(train_prices)
    
    if cv_splits is not None:
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            X_train = train_embeddings[train_idx]
            y_train = log_prices[train_idx]
            X_val = train_embeddings[val_idx]
            
            k_eff = min(k, len(train_idx))
            distances, indices = _search_knn(X_train, X_val, k_eff)
            
            # Gather prices
            neighbor_prices = y_train[indices]  # (len(val_idx), k_eff)
            
            train_features['knn_mean_log_price'][val_idx] = np.mean(neighbor_prices, axis=1)
            train_features['knn_median_log_price'][val_idx] = np.median(neighbor_prices, axis=1)
            train_features['knn_std_log_price'][val_idx] = np.std(neighbor_prices, axis=1)
            train_features['knn_min_log_price'][val_idx] = np.min(neighbor_prices, axis=1)
            train_features['knn_max_log_price'][val_idx] = np.max(neighbor_prices, axis=1)
            train_features['knn_mean_distance'][val_idx] = np.mean(distances, axis=1)
            
    test_features = None
    if test_embeddings is not None:
        test_n = test_embeddings.shape[0]
        test_features = {
            'knn_mean_log_price': np.zeros(test_n),
            'knn_median_log_price': np.zeros(test_n),
            'knn_std_log_price': np.zeros(test_n),
            'knn_min_log_price': np.zeros(test_n),
            'knn_max_log_price': np.zeros(test_n),
            'knn_mean_distance': np.zeros(test_n),
        }
        
        k_eff = min(k, train_n)
        distances, indices = _search_knn(train_embeddings, test_embeddings, k_eff)
        
        neighbor_prices = log_prices[indices]
        
        test_features['knn_mean_log_price'] = np.mean(neighbor_prices, axis=1)
        test_features['knn_median_log_price'] = np.median(neighbor_prices, axis=1)
        test_features['knn_std_log_price'] = np.std(neighbor_prices, axis=1)
        test_features['knn_min_log_price'] = np.min(neighbor_prices, axis=1)
        test_features['knn_max_log_price'] = np.max(neighbor_prices, axis=1)
        test_features['knn_mean_distance'] = np.mean(distances, axis=1)
        
    return {
        'train_features': train_features,
        'test_features': test_features
    }

if __name__ == '__main__':
    np.random.seed(42)
    N_train = 100
    N_test = 20
    D = 16
    train_emb = np.random.randn(N_train, D).astype(np.float32)
    train_prices = np.random.uniform(10, 1000, size=N_train)
    test_emb = np.random.randn(N_test, D).astype(np.float32)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(kf.split(train_emb))
    
    features = build_knn_price_features(train_emb, train_prices, test_emb, cv_splits, k=5)
    print("Train features mean_log_price:", features['train_features']['knn_mean_log_price'][:5])
    print("Test features mean_log_price:", features['test_features']['knn_mean_log_price'][:5])
