"""Integration tests for the training pipeline and model workflows.

Tests verify end-to-end compatibility on small synthetic subsets:
- Config loading
- Feature extraction & matrix combining
- Ridge regression CV execution and positive predictions
- LightGBM CV execution and early stopping
- Submission formatting and file saving
"""
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.train import (
    load_config,
    prepare_features,
    train_ridge_cv,
    train_lgbm_cv,
    save_submission,
)
from src.metrics import smape


@pytest.fixture
def synthetic_data():
    """Create small synthetic train and test DataFrames."""
    rng = np.random.RandomState(42)
    n_train = 50
    n_test = 20

    train_data = {
        "sample_id": list(range(1, n_train + 1)),
        "catalog_content": [
            f"Item Name: Product {i}\nValue: {10.0 + i}\nUnit: Fl Oz\nBullet Point 1: Quality item {i}"
            for i in range(n_train)
        ],
        "image_link": [f"https://images.example.com/{i}.jpg" for i in range(n_train)],
        "price": rng.uniform(5.0, 100.0, size=n_train),
    }

    test_data = {
        "sample_id": list(range(n_train + 1, n_train + n_test + 1)),
        "catalog_content": [
            f"Item Name: Test Product {j}\nValue: {5.0 + j}\nUnit: Fl Oz\nBullet Point 1: Test spec {j}"
            for j in range(n_test)
        ],
        "image_link": [f"https://images.example.com/test_{j}.jpg" for j in range(n_test)],
    }

    return pd.DataFrame(train_data), pd.DataFrame(test_data)


@pytest.fixture
def mock_config():
    return {
        "seed": 42,
        "n_folds": 2,
        "features": {
            "tfidf_max_features": 50,
            "tfidf_ngram_range": [1, 2],
            "tfidf_min_df": 1,
        },
        "target": {
            "transform": "log",
            "clip_min": 0.01,
        },
        "lgbm": {
            "objective": "huber",
            "metric": "mae",
            "boosting_type": "gbdt",
            "n_estimators": 5,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 2,
            "verbose": -1,
            "n_jobs": 1,
            "random_state": 42,
            "early_stopping_rounds": 5,
        },
        "paths": {
            "dataset_dir": "dataset",
            "model_dir": "models",
            "output_file": "dataset/test_out.csv",
        },
    }


def test_load_config():
    """Verify loading YAML config from disk."""
    config_path = "configs/default.yaml"
    if os.path.exists(config_path):
        cfg = load_config(config_path)
        assert isinstance(cfg, dict)
        assert "lgbm" in cfg
        assert "features" in cfg


def test_prepare_features(synthetic_data, mock_config):
    """Test feature preparation pipeline produces correct shapes."""
    train_df, test_df = synthetic_data
    X_train, X_test, y_train, train_struct, test_struct = prepare_features(
        train_df, test_df, mock_config
    )

    assert X_train.shape[0] == len(train_df)
    assert X_test.shape[0] == len(test_df)
    assert X_train.shape[1] == X_test.shape[1]
    assert len(y_train) == len(train_df)
    assert "unit_freq" in train_struct.columns
    assert "unit_freq" in test_struct.columns


def test_train_ridge_cv(synthetic_data, mock_config):
    """Test Ridge regression CV execution and predictions."""
    train_df, test_df = synthetic_data
    X_train, X_test, y_train, _, _ = prepare_features(train_df, test_df, mock_config)

    oof_preds, test_preds, fold_scores = train_ridge_cv(
        X_train, y_train, X_test, mock_config
    )

    assert len(oof_preds) == len(train_df)
    assert len(test_preds) == len(test_df)
    assert len(fold_scores) == mock_config["n_folds"]
    assert (oof_preds > 0).all()
    assert (test_preds > 0).all()

    total_smape = smape(y_train, oof_preds)
    assert 0 <= total_smape <= 200


def test_train_lgbm_cv(synthetic_data, mock_config):
    """Test LightGBM regression CV execution and predictions."""
    train_df, test_df = synthetic_data
    X_train, X_test, y_train, _, _ = prepare_features(train_df, test_df, mock_config)

    oof_preds, test_preds, fold_scores = train_lgbm_cv(
        X_train, y_train, X_test, mock_config
    )

    assert len(oof_preds) == len(train_df)
    assert len(test_preds) == len(test_df)
    assert len(fold_scores) == mock_config["n_folds"]
    assert (oof_preds > 0).all()
    assert (test_preds > 0).all()

    total_smape = smape(y_train, oof_preds)
    assert 0 <= total_smape <= 200


def test_save_submission(synthetic_data):
    """Test saving and validation of output predictions."""
    _, test_df = synthetic_data
    dummy_preds = np.random.uniform(5.0, 50.0, size=len(test_df))

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sub_df = save_submission(test_df, dummy_preds, tmp_path)
        assert os.path.exists(tmp_path)
        assert len(sub_df) == len(test_df)
        assert list(sub_df.columns) == ["sample_id", "price"]
        assert (sub_df["price"] > 0).all()

        loaded_df = pd.read_csv(tmp_path)
        assert len(loaded_df) == len(test_df)
        assert list(loaded_df.columns) == ["sample_id", "price"]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
