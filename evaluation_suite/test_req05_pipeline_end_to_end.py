"""REQ-10..13: End-to-end training pipeline on a synthetic offline dataset.

Verifies that the config-driven pipeline (src.train) ingests train/test CSVs,
trains with cross-validation, and emits a challenge-compliant submission with
positive finite prices for every test row - without touching the network.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.data import load_test, load_train, validate_submission
from src.ensemble import apply_blend, create_price_stratified_folds, find_optimal_blend_weights
from src.knn_features import build_knn_price_features
from src.metrics import smape
from src.train import load_config, prepare_features, save_submission, train_lgbm_cv, train_ridge_cv

from evaluation_suite.conftest import REPO_ROOT

SMALL_CONFIG = {
    "seed": 42,
    "n_folds": 3,
    "features": {"tfidf_max_features": 300, "tfidf_ngram_range": [1, 2], "tfidf_min_df": 1},
    "target": {"transform": "log", "clip_min": 0.01},
    "ridge": {"alpha": 1.0},
    "lgbm": {
        "objective": "huber", "metric": "mae", "n_estimators": 30, "learning_rate": 0.1,
        "num_leaves": 7, "min_child_samples": 5, "verbose": -1, "n_jobs": 1,
        "random_state": 42, "early_stopping_rounds": 10,
    },
}


@pytest.fixture(scope="module")
def prepared(synthetic_frames):
    train_df, test_df = synthetic_frames
    X_tr, X_te, y, _, _ = prepare_features(train_df, test_df, SMALL_CONFIG)
    return train_df, test_df, X_tr, X_te, y


class TestConfig:
    def test_default_config_loads_and_has_required_keys(self):
        cfg = load_config(os.path.join(REPO_ROOT, "configs", "default.yaml"))
        assert cfg["seed"] == 42
        assert cfg["n_folds"] >= 2
        assert cfg["paths"]["output_file"].endswith("test_out.csv")
        assert cfg["target"]["transform"] == "log"

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            load_config(str(tmp_path / "nope.yaml"))


class TestFeaturePreparation:
    def test_shapes_align(self, prepared):
        train_df, test_df, X_tr, X_te, y = prepared
        assert X_tr.shape[0] == len(train_df) == len(y)
        assert X_te.shape[0] == len(test_df)
        assert X_tr.shape[1] == X_te.shape[1]

    def test_target_is_raw_price(self, prepared):
        train_df, _, _, _, y = prepared
        np.testing.assert_allclose(y, train_df["price"].to_numpy())


class TestCrossValidatedModels:
    def test_ridge_cv_outputs(self, prepared):
        _, test_df, X_tr, X_te, y = prepared
        oof, test_pred, scores = train_ridge_cv(X_tr, y, X_te, SMALL_CONFIG)
        assert oof.shape == y.shape and test_pred.shape == (len(test_df),)
        assert (oof > 0).all() and (test_pred > 0).all()
        assert np.isfinite(oof).all() and np.isfinite(test_pred).all()
        assert len(scores) == SMALL_CONFIG["n_folds"]
        assert all(0 <= s <= 200 for s in scores)

    def test_ridge_cv_is_deterministic(self, prepared):
        _, _, X_tr, X_te, y = prepared
        a = train_ridge_cv(X_tr, y, X_te, SMALL_CONFIG)[1]
        b = train_ridge_cv(X_tr, y, X_te, SMALL_CONFIG)[1]
        np.testing.assert_allclose(a, b)

    def test_ridge_beats_constant_median_baseline(self, prepared):
        _, _, X_tr, X_te, y = prepared
        oof, _, _ = train_ridge_cv(X_tr, y, X_te, SMALL_CONFIG)
        baseline = smape(y, np.full_like(y, np.median(y)))
        assert smape(y, oof) < baseline

    def test_lgbm_cv_outputs(self, prepared):
        pytest.importorskip("lightgbm")
        _, test_df, X_tr, X_te, y = prepared
        oof, test_pred, scores = train_lgbm_cv(X_tr, y, X_te, SMALL_CONFIG)
        assert (oof > 0).all() and (test_pred > 0).all()
        assert np.isfinite(test_pred).all()
        assert len(scores) == SMALL_CONFIG["n_folds"]


class TestEndToEndSubmission:
    def test_pipeline_from_csv_to_submission(self, synthetic_dataset_dir, tmp_path):
        train_df = load_train(synthetic_dataset_dir)
        test_df = load_test(synthetic_dataset_dir)
        X_tr, X_te, y, _, _ = prepare_features(train_df, test_df, SMALL_CONFIG)
        _, test_pred, _ = train_ridge_cv(X_tr, y, X_te, SMALL_CONFIG)
        out = tmp_path / "test_out.csv"
        sub = save_submission(test_df, test_pred, str(out))
        validate_submission(sub, expected_count=len(test_df))
        written = pd.read_csv(out)
        assert list(written.columns) == ["sample_id", "price"]
        assert list(written["sample_id"]) == list(test_df["sample_id"])
        assert written["price"].gt(0).all() and np.isfinite(written["price"]).all()

    def test_pipeline_with_single_test_row(self, synthetic_frames, tmp_path):
        train_df, test_df = synthetic_frames
        one = test_df.iloc[:1].reset_index(drop=True)
        X_tr, X_te, y, _, _ = prepare_features(train_df, one, SMALL_CONFIG)
        _, pred, _ = train_ridge_cv(X_tr, y, X_te, SMALL_CONFIG)
        sub = save_submission(one, pred, str(tmp_path / "one.csv"))
        assert len(sub) == 1 and sub["price"].iloc[0] > 0

    def test_pipeline_survives_empty_and_nan_catalogs_in_test(self, synthetic_frames, tmp_path):
        train_df, test_df = synthetic_frames
        te = test_df.copy()
        te.loc[0, "catalog_content"] = np.nan
        te.loc[1, "catalog_content"] = ""
        te.loc[2, "image_link"] = np.nan
        X_tr, X_te, y, _, _ = prepare_features(train_df, te, SMALL_CONFIG)
        _, pred, _ = train_ridge_cv(X_tr, y, X_te, SMALL_CONFIG)
        sub = save_submission(te, pred, str(tmp_path / "nan.csv"))
        assert len(sub) == len(te) and (sub["price"] > 0).all()


class TestEnsembleAndFolds:
    def test_blend_weights_are_convex(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(1, 100, 300)
        oof = {"a": y * rng.uniform(0.8, 1.2, 300), "b": y * rng.uniform(0.5, 1.5, 300), "c": np.full(300, 20.0)}
        w = find_optimal_blend_weights(oof, y)
        assert set(w) == set(oof)
        assert all(v >= -1e-9 for v in w.values())
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
        assert w["a"] > w["c"]  # best model gets most weight

    def test_single_model_gets_weight_one(self):
        assert find_optimal_blend_weights({"only": np.ones(5)}, np.ones(5)) == {"only": 1.0}

    def test_blend_never_below_clip_floor_and_ignores_unknown_weight(self):
        out = apply_blend({"a": np.array([-5.0, 1.0]), "b": np.array([0.0, 3.0])}, {"a": 0.5, "b": 0.5, "zzz": 9.0})
        assert (out > 0).all()
        assert out[1] == pytest.approx(2.0)

    def test_blend_of_positive_predictions_is_not_worse_than_worst(self):
        rng = np.random.default_rng(3)
        y = rng.uniform(1, 100, 200)
        a = y * rng.uniform(0.9, 1.1, 200)
        b = y * rng.uniform(0.6, 1.4, 200)
        w = find_optimal_blend_weights({"a": a, "b": b}, y)
        assert smape(y, apply_blend({"a": a, "b": b}, w)) <= smape(y, b) + 1e-9

    def test_stratified_folds_are_a_partition(self):
        y = np.exp(np.random.default_rng(0).normal(2.5, 1.0, 500))
        folds = create_price_stratified_folds(y, n_folds=5)
        assert len(folds) == 5
        all_val = np.concatenate([v for _, v in folds])
        assert sorted(all_val.tolist()) == list(range(500))
        for tr, va in folds:
            assert len(np.intersect1d(tr, va)) == 0

    def test_stratified_folds_deterministic_under_seed(self):
        y = np.random.default_rng(0).uniform(1, 100, 100)
        f1 = create_price_stratified_folds(y, n_folds=4, seed=7)
        f2 = create_price_stratified_folds(y, n_folds=4, seed=7)
        for (a, b), (c, d) in zip(f1, f2):
            np.testing.assert_array_equal(a, c)
            np.testing.assert_array_equal(b, d)


class TestKnnFeaturesLeakageSafety:
    def test_oof_knn_features_finite_and_do_not_contain_own_price(self):
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(60, 8)).astype(np.float32)
        prices = rng.uniform(1, 100, 60)
        folds = create_price_stratified_folds(prices, n_folds=3, n_bins=3)
        feats = build_knn_price_features(emb, prices, test_embeddings=emb[:5], cv_splits=folds, k=5)
        train_feats = feats["train_features"]
        for name, arr in train_feats.items():
            assert np.isfinite(arr).all(), name
        # If a row's own price leaked in, min == max == its own log1p(price) would occur for k=1 only;
        # with k=5 the neighbour spread must be > 0 for a random embedding.
        assert (train_feats["knn_price_spread"] > 0).any()

    def test_k_larger_than_train_size_is_capped(self):
        rng = np.random.default_rng(1)
        emb = rng.normal(size=(6, 4)).astype(np.float32)
        prices = rng.uniform(1, 10, 6)
        feats = build_knn_price_features(emb, prices, test_embeddings=emb[:2], cv_splits=None, k=50)
        test_feats = feats["test_features"]
        assert np.isfinite(test_feats["knn_mean_log_price"]).all()
