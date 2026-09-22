"""Tests for data loading and validation module.

Verifies train/test CSVs load correctly with expected schema,
dtypes, and distribution properties.
"""
import os

import numpy as np
import pandas as pd
import pytest

# Resolve paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")


class TestLoadTrain:
    """Tests for training data loading."""

    def test_train_loads_successfully(self):
        from src.data import load_train
        df = load_train(DATASET_DIR)
        assert isinstance(df, pd.DataFrame)

    def test_train_has_correct_columns(self):
        from src.data import load_train
        df = load_train(DATASET_DIR)
        expected = {"sample_id", "catalog_content", "image_link", "price"}
        assert set(df.columns) == expected

    def test_train_has_expected_row_count(self):
        from src.data import load_train
        df = load_train(DATASET_DIR)
        # Problem statement says ~75K
        assert len(df) > 70_000
        assert len(df) <= 80_000

    def test_train_no_duplicate_sample_ids(self):
        from src.data import load_train
        df = load_train(DATASET_DIR)
        assert df["sample_id"].is_unique

    def test_train_prices_positive(self):
        from src.data import load_train
        df = load_train(DATASET_DIR)
        assert (df["price"] > 0).all(), "All training prices must be positive"

    def test_train_price_distribution_matches_known_stats(self):
        """Verify against known stats from AGENTS.md: mean ~$23.65, median ~$14.00."""
        from src.data import load_train
        df = load_train(DATASET_DIR)
        assert 20.0 < df["price"].mean() < 30.0, f"Mean {df['price'].mean():.2f} outside expected range"
        assert 10.0 < df["price"].median() < 20.0, f"Median {df['price'].median():.2f} outside expected range"

    def test_train_no_null_catalog_content(self):
        from src.data import load_train
        df = load_train(DATASET_DIR)
        assert df["catalog_content"].notna().all()


class TestLoadTest:
    """Tests for test data loading."""

    def test_test_loads_successfully(self):
        from src.data import load_test
        df = load_test(DATASET_DIR)
        assert isinstance(df, pd.DataFrame)

    def test_test_has_correct_columns(self):
        from src.data import load_test
        df = load_test(DATASET_DIR)
        expected = {"sample_id", "catalog_content", "image_link"}
        assert set(df.columns) == expected

    def test_test_has_expected_row_count(self):
        from src.data import load_test
        df = load_test(DATASET_DIR)
        assert len(df) > 70_000
        assert len(df) <= 80_000

    def test_test_no_duplicate_sample_ids(self):
        from src.data import load_test
        df = load_test(DATASET_DIR)
        assert df["sample_id"].is_unique

    def test_test_has_no_price_column(self):
        from src.data import load_test
        df = load_test(DATASET_DIR)
        assert "price" not in df.columns


class TestValidation:
    """Tests for data validation utilities."""

    def test_validate_submission_format(self):
        from src.data import validate_submission
        sample_out = pd.read_csv(os.path.join(DATASET_DIR, "sample_test_out.csv"))
        # Should not raise
        validate_submission(sample_out)

    def test_validate_submission_rejects_missing_columns(self):
        from src.data import validate_submission
        bad_df = pd.DataFrame({"id": [1], "value": [10.0]})
        with pytest.raises(ValueError, match="column"):
            validate_submission(bad_df)

    def test_validate_submission_rejects_negative_prices(self):
        from src.data import validate_submission
        bad_df = pd.DataFrame({"sample_id": [1], "price": [-5.0]})
        with pytest.raises(ValueError, match="positive"):
            validate_submission(bad_df)

    def test_validate_submission_rejects_nan_prices(self):
        from src.data import validate_submission
        bad_df = pd.DataFrame({"sample_id": [1, 2], "price": [10.0, float("nan")]})
        with pytest.raises(ValueError, match="NaN"):
            validate_submission(bad_df)

    def test_validate_submission_rejects_infinite_prices(self):
        from src.data import validate_submission
        bad_df = pd.DataFrame({"sample_id": [1, 2], "price": [10.0, np.inf]})
        with pytest.raises(ValueError, match="infinite"):
            validate_submission(bad_df)

    def test_validate_submission_rejects_duplicate_sample_ids(self):
        from src.data import validate_submission
        bad_df = pd.DataFrame({"sample_id": [1, 1], "price": [10.0, 20.0]})
        with pytest.raises(ValueError, match="duplicate"):
            validate_submission(bad_df)

    def test_validate_submission_rejects_non_numeric_prices(self):
        from src.data import validate_submission
        bad_df = pd.DataFrame({"sample_id": [1, 2], "price": ["abc", "10.0"]})
        with pytest.raises(ValueError, match="numeric"):
            validate_submission(bad_df)

