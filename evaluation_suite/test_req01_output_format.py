"""REQ-01..04: Output file format, positivity, completeness and ID alignment.

Problem statement:
  * Output CSV has exactly 2 columns: sample_id, price.
  * Formatting must match dataset/sample_test_out.csv exactly.
  * Predicted prices must be positive floats.
  * One prediction for every test sample_id (row count must equal test.csv).
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pandas as pd
import pytest

from evaluation_suite.conftest import (
    CHALLENGE_TEST_ROWS,
    DATASET_DIR,
    REQUIRED_OUTPUT_COLS,
)

SAMPLE_TEST = os.path.join(DATASET_DIR, "sample_test.csv")
SAMPLE_OUT = os.path.join(DATASET_DIR, "sample_test_out.csv")
SUBMISSION = os.path.join(DATASET_DIR, "test_out.csv")


class TestSampleFilesAreChallengeCompliant:
    def test_sample_files_exist(self):
        assert os.path.exists(SAMPLE_TEST)
        assert os.path.exists(SAMPLE_OUT)

    def test_sample_out_has_exact_columns(self):
        df = pd.read_csv(SAMPLE_OUT)
        assert list(df.columns) == REQUIRED_OUTPUT_COLS

    def test_sample_out_ids_match_sample_test(self):
        st = pd.read_csv(SAMPLE_TEST)
        so = pd.read_csv(SAMPLE_OUT)
        assert len(st) == len(so)
        assert list(st["sample_id"]) == list(so["sample_id"])


class TestCommittedSubmissionFormat:
    """dataset/test_out.csv is the artifact the team submits; validate it statically."""

    def test_header_matches_sample_exactly(self):
        with open(SAMPLE_OUT, newline="") as f:
            sample_header = next(csv.reader(f))
        with open(SUBMISSION, newline="") as f:
            sub_header = next(csv.reader(f))
        assert sub_header == sample_header == REQUIRED_OUTPUT_COLS

    def test_only_two_columns_and_no_index_column(self, committed_submission):
        assert list(committed_submission.columns) == REQUIRED_OUTPUT_COLS
        assert committed_submission.shape[1] == 2

    def test_sample_id_is_integer_and_unique(self, committed_submission):
        assert pd.api.types.is_integer_dtype(committed_submission["sample_id"])
        assert committed_submission["sample_id"].is_unique
        assert committed_submission["sample_id"].notna().all()

    def test_price_is_float_dtype(self, committed_submission):
        assert pd.api.types.is_float_dtype(committed_submission["price"])

    def test_prices_are_strictly_positive_and_finite(self, committed_submission):
        p = committed_submission["price"].to_numpy(dtype=np.float64)
        assert not np.isnan(p).any(), "NaN prices present"
        assert np.isfinite(p).all(), "Inf prices present"
        assert (p > 0).all(), f"min price {p.min()} is not > 0"

    def test_row_count_matches_challenge_test_size(self, committed_submission):
        assert len(committed_submission) == CHALLENGE_TEST_ROWS

    def test_price_range_is_plausible_for_ecommerce(self, committed_submission):
        # Training price range documented in AGENTS.md: $0.13 .. $2796.
        p = committed_submission["price"]
        assert p.min() >= 0.01
        assert p.max() <= 10_000

    def test_predictions_are_not_constant_or_random_baseline(self, committed_submission):
        # A trained model must produce >1 distinct value; the sample_code random baseline
        # yields a uniform [5,500] distribution with median ~252 - guard against that.
        p = committed_submission["price"]
        assert p.nunique() > 1000
        assert p.median() < 100, "median looks like the random uniform(5,500) dummy baseline"

    def test_ids_exactly_match_test_csv(self, committed_submission, full_test_csv):
        # Skipped when dataset/test.csv is absent (git-ignored). Exact set AND order.
        assert len(committed_submission) == len(full_test_csv)
        assert list(committed_submission["sample_id"]) == list(full_test_csv["sample_id"])


class TestBackupSubmissionsAlsoCompliant:
    @pytest.mark.parametrize(
        "fname",
        sorted(
            f
            for f in os.listdir(DATASET_DIR)
            if f.startswith("test_out") and f.endswith(".csv") and f != "test_out.csv"
        ),
    )
    def test_backup_file_is_valid(self, fname):
        df = pd.read_csv(os.path.join(DATASET_DIR, fname))
        assert list(df.columns) == REQUIRED_OUTPUT_COLS
        assert len(df) == CHALLENGE_TEST_ROWS
        assert (df["price"] > 0).all()
        assert np.isfinite(df["price"]).all()
