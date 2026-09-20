"""Validation tests for the final submission output.

Ensures that the output file:
- Exists at dataset/test_out.csv
- Contains exactly 2 columns: sample_id, price
- Matches all test sample_ids in exact order
- Has exactly 75,000 rows (matching test.csv)
- Contains only positive finite floats
- Has no nulls, NaNs, or Infs
"""
import os
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
TEST_CSV_PATH = os.path.join(DATASET_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(DATASET_DIR, "test_out.csv")


def test_submission_file_exists():
    """Verify test_out.csv exists."""
    assert os.path.exists(SUBMISSION_PATH), f"Submission file {SUBMISSION_PATH} not found"


def test_submission_structure_and_ids():
    """Verify submission matches test.csv ids, row counts, and positive prices."""
    if not os.path.exists(SUBMISSION_PATH):
        pytest.skip("test_out.csv not yet generated")

    test_df = pd.read_csv(TEST_CSV_PATH)
    sub_df = pd.read_csv(SUBMISSION_PATH)

    # 1. Exact columns
    assert list(sub_df.columns) == ["sample_id", "price"]

    # 2. Row count exact match
    assert len(sub_df) == len(test_df), f"Expected {len(test_df)} rows, got {len(sub_df)}"

    # 3. Exact matching sample_ids in exact order
    assert (sub_df["sample_id"].values == test_df["sample_id"].values).all(), "sample_ids mismatch"

    # 4. No NaNs or Infs
    assert sub_df["price"].notna().all(), "Found NaN in price column"
    assert not np.isinf(sub_df["price"].values).any(), "Found Inf in price column"

    # 5. Strictly positive
    assert (sub_df["price"] > 0).all(), "Found non-positive price"
