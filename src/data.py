"""Data loading and validation for the Amazon ML Challenge 2025.

Provides typed loaders for train/test CSVs and submission validation.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd


def load_train(dataset_dir: str) -> pd.DataFrame:
    """Load the training dataset with schema validation.

    Args:
        dataset_dir: Path to the directory containing train.csv.

    Returns:
        DataFrame with columns: sample_id, catalog_content, image_link, price.

    Raises:
        FileNotFoundError: If train.csv does not exist.
        ValueError: If required columns are missing.
    """
    path = os.path.join(dataset_dir, "train.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training file not found: {path}")

    df = pd.read_csv(path)

    required = {"sample_id", "catalog_content", "image_link", "price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in train.csv: {missing}")

    return df


def load_test(dataset_dir: str) -> pd.DataFrame:
    """Load the test dataset with schema validation.

    Args:
        dataset_dir: Path to the directory containing test.csv.

    Returns:
        DataFrame with columns: sample_id, catalog_content, image_link.

    Raises:
        FileNotFoundError: If test.csv does not exist.
        ValueError: If required columns are missing.
    """
    path = os.path.join(dataset_dir, "test.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test file not found: {path}")

    df = pd.read_csv(path)

    required = {"sample_id", "catalog_content", "image_link"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in test.csv: {missing}")

    return df


def validate_submission(
    df: pd.DataFrame,
    *,
    expected_count: Optional[int] = None,
) -> None:
    """Validate submission DataFrame matches competition format.

    Args:
        df: Submission DataFrame to validate.
        expected_count: If provided, verify exact row count.

    Raises:
        ValueError: If validation fails.
    """
    # Check columns
    required_cols = {"sample_id", "price"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required column(s): {missing}")

    # Check duplicate sample_ids
    if df["sample_id"].duplicated().any():
        dup_count = int(df["sample_id"].duplicated().sum())
        raise ValueError(f"Found {dup_count} duplicate sample_id(s).")

    # Enforce numeric type check
    if pd.api.types.is_bool_dtype(df["price"]):
        raise ValueError("Prices must be numeric floats.")
    try:
        prices = pd.to_numeric(df["price"], errors="raise")
    except (ValueError, TypeError):
        raise ValueError("Prices must be numeric floats.")

    # Check for NaN prices
    if prices.isna().any():
        nan_count = int(prices.isna().sum())
        raise ValueError(f"Found {nan_count} NaN price(s). All prices must be valid.")

    # Check for infinite prices
    if np.isinf(prices).any():
        inf_count = int(np.isinf(prices).sum())
        raise ValueError(f"Found {inf_count} infinite price(s). All prices must be finite.")

    # Check positive prices
    if (prices <= 0).any():
        neg_count = int((prices <= 0).sum())
        raise ValueError(
            f"Found {neg_count} non-positive price(s). All prices must be positive floats."
        )

    # Check expected count
    if expected_count is not None and len(df) != expected_count:
        raise ValueError(
            f"Row count mismatch: got {len(df)}, expected {expected_count}"
        )
