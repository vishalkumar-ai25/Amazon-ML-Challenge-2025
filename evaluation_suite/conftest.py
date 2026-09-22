"""Shared fixtures for the compliance evaluation suite.

All fixtures are deterministic and offline: synthetic data is generated from fixed
seeds and no network access is performed (HTTP clients are mocked in the tests).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DATASET_DIR = os.path.join(REPO_ROOT, "dataset")
SRC_DIR = os.path.join(REPO_ROOT, "src")

REQUIRED_INPUT_COLS = ["sample_id", "catalog_content", "image_link"]
REQUIRED_OUTPUT_COLS = ["sample_id", "price"]
CHALLENGE_TEST_ROWS = 75_000

_UNITS = ["Ounce", "Fl Oz", "Pound", "Count", "Gram", "Liter", "Milliliter"]
_BRANDS = ["Acme", "Amazon Basics", "Globex", "Initech", "Umbrella", "Hooli"]
_CATEGORIES = ["Coffee", "Shampoo", "Cable", "Vitamins", "Protein Powder", "Candle"]


def make_catalog(rng: np.random.Generator, i: int) -> str:
    brand = _BRANDS[i % len(_BRANDS)]
    cat = _CATEGORIES[(i * 7) % len(_CATEGORIES)]
    value = float(rng.choice([4, 8, 12, 16, 24, 32, 64]))
    unit = _UNITS[i % len(_UNITS)]
    pack = int(rng.choice([1, 1, 1, 2, 3, 6, 12]))
    pack_txt = f" (Pack of {pack})" if pack > 1 else ""
    return (
        f"Item Name: {brand} Premium {cat} {value:g} {unit}{pack_txt}\n"
        f"Bullet Point 1: High quality {cat.lower()} from {brand}\n"
        f"Bullet Point 2: Ideal for daily use\n"
        f"Product Description: {brand} {cat} in a {value:g} {unit} container.\n"
        f"Value: {value:g}\nUnit: {unit}\n"
    )


def make_synthetic_frames(n_train: int = 200, n_test: int = 60, seed: int = 123):
    rng = np.random.default_rng(seed)
    train_rows, test_rows = [], []
    for i in range(n_train):
        catalog = make_catalog(rng, i)
        base = 3.0 + (i % 6) * 4.0
        price = round(float(base * rng.uniform(0.8, 1.25)), 2)
        train_rows.append(
            {
                "sample_id": 100_000 + i,
                "catalog_content": catalog,
                "image_link": f"https://images.example.com/train_{i}.jpg",
                "price": max(price, 0.5),
            }
        )
    for j in range(n_test):
        test_rows.append(
            {
                "sample_id": 900_000 + j,
                "catalog_content": make_catalog(rng, j + 17),
                "image_link": f"https://images.example.com/test_{j}.jpg",
            }
        )
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


@pytest.fixture(scope="session")
def synthetic_frames():
    return make_synthetic_frames()


@pytest.fixture
def synthetic_dataset_dir(tmp_path, synthetic_frames):
    train_df, test_df = synthetic_frames
    d = tmp_path / "dataset"
    d.mkdir()
    train_df.to_csv(d / "train.csv", index=False)
    test_df.to_csv(d / "test.csv", index=False)
    return str(d)


@pytest.fixture(scope="session")
def committed_submission() -> pd.DataFrame:
    path = os.path.join(DATASET_DIR, "test_out.csv")
    if not os.path.exists(path):
        pytest.skip("dataset/test_out.csv is not present in the repository")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def full_test_csv() -> pd.DataFrame:
    path = os.path.join(DATASET_DIR, "test.csv")
    if not os.path.exists(path):
        pytest.skip(
            "dataset/test.csv is git-ignored and absent from the clone; "
            "exact sample_id alignment cannot be verified offline"
        )
    return pd.read_csv(path)
