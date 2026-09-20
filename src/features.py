"""Feature engineering for the Amazon ML Challenge 2025.

Extracts structured fields, normalizes units, parses pack quantities,
and builds TF-IDF + numeric feature matrices from catalog_content.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Field extraction from catalog_content
# ---------------------------------------------------------------------------

def extract_field(text: object, field: str) -> str:
    """Extract a named field from catalog_content text.

    Catalog entries use the pattern ``Field Name: value`` on separate lines.

    Args:
        text: Raw catalog_content string (may be NaN).
        field: Field name to extract (e.g., 'Item Name', 'Value', 'Unit').

    Returns:
        Extracted value string, or '' if not found / input is NaN.
    """
    if not isinstance(text, str):
        return ""
    m = re.search(rf"^{re.escape(field)}:\s*(.*)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Unit normalization
# ---------------------------------------------------------------------------

_UNIT_MAP: dict[str, str] = {
    # Fluid ounces
    "fl oz": "fl_oz",
    "fl. oz": "fl_oz",
    "fluid ounce": "fl_oz",
    "fluid ounces": "fl_oz",
    "fl_oz": "fl_oz",
    # Weight ounces
    "ounce": "oz",
    "ounces": "oz",
    "oz": "oz",
    # Pounds
    "pound": "lb",
    "pounds": "lb",
    "lb": "lb",
    "lbs": "lb",
    # Grams / kilograms
    "gram": "g",
    "grams": "g",
    "g": "g",
    "kilogram": "kg",
    "kilograms": "kg",
    "kg": "kg",
    # Milliliters / liters
    "milliliter": "ml",
    "milliliters": "ml",
    "ml": "ml",
    "liter": "l",
    "liters": "l",
    "l": "l",
    # Count
    "count": "count",
    "each": "count",
    "piece": "count",
    "pieces": "count",
    "unit": "count",
    "units": "count",
}


def normalize_unit(raw_unit: str) -> str:
    """Normalize a unit string to a canonical form.

    Args:
        raw_unit: Raw unit string from catalog (e.g., 'Fl Oz', 'Ounce').

    Returns:
        Normalized unit string (e.g., 'fl_oz', 'oz').
    """
    lower = raw_unit.strip().lower()
    return _UNIT_MAP.get(lower, lower)


# ---------------------------------------------------------------------------
# Pack quantity extraction
# ---------------------------------------------------------------------------

_PACK_PATTERNS = [
    # "pack of 6", "case of 12", "box of 10", "set of 3"
    re.compile(r"(?:pack|case|box|set)\s+of\s+(\d+)", re.IGNORECASE),
    # "6 pack", "12 pk", "24 count", "6 ct", "10 per case"
    re.compile(r"(\d+)\s*(?:pack|pk|count|ct|per\s+case|boxes?)", re.IGNORECASE),
]


def extract_pack_quantity(text: object) -> float:
    """Extract the pack/multi-pack quantity from text.

    Searches for patterns like 'Pack of 6', '12 pack', '24 count', etc.

    Args:
        text: Text to search (catalog_content or item name).

    Returns:
        Pack quantity as float. Defaults to 1.0 if no pattern found.
    """
    text_str = str(text) if not isinstance(text, str) else text
    for pattern in _PACK_PATTERNS:
        m = pattern.search(text_str)
        if m:
            return float(m.group(1))
    return 1.0


# ---------------------------------------------------------------------------
# Brand extraction and text parsing helpers
# ---------------------------------------------------------------------------

_ARTICLES = {"la", "le", "el", "the", "de", "del", "san", "santa", "st", "st.", "dr", "dr.", "mr", "mr.", "mrs", "mrs."}


def extract_brand(item_name: object) -> str:
    """Extract approximate brand name from item title.

    Amazon titles typically put the brand name as the first 1-2 words.
    """
    if not isinstance(item_name, str) or not item_name.strip():
        return ""
    clean = re.sub(r"^[^\w\s]+", "", item_name.strip())
    first_chunk = re.split(r"\s+[-–—|/]\s+|,", clean)[0].strip()
    tokens = first_chunk.split()
    if not tokens:
        return ""
    if len(tokens) >= 2 and tokens[0].lower() in _ARTICLES:
        return f"{tokens[0]} {tokens[1]}"
    if len(tokens) >= 2 and (tokens[1].startswith("&") or tokens[1].startswith("+") or "+" in tokens[1] or "&" in tokens[1]):
        if len(tokens) >= 3 and tokens[1] in ["&", "+"]:
            return f"{tokens[0]} {tokens[1]} {tokens[2]}"
        return f"{tokens[0]} {tokens[1]}"
    return tokens[0]


_UNIT_CATEGORIES = {
    "fl_oz": "volume",
    "ml": "volume",
    "l": "volume",
    "oz": "weight",
    "lb": "weight",
    "g": "weight",
    "kg": "weight",
    "count": "count",
}


def get_unit_category(unit: str) -> str:
    """Classify normalized unit into broader physical category."""
    return _UNIT_CATEGORIES.get(str(unit).lower().strip(), "other")


def count_bullet_points(text: object) -> int:
    """Count number of bullet point entries in catalog text."""
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"^Bullet Point \d+:", text, re.MULTILINE))


# ---------------------------------------------------------------------------
# Structured feature extraction
# ---------------------------------------------------------------------------

def extract_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract structured fields from catalog_content into typed columns.

    Args:
        df: DataFrame with 'catalog_content' column.

    Returns:
        DataFrame with structured columns and numeric indicators.
    """
    result = df.copy()

    # Extract text fields
    result["item_name"] = result["catalog_content"].apply(
        lambda x: extract_field(x, "Item Name")
    )
    result["value_raw"] = result["catalog_content"].apply(
        lambda x: extract_field(x, "Value")
    )
    result["unit_raw"] = result["catalog_content"].apply(
        lambda x: extract_field(x, "Unit")
    )
    result["description_raw"] = result["catalog_content"].apply(
        lambda x: extract_field(x, "Product Description")
    )

    # Normalize unit and classify
    result["unit_normalized"] = result["unit_raw"].apply(normalize_unit)
    result["unit_category"] = result["unit_normalized"].apply(get_unit_category)

    # Brand extraction
    result["brand"] = result["item_name"].apply(extract_brand)

    # Numeric value (from Value field)
    result["value_num"] = pd.to_numeric(result["value_raw"], errors="coerce").fillna(1.0)
    result["value_num"] = np.clip(result["value_num"], 0.01, 10_000.0)

    # Pack quantity (from full catalog text)
    result["pack_qty"] = result["catalog_content"].apply(extract_pack_quantity)

    # Derived features
    result["log_value"] = np.log1p(result["value_num"])
    result["log_pack_qty"] = np.log1p(result["pack_qty"])
    result["unit_size"] = result["value_num"] / np.maximum(result["pack_qty"], 1.0)
    result["log_unit_size"] = np.log1p(result["unit_size"])
    result["total_quantity"] = result["value_num"] * result["pack_qty"]
    result["log_total_quantity"] = np.log1p(result["total_quantity"])

    # Bullet points and text length statistics
    result["num_bullets"] = result["catalog_content"].apply(count_bullet_points)
    result["name_len"] = result["item_name"].str.len().fillna(0)
    result["content_len"] = result["catalog_content"].str.len().fillna(0)
    result["desc_len"] = result["description_raw"].str.len().fillna(0)

    # Pricing keyword indicators
    content_lower = result["catalog_content"].str.lower()
    result["is_multipack"] = content_lower.str.contains(r"\b(?:pack|set|case|box|bundle)\b", regex=True).astype(float)
    result["is_premium"] = content_lower.str.contains(r"\b(?:organic|gourmet|pro|premium|luxury|collection)\b", regex=True).astype(float)
    result["is_value_size"] = content_lower.str.contains(r"\b(?:refill|travel|mini|sample)\b", regex=True).astype(float)

    return result


# ---------------------------------------------------------------------------
# TF-IDF text feature extraction
# ---------------------------------------------------------------------------

def extract_text_features(
    texts: pd.Series,
    *,
    max_features: int = 15_000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 3,
    vectorizer: Optional[TfidfVectorizer] = None,
) -> Tuple[csr_matrix, TfidfVectorizer]:
    """Extract TF-IDF features from text.

    Args:
        texts: Series of text strings (catalog_content).
        max_features: Maximum number of TF-IDF features.
        ngram_range: N-gram range for TF-IDF.
        min_df: Minimum document frequency.
        vectorizer: Pre-fitted vectorizer for transform-only mode.
            If None, fits a new one.

    Returns:
        Tuple of (sparse feature matrix, fitted TfidfVectorizer).
    """
    if vectorizer is not None:
        X = vectorizer.transform(texts)
        return X, vectorizer

    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words="english",
        dtype=np.float32,
    )
    X = vec.fit_transform(texts)
    return X, vec


# ---------------------------------------------------------------------------
# Numeric feature matrix builder
# ---------------------------------------------------------------------------

# Columns used as numeric features (order matters for consistency)
NUMERIC_COLS = [
    "log_value",
    "log_pack_qty",
    "log_unit_size",
    "log_total_quantity",
    "name_len",
    "content_len",
    "desc_len",
    "num_bullets",
    "is_multipack",
    "is_premium",
    "is_value_size",
]


def build_numeric_matrix(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
) -> csr_matrix:
    """Build a sparse numeric feature matrix from structured features.

    Args:
        df: DataFrame with structured feature columns (from extract_structured_features).
        columns: List of column names to include. Defaults to NUMERIC_COLS.

    Returns:
        Sparse matrix of shape (n_samples, n_features).
    """
    cols = columns or NUMERIC_COLS
    return csr_matrix(df[cols].fillna(0).values.astype(np.float32))


def build_feature_matrix(
    text_features: csr_matrix,
    numeric_features: csr_matrix,
) -> csr_matrix:
    """Combine text and numeric features into a single matrix.

    Args:
        text_features: Sparse TF-IDF matrix.
        numeric_features: Sparse numeric feature matrix.

    Returns:
        Combined sparse matrix.
    """
    return hstack([text_features, numeric_features]).tocsr()
