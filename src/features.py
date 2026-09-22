"""Feature engineering for the Amazon ML Challenge 2025.

Extracts structured fields, normalizes units, parses pack quantities,
and builds TF-IDF + numeric feature matrices from catalog_content.
"""
from __future__ import annotations

import os
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
# Physical scale conversion factors (Standardized Base Units)
# ---------------------------------------------------------------------------

_WEIGHT_TO_GRAMS: dict[str, float] = {
    "g": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "kg": 1000.0,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
    "oz": 28.3495,
    "ounce": 28.3495,
    "ounces": 28.3495,
    "lb": 453.592,
    "pound": 453.592,
    "pounds": 453.592,
    "lbs": 453.592,
}

_VOLUME_TO_ML: dict[str, float] = {
    "ml": 1.0,
    "milliliter": 1.0,
    "milliliters": 1.0,
    "l": 1000.0,
    "liter": 1000.0,
    "liters": 1000.0,
    "fl_oz": 29.5735,
    "fluid ounce": 29.5735,
    "fluid ounces": 29.5735,
}

_COUNT_TO_PIECES: dict[str, float] = {
    "count": 1.0,
    "each": 1.0,
    "piece": 1.0,
    "pieces": 1.0,
    "unit": 1.0,
    "units": 1.0,
    "pack": 1.0,
    "packs": 1.0,
    "ct": 1.0,
    "can": 1.0,
    "bottle": 1.0,
    "jar": 1.0,
    "bag": 1.0,
}


def convert_to_grams(value: float, unit: str) -> float:
    """Convert quantity to total grams if unit is weight; otherwise 0.0."""
    factor = _WEIGHT_TO_GRAMS.get(str(unit).lower().strip(), 0.0)
    return float(value) * factor


def convert_to_ml(value: float, unit: str) -> float:
    """Convert quantity to total milliliters if unit is volume; otherwise 0.0."""
    factor = _VOLUME_TO_ML.get(str(unit).lower().strip(), 0.0)
    return float(value) * factor


def convert_to_pieces(value: float, unit: str) -> float:
    """Convert quantity to piece count if unit is count; otherwise 0.0."""
    factor = _COUNT_TO_PIECES.get(str(unit).lower().strip(), 0.0)
    return float(value) * factor


_NUM_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
    24: "twenty four", 30: "thirty", 32: "thirty two", 36: "thirty six",
    48: "forty eight", 50: "fifty", 60: "sixty", 72: "seventy two", 100: "one hundred",
}


def number_to_words(val: object) -> str:
    """Convert integer numbers to word representations to improve sub-word tokenization."""
    try:
        n = int(round(float(val)))
        if n in _NUM_WORDS:
            return _NUM_WORDS[n]
        return str(n)
    except Exception:
        return str(val)


def build_llm_prompt(row: pd.Series | dict) -> str:
    """Build a structured text prompt for frozen foundation language models."""
    item_name = str(row.get("item_name", "")).strip()
    unit = str(row.get("unit_normalized", "")).strip()
    val = row.get("value_num", 1.0)
    pack_qty = row.get("pack_qty", 1.0)
    bullets = str(row.get("description_raw", "")).strip()

    pack_word = number_to_words(pack_qty)
    val_str = f"{val:.1f}".rstrip("0").rstrip(".")

    parts = [f"Product: {item_name}"]
    if val > 0:
        parts.append(f"Size: {val_str} {unit}")
    if pack_qty > 1.0:
        parts.append(f"Multipack: {pack_word} pack")
    if bullets:
        parts.append(f"Specifications: {bullets[:300]}")

    return " | ".join(parts)


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
    if "catalog_content" in result.columns:
        result["catalog_content"] = result["catalog_content"].fillna("").astype(str)
    else:
        result["catalog_content"] = ""

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

    # Physical conversions (Standardized Base Units)
    result["total_grams"] = result.apply(
        lambda r: convert_to_grams(r["value_num"] * r["pack_qty"], r["unit_normalized"]), axis=1
    )
    result["total_ml"] = result.apply(
        lambda r: convert_to_ml(r["value_num"] * r["pack_qty"], r["unit_normalized"]), axis=1
    )
    result["total_pieces"] = result.apply(
        lambda r: convert_to_pieces(r["value_num"] * r["pack_qty"], r["unit_normalized"]), axis=1
    )

    result["log_total_grams"] = np.log1p(result["total_grams"])
    result["log_total_ml"] = np.log1p(result["total_ml"])
    result["log_total_pieces"] = np.log1p(result["total_pieces"])

    # Unified standard quantity across physical dimensions
    result["std_quantity"] = result["total_grams"] + result["total_ml"] + result["total_pieces"]
    result["log_std_quantity"] = np.log1p(result["std_quantity"])

    # Structured prompt for foundation models
    result["llm_prompt"] = result.apply(build_llm_prompt, axis=1)

    # Bullet points and text length statistics
    result["num_bullets"] = result["catalog_content"].apply(count_bullet_points)
    result["name_len"] = result["item_name"].astype(str).str.len().fillna(0)
    result["content_len"] = result["catalog_content"].astype(str).str.len().fillna(0)
    result["desc_len"] = result["description_raw"].astype(str).str.len().fillna(0)

    # Pricing keyword indicators
    content_lower = result["catalog_content"].str.lower()
    result["is_multipack"] = content_lower.str.contains(r"\b(?:pack|set|case|box|bundle)\b", regex=True, na=False).astype(float)
    result["is_premium"] = content_lower.str.contains(r"\b(?:organic|gourmet|pro|premium|luxury|collection)\b", regex=True, na=False).astype(float)
    result["is_value_size"] = content_lower.str.contains(r"\b(?:refill|travel|mini|sample)\b", regex=True, na=False).astype(float)

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
    clean_texts = pd.Series(texts).fillna("").astype(str)
    if vectorizer is not None:
        X = vectorizer.transform(clean_texts)
        return X, vectorizer

    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words="english",
        dtype=np.float32,
    )
    X = vec.fit_transform(clean_texts)
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
    "log_total_grams",
    "log_total_ml",
    "log_total_pieces",
    "log_std_quantity",
    "name_len",
    "content_len",
    "desc_len",
    "num_bullets",
    "is_multipack",
    "is_premium",
    "is_value_size",
]

VISUAL_METADATA_COLS = [
    "has_image",
    "img_width",
    "img_height",
    "img_aspect_ratio",
    "img_file_size_kb",
    "img_mean_luminance",
    "img_std_contrast",
    "img_colorfulness",
]


def _extract_single_image_metadata(args: tuple) -> tuple:
    """Process a single image and return its visual metadata.

    Top-level function (not a lambda or closure) so it can be pickled
    by multiprocessing.Pool.

    Args:
        args: Tuple of (index, sample_id, image_dir).

    Returns:
        Tuple of (index, has_image, width, height, aspect_ratio,
                  file_size_kb, mean_luminance, std_contrast, colorfulness).
    """
    idx, sid, image_dir = args
    zeros = (idx, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    img_path = os.path.join(image_dir, f"{sid}.jpg")
    if not os.path.exists(img_path):
        return zeros
    try:
        fsize = os.path.getsize(img_path)
        if fsize <= 1024:
            return zeros

        try:
            from PIL import Image
        except ImportError:
            return (idx, 1.0, 0.0, 0.0, 1.0, np.log1p(float(fsize) / 1024.0), 0.0, 0.0, 0.0)

        with Image.open(img_path) as img:
            w, h = img.size
            width_val = np.log1p(float(w))
            height_val = np.log1p(float(h))
            aspect = float(w) / max(float(h), 1.0)
            fsize_kb = np.log1p(float(fsize) / 1024.0)

            # Compute color statistics on thumbnail
            thumb = img.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
            arr = np.asarray(thumb, dtype=np.float32)

            # Luminance: standard ITU-R BT.601 conversion
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            mean_lum = float(np.mean(lum)) / 255.0
            std_con = float(np.std(lum)) / 128.0

            # Hasler-Süsstrunk colorfulness metric
            rg = np.abs(r - g)
            yb = np.abs(0.5 * (r + g) - b)
            rg_mean, rg_std = np.mean(rg), np.std(rg)
            yb_mean, yb_std = np.mean(yb), np.std(yb)
            std_root = np.sqrt(rg_std ** 2 + yb_std ** 2)
            mean_root = np.sqrt(rg_mean ** 2 + yb_mean ** 2)
            colorful = (std_root + 0.3 * mean_root) / 100.0

            return (idx, 1.0, width_val, height_val, aspect, fsize_kb, mean_lum, std_con, colorful)
    except Exception:
        # Corrupted image or read failure: safely zero-filled
        return zeros


def extract_visual_metadata_features(
    df: pd.DataFrame,
    image_dir: str,
    n_workers: int = 32,
) -> pd.DataFrame:
    """Extract lightweight visual properties from downloaded product images.

    Uses multiprocessing.Pool for parallel image processing across multiple
    CPU cores, reducing runtime from ~87 minutes to ~3-5 minutes on a
    multi-core server.

    Extracts:
        - has_image: 1.0 if image exists, readable, and > 1KB, else 0.0
        - img_width: image width in pixels (log1p scaled)
        - img_height: image height in pixels (log1p scaled)
        - img_aspect_ratio: width / height
        - img_file_size_kb: file size in kilobytes (log1p scaled)
        - img_mean_luminance: average normalized pixel brightness [0, 1]
        - img_std_contrast: standard deviation of luminance (contrast)
        - img_colorfulness: Hasler-Süsstrunk colorfulness metric

    Args:
        df: DataFrame containing 'sample_id'.
        image_dir: Local directory containing '{sample_id}.jpg'.
        n_workers: Number of parallel worker processes (default: 32).

    Returns:
        DataFrame with visual metadata columns aligned with input df.
    """
    from multiprocessing import Pool, cpu_count

    n = len(df)
    has_image = np.zeros(n, dtype=np.float32)
    img_width = np.zeros(n, dtype=np.float32)
    img_height = np.zeros(n, dtype=np.float32)
    img_aspect_ratio = np.ones(n, dtype=np.float32)
    img_file_size_kb = np.zeros(n, dtype=np.float32)
    img_mean_luminance = np.zeros(n, dtype=np.float32)
    img_std_contrast = np.zeros(n, dtype=np.float32)
    img_colorfulness = np.zeros(n, dtype=np.float32)

    sample_ids = df["sample_id"].values
    work_items = [(idx, int(sid), image_dir) for idx, sid in enumerate(sample_ids)]

    # Clamp workers to available CPUs
    actual_workers = min(n_workers, max(1, cpu_count() or 1))

    try:
        with Pool(processes=actual_workers) as pool:
            results = pool.map(_extract_single_image_metadata, work_items, chunksize=256)
    except Exception:
        # Fallback to single-threaded if multiprocessing fails
        results = [_extract_single_image_metadata(item) for item in work_items]

    for (idx, hi, w, h, ar, fs, ml, sc, cf) in results:
        has_image[idx] = hi
        img_width[idx] = w
        img_height[idx] = h
        img_aspect_ratio[idx] = ar
        img_file_size_kb[idx] = fs
        img_mean_luminance[idx] = ml
        img_std_contrast[idx] = sc
        img_colorfulness[idx] = cf

    return pd.DataFrame({
        "has_image": has_image,
        "img_width": img_width,
        "img_height": img_height,
        "img_aspect_ratio": img_aspect_ratio,
        "img_file_size_kb": img_file_size_kb,
        "img_mean_luminance": img_mean_luminance,
        "img_std_contrast": img_std_contrast,
        "img_colorfulness": img_colorfulness,
    }, index=df.index)


def build_vision_svd_features(
    train_vision_emb: np.ndarray,
    test_vision_emb: Optional[np.ndarray] = None,
    n_components: int = 32,
    random_state: int = 42,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Reduce high-dimensional vision embeddings via TruncatedSVD for GBDT ingestion.

    Args:
        train_vision_emb: (N, D_vision) numpy array.
        test_vision_emb: Optional (M, D_vision) numpy array.
        n_components: Number of principal visual components (default 32).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_svd, test_svd).
    """
    from sklearn.decomposition import TruncatedSVD

    n_comp = min(n_components, train_vision_emb.shape[1], train_vision_emb.shape[0])
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    train_svd = svd.fit_transform(train_vision_emb).astype(np.float32)
    test_svd = svd.transform(test_vision_emb).astype(np.float32) if test_vision_emb is not None else None
    return train_svd, test_svd


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
    vision_svd_features: Optional[np.ndarray | csr_matrix] = None,
) -> csr_matrix:
    """Combine text, numeric, and optional vision SVD features into a single matrix.

    Args:
        text_features: Sparse TF-IDF matrix.
        numeric_features: Sparse numeric feature matrix.
        vision_svd_features: Optional dense or sparse vision SVD components.

    Returns:
        Combined sparse matrix.
    """
    matrices = [text_features, numeric_features]
    if vision_svd_features is not None:
        if not isinstance(vision_svd_features, csr_matrix):
            vision_svd_features = csr_matrix(vision_svd_features.astype(np.float32))
        matrices.append(vision_svd_features)
    return hstack(matrices).tocsr()


def extract_advanced_catalog_features(df: pd.DataFrame, brand_tiers: Optional[dict] = None) -> pd.DataFrame:
    """Extract advanced catalog features including category, brand tier, and material score."""
    result = df.copy()
    
    # 1. Product Category
    categories = ['Electronics', 'Beauty', 'Grocery', 'Health', 'Home & Kitchen', 'Toys', 'Sports', 'Automotive', 'Clothing', 'Books', 'Pet Supplies', 'Baby', 'Office', 'Tools', 'Garden', 'Jewelry', 'Musical Instruments', 'Arts & Crafts', 'Industrial']
    
    def get_category(text):
        if not isinstance(text, str):
            return 'Other'
        text_lower = text.lower()
        for cat in categories:
            if cat.lower() in text_lower:
                return cat
        return 'Other'
        
    result['product_category_str'] = result['catalog_content'].apply(get_category)
    cat_map = {cat: i for i, cat in enumerate(categories + ['Other'])}
    result['product_category'] = result['product_category_str'].map(cat_map)
    result.drop(columns=['product_category_str'], inplace=True)
    
    # 2. Brand Tier
    if 'brand' not in result.columns:
        result['brand'] = result['catalog_content'].apply(lambda x: extract_brand(extract_field(x, "Item Name")))
    
    if brand_tiers is not None:
        result['brand_price_tier'] = result['brand'].map(brand_tiers).fillna(2).astype(int)
    elif 'price' in result.columns:
        # Compute from training data
        brand_median_price = result.groupby('brand')['price'].median()
        quantiles = brand_median_price.quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
        def get_tier(price):
            if pd.isna(price): return 2
            if price <= quantiles.get(0.2, 0): return 0
            if price <= quantiles.get(0.4, 0): return 1
            if price <= quantiles.get(0.6, 0): return 2
            if price <= quantiles.get(0.8, 0): return 3
            return 4
        computed_tiers = brand_median_price.apply(get_tier).to_dict()
        result['brand_price_tier'] = result['brand'].map(computed_tiers).fillna(2).astype(int)
        result.attrs['brand_tiers'] = computed_tiers
    else:
        # Test data without precomputed brand tiers - assign middle tier
        result['brand_price_tier'] = 2
        
    # 3. Material Quality Score
    premium_words = ['stainless steel', 'leather', 'organic', 'premium', 'professional', 'titanium', 'ceramic']
    budget_words = ['plastic', 'synthetic', 'basic', 'value', 'economy', 'refill']
    
    def get_material_score(text):
        if not isinstance(text, str): return 0
        text_lower = text.lower()
        prem_count = sum(text_lower.count(w) for w in premium_words)
        budg_count = sum(text_lower.count(w) for w in budget_words)
        return prem_count - budg_count
        
    result['material_quality_score'] = result['catalog_content'].apply(get_material_score)
    
    # 4. Price Keyword Flags
    keywords = ['bulk', 'pack of', 'set of', 'bundle', 'refill', 'travel size', 'sample', 'mini', 'jumbo', 'family size']
    
    content_lower = result['catalog_content'].astype(str).str.lower()
    for kw in keywords:
        col_name = f'flag_{kw.replace(" ", "_")}'
        result[col_name] = content_lower.str.contains(kw, regex=False).astype(int)
        
    return result


def compute_iqr_training_mask(prices: np.ndarray | pd.Series, multiplier: float = 3.0) -> np.ndarray:
    """Compute an IQR-based outlier mask on log(prices)."""
    prices_arr = np.asarray(prices)
    log_prices = np.log(prices_arr)
    
    q1 = np.percentile(log_prices, 25)
    q3 = np.percentile(log_prices, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    mask = (log_prices >= lower_bound) & (log_prices <= upper_bound)
    removed = len(mask) - np.sum(mask)
    print(f"Removed {removed} outliers based on log(price) IQR with multiplier {multiplier}", flush=True)
    
    return mask

