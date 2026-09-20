"""Tests for feature engineering module.

Covers text extraction, unit normalization, pack quantity parsing,
TF-IDF vectorization, and numeric feature construction.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------
SAMPLE_CATALOG_CONTENTS = [
    (
        "Item Name: La Victoria Green Taco Sauce Mild, 12 Ounce (Pack of 6)\n"
        "Value: 72.0\n"
        "Unit: Fl Oz\n"
    ),
    (
        "Item Name: Dove Men+Care Body Wash Clean Comfort 18 oz\n"
        "Bullet Point 1: Body wash for men with micro moisture\n"
        "Value: 18.0\n"
        "Unit: Ounce\n"
    ),
    (
        "Item Name: Apple AirPods Pro (2nd Generation)\n"
        "Bullet Point 1: Active Noise Cancellation\n"
        "Product Description: AirPods Pro feature up to 2x more Active Noise Cancellation\n"
        "Value: 1.0\n"
        "Unit: Count\n"
    ),
    (
        "Item Name: Bounty Quick-Size Paper Towels, 12 Family Rolls\n"
        "Value: 12.0\n"
        "Unit: Count\n"
    ),
]


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "catalog_content": SAMPLE_CATALOG_CONTENTS,
        "sample_id": [1, 2, 3, 4],
    })


# ---------------------------------------------------------------------------
# Field extraction tests
# ---------------------------------------------------------------------------
class TestExtractField:

    def test_extract_item_name(self):
        from src.features import extract_field
        text = SAMPLE_CATALOG_CONTENTS[0]
        result = extract_field(text, "Item Name")
        assert "La Victoria" in result
        assert "Taco Sauce" in result

    def test_extract_value(self):
        from src.features import extract_field
        text = SAMPLE_CATALOG_CONTENTS[0]
        result = extract_field(text, "Value")
        assert result.strip() == "72.0"

    def test_extract_unit(self):
        from src.features import extract_field
        text = SAMPLE_CATALOG_CONTENTS[0]
        result = extract_field(text, "Unit")
        assert "Fl Oz" in result

    def test_extract_missing_field_returns_empty(self):
        from src.features import extract_field
        result = extract_field("Item Name: Test\nValue: 1", "Product Description")
        assert result == ""

    def test_extract_handles_nan_input(self):
        from src.features import extract_field
        result = extract_field(float("nan"), "Item Name")
        assert result == ""


# ---------------------------------------------------------------------------
# Unit normalization tests
# ---------------------------------------------------------------------------
class TestNormalizeUnit:

    @pytest.mark.parametrize("raw,expected", [
        ("Fl Oz", "fl_oz"),
        ("fluid ounce", "fl_oz"),
        ("fl. oz", "fl_oz"),
        ("Ounce", "oz"),
        ("oz", "oz"),
        ("Pound", "lb"),
        ("lb", "lb"),
        ("Count", "count"),
        ("count", "count"),
        ("Gram", "g"),
        ("g", "g"),
        ("Kilogram", "kg"),
        ("kg", "kg"),
        ("Milliliter", "ml"),
        ("ml", "ml"),
        ("Liter", "l"),
        ("liter", "l"),
    ])
    def test_normalize_known_units(self, raw, expected):
        from src.features import normalize_unit
        assert normalize_unit(raw) == expected

    def test_normalize_unknown_unit_lowercases(self):
        from src.features import normalize_unit
        assert normalize_unit("SomeWeirdUnit") == "someweirdunit"


# ---------------------------------------------------------------------------
# Pack quantity extraction tests
# ---------------------------------------------------------------------------
class TestExtractPackQuantity:

    @pytest.mark.parametrize("text,expected", [
        ("Pack of 6", 6.0),
        ("pack of 12", 12.0),
        ("12 pack", 12.0),
        ("24 count", 24.0),
        ("case of 4", 4.0),
        ("box of 10", 10.0),
        ("set of 3", 3.0),
        ("6 pk", 6.0),
        ("Just a regular item", 1.0),  # no pack info → default 1
    ])
    def test_pack_quantity_extraction(self, text, expected):
        from src.features import extract_pack_quantity
        assert extract_pack_quantity(text) == expected


# ---------------------------------------------------------------------------
# Text feature extraction tests
# ---------------------------------------------------------------------------
class TestTextFeatures:

    def test_tfidf_output_shape(self, sample_df):
        from src.features import extract_text_features
        X, vectorizer = extract_text_features(
            sample_df["catalog_content"], max_features=100
        )
        assert X.shape[0] == len(sample_df)
        assert X.shape[1] <= 100
        assert issparse(X)

    def test_tfidf_no_nan(self, sample_df):
        from src.features import extract_text_features
        X, _ = extract_text_features(sample_df["catalog_content"], max_features=100)
        assert not np.any(np.isnan(X.toarray()))

    def test_tfidf_reproducible(self, sample_df):
        from src.features import extract_text_features
        X1, _ = extract_text_features(sample_df["catalog_content"], max_features=100)
        X2, _ = extract_text_features(sample_df["catalog_content"], max_features=100)
        np.testing.assert_array_equal(X1.toarray(), X2.toarray())

    def test_tfidf_transform_mode(self, sample_df):
        """Test that we can fit on train and transform test separately."""
        from src.features import extract_text_features
        X_train, vectorizer = extract_text_features(
            sample_df["catalog_content"], max_features=100
        )
        # Simulate test data — use same vectorizer
        X_test = vectorizer.transform(sample_df["catalog_content"])
        assert X_test.shape[1] == X_train.shape[1]


# ---------------------------------------------------------------------------
# Structured feature extraction tests
# ---------------------------------------------------------------------------
class TestStructuredFeatures:

    def test_extract_structured_fields(self, sample_df):
        from src.features import extract_structured_features
        result = extract_structured_features(sample_df)
        assert "item_name" in result.columns
        assert "value_num" in result.columns
        assert "unit_normalized" in result.columns
        assert "pack_qty" in result.columns
        assert len(result) == len(sample_df)

    def test_structured_no_nan_in_numerics(self, sample_df):
        from src.features import extract_structured_features
        result = extract_structured_features(sample_df)
        assert result["value_num"].notna().all()
        assert result["pack_qty"].notna().all()

    def test_structured_value_num_positive(self, sample_df):
        from src.features import extract_structured_features
        result = extract_structured_features(sample_df)
        assert (result["value_num"] > 0).all()

    def test_structured_pack_qty_at_least_1(self, sample_df):
        from src.features import extract_structured_features
        result = extract_structured_features(sample_df)
        assert (result["pack_qty"] >= 1.0).all()


class TestAdvancedFeatures:

    @pytest.mark.parametrize("item_name,expected_brand", [
        ("La Victoria Green Taco Sauce Mild, 12 Ounce", "La Victoria"),
        ("Apple AirPods Pro (2nd Generation)", "Apple"),
        ("Dove Men+Care Body Wash Clean Comfort 18 oz", "Dove Men+Care"),
        ("Bounty Quick-Size Paper Towels", "Bounty"),
    ])
    def test_extract_brand(self, item_name, expected_brand):
        from src.features import extract_brand
        assert extract_brand(item_name) == expected_brand

    @pytest.mark.parametrize("unit,expected_category", [
        ("fl_oz", "volume"),
        ("ml", "volume"),
        ("l", "volume"),
        ("oz", "weight"),
        ("lb", "weight"),
        ("g", "weight"),
        ("kg", "weight"),
        ("count", "count"),
        ("unknown_unit", "other"),
    ])
    def test_get_unit_category(self, unit, expected_category):
        from src.features import get_unit_category
        assert get_unit_category(unit) == expected_category

    def test_count_bullet_points(self):
        from src.features import count_bullet_points
        text_3_bullets = "Bullet Point 1: A\nBullet Point 2: B\nBullet Point 3: C"
        assert count_bullet_points(text_3_bullets) == 3

        text_no_bullets = "Item Name: Something\nValue: 10\nUnit: Count"
        assert count_bullet_points(text_no_bullets) == 0

    def test_physical_conversions(self):
        from src.features import convert_to_grams, convert_to_ml, convert_to_pieces
        # 1 lb should be ~453.59g
        assert pytest.approx(convert_to_grams(1.0, "lb"), 0.1) == 453.59
        # 16 oz should also be ~453.59g
        assert pytest.approx(convert_to_grams(16.0, "oz"), 0.1) == 453.59
        # 1 fl oz should be ~29.57ml
        assert pytest.approx(convert_to_ml(1.0, "fl_oz"), 0.1) == 29.57
        # 12 count should be 12 pieces
        assert convert_to_pieces(12.0, "count") == 12.0

    def test_number_to_words(self):
        from src.features import number_to_words
        assert number_to_words(1) == "one"
        assert number_to_words(6) == "six"
        assert number_to_words(12) == "twelve"
        assert number_to_words(24) == "twenty four"

    def test_build_llm_prompt(self):
        from src.features import build_llm_prompt
        row = {
            "item_name": "Premium Colombian Coffee Beans",
            "value_num": 2.0,
            "unit_normalized": "lb",
            "pack_qty": 2.0,
            "description_raw": "Single origin dark roast coffee.",
        }
        prompt = build_llm_prompt(row)
        assert "Product: Premium Colombian Coffee Beans" in prompt
        assert "Size: 2 lb" in prompt
        assert "Multipack: two pack" in prompt
        assert "Specifications: Single origin dark roast" in prompt
