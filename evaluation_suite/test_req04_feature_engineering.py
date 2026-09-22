"""REQ-06..09: Feature engineering on catalog_content (title / description / IPQ),
text vectorisation, outlier handling and robustness to malformed catalog payloads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    NUMERIC_COLS,
    build_feature_matrix,
    build_llm_prompt,
    build_numeric_matrix,
    build_vision_svd_features,
    compute_iqr_training_mask,
    convert_to_grams,
    convert_to_ml,
    convert_to_pieces,
    extract_advanced_catalog_features,
    extract_brand,
    extract_field,
    extract_pack_quantity,
    extract_structured_features,
    extract_text_features,
    normalize_unit,
    number_to_words,
)

CATALOG = (
    "Item Name: Acme Organic Coffee Beans 12 Ounce (Pack of 3)\n"
    "Bullet Point 1: Rich flavour\n"
    "Bullet Point 2: Fair trade\n"
    "Product Description: Whole bean arabica coffee.\n"
    "Value: 12\n"
    "Unit: Ounce\n"
)


class TestFieldExtraction:
    def test_named_fields(self):
        assert extract_field(CATALOG, "Item Name").startswith("Acme Organic Coffee")
        assert extract_field(CATALOG, "Value") == "12"
        assert extract_field(CATALOG, "Unit") == "Ounce"

    def test_missing_field_returns_empty(self):
        assert extract_field(CATALOG, "Colour") == ""

    @pytest.mark.parametrize("bad", [None, np.nan, 42, 3.5, b"bytes", [], {}])
    def test_non_string_payload_returns_empty(self, bad):
        assert extract_field(bad, "Item Name") == ""

    def test_field_name_with_regex_metachars_is_escaped(self):
        assert extract_field("A.B: hello\n", "A.B") == "hello"
        assert extract_field("AXB: hello\n", "A.B") == ""


class TestIPQAndPackQuantity:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Pack of 6", 6.0),
            ("12 Pack", 12.0),
            ("24 count", 24.0),
            ("no pack info here", 1.0),
            ("", 1.0),
            (None, 1.0),
            (np.nan, 1.0),
        ],
    )
    def test_pack_quantity(self, text, expected):
        assert extract_pack_quantity(text) == expected

    def test_length_units_are_not_mistaken_for_pack_quantity(self):
        # AGENTS.md "PPU hazard": a 6 ft cable must not be parsed as quantity 6.
        assert extract_pack_quantity("Item Name: 6 ft HDMI cable") == 1.0

    def test_zero_pack_does_not_break_downstream_division(self):
        df = pd.DataFrame({"sample_id": [1], "catalog_content": ["Item Name: Foo 0 pack\nValue: 10\nUnit: Ounce"]})
        out = extract_structured_features(df)
        assert np.isfinite(out[NUMERIC_COLS].to_numpy(dtype=float)).all()


class TestUnitsAndPhysicalConversions:
    @pytest.mark.parametrize(
        "raw,expected",
        [("Ounce", "oz"), ("OUNCES", "oz"), ("Fl Oz", "fl_oz"), ("fluid ounces", "fl_oz"), ("Pound", "lb"), ("lbs", "lb")],
    )
    def test_unit_canonicalisation(self, raw, expected):
        assert normalize_unit(raw) == expected

    def test_unknown_unit_does_not_raise(self):
        assert isinstance(normalize_unit("parsecs"), str)
        assert isinstance(normalize_unit(""), str)

    def test_weight_conversions(self):
        assert convert_to_grams(1, "lb") == pytest.approx(453.592, rel=1e-3)
        assert convert_to_grams(16, "oz") == pytest.approx(convert_to_grams(1, "lb"), rel=1e-3)
        assert convert_to_grams(1, "fl_oz") == 0.0  # not a weight

    def test_volume_conversions(self):
        assert convert_to_ml(1, "fl_oz") == pytest.approx(29.5735, rel=1e-3)
        assert convert_to_ml(1, "oz") == 0.0  # not a volume

    def test_count_conversion(self):
        assert convert_to_pieces(5, "count") == 5.0
        assert convert_to_pieces(5, "oz") == 0.0

    def test_conversions_are_linear_in_value(self):
        assert convert_to_grams(10, "oz") == pytest.approx(10 * convert_to_grams(1, "oz"))


class TestBrandExtraction:
    def test_single_token_brand(self):
        assert extract_brand("Acme Coffee 12 oz") == "Acme"

    def test_article_prefixed_brand_keeps_two_tokens(self):
        assert extract_brand("The North Face Jacket") == "The North"

    def test_non_string_input(self):
        assert extract_brand(None) == ""
        assert extract_brand(np.nan) == ""


class TestStructuredFeatureMatrix:
    def test_happy_path_values(self):
        out = extract_structured_features(pd.DataFrame({"sample_id": [1], "catalog_content": [CATALOG]}))
        row = out.iloc[0]
        assert row["value_num"] == 12.0
        assert row["unit_normalized"] == "oz"
        assert row["pack_qty"] == 3.0
        assert row["total_grams"] == pytest.approx(36 * 28.3495, rel=1e-2)
        assert row["is_multipack"] == 1

    def test_nan_and_empty_catalog_do_not_raise_and_produce_finite_numerics(self):
        df = pd.DataFrame({"sample_id": [1, 2, 3], "catalog_content": [np.nan, "", "garbage without fields"]})
        out = extract_structured_features(df)
        mat = build_numeric_matrix(out)
        assert mat.shape == (3, len(NUMERIC_COLS))
        assert np.isfinite(mat.toarray()).all()

    def test_value_is_clipped_to_bounds(self):
        df = pd.DataFrame({"sample_id": [1, 2], "catalog_content": ["Value: -50\nUnit: Ounce", "Value: 1e12\nUnit: Ounce"]})
        out = extract_structured_features(df)
        assert out["value_num"].iloc[0] >= 0.01
        assert out["value_num"].iloc[1] <= 10_000.0

    def test_non_numeric_value_field_falls_back(self):
        df = pd.DataFrame({"sample_id": [1], "catalog_content": ["Value: twelve\nUnit: Ounce"]})
        out = extract_structured_features(df)
        assert np.isfinite(out["value_num"].iloc[0])

    def test_row_order_and_count_preserved(self, synthetic_frames):
        train_df, _ = synthetic_frames
        out = extract_structured_features(train_df)
        assert len(out) == len(train_df)
        assert list(out["sample_id"]) == list(train_df["sample_id"])

    def test_advanced_features_are_numeric_and_finite(self, synthetic_frames):
        train_df, _ = synthetic_frames
        adv = extract_advanced_catalog_features(train_df)
        num = adv.select_dtypes(include=[np.number]).drop(columns=["sample_id", "price"], errors="ignore")
        assert num.shape[1] >= 10
        assert np.isfinite(num.to_numpy(dtype=float)).all()

    def test_advanced_features_handle_nan_catalog(self):
        adv = extract_advanced_catalog_features(pd.DataFrame({"catalog_content": [np.nan]}))
        assert len(adv) == 1


class TestPromptBuilder:
    def test_number_to_words(self):
        assert number_to_words(12) == "twelve"
        assert number_to_words("not a number") == "not a number"

    def test_prompt_contains_name_size_and_pack(self):
        row = extract_structured_features(pd.DataFrame({"sample_id": [1], "catalog_content": [CATALOG]})).iloc[0]
        prompt = build_llm_prompt(row)
        assert "Acme Organic Coffee" in prompt
        assert "Size: 12 oz" in prompt
        assert "three pack" in prompt

    def test_prompt_for_empty_catalog_is_string(self):
        row = extract_structured_features(pd.DataFrame({"sample_id": [1, 2], "catalog_content": [np.nan, "x"]})).iloc[0]
        assert isinstance(build_llm_prompt(row), str)

    def test_structured_features_on_all_nan_catalog_column(self):
        # Malformed payload: every catalog_content missing -> column dtype float, not object.
        out = extract_structured_features(pd.DataFrame({"sample_id": [1, 2], "catalog_content": [np.nan, np.nan]}))
        assert len(out) == 2


class TestTextVectorisation:
    def test_tfidf_fit_then_transform_shares_vocabulary(self, synthetic_frames):
        train_df, test_df = synthetic_frames
        Xtr, vec = extract_text_features(train_df["catalog_content"], max_features=500, min_df=1)
        Xte, vec2 = extract_text_features(test_df["catalog_content"], vectorizer=vec)
        assert vec2 is vec
        assert Xtr.shape[1] == Xte.shape[1] <= 500
        assert Xtr.shape[0] == len(train_df) and Xte.shape[0] == len(test_df)

    def test_tfidf_tolerates_nan_text(self):
        X, _ = extract_text_features(pd.Series([np.nan, "hello world", ""]), min_df=1)
        assert X.shape[0] == 3

    def test_feature_matrix_hstack_shapes(self, synthetic_frames):
        train_df, _ = synthetic_frames
        Xtxt, _ = extract_text_features(train_df["catalog_content"], max_features=100, min_df=1)
        Xnum = build_numeric_matrix(extract_structured_features(train_df))
        vis = np.zeros((len(train_df), 4), dtype=np.float32)
        X = build_feature_matrix(Xtxt, Xnum, vis)
        assert X.shape == (len(train_df), Xtxt.shape[1] + Xnum.shape[1] + 4)

    def test_vision_svd_is_fit_on_train_only(self):
        rng = np.random.default_rng(0)
        tr = rng.normal(size=(50, 16)).astype(np.float32)
        te = rng.normal(size=(10, 16)).astype(np.float32)
        tr_svd, te_svd = build_vision_svd_features(tr, te, n_components=4)
        tr_svd2, _ = build_vision_svd_features(tr, None, n_components=4)
        assert tr_svd.shape == (50, 4) and te_svd.shape == (10, 4)
        np.testing.assert_allclose(np.abs(tr_svd), np.abs(tr_svd2), atol=1e-4)


class TestOutlierHandling:
    def test_iqr_mask_removes_extreme_log_price_outliers(self):
        prices = np.concatenate([np.full(200, 10.0) * np.linspace(0.8, 1.2, 200), [1e9]])
        mask = compute_iqr_training_mask(prices, multiplier=3.0)
        assert mask.shape == prices.shape
        assert mask[-1] == False  # noqa: E712
        assert mask[:-1].all()

    def test_iqr_mask_keeps_everything_when_no_outliers(self):
        assert compute_iqr_training_mask(np.array([1.0, 2.0, 3.0, 4.0])).all()

    def test_iqr_mask_is_boolean(self):
        assert compute_iqr_training_mask(np.array([1.0, 2.0])).dtype == bool
