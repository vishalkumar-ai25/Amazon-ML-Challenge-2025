"""Tests for Lever 3 (Prompt Enrichment) and Lever 4 (Physical PPU Priors)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    NUMERIC_COLS,
    TIER_LABELS,
    build_llm_prompt,
    build_numeric_matrix,
    extract_category,
    extract_structured_features,
)


class TestCategoryClassification:
    def test_category_accuracy_on_known_domains(self):
        """Verify keyword classification identifies major product categories."""
        samples = [
            ("Whole bean arabica coffee 12 oz", "Grocery"),
            ("Organic Green Tea Bags 100 Count", "Grocery"),
            ("L'Oreal Paris Revitalift Hyaluronic Acid Serum 1 fl oz", "Beauty"),
            ("Matte Velvet Lipstick Long Lasting", "Beauty"),
            ("Vitamin D3 5000 IU Immune Support Softgels", "Health"),
            ("Whey Protein Powder Chocolate 2 lb", "Health"),
            ("Anker USB-C to Lightning Cable 6ft Fast Charging", "Electronics"),
            ("Wireless Bluetooth Earbuds Noise Cancelling", "Electronics"),
            ("Nonstick Frying Pan Cookware 10 Inch", "Home & Kitchen"),
            ("Cotton Bath Towels 4 Pack", "Home & Kitchen"),
            ("Heavy Duty Cordless Power Drill Toolkit", "Tools"),
            ("Men's Athletic Running Shoes Sneakers", "Clothing"),
            ("Sterling Silver Diamond Pendant Necklace", "Jewelry"),
        ]
        for text, expected in samples:
            assert extract_category(text) == expected, f"Failed for text: {text}"

    def test_category_fallback_on_unrecognized_text(self):
        """Verify unclassifiable text safely returns 'Other'."""
        assert extract_category("XYZ Unknown Object") == "Other"
        assert extract_category("") == "Other"
        assert extract_category(None) == "Other"
        assert extract_category(np.nan) == "Other"


class TestPromptEnrichment:
    def test_full_prompt_structure_with_tier(self):
        """Verify complete enriched prompt formatting with semantic tier."""
        row = {
            "item_name": "Gucci Dionysus Leather Shoulder Bag",
            "category": "Clothing",
            "brand": "Gucci",
            "tier": 4,
            "value_num": 1.0,
            "unit_normalized": "count",
            "pack_qty": 1.0,
            "description_raw": "Made in Italy with textured tiger head closure.",
        }
        prompt = build_llm_prompt(row)
        assert "Category: Clothing" in prompt
        assert "Brand: Gucci (Tier: Luxury)" in prompt
        assert "Product: Gucci Dionysus Leather Shoulder Bag" in prompt
        assert "Size: 1 count" in prompt
        assert "Specifications: Made in Italy" in prompt

    def test_prompt_budget_tier_formatting(self):
        """Verify integer 0 maps to 'Budget'."""
        row = {
            "item_name": "Amazon Basics USB-C Cable",
            "category": "Electronics",
            "brand": "Amazon Basics",
            "tier": 0,
            "value_num": 6.0,
            "unit_normalized": "ft",
            "pack_qty": 2.0,
        }
        prompt = build_llm_prompt(row)
        assert "Brand: Amazon Basics (Tier: Budget)" in prompt
        assert "Multipack: two pack" in prompt

    def test_prompt_without_tier_omits_parentheses(self):
        """Verify brand without tier emits clean 'Brand: X' without '(Tier: ...)'."""
        row = {
            "item_name": "Sony Wireless Headphones",
            "category": "Electronics",
            "brand": "Sony",
            "value_num": 1.0,
            "unit_normalized": "count",
        }
        prompt = build_llm_prompt(row)
        assert "Brand: Sony" in prompt
        assert "(Tier:" not in prompt

    def test_backward_compatibility_with_minimal_input(self):
        """Verify compatibility when brand/category are omitted entirely."""
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


class TestPPUAndPhysicalDensity:
    def test_log_unit_density_physical_directionality(self):
        """Verify density separates liquids (negative) from solids (positive)."""
        df = pd.DataFrame({
            "sample_id": [1, 2, 3],
            "catalog_content": [
                "Item Name: Whole Milk 1 Gallon\nValue: 128\nUnit: Fl Oz",
                "Item Name: Steel Barbell Weight Plate\nValue: 45\nUnit: Pound",
                "Item Name: Cotton Socks 6 Pack\nValue: 6\nUnit: Count",
            ],
        })
        out = extract_structured_features(df)
        assert "log_unit_density" in out.columns
        assert "log_unit_density" in NUMERIC_COLS

        # Milk (liquid): ml > 0, grams == 0 -> negative density
        assert out.loc[0, "log_unit_density"] < 0
        # Steel plate (solid): grams > 0, ml == 0 -> positive density
        assert out.loc[1, "log_unit_density"] > 0
        # Socks (count): grams == 0, ml == 0 -> zero density
        assert out.loc[2, "log_unit_density"] == pytest.approx(0.0, abs=1e-5)

    def test_ppu_priors_safe_denominator_and_finite_numerics(self):
        """Verify PPU calculations are strictly finite even with 0 quantity."""
        train_df = pd.DataFrame({
            "sample_id": [1, 2],
            "catalog_content": [
                "Item Name: Acme Coffee\nValue: 12\nUnit: Ounce",
                "Item Name: Beta Cable\nValue: 0\nUnit: Other",
            ],
        })
        out = extract_structured_features(train_df)

        # Mock OOF prices
        out["cat_oof_price"] = np.array([np.log(12.0), np.log(20.0)])
        out["brand_oof_price"] = np.array([np.log(14.0), np.log(18.0)])

        safe_qty = np.maximum(out["std_quantity"].fillna(1.0).astype(float), 1.0)
        safe_log_qty = np.maximum(out["log_std_quantity"].fillna(0.0).astype(float), 0.0)

        out["cat_estimated_ppu"] = np.exp(out["cat_oof_price"]) / safe_qty
        out["brand_estimated_ppu"] = np.exp(out["brand_oof_price"]) / safe_qty
        out["cat_implied_log_ppu"] = out["cat_oof_price"] - safe_log_qty
        out["brand_implied_log_ppu"] = out["brand_oof_price"] - safe_log_qty

        # All values must be finite and positive
        assert np.isfinite(out["cat_estimated_ppu"]).all()
        assert np.isfinite(out["brand_estimated_ppu"]).all()
        assert (out["cat_estimated_ppu"] > 0).all()
        assert (out["brand_estimated_ppu"] > 0).all()
        assert np.isfinite(out["cat_implied_log_ppu"]).all()
        assert np.isfinite(out["brand_implied_log_ppu"]).all()
