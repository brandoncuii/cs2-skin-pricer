"""Tests for cs2pricer.features module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cs2pricer.features import (
    DOPPLER_PHASE,
    FEATURE_COLS,
    KARAMBIT_CH_GEM_TIER,
    TARGET_COL,
    add_features,
)


@pytest.fixture
def feature_df() -> pd.DataFrame:
    """Minimal DataFrame suitable for add_features()."""
    return pd.DataFrame(
        [
            {
                "skin_base": "Karambit | Doppler",
                "weapon": "Karambit",
                "finish": "Doppler",
                "exterior": "Factory New",
                "def_index": 507,
                "is_stattrak": False,
                "float_value": 0.03,
                "paint_index": 418,
                "paint_seed": 100,
                "price_usd": 1500.0,
            },
            {
                "skin_base": "Karambit | Doppler",
                "weapon": "Karambit",
                "finish": "Doppler",
                "exterior": "Factory New",
                "def_index": 507,
                "is_stattrak": False,
                "float_value": 0.05,
                "paint_index": 415,
                "paint_seed": 200,
                "price_usd": 5000.0,
            },
            {
                "skin_base": "Karambit | Case Hardened",
                "weapon": "Karambit",
                "finish": "Case Hardened",
                "exterior": "Minimal Wear",
                "def_index": 507,
                "is_stattrak": False,
                "float_value": 0.12,
                "paint_index": 44,
                "paint_seed": 387,
                "price_usd": 3000.0,
            },
        ]
    )


class TestDopplerPhase:
    def test_all_phases_present(self):
        expected_phases = {
            "Ruby",
            "Sapphire",
            "Black Pearl",
            "Phase 1",
            "Phase 2",
            "Phase 3",
            "Phase 4",
        }
        assert set(DOPPLER_PHASE.values()) == expected_phases

    def test_ruby_mapping(self):
        assert DOPPLER_PHASE[415] == "Ruby"

    def test_sapphire_mapping(self):
        assert DOPPLER_PHASE[416] == "Sapphire"

    def test_black_pearl_mapping(self):
        assert DOPPLER_PHASE[417] == "Black Pearl"


class TestKarambitCHGemTier:
    def test_tier_3_seeds(self):
        for seed in (387, 853, 670):
            assert KARAMBIT_CH_GEM_TIER[seed] == 3

    def test_tier_values_in_range(self):
        for tier in KARAMBIT_CH_GEM_TIER.values():
            assert tier in (1, 2, 3)


class TestAddFeatures:
    def test_adds_expected_columns(self, feature_df):
        out = add_features(feature_df)
        assert "dist_to_boundary" in out.columns
        assert "float_pctile_in_skin" in out.columns
        assert "doppler_phase" in out.columns
        assert "ch_gem_tier" in out.columns
        assert "reference_usd" in out.columns
        assert "log_premium" in out.columns

    def test_doppler_phase_mapped(self, feature_df):
        out = add_features(feature_df)
        # paint_index 418 -> Phase 1
        assert out.iloc[0]["doppler_phase"] == "Phase 1"
        # paint_index 415 -> Ruby
        assert out.iloc[1]["doppler_phase"] == "Ruby"

    def test_ch_gem_tier_set(self, feature_df):
        out = add_features(feature_df)
        # seed 387 for Karambit Case Hardened -> tier 3
        assert out.iloc[2]["ch_gem_tier"] == 3

    def test_non_ch_gem_tier_zero(self, feature_df):
        out = add_features(feature_df)
        # Doppler rows should have ch_gem_tier == 0
        assert out.iloc[0]["ch_gem_tier"] == 0

    def test_dist_to_boundary_range(self, feature_df):
        out = add_features(feature_df)
        # All values should be >= 0
        assert (out["dist_to_boundary"] >= 0).all()

    def test_reference_usd_is_median(self, feature_df):
        out = add_features(feature_df)
        # For the two Karambit Doppler rows, reference = median of their prices
        kd_ref = out[out["skin_base"] == "Karambit | Doppler"]["reference_usd"].iloc[0]
        expected = np.median([1500.0, 5000.0])
        assert kd_ref == pytest.approx(expected)

    def test_log_premium_sign(self, feature_df):
        out = add_features(feature_df)
        # Item priced above median -> positive log_premium
        ruby_row = out.iloc[1]
        assert ruby_row["log_premium"] > 0

    def test_feature_cols_present(self, feature_df):
        out = add_features(feature_df)
        for col in FEATURE_COLS:
            assert col in out.columns
        assert TARGET_COL in out.columns
