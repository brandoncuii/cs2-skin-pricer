"""Tests for cs2pricer.model module (small synthetic data, no real model files)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cs2pricer.features import FEATURE_COLS, add_features
from cs2pricer.model import (
    QUANTILES,
    load_models,
    load_references,
    predict,
    save_models,
    save_references,
    train,
)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Build a small synthetic DataFrame with all required features."""
    rng = np.random.default_rng(42)
    n = 100
    skins = ["Karambit | Doppler", "M9 Bayonet | Doppler"]
    weapons = ["Karambit", "M9 Bayonet"]
    finishes = ["Doppler", "Doppler"]
    exteriors = ["Factory New", "Minimal Wear"]

    data = {
        "skin_base": rng.choice(skins, n),
        "weapon": [
            weapons[0] if s == skins[0] else weapons[1] for s in rng.choice(skins, n)
        ],
        "finish": rng.choice(finishes, n),
        "exterior": rng.choice(exteriors, n),
        "def_index": rng.choice([507, 508], n),
        "is_stattrak": rng.choice([True, False], n),
        "float_value": rng.uniform(0.0, 0.07, n),
        "paint_index": rng.choice([415, 416, 417, 418, 419, 420, 421], n),
        "paint_seed": rng.integers(1, 999, n),
        "price_usd": rng.uniform(500, 5000, n),
    }
    df = pd.DataFrame(data)
    # Ensure weapon matches skin_base
    df["weapon"] = df["skin_base"].apply(lambda s: s.split(" | ")[0])
    return add_features(df)


class TestTrain:
    def test_returns_models_for_all_quantiles(self, synthetic_df):
        result = train(synthetic_df, num_boost_round=10, early_stopping_rounds=5)
        assert set(result["models"].keys()) == set(QUANTILES)

    def test_train_test_split(self, synthetic_df):
        result = train(synthetic_df, num_boost_round=10, early_stopping_rounds=5)
        total = len(result["train_idx"]) + len(result["test_idx"])
        assert total == len(synthetic_df)
        # No overlap
        overlap = set(result["train_idx"]) & set(result["test_idx"])
        assert len(overlap) == 0

    def test_custom_test_indices(self, synthetic_df):
        test_idx = np.array([0, 1, 2, 3, 4])
        result = train(
            synthetic_df,
            num_boost_round=10,
            early_stopping_rounds=5,
            test_indices=test_idx,
        )
        np.testing.assert_array_equal(result["test_idx"], test_idx)

    def test_feature_importance_returned(self, synthetic_df):
        result = train(synthetic_df, num_boost_round=10, early_stopping_rounds=5)
        imp = result["feature_importance"]
        assert "feature" in imp.columns
        assert "gain_mean" in imp.columns
        assert len(imp) == len(FEATURE_COLS)


class TestPredict:
    def test_predict_shape(self, synthetic_df):
        result = train(synthetic_df, num_boost_round=10, early_stopping_rounds=5)
        preds = predict(result["models"], synthetic_df)
        assert len(preds) == len(synthetic_df)
        assert "q10" in preds.columns
        assert "q50" in preds.columns
        assert "q90" in preds.columns
        assert "low_usd" in preds.columns
        assert "mid_usd" in preds.columns
        assert "high_usd" in preds.columns

    def test_quantile_ordering(self, synthetic_df):
        result = train(synthetic_df, num_boost_round=10, early_stopping_rounds=5)
        preds = predict(result["models"], synthetic_df)
        # After clamping: q10 <= q50 <= q90
        assert (preds["q10"] <= preds["q50"] + 1e-9).all()
        assert (preds["q50"] <= preds["q90"] + 1e-9).all()

    def test_usd_positive(self, synthetic_df):
        result = train(synthetic_df, num_boost_round=10, early_stopping_rounds=5)
        preds = predict(result["models"], synthetic_df)
        assert (preds["low_usd"] > 0).all()
        assert (preds["mid_usd"] > 0).all()
        assert (preds["high_usd"] > 0).all()


class TestSaveLoadRoundTrip:
    def test_save_load_models(self, synthetic_df, tmp_path):
        result = train(synthetic_df, num_boost_round=10, early_stopping_rounds=5)
        save_models(result["models"], tmp_path)

        loaded = load_models(tmp_path)
        assert set(loaded.keys()) == set(QUANTILES)

        # Predictions should be identical
        preds_orig = predict(result["models"], synthetic_df)
        preds_loaded = predict(loaded, synthetic_df)
        pd.testing.assert_frame_equal(preds_orig, preds_loaded)

    def test_save_load_references(self, tmp_path):
        refs = {"Karambit | Doppler": 1500.0, "M9 Bayonet | Doppler": 2000.0}
        save_references(refs, tmp_path)
        loaded = load_references(tmp_path)
        assert loaded == refs

    def test_load_references_missing(self, tmp_path):
        result = load_references(tmp_path / "nonexistent")
        assert result is None
