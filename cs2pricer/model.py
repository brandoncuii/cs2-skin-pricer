"""Phase 3 — LightGBM quantile-regression model for log-premium pricing.

Trains three models (q10, q50, q90) on the log-premium target to produce a
fair-value *range* for each listing. Quantile crossings are clamped post-hoc.

Design choices (PLAN.md §8):
  - GBT (LightGBM) because this is tabular with threshold effects (exterior
    cliffs, discrete phase regimes) that trees model naturally.
  - Quantile regression gives a range; point estimates hide uncertainty in a
    fat-tailed market.
  - v1 uses a simple random held-out split for plumbing checks only (no time
    axis in a single snapshot). Do NOT report it as generalization evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import CATEGORICAL_COLS, FEATURE_COLS, TARGET_COL

QUANTILES = (0.10, 0.50, 0.90)
MODEL_DIR = Path("data/model")
MODEL_V15_DIR = Path("data/model_v15")

# LightGBM parameters shared across quantile models.
_BASE_PARAMS: dict[str, Any] = {
    "objective": "quantile",
    "metric": "quantile",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}


def _prepare_dataset(df: pd.DataFrame) -> lgb.Dataset:
    """Build a LightGBM Dataset from the feature matrix."""
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL]
    # LightGBM handles categoricals natively; just cast to 'category'.
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype("category")
    return lgb.Dataset(X, label=y, categorical_feature=CATEGORICAL_COLS, free_raw_data=False)


def train(df: pd.DataFrame, *, num_boost_round: int = 500,
          early_stopping_rounds: int = 50,
          test_fraction: float = 0.15,
          seed: int = 42) -> dict[str, Any]:
    """Train quantile models and return results dict.

    Returns:
        {
            "models": {0.1: Booster, 0.5: Booster, 0.9: Booster},
            "train_idx": ndarray, "test_idx": ndarray,
            "feature_importance": DataFrame,
        }
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    perm = rng.permutation(n)
    n_test = int(n * test_fraction)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    train_ds = _prepare_dataset(df.iloc[train_idx])
    test_ds = _prepare_dataset(df.iloc[test_idx])

    models: dict[float, lgb.Booster] = {}
    for q in QUANTILES:
        params = {**_BASE_PARAMS, "alpha": q}
        bst = lgb.train(
            params,
            train_ds,
            num_boost_round=num_boost_round,
            valid_sets=[test_ds],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        models[q] = bst

    # Feature importance (gain-based, averaged across quantile models).
    imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        **{f"gain_q{int(q*100)}": models[q].feature_importance(importance_type="gain")
           for q in QUANTILES},
    })
    imp["gain_mean"] = imp[[f"gain_q{int(q*100)}" for q in QUANTILES]].mean(axis=1)
    imp = imp.sort_values("gain_mean", ascending=False).reset_index(drop=True)

    return {
        "models": models,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "feature_importance": imp,
    }


def predict(models: dict[float, lgb.Booster], df: pd.DataFrame) -> pd.DataFrame:
    """Predict log-premium quantiles and convert back to USD ranges.

    Returns a DataFrame with columns: q10, q50, q90 (log-premium space) and
    low_usd, mid_usd, high_usd (dollar space, using reference_usd).
    """
    X = df[FEATURE_COLS].copy()
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype("category")

    preds = pd.DataFrame(index=df.index)
    for q in QUANTILES:
        preds[f"q{int(q*100)}"] = models[q].predict(X)

    # Clamp quantile crossings (PLAN.md §8): enforce q10 <= q50 <= q90.
    preds["q10"] = preds[["q10", "q50"]].min(axis=1)
    preds["q90"] = preds[["q50", "q90"]].max(axis=1)

    # Convert from log-premium back to USD.
    ref = df["reference_usd"].values
    preds["low_usd"] = np.exp(preds["q10"]) * ref
    preds["mid_usd"] = np.exp(preds["q50"]) * ref
    preds["high_usd"] = np.exp(preds["q90"]) * ref

    return preds


def save_models(models: dict[float, lgb.Booster], out_dir: Path = MODEL_DIR) -> None:
    """Save trained boosters to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for q, bst in models.items():
        bst.save_model(str(out_dir / f"lgb_q{int(q*100)}.txt"))


def load_models(model_dir: Path = MODEL_DIR) -> dict[float, lgb.Booster]:
    """Load saved boosters from disk."""
    models: dict[float, lgb.Booster] = {}
    for q in QUANTILES:
        path = model_dir / f"lgb_q{int(q*100)}.txt"
        models[q] = lgb.Booster(model_file=str(path))
    return models


def load_models_v15() -> dict[float, lgb.Booster] | None:
    """Load v1.5 models if they exist, else return None."""
    if not MODEL_V15_DIR.exists():
        return None
    try:
        return load_models(MODEL_V15_DIR)
    except Exception:
        return None


def v15_available() -> bool:
    """Check whether v1.5 model artifacts exist on disk."""
    return all(
        (MODEL_V15_DIR / f"lgb_q{int(q*100)}.txt").exists()
        for q in QUANTILES
    )


def save_metadata(result: dict[str, Any], df: pd.DataFrame,
                  out_dir: Path = MODEL_DIR) -> None:
    """Save training metadata (for reproducibility / inspection)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "n_train": len(result["train_idx"]),
        "n_test": len(result["test_idx"]),
        "features": FEATURE_COLS,
        "target": TARGET_COL,
        "quantiles": list(QUANTILES),
        "best_iterations": {
            f"q{int(q*100)}": result["models"][q].best_iteration
            for q in QUANTILES
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    result["feature_importance"].to_csv(out_dir / "feature_importance.csv", index=False)
