"""Phase 3 — train the quantile-regression model + sanity-check report.

Reads data/features/features.parquet, trains LightGBM quantile models (q10/q50/q90),
saves artifacts to data/model/, and prints a sanity-check report per PLAN.md §8:
  - Doppler phase ordering (Ruby/Sapphire/BP should predict higher than low phases)
  - Float monotonicity within Case Hardened (lower float → higher predicted premium)
  - StatTrak premium sign
  - Quantile crossing rate after clamping (should be 0)
  - Held-out coverage (what fraction of test rows fall within [q10, q90])

This is NOT a generalization claim — v1 has no time axis (§8). It's a plumbing check.

Run: PYTHONPATH=. .venv/bin/python scripts/train_model.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cs2pricer.features import FEATURE_COLS, TARGET_COL, add_features
from cs2pricer.model import (MODEL_DIR, predict, save_metadata, save_models,
                             train)

FEATURES_PATH = Path("data/features/features.parquet")
CLEAN_PATH = Path("data/clean/listings.parquet")


def main() -> int:
    # Load the full clean dataset (need reference_usd for USD conversion).
    clean_df = add_features(pd.read_parquet(CLEAN_PATH))

    print("== Training quantile models (q10, q50, q90) ==")
    result = train(clean_df)
    models = result["models"]

    save_models(models)
    save_metadata(result, clean_df)
    print(f"  Models saved to {MODEL_DIR}/")
    print(f"  Best iterations: "
          + ", ".join(f"q{int(q*100)}={models[q].best_iteration}" for q in models))

    # --- Predictions on full dataset for sanity checks ---
    preds = predict(models, clean_df)
    clean_df = clean_df.assign(**preds)

    # --- Sanity Check 1: Doppler phase ordering ---
    print("\n== Sanity Check 1: Doppler phase predicted premium ordering ==")
    print("   (Expected: Ruby/Sapphire/Black Pearl >> Phase 1-4)")
    dop = clean_df[clean_df["finish"] == "Doppler"]
    phase_order = (dop.groupby("doppler_phase")["q50"]
                   .median().sort_values(ascending=False))
    print(phase_order.round(3).to_string())
    # Check Ruby/Sapphire/BP are in the top 3.
    top3 = set(phase_order.index[:3])
    premium_phases = {"Ruby", "Sapphire", "Black Pearl"}
    phase_ok = premium_phases.issubset(top3)
    print(f"   {'PASS' if phase_ok else 'WARN'}: premium phases in top 3 = {phase_ok}")

    # --- Sanity Check 2: Float monotonicity for Case Hardened ---
    print("\n== Sanity Check 2: Case Hardened — lower float → higher predicted premium ==")
    ch = clean_df[clean_df["finish"] == "Case Hardened"].copy()
    ch["float_bin"] = pd.qcut(ch["float_value"], 5, labels=False, duplicates="drop")
    float_med = ch.groupby("float_bin")["q50"].median()
    print(float_med.round(3).to_string())
    # Lower bin (lower float) should generally have higher premium.
    float_ok = float_med.iloc[0] >= float_med.iloc[-1]
    print(f"   {'PASS' if float_ok else 'WARN'}: lowest float bin >= highest = {float_ok}")

    # --- Sanity Check 3: StatTrak premium sign ---
    print("\n== Sanity Check 3: StatTrak predicted premium vs. non-ST ==")
    st_med = clean_df.groupby("is_stattrak")["q50"].median()
    print(st_med.round(3).to_string())
    # ST can go either way in this market (knives are weird); just report.
    print(f"   ST median q50: {st_med.get(True, 0):.3f} vs non-ST: {st_med.get(False, 0):.3f}")

    # --- Sanity Check 4: Quantile crossing rate ---
    print("\n== Sanity Check 4: Quantile crossings (post-clamp, should be 0) ==")
    crossings = ((clean_df["q10"] > clean_df["q50"]) |
                 (clean_df["q50"] > clean_df["q90"])).sum()
    print(f"   Crossings: {crossings}/{len(clean_df)} "
          f"({'PASS' if crossings == 0 else 'WARN'})")

    # --- Sanity Check 5: Held-out coverage ---
    print("\n== Sanity Check 5: Held-out [q10, q90] coverage ==")
    test_mask = np.zeros(len(clean_df), dtype=bool)
    test_mask[result["test_idx"]] = True
    test_df = clean_df[test_mask]
    actual = test_df[TARGET_COL]
    in_range = ((actual >= test_df["q10"]) & (actual <= test_df["q90"])).mean()
    print(f"   {in_range:.1%} of test rows fall within [q10, q90] "
          f"(nominal 80%; close = good plumbing)")

    # --- Sanity Check 6: USD ranges make sense ---
    print("\n== Sanity Check 6: Predicted USD ranges by skin (median) ==")
    usd_summary = (clean_df.groupby("skin_base")[["low_usd", "mid_usd", "high_usd"]]
                   .median().round(0))
    print(usd_summary.to_string())

    # --- Feature importance ---
    print("\n== Feature Importance (gain, averaged across quantiles) ==")
    print(result["feature_importance"][["feature", "gain_mean"]]
          .head(10).to_string(index=False))

    print(f"\nPhase 3 complete. Artifacts in {MODEL_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
