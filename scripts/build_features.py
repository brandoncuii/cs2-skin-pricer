"""Phase 2 — build the feature matrix from the clean table + validation summary.

Reads data/clean/listings.parquet, adds §7 features + the log-premium target, writes
data/features/features.parquet (+ .csv), and prints checks that the engineered
features make sense.

Run: PYTHONPATH=. .venv/bin/python scripts/build_features.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cs2pricer.features import (CATEGORICAL_COLS, FEATURE_COLS, TARGET_COL,
                                add_features)

CLEAN = Path("data/clean/listings.parquet")
OUT_DIR = Path("data/features")


def main() -> int:
    df = add_features(pd.read_parquet(CLEAN))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["id", TARGET_COL, "price_usd", "reference_usd"] + FEATURE_COLS
    feat = df[keep].copy()
    feat.to_parquet(OUT_DIR / "features.parquet", index=False)
    feat.to_csv(OUT_DIR / "features.csv", index=False)

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print(f"rows: {len(feat)} | features: {len(FEATURE_COLS)} | target: {TARGET_COL}")
    print("nulls in feature/target cols:",
          feat[[TARGET_COL] + FEATURE_COLS].isna().sum().sum())

    print("\n== Doppler phase: median price + median log_premium (should rank Ruby/BP/Sapph high) ==")
    dop = df[df.finish == "Doppler"]
    print(dop.groupby("doppler_phase")
            .agg(n=("id", "count"), med_price=("price_usd", "median"),
                 med_log_premium=(TARGET_COL, "median")).round(2)
            .sort_values("med_log_premium").to_string())

    print("\n== Case Hardened gem tier: does higher tier => higher premium? ==")
    ch = df[df.finish == "Case Hardened"]
    print(ch.groupby("ch_gem_tier")
            .agg(n=("id", "count"), med_price=("price_usd", "median"),
                 med_log_premium=(TARGET_COL, "median")).round(2).to_string())
    covered = (ch.ch_gem_tier > 0).sum()
    print(f"   CH rows with a curated tier: {covered}/{len(ch)} "
          f"({covered/len(ch):.0%}) — partial by design (v1).")

    print("\n== StatTrak premium (log_premium should be >0 vs non-ST within skin) ==")
    print(df.groupby(["skin_base", "is_stattrak"])[TARGET_COL].median().round(3).to_string())

    print("\n== Float: Case Hardened premium by exterior (float cliffs should show) ==")
    print(ch.groupby("exterior")
            .agg(n=("id", "count"), med_price=("price_usd", "median"),
                 med_log_premium=(TARGET_COL, "median")).round(2).to_string())

    print("\n== target (log_premium) distribution ==")
    print(df[TARGET_COL].describe(percentiles=[.05, .5, .95]).round(3).to_string())
    print("   (centered near 0 by construction: it's log(price / skin median))")

    print(f"\nwrote {len(feat)} rows -> {OUT_DIR/'features.parquet'} (+ .csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
