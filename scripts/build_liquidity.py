"""Build empirical days-to-sale stats for the Find Deals actionability ranking.

Reads SOLD listings from the collector DB, computes observed days-on-market
(first_seen -> last_seen; 3h-12h poll granularity, right-censored — see
cs2pricer/liquidity.py for the full caveats), buckets by skin and by
ask-vs-v1.5-mid, and writes data/liquidity/days_to_sell.json for app.py.

The rel-to-mid buckets use the v1.5 model's q50 (mid) prediction. The v1.5
model was trained on these same sold rows, so the mids are in-sample — fine
for descriptive bucketing, but don't read the buckets as out-of-sample skill.
If v1.5 artifacts are missing, the bucket level is skipped (skin/global only).

Run: PYTHONPATH=. .venv/bin/python scripts/build_liquidity.py
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from cs2pricer.features import add_features
from cs2pricer.liquidity import (DB_PATH, STATS_PATH, bucket_rel_to_mid,
                                 compute_stats, load_censored_ages,
                                 load_sold_durations, save_stats)
from cs2pricer.model import (MODEL_V15_DIR, load_models_v15, load_references,
                             predict, v15_available)

MIN_SOLD_ROWS = 30


def add_rel_bucket(sold: pd.DataFrame) -> pd.DataFrame:
    """Score sold rows with the v1.5 model and bucket ask vs predicted mid."""
    df = add_features(sold)
    # Serve-time references (per-skin medians of inferred sold prices), same as
    # v1.5 training; fall back to this batch's per-skin median for unseen skins.
    refs = load_references(MODEL_V15_DIR) or {}
    df["reference_usd"] = df["skin_base"].map(refs)
    df["reference_usd"] = df["reference_usd"].fillna(
        df.groupby("skin_base")["price_usd"].transform("median"))
    preds = predict(load_models_v15(), df)
    df = df.assign(**preds)
    df["rel_to_mid"] = df["price_usd"] / df["mid_usd"] - 1
    df["rel_bucket"] = df["rel_to_mid"].apply(bucket_rel_to_mid)
    return df


def main() -> int:
    if not DB_PATH.exists():
        print(f"Collector DB not found at {DB_PATH}. Run the collector first.")
        return 1

    sold = load_sold_durations()
    censored = load_censored_ages()
    print(f"== Sold listings: {len(sold)} | right-censored (still live): {len(censored)} ==")
    if len(sold) < MIN_SOLD_ROWS:
        print(f"  Only {len(sold)} sold rows (need >= {MIN_SOLD_ROWS}). "
              "Let the collector accumulate more data.")
        return 0

    if v15_available():
        sold = add_rel_bucket(sold)
        print("  Bucketed ask vs v1.5 mid (in-sample mids — descriptive only)")
    else:
        print("  v1.5 model not found — skipping rel-to-mid buckets (skin/global only)")

    stats = compute_stats(sold, censored)
    save_stats(stats)
    print(f"  Stats written to {STATS_PATH}\n")

    g = stats["global"]
    print(f"{'group':<48}{'n':>5}{'median':>8}{'q25':>7}{'q75':>7}")
    print("-" * 75)
    print(f"{'GLOBAL (all sold)':<48}{g['n']:>5}{g['median_days']:>8.2f}"
          f"{g['q25_days']:>7.2f}{g['q75_days']:>7.2f}")
    for skin, s in sorted(stats["by_skin"].items()):
        print(f"{skin:<48}{s['n']:>5}{s['median_days']:>8.2f}"
              f"{s['q25_days']:>7.2f}{s['q75_days']:>7.2f}")
        for bucket, b in stats["by_skin_bucket"].get(skin, {}).items():
            print(f"{'  ' + bucket:<48}{b['n']:>5}{b['median_days']:>8.2f}"
                  f"{b['q25_days']:>7.2f}{b['q75_days']:>7.2f}")
    print("-" * 75)
    print(f"(buckets with < {stats['min_bucket_rows']} sold rows omitted; "
          "lookups fall back skin -> global)")
    print(f"Censoring: {g['n_censored']} still-live listings "
          f"(median age so far {g['censored_median_age_days']} days) haven't sold yet — "
          "medians above understate true time-to-sale. Polls every 3-12h; "
          "sales faster than one poll are never observed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
