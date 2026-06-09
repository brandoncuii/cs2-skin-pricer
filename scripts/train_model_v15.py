"""v1.5 — train a sold-price model from collector disappearance data.

Reads disappeared listings from the collector SQLite DB
(data/collector/observations.db), applies the same feature engineering as v1,
and trains LightGBM quantile regression (q10, q50, q90) on log_premium.

Disappeared listings approximate sold prices (noisy — could be delist/expire).
This is the key improvement over v1's asking-price basis.

Uses a time-based train/test split (project rule): rows are sorted by
last_seen and the newest 15% are held out. Per-skin reference prices
(medians of inferred sold prices) are persisted to references.json so the
API can recover them at serve time.

Graceful handling:
  < 30 rows  → exit cleanly (not enough data)
  < 100 rows → warn but proceed

Run: PYTHONPATH=. .venv/bin/python scripts/train_model_v15.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cs2pricer.clean import parse_name
from cs2pricer.features import (CATEGORICAL_COLS, DOPPLER_PHASE, FEATURE_COLS,
                                TARGET_COL, add_features)
from cs2pricer.model import (MODEL_V15_DIR, QUANTILES, predict, save_metadata,
                             save_models, save_references, train)

DB_PATH = Path("data/collector/observations.db")
MIN_ROWS = 30
WARN_ROWS = 100


def _load_disappeared(db_path: Path) -> pd.DataFrame:
    """Load closed listings with their last observed price from the collector DB."""
    con = sqlite3.connect(db_path)

    # Take each closed listing's latest observation as its terminal state.
    # Exclude terminal_state='listed' (get_listing() proved the item is still
    # live — it just fell out of pagination, so it never sold). Keep all other
    # terminal states (gone/sold/...): delist/expire noise is an accepted v1.5
    # limitation. When the terminal observation has no price (collector records
    # state='gone' with NULL price when get_listing() fails), fall back to the
    # listing's latest non-null observed price — the last asking price, which
    # is the sold-price approximation anyway.
    query = """
        WITH terminal AS (
            SELECT listing_id, price_cents, state
            FROM (
                SELECT listing_id, price_cents, state,
                       ROW_NUMBER() OVER (
                           PARTITION BY listing_id ORDER BY observed_at DESC
                       ) AS rn
                FROM observations
            )
            WHERE rn = 1
        ),
        last_priced AS (
            SELECT listing_id, price_cents
            FROM (
                SELECT listing_id, price_cents,
                       ROW_NUMBER() OVER (
                           PARTITION BY listing_id ORDER BY observed_at DESC
                       ) AS rn
                FROM observations
                WHERE price_cents IS NOT NULL
            )
            WHERE rn = 1
        )
        SELECT
            l.id,
            l.market_hash_name,
            l.skin_base,
            l.weapon,
            l.finish,
            l.exterior,
            l.is_stattrak,
            l.def_index,
            l.paint_index,
            l.paint_seed,
            l.float_value,
            COALESCE(t.price_cents, lp.price_cents) AS price_cents,
            t.state AS terminal_state,
            l.last_seen
        FROM listings l
        JOIN terminal t ON t.listing_id = l.id
        LEFT JOIN last_priced lp ON lp.listing_id = l.id
        WHERE l.status = 'closed'
          AND t.state != 'listed'
          AND COALESCE(t.price_cents, lp.price_cents) IS NOT NULL
          AND COALESCE(t.price_cents, lp.price_cents) > 0
        ORDER BY l.last_seen DESC
    """
    df = pd.read_sql_query(query, con)
    con.close()

    if df.empty:
        return df

    df["price_usd"] = df["price_cents"] / 100.0
    df["is_stattrak"] = df["is_stattrak"].astype(bool)

    # Parse name components if skin_base is null (older collector rows).
    mask = df["skin_base"].isna() | df["skin_base"].eq("")
    if mask.any():
        parsed = df.loc[mask, "market_hash_name"].apply(parse_name).apply(pd.Series)
        for col in ["weapon", "finish", "exterior", "skin_base"]:
            df.loc[mask, col] = parsed[col]

    return df


def main() -> int:
    if not DB_PATH.exists():
        print(f"Collector DB not found at {DB_PATH}. Run the collector first.")
        return 1

    print("== Loading disappeared listings from collector DB ==")
    df = _load_disappeared(DB_PATH)
    n = len(df)
    print(f"  Found {n} closed listings with terminal prices")

    if n < MIN_ROWS:
        print(f"  Only {n} rows (need >= {MIN_ROWS}). "
              "Let the collector accumulate more data before training v1.5.")
        return 0  # clean exit, not an error

    if n < WARN_ROWS:
        print(f"  WARNING: Only {n} rows (< {WARN_ROWS}). "
              "Model will be noisy — consider waiting for more data.")

    # Drop malformed rows.
    bad = (
        df["price_usd"].isna() | df["price_usd"].le(0)
        | df["float_value"].isna()
        | df["paint_seed"].isna()
        | df["skin_base"].isna()
    )
    dropped = bad.sum()
    if dropped:
        print(f"  Dropped {dropped} malformed rows")
        df = df[~bad].copy()

    if len(df) < MIN_ROWS:
        print(f"  Only {len(df)} clean rows after filtering. Need >= {MIN_ROWS}.")
        return 0

    # Feature engineering — same pipeline as v1.
    print("\n== Adding features ==")
    df = add_features(df)
    print(f"  Features: {len(FEATURE_COLS)} columns, {len(df)} rows")

    # Time-based split (project rule): newest 15% of rows held out as test.
    df = df.sort_values("last_seen").reset_index(drop=True)
    n_test = max(1, int(len(df) * 0.15))
    test_idx = np.arange(len(df) - n_test, len(df))

    # Train.
    print("\n== Training v1.5 quantile models (q10, q50, q90) ==")
    print("  Basis: inferred sold prices (disappeared listings)")
    print(f"  Time-based split: newest {n_test} rows (by last_seen) held out")
    result = train(df, test_indices=test_idx)
    models = result["models"]

    save_models(models, MODEL_V15_DIR)
    save_metadata(result, df, MODEL_V15_DIR)
    # Persist per-skin reference prices (medians of inferred sold prices) so
    # the API can normalize predictions the same way at serve time.
    refs = {k: float(v) for k, v in
            df.groupby("skin_base")["reference_usd"].first().items()}
    save_references(refs, MODEL_V15_DIR)
    print(f"  References saved to {MODEL_V15_DIR}/references.json "
          f"({len(refs)} skins)")
    print(f"  Models saved to {MODEL_V15_DIR}/")
    print(f"  Best iterations: "
          + ", ".join(f"q{int(q*100)}={models[q].best_iteration}" for q in models))

    # Predictions for sanity checks.
    preds = predict(models, df)
    df = df.assign(**preds)

    # --- Sanity Check 1: Doppler phase ordering ---
    dop = df[df["finish"] == "Doppler"]
    if len(dop) > 0:
        print("\n== Sanity Check: Doppler phase ordering ==")
        phase_order = (dop.groupby("doppler_phase")["q50"]
                       .median().sort_values(ascending=False))
        print(phase_order.round(3).to_string())

    # --- Sanity Check 2: Quantile crossings ---
    print("\n== Sanity Check: Quantile crossings (post-clamp) ==")
    crossings = ((df["q10"] > df["q50"]) | (df["q50"] > df["q90"])).sum()
    print(f"  Crossings: {crossings}/{len(df)} "
          f"({'PASS' if crossings == 0 else 'WARN'})")

    # --- Sanity Check 3: Coverage ---
    print("\n== Sanity Check: Held-out [q10, q90] coverage ==")
    test_mask = np.zeros(len(df), dtype=bool)
    test_mask[result["test_idx"]] = True
    test_df = df[test_mask]
    if len(test_df) > 0:
        actual = test_df[TARGET_COL]
        in_range = ((actual >= test_df["q10"]) & (actual <= test_df["q90"])).mean()
        print(f"  {in_range:.1%} of test rows fall within [q10, q90]")

    # --- USD summary ---
    print("\n== Predicted USD ranges by skin (median) ==")
    usd_summary = (df.groupby("skin_base")[["low_usd", "mid_usd", "high_usd"]]
                   .median().round(0))
    print(usd_summary.to_string())

    # --- Feature importance ---
    print("\n== Feature Importance (gain, averaged across quantiles) ==")
    print(result["feature_importance"][["feature", "gain_mean"]]
          .head(10).to_string(index=False))

    print(f"\nv1.5 training complete. Artifacts in {MODEL_V15_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
