"""Train a v1-style ASKING-price model over ALL knives (not just the locked 4).

This is the broad-coverage sibling of scripts/train_model.py. Where v1 trains on
the 4 locked skins and v1.5 (scripts/train_model_v15.py) trains on inferred SOLD
prices, this script trains on the LATEST ASKING price of every listing the
collector has ever recorded — across all knife types.

The target is therefore an ASKING-price signal, NOT true fair value (PLAN.md §5):
the model learns "priced below comparable current/known asks", which inherits the
asking-price asterisk (selection bias points high). It must be labeled as such.

Row sourcing:
  ONE row per listing = that listing's MOST RECENT observation (max observed_at),
  price_usd = price_cents / 100.0, joined to the listing's attributes. ALL knives
  are included (no restriction to the locked SKINS).

Same feature pipeline as v1 (add_features → reference_usd + log_premium) and the
same quantile regression (q10/q50/q90). Uses a time-based train/test split
(project rule): rows sorted by observation time, newest 15% held out as test.

Writes data/model_all/{lgb_q*.txt, references.json, metadata.json,
feature_importance.csv, supported_skins.json}. The supported_skins.json file
records which skins have enough rows (>= MIN_ROWS) to be priced with confidence.

Run: PYTHONPATH=. .venv/bin/python scripts/train_model_all.py [db_path] [out_dir]
  db_path defaults to data/collector/observations.db (override to point at a
  fixture); out_dir defaults to data/model_all.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cs2pricer.clean import parse_name
from cs2pricer.features import FEATURE_COLS, TARGET_COL, add_features
from cs2pricer.model import (predict, save_metadata, save_models,
                             save_references, train)

DB_PATH = Path("data/collector/observations.db")
MIN_ROWS = 30  # per-skin support threshold for confident pricing
MODEL_ALL_DIR = Path("data/model_all")


def _load_latest_asks(db_path: Path) -> pd.DataFrame:
    """One row per listing: its most-recent observation (the latest asking price).

    This is the asking-price snapshot per live/known listing — for buy_now
    listings the latest observed price IS the current ask. ALL knives included.
    """
    con = sqlite3.connect(db_path)

    # Each listing's latest observation (max observed_at) is its current/known ask.
    query = """
        WITH latest AS (
            SELECT listing_id, observed_at, price_cents, state
            FROM (
                SELECT listing_id, observed_at, price_cents, state,
                       ROW_NUMBER() OVER (
                           PARTITION BY listing_id ORDER BY observed_at DESC
                       ) AS rn
                FROM observations
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
            o.price_cents,
            o.observed_at
        FROM listings l
        JOIN latest o ON o.listing_id = l.id
        WHERE o.price_cents IS NOT NULL
          AND o.price_cents > 0
        ORDER BY o.observed_at DESC
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
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DB_PATH
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else MODEL_ALL_DIR

    if not db_path.exists():
        print(f"Collector DB not found at {db_path}. Run the collector first.")
        return 1

    print("== Loading latest asking prices from collector DB (ALL knives) ==")
    print("  Basis: ASKING prices (latest ask per listing) — NOT true fair value")
    df = _load_latest_asks(db_path)
    n = len(df)
    print(f"  Found {n} listings with a latest ask")

    if n < MIN_ROWS:
        print(f"  Only {n} rows (need >= {MIN_ROWS}). "
              "Let the collector accumulate more data before training.")
        return 0  # clean exit, not an error

    # Drop malformed rows (same as train_model_v15.py).
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

    # Feature engineering — same pipeline as v1 (reference_usd + log_premium).
    print("\n== Adding features ==")
    df = add_features(df)
    print(f"  Features: {len(FEATURE_COLS)} columns, {len(df)} rows")

    # Time-based split (project rule): newest 15% of rows held out as test.
    df = df.sort_values("observed_at").reset_index(drop=True)
    n_test = max(1, int(len(df) * 0.15))
    test_idx = np.arange(len(df) - n_test, len(df))

    # Train.
    print("\n== Training quantile models (q10, q50, q90) ==")
    print("  Basis: ASKING prices (latest ask per listing), all knives")
    print(f"  Time-based split: newest {n_test} rows (by observed_at) held out")
    result = train(df, test_indices=test_idx)
    models = result["models"]

    save_models(models, out_dir)
    save_metadata(result, df, out_dir)
    # Per-skin reference anchors = median ask (reference_usd is already that
    # per-skin median after add_features). Persist so the API normalizes the
    # same way at serve time.
    refs = {k: float(v) for k, v in
            df.groupby("skin_base")["reference_usd"].first().items()}
    save_references(refs, out_dir)
    print(f"  References saved to {out_dir}/references.json ({len(refs)} skins)")
    print(f"  Models saved to {out_dir}/")
    print(f"  Best iterations: "
          + ", ".join(f"q{int(q*100)}={models[q].best_iteration}" for q in models))

    # --- Supported-skins manifest (per-skin row counts used in training) ---
    counts = df.groupby("skin_base").size()
    supported = {k: int(v) for k, v in counts.items() if v >= MIN_ROWS}
    thin = {k: int(v) for k, v in counts.items() if 0 < v < MIN_ROWS}
    manifest = {"min_rows": MIN_ROWS, "supported": supported, "thin": thin}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "supported_skins.json").write_text(json.dumps(manifest, indent=2))
    print(f"  Supported-skins manifest saved to {out_dir}/supported_skins.json")

    # Predictions for sanity checks.
    preds = predict(models, df)
    df = df.assign(**preds)

    # --- Sanity Check: Quantile crossings (post-clamp, should be 0) ---
    print("\n== Sanity Check: Quantile crossings (post-clamp, should be 0) ==")
    crossings = ((df["q10"] > df["q50"]) | (df["q50"] > df["q90"])).sum()
    print(f"  Crossings: {crossings}/{len(df)} "
          f"({'PASS' if crossings == 0 else 'WARN'})")

    # --- Sanity Check: Held-out [q10, q90] coverage ---
    print("\n== Sanity Check: Held-out [q10, q90] coverage ==")
    test_mask = np.zeros(len(df), dtype=bool)
    test_mask[result["test_idx"]] = True
    test_df = df[test_mask]
    if len(test_df) > 0:
        actual = test_df[TARGET_COL]
        in_range = ((actual >= test_df["q10"]) & (actual <= test_df["q90"])).mean()
        print(f"  {in_range:.1%} of test rows fall within [q10, q90] "
              "(nominal 80%)")

    # --- Per-skin median asks (sample) ---
    print("\n== Per-skin median ask (reference_usd), sample ==")
    print(df.groupby("skin_base")["reference_usd"].first()
          .round(2).head(10).to_string())

    # --- One-line summary ---
    print(f"\n{len(supported)} supported / {len(thin)} thin skins "
          f"(threshold MIN_ROWS={MIN_ROWS})")
    print(f"Training complete (ASKING-price basis). Artifacts in {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
