"""Track record — backtest v1 and v1.5 against actual sold prices.

Evaluates both models' predictions against real sales from the collector DB
(terminal_state='sold' — confirmed labeled sales with the sale price).

Leakage rules (the whole point of this script):
  - v1.5 was TRAINED on the earlier portion of these same sold rows, so it is
    only evaluated on rows AFTER its time-split cutoff. metadata.json records
    n_train, so the eval set is everything past the first n_train rows when
    sorted by last_seen — the exact held-out set when the DB is unchanged,
    and still leakage-safe if new rows have arrived since training.
  - v1 was trained on asking-price snapshots, never on sold rows, so ALL sold
    rows are fair game. But honesty caveat: v1 predicts ASKING prices, and
    asks sit above sale prices, so v1 should systematically overestimate.

Reference prices mirror serve time (app.py score_df): v1 converts log-premium
to USD via clean-dataset asking medians; v1.5 via its persisted references.json
(sold medians). Using in-sample medians of the eval rows would leak the answer.

Writes a tidy predictions-vs-actuals parquet + summary.json to data/backtest/
for the app's Track Record view.

Run: PYTHONPATH=. .venv/bin/python scripts/backtest.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cs2pricer.features import add_features
from cs2pricer.model import (MODEL_DIR, MODEL_V15_DIR, load_models,
                             load_models_v15, load_references, predict)

# Reuse the canonical sold-row query from the v1.5 training script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_model_v15 import DB_PATH, _load_disappeared  # noqa: E402

CLEAN_PATH = Path("data/clean/listings.parquet")
OUT_DIR = Path("data/backtest")


def _clean_sold(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same malformed-row filter as train_model_v15, sold rows only."""
    df = df[df["terminal_state"] == "sold"].copy()
    bad = (
        df["price_usd"].isna() | df["price_usd"].le(0)
        | df["float_value"].isna()
        | df["paint_seed"].isna()
        | df["skin_base"].isna()
    )
    return df[~bad].copy()


def _score(df: pd.DataFrame, models, refs: dict[str, float],
           version: str) -> pd.DataFrame:
    """Score sold rows with serve-time references; return predictions vs actuals."""
    df = df.copy()
    # Override the in-sample medians add_features computed — at serve time the
    # model only knows its training-era references, so eval must too.
    df["reference_usd"] = df["skin_base"].map(refs)
    missing = df["reference_usd"].isna()
    if missing.any():
        print(f"  WARNING: dropped {missing.sum()} rows with no {version} "
              f"reference price")
        df = df[~missing].copy()

    preds = predict(models, df)
    df = df.assign(**preds)

    out = df[["id", "market_hash_name", "skin_base", "exterior", "is_stattrak",
              "float_value", "last_seen", "price_usd",
              "low_usd", "mid_usd", "high_usd"]].copy()
    out = out.rename(columns={"price_usd": "actual_usd"})
    out["model_version"] = version
    out["error_usd"] = out["mid_usd"] - out["actual_usd"]
    out["abs_err_usd"] = out["error_usd"].abs()
    out["ape_pct"] = out["abs_err_usd"] / out["actual_usd"] * 100
    out["in_range"] = (out["actual_usd"] >= out["low_usd"]) & \
                      (out["actual_usd"] <= out["high_usd"])
    return out


def _metrics(scored: pd.DataFrame) -> dict:
    return {
        "n": int(len(scored)),
        "mae_usd": float(scored["abs_err_usd"].mean()),
        "median_ape_pct": float(scored["ape_pct"].median()),
        "coverage_q10_q90": float(scored["in_range"].mean()),
        "median_error_usd": float(scored["error_usd"].median()),
    }


def _print_metrics(label: str, m: dict) -> None:
    print(f"  {label}: n={m['n']}  MAE=${m['mae_usd']:.2f}  "
          f"median APE={m['median_ape_pct']:.1f}%  "
          f"[q10,q90] coverage={m['coverage_q10_q90']:.1%}  "
          f"median error=${m['median_error_usd']:+.2f}")


def _print_per_skin(scored: pd.DataFrame) -> None:
    per_skin = scored.groupby("skin_base").agg(
        n=("abs_err_usd", "size"),
        mae_usd=("abs_err_usd", "mean"),
        median_ape_pct=("ape_pct", "median"),
        coverage=("in_range", "mean"),
    ).round(2)
    print(per_skin.to_string())


def main() -> int:
    if not DB_PATH.exists():
        print(f"Collector DB not found at {DB_PATH}. Run the collector first.")
        return 1

    print("== Loading sold listings from collector DB ==")
    df = _clean_sold(_load_disappeared(DB_PATH))
    print(f"  Found {len(df)} sold listings with prices")
    if df.empty:
        print("  Nothing to evaluate.")
        return 0

    # Same ordering as training, then features (same pipeline as v1/v1.5).
    df = df.sort_values("last_seen").reset_index(drop=True)
    df = add_features(df)

    results: list[pd.DataFrame] = []
    summary: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": {},
    }

    # --- v1: asking-price model, all sold rows are fair game ---
    print("\n== v1 (asking-price model) vs actual sold prices ==")
    print("  Basis: v1 never saw sold rows (trained on asking snapshots), so "
          "all sold rows are evaluated.")
    print("  Caveat: v1 predicts ASKING prices — expect it to overestimate sales.")
    refs_v1 = (pd.read_parquet(CLEAN_PATH)
               .groupby("skin_base")["price_usd"].median().to_dict()
               if CLEAN_PATH.exists() else {})
    if not refs_v1:
        print("  Clean dataset missing — skipping v1 (no reference prices).")
    else:
        scored_v1 = _score(df, load_models(), refs_v1, "v1")
        results.append(scored_v1)
        m = _metrics(scored_v1)
        summary["models"]["v1"] = m
        _print_metrics("v1", m)
        print("\n  Per-skin breakdown (v1):")
        _print_per_skin(scored_v1)

    # --- v1.5: sold-price model, ONLY rows after its time-split cutoff ---
    print("\n== v1.5 (sold-price model) vs actual sold prices ==")
    models_v15 = load_models_v15()
    refs_v15 = load_references(MODEL_V15_DIR)
    meta_path = MODEL_V15_DIR / "metadata.json"
    if models_v15 is None or refs_v15 is None or not meta_path.exists():
        print("  v1.5 artifacts missing — skipping. Train with "
              "scripts/train_model_v15.py first.")
    else:
        meta = json.loads(meta_path.read_text())
        n_train, n_test = meta["n_train"], meta["n_test"]
        if len(df) != n_train + n_test:
            print(f"  NOTE: DB has {len(df)} sold rows but v1.5 trained on "
                  f"{n_train + n_test} — evaluating only rows past the "
                  f"training cutoff (still leakage-safe).")
        # Rows past the first n_train (sorted by last_seen) were never trained on.
        test_df = df.iloc[n_train:]
        cutoff = df["last_seen"].iloc[n_train - 1]
        summary["v15_cutoff_last_seen"] = cutoff
        print(f"  Leakage rule: v1.5 trained on the oldest {n_train} sold rows "
              f"(last_seen <= {cutoff});")
        print(f"  evaluating ONLY the {len(test_df)} held-out rows after the cutoff.")
        if len(test_df) == 0:
            print("  No held-out rows yet — skipping v1.5.")
        else:
            scored_v15 = _score(test_df, models_v15, refs_v15, "v1.5")
            results.append(scored_v15)
            m = _metrics(scored_v15)
            summary["models"]["v1.5"] = m
            _print_metrics("v1.5", m)
            print(f"\n  Per-skin breakdown (v1.5, n={len(scored_v15)} — small "
                  "sample, read with caution):")
            _print_per_skin(scored_v15)

    if not results:
        print("\nNo models could be evaluated — nothing written.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tidy = pd.concat(results, ignore_index=True)
    tidy.to_parquet(OUT_DIR / "predictions.parquet", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {len(tidy)} prediction rows to {OUT_DIR}/predictions.parquet")
    print(f"Wrote headline metrics to {OUT_DIR}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
