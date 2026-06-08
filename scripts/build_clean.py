"""Phase 1 — clean the latest raw pull into a tidy table + validation summary.

Reads the newest data/raw/listings_*.jsonl, cleans it (cs2pricer.clean), writes
data/clean/listings.parquet (+ .csv for easy viewing), and prints validation checks.

Run: PYTHONPATH=. .venv/bin/python scripts/build_clean.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cs2pricer.clean import build_clean

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")


def latest_raw() -> Path:
    files = sorted(RAW_DIR.glob("listings_*.jsonl"))
    if not files:
        raise SystemExit("No raw pull found. Run scripts/pull_listings.py first.")
    return files[-1]


def main() -> int:
    raw_path = latest_raw()
    records = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    df, report = build_clean(records)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_DIR / "listings.parquet", index=False)
    df.to_csv(CLEAN_DIR / "listings.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print(f"source: {raw_path.name}")
    print("cleaning report:", json.dumps(report, indent=2))

    print("\n== rows per skin x stattrak ==")
    print(df.groupby(["skin_base", "is_stattrak"]).size().to_string())

    print("\n== rows per skin x exterior ==")
    print(df.pivot_table(index="skin_base", columns="exterior", values="id",
                         aggfunc="count", fill_value=0).to_string())

    print("\n== price_usd by skin (sanity: ranges should look like the real market) ==")
    print(df.groupby("skin_base")["price_usd"]
            .agg(["count", "min", "median", "max"]).round(0).to_string())

    print("\n== float_value coverage by skin (Dopplers/Fades ~all FN; CH should spread) ==")
    print(df.groupby("skin_base")["float_value"]
            .agg(["min", "median", "max"]).round(4).to_string())

    print("\n== flagged price outliers (kept, not dropped) ==")
    out = df[df["price_outlier"]][["skin_base", "exterior", "is_stattrak",
                                   "paint_seed", "float_value", "price_usd", "logprice_z"]]
    print(f"{len(out)} flagged")
    if len(out):
        print(out.sort_values("logprice_z", ascending=False).head(10).to_string(index=False))

    print(f"\nwrote {len(df)} rows -> {CLEAN_DIR/'listings.parquet'} (+ .csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
