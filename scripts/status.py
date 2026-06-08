"""At-a-glance project status — what's built, the data, and the collector.

Run: PYTHONPATH=. .venv/bin/python scripts/status.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

CLEAN = Path("data/clean/listings.parquet")
FEATURES = Path("data/features/features.parquet")
DB = Path("data/collector/observations.db")
LOG = Path("data/collector/collect.log")


def line(label: str, ok: bool, detail: str) -> None:
    mark = "done " if ok else "TODO "
    print(f"  [{mark}] {label:<22} {detail}")


def main() -> int:
    print("=== CS2 knife fair-value model — status ===\n")
    print("Phases (PLAN.md §10):")
    line("0 data access", True, "cs2pricer/client.py, skins.py")
    line("1 data pipeline", CLEAN.exists(), "data/clean/listings.csv")
    line("2 features", FEATURES.exists(), "data/features/features.csv")
    line("3 model (the ML part)", Path("data/model").exists(), "<- NEXT")
    line("4 scoring API", Path("cs2pricer/api.py").exists(), "")
    line("5 frontend", False, "")
    line("collector (v1.5)", DB.exists(), "SQLite, every 12h via launchd")

    if CLEAN.exists():
        df = pd.read_parquet(CLEAN)
        print(f"\nClean listings: {len(df)} rows  (data/clean/listings.csv)")
        print(df.groupby("skin_base").size().to_string())

    if FEATURES.exists():
        f = pd.read_parquet(FEATURES)
        print(f"\nFeatures: {len(f)} rows, target log_premium "
              f"(min {f.log_premium.min():.2f} / med {f.log_premium.median():.2f} "
              f"/ max {f.log_premium.max():.2f})  (data/features/features.csv)")

    print("\nCollector DB (data/collector/observations.db):")
    if DB.exists():
        con = sqlite3.connect(DB)
        nl = con.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        no = con.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        runs = con.execute("SELECT COUNT(DISTINCT observed_at) FROM observations").fetchone()[0]
        span = con.execute("SELECT MIN(observed_at), MAX(observed_at) FROM observations").fetchone()
        opn = con.execute("SELECT COUNT(*) FROM listings WHERE status='open'").fetchone()[0]
        closed = con.execute("SELECT COUNT(*) FROM listings WHERE status='closed'").fetchone()[0]
        print(f"  listings={nl} (open {opn}, closed/likely-sold {closed}) | "
              f"observations={no} across {runs} run(s)")
        print(f"  time span: {span[0]} -> {span[1]}")
        if nl == 0:
            print("  (empty so far — first scheduled run may still be waiting on the rate reset)")
    else:
        print("  not created yet")

    if LOG.exists():
        tail = LOG.read_text().splitlines()[-3:]
        print("\nCollector log tail (data/collector/collect.log):")
        for t in tail:
            print("  " + t)

    print("\nVerify yourself:")
    print("  open data/clean/listings.csv / data/features/features.csv in any viewer")
    print("  launchctl list | grep cs2pricer        # collector scheduled?")
    print("  PYTHONPATH=. .venv/bin/python scripts/status.py   # re-run this")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
