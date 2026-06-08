"""Phase 1 — pull all live buy_now listings for the locked skins (normal + StatTrak).

Saves raw listings (unmodified) to data/raw/listings_<UTC>.jsonl, plus a sidecar
_meta.json recording the pull time and per-name counts. Raw is kept separate from
cleaned data so cleaning stays reproducible (PLAN.md §6).

Run: PYTHONPATH=. .venv/bin/python scripts/pull_listings.py
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

from cs2pricer.client import CSFloatClient
from cs2pricer.skins import market_hash_names

RAW_DIR = Path("data/raw")


def main() -> int:
    client = CSFloatClient()
    names = market_hash_names(stattrak=False) + market_hash_names(stattrak=True)
    pulled_at = datetime.now(timezone.utc).isoformat()
    stamp = pulled_at.replace(":", "").replace("-", "")[:15]  # 20260608T214732

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"listings_{stamp}.jsonl"
    counts: dict[str, int] = {}
    total = 0

    with out_path.open("w") as f:
        for name in names:
            n = 0
            for listing in client.iter_listings(market_hash_name=name, type="buy_now"):
                listing["_pulled_at"] = pulled_at
                f.write(json.dumps(listing) + "\n")
                n += 1
            counts[name] = n
            total += n
            if n:
                print(f"  {n:>4}  {name}")

    meta = {"pulled_at": pulled_at, "total": total, "counts": counts,
            "file": out_path.name}
    (RAW_DIR / f"listings_{stamp}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nTotal {total} listings across {sum(1 for v in counts.values() if v)} "
          f"non-empty names -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
