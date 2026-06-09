"""One collector run — invoked on a schedule (launchd/cron). See cs2pricer.collector.

Flags:
  --full    Collect ALL knife types (expanded set for v1.5). Default: locked 4 only.
  --max-pages N   Max pages per skin name (default: 3 for locked, 1 for --full).
  --min-qty N     (--full only) Skip names whose last liquidity scan saw fewer
                  than N live listings (default: 1 — drops dead combos only).
                  Requires data/collector/full_names.json from scan_full_names.py;
                  without it, all names are polled.

Run manually:
  PYTHONPATH=. .venv/bin/python scripts/collect.py          # locked 4 skins
  PYTHONPATH=. .venv/bin/python scripts/collect.py --full   # all knife types
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

from cs2pricer.client import CSFloatClient
from cs2pricer.collector import collect_once, connect
from cs2pricer.skins import (all_knife_market_hash_names, market_hash_names)

GATE_PATH = Path("data/collector/full_names.json")


def gate_by_liquidity(names: list[str], min_qty: int) -> list[str]:
    """Drop names whose last liquidity scan saw fewer than min_qty live listings.

    Names missing from the scan file are kept (the scan may predate a skins.py
    update); with no scan file at all, everything is kept.
    """
    if not GATE_PATH.exists():
        print(f"  no liquidity scan at {GATE_PATH} — polling all {len(names)} names "
              "(run scripts/scan_full_names.py once to gate)", flush=True)
        return names
    qty = json.loads(GATE_PATH.read_text())
    kept = [n for n in names if qty.get(n, min_qty) >= min_qty]
    print(f"  liquidity gate (min_qty={min_qty}): {len(kept)}/{len(names)} names kept",
          flush=True)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description="CS2 collector run")
    parser.add_argument("--full", action="store_true",
                        help="Collect all knife types (not just locked 4)")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Max pages per skin name (default: 3 for locked, 1 for --full)")
    parser.add_argument("--min-qty", type=int, default=1,
                        help="(--full only) skip names below this live-listing count "
                             "in the last liquidity scan (default: 1)")
    args = parser.parse_args()

    if args.full:
        # Non-StatTrak only: the StatTrak premium measured near-neutral in v1
        # (and was dropped from the feature set), so skipping ST halves the
        # rate-limit cost of the full sweep (~2,920 names vs ~5,840).
        names = all_knife_market_hash_names(stattrak=False)
        names = gate_by_liquidity(names, args.min_qty)
        max_pages = args.max_pages if args.max_pages is not None else 1
        mode = "full"
    else:
        names = market_hash_names(stattrak=False) + market_hash_names(stattrak=True)
        max_pages = args.max_pages if args.max_pages is not None else 3
        mode = "locked"

    start = datetime.now(timezone.utc).isoformat()
    print(f"[{start}] collector run start (mode={mode}, names={len(names)}, "
          f"max_pages={max_pages})", flush=True)

    con = connect()
    summary = collect_once(con, CSFloatClient(), names=names, max_pages=max_pages)
    con.close()

    end = datetime.now(timezone.utc).isoformat()
    print(f"[{end}] done (snapshot {summary['observed_at']}): "
          f"listed_seen={summary['listed_seen']} disappeared={summary['disappeared']}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
