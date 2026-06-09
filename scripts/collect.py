"""One collector run — invoked on a schedule (launchd/cron). See cs2pricer.collector.

Flags:
  --full    Collect ALL knife types (expanded set for v1.5). Default: locked 4 only.
  --max-pages N   Max pages per skin name (default: 3 for locked, 1 for --full).

Run manually:
  PYTHONPATH=. .venv/bin/python scripts/collect.py          # locked 4 skins
  PYTHONPATH=. .venv/bin/python scripts/collect.py --full   # all knife types
"""
from __future__ import annotations

import argparse
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

from cs2pricer.client import CSFloatClient
from cs2pricer.collector import collect_once, connect
from cs2pricer.skins import (all_knife_market_hash_names, market_hash_names)


def main() -> int:
    parser = argparse.ArgumentParser(description="CS2 collector run")
    parser.add_argument("--full", action="store_true",
                        help="Collect all knife types (not just locked 4)")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Max pages per skin name (default: 3 for locked, 1 for --full)")
    args = parser.parse_args()

    if args.full:
        # Non-StatTrak only: the StatTrak premium measured near-neutral in v1
        # (and was dropped from the feature set), so skipping ST halves the
        # rate-limit cost of the full sweep (~2,920 names vs ~5,840).
        names = all_knife_market_hash_names(stattrak=False)
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
