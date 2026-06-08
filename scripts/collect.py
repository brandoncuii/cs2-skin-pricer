"""One collector run — invoked on a schedule (launchd/cron). See cs2pricer.collector.

Run manually: PYTHONPATH=. .venv/bin/python scripts/collect.py
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

from cs2pricer.client import CSFloatClient
from cs2pricer.collector import collect_once, connect


def main() -> int:
    start = datetime.now(timezone.utc).isoformat()
    print(f"[{start}] collector run start", flush=True)
    con = connect()
    summary = collect_once(con, CSFloatClient())
    con.close()
    print(f"[{summary['observed_at']}] done: "
          f"listed_seen={summary['listed_seen']} disappeared={summary['disappeared']}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
