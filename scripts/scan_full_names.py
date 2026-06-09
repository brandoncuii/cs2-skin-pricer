"""Liquidity scan over the FULL knife set — gates --full collector runs.

Probes every (knife x finish x exterior) market_hash_name with limit=1 and
records reference.quantity (live listing count for that name). One request
per name (~2,920), so a scan costs about the same as one ungated --full run —
but afterwards `scripts/collect.py --full` skips dead names entirely, which
cuts recurring cost roughly in half (most knife x finish combos don't exist).

Output: data/collector/full_names.json  {market_hash_name: quantity}

Quantities drift; re-run occasionally (monthly is plenty) to refresh the gate.

Run (on the collector host): PYTHONPATH=. .venv/bin/python scripts/scan_full_names.py
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

from cs2pricer.client import CSFloatClient
from cs2pricer.skins import all_knife_market_hash_names

OUT_PATH = Path("data/collector/full_names.json")


def probe_quantity(client: CSFloatClient, name: str) -> int:
    """Live listing count for a market_hash_name (0 if none listed)."""
    page = client.get_listings(limit=1, market_hash_name=name,
                               sort_by="lowest_price", type="buy_now")
    data = page.get("data", [])
    if not data:
        return 0
    # quantity should always be present alongside a listing; if the reference
    # block is missing, we still know at least one listing exists.
    return (data[0].get("reference") or {}).get("quantity") or 1


def main() -> int:
    client = CSFloatClient()
    names = all_knife_market_hash_names(stattrak=False)
    start = datetime.now(timezone.utc).isoformat()
    print(f"[{start}] probing {len(names)} names (1 request each) ...", flush=True)

    quantities: dict[str, int] = {}
    for i, name in enumerate(names, 1):
        quantities[name] = probe_quantity(client, name)
        if i % 100 == 0:
            print(f"  {i}/{len(names)}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(quantities, indent=0, sort_keys=True))

    nonzero = sum(1 for q in quantities.values() if q > 0)
    end = datetime.now(timezone.utc).isoformat()
    print(f"[{end}] done: {nonzero}/{len(names)} names have live listings; "
          f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
