"""Phase 0 liquidity gate — count live buy_now listings per candidate knife.

Uses reference.quantity (listings count for a market_hash_name) so we don't paginate
everything. Reports per skin x exterior and a skin-level total, plus the lowest ask
(a rough price anchor). Helps lock the final 3-5 skins on liquidity.

Run: PYTHONPATH=. .venv/bin/python scripts/liquidity_scan.py
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from cs2pricer.client import CSFloatClient

STAR = "★"  # ★ prefix on all knife names

# Candidate knives (normal, non-StatTrak). Every Doppler bundles all phases under one
# name (phase isn't in market_hash_name) — fine for a liquidity headline.
CANDIDATES = [
    f"{STAR} Karambit | Doppler",
    f"{STAR} M9 Bayonet | Doppler",
    f"{STAR} Butterfly Knife | Doppler",
    f"{STAR} Bayonet | Doppler",
    f"{STAR} Flip Knife | Doppler",
    f"{STAR} Huntsman Knife | Doppler",
    f"{STAR} Gut Knife | Doppler",
    f"{STAR} Falchion Knife | Doppler",
    f"{STAR} Karambit | Fade",
    f"{STAR} Butterfly Knife | Fade",
    f"{STAR} Karambit | Marble Fade",
    f"{STAR} Karambit | Case Hardened",
]
EXTERIORS = ["Factory New", "Minimal Wear", "Field-Tested"]


def probe(client: CSFloatClient, name: str) -> tuple[int, int | None]:
    """Return (quantity, lowest_price_cents) for a market_hash_name, or (0, None)."""
    page = client.get_listings(
        limit=1, market_hash_name=name, sort_by="lowest_price", type="buy_now"
    )
    data = page.get("data", [])
    if not data:
        return 0, None
    return data[0]["reference"]["quantity"], data[0]["price"]


def main() -> int:
    client = CSFloatClient()
    rows = []
    for base in CANDIDATES:
        total = 0
        lows = []
        per_ext = {}
        for ext in EXTERIORS:
            qty, low = probe(client, f"{base} ({ext})")
            per_ext[ext] = qty
            total += qty
            if low is not None:
                lows.append(low)
        floor = min(lows) / 100 if lows else float("nan")
        rows.append((base, total, per_ext, floor))

    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"{'knife':<32}{'total':>7}{'FN':>6}{'MW':>6}{'FT':>6}{'floor$':>10}")
    print("-" * 67)
    for base, total, per_ext, floor in rows:
        print(f"{base:<32}{total:>7}{per_ext['Factory New']:>6}"
              f"{per_ext['Minimal Wear']:>6}{per_ext['Field-Tested']:>6}{floor:>10.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
