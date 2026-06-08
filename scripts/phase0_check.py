"""Phase 0 — confirm CSFloat data access and inspect the listing schema.

Run: .venv/bin/python scripts/phase0_check.py

Checks:
  1. API key loads and GET /listings returns data.
  2. The fields PLAN.md §5 depends on are present on a listing.
  3. The specific-listing endpoint (GET /listings/<id>) works and exposes `state`.
"""
import json
import sys

from cs2pricer.client import CSFloatClient

# Fields the model/pipeline depend on (PLAN.md §5). Some live on the listing,
# some on the nested `item`.
LISTING_FIELDS = ["id", "price", "state", "type"]
ITEM_FIELDS = [
    "float_value", "paint_seed", "paint_index", "def_index",
    "is_stattrak", "market_hash_name",
]


def main() -> int:
    client = CSFloatClient()

    print("== 1. GET /listings (buy_now sample) ==")
    page = client.get_listings(limit=5, type="buy_now", sort_by="most_recent")
    if not isinstance(page, dict):
        print(f"  Unexpected top-level type: {type(page)}", file=sys.stderr)
        return 1
    print("  top-level keys:", sorted(page.keys()))
    data = page.get("data", [])
    print(f"  cursor present: {'cursor' in page}; items returned: {len(data)}")
    if not data:
        print("  No listings returned — cannot continue.", file=sys.stderr)
        return 1

    sample = data[0]
    item = sample.get("item", {})
    print("\n== 2. Field presence on a sample listing ==")
    print("  listing keys:", sorted(sample.keys()))
    print("  item keys:   ", sorted(item.keys()))
    missing = [f for f in LISTING_FIELDS if f not in sample]
    missing += [f"item.{f}" for f in ITEM_FIELDS if f not in item]
    if missing:
        print("  MISSING expected fields:", missing, file=sys.stderr)
    else:
        print("  All expected §5 fields present.")
    print("\n  sample (trimmed):")
    print(json.dumps({
        "id": sample.get("id"),
        "price_cents": sample.get("price"),
        "state": sample.get("state"),
        "type": sample.get("type"),
        "item": {k: item.get(k) for k in ITEM_FIELDS},
    }, indent=2))

    print("\n== 3. GET /listings/<id> (specific-listing endpoint) ==")
    listing_id = str(sample.get("id"))
    one = client.get_listing(listing_id)
    one_item = one.get("item", {})
    print(f"  fetched id={listing_id}; state={one.get('state')}; "
          f"name={one_item.get('market_hash_name')}")
    print("  -> endpoint works. (Post-`listed` persistence is validated over time by the "
          "v1.5 collector as listings naturally churn; cannot be forced in a one-shot check.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
