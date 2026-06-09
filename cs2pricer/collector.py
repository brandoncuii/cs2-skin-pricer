"""Background sold-price collector (PLAN.md §10 v1.5).

Each run snapshots current buy_now listings for the locked skins into SQLite, then
detects which previously-open listings have left the listed pool. A listing that was
`listed` last run and is absent this run has likely sold near its last asking price
(noisy — could also be delist/expire, per §5.3); we call get_listing() on it to record
its terminal state when the API exposes one.

Storage: data/collector/observations.db (SQLite, under the gitignored data/).
  - listings:     one row per listing id + static attributes + open/closed status
  - observations: append-only time series of (price, state) per poll

The accumulating snapshots also give the time axis that v1.5's time-based split needs.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .client import CSFloatClient
from .clean import flatten_listing
from .skins import market_hash_names

DB_PATH = Path("data/collector/observations.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS listings (
    id TEXT PRIMARY KEY,
    market_hash_name TEXT, skin_base TEXT, weapon TEXT, finish TEXT, exterior TEXT,
    is_stattrak INTEGER, def_index INTEGER, paint_index INTEGER, paint_seed INTEGER,
    float_value REAL, created_at TEXT,
    first_seen TEXT, last_seen TEXT, status TEXT
);
CREATE TABLE IF NOT EXISTS observations (
    listing_id TEXT, observed_at TEXT, price_cents INTEGER, state TEXT
);
CREATE INDEX IF NOT EXISTS idx_obs_listing ON observations(listing_id);
CREATE INDEX IF NOT EXISTS idx_listings_status ON listings(status);
"""


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(_SCHEMA)
    return con


def collect_once(con: sqlite3.Connection, client: CSFloatClient,
                 names: list[str] | None = None,
                 max_pages: int | None = None) -> dict:
    if names is None:
        names = market_hash_names(stattrak=False) + market_hash_names(stattrak=True)
    now = datetime.now(timezone.utc).isoformat()

    seen: set[str] = set()
    for name in names:
        for raw in client.iter_listings(market_hash_name=name, type="buy_now",
                                        max_pages=max_pages):
            r = flatten_listing(raw)
            lid = r["id"]
            seen.add(lid)
            con.execute(
                """INSERT INTO listings
                   (id, market_hash_name, skin_base, weapon, finish, exterior,
                    is_stattrak, def_index, paint_index, paint_seed, float_value,
                    created_at, first_seen, last_seen, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'open')
                   ON CONFLICT(id) DO UPDATE SET last_seen=excluded.last_seen,
                                                 status='open'""",
                (lid, r["market_hash_name"], r["skin_base"], r["weapon"], r["finish"],
                 r["exterior"], int(r["is_stattrak"]), r["def_index"], r["paint_index"],
                 r["paint_seed"], r["float_value"], r["created_at"], now, now),
            )
            con.execute(
                "INSERT INTO observations (listing_id, observed_at, price_cents, state) "
                "VALUES (?,?,?,?)", (lid, now, r["price_cents"], "listed"),
            )
    con.commit()

    # Previously-open listings absent from this poll have left the listed pool.
    open_prev = [row[0] for row in con.execute("SELECT id FROM listings WHERE status='open'")]
    disappeared = [lid for lid in open_prev if lid not in seen]
    for lid in disappeared:
        state, price = "gone", None
        try:
            one = client.get_listing(lid)
            state, price = one.get("state", "gone"), one.get("price")
        except Exception:
            pass  # endpoint failed; record as 'gone' with last price unknown
        con.execute(
            "INSERT INTO observations (listing_id, observed_at, price_cents, state) "
            "VALUES (?,?,?,?)", (lid, now, price, state),
        )
        con.execute("UPDATE listings SET status='closed', last_seen=? WHERE id=?", (now, lid))
    con.commit()

    return {"observed_at": now, "listed_seen": len(seen),
            "disappeared": len(disappeared)}
