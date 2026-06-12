"""Empirical days-on-market stats from the collector DB (Find Deals actionability).

What this measures — and what it honestly does NOT:
  - For SOLD listings (terminal observation state == 'sold'), days-on-market is
    proxied by first_seen -> last_seen. Both ends are coarse: the collector polls
    every 3h (locked skins) / 12h (full sweep), so each endpoint is off by up to
    one poll interval, and listings that sell faster than one interval are never
    seen at all (fast sales are under-sampled).
  - RIGHT-CENSORING: open listings haven't sold YET. Medians here are conditional
    on having sold within the collector's observation window, so they UNDERSTATE
    true time-to-sale. Censored counts/ages are reported alongside as a reality
    check, not folded into the medians.
  - Samples are small (~231 sold). Buckets thinner than MIN_BUCKET_ROWS are
    dropped and the lookup falls back to the skin-level median, then the global
    median — and says which level it used. Plain empirical medians/quantiles on
    purpose: NOT a fitted survival model at this sample size.

Bucketing: by skin_base, and within skin by how far the ask sat vs the v1.5 mid
estimate (rel_to_mid = ask/mid - 1) in coarse buckets — the hypothesis being that
underpriced items move faster. scripts/build_liquidity.py computes the buckets and
writes the stats artifact; app.py reads it via load_stats()/lookup_days_to_sell().
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/collector/observations.db")
STATS_PATH = Path("data/liquidity/days_to_sell.json")

MIN_BUCKET_ROWS = 8  # below this, a (skin x rel-to-mid) bucket is too thin to trust

# Coarse ask-vs-mid buckets (rel_to_mid = ask / v1.5 mid - 1). Coarse on purpose.
_BUCKET_EDGES = (-0.10, 0.0, 0.10)
_BUCKET_LABELS = ("<=-10% vs mid", "-10..0% vs mid", "0..+10% vs mid", ">+10% vs mid")

# Terminal state of a listing = its latest observation's state.
_TERMINAL_CTE = """
    WITH terminal AS (
        SELECT listing_id, price_cents, state
        FROM (
            SELECT listing_id, price_cents, state,
                   ROW_NUMBER() OVER (
                       PARTITION BY listing_id ORDER BY observed_at DESC
                   ) AS rn
            FROM observations
        )
        WHERE rn = 1
    ),
    last_priced AS (
        SELECT listing_id, price_cents
        FROM (
            SELECT listing_id, price_cents,
                   ROW_NUMBER() OVER (
                       PARTITION BY listing_id ORDER BY observed_at DESC
                   ) AS rn
            FROM observations
            WHERE price_cents IS NOT NULL
        )
        WHERE rn = 1
    )
"""


def bucket_rel_to_mid(rel: float | None) -> str | None:
    """Map rel_to_mid (= ask/mid - 1) to a coarse bucket label."""
    if rel is None or pd.isna(rel):
        return None
    for edge, label in zip(_BUCKET_EDGES, _BUCKET_LABELS):
        if rel <= edge:
            return label
    return _BUCKET_LABELS[-1]


def load_sold_durations(db_path: Path = DB_PATH) -> pd.DataFrame:
    """SOLD listings with observed days-on-market (first_seen -> last_seen).

    Strictly terminal state 'sold' — 'gone'/'delisted'/'refunded' are not sales
    (delist = withdrawal, refund = reversed). Price = terminal observation's
    price, falling back to the last non-null observed price (for buy_now the
    sale price IS the last ask).
    """
    con = sqlite3.connect(db_path)
    query = _TERMINAL_CTE + """
        SELECT l.id, l.market_hash_name, l.skin_base, l.weapon, l.finish,
               l.exterior, l.is_stattrak, l.def_index, l.paint_index,
               l.paint_seed, l.float_value,
               COALESCE(t.price_cents, lp.price_cents) AS price_cents,
               (julianday(l.last_seen) - julianday(l.first_seen)) AS duration_days
        FROM listings l
        JOIN terminal t ON t.listing_id = l.id
        LEFT JOIN last_priced lp ON lp.listing_id = l.id
        WHERE t.state = 'sold'
    """
    df = pd.read_sql_query(query, con)
    con.close()
    if not df.empty:
        df["price_usd"] = df["price_cents"] / 100.0
        df["is_stattrak"] = df["is_stattrak"].astype(bool)
    return df


def load_censored_ages(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Right-censored (still live) listings and their observed age so far.

    Censored = still open, OR closed with terminal state 'listed' (get_listing()
    proved it live — it only fell out of pagination). For the latter, age stops
    at the disappearance check, so it's itself an underestimate.
    """
    con = sqlite3.connect(db_path)
    query = _TERMINAL_CTE + """
        SELECT l.skin_base,
               (julianday(l.last_seen) - julianday(l.first_seen)) AS age_days
        FROM listings l
        JOIN terminal t ON t.listing_id = l.id
        WHERE l.status = 'open' OR t.state = 'listed'
    """
    df = pd.read_sql_query(query, con)
    con.close()
    return df


def _summary(durations: pd.Series) -> dict:
    return {
        "n": int(len(durations)),
        "median_days": round(float(durations.median()), 2),
        "q25_days": round(float(durations.quantile(0.25)), 2),
        "q75_days": round(float(durations.quantile(0.75)), 2),
    }


def compute_stats(sold: pd.DataFrame, censored: pd.DataFrame,
                  min_bucket_rows: int = MIN_BUCKET_ROWS) -> dict:
    """Empirical days-to-sale stats keyed for lookup_days_to_sell().

    `sold` needs skin_base + duration_days; an optional rel_bucket column (from
    bucket_rel_to_mid against the v1.5 mid) enables the skin x bucket level.
    Buckets with < min_bucket_rows sold rows are omitted (lookup falls back).
    """
    stats: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "min_bucket_rows": min_bucket_rows,
        "poll_interval_hours": [3, 12],
        "note": (
            "Empirical medians of SOLD listings' observed days-on-market "
            "(first_seen -> last_seen, 3h-12h poll granularity). Right-censored: "
            "still-live listings haven't sold yet, so true times run longer."
        ),
        "global": {
            **_summary(sold["duration_days"]),
            "n_censored": int(len(censored)),
            "censored_median_age_days": (
                round(float(censored["age_days"].median()), 2) if len(censored) else None
            ),
        },
        "by_skin": {},
        "by_skin_bucket": {},
    }
    for skin, grp in sold.groupby("skin_base"):
        stats["by_skin"][skin] = {
            **_summary(grp["duration_days"]),
            "n_censored": int((censored["skin_base"] == skin).sum()),
        }
    if "rel_bucket" in sold.columns:
        with_bucket = sold.dropna(subset=["rel_bucket"])
        for (skin, bucket), grp in with_bucket.groupby(["skin_base", "rel_bucket"]):
            if len(grp) >= min_bucket_rows:
                stats["by_skin_bucket"].setdefault(skin, {})[bucket] = (
                    _summary(grp["duration_days"])
                )
    return stats


def save_stats(stats: dict, path: Path = STATS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))


def load_stats(path: Path = STATS_PATH) -> dict | None:
    """Load the stats artifact; None if missing/unparseable (app degrades)."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def lookup_days_to_sell(stats: dict | None, skin_base: str,
                        rel_to_mid: float | None = None) -> dict | None:
    """Median observed days-on-market for similar SOLD items, with explicit
    fallback: (skin x rel-to-mid bucket) -> skin -> global. Returns
    {"days", "n", "level"} or None when no stats exist at all.
    """
    if not stats:
        return None
    min_rows = stats.get("min_bucket_rows", MIN_BUCKET_ROWS)
    bucket = bucket_rel_to_mid(rel_to_mid)
    if bucket is not None:
        entry = stats.get("by_skin_bucket", {}).get(skin_base, {}).get(bucket)
        if entry:
            return {"days": entry["median_days"], "n": entry["n"],
                    "level": "skin+bucket"}
    entry = stats.get("by_skin", {}).get(skin_base)
    if entry and entry["n"] >= min_rows:
        return {"days": entry["median_days"], "n": entry["n"], "level": "skin"}
    entry = stats.get("global")
    if entry and entry["n"] > 0:
        return {"days": entry["median_days"], "n": entry["n"], "level": "global"}
    return None
