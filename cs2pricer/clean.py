"""Flatten + clean raw CSFloat listings into a tidy table (PLAN.md Phase 1).

Cleaning policy (kept conservative on purpose):
  - Dedup by listing id.
  - Drop only *malformed/impossible* rows (bad price/float/seed). We do NOT drop
    cheap-but-valid rows — those low prices are exactly the deals the model exists to
    find. Dropping them would defeat the tool.
  - Outliers are FLAGGED, not removed: a MAD-based robust z-score of log price within
    (skin x exterior x stattrak). High values are usually real (Ruby/Sapphire Dopplers,
    blue-gem Case Hardened), so modeling handles fat tails via quantile/robust objectives
    (PLAN.md §8) rather than deletion. The flag is for inspection only.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_NAME_RE = re.compile(
    r"^★\s*(?:StatTrak™\s*)?(?P<weapon>[^|]+?)\s*\|\s*(?P<finish>.+?)\s*"
    r"\((?P<exterior>[^()]+)\)$"
)


def parse_name(market_hash_name: str) -> dict[str, str | None]:
    m = _NAME_RE.match(market_hash_name)
    if not m:
        return {"weapon": None, "finish": None, "exterior": None, "skin_base": None}
    weapon, finish, exterior = m.group("weapon"), m.group("finish"), m.group("exterior")
    return {
        "weapon": weapon,
        "finish": finish,
        "exterior": exterior,
        "skin_base": f"{weapon} | {finish}",
    }


def flatten_listing(raw: dict) -> dict:
    item = raw.get("item", {}) or {}
    ref = raw.get("reference", {}) or {}
    name = item.get("market_hash_name", "")
    parts = parse_name(name)
    price_cents = raw.get("price")
    return {
        "id": str(raw.get("id")),
        "created_at": raw.get("created_at"),
        "pulled_at": raw.get("_pulled_at"),
        "state": raw.get("state"),
        "type": raw.get("type"),
        "market_hash_name": name,
        "skin_base": parts["skin_base"],
        "weapon": parts["weapon"],
        "finish": parts["finish"],
        "exterior": parts["exterior"],
        "is_stattrak": bool(item.get("is_stattrak", False)),
        "def_index": item.get("def_index"),
        "paint_index": item.get("paint_index"),
        "paint_seed": item.get("paint_seed"),
        "float_value": item.get("float_value"),
        "price_cents": price_cents,
        "price_usd": price_cents / 100 if price_cents is not None else None,
        # CSFloat's own estimate — kept for sanity checks ONLY, never a feature/target.
        "csfloat_predicted_cents": ref.get("predicted_price"),
        "csfloat_quantity": ref.get("quantity"),
    }


def _robust_logprice_z(df: pd.DataFrame) -> pd.Series:
    """MAD-based z of log price within (skin_base, exterior, is_stattrak)."""
    logp = np.log(df["price_usd"])
    grp = df.groupby(["skin_base", "exterior", "is_stattrak"])["price_usd"]
    med = grp.transform(lambda s: np.median(np.log(s)))
    mad = grp.transform(lambda s: np.median(np.abs(np.log(s) - np.median(np.log(s)))))
    # 1.4826 scales MAD to std for normal data; guard against zero MAD.
    scale = (1.4826 * mad).replace(0, np.nan)
    return (logp - med) / scale


def build_clean(records: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Return (clean_df, report)."""
    raw_df = pd.DataFrame(flatten_listing(r) for r in records)
    report: dict = {"raw_rows": len(raw_df)}

    df = raw_df.drop_duplicates(subset="id")
    report["after_dedup"] = len(df)

    bad = (
        df["price_usd"].isna() | (df["price_usd"] <= 0)
        | df["float_value"].isna() | (df["float_value"] < 0) | (df["float_value"] > 1)
        | df["paint_seed"].isna()
        | df["market_hash_name"].eq("") | df["skin_base"].isna()
    )
    report["dropped_malformed"] = int(bad.sum())
    df = df[~bad].copy()

    # Placeholder / "not actually for sale" listings near CSFloat's $100k price cap.
    # Beyond any genuine buy_now value for this 4-skin set; corrupts medians/quantiles.
    ceiling = df["price_usd"] >= 50_000
    report["dropped_price_ceiling"] = int(ceiling.sum())
    df = df[~ceiling].copy()

    # Fade has NO pattern/seed premium, so price far above CSFloat's estimate is junk
    # (e.g. a ~$39k Fade). We do this ONLY for Fade — Case Hardened's high tail is real
    # blue-gem signal and must be kept (revisit once the seed mapping lands in Phase 2).
    pred_usd = df["csfloat_predicted_cents"] / 100
    fade_junk = (
        df["finish"].eq("Fade") & pred_usd.gt(0) & (df["price_usd"] / pred_usd > 5)
    )
    report["dropped_fade_junk"] = int(fade_junk.sum())
    df = df[~fade_junk].copy()

    report["clean_rows"] = len(df)

    df["logprice_z"] = _robust_logprice_z(df)
    df["price_outlier"] = df["logprice_z"].abs() > 5  # flag only; not dropped
    report["flagged_outliers"] = int(df["price_outlier"].sum())

    return df, report
