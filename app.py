"""Phase 5 — Streamlit frontend for CS2 Knife Fair-Value.

Three views:
  1. **Find Deals** — pulls live CSFloat listings, scores them, shows those priced
     below the model's low estimate (i.e., cheaper than comparable current asks).
  2. **Score a Listing** — manually input a listing's attributes and get its
     fair-value range.
  3. **Track Record** — how well each model predicts actual sold prices, read
     from the backtest artifact produced by scripts/backtest.py.

Supports both v1 (asking-price) and v1.5 (sold-price) models when available.

Run: PYTHONPATH=. .venv/bin/streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

from cs2pricer.clean import flatten_listing
from cs2pricer.client import CSFloatClient, CSFloatError, parse_listing_id
from cs2pricer.features import DOPPLER_PHASE, add_features
from cs2pricer.liquidity import load_stats, lookup_days_to_sell
from cs2pricer.model import (MODEL_DIR, MODEL_V15_DIR, load_models,
                             load_models_v15, load_references, predict,
                             v15_available)
from cs2pricer.skins import EXTERIORS, SKINS, market_hash_names

st.set_page_config(page_title="CS2 Knife Pricer", page_icon="🔪", layout="wide")

# --- Load models ---
@st.cache_resource
def get_models_v1():
    if not MODEL_DIR.exists():
        return None
    return load_models()


@st.cache_resource
def get_models_v15():
    if not v15_available():
        return None
    return load_models_v15()


models_v1 = get_models_v1()
models_v15 = get_models_v15()

if models_v1 is None:
    st.error("Model not trained yet. Run: `PYTHONPATH=. .venv/bin/python scripts/train_model.py`")
    st.stop()

# --- Sidebar ---
st.sidebar.title("🔪 CS2 Knife Pricer")

# Model version selector.
version_options = ["v1 (asking prices)"]
if models_v15 is not None:
    version_options.append("v1.5 (sold prices)")

selected_version_label = st.sidebar.radio("Model Version", version_options)
use_v15 = "v1.5" in selected_version_label and models_v15 is not None
active_models = models_v15 if use_v15 else models_v1
version_tag = "v1.5" if use_v15 else "v1"

if use_v15:
    st.sidebar.caption(
        "**v1.5 — trained on inferred sold prices.**\n\n"
        "Uses collector disappearance data to estimate actual transaction prices. "
        "More accurate than v1's asking-price basis."
    )
else:
    st.sidebar.caption(
        "**v1 — trained on asking prices.**\n\n"
        "A 'deal' means priced below what comparable items are currently listed for. "
        "This is NOT true fair value (requires sold-price data → v1.5)."
    )

if models_v15 is None:
    st.sidebar.info(
        "v1.5 not available yet. Train it with:\n"
        "`PYTHONPATH=. .venv/bin/python scripts/train_model_v15.py`"
    )

page = st.sidebar.radio("View", ["Find Deals", "Score a Listing", "Track Record"])

# --- Helpers ---
CLEAN_PATH = Path("data/clean/listings.parquet")
BACKTEST_DIR = Path("data/backtest")


@st.cache_data(ttl=300)
def get_reference_prices() -> dict[str, float]:
    """Median price per skin_base from the clean dataset."""
    if not CLEAN_PATH.exists():
        return {}
    df = pd.read_parquet(CLEAN_PATH)
    return df.groupby("skin_base")["price_usd"].median().to_dict()


@st.cache_data(ttl=300)
def get_sold_reference_prices() -> dict[str, float]:
    """Per-skin sold-price medians saved at v1.5 training time. These are the
    dollar anchors the v1.5 models were normalized against, so v1.5 predictions
    must use them — not asking-price medians, which run higher and would inflate
    every estimate."""
    return load_references(MODEL_V15_DIR) or {}


def get_active_references() -> dict[str, float]:
    """Reference anchors matching the active model: sold-price medians for v1.5,
    asking-price medians for v1. Falls back to ask medians if the v1.5 refs are
    missing, so the app still runs."""
    if use_v15:
        refs = get_sold_reference_prices()
        if refs:
            return refs
    return get_reference_prices()


@st.cache_data(ttl=300)
def get_liquidity_stats() -> dict | None:
    """Empirical days-to-sale stats (scripts/build_liquidity.py); None if not built."""
    return load_stats()


def score_df(df: pd.DataFrame, refs: dict[str, float]) -> pd.DataFrame:
    """Add features and score a DataFrame."""
    df = add_features(df)
    df["reference_usd"] = df["skin_base"].map(refs)
    df["reference_usd"] = df["reference_usd"].fillna(df["price_usd"])
    df["log_premium"] = np.log(df["price_usd"]) - np.log(df["reference_usd"])
    preds = predict(active_models, df)
    return df.assign(**preds)


# === PAGE: Find Deals ===
if page == "Find Deals":
    st.title("Live Deals — Cheaper Than Comparable Asks")
    st.caption(
        f"Model: **{version_tag}** — Pulls current CSFloat buy_now listings, scores them, "
        "and shows those priced below the model's 10th-percentile estimate."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_skins = st.multiselect(
            "Skins to scan",
            [s["base"] for s in SKINS],
            default=[s["base"] for s in SKINS],
        )
    with col2:
        max_pages = st.slider("Pages per skin (more = slower, more listings)", 1, 10, 3)

    if st.button("🔍 Scan for Deals", type="primary"):
        refs = get_active_references()
        client = CSFloatClient()

        names = []
        for skin in SKINS:
            if skin["base"] in selected_skins:
                for ext in EXTERIORS:
                    names.append(f"{skin['base']} ({ext})")
                    st_name = skin["base"].replace("★ ", "★ StatTrak™ ", 1)
                    names.append(f"{st_name} ({ext})")

        raw_listings: list[dict] = []
        progress = st.progress(0, text="Pulling listings...")
        for i, name in enumerate(names):
            for listing in client.iter_listings(
                market_hash_name=name, type="buy_now", max_pages=max_pages
            ):
                raw_listings.append(listing)
            progress.progress((i + 1) / len(names), text=f"Pulled {len(raw_listings)} listings...")
        progress.empty()

        if not raw_listings:
            st.warning("No listings found.")
            st.stop()

        flat = [flatten_listing(r) for r in raw_listings]
        df = pd.DataFrame(flat)
        df = df[df["price_usd"].gt(0) & df["float_value"].notna() & df["paint_seed"].notna()].copy()

        scored = score_df(df, refs)
        scored["is_deal"] = scored["price_usd"] < scored["low_usd"]
        scored["discount_pct"] = np.where(
            scored["is_deal"],
            ((1 - scored["price_usd"] / scored["low_usd"]) * 100).round(1),
            None,
        )

        deals = scored[scored["is_deal"]].copy()

        # --- Actionability ranking (replaces the raw discount sort) ---
        # Empirical "similar items sold in ~X days" from the collector DB
        # (scripts/build_liquidity.py). Buckets were built against v1.5 mids;
        # here rel_to_mid uses the ACTIVE model's mid — coarse buckets, fine.
        liq_stats = get_liquidity_stats()
        if liq_stats is None:
            st.info(
                "Days-to-sale stats not built yet — ranking by discount and live "
                "supply only. Build with: "
                "`PYTHONPATH=. .venv/bin/python scripts/build_liquidity.py`"
            )
        rel_to_mid = deals["price_usd"] / deals["mid_usd"] - 1
        looked_up = [
            lookup_days_to_sell(liq_stats, sb, rel)
            for sb, rel in zip(deals["skin_base"], rel_to_mid)
        ]
        deals["est_days_to_sell"] = [lk["days"] if lk else np.nan for lk in looked_up]
        deals["days_basis"] = [
            f"{lk['level']} (n={lk['n']})" if lk else "no data" for lk in looked_up
        ]
        deals["live_supply"] = deals["csfloat_quantity"]

        # Actionability = Discount % x (1 + ln(1 + live supply)) / est. days to sell.
        # Simple and visible on purpose: discount per expected day on market, nudged
        # up by live supply (reference.quantity, log-scaled so it's a hint, not a
        # driver). Days floored at 0.25 (~poll granularity) and defaulted to 1.0
        # when no stats exist. Empirical + small-sample — a ranking, not a forecast.
        est_days = deals["est_days_to_sell"].fillna(1.0).clip(lower=0.25)
        supply_boost = 1.0 + np.log1p(deals["live_supply"].fillna(0).clip(lower=0))
        deals["actionability"] = deals["discount_pct"] * supply_boost / est_days

        deals = deals.sort_values("actionability", ascending=False)
        st.success(f"Scored {len(scored)} listings — **{len(deals)} deals** found")

        if len(deals) > 0:
            display_cols = [
                "market_hash_name", "price_usd", "low_usd", "mid_usd", "high_usd",
                "discount_pct", "est_days_to_sell", "days_basis", "live_supply",
                "actionability",
                "float_value", "paint_seed", "doppler_phase", "exterior",
            ]
            show = deals[display_cols].copy()
            show.columns = [
                "Skin", "Price ($)", "Low ($)", "Mid ($)", "High ($)",
                "Discount %", "Est. Days to Sell", "Days Basis", "Live Supply",
                "Actionability",
                "Float", "Seed", "Phase", "Exterior",
            ]
            st.dataframe(
                show.reset_index(drop=True),
                use_container_width=True,
                column_config={
                    "Price ($)": st.column_config.NumberColumn(format="$%.2f"),
                    "Low ($)": st.column_config.NumberColumn(format="$%.2f"),
                    "Mid ($)": st.column_config.NumberColumn(format="$%.2f"),
                    "High ($)": st.column_config.NumberColumn(format="$%.2f"),
                    "Discount %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Est. Days to Sell": st.column_config.NumberColumn(format="%.2f"),
                    "Actionability": st.column_config.NumberColumn(format="%.1f"),
                    "Float": st.column_config.NumberColumn(format="%.6f"),
                },
            )
            st.caption(
                "Ranked by **Actionability** = Discount % × (1 + ln(1 + Live Supply)) "
                "÷ Est. Days to Sell. *Est. Days to Sell* is the empirical median "
                "observed days-on-market of similar SOLD listings from our collector "
                "(3h–12h polls, ~231 sales over a few days — small sample, and "
                "right-censored: still-listed items haven't sold yet, so true times "
                "run longer). *Days Basis* shows the stat's grouping level "
                "(skin+price-bucket → skin → global fallback). *Live Supply* is "
                "CSFloat's count of live listings of the same name — a rough "
                "liquidity hint."
            )
        else:
            st.info("No deals found in the current listings. Try scanning more pages or waiting for new listings.")


# === PAGE: Score a Listing ===
elif page == "Score a Listing":
    st.title("Score a Single Listing")
    st.caption(
        f"Model: **{version_tag}** — Enter a listing's attributes to get its "
        "estimated fair-value range."
    )

    # --- Fetch a live listing by URL or ID ---
    supported_bases = {s["base"].replace("★ ", "") for s in SKINS}

    listing_input = st.text_input(
        "CSFloat listing URL or ID",
        placeholder="https://csfloat.com/item/824952510281353471",
        help="Paste a listing URL (or bare numeric ID) to fetch and score it automatically.",
    )

    if st.button("Fetch & Score", type="primary"):
        listing_id = parse_listing_id(listing_input)
        if listing_id is None:
            st.error(
                "Couldn't read that. Paste a CSFloat listing URL like "
                "`https://csfloat.com/item/824952510281353471` or a bare numeric ID."
            )
        else:
            raw = None
            try:
                raw = CSFloatClient().get_listing(listing_id)
            except (CSFloatError, requests.RequestException) as exc:
                st.error(f"Couldn't fetch listing `{listing_id}`: {exc}")

            if raw is not None:
                flat = flatten_listing(raw)
                if flat["skin_base"] not in supported_bases:
                    supported = ", ".join(s["base"] for s in SKINS)
                    st.error(
                        f"Unsupported skin: **{flat['market_hash_name'] or 'unknown item'}**. "
                        f"The model only supports: {supported}."
                    )
                elif (flat["price_usd"] is None or flat["price_usd"] <= 0
                      or flat["float_value"] is None or flat["paint_seed"] is None):
                    st.error("Listing is missing price, float, or seed data — can't score it.")
                else:
                    refs = get_active_references()
                    scored = score_df(pd.DataFrame([flat]), refs)
                    row = scored.iloc[0]
                    ask_usd = row["price_usd"]

                    st.divider()
                    st.markdown(f"**{row['market_hash_name']}**")
                    if flat["state"] != "listed":
                        st.caption(f"Note: this listing's state is **{flat['state']}**, not currently listed.")

                    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                    col_f1.metric("Ask Price", f"${ask_usd:.2f}")
                    col_f2.metric("Low (10th pctile)", f"${row['low_usd']:.2f}")
                    col_f3.metric("Mid (50th pctile)", f"${row['mid_usd']:.2f}")
                    col_f4.metric("High (90th pctile)", f"${row['high_usd']:.2f}")

                    detail = (
                        f"Float: {row['float_value']:.6f} · Seed: {int(row['paint_seed'])} · "
                        f"Exterior: {row['exterior']}"
                    )
                    phase = row.get("doppler_phase")
                    if phase and phase != "n/a":
                        detail += f" · Doppler Phase: **{phase}**"
                    st.caption(detail)

                    st.divider()
                    if ask_usd < row["low_usd"]:
                        discount = (1 - ask_usd / row["low_usd"]) * 100
                        st.success(
                            f"**Deal!** At ${ask_usd:.2f}, this ask is {discount:.1f}% below "
                            f"the model's low estimate for comparable asks."
                        )
                    elif ask_usd > row["high_usd"]:
                        premium = (ask_usd / row["high_usd"] - 1) * 100
                        st.warning(
                            f"At ${ask_usd:.2f}, this ask is {premium:.1f}% above the model's "
                            f"high estimate for comparable asks."
                        )
                    else:
                        st.info(
                            f"At ${ask_usd:.2f}, this ask is within the model's estimated "
                            f"range for comparable asks."
                        )

                    if use_v15:
                        st.caption(
                            "v1.5 basis: trained on inferred sold prices (collector disappearance data). "
                            "More accurate than asking-price estimates."
                        )
                    else:
                        st.caption(
                            "⚠️ v1 basis: trained on asking prices. 'Deal' = cheaper than comparable "
                            "current asks, not below true fair value."
                        )

    st.divider()
    st.subheader("Or enter attributes manually")

    col1, col2 = st.columns(2)
    with col1:
        skin_base = st.selectbox("Skin", [s["base"] for s in SKINS])
        exterior = st.selectbox("Exterior", EXTERIORS)
        is_stattrak = st.checkbox("StatTrak™")
        price_usd = st.number_input("Asking Price (USD)", min_value=1.0, value=1500.0, step=50.0)

    with col2:
        float_value = st.number_input("Float Value", min_value=0.0, max_value=1.0,
                                      value=0.02, step=0.001, format="%.6f")
        paint_seed = st.number_input("Paint Seed", min_value=0, max_value=999, value=100)
        paint_index = st.number_input("Paint Index", min_value=0, max_value=1000, value=418)
        def_index = st.number_input("Def Index", min_value=0, value=507)

    prefix = "★ StatTrak™ " if is_stattrak else "★ "
    weapon_finish = skin_base.replace("★ ", "")
    market_hash_name = f"{prefix}{weapon_finish} ({exterior})"

    if st.button("Score", type="primary"):
        refs = get_active_references()

        raw = {
            "id": "manual",
            "price": int(price_usd * 100),
            "state": "listed",
            "type": "buy_now",
            "created_at": None,
            "_pulled_at": None,
            "item": {
                "market_hash_name": market_hash_name,
                "float_value": float_value,
                "paint_seed": paint_seed,
                "paint_index": paint_index,
                "def_index": def_index,
                "is_stattrak": is_stattrak,
            },
            "reference": {"predicted_price": None, "quantity": None},
        }
        flat = flatten_listing(raw)
        df = pd.DataFrame([flat])
        scored = score_df(df, refs)
        row = scored.iloc[0]

        is_deal = row["price_usd"] < row["low_usd"]

        st.divider()
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Low (10th pctile)", f"${row['low_usd']:.2f}")
        col_r2.metric("Mid (50th pctile)", f"${row['mid_usd']:.2f}")
        col_r3.metric("High (90th pctile)", f"${row['high_usd']:.2f}")

        st.divider()
        if is_deal:
            discount = (1 - row["price_usd"] / row["low_usd"]) * 100
            st.success(
                f"**Deal!** At ${price_usd:.2f}, this is {discount:.1f}% below the model's "
                f"low estimate for comparable asks."
            )
        else:
            st.info(
                f"At ${price_usd:.2f}, this listing is within or above the model's "
                f"estimated range for comparable asks."
            )

        phase = row.get("doppler_phase")
        if phase and phase != "n/a":
            st.caption(f"Doppler Phase: **{phase}** (from paint_index {paint_index})")

        if use_v15:
            st.caption(
                "v1.5 basis: trained on inferred sold prices (collector disappearance data). "
                "More accurate than asking-price estimates."
            )
        else:
            st.caption(
                "⚠️ v1 basis: trained on asking prices. 'Deal' = cheaper than comparable "
                "current asks, not below true fair value."
            )


# === PAGE: Track Record ===
elif page == "Track Record":
    st.title("Track Record — Predictions vs Actual Sold Prices")
    st.caption(
        "Both models scored against real sales recorded by the collector "
        "(terminal state = 'sold'). Generated by `scripts/backtest.py` — "
        "independent of the Model Version selector."
    )

    @st.cache_data(ttl=300)
    def get_backtest() -> tuple[pd.DataFrame, dict] | None:
        preds_path = BACKTEST_DIR / "predictions.parquet"
        summary_path = BACKTEST_DIR / "summary.json"
        if not preds_path.exists() or not summary_path.exists():
            return None
        return pd.read_parquet(preds_path), json.loads(summary_path.read_text())

    backtest = get_backtest()
    if backtest is None:
        st.info(
            "No backtest results yet. Run:\n"
            "`PYTHONPATH=. .venv/bin/python scripts/backtest.py`"
        )
        st.stop()

    preds, summary = backtest

    FRAMING = {
        "v1": "Trained on **asking prices**, evaluated on all sold rows (it never "
              "saw them). Asks sit above sale prices, so v1 *should* overestimate — "
              "a positive median error here is expected, not a bug.",
        "v1.5": "Trained on **inferred sold prices**. To avoid leakage it is only "
                "evaluated on sales *after* its time-split training cutoff"
                + (f" ({summary.get('v15_cutoff_last_seen', '')[:10]})"
                   if summary.get("v15_cutoff_last_seen") else "")
                + " — a small sample, so read these numbers with caution.",
    }

    for version in ["v1", "v1.5"]:
        if version not in summary["models"]:
            continue
        m = summary["models"][version]
        sub = preds[preds["model_version"] == version]

        st.divider()
        st.subheader(f"{version} — n = {m['n']} sold listings")
        st.caption(FRAMING[version])

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("MAE", f"${m['mae_usd']:.0f}")
        col_m2.metric("Median APE", f"{m['median_ape_pct']:.1f}%")
        col_m3.metric("[q10, q90] coverage", f"{m['coverage_q10_q90']:.0%}",
                      help="Fraction of actual sale prices inside the predicted "
                           "range. Well-calibrated ≈ 80%.")
        col_m4.metric("Median error", f"${m['median_error_usd']:+,.0f}",
                      help="Predicted mid − actual sale price. Positive = "
                           "overestimates sales.")

        scatter = sub.rename(columns={
            "actual_usd": "Actual sale ($)", "mid_usd": "Predicted mid ($)",
            "skin_base": "Skin",
        })
        st.scatter_chart(scatter, x="Actual sale ($)", y="Predicted mid ($)",
                         color="Skin")

        per_skin = sub.groupby("skin_base").agg(
            n=("abs_err_usd", "size"),
            mae_usd=("abs_err_usd", "mean"),
            median_ape_pct=("ape_pct", "median"),
            coverage=("in_range", "mean"),
        ).reset_index()
        per_skin["coverage"] = per_skin["coverage"] * 100
        per_skin.columns = ["Skin", "N", "MAE ($)", "Median APE (%)", "Coverage (%)"]
        st.dataframe(
            per_skin,
            use_container_width=True,
            hide_index=True,
            column_config={
                "MAE ($)": st.column_config.NumberColumn(format="$%.0f"),
                "Median APE (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Coverage (%)": st.column_config.NumberColumn(format="%.0f%%"),
            },
        )

    st.divider()
    st.caption(f"Backtest generated: {summary.get('generated_at', 'unknown')[:19]} UTC")
