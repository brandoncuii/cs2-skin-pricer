"""Phase 5 — Streamlit frontend for CS2 Knife Fair-Value (v1).

Two views:
  1. **Find Deals** — pulls live CSFloat listings, scores them, shows those priced
     below the model's low estimate (i.e., cheaper than comparable current asks).
  2. **Score a Listing** — manually input a listing's attributes and get its
     fair-value range.

Honestly labeled: this model is trained on asking prices. A "deal" means the
listing is priced below what comparable items are currently listed for — NOT below
true fair value (which requires sold-price data, coming in v1.5).

Run: PYTHONPATH=. .venv/bin/streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from cs2pricer.clean import flatten_listing
from cs2pricer.client import CSFloatClient
from cs2pricer.features import DOPPLER_PHASE, add_features
from cs2pricer.model import MODEL_DIR, load_models, predict
from cs2pricer.skins import EXTERIORS, SKINS, market_hash_names

st.set_page_config(page_title="CS2 Knife Pricer (v1)", page_icon="🔪", layout="wide")

# --- Load models ---
@st.cache_resource
def get_models():
    if not MODEL_DIR.exists():
        return None
    return load_models()


models = get_models()
if models is None:
    st.error("Model not trained yet. Run: `PYTHONPATH=. .venv/bin/python scripts/train_model.py`")
    st.stop()

# --- Sidebar ---
st.sidebar.title("🔪 CS2 Knife Pricer")
st.sidebar.caption(
    "**v1 — trained on asking prices.**\n\n"
    "A 'deal' means priced below what comparable items are currently listed for. "
    "This is NOT true fair value (requires sold-price data → v1.5)."
)
page = st.sidebar.radio("View", ["Find Deals", "Score a Listing"])

# --- Helpers ---
CLEAN_PATH = Path("data/clean/listings.parquet")


@st.cache_data(ttl=300)
def get_reference_prices() -> dict[str, float]:
    """Median price per skin_base from the clean dataset."""
    if not CLEAN_PATH.exists():
        return {}
    df = pd.read_parquet(CLEAN_PATH)
    return df.groupby("skin_base")["price_usd"].median().to_dict()


def score_df(df: pd.DataFrame, refs: dict[str, float]) -> pd.DataFrame:
    """Add features and score a DataFrame."""
    df = add_features(df)
    # Override reference with dataset medians for consistency.
    df["reference_usd"] = df["skin_base"].map(refs)
    # Fallback: if a skin isn't in refs, use its own price.
    df["reference_usd"] = df["reference_usd"].fillna(df["price_usd"])
    df["log_premium"] = np.log(df["price_usd"]) - np.log(df["reference_usd"])
    preds = predict(models, df)
    return df.assign(**preds)


# === PAGE: Find Deals ===
if page == "Find Deals":
    st.title("Live Deals — Cheaper Than Comparable Asks")
    st.caption(
        "Pulls current CSFloat buy_now listings, scores them with the model, "
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
        refs = get_reference_prices()
        client = CSFloatClient()

        # Build names for selected skins.
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

        # Flatten and score.
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

        deals = scored[scored["is_deal"]].sort_values("discount_pct", ascending=False)
        st.success(f"Scored {len(scored)} listings — **{len(deals)} deals** found")

        if len(deals) > 0:
            display_cols = [
                "market_hash_name", "price_usd", "low_usd", "mid_usd", "high_usd",
                "discount_pct", "float_value", "paint_seed", "doppler_phase", "exterior",
            ]
            show = deals[display_cols].copy()
            show.columns = [
                "Skin", "Price ($)", "Low ($)", "Mid ($)", "High ($)",
                "Discount %", "Float", "Seed", "Phase", "Exterior",
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
                    "Float": st.column_config.NumberColumn(format="%.6f"),
                },
            )
        else:
            st.info("No deals found in the current listings. Try scanning more pages or waiting for new listings.")


# === PAGE: Score a Listing ===
elif page == "Score a Listing":
    st.title("Score a Single Listing")
    st.caption("Enter a listing's attributes to get its estimated fair-value range.")

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

    # Build market_hash_name.
    prefix = "★ StatTrak™ " if is_stattrak else "★ "
    weapon_finish = skin_base.replace("★ ", "")
    market_hash_name = f"{prefix}{weapon_finish} ({exterior})"

    if st.button("Score", type="primary"):
        refs = get_reference_prices()

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

        # Display result.
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

        # Show Doppler phase if applicable.
        phase = row.get("doppler_phase")
        if phase and phase != "n/a":
            st.caption(f"Doppler Phase: **{phase}** (from paint_index {paint_index})")

        st.caption(
            "⚠️ v1 basis: trained on asking prices. 'Deal' = cheaper than comparable "
            "current asks, not below true fair value."
        )
