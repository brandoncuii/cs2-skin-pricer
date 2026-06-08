"""Phase 4 — FastAPI scoring service.

Loads the trained quantile models and exposes endpoints to:
  1. Score a single listing by its CSFloat attributes → {low, mid, high} USD range.
  2. Pull + score all current buy_now listings for the locked skins and return
     those flagged as "cheaper than comparable asks" (asking < model's low estimate).

Honestly labeled: v1 trains on asking prices (PLAN.md §5). The flag means
"priced below what comparable items are currently listed for" — NOT "below true
fair value."

Run: PYTHONPATH=. .venv/bin/python -m uvicorn cs2pricer.api:app --reload --port 8000
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .client import CSFloatClient
from .clean import flatten_listing
from .features import (CATEGORICAL_COLS, DOPPLER_PHASE, FEATURE_COLS,
                       FLOAT_BOUNDARIES, KARAMBIT_CH_GEM_TIER, add_features)
from .model import MODEL_DIR, QUANTILES, load_models, predict
from .skins import SKINS, market_hash_names

app = FastAPI(
    title="CS2 Knife Fair-Value (v1)",
    description=(
        "Quantile-regression model trained on CSFloat asking prices. "
        "Returns a fair-value *range* relative to comparable current asks. "
        "NOT true fair value — see PLAN.md §5."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load models at startup ---
_models: dict[float, Any] | None = None


def _get_models():
    global _models
    if _models is None:
        if not MODEL_DIR.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. Run: PYTHONPATH=. .venv/bin/python scripts/train_model.py",
            )
        _models = load_models()
    return _models


# --- Request/Response schemas ---


class ListingInput(BaseModel):
    """Attributes of a single listing to score."""
    market_hash_name: str = Field(..., example="★ Karambit | Doppler (Factory New)")
    float_value: float = Field(..., ge=0, le=1)
    paint_seed: int = Field(..., ge=0)
    paint_index: int = Field(..., ge=0)
    def_index: int = Field(..., ge=0)
    is_stattrak: bool = False
    price_cents: int = Field(..., gt=0, description="Asking price in cents")


class ScoredListing(BaseModel):
    id: str | None = None
    market_hash_name: str
    price_usd: float
    low_usd: float
    mid_usd: float
    high_usd: float
    is_deal: bool
    discount_pct: float | None = None
    float_value: float
    paint_seed: int
    paint_index: int
    doppler_phase: str | None = None
    exterior: str | None = None


class ScoreResponse(BaseModel):
    low_usd: float
    mid_usd: float
    high_usd: float
    is_deal: bool
    discount_pct: float | None = None
    basis: str = "v1: trained on asking prices (not sold prices)"


class DealsResponse(BaseModel):
    total_scored: int
    deals: list[ScoredListing]
    basis: str = "v1: trained on asking prices — 'deal' means cheaper than comparable current asks, NOT below true fair value"


# --- Helpers ---


def _build_row(inp: ListingInput) -> pd.DataFrame:
    """Build a single-row DataFrame with all features from ListingInput."""
    raw = {
        "id": "manual",
        "price": inp.price_cents,
        "state": "listed",
        "type": "buy_now",
        "created_at": None,
        "_pulled_at": None,
        "item": {
            "market_hash_name": inp.market_hash_name,
            "float_value": inp.float_value,
            "paint_seed": inp.paint_seed,
            "paint_index": inp.paint_index,
            "def_index": inp.def_index,
            "is_stattrak": inp.is_stattrak,
        },
        "reference": {"predicted_price": None, "quantity": None},
    }
    flat = flatten_listing(raw)
    df = pd.DataFrame([flat])

    # We need a reference_usd. Use the median from the clean dataset if available,
    # otherwise fall back to the listing's own price (makes log_premium = 0).
    clean_path = Path("data/clean/listings.parquet")
    if clean_path.exists():
        clean = pd.read_parquet(clean_path)
        skin_base = flat["skin_base"]
        ref = clean[clean["skin_base"] == skin_base]["price_usd"].median()
        if pd.isna(ref) or ref <= 0:
            ref = flat["price_usd"]
    else:
        ref = flat["price_usd"]

    df = add_features(df)
    # Override reference with the actual dataset median.
    df["reference_usd"] = ref
    df["log_premium"] = np.log(df["price_usd"]) - np.log(ref)
    return df


def _score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Score a full DataFrame and add prediction columns."""
    models = _get_models()
    preds = predict(models, df)
    return df.assign(**preds)


# --- Endpoints ---


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _models is not None}


@app.post("/score", response_model=ScoreResponse)
def score_listing(inp: ListingInput):
    """Score a single listing: returns {low, mid, high} USD fair-value range."""
    df = _build_row(inp)
    scored = _score_dataframe(df)
    row = scored.iloc[0]

    is_deal = row["price_usd"] < row["low_usd"]
    discount = None
    if is_deal:
        discount = round((1 - row["price_usd"] / row["low_usd"]) * 100, 1)

    return ScoreResponse(
        low_usd=round(row["low_usd"], 2),
        mid_usd=round(row["mid_usd"], 2),
        high_usd=round(row["high_usd"], 2),
        is_deal=bool(is_deal),
        discount_pct=discount,
    )


@app.get("/deals", response_model=DealsResponse)
def find_deals(max_pages: int = 3):
    """Pull live listings for locked skins and return those priced below model's low estimate.

    This hits the CSFloat API in real-time (rate-limited), so response time depends
    on how many pages are fetched. Default max_pages=3 (~150 listings per skin name).
    """
    client = CSFloatClient()
    names = market_hash_names(stattrak=False) + market_hash_names(stattrak=True)

    raw_listings: list[dict] = []
    for name in names:
        for listing in client.iter_listings(
            market_hash_name=name, type="buy_now", max_pages=max_pages
        ):
            raw_listings.append(listing)

    if not raw_listings:
        return DealsResponse(total_scored=0, deals=[])

    # Flatten and build features.
    flat = [flatten_listing(r) for r in raw_listings]
    df = pd.DataFrame(flat)

    # Drop malformed rows.
    df = df[df["price_usd"].gt(0) & df["float_value"].notna() & df["paint_seed"].notna()].copy()
    df = add_features(df)

    scored = _score_dataframe(df)
    scored["is_deal"] = scored["price_usd"] < scored["low_usd"]
    scored["discount_pct"] = np.where(
        scored["is_deal"],
        ((1 - scored["price_usd"] / scored["low_usd"]) * 100).round(1),
        None,
    )

    deals = scored[scored["is_deal"]].sort_values("discount_pct", ascending=False)

    deal_list = []
    for _, r in deals.iterrows():
        deal_list.append(ScoredListing(
            id=r.get("id"),
            market_hash_name=r["market_hash_name"],
            price_usd=round(r["price_usd"], 2),
            low_usd=round(r["low_usd"], 2),
            mid_usd=round(r["mid_usd"], 2),
            high_usd=round(r["high_usd"], 2),
            is_deal=True,
            discount_pct=r["discount_pct"],
            float_value=r["float_value"],
            paint_seed=int(r["paint_seed"]),
            paint_index=int(r["paint_index"]),
            doppler_phase=r.get("doppler_phase") if r.get("doppler_phase") != "n/a" else None,
            exterior=r.get("exterior"),
        ))

    return DealsResponse(total_scored=len(scored), deals=deal_list)


@app.get("/skins")
def list_skins():
    """Return the locked skin set for the frontend."""
    return {"skins": SKINS, "exteriors": ["Factory New", "Minimal Wear", "Field-Tested",
                                           "Well-Worn", "Battle-Scarred"]}
