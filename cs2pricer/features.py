"""Feature engineering (PLAN.md §7) + the log-premium-vs-reference target (§7/§8).

Design notes grounded in the actual data (see Phase 2 exploration):
  - Doppler phase is encoded by paint_index (NOT paint_seed) for regular Doppler.
    Confirmed against prices: 415=Ruby, 417=Black Pearl, 416=Sapphire dominate;
    418-421 are Phase 1-4. This is exact and drives most Doppler price variance.
  - Float matters mainly for Case Hardened (the only skin with real float spread);
    Dopplers/Fades sit in the Factory New band so float features are ~flat there.
  - Fade %% is a weak price driver for Karambit Fade in this market (price is ~flat
    across seed), so v1 does NOT build a fade-percent table — paint_seed is passed raw.
  - Case Hardened blue-gem value is seed-specific. The tier map below is curated from
    EXTERNAL knowledge only (never from our prices = no leakage) and is intentionally
    partial for v1; a fuller tier list is a follow-up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# CS2 exterior float cutoffs. Interior boundaries are where price "cliffs" happen.
FLOAT_BOUNDARIES = [0.07, 0.15, 0.38, 0.45]

# Regular Doppler: paint_index -> phase (verified against price ordering in our data).
DOPPLER_PHASE = {
    415: "Ruby", 416: "Sapphire", 417: "Black Pearl",
    418: "Phase 1", 419: "Phase 2", 420: "Phase 3", 421: "Phase 4",
}

# Karambit | Case Hardened blue-gem tiers — CURATED from external community tier lists,
# NOT from our prices. Partial on purpose (v1). 3 = top gem, 2 = high, 1 = notable.
# Populate further from a vetted tier list as a follow-up.
KARAMBIT_CH_GEM_TIER = {
    387: 3, 853: 3, 670: 3,
    868: 2, 905: 2, 555: 2, 179: 2,
    321: 1, 852: 1, 955: 1, 463: 1, 896: 1,
}


def _dist_to_boundary(float_value: float) -> float:
    return min(abs(float_value - b) for b in FLOAT_BOUNDARIES)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # --- Float features (§7) ---
    out["dist_to_boundary"] = out["float_value"].apply(_dist_to_boundary)
    out["float_pctile_in_skin"] = (
        out.groupby("skin_base")["float_value"].rank(pct=True)
    )

    # --- Doppler phase (§7) ---
    out["doppler_phase"] = out["paint_index"].map(DOPPLER_PHASE).fillna("n/a")

    # --- Case Hardened blue-gem tier (§7) ---
    # Blue-gem seeds are knife-specific, so the Karambit tier map only applies to
    # Karambit | Case Hardened rows (other knives' CH rows stay 0).
    is_ch = out["finish"].eq("Case Hardened") & out["weapon"].eq("Karambit")
    out["ch_gem_tier"] = 0
    out.loc[is_ch, "ch_gem_tier"] = (
        out.loc[is_ch, "paint_seed"].map(KARAMBIT_CH_GEM_TIER).fillna(0).astype(int)
    )

    # --- Reference price + target (§7/§8) ---
    # Skin-level robust reference: median asking price per skin (pools ST + non-ST so
    # is_stattrak becomes a premium; pools exteriors so exterior effect is learned).
    out["reference_usd"] = out.groupby("skin_base")["price_usd"].transform("median")
    out["log_premium"] = np.log(out["price_usd"]) - np.log(out["reference_usd"])

    return out


# Columns the model trains on. reference_usd is deliberately EXCLUDED (it's the
# normalizer; feeding it back would let the model relearn absolute price level).
FEATURE_COLS = [
    "skin_base", "weapon", "finish", "exterior", "def_index",
    "is_stattrak",
    "float_value", "dist_to_boundary", "float_pctile_in_skin",
    "doppler_phase", "ch_gem_tier", "paint_seed", "paint_index",
]
TARGET_COL = "log_premium"
CATEGORICAL_COLS = ["skin_base", "weapon", "finish", "exterior", "doppler_phase"]
