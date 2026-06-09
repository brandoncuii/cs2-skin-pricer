"""Knife skin definitions for CS2 Knife Pricer.

SKINS — the locked v1 set (4 skins, Phase 0 deliverable).
ALL_KNIFE_SKINS — every CS2 knife type × finish for the expanded collector.

The locked set is unchanged and used by the v1 model / scoring pipeline.
The full set is used by the collector (--full flag) to accumulate sold-price
data across the entire knife market for v1.5.
"""
from __future__ import annotations

EXTERIORS = [
    "Factory New",
    "Minimal Wear",
    "Field-Tested",
    "Well-Worn",
    "Battle-Scarred",
]

# ---------- Locked v1 set (DO NOT MODIFY) ----------

SKINS = [
    {"base": "★ Karambit | Doppler", "pattern": "doppler_phase"},
    {"base": "★ M9 Bayonet | Doppler", "pattern": "doppler_phase"},
    {"base": "★ Karambit | Case Hardened", "pattern": "case_hardened"},
    {"base": "★ Karambit | Fade", "pattern": "fade"},
]


def market_hash_names(stattrak: bool = False) -> list[str]:
    """Every (skin x exterior) market_hash_name in the locked set."""
    prefix = "★ StatTrak™ " if stattrak else ""
    names = []
    for skin in SKINS:
        base = skin["base"].replace("★ ", prefix, 1) if stattrak else skin["base"]
        for ext in EXTERIORS:
            names.append(f"{base} ({ext})")
    return names


# ---------- All CS2 knife types ----------

KNIFE_TYPES = [
    "Bayonet",
    "Bowie Knife",
    "Butterfly Knife",
    "Classic Knife",
    "Falchion Knife",
    "Flip Knife",
    "Gut Knife",
    "Huntsman Knife",
    "Karambit",
    "Kukri Knife",
    "M9 Bayonet",
    "Navaja Knife",
    "Nomad Knife",
    "Paracord Knife",
    "Shadow Daggers",
    "Skeleton Knife",
    "Stiletto Knife",
    "Survival Knife",
    "Talon Knife",
    "Ursus Knife",
]

# Finishes grouped roughly by era. These lists are a SUPERSET tried against every
# knife type: many knife x finish combos don't exist on the market (e.g. Lore,
# Autotronic, Black Laminate, Freehand, Bright Water exist only on the 5 original
# knives). Invalid combos simply return no listings, at the cost of wasted API
# calls — a curated per-knife availability map is a possible follow-up.
TIER1_FINISHES = [
    "Doppler",
    "Marble Fade",
    "Tiger Tooth",
    "Fade",
    "Gamma Doppler",
    "Autotronic",
    "Bright Water",
    "Freehand",
    "Lore",
    "Black Laminate",
    "Case Hardened",
    "Crimson Web",
    "Slaughter",
    "Blue Steel",
    "Stained",
    "Vanilla",
    "Ultraviolet",
    "Night Stripe",
    "Urban Masked",
    "Boreal Forest",
    "Forest DDPAT",
    "Safari Mesh",
    "Scorched",
    "Rust Coat",
]

# Tier 2: newer finishes (also a superset — not every knife has these).
TIER2_FINISHES = [
    "Damascus Steel",
    "Night",
    "Cobalt Skulls",
    "Cosmic Industrial",
    "Galactic Imperial",
    "Spectral Tundra",
]

# Vanilla has no finish name in market_hash_name — it's just "★ Karambit".
# We handle it specially below.

# Build the full list. Each entry has "base" (market_hash_name stem) and "pattern".
_DOPPLER_FINISHES = {"Doppler", "Gamma Doppler"}
_CH_FINISHES = {"Case Hardened"}
_FADE_FINISHES = {"Fade"}


def _pattern_for(finish: str) -> str:
    if finish in _DOPPLER_FINISHES:
        return "doppler_phase"
    if finish in _CH_FINISHES:
        return "case_hardened"
    if finish in _FADE_FINISHES:
        return "fade"
    return "generic"


def _build_all_knife_skins() -> list[dict]:
    skins = []
    seen = set()
    for knife in KNIFE_TYPES:
        # Vanilla (no finish — just "★ Karambit" etc.)
        vanilla_base = f"★ {knife}"
        if vanilla_base not in seen:
            skins.append({"base": vanilla_base, "pattern": "generic"})
            seen.add(vanilla_base)
        # Regular finishes
        for finish in TIER1_FINISHES + TIER2_FINISHES:
            if finish == "Vanilla":
                continue  # handled above
            base = f"★ {knife} | {finish}"
            if base not in seen:
                skins.append({"base": base, "pattern": _pattern_for(finish)})
                seen.add(base)
    return skins


ALL_KNIFE_SKINS = _build_all_knife_skins()


def all_knife_market_hash_names(stattrak: bool = False) -> list[str]:
    """Every (knife × finish × exterior) market_hash_name across ALL knife types.

    Used by the expanded collector. Vanilla knives have no exterior suffix
    in the API, but we still try all exteriors — the API returns empty for
    invalid combos, which is fine.
    """
    prefix = "★ StatTrak™ " if stattrak else ""
    names = []
    for skin in ALL_KNIFE_SKINS:
        base = skin["base"]
        is_vanilla = "|" not in base
        if stattrak:
            base = base.replace("★ ", prefix, 1)
        if is_vanilla:
            # Vanilla knives: no exterior in name, just "★ Karambit"
            names.append(base)
        else:
            for ext in EXTERIORS:
                names.append(f"{base} ({ext})")
    return names
