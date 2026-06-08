"""The locked v1 knife set (Phase 0 deliverable).

Chosen on liquidity + feature coverage (see PLAN.md Phase 0):
  - Two Dopplers (Karambit, M9) share one phase mapping and let the
    premium-vs-reference framing generalize the phase effect across def_index.
  - Case Hardened carries the paint_seed blue-gem regime AND the only real
    float/exterior spread in the set (Dopplers/Fades are ~all Factory New).
  - Fade exercises the fade-% feature.

`pattern` names the Phase-2 lookup each skin needs. Knife names carry the ★ prefix;
phase/seed regimes are NOT in market_hash_name and get derived in Phase 2.
"""

# All five so the pipeline can iterate; impossible skin x exterior combos simply
# return no listings and get dropped in Phase 1.
EXTERIORS = [
    "Factory New",
    "Minimal Wear",
    "Field-Tested",
    "Well-Worn",
    "Battle-Scarred",
]

SKINS = [
    {"base": "★ Karambit | Doppler", "pattern": "doppler_phase"},
    {"base": "★ M9 Bayonet | Doppler", "pattern": "doppler_phase"},
    {"base": "★ Karambit | Case Hardened", "pattern": "case_hardened"},
    {"base": "★ Karambit | Fade", "pattern": "fade"},
]


def market_hash_names(stattrak: bool = False) -> list[str]:
    """Every (skin x exterior) market_hash_name in the locked set.

    StatTrak knives are a separate name ('★ StatTrak™ ...'); whether to pull them
    is a Phase 1 scope decision (see PLAN.md §7 — is_stattrak is only a live feature
    if ST rows are pulled).
    """
    prefix = "★ StatTrak™ " if stattrak else ""
    names = []
    for skin in SKINS:
        base = skin["base"].replace("★ ", prefix, 1) if stattrak else skin["base"]
        for ext in EXTERIORS:
            names.append(f"{base} ({ext})")
    return names
