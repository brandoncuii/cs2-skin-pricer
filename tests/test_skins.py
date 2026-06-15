"""Tests for cs2pricer.skins module."""

from __future__ import annotations

from cs2pricer.skins import (
    ALL_KNIFE_SKINS,
    EXTERIORS,
    KNIFE_TYPES,
    SKINS,
    all_knife_market_hash_names,
    market_hash_names,
)


class TestMarketHashNames:
    def test_locked_set_count(self):
        # 4 skins x 5 exteriors = 20
        names = market_hash_names(stattrak=False)
        assert len(names) == 20

    def test_stattrak_count(self):
        names = market_hash_names(stattrak=True)
        assert len(names) == 20

    def test_contains_expected_name(self):
        names = market_hash_names()
        assert "★ Karambit | Doppler (Factory New)" in names
        assert "★ M9 Bayonet | Doppler (Minimal Wear)" in names

    def test_stattrak_prefix(self):
        names = market_hash_names(stattrak=True)
        for name in names:
            assert "StatTrak™" in name

    def test_all_exteriors_present(self):
        names = market_hash_names()
        for ext in EXTERIORS:
            assert any(ext in n for n in names)


class TestAllKnifeMarketHashNames:
    def test_non_empty(self):
        names = all_knife_market_hash_names()
        assert len(names) > 0

    def test_covers_all_knife_types(self):
        names = all_knife_market_hash_names()
        combined = " ".join(names)
        for knife in KNIFE_TYPES:
            assert knife in combined, f"Missing knife type: {knife}"

    def test_stattrak_variant(self):
        names = all_knife_market_hash_names(stattrak=True)
        # All non-vanilla names should have StatTrak
        for name in names:
            if "|" in name:
                assert "StatTrak™" in name

    def test_vanilla_knives_included(self):
        names = all_knife_market_hash_names()
        # Vanilla has no finish (no pipe), just "★ Karambit" etc.
        vanilla_names = [n for n in names if "|" not in n]
        assert len(vanilla_names) == len(KNIFE_TYPES)

    def test_no_duplicates(self):
        names = all_knife_market_hash_names()
        assert len(names) == len(set(names))


class TestAllKnifeSkins:
    def test_locked_set_is_subset(self):
        all_bases = {s["base"] for s in ALL_KNIFE_SKINS}
        for skin in SKINS:
            assert skin["base"] in all_bases
