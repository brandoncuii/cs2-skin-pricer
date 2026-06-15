"""Tests for cs2pricer.clean module."""

from __future__ import annotations

from cs2pricer.clean import build_clean, flatten_listing, parse_name


class TestParseName:
    def test_standard_knife(self):
        result = parse_name("★ Karambit | Doppler (Factory New)")
        assert result["weapon"] == "Karambit"
        assert result["finish"] == "Doppler"
        assert result["exterior"] == "Factory New"
        assert result["skin_base"] == "Karambit | Doppler"

    def test_stattrak(self):
        result = parse_name("★ StatTrak™ M9 Bayonet | Doppler (Factory New)")
        assert result["weapon"] == "M9 Bayonet"
        assert result["finish"] == "Doppler"
        assert result["exterior"] == "Factory New"

    def test_case_hardened(self):
        result = parse_name("★ Karambit | Case Hardened (Minimal Wear)")
        assert result["weapon"] == "Karambit"
        assert result["finish"] == "Case Hardened"
        assert result["exterior"] == "Minimal Wear"

    def test_battle_scarred(self):
        result = parse_name("★ Karambit | Fade (Battle-Scarred)")
        assert result["exterior"] == "Battle-Scarred"

    def test_invalid_name_returns_nones(self):
        result = parse_name("Not a valid name")
        assert result == {
            "weapon": None,
            "finish": None,
            "exterior": None,
            "skin_base": None,
        }

    def test_empty_string(self):
        result = parse_name("")
        assert result["weapon"] is None


class TestFlattenListing:
    def test_basic_flatten(self, sample_raw_listing):
        flat = flatten_listing(sample_raw_listing)
        assert flat["id"] == "12345678"
        assert flat["price_cents"] == 150000
        assert flat["price_usd"] == 1500.0
        assert flat["weapon"] == "Karambit"
        assert flat["finish"] == "Doppler"
        assert flat["exterior"] == "Factory New"
        assert flat["is_stattrak"] is False
        assert flat["float_value"] == 0.03
        assert flat["paint_seed"] == 100
        assert flat["csfloat_predicted_cents"] == 140000

    def test_missing_price(self, sample_raw_listing):
        sample_raw_listing["price"] = None
        flat = flatten_listing(sample_raw_listing)
        assert flat["price_cents"] is None
        assert flat["price_usd"] is None

    def test_missing_item(self):
        raw = {"id": "999", "price": 100, "item": None, "reference": None}
        flat = flatten_listing(raw)
        assert flat["id"] == "999"
        assert flat["weapon"] is None

    def test_missing_reference(self, sample_raw_listing):
        sample_raw_listing["reference"] = None
        flat = flatten_listing(sample_raw_listing)
        assert flat["csfloat_predicted_cents"] is None


class TestBuildClean:
    def test_basic_clean(self, sample_raw_listings):
        df, report = build_clean(sample_raw_listings)
        assert report["raw_rows"] == 3
        assert report["after_dedup"] == 3
        assert len(df) == report["clean_rows"]
        assert "logprice_z" in df.columns
        assert "price_outlier" in df.columns

    def test_dedup(self, sample_raw_listing):
        # Same listing twice
        records = [sample_raw_listing, sample_raw_listing.copy()]
        df, report = build_clean(records)
        assert report["after_dedup"] == 1

    def test_drops_malformed(self, sample_raw_listing):
        # Zero price
        bad = sample_raw_listing.copy()
        bad["price"] = 0
        bad["id"] = "bad1"
        records = [sample_raw_listing, bad]
        df, report = build_clean(records)
        assert report["dropped_malformed"] >= 1

    def test_drops_price_ceiling(self, sample_raw_listing):
        expensive = sample_raw_listing.copy()
        expensive["id"] = "expensive1"
        expensive["price"] = 5_000_001  # $50,000.01
        records = [sample_raw_listing, expensive]
        df, report = build_clean(records)
        assert report["dropped_price_ceiling"] >= 1
