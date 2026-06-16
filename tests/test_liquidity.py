"""Tests for cs2pricer.liquidity module."""

from __future__ import annotations

from cs2pricer.liquidity import bucket_rel_to_mid


class TestBucketRelToMid:
    def test_very_negative(self):
        # rel <= -0.10 -> first bucket
        assert bucket_rel_to_mid(-0.15) == "<=-10% vs mid"
        assert bucket_rel_to_mid(-0.10) == "<=-10% vs mid"

    def test_slightly_negative(self):
        # -0.10 < rel <= 0.0
        assert bucket_rel_to_mid(-0.05) == "-10..0% vs mid"
        assert bucket_rel_to_mid(0.0) == "-10..0% vs mid"

    def test_slightly_positive(self):
        # 0.0 < rel <= 0.10
        assert bucket_rel_to_mid(0.05) == "0..+10% vs mid"
        assert bucket_rel_to_mid(0.10) == "0..+10% vs mid"

    def test_very_positive(self):
        # rel > 0.10
        assert bucket_rel_to_mid(0.15) == ">+10% vs mid"
        assert bucket_rel_to_mid(1.0) == ">+10% vs mid"

    def test_none_input(self):
        assert bucket_rel_to_mid(None) is None

    def test_nan_input(self):
        assert bucket_rel_to_mid(float("nan")) is None

    def test_boundary_exact(self):
        # Exact boundary values
        assert bucket_rel_to_mid(-0.10) == "<=-10% vs mid"
        assert bucket_rel_to_mid(0.10) == "0..+10% vs mid"
