"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_raw_listing() -> dict:
    """A realistic raw CSFloat listing dict for test use."""
    return {
        "id": "12345678",
        "created_at": "2024-03-01T12:00:00Z",
        "_pulled_at": "2024-03-02T00:00:00Z",
        "state": "listed",
        "type": "buy_now",
        "price": 150000,
        "item": {
            "market_hash_name": "★ Karambit | Doppler (Factory New)",
            "is_stattrak": False,
            "def_index": 507,
            "paint_index": 418,
            "paint_seed": 100,
            "float_value": 0.03,
        },
        "reference": {
            "predicted_price": 140000,
            "quantity": 50,
        },
    }


@pytest.fixture
def sample_raw_listings(sample_raw_listing: dict) -> list[dict]:
    """Multiple raw listings for build_clean tests."""
    listings = [sample_raw_listing]
    # A second valid listing
    listings.append(
        {
            "id": "12345679",
            "created_at": "2024-03-01T13:00:00Z",
            "_pulled_at": "2024-03-02T01:00:00Z",
            "state": "listed",
            "type": "buy_now",
            "price": 160000,
            "item": {
                "market_hash_name": "★ Karambit | Case Hardened (Minimal Wear)",
                "is_stattrak": False,
                "def_index": 507,
                "paint_index": 44,
                "paint_seed": 387,
                "float_value": 0.12,
            },
            "reference": {
                "predicted_price": 120000,
                "quantity": 30,
            },
        }
    )
    # A StatTrak listing
    listings.append(
        {
            "id": "12345680",
            "created_at": "2024-03-01T14:00:00Z",
            "_pulled_at": "2024-03-02T02:00:00Z",
            "state": "listed",
            "type": "buy_now",
            "price": 200000,
            "item": {
                "market_hash_name": "★ StatTrak™ M9 Bayonet | Doppler (Factory New)",
                "is_stattrak": True,
                "def_index": 508,
                "paint_index": 415,
                "paint_seed": 200,
                "float_value": 0.01,
            },
            "reference": {
                "predicted_price": 180000,
                "quantity": 20,
            },
        }
    )
    return listings
