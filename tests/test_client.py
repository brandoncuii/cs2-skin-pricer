"""Tests for cs2pricer.client module (mocked HTTP, no network)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cs2pricer.client import CSFloatClient, CSFloatError, parse_listing_id


class TestParseListingId:
    def test_bare_numeric(self):
        assert parse_listing_id("123456") == "123456"

    def test_full_url(self):
        url = "https://csfloat.com/item/7891011"
        assert parse_listing_id(url) == "7891011"

    def test_url_without_https(self):
        assert parse_listing_id("csfloat.com/item/999") == "999"

    def test_url_with_www(self):
        assert parse_listing_id("https://www.csfloat.com/item/555") == "555"

    def test_url_with_query_params(self):
        assert parse_listing_id("https://csfloat.com/item/123?foo=bar") == "123"

    def test_url_with_trailing_slash(self):
        assert parse_listing_id("https://csfloat.com/item/123/") == "123"

    def test_invalid_url_returns_none(self):
        assert parse_listing_id("https://example.com/item/123") is None

    def test_non_numeric_returns_none(self):
        assert parse_listing_id("abc") is None

    def test_whitespace_stripped(self):
        assert parse_listing_id("  123456  ") == "123456"


class TestCSFloatClientRateLimit:
    @patch("cs2pricer.client.api_key", return_value="test-key")
    def test_successful_get(self, mock_key):
        client = CSFloatClient(min_interval=0.0)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.ok = True
        mock_resp.json.return_value = {"data": [{"id": "1"}]}
        mock_resp.headers = {
            "x-ratelimit-remaining": "10",
            "x-ratelimit-reset": "0",
        }

        with patch.object(client._session, "get", return_value=mock_resp):
            result = client.get_listings()
            assert result == {"data": [{"id": "1"}]}

    @patch("cs2pricer.client.api_key", return_value="test-key")
    @patch("cs2pricer.client.time.sleep")
    def test_429_retry(self, mock_sleep, mock_key):
        client = CSFloatClient(min_interval=0.0, base_backoff=0.01, max_wait=0.1)

        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.ok = False
        resp_429.headers = {
            "Retry-After": "0.01",
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": "0",
        }

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.ok = True
        resp_200.json.return_value = {"data": []}
        resp_200.headers = {"x-ratelimit-remaining": "10", "x-ratelimit-reset": "0"}

        with patch.object(client._session, "get", side_effect=[resp_429, resp_200]):
            result = client.get_listings()
            assert result == {"data": []}

    @patch("cs2pricer.client.api_key", return_value="test-key")
    @patch("cs2pricer.client.time.sleep")
    def test_500_retry(self, mock_sleep, mock_key):
        client = CSFloatClient(min_interval=0.0, base_backoff=0.01)

        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.ok = False
        resp_500.headers = {}

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.ok = True
        resp_200.json.return_value = {"data": []}
        resp_200.headers = {"x-ratelimit-remaining": "10", "x-ratelimit-reset": "0"}

        with patch.object(client._session, "get", side_effect=[resp_500, resp_200]):
            result = client.get_listings()
            assert result == {"data": []}

    @patch("cs2pricer.client.api_key", return_value="test-key")
    @patch("cs2pricer.client.time.sleep")
    def test_max_retries_exceeded(self, mock_sleep, mock_key):
        client = CSFloatClient(min_interval=0.0, base_backoff=0.01, max_retries=2)

        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.ok = False
        resp_500.headers = {}

        with patch.object(client._session, "get", return_value=resp_500):
            with pytest.raises(CSFloatError, match="Gave up after"):
                client.get_listings()

    @patch("cs2pricer.client.api_key", return_value="test-key")
    def test_4xx_raises_immediately(self, mock_key):
        client = CSFloatClient(min_interval=0.0)

        resp_403 = MagicMock()
        resp_403.status_code = 403
        resp_403.ok = False
        resp_403.reason = "Forbidden"
        resp_403.text = "Access denied"
        resp_403.headers = {}

        with patch.object(client._session, "get", return_value=resp_403):
            with pytest.raises(CSFloatError, match="403"):
                client.get_listings()

    @patch("cs2pricer.client.api_key", return_value="test-key")
    def test_get_listing_by_id(self, mock_key):
        client = CSFloatClient(min_interval=0.0)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.ok = True
        mock_resp.json.return_value = {"id": "42", "price": 1000}
        mock_resp.headers = {"x-ratelimit-remaining": "5", "x-ratelimit-reset": "0"}

        with patch.object(client._session, "get", return_value=mock_resp) as mock_get:
            result = client.get_listing("42")
            assert result == {"id": "42", "price": 1000}
            mock_get.assert_called_once()
            call_url = mock_get.call_args[0][0]
            assert "/listings/42" in call_url

    @patch("cs2pricer.client.api_key", return_value="test-key")
    def test_rate_limit_headers_parsed(self, mock_key):
        client = CSFloatClient(min_interval=0.0)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.ok = True
        mock_resp.json.return_value = {}
        mock_resp.headers = {
            "x-ratelimit-remaining": "3",
            "x-ratelimit-reset": "1700000000.0",
        }

        with patch.object(client._session, "get", return_value=mock_resp):
            client.get_listings()
            assert client._rl_remaining == 3
            assert client._rl_reset == 1700000000.0
