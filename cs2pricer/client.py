"""Minimal, polite client for the official CSFloat Market API.

Docs: https://docs.csfloat.com. We only use documented read endpoints:
  - GET /api/v1/listings           (paginated via opaque cursor, max limit 50)
  - GET /api/v1/listings/<id>      (a single listing, regardless of state)

Be a polite client: respect rate limits, back off on 429/5xx, never scrape the DOM,
never touch FloatDB.
"""
from __future__ import annotations

import re
import time
from typing import Any

import requests

from .config import api_key

BASE_URL = "https://csfloat.com/api/v1"
MAX_LIMIT = 50  # API hard cap per /listings call

_ITEM_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?csfloat\.com/item/(\d+)(?:[/?#].*)?$"
)


def parse_listing_id(text: str) -> str | None:
    """Extract a listing id from a CSFloat item URL or a bare numeric id.

    Accepts 'https://csfloat.com/item/<id>', 'csfloat.com/item/<id>', or '<id>'.
    Returns None for anything else.
    """
    text = text.strip()
    if text.isdigit():
        return text
    m = _ITEM_URL_RE.match(text)
    return m.group(1) if m else None


class CSFloatError(RuntimeError):
    """Raised when the API returns a non-retryable error."""


class CSFloatClient:
    def __init__(self, *, max_retries: int = 5, base_backoff: float = 2.0,
                 min_interval: float = 1.0, max_wait: float = 1900.0):
        # Auth is the raw API key in the Authorization header (no "Bearer").
        self._session = requests.Session()
        self._session.headers.update({"Authorization": api_key()})
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._min_interval = min_interval  # polite floor between calls (seconds)
        self._max_wait = max_wait          # cap on any single rate-limit sleep
        self._last_call = 0.0
        # Updated from x-ratelimit-* headers so we can wait *before* hitting 429.
        self._rl_remaining: int | None = None
        self._rl_reset: float = 0.0

    def _reset_in(self) -> float:
        return max(0.0, min(self._rl_reset - time.time(), self._max_wait))

    def _throttle(self) -> None:
        # Proactive: if the last response said the window is empty, wait for reset.
        if self._rl_remaining is not None and self._rl_remaining <= 0:
            wait = self._reset_in()
            if wait > 0:
                time.sleep(wait + 1)
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def _note_headers(self, resp: requests.Response) -> None:
        rem = resp.headers.get("x-ratelimit-remaining")
        rst = resp.headers.get("x-ratelimit-reset")
        if rem is not None:
            try:
                self._rl_remaining = int(rem)
            except ValueError:
                pass
        if rst is not None:
            try:
                self._rl_reset = float(rst)
            except ValueError:
                pass

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{BASE_URL}{path}"
        for attempt in range(self._max_retries):
            self._throttle()
            try:
                resp = self._session.get(url, params=params, timeout=30)
            except requests.RequestException:
                # Transient network failure (connection reset, DNS, timeout):
                # back off and retry, same as a 5xx.
                self._last_call = time.monotonic()
                time.sleep(self._base_backoff * (2 ** attempt))
                continue
            self._last_call = time.monotonic()
            self._note_headers(resp)

            if resp.status_code == 429:
                # Honor the reset header (Retry-After as fallback), then retry.
                retry_after = resp.headers.get("Retry-After")
                wait = (float(retry_after) if retry_after else self._reset_in()
                        or self._base_backoff * (2 ** attempt))
                time.sleep(min(wait, self._max_wait) + 1)
                continue
            if resp.status_code >= 500:
                time.sleep(self._base_backoff * (2 ** attempt))
                continue
            if not resp.ok:
                raise CSFloatError(f"{resp.status_code} {resp.reason} for {url}: {resp.text[:300]}")

            return resp.json()

        raise CSFloatError(f"Gave up after {self._max_retries} retries for {url}")

    def get_listings(self, **params: Any) -> dict[str, Any]:
        """One page of /listings. Returns the raw response dict (has 'data' + 'cursor')."""
        params.setdefault("limit", MAX_LIMIT)
        return self._get("/listings", params)

    def get_listing(self, listing_id: str) -> dict[str, Any]:
        """A single listing by id (works regardless of listing state)."""
        return self._get(f"/listings/{listing_id}")

    def iter_listings(self, *, max_pages: int | None = None, **params: Any):
        """Yield every listing across pages, following the opaque cursor.

        Stops when a page returns no cursor or fewer than `limit` items.
        """
        params.setdefault("limit", MAX_LIMIT)
        limit = params["limit"]
        pages = 0
        while True:
            page = self.get_listings(**params)
            data = page.get("data", [])
            for listing in data:
                yield listing
            pages += 1
            cursor = page.get("cursor")
            if not cursor or len(data) < limit:
                break
            if max_pages is not None and pages >= max_pages:
                break
            params["cursor"] = cursor
