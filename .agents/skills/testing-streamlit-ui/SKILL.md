---
name: testing-cs2-pricer-ui
description: Test the CS2 Knife Pricer Streamlit frontend end-to-end. Use when verifying model scoring, deal detection, or UI changes.
---

# Testing the CS2 Knife Pricer Streamlit UI

## Prerequisites

1. Model must be trained first:
   ```bash
   cd /home/ubuntu/repos/cs2-skin-pricer
   PYTHONPATH=. .venv/bin/python scripts/train_model.py
   ```
   This creates `data/model/lgb_q10.txt`, `lgb_q50.txt`, `lgb_q90.txt`.

2. `.env` file must contain `CSFLOAT_API_KEY=<key>` (stored as org secret `CSFLOAT_API_KEY`).

3. Clean data must exist at `data/clean/listings.parquet` (run Phase 1 scripts if missing).

## Launching the App

```bash
cd /home/ubuntu/repos/cs2-skin-pricer
PYTHONPATH=. .venv/bin/streamlit run app.py --server.port 8501 --server.headless true
```

Open http://localhost:8501 in the browser.

## Test 1: Score a Listing (Quick, No API Calls)

This test does NOT call the CSFloat API — it only uses the local model. Always test this first.

1. Select "Score a Listing" in the sidebar
2. Set: Skin = ★ Karambit | Doppler, Exterior = Factory New
3. Set: Price = $1300, Float = 0.02, Paint Seed = 100, Paint Index = 418, Def Index = 507
4. Click "Score"
5. Verify:
   - Low ~$1337, Mid ~$1502, High ~$2690 (exact values depend on training data)
   - Green "Deal!" banner with ~2.8% discount
   - Doppler Phase shows "Phase 1" (from paint_index 418)
   - v1 basis warning at bottom

**Not-a-deal variant:** Change price to $5000, click Score again:
- Blue info message: "within or above the model's estimated range"
- No Deal banner
- Low/Mid/High values unchanged (price doesn't affect prediction)

## Test 2: Find Deals (Requires CSFloat API, SLOW)

**WARNING:** This test makes live CSFloat API calls with rate limiting. With all 4 skins × 5 exteriors × 2 (normal + ST) = 40 names, it can take 5-30+ minutes depending on rate limit state.

**Workaround for rate limiting:**
- Select only 1 skin (e.g., just ★ Karambit | Doppler)
- Set pages slider to 1
- This reduces to ~10 API calls (~30-60 seconds if not rate-limited)

**If the API is already rate-limited** (e.g., from earlier testing or data pulls), the scan might take 10-30 minutes as the client waits for the rate limit window to reset. The `max_wait` is configured at 1900 seconds.

**Alternative verification:** Test the scoring/deal logic via the FastAPI endpoint instead:
```bash
# Start API server
PYTHONPATH=. .venv/bin/uvicorn cs2pricer.api:app --port 8000 &

# Score a single listing (no CSFloat API needed)
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"market_hash_name":"★ Karambit | Doppler (Factory New)","float_value":0.02,"paint_seed":100,"paint_index":418,"def_index":507,"is_stattrak":false,"price_cents":130000}'
```

## Known Quirks

- The Streamlit slider for "Pages per skin" is hard to drag precisely. Consider just leaving it at default.
- The multiselect dropdown opens when you click the X to remove a tag if your click lands slightly off.
- If the previous run hit rate limits, subsequent scans in the same session will be slow.
- The app caches reference prices for 5 minutes (`ttl=300`), so model retraining won't immediately reflect in the UI.

## Devin Secrets Needed

- `CSFLOAT_API_KEY` — CSFloat Market API key (required for "Find Deals" and data pulls; not needed for "Score a Listing")
