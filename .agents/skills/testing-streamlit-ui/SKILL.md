---
name: testing-cs2-pricer-ui
description: Test the CS2 Knife Pricer Streamlit frontend end-to-end. Use when verifying model scoring, deal detection, model versioning (v1/v1.5), or UI changes.
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

4. (Optional) For v1.5 testing, train the sold-price model:
   ```bash
   PYTHONPATH=. .venv/bin/python scripts/train_model_v15.py
   ```
   This requires disappeared listings in `data/collector/observations.db` (minimum 30 rows).
   If insufficient data, the script exits gracefully — v1.5 toggle won't appear in the UI.

## Launching the App

```bash
cd /home/ubuntu/repos/cs2-skin-pricer
PYTHONPATH=. .venv/bin/streamlit run app.py --server.port 8501 --server.headless true
```

Open http://localhost:8501 in the browser.

**Port conflicts:** If port 8501 is already in use, kill the existing process:
```bash
fuser -k 8501/tcp
```

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

## Test 2: Model Version Toggle (Sidebar)

### When v1.5 is NOT available (default until collector has enough data):
- Sidebar "Model Version" radio shows ONLY "v1 (asking prices)"
- Blue info box: "v1.5 not available yet. Train it with: PYTHONPATH=. .venv/bin/python scripts/train_model_v15.py"
- Page caption reads "Model: v1"
- v1 basis warning appears after scoring

### When v1.5 IS available (after training on collector data):
- Sidebar shows both "v1 (asking prices)" and "v1.5 (sold prices)" options
- Switching to v1.5 changes page caption to "Model: v1.5"
- Scoring uses v1.5 models; basis text changes to "v1.5 basis: trained on inferred sold prices"

## Test 3: Find Deals (Requires CSFloat API, SLOW)

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

## Test 4: Collector Expansion (Shell, No API Calls)

Verify the expanded skin definitions without hitting the CSFloat API:
```bash
PYTHONPATH=. python -c "
from cs2pricer.skins import SKINS, ALL_KNIFE_SKINS, KNIFE_TYPES, market_hash_names, all_knife_market_hash_names
print(f'KNIFE_TYPES: {len(KNIFE_TYPES)} (expect 20)')
print(f'ALL_KNIFE_SKINS: {len(ALL_KNIFE_SKINS)} (expect 600)')
normal = all_knife_market_hash_names(False)
st = all_knife_market_hash_names(True)
print(f'Total names: {len(normal) + len(st)} (expect 5840)')
print(f'Locked SKINS unchanged: {len(SKINS)} (expect 4)')
"
```

Verify the `--full` CLI flag:
```bash
PYTHONPATH=. python scripts/collect.py --help
# Should show --full and --max-pages flags
```

## Test 5: API Version Parameter (Shell, No CSFloat API)

Verify the API handles the `?version` query parameter correctly:
```bash
PYTHONPATH=. .venv/bin/uvicorn cs2pricer.api:app --port 8000 &
sleep 2

# Health check — should show v15_available: false (unless v1.5 trained)
curl -s http://localhost:8000/health | python -m json.tool

# Score with v1.5 — should fallback to v1 basis text
curl -s -X POST "http://localhost:8000/score?version=v1.5" \
  -H "Content-Type: application/json" \
  -d '{"market_hash_name":"★ Karambit | Doppler (Factory New)","float_value":0.02,"paint_seed":100,"paint_index":418,"def_index":507,"is_stattrak":false,"price_cents":130000}' | python -m json.tool
# basis should be "v1: trained on asking prices (not sold prices)" when v1.5 unavailable

kill %1
```

## Known Quirks

- The Streamlit slider for "Pages per skin" is hard to drag precisely. Consider just leaving it at default.
- The multiselect dropdown opens when you click the X to remove a tag if your click lands slightly off.
- If the previous run hit rate limits, subsequent scans in the same session will be slow.
- The app caches reference prices for 5 minutes (`ttl=300`), so model retraining won't immediately reflect in the UI.
- The app caches v1 and v1.5 models separately via `@st.cache_resource`. If you train a new model, you may need to clear the cache or restart the app.
- Port 8501 might be in use from a previous Streamlit process — use `fuser -k 8501/tcp` to free it.

## Devin Secrets Needed

- `CSFLOAT_API_KEY` — CSFloat Market API key (required for "Find Deals" and data pulls; not needed for "Score a Listing", collector expansion verification, or API version testing)
