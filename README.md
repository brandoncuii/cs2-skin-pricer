# CS2 Knife Fair-Value Pricer

A LightGBM quantile-regression model that prices CS2 knife listings on [CSFloat](https://csfloat.com/) and flags deals — items priced below the model's estimated range for comparable listings.

> **Honest disclaimer:** v1 trains on **asking prices**, not sold prices. A "deal" means *cheaper than what comparable items are currently listed for* — it is **not** true fair value. The v1.5 upgrade retrains on inferred sold prices collected over time.

## Features

- **Quantile regression** (q10 / q50 / q90) → fair-value *range*, not a point estimate
- **Log-premium target** normalized per-skin (hedonic pricing), so float/pattern effects generalize across price levels
- **Doppler phase**, Case Hardened blue-gem tier, float distance-to-boundary, and other domain features
- **Three model versions:**
  - **v1** — asking-price baseline (4 locked skins)
  - **v1.5** — retrained on inferred sold prices from the background collector
  - **v1-all** — expanded to all CS2 knife types × finishes
- **Background collector** on a Raspberry Pi that polls CSFloat listings and detects sales (disappearance = likely sold)
- **Streamlit UI** with Find Deals, Score a Listing (paste a CSFloat URL), and Track Record views
- **FastAPI scoring API**

## Locked v1 skins

| Skin | Pattern type |
|------|-------------|
| ★ Karambit \| Doppler | Doppler phase |
| ★ M9 Bayonet \| Doppler | Doppler phase |
| ★ Karambit \| Case Hardened | Blue-gem tier |
| ★ Karambit \| Fade | Fade |

## Setup

```bash
git clone https://github.com/brandoncuii/cs2-skin-pricer.git
cd cs2-skin-pricer

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# CSFloat API key (get it from your CSFloat profile → developer tab)
echo 'CSFLOAT_API_KEY=<your-key>' > .env
```

## Usage

### 1. Pull listings & build data

```bash
PYTHONPATH=. python scripts/pull_listings.py      # raw listings → data/raw/
PYTHONPATH=. python scripts/build_clean.py        # clean → data/clean/
PYTHONPATH=. python scripts/build_features.py     # features → data/features/
```

### 2. Train the model

```bash
PYTHONPATH=. python scripts/train_model.py        # v1 (asking prices)
# or
PYTHONPATH=. python scripts/train_model_v15.py    # v1.5 (sold prices, needs collector data)
PYTHONPATH=. python scripts/train_model_all.py    # v1-all (all knives)
```

### 3. Run the Streamlit app

```bash
PYTHONPATH=. streamlit run app.py
```

### 4. Run the FastAPI service

```bash
PYTHONPATH=. uvicorn cs2pricer.api:app --reload --port 8000
```

API docs at `http://localhost:8000/docs`.

### 5. Background collector (v1.5 data)

```bash
PYTHONPATH=. python scripts/collect.py            # one-shot collection run
PYTHONPATH=. python scripts/collect.py --full     # all knife types
```

For always-on collection, deploy to a Raspberry Pi:

```bash
bash scripts/pi/setup.sh
```

## Project structure

```
cs2pricer/          Core library
  client.py         CSFloat API client (rate-limited, polite)
  clean.py          Listing flattener + cleaner
  features.py       Feature engineering (float, phase, gem tier, log-premium)
  model.py          LightGBM quantile training + prediction
  collector.py      Background sold-price collector (SQLite)
  liquidity.py      Empirical days-to-sell stats
  skins.py          Skin definitions (locked set + all knives)
  api.py            FastAPI scoring service
  config.py         Env/config loader

scripts/            Pipeline scripts
  pull_listings.py  Pull raw listings from CSFloat
  build_clean.py    Clean raw → parquet
  build_features.py Engineer features
  train_model.py    Train v1 model
  train_model_v15.py Train v1.5 (sold prices)
  train_model_all.py Train all-knife model
  backtest.py       Backtest models vs actual sold prices
  collect.py        Run the collector
  build_liquidity.py Compute days-to-sell stats
  status.py         Print collector DB status

app.py              Streamlit frontend
```

## Tech stack

Python · LightGBM · pandas · FastAPI · Streamlit · SQLite

## License

MIT
