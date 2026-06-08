# PLAN.md — CS2 Knife Fair-Value Model (v1)

## 1. What this is

A supervised regression model that prices CS2 knife listings on **CSFloat** based on their
attributes (float, pattern, exterior, etc.), and flags listings priced **below** the model's
estimated range for comparable items.

The output is a relative-value signal: *"is this listing cheap relative to comparable items
right now?"* It is a tool for **not overpaying**, used to help pick a knife to buy.

## 2. What this explicitly is NOT

- **Not a price predictor.** This does not forecast whether a skin will appreciate. Appreciation
  is driven by exogenous factors (Valve supply changes, meta, market sentiment) that are not in
  the feature set. Do not add "will it go up" framing to scope or UI.
- **Not a multi-market arbitrage tool.** Single market (CSFloat) only for v1.
- **Not a browser extension.** Website/service only for v1. The extension is a possible v2 wrapper
  over the v1 API, not a prerequisite.

## 3. Hard scope boundaries (v1)

**In scope**
- 3–5 specific knife skins, chosen by the owner *and gated on liquidity* (see Phase 0). Must
  include at least one Doppler variant so the phase-as-feature logic gets exercised.
- `buy_now` listings only.
- Predict a fair-value **range** for an item; flag listings below the lower bound.

**Out of scope — do not build these in v1**
- Other skin categories (rifles, gloves, etc.)
- Every-skin coverage
- Multi-market / arbitrage
- Real-time price alerts
- Browser extension
- Auction listings, trade offers, buy orders
- AWS / monorepo / any non-Python infra

If a feature is not in the "In scope" list, it does not get built in v1. Widen only after v1 ships.

## 4. Tech stack (keep it boring)

- **Language:** Python end-to-end. No TypeScript monorepo, no AWS.
- **Data/ML:** pandas, scikit-learn, and **LightGBM or XGBoost** (gradient-boosted trees).
  Do NOT use a neural network — this is a tabular problem with threshold effects (exterior cliffs,
  discrete pattern regimes) that trees model naturally and a net models worse.
- **Serving:** FastAPI service that loads the trained model and returns a fair-value range for a
  given listing.
- **Frontend:** minimal React/Vite page hitting the API, OR Streamlit if skipping frontend for v1.
- **Storage:** local SQLite or Parquet files for v1. No managed DB.

## 5. Data sources and the asking-price → sold-price plan

### What's actually available
- **CSFloat Market API** (`csfloat.com/api/v1/listings`, docs.csfloat.com): real and documented.
  Returns active listings with `float_value`, `paint_seed`, `paint_index`, `def_index`,
  `is_stattrak`, `market_hash_name`, `price` (cents), `collection`, and filters by
  `min_float`/`max_float`, `paint_seed`, `paint_index`, `market_hash_name`, plus `sort_by`
  (`lowest_float`, `float_rank`, `lowest_price`, etc.). This is the data source. **Use this.**
- **FloatDB** (CSFloat's float *search* database): **NO API**, automation actively blocked. Out of
  scope. Do not try to query it.
- The official Market API has **no sales-history endpoint** (only: get all listings, get one
  listing, list an item). So sold prices are NOT directly available from the official API.
- **Paid third-party sold-price APIs (cs2.sh, Pricempire, etc.) are rejected** — too expensive
  (~$75/mo) for a non-monetized learning project.

### The chosen approach: hybrid (asking prices now, sold prices collected in parallel)
Sold prices are the *right* training target, but we don't have a historical set and won't pay for
one. So:

1. **v1 trains on ASKING prices** from the Market API — available immediately. This is a weaker but
   legitimate signal: the model learns "priced below what comparable items are *currently listed*
   for." **This must be labeled honestly in code, docs, and UI** — it is relative-to-asking value,
   NOT true fair value. Do not call it "fair value" without the asterisk.
2. **In parallel, run a background collector** (see §10, v1.5) that polls `/listings` for the chosen
   skins, records each listing ID + attributes + asking price, and periodically re-checks each ID
   via the "get a specific listing" endpoint. When a listing leaves `listed`, treat it as a likely
   sale near its last asking price. This accumulates a real sold-price dataset over weeks at zero
   cost. **Phase 0 must empirically confirm** that the specific-listing endpoint still returns a
   listing after it leaves `listed` — the whole v1.5 strategy depends on it; do not assume from docs.
3. **v1.5 retrains on the collected sold prices.** "Disappeared = sold" is approximate (a listing
   can also be delisted/expired/price-changed) — acknowledge that noise; don't pretend it's clean.

Build v1 against asking prices now; do not block on sold-price collection.

### Known biases of asking-price data (state these, don't hide them)
- **Selection bias points high.** Cheap listings sell and leave; overpriced ones linger. A live
  snapshot over-weights stale, overpriced asks, which biases the estimated range *upward* and makes
  the underpriced flag *conservative* (you'll miss some real deals). The v1.5 sold data fixes this.
- **The "underpriced" flag is circular by construction in v1.** Trained on asks and flagging below
  the model = "this is among the cheapest current asks for comparable items." That is a useful
  not-overpaying filter, but it is NOT "below what it will sell for." The UI must not imply the
  stronger claim.

## 6. Data pipeline rules

- Use the **official CSFloat Market API** with a personal API key (developer tab on profile). No DOM
  scraping, and do not touch FloatDB.
- `GET /api/v1/listings` is paginated via an opaque `cursor`, max `limit` 50 per call. Filter by
  `market_hash_name` / `def_index` to pull only the chosen skins.
- The "get a specific listing" endpoint (`/api/v1/listings/<id>`) is the hook the v1.5 collector
  relies on (see §5.2 — confirm its post-`listed` behavior in Phase 0).
- Be a polite client: respect per-endpoint rate limits (read the HTTP response headers; 429 = back
  off), cache responses, use exponential backoff on errors. Automated actions are at-your-own-risk
  per CSFloat, so do not hammer it.
- Prices from the API are in **cents** — normalize to dollars once, early.
- Store raw pulls separately from cleaned data so the cleaning step is reproducible.
- Dedup, drop malformed rows, and handle outliers explicitly (see modeling notes).

## 7. Feature engineering (this is where the domain knowledge lives)

- **Float**: continuous but NOT smooth. Effect has cliffs at exterior boundaries (FN/MW/FT/WW/BS
  cutoffs) and a collector premium at extremes. Engineer features like:
  - distance to nearest exterior boundary
  - float percentile *within the same skin*
- **Pattern / paint seed**: categorical and lumpy, NOT ordinal. Seed 661 is not "more than" 660.
  Encode the meaningful regimes:
  - Doppler / Gamma Doppler phase (Phase 1–4, Ruby, Sapphire, Black Pearl, Emerald). **Data
    dependency:** phase is NOT in the API; it is derived from `paint_seed`/`paint_index` via an
    external, community-maintained lookup table. Sourcing and validating that table is real Phase-2
    work — treat it as a static, vetted asset checked into the repo, and confirm whether the key is
    seed or index for the chosen knives.
  - Case Hardened blue-gem tier (specific seeds), Fade percentage, Marble Fade placement — same
    pattern: each needs an external mapping. Only build the ones the chosen skins actually need.
- **Skin identity → reference price (the key normalization).** Skin identity dominates price level.
  Do NOT make the model relearn each skin's base price. Model the **log premium relative to a
  per-skin reference**:

      target = log(price) - log(reference_for_skin)

  where `reference_for_skin` = a **robust central tendency (median or trimmed mean) of that skin's
  prices over a recent window**, computed at the **skin level** (not per exterior — that would strip
  the exterior effect out of the target and leave the model nothing to learn there).

  Why this form:
  - **Log** makes attribute effects multiplicative (a Ruby is "~2.3×", not "+$400"), which is how
    this market behaves and lets float/pattern effects generalize across skins of different price
    levels. This is the standard **hedonic pricing** normalization (real estate, cars, wine, art).
  - **Median/trimmed mean** is robust to the fat tails noted in §8.
  - In v1 the reference is median of current *asks*, so it inherits the asking-price asterisk
    (§5). When sold data lands, `reference_for_skin` becomes median *sold* price — same model
    structure, cleaner meaning, no restructuring needed.
- **Flags**: **StatTrak** only. Knives **cannot hold stickers** and **cannot be Souvenir**, so those
  fields are physically constant for this dataset — do not engineer sticker or souvenir features.

## 8. Modeling notes

- **Model:** LightGBM/XGBoost regressor on the log-premium target (§7).
- **Target:** v1 = asking price (honestly labeled per §5); v1.5 = collected sold price. The
  log-premium-vs-reference framing applies either way.
- **Validation — time-based split is a v1.5 deliverable.** A single v1 snapshot has no real time
  axis (every row is scraped at ~the same instant), so a time-based split is degenerate in v1. It
  becomes valid only once the collector has accumulated weeks of history. For v1, use a simple
  held-out split for plumbing checks but **do not report it as evidence of generalization** (see
  evaluation below).
- **Output a range, not a point:** use **quantile regression** (predict the 10th/50th/90th
  percentile) to get a fair-value interval. Note LightGBM/XGBoost train one model per quantile and
  the predicted quantiles can **cross** (low > mid on some rows) — clamp/sort post-hoc.
- **Outliers:** the data is fat-tailed; use robust/quantile objectives and inspect residuals.
- **Evaluation (v1 is honest about having no ground truth):** v1 has no sold prices and no time
  axis, so "$ error on a held-out set" would be measuring asking-price recall against asking
  prices — circular, and indistinguishable from "I memorized the current ask distribution." So v1
  is validated as a **plumbing + feature-engineering milestone**, by:
  - **Sanity checks on learned effects**: float monotonic within a skin, Doppler phase ordering
    (Ruby/Sapphire/Black Pearl > low phases), StatTrak premium has the right sign, etc.
  - **A small hand-built reference set** (~10–20 listings the owner already has a price opinion on)
    to eyeball the model's range against human judgment.
  Real %/$ error against ground truth is a **v1.5 deliverable**, gated on collected sold prices and
  the time-based split.

## 9. Scoring / usage

Given a live listing's attributes, return `{low, mid, high}` estimate.
If `asking_price < low` → flag as **cheaper than comparable current asks** (not "below true value"
— see §5). The UI surfaces flagged listings for the chosen knife skins with that basis disclosed.

## 10. Phases (Claude Code builds these one at a time, stopping between each)

**Phase 0 — Setup + confirm data access + liquidity gate.** Get the API key working; confirm
`/listings` returns the candidate skins with expected fields; **pull per-skin listing counts and
lock the final 3–5 skins on liquidity** (a knife with ~dozens of listings can't support quantile +
Doppler-phase splits — pick partly for volume). Also **empirically confirm** the specific-listing
endpoint's post-`listed` behavior (§5.2) so the v1.5 collector strategy is validated early.

**Phase 1 — Data pipeline.** Pull asking-price listings for the chosen skins, store raw, clean,
dedup, handle outliers. Deliverable: a clean table viewable in a notebook. Validate correctness
before moving on.

**Phase 2 — Feature engineering.** Implement the §7 features, including the seed/index → Doppler
phase lookup and the log-premium-vs-reference target. Deliverable: a feature matrix + a short
notebook showing distributions and that the engineered features make sense.

**Phase 3 — Model.** Train LightGBM/XGBoost on the log-premium target with quantile regression,
validate per §8 (sanity checks + hand-built reference set; NOT $-error claims), save the model
artifact. Deliverable: saved model + a sanity-check report.

**Phase 4 — Scoring service.** FastAPI endpoint that loads the model and returns `{low, mid, high}`
for a given listing. Deliverable: a running local API.

**Phase 5 — Frontend.** Minimal page: browse flagged listings for the chosen skins, or paste a
listing to score it. Must clearly state the model is trained on asking prices in v1 and that the
flag means "cheaper than comparable asks." Deliverable: a usable local UI.

Get each phase working AND understood before starting the next.

### v1.5 milestone — sold-price upgrade (runs in parallel, retrain later)
- **Background collector:** a scheduled job that polls `/listings` for the chosen skins, records
  listing IDs + attributes + asking prices, and periodically re-checks each ID via the
  specific-listing endpoint to detect when it leaves `listed`. Start this collector as early as
  possible (even during Phase 1) so data accumulates while you build.
- After a few weeks of collection: switch the reference to median *sold* price, add the **time-based
  split**, **retrain on sold prices**, and report real %/$ error vs. the asking-price model. Update
  the UI to drop the asking-price asterisk once trained on real sales.

## 11. Definition of done (v1)

- Pulls real CSFloat data for the chosen knife skins (selected on liquidity).
- Trains on asking prices using the log-premium-vs-reference target, honestly labeled.
- Returns a fair-value range and flags below-range listings, with the asking-price basis AND the
  "cheaper than comparable asks (not true value)" meaning clearly disclosed in the UI.
- v1 is validated by sanity checks + a hand-built reference set (NOT $-error claims); real %/$ error
  and the time-based split are explicitly deferred to v1.5.
- The background sold-price collector is running and accumulating data.
- The owner can explain, without notes: why GBT over a neural net; why a log-premium-vs-reference
  target and what the reference is; why the time-based split is deferred to v1.5; what quantile
  regression buys here; how each engineered feature affects price; **why training on asking prices
  is a limitation, why the v1 flag is "cheapest current ask," and what sold prices would change.**
  (If you can't defend it in an interview, it's not done.)
