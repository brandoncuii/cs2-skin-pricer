# CLAUDE.md — CS2 Knife Pricer

## North star
The product is **trustworthy fair-value estimates** for CS2 knives — numbers people
will act on with real money. Success is honest, calibrated **sold-price** coverage,
not feature count. Asking prices are a usable proxy; sold prices are the goal.
Every decision should grow or protect that: more clean sold data, better
calibration, more honest UI.

## Scope
- v1 shipped: the 4 locked knife skins, asking-price basis.
- **In scope now: ALL knife types** — asking model in `data/model_all/`, with each
  skin marked "supported" vs "thin" by training-row count. The sold-price model
  (v1.5) stays scoped to the 4 locked skins until enough sold data accrues
  elsewhere; widen it as data allows. See PLAN.md §12.
- Out of scope (do not build): non-knife categories (rifles, gloves, etc.),
  browser extension, arbitrage, alerts.

## Honesty is the product
- An ask-trained model (v1, v1-all) is an **asking-price** estimator — label it so
  in code/docs/UI. Never call it true "fair value" unqualified. Only the
  sold-price model (v1.5) is fair value.
- Serve each model with its OWN reference anchors: ask-based models use ask
  medians, the sold-based model uses sold medians. `predict()` rebuilds USD as
  `exp(quantile) × reference_usd`, so a mismatched anchor silently biases every
  estimate (this caused a 25–90% v1.5 inflation bug). Keep
  `get_active_references()` version-aware.
- Surface sample size and confidence; warn on thin skins; never present a noisy
  estimate as authoritative.

## Modeling
- LightGBM / XGBoost gradient-boosted trees only. No neural networks.
- Time-based train/test split — never a random split for any generalization claim.
- Quantile regression (q10/q50/q90) for ranges. Enforce 0 quantile crossings and
  report [q10,q90] coverage vs nominal — don't claim a band is calibrated if the
  coverage says otherwise.
- **Leakage (critical):** never use `reference.predicted_price` as a feature or
  target (`reference.quantity` is fine as a liquidity hint). Evaluate the
  sold-price model only on rows past its training time-cutoff. Don't let in-sample
  reference prices leak the eval answers.

## Data & API
- Official CSFloat **Market** API only (docs.csfloat.com, `/api/v1/listings`) with
  an API key. Never scrape the DOM. Never touch FloatDB (no API, blocked). Prices
  are in **cents**.
- Respect rate limits: 429 = slow down and back off. The limit is shared
  **per-IP and per-key**, so collect *smarter* (gate to liquid names) rather than
  just more.
- **Data volume is the critical path.** The collector DB is the moat — protect and
  grow it.

## Collector ops
- One canonical `data/collector/observations.db` on the collector host. Never run
  two collectors at once — they diverge.
- A merged PR is **not** deployed until verified on the host (`crontab -l`).
- Scaling past the home-IP rate ceiling: see `docs/vps-migration.md`.

## Working style
- Python only. No AWS, no monorepo, no TypeScript.
- Stop after each phase or major modeling choice and explain it before continuing.
- Keep changes surgical; report outcomes honestly — failing tests, skipped steps,
  and narrow coverage bands included.
