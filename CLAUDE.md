```
- Python only. No TypeScript, no AWS, no monorepo.
- Use the official CSFloat MARKET API (docs.csfloat.com, /api/v1/listings) with an API key.
  Never scrape the DOM. Never touch FloatDB (no API, blocked).
- v1 trains on ASKING prices and must say so honestly in code/docs/UI. Do NOT call it true "fair
  value" unqualified. Sold prices are the v1.5 upgrade via the background collector.
- Model = LightGBM/XGBoost gradient-boosted trees. Do not use neural networks.
- Use a time-based train/test split and quantile regression for fair-value ranges.
- Respect CSFloat rate limits; cache and back off (429 = slow down). Prices are in cents.
- Stay within v1 scope in PLAN.md. Do not build: extension, arbitrage, alerts, other skin types.
- Stop after each phase and explain modeling choices before continuing.
```