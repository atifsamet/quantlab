# QuantLab

**Quantitative Crypto Market Intelligence**

QuantLab is a quantitative crypto market intelligence platform built to investigate whether technical, strategy-based, machine-learning, and statistical signals can produce a durable predictive edge across multiple crypto assets and timeframes.

**Research finding: NO ROBUST EDGE DETECTED.** Historical validation did not demonstrate a durable predictive edge. QuantLab does **not** execute trades.

---

## What is QuantLab?

A read-only research and market-analysis application that:

- Fetches **public OKX** candle data
- Computes technical indicators and strategy signals
- Produces analytical **LONG / SHORT / NEUTRAL** predictions with signal strength
- Stores prediction snapshots and evaluates them causally (WIN / LOSS / TIMEOUT)
- Surfaces Phases 1–14 research artifacts (backtests, strategies, Monte Carlo, etc.)

It is **not** a trading bot, exchange, investment service, or automated execution platform.

## Features

- Live closed-candle market analysis (BTC / ETH / SOL / XRP × 15m / 1H / 4H)
- Prediction history + historical evaluation performance
- Strategy research tables and backtest artifact viewers
- Statistical validation views (bootstrap, cost sensitivity, Monte Carlo)
- Research timeline and methodology documentation
- Persistent **RESEARCH MODE / LIVE TRADING DISABLED** status

## Architecture

```
Public OKX market data
        ↓
Python research engine (indicators / strategies / risk / ML / stats)
        ↓
Prediction engine + JSONL history store
        ↓
FastAPI (read-only)
        ↓
Next.js QuantLab dashboard
```

## Research pipeline

| Phase | Focus | Outcome |
|-------|--------|---------|
| 1–6 | Data, indicators, baseline, backtest, risk, history | Foundations |
| 7–9 | Robust validation & multi-asset strategy research | No robust strategy |
| 10 | ML signal filter | No durable ML edge |
| 11 | Statistical validation | No evidence of an edge |
| 12–14 | Dashboard, history evaluation, live intelligence | Product layer |
| 15 | Portfolio polish | Presentation / docs |

## Prediction methodology

1. Prefer the **latest closed** public candle (incomplete bars dropped).
2. Reuse existing indicators + breakout-oriented signal logic.
3. Score indicator agreement into **signal strength 0–100** (analytical — **not** P(win)).
4. Derive ATR-based model levels (entry / SL / TP) when directional.
5. Persist snapshot; evaluate later with future candles only (no look-ahead).

## Historical evaluation

| Outcome | Meaning |
|---------|---------|
| WIN | Take-profit before stop-loss |
| LOSS | Stop-loss first (same-candle → SL first) |
| TIMEOUT | Neither within horizon |
| SKIPPED | NEUTRAL / missing levels |

Horizons: 15m→24, 1H→12, 4H→6 bars.

## Research findings

**NO ROBUST EDGE DETECTED**

Across multi-asset strategy grids, ML filtering, Monte Carlo / random benchmarks, and ~7.9k historical prediction snapshots (Phase 13), results did not support a durable predictive edge. Stronger signal-strength bands did not produce meaningfully reliable historical success.

## Tech stack

| Layer | Stack |
|-------|--------|
| Backend | Python, FastAPI, pandas |
| Frontend | Next.js, TypeScript, Tailwind, Recharts |
| Data | Public OKX REST candles |
| Storage | CSV artifacts + `data/predictions/history.jsonl` |

## Project structure

```
crypto-trading-bot/
├── app/                 # Research engine + FastAPI + prediction
├── web/                 # Next.js QuantLab UI
├── tests/
├── data/
│   ├── historical/      # OHLCV CSVs (local)
│   ├── results/         # Phase research artifacts (local)
│   └── predictions/     # Prediction history JSONL (local)
├── main.py
├── requirements.txt
└── README.md
```

## Running locally

### Backend

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --reload
```

API: http://127.0.0.1:8000/docs

### Frontend

```powershell
cd web
npm install
npm run dev
```

Dashboard: http://localhost:3000

Optional: `web/.env.local` with `NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000`

### Historical prediction backfill (optional)

```powershell
.\.venv\Scripts\python.exe -m app.prediction.backfill
```

### Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cd web
npm run lint
npm run build
```

## Safety

| Control | Status |
|---------|--------|
| `LIVE_TRADING_ENABLED` | `false` (hard-checked) |
| Order / private trading APIs | Not implemented |
| Frontend secrets | None required |
| Market data | Public endpoints only |

## Environment variables

**Backend (`.env`)** — see `.env.example`. Public candles work without API keys. Never commit `.env`.

**Frontend (`web/.env.local`)** — optional `NEXT_PUBLIC_API_BASE` only. Never put exchange secrets in Next.js env.

## Deployment readiness

Suggested production layout (not required for local portfolio use):

- **Frontend:** Next.js (`npm run build` / `npm start`) behind HTTPS
- **Backend:** `uvicorn app.api.main:app --host 0.0.0.0 --port 8000`
- **CORS:** local Next.js origins on ports 3000–3003 (`localhost` / `127.0.0.1`) are allowed in `app/api/main.py`
- **Persistence:** mount/write access for `data/predictions/`
- **No Redis / DB required** for the current design

## Limitations

- Not profitable / not a trading system
- Public data may be delayed or unavailable
- Signal strength ≠ calibrated probability
- Backfill uses sampling strides (not every candle)
- Local JSONL is suitable for research volume, not multi-tenant SaaS

## License / intent

Educational and portfolio research project. Outputs are experimental analytical results, not financial advice.
