# Lightweight Backtester

A sparse, event-driven research engine for Laevitas Deribit option archives.
It separates market simulation from accounting and never builds the complete
5-minute option chain in memory.

## Why it is faster

The source archive contains roughly 95,000 BTC and 101,000 ETH contract files.
The engine uses three progressively smaller datasets:

1. `contract_catalog.parquet`: filename and row-group metadata only.
2. A schedule cache: one causal chain snapshot per configured decision time.
3. Full 5-minute paths only for contracts selected by the strategy.

The expensive schedule cache is reusable across strike, structure, stop, fee,
and accounting sweeps. Cache snapshots are strictly as-of: a quote after the
decision timestamp is never used to select a contract.

## Modules

- `data.py`: catalog, snapshot cache, selected-contract path loader.
- `strategies.py`: straddle, strangle, iron butterfly, iron condor, and
  directional call/put selectors. Custom strategies return the same
  `TradeIntent` interface.
- `engine.py`: causal entry, five-minute marks, next-bar stops, and expiry.
- `execution.py`: bid/ask-touch fills plus configurable slippage.
- `accounting.py`: independent FIFO positions, inverse P&L, fees, settlement,
  standard-margin reporting, native equity, and USD equity.
- `metrics.py`: structure/leg ledgers, daily Sharpe, drawdown size/duration,
  fees, margin, and defined-risk maximum-loss diagnostics.

Standard margin is enforced by default: multi-leg entries are checked
atomically, maintenance is recomputed from every five-minute mark, and a breach
liquidates open legs at the executable touch plus slippage and Deribit option
liquidation fees. Rejections and liquidations are included in the metrics.

## Setup

```bash
cd /srv/shared/options_projects/backtester/lightweight_backtester
UV_CACHE_DIR=/tmp/lwbt-uv-cache uv sync --extra dev --extra notebook
```

## Commands

Build or reuse the catalog and decision-snapshot cache:

```bash
uv run python -m lightweight_backtester.cli cache \
  --config configs/btc_weekly_straddle_smoke.toml
```

Run one backtest:

```bash
uv run python -m lightweight_backtester.cli run \
  --config configs/btc_weekly_straddle_smoke.toml \
  --output results/btc_smoke
```

Run the no-stop, 30%, and 50% per-leg stop sweep:

```bash
uv run python -m lightweight_backtester.cli sweep-stops \
  --config configs/btc_weekly_straddle_is.toml \
  --output results/btc_is_stop_sweep
```

Change `structure` to `short_strangle`, `iron_butterfly`, `iron_condor`,
`short_call`, or `short_put`. Set `entry_frequency` to `daily`, `weekly`, or
`monthly`; monthly means the last configured weekday of each month. Daily,
monthly, quarterly, and long-dated options use the same engine by changing the
schedule and DTE range.

## Execution and accounting conventions

- Decision: configured UTC timestamp using the last quote at or before it.
- Entry: next common 5-minute bar; short at bid and long at ask, then slippage.
- Stop: current mark triggers; next common bar executes at ask plus slippage.
- Expiry: inverse payoff using a 30-minute index TWAP ending at 08:00 UTC.
- Trading fee: `min(0.0003, 12.5% × option price)` per contract.
- Friday delivery fee: `min(0.00015, 12.5% × payoff)`; non-Friday daily
  options default to zero delivery fee.
- Primary Sharpe: daily native-equity returns sampled at the last observation
  of each UTC day. USD Sharpe is reported independently.
- Risk: initial margin is checked atomically at entry; maintenance margin and
  intraday drawdown are evaluated at every available five-minute mark.

Outputs include `leg_ledger.csv`, `trade_ledger.csv`, `equity.csv`, and
`metrics.json`. Option-only P&L is kept separate from USD collateral effects.

## Validation

```bash
uv run python -m pytest tests -q
```
