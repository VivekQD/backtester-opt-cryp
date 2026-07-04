# BTC Options Backtester (`btcbt`)

A plug-and-play backtester for **Deribit BTC options**, with spot/perp hedging, a vectorized option-chain engine, a declarative no-code strategy spec, and a local web builder UI.

- **Accuracy first** — European cash settlement to the 30-min TWAP index, inverse vs linear payoff/PnL, Deribit-style option fee math, strict no-lookahead (orders fill on the next bar).
- **Plug-and-play** — a strategy is one object, or a declarative `StrategySpec` built in the UI.
- **Low dependency** — core is `numpy` + `pyarrow`. No backtesting framework, no TA library.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

## Install

```bash
pip install numpy pyarrow            # core
pip install fastapi uvicorn          # web app (optional)
```

## Web app (visual strategy builder)

Build a strategy in the browser — session, indicators, AND/OR conditions, legs, risk — and run it against your data:

```bash
pip install fastapi uvicorn          # already present here
BTCBT_DATA=/path/to/btc_usdc_last7d.parquet python -m webapp.server
# open http://127.0.0.1:8000
```

The page sends a `StrategySpec` to `POST /backtest`; the backend compiles and runs it and returns the equity curve + grouped stats, rendered as metric cards and a chart.

## CLI

```bash
python -m btcbt.run   --strategy strategies.short_put:ShortPut           --config configs/base.toml
python -m btcbt.sweep --strategy strategies.short_straddle:ShortStraddle --grid configs/straddle_grid.toml --workers 8
```

Point `[data].path` in `configs/base.toml` at your local Parquet.

## Define a strategy (three ways)

**1. Tune params** — edit `[strategy]` in the config, no code.

**2. Write a `Strategy` class** — drop a file in `strategies/`:

```python
from btcbt import Strategy, Side, market

class MyVol(Strategy):
    def on_snapshot(self, snap, ctx):
        if not ctx.portfolio().is_flat():
            return
        if ctx.g("atm_iv") < self.params.get("min_iv", 0.5):     # any global chain variable
            return
        sym = (snap.select().options().puts()
               .dte(3, 10).tradable().nearest("delta", -0.25).first())
        if sym:
            ctx.sell(sym, 1)
```

**3. Declarative `StrategySpec`** — conditions + legs as data, compiled to a strategy (this is what the web UI builds).

## Data

5-minute Deribit option bars in Parquet. Columns: `date, instrument_name, option_type, strike, index_price, underlying_price, mark_price, bid/ask_price, mark/bid/ask_iv, open_interest, volume, delta, gamma, theta, vega, currency`. Greeks/IV are used as-is (no pricing model). `currency` selects inverse (`BTC`) vs linear (`BTC_USDC`).

## Fees

Option trading fees follow Deribit's base formulas:

- inverse BTC/ETH options: `min(0.0003, 0.125 * option_price) * contracts`
- USDC linear options: `min(0.0003 * index_price, 0.125 * option_price) * contracts * contract_size`
- delivery fee at expiry: `0.00015` of the same underlying/index basis, capped at `12.5%` of option value
- non-Friday daily option expiries are treated as Deribit daily options with zero delivery fee

## Layout

```
btcbt/        engine core: data, snapshot/selector, venue, execution, portfolio,
              engine, metrics, runner, globals, indicators, conditions, spec, report
strategies/   example strategies (short put/straddle, iron condor, delta hedge)
configs/      run + sweep configs (TOML)
webapp/       FastAPI backend + builder UI
tests/        venue, selector, no-lookahead, spec
docs/         ARCHITECTURE.md
```

## Test

```bash
python -m pytest tests/ -q
```
