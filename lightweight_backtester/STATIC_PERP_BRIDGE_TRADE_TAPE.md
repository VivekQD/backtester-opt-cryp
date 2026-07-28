# Static Perpetual Bridge — Trade-Tape Option Exits

## Purpose

This variant replaces quote-assumed option stop exits with observed Deribit
option trades. It tests whether a perpetual hedge can bridge the interval
between a mark-price stop trigger and an executable option trade.

## Execution sequence

1. Sell the 7-DTE ATM BTC call and put using the existing quote-based entry
   model.
2. Trigger each short leg independently when its mark reaches 150% of its
   entry premium (a 50% loss on that leg).
3. Open a static BTCUSDT perpetual hedge at the next Binance five-minute open.
   The hedge offsets the stopped leg's delta at the trigger and is not
   rebalanced.
4. Observe the Deribit option quotes until either:
   - relative spread falls below the configured limit; or
   - the configured maximum wait expires.
5. From that decision timestamp, find the first subsequent non-block trade in
   the exact option contract, before expiry.
6. Buy back the short option at that observed trade price plus 1 bp adverse
   slippage.
7. Close the perpetual hedge at the next Binance five-minute open after the
   option trade.
8. If no qualifying option trade occurs before expiry, settle the option and
   close the hedge at expiry.

Option entries remain quote-based. Therefore this is specifically a
trade-tape exit study, not a fully tape-executed strategy.

## Data and assumptions

- Sample: 199 weekly BTC expiries, 2021–2024 (in-sample).
- Option exit source: Laevitas Deribit individual trade tape.
- Block trades are excluded.
- Perpetual source: Binance BTCUSDT five-minute OHLCV from Hillroute.
- Perp entry/exit: next five-minute open with 5 bps adverse slippage.
- Perp fee: 5 bps per side.
- Funding, perp order-book spread/depth, and liquidation are not modeled.
- Deribit fees use the lightweight backtester's calibrated option fee model.

## Best grid cell

The highest total P&L was produced by a 5% relative-spread threshold and a
240-minute maximum quote wait:

| Metric | Result |
|---|---:|
| Trades | 199 |
| Total P&L | +0.4520 BTC |
| Ending equity from 1 BTC | 1.4520 BTC |
| Weekly Sharpe | 0.6720 |
| Maximum drawdown | -23.97% |
| Win rate | 70.35% |
| Option P&L component | +0.8071 BTC |
| Perp P&L component | -0.3551 BTC |
| Option fees | 0.1888 BTC |
| Perp fees | 0.1562 BTC |
| Hedged option legs | 221 |
| Tape exits | 218 |
| No-trade-to-expiry cases | 3 |

For this cell, the mean delay from the quote-spread decision to the observed
option trade was 273.1 minutes in the summary aggregation (median trade-level
delay approximately 30.5 minutes). Mean total hedge time was 458.9 minutes.

## Important interpretation

The higher result versus a quote/candle exit is not automatically evidence of
better executable performance. The strategy waits for the next trade after the
spread decision, and that trade arrives several hours later on average. During
that delay the option value can change materially. The resulting fill therefore
mixes execution liquidity with market movement and possible mean reversion.

The trade tape also records completed trades, not the size available to this
strategy. A historical print confirms that someone traded at the price; it
does not prove that the strategy could have filled its required quantity there.
This result should be treated as a liquidity-conditioned scenario and tested
out of sample before selecting the 5%/240-minute parameters.

## Outputs

- `results/static_perp_bridge_trade_tape_grid/summary.csv`
- `results/static_perp_bridge_trade_tape_grid/summary.parquet`
- `results/static_perp_bridge_trade_tape_grid/trade_results.parquet`
- `results/static_perp_bridge_trade_tape_grid/methodology.json`

Run with:

```bash
.venv/bin/python scripts/static_perp_bridge_grid.py \
  --option-execution trade_tape \
  --output results/static_perp_bridge_trade_tape_grid
```
