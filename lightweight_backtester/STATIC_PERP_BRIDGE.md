# Static Perpetual Bridge for Illiquid Option Stops

## Objective

When a 50% per-leg stop triggers on the weekly BTC short straddle, do not
immediately cross a temporarily wide option ask. Open a static BTCUSDT
perpetual hedge for the stopped option's delta, wait for the Deribit option
spread to normalize, then close the option and the hedge.

## Data

- Options: Laevitas Deribit five-minute mark, delta, best bid and best ask.
- Perpetual proxy: Hillroute `propdesk.venue.ohlcv_binanceperp_5`, symbol
  `BTCUSDT`.
- Cached perpetual coverage: 420,768 five-minute candles from 2021-01-01
  through 2024-12-31.
- Database credentials are prompted interactively and are not stored in the
  repository or cache metadata.

The perpetual table contains OHLCV, not historical bid/ask, funding, or order
book depth. Consequently, this remains a candle-fill study.

## Rules

1. Sell the ATM call and put at 7 DTE, one BTC-equivalent each.
2. Trigger each short leg independently when its mark reaches 150% of its entry
   premium.
3. At the next Binance five-minute candle, open a static hedge equal to the
   negative of that short leg's position delta.
4. Do not rebalance the hedge.
5. Beginning after the trigger, calculate the option relative spread:

   `relative_spread = (ask - bid) / ((ask + bid) / 2)`

6. Buy back the option when relative spread is at or below the configured
   threshold.
7. If it has not normalized by the configured maximum wait, buy the option at
   the then-current ask.
8. Close the static perpetual hedge when the option is closed.
9. If the option survives to expiry, settle it and close the hedge.

## Execution assumptions

- Option entry: next five-minute bid plus 1 bp adverse slippage.
- Option exit: observed ask plus 1 bp adverse slippage.
- Perp entry/exit: next BTCUSDT candle open plus 5 bps adverse slippage.
- Perp fee: 5 bps per side.
- Perp P&L: linear USDT P&L converted to BTC at exit.
- Funding: not available and not modeled.
- Perp margin/liquidation: not modeled.
- Option margin: disabled for this isolated trade-level feasibility test.

## Grid

- Relative spread: 5%, 10%, 15%, 20%, 30%.
- Maximum wait: 15, 30, 60, 240, 1,440 minutes.
- Sample: 199 weekly expiries, 2021–2024.

## Leading cells

| Spread limit | Maximum wait | P&L | Weekly Sharpe | Max DD | Win rate | Mean hedge wait |
|---:|---:|---:|---:|---:|---:|---:|
| 5% | 240 min | +0.0917 BTC | 0.12 | -29.5% | 65.3% | 68.8 min |
| 5% | 30 min | +0.0625 BTC | 0.07 | -30.0% | 67.8% | 16.6 min |
| 15% | 240 min | +0.0115 BTC | 0.02 | -30.3% | 65.3% | 16.5 min |

For the leading 5%/240-minute cell:

- option P&L after option fees: +0.6466 BTC;
- static perp P&L after modeled perp fees/slippage: -0.5549 BTC;
- total: +0.0917 BTC;
- 221 option legs triggered a hedge;
- 184 exited after spread normalization;
- 33 exited at the maximum wait;
- the remainder reached expiry.

## Interpretation

The bridge improves on the original immediate next-candle 50%-stop result of
approximately -0.0437 BTC, but the best cell is not yet compelling: its Sharpe
is low and drawdown remains near 30%. The static hedge itself is costly because
it locks in negative-gamma directional adjustments and incurs two perp trades.

The 5%/240-minute result was selected from a 25-cell in-sample grid. It must be
frozen and tested out of sample before being treated as evidence of an edge.
Actual funding and executable perp bid/ask can only reduce or alter the reported
result, so those fields are required for the next validation stage.

## Files

- `scripts/cache_binance_perp.py`: credential-safe database cache builder.
- `scripts/static_perp_bridge_grid.py`: bridge simulation and grid runner.
- `.lwbt_cache/perp/BTCUSDT_5m_2021_2024.parquet`: local OHLCV cache.
- `results/static_perp_bridge_grid/summary.csv`: grid summary.
- `results/static_perp_bridge_grid/trade_results.parquet`: trade-level results.
- `results/static_perp_bridge_grid/methodology.json`: machine-readable assumptions.
