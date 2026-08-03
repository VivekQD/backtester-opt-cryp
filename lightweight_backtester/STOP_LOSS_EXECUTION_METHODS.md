# Stop-Loss Execution Methods

## Scope

This note documents the three stop-loss fill models used by the lightweight
Deribit options backtester. The directly comparable example is the BTC ATM
short straddle entered at 7 DTE and held to Friday expiry, with a 50% per-leg
stop, over the in-sample period from 1 January 2021 through 31 December 2024.

The short call and short put are monitored and stopped independently. If a leg
is sold for entry premium `P`, its 50% stop trigger is:

```text
stop trigger price = P × (1 + 0.50) = 1.5P
```

This is a 50% loss relative to that leg's collected premium. It is not a 50%
loss on account equity.

## Model 1 — Theoretical stop-price fill

The option's five-minute mark triggers the stop when it reaches or exceeds
`1.5P`. The closing fill is then forced to exactly `1.5P` on that same
observation:

```text
trigger: mark >= 1.5P
fill:    1.5P
```

There is no bid/ask spread, gap between observations, queueing, liquidity
constraint, or delay. Deribit fees are still deducted.

This is an optimistic analytical bound. If the observed mark jumps directly
from `1.4P` to `2.0P`, the model still buys the option back at `1.5P`, even
though that price may no longer be executable.

### Theoretical price plus 5% adverse slippage

A separate sensitivity test applies 5% adverse option-price slippage to the
theoretical threshold:

```text
fill = 1.5P × 1.05 = 1.575P
```

This remains theoretical because the price is derived from the threshold, not
from an observed quote or trade.

## Model 2 — Next-candle executable quote

The five-minute mark triggers the stop. The leg is then bought back at the ask
from the following common five-minute observation, with 1 bp adverse slippage:

```text
trigger: mark at time t >= 1.5P
fill:    ask at the next 5-minute observation × 1.0001
```

This model permits the realized loss to exceed 50%. It captures the observed
spread and the price change between the trigger and execution observations,
but it does not model order-book depth or queue position.

This is the default conservative stop model used in the original 1,848-run
parameter grid.

## Model 3 — Nearest subsequent trade

Entries remain quote-based. Only stop-loss exits use the historical trade
tape.

After the stop trigger, the model searches chronologically for the nearest
actual non-block trade in the same option contract, starting from the next
execution snapshot and continuing up to expiry. The exit price is:

```text
fill = next observed trade price × 1.0001
```

The FillEvent is timestamped at the actual later trade time. The position
therefore remains open, marked to market, and subject to margin requirements
during the delay. If no later trade exists before expiry, the implementation
uses the next-candle quote fallback and records the tape miss.

This is an observable trade-price proxy, but it is not the exact counterfactual
fill our order would have received. The historical print belonged to another
market participant, and its displayed amount was not necessarily available to
our order.

More importantly, this is not an immediate stop. Waiting for the next print can
take hours or days, allowing the option price to improve or deteriorate before
the recorded exit.

## In-sample comparison: 7-DTE ATM straddle, 50% stop

| Stop fill model | Trades | BTC return | Daily Sharpe | Max drawdown | Win rate | Profit factor |
|---|---:|---:|---:|---:|---:|---:|
| Exact theoretical threshold | 199 | +135.44% | 0.89 | -17.29% | 77.89% | 2.24 |
| Theoretical threshold + 5% slippage | 199 | +83.45% | 0.58 | -21.28% | 77.39% | 1.66 |
| Next-candle ask + 1 bp | 199 | -4.37% | 0.13 | -44.89% | 68.34% | 0.98 |
| Nearest subsequent trade + 1 bp | 199 | +71.58% | 0.49 | -22.49% | 68.34% | 1.50 |

Fees are included in every row. All returns are based on an initial 1 BTC
account and fixed one-contract-per-leg sizing.

## Trade-tape delay diagnostics

For the nearest-subsequent-trade model:

- 224 stopped-leg tape requests were made.
- 218 found a later trade before expiry.
- 6 had no later trade and used the quote fallback.
- Median delay was 32 minutes.
- 90th-percentile delay was approximately 505 minutes (8.4 hours).
- 75 exits waited at least one hour.
- 7 exits waited at least one day.
- Maximum delay was approximately 8,520 minutes (5.9 days).

These delays explain why the trade-tape result can be materially better than
the next-candle result. It sometimes exits after the stopped option has already
mean-reverted. That improvement should not be interpreted as free execution
alpha.

## Interpretation

The theoretical result is an optimistic bound. The next-candle result is the
most conservative consistently executable approximation available from the
five-minute quote archive. The nearest-trade result is useful for measuring
tape liquidity and stale-exit behavior, but its long delays mean it no longer
represents a strict immediate stop-loss.

For strategy selection, report all three. Use next-candle fills as the primary
backtest, theoretical fills as the optimistic bound, and nearest-trade fills as
a liquidity sensitivity study.

## Result locations

- Next-candle grid: `results/structure_grid/btc/is/`
- Exact theoretical comparison: `results/stop_fill_comparison_margin_enforced_expiry_is/`
- Theoretical plus 5% sensitivity: `results/structure_grid_theoretical_5pct_smoke/`
- Nearest-trade stop results: `results/selected_stop_tape_next_trade/btc/is/`
- Per-exit tape prices and delays: each nearest-trade iteration's
  `execution_ledger.parquet`
