"""Run all plan example strategies on the local USDC dataset and tabulate results."""
from btcbt import DataConfig, Dataset, DefaultFill, Engine, Margin, VenueConfig

from strategies.short_put import ShortPut
from strategies.short_straddle import ShortStraddle
from strategies.iron_condor import IronCondor
from strategies.delta_hedge import DeltaHedge

PATH = "btc_usdc_last7d.parquet"
CASH = 100_000.0
PARAMS = {"dte_min": 3, "dte_max": 10, "short_delta": 0.30, "long_delta": 0.15}


def run(ds, strat_cls, params):
    venue = VenueConfig(margin=Margin.LINEAR)
    eng = Engine(ds, strat_cls(params), DefaultFill(), venue, cash=CASH)
    return eng.run()


def row(name, res):
    s = res.stats
    return (name, s.overview.trades, s.overview.total_pnl, s.overview.max_dd,
            s.overview.win_rate, s.ratios.sharpe, s.ratios.sortino,
            s.ratios.profit_factor, s.costs.total_fees)


def main():
    ds = Dataset.load(DataConfig(PATH))                 # options only
    ds_perp = Dataset.load(DataConfig(PATH, hedge="perp"))  # + synthetic perp hedge

    rows = [
        row("short_put", run(ds, ShortPut, PARAMS)),
        row("short_straddle", run(ds, ShortStraddle, PARAMS)),
        row("iron_condor", run(ds, IronCondor, PARAMS)),
        row("delta_hedge (no underlying)", run(ds, DeltaHedge, PARAMS)),
        row("delta_hedge (perp hedge)", run(ds_perp, DeltaHedge, PARAMS)),
    ]

    h = ("strategy", "trades", "total_pnl", "max_dd", "win%", "sharpe", "sortino", "pf", "fees")
    print(f"{h[0]:<28}{h[1]:>7}{h[2]:>13}{h[3]:>9}{h[4]:>8}{h[5]:>8}{h[6]:>8}{h[7]:>7}{h[8]:>10}")
    print("-" * 98)
    for r in rows:
        print(f"{r[0]:<28}{r[1]:>7}{r[2]:>13.2f}{r[3]:>8.2%}{r[4]:>8.0%}"
              f"{r[5]:>8.2f}{r[6]:>8.2f}{r[7]:>7.2f}{r[8]:>10.5f}")


if __name__ == "__main__":
    main()
