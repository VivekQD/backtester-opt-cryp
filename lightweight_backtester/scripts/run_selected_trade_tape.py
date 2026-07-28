"""Re-run selected IS configurations using actual Deribit trade prints."""
from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import replace
from pathlib import Path

from lightweight_backtester.config import ResearchConfig, to_ns
from lightweight_backtester.data import MarketDataStore
from run_structure_grid import iterations, run_one, write_csv, write_parquet


DEFAULT_IDS = (37, 150, 162, 486, 654, 882, 1217)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/btc_weekly_straddle_is.toml")
    p.add_argument("--output", default="results/selected_trade_tape")
    p.add_argument("--ids", nargs="*", type=int, default=DEFAULT_IDS)
    p.add_argument("--slippage-bps", type=float, default=1.0)
    p.add_argument("--tape-method", choices=("first_trade", "next_trade", "aggressor_vwap"), default="first_trade")
    p.add_argument("--scope", choices=("all", "stops_only"), default="all")
    args = p.parse_args()
    specs = {x["iteration_id"]: x for x in iterations()}
    base = ResearchConfig.from_toml(args.config)
    base = replace(
        base, start_ns=to_ns("2021-01-01T00:00:00+00:00"),
        end_ns=to_ns("2024-12-31T23:59:59+00:00"),
        execution_source="stop_trades" if args.scope == "stops_only" else "trades",
        trade_data_root="/srv/shared/laevitas data/deribit/options_historical_data/trades",
        slippage_bps=args.slippage_bps, stop_fill_mode="next_bar_touch")
    root = Path(args.output) / "btc" / "is"
    root.mkdir(parents=True, exist_ok=True)
    # Passed through ResearchConfig.extra so the generic runner remains usable.
    base.extra["tape_method"] = args.tape_method
    store = MarketDataStore(base.data_root, base.cache_root)
    store.max_cached_paths = 256
    summaries = []
    for iid in args.ids:
        spec = specs[iid]
        cfg, result = run_one(base, spec, store)
        out = root / "iterations" / f"{iid:04d}"
        out.mkdir(parents=True, exist_ok=True)
        metrics = {**result.metrics, "status": "complete"}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
        write_parquet(out / "trades.parquet", result.trade_ledger)
        write_parquet(out / "legs.parquet", result.leg_ledger)
        write_parquet(out / "equity.parquet", [dataclasses.asdict(x) for x in result.equity])
        write_parquet(out / "execution_ledger.parquet", result.execution_ledger)
        summaries.append({**spec, **metrics})
        print(iid, metrics.get("trades", 0), metrics.get("native_return"), flush=True)
    write_csv(root / "summary.csv", summaries)
    (root / "methodology.json").write_text(json.dumps({
        "execution_source": "actual Deribit non-block option trades",
        "scope": args.scope,
        "quote_fallback_on_stop_tape_miss": args.scope == "stops_only",
        "tape_method": args.tape_method,
        "entry": ("quote-based next-bar bid/ask plus slippage" if args.scope == "stops_only"
                  else "first same-instrument non-block print in [decision, decision+5m), adjusted adversely"),
        "stop_exit": ("nearest same-instrument non-block trade after the stop execution snapshot, up to expiry"
                      if args.tape_method == "next_trade" else
                      "first same-instrument non-block print in next five-minute execution window, adjusted adversely"),
        "scheduled_exit": "corresponding adverse-side five-minute trade VWAP",
        "insufficient_volume": ("if no later print exists before expiry, stop exit falls back to next-bar executable quote"
                                if args.scope == "stops_only" else
                                "full trade omitted; never falls back to quote or theoretical fill"),
        "additional_slippage_bps": args.slippage_bps,
        "ids": list(args.ids)}, indent=2))


if __name__ == "__main__":
    main()
