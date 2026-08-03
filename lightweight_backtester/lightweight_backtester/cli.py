from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import replace

from .accounting import AccountingConfig, AccountingModel
from .config import ResearchConfig
from .data import MarketDataStore
from .engine import BacktestEngine
from .execution import ExecutionModel
from .strategies import OptionStructureStrategy
from .trade_tape import TradeTapeStore


def _write_rows(path: str, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_config(cfg: ResearchConfig, output: str):
    store = MarketDataStore(cfg.data_root, cfg.cache_root)
    accounting = AccountingModel(AccountingConfig(
        initial_cash_native=cfg.initial_cash_native,
        trade_fee_rate=cfg.trade_fee_rate,
        settlement_fee_rate=cfg.settlement_fee_rate,
        fee_cap_rate=cfg.fee_cap_rate,
        enforce_margin=cfg.enforce_margin,
        combo_fee_discount=cfg.combo_fee_discount,
    ))
    tape = (TradeTapeStore(cfg.trade_data_root, cfg.asset, cfg.slippage_bps,
                           fill_method=cfg.extra.get("tape_method", "first_trade"))
            if cfg.execution_source in ("trades", "stop_trades") and cfg.trade_data_root else None)
    engine = BacktestEngine(store, ExecutionModel(cfg.slippage_bps), accounting, tape,
                            trade_tape_stops_only=cfg.execution_source == "stop_trades")
    result = engine.run(OptionStructureStrategy(cfg), cfg)
    os.makedirs(output, exist_ok=True)
    _write_rows(os.path.join(output, "leg_ledger.csv"), result.leg_ledger)
    _write_rows(os.path.join(output, "trade_ledger.csv"), result.trade_ledger)
    _write_rows(os.path.join(output, "equity.csv"), [x.__dict__ for x in result.equity])
    _write_rows(os.path.join(output, "execution_ledger.csv"), result.execution_ledger)
    with open(os.path.join(output, "metrics.json"), "w") as f:
        json.dump(result.metrics, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(prog="lwbt")
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache", help="Build contract catalog and decision snapshots")
    cache.add_argument("--config", required=True)
    run = sub.add_parser("run", help="Run one backtest")
    run.add_argument("--config", required=True)
    run.add_argument("--output", default="results/latest")
    sweep = sub.add_parser("sweep-stops", help="Run no-stop, 30%, and 50% variants")
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--output", default="results/stop_sweep")
    args = parser.parse_args()
    cfg = ResearchConfig.from_toml(args.config)
    if args.command == "cache":
        store = MarketDataStore(cfg.data_root, cfg.cache_root)
        print(store.build_catalog())
        print(store.build_snapshot_cache(cfg))
        return
    if args.command == "run":
        result = run_config(cfg, args.output)
        print(json.dumps(result.metrics, indent=2))
        return
    summaries = []
    for stop in (None, 0.30, 0.50):
        variant = replace(cfg, stop_loss_pct=stop)
        label = "none" if stop is None else str(int(stop * 100))
        result = run_config(variant, os.path.join(args.output, f"stop_{label}"))
        summaries.append({"stop_loss_pct": stop, **result.metrics})
    os.makedirs(args.output, exist_ok=True)
    _write_rows(os.path.join(args.output, "summary.csv"), summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
