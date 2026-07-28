"""Run the 1,848 short-volatility structure/holding-period iterations.

Outputs are checkpointed per iteration so the sweep can be resumed safely.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lightweight_backtester.accounting import AccountingConfig, AccountingModel
from lightweight_backtester.config import ResearchConfig, to_ns
from lightweight_backtester.data import MarketDataStore
from lightweight_backtester.engine import BacktestEngine
from lightweight_backtester.execution import ExecutionModel
from lightweight_backtester.strategies import OptionStructureStrategy
from lightweight_backtester.trade_tape import TradeTapeStore


STOPS = (0.50, 0.75, 1.00, 1.50, 2.00, None)
BLOCKS = (
    ("Short Straddle", "Naked", "ATM", "", "short_straddle", "atm", .50, .10),
    ("Short Strangle", "Naked", "2% OTM", "", "short_strangle", "moneyness", .30, .10),
    ("Short Strangle", "Naked", "40D", "", "short_strangle", "delta", .40, .10),
    ("Short Strangle", "Naked", "30D", "", "short_strangle", "delta", .30, .10),
    ("Iron Fly", "Defined-risk", "ATM", "Protective wing", "iron_butterfly", "atm", .50, .10),
    ("Iron Condor", "Defined-risk", "2% OTM", "10D wings", "iron_condor", "moneyness", .30, .10),
    ("Iron Condor", "Defined-risk", "2% OTM", "20D wings", "iron_condor", "moneyness", .30, .20),
    ("Iron Condor", "Defined-risk", "40D", "10D wings", "iron_condor", "delta", .40, .10),
    ("Iron Condor", "Defined-risk", "40D", "20D wings", "iron_condor", "delta", .40, .20),
    ("Iron Condor", "Defined-risk", "30D", "10D wings", "iron_condor", "delta", .30, .10),
    ("Iron Condor", "Defined-risk", "30D", "20D wings", "iron_condor", "delta", .30, .20),
)


def iterations():
    iteration_id = 0
    for structure, risk, body, wing, engine_structure, body_mode, short_delta, wing_delta in BLOCKS:
        for entry_dte in range(7, 0, -1):
            for exit_dte in range(entry_dte - 1, -1, -1):
                for stop in STOPS:
                    iteration_id += 1
                    yield {
                        "iteration_id": iteration_id, "structure": structure,
                        "risk_type": risk, "body": body, "wing": wing,
                        "entry_dte": entry_dte, "exit_dte": exit_dte,
                        "hold_days": entry_dte - exit_dte,
                        "stop_loss": "No Stop" if stop is None else f"{int(stop * 100)}%",
                        "stop_loss_pct": stop, "engine_structure": engine_structure,
                        "body_mode": body_mode, "short_delta": short_delta,
                        "wing_delta": wing_delta,
                    }
    assert iteration_id == 1848


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_parquet(path: Path, rows: list[dict]):
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def run_one(base: ResearchConfig, spec: dict, store: MarketDataStore):
    entry = float(spec["entry_dte"])
    cfg = replace(base, entry_frequency="daily", dte_min=entry, dte_max=entry,
                  expiry_weekday=4,
                  structure=spec["engine_structure"], body_mode=spec["body_mode"],
                  short_delta=spec["short_delta"], wing_delta=spec["wing_delta"],
                  exit_dte=float(spec["exit_dte"]), stop_loss_pct=spec["stop_loss_pct"])
    accounting = AccountingModel(AccountingConfig(
        initial_cash_native=cfg.initial_cash_native,
        trade_fee_rate=cfg.trade_fee_rate, settlement_fee_rate=cfg.settlement_fee_rate,
        fee_cap_rate=cfg.fee_cap_rate, enforce_margin=cfg.enforce_margin,
        combo_fee_discount=cfg.combo_fee_discount))
    tape = (TradeTapeStore(cfg.trade_data_root, cfg.asset, cfg.slippage_bps,
                           fill_method=cfg.extra.get("tape_method", "first_trade"))
            if cfg.execution_source in ("trades", "stop_trades") and cfg.trade_data_root else None)
    engine = BacktestEngine(store, ExecutionModel(cfg.slippage_bps), accounting, tape,
                            trade_tape_stops_only=cfg.execution_source == "stop_trades")
    return cfg, engine.run(OptionStructureStrategy(cfg), cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/btc_weekly_straddle_is.toml")
    p.add_argument("--sample", choices=("is", "oos"), default="is")
    p.add_argument("--output", default="results/structure_grid")
    p.add_argument("--first-id", type=int, default=1)
    p.add_argument("--last-id", type=int, default=1848)
    p.add_argument("--force", action="store_true")
    p.add_argument("--stop-fill-mode", choices=("next_bar_touch", "theoretical", "theoretical_slippage"))
    p.add_argument("--theoretical-stop-slippage-pct", type=float, default=0.0)
    args = p.parse_args()

    base = ResearchConfig.from_toml(args.config)
    if args.stop_fill_mode:
        base = replace(base, stop_fill_mode=args.stop_fill_mode,
                       theoretical_stop_slippage_pct=args.theoretical_stop_slippage_pct)
    if args.sample == "is":
        base = replace(base, start_ns=to_ns("2021-01-01T00:00:00+00:00"),
                       end_ns=to_ns("2024-12-31T23:59:59+00:00"))
    else:
        base = replace(base, start_ns=to_ns("2025-01-01T00:00:00+00:00"),
                       end_ns=to_ns("2026-06-30T23:59:59+00:00"))

    root = Path(args.output) / base.asset.lower() / args.sample
    iteration_root = root / "iterations"
    iteration_root.mkdir(parents=True, exist_ok=True)
    specs = list(iterations())
    write_csv(root / "iteration_manifest.csv", specs)
    methodology = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sample": args.sample, "asset": base.asset,
        "sample_start_ns": base.start_ns, "sample_end_ns": base.end_ns,
        "entry_decision": "08:00 UTC at exact entry DTE; executable fill on next common 5-minute quote",
        "scheduled_exit": "08:00 UTC at exit DTE; executable fill on next common 5-minute quote",
        "naked_stop": "per short leg; mark-price trigger, next common 5-minute ask plus slippage",
        "defined_risk_stop": "whole-structure net mark-to-close reaches entry credit * (1 + stop); all legs close at next common 5-minute executable touch plus slippage",
        "iron_fly_wing_assumption": "10 absolute delta (generic Protective wing resolved to existing default)",
        "fees": {"trade_fee_rate": base.trade_fee_rate,
                 "settlement_fee_rate": base.settlement_fee_rate,
                 "fee_cap_rate": base.fee_cap_rate},
        "slippage_bps": base.slippage_bps, "margin_enforced": base.enforce_margin,
        "stop_fill_mode": base.stop_fill_mode,
        "theoretical_stop_slippage_pct": base.theoretical_stop_slippage_pct,
        "initial_cash_native": base.initial_cash_native, "quantity": base.quantity,
    }
    (root / "methodology.json").write_text(json.dumps(methodology, indent=2))

    store = MarketDataStore(base.data_root, base.cache_root)
    store.max_cached_paths = 256
    summaries = []
    for spec in specs:
        iid = spec["iteration_id"]
        if iid < args.first_id or iid > args.last_id:
            continue
        out = iteration_root / f"{iid:04d}"
        metrics_path = out / "metrics.json"
        if metrics_path.exists() and not args.force:
            metrics = json.loads(metrics_path.read_text())
            summaries.append({**spec, **metrics})
            continue
        out.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            cfg, result = run_one(base, spec, store)
            elapsed = time.perf_counter() - started
            metrics = {**result.metrics, "runtime_seconds": elapsed, "status": "complete"}
            metrics_path.write_text(json.dumps(metrics, indent=2))
            (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
            write_parquet(out / "trades.parquet", result.trade_ledger)
            write_parquet(out / "legs.parquet", result.leg_ledger)
            write_parquet(out / "equity.parquet", [dataclasses.asdict(x) for x in result.equity])
        except Exception as exc:
            metrics = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                       "runtime_seconds": time.perf_counter() - started}
            metrics_path.write_text(json.dumps(metrics, indent=2))
        summaries.append({**spec, **metrics})
        write_csv(root / "summary_partial.csv", summaries)
        print(f"[{args.sample}] {iid:04d}/1848 {spec['structure']} {spec['body']} "
              f"{spec['entry_dte']}->{spec['exit_dte']} {spec['stop_loss']} "
              f"{metrics['status']} {metrics['runtime_seconds']:.1f}s", flush=True)
    write_csv(root / "summary.csv", summaries)


if __name__ == "__main__":
    main()
