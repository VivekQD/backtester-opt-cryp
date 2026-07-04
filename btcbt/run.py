"""python -m btcbt.run --strategy module:Class --config configs/base.toml"""
from __future__ import annotations

import argparse

from .cli import build_runconfig, load_toml, print_stats, resolve_strategy
from .report import export_trades_csv
from .runner import run_one_with_dataset


def main():
    ap = argparse.ArgumentParser(prog="btcbt.run")
    ap.add_argument("--strategy", required=True, help="module:Class")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = build_runconfig(load_toml(args.config), resolve_strategy(args.strategy))
    res, ds = run_one_with_dataset(cfg)
    print_stats(res)
    trade_path = export_trades_csv(res, ds.metas, args.strategy, venue=cfg.venue)
    print(f"Trades CSV saved to {trade_path}")


if __name__ == "__main__":
    main()
