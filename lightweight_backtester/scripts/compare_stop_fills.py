"""Compare exact theoretical stops with next-5-minute executable fills."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from dataclasses import replace

from lightweight_backtester.cli import run_config
from lightweight_backtester.config import ResearchConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/btc_weekly_straddle_is.toml")
    parser.add_argument("--output", default="results/stop_fill_comparison")
    args = parser.parse_args()
    base = ResearchConfig.from_toml(args.config)
    rows = []
    variants = [("hold_to_expiry", None, "next_bar_touch")]
    variants += [(mode, stop, mode) for stop in (.30, .50)
                 for mode in ("theoretical", "next_bar_touch")]
    for label, stop, mode in variants:
        cfg = replace(base, stop_loss_pct=stop, stop_fill_mode=mode)
        name = "no_stop" if stop is None else f"stop_{int(stop*100)}_{mode}"
        result = run_config(cfg, os.path.join(args.output, name))
        stopped = sum(x["exit_reason"].startswith("stop_loss") for x in result.leg_ledger)
        rows.append({"fill_model": label, "stop_loss_pct": stop, "stopped_legs": stopped,
                     **result.metrics})
        del result
        gc.collect()
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
