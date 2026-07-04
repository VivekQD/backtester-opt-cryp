"""python -m btcbt.sweep --strategy module:Class --grid configs/grid.toml --workers 8

Grid TOML = a normal config plus a [grid] table whose keys are strategy params
mapped to lists of values (cartesian product).
"""
from __future__ import annotations

import argparse

from .cli import build_runconfig, load_toml, resolve_strategy
from .runner import Study, expand_grid


def main():
    ap = argparse.ArgumentParser(prog="btcbt.sweep")
    ap.add_argument("--strategy", required=True, help="module:Class")
    ap.add_argument("--grid", required=True)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--metric", default="sharpe")
    args = ap.parse_args()

    d = load_toml(args.grid)
    base = build_runconfig(d, resolve_strategy(args.strategy))
    axes = {k: list(v) for k, v in d.get("grid", {}).items()}
    grid = expand_grid(base, axes) if axes else [base]

    study = Study(grid).run(workers=args.workers)
    print(f"{len(grid)} runs · ranked by {args.metric}")
    for params, score in study.leaderboard(args.metric):
        print(f"  {score:+.3f}   {params}")


if __name__ == "__main__":
    main()
