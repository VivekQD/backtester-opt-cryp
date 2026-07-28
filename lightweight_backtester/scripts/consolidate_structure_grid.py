"""Consolidate completed structure-grid shards and validate iteration coverage."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from run_structure_grid import iterations, write_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", choices=("is", "oos"), default="is")
    p.add_argument("--asset", default="btc")
    p.add_argument("--output", default="results/structure_grid")
    p.add_argument("shards", nargs="+")
    args = p.parse_args()
    root = Path(args.output) / args.asset / args.sample
    target = root / "iterations"
    target.mkdir(parents=True, exist_ok=True)
    for shard in args.shards:
        source_root = Path(shard) / args.asset / args.sample
        for source in (source_root / "iterations").glob("[0-9][0-9][0-9][0-9]"):
            destination = target / source.name
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        for name in ("methodology.json", "iteration_manifest.csv"):
            if (source_root / name).exists():
                shutil.copy2(source_root / name, root / name)

    specs = {x["iteration_id"]: x for x in iterations()}
    rows, errors = [], []
    for iid, spec in specs.items():
        path = target / f"{iid:04d}" / "metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text())
        row = {**spec, **metrics}
        rows.append(row)
        if metrics.get("status") != "complete":
            errors.append(row)
    write_csv(root / "summary.csv", rows)
    audit = {"expected_iterations": 1848, "completed_files": len(rows),
             "successful": sum(x.get("status") == "complete" for x in rows),
             "errors": len(errors),
             "missing_iteration_ids": sorted(set(specs) - {x["iteration_id"] for x in rows})}
    (root / "coverage_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
