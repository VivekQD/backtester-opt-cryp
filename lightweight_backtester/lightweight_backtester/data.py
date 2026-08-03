from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import ResearchConfig
from .types import Contract, OptionType

NS = 1_000_000_000
DAY_NS = 86_400 * NS
NAME_RE = re.compile(r"^(?P<asset>[A-Z0-9_]+)-(?P<expiry>\d{1,2}[A-Z]{3}\d{2})-(?P<strike>[0-9.]+)-(?P<type>[CP])$")
SNAPSHOT_COLUMNS = ["date", "instrument_name", "strike", "index_price", "underlying_price",
                    "mark_price", "bid_price", "ask_price", "mark_iv", "open_interest",
                    "volume", "delta", "gamma", "theta", "vega"]


def parse_contract(path: str) -> Contract | None:
    m = NAME_RE.match(Path(path).stem)
    if not m:
        return None
    expiry = dt.datetime.strptime(m["expiry"], "%d%b%y").replace(hour=8, tzinfo=dt.timezone.utc)
    return Contract(Path(path).stem, m["asset"], int(expiry.timestamp() * NS),
                    float(m["strike"]), OptionType(m["type"]), str(path))


class MarketDataStore:
    """Catalog + schedule snapshots + selected-contract paths.

    The complete 5-minute chain is never assembled. Contract files are scanned
    once when a schedule cache is built, and only selected paths are read during
    simulation.
    """

    def __init__(self, source_root: str, cache_root: str):
        self.source_root = Path(source_root)
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.cache_root / "contract_catalog.parquet"
        self._paths: OrderedDict[tuple, dict[str, np.ndarray]] = OrderedDict()
        self.max_cached_paths = 32

    def build_catalog(self, force: bool = False) -> str:
        if self.catalog_path.exists() and not force:
            return str(self.catalog_path)
        rows = []
        for path in self.source_root.glob("expiry_month=*/*.parquet"):
            contract = parse_contract(str(path))
            if contract is None:
                continue
            try:
                pf = pq.ParquetFile(path)
                date_idx = pf.schema_arrow.get_field_index("date")
                stats = pf.metadata.row_group(0).column(date_idx).statistics
                first = int(np.datetime64(stats.min.replace(tzinfo=None), "ns").astype(np.int64))
                last = int(np.datetime64(stats.max.replace(tzinfo=None), "ns").astype(np.int64))
                count = pf.metadata.num_rows
            except Exception:
                first = last = count = 0
            rows.append({"instrument": contract.instrument, "asset": contract.asset,
                         "expiry_ns": contract.expiry_ns, "strike": contract.strike,
                         "option_type": contract.option_type.value, "path": str(path),
                         "first_quote_ns": first, "last_quote_ns": last, "rows": count,
                         "mtime_ns": path.stat().st_mtime_ns})
        if not rows:
            raise FileNotFoundError(f"No option parquet files below {self.source_root}")
        pq.write_table(pa.Table.from_pylist(rows), self.catalog_path, compression="zstd")
        return str(self.catalog_path)

    def catalog(self) -> list[dict]:
        self.build_catalog()
        return pq.read_table(self.catalog_path).to_pylist()

    @staticmethod
    def schedule(cfg: ResearchConfig) -> np.ndarray:
        start = dt.datetime.fromtimestamp(cfg.start_ns / NS, dt.timezone.utc)
        end = dt.datetime.fromtimestamp(cfg.end_ns / NS, dt.timezone.utc)
        out = []
        frequency = cfg.entry_frequency.lower()
        if frequency == "monthly":
            month = start.replace(day=1, hour=cfg.entry_hour_utc,
                                  minute=cfg.entry_minute_utc, second=0, microsecond=0)
            while month <= end:
                next_month = ((month.replace(year=month.year + 1, month=1)) if month.month == 12
                              else month.replace(month=month.month + 1))
                cursor = next_month - dt.timedelta(days=1)
                cursor -= dt.timedelta(days=(cursor.weekday() - cfg.entry_weekday) % 7)
                if start <= cursor <= end:
                    out.append(int(cursor.timestamp() * NS))
                month = next_month
        else:
            cursor = start.replace(hour=cfg.entry_hour_utc, minute=cfg.entry_minute_utc,
                                   second=0, microsecond=0)
            if frequency == "weekly":
                cursor += dt.timedelta(days=(cfg.entry_weekday - cursor.weekday()) % 7)
                step = dt.timedelta(days=7)
            elif frequency == "daily":
                step = dt.timedelta(days=1)
            else:
                raise ValueError("entry_frequency must be daily, weekly, or monthly")
            if cursor < start:
                cursor += step
            while cursor <= end:
                out.append(int(cursor.timestamp() * NS))
                cursor += step
        return np.asarray(out, dtype=np.int64)

    def _snapshot_key(self, cfg: ResearchConfig) -> str:
        payload = [str(self.source_root), cfg.asset, cfg.start_ns, cfg.end_ns,
                   cfg.entry_frequency, cfg.entry_weekday, cfg.entry_hour_utc, cfg.entry_minute_utc,
                   cfg.dte_min, cfg.dte_max, cfg.snapshot_tolerance_minutes]
        return hashlib.sha1(json.dumps(payload).encode()).hexdigest()[:16]

    def build_snapshot_cache(self, cfg: ResearchConfig, force: bool = False) -> str:
        self.build_catalog()
        target_path = self.cache_root / f"snapshots_{self._snapshot_key(cfg)}.parquet"
        if target_path.exists() and not force:
            return str(target_path)
        targets = self.schedule(cfg)
        rows = []
        lo_exp = cfg.start_ns + int(cfg.dte_min * DAY_NS)
        hi_exp = cfg.end_ns + int(cfg.dte_max * DAY_NS)
        candidates = [r for r in self.catalog() if r["asset"] == cfg.asset
                      and lo_exp <= r["expiry_ns"] <= hi_exp]
        tolerance_ns = cfg.snapshot_tolerance_minutes * 60 * NS
        for meta in candidates:
            eligible = targets[(targets >= meta["first_quote_ns"])
                               & (targets <= meta["last_quote_ns"] + tolerance_ns)]
            dte = (meta["expiry_ns"] - eligible) / DAY_NS
            eligible = eligible[(dte >= cfg.dte_min) & (dte <= cfg.dte_max)]
            if eligible.size == 0:
                continue
            try:
                table = pq.read_table(meta["path"], columns=SNAPSHOT_COLUMNS)
            except Exception:
                continue
            available = set(table.column_names)
            dates = table["date"].to_numpy().astype("datetime64[ns]").astype(np.int64)
            order = np.argsort(dates)
            dates = dates[order]
            for target in eligible:
                idx = int(np.searchsorted(dates, target, side="right") - 1)
                if idx < 0 or target - dates[idx] > tolerance_ns:
                    continue
                raw_idx = int(order[idx])
                row = {name: table[name][raw_idx].as_py() if name in available else None
                       for name in SNAPSHOT_COLUMNS if name != "date"}
                row.update({"target_ns": int(target), "quote_ns": int(dates[idx]),
                            "expiry_ns": int(meta["expiry_ns"]), "path": meta["path"],
                            "asset": cfg.asset, "option_type": meta["option_type"]})
                rows.append(row)
        if not rows:
            raise ValueError("Snapshot cache is empty; check dates, DTE, and source coverage")
        pq.write_table(pa.Table.from_pylist(rows), target_path, compression="zstd")
        return str(target_path)

    def snapshots(self, cfg: ResearchConfig) -> dict[int, list[dict]]:
        path = self.build_snapshot_cache(cfg)
        rows = pq.read_table(path).to_pylist()
        out: dict[int, list[dict]] = {}
        for row in rows:
            out.setdefault(int(row["target_ns"]), []).append(row)
        return out

    def load_contract_path(self, contract: Contract, start_ns: int, end_ns: int) -> dict[str, np.ndarray]:
        key = (contract.path, start_ns, end_ns)
        if key in self._paths:
            self._paths.move_to_end(key)
            return self._paths[key]
        table = pq.read_table(contract.path, columns=SNAPSHOT_COLUMNS)
        date = table["date"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        mask = (date >= start_ns) & (date <= end_ns)
        order = np.argsort(date[mask])
        out = {"date": date[mask][order]}
        for name in SNAPSHOT_COLUMNS[1:]:
            if name in table.column_names:
                values = table[name].to_numpy(zero_copy_only=False)[mask][order]
                out[name] = values
        self._paths[key] = out
        self._paths.move_to_end(key)
        while len(self._paths) > self.max_cached_paths:
            self._paths.popitem(last=False)
        return out
