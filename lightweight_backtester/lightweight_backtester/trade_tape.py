from __future__ import annotations

import datetime as dt
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from .types import Side


@dataclass(frozen=True)
class TapeFill:
    price: float
    amount: float
    first_ts_ns: int
    last_ts_ns: int
    prints: int


class TradeTapeStore:
    """Actual Deribit option prints, partitioned by asset and UTC date."""

    COLUMNS = ["date", "instrument_name", "amount", "price", "direction",
               "block_trade_id"]

    def __init__(self, root: str, asset: str, slippage_bps: float = 1.0,
                 window_minutes: int = 5, max_cached_days: int = 16,
                 fill_method: str = "first_trade"):
        self.root = Path(root)
        self.asset = asset
        self.slippage_bps = slippage_bps
        self.window_ns = window_minutes * 60 * 1_000_000_000
        self.max_cached_days = max_cached_days
        self.fill_method = fill_method
        self._days: OrderedDict[str, list[dict]] = OrderedDict()
        self.audit: list[dict] = []

    def _path(self, day: str) -> Path:
        year = day[:4]
        return self.root / f"currency={self.asset}" / f"year={year}" / f"{self.asset}_trades_{day}.parquet"

    def _load_day(self, day: str) -> list[dict]:
        if day in self._days:
            self._days.move_to_end(day)
            return self._days[day]
        path = self._path(day)
        rows = pq.read_table(path, columns=self.COLUMNS).to_pylist() if path.exists() else []
        self._days[day] = rows
        while len(self._days) > self.max_cached_days:
            self._days.popitem(last=False)
        return rows

    def fill(self, instrument: str, side: Side, start_ns: int, quantity: float,
             max_end_ns: int | None = None) -> TapeFill | None:
        """VWAP actual aggressive prints in [start, start+window), full-fill-or-none.

        Direction 0 is seller aggressor and direction 1 is buyer aggressor in
        this archive. Opening shorts therefore use 0; buy-to-close uses 1.
        Block trades are excluded because they were not generally accessible.
        """
        end_ns = start_ns + self.window_ns
        if self.fill_method == "next_trade" and max_end_ns is not None:
            end_ns = max_end_ns
        start = dt.datetime.fromtimestamp(start_ns / 1e9, dt.timezone.utc)
        end = dt.datetime.fromtimestamp((end_ns - 1) / 1e9, dt.timezone.utc)
        day = start.date()
        days = set()
        while day <= end.date():
            days.add(day.isoformat())
            day += dt.timedelta(days=1)
        direction = 1 if side == Side.BUY else 0
        candidates = []
        for day in sorted(days):
            for row in self._load_day(day):
                if row["instrument_name"] != instrument:
                    continue
                if self.fill_method == "aggressor_vwap" and row["direction"] != direction:
                    continue
                if row.get("block_trade_id") not in (None, ""):
                    continue
                ts = int(row["date"].timestamp() * 1e9)
                if start_ns <= ts < end_ns and row["amount"] and row["price"]:
                    candidates.append((ts, float(row["amount"]), float(row["price"])))
        candidates.sort()
        if candidates and self.fill_method in ("first_trade", "next_trade"):
            ts, amount, raw = candidates[0]
            adverse = 1 + self.slippage_bps * 1e-4 if side == Side.BUY else 1 - self.slippage_bps * 1e-4
            final = raw * adverse
            self.audit.append({"instrument": instrument, "side": side.value,
                "fill_method": self.fill_method, "window_start_ns": start_ns,
                "window_end_ns": end_ns, "requested_amount": quantity,
                "available_amount": sum(x[1] for x in candidates), "filled": True,
                "raw_vwap": raw, "slippage_bps": self.slippage_bps,
                "fill_price": final, "first_trade_ts_ns": ts,
                "last_trade_ts_ns": ts, "prints_used": 1})
            return TapeFill(final, quantity, ts, ts, 1)
        remaining, notional, used, first, last = quantity, 0.0, 0.0, 0, 0
        for ts, amount, price in candidates:
            take = min(remaining, amount)
            if take <= 0:
                continue
            if not first:
                first = ts
            last = ts
            notional += take * price
            remaining -= take
            used += 1
            if remaining <= 1e-12:
                raw = notional / quantity
                adverse = 1 + self.slippage_bps * 1e-4 if side == Side.BUY else 1 - self.slippage_bps * 1e-4
                final = raw * adverse
                self.audit.append({"instrument": instrument, "side": side.value,
                    "fill_method": self.fill_method,
                    "window_start_ns": start_ns, "window_end_ns": end_ns,
                    "requested_amount": quantity, "available_amount": sum(x[1] for x in candidates),
                    "filled": True, "raw_vwap": raw, "slippage_bps": self.slippage_bps,
                    "fill_price": final, "first_trade_ts_ns": first,
                    "last_trade_ts_ns": last, "prints_used": used})
                return TapeFill(final, quantity, first, last, used)
        self.audit.append({"instrument": instrument, "side": side.value,
            "fill_method": self.fill_method,
            "window_start_ns": start_ns, "window_end_ns": end_ns,
            "requested_amount": quantity, "available_amount": sum(x[1] for x in candidates),
            "filled": False, "raw_vwap": None, "slippage_bps": self.slippage_bps,
            "fill_price": None, "first_trade_ts_ns": candidates[0][0] if candidates else None,
            "last_trade_ts_ns": candidates[-1][0] if candidates else None,
            "prints_used": len(candidates)})
        return None
