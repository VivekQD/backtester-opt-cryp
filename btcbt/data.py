"""Parquet -> canonical arrays -> per-bar Snapshot. Phase 1."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pyarrow.dataset as pds

from . import assemble as asm
from .snapshot import Snapshot
from .types import InstKind, Margin, OptType, NS

_MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
           "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}

# stored -> canonical column names
_DEFAULT_MAP = {
    "date": "date", "instrument_name": "instrument_name", "option_type": "option_type",
    "strike": "strike", "index_price": "index_price", "underlying_price": "underlying_price",
    "mark_price": "mark_price", "bid_price": "bid_price", "ask_price": "ask_price",
    "mark_iv": "mark_iv", "bid_iv": "bid_iv", "ask_iv": "ask_iv",
    "open_interest": "open_interest", "volume": "volume",
    "delta": "delta", "gamma": "gamma", "theta": "theta", "vega": "vega",
    "currency": "currency",
}


@dataclass
class Meta:
    sym: int
    name: str
    kind: InstKind
    otype: OptType
    strike: float
    expiry_ts: int
    currency: str
    margin: Margin


@dataclass
class DataConfig:
    path: str
    start: int | None = None       # ns UTC
    end: int | None = None
    column_map: dict | None = None
    hedge: str | None = None        # None | "spot" | "perp": add a synthetic tradable underlying


def _expiry_ns(tok: str) -> int:
    """'12JUN26' -> ns at 08:00 UTC."""
    mon = _MONTHS[tok[-5:-2].upper()]
    yr = 2000 + int(tok[-2:])
    day = int(tok[:-5])
    d = dt.datetime(yr, mon, day, 8, 0, 0, tzinfo=dt.timezone.utc)
    return int(d.timestamp()) * NS


def parse_name(name: str):
    """instrument_name -> (kind, otype, strike, expiry_ts)."""
    parts = name.split("-")
    if len(parts) >= 4 and parts[-1] in ("C", "P"):
        otype = OptType.CALL if parts[-1] == "C" else OptType.PUT
        return InstKind.OPTION, otype, float(parts[-2]), _expiry_ns(parts[-3])
    if "PERP" in name.upper() or "PERPETUAL" in name.upper():
        return InstKind.PERP, OptType.NONE, 0.0, 0
    if len(parts) >= 2:
        try:
            return InstKind.FUTURE, OptType.NONE, 0.0, _expiry_ns(parts[-1])
        except Exception:
            pass
    return InstKind.SPOT, OptType.NONE, 0.0, 0


class Dataset:
    def __init__(self):
        self.metas: dict[int, Meta] = {}
        self.cols: dict[str, np.ndarray] = {}
        self.index_series: np.ndarray | None = None
        self.ts_unique: np.ndarray | None = None
        self._offsets: np.ndarray | None = None
        self._ts_pos: dict[int, int] = {}
        self.margin: Margin = Margin.INVERSE

    # ---- load -------------------------------------------------------------
    @classmethod
    def load(cls, cfg: DataConfig) -> "Dataset":
        cmap = {**_DEFAULT_MAP, **(cfg.column_map or {})}
        ds = pds.dataset(cfg.path)
        want = [v for v in cmap.values() if v in ds.schema.names]
        tbl = ds.to_table(columns=want)

        def arr(stored, dtype=np.float64, default=0.0):
            if stored in tbl.column_names:
                return np.asarray(tbl.column(stored).to_numpy(zero_copy_only=False), dtype=dtype)
            return np.full(tbl.num_rows, default, dtype=dtype)

        date = np.asarray(tbl.column(cmap["date"]).to_numpy(zero_copy_only=False))
        ts = date.astype("datetime64[ns]").astype(np.int64)
        names = tbl.column(cmap["instrument_name"]).to_pylist()
        currency = tbl.column(cmap["currency"]).to_pylist() if cmap["currency"] in tbl.column_names \
            else ["BTC"] * tbl.num_rows

        self = cls()
        self.margin = Margin.LINEAR if any("USDC" in (c or "") for c in currency[:1]) else Margin.INVERSE

        # intern symbols + parse unique names once
        sym_of: dict[str, int] = {}
        for nm in names:
            if nm not in sym_of:
                s = len(sym_of)
                sym_of[nm] = s
                k, ot, strike_n, exp = parse_name(nm)
                cur = "BTC_USDC" if self.margin == Margin.LINEAR else "BTC"
                self.metas[s] = Meta(s, nm, k, ot, strike_n, exp, cur, self.margin)
        sym = np.fromiter((sym_of[nm] for nm in names), dtype=np.int32, count=len(names))

        kind = np.fromiter((int(self.metas[s].kind) for s in sym), dtype=np.int8, count=len(sym))
        expiry = np.fromiter((self.metas[s].expiry_ts for s in sym), dtype=np.int64, count=len(sym))
        otype = np.fromiter((int(self.metas[s].otype) for s in sym), dtype=np.int8, count=len(sym))

        strike = arr(cmap["strike"])
        index = arr(cmap["index_price"])
        forward = arr(cmap["underlying_price"])
        mark = arr(cmap["mark_price"])
        bid = arr(cmap["bid_price"])
        ask = arr(cmap["ask_price"])
        iv = arr(cmap["mark_iv"]) / 100.0
        bid_iv = arr(cmap["bid_iv"]) / 100.0
        ask_iv = arr(cmap["ask_iv"]) / 100.0
        oi = arr(cmap["open_interest"])
        volume = arr(cmap["volume"])
        delta = arr(cmap["delta"])
        gamma = arr(cmap["gamma"])
        theta = arr(cmap["theta"])
        vega = arr(cmap["vega"])

        # optional synthetic tradable underlying (one row per bar)
        if cfg.hedge in ("spot", "perp"):
            spot = cfg.hedge == "spot"
            u = np.unique(ts)
            idx_map, fwd_map = {}, {}
            for t, ix, fw in zip(ts, index, forward):
                if t not in idx_map:
                    idx_map[t], fwd_map[t] = ix, fw
            n = u.size
            hsym = len(sym_of)
            hname = "BTC-SPOT" if spot else "BTC-PERP"
            sym_of[hname] = hsym
            hkind = InstKind.SPOT if spot else InstKind.PERP
            cur = "BTC_USDC" if self.margin == Margin.LINEAR else "BTC"
            self.metas[hsym] = Meta(hsym, hname, hkind, OptType.NONE, 0.0, 0, cur, self.margin)
            h_index = np.array([idx_map[t] for t in u])
            h_fwd = np.array([fwd_map[t] for t in u])
            h_mark = h_index if spot else h_fwd
            full = lambda val, dt=np.float64: np.full(n, val, dtype=dt)
            ts = np.concatenate([ts, u])
            sym = np.concatenate([sym, full(hsym, np.int32)])
            kind = np.concatenate([kind, full(int(hkind), np.int8)])
            otype = np.concatenate([otype, full(int(OptType.NONE), np.int8)])
            expiry = np.concatenate([expiry, full(0, np.int64)])
            strike = np.concatenate([strike, full(0.0)])
            index = np.concatenate([index, h_index])
            forward = np.concatenate([forward, h_fwd])
            mark = np.concatenate([mark, h_mark])
            bid = np.concatenate([bid, h_mark])
            ask = np.concatenate([ask, h_mark])
            iv = np.concatenate([iv, full(0.0)])
            bid_iv = np.concatenate([bid_iv, full(0.0)])
            ask_iv = np.concatenate([ask_iv, full(0.0)])
            oi = np.concatenate([oi, full(1e9)])
            volume = np.concatenate([volume, full(0.0)])
            delta = np.concatenate([delta, full(1.0)])
            gamma = np.concatenate([gamma, full(0.0)])
            theta = np.concatenate([theta, full(0.0)])
            vega = np.concatenate([vega, full(0.0)])

        # optional time window
        if cfg.start is not None or cfg.end is not None:
            lo = cfg.start if cfg.start is not None else ts.min()
            hi = cfg.end if cfg.end is not None else ts.max()
            keep = (ts >= lo) & (ts <= hi)
            (ts, sym, kind, expiry, otype, strike, index, forward, mark, bid, ask, iv,
             bid_iv, ask_iv, oi, volume, delta, gamma, theta, vega) = (
                a[keep] for a in (ts, sym, kind, expiry, otype, strike, index, forward, mark,
                                  bid, ask, iv, bid_iv, ask_iv, oi, volume, delta, gamma, theta, vega))

        # sort by ts so each bar is contiguous
        order = np.argsort(ts, kind="stable")
        cols_raw = dict(sym=sym, kind=kind, otype=otype, strike=strike, expiry=expiry,
                        index=index, forward=forward, mark=mark, bid=bid, ask=ask, iv=iv,
                        bid_iv=bid_iv, ask_iv=ask_iv, oi=oi, volume=volume,
                        delta=delta, gamma=gamma, vega=vega, theta=theta)
        ts = ts[order]
        for k in cols_raw:
            cols_raw[k] = cols_raw[k][order]

        # derived
        cols_raw["tte"] = asm.tte(cols_raw["expiry"], ts)
        cols_raw["moneyness"] = asm.moneyness(cols_raw["forward"], cols_raw["index"], cols_raw["strike"])
        cols_raw["valid"] = asm.valid_mask(cols_raw["bid"], cols_raw["ask"], cols_raw["mark"])
        cols_raw["ts"] = ts
        self.cols = cols_raw

        # bar offsets + index series
        self.ts_unique, self._offsets = np.unique(ts, return_index=True)
        self._ts_pos = {int(t): i for i, t in enumerate(self.ts_unique)}
        idx_at = cols_raw["index"][self._offsets]
        self.index_series = np.column_stack([self.ts_unique.astype(np.float64), idx_at.astype(np.float64)])
        return self

    # ---- access -----------------------------------------------------------
    def timestamps(self) -> np.ndarray:
        return self.ts_unique

    def index_at(self, ts: int) -> float:
        i = self._ts_pos[int(ts)]
        return float(self.index_series[i, 1])

    def _range(self, ts: int):
        i = self._ts_pos[int(ts)]
        start = int(self._offsets[i])
        stop = int(self._offsets[i + 1]) if i + 1 < len(self._offsets) else len(self.cols["ts"])
        return start, stop

    def snapshot_at(self, ts: int) -> Snapshot:
        start, stop = self._range(ts)
        c = self.cols
        sl = slice(start, stop)
        expiry = c["expiry"][sl]
        kind = c["kind"][sl]
        # universe: drop expired options; keep underlying
        live = (expiry > ts) | (kind != int(InstKind.OPTION))
        cols = {k: c[k][sl][live] for k in
                ("sym", "kind", "otype", "strike", "expiry", "forward", "mark", "bid", "ask", "iv",
                 "bid_iv", "ask_iv", "oi", "volume", "delta", "gamma", "vega", "theta",
                 "tte", "moneyness", "valid")}
        return Snapshot(int(ts), self.index_at(ts), cols)

    def expiries_at(self, ts: int) -> np.ndarray:
        start, stop = self._range(ts)
        e = self.cols["expiry"][start:stop]
        return np.unique(self.cols["sym"][start:stop][e == ts])

    def iter_bars(self):
        for ts in self.ts_unique:
            yield self.snapshot_at(int(ts))
