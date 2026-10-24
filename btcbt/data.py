"""Parquet -> canonical arrays -> per-bar Snapshot. Phase 1.

Two ways to feed the engine:

* `Dataset`        - one file, fully in memory. Fine for a few million rows.
* `ChunkedDataset` - many files (e.g. one per year) streamed as ONE continuous
                     backtest on one cash balance. Each source file is split once
                     into time-ordered monthly partitions on disk (cached), then
                     the engine walks them in order with a single month in memory.
                     Symbols, metas and the settlement index series carry across
                     chunk and file boundaries, so positions roll over seamlessly.

`Dataset.load` picks the right one: a list / glob / directory path or a `chunk`
setting returns a `ChunkedDataset`.
"""
from __future__ import annotations

import datetime as dt
import glob as _glob
import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds
import pyarrow.parquet as pq

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

_PARTS_DIR = os.path.join(".btcbt_cache", "parts")
_FLUSH_ROWS = 200_000          # per-month write buffer during partitioning


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
    path: str | list            # file, directory, glob, or list of files
    start: int | None = None       # ns UTC
    end: int | None = None
    column_map: dict | None = None
    hedge: str | None = None        # None | "spot" | "perp": add a synthetic tradable underlying
    # Row pre-filter pushed into the parquet scan. Keys (all optional):
    #   option_types: ["call"] | ["put"] | ["call","put"]
    #   max_dte: float            (needs a `daystogo` column)
    #   strike_index_band: [lo, hi]   keep lo*index <= strike <= hi*index
    filter: dict | None = None
    chunk: str | None = None        # None | "month": stream monthly partitions
    cache_dir: str = _PARTS_DIR


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


def resolve_paths(path) -> list[str]:
    """File | directory | glob | list -> sorted list of parquet files."""
    if isinstance(path, (list, tuple)):
        out = []
        for p in path:
            out.extend(resolve_paths(p))
        return out
    path = str(path)
    if any(ch in path for ch in "*?["):
        return sorted(_glob.glob(path))
    if os.path.isdir(path):
        return sorted(_glob.glob(os.path.join(path, "*.parquet")))
    return [path]


# ---- symbol table (shared across chunks) -------------------------------------
class _SymTable:
    def __init__(self):
        self.sym_of: dict[str, int] = {}
        self.metas: dict[int, Meta] = {}
        self.margin: Margin | None = None
        self._kind = np.zeros(0, np.int8)
        self._otype = np.zeros(0, np.int8)
        self._expiry = np.zeros(0, np.int64)

    def intern(self, name: str, kind=None) -> int:
        s = self.sym_of.get(name)
        if s is not None:
            return s
        s = len(self.sym_of)
        self.sym_of[name] = s
        if kind is None:
            k, ot, strike, exp = parse_name(name)
        else:
            k, ot, strike, exp = kind, OptType.NONE, 0.0, 0
        cur = "BTC_USDC" if self.margin == Margin.LINEAR else "BTC"
        self.metas[s] = Meta(s, name, k, ot, strike, exp, cur, self.margin)
        return s

    def luts(self):
        """Per-sym kind/otype/expiry lookup arrays, grown lazily."""
        n = len(self.metas)
        if self._kind.size != n:
            m = [self.metas[i] for i in range(n)]
            self._kind = np.fromiter((int(x.kind) for x in m), np.int8, n)
            self._otype = np.fromiter((int(x.otype) for x in m), np.int8, n)
            self._expiry = np.fromiter((x.expiry_ts for x in m), np.int64, n)
        return self._kind, self._otype, self._expiry


_FIELDS = ("sym", "kind", "otype", "strike", "expiry", "index", "forward", "mark", "bid",
           "ask", "iv", "bid_iv", "ask_iv", "oi", "volume", "delta", "gamma", "theta", "vega")


def _build(tbl: pa.Table, cmap: dict, st: _SymTable, hedge, start, end) -> dict:
    """Arrow table -> canonical, ts-sorted column dict. Vectorised (no per-row Python)."""
    def arr(stored, default=0.0):
        if stored in tbl.column_names:
            return np.asarray(tbl.column(stored).to_numpy(zero_copy_only=False), dtype=np.float64)
        return np.full(tbl.num_rows, default, dtype=np.float64)

    if st.margin is None:
        cur = ""
        if cmap["currency"] in tbl.column_names and tbl.num_rows:
            cur = tbl.column(cmap["currency"])[0].as_py() or ""
        st.margin = Margin.LINEAR if "USDC" in cur else Margin.INVERSE

    date = np.asarray(tbl.column(cmap["date"]).to_numpy(zero_copy_only=False))
    ts = date.astype("datetime64[ns]").astype(np.int64)

    # intern names via unique + index_in: one Python call per distinct name, not per row
    names = tbl.column(cmap["instrument_name"])
    uniq = pc.unique(names)
    local = pc.index_in(names, value_set=uniq).to_numpy(zero_copy_only=False)
    gmap = np.fromiter((st.intern(n) for n in uniq.to_pylist()), np.int32, len(uniq))
    sym = gmap[local.astype(np.int64)]

    c = dict(sym=sym, strike=arr(cmap["strike"]), index=arr(cmap["index_price"]),
             forward=arr(cmap["underlying_price"]), mark=arr(cmap["mark_price"]),
             bid=arr(cmap["bid_price"]), ask=arr(cmap["ask_price"]),
             iv=arr(cmap["mark_iv"]) / 100.0, bid_iv=arr(cmap["bid_iv"]) / 100.0,
             ask_iv=arr(cmap["ask_iv"]) / 100.0, oi=arr(cmap["open_interest"]),
             volume=arr(cmap["volume"]), delta=arr(cmap["delta"]), gamma=arr(cmap["gamma"]),
             theta=arr(cmap["theta"]), vega=arr(cmap["vega"]))
    del tbl, names, local, date

    # optional synthetic tradable underlying (one row per bar)
    if hedge in ("spot", "perp"):
        spot = hedge == "spot"
        hkind = InstKind.SPOT if spot else InstKind.PERP
        hsym = st.intern("BTC-SPOT" if spot else "BTC-PERP", kind=hkind)
        u, first = np.unique(ts, return_index=True)
        n = u.size
        h_index = c["index"][first]
        h_fwd = c["forward"][first]
        h_mark = h_index if spot else h_fwd
        extra = dict(sym=np.full(n, hsym, np.int32), strike=np.zeros(n), index=h_index,
                     forward=h_fwd, mark=h_mark, bid=h_mark, ask=h_mark, iv=np.zeros(n),
                     bid_iv=np.zeros(n), ask_iv=np.zeros(n), oi=np.full(n, 1e9),
                     volume=np.zeros(n), delta=np.ones(n), gamma=np.zeros(n),
                     theta=np.zeros(n), vega=np.zeros(n))
        ts = np.concatenate([ts, u])
        for k in c:
            c[k] = np.concatenate([c[k], extra[k]])

    # optional time window
    if start is not None or end is not None:
        lo = start if start is not None else ts.min()
        hi = end if end is not None else ts.max()
        keep = (ts >= lo) & (ts <= hi)
        ts = ts[keep]
        for k in c:
            c[k] = c[k][keep]

    # sort by ts so each bar is contiguous
    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    for k in c:
        c[k] = c[k][order]
    del order

    kind_lut, otype_lut, expiry_lut = st.luts()
    c["kind"] = kind_lut[c["sym"]]
    c["otype"] = otype_lut[c["sym"]]
    c["expiry"] = expiry_lut[c["sym"]]
    c["tte"] = asm.tte(c["expiry"], ts)
    c["moneyness"] = asm.moneyness(c["forward"], c["index"], c["strike"])
    c["valid"] = asm.valid_mask(c["bid"], c["ask"], c["mark"])
    c["ts"] = ts
    return c


def _scan_filter(schema: pa.Schema, cmap: dict, f: dict | None):
    if not f:
        return None
    names = schema.names
    expr = None

    def _and(e):
        nonlocal expr
        expr = e if expr is None else expr & e

    if f.get("option_types") and cmap["option_type"] in names:
        _and(pds.field(cmap["option_type"]).isin(list(f["option_types"])))
    if f.get("max_dte") is not None and "daystogo" in names:
        _and(pds.field("daystogo") <= float(f["max_dte"]))
    band = f.get("strike_index_band")
    if band and cmap["strike"] in names and cmap["index_price"] in names:
        lo, hi = float(band[0]), float(band[1])
        k, ix = pds.field(cmap["strike"]), pds.field(cmap["index_price"])
        _and((k >= ix * lo) & (k <= ix * hi))
    return expr


def _read(path: str, cmap: dict, filt: dict | None) -> pa.Table:
    ds = pds.dataset(path)
    want = [v for v in cmap.values() if v in ds.schema.names]
    return ds.to_table(columns=want, filter=_scan_filter(ds.schema, cmap, filt))


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
    def load(cls, cfg: DataConfig):
        paths = resolve_paths(cfg.path)
        if cfg.chunk or len(paths) > 1:
            return ChunkedDataset(cfg)
        cmap = {**_DEFAULT_MAP, **(cfg.column_map or {})}
        st = _SymTable()
        cols = _build(_read(paths[0], cmap, cfg.filter), cmap, st, cfg.hedge, cfg.start, cfg.end)
        self = cls()
        self.metas = st.metas
        self.margin = st.margin
        self._set_cols(cols)
        return self

    def _set_cols(self, cols: dict):
        self.cols = cols
        self.ts_unique, self._offsets = np.unique(cols["ts"], return_index=True)
        self._ts_pos = {int(t): i for i, t in enumerate(self.ts_unique)}
        idx_at = cols["index"][self._offsets]
        self.index_series = np.column_stack([self.ts_unique.astype(np.float64), idx_at.astype(np.float64)])

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


# ---- multi-file / chunked streaming ------------------------------------------
def _part_key(src: str, cmap: dict, filt: dict | None) -> str:
    stt = os.stat(src)
    key = json.dumps([os.path.abspath(src), stt.st_size, int(stt.st_mtime),
                      sorted(cmap.items()), filt or {}], sort_keys=True, default=str)
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def partition_file(src: str, cmap: dict, filt: dict | None, cache_dir: str = _PARTS_DIR,
                   log=print) -> list[str]:
    """Stream `src` once into time-bucketed monthly parquet files. Cached by content key.

    Memory stays bounded: batches are read with no readahead, only needed columns,
    filter pushed down, and each month buffers at most ~_FLUSH_ROWS rows.
    """
    stem = os.path.splitext(os.path.basename(src))[0]
    out = os.path.join(cache_dir, f"{stem}-{_part_key(src, cmap, filt)}")
    done = os.path.join(out, "_DONE")
    if os.path.exists(done):
        return sorted(_glob.glob(os.path.join(out, "*.parquet")))

    os.makedirs(out, exist_ok=True)
    for old in _glob.glob(os.path.join(out, "*.parquet")):
        os.remove(old)
    ds = pds.dataset(src)
    want = [v for v in cmap.values() if v in ds.schema.names]
    scanner = ds.scanner(columns=want, filter=_scan_filter(ds.schema, cmap, filt),
                         batch_size=250_000, batch_readahead=1, fragment_readahead=1)
    writers: dict[str, pq.ParquetWriter] = {}
    bufs: dict[str, list] = {}
    sizes: dict[str, int] = {}
    date_col = cmap["date"]

    def flush(m):
        if not bufs.get(m):
            return
        t = pa.Table.from_batches(bufs[m])
        if m not in writers:
            writers[m] = pq.ParquetWriter(os.path.join(out, f"{m}.parquet"), t.schema,
                                          compression="zstd")
        writers[m].write_table(t)
        bufs[m], sizes[m] = [], 0

    n_in = 0
    for b in scanner.to_batches():
        if b.num_rows == 0:
            continue
        n_in += b.num_rows
        d = np.asarray(b.column(date_col).to_numpy(zero_copy_only=False)).astype("datetime64[M]")
        months, inv = np.unique(d, return_inverse=True)
        for i, mo in enumerate(months):
            m = str(mo)                        # 'YYYY-MM'
            piece = b.filter(pa.array(inv == i))
            bufs.setdefault(m, []).append(piece)
            sizes[m] = sizes.get(m, 0) + piece.num_rows
            if sizes[m] >= _FLUSH_ROWS:
                flush(m)
    for m in list(bufs):
        flush(m)
    for w in writers.values():
        w.close()
    open(done, "w").close()
    if log:
        log(f"[btcbt] partitioned {src}: {n_in:,} rows -> {len(writers)} monthly parts in {out}")
    return sorted(_glob.glob(os.path.join(out, "*.parquet")))


class ChunkedDataset:
    """Many files, one continuous run. Holds one month of rows in memory at a time.

    Exposes what the engine needs: `metas` (grows as new instruments appear),
    `index_series` (accumulated across chunks, used for expiry TWAP settlement)
    and `iter_bars()`.
    """

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.cmap = {**_DEFAULT_MAP, **(cfg.column_map or {})}
        self.paths = resolve_paths(cfg.path)
        if not self.paths:
            raise FileNotFoundError(f"no parquet files match {cfg.path!r}")
        self._st = _SymTable()
        self.metas = self._st.metas
        self._idx_parts: list[np.ndarray] = []
        self._idx_cache: np.ndarray | None = None
        self._chunks: list[tuple[str, list[str]]] | None = None

    @property
    def margin(self) -> Margin:
        return self._st.margin if self._st.margin is not None else Margin.INVERSE

    @property
    def index_series(self) -> np.ndarray:
        if self._idx_cache is None:
            self._idx_cache = (np.concatenate(self._idx_parts) if self._idx_parts
                               else np.zeros((0, 2)))
        return self._idx_cache

    def chunks(self) -> list[tuple[str, list[str]]]:
        """[(month, [part files])] in time order. Partitions sources on first call."""
        if self._chunks is None:
            by_month: dict[str, list[str]] = {}
            for src in self.paths:
                for p in partition_file(src, self.cmap, self.cfg.filter, self.cfg.cache_dir):
                    by_month.setdefault(os.path.splitext(os.path.basename(p))[0], []).append(p)
            lo, hi = self._month_bounds()
            self._chunks = [(m, by_month[m]) for m in sorted(by_month) if lo <= m <= hi]
        return self._chunks

    def _month_bounds(self) -> tuple[str, str]:
        def mon(ns):
            return str(np.datetime64(int(ns), "ns").astype("datetime64[M]"))
        lo = mon(self.cfg.start) if self.cfg.start is not None else "0000-00"
        hi = mon(self.cfg.end) if self.cfg.end is not None else "9999-99"
        return lo, hi

    def load_chunk(self, files: list[str]) -> Dataset:
        tbl = pa.concat_tables([pq.read_table(f) for f in files])
        cols = _build(tbl, self.cmap, self._st, self.cfg.hedge, self.cfg.start, self.cfg.end)
        part = Dataset()
        part.metas = self.metas
        part.margin = self.margin
        part._set_cols(cols)
        return part

    def iter_bars(self):
        self._idx_parts, self._idx_cache = [], None
        last_ts = -1
        for _, files in self.chunks():
            part = self.load_chunk(files)
            if part.ts_unique.size == 0:
                continue
            # duplicate source files (e.g. a quarterly extract of a yearly file) would
            # replay bars; drop anything not strictly after what we already emitted.
            fresh = part.index_series[part.index_series[:, 0] > last_ts]
            self._idx_parts.append(fresh)
            self._idx_cache = None
            for ts in part.ts_unique:
                if ts <= last_ts:
                    continue
                yield part.snapshot_at(int(ts))
            last_ts = int(part.ts_unique[-1])
            del part

    def timestamps(self):
        raise NotImplementedError("ChunkedDataset streams; iterate with iter_bars()")
