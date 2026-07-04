"""Per-bar columnar view (Snapshot) and the fluent Selector. Phase 1."""
from __future__ import annotations

import numpy as np

from .types import InstKind, OptType

_COLS = ("sym", "kind", "otype", "strike", "expiry", "forward", "mark", "bid", "ask",
         "iv", "bid_iv", "ask_iv", "oi", "volume", "delta", "gamma", "vega",
         "theta", "tte", "moneyness", "valid")


class Snapshot:
    """Read-only struct-of-arrays view of one timestamp. Arrays are views into the dataset."""

    __slots__ = ("ts", "index", *_COLS, "_rowof")

    def __init__(self, ts: int, index: float, cols: dict):
        self.ts = ts
        self.index = index
        for k in _COLS:
            setattr(self, k, cols[k])
        self._rowof = None

    def __len__(self) -> int:
        return len(self.sym)

    def col(self, name: str) -> np.ndarray:
        return getattr(self, name)

    def _build_rowof(self):
        self._rowof = {int(s): i for i, s in enumerate(self.sym)}

    def row(self, sym: int):
        if self._rowof is None:
            self._build_rowof()
        return self._rowof.get(int(sym))

    def mid(self, r: int) -> float:
        b, a = self.bid[r], self.ask[r]
        if np.isfinite(b) and np.isfinite(a) and b > 0 and a > 0:
            return 0.5 * (b + a)
        return self.mark[r] if np.isfinite(self.mark[r]) and self.mark[r] > 0 else 0.0

    def spread(self, r: int) -> float:
        b, a = self.bid[r], self.ask[r]
        return (a - b) if (np.isfinite(b) and np.isfinite(a) and b > 0 and a > 0) else 0.0

    def expiries(self) -> np.ndarray:
        return np.unique(self.expiry)

    def atm_strike(self, e: int):
        m = self.expiry == e
        if not m.any():
            return None
        strikes = self.strike[m]
        return float(strikes[np.argmin(np.abs(strikes - self.index))])

    def select(self) -> "Selector":
        return Selector(self, np.arange(len(self.sym)))


class Selector:
    """Composable row filter. Every method returns a new Selector; terminals return syms/rows."""

    def __init__(self, snap: Snapshot, rows: np.ndarray):
        self.s = snap
        self.rows_ = rows

    def _f(self, mask) -> "Selector":
        return Selector(self.s, self.rows_[np.asarray(mask, dtype=bool)])

    # ---- kind / type ----
    def kind(self, k: InstKind):
        return self._f(self.s.kind[self.rows_] == k)

    def options(self):
        return self.kind(InstKind.OPTION)

    def underlying(self):
        return self._f(self.s.kind[self.rows_] != InstKind.OPTION)

    def calls(self):
        return self._f(self.s.otype[self.rows_] == OptType.CALL)

    def puts(self):
        return self._f(self.s.otype[self.rows_] == OptType.PUT)

    def otype(self, ot: OptType):
        return self._f(self.s.otype[self.rows_] == ot)

    # ---- numeric filters ----
    def dte(self, lo: float, hi: float):
        d = self.s.tte[self.rows_] * 365.0
        return self._f((d >= lo) & (d <= hi))

    def expiry(self, e: int):
        return self._f(self.s.expiry[self.rows_] == e)

    def iv(self, lo: float, hi: float):
        v = self.s.iv[self.rows_]
        return self._f((v >= lo) & (v <= hi))

    def delta(self, lo: float, hi: float, abs_: bool = True):
        d = self.s.delta[self.rows_]
        if abs_:
            d = np.abs(d)
        return self._f((d >= lo) & (d <= hi))

    def moneyness(self, lo: float, hi: float):
        v = self.s.moneyness[self.rows_]
        return self._f((v >= lo) & (v <= hi))

    def strike(self, lo: float, hi: float):
        v = self.s.strike[self.rows_]
        return self._f((v >= lo) & (v <= hi))

    def oi_min(self, x: float):
        return self._f(self.s.oi[self.rows_] >= x)

    def volume_min(self, x: float):
        return self._f(self.s.volume[self.rows_] >= x)

    def spread_max(self, x: float):
        b, a = self.s.bid[self.rows_], self.s.ask[self.rows_]
        sp = np.where((b > 0) & (a > 0), a - b, np.inf)
        return self._f(sp <= x)

    def tradable(self):
        b, a = self.s.bid[self.rows_], self.s.ask[self.rows_]
        return self._f((b > 0) | (a > 0))

    def where(self, mask_or_fn):
        m = mask_or_fn(self.s, self.rows_) if callable(mask_or_fn) else mask_or_fn
        return self._f(m)

    # ---- ordering / picking ----
    def sort_by(self, field: str, desc: bool = False):
        v = self.s.col(field)[self.rows_]
        order = np.argsort(v, kind="stable")
        if desc:
            order = order[::-1]
        return Selector(self.s, self.rows_[order])

    def top(self, n: int):
        return Selector(self.s, self.rows_[:n])

    def nearest(self, field: str, target: float):
        if self.rows_.size == 0:
            return self
        v = self.s.col(field)[self.rows_]
        i = int(np.argmin(np.abs(v - target)))
        return Selector(self.s, self.rows_[i:i + 1])

    # ---- terminals ----
    def rows(self) -> np.ndarray:
        return self.rows_

    def syms(self) -> np.ndarray:
        return self.s.sym[self.rows_]

    def first(self):
        if self.rows_.size == 0:
            return None
        return int(self.s.sym[self.rows_[0]])
