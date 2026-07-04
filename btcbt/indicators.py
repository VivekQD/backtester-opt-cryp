"""Indicators on a source series, multi-timeframe. Phase 4.

Source is a single scalar series (the index, or a global). With snapshot-only
data we don't have per-bar OHLC, so ATR/Stochastic are approximated from the
source series (documented). Each indicator exposes value(name, field, offset).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import rolling
from .types import NS


def parse_tf(tf: str) -> int:
    s = tf.strip().lower()
    n = int("".join(c for c in s if c.isdigit()) or "1")
    unit = "".join(c for c in s if c.isalpha())
    mult = {"m": 60, "min": 60, "h": 3600, "hr": 3600, "d": 86400}.get(unit, 60)
    return n * mult * NS


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    kind: str
    params: dict
    timeframe: str = "5m"
    source: str = "index"


def _compute(kind: str, x: np.ndarray, params: dict) -> dict:
    """Return {field: series} for the indicator over source series x."""
    w = int(params.get("window", 14))
    if kind == "sma":
        return {None: rolling.sma(x, w)}
    if kind == "ema":
        return {None: rolling.ema(x, w)}
    if kind == "std":
        return {None: rolling.rolling_std(x, w)}
    if kind == "zscore":
        return {None: rolling.zscore(x, w)}
    if kind in ("rank", "iv_rank"):
        return {None: rolling.rank(x, w)}
    if kind == "rsi":
        return {None: _rsi(x, w)}
    if kind == "macd":
        fast = rolling.ema(x, int(params.get("fast", 12)))
        slow = rolling.ema(x, int(params.get("slow", 26)))
        macd = fast - slow
        sig = rolling.ema(macd, int(params.get("signal", 9)))
        return {None: macd, "macd": macd, "signal": sig}
    if kind == "stochastic":
        k, d = _stochastic(x, w, int(params.get("smooth", 3)))
        return {None: k, "stoch": k, "stoch_signal": d}
    if kind == "atr":
        return {None: _atr_approx(x, w)}
    return {None: rolling.sma(x, w)}


def _rsi(x, n):
    out = np.full(x.size, np.nan)
    if x.size < n + 1:
        return out
    diff = np.diff(x)
    for i in range(n, x.size):
        d = diff[i - n:i]
        up = d[d > 0].sum()
        dn = -d[d < 0].sum()
        rs = up / dn if dn > 0 else np.inf
        out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def _stochastic(x, n, smooth):
    k = np.full(x.size, np.nan)
    for i in range(n - 1, x.size):
        w = x[i - n + 1:i + 1]
        lo, hi = w.min(), w.max()
        k[i] = 100.0 * (x[i] - lo) / (hi - lo) if hi > lo else 50.0
    d = rolling.sma(k, smooth)
    return k, d


def _atr_approx(x, n):
    # no OHLC -> use rolling mean of |delta(source)| as a volatility proxy
    tr = np.abs(np.diff(x, prepend=x[:1]))
    return rolling.sma(tr, n)


class _IndState:
    def __init__(self, spec: IndicatorSpec):
        self.spec = spec
        self.tf = parse_tf(spec.timeframe)
        self.bucket = None
        self.samples: list[float] = []
        self.cache: dict | None = None
        self.dirty = True

    def feed(self, ts: int, value: float):
        b = ts // self.tf
        if self.bucket is None or b > self.bucket:
            self.samples.append(value)     # new timeframe bar
            self.bucket = b
        else:
            self.samples[-1] = value       # update in-progress bar
        self.dirty = True

    def _series(self):
        if self.dirty or self.cache is None:
            self.cache = _compute(self.spec.kind, np.asarray(self.samples, dtype=np.float64), self.spec.params)
            self.dirty = False
        return self.cache

    def value(self, fieldname, offset):
        ser = self._series()
        s = ser.get(fieldname, ser.get(None))
        if s is None or s.size == 0:
            return float("nan")
        i = s.size - 1 - max(0, offset)
        return float(s[i]) if 0 <= i < s.size else float("nan")


class Indicators:
    def __init__(self):
        self._states: dict[str, _IndState] = {}

    def add(self, spec: IndicatorSpec):
        self._states[spec.name] = _IndState(spec)

    def update(self, snap, ctx):
        for st in self._states.values():
            src = st.spec.source
            if src in ("index", "close", "price"):
                v = snap.index
            else:
                v = ctx.g(src)                 # a global series as source
            st.feed(snap.ts, float(v))

    def value(self, name, field=None, offset=0):
        st = self._states.get(name)
        return st.value(field, offset) if st else float("nan")
