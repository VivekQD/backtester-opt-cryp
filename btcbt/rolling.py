"""Rolling helpers over 1D series. Phase 2.

Each returns an array aligned to the input (NaN where the window isn't full).
Use on ctx.index_history(n), a recorded series, or a global's history.
"""
from __future__ import annotations

import numpy as np


def _as1d(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


def sma(x, n: int) -> np.ndarray:
    # NaN-aware: a window with any NaN -> NaN, but NaNs never propagate forward.
    x = _as1d(x)
    out = np.full(x.size, np.nan)
    if n <= 0 or x.size < n:
        return out
    for i in range(n - 1, x.size):
        w = x[i - n + 1:i + 1]
        if not np.isnan(w).any():
            out[i] = w.mean()
    return out


def ema(x, n: int) -> np.ndarray:
    # NaN-aware: skip leading/embedded NaNs, seed on first valid value.
    x = _as1d(x)
    out = np.full(x.size, np.nan)
    if n <= 0:
        return out
    a = 2.0 / (n + 1.0)
    prev = None
    for i in range(x.size):
        v = x[i]
        if np.isnan(v):
            continue
        prev = v if prev is None else a * v + (1 - a) * prev
        out[i] = prev
    return out


def rolling_std(x, n: int) -> np.ndarray:
    x = _as1d(x)
    out = np.full(x.size, np.nan)
    if n <= 1 or x.size < n:
        return out
    for i in range(n - 1, x.size):
        out[i] = x[i - n + 1:i + 1].std(ddof=1)
    return out


def zscore(x, n: int) -> np.ndarray:
    x = _as1d(x)
    m = sma(x, n)
    s = rolling_std(x, n)
    out = np.full(x.size, np.nan)
    ok = s > 0
    out[ok] = (x[ok] - m[ok]) / s[ok]
    return out


def rank(x, n: int) -> np.ndarray:
    """Percentile rank (0..1) of the current value within the trailing window."""
    x = _as1d(x)
    out = np.full(x.size, np.nan)
    if n <= 0 or x.size < n:
        return out
    for i in range(n - 1, x.size):
        w = x[i - n + 1:i + 1]
        out[i] = float((w <= x[i]).sum() - 1) / (n - 1) if n > 1 else 1.0
    return out


def pct_change(x, n: int = 1) -> np.ndarray:
    x = _as1d(x)
    out = np.full(x.size, np.nan)
    if x.size <= n:
        return out
    prev = x[:-n]
    cur = x[n:]
    r = np.full(cur.size, np.nan)
    nz = prev != 0
    r[nz] = (cur[nz] - prev[nz]) / prev[nz]
    out[n:] = r
    return out


def last(fn, x, n: int) -> float:
    """Scalar convenience: the final value of a rolling series (NaN-safe)."""
    s = fn(x, n)
    return float(s[-1]) if s.size and not np.isnan(s[-1]) else float("nan")
