"""Derived per-row fields: tte, moneyness, validity. Computed once at load. Phase 1."""
from __future__ import annotations

import numpy as np

from .types import YEAR_NS


def tte(expiry: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Time to expiry in years. Non-expiring instruments (expiry<=0) -> inf."""
    out = (expiry.astype(np.float64) - ts.astype(np.float64)) / YEAR_NS
    out[expiry <= 0] = np.inf
    return out


def moneyness(forward: np.ndarray, index: np.ndarray, strike: np.ndarray) -> np.ndarray:
    fwd = np.where(np.isfinite(forward) & (forward > 0), forward, index)
    out = np.zeros_like(strike, dtype=np.float64)
    nz = np.isfinite(fwd) & np.isfinite(strike) & (strike > 0)
    out[nz] = fwd[nz] / strike[nz]
    return out


def valid_mask(bid: np.ndarray, ask: np.ndarray, mark: np.ndarray) -> np.ndarray:
    return (np.isfinite(bid) & (bid > 0)) | \
        (np.isfinite(ask) & (ask > 0)) | \
        (np.isfinite(mark) & (mark > 0))
