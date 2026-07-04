"""Global / chain aggregates: one scalar per bar from the whole chain. Phase 2 (basic set)."""
from __future__ import annotations

import numpy as np

from .types import OptType


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


# ---- built-in chain aggregates (snap -> scalar) ---------------------------
def g_pcr_oi(s) -> float:
    p = s.otype == OptType.PUT
    c = s.otype == OptType.CALL
    return _safe_div(s.oi[p].sum(), s.oi[c].sum())


def g_pcr_vol(s) -> float:
    p = s.otype == OptType.PUT
    c = s.otype == OptType.CALL
    return _safe_div(s.volume[p].sum(), s.volume[c].sum())


def g_atm_iv(s) -> float:
    opt = s.otype != OptType.NONE
    if not opt.any():
        return 0.0
    m = s.moneyness[opt]
    iv = s.iv[opt]
    i = int(np.argmin(np.abs(m - 1.0)))
    return float(iv[i])


def g_total_oi(s) -> float:
    return float(s.oi.sum())


def g_total_vol(s) -> float:
    return float(s.volume.sum())


def g_total_gamma(s) -> float:
    return float((s.gamma * s.oi).sum())   # open-interest weighted


def g_total_vega(s) -> float:
    return float((s.vega * s.oi).sum())


def g_skew_25(s) -> float:
    calls = s.otype == OptType.CALL
    puts = s.otype == OptType.PUT
    if not calls.any() or not puts.any():
        return 0.0
    ci = int(np.argmin(np.abs(s.delta[calls] - 0.25)))
    pi = int(np.argmin(np.abs(s.delta[puts] + 0.25)))
    return float(s.iv[puts][pi] - s.iv[calls][ci])


def g_gex(s) -> float:
    calls = s.otype == OptType.CALL
    puts = s.otype == OptType.PUT
    return float((s.gamma[calls] * s.oi[calls]).sum() - (s.gamma[puts] * s.oi[puts]).sum())


def g_term_slope(s) -> float:
    opt = s.otype != OptType.NONE
    exps = np.unique(s.expiry[opt & (s.expiry > 0)])
    if exps.size < 2:
        return 0.0

    def atm_iv_at(e):
        m = (s.expiry == e) & opt
        if not m.any():
            return np.nan
        return s.iv[m][int(np.argmin(np.abs(s.moneyness[m] - 1.0)))]

    return float(atm_iv_at(exps[1]) - atm_iv_at(exps[0]))


def g_max_pain(s) -> float:
    calls = s.otype == OptType.CALL
    puts = s.otype == OptType.PUT
    strikes = np.unique(s.strike[(calls | puts) & (s.strike > 0)])
    if strikes.size == 0:
        return 0.0
    Kc, OIc = s.strike[calls], s.oi[calls]
    Kp, OIp = s.strike[puts], s.oi[puts]
    pain = np.array([(OIc * np.maximum(S - Kc, 0)).sum() + (OIp * np.maximum(Kp - S, 0)).sum()
                     for S in strikes])
    return float(strikes[int(np.argmin(pain))])


def default_registry() -> dict:
    return {
        "pcr_oi": g_pcr_oi,
        "pcr_vol": g_pcr_vol,
        "atm_iv": g_atm_iv,
        "total_oi": g_total_oi,
        "total_vol": g_total_vol,
        "total_gamma": g_total_gamma,
        "total_vega": g_total_vega,
        "skew_25": g_skew_25,
        "gex": g_gex,
        "term_slope": g_term_slope,
        "max_pain": g_max_pain,
    }


class GlobalRegistry:
    def __init__(self, fns: dict | None = None):
        self.fns = dict(fns or default_registry())

    def register(self, name: str, fn) -> None:
        self.fns[name] = fn

    def names(self):
        return list(self.fns)


class Market:
    """Computes registered globals each bar; keeps their history for series()/offset reads."""

    def __init__(self, registry: GlobalRegistry | None = None):
        self.reg = registry or GlobalRegistry()
        self.hist: dict[str, list[float]] = {n: [] for n in self.reg.names()}

    def register(self, name, fn) -> None:
        self.reg.register(name, fn)
        self.hist.setdefault(name, [])

    def update(self, snap) -> None:
        for name, fn in self.reg.fns.items():
            try:
                v = float(fn(snap))
            except Exception:
                v = float("nan")
            self.hist.setdefault(name, []).append(v)

    def value(self, name: str, offset: int = 0) -> float:
        h = self.hist.get(name)
        if not h:
            return float("nan")
        i = len(h) - 1 - max(0, offset)
        return h[max(0, i)]

    def series(self, name: str) -> np.ndarray:
        return np.array(self.hist.get(name, []))
