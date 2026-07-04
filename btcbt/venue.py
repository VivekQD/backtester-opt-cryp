"""Venue conventions: inverse/linear payoff, Deribit-style fees, settlement."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from .types import Margin, OptType, NS


@dataclass
class VenueConfig:
    margin: Margin = Margin.INVERSE
    contract_size: float = 1.0
    option_trade_fee_rate: float = 0.0003      # 0.03% of underlying/index
    option_settle_fee_rate: float = 0.00015    # 0.015% delivery fee
    fee_cap_rate: float = 0.125                # cap = 12.5% of option value
    daily_option_settle_fee_rate: float = 0.0  # Deribit exempts daily options
    twap_mins: int = 30
    funding_series: object = None              # optional (N,2) [ts_ns, rate] for PERP


def payoff(margin: Margin, otype: OptType, strike: float, settle: float) -> float:
    """Cash payoff per contract. Inverse -> BTC, linear -> USDC."""
    if not np.isfinite(strike) or not np.isfinite(settle) or settle <= 0:
        return 0.0
    if otype == OptType.CALL:
        intrinsic = max(settle - strike, 0.0)
    elif otype == OptType.PUT:
        intrinsic = max(strike - settle, 0.0)
    else:
        return 0.0
    if margin == Margin.INVERSE:
        return intrinsic / settle if settle > 0 else 0.0
    return intrinsic


def _fee_basis(cfg: VenueConfig, index_price: float) -> float:
    """Underlying amount used by Deribit option fees, in settlement currency."""
    if cfg.margin == Margin.INVERSE:
        return cfg.contract_size
    index = float(index_price)
    return (index if np.isfinite(index) and index > 0 else 0.0) * cfg.contract_size


def _option_value(cfg: VenueConfig, option_price: float) -> float:
    px = float(option_price)
    return abs(px) * cfg.contract_size if np.isfinite(px) else 0.0


def trade_fee(cfg: VenueConfig, premium: float, index_price: float = 0.0,
              qty: float = 1.0, rate: float | None = None) -> float:
    """Deribit option trading fee for a fill, in settlement currency.

    Inverse options: MIN(0.0003, 12.5% * option price) * amount.
    Linear options:  MIN(0.0003 * index, 12.5% * option price) * amount * contract size.
    """
    r = cfg.option_trade_fee_rate if rate is None else float(rate)
    per_contract = min(r * _fee_basis(cfg, index_price),
                       cfg.fee_cap_rate * _option_value(cfg, premium))
    return abs(float(qty)) * per_contract


def _is_daily_option_expiry(expiry_ts: int | None) -> bool:
    if not expiry_ts:
        return False
    # Deribit's weekly/monthly/quarterly expiries are Fridays at 08:00 UTC.
    # Non-Friday option expiries are daily options and have no delivery fee.
    d = dt.datetime.fromtimestamp(expiry_ts / NS, dt.timezone.utc)
    return d.weekday() != 4


def settle_fee(cfg: VenueConfig, payoff_amt: float, index_price: float = 0.0,
               qty: float = 1.0, expiry_ts: int | None = None) -> float:
    """Deribit option delivery fee for an ITM expiry, in settlement currency."""
    rate = cfg.daily_option_settle_fee_rate if _is_daily_option_expiry(expiry_ts) \
        else cfg.option_settle_fee_rate
    per_contract = min(rate * _fee_basis(cfg, index_price),
                       cfg.fee_cap_rate * _option_value(cfg, payoff_amt))
    return abs(float(qty)) * per_contract


def settlement_index(index_series: np.ndarray, expiry_ts: int, mins: int) -> float:
    """TWAP of the index over `mins` ending at expiry. index_series: (N,2) [ts_ns, index]."""
    if index_series is None or len(index_series) == 0:
        return 0.0
    ts = index_series[:, 0]
    idx = index_series[:, 1]
    lo = expiry_ts - mins * 60 * NS
    start = int(np.searchsorted(ts, lo, side="left"))
    stop = int(np.searchsorted(ts, expiry_ts, side="right"))
    w = idx[start:stop]
    w = w[np.isfinite(w)]
    if w.size:
        return float(w.mean())
    prior = idx[:stop]
    prior = prior[np.isfinite(prior)]
    if prior.size:
        return float(prior[-1])
    any_finite = idx[np.isfinite(idx)]
    return float(any_finite[-1]) if any_finite.size else 0.0


def position_value(qty: float, entry: float, mark: float, contract_size: float) -> float:
    """Unrealized value of a position in settlement currency."""
    return qty * (mark - entry) * contract_size


def funding_rate_at(funding_series, ts: int) -> float:
    """Most recent funding rate at or before ts. 0 if no series."""
    if funding_series is None or len(funding_series) == 0:
        return 0.0
    t = funding_series[:, 0]
    prior = funding_series[t <= ts]
    return float(prior[-1, 1]) if len(prior) else 0.0


def funding(qty: float, mark: float, rate: float, contract_size: float) -> float:
    """Cash delta from one funding accrual. Long pays when rate>0."""
    return -qty * mark * rate * contract_size
