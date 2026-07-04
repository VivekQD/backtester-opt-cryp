"""Fill models. Phase 3.

Snapshot-only data (bid/ask/mark, no OHLC/tick), so all fills are bar-level:
- market: fill against the touch, with spread fallback + slippage + optional size impact
- limit: APPROXIMATE — fills only if the limit is at least as good as the current
  touch this bar. We cannot prove an intrabar touch without OHLC/tick data.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .types import Fill, Order, OrderKind, Side
from .venue import VenueConfig, trade_fee


def _finite_pos(x) -> bool:
    return math.isfinite(float(x)) and float(x) > 0


def _touch_or_mark(snap, r: int, side: Side):
    touch = snap.ask[r] if side == Side.BUY else snap.bid[r]
    if _finite_pos(touch):
        return float(touch)
    mark = snap.mark[r]
    if _finite_pos(mark):
        return float(mark)
    return None


class FillModel:
    def fill(self, o: Order, snap, v: VenueConfig) -> Fill | None:
        raise NotImplementedError


@dataclass
class DefaultFill(FillModel):
    half_spread_bps: float = 5.0     # fallback when the touch is empty
    slippage_bps: float = 1.0
    impact_bps_per_oi: float = 0.0   # extra slippage when qty is large vs OI (0 = off)
    approximate_limit: bool = True

    def _impact(self, o, snap, r, d) -> float:
        if self.impact_bps_per_oi <= 0:
            return 0.0
        oi = snap.oi[r] if snap.oi[r] > 0 else 1.0
        return self.impact_bps_per_oi * (o.qty / oi) * 1e-4 * d

    def fill(self, o: Order, snap, v: VenueConfig) -> Fill | None:
        r = snap.row(o.sym)
        if r is None or not snap.valid[r]:
            return None
        if not _finite_pos(snap.index):
            return None
        d = 1.0 if o.side == Side.BUY else -1.0

        if o.kind == OrderKind.TRIGGERED:
            if o.limit is None or not _finite_pos(o.limit):
                return None
            px = float(o.limit)
            return Fill(o.sym, o.side, o.qty, px,
                        trade_fee(v, px, snap.index, o.qty), snap.ts, o.group, float(snap.index))

        if o.kind == OrderKind.MARKET:
            base = snap.ask[r] if o.side == Side.BUY else snap.bid[r]
            if _finite_pos(base):
                base = float(base)
            elif _finite_pos(snap.mark[r]):
                base = float(snap.mark[r]) * (1.0 + self.half_spread_bps * 1e-4 * d)
            else:
                return None
            px = base * (1.0 + self.slippage_bps * 1e-4 * d + self._impact(o, snap, r, d))
            if not _finite_pos(px):
                return None
            return Fill(o.sym, o.side, o.qty, px,
                        trade_fee(v, px, snap.index, o.qty), snap.ts, o.group, float(snap.index))

        # LIMIT (approximate)
        if not self.approximate_limit or o.limit is None or not _finite_pos(o.limit):
            return None
        touch = _touch_or_mark(snap, r, o.side)
        if touch is None:
            return None
        crossed = (o.side == Side.BUY and touch <= o.limit) or \
                  (o.side == Side.SELL and touch >= o.limit)
        if not crossed:
            return None
        px = o.limit  # conservative: fill at the limit, not the better touch
        return Fill(o.sym, o.side, o.qty, px,
                    trade_fee(v, px, snap.index, o.qty), snap.ts, o.group, float(snap.index))


@dataclass
class MidFill(FillModel):
    """Optimistic sanity-check model: fill at mid (or mark) with no slippage/fees toggle."""
    fee: bool = True

    def fill(self, o: Order, snap, v: VenueConfig) -> Fill | None:
        r = snap.row(o.sym)
        if r is None or not snap.valid[r]:
            return None
        if not _finite_pos(snap.index):
            return None
        if o.kind == OrderKind.TRIGGERED:
            if o.limit is None or not _finite_pos(o.limit):
                return None
            px = float(o.limit)
            f = trade_fee(v, px, snap.index, o.qty) if self.fee else 0.0
            return Fill(o.sym, o.side, o.qty, px, f, snap.ts, o.group, float(snap.index))
        px = snap.mid(r)
        if not _finite_pos(px):
            return None
        f = trade_fee(v, px, snap.index, o.qty) if self.fee else 0.0
        return Fill(o.sym, o.side, o.qty, px, f, snap.ts, o.group, float(snap.index))
