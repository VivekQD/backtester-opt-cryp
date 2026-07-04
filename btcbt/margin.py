"""Pluggable margin models + forced liquidation."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .types import InstKind, Margin, OptType, Side, market


class MarginModel:
    def initial(self, p, snap, v) -> float:
        raise NotImplementedError

    def maintenance(self, p, snap, v) -> float:
        raise NotImplementedError


@dataclass
class DeribitStandardMargin(MarginModel):
    """Deribit standard-margin formulas for option positions.

    This is not portfolio margin. Standard margin sums each position's
    requirement independently and gives no spread/portfolio offsets.
    """

    initial_rate: float = 0.15
    initial_floor: float = 0.10
    maintenance_rate: float = 0.075
    non_option_rate: float = 0.0
    non_option_maint_frac: float = 1.0

    @staticmethod
    def _otm(amount: float, ref: float) -> float:
        return max(float(amount), 0.0) / float(ref) if ref > 0 else 0.0

    @staticmethod
    def _mark(snap, row: int) -> float:
        mark = float(snap.mark[row])
        if math.isfinite(mark) and mark > 0:
            return mark
        mid = float(snap.mid(row))
        return mid if math.isfinite(mid) and mid > 0 else 0.0

    @staticmethod
    def _underlying_mark(snap, row: int) -> float:
        fwd = float(snap.forward[row]) if hasattr(snap, "forward") else 0.0
        if math.isfinite(fwd) and fwd > 0:
            return fwd
        index = float(snap.index)
        return index if math.isfinite(index) and index > 0 else 0.0

    def _inverse_short_call_initial(self, strike: float, index: float, mark: float) -> float:
        otm = self._otm(strike - index, index)
        return max(self.initial_rate - otm, self.initial_floor) + mark

    def _inverse_short_put_maintenance(self, mark: float) -> float:
        return max(self.maintenance_rate, self.maintenance_rate * mark) + mark

    def _inverse_short_put_initial(self, strike: float, index: float, mark: float) -> float:
        otm = self._otm(index - strike, index)
        initial = max(self.initial_rate - otm, self.initial_floor) + mark
        return max(initial, self._inverse_short_put_maintenance(mark))

    def _linear_short_call_initial(self, strike: float, index: float, underlying: float,
                                   mark: float) -> float:
        otm = self._otm(strike - underlying, underlying)
        return max(self.initial_rate - otm, self.initial_floor) * index + mark

    def _linear_short_put_initial(self, strike: float, index: float, underlying: float,
                                  mark: float) -> float:
        otm = self._otm(underlying - strike, underlying)
        return max((self.initial_rate - otm) * index, self.initial_floor * strike) + mark

    def _option_req(self, snap, row: int, v, maintenance: bool) -> float:
        mark = self._mark(snap, row)
        strike = float(snap.strike[row])
        index = float(snap.index)
        index = index if math.isfinite(index) and index > 0 else 0.0
        otype = OptType(int(snap.otype[row]))

        if v.margin == Margin.INVERSE:
            if maintenance:
                if otype == OptType.CALL:
                    return self.maintenance_rate + mark
                if otype == OptType.PUT:
                    return self._inverse_short_put_maintenance(mark)
            elif otype == OptType.CALL:
                return self._inverse_short_call_initial(strike, index, mark)
            elif otype == OptType.PUT:
                return self._inverse_short_put_initial(strike, index, mark)
            return 0.0

        underlying = self._underlying_mark(snap, row)
        if maintenance:
            if otype == OptType.CALL:
                return self.maintenance_rate * index + mark
            if otype == OptType.PUT:
                return self.maintenance_rate * min(index, strike) + mark
        elif otype == OptType.CALL:
            return self._linear_short_call_initial(strike, index, underlying, mark)
        elif otype == OptType.PUT:
            return self._linear_short_put_initial(strike, index, underlying, mark)
        return 0.0

    def _req(self, p, snap, v, maintenance: bool) -> float:
        req = 0.0
        for sym, pos in p.positions.items():
            if pos.qty == 0:
                continue
            r = snap.row(sym)
            if r is None:
                continue
            qty = abs(float(pos.qty))
            if int(snap.kind[r]) == int(InstKind.OPTION):
                if pos.qty < 0:
                    req += qty * float(v.contract_size) * self._option_req(snap, r, v, maintenance)
                continue

            rate = self.non_option_rate
            if maintenance:
                rate *= self.non_option_maint_frac
            req += qty * max(float(snap.mark[r]), 0.0) * float(v.contract_size) * rate
        return req

    def initial(self, p, snap, v) -> float:
        return self._req(p, snap, v, maintenance=False)

    def maintenance(self, p, snap, v) -> float:
        return self._req(p, snap, v, maintenance=True)


@dataclass
class SimpleMargin(MarginModel):
    """Legacy approximate margin model.

    Prefer DeribitStandardMargin for Deribit option backtests.
    """

    short_rate: float = 0.15
    long_rate: float = 0.0
    maint_frac: float = 0.6

    def _req(self, p, snap, v) -> float:
        cs = v.contract_size
        req = 0.0
        for sym, pos in p.positions.items():
            if pos.qty == 0:
                continue
            r = snap.row(sym)
            mark = snap.mark[r] if r is not None else 0.0
            rate = self.short_rate if pos.qty < 0 else self.long_rate
            notional = abs(pos.qty) * max(mark, 0.0) * cs
            req += rate * notional
        return req

    def initial(self, p, snap, v) -> float:
        return self._req(p, snap, v)

    def maintenance(self, p, snap, v) -> float:
        return self.maint_frac * self._req(p, snap, v)


def liquidate(p, snap, v, m: MarginModel):
    """Market orders that flatten the whole book (forced close)."""
    orders = []
    for sym, pos in p.positions.items():
        if pos.qty == 0:
            continue
        side = Side.SELL if pos.qty > 0 else Side.BUY
        orders.append(market(sym, side, abs(pos.qty)))
    return orders
