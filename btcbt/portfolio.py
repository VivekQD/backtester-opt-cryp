"""Accounting: positions, cash, MTM, greeks, settlement. Phase 1."""
from __future__ import annotations

import math

from .types import Equity, ExpiryEvent, Fill, Greeks, InstKind, Margin, Position, Side
from .venue import VenueConfig, funding, funding_rate_at, payoff, settle_fee


class Portfolio:
    def __init__(self, cash: float, venue: VenueConfig, metas: dict):
        self.cash = float(cash)
        self.venue = venue
        self.metas = metas
        self.positions: dict[int, Position] = {}

    def position(self, sym: int):
        return self.positions.get(sym)

    def net_qty(self, sym: int) -> float:
        p = self.positions.get(sym)
        return p.qty if p else 0.0

    def is_flat(self) -> bool:
        return all(p.qty == 0 for p in self.positions.values())

    # ---- fills ----
    def apply_fill(self, f: Fill) -> None:
        cs = self.venue.contract_size
        sq = f.qty if f.side == Side.BUY else -f.qty
        self.cash += -(sq * f.price) * cs - f.fee
        p = self.positions.get(f.sym) or Position(f.sym)
        nq = p.qty + sq
        if p.qty == 0 or (sq > 0) == (p.qty > 0):       # opening / adding
            p.avg_entry = (p.qty * p.avg_entry + sq * f.price) / nq if nq != 0 else 0.0
        else:                                            # reducing / closing
            closed = min(abs(sq), abs(p.qty))
            p.realized += closed * (f.price - p.avg_entry) * (1 if p.qty > 0 else -1) * cs
            if nq != 0 and (nq > 0) != (p.qty > 0):      # flipped
                p.avg_entry = f.price
        p.qty = nq
        if nq == 0:
            p.avg_entry = 0.0
        self.positions[f.sym] = p

    # ---- valuation ----
    def unrealized(self, snap) -> float:
        cs = self.venue.contract_size
        u = 0.0
        for sym, p in self.positions.items():
            if p.qty == 0:
                continue
            r = snap.row(sym)
            if r is None:
                continue
            mark = float(snap.mark[r])
            if not math.isfinite(mark) or mark <= 0:
                mark = snap.mid(r)
            if not math.isfinite(mark) or mark <= 0:
                continue
            u += p.qty * (mark - p.avg_entry) * cs
        return u

    def equity(self, snap) -> Equity:
        base = self.cash + self.unrealized(snap)
        index = float(snap.index)
        valid_index = math.isfinite(index) and index > 0
        if self.venue.margin == Margin.INVERSE:
            return Equity(base, base * index if valid_index else 0.0)
        return Equity(base / index if valid_index else 0.0, base)

    mark_to_market = equity

    def agg_greeks(self, snap) -> Greeks:
        g = [0.0, 0.0, 0.0, 0.0]
        for sym, p in self.positions.items():
            if p.qty == 0:
                continue
            r = snap.row(sym)
            if r is None:
                continue
            g[0] += p.qty * snap.delta[r]
            g[1] += p.qty * snap.gamma[r]
            g[2] += p.qty * snap.vega[r]
            g[3] += p.qty * snap.theta[r]
        return Greeks(*g)

    def accrue_funding(self, snap) -> float:
        """Funding on PERP positions for this bar. 0 if no funding series."""
        fs = getattr(self.venue, "funding_series", None)
        if fs is None:
            return 0.0
        rate = funding_rate_at(fs, snap.ts)
        if rate == 0.0:
            return 0.0
        cs = self.venue.contract_size
        total = 0.0
        for sym, p in self.positions.items():
            if p.qty == 0 or self.metas[sym].kind != InstKind.PERP:
                continue
            r = snap.row(sym)
            if r is None:
                continue
            delta_cash = funding(p.qty, snap.mark[r], rate, cs)
            self.cash += delta_cash
            total += delta_cash
        return total

    # ---- settlement ----
    def settle(self, sym: int, settle_index: float):
        p = self.positions.get(sym)
        if not p or p.qty == 0:
            return None
        m = self.metas[sym]
        cs = self.venue.contract_size
        pay = payoff(self.venue.margin, m.otype, m.strike, settle_index)
        fee = settle_fee(self.venue, pay, settle_index, abs(p.qty), m.expiry_ts) if pay > 0 else 0.0
        self.cash += (p.qty * pay) * cs - fee
        p.realized += p.qty * (pay - p.avg_entry) * cs - fee
        ev = ExpiryEvent(sym, settle_index, pay, fee, m.expiry_ts, settle_index)
        del self.positions[sym]
        return ev
