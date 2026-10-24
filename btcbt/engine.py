"""Chronological backtest loop, Context, Strategy. Phase 1.

Order per bar: settle -> fill prior orders -> mark -> strategy -> queue.
Orders submitted on bar t fill on bar t+1 (no same-bar lookahead).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .globals import Market
from .margin import liquidate
from .metrics import RunResult, compute_stats, match_trades
from .portfolio import Portfolio
from .types import Equity, InstKind, Margin, Order, Side, expand_combo, market
from .venue import VenueConfig, settlement_index


class History:
    def __init__(self):
        self.ts: list[int] = []
        self.idx: list[float] = []

    def push(self, snap):
        self.ts.append(snap.ts)
        self.idx.append(snap.index)

    def index(self, n: int) -> np.ndarray:
        return np.array(self.idx[-n:])


class Strategy:
    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def on_start(self, ctx): ...
    def on_snapshot(self, snap, ctx): ...
    def on_fill(self, fill, ctx): ...
    def on_expiry(self, ev, ctx): ...
    def on_end(self, ctx): ...


class Context:
    """The strategy's only outward channel. Tickets = (orders, atomic) drained each bar."""

    def __init__(self, engine: "Engine"):
        self._e = engine
        self._tickets: list[tuple[list[Order], bool]] = []
        self._state: dict = {}
        self.snap = None

    # submission
    def submit(self, order: Order):
        gid = self._e._next_group()
        self._tickets.append(([replace(order, group=gid)], False))

    def submit_combo(self, c):
        gid = self._e._next_group()
        self._tickets.append((expand_combo(c, gid), c.atomic))

    def buy(self, sym, qty=1.0):
        self.submit(market(sym, Side.BUY, qty))

    def sell(self, sym, qty=1.0):
        self.submit(market(sym, Side.SELL, qty))

    def close(self, sym):
        p = self._e.portfolio.position(sym)
        if p and p.qty != 0:
            side = Side.SELL if p.qty > 0 else Side.BUY
            self.submit(market(sym, side, abs(p.qty)))

    def cancel_all(self):
        self._tickets.clear()

    # reads
    def select(self):
        return self.snap.select()

    def portfolio(self):
        return self._e.portfolio

    def now(self) -> int:
        return self.snap.ts

    def g(self, name: str, offset: int = 0) -> float:
        return self._e.market.value(name, offset)

    def market(self):
        return self._e.market

    def record(self, key: str, v: float):
        self._e.records.setdefault(key, []).append((self.snap.ts, float(v)))

    def get(self, key: str):
        return self._e.records.get(key, [])

    def index_history(self, n: int):
        return self._e.history.index(n)

    def history(self, sym, field, n):
        return np.zeros(0)   # per-instrument lookback: Phase 2

    def state(self) -> dict:
        return self._state

    def drain_orders(self):
        t = self._tickets
        self._tickets = []
        return t


class Engine:
    def __init__(self, ds, strat: Strategy, fill, venue: VenueConfig, cash: float, margin=None):
        self.ds = ds
        self.strat = strat
        self.fill = fill
        self.venue = venue
        self.margin = margin
        self.portfolio = Portfolio(cash, venue, ds.metas)
        self.history = History()
        self.market = Market()
        self.records: dict = {}
        self._gid = 0
        self.ctx = Context(self)

    def _next_group(self) -> int:
        self._gid += 1
        return self._gid

    def run(self) -> RunResult:
        ds, S, P, V, ctx = self.ds, self.strat, self.portfolio, self.venue, self.ctx
        equity, fills, expiries, greeks = [], [], [], []
        Q: list = []
        peak = -1e300
        settlement_cache: dict[int, float] = {}

        S.on_start(ctx)
        for snap in ds.iter_bars():      # Dataset or ChunkedDataset (streams files)
            ts = snap.ts
            self.history.push(snap)
            self.market.update(snap)
            ctx.snap = snap

            # 1. settle expired open positions
            for sym in list(P.positions.keys()):
                m = ds.metas[sym]
                if m.kind in (InstKind.OPTION, InstKind.FUTURE) and m.expiry_ts and m.expiry_ts <= ts:
                    if m.expiry_ts not in settlement_cache:
                        settlement_cache[m.expiry_ts] = settlement_index(ds.index_series, m.expiry_ts, V.twap_mins)
                    sx = settlement_cache[m.expiry_ts]
                    ev = P.settle(sym, sx)
                    if ev:
                        expiries.append(ev)
                        S.on_expiry(ev, ctx)

            # 2. fill prior-bar tickets
            for orders, atomic in Q:
                fs = [self.fill.fill(o, snap, V) for o in orders]
                if atomic and any(f is None for f in fs):
                    continue
                for f in fs:
                    if f is None:
                        continue
                    P.apply_fill(f)
                    fills.append(f)
                    S.on_fill(f, ctx)
            Q = []

            # 3. perp funding, then mark to market
            P.accrue_funding(snap)
            e = P.equity(snap)
            peak = max(peak, e.usd)
            dd = (e.usd - peak) / peak if peak > 0 else 0.0
            equity.append((ts, Equity(e.btc, e.usd, dd)))
            greeks.append((ts, P.agg_greeks(snap)))

            # forced liquidation when below maintenance margin
            breach = False
            if self.margin is not None and not P.is_flat():
                native = e.btc if V.margin == Margin.INVERSE else e.usd
                breach = native < self.margin.maintenance(P, snap, V)

            # 4. strategy decides; orders queue for next bar
            S.on_snapshot(snap, ctx)
            Q = ctx.drain_orders()
            if breach:
                ctx.cancel_all()
                Q = [([Order(o.sym, o.side, o.qty, o.kind, o.limit, self._next_group())], False)
                     for o in liquidate(P, snap, V, self.margin)]

        S.on_end(ctx)
        # fold global chain series into records, aligned to equity timestamps
        eq_ts = [t for t, _ in equity]
        records = dict(self.records)
        for name, h in self.market.hist.items():
            if len(h) == len(eq_ts):
                records[f"g.{name}"] = list(zip(eq_ts, h))
        trades = match_trades(fills, expiries, V, ds.metas)
        stats = compute_stats(equity, trades)
        return RunResult(equity, fills, trades, expiries, greeks, records, stats)
