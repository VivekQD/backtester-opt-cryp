"""Trade matching, summary stats, RunResult. Phase 1."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from .types import DAY_NS, YEAR_NS, Side


@dataclass(frozen=True)
class Trade:
    sym: int
    qty: float
    entry: float
    exit: float
    pnl: float
    fees: float
    entry_ts: int
    exit_ts: int
    side: Side = Side.BUY
    entry_index: float = 0.0
    exit_index: float = 0.0


@dataclass
class Overview:
    total_pnl: float
    max_dd: float
    trades: int
    win_rate: float
    avg_profit: float
    avg_loss: float


@dataclass
class WinLoss:
    pct: float
    avg: float
    max_day: float
    max_streak: int
    max_streak_pnl: float


@dataclass
class Ratios:
    sharpe: float
    sortino: float
    calmar: float
    expectancy: float
    profit_factor: float
    risk_reward: float


@dataclass
class Costs:
    fee_per_trade: float
    total_fees: float
    trades: int


@dataclass
class Stats:
    overview: Overview
    winning: WinLoss
    losing: WinLoss
    ratios: Ratios
    costs: Costs


@dataclass
class RunResult:
    equity: list = field(default_factory=list)        # [(ts, Equity)]
    fills: list = field(default_factory=list)
    trades: list = field(default_factory=list)        # [Trade]
    expiries: list = field(default_factory=list)
    greeks: list = field(default_factory=list)
    records: dict = field(default_factory=dict)
    stats: Stats | None = None


# ---- trade matching (FIFO, pnl net of fees) -------------------------------
def match_trades(fills, expiries, venue, metas) -> list[Trade]:
    cs = venue.contract_size
    trades: list[Trade] = []
    lots: dict[int, deque] = defaultdict(deque)   # sym -> [qty_signed, price, fee_per_unit, ts, index]

    for f in fills:
        sq = f.qty if f.side == Side.BUY else -f.qty
        fee_pu = f.fee / f.qty if f.qty else 0.0
        dq = lots[f.sym]
        # close against opposite lots first
        while sq != 0 and dq and (dq[0][0] > 0) != (sq > 0):
            lot = dq[0]
            m = min(abs(sq), abs(lot[0]))
            dirn = 1 if lot[0] > 0 else -1
            pnl = m * (f.price - lot[1]) * dirn * cs - m * lot[2] - m * fee_pu
            side = Side.BUY if dirn > 0 else Side.SELL
            trades.append(Trade(f.sym, m, lot[1], f.price, pnl, m * lot[2] + m * fee_pu,
                                lot[3], f.ts, side, lot[4], f.index_price))
            lot[0] -= dirn * m
            sq -= (1 if sq > 0 else -1) * m
            if lot[0] == 0:
                dq.popleft()
        if sq != 0:
            dq.append([sq, f.price, fee_pu, f.ts, f.index_price])

    exp_by_sym = {e.sym: e for e in expiries}
    for sym, dq in lots.items():
        e = exp_by_sym.get(sym)
        if e is None:
            continue
        open_qty = sum(abs(lot[0]) for lot in dq if lot[0] != 0)
        settle_fee_pu = e.fee / open_qty if open_qty else 0.0
        for lot in dq:
            if lot[0] == 0:
                continue
            qty = abs(lot[0])
            dirn = 1 if lot[0] > 0 else -1
            fees = qty * (lot[2] + settle_fee_pu)
            pnl = qty * (e.payoff - lot[1]) * dirn * cs - fees
            side = Side.BUY if dirn > 0 else Side.SELL
            trades.append(Trade(sym, qty, lot[1], e.payoff, pnl, fees, lot[3], e.ts,
                                side, lot[4], e.index_price))
    return trades


# ---- helpers --------------------------------------------------------------
def _mean(xs) -> float:
    return float(np.mean(xs)) if len(xs) else 0.0


def equity_to_returns(usd: np.ndarray) -> np.ndarray:
    if usd.size < 2:
        return np.zeros(0)
    prev = usd[:-1]
    out = np.zeros(usd.size - 1)
    nz = prev != 0
    out[nz] = np.diff(usd)[nz] / prev[nz]
    return out


def daily_pnl(ts: np.ndarray, usd: np.ndarray) -> np.ndarray:
    if usd.size == 0:
        return np.zeros(0)
    day = ts // DAY_NS
    last = {}
    for d, v in zip(day, usd):
        last[int(d)] = v
    vals = np.array([last[k] for k in sorted(last)])
    return np.diff(vals) if vals.size > 1 else np.zeros(0)


def max_drawdown(usd: np.ndarray) -> float:
    if usd.size == 0:
        return 0.0
    peak = np.maximum.accumulate(usd)
    dd = np.where(peak > 0, (usd - peak) / peak, 0.0)
    return float(dd.min())


def _bars_per_year(ts: np.ndarray) -> float:
    if ts.size < 2:
        return float(365 * 24 * 12)   # 5-min default
    dtm = np.median(np.diff(ts))
    return YEAR_NS / dtm if dtm > 0 else float(365 * 24 * 12)


def sharpe(r: np.ndarray, bpy: float) -> float:
    if r.size == 0 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(bpy))


def sortino(r: np.ndarray, bpy: float) -> float:
    if r.size == 0:
        return 0.0
    down = r[r < 0]
    dd = down.std() if down.size else 0.0
    if dd == 0:
        return 0.0
    return float(r.mean() / dd * np.sqrt(bpy))


def calmar(usd: np.ndarray, ts: np.ndarray) -> float:
    if usd.size < 2 or usd[0] <= 0:
        return 0.0
    span = ts[-1] - ts[0]
    if span <= 0:
        return 0.0
    cagr = (usd[-1] / usd[0]) ** (YEAR_NS / span) - 1.0
    mdd = abs(max_drawdown(usd))
    return float(cagr / mdd) if mdd > 0 else 0.0


def _streaks(trades):
    win_s = loss_s = 0
    best_w = best_l = 0
    cur_w = cur_l = 0
    wp = lp = 0.0
    cur_wp = cur_lp = 0.0
    for t in trades:
        if t.pnl > 0:
            cur_w += 1; cur_wp += t.pnl; cur_l = 0; cur_lp = 0.0
            if cur_w > best_w:
                best_w, wp = cur_w, cur_wp
        else:
            cur_l += 1; cur_lp += t.pnl; cur_w = 0; cur_wp = 0.0
            if cur_l > best_l:
                best_l, lp = cur_l, cur_lp
    return best_w, wp, best_l, lp


def expectancy(trades) -> float:
    if not trades:
        return 0.0
    W = [t.pnl for t in trades if t.pnl > 0]
    L = [t.pnl for t in trades if t.pnl <= 0]
    n = len(trades)
    return (len(W) / n) * _mean(W) + (len(L) / n) * _mean(L)


def profit_factor(trades) -> float:
    gains = sum(t.pnl for t in trades if t.pnl > 0)
    losses = -sum(t.pnl for t in trades if t.pnl <= 0)
    return float(gains / losses) if losses > 0 else 0.0


def risk_reward(trades) -> float:
    W = [t.pnl for t in trades if t.pnl > 0]
    L = [t.pnl for t in trades if t.pnl <= 0]
    al = abs(_mean(L))
    return float(_mean(W) / al) if al > 0 else 0.0


def compute_stats(equity, trades) -> Stats:
    usd = np.array([e.usd for _, e in equity]) if equity else np.zeros(1)
    ts = np.array([t for t, _ in equity]) if equity else np.zeros(1, dtype=np.int64)
    r = equity_to_returns(usd)
    d = daily_pnl(ts, usd)
    bpy = _bars_per_year(ts)
    W = [t for t in trades if t.pnl > 0]
    L = [t for t in trades if t.pnl <= 0]
    n = len(trades)
    ws, wp, ls, lp = _streaks(trades)
    overview = Overview(
        total_pnl=sum(t.pnl for t in trades),
        max_dd=max_drawdown(usd),
        trades=n,
        win_rate=len(W) / n if n else 0.0,
        avg_profit=_mean([t.pnl for t in W]),
        avg_loss=_mean([t.pnl for t in L]),
    )
    winning = WinLoss(len(W) / n if n else 0.0, _mean([t.pnl for t in W]),
                      float(d[d > 0].max()) if (d > 0).any() else 0.0, ws, wp)
    losing = WinLoss(len(L) / n if n else 0.0, _mean([t.pnl for t in L]),
                     float(d[d < 0].min()) if (d < 0).any() else 0.0, ls, lp)
    ratios = Ratios(sharpe(r, bpy), sortino(r, bpy), calmar(usd, ts),
                    expectancy(trades), profit_factor(trades), risk_reward(trades))
    costs = Costs(fee_per_trade=(sum(t.fees for t in trades) / n if n else 0.0),
                  total_fees=sum(t.fees for t in trades), trades=n)
    return Stats(overview, winning, losing, ratios, costs)
