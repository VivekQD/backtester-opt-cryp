"""Composable condition expressions over any variable. Phase 4.

Expr wraps a function (Scope) -> scalar. Builders resolve named variables;
operators compose. Used directly (imperative) or built from a declarative Group
via spec.to_expr.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .types import OptType


class Scope:
    """Everything readable inside a condition for one evaluation."""
    __slots__ = ("snap", "ctx", "market", "indicators", "row", "entry_ts")

    def __init__(self, snap, ctx, market=None, indicators=None, row=None, entry_ts=None):
        self.snap = snap
        self.ctx = ctx
        self.market = market
        self.indicators = indicators
        self.row = row
        self.entry_ts = entry_ts


def _val(x, scope):
    return x(scope) if isinstance(x, Expr) else x


class Expr:
    def __init__(self, fn: Callable):
        self.fn = fn

    def __call__(self, scope: Scope):
        return self.fn(scope)

    def _bin(self, other, op):
        return Expr(lambda s: op(self(s), _val(other, s)))

    # arithmetic
    def __add__(self, o): return self._bin(o, lambda a, b: a + b)
    def __radd__(self, o): return self._bin(o, lambda a, b: b + a)
    def __sub__(self, o): return self._bin(o, lambda a, b: a - b)
    def __rsub__(self, o): return self._bin(o, lambda a, b: b - a)
    def __mul__(self, o): return self._bin(o, lambda a, b: a * b)
    def __rmul__(self, o): return self._bin(o, lambda a, b: b * a)
    def __truediv__(self, o): return self._bin(o, lambda a, b: a / b if b else float("nan"))

    # comparison -> bool
    def __lt__(self, o): return self._bin(o, lambda a, b: a < b)
    def __le__(self, o): return self._bin(o, lambda a, b: a <= b)
    def __gt__(self, o): return self._bin(o, lambda a, b: a > b)
    def __ge__(self, o): return self._bin(o, lambda a, b: a >= b)
    def __eq__(self, o): return self._bin(o, lambda a, b: a == b)

    # boolean
    def __and__(self, o): return Expr(lambda s: bool(self(s)) and bool(_val(o, s)))
    def __or__(self, o): return Expr(lambda s: bool(self(s)) or bool(_val(o, s)))
    def __invert__(self): return Expr(lambda s: not bool(self(s)))

    def abs(self): return Expr(lambda s: abs(self(s)))
    def between(self, lo, hi): return Expr(lambda s: lo <= self(s) <= hi)


# ---- builders -------------------------------------------------------------
def CONST(x) -> Expr:
    return Expr(lambda s: x)


def F(name: str) -> Expr:
    def fn(s):
        col = s.snap.col(name)
        return col[s.row] if s.row is not None else col
    return Expr(fn)


def U(name: str) -> Expr:
    def fn(s):
        if name == "index":
            return s.snap.index
        col = getattr(s.snap, name, None)
        if col is not None and s.row is not None:
            return col[s.row]
        return s.snap.index
    return Expr(fn)


def G(name: str, offset: int = 0) -> Expr:
    return Expr(lambda s: s.market.value(name, offset) if s.market else float("nan"))


def PF(name: str) -> Expr:
    def fn(s):
        p = s.ctx.portfolio()
        if name in ("delta", "gamma", "vega", "theta"):
            return getattr(p.agg_greeks(s.snap), name)
        if name == "cash":
            return p.cash
        if name == "unrealized":
            return p.unrealized(s.snap)
        if name == "equity":
            return p.equity(s.snap).usd
        return 0.0
    return Expr(fn)


def IND(name: str, field: str = None, offset: int = 0) -> Expr:
    return Expr(lambda s: s.indicators.value(name, field, offset) if s.indicators else float("nan"))


def REC(key: str) -> Expr:
    def fn(s):
        h = s.ctx.get(key)
        return h[-1][1] if h else float("nan")
    return Expr(fn)


def T(field: str) -> Expr:
    from .spec import time_fields
    def fn(s):
        tf = time_fields(s.snap.ts, "UTC", s.entry_ts, snap=s.snap, row=s.row)
        return tf.get(field, 0)
    return Expr(fn)


def VAR(name: str, offset: int = 0) -> Expr:
    ns, _, rest = name.partition(".")
    if ns == "g":
        return G(rest, offset)
    if ns == "pf":
        return PF(rest)
    if ns == "t":
        return T(rest)
    if ns in ("u", "market"):
        return U(rest)
    if ns == "ind":
        return IND(rest, offset=offset)
    return F(name)


# ---- reducers (vector -> scalar over a Selector) --------------------------
def SUM(sel, field): return float(sel.s.col(field)[sel.rows_].sum())
def MEAN(sel, field): import numpy as np; return float(np.mean(sel.s.col(field)[sel.rows_])) if sel.rows_.size else 0.0
def MAX(sel, field): return float(sel.s.col(field)[sel.rows_].max()) if sel.rows_.size else 0.0
def MIN(sel, field): return float(sel.s.col(field)[sel.rows_].min()) if sel.rows_.size else 0.0
def COUNT(sel): return int(sel.rows_.size)


# ---- rules ----------------------------------------------------------------
Action = Callable  # (ctx, scope) -> None


@dataclass
class Rule:
    when: Expr
    do: Action


@dataclass
class Sizing:
    mode: str = "fixed"     # fixed | pct_equity
    value: float = 1.0

    def qty(self, ctx, snap, price: float) -> float:
        if self.mode == "pct_equity" and price > 0:
            eq = ctx.portfolio().equity(snap).usd
            return max(0.0, self.value * eq / price)
        return self.value


@dataclass
class RuleStrategy:
    """Imperative-but-declarative: selector candidates + entry/exit rules + sizing."""
    select: list
    entry: list
    exit: list
    sizing: Sizing
