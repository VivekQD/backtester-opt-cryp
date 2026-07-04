"""Core value types and enums. Phase 1."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum

SCHEMA_VERSION = 1

NS = 1_000_000_000
DAY_NS = 24 * 3600 * NS
YEAR_NS = 365 * DAY_NS


class InstKind(IntEnum):
    OPTION = 0
    PERP = 1
    FUTURE = 2
    SPOT = 3


class OptType(IntEnum):
    CALL = 0
    PUT = 1
    NONE = 2


class Side(IntEnum):
    BUY = 1
    SELL = -1


class Margin(IntEnum):
    INVERSE = 0
    LINEAR = 1


class OrderKind(IntEnum):
    MARKET = 0
    LIMIT = 1
    TRIGGERED = 2


@dataclass(frozen=True)
class Order:
    sym: int
    side: Side
    qty: float
    kind: OrderKind = OrderKind.MARKET
    limit: float | None = None
    group: int = 0


@dataclass(frozen=True)
class Leg:
    sym: int
    side: Side
    ratio: float = 1.0


@dataclass(frozen=True)
class Combo:
    legs: list
    qty: float = 1.0
    kind: OrderKind = OrderKind.MARKET
    limit: float | None = None
    atomic: bool = False


@dataclass(frozen=True)
class Fill:
    sym: int
    side: Side
    qty: float
    price: float
    fee: float
    ts: int
    group: int = 0
    index_price: float = 0.0


@dataclass
class Position:
    sym: int
    qty: float = 0.0
    avg_entry: float = 0.0
    realized: float = 0.0


@dataclass(frozen=True)
class ExpiryEvent:
    sym: int
    settle_index: float
    payoff: float
    fee: float
    ts: int
    index_price: float = 0.0


@dataclass(frozen=True)
class Equity:
    btc: float
    usd: float
    dd: float = 0.0


@dataclass(frozen=True)
class Greeks:
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0


# ---- order / combo constructors -------------------------------------------
def market(sym: int, side: Side, qty: float) -> Order:
    return Order(int(sym), side, float(qty), OrderKind.MARKET)


def limit(sym: int, side: Side, qty: float, px: float) -> Order:
    return Order(int(sym), side, float(qty), OrderKind.LIMIT, float(px))


def triggered(sym: int, side: Side, qty: float, px: float) -> Order:
    return Order(int(sym), side, float(qty), OrderKind.TRIGGERED, float(px))


def close(pos: Position) -> Order:
    side = Side.SELL if pos.qty > 0 else Side.BUY
    return Order(pos.sym, side, abs(pos.qty), OrderKind.MARKET)


def combo(legs, qty: float = 1.0, kind: OrderKind = OrderKind.MARKET,
          limit: float | None = None, atomic: bool = False) -> Combo:
    return Combo(list(legs), float(qty), kind, limit, atomic)


def expand_combo(c: Combo, group: int) -> list[Order]:
    """Multi-leg -> per-instrument orders sharing one group id."""
    return [
        Order(leg.sym, leg.side, c.qty * leg.ratio, c.kind, c.limit, group)
        for leg in c.legs
    ]
