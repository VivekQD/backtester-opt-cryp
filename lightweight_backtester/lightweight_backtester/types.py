from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OptionType(StrEnum):
    CALL = "C"
    PUT = "P"


@dataclass(frozen=True)
class Contract:
    instrument: str
    asset: str
    expiry_ns: int
    strike: float
    option_type: OptionType
    path: str


@dataclass(frozen=True)
class LegIntent:
    contract: Contract
    side: Side
    quantity: float = 1.0
    stop_loss_pct: float | None = None
    profit_target_pct: float | None = None


@dataclass(frozen=True)
class TradeIntent:
    trade_id: str
    entry_target_ns: int
    legs: tuple[LegIntent, ...]
    exit_at_expiry: bool = True
    structure: str = "custom"
    exit_target_ns: int | None = None
    structure_stop_pct: float | None = None
    structure_profit_target_pct: float | None = None
    survivor_profit_target_pct: float | None = None
    survivor_breakeven_stop: bool = False
    disable_structure_target_after_loss_stop: bool = False


@dataclass(frozen=True)
class FillEvent:
    ts_ns: int
    trade_id: str
    contract: Contract
    side: Side
    quantity: float
    price: float
    spot: float
    reason: str


@dataclass(frozen=True)
class MarkEvent:
    ts_ns: int
    spot: float
    marks: dict[str, float]
    bids: dict[str, float] = field(default_factory=dict)
    asks: dict[str, float] = field(default_factory=dict)
    record_equity: bool = True


@dataclass(frozen=True)
class SettlementEvent:
    ts_ns: int
    trade_id: str
    contract: Contract
    quantity: float
    payoff: float
    spot: float


Event = FillEvent | MarkEvent | SettlementEvent


@dataclass
class Position:
    contract: Contract
    quantity: float = 0.0
    average_price: float = 0.0
    entry_ts_ns: int = 0
    opening_fees: float = 0.0


@dataclass(frozen=True)
class EquityPoint:
    ts_ns: int
    native_equity: float
    usd_equity: float
    option_pnl_native: float
    initial_margin_native: float
    maintenance_margin_native: float


@dataclass
class BacktestResult:
    events: list[Event]
    equity: list[EquityPoint]
    leg_ledger: list[dict]
    trade_ledger: list[dict]
    metrics: dict = field(default_factory=dict)
    execution_ledger: list[dict] = field(default_factory=list)
