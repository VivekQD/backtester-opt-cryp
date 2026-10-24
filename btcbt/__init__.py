"""BTC Options Backtester — Phase 1 package."""
from .types import (
    SCHEMA_VERSION, NS, DAY_NS, YEAR_NS,
    InstKind, OptType, Side, Margin, OrderKind,
    Order, Leg, Combo, Fill, Position, ExpiryEvent, Equity, Greeks,
    market, limit, triggered, close, combo, expand_combo,
)
from .venue import VenueConfig, payoff, trade_fee, settle_fee, settlement_index
from .assemble import tte, moneyness, valid_mask
from .snapshot import Snapshot, Selector
from .data import ChunkedDataset, Dataset, DataConfig, Meta, parse_name, partition_file
from .execution import FillModel, DefaultFill, MidFill
from .portfolio import Portfolio
from .margin import DeribitStandardMargin, MarginModel, SimpleMargin, liquidate
from .globals import GlobalRegistry, Market, default_registry
from .engine import Engine, Context, Strategy, History
from . import rolling, report
from .conditions import (
    Scope, Expr, F, U, G, PF, T, IND, REC, VAR, CONST,
    SUM, MEAN, MAX, MIN, COUNT, Rule, Sizing, RuleStrategy,
)
from .indicators import IndicatorSpec, Indicators, parse_tf
from .spec import (
    Operand, Compare, Group, StrikeSpec, ExpirySpec, LegRisk, LegSpec,
    CalendarSpec, SessionSpec, RiskSpec, StrategySpec,
    compile_spec, to_expr, resolve_strike, resolve_expiry, in_session,
    time_fields, load_spec, save_spec, spec_from_dict,
)
from .metrics import (
    Trade, Overview, WinLoss, Ratios, Costs, Stats, RunResult,
    match_trades, compute_stats,
)
from .runner import (
    RunConfig, run_one, run_sweep, expand_grid, config_hash,
    save_result, load_result, Study, StudyResult,
)

__all__ = [
    "InstKind", "OptType", "Side", "Margin", "OrderKind",
    "Order", "Leg", "Combo", "Fill", "Position", "ExpiryEvent", "Equity", "Greeks",
    "market", "limit", "triggered", "close", "combo", "expand_combo",
    "VenueConfig", "payoff", "trade_fee", "settle_fee", "settlement_index",
    "Snapshot", "Selector", "Dataset", "ChunkedDataset", "DataConfig", "Meta", "parse_name",
    "partition_file",
    "FillModel", "DefaultFill", "MidFill", "Portfolio",
    "MarginModel", "DeribitStandardMargin", "SimpleMargin", "liquidate",
    "GlobalRegistry", "Market", "default_registry",
    "Engine", "Context", "Strategy", "History", "rolling", "report",
    "Trade", "Stats", "RunResult", "match_trades", "compute_stats",
    "RunConfig", "run_one", "run_sweep", "expand_grid", "config_hash", "Study",
    "Scope", "Expr", "F", "U", "G", "PF", "T", "IND", "REC", "VAR", "CONST",
    "Rule", "Sizing", "RuleStrategy",
    "IndicatorSpec", "Indicators",
    "Operand", "Compare", "Group", "StrikeSpec", "ExpirySpec", "LegRisk", "LegSpec",
    "CalendarSpec", "SessionSpec", "RiskSpec", "StrategySpec", "compile_spec",
    "to_expr", "resolve_strike", "resolve_expiry", "in_session", "time_fields",
    "load_spec", "save_spec",
]
