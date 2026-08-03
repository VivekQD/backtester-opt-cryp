from __future__ import annotations

import datetime as dt
import tomllib
from dataclasses import dataclass, field


def to_ns(value: str | int) -> int:
    if isinstance(value, int):
        return value
    x = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if x.tzinfo is None:
        x = x.replace(tzinfo=dt.timezone.utc)
    return int(x.timestamp() * 1_000_000_000)


@dataclass
class ResearchConfig:
    data_root: str
    cache_root: str
    asset: str
    start_ns: int
    end_ns: int
    entry_weekday: int = 4
    entry_frequency: str = "weekly"
    entry_hour_utc: int = 8
    entry_minute_utc: int = 0
    dte_min: float = 6
    dte_max: float = 8
    structure: str = "short_straddle"
    quantity: float = 1.0
    short_delta: float = 0.30
    wing_delta: float = 0.10
    body_mode: str = "delta"
    body_otm_pct: float = 0.02
    exit_dte: float = 0.0
    expiry_weekday: int | None = None
    stop_loss_pct: float | None = None
    profit_target_pct: float | None = None
    leg_profit_target_pct: float | None = None
    survivor_profit_target_pct: float | None = None
    survivor_breakeven_stop: bool = False
    disable_structure_target_after_loss_stop: bool = False
    min_open_interest: float = 10
    max_relative_spread: float = 0.15
    initial_cash_native: float = 1.0
    slippage_bps: float = 1.0
    stop_fill_mode: str = "next_bar_touch"
    theoretical_stop_slippage_pct: float = 0.0
    trade_data_root: str | None = None
    execution_source: str = "quotes"
    snapshot_tolerance_minutes: int = 30
    trade_fee_rate: float = 0.0003
    settlement_fee_rate: float = 0.00015
    fee_cap_rate: float = 0.125
    enforce_margin: bool = True
    combo_fee_discount: bool = True
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_toml(cls, path: str) -> "ResearchConfig":
        with open(path, "rb") as f:
            d = tomllib.load(f)
        data, s, run, execution = d["data"], d["strategy"], d.get("run", {}), d.get("execution", {})
        accounting = d.get("accounting", {})
        known = dict(
            data_root=data["root"], cache_root=data.get("cache", ".lwbt_cache"),
            asset=data.get("asset", "BTC"), start_ns=to_ns(data["start"]), end_ns=to_ns(data["end"]),
            entry_weekday=int(s.get("entry_weekday", 4)),
            entry_frequency=str(s.get("entry_frequency", "weekly")),
            entry_hour_utc=int(s.get("entry_hour_utc", 8)),
            entry_minute_utc=int(s.get("entry_minute_utc", 0)), dte_min=float(s.get("dte_min", 6)),
            dte_max=float(s.get("dte_max", 8)), structure=s.get("structure", "short_straddle"),
            quantity=float(s.get("quantity", 1)), short_delta=float(s.get("short_delta", .30)),
            wing_delta=float(s.get("wing_delta", .10)), stop_loss_pct=s.get("stop_loss_pct"),
            profit_target_pct=s.get("profit_target_pct"),
            leg_profit_target_pct=s.get("leg_profit_target_pct"),
            survivor_profit_target_pct=s.get("survivor_profit_target_pct"),
            survivor_breakeven_stop=bool(s.get("survivor_breakeven_stop", False)),
            disable_structure_target_after_loss_stop=bool(s.get("disable_structure_target_after_loss_stop", False)),
            body_mode=str(s.get("body_mode", "delta")),
            body_otm_pct=float(s.get("body_otm_pct", .02)),
            exit_dte=float(s.get("exit_dte", 0)),
            expiry_weekday=(int(s["expiry_weekday"]) if "expiry_weekday" in s else None),
            min_open_interest=float(s.get("min_open_interest", 10)),
            max_relative_spread=float(s.get("max_relative_spread", .15)),
            initial_cash_native=float(run.get("initial_cash_native", 1)),
            slippage_bps=float(execution.get("slippage_bps", 1)),
            stop_fill_mode=str(execution.get("stop_fill_mode", "next_bar_touch")),
            theoretical_stop_slippage_pct=float(execution.get("theoretical_stop_slippage_pct", 0)),
            trade_data_root=execution.get("trade_data_root"),
            execution_source=str(execution.get("execution_source", "quotes")),
            snapshot_tolerance_minutes=int(data.get("snapshot_tolerance_minutes", 30)),
            trade_fee_rate=float(accounting.get("trade_fee_rate", .0003)),
            settlement_fee_rate=float(accounting.get("settlement_fee_rate", .00015)),
            fee_cap_rate=float(accounting.get("fee_cap_rate", .125)),
            enforce_margin=bool(accounting.get("enforce_margin", True)),
            combo_fee_discount=bool(accounting.get("combo_fee_discount", True)),
        )
        return cls(**known)
