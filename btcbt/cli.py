"""Config loading + RunConfig assembly shared by run.py / sweep.py. Phase 1."""
from __future__ import annotations

import datetime as dt
import importlib
import tomllib

from .data import DataConfig
from .execution import DefaultFill
from .margin import DeribitStandardMargin, SimpleMargin
from .runner import RunConfig
from .types import NS, Margin
from .venue import VenueConfig


def load_toml(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve_strategy(spec: str) -> type:
    """'strategies.short_put:ShortPut' -> the class."""
    mod, _, cls = spec.partition(":")
    return getattr(importlib.import_module(mod), cls)


def _to_ns(v) -> int | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    d = dt.datetime.fromisoformat(str(v))
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp()) * NS


def build_venue(d: dict) -> VenueConfig:
    v = d.get("venue", {})
    margin = Margin.LINEAR if str(v.get("margin", "inverse")).lower() == "linear" else Margin.INVERSE
    return VenueConfig(
        margin=margin,
        contract_size=float(v.get("contract_size", 1.0)),
        option_trade_fee_rate=float(v.get("option_trade_fee_rate",
                                          v.get("trade_fee_per_contract", 0.0003))),
        fee_cap_rate=float(v.get("fee_cap_rate", 0.125)),
        option_settle_fee_rate=float(v.get("option_settle_fee_rate",
                                           v.get("settle_fee_per_contract", 0.00015))),
        daily_option_settle_fee_rate=float(v.get("daily_option_settle_fee_rate", 0.0)),
        twap_mins=int(v.get("twap_mins", 30)),
    )


def build_fill(d: dict) -> DefaultFill:
    f = d.get("fill", {})
    return DefaultFill(
        half_spread_bps=float(f.get("half_spread_bps", 5.0)),
        slippage_bps=float(f.get("slippage_bps", 1.0)),
        impact_bps_per_oi=float(f.get("impact_bps_per_oi", 0.0)),
    )


def build_data(d: dict) -> DataConfig:
    dd = d.get("data", {})
    return DataConfig(path=dd["path"], start=_to_ns(dd.get("start")),
                      end=_to_ns(dd.get("end")), hedge=dd.get("hedge"))


def build_margin(d: dict):
    m = d.get("margin")
    if not m or not m.get("enabled", False):
        return None
    model = str(m.get("model", m.get("type", "deribit"))).lower()
    if model in ("simple", "legacy"):
        return SimpleMargin(
            short_rate=float(m.get("short_rate", 0.15)),
            long_rate=float(m.get("long_rate", 0.0)),
            maint_frac=float(m.get("maint_frac", 0.6)),
        )
    return DeribitStandardMargin(
        initial_rate=float(m.get("initial_rate", m.get("short_rate", 0.15))),
        initial_floor=float(m.get("initial_floor", 0.10)),
        maintenance_rate=float(m.get("maintenance_rate", 0.075)),
        non_option_rate=float(m.get("non_option_rate", 0.0)),
        non_option_maint_frac=float(m.get("non_option_maint_frac", 1.0)),
    )


def build_runconfig(d: dict, strategy: type) -> RunConfig:
    return RunConfig(
        data=build_data(d),
        venue=build_venue(d),
        fill=build_fill(d),
        strategy=strategy,
        params=dict(d.get("strategy", {})),
        cash=float(d.get("run", {}).get("cash", 1.0)),
        seed=int(d.get("run", {}).get("seed", 0)),
        margin=build_margin(d),
    )


def print_stats(res) -> None:
    s = res.stats
    o, w, l, r, c = s.overview, s.winning, s.losing, s.ratios, s.costs
    print("=" * 52)
    print(f"  total_pnl   {o.total_pnl:+.6f}")
    print(f"  max_dd      {o.max_dd:.2%}")
    print(f"  trades      {o.trades}")
    print(f"  win_rate    {o.win_rate:.2%}   avg_profit {o.avg_profit:+.6f}   avg_loss {o.avg_loss:+.6f}")
    print(f"  winning     streak {w.max_streak} ({w.max_streak_pnl:+.4f})  max_day {w.max_day:+.4f}")
    print(f"  losing      streak {l.max_streak} ({l.max_streak_pnl:+.4f})  max_day {l.max_day:+.4f}")
    print(f"  sharpe {r.sharpe:.2f}  sortino {r.sortino:.2f}  calmar {r.calmar:.2f}")
    print(f"  expectancy {r.expectancy:+.6f}  profit_factor {r.profit_factor:.2f}  risk_reward {r.risk_reward:.2f}")
    print(f"  fees        per_trade {c.fee_per_trade:.6f}   total {c.total_fees:.6f}")
    print("=" * 52)
