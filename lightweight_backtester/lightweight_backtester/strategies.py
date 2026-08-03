from __future__ import annotations

import math
import datetime as dt

from .config import ResearchConfig
from .data import parse_contract
from .types import LegIntent, OptionType, Side, TradeIntent


def _number(row: dict, name: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(name, default))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


class OptionStructureStrategy:
    """Configurable selector for common short-premium and directional structures."""

    def __init__(self, cfg: ResearchConfig):
        self.cfg = cfg

    def _liquid(self, rows: list[dict]) -> list[dict]:
        out = []
        for r in rows:
            bid, ask, mark = (_number(r, "bid_price"), _number(r, "ask_price"),
                              _number(r, "mark_price"))
            oi = _number(r, "open_interest")
            if bid <= 0 or ask < bid or mark <= 0 or oi < self.cfg.min_open_interest:
                continue
            if (ask - bid) / mark > self.cfg.max_relative_spread:
                continue
            out.append(r)
        return out

    @staticmethod
    def _nearest(rows: list[dict], field: str, target: float, absolute: bool = False):
        if not rows:
            return None
        def distance(row):
            value = _number(row, field, float("nan"))
            if absolute:
                value = abs(value)
            return abs(value - target) if math.isfinite(value) else float("inf")
        return min(rows, key=distance)

    @staticmethod
    def _contract(row: dict):
        return parse_contract(row["path"])

    def _leg(self, row, side, stop=None):
        contract = self._contract(row)
        target = self.cfg.leg_profit_target_pct if side == Side.SELL else None
        return LegIntent(contract, side, self.cfg.quantity, stop, target) if contract else None

    def on_entry(self, target_ns: int, chain: list[dict]) -> TradeIntent | None:
        rows = self._liquid(chain)
        if self.cfg.expiry_weekday is not None:
            rows = [r for r in rows if dt.datetime.fromtimestamp(
                int(r["expiry_ns"]) / 1e9, dt.timezone.utc).weekday()
                == self.cfg.expiry_weekday]
        if not rows:
            return None
        expiry = min(int(r["expiry_ns"]) for r in rows)
        rows = [r for r in rows if int(r["expiry_ns"]) == expiry]
        calls = [r for r in rows if r["option_type"] == "C"]
        puts = [r for r in rows if r["option_type"] == "P"]
        structure = self.cfg.structure.lower()
        defined_risk = structure in ("iron_butterfly", "iron_condor",
                                     "bull_put_spread", "bear_call_spread")
        vertical_leg_stops = (defined_risk
                              and self.cfg.extra.get("defined_risk_stop_mode")
                              == "short_leg_vertical")
        short_stop = self.cfg.stop_loss_pct if (not defined_risk or vertical_leg_stops) else None
        selected: list[tuple[dict, Side, float | None]] = []

        if structure in ("short_straddle", "iron_butterfly"):
            common = sorted(set(_number(r, "strike") for r in calls)
                            & set(_number(r, "strike") for r in puts))
            if not common:
                return None
            spot = _number(rows[0], "index_price", _number(rows[0], "underlying_price"))
            strike = min(common, key=lambda x: abs(x - spot))
            c = next(r for r in calls if _number(r, "strike") == strike)
            p = next(r for r in puts if _number(r, "strike") == strike)
            selected = [(c, Side.SELL, short_stop),
                        (p, Side.SELL, short_stop)]
            if structure == "iron_butterfly":
                wc = self._nearest([r for r in calls if _number(r, "strike") > strike],
                                   "delta", self.cfg.wing_delta, True)
                wp = self._nearest([r for r in puts if _number(r, "strike") < strike],
                                   "delta", self.cfg.wing_delta, True)
                if wc is None or wp is None:
                    return None
                selected += [(wc, Side.BUY, None), (wp, Side.BUY, None)]
        elif structure in ("short_strangle", "iron_condor"):
            spot = _number(rows[0], "index_price", _number(rows[0], "underlying_price"))
            if self.cfg.body_mode == "moneyness":
                c = self._nearest(calls, "strike", spot * (1 + self.cfg.body_otm_pct))
                p = self._nearest(puts, "strike", spot * (1 - self.cfg.body_otm_pct))
            else:
                c = self._nearest(calls, "delta", self.cfg.short_delta, True)
                p = self._nearest(puts, "delta", self.cfg.short_delta, True)
            if c is None or p is None:
                return None
            selected = [(c, Side.SELL, short_stop),
                        (p, Side.SELL, short_stop)]
            if structure == "iron_condor":
                wc = self._nearest([r for r in calls if _number(r, "strike") > _number(c, "strike")],
                                   "delta", self.cfg.wing_delta, True)
                wp = self._nearest([r for r in puts if _number(r, "strike") < _number(p, "strike")],
                                   "delta", self.cfg.wing_delta, True)
                if wc is None or wp is None:
                    return None
                selected += [(wc, Side.BUY, None), (wp, Side.BUY, None)]
        elif structure in ("bull_put_spread", "bear_call_spread"):
            pool = puts if structure == "bull_put_spread" else calls
            short = self._nearest(pool, "delta", self.cfg.short_delta, True)
            if short is None:
                return None
            short_strike = _number(short, "strike")
            wings = ([r for r in puts if _number(r, "strike") < short_strike]
                     if structure == "bull_put_spread" else
                     [r for r in calls if _number(r, "strike") > short_strike])
            wing = self._nearest(wings, "delta", self.cfg.wing_delta, True)
            if wing is None:
                return None
            selected = [(short, Side.SELL, None), (wing, Side.BUY, None)]
        elif structure in ("short_call", "short_put"):
            pool = calls if structure == "short_call" else puts
            leg = self._nearest(pool, "delta", self.cfg.short_delta, True)
            if leg is None:
                return None
            selected = [(leg, Side.SELL, self.cfg.stop_loss_pct)]
        else:
            raise ValueError(f"Unsupported structure: {self.cfg.structure}")

        legs = tuple(self._leg(*item) for item in selected)
        if any(x is None for x in legs):
            return None
        trade_id = f"{self.cfg.asset}-{target_ns}-{structure}"
        exit_target = expiry - int(self.cfg.exit_dte * 86_400_000_000_000)
        return TradeIntent(trade_id, target_ns, legs, self.cfg.exit_dte <= 0,
                           structure, exit_target,
                           (self.cfg.stop_loss_pct
                            if defined_risk and not vertical_leg_stops else None),
                           self.cfg.profit_target_pct,
                           self.cfg.survivor_profit_target_pct,
                           self.cfg.survivor_breakeven_stop,
                           self.cfg.disable_structure_target_after_loss_stop)
