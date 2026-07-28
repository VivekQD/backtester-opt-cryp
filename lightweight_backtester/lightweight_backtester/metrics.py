from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def _drawdown(values: np.ndarray):
    if values.size == 0:
        return 0.0, 0
    peak = np.maximum.accumulate(values)
    dd = np.divide(values - peak, peak, out=np.zeros_like(values), where=peak != 0)
    longest = current = 0
    for value in dd:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return float(dd.min()), longest


def _sharpe(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    returns = np.diff(values) / values[:-1]
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * math.sqrt(365))


def build_trade_ledger(legs: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for leg in legs:
        grouped[leg["trade_id"]].append(leg)
    out = []
    for trade_id, rows in grouped.items():
        pnl = sum(x["net_pnl_native"] for x in rows)
        defined_max_loss = _defined_max_loss(rows)
        out.append({"trade_id": trade_id, "entry_ts_ns": min(x["entry_ts_ns"] for x in rows),
                    "exit_ts_ns": max(x["exit_ts_ns"] for x in rows), "legs": len(rows),
                    "net_pnl_native": pnl, "fees_native": sum(x["fees_native"] for x in rows),
                    "win": pnl > 0, "all_expired": all(x["exit_reason"] == "expiry" for x in rows),
                    "defined_max_loss_native": defined_max_loss,
                    "return_on_defined_max_loss": (pnl / defined_max_loss
                                                   if defined_max_loss else None)})
    return sorted(out, key=lambda x: x["entry_ts_ns"])


def _defined_max_loss(rows: list[dict]):
    """Dense expiry-payoff diagnostic for structures containing long protection."""
    if not rows or not any(r["side"] == "long" for r in rows):
        return None
    strikes = np.asarray([r["strike"] for r in rows], dtype=float)
    grid = np.unique(np.r_[np.geomspace(max(strikes.min() * .02, 1e-6), strikes.max() * 4, 8000),
                           strikes])
    pnl = np.zeros_like(grid)
    entry_fees = 0.0
    for r in rows:
        signed = r["quantity"] if r["side"] == "long" else -r["quantity"]
        payoff = (np.maximum(grid - r["strike"], 0) / grid if r["option_type"] == "C"
                  else np.maximum(r["strike"] - grid, 0) / grid)
        pnl += signed * (payoff - r["entry_price"])
        entry_fees += r["quantity"] * min(.0003, .125 * r["entry_price"])
    return float(max(-pnl.min() + entry_fees, 0.0))


def compute_metrics(equity, leg_ledger, trade_ledger) -> dict:
    if not equity:
        return {}
    # One close per UTC day; timestamps are nanoseconds.
    daily = {}
    for point in equity:
        daily[point.ts_ns // 86_400_000_000_000] = point
    points = [daily[k] for k in sorted(daily)]
    native = np.asarray([p.native_equity for p in points], dtype=float)
    usd = np.asarray([p.usd_equity for p in points], dtype=float)
    option = np.asarray([p.option_pnl_native for p in points], dtype=float)
    initial_native = float(native[0] - option[0])
    first_spot = float(usd[0] / native[0]) if native[0] else 0.0
    initial_usd = initial_native * first_spot
    native_dd, native_duration = _drawdown(native)
    usd_dd, usd_duration = _drawdown(usd)
    wins = sum(t["win"] for t in trade_ledger)
    gains = sum(max(t["net_pnl_native"], 0) for t in trade_ledger)
    losses = -sum(min(t["net_pnl_native"], 0) for t in trade_ledger)
    return {
        "start_native_equity": initial_native, "end_native_equity": float(native[-1]),
        "total_option_pnl_native": float(option[-1]),
        "native_return": float(native[-1] / initial_native - 1) if initial_native else 0.0,
        "usd_return": float(usd[-1] / initial_usd - 1) if initial_usd else 0.0,
        "native_sharpe_daily": _sharpe(native), "usd_sharpe_daily": _sharpe(usd),
        "native_max_drawdown": native_dd, "usd_max_drawdown": usd_dd,
        "native_drawdown_duration_days": native_duration,
        "usd_drawdown_duration_days": usd_duration,
        "trades": len(trade_ledger), "option_legs": len(leg_ledger),
        "win_rate": wins / len(trade_ledger) if trade_ledger else 0.0,
        "profit_factor": gains / losses if losses else 0.0,
        "total_fees_native": sum(x["fees_native"] for x in leg_ledger),
        "peak_initial_margin_native": max(p.initial_margin_native for p in equity),
        "peak_maintenance_margin_native": max(p.maintenance_margin_native for p in equity),
    }
