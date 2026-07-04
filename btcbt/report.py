"""Research output: tidy series, optional plots, CSV/Parquet export. Phase 2.

No hard plotting dependency — matplotlib is imported lazily inside plot_* only.
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import re

import numpy as np

from .types import NS, Side


def report(res) -> dict:
    """All result series as plain numpy arrays, ready for a notebook."""
    eq = res.equity
    ts = np.array([t for t, _ in eq], dtype=np.int64)
    usd = np.array([e.usd for _, e in eq])
    btc = np.array([e.btc for _, e in eq])
    dd = np.array([e.dd for _, e in eq])
    g = res.greeks
    out = {
        "ts": ts,
        "equity_usd": usd,
        "equity_btc": btc,
        "drawdown": dd,
        "greeks_ts": np.array([t for t, _ in g], dtype=np.int64),
        "delta": np.array([x.delta for _, x in g]),
        "gamma": np.array([x.gamma for _, x in g]),
        "vega": np.array([x.vega for _, x in g]),
        "theta": np.array([x.theta for _, x in g]),
        "trade_pnl": np.array([t.pnl for t in res.trades]),
        "trade_exit_ts": np.array([t.exit_ts for t in res.trades], dtype=np.int64),
    }
    for key, series in res.records.items():
        out[f"rec:{key}"] = np.array([v for _, v in series])
    return out


def _daily(ts: np.ndarray, usd: np.ndarray):
    from .types import DAY_NS
    day = ts // DAY_NS
    last = {}
    for d, v in zip(day, usd):
        last[int(d)] = v
    days = sorted(last)
    vals = np.array([last[k] for k in days])
    pnl = np.diff(vals) if vals.size > 1 else np.zeros(0)
    return np.array(days[1:]), pnl


def _iso_utc(ts: int) -> str:
    if not ts:
        return ""
    return dt.datetime.fromtimestamp(int(ts) / NS, dt.timezone.utc).isoformat()


def _enum_name(value) -> str:
    return value.name if hasattr(value, "name") else str(value)


def _strategy_file_name(strategy_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", strategy_name.strip())
    return f"{safe.strip('_') or 'strategy'}.csv"


def export_trades_csv(res, metas: dict, strategy_name: str, out_dir: str = "trades",
                      venue=None) -> str:
    """Write a detailed matched-trade CSV and return its path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, _strategy_file_name(strategy_name))
    contract_size = float(getattr(venue, "contract_size", 1.0)) if venue is not None else 1.0
    margin = getattr(venue, "margin", None)
    fields = [
        "trade_id", "strategy", "sym", "instrument_name", "instrument_kind",
        "option_type", "currency", "margin", "strike", "expiry_ts", "expiry_utc",
        "side", "qty", "contract_size", "entry_price", "exit_price",
        "entry_spot", "exit_spot", "gross_pnl", "fees", "net_pnl",
        "entry_ts", "entry_utc", "exit_ts", "exit_utc",
        "holding_minutes", "minutes_to_expiry_at_entry",
        "settled_at_expiry",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, t in enumerate(res.trades, start=1):
            meta = metas.get(t.sym)
            side = getattr(t, "side", Side.BUY)
            dirn = 1.0 if side == Side.BUY else -1.0
            expiry_ts = int(getattr(meta, "expiry_ts", 0) or 0) if meta is not None else 0
            gross_pnl = float(t.qty) * (float(t.exit) - float(t.entry)) * dirn * contract_size
            holding_minutes = (int(t.exit_ts) - int(t.entry_ts)) / (60 * NS)
            entry_tte = (expiry_ts - int(t.entry_ts)) / (60 * NS) if expiry_ts else ""
            settled = bool(expiry_ts and int(t.exit_ts) >= expiry_ts)
            w.writerow({
                "trade_id": i,
                "strategy": strategy_name,
                "sym": t.sym,
                "instrument_name": getattr(meta, "name", ""),
                "instrument_kind": _enum_name(getattr(meta, "kind", "")),
                "option_type": _enum_name(getattr(meta, "otype", "")),
                "currency": getattr(meta, "currency", ""),
                "margin": _enum_name(margin if margin is not None else getattr(meta, "margin", "")),
                "strike": getattr(meta, "strike", ""),
                "expiry_ts": expiry_ts or "",
                "expiry_utc": _iso_utc(expiry_ts),
                "side": _enum_name(side),
                "qty": t.qty,
                "contract_size": contract_size,
                "entry_price": t.entry,
                "exit_price": t.exit,
                "entry_spot": getattr(t, "entry_index", 0.0),
                "exit_spot": getattr(t, "exit_index", 0.0),
                "gross_pnl": gross_pnl,
                "fees": t.fees,
                "net_pnl": t.pnl,
                "entry_ts": t.entry_ts,
                "entry_utc": _iso_utc(t.entry_ts),
                "exit_ts": t.exit_ts,
                "exit_utc": _iso_utc(t.exit_ts),
                "holding_minutes": holding_minutes,
                "minutes_to_expiry_at_entry": entry_tte,
                "settled_at_expiry": settled,
            })
    return path


# ---- export ---------------------------------------------------------------
def export_result(res, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "equity.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "equity_usd", "equity_btc", "drawdown"])
        for t, e in res.equity:
            w.writerow([t, e.usd, e.btc, e.dd])
    with open(os.path.join(out_dir, "trades.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sym", "qty", "entry", "exit", "pnl", "fees", "entry_ts", "exit_ts"])
        for t in res.trades:
            w.writerow([t.sym, t.qty, t.entry, t.exit, t.pnl, t.fees, t.entry_ts, t.exit_ts])


# ---- plots (lazy matplotlib) ----------------------------------------------
def _ax(ax, title):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    ax.set_title(title)
    return ax


def plot_equity(res, ax=None):
    r = report(res)
    ax = _ax(ax, "Equity (USD)")
    ax.plot(r["ts"], r["equity_usd"])
    return ax


def plot_drawdown(res, ax=None):
    r = report(res)
    ax = _ax(ax, "Drawdown")
    ax.fill_between(r["ts"], r["drawdown"], 0.0, step="pre")
    return ax


def plot_trade_pnl(res, ax=None):
    r = report(res)
    ax = _ax(ax, "Trade PnL")
    ax.scatter(r["trade_exit_ts"], r["trade_pnl"], s=8)
    return ax


def plot_pnl_hist(res, ax=None):
    r = report(res)
    ax = _ax(ax, "PnL distribution")
    if r["trade_pnl"].size:
        ax.hist(r["trade_pnl"], bins=30)
    return ax


def plot_daily_pnl(res, ax=None):
    r = report(res)
    days, pnl = _daily(r["ts"], r["equity_usd"])
    ax = _ax(ax, "Daily PnL")
    ax.bar(days, pnl)
    return ax


def plot_greeks(res, ax=None):
    r = report(res)
    ax = _ax(ax, "Portfolio greeks")
    for k in ("delta", "gamma", "vega", "theta"):
        ax.plot(r["greeks_ts"], r[k], label=k)
    ax.legend()
    return ax
