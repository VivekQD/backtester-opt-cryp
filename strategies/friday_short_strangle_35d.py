"""Short 35-delta strangle every Friday, target 2-DTE expiry, exit Monday morning.

Weekly pattern (all times UTC):
  Fri 08:00  sell one 35-delta call and one 35-delta put on the ~2-DTE expiry
  Sun 08:00  Deribit settlement closes the position via on_expiry
  Mon 08:00  fallback close for anything still open (rare with 2-DTE)

One entry per ISO week; controlled by `entry_dow` / `entry_hour_utc` / `entry_window_mins`.
Legs picked by |delta| closest to `target_delta_abs` on tradable rows of the chosen expiry.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from btcbt import Leg, OptType, Side, Strategy as BaseStrategy, combo
from btcbt.types import DAY_NS, NS


DEFAULT_PARAMS = {
    "entry_dow": 4,             # 0=Mon .. 4=Fri
    "entry_hour_utc": 8,        # Deribit daily roll
    "entry_window_mins": 60,    # bars after entry_hour still eligible for entry
    "target_dte_days": 2.0,     # pick expiry closest to this many days from entry
    "exit_dow": 0,              # 0=Mon
    "exit_hour_utc": 8,
    "target_delta_abs": 0.35,
    "qty": 1.0,
    # Delta risk rules:
    "max_entry_delta_dev": 0.10,   # pre-entry: skip week if |selected Δ| deviates > this from target
    "leg_delta_stop_abs": 0.70,    # in-trade: buy back a leg when |Δ| >= this
}


class FridayShortStrangle35d(BaseStrategy):
    def __init__(self, params: dict | None = None):
        merged = {**DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)
        self.entry_dow = int(self.params["entry_dow"])
        self.entry_hour_utc = int(self.params["entry_hour_utc"])
        self.entry_window_mins = float(self.params["entry_window_mins"])
        self.target_dte_days = float(self.params["target_dte_days"])
        self.exit_dow = int(self.params["exit_dow"])
        self.exit_hour_utc = int(self.params["exit_hour_utc"])
        self.target_delta_abs = abs(float(self.params["target_delta_abs"]))
        self.qty = float(self.params["qty"])
        self.max_entry_delta_dev = float(self.params["max_entry_delta_dev"])
        self.leg_delta_stop_abs = float(self.params["leg_delta_stop_abs"])
        self._entered_weeks: set[tuple[int, int]] = set()
        self._active: dict | None = None

    def _as_utc(self, ts: int) -> dt.datetime:
        return dt.datetime.fromtimestamp(ts / NS, tz=dt.timezone.utc)

    def _week_key(self, ts: int) -> tuple[int, int]:
        y, w, _ = self._as_utc(ts).isocalendar()
        return (y, w)

    def _in_entry_window(self, ts: int) -> bool:
        d = self._as_utc(ts)
        if d.weekday() != self.entry_dow:
            return False
        entry_open = d.replace(hour=self.entry_hour_utc, minute=0, second=0, microsecond=0)
        entry_close = entry_open + dt.timedelta(minutes=self.entry_window_mins)
        return entry_open <= d < entry_close

    def _next_exit_ts(self, entry_ts: int) -> int:
        d = self._as_utc(entry_ts)
        days_forward = (self.exit_dow - d.weekday()) % 7
        if days_forward == 0:
            days_forward = 7
        target = d + dt.timedelta(days=days_forward)
        target = target.replace(hour=self.exit_hour_utc, minute=0, second=0, microsecond=0)
        return int(target.timestamp() * NS)

    def _pick_expiry(self, snap) -> int | None:
        opt_mask = (snap.otype != int(OptType.NONE)) & (snap.expiry > snap.ts)
        expiries = np.unique(snap.expiry[opt_mask])
        if expiries.size == 0:
            return None
        target = snap.ts + int(self.target_dte_days * DAY_NS)
        return int(expiries[int(np.argmin(np.abs(expiries - target)))])

    def _tradable_delta_rows(self, snap, rows):
        return rows[
            np.isfinite(snap.delta[rows])
            & np.isfinite(snap.bid[rows])
            & (snap.bid[rows] > 0)
        ]

    def _pick_pair(self, snap, expiry: int) -> tuple[int, int] | None:
        rows = snap.select().options().expiry(expiry).tradable().rows()
        if rows.size == 0:
            return None
        calls = self._tradable_delta_rows(snap, rows[snap.otype[rows] == int(OptType.CALL)])
        puts = self._tradable_delta_rows(snap, rows[snap.otype[rows] == int(OptType.PUT)])
        if calls.size == 0 or puts.size == 0:
            return None
        call_row = calls[int(np.argmin(np.abs(snap.delta[calls] - self.target_delta_abs)))]
        put_row = puts[int(np.argmin(np.abs(snap.delta[puts] + self.target_delta_abs)))]
        # pre-entry sanity check: |selected Δ| must be within tolerance of target.
        # Kills stale/thin-quote weeks where the "closest to 35Δ" pick is far off (e.g. 68Δ).
        if self.max_entry_delta_dev > 0:
            call_dev = abs(abs(float(snap.delta[call_row])) - self.target_delta_abs)
            put_dev = abs(abs(float(snap.delta[put_row])) - self.target_delta_abs)
            if call_dev > self.max_entry_delta_dev or put_dev > self.max_entry_delta_dev:
                return None
        call_sym = int(snap.sym[call_row])
        put_sym = int(snap.sym[put_row])
        if call_sym == put_sym:
            return None
        return call_sym, put_sym

    def _check_leg_delta_stops(self, snap, ctx) -> None:
        if self._active is None or self.leg_delta_stop_abs <= 0:
            return
        stopped = self._active.setdefault("stopped", set())
        for sym in (self._active["call"], self._active["put"]):
            if sym in stopped:
                continue
            pos = ctx.portfolio().position(sym)
            if not pos or pos.qty >= 0:
                continue
            row = snap.row(sym)
            if row is None or not bool(snap.valid[row]):
                continue
            d = float(snap.delta[row])
            if not np.isfinite(d):
                continue
            if abs(d) >= self.leg_delta_stop_abs:
                ctx.close(sym)
                stopped.add(sym)

    def _maybe_clear_active(self, ctx) -> None:
        if self._active is None:
            return
        call, put = self._active["call"], self._active["put"]
        if ctx.portfolio().net_qty(call) == 0 and ctx.portfolio().net_qty(put) == 0:
            self._active = None

    def _submit_close(self, ctx) -> None:
        if self._active is None:
            return
        call, put = self._active["call"], self._active["put"]
        cpos = ctx.portfolio().position(call)
        ppos = ctx.portfolio().position(put)
        if (
            cpos and ppos
            and cpos.qty < 0 and ppos.qty < 0
            and np.isclose(abs(cpos.qty), abs(ppos.qty))
        ):
            ctx.submit_combo(combo([
                Leg(call, Side.BUY, 1.0),
                Leg(put, Side.BUY, 1.0),
            ], qty=abs(cpos.qty), atomic=True))
            return
        if cpos and cpos.qty != 0:
            ctx.close(call)
        if ppos and ppos.qty != 0:
            ctx.close(put)

    def on_snapshot(self, snap, ctx):
        self._maybe_clear_active(ctx)

        if self._active is not None:
            if snap.ts >= self._active["exit_ts"]:
                self._submit_close(ctx)
                return
            self._check_leg_delta_stops(snap, ctx)
            return

        if not ctx.portfolio().is_flat():
            return
        wk = self._week_key(snap.ts)
        if wk in self._entered_weeks:
            return
        if not self._in_entry_window(snap.ts):
            return

        expiry = self._pick_expiry(snap)
        if expiry is None:
            return
        pair = self._pick_pair(snap, expiry)
        if pair is None:
            return
        call, put = pair
        self._active = {
            "week": wk,
            "expiry": expiry,
            "call": call,
            "put": put,
            "exit_ts": self._next_exit_ts(snap.ts),
        }
        self._entered_weeks.add(wk)
        ctx.submit_combo(combo([
            Leg(call, Side.SELL, 1.0),
            Leg(put, Side.SELL, 1.0),
        ], qty=self.qty, atomic=True))

    def on_fill(self, fill, ctx):
        if fill.side == Side.BUY:
            self._maybe_clear_active(ctx)

    def on_expiry(self, ev, ctx):
        self._maybe_clear_active(ctx)


Strategy = FridayShortStrangle35d
