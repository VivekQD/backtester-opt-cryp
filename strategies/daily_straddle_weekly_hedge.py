"""Sell 1-DTE ATM straddle daily, hedged with a long weekly Friday 0.20-delta strangle.

Weekly pattern (all times UTC):
  Fri 08:00 (first Friday)  INITIAL ENTRY. Buy next Friday's 0.20-delta call+put
                            (long strangle hedge). Sell a 1-DTE ATM straddle
                            expiring Sat 08:00.
  Fri 08:00 (thereafter)    Old hedge (Fri weekly) and old daily straddle settle
                            via the engine. Buy the *following* Friday's 0.20-delta
                            strangle. Sell a fresh 1-DTE ATM straddle.
  Sat-Thu 08:00             Previous day's 1-DTE straddle settled at open.
                            Sell a fresh 1-DTE ATM straddle. Hedge stays open.

Sizing: hedge qty = straddle qty = `qty` (1:1).
Straddle legs picked by nearest moneyness=1.0 on the ~1-DTE expiry.
Hedge legs picked by nearest |delta|=hedge_delta_abs on the target Friday weekly expiry.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from btcbt import Leg, OptType, Side, Strategy as BaseStrategy, combo
from btcbt.types import DAY_NS, NS


DEFAULT_PARAMS = {
    "entry_hour_utc": 8,
    "entry_window_mins": 60,
    "hedge_dow": 4,                    # Friday
    "hedge_delta_abs": 0.20,
    "hedge_target_dte_days": 7.0,
    "straddle_target_dte_days": 1.0,
    "qty": 1.0,
}


class DailyStraddleWeeklyHedge(BaseStrategy):
    def __init__(self, params: dict | None = None):
        merged = {**DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)
        self.entry_hour_utc = int(self.params["entry_hour_utc"])
        self.entry_window_mins = float(self.params["entry_window_mins"])
        self.hedge_dow = int(self.params["hedge_dow"])
        self.hedge_delta_abs = abs(float(self.params["hedge_delta_abs"]))
        self.hedge_target_dte_days = float(self.params["hedge_target_dte_days"])
        self.straddle_target_dte_days = float(self.params["straddle_target_dte_days"])
        self.qty = float(self.params["qty"])
        self._hedge: dict | None = None       # {"call", "put", "expiry"}
        self._straddle: dict | None = None    # {"call", "put", "expiry"}
        self._entered_days: set[dt.date] = set()

    def _as_utc(self, ts: int) -> dt.datetime:
        return dt.datetime.fromtimestamp(ts / NS, tz=dt.timezone.utc)

    def _in_entry_window(self, ts: int) -> bool:
        d = self._as_utc(ts)
        entry_open = d.replace(hour=self.entry_hour_utc, minute=0, second=0, microsecond=0)
        entry_close = entry_open + dt.timedelta(minutes=self.entry_window_mins)
        return entry_open <= d < entry_close

    def _tradable_delta_rows(self, snap, rows):
        return rows[
            np.isfinite(snap.delta[rows])
            & np.isfinite(snap.bid[rows])
            & (snap.bid[rows] > 0)
        ]

    def _pick_straddle_expiry(self, snap) -> int | None:
        """Nearest option expiry after now, closest to `straddle_target_dte_days`."""
        opt_mask = (snap.otype != int(OptType.NONE)) & (snap.expiry > snap.ts)
        expiries = np.unique(snap.expiry[opt_mask])
        if expiries.size == 0:
            return None
        target = snap.ts + int(self.straddle_target_dte_days * DAY_NS)
        return int(expiries[int(np.argmin(np.abs(expiries - target)))])

    def _pick_hedge_expiry(self, snap) -> int | None:
        """Nearest Friday-weekly expiry >= ~1 day out, closest to `hedge_target_dte_days`.

        The 1-day floor ensures we skip today's Friday expiry (which just settled) and
        pick the *following* Friday.
        """
        opt_mask = (snap.otype != int(OptType.NONE)) & (snap.expiry > snap.ts)
        expiries = np.unique(snap.expiry[opt_mask])
        if expiries.size == 0:
            return None
        keep = []
        for e in expiries:
            d = dt.datetime.fromtimestamp(int(e) / NS, tz=dt.timezone.utc)
            if d.weekday() == self.hedge_dow and d.hour == self.entry_hour_utc:
                keep.append(int(e))
        fridays = np.array(keep, dtype=np.int64)
        fridays = fridays[fridays >= snap.ts + DAY_NS]
        if fridays.size == 0:
            return None
        target = snap.ts + int(self.hedge_target_dte_days * DAY_NS)
        return int(fridays[int(np.argmin(np.abs(fridays - target)))])

    def _pick_straddle_pair(self, snap, expiry: int) -> tuple[int, int] | None:
        sel = snap.select().options().expiry(expiry).tradable()
        call = sel.calls().nearest("moneyness", 1.0).first()
        put = sel.puts().nearest("moneyness", 1.0).first()
        if call is None or put is None or call == put:
            return None
        return int(call), int(put)

    def _pick_hedge_pair(self, snap, expiry: int) -> tuple[int, int] | None:
        rows = snap.select().options().expiry(expiry).tradable().rows()
        if rows.size == 0:
            return None
        calls = self._tradable_delta_rows(snap, rows[snap.otype[rows] == int(OptType.CALL)])
        puts = self._tradable_delta_rows(snap, rows[snap.otype[rows] == int(OptType.PUT)])
        if calls.size == 0 or puts.size == 0:
            return None
        call_row = calls[int(np.argmin(np.abs(snap.delta[calls] - self.hedge_delta_abs)))]
        put_row = puts[int(np.argmin(np.abs(snap.delta[puts] + self.hedge_delta_abs)))]
        call = int(snap.sym[call_row])
        put = int(snap.sym[put_row])
        if call == put:
            return None
        return call, put

    def _clear_expired(self, ctx) -> None:
        if self._hedge is not None:
            c, p = self._hedge["call"], self._hedge["put"]
            if ctx.portfolio().net_qty(c) == 0 and ctx.portfolio().net_qty(p) == 0:
                self._hedge = None
        if self._straddle is not None:
            c, p = self._straddle["call"], self._straddle["put"]
            if ctx.portfolio().net_qty(c) == 0 and ctx.portfolio().net_qty(p) == 0:
                self._straddle = None

    def on_snapshot(self, snap, ctx):
        self._clear_expired(ctx)

        if not self._in_entry_window(snap.ts):
            return
        d = self._as_utc(snap.ts)
        day = d.date()
        if day in self._entered_days:
            return

        is_friday = (d.weekday() == self.hedge_dow)

        # Initial entry and every subsequent hedge-roll must land on a Friday.
        if self._hedge is None and not is_friday:
            return

        # 1) Buy hedge if missing (initial entry or old hedge just settled today).
        if self._hedge is None:
            hedge_expiry = self._pick_hedge_expiry(snap)
            if hedge_expiry is None:
                return
            hp = self._pick_hedge_pair(snap, hedge_expiry)
            if hp is None:
                return
            hc, hpu = hp
            ctx.submit_combo(combo([
                Leg(hc, Side.BUY, 1.0),
                Leg(hpu, Side.BUY, 1.0),
            ], qty=self.qty, atomic=True))
            self._hedge = {"call": hc, "put": hpu, "expiry": hedge_expiry}

        # 2) Sell fresh 1-DTE ATM straddle.
        if self._straddle is None:
            se = self._pick_straddle_expiry(snap)
            if se is None:
                return
            sp = self._pick_straddle_pair(snap, se)
            if sp is None:
                return
            sc, spu = sp
            ctx.submit_combo(combo([
                Leg(sc, Side.SELL, 1.0),
                Leg(spu, Side.SELL, 1.0),
            ], qty=self.qty, atomic=True))
            self._straddle = {"call": sc, "put": spu, "expiry": se}

        self._entered_days.add(day)

    def on_expiry(self, ev, ctx):
        del ev
        self._clear_expired(ctx)


Strategy = DailyStraddleWeeklyHedge
