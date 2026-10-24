"""Weekly OTM butterflies (1x2x1), held to expiry. No TP / SL.

Every Friday at the Deribit roll (08:00 UTC) pick the expiry closest to
`target_dte_days` and, for each side in `sides`:

    call side (upside):   buy 1x OTM1 call, sell 2x OTM2 call, buy 1x OTM3 call
    put side (downside):  buy 1x OTM1 put,  sell 2x OTM2 put,  buy 1x OTM3 put

The position is left to cash-settle at expiry. "OTMn" means the n-th listed
strike away from spot (above for calls, below for puts) for that expiry, so
wing width follows the chain's grid. Each side is its own atomic combo.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from btcbt import Leg, OptType, Side, Strategy as BaseStrategy, combo
from btcbt.types import DAY_NS, NS


DEFAULT_PARAMS = {
    "entry_dow": 4,                # 0=Mon .. 4=Fri
    "entry_hour_utc": 8,
    "entry_window_mins": 60,
    "target_dte_days": 7.0,
    "otm_offsets": [1, 2, 3],      # strike steps away from ATM for buy / sell / buy
    "qty": 1.0,                    # lots of the 1x2x1 unit
    "sides": ["call"],             # "call" (upside), "put" (downside), or both
}


class WeeklyButterflyOTM(BaseStrategy):
    def __init__(self, params: dict | None = None):
        merged = {**DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)
        self.entry_dow = int(self.params["entry_dow"])
        self.entry_hour_utc = int(self.params["entry_hour_utc"])
        self.entry_window_mins = float(self.params["entry_window_mins"])
        self.target_dte_days = float(self.params["target_dte_days"])
        self.otm_offsets = [int(x) for x in self.params["otm_offsets"]]
        self.qty = float(self.params["qty"])
        self.sides = [str(x).lower() for x in self.params["sides"]]
        self._entered_weeks: set[tuple[int, int]] = set()

    def _as_utc(self, ts: int) -> dt.datetime:
        return dt.datetime.fromtimestamp(ts / NS, tz=dt.timezone.utc)

    def _in_entry_window(self, ts: int) -> bool:
        d = self._as_utc(ts)
        if d.weekday() != self.entry_dow:
            return False
        entry_open = d.replace(hour=self.entry_hour_utc, minute=0, second=0, microsecond=0)
        entry_close = entry_open + dt.timedelta(minutes=self.entry_window_mins)
        return entry_open <= d < entry_close

    def _pick_expiry(self, snap) -> int | None:
        opt_mask = (snap.otype != int(OptType.NONE)) & (snap.expiry > snap.ts)
        expiries = np.unique(snap.expiry[opt_mask])
        if expiries.size == 0:
            return None
        target = snap.ts + int(self.target_dte_days * DAY_NS)
        return int(expiries[int(np.argmin(np.abs(expiries - target)))])

    def _pick_legs(self, snap, expiry: int, side: str) -> tuple[int, int, int] | None:
        sel = snap.select().options().expiry(expiry)
        if side == "call":
            rows = sel.calls().rows()
            rows = rows[snap.strike[rows] > snap.index]
            rows = rows[np.argsort(snap.strike[rows], kind="stable")]
        else:
            rows = sel.puts().rows()
            rows = rows[snap.strike[rows] < snap.index]
            rows = rows[np.argsort(-snap.strike[rows], kind="stable")]
        if rows.size == 0:
            return None
        need = max(self.otm_offsets)
        if rows.size < need:
            return None
        lo, mid, hi = (rows[k - 1] for k in self.otm_offsets)
        # buys need an ask, the short needs a bid
        if not (snap.ask[lo] > 0 and snap.bid[mid] > 0 and snap.ask[hi] > 0):
            return None
        return int(snap.sym[lo]), int(snap.sym[mid]), int(snap.sym[hi])

    def on_snapshot(self, snap, ctx):
        if not ctx.portfolio().is_flat():
            return
        y, w, _ = self._as_utc(snap.ts).isocalendar()
        if (y, w) in self._entered_weeks or not self._in_entry_window(snap.ts):
            return
        expiry = self._pick_expiry(snap)
        if expiry is None:
            return
        flies = [legs for side in self.sides
                 if (legs := self._pick_legs(snap, expiry, side)) is not None]
        if not flies:
            return
        self._entered_weeks.add((y, w))
        for near, mid, far in flies:
            ctx.submit_combo(combo([
                Leg(near, Side.BUY, 1.0),
                Leg(mid, Side.SELL, 2.0),
                Leg(far, Side.BUY, 1.0),
            ], qty=self.qty, atomic=True))


WeeklyCallButterflyOTM = WeeklyButterflyOTM   # backwards-compatible name

Strategy = WeeklyButterflyOTM
