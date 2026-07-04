"""0DTE short ATM straddle held until expiry.

The tunable defaults live in this file on purpose. TOML strategy params can
still override them, but this file is self-contained for the common run.

Deribit BTC daily options expire at 08:00 UTC. New daily expiries are listed
around that same 08:00 UTC roll, but a same-calendar-date 0DTE trade must enter
before that expiry moment. The default therefore uses 00:00 UTC, the first
same-date 0DTE bar, and lets the engine settle at the 08:00 UTC expiry.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np

from btcbt import Leg, OptType, Side, Strategy as BaseStrategy, combo
from btcbt.types import NS


DEFAULT_PARAMS = {
    # For Deribit 0DTE, same-date expiry runs from 00:00 UTC to 08:00 UTC.
    # Using 08:00 here would be the expiry/settlement moment, not an entry.
    "entry_time": "00:00",
    "timezone": "UTC",
    "entry_window_mins": 5,

    # One short ATM call + one short ATM put.
    "qty": 1.0,

    "min_minutes_to_expiry": 1,

    # Optional ISO datetimes, e.g. "2026-06-12T00:00:00+00:00".
    "start_datetime": None,
    "end_datetime": None,
}


def _hm(value: str) -> tuple[int, int]:
    h, _, m = str(value).partition(":")
    return int(h), int(m or 0)


def _parse_dt(value, tz: ZoneInfo) -> int | None:
    if value in (None, ""):
        return None
    out = dt.datetime.fromisoformat(str(value))
    if out.tzinfo is None:
        out = out.replace(tzinfo=tz)
    return int(out.timestamp() * NS)


class ZeroDteShortStraddle(BaseStrategy):
    """Sell today's ATM straddle and hold both legs until expiry."""

    def __init__(self, params: dict | None = None):
        merged = {**DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)
        self.tz = ZoneInfo(str(self.params["timezone"]))
        self.entry_hm = _hm(self.params["entry_time"])
        self.entry_window_mins = float(self.params["entry_window_mins"])
        self.qty = float(self.params["qty"])
        self.min_minutes_to_expiry = float(self.params["min_minutes_to_expiry"])
        self.start_ns = _parse_dt(self.params.get("start_datetime"), self.tz)
        self.end_ns = _parse_dt(self.params.get("end_datetime"), self.tz)
        self._entered_expiries: set[int] = set()

    def _local(self, ts: int) -> dt.datetime:
        return dt.datetime.fromtimestamp(ts / NS, self.tz)

    def _within_date_bounds(self, ts: int) -> bool:
        if self.start_ns is not None and ts < self.start_ns:
            return False
        if self.end_ns is not None and ts > self.end_ns:
            return False
        return True

    def _entry_window_open(self, ts: int, expiry: int) -> bool:
        now = self._local(ts)
        exp = self._local(expiry)
        if now.date() != exp.date():
            return False
        entry = dt.datetime.combine(exp.date(), dt.time(*self.entry_hm), tzinfo=self.tz)
        if now < entry:
            return False
        if self.entry_window_mins >= 0:
            delay_mins = (now - entry).total_seconds() / 60.0
            if delay_mins > self.entry_window_mins:
                return False
        mins_to_expiry = (expiry - ts) / (60 * NS)
        return mins_to_expiry >= self.min_minutes_to_expiry

    def _candidate_expiry(self, snap) -> int | None:
        if not self._within_date_bounds(snap.ts):
            return None
        option_rows = np.flatnonzero(
            (snap.otype != int(OptType.NONE)) & (snap.expiry > snap.ts)
        )
        for expiry in sorted(int(x) for x in np.unique(snap.expiry[option_rows])):
            if expiry in self._entered_expiries:
                continue
            if self._entry_window_open(snap.ts, expiry):
                return expiry
        return None

    def _atm_pair(self, snap, expiry: int) -> tuple[int, int] | None:
        rows = snap.select().options().expiry(expiry).tradable().rows()
        if rows.size == 0:
            return None

        calls = rows[snap.otype[rows] == int(OptType.CALL)]
        puts = rows[snap.otype[rows] == int(OptType.PUT)]
        if calls.size == 0 or puts.size == 0:
            return None

        common_strikes = np.intersect1d(snap.strike[calls], snap.strike[puts])
        if common_strikes.size == 0:
            return None

        strike = float(common_strikes[np.argmin(np.abs(common_strikes - snap.index))])
        call_matches = calls[np.isclose(snap.strike[calls], strike)]
        put_matches = puts[np.isclose(snap.strike[puts], strike)]
        if call_matches.size == 0 or put_matches.size == 0:
            return None
        return int(snap.sym[call_matches[0]]), int(snap.sym[put_matches[0]])

    def on_snapshot(self, snap, ctx):
        if not ctx.portfolio().is_flat():
            return

        expiry = self._candidate_expiry(snap)
        if expiry is None:
            return

        pair = self._atm_pair(snap, expiry)
        if pair is None:
            self._entered_expiries.add(expiry)
            return

        call, put = pair
        ctx.submit_combo(combo([
            Leg(call, Side.SELL, 1.0),
            Leg(put, Side.SELL, 1.0),
        ], qty=self.qty, atomic=True))
        self._entered_expiries.add(expiry)


# Allows: python -m btcbt.run --strategy strategies.zero_dte_iv_straddle:Strategy ...
ZeroDteIvStraddle = ZeroDteShortStraddle
Strategy = ZeroDteShortStraddle
