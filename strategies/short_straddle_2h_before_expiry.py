"""Short the ATM straddle about two hours before expiry with risk controls."""
from __future__ import annotations

import datetime as dt

import numpy as np

from btcbt import Leg, OptType, Side, Strategy as BaseStrategy, combo, triggered
from btcbt.types import NS


DEFAULT_PARAMS = {
    # Orders submitted on this bar fill on the next bar in the engine.
    "entry_minutes_before_expiry": 120,
    "entry_window_mins": 5,
    "min_minutes_to_expiry": 1,
    "qty": 1.0,

    # Risk controls, all measured per one straddle contract.
    "min_total_credit": 200.0,
    "take_profit_pct": 0.50,                 # close when buyback <= 50% of entry credit
    "stop_loss_mult": 2.0,                   # close when buyback >= 2x entry credit
    "leg_take_profit_pct": 0.40,             # close leg when buyback <= 60% of leg entry
    "leg_stop_loss_pct": 0.20,               # close leg when buyback >= 120% of leg entry
    "force_close_minutes_before_expiry": 15,

    # Optional ISO datetimes, e.g. "2026-06-12T00:00:00+00:00".
    "start_datetime": None,
    "end_datetime": None,
}


def _parse_dt(value) -> int | None:
    if value in (None, ""):
        return None
    out = dt.datetime.fromisoformat(str(value))
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return int(out.timestamp() * NS)


def _finite_pos(x) -> bool:
    return bool(np.isfinite(x) and x > 0)


class ShortStraddleTwoHoursBeforeExpiry(BaseStrategy):
    """Sell the common-strike ATM call and put once per expiry."""

    def __init__(self, params: dict | None = None):
        merged = {**DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)
        self.entry_minutes_before_expiry = float(self.params["entry_minutes_before_expiry"])
        self.entry_window_mins = float(self.params["entry_window_mins"])
        self.min_minutes_to_expiry = float(self.params["min_minutes_to_expiry"])
        self.qty = float(self.params["qty"])
        self.min_total_credit = float(self.params["min_total_credit"])
        self.take_profit_pct = float(self.params["take_profit_pct"])
        self.stop_loss_mult = float(self.params["stop_loss_mult"])
        self.leg_take_profit_pct = float(self.params["leg_take_profit_pct"])
        self.leg_stop_loss_pct = float(self.params["leg_stop_loss_pct"])
        self.force_close_minutes_before_expiry = float(
            self.params["force_close_minutes_before_expiry"]
        )
        self.start_ns = _parse_dt(self.params.get("start_datetime"))
        self.end_ns = _parse_dt(self.params.get("end_datetime"))
        self._entered_expiries: set[int] = set()
        self._sym_expiry: dict[int, int] = {}
        self._expiry_pair: dict[int, tuple[int, int]] = {}
        self._active: dict | None = None

    def _within_date_bounds(self, ts: int) -> bool:
        if self.start_ns is not None and ts < self.start_ns:
            return False
        if self.end_ns is not None and ts > self.end_ns:
            return False
        return True

    def _entry_window_open(self, ts: int, expiry: int) -> bool:
        mins_to_expiry = (expiry - ts) / (60 * NS)
        if mins_to_expiry < self.min_minutes_to_expiry:
            return False
        if mins_to_expiry > self.entry_minutes_before_expiry:
            return False
        if self.entry_window_mins >= 0:
            earliest = self.entry_minutes_before_expiry - self.entry_window_mins
            if mins_to_expiry < earliest:
                return False
        return True

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

    def _sell_credit(self, snap, sym: int) -> float | None:
        row = snap.row(sym)
        if row is None:
            return None
        bid = float(snap.bid[row])
        return bid if _finite_pos(bid) else None

    def _buyback_price(self, snap, sym: int) -> float | None:
        row = snap.row(sym)
        if row is None:
            return None
        ask = float(snap.ask[row])
        if _finite_pos(ask):
            return ask
        mark = float(snap.mark[row])
        return mark if _finite_pos(mark) else None

    def _entry_credit(self, snap, call: int, put: int) -> float | None:
        call_credit = self._sell_credit(snap, call)
        put_credit = self._sell_credit(snap, put)
        if call_credit is None or put_credit is None:
            return None
        return call_credit + put_credit

    def _buyback_value(self, snap, call: int, put: int) -> float | None:
        call_debit = self._buyback_price(snap, call)
        put_debit = self._buyback_price(snap, put)
        if call_debit is None or put_debit is None:
            return None
        return call_debit + put_debit

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
        if cpos and ppos and cpos.qty < 0 and ppos.qty < 0 and np.isclose(abs(cpos.qty), abs(ppos.qty)):
            ctx.submit_combo(combo([
                Leg(call, Side.BUY, 1.0),
                Leg(put, Side.BUY, 1.0),
            ], qty=abs(cpos.qty), atomic=True))
            return
        if cpos and cpos.qty != 0:
            ctx.close(call)
        if ppos and ppos.qty != 0:
            ctx.close(put)

    def _submit_close_leg(self, ctx, sym: int) -> None:
        pos = ctx.portfolio().position(sym)
        if pos and pos.qty != 0:
            ctx.close(sym)

    def _submit_triggered_close_leg(self, ctx, sym: int, px: float) -> None:
        pos = ctx.portfolio().position(sym)
        if pos and pos.qty < 0 and _finite_pos(px):
            ctx.submit(triggered(sym, Side.BUY, abs(pos.qty), px))

    def _both_legs_open_short(self, ctx) -> bool:
        if self._active is None:
            return False
        cpos = ctx.portfolio().position(self._active["call"])
        ppos = ctx.portfolio().position(self._active["put"])
        return bool(cpos and ppos and cpos.qty < 0 and ppos.qty < 0)

    def _manage_leg_exits(self, snap, ctx) -> bool:
        if self._active is None:
            return False
        did_trigger = False
        entry_prices = self._active.get("entry_prices", {})
        for sym in (self._active["call"], self._active["put"]):
            pos = ctx.portfolio().position(sym)
            if not pos or pos.qty >= 0:
                continue
            entry = float(entry_prices.get(sym, 0.0))
            if entry <= 0:
                continue
            buyback = self._buyback_price(snap, sym)
            if buyback is None:
                continue
            take_profit_px = entry * (1.0 - self.leg_take_profit_pct)
            stop_loss_px = entry * (1.0 + self.leg_stop_loss_pct)
            if buyback <= take_profit_px:
                self._submit_triggered_close_leg(ctx, sym, take_profit_px)
                did_trigger = True
            elif buyback >= stop_loss_px:
                self._submit_triggered_close_leg(ctx, sym, stop_loss_px)
                did_trigger = True
        return did_trigger

    def _manage_active(self, snap, ctx) -> None:
        if self._active is None:
            return

        expiry = self._active["expiry"]
        mins_to_expiry = (expiry - snap.ts) / (60 * NS)
        if 0 < mins_to_expiry <= self.force_close_minutes_before_expiry:
            self._submit_close(ctx)
            return

        entry_credit = self._active.get("entry_credit", 0.0)
        if entry_credit <= 0:
            return
        if self._manage_leg_exits(snap, ctx):
            return
        if self._active.get("under_min_credit", False):
            self._submit_close(ctx)
            return
        if not self._both_legs_open_short(ctx):
            return
        buyback = self._buyback_value(snap, self._active["call"], self._active["put"])
        if buyback is None:
            return
        if buyback <= entry_credit * (1.0 - self.take_profit_pct):
            self._submit_close(ctx)
            return
        if buyback >= entry_credit * self.stop_loss_mult:
            self._submit_close(ctx)

    def on_snapshot(self, snap, ctx):
        self._maybe_clear_active(ctx)
        if self._active is not None and not ctx.portfolio().is_flat():
            self._manage_active(snap, ctx)
            return
        if not ctx.portfolio().is_flat():
            return

        expiry = self._candidate_expiry(snap)
        if expiry is None:
            return

        pair = self._atm_pair(snap, expiry)
        if pair is None:
            return

        call, put = pair
        entry_credit = self._entry_credit(snap, call, put)
        if entry_credit is None or entry_credit < self.min_total_credit:
            return
        self._sym_expiry[call] = expiry
        self._sym_expiry[put] = expiry
        self._expiry_pair[expiry] = pair
        ctx.submit_combo(combo([
            Leg(call, Side.SELL, 1.0),
            Leg(put, Side.SELL, 1.0),
        ], qty=self.qty, atomic=True))

    def on_fill(self, fill, ctx):
        expiry = self._sym_expiry.get(fill.sym)
        if expiry is None:
            return
        if fill.side == Side.SELL:
            call, put = self._expiry_pair.get(expiry, (None, None))
            if call is None or put is None:
                return
            if self._active is None or self._active.get("expiry") != expiry:
                self._active = {
                    "expiry": expiry,
                    "call": call,
                    "put": put,
                    "entry_prices": {},
                    "entry_credit": 0.0,
                    "under_min_credit": False,
                }
            self._active["entry_prices"][fill.sym] = float(fill.price)
            if len(self._active["entry_prices"]) == 2:
                self._active["entry_credit"] = sum(self._active["entry_prices"].values())
                self._active["under_min_credit"] = self._active["entry_credit"] < self.min_total_credit
                self._entered_expiries.add(expiry)
            return
        if fill.side == Side.BUY:
            self._maybe_clear_active(ctx)

    def on_expiry(self, ev, ctx):
        self._maybe_clear_active(ctx)


Strategy = ShortStraddleTwoHoursBeforeExpiry
