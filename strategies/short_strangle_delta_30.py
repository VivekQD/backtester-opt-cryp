"""Roll a short 30-delta strangle through each daily expiry cycle."""
from __future__ import annotations

import datetime as dt

import numpy as np

from btcbt import Leg, OptType, Side, Strategy as BaseStrategy, combo, triggered
from btcbt.types import DAY_NS, NS


DEFAULT_PARAMS = {
    # Deribit daily options expire at 08:00 UTC. At each 08:00 UTC roll,
    # target the next day's 08:00 UTC daily expiry.
    "daily_roll_hour_utc": 8,
    "roll_interval_mins": 240,
    "entry_delay_mins": 0,
    "entry_window_mins": 15,
    "min_minutes_to_expiry": 1,
    "qty": 1.0,

    # Short strangle legs: sell call near +delta and put near -delta.
    "target_delta": 0.30,

    # Per-leg stop. Set to 0 to disable; open legs otherwise roll every 2 hours
    # or settle at expiry.
    "leg_stop_loss_pct": 0.0,

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


class ShortStrangleDelta30(BaseStrategy):
    """Sell 30-delta call/put every 2 hours for each daily expiry."""

    def __init__(self, params: dict | None = None):
        merged = {**DEFAULT_PARAMS, **(params or {})}
        super().__init__(merged)
        self.daily_roll_hour_utc = int(self.params["daily_roll_hour_utc"])
        self.roll_interval_mins = float(self.params["roll_interval_mins"])
        if self.roll_interval_mins <= 0:
            raise ValueError("roll_interval_mins must be positive")
        self.roll_interval_ns = int(self.roll_interval_mins * 60 * NS)
        self.entry_delay_mins = float(self.params["entry_delay_mins"])
        self.entry_window_mins = float(self.params["entry_window_mins"])
        self.min_minutes_to_expiry = float(self.params["min_minutes_to_expiry"])
        self.qty = float(self.params["qty"])
        self.target_delta = abs(float(self.params["target_delta"]))
        self.leg_stop_loss_pct = float(self.params["leg_stop_loss_pct"])
        self.start_ns = _parse_dt(self.params.get("start_datetime"))
        self.end_ns = _parse_dt(self.params.get("end_datetime"))
        self._entered_slots: set[tuple[int, int]] = set()
        self._sym_expiry: dict[int, int] = {}
        self._expiry_pair: dict[int, tuple[int, int]] = {}
        self._pending_entry: dict | None = None
        self._active: dict | None = None

    def _within_date_bounds(self, ts: int) -> bool:
        if self.start_ns is not None and ts < self.start_ns:
            return False
        if self.end_ns is not None and ts > self.end_ns:
            return False
        return True

    def _daily_roll_ns(self, ts: int) -> int:
        cur = dt.datetime.fromtimestamp(ts / NS, tz=dt.timezone.utc)
        roll = cur.replace(
            hour=self.daily_roll_hour_utc,
            minute=0,
            second=0,
            microsecond=0,
        )
        if cur < roll:
            roll -= dt.timedelta(days=1)
        return int(roll.timestamp() * NS)

    def _daily_expiry_for_ts(self, ts: int) -> int:
        return self._daily_roll_ns(ts) + DAY_NS

    def _slot_start_for_ts(self, ts: int, expiry: int) -> int | None:
        roll = expiry - DAY_NS
        if ts < roll or ts >= expiry:
            return None
        slot_index = int((ts - roll) // self.roll_interval_ns)
        slot = roll + slot_index * self.roll_interval_ns
        return slot if roll <= slot < expiry else None

    def _entry_window_open(self, ts: int, expiry: int, slot: int) -> bool:
        mins_since_slot = (ts - slot) / (60 * NS)
        if mins_since_slot < self.entry_delay_mins:
            return False
        if self.entry_window_mins >= 0:
            latest = self.entry_delay_mins + self.entry_window_mins
            if mins_since_slot > latest:
                return False
        mins_to_expiry = (expiry - ts) / (60 * NS)
        if mins_to_expiry < self.min_minutes_to_expiry:
            return False
        return True

    def _candidate_slot(self, snap) -> tuple[int, int] | None:
        if not self._within_date_bounds(snap.ts):
            return None
        expiry = self._daily_expiry_for_ts(snap.ts)
        slot = self._slot_start_for_ts(snap.ts, expiry)
        if slot is None:
            return None
        key = (expiry, slot)
        if key in self._entered_slots:
            return None
        if not self._entry_window_open(snap.ts, expiry, slot):
            return None
        rows = np.flatnonzero(
            (snap.otype != int(OptType.NONE))
            & (snap.expiry == expiry)
            & (snap.expiry > snap.ts)
        )
        return (expiry, slot) if rows.size else None

    def _quoted_delta_rows(self, snap, rows):
        return rows[
            np.isfinite(snap.delta[rows])
            & np.isfinite(snap.bid[rows])
            & (snap.bid[rows] > 0)
        ]

    def _delta_pair(self, snap, expiry: int) -> tuple[int, int] | None:
        rows = snap.select().options().expiry(expiry).tradable().rows()
        if rows.size == 0:
            return None

        calls = self._quoted_delta_rows(snap, rows[snap.otype[rows] == int(OptType.CALL)])
        puts = self._quoted_delta_rows(snap, rows[snap.otype[rows] == int(OptType.PUT)])
        if calls.size == 0 or puts.size == 0:
            return None

        call_row = calls[int(np.argmin(np.abs(snap.delta[calls] - self.target_delta)))]
        put_row = puts[int(np.argmin(np.abs(snap.delta[puts] + self.target_delta)))]
        call = int(snap.sym[call_row])
        put = int(snap.sym[put_row])
        if call == put:
            return None
        return call, put

    def _stop_reference_price(self, snap, sym: int) -> float | None:
        row = snap.row(sym)
        if row is None:
            return None
        mark = float(snap.mark[row])
        if _finite_pos(mark):
            return mark
        ask = float(snap.ask[row])
        return ask if _finite_pos(ask) else None

    def _maybe_clear_active(self, ctx) -> None:
        if self._active is None:
            return
        call, put = self._active["call"], self._active["put"]
        if ctx.portfolio().net_qty(call) == 0 and ctx.portfolio().net_qty(put) == 0:
            self._active = None

    def _active_roll_due(self, ts: int) -> bool:
        if self._active is None:
            return False
        expiry = int(self._active["expiry"])
        slot = int(self._active["slot"])
        if ts >= expiry:
            return False
        if (expiry - ts) / (60 * NS) < self.min_minutes_to_expiry:
            return False
        return ts >= slot + self.roll_interval_ns

    def _active_closeable(self, snap, ctx) -> bool:
        if self._active is None:
            return False
        checked = False
        for sym in (self._active["call"], self._active["put"]):
            pos = ctx.portfolio().position(sym)
            if not pos or pos.qty == 0:
                continue
            checked = True
            row = snap.row(sym)
            if row is None or not bool(snap.valid[row]):
                return False
        return checked

    def _submit_close_active(self, ctx) -> bool:
        if self._active is None:
            return False
        submitted = False
        for sym in (self._active["call"], self._active["put"]):
            pos = ctx.portfolio().position(sym)
            if pos and pos.qty != 0:
                ctx.close(sym)
                submitted = True
        return submitted

    def _submit_entry(self, ctx, expiry: int, slot: int, pair: tuple[int, int]) -> None:
        call, put = pair
        self._sym_expiry[call] = expiry
        self._sym_expiry[put] = expiry
        self._expiry_pair[expiry] = pair
        self._pending_entry = {
            "expiry": expiry,
            "slot": slot,
            "call": call,
            "put": put,
            "entry_prices": {},
            "entry_indices": {},
        }
        ctx.submit_combo(combo([
            Leg(call, Side.SELL, 1.0),
            Leg(put, Side.SELL, 1.0),
        ], qty=self.qty, atomic=True))

    def _submit_stop_leg(self, ctx, sym: int, px: float) -> None:
        pos = ctx.portfolio().position(sym)
        if pos and pos.qty < 0 and _finite_pos(px):
            ctx.submit(triggered(sym, Side.BUY, abs(pos.qty), px))

    def _spot_allows_stop(self, snap, sym: int) -> bool:
        if self._active is None:
            return False
        entry_index = float(self._active.get("entry_index", 0.0))
        if not _finite_pos(entry_index) or not _finite_pos(snap.index):
            return True
        if sym == self._active["call"]:
            return float(snap.index) > entry_index
        if sym == self._active["put"]:
            return float(snap.index) < entry_index
        return True

    def _manage_stops(self, snap, ctx) -> None:
        if self._active is None:
            return
        if self.leg_stop_loss_pct <= 0:
            return
        entry_prices = self._active.get("entry_prices", {})
        for sym in (self._active["call"], self._active["put"]):
            pos = ctx.portfolio().position(sym)
            if not pos or pos.qty >= 0:
                continue
            entry = float(entry_prices.get(sym, 0.0))
            if entry <= 0:
                continue
            stop_ref = self._stop_reference_price(snap, sym)
            if stop_ref is None or not self._spot_allows_stop(snap, sym):
                continue
            stop_px = entry * (1.0 + self.leg_stop_loss_pct)
            if stop_ref >= stop_px:
                self._submit_stop_leg(ctx, sym, stop_px)

    def on_snapshot(self, snap, ctx):
        self._maybe_clear_active(ctx)
        if self._active is not None and not ctx.portfolio().is_flat():
            if self._active_roll_due(snap.ts):
                candidate = self._candidate_slot(snap)
                if candidate is None:
                    self._submit_close_active(ctx)
                    return
                expiry, slot = candidate
                pair = self._delta_pair(snap, expiry)
                if pair is None or not self._active_closeable(snap, ctx):
                    self._submit_close_active(ctx)
                    return
                self._submit_close_active(ctx)
                self._submit_entry(ctx, expiry, slot, pair)
                return
            self._manage_stops(snap, ctx)
            return
        if not ctx.portfolio().is_flat():
            return

        candidate = self._candidate_slot(snap)
        if candidate is None:
            return
        expiry, slot = candidate

        pair = self._delta_pair(snap, expiry)
        if pair is None:
            return

        self._submit_entry(ctx, expiry, slot, pair)

    def on_fill(self, fill, ctx):
        if fill.side == Side.SELL:
            pending = self._pending_entry
            if pending is None or fill.sym not in (pending["call"], pending["put"]):
                return
            expiry = int(pending["expiry"])
            slot = int(pending["slot"])
            if self._active is None or self._active.get("slot") != slot:
                self._active = {
                    "expiry": expiry,
                    "slot": slot,
                    "call": pending["call"],
                    "put": pending["put"],
                    "entry_prices": {},
                    "entry_indices": {},
                    "entry_index": 0.0,
                }
            self._active["entry_prices"][fill.sym] = float(fill.price)
            self._active["entry_indices"][fill.sym] = float(fill.index_price)
            if len(self._active["entry_prices"]) == 2:
                self._active["entry_index"] = sum(self._active["entry_indices"].values()) / 2.0
                self._entered_slots.add((expiry, slot))
                self._pending_entry = None
            return
        if fill.side == Side.BUY:
            self._maybe_clear_active(ctx)

    def on_expiry(self, ev, ctx):
        self._maybe_clear_active(ctx)


Strategy = ShortStrangleDelta30
