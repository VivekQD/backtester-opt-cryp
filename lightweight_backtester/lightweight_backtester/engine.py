from __future__ import annotations

from collections import defaultdict

import numpy as np

from .accounting import AccountingModel
from .config import ResearchConfig
from .data import DAY_NS, MarketDataStore
from .execution import ExecutionModel
from .metrics import build_trade_ledger, compute_metrics
from .types import (BacktestResult, FillEvent, MarkEvent, OptionType, SettlementEvent,
                    Side, TradeIntent)


def _float(path, name, idx, default=0.0):
    try:
        value = float(path[name][idx])
        return value if np.isfinite(value) else default
    except (KeyError, IndexError, TypeError, ValueError):
        return default


class BacktestEngine:
    def __init__(self, store: MarketDataStore, execution: ExecutionModel,
                 accounting: AccountingModel, trade_tape=None,
                 trade_tape_stops_only: bool = False,
                 trade_tape_quote_fallback: bool = True):
        self.store = store
        self.execution = execution
        self.accounting = accounting
        self.trade_tape = trade_tape
        self.trade_tape_stops_only = trade_tape_stops_only
        self.trade_tape_quote_fallback = trade_tape_quote_fallback

    def _fill_price(self, leg, side, path, idx, window_start_ns, is_stop=False):
        use_tape = (self.trade_tape is not None
                    and (not self.trade_tape_stops_only or is_stop))
        if use_tape:
            fill = self.trade_tape.fill(leg.contract.instrument, side,
                                        window_start_ns, leg.quantity)
            if fill is not None:
                return fill.price
            if not self.trade_tape_quote_fallback:
                return None
        price = self.execution.fill_price(side, _float(path, "bid_price", idx),
                                          _float(path, "ask_price", idx),
                                          _float(path, "mark_price", idx))
        # A far-OTM protective long can legitimately have no bid and a zero
        # mark near expiry.  When explicitly enabled, abandon that long wing
        # at zero instead of deleting the complete multi-leg trade.  This is
        # limited to closing an originally long option; entries and short-leg
        # buybacks still require a positive executable/model price.
        if (price is None and getattr(self, "abandon_zero_value_long_wings", False)
                and leg.side == Side.BUY and side == Side.SELL):
            return 0.0
        return price

    def _simulate_intent(self, intent: TradeIntent) -> list:
        expiry = min(leg.contract.expiry_ns for leg in intent.legs)
        exit_target = intent.exit_target_ns if intent.exit_target_ns is not None else expiry
        paths = [self.store.load_contract_path(leg.contract, intent.entry_target_ns,
                                               expiry) for leg in intent.legs]
        if any(len(p.get("date", [])) == 0 for p in paths):
            return []
        common = paths[0]["date"]
        for path in paths[1:]:
            common = np.intersect1d(common, path["date"], assume_unique=False)
        common = common[(common > intent.entry_target_ns) & (common <= expiry)]
        if common.size == 0:
            return []
        indices = [{int(t): i for i, t in enumerate(path["date"])} for path in paths]
        entry_ts = int(common[0])
        events = []
        entry_prices = {}
        for leg, path, lookup in zip(intent.legs, paths, indices):
            i = lookup[entry_ts]
            price = self._fill_price(leg, leg.side, path, i, intent.entry_target_ns)
            if price is None:
                return []
            spot = _float(path, "index_price", i, _float(path, "underlying_price", i))
            entry_prices[leg.contract.instrument] = price
            events.append(FillEvent(entry_ts, intent.trade_id, leg.contract, leg.side,
                                    leg.quantity, price, spot, "entry"))

        stopped = set()
        pending: dict[str, int] = {}
        pending_leg_profit: dict[str, int] = {}
        pending_breakeven: dict[str, int] = {}
        loss_stop_hit = False
        pending_structure = None
        pending_profit_target = None
        closed_debit = 0.0
        scheduled_structure_exit_max = None
        entry_credit = sum((1 if leg.side == Side.SELL else -1)
                           * entry_prices[leg.contract.instrument] * leg.quantity
                           for leg in intent.legs)
        for n, timestamp in enumerate(common):
            timestamp = int(timestamp)
            if (scheduled_structure_exit_max is not None
                    and timestamp > scheduled_structure_exit_max):
                break
            if timestamp > exit_target and not intent.exit_at_expiry:
                break
            marks, bids, asks, spots = {}, {}, {}, []
            for leg, path, lookup in zip(intent.legs, paths, indices):
                i = lookup[timestamp]
                marks[leg.contract.instrument] = _float(path, "mark_price", i)
                bids[leg.contract.instrument] = _float(path, "bid_price", i)
                asks[leg.contract.instrument] = _float(path, "ask_price", i)
                spot = _float(path, "index_price", i, _float(path, "underlying_price", i))
                if spot > 0:
                    spots.append(spot)
            spot = float(np.median(spots)) if spots else 0.0
            # Stops are evaluated at every native 5-minute observation, but the
            # retained equity curve needs only the entry/exit bars and each UTC
            # day's final mark. This is the main memory bound for long studies.
            day = timestamp // DAY_NS
            next_day = (int(common[n + 1]) // DAY_NS) if n + 1 < len(common) else None
            record_mark = n == 0 or next_day != day or timestamp in pending.values()
            events.append(MarkEvent(timestamp, spot, marks, bids, asks, record_mark))
            if pending_profit_target == timestamp:
                for leg, path, lookup in zip(intent.legs, paths, indices):
                    if leg.contract.instrument in stopped:
                        continue
                    i = lookup[timestamp]
                    side = Side.BUY if leg.side == Side.SELL else Side.SELL
                    price = self._fill_price(leg, side, path, i, timestamp)
                    if price is None:
                        return []
                    events.append(FillEvent(timestamp, intent.trade_id, leg.contract, side,
                                            leg.quantity, price, spot, "profit_target"))
                    if leg.side == Side.SELL:
                        closed_debit += price * leg.quantity
                    stopped.add(leg.contract.instrument)
                break
            if pending_structure == timestamp:
                fill_times = []
                for leg, path, lookup in zip(intent.legs, paths, indices):
                    i = lookup[timestamp]
                    side = Side.BUY if leg.side == Side.SELL else Side.SELL
                    tape_fill = None
                    if (self.trade_tape is not None and self.trade_tape_stops_only
                            and self.trade_tape.fill_method == "next_trade"):
                        tape_fill = self.trade_tape.fill(leg.contract.instrument, side,
                                                         timestamp, leg.quantity, expiry)
                    price = (tape_fill.price if tape_fill is not None else
                             self._fill_price(leg, side, path, i, timestamp, is_stop=True))
                    if price is None:
                        return []
                    fill_ts = tape_fill.first_ts_ns if tape_fill is not None else timestamp
                    fill_times.append(fill_ts)
                    events.append(FillEvent(fill_ts, intent.trade_id, leg.contract, side,
                                            leg.quantity, price, spot, "structure_stop_loss"))
                    stopped.add(leg.contract.instrument)
                pending_structure = None
                scheduled_structure_exit_max = max(fill_times) if fill_times else timestamp
                continue
            if (intent.structure_stop_pct is not None and pending_structure is None
                    and scheduled_structure_exit_max is None
                    and entry_credit > 0):
                close_mark = sum((1 if leg.side == Side.SELL else -1)
                                 * marks[leg.contract.instrument] * leg.quantity
                                 for leg in intent.legs)
                if close_mark >= entry_credit * (1 + intent.structure_stop_pct):
                    if self.stop_fill_mode == "theoretical_slippage":
                        desired_debit = (entry_credit * (1 + intent.structure_stop_pct)
                                         * (1 + self.theoretical_stop_slippage_pct))
                        long_credit = sum(marks[leg.contract.instrument] * leg.quantity
                                          for leg in intent.legs if leg.side == Side.BUY)
                        short_debit = sum(marks[leg.contract.instrument] * leg.quantity
                                         for leg in intent.legs if leg.side == Side.SELL)
                        short_scale = ((desired_debit + long_credit) / short_debit
                                       if short_debit > 0 else 1.0)
                        for leg in intent.legs:
                            mark = marks[leg.contract.instrument]
                            side = Side.BUY if leg.side == Side.SELL else Side.SELL
                            price = mark * short_scale if leg.side == Side.SELL else mark
                            events.append(FillEvent(timestamp, intent.trade_id, leg.contract,
                                                    side, leg.quantity, price, spot,
                                                    "structure_stop_theoretical_slippage"))
                            stopped.add(leg.contract.instrument)
                        break
                    elif n + 1 < len(common):
                        pending_structure = int(common[n + 1])
                    continue
            if (intent.structure_profit_target_pct is not None
                    and pending_profit_target is None and entry_credit > 0
                    and not (loss_stop_hit and intent.disable_structure_target_after_loss_stop)):
                close_mark = sum((1 if leg.side == Side.SELL else -1)
                                 * marks[leg.contract.instrument] * leg.quantity
                                 for leg in intent.legs
                                 if leg.contract.instrument not in stopped)
                target_debit = entry_credit * (1 - intent.structure_profit_target_pct)
                if closed_debit + close_mark <= target_debit and n + 1 < len(common):
                    pending_profit_target = int(common[n + 1])
                    continue
            for leg, path, lookup in zip(intent.legs, paths, indices):
                instrument = leg.contract.instrument
                if instrument in stopped or leg.side != Side.SELL:
                    continue
                if pending_leg_profit.get(instrument) == timestamp:
                    i = lookup[timestamp]
                    price = self._fill_price(leg, Side.BUY, path, i, timestamp)
                    if price is not None:
                        events.append(FillEvent(timestamp, intent.trade_id, leg.contract,
                                                Side.BUY, leg.quantity, price, spot,
                                                "leg_profit_target"))
                        closed_debit += price * leg.quantity
                        stopped.add(instrument)
                    continue
                if pending_breakeven.get(instrument) == timestamp:
                    i = lookup[timestamp]
                    price = self._fill_price(leg, Side.BUY, path, i, timestamp)
                    if price is not None:
                        events.append(FillEvent(timestamp, intent.trade_id, leg.contract,
                                                Side.BUY, leg.quantity, price, spot,
                                                "survivor_breakeven_stop"))
                        closed_debit += price * leg.quantity
                        stopped.add(instrument)
                    continue
                if pending.get(instrument) == timestamp:
                    i = lookup[timestamp]
                    tape_fill = None
                    if (self.trade_tape is not None and self.trade_tape_stops_only
                            and self.trade_tape.fill_method == "next_trade"):
                        tape_fill = self.trade_tape.fill(instrument, Side.BUY, timestamp,
                                                         leg.quantity, expiry)
                    price = (tape_fill.price if tape_fill is not None else
                             self._fill_price(leg, Side.BUY, path, i, timestamp, is_stop=True))
                    if price is not None:
                        fill_ts = tape_fill.first_ts_ns if tape_fill is not None else timestamp
                        events.append(FillEvent(fill_ts, intent.trade_id, leg.contract,
                                                Side.BUY, leg.quantity, price, spot, "stop_loss"))
                        closed_debit += price * leg.quantity
                        stopped.add(instrument)
                        # In short-leg vertical stop mode, close the protective
                        # long option on the same side and same common candle.
                        # This realizes the bounded vertical-spread value rather
                        # than leaving an orphan long wing behind.
                        if (intent.structure in ("iron_butterfly", "iron_condor")
                                and intent.structure_stop_pct is None):
                            for wing, wing_path, wing_lookup in zip(intent.legs, paths, indices):
                                wing_instrument = wing.contract.instrument
                                if (wing_instrument in stopped or wing.side != Side.BUY
                                        or wing.contract.option_type != leg.contract.option_type):
                                    continue
                                wi = wing_lookup[timestamp]
                                wing_price = self._fill_price(wing, Side.SELL, wing_path, wi,
                                                              timestamp, is_stop=True)
                                if wing_price is not None:
                                    events.append(FillEvent(timestamp, intent.trade_id,
                                                            wing.contract, Side.SELL,
                                                            wing.quantity, wing_price, spot,
                                                            "vertical_wing_stop_exit"))
                                    closed_debit -= wing_price * wing.quantity
                                    stopped.add(wing_instrument)
                        loss_stop_hit = True
                    continue
                target_pct = (intent.survivor_profit_target_pct
                              if loss_stop_hit and intent.survivor_profit_target_pct is not None
                              else leg.profit_target_pct)
                mark = marks[instrument]
                if target_pct is not None:
                    target_price = entry_prices[instrument] * (1 - float(target_pct))
                    if mark <= target_price and n + 1 < len(common):
                        pending_leg_profit[instrument] = int(common[n + 1])
                        continue
                if (loss_stop_hit and intent.survivor_breakeven_stop
                        and mark >= entry_prices[instrument] and n + 1 < len(common)):
                    pending_breakeven[instrument] = int(common[n + 1])
                    continue
                if leg.stop_loss_pct is None:
                    continue
                threshold = entry_prices[instrument] * (1 + float(leg.stop_loss_pct))
                if mark >= threshold:
                    if getattr(self, "stop_fill_mode", "next_bar_touch") in ("theoretical", "theoretical_slippage"):
                        fill = threshold
                        reason = "stop_loss_theoretical"
                        if self.stop_fill_mode == "theoretical_slippage":
                            fill *= 1 + self.theoretical_stop_slippage_pct
                            reason = "stop_loss_theoretical_slippage"
                        events.append(FillEvent(timestamp, intent.trade_id, leg.contract,
                                                Side.BUY, leg.quantity, fill, spot, reason))
                        closed_debit += fill * leg.quantity
                        stopped.add(instrument)
                        # A defined-risk short-leg stop closes the matching
                        # protective wing at the trigger snapshot.  The short
                        # is filled at its theoretical stop price; the long
                        # wing is sold at the executable quote available at
                        # that same timestamp.
                        if (intent.structure in ("iron_butterfly", "iron_condor")
                                and intent.structure_stop_pct is None):
                            for wing, wing_path, wing_lookup in zip(intent.legs, paths, indices):
                                wing_instrument = wing.contract.instrument
                                if (wing_instrument in stopped or wing.side != Side.BUY
                                        or wing.contract.option_type != leg.contract.option_type):
                                    continue
                                wi = wing_lookup[timestamp]
                                wing_price = self._fill_price(wing, Side.SELL, wing_path, wi,
                                                              timestamp, is_stop=True)
                                if wing_price is not None:
                                    events.append(FillEvent(timestamp, intent.trade_id,
                                                            wing.contract, Side.SELL,
                                                            wing.quantity, wing_price, spot,
                                                            "vertical_wing_theoretical_stop_exit"))
                                    closed_debit -= wing_price * wing.quantity
                                    stopped.add(wing_instrument)
                        loss_stop_hit = True
                    elif n + 1 < len(common):
                        pending[instrument] = int(common[n + 1])

        if len(stopped) == len(intent.legs):
            return events
        if not intent.exit_at_expiry:
            exits = common[common > exit_target]
            if exits.size == 0:
                return []
            exit_ts = int(exits[0])
            for leg, path, lookup in zip(intent.legs, paths, indices):
                if leg.contract.instrument in stopped:
                    continue
                i = lookup[exit_ts]
                side = Side.BUY if leg.side == Side.SELL else Side.SELL
                price = self._fill_price(leg, side, path, i, exit_target)
                if price is None:
                    return []
                spot = _float(path, "index_price", i, _float(path, "underlying_price", i))
                events.append(FillEvent(exit_ts, intent.trade_id, leg.contract, side,
                                        leg.quantity, price, spot, "scheduled_exit"))
            return events

        # Deribit-style 30-minute index TWAP ending at 08:00 expiry.
        reference = paths[0]
        dates = reference["date"]
        mask = (dates > expiry - 30 * 60 * 1_000_000_000) & (dates <= expiry)
        spots = np.asarray(reference.get("index_price", reference.get("underlying_price")), dtype=float)[mask]
        spots = spots[np.isfinite(spots) & (spots > 0)]
        settle = float(spots.mean()) if spots.size else _float(reference, "index_price", -1)
        for leg in intent.legs:
            if leg.contract.instrument in stopped:
                continue
            if leg.contract.option_type == OptionType.CALL:
                payoff = max(settle - leg.contract.strike, 0.0) / settle if settle > 0 else 0.0
            else:
                payoff = max(leg.contract.strike - settle, 0.0) / settle if settle > 0 else 0.0
            events.append(SettlementEvent(expiry, intent.trade_id, leg.contract,
                                          leg.quantity, payoff, settle))
        return events

    @staticmethod
    def _merge_marks(events: list):
        other, marks, bids, asks, recorded = [], defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(bool)
        spots = defaultdict(list)
        for event in events:
            if isinstance(event, MarkEvent):
                marks[event.ts_ns].update(event.marks)
                bids[event.ts_ns].update(event.bids)
                asks[event.ts_ns].update(event.asks)
                recorded[event.ts_ns] = recorded[event.ts_ns] or event.record_equity
                if event.spot > 0:
                    spots[event.ts_ns].append(event.spot)
            else:
                other.append(event)
        for ts, values in marks.items():
            other.append(MarkEvent(ts, float(np.median(spots[ts])) if spots[ts] else 0.0,
                                   values, bids[ts], asks[ts], recorded[ts]))
        return other

    def run(self, strategy, cfg: ResearchConfig) -> BacktestResult:
        if cfg.stop_fill_mode not in ("next_bar_touch", "theoretical", "theoretical_slippage"):
            raise ValueError("invalid stop_fill_mode")
        self.stop_fill_mode = cfg.stop_fill_mode
        self.theoretical_stop_slippage_pct = cfg.theoretical_stop_slippage_pct
        self.abandon_zero_value_long_wings = bool(
            cfg.extra.get("abandon_zero_value_long_wings", False))
        snapshots = self.store.snapshots(cfg)
        intents = []
        for target in sorted(snapshots):
            intent = strategy.on_entry(target, snapshots[target])
            if intent is not None and max(leg.contract.expiry_ns for leg in intent.legs) <= cfg.end_ns:
                intents.append(intent)
        events = []
        for intent in intents:
            events.extend(self._simulate_intent(intent))
        events = self._merge_marks(events)
        equity, legs = self.accounting.process(events)
        trades = build_trade_ledger(legs)
        metrics = compute_metrics(equity, legs, trades)
        metrics.update({"intraday_max_drawdown": self.accounting.intraday_max_drawdown,
                        "peak_initial_margin_native": self.accounting.peak_initial_margin,
                        "peak_maintenance_margin_native": self.accounting.peak_maintenance_margin,
                        "rejected_entry_legs": len(self.accounting.rejections),
                        "liquidated_legs": self.accounting.liquidations})
        retained = [x for x in events if not isinstance(x, MarkEvent) or x.record_equity]
        execution_ledger = self.trade_tape.audit if self.trade_tape is not None else []
        if execution_ledger:
            metrics.update({"tape_fill_requests": len(execution_ledger),
                            "tape_filled_requests": sum(x["filled"] for x in execution_ledger),
                            "tape_unfilled_requests": sum(not x["filled"] for x in execution_ledger)})
        return BacktestResult(retained, equity, legs, trades, metrics, execution_ledger)
