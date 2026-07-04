"""Sell the ATM call+put of the nearest in-window expiry, as one atomic combo."""
import numpy as np

from btcbt import Leg, Side, Strategy, combo


class ShortStraddle(Strategy):
    def on_snapshot(self, snap, ctx):
        if not ctx.portfolio().is_flat():
            return
        dmin = self.params.get("dte_min", 3)
        dmax = self.params.get("dte_max", 10)
        qty = self.params.get("qty", 1)

        opts = snap.select().options().dte(dmin, dmax)
        rows = opts.rows()
        if rows.size == 0:
            return
        e = int(snap.expiry[rows].min())   # nearest in-window expiry

        call = (snap.select().options().calls().expiry(e).tradable()
                .nearest("moneyness", 1.0).first())
        put = (snap.select().options().puts().expiry(e).tradable()
               .nearest("moneyness", 1.0).first())
        if call is None or put is None:
            return
        ctx.submit_combo(combo([Leg(call, Side.SELL, 1.0), Leg(put, Side.SELL, 1.0)],
                               qty=qty, atomic=True))
