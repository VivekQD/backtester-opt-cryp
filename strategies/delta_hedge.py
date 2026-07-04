"""Short ATM straddle, delta-hedged with the underlying when one is available.

If the dataset has no tradable underlying (options-only data), the hedge leg is
skipped and this behaves like a short straddle. A real hedge needs a PERP/SPOT
instrument in the book (Phase 3).
"""
from btcbt import Leg, Side, Strategy, combo, market


class DeltaHedge(Strategy):
    def on_snapshot(self, snap, ctx):
        P = ctx.portfolio()
        # open the straddle once
        if P.is_flat():
            dmin = self.params.get("dte_min", 3)
            dmax = self.params.get("dte_max", 10)
            opts = snap.select().options().dte(dmin, dmax)
            rows = opts.rows()
            if rows.size == 0:
                return
            e = int(snap.expiry[rows].min())
            c = snap.select().options().calls().expiry(e).tradable().nearest("moneyness", 1.0).first()
            p = snap.select().options().puts().expiry(e).tradable().nearest("moneyness", 1.0).first()
            if c is None or p is None:
                return
            ctx.submit_combo(combo([Leg(c, Side.SELL, 1.0), Leg(p, Side.SELL, 1.0)], atomic=True))
            return

        # re-hedge residual delta with the underlying, if one exists
        band = self.params.get("band", 0.1)
        d = P.agg_greeks(snap).delta
        if abs(d) <= band:
            return
        hedge = snap.select().underlying().tradable().first()
        if hedge is None:
            return  # options-only dataset: nothing to hedge with
        side = Side.SELL if d > 0 else Side.BUY
        ctx.submit(market(hedge, side, abs(d)))
