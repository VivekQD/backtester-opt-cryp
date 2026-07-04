"""Sell a delta-target put when flat; hold to expiry (no exit rule)."""
from btcbt import Side, Strategy, market


class ShortPut(Strategy):
    def on_snapshot(self, snap, ctx):
        if not ctx.portfolio().is_flat():
            return
        dmin = self.params.get("dte_min", 3)
        dmax = self.params.get("dte_max", 10)
        delta = self.params.get("delta", 0.30)
        sym = (snap.select().options().puts()
               .dte(dmin, dmax).tradable()
               .nearest("delta", -delta).first())     # put deltas are negative
        if sym is None:
            return
        ctx.submit(market(sym, Side.SELL, self.params.get("qty", 1)))
