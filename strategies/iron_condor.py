"""Short iron condor: sell ~delta call+put, buy further-OTM wings. Held to expiry."""
from btcbt import Leg, Side, Strategy, combo


class IronCondor(Strategy):
    def on_snapshot(self, snap, ctx):
        if not ctx.portfolio().is_flat():
            return
        dmin = self.params.get("dte_min", 3)
        dmax = self.params.get("dte_max", 10)
        short_d = self.params.get("short_delta", 0.30)
        long_d = self.params.get("long_delta", 0.15)
        qty = self.params.get("qty", 1)

        opts = snap.select().options().dte(dmin, dmax)
        rows = opts.rows()
        if rows.size == 0:
            return
        e = int(snap.expiry[rows].min())

        sc = snap.select().options().calls().expiry(e).tradable().nearest("delta", short_d).first()
        sp = snap.select().options().puts().expiry(e).tradable().nearest("delta", -short_d).first()
        lc = snap.select().options().calls().expiry(e).tradable().nearest("delta", long_d).first()
        lp = snap.select().options().puts().expiry(e).tradable().nearest("delta", -long_d).first()
        legs = [sc, sp, lc, lp]
        if any(x is None for x in legs) or len(set(legs)) < 4:
            return
        ctx.submit_combo(combo([Leg(sc, Side.SELL, 1.0), Leg(sp, Side.SELL, 1.0),
                                Leg(lc, Side.BUY, 1.0), Leg(lp, Side.BUY, 1.0)],
                               qty=qty, atomic=True))
