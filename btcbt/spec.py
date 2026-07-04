"""Declarative StrategySpec -> compiled Strategy. Phase 4.

A serializable mirror of the primitives: condition Groups lower to Expr, LegSpecs
resolve to concrete syms -> Combo, session/risk become guards. Anything the spec
can't express drops to a hand-written Strategy on the same engine.
"""
from __future__ import annotations

import datetime as dt
import pickle
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import numpy as np

from .conditions import PF, T, U, Expr, Scope
from .engine import Strategy
from .indicators import IndicatorSpec, Indicators
from .types import DAY_NS, InstKind, Leg, OptType, Side, combo

_DAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]


# ---- declarative dataclasses ----------------------------------------------
@dataclass(frozen=True)
class Operand:
    ref: str = None
    fieldn: str = None
    offset: int = 0
    const: float = None


@dataclass(frozen=True)
class Compare:
    left: Operand
    op: str
    right: Operand


@dataclass(frozen=True)
class Group:
    match: str
    items: list


@dataclass(frozen=True)
class StrikeSpec:
    method: str           # moneyness | delta | premium | none
    value: object = None  # "ATM"/"OTM+1" | delta magnitude | premium


@dataclass(frozen=True)
class ExpirySpec:
    bucket: str = "this_week"   # this_week/next_week/this_month/next_month/none/dte<=n


@dataclass(frozen=True)
class LegRisk:
    sl: float = None
    sl_unit: str = "%"
    tp: float = None
    tp_unit: str = "%"
    reentry: int = 0


@dataclass(frozen=True)
class LegSpec:
    side: Side
    kind: InstKind
    otype: OptType
    lots: float
    lot_size: int
    expiry: ExpirySpec
    strike: StrikeSpec
    risk: LegRisk = None


@dataclass(frozen=True)
class CalendarSpec:
    tz: str = "UTC"
    windows: list = None
    date_from: str = None
    date_to: str = None
    exclude: list = None
    months: list = None


@dataclass(frozen=True)
class SessionSpec:
    holding: str = "overnight"
    direction: str = "both"
    start: str = "00:00"
    squareoff: str = "00:00"
    days: list = field(default_factory=lambda: list(_DAYS))
    linked_exit: bool = True
    calendar: CalendarSpec = None


@dataclass(frozen=True)
class RiskSpec:
    daily_profit_cap: float = None
    daily_loss_limit: float = None
    max_trades_day: int = None


@dataclass(frozen=True)
class StrategySpec:
    name: str
    indicators: list = field(default_factory=list)
    session: SessionSpec = field(default_factory=SessionSpec)
    entry: Group = None
    exit: Group = None
    legs: list = field(default_factory=list)
    risk: RiskSpec = None


# ---- time / session -------------------------------------------------------
def _tzdt(ts: int, tz: str):
    return dt.datetime.fromtimestamp(ts / 1e9, ZoneInfo(tz))


def time_fields(ts: int, tz: str, entry_ts=None, snap=None, row=None) -> dict:
    d = _tzdt(ts, tz)
    out = {
        "minute": d.minute, "hour": d.hour, "dow": d.weekday(), "dom": d.day,
        "week_of_month": (d.day - 1) // 7 + 1, "month": d.month,
        "date": d.year * 10000 + d.month * 100 + d.day,
    }
    if snap is not None and row is not None:
        out["dte"] = float(snap.tte[row] * 365.0)
    if entry_ts:
        out["days_in_trade"] = (ts - entry_ts) / DAY_NS
        out["since_entry"] = out["days_in_trade"]
    return out


def _hm(s: str) -> tuple[int, int]:
    h, _, m = s.partition(":")
    return int(h), int(m or 0)


def in_session(ts: int, session: SessionSpec) -> bool:
    cal = session.calendar or CalendarSpec()
    d = _tzdt(ts, cal.tz)
    if _DAYS[d.weekday()] not in [x.upper() for x in session.days]:
        return False
    if cal.months and d.month not in cal.months:
        return False
    iso = d.date().isoformat()
    if cal.exclude and iso in cal.exclude:
        return False
    if cal.date_from and iso < cal.date_from:
        return False
    if cal.date_to and iso > cal.date_to:
        return False
    cur = d.hour * 60 + d.minute
    windows = cal.windows or [(session.start, session.squareoff)]
    for lo, hi in windows:
        lh, lm = _hm(lo); hh, hm = _hm(hi)
        a, b = lh * 60 + lm, hh * 60 + hm
        if a == b or (a <= cur <= b):
            return True
    return False


def _past_squareoff(ts: int, session: SessionSpec) -> bool:
    cal = session.calendar or CalendarSpec()
    d = _tzdt(ts, cal.tz)
    hh, hm = _hm(session.squareoff)
    return (d.hour * 60 + d.minute) >= (hh * 60 + hm)


# ---- Group -> Expr --------------------------------------------------------
_OPS = {
    "gt": lambda a, b: a > b, "lt": lambda a, b: a < b,
    "ge": lambda a, b: a >= b, "le": lambda a, b: a <= b, "eq": lambda a, b: a == b,
    "above": lambda a, b: a > b, "below": lambda a, b: a < b,
    "above_or_equal": lambda a, b: a >= b, "below_or_equal": lambda a, b: a <= b,
}


def _resolve(op: Operand, scope: Scope, extra: int = 0):
    if op.const is not None:
        return op.const
    ref, off = op.ref, op.offset + extra
    if ref == "G":
        return scope.market.value(op.fieldn, off) if scope.market else float("nan")
    if ref == "T":
        return T(op.fieldn)(scope)
    if ref == "PF":
        return PF(op.fieldn)(scope)
    if ref in ("U", "index", "forward", "market"):
        return U(op.fieldn or (ref if ref in ("index", "forward") else "index"))(scope)
    v = scope.indicators.value(ref, op.fieldn, off) if scope.indicators else float("nan")
    if v != v and scope.market:        # nan -> try a global by that name
        v = scope.market.value(ref, off)
    return v


def _cmp(c: Compare, scope: Scope) -> bool:
    if c.op in ("cross_above", "cross_below"):
        ln, rn = _resolve(c.left, scope, 0), _resolve(c.right, scope, 0)
        lp, rp = _resolve(c.left, scope, 1), _resolve(c.right, scope, 1)
        if any(x != x for x in (ln, rn, lp, rp)):
            return False
        return (lp <= rp and ln > rn) if c.op == "cross_above" else (lp >= rp and ln < rn)
    a, b = _resolve(c.left, scope), _resolve(c.right, scope)
    if a != a or b != b:               # nan during warmup
        return False
    return bool(_OPS[c.op](a, b))


def to_expr(group: Group) -> Expr:
    if group is None:
        return Expr(lambda s: True)

    def ev(s):
        res = [to_expr(it)(s) if isinstance(it, Group) else _cmp(it, s) for it in group.items]
        return all(res) if group.match.upper() == "AND" else any(res)
    return Expr(ev)


# ---- leg resolution -------------------------------------------------------
def resolve_expiry(snap, e: ExpirySpec):
    if e is None or e.bucket == "none":
        return 0
    opt = snap.select().options().rows()
    if opt.size == 0:
        return 0
    exps = np.unique(snap.expiry[opt])
    if e.bucket.startswith("dte<="):
        n = float(e.bucket.split("<=")[1])
        ok = exps[(exps - snap.ts) / DAY_NS <= n]
        return int(ok.min()) if ok.size else int(exps[0])
    idx = 1 if e.bucket in ("next_week", "next_month") and exps.size > 1 else 0
    return int(exps[idx])


def resolve_strike(snap, otype: OptType, expiry_ts: int, s: StrikeSpec):
    sel = snap.select().options().otype(otype).expiry(expiry_ts).tradable()
    if s is None or s.method == "none":
        return sel.first()
    if s.method == "delta":
        target = float(s.value) if otype == OptType.CALL else -float(s.value)
        return sel.nearest("delta", target).first()
    if s.method == "premium":
        return sel.nearest("mark", float(s.value)).first()
    # moneyness: ATM | OTM+n | ITM-n
    rows = sel.rows()
    if rows.size == 0:
        return None
    strikes = np.sort(np.unique(snap.strike[rows]))
    atm_i = int(np.argmin(np.abs(strikes - snap.index)))
    spec = str(s.value).upper()
    step = 0
    if spec.startswith("OTM"):
        step = int(spec[3:] or 1)
        step = step if otype == OptType.CALL else -step
    elif spec.startswith("ITM"):
        step = -int(spec[3:] or 1)
        step = step if otype == OptType.CALL else -step
    target_strike = strikes[min(max(atm_i + step, 0), strikes.size - 1)]
    return sel.nearest("strike", float(target_strike)).first()


# ---- compile --------------------------------------------------------------
def compile_spec(spec: StrategySpec) -> Strategy:
    class _Compiled(Strategy):
        def __init__(self, params=None):
            super().__init__(params)
            self.spec = spec

        def on_start(self, ctx):
            self.ind = Indicators()
            for isp in spec.indicators:
                self.ind.add(isp)
            self.entry_expr = to_expr(spec.entry)
            self.exit_expr = to_expr(spec.exit) if spec.exit else None
            self._entry_ts = {}
            self._day = None
            self._day_start_eq = None
            self._trades_today = 0

        def _risk_tripped(self, ctx, snap) -> bool:
            r = spec.risk
            if r is None:
                return False
            day = snap.ts // DAY_NS
            eq = ctx.portfolio().equity(snap).usd
            if day != self._day:
                self._day, self._day_start_eq, self._trades_today = day, eq, 0
            pnl = eq - (self._day_start_eq or eq)
            if r.daily_loss_limit is not None and pnl <= -abs(r.daily_loss_limit):
                return True
            if r.daily_profit_cap is not None and pnl >= abs(r.daily_profit_cap):
                return True
            if r.max_trades_day is not None and self._trades_today >= r.max_trades_day:
                return True
            return False

        def on_snapshot(self, snap, ctx):
            self.ind.update(snap, ctx)
            sess = spec.session
            if not in_session(snap.ts, sess):
                if sess.holding == "intraday" and _past_squareoff(snap.ts, sess):
                    for sym in list(ctx.portfolio().positions):
                        ctx.close(sym)
                return
            P = ctx.portfolio()
            mkt = ctx.market()

            if self.exit_expr is not None:
                for sym, pos in list(P.positions.items()):
                    if pos.qty == 0:
                        continue
                    sc = Scope(snap, ctx, mkt, self.ind, row=snap.row(sym),
                               entry_ts=self._entry_ts.get(sym))
                    if self.exit_expr(sc):
                        ctx.close(sym)

            if self._risk_tripped(ctx, snap):
                return

            if P.is_flat() and self.entry_expr(Scope(snap, ctx, mkt, self.ind)):
                legs = []
                for L in spec.legs:
                    if L.kind != InstKind.OPTION:
                        sym = snap.select().underlying().tradable().first()
                    else:
                        e = resolve_expiry(snap, L.expiry)
                        sym = resolve_strike(snap, L.otype, e, L.strike)
                    if sym is None:
                        return
                    legs.append(Leg(sym, L.side, float(L.lots) * float(L.lot_size)))
                    self._entry_ts[sym] = snap.ts
                ctx.submit_combo(combo(legs, qty=1.0, atomic=sess.linked_exit))
                self._trades_today += 1

    # return the class (instantiated by the engine/runner with params)
    return _Compiled


# ---- JSON / dict construction (frontend <-> backend) ----------------------
def _isnum(s) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _operand(x) -> Operand:
    if isinstance(x, dict):
        if x.get("const") is not None:
            return Operand(const=float(x["const"]))
        return Operand(ref=x.get("ref"), fieldn=x.get("field"),
                       offset=int(x.get("offset") or 0))
    if isinstance(x, (int, float)):
        return Operand(const=float(x))
    s = str(x).strip()
    if s != "" and _isnum(s):
        return Operand(const=float(s))
    if "." in s:
        r, f = s.split(".", 1)
        return Operand(ref=r, fieldn=f)
    return Operand(ref=s)


def spec_from_dict(d: dict) -> StrategySpec:
    ses = d.get("session", {})
    session = SessionSpec(
        holding=ses.get("holding", "overnight"),
        direction=ses.get("direction", "both"),
        start=ses.get("start", "00:00"),
        squareoff=ses.get("squareoff", "00:00"),
        days=list(ses.get("days", _DAYS)),
        linked_exit=bool(ses.get("linked_exit", True)),
    )
    inds = [IndicatorSpec(i["name"], i["kind"], {"window": int(i.get("window", 14))},
                          i.get("timeframe", i.get("tf", "5m")), i.get("source", "index"))
            for i in d.get("indicators", [])]
    e = d.get("entry", {})
    entry = (Group(e.get("match", "AND"),
                   [Compare(_operand(r["l"]), r["op"], _operand(r["r"])) for r in e["rows"]])
             if e.get("rows") else None)
    x = d.get("exit", {})
    exit_g = (Group(x.get("match", "OR"),
                    [Compare(_operand(r["l"]), r["op"], _operand(r["r"])) for r in x["rows"]])
              if x.get("rows") else None)
    legs = []
    for l in d.get("legs", []):
        sl = l.get("sl")
        legs.append(LegSpec(
            Side[l["side"]], InstKind[l.get("kind", "OPTION")], OptType[l["otype"]],
            float(l.get("lots", 1)), int(l.get("lot_size", 1)),
            ExpirySpec(l.get("expiry", "this_week")),
            StrikeSpec(l.get("method", "delta"), l.get("value", "0.3")),
            LegRisk(float(sl)) if sl not in (None, "") else None))
    r = d.get("risk", {})
    num = lambda v: float(v) if _isnum(v) else None
    risk = RiskSpec(num(r.get("profit_cap")), num(r.get("loss_limit")),
                    int(float(r["max_trades"])) if _isnum(r.get("max_trades")) else None)
    return StrategySpec(name=d.get("name", "spec"), indicators=inds, session=session,
                        entry=entry, exit=exit_g, legs=legs, risk=risk)


# ---- persistence ----------------------------------------------------------
def save_spec(spec: StrategySpec, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(spec, f)


def load_spec(path: str) -> StrategySpec:
    with open(path, "rb") as f:
        return pickle.load(f)
