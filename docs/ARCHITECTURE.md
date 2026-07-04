# BTC Options Backtester

Python backtester for Deribit BTC options. Strategy = one object. Engine = everything else. Works for any holding pattern: close early via orders, or hold to expiry via settlement.

> This doc specifies the **full target**. What gets built **first (MVP) vs deferred** is sequenced in [Build phases](#build-phases) and [Defer / simplify](#defer--simplify). Start coding from Phase 1.

## Product shape

```text
1. drop a strategy in strategies/  (a Strategy class; later, a StrategySpec)
2. set data range, cash, fees, params in a config (TOML)
3. run one backtest, or a parameter sweep
4. get equity, fills, trades, greeks, records, stats
```

```bash
python -m btcbt.run   --strategy strategies.short_straddle:Strategy --config configs/base.toml
python -m btcbt.sweep --strategy strategies.short_straddle:Strategy --grid configs/straddle_grid.toml --workers 8
```

```text
strategies/
  short_put.py
  short_straddle.py
  iron_condor.py
```

## Data

Stored schema:

```
date              timestamp[us, UTC]
instrument_name   str          e.g. BTC_USDC-12JUN26-52000-C
option_type       str          call | put
strike            f64
index_price       f64
underlying_price  f64          forward
mark_price        f64
bid_price         f64
ask_price         f64
mark_iv           f64          percent
bid_iv            f64          percent
ask_iv            f64          percent
open_interest     f64
volume            f64
delta             f64
gamma             f64
theta             f64
vega              f64
daystogo          f64          time to expiry, days
currency          str          BTC (inverse) | BTC_USDC (linear)
expiry_month      str          YYYY-MM
```

- Bars: 5-min, `date` = UTC bar time.
- IV in percent; `iv/100` internally.
- `0.0` quote/iv/oi/volume = absent.
- Greeks precomputed; used as-is. No rho.
- Expiry, strike, type parsed from `instrument_name`; expiry at 08:00 UTC.

Canonical (after load) — every stored variable is preserved and reachable:

```
ts i64  sym u32  kind enum  otype enum  strike f64  expiry_ts i64
index f64  forward f64  mark f64  bid f64  ask f64
iv f64  bid_iv f64  ask_iv f64  oi f64  volume f64
delta f64  gamma f64  vega f64  theta f64  tte f64  moneyness f64  valid bool
```

Underlying instruments (SPOT/PERP/FUTURE) are synthesized as rows: `mark` = index/forward, `delta` = 1, other greeks 0, `otype = NONE`. They are tradable like any option.

Storage layout:

```text
data/deribit_btc_options/
  date_month=2026-01/
    expiry_month=2026-01/*.parquet
    expiry_month=2026-02/*.parquet
```

Partition by `date_month` first (most runs filter by time range), then `expiry_month` (expiry scans are common). Load via `pyarrow.dataset` with column projection + date filter; convert to numpy only after filtering. Build integer `sym` ids once and use ids everywhere; `bar_offsets`: ts → contiguous row range so each snapshot is a zero-copy view.

## Venue

```
inverse  call payoff = max(S_T-K,0)/S_T      put = max(K-S_T,0)/S_T     unit BTC
linear   call payoff = max(S_T-K,0)          put = max(K-S_T,0)         unit USDC
S_T            = TWAP of index over twap_mins ending at expiry (08:00 UTC)
trade fee      = Deribit option base fee, capped at 12.5% of option value:
                 inverse min(0.0003, 0.125*premium) * qty
                 linear  min(0.0003*index, 0.125*premium) * qty * contract_size
settle fee     = 0.015% of the same basis, capped at 12.5% of payoff; daily options exempt
```

`currency` selects inverse vs linear. USD = btc*index.

## Layers

```
6 runner     config, param grid, sweep
5 metrics    equity, trades, greeks, stats
4 portfolio  positions, PnL, margin, agg greeks, settlement
3 exec       order -> fill vs bar (spread, slippage, fee)
2 strategy   on_snapshot / on_fill / on_expiry   <- plug in
1 engine     chronological clock + lifecycle
0 data       Parquet -> canonical -> per-ts Snapshot
```

Strategy sees read-only `Snapshot`, emits `Order`; no access to files/clock/portfolio internals.

## Execution model

All CE/PE across every strike and expiry are processed **together, per bar** — never instrument-by-instrument loops.

```
within a bar   Snapshot is struct-of-arrays (columnar numpy). Every per-bar op runs over the
               whole chain at once — one vector op over all strikes × CE/PE:
                 assemble (tte/moneyness/valid) · global aggregates · condition EVAL ·
                 selector filters · MTM · aggregate greeks
               This is the parallel form within a bar (numpy/BLAS/SIMD; GIL makes vectorization,
               not Python threads, the right tool). Optional numba for any non-vectorizable kernel.
across runs    sweeps/studies fan out over processes (multiprocessing pool) — true parallel.
ordering       bars are strictly sequential (chronological clock) to preserve causality/no-lookahead;
               the parallelism is across instruments (within a bar) and across runs, never across time.
```

## Variable model (core invariant)

One namespace, one resolver. Every value the system can see is a **named variable** in a single space, and any variable composes with any other in any expression — this is the design's load-bearing guarantee.

```
namespaces   F per-instrument  ·  U market  ·  G global/chain  ·  PF portfolio
             T time  ·  HIST history  ·  IND indicator  ·  REC recorded   (VAR("ns.name") resolves all)
open         stored columns auto-register; derived (G/IND/REC) register once → read like any other
uniform      same name works in: conditions · selectors(where) · indicator source · sizing · risk · record · plots
composable   full algebra (arith + compare + bool + reducers + offset); scalar↔vector broadcasts
             → any combination expressible; nothing is a special case
escape       a raw lambda(Scope)->bool is allowed, never required
```

If a new variable can be named, it is usable everywhere the moment it is registered. No call site enumerates variables; all dispatch through the resolver.

## Strategy surface

Strategy logic is arbitrary Python over the full state. No model, structure, or signal type is imposed by the engine.

Readable each bar (`Snapshot` + `Context`):

```
per instrument   kind mark bid ask mid spread iv bid_iv ask_iv oi volume          F(name) / col / where
                 delta gamma vega theta strike expiry tte moneyness valid
market           index/spot, forward                                              U(name)
global (chain)   pcr_oi pcr_vol atm_iv skew_25 rr_25 fly_25 term_slope vwiv        G(name)
                 total_oi total_vol total_gamma total_vega gex max_pain basis rv_n iv_rank  + any registered
portfolio        positions, cash, realized/unrealized, net qty, aggregate greeks   PF(name)
history          past n bars per instrument, past n index, own recorded series     HIST / REC
time             minute hour dow dom week_of_month month date dte days_in_trade     T(name)
state            free-form per-strategy scratch
```

Every variable is name-addressable and open: stored columns auto-expose; derived globals/indicators register once, then read identically (`F`/`U`/`G`/`PF`/`HIST`/`T`/`IND`) in conditions, selectors (`where`), indicator `source`, sizing, recording, and plots.

Emitted: `Order` / `Combo` (market/limit, buy/sell, any qty, any instrument incl. underlying), `cancel_all`, `record`.

Entry/exit: arbitrary `Expr` over any of the above (`F`/`U`/`PF`/`HIST`/`REC`/`IND` + combinators), or a raw `lambda(Scope)->bool`. Hedge legs (spot/perp/future) compose with option legs in one `Combo`.

Two equivalent forms, same engine:
- declarative `StrategySpec` (indicators, condition groups, legs, session, risk) compiled via `compile_spec` — the no-code builder surface.
- hand-written `Strategy` class — escape hatch for anything the spec can't express.

Modularity / reliability:

```
swappable   Strategy, FillModel, MarginModel, VenueConfig, DataConfig — all injected
contracts   stable Snapshot / Order / Fill across strategies; SCHEMA_VERSION pinned
determinism same data + config + seed → identical RunResult
no lookahead Snapshot at ts holds only data ≤ ts; orders fill on the next bar
accounting  double-entry cash; equity = cash + unrealized; reconciles to trade log
isolation   strategy cannot touch files, clock, or mutate portfolio
```

## Code schema

```python
SCHEMA_VERSION = 1
YEAR_NS = 365*24*3600*1_000_000_000

class InstKind(Enum):  OPTION; PERP; FUTURE; SPOT      # all tradable, all in the book
class OptType(Enum):   CALL; PUT; NONE                 # NONE for non-options
class Side(Enum):      BUY; SELL
class Margin(Enum):    INVERSE; LINEAR
class OrderKind(Enum): MARKET; LIMIT

# ---- value types ----------------------------------------------------------
@dataclass(frozen=True)
class Order:        sym: int; side: Side; qty: float; kind: OrderKind; limit: float | None = None; group: int = 0
@dataclass(frozen=True)
class Leg:          sym: int; side: Side; ratio: float                  # ratio per 1 combo unit
@dataclass(frozen=True)
class Combo:        legs: list[Leg]; qty: float; kind: OrderKind; limit: float | None = None; atomic: bool = False
@dataclass(frozen=True)
class Fill:         sym: int; side: Side; qty: float; price: float; fee: float; ts: int; group: int = 0
@dataclass
class Position:     sym: int; qty: float; avg_entry: float; realized: float
@dataclass(frozen=True)
class ExpiryEvent:  sym: int; settle_index: float; payoff: float; fee: float; ts: int
@dataclass(frozen=True)
class Equity:       btc: float; usd: float; dd: float
@dataclass(frozen=True)
class Greeks:       delta: float; gamma: float; vega: float; theta: float

# order / combo constructors
def market(sym, side, qty) -> Order: ...
def limit(sym, side, qty, px) -> Order: ...
def close(pos: Position) -> Order: ...                 # opposite side, |qty|
def combo(legs: list[Leg], qty, kind=MARKET, limit=None, atomic=False) -> Combo: ...
def expand_combo(c: Combo, snap: Snapshot) -> list[Order]: ...   # legs → per-instrument orders

# ---- instrument meta ------------------------------------------------------
@dataclass(frozen=True)
class Meta:         sym: int; name: str; kind: InstKind; otype: OptType
                    strike: float; expiry_ts: int; currency: str; margin: Margin
def parse_name(name: str) -> tuple[InstKind, OptType, float, int]:   # from instrument_name
    ...

# underlying instruments synthesized at load and added to the universe/book:
#   SPOT  mark = index            (no expiry, delta=1, gamma/vega/theta=0)
#   PERP  mark = forward          (no expiry, funding accrual, delta=1)
#   FUTURE mark = forward         (dated, settles to index, delta≈1)
def synth_underlying(index_series, forward_series, cfg) -> list[Meta]: ...

# ---- snapshot (one ts, columnar numpy) ------------------------------------
class Snapshot:
    ts: int; index: float
    sym, kind, otype, strike, expiry: np.ndarray
    mark, bid, ask: np.ndarray
    iv, bid_iv, ask_iv, oi, volume: np.ndarray
    delta, gamma, vega, theta: np.ndarray
    tte, moneyness, valid: np.ndarray
    def __len__(self) -> int: ...
    def col(self, name: str) -> np.ndarray: ...         # any variable by name
    def row(self, sym: int) -> int | None: ...
    def mid(self, r: int) -> float: ...                 # (bid+ask)/2 or mark
    def spread(self, r: int) -> float: ...
    def expiries(self) -> np.ndarray: ...               # sorted unique expiry_ts
    def atm_strike(self, e: int) -> float | None: ...
    def select(self) -> "Selector": ...                 # composable filter, entry point

# fluent row selector — every filter chains, any variable, single or combined.
# returns row indices / syms / a sub-Snapshot; this is how legs are chosen.
class Selector:
    def kind(self, k: InstKind) -> "Selector": ...      # OPTION / PERP / FUTURE / SPOT
    def options(self) -> "Selector": ...
    def underlying(self) -> "Selector": ...             # spot/perp/future legs
    def calls(self) -> "Selector": ...
    def puts(self) -> "Selector": ...
    def otype(self, ot: OptType) -> "Selector": ...
    def dte(self, min_d: float, max_d: float) -> "Selector": ...        # days to expiry
    def expiry(self, e: int) -> "Selector": ...
    def iv(self, lo: float, hi: float) -> "Selector": ...
    def delta(self, lo: float, hi: float, abs: bool = True) -> "Selector": ...
    def moneyness(self, lo: float, hi: float) -> "Selector": ...
    def strike(self, lo: float, hi: float) -> "Selector": ...
    def oi_min(self, x: float) -> "Selector": ...
    def volume_min(self, x: float) -> "Selector": ...
    def spread_max(self, x: float) -> "Selector": ...
    def tradable(self) -> "Selector": ...                               # valid quotes
    def where(self, mask_or_fn) -> "Selector": ...                      # arbitrary predicate over any column
    def sort_by(self, field: str, desc: bool = False) -> "Selector": ...
    def top(self, n: int) -> "Selector": ...
    def nearest(self, field: str, target: float) -> "Selector": ...     # e.g. nearest delta / strike
    def rows(self) -> np.ndarray: ...
    def syms(self) -> np.ndarray: ...
    def first(self) -> int | None: ...
    def snapshot(self) -> "Snapshot": ...                               # sub-view of the selection

# ---- venue ----------------------------------------------------------------
@dataclass
class VenueConfig:  margin: Margin; contract_size: float; trade_fee: float; fee_cap: float
                    settle_fee: float; twap_mins: int; funding_series: np.ndarray | None
def payoff(margin, otype, strike, settle) -> float: ...
def trade_fee(cfg, premium) -> float: ...
def settle_fee(cfg, premium) -> float: ...
def settlement_index(index_series, expiry_ts, mins) -> float: ...   # TWAP window
# PnL per instrument kind (linear vs inverse handled inside):
def position_value(meta: Meta, qty, entry, mark, index, v: VenueConfig) -> float: ...
def funding(meta: Meta, qty, mark, rate, v: VenueConfig) -> float: ...   # PERP only, per bar

# ---- data -----------------------------------------------------------------
@dataclass
class DataConfig:   path: str; start: int | None; end: int | None
@dataclass
class ColumnMap:    ...                                 # stored -> canonical names
class Dataset:
    metas: dict[int, Meta]
    index_series: np.ndarray                            # (ts, index)
    def load(cfg: DataConfig, cols: ColumnMap = None) -> "Dataset": ...
    def normalize(self) -> None: ...                    # iv/100, parse names, derive expiry_ts
    def timestamps(self) -> np.ndarray: ...             # sorted unique
    def universe_at(self, ts: int) -> np.ndarray: ...   # live rows: listed, not expired, present
    def snapshot_at(self, ts: int) -> Snapshot: ...
    def expiries_at(self, ts: int) -> np.ndarray: ...   # syms expiring exactly at ts
    def index_at(self, ts: int) -> float: ...
    def iter_bars(self) -> Iterator[Snapshot]: ...      # streaming variant

# ---- assemble -------------------------------------------------------------
def assemble(ds: Dataset, ts: int, rows: np.ndarray) -> Snapshot: ...   # fills tte/moneyness/valid
def tte(expiry: np.ndarray, ts: int) -> np.ndarray: ...
def moneyness(forward: np.ndarray, index: float, strike: np.ndarray) -> np.ndarray: ...
def valid_mask(bid: np.ndarray, ask: np.ndarray, mark: np.ndarray) -> np.ndarray: ...

# ---- execution ------------------------------------------------------------
class FillModel(Protocol):
    def fill(self, o: Order, s: Snapshot, v: VenueConfig) -> Fill | None: ...
@dataclass
class DefaultFill:  half_spread_bps: float; slippage_bps: float; next_bar: bool
    def quote_price(self, o, s, r) -> float: ...
    def limit_reached(self, o, s, r) -> bool: ...

# ---- portfolio ------------------------------------------------------------
class Portfolio:
    cash: float; positions: dict[int, Position]
    def apply_fill(self, f: Fill, v: VenueConfig) -> None: ...
    def position(self, sym: int) -> Position | None: ...
    def net_qty(self, sym: int) -> float: ...
    def unrealized(self, s: Snapshot, v: VenueConfig) -> float: ...
    def equity(self, s: Snapshot, v: VenueConfig) -> Equity: ...
    def mark_to_market(self, s: Snapshot, v: VenueConfig) -> Equity: ...
    def agg_greeks(self, s: Snapshot) -> Greeks: ...    # underlying contributes delta=1
    def accrue_funding(self, s: Snapshot, v: VenueConfig) -> float: ...   # PERP positions
    def settle(self, sym: int, settle_index: float, v: VenueConfig) -> ExpiryEvent: ...  # options + dated futures
    def is_flat(self) -> bool: ...

class MarginModel(Protocol):
    def initial(self, p: Portfolio, s: Snapshot, v: VenueConfig) -> float: ...
    def maintenance(self, p: Portfolio, s: Snapshot, v: VenueConfig) -> float: ...
def liquidate(p: Portfolio, s: Snapshot, v: VenueConfig, m: MarginModel) -> list[Order]: ...

# ---- strategy / context ---------------------------------------------------
class Strategy(Protocol):
    def on_start(self, ctx: Context) -> None: ...
    def on_snapshot(self, s: Snapshot, ctx: Context) -> None: ...
    def on_fill(self, f: Fill, ctx: Context) -> None: ...
    def on_expiry(self, ev: ExpiryEvent, ctx: Context) -> None: ...
    def on_end(self, ctx: Context) -> None: ...
class Context:
    def submit(self, o: Order) -> None: ...
    def submit_combo(self, c: Combo) -> None: ...       # multi-leg, shared group id
    def cancel_all(self) -> None: ...
    def select(self) -> Selector: ...                   # live universe selector (this bar)
    def portfolio(self) -> Portfolio: ...               # read-only positions, cash, greeks
    def now(self) -> int: ...
    def g(self, name: str, offset: int = 0) -> float: ...   # any global chain aggregate
    def col(self, name: str) -> np.ndarray: ...             # any per-instrument variable, full chain
    def record(self, key: str, v: float) -> None: ...   # custom series, kept in RunResult
    def get(self, key: str) -> list: ...                # own recorded history
    def history(self, sym: int, field: str, n: int) -> np.ndarray: ...   # past n bars, one instrument
    def index_history(self, n: int) -> np.ndarray: ...  # past n index/spot values
    def state(self) -> dict: ...                        # free-form per-strategy scratch
    def drain_orders(self) -> list[Order]: ...

# rolling buffer feeding ctx.history / ctx.index_history (lookback bounded by config)
class History:
    def push(self, snap: Snapshot) -> None: ...
    def field(self, sym: int, field: str, n: int) -> np.ndarray: ...
    def index(self, n: int) -> np.ndarray: ...

# ---- conditions (arbitrary entry/exit logic over any variable) ------------
# Expr evaluates against a Scope; vectorized per-row or scalar. Fully composable.
class Scope:                                            # everything readable in a condition
    snap: Snapshot; ctx: Context; rows: np.ndarray; pos: Position | None
    def __getitem__(self, k): ...                       # any column / portfolio field by name
def F(name: str) -> Expr: ...                           # per-instrument field: F("iv"), F("delta"), F("tte")...
def U(name: str) -> Expr: ...                           # raw market: U("index"), U("forward")
def G(name: str, offset: int = 0) -> Expr: ...          # GLOBAL chain aggregate: G("pcr_oi"), G("atm_iv"), G("skew_25")
def PF(name: str) -> Expr: ...                          # portfolio: PF("delta"), PF("cash"), PF("unrealized")
def HIST(field: str, n: int, agg: str) -> Expr: ...     # history: sma/std/min/max/change/rank/zscore
def REC(key: str) -> Expr: ...                          # own recorded series
def IND(name: str, field: str = None, offset: int = 0) -> Expr: ...  # indicator value, e.g. IND("stoch","signal",1)
def T(field: str) -> Expr: ...   # time/calendar, any combination: minute hour dow dom week_of_month
                                 # month date dte days_in_trade bars_in_trade since_entry  (tz-aware)
def CONST(x) -> Expr: ...
def VAR(name: str, offset: int = 0) -> Expr: ...  # universal resolver: "iv" | "index" | "pcr_oi" | "pf.delta" | "t.hour" | indicator
# Expr algebra (every variable composes with every other, scalar or vector):
#   arithmetic   + - * /  abs min max  log  pct_change(n)  zscore(n)  rank(n)
#   comparison   < <= > >= ==  between(lo,hi)  cross_above  cross_below
#   boolean      &  |  ~
#   reducers     SUM MEAN MAX MIN COUNT over a Selector → scalar (vector → global)
#   offset       any operand [n] = n bars back
# broadcasting   scalar (G/U/PF/T/IND/CONST) auto-broadcasts across chain rows (F);
#                vector⊕vector is elementwise; reduce a vector to scalar with a reducer.
# drop to a raw lambda(Scope)->bool only when you want imperative escape; nothing else requires it

Action = Callable[[Context, Scope], None]               # submit / submit_combo / close / cancel / record
@dataclass(frozen=True)
class Rule:         when: Expr; do: Action
@dataclass(frozen=True)
class Sizing:       mode: str; value: float             # fixed qty / by greek / by % equity / by margin

# config-driven strategy: pick legs by Selector, fire on arbitrary entry/exit Rules.
# no code needed for a new structure — just selectors + rules + sizing as params.
@dataclass
class RuleStrategy(Strategy):
    select: list                                        # Selector specs → candidate legs
    entry: list[Rule]                                   # evaluated over candidates each bar
    exit:  list[Rule]                                   # evaluated over open positions each bar
    sizing: Sizing

# ---- variable access (every data variable, in any form) -------------------
# Every stored column is auto-exposed by name (F/G/col). Derived variables are
# registered once and then usable identically — no special-casing per variable.
VARS = {
  "instrument": [...stored option/underlying columns...],   # F(name) / Snapshot.col(name) / Selector.where
  "market":     ["index","forward"],                        # U(name)
  "global":     [...registered chain aggregates...],         # G(name) — see GlobalRegistry
}

# ---- global / chain aggregates (one scalar per bar from the whole chain) ---
class GlobalRegistry:
    def register(self, name: str, fn: Callable[[Snapshot], float]) -> None: ...  # any chain → scalar
    def names(self) -> list[str]: ...
class Market:
    def update(self, snap: Snapshot) -> None: ...           # compute all registered globals this bar
    def value(self, name: str, offset: int = 0) -> float: ..# backs G(name)
    def series(self, name: str) -> np.ndarray: ...          # full history → indicators / plots / records
# built-in globals (vectorized over chain), all overridable / extensible:
#   pcr_oi pcr_vol  atm_iv  skew_25 rr_25 fly_25  term_slope  vwiv
#   total_oi total_vol total_gamma total_vega  gex  max_pain  basis  rv_n  iv_rank

# ---- indicators (computed on any variable, multi-timeframe) ----------------
@dataclass(frozen=True)
class IndicatorSpec:  name: str; kind: str; params: dict; timeframe: str; source: str  # sma/ema/rsi/atr/stochastic/macd/bbands
#   source = any variable name: price/close, a market var ("index"), or a global ("atm_iv","pcr_oi")
#   → e.g. rank of atm_iv = iv_rank; sma of pcr_oi; atr of index
class Indicators:
    def add(self, spec: IndicatorSpec) -> None: ...
    def update(self, snap: Snapshot) -> None: ...       # per bar; resamples source to spec.timeframe
    def value(self, name, field=None, offset=0) -> float: ...   # backs IND(...)
def make_indicator(kind, params) -> Callable: ...       # registry; add new kinds here

# ---- declarative strategy spec (no-code config → compiled Strategy) --------
# serializable mirror of the primitives above; compiles to an engine Strategy.
@dataclass(frozen=True)
class Operand:        ref: str; field: str = None; offset: int = 0; const: float = None  # indicator|price|leg-field|const
@dataclass(frozen=True)
class Compare:        left: Operand; op: str; right: Operand        # above/below/ge/le/eq/cross_above/cross_below
@dataclass(frozen=True)
class Group:          match: str; items: list                       # "AND"|"OR"; items = Compare | Group (nested)
@dataclass(frozen=True)
class StrikeSpec:     method: str; value                            # moneyness:ATM/OTM+1 | premium:x | delta:x
@dataclass(frozen=True)
class ExpirySpec:     bucket: str                                   # this_week/next_week/this_month/next_month/dte<=n
@dataclass(frozen=True)
class LegRisk:        sl: float; sl_unit: str; tp: float; tp_unit: str; reentry: int    # %, premium, points; per-leg
@dataclass(frozen=True)
class LegSpec:        side: Side; kind: InstKind; otype: OptType; lots: float; lot_size: int
                      expiry: ExpirySpec; strike: StrikeSpec; risk: LegRisk = None
@dataclass(frozen=True)
class CalendarSpec:   tz: str = "UTC"; windows: list[tuple[str,str]] = None   # multiple intraday windows; None → all-day
                      date_from: str = None; date_to: str = None
                      exclude: list[str] = None; months: list[int] = None     # holiday dates; allowed months
@dataclass(frozen=True)
class SessionSpec:    holding: str; direction: str; start: str; squareoff: str
                      days: list[str]; linked_exit: bool            # holding intraday/overnight; linked → atomic combo
                      calendar: CalendarSpec = None                 # tz/windows/date-range/holidays; None → 24/7 UTC
@dataclass(frozen=True)
class RiskSpec:       daily_profit_cap: float = None; daily_loss_limit: float = None; max_trades_day: int = None
@dataclass(frozen=True)
class StrategySpec:   name: str; indicators: list[IndicatorSpec]; session: SessionSpec
                      entry: Group; exit: Group = None; legs: list[LegSpec] = []; risk: RiskSpec = None

def compile_spec(spec: StrategySpec) -> Strategy: ...   # lowers Group→Expr, LegSpec→Combo, caps→guards
def to_expr(g: Group) -> Expr: ...                      # Group/Compare → Expr tree
def resolve_expiry(snap, e: ExpirySpec) -> int: ...
def resolve_strike(snap, otype, expiry_ts, s: StrikeSpec) -> int: ...   # by moneyness/premium/delta
def in_session(ts, session) -> bool: ...                # tz, windows, days, date range, holidays, months
def time_fields(ts, tz, entry_ts=None) -> dict: ...     # backs T(...): minute/hour/dow/dom/week_of_month/month/date/dte/days_in_trade/...
def load_spec(path) -> StrategySpec: ...                # TOML/JSON
def save_spec(spec, path) -> None: ...

# ---- engine / metrics / runner --------------------------------------------
class Engine:
    def __init__(self, ds, strat, fill, venue, cash, margin=None): ...
    def run(self) -> RunResult: ...

# closed round-trip (FIFO match of fills + settlements), pnl net of fees
@dataclass(frozen=True)
class Trade:        sym: int; qty: float; entry: float; exit: float
                    pnl: float; fees: float; entry_ts: int; exit_ts: int

@dataclass
class Overview:     total_pnl: float; max_dd: float; trades: int
                    win_rate: float; avg_profit: float; avg_loss: float
@dataclass
class Side_:        pct: float; avg: float; max_day: float; max_streak: int; max_streak_pnl: float
@dataclass
class Ratios:       sharpe: float; sortino: float; calmar: float
                    expectancy: float; profit_factor: float; risk_reward: float
@dataclass
class Costs:        fee_per_trade: float; total_fees: float; trades: int
@dataclass
class Stats:        overview: Overview; winning: Side_; losing: Side_; ratios: Ratios; costs: Costs

@dataclass
class RunResult:    equity: list; fills: list; trades: list[Trade]; expiries: list
                    greeks: list; records: dict; stats: Stats

# ---- metrics --------------------------------------------------------------
def match_trades(fills, expiries) -> list[Trade]: ...   # FIFO pairing, pnl net of fees
def compute_stats(equity, trades) -> Stats: ...
def equity_to_returns(equity) -> np.ndarray: ...
def daily_pnl(equity) -> np.ndarray: ...                # per-day P&L (for max profit/loss day)
def max_drawdown(equity) -> float: ...
def streaks(trades) -> tuple[int, float, int, float]: ..# (win_streak, win_pnl, loss_streak, loss_pnl)
def sharpe(r) -> float: ...
def sortino(r) -> float: ...
def calmar(equity) -> float: ...                        # cagr / max_dd
def expectancy(trades) -> float: ...                    # win%*avg_win − loss%*|avg_loss|
def profit_factor(trades) -> float: ...                 # Σwin / Σ|loss|
def risk_reward(trades) -> float: ...                   # avg_win / |avg_loss|

# ---- plots (research): return tidy series; rendering optional, no core dep -
def report(res: RunResult) -> dict: ...                 # all series as frames, for notebooks
def plot_equity(res): ...        # equity BTC/USD
def plot_drawdown(res): ...
def plot_trade_pnl(res): ...     # per-trade P&L over time
def plot_pnl_hist(res): ...      # win/loss distribution
def plot_daily_pnl(res): ...     # daily bars
def plot_rolling_sharpe(res, n): ...
def plot_greeks(res): ...        # delta/gamma/vega/theta exposure over time
def plot_exposure(res): ...      # position count / notional over time
def plot_streaks(res): ...
def plot_monthly(res): ...       # monthly returns heatmap

@dataclass
class RunConfig:    data: DataConfig; venue: VenueConfig; fill: FillModel; strategy: type; params: dict; cash: float; seed: int
def run_one(cfg: RunConfig) -> RunResult: ...
def run_sweep(grid: list[RunConfig], workers: int = 0) -> list[RunResult]: ...   # multiprocessing pool
def expand_grid(base: RunConfig, axes: dict[str, list]) -> list[RunConfig]: ...  # cartesian over params
def cross(strategies: list[tuple[type, dict]], data, venue, fill, cash) -> list[RunConfig]: ...  # strategies × param grids
def config_hash(cfg: RunConfig) -> str: ...
def save_result(h: str, res: RunResult) -> None: ...
def load_result(h: str) -> RunResult | None: ...

@dataclass
class Study:        configs: list[RunConfig]
    def run(self, workers: int = 0) -> StudyResult: ...      # cached, resumable
@dataclass
class StudyResult:  results: list[RunResult]
    def leaderboard(self, metric: str) -> list: ...          # rank by any Stats field
    def frame(self) -> "DataFrame": ...                      # params × metrics table
```

## Pseudocode

```
RUN(D, S, F, V, c0, M)
  P ← Portfolio(c0, {}); Q ← []; res ← {}; H ← History()
  S.on_start(ctx)
  for ts in D.timestamps():
    snap ← ASSEMBLE(D, ts); H.push(snap); MKT.update(snap)   # 0. lookback + global chain aggregates
    for sym in D.expiries_at(ts):                 # 1. settle expiries
      ev ← SETTLE(P, sym, D, V); res.expiries += ev; S.on_expiry(ev)
    for grp in group_by(Q, .group):              # 2. fill prior-bar orders
      fills ← [F.fill(o, snap, V) for o in grp]
      if atomic(grp) and any(f is None for f in fills): continue   # all-or-none
      for f in fills if f:
        APPLY_FILL(P, f, V); res.fills += f; S.on_fill(f)
    Q ← []
    P.accrue_funding(snap, V)                     # 3. perp funding (cash)
    eq ← P.mark_to_market(snap, V)                #    mark (options + underlying)
    if M and eq.btc < M.maintenance(P,snap,V):    #    forced liquidation
      for o in liquidate(P,snap,V,M): Q += o
    res.equity += (ts, eq); res.greeks += (ts, P.agg_greeks(snap))
    S.on_snapshot(snap, ctx)                      # 4. strategy → Q  (full state + H)
    Q ← ctx.drain_orders()
  S.on_end(ctx)
  res.trades ← match_trades(res.fills, res.expiries)
  res.stats  ← compute_stats(res.equity, res.trades); return res
  # order: settle → fill → funding → mark → strategy; orders fill on the next bar

LOAD(cfg, cols)
  tbl ← read_parquet(cfg.path, filter date in [start,end])
  rename(tbl, cols); tbl.iv /= 100
  for name in unique(instrument_name):
    otype,strike,expiry ← parse_name(name); sym ← intern(name)
    metas[sym] ← Meta(sym,name,otype,strike,expiry,currency)
  index_series ← unique (ts, index_price) sorted
  build time index: ts -> row range
  return Dataset

UNIVERSE_AT(D, ts)
  rows ← D.rows_at(ts)
  return [r for r in rows if expiry[r] > ts]      # listed, not expired

ASSEMBLE(D, ts)
  rows ← UNIVERSE_AT(D, ts); idx ← D.index_at(ts)
  tte ← (expiry[rows]-ts)/YEAR_NS                 # vectorized
  fwd ← forward[rows] where>0 else idx
  moneyness ← fwd/strike[rows]
  valid ← (bid[rows]>0) | (ask[rows]>0)
  return Snapshot{cols[rows], idx, tte, moneyness, valid}

FILL(o, snap, V)
  r ← snap.row(o.sym); if r is None or not valid[r]: return None
  if o.kind==MARKET:
    base ← ask[r] if o.side==BUY else bid[r]
    if base==0: base ← mark[r]*(1 + half_spread*dir)        # dir=+1 buy, -1 sell
    px ← base*(1 + slippage*dir)
  else:                                                     # LIMIT(o.limit)
    hit ← (o.side==BUY and low[r]<=o.limit) or (o.side==SELL and high[r]>=o.limit)
    if not hit: return None
    px ← o.limit
  return Fill(o.sym, o.side, o.qty, px, trade_fee(V,px), snap.ts)

APPLY_FILL(P, f, V)
  sq ← f.qty if f.side==BUY else -f.qty           # signed
  P.cash += -(sq*f.price)*V.contract_size - f.fee # pay to buy, receive to sell
  pos ← P.positions.get(f.sym) or Position(f.sym,0,0,0)
  nq ← pos.qty + sq
  if pos.qty==0 or sign(sq)==sign(pos.qty):       # opening / adding
    pos.avg_entry ← (pos.qty*pos.avg_entry + sq*f.price)/nq if nq!=0 else 0
  else:                                           # reducing / closing
    closed ← min(|sq|,|pos.qty|)
    pos.realized += closed*(f.price-pos.avg_entry)*sign(pos.qty)*V.contract_size
    if sign(nq)!=sign(pos.qty) and nq!=0: pos.avg_entry ← f.price   # flipped
  pos.qty ← nq
  if nq==0: pos.avg_entry ← 0
  P.positions[f.sym] ← pos

UNREALIZED(P, snap, V)
  u ← 0
  for sym,pos in P.positions:
    r ← snap.row(sym); if r is None: continue
    u += pos.qty*(mark[r]-pos.avg_entry)*V.contract_size
  return u                                         # unit = BTC (inverse) | USDC (linear)

MARK_TO_MARKET(P, snap, V)
  base ← P.cash + UNREALIZED(P, snap, V)
  btc,usd ← (base, base*snap.index) if V.margin==INVERSE else (base/snap.index, base)
  return Equity(btc, usd, dd=running_drawdown(btc))

AGG_GREEKS(P, snap)
  g ← Greeks(0,0,0,0)
  for sym,pos in P.positions:
    r ← snap.row(sym); if r is None: continue
    g.delta += pos.qty*delta[r]; g.gamma += pos.qty*gamma[r]
    g.vega  += pos.qty*vega[r];  g.theta += pos.qty*theta[r]
  return g

SETTLE(P, sym, D, V)
  pos ← P.positions[sym]; if pos.qty==0: return None
  Sx ← settlement_index(D.index_series, expiry(sym), V.twap_mins)
  pay ← payoff(V.margin, otype(sym), strike(sym), Sx)
  fee ← settle_fee(V, pay) if pay>0 else 0
  P.cash += (pos.qty*pay)*V.contract_size - sign(pos.qty)*fee
  pos.realized += pos.qty*(pay-pos.avg_entry)*V.contract_size - fee
  del P.positions[sym]
  return ExpiryEvent(sym, Sx, pay, fee, expiry(sym))

SETTLEMENT_INDEX(series, expiry_ts, mins)
  w ← series[expiry_ts-mins*60e9 <= ts <= expiry_ts]
  return time_weighted_mean(w.index)

COMPUTE_STATS(equity, trades)
  r ← equity_to_returns(equity); d ← daily_pnl(equity)
  W ← [t for t in trades if t.pnl>0]; L ← [t for t in trades if t.pnl<=0]
  ws,wp,ls,lp ← streaks(trades)
  overview ← Overview(total_pnl=Σt.pnl, max_dd=max_drawdown(equity), trades=|trades|,
                      win_rate=|W|/|trades|, avg_profit=mean(W.pnl), avg_loss=mean(L.pnl))
  winning  ← Side_(pct=|W|/|trades|, avg=mean(W.pnl), max_day=max(d>0), max_streak=ws, max_streak_pnl=wp)
  losing   ← Side_(pct=|L|/|trades|, avg=mean(L.pnl), max_day=min(d<0), max_streak=ls, max_streak_pnl=lp)
  ratios   ← Ratios(sharpe(r), sortino(r), calmar(equity),
                    expectancy(trades), profit_factor(trades), risk_reward(trades))
  costs    ← Costs(fee_per_trade=Σt.fees/|trades|, total_fees=Σt.fees, trades=|trades|)
  return Stats(overview, winning, losing, ratios, costs)

EXPAND_COMBO(c, snap)                              # multi-leg → orders, shared group
  g ← new_group_id()
  return [Order(leg.sym, leg.side, c.qty*leg.ratio, c.kind, c.limit, group=g) for leg in c.legs]

SELECT(snap, filters)                             # composable; any variable, any combination
  rows ← all rows                                 # options + underlying (spot/perp/future)
  for f in filters: rows ← rows[f(snap, rows)]    # kind/dte/iv/delta/moneyness/oi/where/...
  if sort: rows ← argsort(snap.col(field)[rows], desc)
  if top:  rows ← rows[:n]
  return rows

RULE_STRATEGY.on_snapshot(snap, ctx)              # config-driven, any condition any variable
  cands ← SELECT(snap, self.select)
  for r in self.exit:                             # exits first
    scope ← Scope(snap, ctx, rows=open_position_rows)
    for p where r.when(scope): r.do(ctx, scope)   # e.g. close on PnL / delta / iv / time / any expr
  for r in self.entry:
    scope ← Scope(snap, ctx, rows=cands)
    if r.when(scope): r.do(ctx, scope)            # size via self.sizing, submit order/combo

COMPILED_SPEC.on_snapshot(snap, ctx)              # what compile_spec(StrategySpec) runs
  mkt.update(snap); ind.update(snap)              # globals first, then indicators (may source globals)
  if not in_session(snap.ts, session) or risk_tripped(ctx, risk):
    if holding==intraday and past_squareoff: close_all(ctx)
    return
  # in_session(ts): tz-convert → day in days → ts in any window → date in [from,to] → not holiday → month allowed
  for pos in ctx.portfolio().positions:           # per-leg SL/TP/re-entry + exit group
    if leg_risk_hit(pos, snap) or to_expr(exit)(scope(pos)): close(ctx, pos)
  if flat_or_allowed(direction) and to_expr(entry)(scope(snap)):
    legs ← []
    for L in spec.legs:                           # resolve each declarative leg → concrete sym
      e ← resolve_expiry(snap, L.expiry); k ← resolve_strike(snap, L.otype, e, L.strike)
      legs += Leg(k, L.side, L.lots*L.lot_size)
    ctx.submit_combo(Combo(legs, qty=1, atomic=session.linked_exit))

EVAL(expr, scope)                                 # vectorized over scope.rows, or scalar
  F→chain col | U→market | G→global(mkt) | PF→portfolio | HIST→history | T→time | IND→indicator | REC→recorded
  combinators → boolean array/scalar
  return expr(scope)

GLOBALS(snap, mkt)                                # one scalar per registered name, per bar
  for name, fn in registry: mkt[name] ← fn(snap)  # e.g. pcr_oi = Σoi(puts)/Σoi(calls)
  # built-ins are just pre-registered fns; users register(name, fn) for anything else
  # every global is series()-able → usable as an indicator source, plot, or recorded var

RUN_SWEEP(grid, workers)                           # params × strategies, cached, resumable
  pool(workers): for g in grid:
    h ← config_hash(g)
    res ← load_result(h) or run_one(g); save_result(h, res)
  return res[]
```

## Examples

Stability Condor (as a `StrategySpec` literal):

```python
StrategySpec(
  name="Stability Condor",
  indicators=[
    IndicatorSpec("sma_20",   "sma",        {"window":20},            "15m", "close"),
    IndicatorSpec("atr_14",   "atr",        {"window":14},            "15m", "close"),
    IndicatorSpec("stoch",    "stochastic", {"window":14,"smooth":3}, "15m", "close"),
  ],
  session=SessionSpec(holding="intraday", direction="long",
                      start="09:20", squareoff="15:15",
                      days=["MON","TUE","WED","THU","FRI"], linked_exit=True),
  entry=Group("AND", [
    Compare(Operand("stoch","stoch",1), "ge",         Operand(const=35)),
    Compare(Operand("stoch","stoch",1), "le",         Operand(const=65)),
    Compare(Operand("stoch","stoch",1), "gt",         Operand("stoch","stoch_signal",1)),
    Compare(Operand("atr_14",offset=1), "lt",         Operand("sma_20",offset=1)),
  ]),
  exit=None,                                          # held to square-off
  legs=[
    LegSpec(SELL, OPTION, CALL, 1, 65, ExpirySpec("this_month"), StrikeSpec("moneyness","ATM"),   LegRisk(25,"%",None,"",0)),
    LegSpec(SELL, OPTION, PUT,  1, 65, ExpirySpec("this_month"), StrikeSpec("moneyness","ATM"),   LegRisk(25,"%",None,"",0)),
    LegSpec(BUY,  OPTION, CALL, 1, 65, ExpirySpec("this_month"), StrikeSpec("moneyness","OTM+1"), LegRisk(25,"%",None,"",0)),
    LegSpec(BUY,  OPTION, PUT,  1, 65, ExpirySpec("this_month"), StrikeSpec("moneyness","OTM+1"), LegRisk(25,"%",None,"",0)),
  ],
  risk=RiskSpec(daily_profit_cap=50000, daily_loss_limit=8000, max_trades_day=100),
)
```

Run it:

```
spec ← load_spec("condor.toml")
res  ← run_one(RunConfig(data, venue, fill, strategy=compile_spec(spec), params={}, cash=c0, seed=0))
print(res.stats.overview, res.stats.ratios); plot_equity(res)
```

Same condor, imperative escape hatch (when the spec can't express something):

```
class Condor(Strategy):
  def on_snapshot(snap, ctx):
    if not session_open(ctx.now()) or not ctx.portfolio().is_flat(): return
    if not (35 <= ind("stoch","stoch",1) <= 65 and ind("stoch","stoch",1) > ind("stoch","stoch_signal",1)
            and ind("atr_14",o=1) < ind("sma_20",o=1)): return
    e   ← snap.select().options().expiry(this_month()).first_expiry()
    atm ← snap.select().expiry(e).nearest("moneyness", 1.0)            # ATM
    ce1 ← snap.select().calls().expiry(e).nearest("strike", strike(atm)+step)   # OTM+1
    pe1 ← snap.select().puts().expiry(e).nearest("strike", strike(atm)-step)
    ctx.submit_combo(Combo([Leg(call(atm),SELL,1), Leg(put(atm),SELL,1),
                            Leg(ce1,BUY,1),         Leg(pe1,BUY,1)], qty=1, atomic=True))
```

Delta-hedged short straddle (mixes option + perp legs in one combo):

```
atm ← snap.select().options().expiry(weekly).nearest("delta", 0.5)
d   ← ctx.portfolio().agg_greeks(snap).delta
hedge ← snap.select().kind(PERP).first()
ctx.submit_combo(Combo([Leg(atm_call,SELL,1), Leg(atm_put,SELL,1),
                        Leg(hedge, BUY if d<0 else SELL, abs(d))], qty=1))
```

Custom exit on any variable (close when IV spikes or book delta breaches):

```
exit_rule ← Rule(when=(PF("delta").abs() > 5) | (F("iv") > IND("iv_ma",o=0)*1.5),
                 do=lambda ctx,sc: ctx.cancel_all_and_close())
```

Sweep — Condor across strike distance × SL (parallel, ranked):

```
grid ← expand_grid(base_condor_cfg, {"otm_dist":[1,2,3], "sl_pct":[20,25,30]})
study ← Study(grid).run(workers=8)
study.leaderboard("sharpe")[:10]; study.frame()
```

Short straddle, IV-rank gated, overnight, premium-strike (declarative — shows `IND` on iv, premium method, per-leg TP/SL):

```python
StrategySpec(
  name="IV-rank short straddle",
  indicators=[IndicatorSpec("ivrank","rank",{"window":480},"5m","mark_iv")],
  session=SessionSpec("overnight","short","00:00","00:00",["MON","TUE","WED","THU","FRI","SAT","SUN"], linked_exit=False),
  entry=Group("AND",[ Compare(Operand("ivrank"),"gt",Operand(const=0.8)) ]),   # sell vol only when rich
  exit =Group("OR", [ Compare(Operand("ivrank"),"lt",Operand(const=0.4)),
                      Compare(Operand("dte"),"le",Operand(const=1)) ]),        # cover or roll near expiry
  legs=[
    LegSpec(SELL,OPTION,CALL,1,1,ExpirySpec("this_week"),StrikeSpec("delta",0.50),LegRisk(50,"%",100,"%",1)),
    LegSpec(SELL,OPTION,PUT, 1,1,ExpirySpec("this_week"),StrikeSpec("delta",0.50),LegRisk(50,"%",100,"%",1)),
  ],
  risk=RiskSpec(daily_loss_limit=5000, max_trades_day=4),
)
```

Calendar spread — sell near, buy far, same strike (imperative — two expiries):

```
near ← snap.select().options().expiry(this_week()).first_expiry()
far  ← snap.select().options().expiry(next_week()).first_expiry()
k    ← snap.select().calls().expiry(near).nearest("delta", 0.30)
kf   ← snap.select().calls().expiry(far ).nearest("strike", strike(k))
ctx.submit_combo(Combo([Leg(k,SELL,1), Leg(kf,BUY,1)], qty=1, atomic=True))
```

Gamma scalp — short straddle, re-hedge delta with perp every bar past a band (imperative — per-bar rebalance):

```
class GammaScalp(Strategy):
  def on_snapshot(snap, ctx):
    if ctx.portfolio().is_flat():
      atm ← snap.select().options().expiry(weekly).nearest("delta",0.5)
      ctx.submit_combo(Combo([Leg(call(atm),SELL,1),Leg(put(atm),SELL,1)],1))
    d ← ctx.portfolio().agg_greeks(snap).delta
    if abs(d) > band:                                  # flatten residual delta with perp
      perp ← snap.select().kind(PERP).first()
      ctx.submit(market(perp, SELL if d>0 else BUY, abs(d)))
```

Protective collar — long perp + long put + short call (declarative — mixes underlying and options):

```python
StrategySpec(
  name="Collar",
  indicators=[],
  session=SessionSpec("overnight","long","00:00","00:00",["MON","TUE","WED","THU","FRI","SAT","SUN"], linked_exit=True),
  entry=Group("AND",[ Compare(Operand("index"),"cross_above",Operand("sma_200")) ]),
  legs=[
    LegSpec(BUY, PERP,  NONE, 1,1, ExpirySpec("none"),       StrikeSpec("none",0),        None),
    LegSpec(BUY, OPTION,PUT,  1,1, ExpirySpec("this_month"), StrikeSpec("moneyness","OTM+2"), None),
    LegSpec(SELL,OPTION,CALL, 1,1, ExpirySpec("this_month"), StrikeSpec("moneyness","OTM+2"), None),
  ],
  risk=None,
)
```

Ratio backspread — sell 1 ATM, buy 2 OTM (imperative — `Leg.ratio`):

```
atm  ← snap.select().calls().expiry(monthly).nearest("delta",0.50)
otm  ← snap.select().calls().expiry(monthly).nearest("delta",0.25)
ctx.submit_combo(Combo([Leg(atm,SELL,1.0), Leg(otm,BUY,2.0)], qty=1, atomic=True))
```

Hold-to-expiry — sell a put, no exit rule; engine settles it (shows settlement path):

```
if ctx.portfolio().is_flat():
  p ← snap.select().puts().expiry(this_week()).nearest("delta",0.20)
  ctx.submit(market(p, SELL, 1))         # on_expiry fires with inverse/linear payoff at S_T
```

Global chain variables in any form — condition on PCR + skew, register a custom global, take an indicator of it, record it:

```
mkt.register("call_oi_share", lambda s: sum(s.oi[s.calls()]) / sum(s.oi))   # any chain → scalar
ind.add(IndicatorSpec("pcr_ma","sma",{"window":48},"5m","pcr_oi"))          # indicator sourced from a global
entry ← Group("AND",[ Compare(Operand("G","pcr_oi"),"gt",Operand(const=1.2)),
                      Compare(Operand("G","skew_25"),"gt",Operand("pcr_ma")),
                      Compare(Operand("G","iv_rank"),"gt",Operand(const=0.7)) ])
# imperative: ctx.g("gex"), ctx.g("atm_iv",offset=1); ctx.record("pcr", ctx.g("pcr_oi")) → plots
# selector: snap.select().where(F("iv") > G("atm_iv")*1.1)                     # per-leg vs global
```

Time as a first-class condition — enter only late morning, on expiry day, after 3 bars in trade exit:

```
entry ← Group("AND",[ Compare(Operand("T","hour"),"ge",Operand(const=10)),
                      Compare(Operand("T","dte"),"le",Operand(const=1)),
                      Compare(Operand("stoch","stoch"),"lt",Operand(const=20)) ])
exit  ← Group("OR", [ Compare(Operand("T","days_in_trade"),"ge",Operand(const=3)),
                      Compare(Operand("T","hour"),"ge",Operand(const=15)) ])
# session: tz="Asia/Kolkata", two windows, weekdays only, skip holidays
SessionSpec("intraday","both","09:20","15:15",["MON","TUE","WED","THU","FRI"], True,
  CalendarSpec(tz="Asia/Kolkata", windows=[("09:20","11:30"),("13:30","15:15")],
               date_from="2026-01-01", exclude=["2026-03-21"], months=[1,2,3]))
```

## Deps

`numpy`, `pyarrow`. `pandas` optional (results only).

## Module plan

```text
btcbt/
  __init__.py
  types.py          # enums + dataclasses: Order, Leg, Combo, Fill, Position, ExpiryEvent, Equity, Greeks
  venue.py          # payoff, fees, settlement_index, position_value, funding, currency conversion
  data.py           # Parquet load, normalize, parse_name, Dataset, symbol map, bar_offsets
  snapshot.py       # Snapshot, Selector
  assemble.py       # tte, moneyness, valid_mask
  execution.py      # FillModel, DefaultFill (next-bar market fills)
  portfolio.py      # apply_fill, MTM, unrealized, agg_greeks, settle
  engine.py         # chronological loop, order queue, lifecycle
  metrics.py        # match_trades, compute_stats, streaks, drawdown, ratios
  runner.py         # run_one, run_sweep, config_hash, result cache
  cli.py            # python -m btcbt.run / .sweep
  # ---- deferred (Phase 2-4): globals.py indicators.py conditions.py spec.py plots.py margin.py
strategies/
  short_put.py  short_straddle.py
configs/
  base.toml  straddle_grid.toml
tests/
  test_venue.py test_data.py test_selector.py test_execution.py test_portfolio.py test_engine_no_lookahead.py
```

## Build phases

Build Phase 1 fully before anything else. The rich variable model / DSL / globals are the destination, not the start.

```text
Phase 1 — minimal fast backtest   (one Python strategy → fills/equity/stats)
  types · venue · data+snapshot+selector · portfolio · execution (next-bar market) ·
  engine (no-lookahead queue) · metrics · runner+CLI · one strategy
  market orders only; options + at most one optional hedge instrument

Phase 2 — research ergonomics
  multiprocessing sweeps · config-hash cache · param grids · RunResult export ·
  records API · rolling helpers (sma/ema/std/zscore/rank) ·
  basic globals (pcr_oi/pcr_vol/atm_iv/total_oi/total_vol/agg greeks) · plots/report ·
  example strategies (short put, straddle, condor, delta hedge)

Phase 3 — accuracy
  richer slippage · full hedge instrument (perp/spot) + funding · margin model + liquidation ·
  approximate limit orders (clearly marked)

Phase 4 — platform
  declarative StrategySpec + compile_spec · condition DSL (Expr/VAR/Group) ·
  indicator registry + multi-timeframe · full global set (skew/gex/max_pain/term_slope/...) ·
  streaming Dataset.iter_bars · notebook/UI templates
```

## Defer / simplify

| Item | Decision | Reason |
|---|---|---|
| `Expr`/`VAR`/universal resolver, `StrategySpec`/`compile_spec`, nested groups | Phase 4 | Second product layer; the Python strategy API solves the immediate need with less complexity. |
| Indicator framework, multi-timeframe resampling | Phase 4 (rolling helpers in P2) | Start with small rolling windows; add framework when a strategy needs it. |
| Full global set (`skew_25`, `gex`, `max_pain`, `fly_25`) | Phase 4 (basics in P2) | Keep `atm_iv`, totals, PCR, agg greeks first. |
| Plots as core module | Phase 2 | Return tidy result frames first; plotting can live in notebooks. |
| Margin model + forced liquidation | Phase 3 | Not needed to validate research; add after accounting is trusted. |
| Spot/perp/future full instrument system | Simplify → one hedge instrument (Phase 1 optional), full in Phase 3 | Start with options + one optional `BTC-PERP`/spot hedge; dated futures later. |
| Limit orders (high/low touch) | Phase 3, marked approximate | Snapshot-only bid/ask/mark can't prove an intrabar limit was touched; needs tick/OHLC. |
| Out-of-core streaming | Phase 4 | In-memory canonical arrays are simpler/faster first; add if data size demands. |

## Fast path

```text
1. each canonical column = one numpy array
2. rows for a timestamp are contiguous; snapshots are views, not copied frames
3. integer ids for sym and enum fields
4. boolean masks for selection; no per-instrument Python loops in strategy helpers
5. portfolio loops iterate only open positions
6. parallelize sweeps with multiprocessing, never one run's time loop
7. cache normalized datasets and results by config hash
8. profile before Numba; add Numba only for confirmed hot kernels
```

Expected hotspots: snapshot assembly · selector/nearest · MTM over open positions · trade matching · stats over long equity. Not hotspots: strategy dispatch (once/bar) · fee calc · small open-position loop.

## Correctness tests

```text
venue       inverse + linear call/put payoff; TWAP window; fee cap
data        instrument parser; iv%→decimal; expiry 08:00 UTC; zero bid/ask validity; sorted ts + contiguous bar_offsets
selector    calls/puts; dte bounds; nearest strike/delta; tradable
execution   buy@ask / sell@bid; invalid row no fill; order on t cannot fill before t+1; atomic combo all-or-none
portfolio   open/add/reduce/close; flip long↔short; realized/unrealized; settlement closes; equity reconciliation
engine      no-lookahead (two-bar fixture); hold-to-expiry; early-close; deterministic (same seed+data)
```

## Open questions

Resolve before Phase 1:

```text
1. cash settlement currency: BTC, USDC, or separate wallets per currency?
2. data shape: one row per 5-min bar per LISTED option, or only active/quoted instruments?
3. tick/OHLC bid/ask anywhere? if not, limit orders stay out of MVP.
4. first hedge instrument needed: spot index only, BTC-PERP, or none?
5. full dataset size (rows / GB)? decides whether in-memory arrays suffice for Phase 1.
```

## First implementation order

```text
1. data.py + snapshot.py        5. engine.py                     9. multiprocessing sweeps
2. venue.py                     6. strategies/short_put.py      10. cache + results export
3. portfolio.py                 7. metrics.py
4. execution.py                 8. runner.py + cli.py
```

Do not start with the DSL. Fastest route to useful research: a stable Python strategy API on a vectorized chain engine.
