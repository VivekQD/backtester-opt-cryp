"""Static perp bridge after a 50% per-leg stop trigger, with spread-exit grid."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lightweight_backtester.config import ResearchConfig, to_ns
from lightweight_backtester.data import MarketDataStore
from lightweight_backtester.execution import ExecutionModel
from lightweight_backtester.strategies import OptionStructureStrategy
from lightweight_backtester.types import OptionType, Side
from lightweight_backtester.trade_tape import TapeFill

NS = 1_000_000_000; DAY_NS = 86_400 * NS


def val(path, name, i, default=0.0):
    try:
        x = float(path[name][i]); return x if np.isfinite(x) else default
    except (KeyError, IndexError, TypeError, ValueError): return default


class PerpBars:
    def __init__(self, path):
        t = pq.read_table(path)
        self.ts = t["datetime"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        self.open = t["open"].to_numpy(zero_copy_only=False).astype(float)

    def next_open(self, ts):
        i = int(np.searchsorted(self.ts, ts, side="right"))
        return (int(self.ts[i]), float(self.open[i])) if i < len(self.ts) else (None, None)


class OptionTradeIndex:
    """One-pass index of non-block prints for only the selected contracts."""
    def __init__(self, root, instruments, start_ns, end_ns, slippage_bps=1.):
        self.slippage_bps=slippage_bps; grouped={x:[] for x in instruments}
        root=Path(root); values=pa.array(sorted(instruments),type=pa.large_string())
        start=dt.datetime.fromtimestamp(start_ns/NS,dt.timezone.utc).date()
        end=dt.datetime.fromtimestamp(end_ns/NS,dt.timezone.utc).date()
        for year in range(start.year,end.year+1):
            for path in sorted((root/f"currency=BTC"/f"year={year}").glob("BTC_trades_*.parquet")):
                day=dt.date.fromisoformat(path.stem[-10:])
                if day<start or day>end:continue
                # Read the physical file directly. ``pq.read_table(path)`` may
                # infer Hive partitions from ``currency=BTC`` and attempt to
                # merge that partition field with an encoded in-file currency
                # column, which fails on newer PyArrow versions.
                t=pq.ParquetFile(path).read(columns=["date","instrument_name","price","block_trade_id"])
                mask=pc.and_(pc.is_in(t["instrument_name"],value_set=values),pc.is_null(t["block_trade_id"]))
                t=t.filter(mask)
                if not len(t):continue
                dates=t["date"].to_numpy().astype("datetime64[ns]").astype(np.int64)
                names=t["instrument_name"].to_pylist();prices=t["price"].to_numpy(zero_copy_only=False)
                for ts,name,price in zip(dates,names,prices):
                    if price and np.isfinite(price):grouped[name].append((int(ts),float(price)))
        self.data={}
        for name,rows in grouped.items():
            rows.sort();self.data[name]=(np.asarray([x[0] for x in rows],dtype=np.int64),
                                         np.asarray([x[1] for x in rows],dtype=float))

    def fill(self,instrument,side,start_ns,quantity,max_end_ns=None):
        ts,px=self.data.get(instrument,(np.array([],dtype=np.int64),np.array([],dtype=float)))
        i=int(np.searchsorted(ts,start_ns,side="left"))
        if i>=len(ts) or (max_end_ns is not None and ts[i]>=max_end_ns):return None
        price=float(px[i])*(1+self.slippage_bps*1e-4 if side==Side.BUY else 1-self.slippage_bps*1e-4)
        return TapeFill(price,quantity,int(ts[i]),int(ts[i]),1)


def perp_round_trip(qty, entry, exit_, fee_bps, slip_bps):
    if qty > 0:
        ep, xp = entry * (1 + slip_bps * 1e-4), exit_ * (1 - slip_bps * 1e-4)
    else:
        ep, xp = entry * (1 - slip_bps * 1e-4), exit_ * (1 + slip_bps * 1e-4)
    # Linear USDT-perp P&L converted to BTC at exit; fees converted per fill.
    gross = qty * (xp - ep) / xp
    fees = abs(qty) * fee_bps * 1e-4 * 2
    return gross - fees, ep, xp, fees


def config():
    p = Path(__file__).resolve().parents[1] / "configs/btc_weekly_straddle_is.toml"
    c = ResearchConfig.from_toml(str(p))
    return replace(c, start_ns=to_ns("2021-01-01T00:00:00+00:00"),
                   end_ns=to_ns("2024-12-31T23:59:59+00:00"),
                   entry_frequency="daily", dte_min=7, dte_max=7,
                   expiry_weekday=4, structure="short_straddle",
                   stop_loss_pct=.50, enforce_margin=False)


def simulate(intent, store, perp, spread_limit, max_wait_minutes,
             option_slip_bps=1., perp_slip_bps=5., perp_fee_bps=5., tape=None):
    expiry = min(x.contract.expiry_ns for x in intent.legs)
    paths = [store.load_contract_path(x.contract, intent.entry_target_ns, expiry) for x in intent.legs]
    common = paths[0]["date"]
    for p in paths[1:]: common = np.intersect1d(common, p["date"], assume_unique=False)
    common = common[(common > intent.entry_target_ns) & (common <= expiry)]
    if not common.size: return None
    maps = [{int(t): i for i, t in enumerate(p["date"])} for p in paths]
    entry_ts = int(common[0]); ex = ExecutionModel(option_slip_bps)
    entry_price = {}; option_cash = 0.; fees = 0.
    for leg, path, mp in zip(intent.legs, paths, maps):
        i = mp[entry_ts]
        px = ex.fill_price(Side.SELL, val(path,"bid_price",i), val(path,"ask_price",i), val(path,"mark_price",i))
        if px is None: return None
        entry_price[leg.contract.instrument] = px; option_cash += px
        fees += min(.0003, .125 * px)
    states = {x.contract.instrument: {"trigger":None,"delta":0.,"perp_entry_ts":None,
              "perp_entry":None,"exit_ts":None,"option_exit":None,"perp_exit":None,
              "perp_pnl":0.,"perp_fees":0.,"exit_reason":None,"wait_minutes":0.}
              for x in intent.legs}
    for ts in common:
        ts = int(ts)
        for leg, path, mp in zip(intent.legs, paths, maps):
            st=states[leg.contract.instrument]
            if st["exit_ts"] is not None: continue
            if st.get("tape_exhausted"): continue
            i=mp[ts]; mark=val(path,"mark_price",i); bid=val(path,"bid_price",i); ask=val(path,"ask_price",i)
            if st["trigger"] is None and mark >= 1.5 * entry_price[leg.contract.instrument]:
                st["trigger"]=ts
                long_delta=val(path,"delta",i); st["delta"]=-long_delta
                ht,hp=perp.next_open(ts)
                if hp is None: return None
                st["perp_entry_ts"],st["perp_entry"]=ht,hp
            if st["trigger"] is None: continue
            mid=(bid+ask)/2 if bid>0 and ask>=bid else 0
            rel=(ask-bid)/mid if mid>0 else float("inf")
            waited=(ts-st["trigger"])/60/NS
            normalized=rel <= spread_limit
            timed_out=waited >= max_wait_minutes
            if normalized or timed_out:
                decision_reason="spread_normalized" if normalized else "max_wait"
                tape_fill = (tape.fill(leg.contract.instrument, Side.BUY, ts,
                                       leg.quantity, expiry) if tape is not None else None)
                if tape is not None and tape_fill is None:
                    # No later print exists before expiry: retain option and hedge to settlement.
                    st["exit_reason"]="expiry_no_tape"
                    st["tape_exhausted"]=True
                    continue
                option_exit_ts = tape_fill.first_ts_ns if tape_fill is not None else ts
                option_exit_px = tape_fill.price if tape_fill is not None else ask*(1+option_slip_bps*1e-4)
                if option_exit_px <= 0: continue
                st["option_exit"]=option_exit_px
                option_cash-=st["option_exit"]; fees+=min(.0003,.125*st["option_exit"])
                st["exit_reason"]=decision_reason+("_tape" if tape_fill is not None else "_quote")
                xt,xp=perp.next_open(option_exit_ts)
                if xp is None: return None
                hedge_qty=-st["delta"]
                pp,ep,xpx,pfees=perp_round_trip(hedge_qty,st["perp_entry"],xp,perp_fee_bps,perp_slip_bps)
                st.update({"exit_ts":xt,"perp_exit":xpx,"perp_pnl":pp,"perp_fees":pfees,
                           "wait_minutes":(option_exit_ts-st["trigger"])/60/NS,
                           "tape_delay_minutes":(option_exit_ts-ts)/60/NS})
    ref=paths[0]; mask=(ref["date"]>expiry-30*60*NS)&(ref["date"]<=expiry)
    ss=np.asarray(ref.get("index_price",ref.get("underlying_price")),float)[mask]
    ss=ss[np.isfinite(ss)&(ss>0)]; settle=float(ss.mean()) if ss.size else 0.
    for leg in intent.legs:
        st=states[leg.contract.instrument]
        if st["option_exit"] is None:
            intrinsic=(max(settle-leg.contract.strike,0)/settle if leg.contract.option_type==OptionType.CALL
                       else max(leg.contract.strike-settle,0)/settle) if settle else 0
            option_cash-=intrinsic; fees+=min(.00015,.125*intrinsic)
            if st["trigger"] is not None and st["exit_ts"] is None:
                xt,xp=perp.next_open(expiry)
                if xp is None:return None
                hedge_qty=-st["delta"]
                pp,ep,xpx,pfees=perp_round_trip(hedge_qty,st["perp_entry"],xp,perp_fee_bps,perp_slip_bps)
                st.update({"exit_ts":xt,"perp_exit":xpx,"perp_pnl":pp,"perp_fees":pfees,
                           "wait_minutes":(expiry-st["trigger"])/60/NS,
                           "tape_delay_minutes":None,"exit_reason":"expiry_no_tape"})
    option_pnl=option_cash-fees; perp_pnl=sum(x["perp_pnl"] for x in states.values())
    return {"trade_id":intent.trade_id,"entry_ts_ns":entry_ts,"expiry_ts_ns":expiry,
            "spread_limit":spread_limit,"max_wait_minutes":max_wait_minutes,
            "option_pnl_native":option_pnl,"perp_pnl_native":perp_pnl,
            "total_pnl_native":option_pnl+perp_pnl,"option_fees_native":fees,
            "perp_fees_native":sum(x["perp_fees"] for x in states.values()),
            "hedged_legs":sum(x["trigger"] is not None for x in states.values()),
            "normalized_exits":sum(str(x["exit_reason"]).startswith("spread_normalized") for x in states.values()),
            "timeout_exits":sum(str(x["exit_reason"]).startswith("max_wait") for x in states.values()),
            "tape_exits":sum((str(x["exit_reason"]).startswith("spread_normalized")
                              or str(x["exit_reason"]).startswith("max_wait"))
                             and str(x["exit_reason"]).endswith("_tape")
                             for x in states.values()),
            "no_tape_expiries":sum(x["exit_reason"]=="expiry_no_tape" for x in states.values()),
            "mean_tape_delay_minutes":float(np.mean([x.get("tape_delay_minutes",0) for x in states.values() if x.get("tape_delay_minutes") is not None])) if any(x.get("tape_delay_minutes") is not None for x in states.values()) else 0.,
            "mean_wait_minutes":float(np.mean([x["wait_minutes"] for x in states.values() if x["trigger"] is not None])) if any(x["trigger"] is not None for x in states.values()) else 0.}


def metrics(rows):
    p=np.asarray([x["total_pnl_native"] for x in rows]); eq=1+np.cumsum(p); curve=np.r_[1.,eq]
    dd=(curve-np.maximum.accumulate(curve))/np.maximum.accumulate(curve); sd=p.std(ddof=1)
    return {"trades":len(rows),"total_pnl_native":float(p.sum()),"ending_equity_native":float(eq[-1]),
            "weekly_sharpe":float(p.mean()/sd*math.sqrt(52)) if sd else 0.,"max_drawdown":float(dd.min()),
            "win_rate":float(np.mean(p>0)),"hedged_legs":sum(x["hedged_legs"] for x in rows),
            "normalized_exits":sum(x["normalized_exits"] for x in rows),"timeout_exits":sum(x["timeout_exits"] for x in rows),
            "tape_exits":sum(x["tape_exits"] for x in rows),"no_tape_expiries":sum(x["no_tape_expiries"] for x in rows),
            "mean_tape_delay_minutes":float(np.mean([x["mean_tape_delay_minutes"] for x in rows if x["tape_exits"]])) if any(x["tape_exits"] for x in rows) else 0.,
            "mean_wait_minutes":float(np.mean([x["mean_wait_minutes"] for x in rows if x["hedged_legs"]]))}


def main():
    a=argparse.ArgumentParser();a.add_argument("--perp",default=".lwbt_cache/perp/BTCUSDT_5m_2021_2024.parquet")
    a.add_argument("--output",default="results/static_perp_bridge_grid")
    a.add_argument("--option-execution",choices=("quotes","trade_tape"),default="quotes")
    a.add_argument("--trade-root",default="/srv/shared/laevitas data/deribit/options_historical_data/trades")
    args=a.parse_args()
    c=config();store=MarketDataStore(c.data_root,c.cache_root);store.max_cached_paths=256;perp=PerpBars(args.perp)
    snaps=store.snapshots(c);s=OptionStructureStrategy(c);intents=[x for t in sorted(snaps) if (x:=s.on_entry(t,snaps[t])) and max(y.contract.expiry_ns for y in x.legs)<=c.end_ns]
    tape=(OptionTradeIndex(args.trade_root,{y.contract.instrument for x in intents for y in x.legs},
                           c.start_ns,c.end_ns,slippage_bps=1.)
          if args.option_execution=="trade_tape" else None)
    detail=[];summary=[]
    for spread in (.05,.10,.15,.20,.30):
      for wait in (15,30,60,240,1440):
        rows=[r for x in intents if (r:=simulate(x,store,perp,spread,wait,tape=tape))]
        detail.extend(rows);summary.append({"relative_spread_limit":spread,"max_wait_minutes":wait,**metrics(rows)})
        print(summary[-1],flush=True)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.Table.from_pylist(detail),out/"trade_results.parquet",compression="zstd")
    pq.write_table(pa.Table.from_pylist(summary),out/"summary.parquet",compression="zstd")
    with (out/"summary.csv").open("w",newline="") as f:
      w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    (out/"methodology.json").write_text(json.dumps({"option_stop_trigger":"per-leg mark >= 1.5 * entry premium","static_hedge":"minus stopped short-leg delta","perp_fill":"next BTCUSDT 5m open","perp_slippage_bps":5,"perp_fee_bps_each_side":5,"option_execution":args.option_execution,"option_exit":("nearest subsequent non-block trade after spread/max-wait decision" if tape else "ask when spread normalizes, otherwise max-wait ask"),"no_trade_before_expiry":"option settlement and perp expiry cleanup","funding":"not available/not modeled","perp_margin":"not modeled"},indent=2))

if __name__=="__main__":main()
