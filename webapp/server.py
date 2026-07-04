"""Local web app: serve the strategy builder and run backtests.

    pip install fastapi uvicorn
    BTCBT_DATA=/path/to/btc_usdc_last7d.parquet python -m webapp.server
    open http://127.0.0.1:8000
"""
from __future__ import annotations

import os
import sys
import traceback

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# allow `python webapp/server.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btcbt import DataConfig, DefaultFill, Margin, RunConfig, VenueConfig, run_one  # noqa: E402
from btcbt.runner import _load_dataset  # noqa: E402
from btcbt.spec import compile_spec, spec_from_dict  # noqa: E402

DATA_PATH = os.environ.get("BTCBT_DATA", "/Users/nabi/Downloads/btc_usdc_last7d.parquet")
HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="btcbt builder")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "index.html")) as f:
        return f.read()


@app.get("/meta")
def meta():
    import datetime as dt
    try:
        ds = _load_dataset(DataConfig(DATA_PATH))
        ts = ds.timestamps()
        iso = lambda ns: dt.datetime.fromtimestamp(int(ns) / 1e9, dt.timezone.utc).strftime("%Y-%m-%d")
        return {"ok": True, "file": os.path.basename(DATA_PATH),
                "bars": int(len(ts)), "instruments": int(len(ds.metas)),
                "start": iso(ts[0]), "end": iso(ts[-1]),
                "margin": "linear" if int(ds.margin) == 1 else "inverse"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


@app.post("/backtest")
def backtest(payload: dict):
    try:
        spec = spec_from_dict(payload)
        cash = float(payload.get("run", {}).get("cash", 100000.0))
        margin = Margin.LINEAR if payload.get("run", {}).get("margin", "linear") == "linear" else Margin.INVERSE
        cfg = RunConfig(DataConfig(DATA_PATH), VenueConfig(margin=margin),
                        DefaultFill(), compile_spec(spec), cash=cash)
        res = run_one(cfg)
        s = res.stats
        metas = _load_dataset(DataConfig(DATA_PATH)).metas
        name_of = lambda sym: metas[sym].name if sym in metas else str(sym)

        eq = [(int(t // 1_000_000), round(e.usd, 2)) for t, e in res.equity]
        dd = [round(e.dd, 5) for _, e in res.equity]
        idx = _load_dataset(DataConfig(DATA_PATH)).index_series
        price = [(int(t // 1_000_000), round(float(v), 2)) for t, v in idx]
        step = max(1, len(eq) // 1500)
        return {
            "ok": True,
            "overview": vars(s.overview), "winning": vars(s.winning),
            "losing": vars(s.losing), "ratios": vars(s.ratios), "costs": vars(s.costs),
            "equity": eq[::step], "drawdown": dd[::step], "price": price[::step],
            "trades": [{"name": name_of(t.sym), "qty": round(t.qty, 4),
                        "entry": round(t.entry, 6), "exit": round(t.exit, 6),
                        "pnl": round(t.pnl, 2), "fees": round(t.fees, 6),
                        "entry_ts": int(t.entry_ts // 1_000_000),
                        "exit_ts": int(t.exit_ts // 1_000_000)}
                       for t in res.trades[:3000]],
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-1200:]}


def main():
    import uvicorn
    if not os.path.exists(DATA_PATH):
        print(f"warning: dataset not found at {DATA_PATH} (set BTCBT_DATA)", file=sys.stderr)
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
