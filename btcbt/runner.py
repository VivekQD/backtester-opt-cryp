"""Single run, parameter sweeps, config hashing + result cache. Phase 1 / 2."""
from __future__ import annotations

import hashlib
import itertools
import os
import pickle
from dataclasses import dataclass, field, replace

from .data import DataConfig, Dataset
from .engine import Engine
from .execution import DefaultFill, FillModel
from .metrics import RunResult
from .venue import VenueConfig

CACHE_DIR = ".btcbt_cache"
_DATASET_CACHE: dict = {}


@dataclass
class RunConfig:
    data: DataConfig
    venue: VenueConfig
    fill: FillModel
    strategy: type
    params: dict = field(default_factory=dict)
    cash: float = 1.0
    seed: int = 0
    margin: object = None       # optional MarginModel


def _load_dataset(dc: DataConfig) -> Dataset:
    key = (dc.path, dc.start, dc.end)
    ds = _DATASET_CACHE.get(key)
    if ds is None:
        ds = Dataset.load(dc)
        _DATASET_CACHE[key] = ds
    return ds


def run_one(cfg: RunConfig) -> RunResult:
    res, _ = run_one_with_dataset(cfg)
    return res


def run_one_with_dataset(cfg: RunConfig) -> tuple[RunResult, Dataset]:
    ds = _load_dataset(cfg.data)
    strat = cfg.strategy(cfg.params)
    eng = Engine(ds, strat, cfg.fill, cfg.venue, cfg.cash, margin=cfg.margin)
    return eng.run(), ds


def config_hash(cfg: RunConfig) -> str:
    key = (
        cfg.data.path, cfg.data.start, cfg.data.end,
        f"{cfg.strategy.__module__}.{cfg.strategy.__qualname__}",
        tuple(sorted(cfg.params.items())),
        cfg.venue.__dict__ if hasattr(cfg.venue, "__dict__") else str(cfg.venue),
        cfg.cash, cfg.seed,
    )
    return hashlib.sha1(repr(key).encode()).hexdigest()[:16]


def save_result(h: str, res: RunResult) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, f"{h}.pkl"), "wb") as f:
        pickle.dump(res, f)


def load_result(h: str) -> RunResult | None:
    p = os.path.join(CACHE_DIR, f"{h}.pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def expand_grid(base: RunConfig, axes: dict) -> list[RunConfig]:
    keys = list(axes)
    out = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        params = {**base.params, **dict(zip(keys, combo))}
        out.append(replace(base, params=params))
    return out


def _run_cached(cfg: RunConfig) -> RunResult:
    h = config_hash(cfg)
    res = load_result(h)
    if res is None:
        res = run_one(cfg)
        save_result(h, res)
    return res


def run_sweep(grid: list[RunConfig], workers: int = 0) -> list[RunResult]:
    if workers and workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            return pool.map(_run_cached, grid)
    return [_run_cached(g) for g in grid]


@dataclass
class StudyResult:
    configs: list
    results: list

    def leaderboard(self, metric: str = "sharpe"):
        def key(i):
            s = self.results[i].stats
            for grp in (s.ratios, s.overview, s.costs):
                if hasattr(grp, metric):
                    return getattr(grp, metric)
            return 0.0
        order = sorted(range(len(self.results)), key=key, reverse=True)
        return [(self.configs[i].params, key(i)) for i in order]

    def frame(self) -> list[dict]:
        rows = []
        for c, r in zip(self.configs, self.results):
            s = r.stats
            rows.append({**c.params, "total_pnl": s.overview.total_pnl,
                         "sharpe": s.ratios.sharpe, "max_dd": s.overview.max_dd,
                         "trades": s.overview.trades, "win_rate": s.overview.win_rate})
        return rows


@dataclass
class Study:
    configs: list

    def run(self, workers: int = 0) -> StudyResult:
        return StudyResult(self.configs, run_sweep(self.configs, workers))
