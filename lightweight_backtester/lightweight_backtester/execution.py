from __future__ import annotations

from dataclasses import dataclass

from .types import Side


@dataclass
class ExecutionModel:
    slippage_bps: float = 1.0

    def fill_price(self, side: Side, bid: float, ask: float, mark: float) -> float | None:
        touch = ask if side == Side.BUY else bid
        if not touch or touch <= 0:
            touch = mark
        if not touch or touch <= 0:
            return None
        direction = 1 if side == Side.BUY else -1
        return float(touch) * (1 + direction * self.slippage_bps * 1e-4)
