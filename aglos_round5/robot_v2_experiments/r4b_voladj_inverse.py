"""H4' Vol-adjusted z, inverse: when vol is LOW, lower entry_z (more trades).
Clamp tight."""
from __future__ import annotations
from typing import Dict, List, Tuple
import jsonpickle
from datamodel import Order, OrderDepth, TradingState

PAIRS: List[Tuple[str, str, str, float]] = [
    ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.0),
    ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.5),
    ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
    ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
    ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
    ("ROBOT_MOPPING", "ROBOT_VACUUMING", "sum", 1.2),
]
EXIT_Z = 0.3
SIZE = 10
WARMUP = 500
LIMIT = 10
ALPHA_SHORT = 0.005
RATIO_CLAMP_LO = 0.7
RATIO_CLAMP_HI = 1.5


def mid_of(d: OrderDepth):
    if not d.buy_orders or not d.sell_orders:
        return None
    return (max(d.buy_orders.keys()) + min(d.sell_orders.keys())) / 2.0


def pair_key(a, b, sign):
    return f"{a}|{b}|{sign}"


class Trader:
    def run(self, state: TradingState):
        store = jsonpickle.decode(state.traderData) if state.traderData else {}
        stats = store.setdefault("stats", {})
        targets = store.setdefault("target", {})

        leg_targets: Dict[str, int] = {}
        for a, b, sign, entry_z in PAIRS:
            da = state.order_depths.get(a); db = state.order_depths.get(b)
            if da is None or db is None: continue
            ma, mb = mid_of(da), mid_of(db)
            if ma is None or mb is None: continue
            sig = ma - mb if sign == "spread" else ma + mb
            key = pair_key(a, b, sign)
            st = stats.setdefault(key, {"n": 0, "mean": 0.0, "M2": 0.0, "short_var": 0.0})
            st["n"] += 1; n = st["n"]
            delta = sig - st["mean"]
            st["mean"] += delta / n
            st["M2"] += delta * (sig - st["mean"])
            sq = (sig - st["mean"]) ** 2
            st["short_var"] = (1 - ALPHA_SHORT) * st["short_var"] + ALPHA_SHORT * sq
            if n < WARMUP:
                targets[key] = 0; continue
            long_var = st["M2"] / (n - 1) if n > 1 else 0.0
            sd = long_var ** 0.5
            if sd <= 1e-9: continue
            z = (sig - st["mean"]) / sd
            if long_var > 1e-9:
                ratio = (st["short_var"] / long_var) ** 0.5
                ratio = max(RATIO_CLAMP_LO, min(RATIO_CLAMP_HI, ratio))
            else:
                ratio = 1.0
            adj_entry_z = entry_z * ratio
            cur = targets.get(key, 0)
            if cur == 0:
                if z > adj_entry_z: cur = -1
                elif z < -adj_entry_z: cur = +1
            else:
                if abs(z) < EXIT_Z: cur = 0
            targets[key] = cur
            ta = SIZE * cur
            tb = (-SIZE if sign == "spread" else SIZE) * cur
            leg_targets[a] = leg_targets.get(a, 0) + ta
            leg_targets[b] = leg_targets.get(b, 0) + tb

        result: Dict[str, List[Order]] = {}
        for p, tgt in leg_targets.items():
            tgt = max(-LIMIT, min(LIMIT, tgt))
            depth = state.order_depths.get(p)
            if depth is None: continue
            current = state.position.get(p, 0)
            d = tgt - current
            if d == 0: continue
            if d > 0 and depth.sell_orders:
                result[p] = [Order(p, min(depth.sell_orders.keys()), d)]
            elif d < 0 and depth.buy_orders:
                result[p] = [Order(p, max(depth.buy_orders.keys()), d)]

        return result, 0, jsonpickle.encode(store)
