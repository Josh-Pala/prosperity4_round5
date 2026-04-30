"""
COMBINED v5 (CV-robust): pairs picked via leave-one-day-out cross-validation,
with per-pair best ENTRY_Z hyperparameter chosen by CV.

Built from analysis/24_build_cv_combined.py — see analysis/out/cv_picked_pairs.csv.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import jsonpickle
from datamodel import Order, OrderDepth, TradingState

# (A, B, sign, ENTRY_Z) — ENTRY_Z is per-pair, chosen by CV.
PAIRS: List[Tuple[str, str, str, float]] = [
    ("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE", "spread", 1.2),  # cv_mean=23979  cv_min=19331
    ("PEBBLES_XS", "PEBBLES_S", "spread", 2.0),  # cv_mean=23126  cv_min=18240
    ("TRANSLATOR_SPACE_GRAY", "TRANSLATOR_GRAPHITE_MIST", "spread", 1.2),  # cv_mean=14688  cv_min=10361
    ("MICROCHIP_CIRCLE", "MICROCHIP_TRIANGLE", "sum", 1.5),  # cv_mean=12018  cv_min=4209
    ("PANEL_2X4", "PANEL_4X4", "sum", 2.0),  # cv_mean=10507  cv_min=3003
    ("ROBOT_VACUUMING", "ROBOT_LAUNDRY", "spread", 1.0),  # cv_mean=9708  cv_min=6937
    ("SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY", "sum", 1.5),  # cv_mean=9625  cv_min=8240
    ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.0),  # cv_mean=9007  cv_min=5305
    ("ROBOT_MOPPING", "ROBOT_LAUNDRY", "sum", 2.0),  # cv_mean=8181  cv_min=2050
    ("SLEEP_POD_POLYESTER", "SLEEP_POD_COTTON", "spread", 1.8),  # cv_mean=11616  cv_min=6750
    ("SNACKPACK_VANILLA", "SNACKPACK_RASPBERRY", "spread", 1.2),  # cv_mean=6970  cv_min=4340
    ("SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO", "sum", 1.5),  # cv_mean=7130  cv_min=4280
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO", "spread", 1.8),  # cv_mean=5953  cv_min=3440
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_STRAWBERRY", "sum", 1.8),  # cv_mean=6170  cv_min=4375
    ("MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE", "spread", 2.0),  # cv_mean=4979  cv_min=122
    ("TRANSLATOR_SPACE_GRAY", "TRANSLATOR_VOID_BLUE", "sum", 1.2),  # cv_mean=6880  cv_min=4116
    ("ROBOT_LAUNDRY", "ROBOT_IRONING", "spread", 1.2),  # cv_mean=3998  cv_min=2090
    ("ROBOT_MOPPING", "ROBOT_IRONING", "sum", 1.5),  # cv_mean=5836  cv_min=2622

]

LIMIT = 10
WARMUP = 500
EXIT_Z = 0.3
SIZE = 10


def mid_of(d: OrderDepth):
    if not d.buy_orders or not d.sell_orders: return None
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
            st = stats.setdefault(key, {"n": 0, "mean": 0.0, "M2": 0.0})
            st["n"] += 1; n = st["n"]
            delta = sig - st["mean"]; st["mean"] += delta / n; st["M2"] += delta * (sig - st["mean"])
            if n < WARMUP:
                targets[key] = 0
                continue
            var = st["M2"] / (n - 1) if n > 1 else 0.0; sd = var ** 0.5
            if sd <= 1e-9: continue
            z = (sig - st["mean"]) / sd
            cur = targets.get(key, 0)
            if cur == 0:
                if z > entry_z: cur = -1
                elif z < -entry_z: cur = +1
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
