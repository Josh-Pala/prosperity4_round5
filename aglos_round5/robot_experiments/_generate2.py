"""Round 2 — neighbourhood of H2 winner."""
from pathlib import Path

TEMPLATE_HEADER = '''"""ROBOT exp — {name}
{desc}
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import jsonpickle
from datamodel import Order, OrderDepth, TradingState

PAIRS: List[Tuple[str, str, str, float]] = {pairs!r}
EXIT_Z = {exit_z}
SIZE = 10
WARMUP = 500
LIMIT = 10


def mid_of(d: OrderDepth):
    if not d.buy_orders or not d.sell_orders:
        return None
    return (max(d.buy_orders.keys()) + min(d.sell_orders.keys())) / 2.0


def pair_key(a, b, sign):
    return f"{{a}}|{{b}}|{{sign}}"


class Trader:
    def run(self, state: TradingState):
        store = jsonpickle.decode(state.traderData) if state.traderData else {{}}
        stats = store.setdefault("stats", {{}})
        targets = store.setdefault("target", {{}})

        leg_targets: Dict[str, int] = {{}}
        for a, b, sign, entry_z in PAIRS:
            da = state.order_depths.get(a)
            db = state.order_depths.get(b)
            if da is None or db is None:
                continue
            ma, mb = mid_of(da), mid_of(db)
            if ma is None or mb is None:
                continue
            sig = ma - mb if sign == "spread" else ma + mb
            key = pair_key(a, b, sign)
            st = stats.setdefault(key, {{"n": 0, "mean": 0.0, "M2": 0.0}})
            st["n"] += 1
            n = st["n"]
            delta = sig - st["mean"]
            st["mean"] += delta / n
            st["M2"] += delta * (sig - st["mean"])
            if n < WARMUP:
                targets[key] = 0
                continue
            var = st["M2"] / (n - 1) if n > 1 else 0.0
            sd = var ** 0.5
            if sd <= 1e-9:
                continue
            z = (sig - st["mean"]) / sd
            cur = targets.get(key, 0)
            if cur == 0:
                if z > entry_z:
                    cur = -1
                elif z < -entry_z:
                    cur = +1
            else:
                if abs(z) < EXIT_Z:
                    cur = 0
            targets[key] = cur
            ta = SIZE * cur
            tb = (-SIZE if sign == "spread" else SIZE) * cur
            leg_targets[a] = leg_targets.get(a, 0) + ta
            leg_targets[b] = leg_targets.get(b, 0) + tb

        result: Dict[str, List[Order]] = {{}}
        for p, tgt in leg_targets.items():
            tgt = max(-LIMIT, min(LIMIT, tgt))
            depth = state.order_depths.get(p)
            if depth is None:
                continue
            current = state.position.get(p, 0)
            d = tgt - current
            if d == 0:
                continue
            if d > 0 and depth.sell_orders:
                result[p] = [Order(p, min(depth.sell_orders.keys()), d)]
            elif d < 0 and depth.buy_orders:
                result[p] = [Order(p, max(depth.buy_orders.keys()), d)]

        return result, 0, jsonpickle.encode(store)
'''

# H2 was: LV=1.2, DV=1.5, DL=1.5, IM=1.5, LM=2.0, exit=0.3
# Pair labels: LV=Laundry|Vacuuming spread, DV=Dishes|Vacuuming sum, DL=Dishes|Laundry sum,
#              IM=Ironing|Mopping sum, LM=Laundry|Mopping sum

EXPERIMENTS = {
    # Round 2: tweak z values around H2
    "h9_h2_tighter_exit": dict(
        desc="H2 with EXIT_Z=0.1",
        exit_z=0.1,
        pairs=[
            ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.2),
            ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.5),
            ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
            ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
            ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
        ],
    ),
    "h10_h2_higher_LV": dict(
        desc="H2 but LV z=1.5 (was 1.2)",
        exit_z=0.3,
        pairs=[
            ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.5),
            ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.5),
            ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
            ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
            ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
        ],
    ),
    "h11_h2_lower_LV": dict(
        desc="H2 but LV z=1.0",
        exit_z=0.3,
        pairs=[
            ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.0),
            ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.5),
            ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
            ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
            ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
        ],
    ),
    "h12_h2_higher_LM": dict(
        desc="H2 but LM z=2.5 (LM had z>2.0 freq=6.6%)",
        exit_z=0.3,
        pairs=[
            ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.2),
            ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.5),
            ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
            ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
            ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.5),
        ],
    ),
    "h13_h2_lower_LM": dict(
        desc="H2 but LM z=1.5",
        exit_z=0.3,
        pairs=[
            ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.2),
            ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.5),
            ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
            ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
            ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 1.5),
        ],
    ),
    "h14_drop_DL": dict(
        desc="H2 minus DISHES|LAUNDRY sum (LAUNDRY was in 3 pairs, reduce to 2)",
        exit_z=0.3,
        pairs=[
            ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.2),
            ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.5),
            ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
            ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
        ],
    ),
    "h15_h2_plus_IV": dict(
        desc="H2 + IRONING|VACUUMING spread (rank 6, z=1.2)",
        exit_z=0.3,
        pairs=[
            ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.2),
            ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.5),
            ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
            ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
            ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
            ("ROBOT_IRONING", "ROBOT_VACUUMING", "spread", 1.2),
        ],
    ),
    "h16_h2_plus_MV": dict(
        desc="H2 + MOPPING|VACUUMING sum (rank 5, z=1.2)",
        exit_z=0.3,
        pairs=[
            ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.2),
            ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.5),
            ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
            ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
            ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
            ("ROBOT_MOPPING", "ROBOT_VACUUMING", "sum", 1.2),
        ],
    ),
    "h17_h2_higher_DV": dict(
        desc="H2 but DV z=1.0 (DV had high crossings 13.8/1k)",
        exit_z=0.3,
        pairs=[
            ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.2),
            ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.0),
            ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
            ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
            ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
        ],
    ),
    "h18_h2_lower_IM": dict(
        desc="H2 but IM z=1.2",
        exit_z=0.3,
        pairs=[
            ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.2),
            ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.5),
            ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
            ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.2),
            ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
        ],
    ),
}

OUT = Path(__file__).parent
for name, cfg in EXPERIMENTS.items():
    p = OUT / f"{name}.py"
    p.write_text(TEMPLATE_HEADER.format(name=name, desc=cfg["desc"], pairs=cfg["pairs"], exit_z=cfg["exit_z"]))
    print("wrote", p.name)
