"""Round 3 — neighbourhood of H16 winner (96.652)."""
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

# H16 baseline:
# LV (Laundry|Vacuuming spread) z=1.2
# DV (Dishes|Vacuuming sum) z=1.5
# DL (Dishes|Laundry sum) z=1.5
# IM (Ironing|Mopping sum) z=1.5
# LM (Laundry|Mopping sum) z=2.0
# MV (Mopping|Vacuuming sum) z=1.2

H16 = [
    ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.2),
    ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.5),
    ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
    ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
    ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
    ("ROBOT_MOPPING", "ROBOT_VACUUMING", "sum", 1.2),
]

def with_z(pairs, sym_a, sym_b, sign, new_z):
    return [(a,b,s, new_z if (a==sym_a and b==sym_b and s==sign) else z) for (a,b,s,z) in pairs]

EXPERIMENTS = {
    # Sweep around H16 — vary MV
    "h19_h16_MV_1.0": dict(desc="H16 MV z=1.0", exit_z=0.3,
                            pairs=with_z(H16,"ROBOT_MOPPING","ROBOT_VACUUMING","sum",1.0)),
    "h20_h16_MV_1.5": dict(desc="H16 MV z=1.5", exit_z=0.3,
                            pairs=with_z(H16,"ROBOT_MOPPING","ROBOT_VACUUMING","sum",1.5)),
    "h21_h16_MV_2.0": dict(desc="H16 MV z=2.0", exit_z=0.3,
                            pairs=with_z(H16,"ROBOT_MOPPING","ROBOT_VACUUMING","sum",2.0)),

    # Add 7th pair
    "h22_h16_plus_IV": dict(desc="H16 + IRONING|VACUUMING spread z=1.2", exit_z=0.3,
                            pairs=H16 + [("ROBOT_IRONING","ROBOT_VACUUMING","spread",1.2)]),
    "h23_h16_plus_IL": dict(desc="H16 + IRONING|LAUNDRY spread z=1.2", exit_z=0.3,
                            pairs=H16 + [("ROBOT_IRONING","ROBOT_LAUNDRY","spread",1.2)]),
    "h24_h16_plus_DI": dict(desc="H16 + DISHES|IRONING sum z=1.5", exit_z=0.3,
                            pairs=H16 + [("ROBOT_DISHES","ROBOT_IRONING","sum",1.5)]),

    # Combine: more pairs + tweak existing
    "h25_h16_lower_LV": dict(desc="H16 LV z=1.0", exit_z=0.3,
                              pairs=with_z(H16,"ROBOT_LAUNDRY","ROBOT_VACUUMING","spread",1.0)),
    "h26_h16_higher_LV": dict(desc="H16 LV z=1.5", exit_z=0.3,
                              pairs=with_z(H16,"ROBOT_LAUNDRY","ROBOT_VACUUMING","spread",1.5)),

    # Variations of EXIT_Z on H16
    "h27_h16_exit_0.1": dict(desc="H16 EXIT_Z=0.1", exit_z=0.1, pairs=H16),
    "h28_h16_exit_0.5": dict(desc="H16 EXIT_Z=0.5", exit_z=0.5, pairs=H16),

    # Drop weakest
    "h29_h16_drop_DL": dict(desc="H16 minus DL", exit_z=0.3,
                             pairs=[p for p in H16 if not (p[0]=="ROBOT_DISHES" and p[1]=="ROBOT_LAUNDRY")]),
    "h30_h16_drop_LM": dict(desc="H16 minus LM", exit_z=0.3,
                             pairs=[p for p in H16 if not (p[0]=="ROBOT_LAUNDRY" and p[1]=="ROBOT_MOPPING")]),

    # H16 plus everything
    "h31_h16_full7": dict(desc="H16 + IV + IL (top-8 EDA pairs)", exit_z=0.3,
                          pairs=H16 + [
                              ("ROBOT_IRONING","ROBOT_VACUUMING","spread",1.2),
                              ("ROBOT_IRONING","ROBOT_LAUNDRY","spread",1.2),
                          ]),
}

OUT = Path(__file__).parent
for name, cfg in EXPERIMENTS.items():
    p = OUT / f"{name}.py"
    p.write_text(TEMPLATE_HEADER.format(name=name, desc=cfg["desc"], pairs=cfg["pairs"], exit_z=cfg["exit_z"]))
    print("wrote", p.name)
