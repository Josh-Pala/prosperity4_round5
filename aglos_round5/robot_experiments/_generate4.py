"""Round 4 — combine winners H25 + drop DL + sweep."""
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

# Best so far: H25 = H16 with LV z=1.0 -> 97.216
# H25 pairs:
H25 = [
    ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.0),
    ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.5),
    ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
    ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
    ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
    ("ROBOT_MOPPING", "ROBOT_VACUUMING", "sum", 1.2),
]
# H29 minus DL got 92.209 - similar magnitude. Let's try H25 minus DL.

EXPERIMENTS = {
    # Combine winners
    "h32_h25_drop_DL": dict(desc="H25 minus DL", exit_z=0.3,
                             pairs=[p for p in H25 if not (p[0]=="ROBOT_DISHES" and p[1]=="ROBOT_LAUNDRY")]),
    "h33_h25_LV_0.8": dict(desc="H25 LV z=0.8", exit_z=0.3,
                            pairs=[(a,b,s,0.8 if (a=="ROBOT_LAUNDRY" and b=="ROBOT_VACUUMING") else z) for (a,b,s,z) in H25]),
    "h34_h25_LV_1.2": dict(desc="H25 LV z=1.2 (=H16)", exit_z=0.3, pairs=H25.copy()),

    # Sweep MV around H25
    "h35_h25_MV_0.8": dict(desc="H25 MV z=0.8", exit_z=0.3,
                            pairs=[(a,b,s,0.8 if (a=="ROBOT_MOPPING" and b=="ROBOT_VACUUMING") else z) for (a,b,s,z) in H25]),
    "h36_h25_MV_1.0": dict(desc="H25 MV z=1.0", exit_z=0.3,
                            pairs=[(a,b,s,1.0 if (a=="ROBOT_MOPPING" and b=="ROBOT_VACUUMING") else z) for (a,b,s,z) in H25]),
    "h37_h25_MV_1.5": dict(desc="H25 MV z=1.5", exit_z=0.3,
                            pairs=[(a,b,s,1.5 if (a=="ROBOT_MOPPING" and b=="ROBOT_VACUUMING") else z) for (a,b,s,z) in H25]),

    # H25 + IL
    "h38_h25_plus_IL": dict(desc="H25 + IL spread z=1.2", exit_z=0.3,
                             pairs=H25 + [("ROBOT_IRONING","ROBOT_LAUNDRY","spread",1.2)]),

    # EXIT tweaks
    "h39_h25_exit_0.5": dict(desc="H25 EXIT_Z=0.5", exit_z=0.5, pairs=H25.copy()),
    "h40_h25_exit_0.2": dict(desc="H25 EXIT_Z=0.2", exit_z=0.2, pairs=H25.copy()),

    # Combined
    "h41_h25_drop_DL_LV0.8_MV1.0": dict(desc="H25 minus DL, LV=0.8, MV=1.0", exit_z=0.3,
        pairs=[(a,b,s,
                0.8 if (a=="ROBOT_LAUNDRY" and b=="ROBOT_VACUUMING") else
                1.0 if (a=="ROBOT_MOPPING" and b=="ROBOT_VACUUMING") else z)
                for (a,b,s,z) in H25 if not (a=="ROBOT_DISHES" and b=="ROBOT_LAUNDRY")]),
    "h42_h25_drop_DL_plus_IL": dict(desc="H25 minus DL plus IL", exit_z=0.3,
        pairs=[p for p in H25 if not (p[0]=="ROBOT_DISHES" and p[1]=="ROBOT_LAUNDRY")] +
              [("ROBOT_IRONING","ROBOT_LAUNDRY","spread",1.2)]),

    # Looser MV (Mopping/Vacuuming had std=479, 8.6 crossings/1k -> potentially higher z helps)
    "h43_h25_MV_2.0": dict(desc="H25 MV z=2.0", exit_z=0.3,
                            pairs=[(a,b,s,2.0 if (a=="ROBOT_MOPPING" and b=="ROBOT_VACUUMING") else z) for (a,b,s,z) in H25]),
}

OUT = Path(__file__).parent
for name, cfg in EXPERIMENTS.items():
    p = OUT / f"{name}.py"
    p.write_text(TEMPLATE_HEADER.format(name=name, desc=cfg["desc"], pairs=cfg["pairs"], exit_z=cfg["exit_z"]))
    print("wrote", p.name)
