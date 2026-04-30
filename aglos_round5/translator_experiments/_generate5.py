"""Round 5 — neighbourhood of T42 (99,916 = +97.1%). Tune EXIT_Z fine."""
from pathlib import Path

TEMPLATE = '''"""TRANSLATOR exp — {name}: {desc}"""
from __future__ import annotations
from typing import Dict, List, Tuple
import jsonpickle
from datamodel import Order, OrderDepth, TradingState

PAIRS: List[Tuple[str, str, str, float]] = {pairs!r}
EXIT_Z = {exit_z}
SIZE = 10
WARMUP = 500
LIMIT = 10
PASSIVE_ENTRY = {passive}


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
            da = state.order_depths.get(a); db = state.order_depths.get(b)
            if da is None or db is None: continue
            ma, mb = mid_of(da), mid_of(db)
            if ma is None or mb is None: continue
            sig = ma - mb if sign == "spread" else ma + mb
            key = pair_key(a, b, sign)
            st = stats.setdefault(key, {{"n": 0, "mean": 0.0, "M2": 0.0}})
            st["n"] += 1; n = st["n"]
            delta = sig - st["mean"]
            st["mean"] += delta / n
            st["M2"] += delta * (sig - st["mean"])
            if n < WARMUP:
                targets[key] = 0; continue
            var = st["M2"] / (n - 1) if n > 1 else 0.0
            sd = var ** 0.5
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

        result: Dict[str, List[Order]] = {{}}
        for p, tgt in leg_targets.items():
            tgt = max(-LIMIT, min(LIMIT, tgt))
            depth = state.order_depths.get(p)
            if depth is None: continue
            current = state.position.get(p, 0)
            d = tgt - current
            if d == 0: continue
            is_reducing = (current > 0 and d < 0) or (current < 0 and d > 0)
            if PASSIVE_ENTRY and not is_reducing:
                if d > 0 and depth.buy_orders:
                    result[p] = [Order(p, max(depth.buy_orders.keys()) + 1, d)]
                elif d < 0 and depth.sell_orders:
                    result[p] = [Order(p, min(depth.sell_orders.keys()) - 1, d)]
            else:
                if d > 0 and depth.sell_orders:
                    result[p] = [Order(p, min(depth.sell_orders.keys()), d)]
                elif d < 0 and depth.buy_orders:
                    result[p] = [Order(p, max(depth.buy_orders.keys()), d)]

        return result, 0, jsonpickle.encode(store)
'''

T = lambda x: "TRANSLATOR_" + x

T22 = [
    (T("SPACE_GRAY"), T("GRAPHITE_MIST"), "spread", 1.2),
    (T("SPACE_GRAY"), T("VOID_BLUE"), "sum", 1.2),
    (T("ASTRO_BLACK"), T("VOID_BLUE"), "sum", 1.2),
    (T("ECLIPSE_CHARCOAL"), T("VOID_BLUE"), "spread", 1.2),
    (T("ECLIPSE_CHARCOAL"), T("SPACE_GRAY"), "sum", 1.2),
]

EXPS = {
    "t46_t22_exit015": dict(desc="T22 EXIT_Z=0.15", exit_z=0.15, passive=False, pairs=T22),
    "t47_t22_exit025": dict(desc="T22 EXIT_Z=0.25", exit_z=0.25, passive=False, pairs=T22),
    "t48_t42_z10": dict(desc="T42 (exit=0.2) z=1.0", exit_z=0.2, passive=False,
                       pairs=[(a,b,s,1.0) for (a,b,s,z) in T22]),
    "t49_t42_z15": dict(desc="T42 (exit=0.2) z=1.5", exit_z=0.2, passive=False,
                       pairs=[(a,b,s,1.5) for (a,b,s,z) in T22]),
    "t50_t42_drop_sg_vb": dict(desc="T42 minus SPACE_GRAY|VOID_BLUE",
                               exit_z=0.2, passive=False,
                               pairs=[p for p in T22 if not (p[0]==T("SPACE_GRAY") and p[1]==T("VOID_BLUE"))]),
    "t51_t42_drop_sg_gm": dict(desc="T42 minus SPACE_GRAY|GRAPHITE_MIST",
                               exit_z=0.2, passive=False,
                               pairs=[p for p in T22 if not (p[0]==T("SPACE_GRAY") and p[1]==T("GRAPHITE_MIST"))]),
    "t52_t42_plus_ae": dict(desc="T42 + ASTRO|ECLIPSE sum",
                             exit_z=0.2, passive=False,
                             pairs=T22 + [(T("ASTRO_BLACK"), T("ECLIPSE_CHARCOAL"), "sum", 1.2)]),
    "t53_t42_plus_eg": dict(desc="T42 + ECLIPSE|GRAPHITE_MIST sum",
                             exit_z=0.2, passive=False,
                             pairs=T22 + [(T("ECLIPSE_CHARCOAL"), T("GRAPHITE_MIST"), "sum", 1.2)]),
    "t54_t42_passive": dict(desc="T42 with passive entry",
                             exit_z=0.2, passive=True, pairs=T22),
    "t55_t42_mixed_z": dict(desc="T42 z by EDA rank", exit_z=0.2, passive=False,
                            pairs=[
                                (T("SPACE_GRAY"), T("GRAPHITE_MIST"), "spread", 1.5),
                                (T("SPACE_GRAY"), T("VOID_BLUE"), "sum", 1.2),
                                (T("ASTRO_BLACK"), T("VOID_BLUE"), "sum", 1.0),
                                (T("ECLIPSE_CHARCOAL"), T("VOID_BLUE"), "spread", 1.0),
                                (T("ECLIPSE_CHARCOAL"), T("SPACE_GRAY"), "sum", 1.0),
                            ]),
}

OUT = Path(__file__).parent
for name, cfg in EXPS.items():
    p = OUT / f"{name}.py"
    p.write_text(TEMPLATE.format(name=name, desc=cfg["desc"], pairs=cfg["pairs"], exit_z=cfg["exit_z"], passive=cfg["passive"]))
    print("wrote", p.name)
