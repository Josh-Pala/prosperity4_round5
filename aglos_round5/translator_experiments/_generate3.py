"""Round 3 — neighbourhood of T17 (T6 aggressive +67.7%) and T11."""
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

T17 = [
    (T("SPACE_GRAY"), T("GRAPHITE_MIST"), "spread", 1.2),
    (T("SPACE_GRAY"), T("VOID_BLUE"), "sum", 1.2),
    (T("ASTRO_BLACK"), T("VOID_BLUE"), "sum", 1.2),
    (T("ECLIPSE_CHARCOAL"), T("VOID_BLUE"), "spread", 1.2),
]

EXPS = {
    # T22: T17 + ECLIPSE|SPACE_GRAY sum (T11+aggressive)
    "t22_t17_plus_eg_sum": dict(
        desc="T17 + ECLIPSE|SPACE_GRAY sum (aggr)",
        exit_z=0.3, passive=False,
        pairs=T17 + [(T("ECLIPSE_CHARCOAL"), T("SPACE_GRAY"), "sum", 1.2)]),
    # T23: T17 + ASTRO|ECLIPSE sum
    "t23_t17_plus_ae_sum": dict(
        desc="T17 + ASTRO|ECLIPSE sum (aggr)",
        exit_z=0.3, passive=False,
        pairs=T17 + [(T("ASTRO_BLACK"), T("ECLIPSE_CHARCOAL"), "sum", 1.2)]),
    # T24: T17 + ECLIPSE|GRAPHITE sum
    "t24_t17_plus_eg_graph_sum": dict(
        desc="T17 + ECLIPSE|GRAPHITE_MIST sum (aggr)",
        exit_z=0.3, passive=False,
        pairs=T17 + [(T("ECLIPSE_CHARCOAL"), T("GRAPHITE_MIST"), "sum", 1.2)]),
    # T25: T17 + 3 EDA pairs (top-7 aggr)
    "t25_t17_plus_3": dict(
        desc="T17 + 3 EDA pairs (aggr)",
        exit_z=0.3, passive=False,
        pairs=T17 + [
            (T("ECLIPSE_CHARCOAL"), T("SPACE_GRAY"), "sum", 1.2),
            (T("ASTRO_BLACK"), T("ECLIPSE_CHARCOAL"), "sum", 1.2),
            (T("ECLIPSE_CHARCOAL"), T("GRAPHITE_MIST"), "sum", 1.2),
        ]),
    # T26: T17 z=1.0
    "t26_t17_z10": dict(
        desc="T17 z=1.0",
        exit_z=0.3, passive=False,
        pairs=[(a,b,s,1.0) for (a,b,s,z) in T17]),
    # T27: T17 z=1.5
    "t27_t17_z15": dict(
        desc="T17 z=1.5",
        exit_z=0.3, passive=False,
        pairs=[(a,b,s,1.5) for (a,b,s,z) in T17]),
    # T28: T17 z=0.8
    "t28_t17_z08": dict(
        desc="T17 z=0.8 (very aggressive)",
        exit_z=0.3, passive=False,
        pairs=[(a,b,s,0.8) for (a,b,s,z) in T17]),
    # T29: T17 EXIT_Z=0.1
    "t29_t17_exit01": dict(
        desc="T17 EXIT_Z=0.1",
        exit_z=0.1, passive=False, pairs=T17),
    # T30: T17 EXIT_Z=0.5
    "t30_t17_exit05": dict(
        desc="T17 EXIT_Z=0.5",
        exit_z=0.5, passive=False, pairs=T17),
    # T31: T17 drop SPACE_GRAY|GRAPHITE_MIST
    "t31_t17_drop_sg_gm": dict(
        desc="T17 minus SPACE_GRAY|GRAPHITE_MIST",
        exit_z=0.3, passive=False,
        pairs=[p for p in T17 if not (p[0]==T("SPACE_GRAY") and p[1]==T("GRAPHITE_MIST"))]),
    # T32: T17 drop SPACE_GRAY|VOID_BLUE
    "t32_t17_drop_sg_vb": dict(
        desc="T17 minus SPACE_GRAY|VOID_BLUE",
        exit_z=0.3, passive=False,
        pairs=[p for p in T17 if not (p[0]==T("SPACE_GRAY") and p[1]==T("VOID_BLUE"))]),
    # T33: T17 mixed z (rank-aware)
    "t33_t17_rankz": dict(
        desc="T17 z by EDA rank (high-std → high z)",
        exit_z=0.3, passive=False,
        pairs=[
            (T("ASTRO_BLACK"), T("VOID_BLUE"), "sum", 1.0),       # rank 1
            (T("ECLIPSE_CHARCOAL"), T("VOID_BLUE"), "spread", 1.0),  # rank 2
            (T("SPACE_GRAY"), T("VOID_BLUE"), "sum", 1.2),         # rank 3
            (T("SPACE_GRAY"), T("GRAPHITE_MIST"), "spread", 1.5),  # rank 12
        ]),
}

OUT = Path(__file__).parent
for name, cfg in EXPS.items():
    p = OUT / f"{name}.py"
    p.write_text(TEMPLATE.format(name=name, desc=cfg["desc"], pairs=cfg["pairs"], exit_z=cfg["exit_z"], passive=cfg["passive"]))
    print("wrote", p.name)
