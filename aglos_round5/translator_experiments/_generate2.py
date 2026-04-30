"""Round 2 — neighbourhood of T6 winner (current 2 + top-2 EDA = +17.7k)."""
from pathlib import Path

TEMPLATE = open(Path(__file__).parent / "_generate.py").read().split("EXPS")[0]
# Reuse template: hack — extract template literal
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

# T6 winner pairs:
T6 = [
    (T("SPACE_GRAY"), T("GRAPHITE_MIST"), "spread", 1.2),
    (T("SPACE_GRAY"), T("VOID_BLUE"), "sum", 1.2),
    (T("ASTRO_BLACK"), T("VOID_BLUE"), "sum", 1.2),
    (T("ECLIPSE_CHARCOAL"), T("VOID_BLUE"), "spread", 1.2),
]

EXPS = {
    # T11: T6 + ECLIPSE|SPACE_GRAY sum (rank 5)
    "t11_t6_plus_eg_sum": dict(
        desc="T6 + ECLIPSE|SPACE_GRAY sum",
        exit_z=0.3, passive=True,
        pairs=T6 + [(T("ECLIPSE_CHARCOAL"), T("SPACE_GRAY"), "sum", 1.2)]),
    # T12: T6 + ECLIPSE|GRAPHITE sum (rank 6)
    "t12_t6_plus_egraph_sum": dict(
        desc="T6 + ECLIPSE|GRAPHITE_MIST sum",
        exit_z=0.3, passive=True,
        pairs=T6 + [(T("ECLIPSE_CHARCOAL"), T("GRAPHITE_MIST"), "sum", 1.2)]),
    # T13: T6 + ASTRO|ECLIPSE sum (rank 4)
    "t13_t6_plus_ae_sum": dict(
        desc="T6 + ASTRO_BLACK|ECLIPSE_CHARCOAL sum",
        exit_z=0.3, passive=True,
        pairs=T6 + [(T("ASTRO_BLACK"), T("ECLIPSE_CHARCOAL"), "sum", 1.2)]),
    # T14: T6 with z=1.0 (more aggressive)
    "t14_t6_z10": dict(
        desc="T6 all entry_z=1.0",
        exit_z=0.3, passive=True,
        pairs=[(a,b,s,1.0) for (a,b,s,z) in T6]),
    # T15: T6 with z=1.5
    "t15_t6_z15": dict(
        desc="T6 all entry_z=1.5",
        exit_z=0.3, passive=True,
        pairs=[(a,b,s,1.5) for (a,b,s,z) in T6]),
    # T16: T6 + 3 best EDA (top-7)
    "t16_t6_plus_3": dict(
        desc="T6 + top-3 unused EDA pairs",
        exit_z=0.3, passive=True,
        pairs=T6 + [
            (T("ASTRO_BLACK"), T("ECLIPSE_CHARCOAL"), "sum", 1.2),
            (T("ECLIPSE_CHARCOAL"), T("SPACE_GRAY"), "sum", 1.2),
            (T("ECLIPSE_CHARCOAL"), T("GRAPHITE_MIST"), "sum", 1.2),
        ]),
    # T17: T6 aggressive instead of passive
    "t17_t6_aggressive": dict(
        desc="T6 with taker execution",
        exit_z=0.3, passive=False,
        pairs=T6),
    # T18: only the 2 best EDA pairs (drop current)
    "t18_only_top2": dict(
        desc="just rank 1+2 (no current pairs)",
        exit_z=0.3, passive=True,
        pairs=[
            (T("ASTRO_BLACK"), T("VOID_BLUE"), "sum", 1.2),
            (T("ECLIPSE_CHARCOAL"), T("VOID_BLUE"), "spread", 1.2),
        ]),
    # T19: drop SPACE_GRAY|GRAPHITE_MIST (rank 12 = lowest of current)
    "t19_t6_drop_sg_gm": dict(
        desc="T6 minus SPACE_GRAY|GRAPHITE_MIST",
        exit_z=0.3, passive=True,
        pairs=[p for p in T6 if not (p[0]==T("SPACE_GRAY") and p[1]==T("GRAPHITE_MIST"))]),
    # T20: drop SPACE_GRAY|VOID_BLUE (rank 3)
    "t20_t6_drop_sg_vb": dict(
        desc="T6 minus SPACE_GRAY|VOID_BLUE",
        exit_z=0.3, passive=True,
        pairs=[p for p in T6 if not (p[0]==T("SPACE_GRAY") and p[1]==T("VOID_BLUE"))]),
    # T21: T6 with mixed z (selective for high-std pairs)
    "t21_t6_mixed_z": dict(
        desc="T6 with z scaled by std rank",
        exit_z=0.3, passive=True,
        pairs=[
            (T("SPACE_GRAY"), T("GRAPHITE_MIST"), "spread", 1.5),  # rank 12, selective
            (T("SPACE_GRAY"), T("VOID_BLUE"), "sum", 1.2),         # rank 3
            (T("ASTRO_BLACK"), T("VOID_BLUE"), "sum", 1.0),        # rank 1, aggressive
            (T("ECLIPSE_CHARCOAL"), T("VOID_BLUE"), "spread", 1.0),  # rank 2, aggressive
        ]),
}

OUT = Path(__file__).parent
for name, cfg in EXPS.items():
    p = OUT / f"{name}.py"
    p.write_text(TEMPLATE.format(name=name, desc=cfg["desc"], pairs=cfg["pairs"], exit_z=cfg["exit_z"], passive=cfg["passive"]))
    print("wrote", p.name)
