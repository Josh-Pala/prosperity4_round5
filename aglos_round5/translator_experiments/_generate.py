"""Generate TRANSLATOR experiment files with passive-entry execution
(consistent with v3.1 routing for this family)."""
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

EXPS = {
    # T0: baseline = current v3.1
    "t0_baseline": dict(
        desc="current v3.1 (2 pairs, passive entry)",
        exit_z=0.3, passive=True,
        pairs=[
            (T("SPACE_GRAY"), T("GRAPHITE_MIST"), "spread", 1.2),
            (T("SPACE_GRAY"), T("VOID_BLUE"), "sum", 1.2),
        ]),
    # T1: top-2 by EDA std
    "t1_top2": dict(
        desc="top-2 EDA pairs (rank 1+2)",
        exit_z=0.3, passive=True,
        pairs=[
            (T("ASTRO_BLACK"), T("VOID_BLUE"), "sum", 1.2),
            (T("ECLIPSE_CHARCOAL"), T("VOID_BLUE"), "spread", 1.2),
        ]),
    # T2: top-3 by EDA std (anchor diversified)
    "t2_top3": dict(
        desc="top-3 EDA pairs",
        exit_z=0.3, passive=True,
        pairs=[
            (T("ASTRO_BLACK"), T("VOID_BLUE"), "sum", 1.2),
            (T("ECLIPSE_CHARCOAL"), T("VOID_BLUE"), "spread", 1.2),
            (T("SPACE_GRAY"), T("VOID_BLUE"), "sum", 1.2),
        ]),
    # T3: top-4
    "t3_top4": dict(
        desc="top-4 EDA pairs",
        exit_z=0.3, passive=True,
        pairs=[
            (T("ASTRO_BLACK"), T("VOID_BLUE"), "sum", 1.2),
            (T("ECLIPSE_CHARCOAL"), T("VOID_BLUE"), "spread", 1.2),
            (T("SPACE_GRAY"), T("VOID_BLUE"), "sum", 1.2),
            (T("ASTRO_BLACK"), T("ECLIPSE_CHARCOAL"), "sum", 1.2),
        ]),
    # T4: top-5 (covers all 5 symbols with 5 pairs)
    "t4_top5": dict(
        desc="top-5 EDA pairs",
        exit_z=0.3, passive=True,
        pairs=[
            (T("ASTRO_BLACK"), T("VOID_BLUE"), "sum", 1.2),
            (T("ECLIPSE_CHARCOAL"), T("VOID_BLUE"), "spread", 1.2),
            (T("SPACE_GRAY"), T("VOID_BLUE"), "sum", 1.2),
            (T("ASTRO_BLACK"), T("ECLIPSE_CHARCOAL"), "sum", 1.2),
            (T("ECLIPSE_CHARCOAL"), T("SPACE_GRAY"), "sum", 1.2),
        ]),
    # T5: diversified anchor (avoid VOID_BLUE always being one leg)
    "t5_diversified": dict(
        desc="diversified anchor (no symbol used >2 times)",
        exit_z=0.3, passive=True,
        pairs=[
            (T("ASTRO_BLACK"), T("VOID_BLUE"), "sum", 1.2),       # rank 1
            (T("ECLIPSE_CHARCOAL"), T("VOID_BLUE"), "spread", 1.2),  # rank 2
            (T("ASTRO_BLACK"), T("ECLIPSE_CHARCOAL"), "sum", 1.2),  # rank 4
            (T("SPACE_GRAY"), T("ECLIPSE_CHARCOAL"), "sum", 1.2),  # rank 5
            (T("ECLIPSE_CHARCOAL"), T("GRAPHITE_MIST"), "sum", 1.2),  # rank 6
        ]),
    # T6: include one current pair we'd be replacing (sanity)
    "t6_extend_current": dict(
        desc="add top-2 EDA pairs to current 2",
        exit_z=0.3, passive=True,
        pairs=[
            (T("SPACE_GRAY"), T("GRAPHITE_MIST"), "spread", 1.2),
            (T("SPACE_GRAY"), T("VOID_BLUE"), "sum", 1.2),
            (T("ASTRO_BLACK"), T("VOID_BLUE"), "sum", 1.2),
            (T("ECLIPSE_CHARCOAL"), T("VOID_BLUE"), "spread", 1.2),
        ]),
    # T7: aggressive entry on T5
    "t7_t5_z1.0": dict(
        desc="T5 with all entry_z=1.0",
        exit_z=0.3, passive=True,
        pairs=[
            (T("ASTRO_BLACK"), T("VOID_BLUE"), "sum", 1.0),
            (T("ECLIPSE_CHARCOAL"), T("VOID_BLUE"), "spread", 1.0),
            (T("ASTRO_BLACK"), T("ECLIPSE_CHARCOAL"), "sum", 1.0),
            (T("SPACE_GRAY"), T("ECLIPSE_CHARCOAL"), "sum", 1.0),
            (T("ECLIPSE_CHARCOAL"), T("GRAPHITE_MIST"), "sum", 1.0),
        ]),
    # T8: selective entry on T5
    "t8_t5_z1.5": dict(
        desc="T5 with all entry_z=1.5",
        exit_z=0.3, passive=True,
        pairs=[
            (T("ASTRO_BLACK"), T("VOID_BLUE"), "sum", 1.5),
            (T("ECLIPSE_CHARCOAL"), T("VOID_BLUE"), "spread", 1.5),
            (T("ASTRO_BLACK"), T("ECLIPSE_CHARCOAL"), "sum", 1.5),
            (T("SPACE_GRAY"), T("ECLIPSE_CHARCOAL"), "sum", 1.5),
            (T("ECLIPSE_CHARCOAL"), T("GRAPHITE_MIST"), "sum", 1.5),
        ]),
    # T9: T5 aggressive (taker) instead of passive
    "t9_t5_aggressive": dict(
        desc="T5 with taker execution (test family routing)",
        exit_z=0.3, passive=False,
        pairs=[
            (T("ASTRO_BLACK"), T("VOID_BLUE"), "sum", 1.2),
            (T("ECLIPSE_CHARCOAL"), T("VOID_BLUE"), "spread", 1.2),
            (T("ASTRO_BLACK"), T("ECLIPSE_CHARCOAL"), "sum", 1.2),
            (T("SPACE_GRAY"), T("ECLIPSE_CHARCOAL"), "sum", 1.2),
            (T("ECLIPSE_CHARCOAL"), T("GRAPHITE_MIST"), "sum", 1.2),
        ]),
    # T10: top-6 (add ECLIPSE_CHARCOAL|GRAPHITE_MIST spread)
    "t10_top6": dict(
        desc="top-6 EDA pairs",
        exit_z=0.3, passive=True,
        pairs=[
            (T("ASTRO_BLACK"), T("VOID_BLUE"), "sum", 1.2),
            (T("ECLIPSE_CHARCOAL"), T("VOID_BLUE"), "spread", 1.2),
            (T("SPACE_GRAY"), T("VOID_BLUE"), "sum", 1.2),
            (T("ASTRO_BLACK"), T("ECLIPSE_CHARCOAL"), "sum", 1.2),
            (T("ECLIPSE_CHARCOAL"), T("SPACE_GRAY"), "sum", 1.2),
            (T("ECLIPSE_CHARCOAL"), T("GRAPHITE_MIST"), "sum", 1.2),
        ]),
}

OUT = Path(__file__).parent
for name, cfg in EXPS.items():
    p = OUT / f"{name}.py"
    p.write_text(TEMPLATE.format(name=name, desc=cfg["desc"], pairs=cfg["pairs"], exit_z=cfg["exit_z"], passive=cfg["passive"]))
    print("wrote", p.name)
