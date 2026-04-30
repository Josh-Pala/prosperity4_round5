"""Generate experiment files. Each file is template + custom PAIRS list."""
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

EXPERIMENTS = {
    # H0 — baseline (current 5 pairs from mm_pairs_full)
    "h0_baseline": dict(
        desc="Baseline: 5 pairs from mm_pairs_full as-is.",
        exit_z=0.3,
        pairs=[
            ("ROBOT_VACUUMING", "ROBOT_LAUNDRY", "spread", 1.0),
            ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.0),
            ("ROBOT_MOPPING", "ROBOT_LAUNDRY", "sum", 2.0),
            ("ROBOT_LAUNDRY", "ROBOT_IRONING", "spread", 1.2),
            ("ROBOT_MOPPING", "ROBOT_IRONING", "sum", 1.5),
        ],
    ),

    # H1 — proposed A: 5 pairs with diversified anchor (top-5 by EDA std)
    "h1_diversified": dict(
        desc="Diversified anchor: top 5 pairs by EDA residual std. LAUNDRY appears in 2 (was 4).",
        exit_z=0.3,
        pairs=[
            ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.0),
            ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.0),
            ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
            ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
            ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
        ],
    ),

    # H2 — A + B: diversified + tuned z (higher z on pairs with high z>2.0 freq)
    "h2_diversified_tuned": dict(
        desc="Diversified + tuned entry_z based on z>2.0 frequency.",
        exit_z=0.3,
        pairs=[
            ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.2),
            ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.5),
            ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
            ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
            ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
        ],
    ),

    # H3 — top-6 by EDA: add IRONING|VACUUMING spread (rank 6) for full coverage
    "h3_top6": dict(
        desc="Top 6 pairs by EDA: balanced 6-pair set covering all symbols evenly.",
        exit_z=0.3,
        pairs=[
            ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.0),
            ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.0),
            ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
            ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
            ("ROBOT_IRONING", "ROBOT_VACUUMING", "spread", 1.2),
            ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
        ],
    ),

    # H4 — minimalist 3 pairs (top by std and crossings)
    "h4_top3": dict(
        desc="Top-3 minimal: less leg-sharing, smaller portfolio.",
        exit_z=0.3,
        pairs=[
            ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.0),
            ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.0),
            ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
        ],
    ),

    # H5 — H1 with tighter exit (more cycles)
    "h5_tighter_exit": dict(
        desc="H1 set with EXIT_Z=0.1 (close earlier, more cycles).",
        exit_z=0.1,
        pairs=[
            ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.0),
            ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.0),
            ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
            ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
            ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
        ],
    ),

    # H6 — H1 with looser exit
    "h6_looser_exit": dict(
        desc="H1 set with EXIT_Z=0.5 (hold longer).",
        exit_z=0.5,
        pairs=[
            ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.0),
            ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.0),
            ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
            ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
            ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
        ],
    ),

    # H7 — aggressive entry (lower z, more trades)
    "h7_aggressive_z": dict(
        desc="H1 with all entry_z = 0.8 (more trades, weaker signals).",
        exit_z=0.3,
        pairs=[
            ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 0.8),
            ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 0.8),
            ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 0.8),
            ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 0.8),
            ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 0.8),
        ],
    ),

    # H8 — selective entry (higher z, fewer but better trades)
    "h8_selective_z": dict(
        desc="H1 with all entry_z = 2.0 (selective, higher quality).",
        exit_z=0.3,
        pairs=[
            ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 2.0),
            ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 2.0),
            ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 2.0),
            ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 2.0),
            ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
        ],
    ),
}

OUT = Path(__file__).parent
for name, cfg in EXPERIMENTS.items():
    p = OUT / f"{name}.py"
    p.write_text(TEMPLATE_HEADER.format(name=name, desc=cfg["desc"], pairs=cfg["pairs"], exit_z=cfg["exit_z"]))
    print("wrote", p.name)
