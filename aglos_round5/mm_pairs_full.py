"""
mm_pairs_full — pair trading + MM on the FULL 27-symbol residual.

Same pair layer as mm_pairs.py / combined_v5_cv_robust. The MM layer is
expanded from 14 to 27 symbols: every product in the round-5 universe
that is NOT touched by any pair leg. Symbol sets remain disjoint, so
position limits don't collide.

Residual derivation: full 50-product universe minus the 23 unique pair
legs from PAIRS below. Verified by analysis/_carveout.py reconciliation:
mm_v3 on residual-27 alone produces 206,243; pair layer alone produces
355,621. Theoretical combined: 561,864.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import jsonpickle
from datamodel import Order, OrderDepth, TradingState


PAIRS: List[Tuple[str, str, str, float]] = [
    ("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE", "spread", 1.2),
    ("PEBBLES_XS", "PEBBLES_S", "spread", 2.0),
    ("TRANSLATOR_SPACE_GRAY", "TRANSLATOR_GRAPHITE_MIST", "spread", 1.2),
    ("MICROCHIP_CIRCLE", "MICROCHIP_TRIANGLE", "sum", 1.5),
    ("PANEL_2X4", "PANEL_4X4", "sum", 2.0),
    ("ROBOT_VACUUMING", "ROBOT_LAUNDRY", "spread", 1.0),
    ("SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY", "sum", 1.5),
    ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.0),
    ("ROBOT_MOPPING", "ROBOT_LAUNDRY", "sum", 2.0),
    ("SLEEP_POD_POLYESTER", "SLEEP_POD_COTTON", "spread", 1.8),
    ("SNACKPACK_VANILLA", "SNACKPACK_RASPBERRY", "spread", 1.2),
    ("SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO", "sum", 1.5),
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO", "spread", 1.8),
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_STRAWBERRY", "sum", 1.8),
    ("MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE", "spread", 2.0),
    ("TRANSLATOR_SPACE_GRAY", "TRANSLATOR_VOID_BLUE", "sum", 1.2),
    ("ROBOT_LAUNDRY", "ROBOT_IRONING", "spread", 1.2),
    ("ROBOT_MOPPING", "ROBOT_IRONING", "sum", 1.5),
]

# All 27 round-5 products NOT used by any pair leg, sorted DESC by mean
# quoted half-spread (analysis/out/mm_spread_scan.csv).
MM_UNIVERSE: List[str] = [
    "PEBBLES_XL",
    "OXYGEN_SHAKE_GARLIC",
    "GALAXY_SOUNDS_BLACK_HOLES",
    "UV_VISOR_MAGENTA",
    "GALAXY_SOUNDS_SOLAR_FLAMES",
    "UV_VISOR_RED",
    "UV_VISOR_YELLOW",
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "GALAXY_SOUNDS_SOLAR_WINDS",
    "UV_VISOR_ORANGE",
    "PEBBLES_M",
    "GALAXY_SOUNDS_DARK_MATTER",
    "PEBBLES_L",
    "OXYGEN_SHAKE_MORNING_BREATH",
    "OXYGEN_SHAKE_MINT",
    "OXYGEN_SHAKE_CHOCOLATE",
    "OXYGEN_SHAKE_EVENING_BREATH",
    "MICROCHIP_SQUARE",
    "PANEL_1X2",
    "UV_VISOR_AMBER",
    "SLEEP_POD_SUEDE",
    "SLEEP_POD_LAMB_WOOL",
    "TRANSLATOR_ECLIPSE_CHARCOAL",
    "SLEEP_POD_NYLON",
    "PANEL_2X2",
    "PANEL_1X4",
    "TRANSLATOR_ASTRO_BLACK",
]

LIMIT = 10
WARMUP = 500
EXIT_Z = 0.3
SIZE = 10

MM_QUOTE_SIZE = 5
MM_SKEW_PER_UNIT = 0.0
MM_MIN_SPREAD = 3


def mid_of(d: OrderDepth):
    if not d.buy_orders or not d.sell_orders:
        return None
    return (max(d.buy_orders.keys()) + min(d.sell_orders.keys())) / 2.0


def best_levels(d: OrderDepth):
    if not d.buy_orders or not d.sell_orders:
        return None, None
    return max(d.buy_orders.keys()), min(d.sell_orders.keys())


def pair_key(a, b, sign):
    return f"{a}|{b}|{sign}"


class Trader:
    def run(self, state: TradingState):
        store = jsonpickle.decode(state.traderData) if state.traderData else {}
        stats = store.setdefault("stats", {})
        targets = store.setdefault("target", {})

        # ---- Pair-trading layer ----
        leg_targets: Dict[str, int] = {}
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
            st = stats.setdefault(key, {"n": 0, "mean": 0.0, "M2": 0.0})
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

        result: Dict[str, List[Order]] = {}
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

        # ---- MM layer (disjoint product set) ----
        for sym in MM_UNIVERSE:
            depth = state.order_depths.get(sym)
            if depth is None:
                continue
            best_bid, best_ask = best_levels(depth)
            if best_bid is None or best_ask is None:
                continue
            spread = best_ask - best_bid
            if spread < MM_MIN_SPREAD:
                continue

            position = state.position.get(sym, 0)
            skew = int(round(position * MM_SKEW_PER_UNIT))
            bid_px = best_bid + 1 - skew
            ask_px = best_ask - 1 - skew
            if bid_px >= best_ask:
                bid_px = best_ask - 1
            if ask_px <= best_bid:
                ask_px = best_bid + 1

            buy_capacity = max(0, LIMIT - position)
            sell_capacity = max(0, LIMIT + position)
            buy_qty = min(MM_QUOTE_SIZE, buy_capacity)
            sell_qty = min(MM_QUOTE_SIZE, sell_capacity)

            if bid_px >= ask_px:
                if position > 0:
                    buy_qty = 0
                    ask_px = max(ask_px, best_bid + 1)
                elif position < 0:
                    sell_qty = 0
                    bid_px = min(bid_px, best_ask - 1)
                else:
                    bid_px = best_bid + 1
                    ask_px = best_ask - 1

            mm_orders: List[Order] = []
            if buy_qty > 0:
                mm_orders.append(Order(sym, int(bid_px), int(buy_qty)))
            if sell_qty > 0:
                mm_orders.append(Order(sym, int(ask_px), -int(sell_qty)))
            if mm_orders:
                result[sym] = mm_orders

        return result, 0, jsonpickle.encode(store)
