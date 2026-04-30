"""glauco_v4_trader — v1 + ONLY the PANEL pair tuning (entry_z 2.0 → 1.5).

Hypothesis after v2 (basket residual: -41k) and v3 (full PANEL MM block: -14k):
  - Both attempts failed because they OVER-engineered the family.
  - The single change with positive evidence is the realistic taker sim on
    PANEL_2X4|PANEL_4X4 sum: entry_z=1.5, exit_z=0.3 → 29,620 vs 27,850 at z=2.0.
  - Keep everything else from v1 untouched (PANEL_1X2/2X2/1X4 stay in MM_UNIVERSE
    with QUOTE_SIZE=5 — that's what worked).

Single delta vs v1:
  PAIRS line: ("PANEL_2X4", "PANEL_4X4", "sum", 1.5)  # was 2.0
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import jsonpickle
from datamodel import Order, OrderDepth, TradingState


PAIRS: List[Tuple[str, str, str, float]] = [
    ("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE", "spread", 1.2),
    ("MICROCHIP_CIRCLE", "MICROCHIP_TRIANGLE", "sum", 1.5),
    ("MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE", "spread", 2.0),
    # PANEL pair: z=1.5 (was 2.0). Realistic taker sim on 3 days = 29,620.
    ("PANEL_2X4", "PANEL_4X4", "sum", 1.5),
    ("SLEEP_POD_POLYESTER", "SLEEP_POD_COTTON", "spread", 1.8),
    ("SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY", "sum", 1.5),
    ("SNACKPACK_VANILLA", "SNACKPACK_RASPBERRY", "spread", 1.2),
    ("SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO", "sum", 1.5),
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO", "spread", 1.8),
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_STRAWBERRY", "sum", 1.8),
    ("TRANSLATOR_SPACE_GRAY", "TRANSLATOR_GRAPHITE_MIST", "spread", 1.2),
    ("TRANSLATOR_SPACE_GRAY", "TRANSLATOR_VOID_BLUE", "sum", 1.2),
    ("TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_VOID_BLUE", "sum", 1.2),
    ("TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_VOID_BLUE", "spread", 1.2),
    ("TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_SPACE_GRAY", "sum", 1.2),
    ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.0),
    ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.5),
    ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
    ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
    ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
    ("ROBOT_MOPPING", "ROBOT_VACUUMING", "sum", 1.2),
]

PASSIVE_ENTRY_FAMILIES = {"MICROCHIP", "SNACKPACK", "SLEEP_POD"}
EXIT_Z_BY_FAMILY: Dict[str, float] = {"TRANSLATOR": 0.2}

MM_UNIVERSE: List[str] = [
    "OXYGEN_SHAKE_GARLIC",
    "GALAXY_SOUNDS_BLACK_HOLES",
    "UV_VISOR_MAGENTA",
    "GALAXY_SOUNDS_SOLAR_FLAMES",
    "UV_VISOR_RED",
    "UV_VISOR_YELLOW",
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "GALAXY_SOUNDS_SOLAR_WINDS",
    "UV_VISOR_ORANGE",
    "GALAXY_SOUNDS_DARK_MATTER",
    "OXYGEN_SHAKE_MORNING_BREATH",
    "OXYGEN_SHAKE_MINT",
    "OXYGEN_SHAKE_CHOCOLATE",
    "OXYGEN_SHAKE_EVENING_BREATH",
    "MICROCHIP_SQUARE",
    "PANEL_1X2",
    "UV_VISOR_AMBER",
    "SLEEP_POD_SUEDE",
    "SLEEP_POD_LAMB_WOOL",
    "SLEEP_POD_NYLON",
    "PANEL_2X2",
    "PANEL_1X4",
    "TRANSLATOR_ASTRO_BLACK",
]

LIMIT = 10
WARMUP = 500
EXIT_Z_DEFAULT = 0.3
SIZE = 10

MM_QUOTE_SIZE = 5
MM_SKEW_PER_UNIT = 0.0
MM_MIN_SPREAD = 3

PEB_ENTRY_Z_P1 = 2.0
PEB_ENTRY_Z_P2 = 2.8
PEB_HALF_QUOTE = 10
PEB_EDGE_TAKE = 2
PEB_QUOTE_SIZE = 5
PEB_PAIRS: List[Tuple[str, str, str, float]] = [
    ("PEBBLES_XS", "PEBBLES_S", "spread", PEB_ENTRY_Z_P1),
    ("PEBBLES_S", "PEBBLES_M", "sum", PEB_ENTRY_Z_P2),
]
ALL_PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
PEB_SUM_INVARIANT = 50_000
PEB_PAIR_SIZE = 10
PEB_SOFT_POS = 6


def mid_of(d: OrderDepth):
    if not d.buy_orders or not d.sell_orders:
        return None
    return (max(d.buy_orders.keys()) + min(d.sell_orders.keys())) / 2.0


def best_levels(d: OrderDepth):
    if not d.buy_orders or not d.sell_orders:
        return None, None
    return max(d.buy_orders.keys()), min(d.sell_orders.keys())


def _bba(od: OrderDepth):
    bb = max(od.buy_orders.keys()) if od.buy_orders else None
    ba = min(od.sell_orders.keys()) if od.sell_orders else None
    bq = od.buy_orders.get(bb, 0) if bb is not None else 0
    aq = abs(od.sell_orders.get(ba, 0)) if ba is not None else 0
    return bb, ba, bq, aq


def pair_key(a, b, sign):
    return f"{a}|{b}|{sign}"


def family_of(symbol: str) -> str:
    for f in (
        "GALAXY_SOUNDS", "OXYGEN_SHAKE", "SLEEP_POD",
        "MICROCHIP", "PANEL", "PEBBLES", "ROBOT",
        "SNACKPACK", "TRANSLATOR", "UV_VISOR",
    ):
        if symbol.startswith(f + "_"):
            return f
    return ""


def exit_z_for(a: str, b: str) -> float:
    fa = family_of(a)
    fb = family_of(b)
    return max(
        EXIT_Z_BY_FAMILY.get(fa, EXIT_Z_DEFAULT),
        EXIT_Z_BY_FAMILY.get(fb, EXIT_Z_DEFAULT),
    )


class Trader:
    def run(self, state: TradingState):
        store = jsonpickle.decode(state.traderData) if state.traderData else {}
        stats = store.setdefault("stats", {})
        targets = store.setdefault("target", {})

        result: Dict[str, List[Order]] = {}

        leg_targets: Dict[str, int] = {}
        leg_passive_entry: Dict[str, bool] = {}
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
            exit_z_pair = exit_z_for(a, b)
            if cur == 0:
                if z > entry_z:
                    cur = -1
                elif z < -entry_z:
                    cur = +1
            else:
                if abs(z) < exit_z_pair:
                    cur = 0
            targets[key] = cur
            ta = SIZE * cur
            tb = (-SIZE if sign == "spread" else SIZE) * cur
            leg_targets[a] = leg_targets.get(a, 0) + ta
            leg_targets[b] = leg_targets.get(b, 0) + tb
            if family_of(a) in PASSIVE_ENTRY_FAMILIES:
                leg_passive_entry[a] = True
            if family_of(b) in PASSIVE_ENTRY_FAMILIES:
                leg_passive_entry[b] = True

        for p, tgt in leg_targets.items():
            tgt = max(-LIMIT, min(LIMIT, tgt))
            depth = state.order_depths.get(p)
            if depth is None:
                continue
            current = state.position.get(p, 0)
            d = tgt - current
            if d == 0:
                continue
            is_reducing = (current > 0 and d < 0) or (current < 0 and d > 0)
            use_passive = leg_passive_entry.get(p, False) and not is_reducing
            if use_passive:
                if d > 0 and depth.buy_orders:
                    result[p] = [Order(p, max(depth.buy_orders.keys()) + 1, d)]
                elif d < 0 and depth.sell_orders:
                    result[p] = [Order(p, min(depth.sell_orders.keys()) - 1, d)]
            else:
                if d > 0 and depth.sell_orders:
                    result[p] = [Order(p, min(depth.sell_orders.keys()), d)]
                elif d < 0 and depth.buy_orders:
                    result[p] = [Order(p, max(depth.buy_orders.keys()), d)]

        engaged_pair_legs = set(leg_targets.keys())

        for sym in MM_UNIVERSE:
            if sym in engaged_pair_legs:
                continue
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

        books = {}
        peb_ok = True
        for s in ALL_PEBBLES:
            od = state.order_depths.get(s)
            if od is None:
                peb_ok = False
                break
            bb, ba, bq, aq = _bba(od)
            if bb is None or ba is None:
                peb_ok = False
                break
            books[s] = (bb, ba, bq, aq, (bb + ba) / 2.0)

        if peb_ok:
            peb_leg_targets: Dict[str, int] = {s: 0 for s in ALL_PEBBLES}
            for a, b, sign, entry_z in PEB_PAIRS:
                ma, mb = books[a][4], books[b][4]
                sig = ma - mb if sign == "spread" else ma + mb
                key = pair_key(a, b, sign)
                st = stats.setdefault(key, {"n": 0, "mean": 0.0, "M2": 0.0})
                st["n"] += 1
                n = st["n"]
                delta = sig - st["mean"]
                st["mean"] += delta / n
                st["M2"] += delta * (sig - st["mean"])
                cur = targets.get(key, 0)
                if n >= WARMUP:
                    var = st["M2"] / (n - 1) if n > 1 else 0.0
                    sd = var ** 0.5
                    if sd > 1e-9:
                        z = (sig - st["mean"]) / sd
                        if cur == 0:
                            cur = -1 if z > entry_z else (+1 if z < -entry_z else 0)
                        else:
                            if abs(z) < EXIT_Z_DEFAULT:
                                cur = 0
                targets[key] = cur
                if cur != 0:
                    peb_leg_targets[a] += PEB_PAIR_SIZE * cur
                    peb_leg_targets[b] += (
                        -PEB_PAIR_SIZE if sign == "spread" else PEB_PAIR_SIZE
                    ) * cur

            peb_orders: Dict[str, List[Order]] = {s: [] for s in ALL_PEBBLES}
            engaged_legs = set()
            for sym, tgt in peb_leg_targets.items():
                if tgt == 0:
                    continue
                engaged_legs.add(sym)
                tgt = max(-LIMIT, min(LIMIT, tgt))
                current = state.position.get(sym, 0)
                d = tgt - current
                if d == 0:
                    continue
                bb, ba, *_ = books[sym]
                is_reducing = (current > 0 and d < 0) or (current < 0 and d > 0)
                if is_reducing:
                    if d > 0:
                        peb_orders[sym].append(Order(sym, ba, d))
                    else:
                        peb_orders[sym].append(Order(sym, bb, d))
                else:
                    if d > 0:
                        peb_orders[sym].append(Order(sym, bb + 1, d))
                    else:
                        peb_orders[sym].append(Order(sym, ba - 1, d))

            mm_legs = [s for s in ALL_PEBBLES if s not in engaged_legs]
            sum_mid_all = sum(books[s][4] for s in ALL_PEBBLES)
            for sym in mm_legs:
                bb, ba, bq, aq, mid = books[sym]
                spread = ba - bb
                pos = state.position.get(sym, 0)
                buy_cap = LIMIT - pos
                sell_cap = LIMIT + pos
                fair = PEB_SUM_INVARIANT - (sum_mid_all - mid)
                scale_long = max(
                    0.0, 1.0 - max(0, pos - PEB_SOFT_POS) / float(LIMIT - PEB_SOFT_POS + 1e-9)
                )
                scale_short = max(
                    0.0, 1.0 - max(0, -pos - PEB_SOFT_POS) / float(LIMIT - PEB_SOFT_POS + 1e-9)
                )
                if buy_cap > 0:
                    edge = fair - ba
                    if edge >= PEB_EDGE_TAKE:
                        qty = min(buy_cap, aq, LIMIT)
                        if qty > 0:
                            peb_orders[sym].append(Order(sym, ba, qty))
                            buy_cap -= qty
                            pos += qty
                if sell_cap > 0:
                    edge = bb - fair
                    if edge >= PEB_EDGE_TAKE:
                        qty = min(sell_cap, bq, LIMIT)
                        if qty > 0:
                            peb_orders[sym].append(Order(sym, bb, -qty))
                            sell_cap -= qty
                            pos -= qty
                if spread >= 2:
                    target_bid = int(fair - PEB_HALF_QUOTE)
                    quote_bid = min(target_bid, ba - 1)
                    quote_bid = max(quote_bid, bb + 1)
                    if quote_bid <= bb or quote_bid >= ba:
                        quote_bid = None
                    target_ask = int(fair + PEB_HALF_QUOTE + 0.999)
                    quote_ask = max(target_ask, bb + 1)
                    quote_ask = min(quote_ask, ba - 1)
                    if quote_ask >= ba or quote_ask <= bb:
                        quote_ask = None
                    if quote_bid is not None and buy_cap > 0:
                        size = max(1, int(PEB_QUOTE_SIZE * scale_long))
                        size = min(size, buy_cap)
                        if size > 0:
                            peb_orders[sym].append(Order(sym, quote_bid, size))
                    if quote_ask is not None and sell_cap > 0:
                        size = max(1, int(PEB_QUOTE_SIZE * scale_short))
                        size = min(size, sell_cap)
                        if size > 0:
                            peb_orders[sym].append(Order(sym, quote_ask, -size))

            for sym, orders in peb_orders.items():
                if orders:
                    result[sym] = orders

        return result, 0, jsonpickle.encode(store)
