"""
v10 — v08 with tunable entry_z per pair (will be set via param sweep).
Same logic as v08; constants will be patched from outside for the sweep.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import jsonpickle
from datamodel import Order, OrderDepth, TradingState

# >>>SWEEP-START<<<
ENTRY_Z_P1 = 2.0
ENTRY_Z_P2 = 2.0
EXIT_Z = 0.3
HALF_QUOTE = 10
EDGE_TAKE = 2
QUOTE_SIZE = 5
# >>>SWEEP-END<<<

PAIRS: List[Tuple[str, str, str]] = [
    ("PEBBLES_XS", "PEBBLES_S", "spread"),
    ("PEBBLES_S", "PEBBLES_M", "sum"),
]
WARMUP = 500
PAIR_SIZE = 10
ALL_PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
SUM_INVARIANT = 50_000
LIMIT = 10
SOFT_POS = 6


def _bba(od: OrderDepth):
    bb = max(od.buy_orders.keys()) if od.buy_orders else None
    ba = min(od.sell_orders.keys()) if od.sell_orders else None
    bq = od.buy_orders.get(bb, 0) if bb is not None else 0
    aq = abs(od.sell_orders.get(ba, 0)) if ba is not None else 0
    return bb, ba, bq, aq


class Trader:
    def run(self, state: TradingState):
        store = jsonpickle.decode(state.traderData) if state.traderData else {}
        stats = store.setdefault("stats", {})
        targets = store.setdefault("target", {})

        books = {}
        for s in ALL_PEBBLES:
            od = state.order_depths.get(s)
            if od is None:
                return {}, 0, jsonpickle.encode(store)
            bb, ba, bq, aq = _bba(od)
            if bb is None or ba is None:
                return {}, 0, jsonpickle.encode(store)
            books[s] = (bb, ba, bq, aq, (bb + ba) / 2.0)

        leg_targets: Dict[str, int] = {s: 0 for s in ALL_PEBBLES}
        entry_zs = [ENTRY_Z_P1, ENTRY_Z_P2]
        for (a, b, sign), entry_z in zip(PAIRS, entry_zs):
            ma, mb = books[a][4], books[b][4]
            sig = ma - mb if sign == "spread" else ma + mb
            key = f"{a}|{b}|{sign}"
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
                        if abs(z) < EXIT_Z:
                            cur = 0
            targets[key] = cur
            if cur != 0:
                leg_targets[a] += PAIR_SIZE * cur
                leg_targets[b] += (-PAIR_SIZE if sign == "spread" else PAIR_SIZE) * cur

        result: Dict[str, List[Order]] = {s: [] for s in ALL_PEBBLES}
        engaged_legs = set()
        for sym, tgt in leg_targets.items():
            if tgt == 0:
                continue
            engaged_legs.add(sym)
            tgt = max(-LIMIT, min(LIMIT, tgt))
            current = state.position.get(sym, 0)
            d = tgt - current
            if d == 0:
                continue
            bb, ba, *_ = books[sym]
            if d > 0:
                result[sym].append(Order(sym, ba, d))
            else:
                result[sym].append(Order(sym, bb, d))

        mm_legs = [s for s in ALL_PEBBLES if s not in engaged_legs]
        sum_mid_all = sum(books[s][4] for s in ALL_PEBBLES)
        for sym in mm_legs:
            bb, ba, bq, aq, mid = books[sym]
            spread = ba - bb
            pos = state.position.get(sym, 0)
            buy_cap = LIMIT - pos
            sell_cap = LIMIT + pos
            fair = SUM_INVARIANT - (sum_mid_all - mid)
            scale_long = max(0.0, 1.0 - max(0, pos - SOFT_POS) / float(LIMIT - SOFT_POS + 1e-9))
            scale_short = max(0.0, 1.0 - max(0, -pos - SOFT_POS) / float(LIMIT - SOFT_POS + 1e-9))
            if buy_cap > 0:
                edge = fair - ba
                if edge >= EDGE_TAKE:
                    qty = min(buy_cap, aq, LIMIT)
                    if qty > 0:
                        result[sym].append(Order(sym, ba, qty))
                        buy_cap -= qty
                        pos += qty
            if sell_cap > 0:
                edge = bb - fair
                if edge >= EDGE_TAKE:
                    qty = min(sell_cap, bq, LIMIT)
                    if qty > 0:
                        result[sym].append(Order(sym, bb, -qty))
                        sell_cap -= qty
                        pos -= qty
            if spread >= 2:
                target_bid = int(fair - HALF_QUOTE)
                quote_bid = min(target_bid, ba - 1)
                quote_bid = max(quote_bid, bb + 1)
                if quote_bid <= bb or quote_bid >= ba:
                    quote_bid = None
                target_ask = int(fair + HALF_QUOTE + 0.999)
                quote_ask = max(target_ask, bb + 1)
                quote_ask = min(quote_ask, ba - 1)
                if quote_ask >= ba or quote_ask <= bb:
                    quote_ask = None
                if quote_bid is not None and buy_cap > 0:
                    size = max(1, int(QUOTE_SIZE * scale_long))
                    size = min(size, buy_cap)
                    if size > 0:
                        result[sym].append(Order(sym, quote_bid, size))
                if quote_ask is not None and sell_cap > 0:
                    size = max(1, int(QUOTE_SIZE * scale_short))
                    size = min(size, sell_cap)
                    if size > 0:
                        result[sym].append(Order(sym, quote_ask, -size))

        result = {k: v for k, v in result.items() if v}
        return result, 0, jsonpickle.encode(store)
