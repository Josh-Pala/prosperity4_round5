"""
v04 — Co-tenancy on XS/S:
  - Pair z-score targets ±5 on XS/S (instead of ±10 → leaves 5 cap headroom).
  - Constant-sum MM operates on all 5 legs.

When pair is flat, MM can use full ±10 on XS/S. When pair is engaged at ±5,
MM still has headroom ±5 in the *opposite* direction or +0 in same direction.
"""
from __future__ import annotations
from typing import Dict, List
import jsonpickle
from datamodel import Order, OrderDepth, TradingState

PAIR = ("PEBBLES_XS", "PEBBLES_S", "spread", 2.0)
WARMUP = 500
EXIT_Z = 0.3
PAIR_SIZE = 5

PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
SUM_INVARIANT = 50_000
LIMIT = 10
EDGE_TAKE = 2
HALF_QUOTE = 10
QUOTE_SIZE = 5
SOFT_POS = 6


def _best_bid_ask(od: OrderDepth):
    bb = max(od.buy_orders.keys()) if od.buy_orders else None
    ba = min(od.sell_orders.keys()) if od.sell_orders else None
    bq = od.buy_orders.get(bb, 0) if bb is not None else 0
    aq = abs(od.sell_orders.get(ba, 0)) if ba is not None else 0
    return bb, ba, bq, aq


def _pair_key(a, b, sign):
    return f"{a}|{b}|{sign}"


class Trader:
    def run(self, state: TradingState):
        store = jsonpickle.decode(state.traderData) if state.traderData else {}
        stats = store.setdefault("stats", {})
        targets = store.setdefault("target", {})

        books = {}
        for s in PEBBLES:
            od = state.order_depths.get(s)
            if od is None:
                return {}, 0, jsonpickle.encode(store)
            bb, ba, bq, aq = _best_bid_ask(od)
            if bb is None or ba is None:
                return {}, 0, jsonpickle.encode(store)
            books[s] = (bb, ba, bq, aq, (bb + ba) / 2.0)

        result: Dict[str, List[Order]] = {s: [] for s in PEBBLES}

        # ── 1) PAIR component (XS|S z-score, size ±PAIR_SIZE) ─────────────
        a, b, sign, entry_z = PAIR
        ma = books[a][4]
        mb = books[b][4]
        sig = ma - mb if sign == "spread" else ma + mb
        key = _pair_key(a, b, sign)
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
                    if z > entry_z:
                        cur = -1
                    elif z < -entry_z:
                        cur = +1
                else:
                    if abs(z) < EXIT_Z:
                        cur = 0
        targets[key] = cur

        # Pair targets (apply only if non-zero)
        pair_target = {a: PAIR_SIZE * cur, b: -PAIR_SIZE * cur if sign == "spread" else PAIR_SIZE * cur}
        for sym, tgt in pair_target.items():
            tgt = max(-LIMIT, min(LIMIT, tgt))
            current = state.position.get(sym, 0)
            d = tgt - current
            if d == 0 or cur == 0:
                # If pair flat, do not place pair orders (let MM do its thing)
                continue
            bb, ba, bq, aq, _ = books[sym]
            if d > 0:
                result[sym].append(Order(sym, ba, d))
            else:
                result[sym].append(Order(sym, bb, d))

        # ── 2) MM component on all 5 legs ─────────────────────────────────
        sum_mid = sum(books[s][4] for s in PEBBLES)
        for sym in PEBBLES:
            bb, ba, bq, aq, mid = books[sym]
            spread = ba - bb
            pos = state.position.get(sym, 0)
            # Reserve cap for the pair on XS/S
            reserved = abs(pair_target.get(sym, 0)) if cur != 0 else 0
            # Capacity left for MM after pair commitment
            buy_cap = LIMIT - pos - reserved if pair_target.get(sym, 0) > 0 else LIMIT - pos
            sell_cap = LIMIT + pos - reserved if pair_target.get(sym, 0) < 0 else LIMIT + pos
            buy_cap = max(0, buy_cap)
            sell_cap = max(0, sell_cap)

            fair = SUM_INVARIANT - (sum_mid - mid)

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
