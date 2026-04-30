"""
v03 — Disjoint combo:
  - Pair z-score on XS|S (target ±10 each, opposite sign).
  - Constant-sum-aware MM on M, L, XL only.

Rationale: v01 and v02 both touch XS/S but via different mechanics (active z-score vs
passive MM). To avoid signal cancellation, dedicate XS/S to v01 (which made +69k) and
restrict v02's MM logic to the 3 leftover legs M, L, XL.

The MM "fair" for each of {M,L,XL} is derived from the constant-sum invariant
restricted to those 3:
    fair_M = (50_000 - mid_XS - mid_S) - mid_L - mid_XL
i.e. fair_i = (50_000 - mid_XS - mid_S) - sum_{j in {M,L,XL}, j!=i} mid_j

Note: 50_000 - mid_XS - mid_S floats with the pair, but the invariant still holds:
mid_M + mid_L + mid_XL ≈ 50_000 - mid_XS - mid_S, so each leg's fair is well-defined.
"""
from __future__ import annotations
from typing import Dict, List
import jsonpickle
from datamodel import Order, OrderDepth, TradingState

# ── Pair component ─────────────────────────────────────────────────────────
PAIR = ("PEBBLES_XS", "PEBBLES_S", "spread", 2.0)
WARMUP = 500
EXIT_Z = 0.3
PAIR_SIZE = 10

# ── MM component (M, L, XL) ─────────────────────────────────────────────────
MM_LEGS = ["PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
ALL_PEBBLES = ["PEBBLES_XS", "PEBBLES_S"] + MM_LEGS
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


def _mid(od: OrderDepth):
    bb, ba, _, _ = _best_bid_ask(od)
    if bb is None or ba is None:
        return None
    return (bb + ba) / 2.0


def _pair_key(a, b, sign):
    return f"{a}|{b}|{sign}"


class Trader:
    def run(self, state: TradingState):
        store = jsonpickle.decode(state.traderData) if state.traderData else {}
        stats = store.setdefault("stats", {})
        targets = store.setdefault("target", {})

        # Snapshot all PEBBLES order books
        books = {}
        for s in ALL_PEBBLES:
            od = state.order_depths.get(s)
            if od is None:
                return {}, 0, jsonpickle.encode(store)
            bb, ba, bq, aq = _best_bid_ask(od)
            if bb is None or ba is None:
                return {}, 0, jsonpickle.encode(store)
            books[s] = (bb, ba, bq, aq, (bb + ba) / 2.0)

        result: Dict[str, List[Order]] = {s: [] for s in ALL_PEBBLES}

        # ── 1) PAIR component on XS|S ─────────────────────────────────────
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

        # Place pair orders (taker)
        ta = PAIR_SIZE * cur
        tb = (-PAIR_SIZE if sign == "spread" else PAIR_SIZE) * cur
        for sym, tgt in [(a, ta), (b, tb)]:
            tgt = max(-LIMIT, min(LIMIT, tgt))
            current = state.position.get(sym, 0)
            d = tgt - current
            if d == 0:
                continue
            bb, ba, bq, aq, _ = books[sym]
            if d > 0:
                result[sym].append(Order(sym, ba, d))
            else:
                result[sym].append(Order(sym, bb, d))

        # ── 2) MM component on M, L, XL ───────────────────────────────────
        # Invariant restricted to MM_LEGS:
        #   sum_mm_mid := mid_M + mid_L + mid_XL
        # We'd like sum_mm_mid ≈ 50_000 - mid_XS - mid_S, so:
        sum_xs_s = books["PEBBLES_XS"][4] + books["PEBBLES_S"][4]
        sum_mm_target = SUM_INVARIANT - sum_xs_s
        sum_mm_mid = sum(books[s][4] for s in MM_LEGS)

        for sym in MM_LEGS:
            bb, ba, bq, aq, mid = books[sym]
            spread = ba - bb
            pos = state.position.get(sym, 0)
            buy_cap = LIMIT - pos
            sell_cap = LIMIT + pos
            # fair leg = sum_mm_target - sum_others_mid
            others = sum(books[s][4] for s in MM_LEGS if s != sym)
            fair = sum_mm_target - others

            scale_long = max(0.0, 1.0 - max(0, pos - SOFT_POS) / float(LIMIT - SOFT_POS + 1e-9))
            scale_short = max(0.0, 1.0 - max(0, -pos - SOFT_POS) / float(LIMIT - SOFT_POS + 1e-9))

            # Taker
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

            # Maker (passive quotes inside the spread)
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

        # Drop empty entries
        result = {k: v for k, v in result.items() if v}
        return result, 0, jsonpickle.encode(store)
