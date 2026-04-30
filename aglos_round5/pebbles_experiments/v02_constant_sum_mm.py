import json
from typing import Any, Dict, List

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)

# ══════════════════════════════════════════════════════════════════════════════
# Logger (standard Prosperity 4 format — do not modify)
# ══════════════════════════════════════════════════════════════════════════════

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json([
                self.compress_state(state, ""),
                self.compress_orders(orders),
                conversions, "", "",
            ])
        )
        max_item_length = (self.max_log_length - base_length) // 3
        print(
            self.to_json([
                self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                self.compress_orders(orders),
                conversions,
                self.truncate(trader_data, max_item_length),
                self.truncate(self.logs, max_item_length),
            ])
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp, trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for t in arr:
                compressed.append(
                    [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        co = {}
        for product, obs in observations.conversionObservations.items():
            co[product] = [
                obs.bidPrice, obs.askPrice, obs.transportFees,
                obs.exportTariff, obs.importTariff, obs.sugarPrice, obs.sunlightIndex,
            ]
        return [observations.plainValueObservations, co]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for o in arr:
                compressed.append([o.symbol, o.price, o.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            if len(json.dumps(candidate)) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


# ══════════════════════════════════════════════════════════════════════════════
# Trader — PEBBLES constant-sum aware market maker / opportunistic taker
#
# Empirical (Round 5 days 2-4):
#   • Σ mid_i ≈ 50_000  (invariant, R²=1.0)
#   • Σ best_ask − 50_000 ≈ +33 always (median, σ < 1)
#   • Σ best_bid − 50_000 ≈ −33 always
#   • Per-leg spread ≈ 13, half-spread ≈ 6-7
#
# Therefore taking the spread leg-by-leg is unprofitable. Instead:
#
#   1. TAKE-when-mispriced: for each leg, define
#         fair_i = 50_000 − Σ_{j≠i} mid_j
#      Take best_ask if fair_i − best_ask >= EDGE_TAKE
#      Take best_bid if best_bid − fair_i >= EDGE_TAKE
#      EDGE_TAKE includes a buffer over half-spread.
#
#   2. PASSIVE QUOTING inside the spread, *anchored* on fair_i so that a fill
#      already locks in positive expected PnL vs invariant:
#         buy quote  = min(best_bid+1, floor(fair_i − HALF))
#         sell quote = max(best_ask-1, ceil (fair_i + HALF))
#      Quote only when it improves the BBO and stays inside the spread.
#
#   3. Position-aware sizing: skew quotes/sizes toward flat as |position| grows.
# ══════════════════════════════════════════════════════════════════════════════

PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
SUM_INVARIANT = 50_000
POSITION_LIMIT = 10
EDGE_TAKE = 2          # min seashells of edge vs fair to lift/hit (taker branch rarely fires)
HALF_QUOTE = 10        # half-distance from fair for passive quotes (tuned on R5 days 2-4)
QUOTE_SIZE = 5         # base size for passive quotes
SOFT_POS = 6           # start scaling quote size down past this |position|


def _best_bid_ask(od: OrderDepth) -> tuple[int | None, int | None, int, int]:
    best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
    best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
    bid_qty = od.buy_orders.get(best_bid, 0) if best_bid is not None else 0
    ask_qty = abs(od.sell_orders.get(best_ask, 0)) if best_ask is not None else 0
    return best_bid, best_ask, bid_qty, ask_qty


def _mid(od: OrderDepth) -> float | None:
    bb, ba, _, _ = _best_bid_ask(od)
    if bb is None or ba is None:
        return None
    return (bb + ba) / 2.0


class Trader:

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: Dict[str, List[Order]] = {s: [] for s in PEBBLES}
        conversions = 0

        # Snapshot
        mids: Dict[str, float] = {}
        bids: Dict[str, int] = {}
        asks: Dict[str, int] = {}
        bid_q: Dict[str, int] = {}
        ask_q: Dict[str, int] = {}
        for s in PEBBLES:
            od = state.order_depths.get(s)
            if od is None:
                logger.flush(state, result, conversions, "")
                return result, conversions, ""
            bb, ba, bq, aq = _best_bid_ask(od)
            if bb is None or ba is None:
                logger.flush(state, result, conversions, "")
                return result, conversions, ""
            bids[s] = bb
            asks[s] = ba
            bid_q[s] = bq
            ask_q[s] = aq
            mids[s] = (bb + ba) / 2.0

        total_mid = sum(mids.values())
        positions = {s: state.position.get(s, 0) for s in PEBBLES}

        for sym in PEBBLES:
            od = state.order_depths[sym]
            pos = positions[sym]
            buy_cap = POSITION_LIMIT - pos
            sell_cap = POSITION_LIMIT + pos
            fair = SUM_INVARIANT - (total_mid - mids[sym])
            best_bid, best_ask = bids[sym], asks[sym]
            spread = best_ask - best_bid

            # Buy-side capacity skew: smaller as we grow long
            scale_long = max(0.0, 1.0 - max(0, pos - SOFT_POS) / float(POSITION_LIMIT - SOFT_POS + 1e-9))
            scale_short = max(0.0, 1.0 - max(0, -pos - SOFT_POS) / float(POSITION_LIMIT - SOFT_POS + 1e-9))

            # ── 1) TAKER: cross only with positive edge vs fair ────────────────
            if buy_cap > 0:
                edge = fair - best_ask
                if edge >= EDGE_TAKE:
                    qty = min(buy_cap, ask_q[sym], POSITION_LIMIT)
                    if qty > 0:
                        result[sym].append(Order(sym, best_ask, qty))
                        buy_cap -= qty
                        pos += qty

            if sell_cap > 0:
                edge = best_bid - fair
                if edge >= EDGE_TAKE:
                    qty = min(sell_cap, bid_q[sym], POSITION_LIMIT)
                    if qty > 0:
                        result[sym].append(Order(sym, best_bid, -qty))
                        sell_cap -= qty
                        pos -= qty

            # ── 2) PASSIVE QUOTES anchored on fair, inside the spread ─────────
            if spread >= 2:
                # Buy quote: at most floor(fair - HALF), at least best_bid + 1, must stay below ask
                target_bid = int(fair - HALF_QUOTE)
                quote_bid = min(target_bid, best_ask - 1)
                quote_bid = max(quote_bid, best_bid + 1)
                if quote_bid <= best_bid or quote_bid >= best_ask:
                    quote_bid = None

                # Sell quote: at least ceil(fair + HALF), at most best_ask - 1
                target_ask = int(fair + HALF_QUOTE + 0.999)
                quote_ask = max(target_ask, best_bid + 1)
                quote_ask = min(quote_ask, best_ask - 1)
                if quote_ask >= best_ask or quote_ask <= best_bid:
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

        logger.print(f"sum_mid={total_mid:.1f} pos={positions}")
        logger.flush(state, result, conversions, "")
        return result, conversions, ""
