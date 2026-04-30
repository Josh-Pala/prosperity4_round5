"""
mm_pairs_full_v3_2_volsignal — v3_1 + TRANSLATOR volume-signal layer (skew mode).

Adds informed-trader detection on TRANSLATOR_* (eda5/volume_signals/FINDINGS.md):
  bid_vol_1 >= THR AND ask_vol_1 < THR -> drift up over h=200 ticks.

Earlier override-LIMIT behavior dropped total PnL by ~25k vs v3_1 because the
signal cannibalizes existing pair-trading and MM PnL on those legs. This
version uses the signal as a *skew* only:

  1. When signal LONG fires on a TRANSLATOR symbol, force MM bid up by
     +SKEW_TICKS and lift the ask similarly (or skip ask quote entirely);
     bias pair-leg execution to favor LONG fills.
  2. No forced position, no LIMIT exposure.
  3. Hold the skew for VOLSIG_HOLD ticks after last fire.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import jsonpickle
from datamodel import Order, OrderDepth, TradingState


PAIRS: List[Tuple[str, str, str, float]] = [
    ("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE", "spread", 1.2),
    ("TRANSLATOR_SPACE_GRAY", "TRANSLATOR_GRAPHITE_MIST", "spread", 1.2),
    ("MICROCHIP_CIRCLE", "MICROCHIP_TRIANGLE", "sum", 1.5),
    ("PANEL_2X4", "PANEL_4X4", "sum", 2.0),
    ("SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY", "sum", 1.5),
    ("SLEEP_POD_POLYESTER", "SLEEP_POD_COTTON", "spread", 1.8),
    ("SNACKPACK_VANILLA", "SNACKPACK_RASPBERRY", "spread", 1.2),
    ("SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO", "sum", 1.5),
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO", "spread", 1.8),
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_STRAWBERRY", "sum", 1.8),
    ("MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE", "spread", 2.0),
    ("TRANSLATOR_SPACE_GRAY", "TRANSLATOR_VOID_BLUE", "sum", 1.2),
    ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.0),
    ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.5),
    ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
    ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
    ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
    ("ROBOT_MOPPING", "ROBOT_VACUUMING", "sum", 1.2),
]

# Families where passive-entry execution beats aggressive (per backtest above)
PASSIVE_ENTRY_FAMILIES = {"MICROCHIP", "SNACKPACK", "SLEEP_POD", "TRANSLATOR"}

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

# ---- TRANSLATOR volume-signal layer ----
# Per-symbol bid/ask volume thresholds (from FINDINGS.md). Symbols not listed
# are not driven by this layer. Per FINDINGS, only the BIG_BID -> LONG side is
# stable across days; the symmetric short side is disabled (enable_short=False).
VOLSIG_RULES: Dict[str, Dict[str, int]] = {
    "TRANSLATOR_GRAPHITE_MIST": {"thr_bid": 15, "thr_opp": 10, "enable_short": False},
    "TRANSLATOR_ECLIPSE_CHARCOAL": {"thr_bid": 15, "thr_opp": 10, "enable_short": False},
    "TRANSLATOR_VOID_BLUE": {"thr_bid": 15, "thr_opp": 10, "enable_short": False},
    "TRANSLATOR_ASTRO_BLACK": {"thr_bid": 33, "thr_opp": 33, "enable_short": False},
}
VOLSIG_HOLD = 200      # ticks to keep skew active after signal fires
VOLSIG_MM_SKEW = 1     # MM bid/ask skew (in ticks) in signal direction
VOLSIG_LEG_BIAS = 7    # additive bias applied to pair-trading leg target

# ---- PEBBLES block (from v11_final_two_pairs_tuned) ----
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
    # Family is the prefix up to last underscore, but symbols use FAMILY_VARIANT
    # where FAMILY itself may contain '_' (e.g. GALAXY_SOUNDS, OXYGEN_SHAKE).
    # Map by known prefixes.
    for f in (
        "GALAXY_SOUNDS", "OXYGEN_SHAKE", "SLEEP_POD",
        "MICROCHIP", "PANEL", "PEBBLES", "ROBOT",
        "SNACKPACK", "TRANSLATOR", "UV_VISOR",
    ):
        if symbol.startswith(f + "_"):
            return f
    return ""


class Trader:
    def run(self, state: TradingState):
        store = jsonpickle.decode(state.traderData) if state.traderData else {}
        stats = store.setdefault("stats", {})
        targets = store.setdefault("target", {})
        volsig = store.setdefault("volsig", {})  # sym -> {dir: +/-1, ttl: int}

        result: Dict[str, List[Order]] = {}

        # ---- TRANSLATOR volume-signal layer (informed-trader detection) ----
        # Updates volsig[sym] = {dir, ttl}; the MM layer reads dir to skew its
        # quotes in the predicted direction. No hard target / no override of
        # pair-trading leg targets (skew-only mode).
        for sym, rule in VOLSIG_RULES.items():
            depth = state.order_depths.get(sym)
            if depth is None or not depth.buy_orders or not depth.sell_orders:
                continue
            best_bid_px = max(depth.buy_orders.keys())
            best_ask_px = min(depth.sell_orders.keys())
            bid_vol = depth.buy_orders.get(best_bid_px, 0)
            ask_vol = abs(depth.sell_orders.get(best_ask_px, 0))

            state_entry = volsig.get(sym, {"dir": 0, "ttl": 0})
            cur_dir = state_entry.get("dir", 0)
            cur_ttl = state_entry.get("ttl", 0)

            new_dir = cur_dir
            new_ttl = max(0, cur_ttl - 1)

            big_bid = bid_vol >= rule["thr_bid"] and ask_vol < rule["thr_opp"]
            big_ask = (
                rule.get("enable_short", False)
                and ask_vol >= rule.get("thr_ask", rule["thr_bid"])
                and bid_vol < rule["thr_opp"]
            )

            if big_bid:
                new_dir = +1
                new_ttl = VOLSIG_HOLD
            elif big_ask:
                new_dir = -1
                new_ttl = VOLSIG_HOLD
            elif new_ttl == 0:
                new_dir = 0

            volsig[sym] = {"dir": new_dir, "ttl": new_ttl}

        # ---- Pair-trading layer ----
        # Track per-leg execution mode (passive entry if any pair leg from
        # whitelisted family touches that symbol)
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
            # Mark legs as passive-entry eligible based on family
            if family_of(a) in PASSIVE_ENTRY_FAMILIES:
                leg_passive_entry[a] = True
            if family_of(b) in PASSIVE_ENTRY_FAMILIES:
                leg_passive_entry[b] = True

        # Apply volume-signal bias additively on pair-leg targets so we lean
        # into the predicted direction without abandoning the pair-trading
        # mean-reversion logic.
        for sym in VOLSIG_RULES:
            sd = volsig.get(sym, {}).get("dir", 0)
            if sd == 0:
                continue
            leg_targets[sym] = leg_targets.get(sym, 0) + sd * VOLSIG_LEG_BIAS

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

        # ---- MM layer ----
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
            # Signal-driven skew: when a directional volume signal is active
            # on this MM symbol, lift bid+ask by signal_skew. This biases the
            # MM toward the predicted direction without abandoning the spread.
            sig_skew = 0
            sig_dir = volsig.get(sym, {}).get("dir", 0)
            if sig_dir != 0:
                sig_skew = sig_dir * VOLSIG_MM_SKEW
            bid_px = best_bid + 1 - skew + sig_skew
            ask_px = best_ask - 1 - skew + sig_skew
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

        # ---- PEBBLES dedicated block (v11) — passive entry on pair legs ----
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
                            if abs(z) < EXIT_Z:
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
