"""
FINAL_GLAUCO — final submission for IMC Prosperity 4 Round 5.

OXYGEN_SHAKE changes (this session):
  - MM offset 1 -> 0 (join best instead of improve+1) on all 5 OXY symbols
    (+7.55k, 787.931 -> 795.482).
  - MINT cross-family fair-value taker: fair = α + β·PANEL basket. Take when
    |bid/ask - fair| > 200 ticks. PANEL-only basket avoids the overfitting of
    larger baskets (OOS resid sd 230-440 vs 430-660 for top8). +55.99k
    (795.482 -> 851.472). MINT goes from +0.95k -> +56.94k across 3 days.
  - Pair-trading rejected (mid-price simulator over-estimates with wide BBO;
    see eda5/oxygen_shake/VERDICT.md and MM_VERDICT.md).

SLEEP_POD change (prior session):
  - POLY-COTTON pair: entry_z 1.8 -> 1.0 (+12.0k, more frequent re-entries).
  - Added POLY-SUEDE spread @ z=1.8 (unlocks SUEDE leg from -6.3k to +2.9k).
  - Combined gain: +12.65k (backtester: 775.280 -> 787.931).
  - Validated via eda5/sleep_pod/pair_scanner_v2.py + combo_sim.py.

PANEL change (prior session):
  - PANEL_1X2 removed from MM_UNIVERSE.
  - Rationale: deep EDA (eda5/panel/) showed PANEL mid is a random walk,
    and PANEL_1X2 has the widest spread (~12 ticks) but strongest intraday
    drift; it bled ~9k/day in v1's MM layer. Removing it gave +18.7k total.

UV_VISOR changes (prior session):
  - Added pair (MAGENTA, RED, sum, 1.0) — v13 winner.
  - Fair-value edge taker on UV_VISOR_AMBER — v14.
  - Size-skew on UV_VISOR_* MM quotes — v17.
  - Join-best execution on UV_VISOR_* (offset=0 instead of +1/-1) — v18.

Asset classes UNCHANGED:
  PEBBLES (v11), ROBOT (H25), TRANSLATOR (T42), MICROCHIP, SNACKPACK,
  GALAXY_SOUNDS.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import jsonpickle
from datamodel import Order, OrderDepth, TradingState


PAIRS: List[Tuple[str, str, str, float]] = [
    # MICROCHIP
    ("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE", "spread", 1.2),
    ("MICROCHIP_CIRCLE", "MICROCHIP_TRIANGLE", "sum", 1.5),
    ("MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE", "spread", 2.0),
    # PANEL
    ("PANEL_2X4", "PANEL_4X4", "sum", 2.0),
    # SLEEP_POD — pair scanner gain (+12.6k vs prior 1-pair config)
    # POLY-COTTON: lowered z 1.8 -> 1.0 (single change worth +12.0k)
    # POLY-SUEDE: new pair, unlocks SUEDE leg (was -6.3k, becomes +2.9k)
    ("SLEEP_POD_POLYESTER", "SLEEP_POD_COTTON", "spread", 1.0),
    ("SLEEP_POD_POLYESTER", "SLEEP_POD_SUEDE", "spread", 1.8),
    # SNACKPACK
    ("SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY", "sum", 1.5),
    ("SNACKPACK_VANILLA", "SNACKPACK_RASPBERRY", "spread", 1.2),
    ("SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO", "sum", 1.5),
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO", "spread", 1.8),
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_STRAWBERRY", "sum", 1.8),
    # TRANSLATOR (T42 optimized: 5 pairs, was 2)
    ("TRANSLATOR_SPACE_GRAY", "TRANSLATOR_GRAPHITE_MIST", "spread", 1.2),
    ("TRANSLATOR_SPACE_GRAY", "TRANSLATOR_VOID_BLUE", "sum", 1.2),
    ("TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_VOID_BLUE", "sum", 1.2),
    ("TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_VOID_BLUE", "spread", 1.2),
    ("TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_SPACE_GRAY", "sum", 1.2),
    # ROBOT (H25)
    ("ROBOT_LAUNDRY", "ROBOT_VACUUMING", "spread", 1.0),
    ("ROBOT_DISHES", "ROBOT_VACUUMING", "sum", 1.5),
    ("ROBOT_DISHES", "ROBOT_LAUNDRY", "sum", 1.5),
    ("ROBOT_IRONING", "ROBOT_MOPPING", "sum", 1.5),
    ("ROBOT_LAUNDRY", "ROBOT_MOPPING", "sum", 2.0),
    ("ROBOT_MOPPING", "ROBOT_VACUUMING", "sum", 1.2),
    # UV_VISOR — v13 winner
    ("UV_VISOR_MAGENTA", "UV_VISOR_RED", "sum", 1.0),
]

# Families where passive-entry execution beats aggressive
# TRANSLATOR removed: T42 sweep showed aggressive beats passive (+11k) for the 5-pair set
PASSIVE_ENTRY_FAMILIES = {"MICROCHIP", "SNACKPACK", "SLEEP_POD"}

# Per-family EXIT_Z override (default 0.3)
EXIT_Z_BY_FAMILY: Dict[str, float] = {
    "TRANSLATOR": 0.2,  # T42 sweep
}

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
    # PANEL_1X2 removed: bleeds ~9k/day in MM (drift > MM edge, see eda5/panel/)
    "UV_VISOR_AMBER",
    "SLEEP_POD_SUEDE",
    "SLEEP_POD_LAMB_WOOL",
    # TRANSLATOR_ECLIPSE_CHARCOAL removed: now used in pairs
    "SLEEP_POD_NYLON",
    "PANEL_2X2",
    "PANEL_1X4",
    "TRANSLATOR_ASTRO_BLACK",  # still here, but is also in pair — pair layer wins on entry
]

LIMIT = 10
WARMUP = 500
EXIT_Z_DEFAULT = 0.3
SIZE = 10

MM_QUOTE_SIZE = 5
MM_SKEW_PER_UNIT = 0.0
MM_MIN_SPREAD = 3

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

# v14: AMBER fair-value model
UV_AMBER_INTERCEPT = 31353.3
UV_AMBER_BETAS = {
    "UV_VISOR_MAGENTA": -0.970,
    "UV_VISOR_ORANGE": -0.385,
    "UV_VISOR_RED": -0.617,
    "UV_VISOR_YELLOW": -0.167,
}
UV_AMBER_TAKE_THRESHOLD = 400

# OXYGEN_SHAKE_MINT fair-value model (eda5/oxygen_shake/mint_deep_dive.py).
# PANEL-only basket: full sample R²=0.644, in-sample resid_sd=303.
# Holdout sd 230-440 (more stable than top8's 430-660).
# Threshold 200 was the backtest peak (sweep 100-400, see MINT_VERDICT.md).
MINT_INTERCEPT = 7551.93
MINT_BETAS = {
    "PANEL_2X2": 0.3942,
    "PANEL_1X4": 0.1190,
    "PANEL_4X4": -0.2639,
}
MINT_TAKE_THRESHOLD = 200

# OXYGEN_SHAKE_MORNING_BREATH fair-value model (eda5/oxygen_shake/morning_breath/).
# Basket: pebbles.  Threshold: 400.
MB_INTERCEPT = 14866.29
MB_BETAS = {
    "PEBBLES_M": -0.5864,
    "PEBBLES_XS": +0.1556,
}
MB_TAKE_THRESHOLD = 400


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
    """Pair EXIT_Z = max of per-family overrides for the two legs.
    Conservative: if either leg's family has a higher (looser) exit_z, use it."""
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

        # ---- Pair-trading layer ----
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

        # Pair legs reserve symbols: skip them in MM
        engaged_pair_legs = set(leg_targets.keys())

        # ---- MM layer ----
        for sym in MM_UNIVERSE:
            if sym in engaged_pair_legs:
                # Symbol is already a pair leg this tick; pair owns the position
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
            # v18: UV_VISOR join-best (offset 0)
            # OXY: same trick — BBO spread is 12-15 ticks and trades only fire
            # at BBO, so joining captures the full spread (+7.55k validated).
            depth_off = 0 if (sym.startswith("UV_VISOR_") or sym.startswith("OXYGEN_SHAKE_")) else 1
            bid_px = best_bid + depth_off - skew
            ask_px = best_ask - depth_off - skew
            if bid_px >= best_ask:
                bid_px = best_ask - 1
            if ask_px <= best_bid:
                ask_px = best_bid + 1

            buy_capacity = max(0, LIMIT - position)
            sell_capacity = max(0, LIMIT + position)
            buy_qty = min(MM_QUOTE_SIZE, buy_capacity)
            sell_qty = min(MM_QUOTE_SIZE, sell_capacity)
            # v17: UV_VISOR size-skew
            if sym.startswith("UV_VISOR_"):
                if position > 3:
                    buy_qty = max(0, buy_qty - position + 3)
                elif position < -3:
                    sell_qty = max(0, sell_qty + position + 3)

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

        # ---- v14: UV_VISOR_AMBER fair-value edge taker ----
        amber_dep = state.order_depths.get("UV_VISOR_AMBER")
        if amber_dep is not None and "UV_VISOR_AMBER" not in engaged_pair_legs:
            other_mids = {}
            for s in UV_AMBER_BETAS:
                d = state.order_depths.get(s)
                if d is not None:
                    m = mid_of(d)
                    if m is not None:
                        other_mids[s] = m
            if len(other_mids) == len(UV_AMBER_BETAS):
                fair = UV_AMBER_INTERCEPT + sum(
                    UV_AMBER_BETAS[s] * other_mids[s] for s in UV_AMBER_BETAS
                )
                bb, ba = best_levels(amber_dep)
                if bb is not None and ba is not None:
                    pos_a = state.position.get("UV_VISOR_AMBER", 0)
                    existing = result.get("UV_VISOR_AMBER", [])
                    extra: List[Order] = []
                    if fair - ba > UV_AMBER_TAKE_THRESHOLD:
                        cap = LIMIT - pos_a
                        for o in existing:
                            if o.quantity > 0:
                                cap -= o.quantity
                        qty = min(5, max(0, cap))
                        if qty > 0:
                            extra.append(Order("UV_VISOR_AMBER", ba, qty))
                    elif bb - fair > UV_AMBER_TAKE_THRESHOLD:
                        cap = LIMIT + pos_a
                        for o in existing:
                            if o.quantity < 0:
                                cap -= -o.quantity
                        qty = min(5, max(0, cap))
                        if qty > 0:
                            extra.append(Order("UV_VISOR_AMBER", bb, -qty))
                    if extra:
                        result["UV_VISOR_AMBER"] = existing + extra

        # ---- OXYGEN_SHAKE_MINT cross-family fair-value edge taker ----
        # Lifts MINT from +0.95k -> +56.94k in 3-day backtest. Same template as
        # AMBER: take when bid/ask deviates from PANEL-implied fair by > 200.
        mint_dep = state.order_depths.get("OXYGEN_SHAKE_MINT")
        if mint_dep is not None and "OXYGEN_SHAKE_MINT" not in engaged_pair_legs:
            other_mids = {}
            for s in MINT_BETAS:
                d = state.order_depths.get(s)
                if d is not None:
                    m = mid_of(d)
                    if m is not None:
                        other_mids[s] = m
            if len(other_mids) == len(MINT_BETAS):
                fair = MINT_INTERCEPT + sum(
                    MINT_BETAS[s] * other_mids[s] for s in MINT_BETAS
                )
                bb, ba = best_levels(mint_dep)
                if bb is not None and ba is not None:
                    pos_m = state.position.get("OXYGEN_SHAKE_MINT", 0)
                    existing = result.get("OXYGEN_SHAKE_MINT", [])
                    extra: List[Order] = []
                    if fair - ba > MINT_TAKE_THRESHOLD:
                        cap = LIMIT - pos_m
                        for o in existing:
                            if o.quantity > 0:
                                cap -= o.quantity
                        qty = min(5, max(0, cap))
                        if qty > 0:
                            extra.append(Order("OXYGEN_SHAKE_MINT", ba, qty))
                    elif bb - fair > MINT_TAKE_THRESHOLD:
                        cap = LIMIT + pos_m
                        for o in existing:
                            if o.quantity < 0:
                                cap -= -o.quantity
                        qty = min(5, max(0, cap))
                        if qty > 0:
                            extra.append(Order("OXYGEN_SHAKE_MINT", bb, -qty))
                    if extra:
                        result["OXYGEN_SHAKE_MINT"] = existing + extra


        # ---- OXYGEN_SHAKE_MORNING_BREATH cross-family fair-value edge taker ----
        # Same template as MINT/AMBER. Independent of MINT taker.
        mb_dep = state.order_depths.get("OXYGEN_SHAKE_MORNING_BREATH")
        if mb_dep is not None and "OXYGEN_SHAKE_MORNING_BREATH" not in engaged_pair_legs:
            mb_other = {}
            for s in MB_BETAS:
                d = state.order_depths.get(s)
                if d is not None:
                    m = mid_of(d)
                    if m is not None:
                        mb_other[s] = m
            if len(mb_other) == len(MB_BETAS):
                fair_mb = MB_INTERCEPT + sum(
                    MB_BETAS[s] * mb_other[s] for s in MB_BETAS
                )
                bb_mb, ba_mb = best_levels(mb_dep)
                if bb_mb is not None and ba_mb is not None:
                    pos_mb = state.position.get("OXYGEN_SHAKE_MORNING_BREATH", 0)
                    existing_mb = result.get("OXYGEN_SHAKE_MORNING_BREATH", [])
                    extra_mb: List[Order] = []
                    if fair_mb - ba_mb > MB_TAKE_THRESHOLD:
                        cap = LIMIT - pos_mb
                        for o in existing_mb:
                            if o.quantity > 0:
                                cap -= o.quantity
                        qty = min(5, max(0, cap))
                        if qty > 0:
                            extra_mb.append(Order("OXYGEN_SHAKE_MORNING_BREATH", ba_mb, qty))
                    elif bb_mb - fair_mb > MB_TAKE_THRESHOLD:
                        cap = LIMIT + pos_mb
                        for o in existing_mb:
                            if o.quantity < 0:
                                cap -= -o.quantity
                        qty = min(5, max(0, cap))
                        if qty > 0:
                            extra_mb.append(Order("OXYGEN_SHAKE_MORNING_BREATH", bb_mb, -qty))
                    if extra_mb:
                        result["OXYGEN_SHAKE_MORNING_BREATH"] = existing_mb + extra_mb

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
