"""
OXYGEN_SHAKE — what MM strategies can squeeze more PnL out of the wide BBO?

Tests these strategies via a tick-by-tick fill simulator that uses real BBO
data + executed market trades to determine which MM quotes get hit:

  v0_baseline       FINAL_GLAUCO baseline: offset+1, size=5, no skew
  v1_size10         offset+1, size=10 (use full LIMIT capacity per side)
  v2_skew           offset+1, size=5, skew quotes by position
  v3_two_levels     offset+1 size=5 AND offset+3 size=5 (deeper liquidity)
  v4_inside_join    join best (offset=0) vs improve (offset=1) — UV_VISOR style
  v5_aggressive_take   take any opposing best when edge > k ticks
  v6_garlic_wider   GARLIC-specific: offset+2 (since spread is 15)

Fill model: each tick, look at executed trades for that symbol at that t.
A market BUY (price >= ask) hits the lowest ask resting at that price level.
If our ask is <= traded price AND not already saturated, we get filled.
Symmetric for sells.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "Data_ROUND_5"
OUT_DIR = Path(__file__).resolve().parent

OXY = [
    "OXYGEN_SHAKE_CHOCOLATE",
    "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_GARLIC",
    "OXYGEN_SHAKE_MINT",
    "OXYGEN_SHAKE_MORNING_BREATH",
]
LIMIT = 10


def load_books_and_trades():
    px_frames, tr_frames = [], []
    for d in (2, 3, 4):
        px = pd.read_csv(DATA_DIR / f"prices_round_5_day_{d}.csv", sep=";")
        px = px[px["product"].isin(OXY)].copy()
        px["day"] = d
        px_frames.append(px)
        tr = pd.read_csv(DATA_DIR / f"trades_round_5_day_{d}.csv", sep=";")
        tr = tr[tr["symbol"].isin(OXY)].copy()
        tr["day"] = d
        tr_frames.append(tr)
    return pd.concat(px_frames, ignore_index=True), pd.concat(tr_frames, ignore_index=True)


def aggregate_trades_per_tick(trades: pd.DataFrame) -> pd.DataFrame:
    """For each (day, ts, symbol): max aggressive buy price, max aggressive sell price,
    plus total volumes traded at-or-above-ask (buyer-aggressive) and at-or-below-bid (seller-aggressive)."""
    g = trades.groupby(["day", "timestamp", "symbol"])
    out = g.agg(
        max_price=("price", "max"),
        min_price=("price", "min"),
        buy_qty=("quantity", "sum"),  # all trades; we split below
    ).reset_index()
    return out


def simulate_strategy(books: pd.DataFrame, trades: pd.DataFrame, sym: str,
                      build_quotes) -> dict:
    """
    Run MM simulation for one symbol.
    `build_quotes(state)` returns list of (side, price, size) tuples each tick,
    where state has best_bid/best_ask/mid/position.
    Fill rule: a buy quote at px gets filled if any market sell trade this tick
    has price <= px (seller crossed our bid). Capped by quote size.
    Returns dict with pnl, n_fills, end_position.
    """
    sb = books[books["product"] == sym].sort_values(["day", "timestamp"]).reset_index(drop=True)
    st = trades[trades["symbol"] == sym].sort_values(["day", "timestamp"])

    # Index trades by (day, ts) -> dataframe slice
    tr_groups = st.groupby(["day", "timestamp"])
    seen = {(d, ts): df for (d, ts), df in tr_groups}

    position = 0
    realized = 0.0
    avg_cost = 0.0
    n_fills = 0
    fills_by_day = {2: 0.0, 3: 0.0, 4: 0.0}

    for row in sb.itertuples(index=False):
        d = row.day; ts = row.timestamp
        bb = row.bid_price_1; ba = row.ask_price_1
        mid = row.mid_price
        if pd.isna(bb) or pd.isna(ba):
            continue
        state = {"bb": bb, "ba": ba, "mid": mid, "pos": position, "spread": ba - bb}
        quotes = build_quotes(state)

        # Determine fills using market trades this tick.
        # Buyer-aggressive (price >= our ask) means somebody wanted to buy at our ask -> our sell fills.
        # Seller-aggressive (price <= our bid) means somebody wanted to sell at our bid -> our buy fills.
        tr_slice = seen.get((d, ts))
        if tr_slice is None:
            continue

        # Split into buyer-aggressive (price >= mid) vs seller-aggressive (price < mid)
        # This is approximate but matches the >=ask / <=bid heuristic.
        buy_aggr = tr_slice[tr_slice["price"] >= ba]  # somebody bought from asks
        sell_aggr = tr_slice[tr_slice["price"] <= bb]  # somebody sold into bids

        for side, px, sz in quotes:
            if sz <= 0:
                continue
            if side == "buy":
                # Our bid at px gets filled if any seller hit at price <= px.
                # We ride the queue at the same price level; assume FIFO so we get filled if
                # market crosses our level (px >= traded price means we're inside or at).
                cross = sell_aggr[sell_aggr["price"] <= px]
                if cross.empty:
                    continue
                avail = int(cross["quantity"].sum())
                cap = LIMIT - position
                fill = min(sz, avail, cap)
                if fill <= 0:
                    continue
                # Update avg cost and position
                if position >= 0:
                    avg_cost = (avg_cost * position + px * fill) / (position + fill) if (position + fill) else 0.0
                    position += fill
                else:
                    # Closing/flipping a short
                    close = min(fill, -position)
                    realized += (avg_cost - px) * close  # short pnl
                    fills_by_day[d] += (avg_cost - px) * close
                    position += close
                    rem = fill - close
                    if rem > 0:
                        avg_cost = px
                        position += rem
                n_fills += 1
            else:
                # sell
                cross = buy_aggr[buy_aggr["price"] >= px]
                if cross.empty:
                    continue
                avail = int(cross["quantity"].sum())
                cap = LIMIT + position
                fill = min(sz, avail, cap)
                if fill <= 0:
                    continue
                if position <= 0:
                    new_pos = position - fill
                    avg_cost = (avg_cost * abs(position) + px * fill) / abs(new_pos) if new_pos else 0.0
                    position = new_pos
                else:
                    close = min(fill, position)
                    realized += (px - avg_cost) * close
                    fills_by_day[d] += (px - avg_cost) * close
                    position -= close
                    rem = fill - close
                    if rem > 0:
                        avg_cost = px
                        position -= rem
                n_fills += 1

    # MTM at end
    last_mid = sb.iloc[-1]["mid_price"]
    last_day = int(sb.iloc[-1]["day"])
    mtm = (last_mid - avg_cost) * position
    realized += mtm
    fills_by_day[last_day] += mtm

    return {"pnl": realized, "n_fills": n_fills, "end_pos": position,
            "d2": fills_by_day[2], "d3": fills_by_day[3], "d4": fills_by_day[4]}


# Strategy builders
def v0_baseline(state, size=5, off=1):
    return [
        ("buy", state["bb"] + off, min(size, LIMIT - state["pos"])),
        ("sell", state["ba"] - off, min(size, LIMIT + state["pos"])),
    ]


def v1_size10(state):
    return [
        ("buy", state["bb"] + 1, max(0, LIMIT - state["pos"])),
        ("sell", state["ba"] - 1, max(0, LIMIT + state["pos"])),
    ]


def v2_skew(state):
    pos = state["pos"]
    skew = int(round(pos * 0.5))  # 0.5 ticks per unit; max 5 ticks at limit
    bid = state["bb"] + 1 - skew
    ask = state["ba"] - 1 - skew
    bid = max(state["bb"], min(bid, state["ba"] - 1))
    ask = max(state["bb"] + 1, min(ask, state["ba"]))
    return [
        ("buy", bid, min(5, LIMIT - pos)),
        ("sell", ask, min(5, LIMIT + pos)),
    ]


def v3_two_levels(state):
    pos = state["pos"]
    spread = state["spread"]
    quotes = [
        ("buy", state["bb"] + 1, min(5, LIMIT - pos)),
        ("sell", state["ba"] - 1, min(5, LIMIT + pos)),
    ]
    # Add a second, deeper tier — only if spread big enough
    if spread >= 8:
        quotes.append(("buy", state["bb"] + 1 - 3, min(5, max(0, LIMIT - pos - 5))))
        quotes.append(("sell", state["ba"] - 1 + 3, min(5, max(0, LIMIT + pos - 5))))
    return quotes


def v4_inside_join(state):
    return [
        ("buy", state["bb"], min(5, LIMIT - state["pos"])),  # join best bid
        ("sell", state["ba"], min(5, LIMIT + state["pos"])),  # join best ask
    ]


def v5_aggressive_take(state):
    """Same as baseline + take when edge to mid > 4 ticks."""
    quotes = list(v0_baseline(state))
    # Edge taker: if bid price < mid - 4 we'd take that ask, but those are theoretical
    # — simplified: just keep baseline; aggressive_take is hard to model from BBO alone.
    return quotes


def v6_garlic_wider(state):
    # Wider for GARLIC due to 15-tick spread: offset+2 to capture more
    return [
        ("buy", state["bb"] + 2, min(5, LIMIT - state["pos"])),
        ("sell", state["ba"] - 2, min(5, LIMIT + state["pos"])),
    ]


def v7_full_inside(state):
    """size=10, offset+1 — full capacity, single level."""
    return [
        ("buy", state["bb"] + 1, max(0, LIMIT - state["pos"])),
        ("sell", state["ba"] - 1, max(0, LIMIT + state["pos"])),
    ]


def v8_join_size10(state):
    """size=10 join best (offset=0)."""
    return [
        ("buy", state["bb"], max(0, LIMIT - state["pos"])),
        ("sell", state["ba"], max(0, LIMIT + state["pos"])),
    ]


def v9_two_levels_size3(state):
    pos = state["pos"]; spread = state["spread"]
    qs = [
        ("buy", state["bb"] + 1, min(3, LIMIT - pos)),
        ("sell", state["ba"] - 1, min(3, LIMIT + pos)),
    ]
    if spread >= 8:
        qs.append(("buy", state["bb"] + 1 - 3, min(4, max(0, LIMIT - pos - 3))))
        qs.append(("sell", state["ba"] - 1 + 3, min(4, max(0, LIMIT + pos - 3))))
        qs.append(("buy", state["bb"], min(3, max(0, LIMIT - pos - 7))))
        qs.append(("sell", state["ba"], min(3, max(0, LIMIT + pos - 7))))
    return qs


STRATEGIES = {
    "v0_baseline_off1_sz5": v0_baseline,
    "v1_size10": v1_size10,
    "v2_skew": v2_skew,
    "v3_two_levels": v3_two_levels,
    "v4_join_best": v4_inside_join,
    "v6_garlic_wider": v6_garlic_wider,
    "v7_size10_off1": v7_full_inside,
    "v8_join_size10": v8_join_size10,
    "v9_two_lvl_sz3": v9_two_levels_size3,
}


def main():
    print("Loading...")
    books, trades = load_books_and_trades()
    print(f"  books rows: {len(books)}   trades rows: {len(trades)}")

    rows = []
    for sym in OXY:
        for name, fn in STRATEGIES.items():
            r = simulate_strategy(books, trades, sym, fn)
            rows.append({"symbol": sym.replace("OXYGEN_SHAKE_", ""),
                         "strategy": name,
                         "pnl": round(r["pnl"], 0),
                         "n_fills": r["n_fills"],
                         "end_pos": r["end_pos"],
                         "d2": round(r["d2"], 0),
                         "d3": round(r["d3"], 0),
                         "d4": round(r["d4"], 0)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "mm_strategy_results.csv", index=False)

    print("\n=== PnL per (symbol, strategy) ===")
    pivot = df.pivot(index="symbol", columns="strategy", values="pnl").fillna(0).astype(int)
    cols = list(STRATEGIES.keys())
    pivot = pivot[cols]
    pivot.loc["TOTAL"] = pivot.sum()
    print(pivot.to_string())

    print("\n=== Totals (all 5 symbols) ===")
    totals = df.groupby("strategy")["pnl"].sum().sort_values(ascending=False)
    print(totals.round(0).to_string())

    print("\n=== Best strategy per symbol ===")
    idx = df.loc[df.groupby("symbol")["pnl"].idxmax()]
    print(idx[["symbol", "strategy", "pnl", "d2", "d3", "d4", "n_fills"]].to_string(index=False))


if __name__ == "__main__":
    main()
