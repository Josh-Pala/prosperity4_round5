"""Conservative market-making backtest with strict ±10 inventory.

Fill model: a posted bid at price B fills only if the NEXT tick has best_ask <= B
(i.e., someone genuinely crossed our quote). Same for the ask side.
Quotes posted: bid = mid - K*half_spread, ask = mid + K*half_spread for K in {0.4, 0.6, 0.8, 1.0}.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

DATA = "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5"
OUT = "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/snackpack"
SYMBOLS = ["SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA","SNACKPACK_PISTACHIO",
           "SNACKPACK_STRAWBERRY","SNACKPACK_RASPBERRY"]
LIMIT = 10

frames = []
for d in (2,3,4):
    df = pd.read_csv(f"{DATA}/prices_round_5_day_{d}.csv", sep=";")
    df = df[df["product"].isin(SYMBOLS)].copy()
    df["t"] = d*1_000_000 + df["timestamp"]
    frames.append(df[["product","t","mid_price","bid_price_1","ask_price_1"]])
raw = pd.concat(frames, ignore_index=True).sort_values(["product","t"]).reset_index(drop=True)

results = []
for s in SYMBOLS:
    sub = raw[raw["product"]==s].sort_values("t").reset_index(drop=True)
    bid = sub["bid_price_1"].values; ask = sub["ask_price_1"].values
    mid_v = sub["mid_price"].values
    spread = ask - bid
    for K in (0.3, 0.5, 0.7, 1.0):
        # Quote: bid = best_bid + 1, ask = best_ask - 1 if K==1.0 (1 inside)
        # else bid = mid - K*half, ask = mid + K*half rounded to int (Prosperity uses int prices)
        if K == 1.0:
            qbid = bid + 1; qask = ask - 1
        else:
            half = (ask - bid)/2.0
            qbid = np.round(mid_v - K*half).astype(int)
            qask = np.round(mid_v + K*half).astype(int)
        # ensure qbid<qask (skip otherwise)
        valid = qbid < qask

        pos = 0
        cash = 0.0
        n_fills = 0
        for i in range(len(sub)-1):
            if not valid[i]: continue
            nxt_bid = bid[i+1]; nxt_ask = ask[i+1]
            # Fill ASK side if next-tick best bid crossed our ask
            if pos > -LIMIT and nxt_bid >= qask[i]:
                cash += qask[i]; pos -= 1; n_fills += 1
            # Fill BID side if next-tick best ask crossed our bid
            if pos < LIMIT and nxt_ask <= qbid[i]:
                cash -= qbid[i]; pos += 1; n_fills += 1
        # mark to market at last mid
        final = cash + pos * mid_v[-1]
        results.append({"symbol":s, "K":K, "fills":n_fills,
                        "end_pos":pos, "final_pnl":round(float(final),2)})

mm = pd.DataFrame(results); print(mm.to_string(index=False))
mm.to_csv(f"{OUT}/p4_mm_strict.csv", index=False)
