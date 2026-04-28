"""Realistic strategy backtest: cost-aware momentum at slower frequency,
plus a passive market-making baseline."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

OUT = "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/snackpack"
DATA = "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5"
SYMBOLS = ["SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA","SNACKPACK_PISTACHIO",
           "SNACKPACK_STRAWBERRY","SNACKPACK_RASPBERRY"]
LIMIT = 10

frames = []
for d in (2,3,4):
    df = pd.read_csv(f"{DATA}/prices_round_5_day_{d}.csv", sep=";")
    df = df[df["product"].isin(SYMBOLS)].copy()
    df["t"] = d*1_000_000 + df["timestamp"]
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    df["half_spread"] = df["spread"]/2.0
    frames.append(df[["product","t","mid_price","bid_price_1","ask_price_1","spread","half_spread"]])
raw = pd.concat(frames, ignore_index=True).sort_values(["product","t"])
mid = raw.pivot(index="t", columns="product", values="mid_price").sort_index()
half = raw.pivot(index="t", columns="product", values="half_spread").sort_index()

def momentum_cost(price, half_sp, N, M):
    p = price.values; hs = half_sp.values
    sig = np.sign(np.concatenate([np.full(N, np.nan), p[N:]-p[:-N]]))
    pos = np.zeros(len(p))
    i = 0
    while i < len(p):
        si = 0 if np.isnan(sig[i]) else sig[i]
        pos[i:i+M] = si*LIMIT
        i += M
    dp = np.diff(p, prepend=p[0])
    pnl = pos*dp
    dpos = np.diff(pos, prepend=0)
    cost = np.abs(dpos)*hs
    net = pnl - cost
    return float(pnl.sum()), float(cost.sum()), float(net.sum()), float(net.mean()/net.std()*np.sqrt(len(net))) if net.std()>0 else 0

# Sweep momentum (longer holds)
print("=== Cost-aware momentum sweep ===")
sweep_rows = []
for s in SYMBOLS:
    p = mid[s].dropna(); hs = half[s].reindex(p.index).ffill().fillna(0)
    best = None
    for N in (5,10,20,50,100,200):
        for M in (5,10,20,50,100,200,500):
            g, c, n, sh = momentum_cost(p, hs, N, M)
            row = {"symbol":s,"N":N,"M":M,"gross":round(g),"cost":round(c),"net":round(n),"sharpe":round(sh,2)}
            if best is None or row["net"] > best["net"]:
                best = row
    sweep_rows.append(best)
sw = pd.DataFrame(sweep_rows); print(sw.to_string(index=False))
sw.to_csv(f"{OUT}/p4_momentum_costaware.csv", index=False)

# Z-score mean reversion with longer window + cost
def zscore_cost(price, half_sp, win, entry, exit_z):
    p = price.values; hs = half_sp.values
    pmean = pd.Series(p).rolling(win).mean().values
    pstd  = pd.Series(p).rolling(win).std().values
    z = (p-pmean)/pstd
    pos = np.zeros(len(p)); cur = 0
    for i in range(len(p)):
        zi = z[i]
        if not np.isnan(zi):
            if cur==0:
                if zi<-entry: cur = LIMIT
                elif zi>entry: cur = -LIMIT
            else:
                if abs(zi)<exit_z: cur = 0
        pos[i] = cur
    dp = np.diff(p, prepend=p[0])
    pnl = pos*dp
    dpos = np.diff(pos, prepend=0)
    cost = np.abs(dpos)*hs
    net = pnl-cost
    return float(pnl.sum()), float(cost.sum()), float(net.sum()), float(net.mean()/net.std()*np.sqrt(len(net))) if net.std()>0 else 0

print("\n=== Cost-aware z-score sweep ===")
zr_rows = []
for s in SYMBOLS:
    p = mid[s].dropna(); hs = half[s].reindex(p.index).ffill().fillna(0)
    best = None
    for win in (50,100,200,500):
        for ent in (1.5, 2.0, 2.5):
            for ex in (0.0, 0.3, 0.5):
                g,c,n,sh = zscore_cost(p, hs, win, ent, ex)
                row = {"symbol":s,"win":win,"entry":ent,"exit":ex,
                       "gross":round(g),"cost":round(c),"net":round(n),"sharpe":round(sh,2)}
                if best is None or row["net"] > best["net"]:
                    best = row
    zr_rows.append(best)
zr = pd.DataFrame(zr_rows); print(zr.to_string(index=False))
zr.to_csv(f"{OUT}/p4_zscore_costaware.csv", index=False)

# Passive market-making baseline:
# At each tick, post a bid 1 tick above best bid AND ask 1 tick below best ask, of size 1.
# Fill probability proxy: count the next-tick move — if mid moves down past our ask, we sold; if it moves up past our bid, we bought.
# This is rough but gives the order of magnitude of MM PnL.
print("\n=== Passive market-making (1-lot, capture spread) ===")
mm_rows = []
for s in SYMBOLS:
    sub = raw[raw["product"]==s].sort_values("t").reset_index(drop=True)
    bid = sub["bid_price_1"].values; ask = sub["ask_price_1"].values
    mid_v = sub["mid_price"].values
    # post 1 tick inside: post_bid = bid+1, post_ask = ask-1
    post_bid = bid + 1
    post_ask = ask - 1
    pos = 0
    cash = 0
    pnl_series = []
    for i in range(len(sub)-1):
        # detect fill in next tick
        nxt_bid = bid[i+1]; nxt_ask = ask[i+1]
        if nxt_bid <= post_ask[i] and pos > -LIMIT:
            # market traded up to our ask — we sell 1
            cash += post_ask[i]; pos -= 1
        if nxt_ask >= post_bid[i] and pos < LIMIT:
            # market traded down to our bid — we buy 1
            cash -= post_bid[i]; pos += 1
        pnl_series.append(cash + pos*mid_v[i+1])
    pnl_series = np.array(pnl_series)
    final = pnl_series[-1] if len(pnl_series)>0 else 0
    mm_rows.append({"symbol":s,"final_pnl":round(float(final)),
                    "max_pos_excursion":int(np.abs(pd.Series(pnl_series).diff()).max() if len(pnl_series)>1 else 0)})
mm = pd.DataFrame(mm_rows); print(mm.to_string(index=False))
mm.to_csv(f"{OUT}/p4_mm.csv", index=False)
