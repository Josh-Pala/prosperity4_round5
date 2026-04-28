"""Add realistic execution-cost sensitivity to the winning strategies."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

OUT = "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/snackpack"
DATA = "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5"
SYMBOLS = ["SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA","SNACKPACK_PISTACHIO",
           "SNACKPACK_STRAWBERRY","SNACKPACK_RASPBERRY"]
LIMIT = 10

# load full books for spreads
frames = []
for d in (2,3,4):
    df = pd.read_csv(f"{DATA}/prices_round_5_day_{d}.csv", sep=";")
    df = df[df["product"].isin(SYMBOLS)].copy()
    df["t"] = d*1_000_000 + df["timestamp"]
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    df["half_spread"] = df["spread"]/2.0
    frames.append(df[["product","t","mid_price","bid_price_1","ask_price_1","spread","half_spread"]])
raw = pd.concat(frames, ignore_index=True).sort_values(["product","t"])

print("Mean bid-ask spread per symbol (in price units):")
sp = raw.groupby("product")["spread"].agg(["mean","median","std"]).round(3)
print(sp)
sp.to_csv(f"{OUT}/p4_spread_summary.csv")

mid = raw.pivot(index="t", columns="product", values="mid_price").sort_index()
half = raw.pivot(index="t", columns="product", values="half_spread").sort_index()

def momentum_pnl(price, half_sp, N=5, M=5):
    p = price.values
    hs = half_sp.values
    sig = np.sign(np.concatenate([np.full(N, np.nan), p[N:]-p[:-N]]))
    pos = np.zeros(len(p))
    i = 0
    while i < len(p):
        si = 0 if np.isnan(sig[i]) else sig[i]
        pos[i:i+M] = si*LIMIT
        i += M
    dp = np.diff(p, prepend=p[0])
    pnl = pos*dp
    # cost: every position change |dpos| pays half-spread per contract
    dpos = np.diff(pos, prepend=0)
    cost = np.abs(dpos) * hs
    pnl_net = pnl - cost
    return pnl, pnl_net, cost

print("\n=== Momentum N=M=5 with bid-ask cost ===")
for s in SYMBOLS:
    p = mid[s].dropna()
    hs = half[s].reindex(p.index).ffill().fillna(0)
    g, n, c = momentum_pnl(p, hs)
    print(f"  {s:<22} gross={g.sum():>9.0f}  cost={c.sum():>9.0f}  NET={n.sum():>9.0f}  net_sharpe={n.mean()/n.std()*np.sqrt(len(n)):.2f}")

# Pair PISTACHIO|STRAWBERRY with bid-ask cost
print("\n=== Pair PISTACHIO|STRAWBERRY with bid-ask cost ===")
import statsmodels.api as sm
a, b = "SNACKPACK_PISTACHIO","SNACKPACK_STRAWBERRY"
pa = np.log(mid[a].dropna()); pb = np.log(mid[b].dropna())
common = pa.index.intersection(pb.index)
pa, pb = pa.loc[common], pb.loc[common]
p_a = mid[a].loc[common]; p_b = mid[b].loc[common]
hs_a = half[a].reindex(common).ffill().fillna(0)
hs_b = half[b].reindex(common).ffill().fillna(0)
mask_tr = pd.Series((common // 1_000_000) == 2, index=common)
Xb = sm.add_constant(pb[mask_tr]); ols = sm.OLS(pa[mask_tr], Xb).fit()
beta = ols.params.iloc[1]
spr = pa - beta*pb
mu = spr.rolling(200).mean(); sd = spr.rolling(200).std()
z = (spr-mu)/sd
pos_a = np.zeros(len(common)); pos_b = np.zeros(len(common))
cur = 0
for i in range(len(common)):
    zi = z.iloc[i]
    if not np.isnan(zi):
        if cur==0:
            if zi<-1.5: cur=1
            elif zi>1.5: cur=-1
        else:
            if abs(zi)<0.3: cur=0
    pos_a[i]=cur*LIMIT; pos_b[i]=-cur*LIMIT
da = np.diff(p_a.values, prepend=p_a.values[0])
db = np.diff(p_b.values, prepend=p_b.values[0])
gross = pos_a*da + pos_b*db
dpos_a = np.diff(pos_a, prepend=0); dpos_b = np.diff(pos_b, prepend=0)
cost = np.abs(dpos_a)*hs_a.values + np.abs(dpos_b)*hs_b.values
net = gross - cost
print(f"  gross={gross.sum():>9.0f}  cost={cost.sum():>9.0f}  NET={net.sum():>9.0f}  net_sharpe={net.mean()/net.std()*np.sqrt(len(net)):.2f}")
