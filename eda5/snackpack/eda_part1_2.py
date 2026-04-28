"""SNACKPACK EDA — Parts 1 & 2: Health check + single-instrument structure."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
import os, json

DATA_DIR = "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/Data_ROUND_5"
OUT_DIR = "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5/eda5/snackpack"
SYMBOLS = ["SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO",
           "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY"]

def load():
    frames = []
    for d in (2, 3, 4):
        df = pd.read_csv(f"{DATA_DIR}/prices_round_5_day_{d}.csv", sep=";")
        df = df[df["product"].isin(SYMBOLS)].copy()
        df["day"] = d
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["t"] = df["day"] * 1_000_000 + df["timestamp"]
    df = df.sort_values(["product", "t"]).reset_index(drop=True)
    return df

def pivot_mid(df):
    return df.pivot(index="t", columns="product", values="mid_price").sort_index()

def vol_per_tick(df):
    bidv = df[["bid_volume_1","bid_volume_2","bid_volume_3"]].fillna(0).sum(axis=1)
    askv = df[["ask_volume_1","ask_volume_2","ask_volume_3"]].fillna(0).sum(axis=1)
    df["total_vol"] = bidv + askv
    return df.pivot(index="t", columns="product", values="total_vol").sort_index()

def main():
    out = {}
    df = load()
    print("Loaded:", df.shape)
    print("Per-symbol counts:\n", df.groupby("product").size())
    print("Missing mid_price per symbol:\n", df.groupby("product")["mid_price"].apply(lambda s: s.isna().sum()))

    # duplicate timestamps
    dup = df.duplicated(["product","t"]).sum()
    print(f"Duplicate (product,t) rows: {dup}")

    mid = pivot_mid(df)
    vol = vol_per_tick(df)
    logp = np.log(mid)
    ret  = logp.diff().dropna(how="all")

    # -- 1.1 descriptives
    desc = []
    for s in SYMBOLS:
        p = mid[s].dropna(); r = ret[s].dropna()
        desc.append({
            "symbol": s,
            "n": int(len(p)),
            "missing_pct": round(100*(1 - len(p)/len(mid)), 4),
            "p_mean": round(p.mean(),3), "p_std": round(p.std(),3),
            "p_min": round(p.min(),3), "p_max": round(p.max(),3),
            "r_mean": float(f"{r.mean():.3e}"), "r_std": float(f"{r.std():.3e}"),
            "r_skew": round(stats.skew(r),3), "r_kurt": round(stats.kurtosis(r),3),
            "p_skew": round(stats.skew(p),3), "p_kurt": round(stats.kurtosis(p),3),
        })
    desc_df = pd.DataFrame(desc)
    print("\n=== 1.1 Descriptives ===")
    print(desc_df.to_string(index=False))
    desc_df.to_csv(f"{OUT_DIR}/p1_descriptives.csv", index=False)

    # -- 1.2 plots
    norm = mid.divide(mid.iloc[0]).mul(100)
    plt.figure(figsize=(12,5))
    for s in SYMBOLS: plt.plot(norm.index, norm[s], label=s, lw=0.7)
    plt.legend(fontsize=7); plt.title("Mid prices normalised to 100"); plt.xlabel("t")
    plt.savefig(f"{OUT_DIR}/p1_prices_norm.png", dpi=110, bbox_inches="tight"); plt.close()

    fig, axes = plt.subplots(5,1, figsize=(12,10), sharex=True)
    for ax,s in zip(axes,SYMBOLS):
        ax.plot(ret.index, ret[s], lw=0.3); ax.set_ylabel(s, fontsize=7)
    axes[-1].set_xlabel("t"); plt.suptitle("Log-returns")
    plt.savefig(f"{OUT_DIR}/p1_returns.png", dpi=110, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(12,5))
    for s in SYMBOLS: plt.plot(vol.index, vol[s].rolling(50).mean(), label=s, lw=0.6)
    plt.legend(fontsize=7); plt.title("Rolling mean (50) total order-book volume per tick")
    plt.savefig(f"{OUT_DIR}/p1_volume.png", dpi=110, bbox_inches="tight"); plt.close()

    # -- 1.3 stationarity (sample 5000 to keep ADF/KPSS fast)
    print("\n=== 1.3 Stationarity ===")
    stat = []
    for s in SYMBOLS:
        p = mid[s].dropna()
        r = ret[s].dropna()
        # subsample uniformly for ADF (it's O(n))
        ps = p.iloc[::max(1,len(p)//5000)]
        rs = r.iloc[::max(1,len(r)//5000)]
        adf_p = adfuller(ps, autolag="AIC")[1]
        adf_r = adfuller(rs, autolag="AIC")[1]
        try: kpss_p = kpss(ps, regression="c", nlags="auto")[1]
        except: kpss_p = np.nan
        try: kpss_r = kpss(rs, regression="c", nlags="auto")[1]
        except: kpss_r = np.nan
        stat.append({"symbol":s, "ADF_price_p":round(adf_p,4), "KPSS_price_p":round(kpss_p,4),
                     "ADF_ret_p":round(adf_r,4), "KPSS_ret_p":round(kpss_r,4),
                     "I(1)?": "yes" if (adf_p>0.05 and adf_r<0.05) else "uncertain"})
    stat_df = pd.DataFrame(stat)
    print(stat_df.to_string(index=False))
    stat_df.to_csv(f"{OUT_DIR}/p1_stationarity.csv", index=False)

    # ===== PART 2 =====
    # 2.1 ACF/PACF + Ljung-Box
    print("\n=== 2.1 ACF / Ljung-Box on returns ===")
    fig, axes = plt.subplots(5,2, figsize=(12,12))
    lb_rows = []
    for i,s in enumerate(SYMBOLS):
        r = ret[s].dropna().values
        a = acf(r, nlags=40, fft=True)
        p = pacf(r, nlags=40, method="ywm")
        axes[i,0].bar(range(41), a); axes[i,0].set_title(f"{s} ACF", fontsize=8); axes[i,0].axhline(0,color="k",lw=0.3)
        axes[i,1].bar(range(41), p); axes[i,1].set_title(f"{s} PACF", fontsize=8); axes[i,1].axhline(0,color="k",lw=0.3)
        lb = acorr_ljungbox(r, lags=[5,10,20], return_df=True)
        lb_rows.append({"symbol":s, "acf1":round(a[1],4), "acf2":round(a[2],4), "acf5":round(a[5],4),
                        "LB_p_5":round(lb["lb_pvalue"].iloc[0],4),
                        "LB_p_10":round(lb["lb_pvalue"].iloc[1],4),
                        "LB_p_20":round(lb["lb_pvalue"].iloc[2],4)})
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/p2_acf.png", dpi=110); plt.close()
    lb_df = pd.DataFrame(lb_rows); print(lb_df.to_string(index=False))
    lb_df.to_csv(f"{OUT_DIR}/p2_ljungbox.csv", index=False)

    # 2.2 Trend + Hurst
    def hurst(ts):
        ts = np.asarray(ts); n=len(ts)
        lags = np.unique(np.round(np.logspace(0.7, np.log10(n//4), 20)).astype(int))
        tau = []
        for lag in lags:
            diff = ts[lag:] - ts[:-lag]
            tau.append(np.std(diff))
        tau = np.array(tau); ok = tau>0
        slope, _ = np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)
        return slope

    print("\n=== 2.2 Trend / Hurst ===")
    hurst_rows = []
    fig, axes = plt.subplots(5,1, figsize=(12,10), sharex=True)
    for ax,s in zip(axes,SYMBOLS):
        p = mid[s].dropna()
        ax.plot(p.index, p, lw=0.4, label="price")
        for w,c in [(20,"C1"),(50,"C2"),(100,"C3")]:
            ax.plot(p.index, p.rolling(w).mean(), lw=0.6, label=f"MA{w}", color=c)
        ax.set_title(s, fontsize=8); ax.legend(fontsize=6, loc="upper right")
        h = hurst(p.values)
        hurst_rows.append({"symbol":s, "hurst":round(h,4),
                           "regime":"mean-revert" if h<0.45 else ("trend" if h>0.55 else "random")})
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/p2_trend.png", dpi=110); plt.close()
    h_df = pd.DataFrame(hurst_rows); print(h_df.to_string(index=False))
    h_df.to_csv(f"{OUT_DIR}/p2_hurst.csv", index=False)

    # 2.3 Volatility / GARCH
    print("\n=== 2.3 Volatility / GARCH ===")
    garch_rows = []
    fig, axes = plt.subplots(5,1, figsize=(12,10), sharex=True)
    for ax,s in zip(axes,SYMBOLS):
        r = ret[s].dropna()
        rv = r.rolling(20).std()
        ax.plot(rv.index, rv, lw=0.4); ax.set_title(f"{s} rolling 20 sd", fontsize=8)
        # Ljung-Box on squared returns
        lb_sq = acorr_ljungbox(r.values**2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
        # GARCH(1,1) on returns scaled by 1e4 to avoid convergence issues
        try:
            am = arch_model(r.values*1e4, vol="GARCH", p=1, q=1, mean="constant", dist="normal")
            res = am.fit(disp="off")
            a = res.params.get("alpha[1]", np.nan); b = res.params.get("beta[1]", np.nan)
        except Exception as e:
            a, b = np.nan, np.nan
        garch_rows.append({"symbol":s, "LB_sq_p10":round(lb_sq,4),
                           "alpha":round(a,4), "beta":round(b,4),
                           "alpha+beta":round(a+b,4) if not np.isnan(a+b) else np.nan})
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/p2_volatility.png", dpi=110); plt.close()
    g_df = pd.DataFrame(garch_rows); print(g_df.to_string(index=False))
    g_df.to_csv(f"{OUT_DIR}/p2_garch.csv", index=False)

    # 2.4 Distribution
    print("\n=== 2.4 Return distribution ===")
    fig, axes = plt.subplots(5,2, figsize=(12,14))
    dist_rows = []
    for i,s in enumerate(SYMBOLS):
        r = ret[s].dropna().values
        mu, sd = r.mean(), r.std()
        axes[i,0].hist(r, bins=80, density=True, alpha=0.6)
        x = np.linspace(r.min(), r.max(), 300)
        axes[i,0].plot(x, stats.norm.pdf(x, mu, sd), 'r', lw=1)
        axes[i,0].set_title(f"{s} hist", fontsize=8)
        stats.probplot(r, dist="norm", plot=axes[i,1]); axes[i,1].set_title(f"{s} Q-Q", fontsize=8)
        jb_stat, jb_p = stats.jarque_bera(r)
        dist_rows.append({"symbol":s, "skew":round(stats.skew(r),3),
                          "excess_kurt":round(stats.kurtosis(r),3),
                          "JB_p":round(jb_p,6)})
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/p2_distribution.png", dpi=110); plt.close()
    d_df = pd.DataFrame(dist_rows); print(d_df.to_string(index=False))
    d_df.to_csv(f"{OUT_DIR}/p2_distribution.csv", index=False)

    # 2.5 Volume analysis
    print("\n=== 2.5 Volume analysis ===")
    vrows = []
    for s in SYMBOLS:
        v = vol[s].dropna()
        r = ret[s].dropna()
        common = v.index.intersection(r.index)
        v_c, r_c = v.loc[common], r.loc[common]
        # ACF of volume
        v_acf1 = acf(v_c.values, nlags=5, fft=True)[1]
        # corr(|return|, volume)
        corr_abs = np.corrcoef(np.abs(r_c.values), v_c.values)[0,1]
        # logistic: sign(r_{t+1}) ~ log(volume_t)
        from sklearn.linear_model import LogisticRegression
        try:
            sign = (r_c.shift(-1).dropna() > 0).astype(int).values
            X = np.log(v_c.iloc[:-1].clip(lower=1).values).reshape(-1,1)
            mask = ~np.isnan(X.ravel())
            lr = LogisticRegression().fit(X[mask], sign[mask])
            beta = lr.coef_[0,0]
            acc = lr.score(X[mask], sign[mask])
        except Exception:
            beta, acc = np.nan, np.nan
        # Amihud
        amihud = (np.abs(r_c) / v_c.clip(lower=1)).mean()
        vrows.append({"symbol":s, "vol_acf1":round(v_acf1,4),
                      "corr_|r|_v":round(corr_abs,4),
                      "logit_beta":round(beta,4), "logit_acc":round(acc,4),
                      "amihud":float(f"{amihud:.3e}")})
    v_df = pd.DataFrame(vrows); print(v_df.to_string(index=False))
    v_df.to_csv(f"{OUT_DIR}/p2_volume.csv", index=False)

    # save mid/ret/vol for later parts
    mid.to_parquet(f"{OUT_DIR}/_mid.parquet")
    ret.to_parquet(f"{OUT_DIR}/_ret.parquet")
    vol.to_parquet(f"{OUT_DIR}/_vol.parquet")
    print("\nSaved intermediate parquets.")

if __name__ == "__main__":
    main()
