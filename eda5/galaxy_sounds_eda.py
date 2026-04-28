"""
Galaxy Sounds Recorders — Full EDA
Round 5 / Days 2-4
Symbols: GALAXY_SOUNDS_{BLACK_HOLES, DARK_MATTER, PLANETARY_RINGS, SOLAR_FLAMES, SOLAR_WINDS}
Position limit: +/- 10 per instrument

Outputs go to ./galaxy_sounds_output/
- plots saved as PNG
- numerical results saved as JSON / CSV
- final verdict printed to stdout
"""

from __future__ import annotations

import json
import os
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import (
    acf,
    adfuller,
    coint,
    grangercausalitytests,
    kpss,
    pacf,
)
from statsmodels.tsa.vector_ar.vecm import coint_johansen

warnings.filterwarnings("ignore")

ROOT = Path("/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5")
DATA_DIR = ROOT / "Data_ROUND_5"
OUT_DIR = ROOT / "eda5" / "galaxy_sounds_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "GALAXY_SOUNDS_BLACK_HOLES",
    "GALAXY_SOUNDS_DARK_MATTER",
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "GALAXY_SOUNDS_SOLAR_FLAMES",
    "GALAXY_SOUNDS_SOLAR_WINDS",
]
SHORT = {s: s.replace("GALAXY_SOUNDS_", "") for s in SYMBOLS}
POS_LIMIT = 10


def load_prices() -> pd.DataFrame:
    parts = []
    for d in (2, 3, 4):
        df = pd.read_csv(DATA_DIR / f"prices_round_5_day_{d}.csv", sep=";")
        df = df[df["product"].isin(SYMBOLS)].copy()
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    full["t"] = full["day"] * 1_000_000 + full["timestamp"]
    full = full.sort_values(["product", "t"]).reset_index(drop=True)
    return full


def load_trades() -> pd.DataFrame:
    parts = []
    for d in (2, 3, 4):
        df = pd.read_csv(DATA_DIR / f"trades_round_5_day_{d}.csv", sep=";")
        df = df[df["symbol"].isin(SYMBOLS)].copy()
        df["day"] = d
        df["t"] = d * 1_000_000 + df["timestamp"]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def pivot_mid(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.pivot_table(index="t", columns="product", values="mid_price", aggfunc="last")
    p = p[SYMBOLS].sort_index().ffill()
    return p


def pivot_volume_per_tick(trades: pd.DataFrame, mid_index) -> pd.DataFrame:
    """Aggregate trade quantity per tick per symbol, reindex to mid_price tick grid."""
    g = trades.groupby(["t", "symbol"])["quantity"].sum().unstack(fill_value=0)
    g = g.reindex(columns=SYMBOLS, fill_value=0)
    g = g.reindex(mid_index, fill_value=0)
    return g


def section_header(title: str):
    print("\n" + "=" * 78, flush=True)
    print(title, flush=True)
    print("=" * 78, flush=True)


def step(msg: str):
    print(f"  [step] {msg}", flush=True)


def save_json(name: str, obj):
    with open(OUT_DIR / name, "w") as f:
        json.dump(obj, f, indent=2, default=str)


# --------------------------------------------------------------------------
# PART 1 - DATA HEALTH
# --------------------------------------------------------------------------


def part1(prices: pd.DataFrame, mid: pd.DataFrame, vol: pd.DataFrame):
    section_header("PART 1 - RAW DATA HEALTH CHECK")

    # 1.1 descriptives
    out = {}
    out["shape_prices_filtered"] = list(prices.shape)
    out["dtypes"] = {c: str(t) for c, t in prices.dtypes.items()}
    out["missing_per_col"] = prices.isna().sum().to_dict()
    out["duplicate_keys"] = int(prices.duplicated(["t", "product"]).sum())

    desc = {}
    log_ret = np.log(mid).diff()
    for s in SYMBOLS:
        p = mid[s].dropna()
        r = log_ret[s].dropna()
        desc[SHORT[s]] = {
            "n_ticks": int(len(p)),
            "price_mean": float(p.mean()),
            "price_std": float(p.std()),
            "price_min": float(p.min()),
            "price_max": float(p.max()),
            "price_skew": float(p.skew()),
            "price_kurt": float(p.kurt()),
            "ret_mean": float(r.mean()),
            "ret_std": float(r.std()),
            "ret_skew": float(r.skew()),
            "ret_kurt": float(r.kurt()),
            "zero_price": int((p == 0).sum()),
            "missing_pct": float(mid[s].isna().mean() * 100),
        }
    out["descriptives"] = desc
    save_json("part1_descriptives.json", out)

    # 1.2 plots
    fig, ax = plt.subplots(figsize=(12, 5))
    norm = mid / mid.iloc[0] * 100
    for s in SYMBOLS:
        ax.plot(norm.index, norm[s], label=SHORT[s], lw=0.8)
    ax.set_title("Normalised mid prices (t=0 base 100)")
    ax.set_xlabel("t")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "1_2_prices_normalised.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(5, 1, figsize=(12, 9), sharex=True)
    for i, s in enumerate(SYMBOLS):
        axes[i].plot(log_ret.index, log_ret[s], lw=0.4)
        axes[i].set_ylabel(SHORT[s], fontsize=8)
    axes[-1].set_xlabel("t")
    fig.suptitle("Log returns")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "1_2_logret_stack.png", dpi=110)
    plt.close(fig)

    if vol is not None:
        fig, axes = plt.subplots(5, 1, figsize=(12, 9), sharex=True)
        for i, s in enumerate(SYMBOLS):
            axes[i].bar(vol.index, vol[s], width=80, color="steelblue")
            axes[i].set_ylabel(SHORT[s], fontsize=8)
        axes[-1].set_xlabel("t")
        fig.suptitle("Trade volume per tick")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "1_2_volume_per_tick.png", dpi=110)
        plt.close(fig)

    # 1.3 stationarity (fixed maxlag for speed on long series)
    stationarity = {}
    KP_LAGS = 20
    ADF_MAXLAG = 20
    for s in SYMBOLS:
        p = mid[s].dropna()
        r = log_ret[s].dropna()
        # downsample for KPSS to keep it fast (still representative)
        p_s = p.iloc[:: max(1, len(p) // 5000)]
        r_s = r.iloc[:: max(1, len(r) // 5000)]
        adf_p = adfuller(p, maxlag=ADF_MAXLAG, autolag=None)
        adf_r = adfuller(r, maxlag=ADF_MAXLAG, autolag=None)
        try:
            kpss_p = kpss(p_s, regression="c", nlags=KP_LAGS)
        except Exception:
            kpss_p = (np.nan, np.nan)
        try:
            kpss_r = kpss(r_s, regression="c", nlags=KP_LAGS)
        except Exception:
            kpss_r = (np.nan, np.nan)
        stationarity[SHORT[s]] = {
            "ADF_price_pvalue": float(adf_p[1]),
            "ADF_logret_pvalue": float(adf_r[1]),
            "KPSS_price_pvalue": float(kpss_p[1]) if kpss_p[1] == kpss_p[1] else None,
            "KPSS_logret_pvalue": float(kpss_r[1]) if kpss_r[1] == kpss_r[1] else None,
            "I(1)?": bool(adf_p[1] > 0.05 and adf_r[1] < 0.05),
        }
    save_json("part1_stationarity.json", stationarity)
    return desc, stationarity


# --------------------------------------------------------------------------
# PART 2 - SINGLE INSTRUMENT
# --------------------------------------------------------------------------


def hurst_rs(series: np.ndarray) -> float:
    """R/S Hurst exponent."""
    series = np.asarray(series)
    series = series[~np.isnan(series)]
    if len(series) < 100:
        return float("nan")
    lags = np.unique(np.logspace(1, np.log10(len(series) // 4), 20).astype(int))
    rs = []
    for lag in lags:
        if lag < 8:
            continue
        n_chunks = len(series) // lag
        if n_chunks < 2:
            continue
        chunks = series[: n_chunks * lag].reshape(n_chunks, lag)
        chunk_mean = chunks.mean(axis=1, keepdims=True)
        dev = chunks - chunk_mean
        cum = dev.cumsum(axis=1)
        R = cum.max(axis=1) - cum.min(axis=1)
        S = chunks.std(axis=1, ddof=1)
        valid = S > 1e-12
        if valid.sum() == 0:
            continue
        rs.append((lag, np.mean(R[valid] / S[valid])))
    if len(rs) < 4:
        return float("nan")
    lags_a = np.array([x[0] for x in rs])
    rs_a = np.array([x[1] for x in rs])
    slope, _ = np.polyfit(np.log(lags_a), np.log(rs_a), 1)
    return float(slope)


def part2(mid: pd.DataFrame, vol: pd.DataFrame):
    section_header("PART 2 - SINGLE-INSTRUMENT STRUCTURE")
    log_ret = np.log(mid).diff()
    step("2.1 ACF/PACF + Ljung-Box")

    # 2.1 ACF / PACF + Ljung-Box
    fig, axes = plt.subplots(5, 2, figsize=(12, 11))
    serial = {}
    for i, s in enumerate(SYMBOLS):
        r = log_ret[s].dropna()
        a = acf(r, nlags=40, fft=True)
        p = pacf(r, nlags=40, method="ywm")
        axes[i, 0].bar(range(len(a)), a)
        axes[i, 0].set_title(f"ACF {SHORT[s]}", fontsize=9)
        axes[i, 1].bar(range(len(p)), p)
        axes[i, 1].set_title(f"PACF {SHORT[s]}", fontsize=9)
        r_ds = r.iloc[:: max(1, len(r) // 5000)]
        lb = acorr_ljungbox(r_ds, lags=[5, 10, 20], return_df=True)
        serial[SHORT[s]] = {
            "acf_lag1": float(a[1]),
            "acf_lag2": float(a[2]),
            "LB_lag5_p": float(lb["lb_pvalue"].iloc[0]),
            "LB_lag10_p": float(lb["lb_pvalue"].iloc[1]),
            "LB_lag20_p": float(lb["lb_pvalue"].iloc[2]),
        }
    fig.tight_layout()
    fig.savefig(OUT_DIR / "2_1_acf_pacf.png", dpi=110)
    plt.close(fig)
    save_json("part2_serial.json", serial)

    step("2.2 trend + Hurst")
    # 2.2 trend + Hurst
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    hurst = {}
    for i, s in enumerate(SYMBOLS):
        p = mid[s].dropna()
        axes[i].plot(p.index, p, lw=0.6, label="price")
        for w in (20, 50, 100):
            axes[i].plot(p.index, p.rolling(w).mean(), lw=0.6, label=f"MA{w}")
        axes[i].set_title(SHORT[s], fontsize=9)
        axes[i].legend(fontsize=7)
        hurst[SHORT[s]] = hurst_rs(p.values)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "2_2_trend.png", dpi=110)
    plt.close(fig)
    save_json("part2_hurst.json", hurst)

    step("2.3 volatility + GARCH")
    # 2.3 vol structure
    vol_struct = {}
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    for i, s in enumerate(SYMBOLS):
        r = log_ret[s].dropna()
        rv = r.rolling(20).std()
        axes[i].plot(rv.index, rv, lw=0.5)
        axes[i].set_title(f"{SHORT[s]} 20-tick rolling std of log-returns", fontsize=9)
        # ARCH (LB on r^2) - downsample
        r2_ds = (r ** 2).iloc[:: max(1, len(r) // 5000)]
        lb2 = acorr_ljungbox(r2_ds, lags=[10], return_df=True)
        vol_struct[SHORT[s]] = {
            "LB_sq_lag10_p": float(lb2["lb_pvalue"].iloc[0]),
        }
        # GARCH(1,1) — downsample for speed
        try:
            from arch import arch_model

            r_g = r.iloc[:: max(1, len(r) // 5000)]
            am = arch_model(r_g * 100, vol="GARCH", p=1, q=1, mean="Zero", rescale=False)
            res = am.fit(disp="off", show_warning=False, options={"maxiter": 50})
            params = res.params
            vol_struct[SHORT[s]]["garch_alpha"] = float(params.get("alpha[1]", np.nan))
            vol_struct[SHORT[s]]["garch_beta"] = float(params.get("beta[1]", np.nan))
            vol_struct[SHORT[s]]["garch_persistence"] = float(
                params.get("alpha[1]", 0) + params.get("beta[1]", 0)
            )
        except Exception as e:
            vol_struct[SHORT[s]]["garch_error"] = str(e)[:60]
    fig.tight_layout()
    fig.savefig(OUT_DIR / "2_3_volatility.png", dpi=110)
    plt.close(fig)
    save_json("part2_vol.json", vol_struct)

    step("2.4 distribution")
    # 2.4 distribution
    distro = {}
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for i, s in enumerate(SYMBOLS):
        r = log_ret[s].dropna()
        axes[0, i].hist(r, bins=80, density=True, color="steelblue", alpha=0.7)
        x = np.linspace(r.min(), r.max(), 200)
        axes[0, i].plot(x, stats.norm.pdf(x, r.mean(), r.std()), "r-", lw=1)
        axes[0, i].set_title(SHORT[s], fontsize=9)
        stats.probplot(r, dist="norm", plot=axes[1, i])
        axes[1, i].set_title("")
        jb = stats.jarque_bera(r)
        distro[SHORT[s]] = {
            "skew": float(r.skew()),
            "exc_kurt": float(r.kurt()),
            "JB_pvalue": float(jb.pvalue),
        }
    fig.tight_layout()
    fig.savefig(OUT_DIR / "2_4_distribution.png", dpi=110)
    plt.close(fig)
    save_json("part2_distribution.json", distro)

    step("2.5 volume analysis")
    # 2.5 volume analysis
    vol_an = {}
    if vol is not None:
        for s in SYMBOLS:
            v = vol[s]
            r = log_ret[s]
            df = pd.DataFrame({"v": v, "r": r}).dropna()
            df = df[df["v"] > 0]  # only ticks with trades
            if len(df) < 20:
                vol_an[SHORT[s]] = {"note": "insufficient trade data"}
                continue
            v_acf = acf(df["v"].values, nlags=20, fft=True)
            corr_abs = float(df["r"].abs().corr(df["v"]))
            # logistic: sign(r_{t+1}) ~ log(v_t)
            df2 = pd.DataFrame({"v": v, "r": r})
            df2["next_r"] = df2["r"].shift(-1)
            df2 = df2.dropna()
            df2 = df2[df2["v"] > 0]
            if len(df2) > 30:
                from sklearn.linear_model import LogisticRegression

                X = np.log(df2["v"].values).reshape(-1, 1)
                y = (df2["next_r"] > 0).astype(int).values
                if len(np.unique(y)) > 1:
                    lr = LogisticRegression(max_iter=200).fit(X, y)
                    coef = float(lr.coef_[0, 0])
                    acc = float(lr.score(X, y))
                else:
                    coef, acc = float("nan"), float("nan")
            else:
                coef, acc = float("nan"), float("nan")
            amihud = float((df["r"].abs() / df["v"]).mean()) if (df["v"] > 0).any() else float("nan")
            vol_an[SHORT[s]] = {
                "vol_acf_lag1": float(v_acf[1]),
                "corr_absret_vol": corr_abs,
                "logit_coef_log_v": coef,
                "logit_train_acc": acc,
                "amihud": amihud,
                "n_trade_ticks": int(len(df)),
            }
    save_json("part2_volume.json", vol_an)
    return serial, hurst, vol_struct, distro, vol_an


# --------------------------------------------------------------------------
# PART 3 - CROSS-INSTRUMENT
# --------------------------------------------------------------------------


def part3(mid: pd.DataFrame, vol: pd.DataFrame):
    section_header("PART 3 - CROSS-INSTRUMENT STRUCTURE")
    log_ret = np.log(mid).diff().dropna()
    step("3.1 correlation")
    # 3.1 corr
    pearson = log_ret.corr(method="pearson")
    spearman = log_ret.corr(method="spearman")
    pearson.to_csv(OUT_DIR / "3_1_pearson.csv")
    spearman.to_csv(OUT_DIR / "3_1_spearman.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pearson.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([SHORT[s] for s in SYMBOLS], rotation=45)
    ax.set_yticklabels([SHORT[s] for s in SYMBOLS])
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{pearson.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title("Pearson correlation of log-returns")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "3_1_pearson_heatmap.png", dpi=110)
    plt.close(fig)

    rolling_corr_stats = {}
    fig, ax = plt.subplots(figsize=(12, 6))
    for a, b in combinations(SYMBOLS, 2):
        rc = log_ret[a].rolling(50).corr(log_ret[b])
        ax.plot(rc.index, rc, lw=0.4, label=f"{SHORT[a][:3]}-{SHORT[b][:3]}")
        rolling_corr_stats[f"{SHORT[a]}__{SHORT[b]}"] = {
            "mean_abs": float(rc.abs().mean()),
            "std": float(rc.std()),
            "mean": float(rc.mean()),
        }
    ax.legend(fontsize=6, ncol=5)
    ax.set_title("Rolling-50 pairwise correlation")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "3_1_rolling_corr.png", dpi=110)
    plt.close(fig)
    save_json("part3_correlation.json", {
        "pearson": pearson.round(3).to_dict(),
        "rolling_stats": rolling_corr_stats,
    })

    step("3.2 cointegration (downsampled)")
    # 3.2 cointegration
    coint_results = {}
    log_p = np.log(mid).dropna()
    # downsample for EG coint test (statsmodels is O(n^2) for some auto-lag paths)
    ds_step = max(1, len(log_p) // 5000)
    log_p_ds = log_p.iloc[::ds_step]
    for a, b in combinations(SYMBOLS, 2):
        try:
            t_stat, p_val, _ = coint(log_p_ds[a], log_p_ds[b], maxlag=10, autolag=None)
        except Exception:
            t_stat, p_val = np.nan, np.nan
        # OLS hedge ratio
        beta = float(np.cov(log_p[a], log_p[b])[0, 1] / np.var(log_p[b]))
        spread = log_p[a] - beta * log_p[b]
        try:
            adf_sp = adfuller(spread.dropna(), maxlag=20, autolag=None)
            adf_sp_p = float(adf_sp[1])
        except Exception:
            adf_sp_p = float("nan")
        # half life: spread_t = a + rho*spread_{t-1}; HL = -ln(2)/ln(rho)
        sp = spread.dropna()
        sp_lag = sp.shift(1).dropna()
        sp_now = sp.loc[sp_lag.index]
        rho = np.cov(sp_now, sp_lag)[0, 1] / np.var(sp_lag)
        hl = float(-np.log(2) / np.log(rho)) if 0 < rho < 1 else float("nan")
        coint_results[f"{SHORT[a]}__{SHORT[b]}"] = {
            "EG_pvalue": float(p_val),
            "hedge_ratio": beta,
            "spread_ADF_p": adf_sp_p,
            "halflife": hl,
            "spread_std": float(sp.std()),
        }

    # Johansen
    try:
        joh = coint_johansen(log_p.values, det_order=0, k_ar_diff=1)
        joh_out = {
            "trace_stat": joh.lr1.tolist(),
            "crit_95": joh.cvt[:, 1].tolist(),
            "rank_at_95": int((joh.lr1 > joh.cvt[:, 1]).sum()),
        }
    except Exception as e:
        joh_out = {"error": str(e)[:100]}
    save_json("part3_coint.json", {"pairs": coint_results, "johansen": joh_out})

    # plot best pair spread
    best = min(coint_results.items(), key=lambda kv: kv[1]["EG_pvalue"] if kv[1]["EG_pvalue"] == kv[1]["EG_pvalue"] else 1.0)
    pair_name = best[0]
    a_name, b_name = pair_name.split("__")
    a_sym = "GALAXY_SOUNDS_" + a_name
    b_sym = "GALAXY_SOUNDS_" + b_name
    beta = best[1]["hedge_ratio"]
    spread = log_p[a_sym] - beta * log_p[b_sym]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(spread.index, spread, lw=0.6)
    ax.axhline(spread.mean(), color="black", lw=0.6)
    ax.axhline(spread.mean() + spread.std(), color="orange", lw=0.5, ls="--")
    ax.axhline(spread.mean() - spread.std(), color="orange", lw=0.5, ls="--")
    ax.axhline(spread.mean() + 2 * spread.std(), color="red", lw=0.5, ls="--")
    ax.axhline(spread.mean() - 2 * spread.std(), color="red", lw=0.5, ls="--")
    ax.set_title(f"Best cointegrated spread: {a_name} vs {b_name} (EG p={best[1]['EG_pvalue']:.3g})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "3_2_best_spread.png", dpi=110)
    plt.close(fig)

    step("3.3 lead-lag + Granger")
    # 3.3 lead-lag
    leadlag = {}
    granger = {}
    fig, axes = plt.subplots(5, 2, figsize=(12, 11))
    pair_idx = 0
    for a, b in combinations(SYMBOLS, 2):
        ra, rb = log_ret[a].values, log_ret[b].values
        n = min(len(ra), len(rb))
        ra, rb = ra[-n:], rb[-n:]
        # cross-correlation
        lags = np.arange(-20, 21)
        ccf = []
        for L in lags:
            if L < 0:
                ccf.append(float(np.corrcoef(ra[-L:], rb[: n + L])[0, 1]))
            elif L > 0:
                ccf.append(float(np.corrcoef(ra[: n - L], rb[L:])[0, 1]))
            else:
                ccf.append(float(np.corrcoef(ra, rb)[0, 1]))
        leadlag[f"{SHORT[a]}__{SHORT[b]}"] = {
            "max_abs_lag": int(lags[int(np.argmax(np.abs(ccf)))]),
            "max_abs_corr": float(max(np.abs(ccf))),
        }
        if pair_idx < 5:
            axes[pair_idx, 0].bar(lags, ccf)
            axes[pair_idx, 0].set_title(f"CCF {SHORT[a][:3]}->{SHORT[b][:3]}", fontsize=8)
        # Granger (small max lag for speed; downsample to keep <5k)
        try:
            df_g = pd.DataFrame({"a": ra, "b": rb}).dropna()
            stp = max(1, len(df_g) // 5000)
            df_g = df_g.iloc[::stp]
            res = grangercausalitytests(df_g[["a", "b"]], maxlag=5, verbose=False)
            p_a_to_b = min(res[k][0]["ssr_chi2test"][1] for k in res)
            res2 = grangercausalitytests(df_g[["b", "a"]], maxlag=5, verbose=False)
            p_b_to_a = min(res2[k][0]["ssr_chi2test"][1] for k in res2)
            granger[f"{SHORT[a]}__{SHORT[b]}"] = {
                "p_A_causes_B": float(p_a_to_b),
                "p_B_causes_A": float(p_b_to_a),
            }
        except Exception as e:
            granger[f"{SHORT[a]}__{SHORT[b]}"] = {"error": str(e)[:60]}
        pair_idx += 1
    fig.tight_layout()
    fig.savefig(OUT_DIR / "3_3_ccf.png", dpi=110)
    plt.close(fig)
    save_json("part3_leadlag.json", {"ccf": leadlag, "granger": granger})

    step("3.4 order flow")
    # 3.4 order flow impact (using trades pivot)
    flow = {}
    if vol is not None:
        for s in SYMBOLS:
            v = vol[s].values.astype(float)
            r = log_ret[s].reindex(vol.index).fillna(0).values
            mask = v > 0
            if mask.sum() < 30:
                flow[SHORT[s]] = {"note": "insufficient signed-flow proxy"}
                continue
            # signed volume proxy: sign of contemporaneous return * volume
            signed = np.sign(r) * v
            cum_ret = r.cumsum()
            slope, intercept, rval, pval, _ = stats.linregress(signed[mask], cum_ret[mask])
            # impact decay after volume spike (top 5%)
            thr = np.quantile(v[mask], 0.95)
            spike_idx = np.where(v >= thr)[0]
            spike_idx = spike_idx[spike_idx + 30 < len(v)]
            if len(spike_idx) > 0:
                paths = np.zeros(31)
                for idx in spike_idx:
                    paths += r[idx : idx + 31].cumsum() * np.sign(r[idx])
                paths /= len(spike_idx)
                decay = paths.tolist()
            else:
                decay = []
            flow[SHORT[s]] = {
                "kyle_lambda": float(slope),
                "kyle_r2": float(rval ** 2),
                "kyle_p": float(pval),
                "n_spikes": int(len(spike_idx)),
                "post_spike_signed_cumret_30": decay[-1] if decay else float("nan"),
            }
    save_json("part3_flow.json", flow)
    return coint_results, leadlag, granger, flow, pair_name


# --------------------------------------------------------------------------
# PART 4 - BACKTESTS
# --------------------------------------------------------------------------


def sharpe(pnl: pd.Series) -> float:
    pnl = pnl.dropna()
    if pnl.std() == 0 or len(pnl) < 10:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(len(pnl)))


def stats_from_trades(trade_pnls: list[float], equity: pd.Series) -> dict:
    if len(trade_pnls) == 0:
        return {"sharpe": 0.0, "win_rate": 0.0, "mean_pnl": 0.0, "max_dd": 0.0, "n_trades": 0}
    eq = equity.fillna(method="ffill").fillna(0)
    dd = (eq - eq.cummax()).min()
    return {
        "sharpe": sharpe(equity.diff().dropna()),
        "win_rate": float(np.mean([1 if p > 0 else 0 for p in trade_pnls])),
        "mean_pnl": float(np.mean(trade_pnls)),
        "max_dd": float(dd),
        "n_trades": int(len(trade_pnls)),
    }


def backtest_zscore(price: pd.Series, window: int = 50, entry: float = 1.5, exit_z: float = 0.3, size: int = POS_LIMIT):
    """Mean reversion: buy when z<-entry, sell when z>+entry. Capped to ±size."""
    z = (price - price.rolling(window).mean()) / price.rolling(window).std()
    pos = 0
    positions = []
    trade_pnls = []
    entry_px = None
    entry_pos = 0
    for t, (px, zt) in enumerate(zip(price.values, z.values)):
        if np.isnan(zt):
            positions.append(pos)
            continue
        if pos == 0:
            if zt < -entry:
                pos = size
                entry_px = px
                entry_pos = pos
            elif zt > entry:
                pos = -size
                entry_px = px
                entry_pos = pos
        else:
            if abs(zt) < exit_z:
                trade_pnls.append((px - entry_px) * entry_pos)
                pos = 0
                entry_px = None
        positions.append(pos)
    positions = pd.Series(positions, index=price.index)
    pnl = positions.shift(1) * price.diff()
    return trade_pnls, pnl.cumsum(), positions


def backtest_momentum(price: pd.Series, N: int, M: int, size: int = POS_LIMIT):
    sig = np.sign(price.diff(N))
    pos = sig.fillna(0) * size
    # cap holding to M ticks: refresh every M ticks
    refresh = (np.arange(len(price)) // M)
    pos_arr = pos.values.copy()
    for i in range(1, len(pos_arr)):
        if refresh[i] == refresh[i - 1]:
            pos_arr[i] = pos_arr[i - 1]
    pos_s = pd.Series(pos_arr, index=price.index)
    pnl = pos_s.shift(1) * price.diff()
    # naive "trades": every direction change
    changes = pos_s.diff().fillna(0)
    trade_idx = changes[changes != 0].index
    trade_pnls = []
    last_px, last_pos = None, 0
    for idx in trade_idx:
        px = price.loc[idx]
        if last_px is not None and last_pos != 0:
            trade_pnls.append((px - last_px) * last_pos)
        last_px = px
        last_pos = pos_s.loc[idx]
    return trade_pnls, pnl.cumsum(), pos_s


def backtest_volume_momentum(price: pd.Series, vol: pd.Series, size: int = POS_LIMIT, hold: int = 10):
    r = price.diff()
    vmean = vol.rolling(20).mean()
    trigger = (vol > 2 * vmean) & (vmean > 0)
    pos = pd.Series(0, index=price.index, dtype=float)
    cooldown = 0
    direction = 0
    held = 0
    for i in range(len(price)):
        if held > 0:
            pos.iloc[i] = direction * size
            held -= 1
            continue
        if trigger.iloc[i] and r.iloc[i] != 0 and not np.isnan(r.iloc[i]):
            direction = np.sign(r.iloc[i])
            held = hold - 1
            pos.iloc[i] = direction * size
    pnl = pos.shift(1) * price.diff()
    # trades
    changes = pos.diff().fillna(0)
    trade_pnls = []
    last_px, last_pos = None, 0
    for idx in changes[changes != 0].index:
        px = price.loc[idx]
        if last_pos != 0 and last_px is not None:
            trade_pnls.append((px - last_px) * last_pos)
        last_px = px
        last_pos = pos.loc[idx]
    return trade_pnls, pnl.cumsum(), pos


def backtest_leadlag(leader: pd.Series, follower: pd.Series, size: int = POS_LIMIT, hold: int = 3):
    r_lead = leader.diff()
    sig = np.sign(r_lead.shift(1)).fillna(0)
    pos = pd.Series(0, index=follower.index, dtype=float)
    held = 0
    direction = 0
    for i in range(len(follower)):
        if held > 0:
            pos.iloc[i] = direction * size
            held -= 1
            continue
        if sig.iloc[i] != 0:
            direction = sig.iloc[i]
            held = hold - 1
            pos.iloc[i] = direction * size
    pnl = pos.shift(1) * follower.diff()
    changes = pos.diff().fillna(0)
    trade_pnls = []
    last_px, last_pos = None, 0
    for idx in changes[changes != 0].index:
        px = follower.loc[idx]
        if last_pos != 0 and last_px is not None:
            trade_pnls.append((px - last_px) * last_pos)
        last_px = px
        last_pos = pos.loc[idx]
    return trade_pnls, pnl.cumsum(), pos


def backtest_pairs(price_a: pd.Series, price_b: pd.Series, beta: float, size: int = POS_LIMIT, entry: float = 1.5, exit_z: float = 0.3):
    spread = np.log(price_a) - beta * np.log(price_b)
    mu = spread.rolling(200).mean()
    sd = spread.rolling(200).std()
    z = (spread - mu) / sd
    pos = 0
    pos_a = []
    pos_b = []
    trade_pnls = []
    entry_px_a, entry_px_b, entry_dir = None, None, 0
    for t in range(len(spread)):
        zt = z.iloc[t]
        if np.isnan(zt):
            pos_a.append(0); pos_b.append(0); continue
        if pos == 0:
            if zt < -entry:
                pos = 1
                entry_px_a = price_a.iloc[t]; entry_px_b = price_b.iloc[t]
                entry_dir = 1
            elif zt > entry:
                pos = -1
                entry_px_a = price_a.iloc[t]; entry_px_b = price_b.iloc[t]
                entry_dir = -1
        else:
            if abs(zt) < exit_z:
                p = (price_a.iloc[t] - entry_px_a) * entry_dir * size + (price_b.iloc[t] - entry_px_b) * (-entry_dir) * size * beta
                trade_pnls.append(float(p))
                pos = 0
        pos_a.append(pos * size)
        pos_b.append(-pos * size * beta)
    pos_a = pd.Series(pos_a, index=price_a.index)
    pos_b = pd.Series(pos_b, index=price_b.index)
    pnl = pos_a.shift(1) * price_a.diff() + pos_b.shift(1) * price_b.diff()
    return trade_pnls, pnl.cumsum(), pos_a


def part4(mid: pd.DataFrame, vol: pd.DataFrame, coint_results: dict, granger: dict):
    section_header("PART 4 - STRATEGY BACKTESTS")
    results = []

    # 4.1 z-score mean reversion per symbol
    for s in SYMBOLS:
        trades, eq, _ = backtest_zscore(mid[s])
        st = stats_from_trades(trades, eq)
        results.append({"symbol": SHORT[s], "strategy": "zscore_w50_e1.5", **st})

    # 4.2 momentum sweep
    for s in SYMBOLS:
        best = None
        for N in (5, 10, 20):
            for M in (5, 10, 20):
                trades, eq, _ = backtest_momentum(mid[s], N, M)
                st = stats_from_trades(trades, eq)
                if best is None or st["sharpe"] > best["sharpe"]:
                    best = {**st, "N": N, "M": M}
        results.append({
            "symbol": SHORT[s],
            "strategy": f"mom_N{best['N']}_M{best['M']}",
            **{k: v for k, v in best.items() if k not in ("N", "M")},
        })

    # 4.3 volume-triggered momentum
    if vol is not None:
        for s in SYMBOLS:
            trades, eq, _ = backtest_volume_momentum(mid[s], vol[s])
            st = stats_from_trades(trades, eq)
            results.append({"symbol": SHORT[s], "strategy": "vol_mom_2x_h10", **st})

    # 4.4 lead-lag: pick strongest Granger directional pair
    best_g = None
    for k, v in granger.items():
        if "p_A_causes_B" not in v:
            continue
        p_min = min(v["p_A_causes_B"], v["p_B_causes_A"])
        if best_g is None or p_min < best_g[1]:
            best_g = (k, p_min, v)
    if best_g is not None:
        a, b = best_g[0].split("__")
        if best_g[2]["p_A_causes_B"] < best_g[2]["p_B_causes_A"]:
            leader, follower = "GALAXY_SOUNDS_" + a, "GALAXY_SOUNDS_" + b
        else:
            leader, follower = "GALAXY_SOUNDS_" + b, "GALAXY_SOUNDS_" + a
        trades, eq, _ = backtest_leadlag(mid[leader], mid[follower])
        st = stats_from_trades(trades, eq)
        results.append({
            "symbol": f"{SHORT[follower]}<-{SHORT[leader]}",
            "strategy": "leadlag_h3",
            **st,
        })

    # 4.5 pairs trade on best cointegrated
    best_c = min(coint_results.items(), key=lambda kv: kv[1]["EG_pvalue"])
    a, b = best_c[0].split("__")
    a_sym, b_sym = "GALAXY_SOUNDS_" + a, "GALAXY_SOUNDS_" + b
    beta = best_c[1]["hedge_ratio"]
    trades, eq, _ = backtest_pairs(mid[a_sym], mid[b_sym], beta)
    st = stats_from_trades(trades, eq)
    results.append({"symbol": f"{a}/{b}", "strategy": f"pairs_beta{beta:.2f}", **st})

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "part4_backtests.csv", index=False)
    return df


# --------------------------------------------------------------------------
# PART 5 - VERDICT
# --------------------------------------------------------------------------


def verdict(s: dict) -> str:
    if s["sharpe"] > 2.0 and s["mean_pnl"] > 0 and s["n_trades"] >= 5:
        return "STRONG EDGE"
    if s["sharpe"] > 1.0 and s["mean_pnl"] > 0:
        return "WEAK EDGE"
    if s["sharpe"] < -0.5 or s["mean_pnl"] < 0:
        return "AVOID"
    return "NO EDGE"


def part5(bt: pd.DataFrame):
    section_header("PART 5 - STRATEGY VERDICT")
    bt["verdict"] = bt.apply(lambda r: verdict(r.to_dict()), axis=1)
    cols = ["symbol", "strategy", "sharpe", "win_rate", "mean_pnl", "max_dd", "n_trades", "verdict"]
    bt = bt[cols]
    bt.to_csv(OUT_DIR / "part5_verdict.csv", index=False)
    print(bt.to_string(index=False))
    return bt


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------


def main():
    print("Loading data...")
    prices = load_prices()
    trades = load_trades()
    mid = pivot_mid(prices)
    vol = pivot_volume_per_tick(trades, mid.index)
    print(f"  prices rows={len(prices)}, ticks per symbol={len(mid)}, trade events={len(trades)}")

    desc, station = part1(prices, mid, vol)
    print("[main] Part 1 done", flush=True)
    serial, hurst, vol_struct, distro, vol_an = part2(mid, vol)
    print("[main] Part 2 done", flush=True)
    coint_results, leadlag, granger, flow, best_pair = part3(mid, vol)
    print("[main] Part 3 done", flush=True)
    bt = part4(mid, vol, coint_results, granger)
    print("[main] Part 4 done", flush=True)
    final = part5(bt)
    print("\nDone. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
