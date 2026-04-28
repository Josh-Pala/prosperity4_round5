# SNACKPACK EDA — Quantitative Strategy Analysis

**Category**: Protein Snack Packs
**Symbols**: SNACKPACK_CHOCOLATE, SNACKPACK_VANILLA, SNACKPACK_PISTACHIO, SNACKPACK_STRAWBERRY, SNACKPACK_RASPBERRY
**Position limit**: ±10 per instrument
**Data**: prices_round_5_day_{2,3,4}.csv — 30,000 ticks/symbol (10,000 per day, timestamps every 100)

Each section gives the *interpretation guide* first (what to look for and why), then the empirical results.

---

## PART 1 — RAW DATA HEALTH CHECK

### 1.1 Basic descriptives

**Interpretation guide.** We look at `mean`, `std`, `min/max`, `skew`, `kurtosis` of price and log-return. The `mean` of returns should be near zero (efficient prices); a non-zero mean implies drift. `std` of returns measures unconditional volatility — sets the realistic profit-per-trade scale (you can't earn more than ~σ per round-trip without leverage). `skew` and `kurtosis` of returns indicate fat tails; `kurtosis>3` warns of large rare moves and the need for stops. We also flag `>1% missing` ticks (synchronization issues) or zero prices (data corruption).

**Results (see [p1_descriptives.csv](p1_descriptives.csv)):**

| Symbol | n | miss% | mean P | std P | min P | max P | r std | r skew | r kurt |
|--------|---|-------|--------|-------|-------|-------|-------|--------|--------|
| CHOCOLATE | 30000 | 0 | 9843.4 | 200.7 | 9268 | 10316 | 6.69e-4 | 0.011 | 0.071 |
| VANILLA | 30000 | 0 | 10097.3 | 178.5 | 9676 | 10563 | 6.45e-4 | -0.002 | 0.056 |
| PISTACHIO | 30000 | 0 | 9495.8 | 187.5 | 9013 | 10107 | 5.52e-4 | -0.002 | -0.028 |
| STRAWBERRY | 30000 | 0 | 10706.6 | 363.6 | 9554 | 11391 | 7.61e-4 | -0.007 | -0.019 |
| RASPBERRY | 30000 | 0 | 10077.8 | 169.8 | 9683 | 10647 | 8.03e-4 | 0.000 | -0.009 |

- **No missing data, no duplicates, no zero prices.** Clean.
- All symbols trade ~10,000 in price units. STRAWBERRY has the widest range (~9550-11400, std 364) — most trending behavior.
- Per-tick log-return std ≈ 5.5–8.0 bp ⇒ typical ±10-position MTM swing per tick = `10 contracts × ~7000 × 6e-4 ≈ ±42 PnL/tick`.
- Returns are symmetric (skew ≈ 0) and **near-Gaussian** (excess kurtosis < 0.1) — unusual for finance, suggests synthetic noise.

### 1.2 Visual inspection

**Interpretation guide.** A normalized price plot (all start at 100) reveals which symbols are trending vs range-bound and whether they share a common factor. Stacked log-return panels show whether volatility is constant or clustered. Volume bars expose activity regimes that signal information arrival.

**Plots**: [p1_prices_norm.png](p1_prices_norm.png), [p1_returns.png](p1_returns.png), [p1_volume.png](p1_volume.png).
- STRAWBERRY drifts the most. Other four oscillate ~±5%. A clear visual **anti-correlation between CHOC and VAN** is apparent (when one is up, the other is down).
- Returns appear homoskedastic — no obvious vol clusters.
- Order-book volume is low and stable; no spikes.

### 1.3 Stationarity

**Interpretation guide.** Trading strategies need to know whether to model the **level** (price) or the **first difference** (return). ADF tests *H0: unit root (non-stationary)*; KPSS tests *H0: stationary*. For an I(1) process, we expect: ADF on price fails to reject (p>0.05), KPSS on price rejects (p<0.05); ADF on returns rejects, KPSS on returns fails to reject. I(1) means trade differences/spreads, not levels.

**Results (see [p1_stationarity.csv](p1_stationarity.csv)):**

| Symbol | ADF price p | KPSS price p | ADF return p | KPSS ret p | I(1)? |
|--------|-------------|--------------|--------------|------------|-------|
| CHOCOLATE | 0.067 | 0.01 | 0.000 | 0.10 | yes |
| VANILLA | 0.037 | 0.01 | 0.000 | 0.10 | borderline |
| PISTACHIO | 0.070 | 0.01 | 0.000 | 0.10 | yes |
| STRAWBERRY | 0.151 | 0.01 | 0.000 | 0.10 | yes |
| RASPBERRY | 0.003 | 0.10 | 0.000 | 0.10 | possibly stationary |

Conclusion: prices are essentially **I(1) random walks**, returns are stationary. RASPBERRY shows mild evidence of level-stationarity (KPSS doesn't reject), suggesting partial mean-reversion in price — relevant later.

---

## PART 2 — SINGLE-INSTRUMENT STRUCTURE

### 2.1 Autocorrelation of returns

**Interpretation guide.** A negative ACF(1) in returns indicates short-term mean reversion (price overshoots and corrects); positive ACF(1) indicates short-term momentum. Ljung-Box rejects the null of "no autocorrelation up to lag k". If LB rejects strongly at small lags, there's exploitable structure.

**Results (see [p2_ljungbox.csv](p2_ljungbox.csv), [p2_acf.png](p2_acf.png)):**

| Symbol | ACF(1) | ACF(2) | LB p (lag 5) | LB p (lag 10) |
|--------|--------|--------|--------------|---------------|
| CHOCOLATE | -0.031 | 0.002 | <0.001 | <0.001 |
| VANILLA | -0.027 | -0.004 | <0.001 | 0.002 |
| PISTACHIO | -0.025 | 0.006 | <0.001 | <0.001 |
| STRAWBERRY | -0.014 | 0.008 | 0.149 | 0.054 |
| RASPBERRY | -0.017 | 0.011 | 0.029 | 0.013 |

All ACF(1) are **slightly negative** — weak mean reversion at lag 1. Statistically significant in CHOC/VAN/PIST/RASP. STRAWBERRY shows no autocorrelation. The effect size (|ACF| ≈ 0.02-0.03) is too small to overcome trading costs at lag-1 frequency, but it is the signature of **noisy market microstructure** (bid-ask bounce).

### 2.2 Trend detection (Hurst)

**Interpretation guide.** The Hurst exponent measures the long-range memory of a time series. `H<0.5` = mean-reverting, `H≈0.5` = random walk, `H>0.5` = persistent/trending. Combined with rolling means, this tells us whether to bet on continuation or reversion at longer horizons.

**Results (see [p2_hurst.csv](p2_hurst.csv), [p2_trend.png](p2_trend.png)):**

| Symbol | Hurst | Regime |
|--------|-------|--------|
| CHOCOLATE | 0.41 | mean-revert |
| VANILLA | 0.42 | mean-revert |
| PISTACHIO | 0.40 | mean-revert |
| STRAWBERRY | 0.41 | mean-revert |
| RASPBERRY | 0.39 | mean-revert |

All H are **between 0.39 and 0.42** — clearly < 0.5. The category is **mean-reverting at multiple horizons**. This is consistent with prices oscillating around a slowly-moving fundamental level rather than trending.

### 2.3 Volatility structure (GARCH)

**Interpretation guide.** Ljung-Box on squared returns tests for ARCH effects (volatility clustering). GARCH(1,1) decomposes vol into a "shock" term `α` (sensitivity to recent surprises) and a "persistence" term `β` (memory of past variance). `α+β` close to 1 ⇒ high vol persistence ⇒ scale positions inversely to recent volatility.

**Results (see [p2_garch.csv](p2_garch.csv)):**

| Symbol | LB sq p (lag 10) | α | β | α+β |
|--------|------------------|---|---|-----|
| CHOCOLATE | 0.99 | 0.002 | 0.991 | 0.993 |
| VANILLA | 0.90 | 0.001 | 0.971 | 0.973 |
| PISTACHIO | 0.10 | 0.014 | 0.008 | 0.022 |
| STRAWBERRY | 0.006 | 0.016 | 0.658 | 0.674 |
| RASPBERRY | 0.069 | 0.013 | 0.649 | 0.662 |

CHOC/VAN have suspiciously high `β` with LB-sq p ≈ 1 (no actual clustering — GARCH overfit a constant). PIST has near-zero α+β (essentially homoskedastic). STRAWBERRY/RASPBERRY have moderate persistence (~0.67). **Practical takeaway**: vol clustering is weak; dynamic position sizing yields little benefit here.

### 2.4 Return distribution

**Interpretation guide.** Skew + kurtosis + Jarque-Bera tell us whether returns are Gaussian. Fat tails ⇒ stops set on Gaussian assumptions get hit too often; extreme P&L is dominated by rare events. Q-Q plots make this visible.

**Results (see [p2_distribution.csv](p2_distribution.csv), [p2_distribution.png](p2_distribution.png)):**

| Symbol | Skew | Excess Kurt | JB p |
|--------|------|-------------|------|
| CHOCOLATE | 0.011 | 0.071 | 0.032 |
| VANILLA | -0.002 | 0.056 | 0.137 |
| PISTACHIO | -0.002 | -0.028 | 0.599 |
| STRAWBERRY | -0.007 | -0.019 | 0.714 |
| RASPBERRY | 0.000 | -0.009 | 0.952 |

Returns are **essentially Gaussian** (PIST/STRAW/RASP fail to reject normality). No fat tails. No extreme jump risk to budget for. This is unusual and consistent with a **synthetic Gaussian generator + slight microstructure noise**.

### 2.5 Volume analysis

**Interpretation guide.** ACF(volume) tells us if heavy ticks cluster. `corr(|r|, v)` measures whether volume spikes coincide with big moves (the standard volatility-volume link). Logistic `sign(r_{t+1}) ~ log(v_t)` checks whether volume predicts the next move's direction. Amihud illiquidity = `mean(|r|/v)` measures market impact per unit volume.

**Results (see [p2_volume.csv](p2_volume.csv)):**

| Symbol | ACF(v,1) | corr(|r|, v) | logit β | logit acc | Amihud |
|--------|----------|--------------|---------|-----------|--------|
| CHOCOLATE | -0.002 | -0.001 | -0.015 | 0.520 | 3e-6 |
| VANILLA | -0.002 | 0.000 | -0.014 | 0.519 | 3e-6 |
| PISTACHIO | -0.002 | 0.005 | -0.022 | 0.528 | 3e-6 |
| STRAWBERRY | -0.002 | 0.006 | -0.010 | 0.513 | 4e-6 |
| RASPBERRY | -0.002 | 0.007 | -0.013 | 0.517 | 4e-6 |

Volume is **white noise** (ACF ≈ 0), uncorrelated with absolute returns, and a useless predictor of next-tick direction (logit accuracy ~52% ≈ random). Amihud is uniformly tiny — order-book is deep relative to typical moves. **Volume signals are not actionable on this category.**

---

## PART 3 — CROSS-INSTRUMENT STRUCTURE

### 3.1 Correlation matrix

**Interpretation guide.** High |corr| → candidate for hedged/pair trading. The Pearson matrix reveals the dominant common factors. Rolling correlation tells us if the relationship is stable; high std of rolling corr ⇒ regime instability.

**Results (see [p3_pearson.csv](p3_pearson.csv), [p3_rolling_corr.csv](p3_rolling_corr.csv), [p3_rolling_corr.png](p3_rolling_corr.png)):**

Pairs with |corr| > 0.6 (strong cross-section structure):

| Pair | Pearson | Rolling-50 mean | Rolling std |
|------|---------|-----------------|-------------|
| CHOC ↔ VAN | **−0.92** | −0.91 | 0.05 |
| PIST ↔ STRAW | **+0.91** | +0.91 | 0.03 |
| PIST ↔ RASP | **−0.83** | −0.83 | 0.06 |
| STRAW ↔ RASP | **−0.92** | −0.92 | 0.04 |

The basket has **two clear clusters** with extremely stable cross-relationships:
- **Cluster A**: CHOC ↔ VAN are mirror images (anti-correlated).
- **Cluster B**: PIST ≈ STRAW (same direction); both anti-correlated with RASP.

Other pairs (CHOC×PIST, etc.) show ~0 correlation: the two clusters are **orthogonal**.

### 3.2 Cointegration

**Interpretation guide.** Cointegration = two I(1) series whose linear combination is stationary, i.e. they share a common stochastic trend and the spread mean-reverts. EG p<0.05 ⇒ pair is cointegrated. Hedge ratio β from OLS fixes the mix. Half-life of mean reversion via OLS regression of `Δspread ~ spread_{t-1}` tells how fast the spread returns to its mean. Johansen on the basket reports how many cointegrating vectors exist (rank).

**Results (see [p3_cointegration.csv](p3_cointegration.csv), [p3_johansen.csv](p3_johansen.csv), [p3_best_spread.png](p3_best_spread.png)):**

| Pair | EG p | hedge β | ADF spread p | half-life (ticks) |
|------|------|---------|--------------|-------------------|
| **PIST/STRAW** | 0.036 | -0.254 | 0.009 | 812 |
| CHOC/STRAW | 0.040 | -0.320 | 0.010 | 814 |
| VAN/PIST | 0.044 | -0.283 | 0.011 | 875 |
| CHOC/PIST | 0.048 | 0.492 | 0.013 | 876 |
| VAN/STRAW | 0.074 | 0.143 | 0.021 | 972 |
| VAN/RASP | 0.110 | 0.024 | 0.035 | 1073 |
| CHOC/RASP | 0.194 | 0.057 | 0.072 | 1308 |
| STRAW/RASP | 0.329 | -0.837 | 0.143 | 6069 |
| PIST/RASP | 0.383 | -0.585 | 0.178 | 2819 |
| CHOC/VAN | 0.635 | -1.070 | 0.386 | 1076 |

Surprising: **CHOC/VAN do NOT cointegrate** (p=0.63) despite their −0.92 correlation. The hedge ratio is ≈ −1 but the spread is *non-stationary* — they move together in returns but diverge in cumulative drift. Same for STRAW/RASP.

The strongest cointegrated pair is **PIST/STRAW** (EG p=0.036, half-life ~800 ticks). The Johansen test on the full basket finds **rank ≥ 1** (trace 71.7 > 95% CV 69.8) — at least one cointegrating vector across the basket.

### 3.3 Lead-lag relationships

**Interpretation guide.** Cross-correlation at non-zero lag identifies which instrument moves first. Granger causality tests whether `x_{t-k}` improves the forecast of `y_t` over `y`'s own history alone. A small p-value at lag k means `x` leads `y` by k ticks — trade `y` in `x`'s direction.

**Results (see [p3_ccf.csv](p3_ccf.csv), [p3_granger.csv](p3_granger.csv), [p3_ccf.png](p3_ccf.png)):**

- All non-zero-lag CCF magnitudes are **< 0.025** — no economically meaningful lead-lag at the 100-ms tick level.
- Granger causality finds suggestive patterns: CHOC→STRAW (p=0.04), STRAW→PIST (p=0.05), VAN→CHOC (p=0.06) — all marginally significant, all at lag 1-2.
- Effect sizes are tiny: lead-lag is **statistically detectable but not economically tradable** at this frequency without ultra-low transaction costs.

### 3.4 Order flow impact (Kyle's λ)

**Interpretation guide.** Kyle's λ = price-change per unit signed volume. We use OBI (bid_volume - ask_volume) as a signed-flow proxy. A positive λ means buying pressure pushes price up. The post-shock decay reveals **whether impact persists (momentum) or reverts (overshoot/fade)**.

**Results (see [p3_kyle.csv](p3_kyle.csv)):**

| Symbol | λ | R² | d1 | d5 | d10 | d20 | d30 |
|--------|---|----|----|----|-----|-----|-----|
| CHOCOLATE | 0.95 | 0.013 | -0.143 | -0.140 | -0.150 | -0.138 | -0.098 |
| VANILLA | 0.90 | 0.012 | -0.136 | -0.136 | -0.125 | -0.133 | -0.173 |
| PISTACHIO | 0.87 | 0.017 | -0.129 | -0.131 | -0.128 | -0.123 | -0.121 |
| STRAWBERRY | 0.99 | 0.009 | -0.147 | -0.145 | -0.138 | -0.134 | -0.123 |
| RASPBERRY | 0.94 | 0.008 | -0.149 | -0.152 | -0.164 | -0.160 | -0.152 |

λ is **positive** (contemporaneous OBI shock moves price in same direction), but the post-shock 30-tick decay is **uniformly negative**: after a top-5%-OBI imbalance, the signed price move on the next tick reverses by ~0.14 and stays reverted. This is classic **temporary impact / overshoot** behavior — the OBI shock signals a transient liquidity demand that subsequently corrects. It hints at a fade strategy, but R² is tiny (<2%) so the signal is weak.

---

## PART 4 — STRATEGY SIGNAL BACKTESTS

All backtests use **mid price** with a position limit of ±10. Two cost regimes are reported:
- **Gross** = no transaction costs (pure signal value, optimistic upper bound).
- **Net** = pay half-spread per unit position change (≈ 8 price units per contract turnover; pessimistic lower bound for an aggressive taker).

### 4.1 Mean reversion (z-score, window=50, ±1.5 / 0.3)

| Symbol | Sharpe | Win | Mean PnL/trade | Max DD | n trades | Total PnL (gross) |
|--------|--------|-----|----------------|--------|----------|-------------------|
| CHOCOLATE | -14.15 | 0.68 | -3.7 | -131,010 | 776 | **-130,820** |
| VANILLA | -14.05 | 0.68 | -5.2 | -128,835 | 760 | **-128,670** |
| PISTACHIO | -13.73 | 0.72 | 15.5 | -99,220 | 827 | -99,200 |
| STRAWBERRY | -13.16 | 0.68 | 25.2 | -148,015 | 826 | -147,880 |
| RASPBERRY | -11.62 | 0.72 | 39.6 | -131,295 | 826 | **-130,805** |

**Disastrous**: prices are I(1) random walks at this window — z-score reverts work only on stationary spreads, not standalone prices. Win rate is high (68-72%) but losses are enormous when the price fails to revert. **Avoid.**

### 4.2 Momentum (best N=M per symbol)

| Symbol | N=M | Sharpe | Win | Max DD | Total PnL (gross) | Total PnL (NET) |
|--------|-----|--------|-----|--------|-------------------|-----------------|
| CHOCOLATE | 5 | 12.6 | 0.34 | -3,065 | **142,095** | **-354,330** |
| VANILLA | 5 | 12.7 | 0.33 | -2,445 | **141,160** | -368,735 |
| PISTACHIO | 5 | 12.7 | 0.34 | -1,880 | 113,545 | -358,620 |
| STRAWBERRY | 5 | 13.2 | 0.34 | -3,150 | **183,775** | -343,390 |
| RASPBERRY | 5 | 12.9 | 0.34 | -2,885 | **178,750** | -315,190 |

**Out-of-sample** (train day 2, test days 3-4): consistently positive on every day — see below. Per-day breakdown:

| Symbol | Day 2 | Day 3 | Day 4 |
|--------|-------|-------|-------|
| CHOCOLATE | 56,195 | 41,180 | 44,790 |
| VANILLA | 54,535 | 40,340 | 46,355 |
| PISTACHIO | 41,530 | 36,105 | 35,805 |
| STRAWBERRY | 70,300 | 57,070 | 56,330 |
| RASPBERRY | 66,750 | 57,910 | 53,980 |

**The signal is real (33% win rate × big right-tail trades = positive expectancy), generalizes OOS — but is destroyed by the bid-ask spread when traded as a market taker.** A taker-only momentum strategy is unviable.

### 4.3 Volume-triggered momentum

Threshold (volume > 2× rolling 20-tick mean) **never fires**: volume series is white-noise with very tight tails. Even at 1.5× threshold, no fires. Useless on this category.

### 4.3b OBI fade (Part 3.4 motivated)

| Symbol | Sharpe | Win | Total PnL |
|--------|--------|-----|-----------|
| CHOCOLATE | -0.33 | 0.58 | -1,960 |
| VANILLA | 0.25 | 0.59 | 1,460 |
| PISTACHIO | 0.26 | 0.57 | 1,250 |
| STRAWBERRY | -0.12 | 0.54 | -915 |
| RASPBERRY | 0.59 | 0.59 | 4,330 |

Mixed and small. Not a robust edge.

### 4.4 Lead-lag cross signal

| Follower ← Leader | Sharpe | Win | Total PnL |
|---|---|---|---|
| STRAW ← CHOC | -1.86 | 0.49 | -25,965 |
| PIST ← STRAW | -1.48 | 0.45 | -13,340 |
| **CHOC ← VAN** | **0.69** | 0.53 | **+7,800** |

VAN does mildly lead CHOC (consistent with their −0.92 correlation: when VAN moves up, CHOC follows down). PnL is small and almost certainly cost-negative as a taker. CCF magnitudes (<0.025) cap the realistic edge.

### 4.5 Pairs / stat-arb (cointegrated spread, ±1.5σ entry, 0.3σ exit)

| Pair | β | Sharpe | Win | Total PnL (gross) | Total PnL (NET) |
|------|---|--------|-----|-------------------|-----------------|
| **PIST/STRAW** | -0.25 | **2.50** | 0.37 | **+13,640** | -85,670 |
| CHOC/STRAW | -0.32 | -1.25 | 0.58 | -17,980 | — |
| VAN/PIST | -0.28 | -2.90 | 0.64 | -32,690 | — |
| CHOC/VAN | -1.07 | 0.20 | 0.43 | +3,510 | — |
| STRAW/RASP | -0.84 | -1.17 | 0.60 | -25,415 | — |
| PIST/RASP | -0.58 | -1.15 | 0.55 | -19,695 | — |

PIST/STRAW pairs trade is the best **gross-PnL** pair signal — Sharpe 2.5, OOS-positive (train PnL 5,750, test PnL 4,975 with hedge ratio fixed on train only). But again, taker costs eat all of it (NET = -86k).

### 4.6 Cost-aware optimum search

Sweeping (N, M) and z-score (window, entry, exit) under bid-ask costs:

| Symbol | Best taker strategy | NET PnL | Sharpe |
|--------|---------------------|---------|--------|
| CHOCOLATE | z-score win=200, ent=1.5, exit=0 | 4,130 | 0.36 |
| VANILLA | z-score win=200, ent=1.5, exit=0 | 3,720 | 0.33 |
| PISTACHIO | **z-score win=500, ent=2.0, exit=0** | **9,800** | **1.09** |
| STRAWBERRY | z-score win=50, ent=1.5, exit=0 | 9,435 | 0.67 |
| RASPBERRY | z-score win=500, ent=2.0, exit=0 | 5,240 | 0.38 |

When you slow trading to fire only on extreme z-scores (entries every few thousand ticks) and don't pay for exits, **a slow mean-reversion strategy clears costs by 4–10k PnL per symbol over the full sample**. Not large, but positive. PISTACHIO is the standout (Sharpe 1.09, 9,800 PnL).

### 4.7 Market-making (passive baseline)

A bid-ask spread of ~17 in price units per pair with 30,000 ticks and ~5–8 bps return std means the **theoretical MM revenue is enormous** (10s–100s of thousand units of PnL). Two backtests:

| Fill model | Result |
|-----------|--------|
| **Optimistic (touch)**: fill if next bid ≤ our ask (or vice versa) | +413k–478k per symbol |
| **Strict (cross)**: fill only if next-tick best bid/ask actually crosses our quote | -52k to +3k per symbol |

The truth lies in between and depends on the simulator's bot fill behavior, which cannot be deduced from CSV data alone. **Market-making is the most likely-viable strategy on this category but must be validated via `prosperity4btx`**, not by historical price reconstruction.

### 4.8 Counterparty intel (Round 5 feature)

The `trades_round_5_day_*.csv` files have **buyer = NaN, seller = NaN** for every SNACKPACK trade in the dataset — the counterparty-revelation feature does not apply to this category. There is no specific bot to fade or follow.

---

## PART 5 — STRATEGY VERDICT

### Per-symbol × per-strategy verdict (taker, NET of bid-ask)

| Symbol | Strategy | Sharpe | Win rate | Mean PnL/trade | Max DD | Verdict |
|--------|----------|--------|----------|----------------|--------|---------|
| CHOC | z-score mean-revert (50, 1.5/0.3) | -14.2 | 0.68 | -3.7 | -131,010 | **AVOID** |
| CHOC | momentum N=M=5 (gross) | 12.6 | 0.34 | 1.3 | -3,065 | NO EDGE (taker) |
| CHOC | momentum N=M=5 (NET) | -26.3 | — | — | — | **AVOID** as taker |
| CHOC | OBI fade | -0.3 | 0.58 | — | -6,885 | NO EDGE |
| CHOC | lead-lag (leader=VAN) | 0.7 | 0.53 | 0.2 | -6,570 | WEAK EDGE (gross only) |
| CHOC | z-score win=200, ent=1.5, exit=0 (NET) | 0.4 | — | — | — | WEAK EDGE |
| CHOC | market-making (Prosperity-sim dependent) | — | — | — | — | **STRONG EDGE if MM legal** |
| VAN | z-score (50, 1.5/0.3) | -14.1 | 0.68 | -5.2 | -128,835 | **AVOID** |
| VAN | momentum N=M=5 (gross) | 12.7 | 0.33 | -1.2 | -2,445 | NO EDGE (taker) |
| VAN | momentum N=M=5 (NET) | -27.4 | — | — | — | **AVOID** as taker |
| VAN | z-score win=200, ent=1.5, exit=0 (NET) | 0.3 | — | — | — | WEAK EDGE |
| VAN | OBI fade | 0.2 | 0.59 | — | -5,955 | NO EDGE |
| VAN | market-making | — | — | — | — | **STRONG EDGE if MM legal** |
| PIST | z-score (50, 1.5/0.3) | -13.7 | 0.72 | 15.5 | -99,220 | **AVOID** |
| PIST | momentum N=M=5 (gross) | 12.7 | 0.34 | -0.5 | -1,880 | NO EDGE (taker) |
| PIST | momentum N=M=5 (NET) | -31.2 | — | — | — | **AVOID** as taker |
| PIST | OBI fade | 0.3 | 0.57 | — | -4,670 | NO EDGE |
| PIST | **z-score win=500, ent=2.0, exit=0 (NET)** | **1.09** | — | — | — | **WEAK EDGE → most usable taker signal** |
| PIST | pairs vs STRAW (gross) | 2.5 | 0.37 | -18.0 | -2,380 | WEAK EDGE (gross only) |
| PIST | market-making | — | — | — | — | **STRONG EDGE if MM legal** |
| STRAW | z-score (50, 1.5/0.3) | -13.2 | 0.68 | 25.2 | -148,015 | **AVOID** |
| STRAW | momentum N=M=5 (gross) | 13.2 | 0.34 | 1.3 | -3,150 | NO EDGE (taker) |
| STRAW | momentum N=M=5 (NET) | -21.6 | — | — | — | **AVOID** as taker |
| STRAW | z-score win=50, ent=1.5, exit=0 (NET) | 0.7 | — | — | — | WEAK EDGE |
| STRAW | OBI fade | -0.1 | 0.54 | — | -7,305 | NO EDGE |
| STRAW | pairs vs PIST (gross) | 2.5 | 0.37 | -18.0 | -2,380 | WEAK EDGE (gross only) |
| STRAW | market-making | — | — | — | — | **STRONG EDGE if MM legal** |
| RASP | z-score (50, 1.5/0.3) | -11.6 | 0.72 | 39.6 | -131,295 | **AVOID** |
| RASP | momentum N=M=5 (gross) | 12.9 | 0.34 | -0.4 | -2,885 | NO EDGE (taker) |
| RASP | momentum N=M=5 (NET) | -20.2 | — | — | — | **AVOID** as taker |
| RASP | z-score win=500, ent=2.0, exit=0 (NET) | 0.4 | — | — | — | WEAK EDGE |
| RASP | OBI fade | 0.6 | 0.59 | — | -8,310 | WEAK EDGE |
| RASP | market-making | — | — | — | — | **STRONG EDGE if MM legal** |

### Category summary

**Dominant regime**: **mean-reverting microstructure on top of an I(1) random walk**. Hurst ≈ 0.40 across all five symbols, mildly negative ACF(1) on returns (bid-ask bounce), no fat tails, no volatility clustering, near-Gaussian returns. The basket has two strong correlation clusters (CHOC↔VAN; PIST↔STRAW↔−RASP) but only one **pair (PIST/STRAW)** is properly cointegrated — the other anti-correlated pairs share return sign but not cumulative trend.

**The single most important finding**: **the bid-ask spread (~17 price units, ~17 bps) is much larger than the predictability of any per-tick return signal**. Every taker-style strategy that crosses the spread to enter loses money. Only **passive (market-making) strategies that earn the spread** can reliably profit on this category.

**Recommended live strategies, in priority order:**

1. **Symmetric market-making on all 5 symbols** — quote 1 tick inside the best bid/ask with strict ±10 inventory cap. Validate the actual fill rate via `prosperity4btx` — the simulator's bot behavior, not historical CSV data, will determine PnL. Expected upside (if Prosperity bots fill at-touch like in earlier rounds) is by far the largest available edge here. Add inventory-skew: when long, lean ask aggressive / bid passive; when short, the reverse. This is the dominant strategy for this category.

2. **Slow z-score mean-reversion as a taker overlay**, only on **PISTACHIO** (Sharpe 1.09 net of cost) with window=500, entry=±2.0, exit=0. This is small (~10k PnL over 30k ticks) but additive to MM. Optionally also on STRAW (window=50) and CHOC/VAN (window=200) for a few thousand each. Skip RASP for taker overlay — its signal is the weakest.

3. **PISTACHIO/STRAWBERRY pair as a long-horizon overlay**: when |z-spread| ≥ 1.5σ on the 200-tick rolling spread, take a hedged position. Half-life ~800 ticks means trades held ~13 min. The strategy is gross-positive (Sharpe 2.5) but only net-positive if at least one leg can be filled passively (e.g. enter STRAW passively on the move, take PIST aggressively, or vice versa). Worth experimenting with in `prosperity4btx`.

**Symbols to focus on**: STRAWBERRY and RASPBERRY (highest gross momentum PnL, widest price range — best raw MM opportunity given their spreads). PISTACHIO is the only one with a clear net-of-cost taker edge. CHOC and VAN should be quoted in MM but are weakest as taker targets.

**Symbols to skip**: none — all 5 contribute MM revenue. But avoid CHOC/VAN with any taker strategy at high frequency.

**Risks and caveats**:
- The MM PnL estimate from CSV alone is **wildly uncertain**: optimistic touch-fill says +450k/symbol, strict-cross-fill says ~-30k. The Prosperity simulator's actual bot fill model determines reality. **All MM claims must be validated with `prosperity4btx`** before scaling.
- Counterparty data is missing for SNACKPACKs, so Round-5-specific bot-fading strategies are not available here.
- The high momentum Sharpe (~12) is **not a real arb** — it's a 33%-win/67%-loss expectancy that requires zero transaction cost. Taking it as a justification for aggressive strategies will lose money badly.
- All cointegration/correlation results are **measured over only 30k ticks** (3 days). Test for regime stability daily before sizing up the pair trade.
- Volume-based signals are useless on this category — order-book volume is white noise. Any volume-conditional logic in the trader should be turned off for SNACKPACKs.

---

### Files produced (all in `eda5/snackpack/`)

| File | Content |
|------|---------|
| `eda_part1_2.py` | Parts 1 + 2 driver |
| `eda_part3.py` | Part 3 driver |
| `eda_part4.py`, `eda_part4_v2.py` | Part 4 backtests (raw + cleaned) |
| `eda_part4_oos.py` | Out-of-sample validation |
| `eda_part4_costs.py` | Cost-aware sweep with bid-ask spread |
| `eda_part4_realistic.py` | Cost-aware (N,M) and z-score sweep + naive MM |
| `eda_part4_mm.py` | Strict MM backtest |
| `p1_*.csv`, `p2_*.csv`, `p3_*.csv`, `p4_*.csv` | numerical results |
| `p1_*.png`, `p2_*.png`, `p3_*.png` | plots |
| `_mid.parquet`, `_ret.parquet`, `_vol.parquet` | cached pivot tables |
