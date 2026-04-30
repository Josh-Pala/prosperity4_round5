# OXYGEN_SHAKE_GARLIC — cross-family fair-value taker

**Date:** 2026-04-30
**Question:** can we replicate the MINT taker on GARLIC and lift it from
+10.9k to a much larger figure?
**Answer: YES — +48.25k via PEBBLES-only basket at threshold 250.**

---

## 1. Why GARLIC was leaving money on the table

After the OXY join-best fix, GARLIC sat at +10.9k across 3 days (median per
OXY). Yet the symbol has the **strongest intraday drift of any OXY** (D2:
-1.8k starting drift, D4: +8.5k drift) — the perfect target for a directional
fair-value taker, since the MM layer alone cannot absorb sustained
mid-price movement.

Per-day baseline GARLIC PnL (FINAL_GLAUCO):

| Day | GARLIC | OXY total | Total |
|-----|-------:|----------:|------:|
| 2   | -1,803 | 32,777    | 253,374 |
| 3   |  4,170 | 47,477    | 298,922 |
| 4   |  8,544 | 50,279    | 299,176 |
| Tot | **10,911** | 130,533 | **851,472** |

## 2. Discovery — GARLIC has a much higher in-sample R² than MINT

`garlic_deep_dive.py` regressed GARLIC mid on the top-8 most-correlated
symbols. Hits **R²=0.901** with residual sd=300 (vs MINT top-8 R²=0.675,
sd=290). Stronger signal in-sample.

**Top correlations with GARLIC mid (full sample):**

| Symbol                    | Level corr |
|---------------------------|-----------:|
| GALAXY_SOUNDS_BLACK_HOLES | 0.885 |
| PEBBLES_S                 | 0.884 |
| PEBBLES_XL                | 0.853 |
| MICROCHIP_OVAL            | 0.847 |
| PANEL_2X4                 | 0.843 |
| PEBBLES_XS                | 0.831 |
| UV_VISOR_AMBER            | 0.822 |
| ROBOT_IRONING             | 0.816 |

The residual mean-reverts with half-life 445 ticks (vs MINT's 800 → faster
reversion, even better for the taker).

Note no useful lead-lag (max |corr|<0.018) and ACF on returns ~0.01 across
all lags — the alpha is purely cross-sectional level mean-reversion, same
shape as MINT.

## 3. Overfitting check led to PEBBLES-only basket

Walk-forward holdout (train on 2 days, test on the 3rd):

| Basket             | k | R² in | sd in | sd OOS d2 | sd OOS d3 | sd OOS d4 |
|--------------------|--:|------:|------:|----------:|----------:|----------:|
| top8 (full)        | 8 | 0.901 | 299.8 | 483.0 | 536.3 | 505.3 |
| top5               | 5 | 0.871 | 342.1 | 447.4 | 339.2 | 280.9 |
| top4               | 4 | 0.871 | 342.1 | 444.7 | 332.8 | 276.1 |
| top3               | 3 | 0.862 | 353.8 | 465.0 | 325.8 | 297.4 |
| **PEBBLES-only**   | 3 | 0.840 | 381.3 | **410.6** | **410.4** | **416.2** |

`PEBBLES_S + PEBBLES_XL + PEBBLES_XS` has slightly lower in-sample R² but
**exceptionally stable OOS** (410±3 across all 3 holdouts), exactly the
same pattern that drove the MINT decision toward PANEL-only. Top4 has
better sd_OOS on d3/d4 (333/276) but pays a 444 sd on d2 — i.e. it is
non-stationary across days. PEBBLES-only is the safer bet.

Final coefficients (full-sample fit, used in production candidate):
- intercept = 16165.04
- PEBBLES_S:  -0.5554
- PEBBLES_XL: +0.1388
- PEBBLES_XS: -0.1505

(top5 was discarded because PANEL_2X4 collapsed to β=0.0000 — collinear
with PEBBLES family in this sample.)

## 4. Threshold sweep (3-day backtester totals)

PEBBLES-only basket:

| Threshold | GARLIC 3-day | Total 3-day | Δ vs base 851,472 |
|-----------|-------------:|------------:|------------------:|
|   100     |  50,742      |   891,302   |        +39,830    |
|   150     |  57,878      |   898,440   |        +46,968    |
|   200     |  56,979      |   897,541   |        +46,069    |
| **250**   | **59,158**   | **899,720** |    **+48,248**    |
|   300     |  56,927      |   897,489   |        +46,017    |
|   400     |  51,164      |   891,725   |        +40,253    |

**Concave with peak at threshold 250.** Coherent with sd_OOS ≈ 410:
threshold 250 is ~0.6σ OOS, the same regime that worked for MINT at 200
(0.66σ OOS). Below 100 picks marginal trades, above 300 misses real edges.

Top4 sweep (alternative basket, for reference):

| Threshold | GARLIC 3-day | Total 3-day | Δ vs base |
|-----------|-------------:|------------:|----------:|
|   100     |  59,668      |   900,228   |  +48,756  |
|   150     |  51,007      |   891,568   |  +40,096  |
|   200     |  54,003      |   894,566   |  +43,094  |
|   250     |  54,047      |   894,608   |  +43,136  |
|   300     |  54,509      |   895,070   |  +43,598  |
|   400     |  48,502      |   889,064   |  +37,592  |

Top4_t100 is technically the absolute peak across the full sweep (+48,756
vs +48,248), but the curve is **non-monotonic**: t100 is an isolated
maximum, t150 drops to 51k, then 54k plateau. PEBBLES-only's curve is a
clean bell — much more trustworthy.

## 5. Per-day PnL of the recommended candidate (pebbles_t250)

| Day | GARLIC before | GARLIC after | Δ GARLIC | Total before | Total after | Δ Total |
|-----|-------------:|-------------:|---------:|-------------:|-------------:|--------:|
| 2   | -1,803       | 19,851       | +21,654  | 253,374      | 275,028      | +21,654 |
| 3   |  4,170       | 23,798       | +19,628  | 298,922      | 318,550      | +19,628 |
| 4   |  8,544       | 15,509       | +6,965   | 299,176      | 306,142      | +6,966  |
| Tot | **10,911**   | **59,158**   | **+48,247** | **851,472** | **899,720** | **+48,248** |

Day-2 (the largest baseline drift day) gains the most, day-4 the least —
consistent with the model: when GARLIC drifts away from the basket, the
taker captures the mispricing; when it tracks, less to capture.

## 6. Conflicts / interactions with existing strategies

The PEBBLES-only basket reads `mid_price` only — it does **not** trade
PEBBLES, so no position-limit interaction. PEB pair-trading uses
PEBBLES_XS/S/M (the GARLIC basket touches XS/S/XL) — even at the worst-case
overlap on PEBBLES_S/XS, only the mids are read; the PEB pair quotes and
PEB position limits are unaffected.

The MINT taker (PANEL family) and the GARLIC taker (PEBBLES + GALAXY)
operate on disjoint regressor families → no model interference.

`engaged_pair_legs` exclusion clause already in the template handles the
case where GARLIC ever becomes part of a future pair trade (currently it
isn't).

## 7. Why backtest PnL > naive expectation

In-sample sd 381 vs OOS 410 (gap of 8%, much smaller than MINT's 303 vs
440 = 45% gap). Threshold 250 ≈ 0.61σ OOS, and the residual half-life is
445 ticks (≈half MINT's 800), so the taker fires more often and closes
faster.

Risk: if GARLIC's relationship to PEBBLES family breaks (e.g. PEBLLES
constant-sum invariant changes regime — see project memory), this strategy
loses. PEBBLES is one of the more stable families in this game, so the
risk is moderate.

## 8. Decision

**Recommend promoting `oxy_garlic_pebbles_t250.py` to FINAL_GLAUCO.py.**

- Add `GARLIC_INTERCEPT`, `GARLIC_BETAS`, `GARLIC_TAKE_THRESHOLD` constants.
- Add GARLIC taker block right after the MINT taker (same template).
- Expected gain: +48.25k across days 2-4 (validated with backtester).

Top4_t100 was rejected despite slightly higher absolute PnL because:
1. its threshold curve is non-monotonic (the +800 over pebbles_t250 looks
   like an in-sample artifact),
2. top4 sd_OOS is unstable across days (276-444 vs PEBBLES 410±3),
3. PEBBLES-only echoes the MINT lesson (single-family > top-K composite)
   and is therefore more defensible.

If the user disagrees with the conservative call, top4_t100 remains the
runner-up.

## Files

- `garlic_deep_dive.py`           — cross-family corr / regression / ACF / walk-forward
- `garlic_lagged_corr.csv`        — lead-lag correlations (all noise)
- `garlic_basket_regression.txt`  — top-8 OLS coefs
- `garlic_holdout_results.csv`    — walk-forward sd per candidate basket
- `garlic_autocorr.csv`           — ACF (no exploitable autocorrelation)
- `sweep_results_clean.csv`       — full per-day sweep results
- `sweep_summary.csv`             — 3-day totals per variant
- `../../aglos_round5/mint_clones/garlic/oxy_garlic_{pebbles,top4}_t{100,150,200,250,300,400}.py`
                                  — 12 backtester-validated variants
- `GARLIC_VERDICT.md`             — this file
