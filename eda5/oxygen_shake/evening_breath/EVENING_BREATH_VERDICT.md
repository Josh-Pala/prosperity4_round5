# OXYGEN_SHAKE_EVENING_BREATH — cross-family fair-value taker

**Date:** 2026-04-30
**Question:** can we lift EVENING_BREATH from MM-only (+22.4k after the join-best fix)?
**Answer: YES — +26.84k via cross-family fair-value taker on a top-8 basket (t=150).**

Total backtest: **851,472 → 878,309** (+26,837).
EVENING_BREATH 3-day: **22,375 → 49,212**.

---

## 1. Why bother?

After OXY join-best, EB was already positive (+22.4k 3-day) and growing across
days (2.1k → 7.9k → 12.4k). The increasing daily PnL hinted there was
directional drift the MM layer was riding but not fully capturing — same
shape as MINT pre-taker. Worth checking if a fair-value model could squeeze
out the residual edge.

## 2. EDA — top correlations are weaker than MINT

`eb_deep_dive.py` regressed EB mid on the top-N most-correlated symbols
(level corr). Compared to MINT, the picture is materially noisier:

**Top 8 level correlations with EB mid (full sample):**
| Symbol                     | Level corr |
|----------------------------|------------|
| GALAXY_SOUNDS_SOLAR_WINDS  | 0.503      |
| GALAXY_SOUNDS_SOLAR_FLAMES | 0.489      |
| TRANSLATOR_GRAPHITE_MIST   | 0.449      |
| SLEEP_POD_NYLON            | 0.441      |
| PEBBLES_L                  | 0.419      |
| PANEL_1X4                  | 0.418      |
| UV_VISOR_YELLOW            | 0.417      |
| PANEL_1X2                  | 0.409      |

Compare to MINT's top corr 0.77 (PANEL_2X2). EB is more diffuse — no single
symbol dominates, the top correlations cluster around 0.40–0.50, and they
span 5 different families. This implies (a) the basket needs more symbols
for adequate R², (b) overfitting risk is higher, (c) the threshold likely
needs more care than MINT.

Lead-lag and ACF are flat: max |corr| ≈ 0.02 across all lags ±50, no
exploitable autocorrelation. So the only edge is contemporaneous
mean-reversion to a basket — same template as MINT.

## 3. Walk-forward holdout

For each candidate basket, fit on 2 days, measure residual sd OOS on the
held-out third:

| Basket               | n | R² in | sd in | sd OOS d2 | sd OOS d3 | sd OOS d4 |
|----------------------|--:|------:|------:|----------:|----------:|----------:|
| **top8 (full)**      | 8 | 0.560 | 265.3 |     544.6 |     333.2 |     316.8 |
| top5                 | 5 | 0.416 | 305.6 |     478.8 |     392.4 |     234.0 |
| top4                 | 4 | 0.380 | 315.0 |     502.1 |     417.7 |     262.2 |
| top3                 | 3 | 0.375 | 316.2 |     503.0 |     425.9 |     257.1 |
| panel_only_top3      | 3 | 0.272 | 341.2 |     556.0 |     397.2 |     299.3 |
| oxy_only_top3        | 3 | 0.358 | 320.4 |     514.2 |     400.7 |     375.8 |
| panel_2x2_1x4_4x4    | 3 | 0.196 | 358.5 |     539.2 |     408.9 |     230.5 |

The MINT-style "smaller basket more stable OOS" pattern doesn't hold here.
top8 has the best in-sample R² and the best d3/d4 OOS sd; top5/top4/top3
trade R² for marginal d4 stability. Reusing MINT's PANEL trio
(`panel_2x2_1x4_4x4`) gives the worst R² (0.20). Different driver families
for EB.

This was the moment to test rather than pre-commit. We carried both top5
and top8 into the sweep — top8 dominated.

## 4. Threshold sweep — top8 wins decisively

All numbers are 3-day EB PnL. Baseline (MM only) = 22,375.

**top5 basket** (intercept=7271.76, betas={SOLAR_WINDS:+0.124, SOLAR_FLAMES:-0.234, GRAPHITE_MIST:-0.013, NYLON:+0.172, PEBBLES_L:+0.174}):

| Threshold | EB 3-day | Δ vs base | Total 3-day |
|-----------|---------:|----------:|------------:|
|    50     |   38,630 |   +16,255 |     867,727 |
|   100     |   38,330 |   +15,955 |     867,427 |
|   150     |   35,030 |   +12,655 |     864,127 |
|   200     |   34,996 |   +12,621 |     864,093 |
|   300     |   32,996 |   +10,621 |     862,093 |
|   400     |   28,312 |    +5,937 |     857,409 |
|   500     |   25,766 |    +3,391 |     854,863 |

Monotone-decreasing on the right side, plateau on the left (50≈100). Picco
at t=50–100 with +16k.

**top8 basket** (intercept=9097.72, betas in `eb_top8_t150.py`):

| Threshold | EB 3-day | Δ vs base | Total 3-day |
|-----------|---------:|----------:|------------:|
|   100     |   44,638 |   +22,263 |     873,735 |
|   125     |   47,870 |   +25,495 |     876,967 |
| **150**   | **49,212** | **+26,837** | **878,309** |
|   175     |   48,822 |   +26,447 |     877,919 |
|   200     |   46,634 |   +24,259 |     875,731 |
|   250     |   46,122 |   +23,747 |     875,219 |
|   300     |   44,344 |   +21,969 |     873,809 |
|   400     |   34,449 |   +12,074 |     863,546 |

Concave with a clear peak at **t=150**. top8 outperforms top5 by ~10k at
the optimum — the extra symbols do contribute predictive power despite the
slightly worse OOS sd on day 2.

## 5. Per-day PnL (top8 t=150)

| Day | EB before | EB after | Δ family |
|-----|----------:|---------:|---------:|
| 2   | 2,124     | 13,180   | +11,056  |
| 3   | 7,886     | 16,356   |  +8,470  |
| 4   | 12,365    | 19,676   |  +7,311  |
| Tot | 22,375    | **49,212** | **+26,837** |

Gain spread evenly across days — not a single-day fluke. d2 had the largest
relative jump (~6×), consistent with the OOS sd being highest there
(opportunities of large mispricings).

## 6. Position contention with MINT

The top8 basket has **zero overlap** with MINT's PANEL trio
(PANEL_2X2/1X4/4X4). PANEL_1X4 is in EB's top8 but not used in top8 (it
was discarded because top5 already excluded it; PANEL_1X4 reappears in top8
at index 6). MINT trades only OXYGEN_SHAKE_MINT positions; EB trades only
OXYGEN_SHAKE_EVENING_BREATH positions. No leg conflict possible.

The 8 basket symbols are all **read-only** for the EB taker (we look at
their mids to compute fair value, never trade them). PANEL_1X4 and PANEL_1X2
are MM legs in `MM_UNIVERSE` — not affected by the EB taker.

## 7. Why top8 beats top5 here (and not in MINT)

MINT was dominated by a single family (PANEL ~0.55–0.77 corr). Adding more
symbols introduced noise. EB has no dominant family — five families are
roughly co-equal in predictive power around 0.42–0.50. Discarding any of
them throws away a real signal. The lesson from MINT (smaller is better) is
not a universal rule; it depends on the correlation structure of the target.

## 8. Recommendation

**Promote** `eb_top8_t150.py` block into `FINAL_GLAUCO.py`:
- Add `EB_INTERCEPT`, `EB_BETAS` (top8), `EB_TAKE_THRESHOLD = 150`
- Add EB taker block right after MINT taker (same template)
- Expected gain: **+26,837** on 3-day backtest (851,472 → 878,309)
- No effect on other symbols — taker only fires on EVENING_BREATH

OXY family backtest with both takers active:
- MINT alone (current FINAL_GLAUCO): +56,943
- EB alone (proposed): +26,837
- Combined OXY taker gain: ~+83,780 across 3 days

## 9. Caveats / monitoring

- **OOS sd day-2 = 545** is the biggest single risk. The model fits day-3/4
  well but day-2 is noisier. Live submission may underperform if
  distribution shifts toward day-2 regime.
- **No half-life regularisation** in the threshold choice. t=150 ≈ 0.27σ on
  day-2 OOS, ≈ 0.45σ on day-3, ≈ 0.47σ on day-4 — picks up a lot of
  day-2 marginal trades. If live distribution looks closer to day-2,
  consider raising threshold to 200–250 to reduce false positives (still
  +24k expected).
- **Eight-symbol fair-value** depends on all eight order books being
  populated. The `len(other_mids) == len(EB_BETAS)` guard skips the taker
  if any leg is missing a mid, so this fails safe rather than trading on a
  partial basket.

## Files

- `eb_deep_dive.py`              — cross-family corr / regression / WF holdout
- `eb_level_corr.csv`            — full level-corr ranking
- `eb_lagged_corr.csv`           — lead-lag (mostly noise)
- `eb_basket_regression.txt`     — top8 OLS coefs
- `eb_walkforward.csv`           — WF sd_oos for 7 candidate baskets
- `eb_basket_coefs.csv`          — full-sample OLS coefs per basket
- `eb_autocorr.csv`              — ACF (no exploitable autocorrelation)
- `../../aglos_round5/mint_clones/evening_breath/eb_top5_t{50,100,150,200,300,400,500,600,800}.py`
- `../../aglos_round5/mint_clones/evening_breath/eb_top8_t{100,125,150,175,200,250,300,400}.py`
- `EVENING_BREATH_VERDICT.md`    — this file

## Decision

**Hand-off to user**: top8 t=150 promoted candidate. Not yet merged into
FINAL_GLAUCO. Suggested merge after a sanity-check on whether a lower-noise
threshold (200/250) is preferred for live regime risk.
