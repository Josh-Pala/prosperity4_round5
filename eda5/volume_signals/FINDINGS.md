# Volume-signal predictability — Findings

## Domanda
Esiste un volume "anomalo" nell'order book o nei trades che permetta di
predire il movimento del mid_price? Cerca un trader informato con ordini
"firma" (taglie tonde tipo 50/100/200/etc).

## Dataset
- 3 giorni di Round 5 (day 2, 3, 4)
- 50 prodotti spot
- 1.5M righe order book + 35k trades (tutti anonimi)

## Risultati chiave

### Distribuzioni
- **Trades**: tutti molto piccoli — max 5 (PEBBLES) o 4 (altri).
  Nessun "whale trade" identificabile per dimensione.
- **Order book**: alcune asimmetrie sì.
  - SNACKPACK_*: tutto livello costante a 60 (no signal)
  - PEBBLES_*: livello 1 capped a 30, picchi 38-45 su PEBBLES_XS
  - ROBOT_*, MICROCHIP_*, TRANSLATOR_*: distribuzione bid/ask_volume_1
    asimmetrica, mediana 11, p95=15, p99=25-33 → c'è chi mette ordini
    grossi episodicamente.

### Test predittivo (h=1,5,10,30,100 tick)
Top |t-stat| signals (mean fut_ret quando il segnale è ON, vs OFF):

| symbol | signal | h | t | dir |
|---|---|---|---|---|
| ROBOT_IRONING | BIG_BID_LVL1_P95 | 100 | -7.0 | DOWN |
| TRANSLATOR_GRAPHITE_MIST | BIG_BID_LVL1_P99 | 100 | +5.6 | UP |
| TRANSLATOR_ASTRO_BLACK | BIG_BID_LVL1_P99 | 100 | +4.1 | UP |
| MICROCHIP_RECTANGLE | BIG_BID_LVL1_P95 | 30 | +4.1 | UP |
| TRANSLATOR_ECLIPSE_CHARCOAL | BIG_BID_LVL1_P99 | 100 | +3.9 | UP |

### Stabilità per giorno (CRUCIALE)
La maggior parte dei top signal **non è stabile** day-by-day → overfit:
- ROBOT_IRONING: d2/d3 +26k, d4 -21k → no good
- MICROCHIP_OVAL: d2 -2.5k, d3/d4 +15k → fragile
- PEBBLES_XS: d2/d3 +22k, d4 -1.8k → marginale

### ✅ Segnali stabili (tutti e 3 i giorni positivi)

**Famiglia TRANSLATOR**, regola `BIG_BID_ONLY → LONG h=200`:

| symbol | thr | n | tot PnL | per-day |
|---|---|---|---|---|
| TRANSLATOR_GRAPHITE_MIST | bid_vol_1 ≥ 15 ∧ ask_vol_1 < 15 | 416 | **+14,649** | d2 +1248, d3 +10204, d4 +1569 |
| TRANSLATOR_ECLIPSE_CHARCOAL | bid_vol_1 ≥ 15 ∧ ask_vol_1 < 15 | 732 | **+12,785** | d2 +6343, d3 +877, d4 +5725 |
| TRANSLATOR_ASTRO_BLACK | bid_vol_1 ≥ 33 ∧ ask_vol_1 normale | 327 | **+3,643** | tutti i giorni positivi |
| TRANSLATOR_VOID_BLUE | bid_vol_1 ≥ 15 only | 115 | +1,748 | piccolo ma positivo |

**Totale stimato (3 giorni): ~32k XIRECs** dopo half-spread costs.

Per simmetria, vale anche `BIG_ASK_ONLY → SHORT h=200` ma il segnale long
funziona meglio (probabilmente trend rialzista nei dati di training).

## Interpretazione
- I trade aggressivi sono troppo piccoli per essere informativi.
- Quello che conta è la **asimmetria delle resting order** nel livello 1:
  qualcuno (probabilmente lo stesso "trader informato") posta ordini
  bid grossi (≥15) quando vuole comprare. È un segnale lento (200 tick di
  holding) ma robusto.
- La famiglia TRANSLATOR è dove il pattern emerge più chiaramente. Le altre
  famiglie hanno segnali più rumorosi o overfit a un singolo giorno.

## Raccomandazioni implementative

1. **Strategia volume-skew TRANSLATOR**: nel `Trader`, per ognuno dei 5
   simboli TRANSLATOR_*, monitora `bid_volume_1` e `ask_volume_1`:
   - Se `bid_volume_1 ≥ 15` e `ask_volume_1 < 15` → apri LONG fino al limite
   - Se `ask_volume_1 ≥ 15` e `bid_volume_1 < 15` → apri SHORT
   - Tieni la posizione 200 tick poi flatten (o trail con un nuovo segnale)
   - Position cap: 10 (limite di gioco)

2. **Considerazioni di liquidità**: half-spread medio 4-5 cents.
   La regola è già robusta a questo costo.

3. **Da NON usare**: pattern su ROBOT, MICROCHIP_OVAL, PEBBLES_XS — sembrano
   profittevoli aggregati ma falliscono in almeno 1 giorno su 3.

## File generati
- `trade_volume_stats.csv`, `ob_volume_stats.csv` — distribuzioni
- `trade_round_sizes.csv`, `ob_round_sizes.csv` — taglie tonde
- `predictive_power_all.csv`, `predictive_power_agg.csv` — t-stat per segnali
- `trading_rules_pnl.csv`, `best_rule_per_symbol.csv` — backtest naive
- `stability_per_day.csv` — controllo overfit
- `translator_grid.csv`, `translator_top_stability.csv` — drilldown finale
