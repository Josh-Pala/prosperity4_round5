#!/usr/bin/env bash
# Run all CHOCOLATE sweep variants × days, collect CHOCOLATE PnL + Total.
set -u
ROOT="/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5"
BTX="/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_windteam/.venv/bin/prosperity4btx"
DATA="/tmp/p4data"
OUT="$ROOT/eda5/oxygen_shake/chocolate/sweep_results.csv"
echo "variant,day,choc_pnl,total" > "$OUT"

cd "$ROOT" || exit 1
for f in aglos_round5/mint_clones/chocolate/choc_*.py; do
  name=$(basename "$f" .py)
  for day in 5-2 5-3 5-4; do
    log=$( "$BTX" "$f" "$day" --no-out --no-progress --data "$DATA" 2>&1 )
    choc=$(echo "$log" | grep -E "^OXYGEN_SHAKE_CHOCOLATE:" | awk '{print $2}' | tr -d ',')
    total=$(echo "$log" | grep -E "^Total profit:" | awk '{print $3}' | tr -d ',')
    echo "$name,$day,$choc,$total" >> "$OUT"
    echo "$name $day choc=$choc total=$total"
  done
done
echo "Done. Results in $OUT"
