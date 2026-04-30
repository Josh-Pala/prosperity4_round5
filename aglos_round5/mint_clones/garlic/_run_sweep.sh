#!/bin/bash
# Run threshold sweep on GARLIC variants. Output CSV: file,day,garlic_pnl,total_pnl
set -e
PROSPERITY="/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_windteam/.venv/bin/prosperity4btx"
cd "/Users/glaucorampone/Progetti/IMC Prosperity/prosperity4_round5"
OUT="${1:-/dev/stdout}"
echo "file,day,garlic_pnl,total_pnl" > "$OUT"
for f in aglos_round5/mint_clones/garlic/oxy_garlic_*.py; do
    name=$(basename "$f" .py)
    for d in 5-2 5-3 5-4; do
        out=$("$PROSPERITY" "$f" "$d" --no-out --data /tmp/p4data 2>&1)
        garlic=$(echo "$out" | grep "OXYGEN_SHAKE_GARLIC:" | awk '{print $2}' | tr -d ',')
        total=$(echo "$out" | grep "Total profit:" | awk '{print $3}' | tr -d ',')
        echo "$name,$d,$garlic,$total" >> "$OUT"
        echo "  $name $d garlic=$garlic total=$total" >&2
    done
done
