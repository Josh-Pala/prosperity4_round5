"""
Generate GARLIC trader variants by patching FINAL_GLAUCO.py.

For each (basket, threshold) combo, write a copy of FINAL_GLAUCO.py with:
  - new constants GARLIC_INTERCEPT, GARLIC_BETAS, GARLIC_TAKE_THRESHOLD
  - new edge-taker block right after the MINT block
"""
from __future__ import annotations
from pathlib import Path
import re
import textwrap

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "aglos_round5" / "FINAL_GLAUCO.py"
OUT = Path(__file__).resolve().parent

# Basket coefficients from garlic_deep_dive.py (full-sample OLS).
BASKETS = {
    "top4": {
        "intercept": 7857.22,
        "betas": {
            "GALAXY_SOUNDS_BLACK_HOLES": 0.3308,
            "PEBBLES_S": -0.1264,
            "PEBBLES_XL": 0.1875,
            "MICROCHIP_OVAL": -0.1314,
        },
    },
    "pebbles": {
        "intercept": 16165.04,
        "betas": {
            "PEBBLES_S": -0.5554,
            "PEBBLES_XL": 0.1388,
            "PEBBLES_XS": -0.1505,
        },
    },
}


def make_block(basket_name: str) -> str:
    return textwrap.dedent(f"""
        # ---- OXYGEN_SHAKE_GARLIC cross-family fair-value edge taker ({basket_name}) ----
        garlic_dep = state.order_depths.get("OXYGEN_SHAKE_GARLIC")
        if garlic_dep is not None and "OXYGEN_SHAKE_GARLIC" not in engaged_pair_legs:
            other_mids_g = {{}}
            for s in GARLIC_BETAS:
                d = state.order_depths.get(s)
                if d is not None:
                    m = mid_of(d)
                    if m is not None:
                        other_mids_g[s] = m
            if len(other_mids_g) == len(GARLIC_BETAS):
                fair = GARLIC_INTERCEPT + sum(
                    GARLIC_BETAS[s] * other_mids_g[s] for s in GARLIC_BETAS
                )
                bb, ba = best_levels(garlic_dep)
                if bb is not None and ba is not None:
                    pos_g = state.position.get("OXYGEN_SHAKE_GARLIC", 0)
                    existing = result.get("OXYGEN_SHAKE_GARLIC", [])
                    extra: List[Order] = []
                    if fair - ba > GARLIC_TAKE_THRESHOLD:
                        cap = LIMIT - pos_g
                        for o in existing:
                            if o.quantity > 0:
                                cap -= o.quantity
                        qty = min(5, max(0, cap))
                        if qty > 0:
                            extra.append(Order("OXYGEN_SHAKE_GARLIC", ba, qty))
                    elif bb - fair > GARLIC_TAKE_THRESHOLD:
                        cap = LIMIT + pos_g
                        for o in existing:
                            if o.quantity < 0:
                                cap -= -o.quantity
                        qty = min(5, max(0, cap))
                        if qty > 0:
                            extra.append(Order("OXYGEN_SHAKE_GARLIC", bb, -qty))
                    if extra:
                        result["OXYGEN_SHAKE_GARLIC"] = existing + extra
        """).rstrip("\n")


def make_constants(basket: dict, threshold: int, basket_name: str) -> str:
    betas_lines = "\n".join(f'    "{s}": {b:+.4f},' for s, b in basket["betas"].items())
    return f"""
# OXYGEN_SHAKE_GARLIC fair-value model ({basket_name}). See eda5/oxygen_shake/garlic/.
GARLIC_INTERCEPT = {basket["intercept"]:.2f}
GARLIC_BETAS = {{
{betas_lines}
}}
GARLIC_TAKE_THRESHOLD = {threshold}
"""


def patch(src_text: str, basket: dict, threshold: int, basket_name: str) -> str:
    # Add constants right after the MINT_TAKE_THRESHOLD line.
    constants = make_constants(basket, threshold, basket_name)
    src_text = re.sub(
        r"(MINT_TAKE_THRESHOLD\s*=\s*\d+\s*\n)",
        r"\1" + constants,
        src_text,
        count=1,
    )

    # Insert new GARLIC block right after the MINT taker block.
    # Locate the closing of the MINT block: the line that says
    #   result["OXYGEN_SHAKE_MINT"] = existing + extra
    # followed by a blank line and the next comment header.
    mint_end_pattern = (
        r"(result\[\"OXYGEN_SHAKE_MINT\"\] = existing \+ extra\s*\n)"
    )
    block = make_block(basket_name)
    # Indent the block to 8 spaces (it lives inside the run() method body).
    indented = "\n".join(("        " + ln if ln.strip() else "") for ln in block.splitlines())
    src_text = re.sub(
        mint_end_pattern,
        r"\1" + indented + "\n",
        src_text,
        count=1,
    )
    return src_text


def main():
    src_text = SRC.read_text()
    thresholds = [100, 150, 200, 250, 300, 400]
    for basket_name, basket in BASKETS.items():
        for thr in thresholds:
            patched = patch(src_text, basket, thr, basket_name)
            out_path = OUT / f"oxy_garlic_{basket_name}_t{thr}.py"
            out_path.write_text(patched)
            print(f"wrote {out_path.name}")


if __name__ == "__main__":
    main()
