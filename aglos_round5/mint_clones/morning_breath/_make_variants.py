"""Generate trader variants for OXYGEN_SHAKE_MORNING_BREATH cross-family taker.

Each variant = FINAL_GLAUCO.py + an MB taker block (template = MINT block) with
a specific basket and threshold. The original MINT block is left untouched.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "FINAL_GLAUCO.py"
OUT_DIR = Path(__file__).resolve().parent

# Three candidate baskets (full-sample OLS coefs from mb_deep_dive.py).
BASKETS: dict[str, tuple[float, dict[str, float]]] = {
    "top3": (
        16775.71,
        {"PEBBLES_M": -0.4927, "ROBOT_MOPPING": -0.2479, "PANEL_1X4": +0.1099},
    ),
    "top4": (
        15885.59,
        {
            "PEBBLES_M": -0.4628,
            "ROBOT_MOPPING": -0.2322,
            "PANEL_1X4": +0.0807,
            "ROBOT_IRONING": +0.0785,
        },
    ),
    "pebbles": (
        14866.29,
        {"PEBBLES_M": -0.5864, "PEBBLES_XS": +0.1556},
    ),
}

THRESHOLDS = [100, 150, 200, 250, 300, 400]


MB_HEADER_TEMPLATE = '''
# OXYGEN_SHAKE_MORNING_BREATH fair-value model (eda5/oxygen_shake/morning_breath/).
# Basket: {basket_name}.  Threshold: {threshold}.
MB_INTERCEPT = {intercept}
MB_BETAS = {{
{betas_block}
}}
MB_TAKE_THRESHOLD = {threshold}
'''

MB_TAKER_BLOCK = '''
        # ---- OXYGEN_SHAKE_MORNING_BREATH cross-family fair-value edge taker ----
        # Same template as MINT/AMBER. Independent of MINT taker.
        mb_dep = state.order_depths.get("OXYGEN_SHAKE_MORNING_BREATH")
        if mb_dep is not None and "OXYGEN_SHAKE_MORNING_BREATH" not in engaged_pair_legs:
            mb_other = {}
            for s in MB_BETAS:
                d = state.order_depths.get(s)
                if d is not None:
                    m = mid_of(d)
                    if m is not None:
                        mb_other[s] = m
            if len(mb_other) == len(MB_BETAS):
                fair_mb = MB_INTERCEPT + sum(
                    MB_BETAS[s] * mb_other[s] for s in MB_BETAS
                )
                bb_mb, ba_mb = best_levels(mb_dep)
                if bb_mb is not None and ba_mb is not None:
                    pos_mb = state.position.get("OXYGEN_SHAKE_MORNING_BREATH", 0)
                    existing_mb = result.get("OXYGEN_SHAKE_MORNING_BREATH", [])
                    extra_mb: List[Order] = []
                    if fair_mb - ba_mb > MB_TAKE_THRESHOLD:
                        cap = LIMIT - pos_mb
                        for o in existing_mb:
                            if o.quantity > 0:
                                cap -= o.quantity
                        qty = min(5, max(0, cap))
                        if qty > 0:
                            extra_mb.append(Order("OXYGEN_SHAKE_MORNING_BREATH", ba_mb, qty))
                    elif bb_mb - fair_mb > MB_TAKE_THRESHOLD:
                        cap = LIMIT + pos_mb
                        for o in existing_mb:
                            if o.quantity < 0:
                                cap -= -o.quantity
                        qty = min(5, max(0, cap))
                        if qty > 0:
                            extra_mb.append(Order("OXYGEN_SHAKE_MORNING_BREATH", bb_mb, -qty))
                    if extra_mb:
                        result["OXYGEN_SHAKE_MORNING_BREATH"] = existing_mb + extra_mb
'''


# Anchor right after the MINT taker block. We insert just before the line
# that begins the PEBBLES dedicated block.
MB_INSERTION_ANCHOR = "        # ---- PEBBLES dedicated block (v11) — passive entry on pair legs ----"

# Anchor for MB constants — insert right after MINT_TAKE_THRESHOLD = 200 block
MB_CONST_ANCHOR = "MINT_TAKE_THRESHOLD = 200\n"


def render_betas(betas: dict[str, float]) -> str:
    return "\n".join(f'    "{s}": {b:+.4f},' for s, b in betas.items())


def make_variant(basket_name: str, threshold: int) -> Path:
    template_text = TEMPLATE.read_text()
    intercept, betas = BASKETS[basket_name]
    header = MB_HEADER_TEMPLATE.format(
        basket_name=basket_name,
        threshold=threshold,
        intercept=f"{intercept:.2f}",
        betas_block=render_betas(betas),
    )
    new_text = template_text.replace(
        MB_CONST_ANCHOR,
        MB_CONST_ANCHOR + header,
        1,
    )
    new_text = new_text.replace(
        MB_INSERTION_ANCHOR,
        MB_TAKER_BLOCK + "\n" + MB_INSERTION_ANCHOR,
        1,
    )
    out_path = OUT_DIR / f"mb_{basket_name}_t{threshold}.py"
    out_path.write_text(new_text)
    return out_path


def main():
    for basket in BASKETS:
        for thr in THRESHOLDS:
            p = make_variant(basket, thr)
            print(f"wrote {p.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
