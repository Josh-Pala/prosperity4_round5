"""Generate trader variants for CHOCOLATE edge-taker threshold sweep.

Each variant is a copy of FINAL_GLAUCO.py with an additional CHOCOLATE block
right after the MINT block. The MINT block is left untouched.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "FINAL_GLAUCO.py"
OUT_DIR = Path(__file__).resolve().parent

BASKETS = {
    "microchip": {
        "intercept": 6012.9230,
        "betas": {
            "MICROCHIP_CIRCLE": +0.5873,
            "MICROCHIP_OVAL": -0.2003,
            "MICROCHIP_RECTANGLE": +0.0603,
            "MICROCHIP_SQUARE": -0.0415,
            "MICROCHIP_TRIANGLE": -0.0197,
        },
    },
    "top3": {
        "intercept": 4655.1471,
        "betas": {
            "MICROCHIP_CIRCLE": +0.5769,
            "MICROCHIP_TRIANGLE": -0.2085,
            "SLEEP_POD_NYLON": +0.1666,
        },
    },
}

THRESHOLDS = (100, 150, 200, 250, 300, 400)


def render_constants(name: str, intercept: float, betas: dict[str, float], thr: int) -> str:
    lines = [
        f"# CHOCOLATE fair-value model (basket={name}, thr={thr})",
        f"CHOC_INTERCEPT = {intercept}",
        "CHOC_BETAS = {",
    ]
    for s, b in betas.items():
        lines.append(f'    "{s}": {b:+.4f},')
    lines.append("}")
    lines.append(f"CHOC_TAKE_THRESHOLD = {thr}")
    return "\n".join(lines) + "\n\n\n"


CHOC_BLOCK = """
        # ---- OXYGEN_SHAKE_CHOCOLATE cross-family fair-value edge taker ----
        # Same template as MINT, basket via eda5/oxygen_shake/chocolate/.
        choc_dep = state.order_depths.get("OXYGEN_SHAKE_CHOCOLATE")
        if choc_dep is not None and "OXYGEN_SHAKE_CHOCOLATE" not in engaged_pair_legs:
            other_mids_c = {}
            for s in CHOC_BETAS:
                d = state.order_depths.get(s)
                if d is not None:
                    m = mid_of(d)
                    if m is not None:
                        other_mids_c[s] = m
            if len(other_mids_c) == len(CHOC_BETAS):
                fair_c = CHOC_INTERCEPT + sum(
                    CHOC_BETAS[s] * other_mids_c[s] for s in CHOC_BETAS
                )
                bb_c, ba_c = best_levels(choc_dep)
                if bb_c is not None and ba_c is not None:
                    pos_c = state.position.get("OXYGEN_SHAKE_CHOCOLATE", 0)
                    existing_c = result.get("OXYGEN_SHAKE_CHOCOLATE", [])
                    extra_c: List[Order] = []
                    if fair_c - ba_c > CHOC_TAKE_THRESHOLD:
                        cap = LIMIT - pos_c
                        for o in existing_c:
                            if o.quantity > 0:
                                cap -= o.quantity
                        qty = min(5, max(0, cap))
                        if qty > 0:
                            extra_c.append(Order("OXYGEN_SHAKE_CHOCOLATE", ba_c, qty))
                    elif bb_c - fair_c > CHOC_TAKE_THRESHOLD:
                        cap = LIMIT + pos_c
                        for o in existing_c:
                            if o.quantity < 0:
                                cap -= -o.quantity
                        qty = min(5, max(0, cap))
                        if qty > 0:
                            extra_c.append(Order("OXYGEN_SHAKE_CHOCOLATE", bb_c, -qty))
                    if extra_c:
                        result["OXYGEN_SHAKE_CHOCOLATE"] = existing_c + extra_c

"""


def main() -> None:
    base = BASE.read_text()

    # Anchors:
    # 1) constants: insert just BEFORE "def mid_of(d: OrderDepth):" (line 162)
    const_anchor = "def mid_of(d: OrderDepth):"
    # 2) block: insert just BEFORE the PEBBLES dedicated block comment
    block_anchor = "        # ---- PEBBLES dedicated block (v11)"

    if const_anchor not in base or block_anchor not in base:
        raise SystemExit("Anchors not found in FINAL_GLAUCO.py — adjust generator.")

    for basket_name, cfg in BASKETS.items():
        for thr in THRESHOLDS:
            consts = render_constants(basket_name, cfg["intercept"], cfg["betas"], thr)
            patched = base.replace(const_anchor, consts + const_anchor, 1)
            patched = patched.replace(block_anchor, CHOC_BLOCK + block_anchor, 1)
            out = OUT_DIR / f"choc_{basket_name}_t{thr}.py"
            out.write_text(patched)
            print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
