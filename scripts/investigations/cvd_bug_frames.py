"""How much of the bug survives colour vision deficiency, measured on a pair of frames.

The bug card is one frame against another: the same points, the old checkpoint, the input
as delivered against the return ratio set to 0. For a reader with a red-green deficiency
the difference between the two frames is what is left of the finding. This measures it:

    changed pixels   pixels whose CIELAB delta E76 between the two frames, in normal
                     vision, exceeds CHANGED_DE = 20: a change of class colour, not of
                     shading. 20 is the definition that reproduces the figure first
                     measured on these frames (20.2 percent changed, medians 69.1, 19.7
                     and 36.5); one just noticeable difference, 2.3, is kept as a
                     sensitivity, where eye-dome shading differences dominate the share
    difference       the median delta E76 over those pixels, in normal vision and after
                     Machado 2009 full-severity simulation of deuteranopia and protanopia
                     (surrey.colour_vision), and the share of the normal median that remains

Reproduces the "before" figure from a script in the repository before the palette changes,
and measures the "after" the same way.

Run:
    python scripts/investigations/cvd_bug_frames.py --a docs/design/frames/ggpark_ret0_oblique_e6e9e6.webp \\
        --b docs/design/frames/ggpark_returns_ood_oblique_e6e9e6.webp --label before_light
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from surrey.colour_vision import delta_e76, simulate  # noqa: E402

OUT = ROOT / "docs" / "design" / "cvd_bug_frames.json"
CHANGED_DE = 20.0
SENSITIVITY_DE = 2.3
KINDS = ("normal", "deuteranopia", "protanopia")


def measure(a: np.ndarray, b: np.ndarray, changed_de: float = CHANGED_DE) -> dict:
    """a, b: (h, w, 3) sRGB in [0, 1], the same size."""
    if a.shape != b.shape:
        raise ValueError(f"frames differ in size: {a.shape} and {b.shape}")
    normal = delta_e76(a, b)
    changed = normal > changed_de
    out = {
        "pixels": int(normal.size),
        "changed_share": round(float(changed.mean()), 4),
        "changed_de": changed_de,
    }
    base = float(np.median(normal[changed])) if changed.any() else 0.0
    for kind in KINDS:
        d = (
            delta_e76(simulate(a[changed], kind), simulate(b[changed], kind))
            if changed.any()
            else np.zeros(0)
        )
        med = float(np.median(d)) if d.size else 0.0
        out[kind] = {
            "median_delta_e76": round(med, 1),
            "share_of_normal": round(med / base, 3) if base else None,
        }
    return out


def load(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), np.float64) / 255.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    a, b = load(ROOT / args.a), load(ROOT / args.b)
    row = measure(a, b) | {
        "sensitivity_changed_de_2_3": measure(a, b, SENSITIVITY_DE),
        "a": str(args.a).replace("\\", "/"),
        "b": str(args.b).replace("\\", "/"),
    }
    doc = (
        json.loads(OUT.read_text(encoding="utf-8"))
        if OUT.exists()
        else {"method": __doc__.split("\n\n")[1], "pairs": {}}
    )
    doc["pairs"][args.label] = row
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(args.label, json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
