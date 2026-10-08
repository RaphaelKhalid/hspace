"""Descriptive (post hoc, NOT preregistered) numbers from the v11 W records, for the write-up.
    python hs_v11_descriptive.py        (reads ../results/v11/hs_v11_full_raw.pt; CPU only)
Prints: the canonical window (L16, window 13, the "." after "the Pagan Dynasty itself"): single nudges, sum, both and the
leftover interaction for H, the median decoy (tw1) and J5; per-layer interaction-per-main-effect ("specificity") for
H, J5 and the decoys; and how many windows J5's interaction exceeds H's.
"""
import math
from pathlib import Path

import numpy as np
import torch

RAW = Path(__file__).resolve().parent.parent / "results" / "v11" / "hs_v11_full_raw.pt"
MP = ["01", "23", "14"]
TW = [f"tw{j}" for j in range(16)]


def pct(x):
    return (math.exp(x) - 1) * 100


def leftover(rec, arm):
    return float(np.mean([abs(rec["pairs"][f"{arm}|{m}"][0]) for m in MP]))


def single(rec, arm):
    return float(np.mean([(abs(rec["pairs"][f"{arm}|{m}"][3]) + abs(rec["pairs"][f"{arm}|{m}"][4])) / 2 for m in MP]))


def main():
    r = torch.load(RAW, map_location="cpu", weights_only=False)
    rec = r["W"][16][13]
    print("Canonical window: layer 16, window 13, nudged at the '.' after 'the Pagan Dynasty itself'")
    for arm, name in (("tw1", "median decoy"), ("H", "H-space"), ("J5", "J-space (J5)")):
        print(f"  {name}: mean |leftover| {pct(leftover(rec, arm)):.3f}%")
        for m in MP:
            I, _, _, a, b = rec["pairs"][f"{arm}|{m}"]
            print(f"    pair {m}: A {pct(a):+.2f}%  B {pct(b):+.2f}%  A+B {pct(a + b):+.2f}%  both {pct(a + b + I):+.2f}%  leftover {pct(I):+.3f}%")
    for l in (16, 40):
        W = r["W"][l]
        sp = {k: float(np.exp(np.mean([math.log(leftover(x, k) / single(x, k)) for x in W]))) for k in ["H", "J5"] + TW}
        tw_med = float(np.median([sp[t] for t in TW]))
        j_gt_h = sum(leftover(x, "J5") > leftover(x, "H") for x in W)
        print(f"Layer {l}: interaction per unit main effect  H {sp['H']:.4f}  J5 {sp['J5']:.4f}  decoy median {tw_med:.4f}  "
              f"(H/decoys {sp['H'] / tw_med:.2f}x, J5/H {sp['J5'] / sp['H']:.2f}x); J5 leftover > H leftover in {j_gt_h}/{len(W)} windows")


if __name__ == "__main__":
    main()
