"""Softmax-saturation check on the v11 W records (post hoc, descriptive, NOT preregistered; CPU only, $0).
    python hs_v11_softmax.py        (reads ../results/v11/hs_v11_full_raw.pt, writes ../results/v11/hs_v11_softmax.json)

Question: J5 carries more pairwise interaction than H-space in 39/40 (L16) and 40/40 (L40) v11 W windows. Is that
edge final-readout saturation (final norm + log-softmax), which the H-lens readout excludes by design?

hs_v10.rect stores, per pair, (I, I_trunk, I_head, A, B) with I = I_trunk + I_head exactly:
  I_trunk = f(both) - f(rows_A + rows_B - rows_0)   non-additivity of the final residual (the trunk)
  I_head  = f(rows_A + rows_B - rows_0) - f(A) - f(B) + f(0)   non-additivity added by the final norm + softmax readout

Rule (written and committed before this script was first run):
  Per window, per arm: |I_c| = mean over the 3 pairs of |component c|.
  D_c = per-window log(|I_c|(J5) / |I_c|(H)), averaged; 95% CI by article-cluster bootstrap (2000 resamples, seed 0).
  Per layer:
    "saturation explains J5's edge"  if D_trunk CI includes 0 or is below 0 (J5 no longer ahead in the trunk);
    "saturation explains part"       if D_trunk CI is above 0 and D_trunk <= 0.5 * D_I;
    "not saturation"                 otherwise.
  Also reported: head share |I_head| / (|I_trunk| + |I_head|) for H, J5 and the median of the 16 tw decoys;
  windows where J5's |I_trunk| exceeds H's.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "v11" / "hs_v11_full_raw.pt"
OUTF = ROOT / "results" / "v11" / "hs_v11_softmax.json"
MP = ["01", "23", "14"]
TW = [f"tw{j}" for j in range(16)]
COMP = {"I": 0, "I_trunk": 1, "I_head": 2}


def absc(rec, arm, c):
    return float(np.mean([abs(rec["pairs"][f"{arm}|{m}"][COMP[c]]) for m in MP]))


def cluster_ci(v, clusters, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ids = sorted(set(clusters))
    by = {a: [i for i, c in enumerate(clusters) if c == a] for a in ids}
    stats = []
    for _ in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = [i for a in pick for i in by[a]]
        stats.append(float(np.mean(v[idx])))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def head_share(W, arm):
    return float(np.median([absc(r, arm, "I_head") / (absc(r, arm, "I_trunk") + absc(r, arm, "I_head") + 1e-12) for r in W]))


def main():
    r = torch.load(RAW, map_location="cpu", weights_only=False)
    res = {"source": str(RAW.relative_to(ROOT)), "preregistered": False, "layers": {}}
    for l in (16, 40):
        W = r["W"][l]
        cl = [x["article"] for x in W]
        L = {"n": len(W)}
        for c in COMP:
            v = np.array([math.log((absc(x, "J5", c) + 1e-12) / (absc(x, "H", c) + 1e-12)) for x in W])
            L[f"D_{c}"] = {"mean": float(v.mean()), "ratio": float(math.exp(v.mean())), "ci95": cluster_ci(v, cl)}
        L["J5_trunk_gt_H_trunk"] = int(sum(absc(x, "J5", "I_trunk") > absc(x, "H", "I_trunk") for x in W))
        L["J5_total_gt_H_total"] = int(sum(absc(x, "J5", "I") > absc(x, "H", "I") for x in W))
        L["head_share_median"] = {"H": head_share(W, "H"), "J5": head_share(W, "J5"),
                                  "tw_median": float(np.median([head_share(W, t) for t in TW]))}
        dt, di = L["D_I_trunk"], L["D_I"]
        if dt["ci95"][0] <= 0:
            L["verdict"] = "saturation explains J5's edge"
        elif dt["mean"] <= 0.5 * di["mean"]:
            L["verdict"] = "saturation explains part"
        else:
            L["verdict"] = "not saturation"
        res["layers"][str(l)] = L
        print(f"L{l} (n={len(W)}): J5/H total {di['ratio']:.2f}x {[round(math.exp(x), 2) for x in di['ci95']]} | "
              f"trunk {dt['ratio']:.2f}x {[round(math.exp(x), 2) for x in dt['ci95']]} | "
              f"head {L['D_I_head']['ratio']:.2f}x | J5>H trunk {L['J5_trunk_gt_H_trunk']}/{len(W)} | "
              f"head share H {L['head_share_median']['H']:.2f} J5 {L['head_share_median']['J5']:.2f} "
              f"decoys {L['head_share_median']['tw_median']:.2f} -> {L['verdict']}")
    json.dump(res, open(OUTF, "w"), indent=1)
    print(f"wrote {OUTF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
