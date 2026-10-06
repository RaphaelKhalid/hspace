"""Descriptive robustness of the v9 result, recomputed from the saved per-window records (laptop, $0).

    python hs_v9_robust.py [out/pod/live/hs_v9_full_raw.pt] [out/hs_v9_robust.json]

Per window: ex = mean |I| over the matched EX pairs (0,1),(2,3),(1,4); tw_j = the same for twin j;
T1_w = mean_j log(ex / tw_j). Reports the geometric-mean ratio exp(mean T1_w), the ratio of mean |I|,
the median per-window ratio ex / mean_j tw_j, the fraction of windows where EX beats every twin, and the
same ratios after dropping the 5 windows with the largest T1_w. The raw file holds 88 of the 89 windows
scored in hs_v9_full.json, so the full-sample figures differ from the headline in the third digit.
"""
import json
import sys

import numpy as np
import torch

MP = [(0, 1), (2, 3), (1, 4)]


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "out/pod/live/hs_v9_full_raw.pt"
    dst = sys.argv[2] if len(sys.argv) > 2 else "out/hs_v9_robust.json"
    D = torch.load(src, weights_only=False)
    res = {"source": src, "layers": {}}
    for l in sorted(D["rec"].keys()):
        rec = D["rec"][l]
        tw = sorted({k.split("|")[0] for k in rec[0]["pairs"] if k.startswith("tw")})
        ex, tws = [], []
        for r in rec:
            ex.append(np.mean([abs(r["pairs"][f"EX|{i}{j}"][0]) for i, j in MP]))
            tws.append([np.mean([abs(r["pairs"][f"{t}|{i}{j}"][0]) for i, j in MP]) for t in tw])
        ex, tws = np.array(ex), np.array(tws)
        T1 = np.log(ex[:, None] / tws).mean(1)
        keep = np.argsort(T1)[::-1][5:]
        res["layers"][str(l)] = {
            "n_windows": int(len(ex)),
            "geomean_ratio": float(np.exp(T1.mean())),
            "ratio_of_means": float(ex.mean() / tws.mean()),
            "median_window_ratio": float(np.median(ex / tws.mean(1))),
            "frac_beats_every_twin": float((ex[:, None] > tws).all(1).mean()),
            "frac_beats_mean_twin": float((ex > tws.mean(1)).mean()),
            "drop_top5_geomean_ratio": float(np.exp(T1[keep].mean())),
            "drop_top5_ratio_of_means": float(ex[keep].mean() / tws[keep].mean()),
            "mean_absI_EX": float(ex.mean()),
            "mean_absI_twins": float(tws.mean()),
        }
    json.dump(res, open(dst, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
