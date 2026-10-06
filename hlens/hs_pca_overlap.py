"""Overlap of run 2's learned top-25 subspaces (and J25) with PCA-25, in whitened coordinates (laptop, $0).

    python hs_pca_overlap.py [out/pod/hf/out/hspace2_full.pt] [out/hs_pca_overlap_run2.json]

PCA-25 = the top-25 eigenvectors of Sh = Sigma^{1/2} (the same axes as Sigma's). Overlap of a k-dim subspace Q
with PCA-25 is ||P25^T orth(Q)||_F^2 / k (mean squared cosine); a random 25-dim subspace scores 25/5120 = 0.005.
This is the source of the paper's "the spatial-sign top-25 overlaps PCA-25 by 0.78" (loc_norm, L16).
"""
import json
import sys

import torch


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "out/pod/hf/out/hspace2_full.pt"
    dst = sys.argv[2] if len(sys.argv) > 2 else "out/hs_pca_overlap_run2.json"
    T = torch.load(src, map_location="cpu", weights_only=False)
    res = {"source": src, "metric": "||P25^T orth(Q)||_F^2 / k", "random_25": 25 / 5120, "layers": {}}
    for l in (16, 28, 40, 52):
        if f"Sh{l}" not in T:
            continue
        ev, V = torch.linalg.eigh(T[f"Sh{l}"].double())
        P25 = V[:, ev.argsort(descending=True)[:25]]

        def ov(Q):
            Q = torch.linalg.qr(Q.double())[0]
            return float((P25.T @ Q).pow(2).sum() / Q.shape[1])

        row = {k: ov(v) for k, v in T[f"subs_{l}"].items()}
        row["J25w"] = ov(T[f"J25w_{l}"])
        row["run1H25w"] = ov(T[f"run1H25w_{l}"])
        res["layers"][str(l)] = row
    json.dump(res, open(dst, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
