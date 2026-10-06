"""CPU-only diagnostic of run 1's dominant H-directions (Oct 6 2026): are the top eigenvectors of the run-1 operator the
RMSNorm radial direction (whitened mean / massive-activation dims)? Reads out/hspace_full.pt (or _small); no GPU.

    python hs_diag_top.py full
"""
import json
import sys

import torch

tag = sys.argv[1] if len(sys.argv) > 1 else "small"
T = torch.load(f"out/hspace_{tag}.pt", map_location="cpu", weights_only=False)
R = json.load(open(f"out/hspace_{tag}.json"))
out = {}
for l in sorted({int(k[2:]) for k in T if k.startswith("mu")}):
    mu, Sh, U = T[f"mu{l}"].double(), T[f"Sh{l}"].double(), T[f"Hevec64_{l}"].double()
    S = Sh @ Sh
    rad = Sh @ mu; rad /= rad.norm()                       # whitened radial (mean) direction, vector convention
    radc = torch.linalg.solve(Sh, mu); radc /= radc.norm()  # covector convention
    ev, V = torch.linalg.eigh(S)
    top_var = V[:, -1]                                    # top raw-variance direction mapped through Sh: Sh v ∝ v
    big = torch.topk(mu.abs(), 4).indices                 # massive-activation coordinates
    E = torch.eye(Sh.shape[0], dtype=torch.float64)[:, big]
    Ew = torch.linalg.qr(Sh @ E)[0]
    eig = R["layers"][str(l)]["H_eigs_top64"]
    row = {"top_eigs": [round(x, 1) for x in eig[:6]], "massive_dims": big.tolist(), "mu_abs_top": [round(float(mu[i]), 1) for i in big],
           "cos2_with_radial": [round(float((U[:, i] @ rad) ** 2), 3) for i in range(6)],
           "cos2_with_radial_covector": [round(float((U[:, i] @ radc) ** 2), 3) for i in range(6)],
           "cos2_with_top_var": [round(float((U[:, i] @ top_var) ** 2), 3) for i in range(6)],
           "frac_in_massive_dims_span": [round(float((Ew.T @ U[:, i]).pow(2).sum()), 3) for i in range(6)]}
    out[l] = row
    print(l, json.dumps(row))
json.dump(out, open(f"out/hs_diag_top_{tag}.json", "w"), indent=1)
