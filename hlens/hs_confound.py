"""Massive-activation / radial confound check for H-space (exploratory, post-hoc, NOT preregistered; Oct 6 2026).

Question: is the high-curvature subspace just the known massive-activation / RMSNorm-radial machinery?
Confound span C (whitened z-space, z = Sh^-1 (x - mu)), built from the top-m |mu| coordinates E and the mean mu:
    C = span(Sh E, Sh^-1 E, Sh mu, Sh^-1 mu)      (covector and vector conventions both included)

  run1:  uses run-1 eigvecs/eigs (out/.../hspace_full.pt + hspace_full.json): share of top-64 eigen-energy inside C
  dumps: uses run-2 per-row samples (out/hspace2_full_dump_L*.pt): for raw / spatial-sign / cross-moment estimators,
         energy share inside C vs a random subspace of the same dimension, and split-half (probe parity) at
         k = 1, 2, 3, 5, 10, 25 BEFORE and AFTER projecting C out of both sides.

    python hs_confound.py run1 <hspace_full.pt> <hspace_full.json>
    python hs_confound.py dumps <hspace_full.pt> <dump_dir> [out.json]
"""
import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(8)
KS = (1, 2, 3, 5, 10, 25)


def conf_basis(mu, Sh, m):
    d = mu.shape[0]
    E = torch.eye(d, dtype=torch.float64)[:, torch.topk(mu.abs(), m).indices]
    Sih = torch.linalg.inv(Sh)
    cols = torch.cat([Sh @ E, Sih @ E, (Sh @ mu)[:, None], (Sih @ mu)[:, None]], 1)
    cols = cols / cols.norm(dim=0, keepdim=True)
    U, S, _ = torch.linalg.svd(cols, full_matrices=False)
    return U[:, S > 1e-6 * S[0]]


def top(M, k):
    ev, U = torch.linalg.eigh(M)
    o = torch.argsort(ev, descending=True)
    return ev[o], U[:, o[:k]]


def ov(A, B):
    return float((A.T @ B).pow(2).sum() / A.shape[1])


def run1(pt, js):
    T = torch.load(pt, map_location="cpu", weights_only=False)
    R = json.load(open(js))
    out = {}
    for l in sorted({int(k[2:]) for k in T if k.startswith("mu")}):
        mu, Sh, U = T[f"mu{l}"].double(), T[f"Sh{l}"].double(), T[f"Hevec64_{l}"].double()
        lam = torch.tensor(R["layers"][str(l)]["H_eigs_top64"], dtype=torch.float64).clamp_min(0)
        row = {}
        for m in (4, 8, 16):
            Qc = conf_basis(mu, Sh, m)
            f = (Qc.T @ U).pow(2).sum(0)                             # per-eigvec mass in C
            row[f"m{m}"] = {"dimC": Qc.shape[1], "mass_top10": [round(float(x), 3) for x in f[:10]],
                            "energy_share_top25_in_C": round(float((lam[:25] * f[:25]).sum() / lam[:25].sum()), 3),
                            "n_top25_with_mass_lt_0.1": int((f[:25] < 0.1).sum()),
                            "random_expectation": round(Qc.shape[1] / mu.shape[0], 4)}
        out[l] = row
        print(f"L{l}", json.dumps(row), flush=True)
    return out


def dumps(pt, ddir, outp):
    T = torch.load(pt, map_location="cpu", weights_only=False)
    res = {}
    for f in sorted(Path(ddir).glob("hspace2_full_dump_L*.pt")):
        l = int(f.stem.split("_L")[-1])
        D = torch.load(f, map_location="cpu", weights_only=False)
        mu, Sh = T[f"mu{l}"].double(), T[f"Sh{l}"].double()
        d = mu.shape[0]
        R = {}
        for kind in ("loc", "int"):
            if f"{kind}_y1" not in D:
                continue
            y1, y2, pr = D[f"{kind}_y1"].double(), D[f"{kind}_y2"].double(), D[f"{kind}_probe"]
            half = (pr % 2 == 0)

            def est(mask, var, P=None):
                a, b = y1[mask], y2[mask]
                if P is not None:
                    a, b = a @ P, b @ P
                if var == "raw":
                    return a.T @ a / len(a)
                if var == "norm":
                    u = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    return u.T @ u / len(a)
                if var == "xnorm":
                    u = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    w = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    x = u.T @ w / len(a)
                    return 0.5 * (x + x.T)
                x = a.T @ b / len(a)
                return 0.5 * (x + x.T)

            r = {}
            g = torch.Generator().manual_seed(l)
            memo = {}
            for var in ("raw", "norm", "x", "xnorm"):                # m-independent pieces, computed once
                Mf = est(torch.ones_like(half), var)
                memo[var] = (Mf, top(Mf, 25), top(est(half, var), 25)[1], top(est(~half, var), 25)[1])
            for m in (4, 16):
                Qc = conf_basis(mu, Sh, m)
                Qr = torch.linalg.qr(torch.randn(d, Qc.shape[1], generator=g, dtype=torch.float64))[0]
                P = torch.eye(d, dtype=torch.float64) - Qc @ Qc.T
                for var in ("raw", "norm", "x", "xnorm"):
                    Mf, (ev, Uf), A0, B0 = memo[var]
                    pos = ev.clamp_min(0)
                    share_C = float(torch.trace(Qc.T @ Mf @ Qc) / pos.sum())
                    share_R = float(torch.trace(Qr.T @ Mf @ Qr) / pos.sum())
                    A1, B1 = top(est(half, var, P), 25)[1], top(est(~half, var, P), 25)[1]
                    Mp = P @ Mf @ P
                    evp, Up = top(Mp, 25)
                    r[f"m{m}_{var}"] = {
                        "dimC": Qc.shape[1],
                        "energy_share_in_C": round(share_C, 3), "energy_share_in_random_same_dim": round(share_R, 4),
                        "top10_mass_in_C": [round(float(x), 3) for x in (Qc.T @ Uf[:, :10]).pow(2).sum(0)],
                        "deflated_energy_retained_top25": round(float(evp[:25].clamp_min(0).sum() / ev[:25].clamp_min(0).sum()), 3),
                        "PR_before": round(float(pos.sum() ** 2 / pos.pow(2).sum()), 2),
                        "PR_after": round(float(evp.clamp_min(0).sum() ** 2 / evp.clamp_min(0).pow(2).sum()), 2),
                        "split_before": {f"k{k}": round(ov(A0[:, :k], B0[:, :k]), 3) for k in KS},
                        "split_after": {f"k{k}": round(ov(A1[:, :k], B1[:, :k]), 3) for k in KS},
                        "overlap_after_vs_before_top10": round(ov(Up[:, :10], Uf[:, :10]), 3)}
                    x = r[f"m{m}_{var}"]
                    print(f"L{l} {kind} m{m} {var:5s}: C-share {x['energy_share_in_C']:.3f} (rand {x['energy_share_in_random_same_dim']:.4f}) "
                          f"PR {x['PR_before']}->{x['PR_after']} | split before " + " ".join(f"{k}={v}" for k, v in x["split_before"].items())
                          + " | after " + " ".join(f"{k}={v}" for k, v in x["split_after"].items()), flush=True)
            R[kind] = r
        res[l] = R
        json.dump(res, open(outp, "w"), indent=1)
    return res


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "run1":
        o = run1(sys.argv[2], sys.argv[3])
        json.dump(o, open("out/hs_confound_run1.json", "w"), indent=1)
    else:
        dumps(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "out/hs_confound_dumps_full.json")
