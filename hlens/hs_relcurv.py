"""RETRACTED (Oct 6 2026, 07:35 UTC); see results/CORRECTIONS.md in the public repo. geig() returns FILTERS M0^-1/2 V, not
patterns; the simulated null has rank <= 256 (one z per probe), so "data beats null" is guaranteed; M0 = cI is rejected by
the data and the ridge eps decides which PCA band the "excess" lands in. Kept for the record only.

Relative-curvature H-space: second-order structure BEYOND the isotropic-Hessian null (exploratory, post-hoc; Oct 6 2026).

Problem found in run 2: the whitened lens y = Sh H Sh z has a built-in PCA prior. If H = cI (no special second-order
structure at all), y = c Sigma z, so every energy estimator returns the top PCA directions of Sigma. The reproducible
run-2 estimators (loc_norm / loc_xnorm) overlap PCA-25 by ~0.78 at L16.

Fix: compare the estimator M against its exact expectation M0 under the null H = cI and look for EXCESS directions,
    M x = lambda M0 x      (generalized eigenproblem; R = M0^-1/2 M M0^-1/2 has R = c I under the null).
For the spatial-sign estimator, M0 = E[Sigma z z' Sigma / |Sigma z|^2] is diagonal in the PCA basis (sign symmetry);
it is estimated by Monte Carlo. The null is ALSO run through the identical pipeline on simulated rows Sigma z_probe
(same probe structure: one z per probe, 64 rows/probe), so split-halves and top eigenvalues have a calibrated null.

Reported per layer and ridge eps (relative to mean m0): top generalized eigenvalues (data vs null), split-half of the
top-k generalized eigenspaces (data vs null), and overlaps of the top-25 excess subspace with PCA-25 / J25 / loc_norm.
Saves the orthonormalized top-25 excess subspace (whitened frame) to out/relcurv_subs.pt for a causal test.

    python hs_relcurv.py <hspace2_full.pt> <dump_dir> [layers ...]
"""
import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(8)
KS = (1, 3, 5, 10, 25)
EPS = (0.0, 0.01, 0.1)


def ov(A, B):
    return float((A.T @ B).pow(2).sum() / A.shape[1])


def geig(M, m0, eps, k=25):
    """top-k generalized eigvecs of M x = lam diag(m0 + eps*mean) x, in the (PCA) basis of M; returns lam, W (orthonormal span)."""
    dd = (m0 + eps * m0.mean()).rsqrt()
    R = dd[:, None] * M * dd[None, :]
    lam, V = torch.linalg.eigh(R)
    o = torch.argsort(lam, descending=True)
    X = dd[:, None] * V[:, o[:k]]                          # generalized eigvecs (not orthonormal)
    return lam[o], torch.linalg.qr(X)[0], V[:, o[:k]]


def main(t2p, ddir, layers):
    T2 = torch.load(t2p, map_location="cpu", weights_only=False)
    outj, outs = Path("out/relcurv_full.json"), Path("out/relcurv_subs.pt")
    res = json.load(open(outj)) if outj.exists() else {}            # merge with layers already done
    subs = torch.load(outs, map_location="cpu", weights_only=False) if outs.exists() else {}
    for l in layers:
        f = Path(ddir) / f"hspace2_full_dump_L{l}.pt"
        if not f.exists():
            print(f"L{l}: no dump", flush=True); continue
        D = torch.load(f, map_location="cpu", weights_only=False)
        Sh = T2[f"Sh{l}"].double()
        s, U = torch.linalg.eigh(Sh)
        o = torch.argsort(s, descending=True)
        s, P = s[o], U[:, o]                              # Sh eigvals (Sigma^1/2), PCA basis (whitened frame)
        sig2 = s ** 2                                     # Sigma eigenvalues
        d = len(s)
        g = torch.Generator().manual_seed(1000 + l)
        m0 = torch.zeros(d, dtype=torch.float64)          # spatial-sign null, MC in the PCA basis
        nmc = 0
        for _ in range(20):
            z = torch.randn(4096, d, generator=g, dtype=torch.float64)
            y0 = z * sig2
            u0 = y0 / y0.norm(dim=1, keepdim=True)
            m0 += u0.pow(2).sum(0); nmc += len(u0)
        m0 /= nmc
        y1 = D["loc_y1"].double() @ P
        y2 = D["loc_y2"].double() @ P
        pr = D["loc_probe"]
        half = (pr % 2 == 0)
        npb = int(pr.max()) + 1
        zp = torch.randn(npb, d, generator=g, dtype=torch.float64) * sig2       # null rows: Sigma z_probe, same structure
        n0 = zp[pr]
        u = {"norm": (y1 / y1.norm(dim=1, keepdim=True), None),
             "xnorm": (y1 / y1.norm(dim=1, keepdim=True), y2 / y2.norm(dim=1, keepdim=True)),
             "null": (n0 / n0.norm(dim=1, keepdim=True), None)}

        def est(var, mask):
            a, b = u[var]
            a = a[mask]
            if b is None:
                return a.T @ a / len(a)
            x = a.T @ b[mask] / len(a)
            return 0.5 * (x + x.T)

        allm = torch.ones_like(half)
        Pw = P                                             # PCA basis -> whitened frame
        loc_norm = T2[f"subs_{l}"]["loc_norm"].double() if f"subs_{l}" in T2 else None
        J = torch.linalg.qr(T2[f"J25w_{l}"].double())[0] if f"J25w_{l}" in T2 else None
        L = {"m0_top5": [float(x) for x in m0[:5]], "m0_PR": float(m0.sum() ** 2 / m0.pow(2).sum())}
        mu = T2[f"mu{l}"].double()
        Sih = torch.linalg.inv(Sh)
        Cq = {}
        for mm in (4, 16):                                 # massive-activation / radial confound span (hs_confound.py)
            E = torch.eye(d, dtype=torch.float64)[:, torch.topk(mu.abs(), mm).indices]
            cols = torch.cat([Sh @ E, Sih @ E, (Sh @ mu)[:, None], (Sih @ mu)[:, None]], 1)
            cols = cols / cols.norm(dim=0, keepdim=True)
            Uc, Sc, _ = torch.linalg.svd(cols, full_matrices=False)
            Cq[mm] = Uc[:, Sc > 1e-6 * Sc[0]]
        for var in ("norm", "xnorm", "null"):
            Mf, Ma, Mb = est(var, allm), est(var, half), est(var, ~half)
            for eps in EPS:
                lam, Wf, _ = geig(Mf, m0, eps)
                _, Wa, _ = geig(Ma, m0, eps)
                _, Wb, _ = geig(Mb, m0, eps)
                Ww = Pw @ Wf                               # whitened-frame orthonormal excess subspace
                r = {"lam_top10": [round(float(x), 2) for x in lam[:10]], "lam_median": round(float(lam.median()), 3),
                     "split": {f"k{k}": round(ov(Wa[:, :k], Wb[:, :k]), 3) for k in KS},
                     "vs_pca25": round(ov(Wf, torch.eye(d, dtype=torch.float64)[:, :25]), 3),
                     "vs_pca100": round(ov(Wf, torch.eye(d, dtype=torch.float64)[:, :100]), 3),
                     "mean_pca_rank_of_mass": round(float(((Wf.pow(2).sum(1)) * torch.arange(d, dtype=torch.float64)).sum() / 25), 1)}
                for mm, Q_ in Cq.items():
                    r[f"mass_in_C{mm}_top25"] = round(float((Q_.T @ Ww).pow(2).sum() / 25), 3)
                    r[f"mass_in_C{mm}_top5"] = round(float((Q_.T @ Ww[:, :5]).pow(2).sum() / 5), 3)
                big = torch.topk(mu.abs(), 16).indices
                r["raw_mass_on_top16_mu_dims_top5"] = round(float(((Sh @ Ww[:, :5]) / (Sh @ Ww[:, :5]).norm(dim=0, keepdim=True))[big].pow(2).sum() / 5), 3)
                if loc_norm is not None:
                    r["vs_loc_norm25"] = round(ov(Ww, loc_norm), 3)
                if J is not None:
                    r["vs_J25"] = round(ov(Ww, J), 3)
                L[f"{var}_eps{eps}"] = r
                print(f"L{l} {var:5s} eps={eps:<5}: lam {r['lam_top10'][:5]} med {r['lam_median']} | split "
                      + " ".join(f"{k_}={v_}" for k_, v_ in r["split"].items())
                      + f" | vsPCA25 {r['vs_pca25']} vsPCA100 {r['vs_pca100']} meanPCArank {r['mean_pca_rank_of_mass']}"
                      + (f" vsJ25 {r.get('vs_J25')} vs_loc_norm {r.get('vs_loc_norm25')}")
                      + f" | C4 {r['mass_in_C4_top25']} C16 {r['mass_in_C16_top25']} (top5 {r['mass_in_C16_top5']}) rawmass@mu16 {r['raw_mass_on_top16_mu_dims_top5']}", flush=True)
                if var in ("norm", "xnorm") and eps == 0.1:
                    subs[f"rel25_{var}_{l}"] = Ww.float()
        res[str(l)] = L
        json.dump(res, open(outj, "w"), indent=1)
        torch.save(subs, outs.with_name(outs.name + ".tmp")); outs.with_name(outs.name + ".tmp").replace(outs)
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]] or [16, 28, 40, 52])
