"""Matched-flatness misspecification null for the v5 flip test (post-hoc robustness; Oct 6 2026).

hs_flip5_misspec.py showed that Sigma1-commuting rows seen through the run-2 basis pass v5 when raw curvature grows with
variance (f = Sigma), but not for flatter f (Sigma^-1/2, Sigma^-1). Which regime is the real Hessian in?
Calibrate f = Sigma^a on a grid so the simulated spatial-sign spectrum has the SAME participation ratio as the data's
loc_norm operator at that layer (hspace2_full.json), then run the identical v5 pipeline (100 flips) on the two grid points
that bracket the data's PR. Rule written before running: the v5 pass survives basis misspecification at a layer iff, at both
bracketing a values, split3 of the simulation (norm and xnorm) is < 0.5 (the data's is 0.82-0.94).

    python hs_flip5_misspec2.py <hspace_full.pt> <hspace2_full.pt> <hspace2_full.json> 40 52
"""
import json
import sys

import torch

sys.argv, _a = sys.argv[:1], sys.argv[1:]
import hs_flip5 as F  # noqa: E402

GRID = (1.0, 0.5, 0.25, 0.0, -0.25, -0.5, -1.0)


def rows(A, P1, Mmap, fl, probe, g):
    mag = torch.exp(torch.randn(len(probe), 1, generator=g, dtype=torch.float64) * 1.5)
    out = []
    for _ in range(2):
        r = torch.exp(torch.randn(len(probe), A.shape[1], generator=g, dtype=torch.float64) * 0.5)
        out.append(((mag * A * fl * r) @ Mmap).float())
    return out


def pr_norm(Y):
    U = Y / Y.norm(dim=1, keepdim=True)
    ev = torch.linalg.eigvalsh((U.T @ U / len(U)).double()).clamp_min(0)
    return float(ev.sum() ** 2 / ev.pow(2).sum())


def main(t1, t2, j2, layers):
    torch.set_num_threads(10)
    T1 = torch.load(t1, map_location="cpu", weights_only=False)
    T2 = torch.load(t2, map_location="cpu", weights_only=False)
    R2 = json.load(open(j2))
    out = {}
    for l in layers:
        target = R2["layers"][str(l)]["loc_norm"]["PR"]
        Sh1, Sh2, mu = T1[f"Sh{l}"].double(), T2[f"Sh{l}"].double(), T2[f"mu{l}"].double()
        e1, P1 = torch.linalg.eigh(Sh1); o1 = torch.argsort(e1, descending=True); e1, P1 = e1[o1], P1[:, o1]
        e2, P2 = torch.linalg.eigh(Sh2); P2 = P2[:, torch.argsort(e2, descending=True)]
        lam1 = e1 ** 2
        d, K = Sh2.shape[0], 1000
        C = F.conf_basis(mu, Sh2, 4)
        wC = (P2.T @ C).pow(2).sum(1)
        drop = torch.zeros(d, dtype=torch.bool); drop[:5] = True
        tot = float(wC[:K].sum())
        for i_ in torch.argsort(wC[:K], descending=True):
            if float(wC[:K][drop[:K]].sum()) >= 0.9 * tot:
                break
            drop[int(i_)] = True
        keep = (~drop).float()
        npb, rpp = 256, 64
        probe = torch.arange(npb).repeat_interleave(rpp)
        g = torch.Generator().manual_seed(500 + l)
        z = torch.randn(npb, d, generator=g, dtype=torch.float64)
        A = (z @ Sh2 @ P1)[probe]
        Mmap = P1.T @ Sh2 @ P2
        prs = {}
        for a in GRID:
            fl = lam1 ** a; fl = fl / fl.max()
            Y1, _ = rows(A, P1, Mmap, fl, probe, torch.Generator().manual_seed(600 + l))
            prs[a] = pr_norm(Y1.double())
        print(f"L{l}: data loc_norm PR {target:.1f}; simulated PR by a: " + ", ".join(f"{a:+.2f}:{p:.1f}" for a, p in prs.items()), flush=True)
        above = [a for a in GRID if prs[a] >= target]
        below = [a for a in GRID if prs[a] < target]
        bracket = sorted({min(above, key=lambda a: prs[a] - target) if above else GRID[-1],
                          max(below, key=lambda a: prs[a] - target) if below else GRID[0]})
        R = {"data_PR": target, "sim_PR": {str(a): p for a, p in prs.items()}, "bracket": bracket, "runs": {}}
        for a in bracket:
            fl = lam1 ** a; fl = fl / fl.max()
            Y1, Y2 = rows(A, P1, Mmap, fl, probe, torch.Generator().manual_seed(700 + l))
            res = F.run_layer(Y1, Y2, probe, {"d": keep}, K, 100, 20, "cpu", 800 + l, f"L{l} matched a={a:+.2f}")
            R["runs"][str(a)] = {e: {k: res["d"][e][k] for k in ("obs_lam", "null_lam1_max", "obs_split", "null_split3_max")}
                                 for e in ("norm", "xnorm")}
        R["v5_survives_misspec"] = all(R["runs"][str(a)][e]["obs_split"]["k3"] < 0.5 for a in bracket for e in ("norm", "xnorm"))
        print(f"L{l}: v5 pass survives matched-flatness misspecification: {R['v5_survives_misspec']}", flush=True)
        out[str(l)] = R
        json.dump(out, open("out/hs_flip5_misspec2.json", "w"), indent=1)


if __name__ == "__main__":
    main(_a[0], _a[1], _a[2], [int(x) for x in _a[3:]] or [40, 52])
