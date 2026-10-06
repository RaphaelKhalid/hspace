"""Can PCA-basis ESTIMATION ERROR fake the v5 flip-test pass? (post-hoc robustness check; Oct 6 2026)

The flip test's H0 is "Sigma-commuting in the ESTIMATED run-2 PCA basis P2". A Hessian that is a function of the TRUE Sigma
would look slightly non-commuting in P2 because P2 is rotated relative to the truth (run-1 vs run-2 Sigma^1/2: eigenvalues
agree to 1-4%, but PC6..1000 subspaces overlap only 0.67-0.88).

Simulation: take run-1's independent estimate Sigma1 as a stand-in for the truth. Rows
    y = Sh2 P1 diag(f(lam1) * r_row) P1' Sh2 z_probe
with z shared within a probe (256 probes x 64 rows), r_row iid lognormal (position diversity) and a heavy-tailed per-row
magnitude, f in {Sigma, Sigma^-1, Sigma^-1/2}. These rows are exactly Sigma1-commuting. They are then analysed with the
IDENTICAL v5 pipeline (run-2 basis P2, the same axis-aligned drop, 100 flips). If misspecification could explain the real
result, these simulations would show lambda_1 and split3 far above their flip nulls, like the data (L40 norm: lambda_1 5.1
vs 1.6, split3 0.90 vs 0.01; L52: 12.5 vs 1.9, 0.94 vs 0.02).

    python hs_flip5_misspec.py <hspace_full.pt (run 1)> <hspace2_full.pt (run 2)> 40 52
"""
import json
import sys

import torch

sys.argv, _args = sys.argv[:1], sys.argv[1:]
import hs_flip5 as F  # noqa: E402


def main(t1, t2, layers):
    torch.set_num_threads(10)
    T1 = torch.load(t1, map_location="cpu", weights_only=False)
    T2 = torch.load(t2, map_location="cpu", weights_only=False)
    out = {}
    for l in layers:
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
        g = torch.Generator().manual_seed(77 + l)
        npb, rpp = 256, 64
        probe = torch.arange(npb).repeat_interleave(rpp)
        z = torch.randn(npb, d, generator=g, dtype=torch.float64)
        A = (z @ Sh2 @ P1)[probe]                          # rows of (Sh2 z) in the P1 basis
        Mmap = P1.T @ Sh2 @ P2                             # P1 coords -> run-2 PCA coords
        R = {}
        for fname, fl in (("Sigma", lam1), ("Sigma^-1", 1 / lam1), ("Sigma^-1/2", lam1.rsqrt())):
            fl = fl / fl.max()
            mag = torch.exp(torch.randn(len(probe), 1, generator=g, dtype=torch.float64) * 1.5)
            Y = []
            for _ in range(2):
                r = torch.exp(torch.randn(len(probe), d, generator=g, dtype=torch.float64) * 0.5)
                Y.append(((mag * A * fl * r) @ Mmap).float())
            res = F.run_layer(Y[0], Y[1], probe, {"d": keep}, K, 100, 20, "cpu", 900 + l, f"L{l} misspec f={fname}")
            R[fname] = {e: {k: res["d"][e][k] for k in ("obs_lam", "null_lam1_max", "obs_split", "null_split3_max", "p_lam1", "p_split3")}
                        for e in ("norm", "xnorm")}
        out[str(l)] = R
        json.dump(out, open("out/hs_flip5_misspec.json", "w"), indent=1)


if __name__ == "__main__":
    main(_args[0], _args[1], [int(x) for x in _args[2:]] or [40, 52])
