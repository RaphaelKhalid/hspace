"""Offline, unbiased MEAN-Hessian lens (M^lens) from the existing 27B HVP dumps. CPU only, $0. Exploratory (not preregistered).

theory-v2 section 0 defines the H-lens as the context-AVERAGED Hessian T[c] = E_x mean_p H_x[c]_pp (the J-lens one
derivative up) and its energy M^lens = E_c T~[c]^2. Every estimator run so far converges to a second moment over
contexts (M^row, M^diag, spatial-sign shapes; Prop. 6), which the context variance and heavy tails dominate; run 1's
"M_lens" carried a Cov_x/B bias with B = 4 (Prop. 6f). Nobody has estimated M^lens without that bias.

Within one probe (covector c and direction z shared), rows from DIFFERENT windows are independent contexts, and
E[y | x, c, z] = H~_pp(x) z. So with Ybar_kw = mean over probe k's rows in window w of (y1 + y2)/2:
    M^lens_hat = sum_k [S_k S_k^T - sum_w Ybar_kw Ybar_kw^T] / sum_k n_k(n_k - 1),   S_k = sum_w Ybar_kw
is unbiased for E_c T~[c] z z^T T~[c] = M^lens (z ~ N(0, I)). The between-window products remove the context variance.
Reported per layer:
  phi        = tr M^lens / tr M^diag: the share of whitened curvature energy that is context-invariant (the J-like part),
               with a probe-bootstrap CI. M^diag_hat = mean over rows of sym(y1 y2^T) (unbiased, Prop. 6b).
  offdiag    = ||offdiag_P(M^lens)||_F^2 / ||M^lens||_F^2 in the PCA basis P, against the exact per-probe PCA sign-flip null
               (Cor. 8.1): flips leave any Sigma-commuting law invariant, so beating them = non-Sigma mean curvature.
  lam1_R     = top eigenvalue of R = D^-1/2 M^lens D^-1/2 (D = diag), vs flips.
  split_k    = top-k eigenspace overlap of M^lens between probe halves (parity), k = 1, 3, 5, vs flips.
  overlap of M^lens top-5 with v11a H_all, the v9 punctuation H-space (EX), and J25 (all in whitened coordinates).
Preprocessing: v5 variant d axis-aligned drop (PCA axes 1-5 + axes with 90% of the massive/radial span mass), which
commutes with the flips.
    python hs_mlens_offline.py [--n 200] [--layers 16 28 40 52] [--v8]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from hs_flip5 import conf_basis

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RES = HERE.parent / "results" / "mlens"


NODROP = False


def axis_drop(T2, l, K=1000):
    if NODROP:
        Sh = T2[f"Sh{l}"].double()
        ev, P = torch.linalg.eigh(Sh)
        return P[:, torch.argsort(ev, descending=True)], torch.ones(P.shape[0], dtype=torch.bool)
    return _axis_drop(T2, l, K)


def _axis_drop(T2, l, K=1000):
    Sh, mu = T2[f"Sh{l}"].double(), T2[f"mu{l}"].double()
    ev, P = torch.linalg.eigh(Sh)
    P = P[:, torch.argsort(ev, descending=True)]
    C = conf_basis(mu, Sh, 4)
    wC = (P.T @ C).pow(2).sum(1)
    Kd = min(K, P.shape[0])
    drop = torch.zeros(P.shape[0], dtype=torch.bool); drop[:5] = True
    tot = float(wC[:Kd].sum())
    for i_ in torch.argsort(wC[:Kd], descending=True):
        if float(wC[:Kd][drop[:Kd]].sum()) >= 0.9 * tot:
            break
        drop[int(i_)] = True
    return P, ~drop


def group_means(Y, probe, win):
    """-> list over probes of [n_w, d] window means (probes with >= 2 windows only)."""
    out = []
    key = probe * 100000 + win
    for k in torch.unique(probe):
        mk = probe == k
        ws = torch.unique(win[mk])
        if len(ws) < 2:
            continue
        out.append(torch.stack([Y[mk & (win == w)].mean(0) for w in ws]))
    return out


def mlens_parts(G, flips=None):
    """G: list of [n_w, d]. Returns A (S_k rows), B (all window means), N (sum n(n-1)); flips: list of [d] sign vectors."""
    A, B, N = [], [], 0
    for k, g in enumerate(G):
        if flips is not None:
            g = g * flips[k]
        A.append(g.sum(0)); B.append(g); N += g.shape[0] * (g.shape[0] - 1)
    return torch.stack(A), torch.cat(B), N


def matvec(A, B, N, v):
    return (A.T @ (A @ v) - B.T @ (B @ v)) / N


def top_eig(A, B, N, k=5, iters=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(A.shape[1], k, generator=g, dtype=A.dtype))[0]
    for _ in range(iters):
        Q = torch.linalg.qr(matvec(A, B, N, Q))[0]
    Tm = Q.T @ matvec(A, B, N, Q)
    ev, U = torch.linalg.eigh(Tm)
    o = torch.argsort(ev, descending=True)
    return ev[o], Q @ U[:, o]


def full_M(A, B, N):
    return (A.T @ A - B.T @ B) / N


def ov(Qa, Qb):
    return float((Qa.T @ Qb).pow(2).sum() / Qa.shape[1])


def op_eigs(A, B, N, k, which="LA", scale=None):
    """Top-k (algebraic) eigenpairs of M = (A^T A - B^T B)/N, or of R = S M S with S = diag(scale), matrix-free."""
    from scipy.sparse.linalg import LinearOperator, eigsh
    An, Bn = A.numpy(), B.numpy()
    s = None if scale is None else scale.numpy()

    def mv(v):
        v = v.ravel() if s is None else s * v.ravel()
        w = (An.T @ (An @ v) - Bn.T @ (Bn @ v)) / N
        return w if s is None else s * w
    d = A.shape[1]
    ev, V = eigsh(LinearOperator((d, d), matvec=mv, dtype=np.float64), k=k, which=which, tol=1e-6, maxiter=5000)
    o = np.argsort(-ev)
    return ev[o], V[:, o]


def fro2(A, B, N):
    """||(A^T A - B^T B)/N||_F^2 via small Gram matrices."""
    return float(((A @ A.T).pow(2).sum() + (B @ B.T).pow(2).sum() - 2 * (A @ B.T).pow(2).sum()) / N ** 2)


def stats(G, flips=None, half=None):
    A, B, N = mlens_parts(G, flips)
    dg = (A.pow(2).sum(0) - B.pow(2).sum(0)) / N
    F2 = fro2(A, B, N)
    off = float((F2 - float(dg.pow(2).sum())) / F2)
    Dm = dg.abs().clamp_min(1e-12 * float(dg.abs().max())).rsqrt()
    lam1_R = float(op_eigs(A, B, N, 1, scale=Dm)[0][0])
    out = {"tr": float(dg.sum()), "fro2": F2, "offdiag": off, "lam1_R": lam1_R}
    if half is not None:
        Vs = []
        for h in (0, 1):
            Gh = [g for k, g in enumerate(G) if half[k] == h]
            fh = None if flips is None else [f for k, f in enumerate(flips) if half[k] == h]
            Ah, Bh, Nh = mlens_parts(Gh, fh)
            Vs.append(torch.as_tensor(op_eigs(Ah, Bh, Nh, 5)[1]))
        for kk in (1, 3, 5):
            out[f"split{kk}"] = ov(Vs[0][:, :kk], Vs[1][:, :kk])
    return out, (A, B, N)


def run(name, Y1, Y2, probe, win, tok, T2, l, refs, n_flip, seed):
    t0 = time.time()
    P, keep = axis_drop(T2, l)
    Pk = P[:, keep]                                                          # whitened coords -> kept PCA coords
    Y = ((Y1 + Y2) / 2).double() @ Pk
    Yd1, Yd2 = Y1.double() @ Pk, Y2.double() @ Pk
    G = group_means(Y, probe, win)
    nprobe = len(G)
    # M^diag trace (unbiased cross-moment, Prop. 6b) and phi with a probe bootstrap
    trd_rows = (Yd1 * Yd2).sum(1)
    pr_ids = torch.unique(probe)
    tr_diag_k = torch.stack([trd_rows[probe == k].mean() for k in pr_ids]).numpy()
    tr_lens_k = []
    for g in G:
        s = g.sum(0); n = g.shape[0]
        tr_lens_k.append(float((s @ s - (g * g).sum()) / (n * (n - 1))))
    tr_lens_k = np.array(tr_lens_k)
    rng = np.random.default_rng(seed)
    phi = float(tr_lens_k.mean() / tr_diag_k.mean())
    bs = []
    for _ in range(2000):
        a = rng.integers(0, len(tr_lens_k), len(tr_lens_k)); b = rng.integers(0, len(tr_diag_k), len(tr_diag_k))
        bs.append(tr_lens_k[a].mean() / tr_diag_k[b].mean())
    phi_ci = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    # per token class (digits / punctuation / other) phi, descriptive
    half = [k % 2 for k in range(nprobe)]
    obs, (A, B, N) = stats(G, None, half)
    ev, V = op_eigs(A, B, N, 10)
    ev_min = float(op_eigs(A, B, N, 1, which="SA")[0][0])
    ev, V = torch.as_tensor(ev), torch.as_tensor(V)
    pr = float(obs["tr"] ** 2 / obs["fro2"])                                  # (sum lambda)^2 / sum lambda^2
    top5_w = Pk @ V[:, :5]                                                    # back to whitened coordinates
    ovl = {k: ov(top5_w, torch.linalg.qr(R_.double())[0]) for k, R_ in refs.items() if R_ is not None}
    # flip null (Cor. 8.1): one random sign vector in PCA coords per probe
    g = torch.Generator().manual_seed(seed)
    null = {"offdiag": [], "lam1_R": [], "split1": [], "split3": [], "split5": []}
    for r in range(n_flip):
        fl = [torch.randint(0, 2, (Pk.shape[1],), generator=g).double() * 2 - 1 for _ in range(nprobe)]
        st, _ = stats(G, fl, half)
        for kk in null:
            null[kk].append(st[kk])
    pv = {kk: float((1 + sum(x >= obs[kk] for x in null[kk])) / (n_flip + 1)) for kk in null}
    res = {"name": name, "layer": l, "n_probes": nprobe, "rows": int(Y.shape[0]), "kept_axes": int(keep.sum()),
           "phi_lens_over_diag": phi, "phi_ci95": phi_ci, "tr_lens": float(tr_lens_k.mean()), "tr_diag": float(tr_diag_k.mean()),
           "observed": obs, "null_max": {kk: float(max(v)) for kk, v in null.items()},
           "null_median": {kk: float(np.median(v)) for kk, v in null.items()}, "p_flip": pv,
           "eig_top10": [float(x) for x in ev[:10]], "eig_min": ev_min,
           "participation_ratio": pr, "top5_overlap": ovl, "minutes": (time.time() - t0) / 60}
    print(f"{name} L{l}: probes {nprobe} | phi = tr M^lens / tr M^diag = {phi:.4f} CI {[round(x, 4) for x in phi_ci]} | "
          f"offdiag {obs['offdiag']:.3f} (flip max {res['null_max']['offdiag']:.3f}, p {pv['offdiag']:.3f}) | lam1_R {obs['lam1_R']:.2f} "
          f"(max {res['null_max']['lam1_R']:.2f}, p {pv['lam1_R']:.3f}) | split1/3/5 {obs['split1']:.2f}/{obs['split3']:.2f}/{obs['split5']:.2f} "
          f"(null max {res['null_max']['split1']:.2f}/{res['null_max']['split3']:.2f}/{res['null_max']['split5']:.2f}) | PR {pr:.1f} | "
          f"lam1 {float(ev[0]):.3e} lam_min {ev_min:.3e} | top5 overlap {ovl} | {res['minutes']:.1f} min", flush=True)
    return res, top5_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--layers", type=int, nargs="*", default=[16, 28, 40, 52])
    ap.add_argument("--v8", action="store_true")
    ap.add_argument("--nodrop", action="store_true")
    a = ap.parse_args()
    global NODROP, RES
    NODROP = a.nodrop
    if a.nodrop:
        RES = RES.parent / "mlens_nodrop"
    torch.set_num_threads(12)
    RES.mkdir(parents=True, exist_ok=True)
    T2 = torch.load(OUT / "pod/hf/out/hspace2_full.pt", map_location="cpu", weights_only=False)
    SUB = torch.load(OUT / "v11a_subspaces.pt", map_location="cpu", weights_only=False)
    V9 = torch.load(OUT / "v9_subspaces.pt", map_location="cpu", weights_only=False) if (OUT / "v9_subspaces.pt").exists() else {}
    allres, tops = {}, {}
    jobs = [("run2", l, OUT / f"pod/hf/out/hspace2_full_dump_L{l}.pt") for l in a.layers]
    if a.v8:
        jobs += [("v8", l, OUT / f"pod/hf/v8/out/hs_v8_full_L{l}.pt") for l in (16, 40)]
    for name, l, f in jobs:
        D = torch.load(f, map_location="cpu", weights_only=False)
        win = D["loc_win"] if "loc_win" in D else D["win"]
        refs = {"H_all_v11a": SUB.get(f"H_all_{l}"), "EX_v9": V9.get(f"EX_{l}"), "J25w": T2.get(f"J25w_{l}")}
        r, top = run(name, D["loc_y1"].float(), D["loc_y2"].float(), D["loc_probe"], win, D["loc_tok"], T2, l, refs, a.n, 1000 + l)
        allres[f"{name}_L{l}"] = r; tops[f"{name}_L{l}"] = top
        json.dump(allres, open(RES / "mlens_offline.json", "w"), indent=1)
        del D
    for l in (16, 40):
        if f"run2_L{l}" in tops and f"v8_L{l}" in tops:
            allres[f"cross_run_top5_overlap_L{l}"] = ov(torch.linalg.qr(tops[f"run2_L{l}"])[0], torch.linalg.qr(tops[f"v8_L{l}"])[0])
            print(f"L{l}: M^lens top-5 overlap run2 vs v8 = {allres[f'cross_run_top5_overlap_L{l}']:.3f}")
    json.dump(allres, open(RES / "mlens_offline.json", "w"), indent=1)
    torch.save({k: v.float() for k, v in tops.items()}, OUT / "mlens_top5.pt")


if __name__ == "__main__":
    main()
