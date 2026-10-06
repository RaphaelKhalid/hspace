"""v5 (axis-aligned confound removal; frozen before any flip computation on L40/L52). Variant d: zero PCA axes 1-5 and
the axes carrying the most massive/radial-span mass until 90% of that mass inside PCA 1..K is removed; this commutes with
the sign flips, so (unlike variant b) the deflation cannot create reproducible null patterns.

Per-probe PCA sign-flip randomization test: is there H-structure that is NOT Sigma-commuting? (Oct 6 2026)

Design from the adversarial review (wf_0874b041). Rows y_p = sum_q s_p s_q Sh H_pq Sh z (hspace2.py dumps). If the joint law
of the whitened Hessian blocks is invariant under simultaneous sign conjugation along PCA axes (H = cI, any f(Sigma),
GOE-like random curvature, the RMSNorm isotropic term: everything "Sigma-commuting"), then flipping the PCA-coordinate
signs of every row of a probe with one shared random sign vector leaves the law of the data unchanged. Off-diagonal
(non-Sigma-commuting) structure that is consistent across windows is destroyed by the flips. So the flip replicates are an
exact null for "the reproducible structure is a function of Sigma".

Per layer, variant (a none, b massive/radial span C, c C + J25w deflated), replicate r (0 = observed, 1..N flips):
  rows in PCA coords -> flip (r > 0) -> deflate -> unit-normalize full rows -> keep PCA coords 1..K (K = 1000)
  M_norm = U1'U1/n, M_xnorm = sym(U1'U2)/n, D = diag(M_norm), R = D^-1/2 M D^-1/2 (per half, own D)
  record lambda_1..5 of R_full and split-half (probe parity) of PATTERNS QR(D^1/2 V_k), k = 1, 3, 5, 10.
Power control: plant a random rank-3 component in PCA ranks 26..1000 at 0.1% of row energy per direction (same per-row
coefficient in y1 and y2), run variant b on it with N_plant flips; it must beat its own flip-null maximum.
Descriptive: raw_trim5 (raw second moment of rows below the 95% norm quantile, same normalization), Csur25 overlap.

    python hs_flip.py <hspace2_full.pt> <dump_dir> <layers...> [--n 200] [--nplant 50] [--device cpu|cuda]
    python hs_flip.py --synthetic        # size/power self-check on simulated Sigma-commuting rows
"""
import argparse
import json
import time
from pathlib import Path

import torch

KS = (1, 3, 5, 10)


def conf_basis(mu, Sh, m=4):
    d = mu.shape[0]
    E = torch.eye(d, dtype=torch.float64)[:, torch.topk(mu.abs(), m).indices]
    Sih = torch.linalg.inv(Sh)
    cols = torch.cat([Sh @ E, Sih @ E, (Sh @ mu)[:, None], (Sih @ mu)[:, None]], 1)
    cols = cols / cols.norm(dim=0, keepdim=True)
    U, S, _ = torch.linalg.svd(cols, full_matrices=False)
    return U[:, S > 1e-6 * S[0]]


def ov(A, B):
    return float((A.T @ B).pow(2).sum() / A.shape[1])


def stats(Y1, Y2, half, K, Qp, keep_raw):
    """Y1, Y2: rows in PCA coords (already flipped). Returns lambda_1..5 and pattern split-halves per estimator."""
    if Qp is not None and Qp.dim() == 1:          # axis-aligned drop mask: commutes with PCA sign flips (no null artifact)
        Y1 = Y1 * Qp
        Y2 = Y2 * Qp
    elif Qp is not None:
        Y1 = Y1 - (Y1 @ Qp) @ Qp.T
        Y2 = Y2 - (Y2 @ Qp) @ Qp.T
    cols = torch.arange(K, device=Y1.device) if (Qp is None or Qp.dim() != 1) else torch.nonzero(Qp[:K] > 0).squeeze(1)
    U1 = (Y1 / Y1.norm(dim=1, keepdim=True).clamp_min(1e-30))[:, cols]
    U2 = (Y2 / Y2.norm(dim=1, keepdim=True).clamp_min(1e-30))[:, cols]
    R1 = Y1[:, cols]
    out = {}
    Ms = {}
    for h, msk in (("A", half), ("B", ~half)):
        a, b, n = U1[msk], U2[msk], int(msk.sum())
        Mn = a.T @ a / n
        x = a.T @ b / n
        Mx = 0.5 * (x + x.T)
        r_ = R1[msk & keep_raw]
        Mr = r_.T @ r_ / max(1, len(r_))
        Ms[h] = {"norm": Mn, "xnorm": Mx, "raw_trim5": Mr, "n": n, "nr": len(r_)}
    for est in ("norm", "xnorm", "raw_trim5"):
        dkey = "raw_trim5" if est == "raw_trim5" else "norm"
        nA, nB = (Ms["A"]["nr"], Ms["B"]["nr"]) if est == "raw_trim5" else (Ms["A"]["n"], Ms["B"]["n"])
        Mf = (nA * Ms["A"][est] + nB * Ms["B"][est]) / (nA + nB)
        Df = (nA * Ms["A"][dkey] + nB * Ms["B"][dkey]) / (nA + nB)
        df = torch.diagonal(Df).clamp_min(1e-30).rsqrt()
        lam = torch.linalg.eigvalsh(df[:, None] * Mf * df[None, :]).flip(0)[:5]
        pats = {}
        for h in ("A", "B"):
            dh = torch.diagonal(Ms[h][dkey]).clamp_min(1e-30)
            Rh = dh.rsqrt()[:, None] * Ms[h][est] * dh.rsqrt()[None, :]
            ev, V = torch.linalg.eigh(Rh)
            V = V[:, torch.argsort(ev, descending=True)[:max(KS)]]
            pats[h] = torch.linalg.qr(dh.sqrt()[:, None] * V)[0]
        out[est] = {"lam": [float(x) for x in lam], "split": {f"k{k}": ov(pats["A"][:, :k], pats["B"][:, :k]) for k in KS},
                    "_pat": pats}
    return out


def run_layer(Y1, Y2, probe, Qs, K, N, nplant, dev, seed, tag, extra_desc=None):
    half = (probe % 2 == 0).to(dev)
    nrm = Y1.norm(dim=1)
    keep_raw = nrm <= torch.quantile(nrm.float().cpu(), 0.95).to(dev)
    npb = int(probe.max()) + 1
    pr = probe.to(dev)
    d = Y1.shape[1]
    res = {}
    t0 = time.time()
    for vname, Qp in Qs.items():
        obs = stats(Y1, Y2, half, K, Qp, keep_raw)
        g = torch.Generator(device=dev).manual_seed(seed)
        nulls = []
        for r in range(N):
            F = (torch.randint(0, 2, (npb, d), generator=g, device=dev).to(Y1.dtype) * 2 - 1)[pr]
            nulls.append(stats(Y1 * F, Y2 * F, half, K, Qp, keep_raw))
        V = {}
        for est in ("norm", "xnorm", "raw_trim5"):
            o = obs[est]
            nl1 = [x[est]["lam"][0] for x in nulls]
            ns3 = [x[est]["split"]["k3"] for x in nulls]
            V[est] = {"obs_lam": o["lam"], "obs_split": o["split"],
                      "null_lam1_max": max(nl1), "null_lam1_q95": sorted(nl1)[int(0.95 * (N - 1))],
                      "null_split3_max": max(ns3), "null_split_k1_max": max(x[est]["split"]["k1"] for x in nulls),
                      "null_split5_max": max(x[est]["split"]["k5"] for x in nulls),
                      "p_lam1": (1 + sum(x >= o["lam"][0] for x in nl1)) / (N + 1),
                      "p_split3": (1 + sum(x >= o["split"]["k3"] for x in ns3)) / (N + 1)}
            if extra_desc is not None:
                V[est].update(extra_desc(o["_pat"]["A"], vname, est))
        res[vname] = V
        for est in ("norm", "xnorm", "raw_trim5"):
            v = V[est]
            print(f"[{tag}] {vname:4s} {est:9s}: lam1 {v['obs_lam'][0]:.3f} (null max {v['null_lam1_max']:.3f}, p {v['p_lam1']:.3f}) | "
                  f"split k1/3/5/10 " + "/".join(f"{v['obs_split'][f'k{k}']:.2f}" for k in KS)
                  + f" (null max k3 {v['null_split3_max']:.2f}, p {v['p_split3']:.3f}) ({time.time()-t0:.0f}s)", flush=True)
    # power control on variant b (or the first variant if b is absent)
    vb = "b" if "b" in Qs else ("d" if "d" in Qs else next(iter(Qs)))
    g = torch.Generator(device=dev).manual_seed(seed + 7)
    Wp = torch.zeros(d, 3, device=dev, dtype=Y1.dtype)
    Wp[25:min(1000, d)] = torch.randn(min(1000, d) - 25, 3, generator=g, device=dev, dtype=Y1.dtype)
    Wp = torch.linalg.qr(Wp)[0]
    coef = torch.randn(Y1.shape[0], 3, generator=g, device=dev, dtype=Y1.dtype) * (0.001 ** 0.5) * nrm[:, None]
    P1, P2 = Y1 + coef @ Wp.T, Y2 + coef @ Wp.T
    obs = stats(P1, P2, half, K, Qs[vb], keep_raw)
    nulls = []
    for r in range(nplant):
        F = (torch.randint(0, 2, (npb, d), generator=g, device=dev).to(Y1.dtype) * 2 - 1)[pr]
        nulls.append(stats(P1 * F, P2 * F, half, K, Qs[vb], keep_raw))
    res["planted"] = {}
    for est in ("norm", "xnorm"):
        nl1 = max(x[est]["lam"][0] for x in nulls); ns3 = max(x[est]["split"]["k3"] for x in nulls)
        res["planted"][est] = {"variant": vb, "obs_lam1": obs[est]["lam"][0], "null_lam1_max": nl1,
                               "obs_split3": obs[est]["split"]["k3"], "null_split3_max": ns3,
                               "planted_recovered_top3": ov(Wp[:K][(Qs[vb][:K] > 0) if (Qs[vb] is not None and Qs[vb].dim() == 1) else slice(None)],
                                                            obs[est]["_pat"]["A"][:, :3]),
                               "detected": bool(obs[est]["lam"][0] > nl1)}
        print(f"[{tag}] PLANTED {est}: lam1 {obs[est]['lam'][0]:.3f} vs null max {nl1:.3f} -> "
              f"{'DETECTED' if res['planted'][est]['detected'] else 'missed'}; recovered {res['planted'][est]['planted_recovered_top3']:.2f}", flush=True)
    return res


def decide(res_by_layer):
    """Preregistered rule (runlog freeze line): variant b, both layers, both norm and xnorm: lam1 and split3 beat all
    replicates, split3 >= 0.7, planted detected."""
    ok = True
    for l, R in res_by_layer.items():
        for est in ("norm", "xnorm"):
            v = R["b"][est]
            ok &= v["obs_lam"][0] > v["null_lam1_max"] and v["obs_split"]["k3"] > v["null_split3_max"] and v["obs_split"]["k3"] >= 0.7
            ok &= R["planted"][est]["detected"]
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("t2", nargs="?"); ap.add_argument("ddir", nargs="?"); ap.add_argument("layers", nargs="*", type=int)
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--nplant", type=int, default=50)
    ap.add_argument("--K", type=int, default=1000); ap.add_argument("--device", default="cpu")
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--out", default="out/hs_flip5_full.json")
    a = ap.parse_args()
    dev = a.device
    torch.set_num_threads(12)
    if a.synthetic:
        # Sigma-commuting null: y = (diag(sigma^2) * per-row random positive diag scales) z, z shared within probe
        d, npb, rpp = 600, 128, 32
        g = torch.Generator().manual_seed(0)
        sig2 = torch.logspace(1, -3, d)
        z = torch.randn(npb, d, generator=g)
        probe = torch.arange(npb).repeat_interleave(rpp)
        heavy = torch.exp(torch.randn(npb * rpp, 1, generator=g) * 2)          # heavy-tailed row magnitudes
        Y1 = heavy * torch.exp(torch.randn(npb * rpp, d, generator=g) * 0.5) * sig2 * z[probe]
        Y2 = heavy * torch.exp(torch.randn(npb * rpp, d, generator=g) * 0.5) * sig2 * z[probe]
        res = run_layer(Y1, Y2, probe, {"a": None}, min(a.K, d), a.n, a.nplant, "cpu", 1, "synthetic-null")
        print("synthetic Sigma-commuting null: norm p_lam1", res["a"]["norm"]["p_lam1"], "p_split3", res["a"]["norm"]["p_split3"],
              "| xnorm p_lam1", res["a"]["xnorm"]["p_lam1"], "p_split3", res["a"]["xnorm"]["p_split3"])
        return
    T2 = torch.load(a.t2, map_location="cpu", weights_only=False)
    outp = Path(a.out)
    allres = json.load(open(outp)) if outp.exists() else {}
    for l in a.layers:
        f = Path(a.ddir) / f"hspace2_full_dump_L{l}.pt"
        D = torch.load(f, map_location="cpu", weights_only=False)
        Sh, mu = T2[f"Sh{l}"].double(), T2[f"mu{l}"].double()
        ev, P = torch.linalg.eigh(Sh)
        P = P[:, torch.argsort(ev, descending=True)]
        C = conf_basis(mu, Sh, 4)
        J = T2[f"J25w_{l}"].double()
        CJ = torch.linalg.qr(torch.cat([C, J], 1))[0]
        Kd = min(a.K, P.shape[0])
        wC = (P.T @ C).pow(2).sum(1)                                  # C mass per PCA axis
        drop = torch.zeros(P.shape[0], dtype=torch.bool); drop[:5] = True
        tot = float(wC[:Kd].sum()); order_ = torch.argsort(wC[:Kd], descending=True)
        for i_ in order_:
            if float(wC[:Kd][drop[:Kd]].sum()) >= 0.9 * tot:
                break
            drop[int(i_)] = True
        keep = (~drop).float().to(dev)
        print(f"L{l}: variant d drops {int(drop.sum())} PCA axes (1-5 + top C-mass axes; C mass removed in window "
              f"{float(wC[:Kd][drop[:Kd]].sum())/tot:.2f})", flush=True)
        Qs = {"d": keep}
        Pf = P.float().to(dev)
        Y1 = D["loc_y1"].float().to(dev) @ Pf
        Y2 = D["loc_y2"].float().to(dev) @ Pf
        probe = D["loc_probe"]
        loc_norm = T2[f"subs_{l}"]["loc_norm"].double() if f"subs_{l}" in T2 else None
        PC, PJ = (P.T @ C).float().to(dev), (P.T @ J).float().to(dev)
        Pl = (P.T @ loc_norm).float().to(dev) if loc_norm is not None else None

        cols_d = torch.nonzero(Qs["d"][:a.K] > 0).squeeze(1)

        def desc(pat, vname, est):
            X = pat[:, :10]
            pos = cols_d
            o = {"pat10_vs_pca25": float(X[pos < 25].pow(2).sum() / 10), "pat10_vs_pca100": float(X[pos < 100].pow(2).sum() / 10),
                 "pat10_in_C": float((PC[pos].T @ X).pow(2).sum() / 10), "pat10_in_J25": float((PJ[pos].T @ X).pow(2).sum() / 10),
                 "pat_PR_pca_coords": [round(float(1 / (X[:, i] ** 4).sum()), 1) for i in range(5)]}
            return o

        t0 = time.time()
        R = run_layer(Y1, Y2, probe, Qs, a.K, a.n, a.nplant, dev, 1000 + l, f"L{l}", desc)
        # Csur25: the 25 PCA axes with the largest diag(M_norm) -> reweighted-PCA check against loc_norm
        U1 = (Y1 / Y1.norm(dim=1, keepdim=True))
        dg = U1.pow(2).mean(0)
        idx = torch.topk(dg, 25).indices
        if Pl is not None:
            R["Csur25_vs_loc_norm"] = float(Pl[idx].pow(2).sum() / 25)
        R["seconds"] = time.time() - t0
        allres[str(l)] = R
        for est in ("norm", "xnorm", "raw_trim5"):
            for v in Qs:
                R[v][est].pop("_pat", None)
        R["n_dropped_axes"] = int(drop.sum())
        json.dump(allres, open(outp, "w"), indent=1)
        print(f"L{l}: Csur25 vs loc_norm {R.get('Csur25_vs_loc_norm')}; saved ({R['seconds']:.0f}s)", flush=True)
    done = {l: allres[l] for l in allres if l in {str(x) for x in (40, 52)}}
    if len(done) == 2:
        ok = all(R_["d"][e]["obs_lam"][0] > R_["d"][e]["null_lam1_max"] and R_["d"][e]["obs_split"]["k3"] > R_["d"][e]["null_split3_max"]
                 and R_["d"][e]["obs_split"]["k3"] >= 0.7 and R_["planted"][e]["detected"] for R_ in done.values() for e in ("norm", "xnorm"))
        print("DECISION v5 (variant d axis-aligned, L40 & L52, norm & xnorm):", "PASS: non-Sigma-commuting H-structure" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
