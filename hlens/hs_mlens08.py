"""The MEAN H-lens on Qwen3.5-0.8B (CPU, $0, exploratory, not preregistered): estimate it, test it, decode it.

J-lens recipe, one derivative up. hs_phi_cpu.py found phi_H = 0.03-0.11 at 0.8B (CI above 0): the context-averaged
Hessian exists there. This script builds its top eigenspace and asks whether it is stable and readable.
 Stage A (estimate): P probes x W windows. Per probe: c ~ N(0, Sigma_U), z ~ N(0, I); per window one HVP with
   Rademacher position signs (two sign vectors); rows y_p = s_p Sigma^1/2 (H (s x Sigma^1/2 z))_p, p in [16, T-2].
   M^lens_hat = sum_k [S_k S_k^T - sum_w Ybar_kw Ybar_kw^T] / sum_k n_k(n_k-1)   (unbiased, whitened)
 Stage B (test): phi; top-5 eigenspace split-half (probe parity) vs the exact per-probe PCA sign-flip null (Cor. 8.1);
   overlap with PCA top-5/25 and with the 0.8B J25 (v10_inputs_small J25w_l when l is 6 or 12; else skipped).
 Stage C (decode): u_k = Sigma^1/2 v_k (1 sigma) for the top-5 eigenvectors v_k and for 4 Sigma-orbit twins of them.
   On N fresh windows at one random position q: r_ij = h_L(x+u_i+u_j) - h_L(x+u_i) - h_L(x+u_j) + h_L(x) at q,
   m_i = h_L(x+u_i) - h_L(x) at q (pre-final-norm, / rms(h_L(x)_q)). Report the invariant fraction F of r per arm
   (unbiased between-window), and decode mean r_ij and mean m_i through gain * W_U (top / bottom 12 tokens).
    python hs_mlens08.py [--layer 18] [--probes 256] [--windows 4] [--ndec 48] [--flips 200]
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from hs_phi_cpu import Readout, load, windows, SKIP

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "mlens"
PROG = HERE / "out" / "progress" / "mlens08.json"


def progress(stage, idx, done, total, note=""):
    try:
        PROG.parent.mkdir(parents=True, exist_ok=True)
        PROG.write_text(json.dumps({"job": "mlens08", "title": "Mean H-lens 0.8B (CPU)", "status": "running" if stage != "done" else "done",
                                    "stage": stage, "stage_index": idx, "stage_count": 3, "done": done, "total": total, "unit": "steps",
                                    "started": T0, "updated": time.time(), "note": note}))
    except OSError:
        pass


T0 = time.time()


def between_parts(G):
    A = torch.stack([g.sum(0) for g in G]); B = torch.cat(G); N = sum(g.shape[0] * (g.shape[0] - 1) for g in G)
    return A, B, N


def M_of(G):
    A, B, N = between_parts(G)
    return (A.T @ A - B.T @ B) / N


def topk(M, k):
    ev, V = torch.linalg.eigh(M)
    o = torch.argsort(ev, descending=True)
    return ev[o], V[:, o[:k]]


def ov(A, B):
    return float((A.T @ B).pow(2).sum() / A.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--probes", type=int, default=256)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--ndec", type=int, default=48)
    ap.add_argument("--flips", type=int, default=200)
    ap.add_argument("--T", type=int, default=64)
    a = ap.parse_args()
    torch.set_num_threads(12)
    RES.mkdir(parents=True, exist_ok=True)
    m, tok = load("Qwen/Qwen3.5-0.8B")
    d = m.config.get_text_config().hidden_size
    l, T = a.layer, a.T
    W_U = m.lm_head.weight.float(); gain = 1.0 + m.model.norm.weight.float()
    Ug = W_U * gain
    SigU = (Ug.T @ Ug) / Ug.shape[0]
    LU = torch.linalg.cholesky(SigU + 1e-8 * torch.eye(d) * SigU.diagonal().mean())
    R = Readout(m, l)
    cov = windows(tok, "train", 48, T, seed=5)
    hs = []
    with torch.no_grad():
        for w in cov:
            o = m.model(input_ids=w[None], output_hidden_states=True, use_cache=False)
            hs.append(o.hidden_states[l + 1][0, SKIP:T - 1])
    X = torch.cat(hs).double()
    S = torch.cov(X.T); S = 0.95 * S + 0.05 * torch.trace(S) / d * torch.eye(d, dtype=S.dtype)
    evS, PS = torch.linalg.eigh(S); o = torch.argsort(evS, descending=True); evS, PS = evS[o], PS[:, o]
    Sh = (PS * evS.clamp_min(1e-12).sqrt()) @ PS.T; Shf = Sh.float()
    ids_all = windows(tok, "train", a.probes * a.windows + a.ndec, T, seed=7)
    assert ids_all.shape[0] == a.probes * a.windows + a.ndec, ids_all.shape
    ev_ids, dec_ids = ids_all[:a.probes * a.windows], ids_all[a.probes * a.windows:]
    out = {"layer": l, "probes": a.probes, "windows": a.windows, "T": T}

    # ---------------- Stage A: estimate
    g = torch.Generator().manual_seed(500 + l)
    G, rowE = [], []
    for k in range(a.probes):
        c = LU @ torch.randn(d, generator=g); z = torch.randn(d, generator=g); v = Shf @ z
        gw = []
        for w_i in range(a.windows):
            ids = ev_ids[k * a.windows + w_i]
            delta = torch.zeros(1, T, d, requires_grad=True)
            (gr,) = torch.autograd.grad(R.F(ids, c, delta), delta, create_graph=True)
            ys = []
            for rep in range(2):
                s = (torch.randint(0, 2, (T, 1), generator=g).float() * 2 - 1); s[:SKIP] = 0; s[T - 1:] = 0
                (hv,) = torch.autograd.grad(gr, delta, grad_outputs=(s * v)[None], retain_graph=(rep == 0))
                ys.append(((s * hv[0])[SKIP:T - 1].detach() @ Shf).double())
            rowE.append(float((ys[0] * ys[1]).sum(1).mean()))
            gw.append(((ys[0] + ys[1]) / 2).mean(0))
        G.append(torch.stack(gw))
        if (k + 1) % 8 == 0:
            progress("A estimate mean H-lens", 1, k + 1, a.probes, f"{(time.time()-T0)/60:.0f} min")
            print(f"A: probe {k+1}/{a.probes} ({(time.time()-T0)/60:.1f} min)", flush=True)
    M = M_of(G)
    phi = float(torch.trace(M) / np.mean(rowE))
    ev, V5 = topk(M, 5)
    out["phi"] = phi
    out["eig_top10"] = [float(x) for x in ev[:10]]
    out["eig_bottom3"] = [float(x) for x in ev[-3:]]
    print(f"A: phi = {phi:.4f} | top eig {[round(float(x), 4) for x in ev[:5]]} | bottom {[round(float(x), 4) for x in ev[-3:]]}", flush=True)

    # ---------------- Stage B: test (split-half vs per-probe PCA sign-flip null; overlaps)
    half = [k % 2 for k in range(a.probes)]

    def split(Gs):
        Va = topk(M_of([x for x, h in zip(Gs, half) if h == 0]), 5)[1]
        Vb = topk(M_of([x for x, h in zip(Gs, half) if h == 1]), 5)[1]
        return {kk: ov(Va[:, :kk], Vb[:, :kk]) for kk in (1, 3, 5)}
    obs = split(G)
    Gp = [x @ PS for x in G]                                                 # PCA coordinates for the flips
    gf = torch.Generator().manual_seed(9)
    null = {kk: [] for kk in (1, 3, 5)}
    for r in range(a.flips):
        fl = [x * (torch.randint(0, 2, (d,), generator=gf).double() * 2 - 1) for x in Gp]
        sp = split(fl)
        for kk in null:
            null[kk].append(sp[kk])
        if (r + 1) % 25 == 0:
            progress("B flip null", 2, r + 1, a.flips)
    out["split"] = obs
    out["split_null_max"] = {kk: float(max(v)) for kk, v in null.items()}
    out["split_p"] = {kk: float((1 + sum(x >= obs[kk] for x in null[kk])) / (a.flips + 1)) for kk in null}
    out["overlap_pca5"] = ov(V5, PS[:, :5]); out["overlap_pca25"] = ov(V5, PS[:, :25])
    try:
        INP = torch.load(HERE / "out" / "v10_inputs_small.pt", map_location="cpu", weights_only=False)
        if f"J25w_{l}" in INP:
            out["overlap_J25"] = ov(V5, torch.linalg.qr(INP[f"J25w_{l}"].double())[0])
    except OSError:
        pass
    print(f"B: split1/3/5 {obs[1]:.2f}/{obs[3]:.2f}/{obs[5]:.2f} vs flip max {out['split_null_max'][1]:.2f}/{out['split_null_max'][3]:.2f}/"
          f"{out['split_null_max'][5]:.2f} (p {out['split_p'][1]:.3f}/{out['split_p'][3]:.3f}/{out['split_p'][5]:.3f}) | overlap PCA5 "
          f"{out['overlap_pca5']:.2f} PCA25 {out['overlap_pca25']:.2f} J25 {out.get('overlap_J25', 'n/a')}", flush=True)
    json.dump(out, open(RES / f"mlens08_L{l}.json", "w"), indent=1)

    # ---------------- Stage C: decode pairs of the mean-H directions vs Sigma-orbit twins
    def twins(Hw, n, seed):
        gt = torch.Generator().manual_seed(seed); C = PS.T @ Hw
        return [PS @ ((torch.randint(0, 2, (d, 1), generator=gt).double() * 2 - 1) * C) for _ in range(n)]
    arms = {"Hmean": V5, **{f"tw{j}": t for j, t in enumerate(twins(V5, 4, 77))}}
    moves = {k: (Sh @ v).float() for k, v in arms.items()}
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    capt = {}
    m.model.layers[-1].register_forward_hook(lambda mod, inp, o_: capt.__setitem__("hL", (o_[0] if isinstance(o_, tuple) else o_)))
    rng = random.Random(3)
    recs = {k: {"r": [], "m": []} for k in arms}
    for wi, ids in enumerate(dec_ids):
        q = rng.randrange(SKIP + 8, T - 1)

        def hq(delta):
            R.delta = delta
            with torch.no_grad():
                m.model(input_ids=ids[None], use_cache=False)
            R.delta = None
            return capt["hL"][0, q].double()
        h0 = hq(None); rms = float(h0.pow(2).mean().sqrt())
        for k, U in moves.items():
            hs_ = []
            for i in range(5):
                dl = torch.zeros(1, T, d); dl[0, q] = U[:, i]; hs_.append(hq(dl))
            recs[k]["m"].append(torch.stack([(hs_[i] - h0) / rms for i in range(5)]))
            rr = []
            for (i, j) in pairs:
                dl = torch.zeros(1, T, d); dl[0, q] = U[:, i] + U[:, j]
                rr.append((hq(dl) - hs_[i] - hs_[j] + h0) / rms)
            recs[k]["r"].append(torch.stack(rr))
        if (wi + 1) % 4 == 0:
            progress("C decode pairs", 3, wi + 1, len(dec_ids))
            print(f"C: window {wi+1}/{len(dec_ids)} ({(time.time()-T0)/60:.1f} min)", flush=True)

    def F(Rn):
        s = Rn.sum(0); n = Rn.shape[0]
        return float(((s @ s - (Rn * Rn).sum()) / (n * (n - 1))) / (Rn * Rn).sum(1).mean())

    def decode(vec, k=12):
        lg = (vec.float() * gain) @ W_U.T; lg = lg - lg.mean()
        return {"top": [tok.decode([t]) for t in torch.topk(lg, k).indices.tolist()],
                "bottom": [tok.decode([t]) for t in torch.topk(-lg, k).indices.tolist()]}
    dec = {}
    for k in arms:
        Rr = torch.stack(recs[k]["r"]); Mm = torch.stack(recs[k]["m"])
        dec[k] = {"F_pairs": [F(Rr[:, p]) for p in range(len(pairs))], "F_singles": [F(Mm[:, i]) for i in range(5)]}
        dec[k]["F_pairs_mean"] = float(np.mean(dec[k]["F_pairs"])); dec[k]["F_singles_mean"] = float(np.mean(dec[k]["F_singles"]))
        if k in ("Hmean", "tw0"):
            dec[k]["decode_pairs"] = {f"{i}{j}": decode(Rr[:, p].mean(0)) for p, (i, j) in enumerate(pairs)}
            dec[k]["decode_singles"] = [decode(Mm[:, i].mean(0)) for i in range(5)]
    out["decode"] = dec
    json.dump(out, open(RES / f"mlens08_L{l}.json", "w"), indent=1)
    print("C: invariant fraction of pair cross-terms F: " + " | ".join(f"{k} {dec[k]['F_pairs_mean']:.3f} (singles {dec[k]['F_singles_mean']:.3f})" for k in arms), flush=True)
    for p, (i, j) in enumerate(pairs[:4]):
        print(f"   Hmean pair {i}{j}: top {dec['Hmean']['decode_pairs'][f'{i}{j}']['top'][:8]}", flush=True)
    json.dump(out, open(RES / f"mlens08_L{l}.json", "w"), indent=1)
    progress("done", 3, 1, 1, f"phi {phi:.3f}, split5 {obs[5]:.2f}")


if __name__ == "__main__":
    main()
