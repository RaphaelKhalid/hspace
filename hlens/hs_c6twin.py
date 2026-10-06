"""Sigma-orbit twin ablation: is the non-Sigma-commuting H-structure (hs_flip.py, variant b) causally interaction-specific?
(post-hoc, runs only if the frozen flip-test decision rule PASSES; Oct 6 2026)

EX  = top-k PATTERNS of the spatial-sign estimator after deflating the massive/radial span C (PCA coords 1..K, D-normalized
      correlation operator R = D^-1/2 M D^-1/2, pattern = QR(D^1/2 V)), mapped to the run-2 whitened frame. k = largest of
      1/3/5 whose observed split-half beat every flip replicate in hs_flip_full.json (norm, variant b).
twins = 8 Sigma-orbit twins Q_j = P (s_j * P'EX), s_j iid +-1 (seed 4242 + l): same raw variance and the same per-PCA-axis
      mass profile as EX exactly, and exchangeable with EX under the Sigma-commuting null.
Ablation: whitened mean-ablation at layer l, positions >= SKIP only (mu/Sigma were estimated there), run-2 frame.
Windows: the held-out eval windows 228..454 (C6' used 0..227), labels from the identical Random(7) draw sequence.
Statistic: S = r_M - r_I (hs_c6prime.py definitions). Real only if ALL hold (per layer):
  S(EX) ranks first among EX + 8 twins; paired-window bootstrap 95% CI of dS = S(EX) - mean_j S(tw_j) > 0;
  CI of dr_I = r_I(EX) - mean r_I(tw) < 0 and lower bound of dr_M > -0.05; mean KL(EX) inside the twins' KL range.

    python hs_c6twin.py smoke|full [layers...]
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bf16w  # noqa: E402
from hspace import CFG, SKIP, Model, log, wikitext_windows  # noqa: E402
from hs_flip import conf_basis  # noqa: E402

OUT = HERE / "out"
DEADLINE = 1791286200 if "--late" in sys.argv else 1791283800   # 11:30 / 10:50 UTC; checked every window
N_TWIN, K_PCA = 8, 1000


def ex_patterns(T2, l, k, dump):
    Sh, mu = T2[f"Sh{l}"].double(), T2[f"mu{l}"].double()
    ev, P = torch.linalg.eigh(Sh)
    P = P[:, torch.argsort(ev, descending=True)]
    Qp = P.T @ conf_basis(mu, Sh, 4)
    Y = dump["loc_y1"].double() @ P
    Y = Y - (Y @ Qp) @ Qp.T
    U = (Y / Y.norm(dim=1, keepdim=True))[:, :K_PCA]
    M = U.T @ U / len(U)
    dg = torch.diagonal(M)
    R = dg.rsqrt()[:, None] * M * dg.rsqrt()[None, :]
    e, V = torch.linalg.eigh(R)
    V = V[:, torch.argsort(e, descending=True)[:k]]
    pat = torch.linalg.qr(dg.sqrt()[:, None] * V)[0]            # PCA coords 1..K
    EX = P[:, :K_PCA] @ pat                                     # whitened frame, orthonormal
    g = torch.Generator().manual_seed(4242 + l)
    tw = [P @ ((torch.randint(0, 2, (P.shape[0], 1), generator=g).double() * 2 - 1) * (P.T @ EX)) for _ in range(N_TWIN)]
    pca25 = P[:, :25]
    return EX, tw, pca25


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    lay = [int(x) for x in sys.argv[2:] if x.isdigit()]
    cfg = CFG[tag]
    t0 = time.time()
    T2 = torch.load(OUT / f"hspace2_{tag}.pt", map_location="cpu", weights_only=False)
    FL = json.load(open(OUT / f"hs_flip_{tag}.json")) if (OUT / f"hs_flip_{tag}.json").exists() else {}
    m = Model(cfg["model"], cfg["T"])
    bf16w.MODE = "bf16"
    T, d = cfg["T"], m.d
    n_eval_all = cfg["n_eval"] if tag != "full" else 455
    ev_ids = wikitext_windows(m.tok, "test" if tag == "full" else "validation", cfg["n_eval"] + cfg["n_jact"] // (T - SKIP - 1) + 8,
                              T, seed=1, skip_articles=0 if tag == "full" else 30).cuda()
    n_all = min(n_eval_all, len(ev_ids))
    filler = m.tok(" the", add_special_tokens=False)["input_ids"][0]
    rng = random.Random(7)
    labs = []
    for wi in range(n_all):
        for _ in range(cfg["n_lab_targets"]):
            t = rng.randrange(max(SKIP + 44, T // 2), T - 2)
            s1 = rng.randrange(SKIP, t - 40); s2 = rng.randrange(s1 + 10, t - 8)
            labs.append((wi, t, s1, s2))
    w_lo = 228 if tag == "full" else n_all // 2
    labs = [x for x in labs if x[0] >= w_lo]
    ws = sorted({x[0] for x in labs})
    log(f"[c6t] {tag}: held-out windows {w_lo}..{n_all-1} ({len(ws)}), {len(labs)} targets ({time.time()-t0:.0f}s)")
    res_path = OUT / f"c6twin_{tag}.json"
    res = json.load(open(res_path)) if res_path.exists() else {"tag": tag, "layers": {}}
    for l in (lay or cfg["layers"]):
        fl = (FL.get(str(l)) or {}).get("b", {}).get("norm")
        if tag == "full":
            if not fl:
                log(f"[c6t] L{l}: no flip result, skipped"); continue
            ks = [k for k in (1, 3, 5) if fl["obs_split"][f"k{k}"] > fl[{1: "null_split_k1_max", 3: "null_split3_max", 5: "null_split5_max"}[k]]]
            if not ks:
                log(f"[c6t] L{l}: no pattern split beats the flip null, skipped"); continue
            k = max(ks)
        else:
            k = 3
        dump = torch.load(OUT / f"hspace2_{tag}_dump_L{l}.pt", map_location="cpu", weights_only=False)
        EX, tw, pca25 = ex_patterns(T2, l, k, dump)
        mu2, Sh2 = T2[f"mu{l}"].cuda(), T2[f"Sh{l}"].cuda()
        Sih2 = torch.linalg.inv(Sh2.double()).float()
        conds = {"none": None, "EX": EX.float().cuda(), **{f"tw_{j}": q.float().cuda() for j, q in enumerate(tw)}, "pca25": pca25.float().cuda()}
        names = list(conds)
        F = {n_: {} for n_ in names}
        KL = {n_: [] for n_ in names}
        by_w = {}
        for ti, (wi, t, s1, s2) in enumerate(labs):
            by_w.setdefault(wi, []).append((ti, t, s1, s2))
        done_w = []
        stopped = False
        for wi in ws:
            if time.time() > DEADLINE:
                log(f"[c6t] L{l}: deadline after {len(done_w)} windows"); stopped = True; break
            ids = ev_ids[wi:wi + 1]
            seqs, need = [ids[0]], []
            for (ti, t, s1, s2) in by_w[wi]:
                for v, (r1, r2) in enumerate(((1, 0), (0, 1), (1, 1))):
                    mod = ids[0].clone()
                    if r1: mod[s1:s1 + 8] = filler
                    if r2: mod[s2:s2 + 8] = filler
                    seqs.append(mod); need.append((ti, v + 1, len(seqs) - 1, t))
                need.append((ti, 0, 0, t))
            S = torch.stack(seqs)
            base_rows = [j for j, (_, v, _, _) in enumerate(need) if v == 0]
            with torch.no_grad():
                X = m.run(m.embed(S), 0, l + 1)
                lp0 = None
                for name in names:
                    if conds[name] is None:
                        Xa = X
                    else:
                        Q = conds[name]
                        z = (X - mu2) @ Sih2
                        Xa = X.clone()
                        Xa[:, SKIP:] = (mu2 + (z - (z @ Q) @ Q.T) @ Sh2)[:, SKIP:]
                    hL = m.text.norm(m.run(Xa, l + 1))
                    rows = torch.stack([hL[si, t - 1] for (_, _, si, t) in need])
                    lp = torch.log_softmax(m.logits(rows).float(), -1)
                    for j, (ti, v, si, t) in enumerate(need):
                        F[name][(ti, v)] = float(lp[j, S[si, t]])
                    if name == "none":
                        lp0 = lp[base_rows]
                    KL[name].append(float((lp0.exp() * (lp0 - lp[base_rows])).sum(-1).mean()))
            done_w.append(wi)
            if len(done_w) % 25 == 0:
                log(f"[c6t] L{l} window {len(done_w)}/{len(ws)} ({time.time()-t0:.0f}s)")
        tis = [ti for ti, x in enumerate(labs) if x[0] in set(done_w)]
        if len(done_w) < (10 if tag == "full" else 2):
            log(f"[c6t] L{l}: too few windows ({len(done_w)}), no statistics"); continue
        I = {n_: np.array([F[n_][(i, 0)] - F[n_][(i, 1)] - F[n_][(i, 2)] + F[n_][(i, 3)] for i in tis]) for n_ in names}
        M = {n_: np.array([[F[n_][(i, 0)] - F[n_][(i, 1)], F[n_][(i, 0)] - F[n_][(i, 2)]] for i in tis]) for n_ in names}
        win = np.array([labs[i][0] for i in tis])
        TW = [n_ for n_ in names if n_.startswith("tw_")]

        def st(idx):
            I0, M0 = I["none"][idx], M["none"][idx]
            o = {}
            for n_ in names[1:]:
                rI = float((I[n_][idx] * I0).sum() / max((I0 ** 2).sum(), 1e-12))
                rM = float((M[n_][idx] * M0).sum() / max((M0 ** 2).sum(), 1e-12))
                o[n_] = (rI, rM, rM - rI)
            return o

        full = st(np.arange(len(tis)))
        rb = np.random.default_rng(0)
        uw = np.unique(win)
        dS, dI, dM = [], [], []
        for _ in range(1000):
            wsb = rb.choice(uw, len(uw))
            idx = np.concatenate([np.where(win == w)[0] for w in wsb])
            b = st(idx)
            dS.append(b["EX"][2] - np.mean([b[t_][2] for t_ in TW]))
            dI.append(b["EX"][0] - np.mean([b[t_][0] for t_ in TW]))
            dM.append(b["EX"][1] - np.mean([b[t_][1] for t_ in TW]))
        q = lambda a: [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]
        S_all = {n_: full[n_][2] for n_ in ["EX"] + TW}
        kl = {n_: float(np.mean(KL[n_])) for n_ in names}
        L = {"k": k, "n_windows": len(done_w), "stopped_at_deadline": stopped,
             "conds": {n_: {"r_I": full[n_][0], "r_M": full[n_][1], "S": full[n_][2], "KL": kl[n_]} for n_ in names[1:]},
             "dS_ci95": q(dS), "drI_ci95": q(dI), "drM_ci95": q(dM),
             "rank_EX_among_9": int(1 + sum(S_all[t_] > S_all["EX"] for t_ in TW)),
             "KL_EX_in_twin_range": bool(min(kl[t_] for t_ in TW) <= kl["EX"] <= max(kl[t_] for t_ in TW))}
        L["REAL"] = bool(L["rank_EX_among_9"] == 1 and L["dS_ci95"][0] > 0 and L["drI_ci95"][1] < 0
                         and L["drM_ci95"][0] > -0.05 and L["KL_EX_in_twin_range"])
        res["layers"][str(l)] = L
        log(f"[c6t] L{l} (k={k}, {len(done_w)} windows): S(EX)={full['EX'][2]:+.3f} twins mean {np.mean([full[t_][2] for t_ in TW]):+.3f} "
            f"rank {L['rank_EX_among_9']}/9 dS CI [{L['dS_ci95'][0]:+.3f},{L['dS_ci95'][1]:+.3f}] drI CI [{L['drI_ci95'][0]:+.3f},{L['drI_ci95'][1]:+.3f}] "
            f"drM lo {L['drM_ci95'][0]:+.3f} KL EX {kl['EX']:.4f} twins [{min(kl[t_] for t_ in TW):.4f},{max(kl[t_] for t_ in TW):.4f}] "
            f"pca25 S {full['pca25'][2]:+.3f} -> {'REAL' if L['REAL'] else 'not real'} ({time.time()-t0:.0f}s)")
        tmp = res_path.with_name(res_path.name + ".tmp")
        json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, res_path)
    bf16w.MODE = "fp32"
    log(f"[c6t] DONE ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
