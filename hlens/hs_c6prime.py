"""C6' interaction-ablation test (prereg paper/SCOPE-hspace-v3.md, frozen before run-2 results).

For each target (same labels as run-1 S7: Random(7), 8-token span removals) and ablation A (whitened mean-ablation of a
25-dim subspace at layer l, all positions), measure the 2x2 interaction of the two spans ON the target log-prob WITH the
ablation applied in all four forwards:
    I_A = f(S1,S2) - f(-S1) - f(-S2) + f(-S1-S2),   M_A = (f(S1,S2) - f(-S1), f(S1,S2) - f(-S2))
Energy-weighted retentions r_I(A) = sum I_A I_0 / sum I_0^2, r_M(A) = sum <M_A, M_0> / sum |M_0|^2 and the
AND-specificity S(A) = r_M(A) - r_I(A) (positive = A removes interactions more than main effects).
Each subspace is ablated in its own whitening (run-2 subspaces with run-2 mu/Sigma, run-1 with run-1), as defined.

    python hs_c6prime.py smoke|small|full
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
from hspace import CFG, SKIP, Model, log, sym_sqrt, wikitext_windows  # noqa: E402

OUT = HERE / "out"
DEADLINE = 1791282600   # 2026-10-06 10:30 UTC
N_WIN = {"smoke": 6, "small": 48, "full": 228}


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    cfg = CFG[tag]
    t0 = time.time()
    T1 = torch.load(OUT / f"hspace_{tag}.pt", map_location="cpu", weights_only=False)
    T2 = torch.load(OUT / f"hspace2_{tag}.pt", map_location="cpu", weights_only=False)
    m = Model(cfg["model"], cfg["T"])
    bf16w.MODE = "bf16"
    T, d, k = cfg["T"], m.d, cfg["k"]
    n_eval_all = cfg["n_eval"] if tag != "full" else 455
    ev_ids = wikitext_windows(m.tok, "test" if tag == "full" else "validation", cfg["n_eval"] + cfg["n_jact"] // (T - SKIP - 1) + 8,
                              T, seed=1, skip_articles=0 if tag == "full" else 30).cuda()
    n_all = min(n_eval_all, len(ev_ids))
    filler = m.tok(" the", add_special_tokens=False)["input_ids"][0]
    rng = random.Random(7)
    labs = []                                             # identical draw sequence to run-1 S7 / hs_s7.py
    for wi in range(n_all):
        for _ in range(cfg["n_lab_targets"]):
            t = rng.randrange(max(SKIP + 44, T // 2), T - 2)
            s1 = rng.randrange(SKIP, t - 40); s2 = rng.randrange(s1 + 10, t - 8)
            labs.append((wi, t, s1, s2))
    nw = min(N_WIN[tag], n_all)
    labs = [x for x in labs if x[0] < nw]
    log(f"[c6p] {tag}: {nw} windows, {len(labs)} targets, layers {cfg['layers']} ({time.time()-t0:.0f}s)")
    res_path = OUT / f"c6prime_{tag}.json"
    res = {"tag": tag, "n_windows": nw, "n_targets": len(labs), "layers": {}}
    gr = torch.Generator(device="cuda").manual_seed(99)
    for l in cfg["layers"]:
        if time.time() > DEADLINE:
            log(f"[c6p] stopped at deadline before L{l}"); break
        if f"subs_{l}" not in T2:
            log(f"[c6p] L{l}: no run-2 subspaces, skipped"); continue
        mu1, Sh1 = T1[f"mu{l}"].cuda(), T1[f"Sh{l}"].cuda()
        mu2, Sh2 = T2[f"mu{l}"].cuda(), T2[f"Sh{l}"].cuda()
        Sih1, Sih2 = sym_sqrt(Sh1 @ Sh1, inv=True), sym_sqrt(Sh2 @ Sh2, inv=True)
        conds = {"none": None,
                 "loc_x": (T2[f"subs_{l}"]["loc_x"].cuda(), mu2, Sh2, Sih2),
                 "loc_norm": (T2[f"subs_{l}"]["loc_norm"].cuda(), mu2, Sh2, Sih2),
                 "run1_H25": (T1[f"H25_{l}"].cuda(), mu1, Sh1, Sih1),
                 "J25": (T1[f"J25_{l}"].cuda(), mu1, Sh1, Sih1)}
        for r in range(2):
            conds[f"rand_{r}"] = (torch.linalg.qr(torch.randn(d, k, device="cuda", generator=gr))[0], mu1, Sh1, Sih1)
        names = list(conds)
        F = {n_: {} for n_ in names}                      # F[cond][(target_idx, variant)] = log p
        by_w = {}
        for ti, (wi, t, s1, s2) in enumerate(labs):
            by_w.setdefault(wi, []).append((ti, t, s1, s2))
        for wi in range(nw):
            ids = ev_ids[wi:wi + 1]
            seqs, need = [ids[0]], []                     # sequence 0 = unmodified
            for (ti, t, s1, s2) in by_w.get(wi, []):
                for v, (r1, r2) in enumerate(((1, 0), (0, 1), (1, 1))):
                    mod = ids[0].clone()
                    if r1: mod[s1:s1 + 8] = filler
                    if r2: mod[s2:s2 + 8] = filler
                    seqs.append(mod); need.append((ti, v + 1, len(seqs) - 1, t))
                need.append((ti, 0, 0, t))
            S = torch.stack(seqs)
            with torch.no_grad():
                X = m.run(m.embed(S), 0, l + 1)
                for name in names:
                    if conds[name] is None:
                        Xa = X
                    else:
                        Q, mu, Sh, Sih = conds[name]
                        z = (X - mu) @ Sih
                        Xa = mu + (z - (z @ Q) @ Q.T) @ Sh
                    hL = m.text.norm(m.run(Xa, l + 1))
                    rows = torch.stack([hL[si, t - 1] for (_, _, si, t) in need])
                    lp = torch.log_softmax(m.logits(rows), -1)
                    for j, (ti, v, si, t) in enumerate(need):
                        F[name][(ti, v)] = float(lp[j, S[si, t]])
            if wi % max(1, nw // 6) == 0:
                log(f"[c6p] L{l} window {wi+1}/{nw} ({time.time()-t0:.0f}s)")
        n = len(labs)
        I = {n_: np.array([F[n_][(i, 0)] - F[n_][(i, 1)] - F[n_][(i, 2)] + F[n_][(i, 3)] for i in range(n)]) for n_ in names}
        M = {n_: np.array([[F[n_][(i, 0)] - F[n_][(i, 1)], F[n_][(i, 0)] - F[n_][(i, 2)]] for i in range(n)]) for n_ in names}
        win = np.array([x[0] for x in labs])

        def stats(idx):
            I0, M0 = I["none"][idx], M["none"][idx]
            out = {}
            for n_ in names[1:]:
                rI = float((I[n_][idx] * I0).sum() / max((I0 ** 2).sum(), 1e-12))
                rM = float((M[n_][idx] * M0).sum() / max((M0 ** 2).sum(), 1e-12))
                out[n_] = (rI, rM, rM - rI)
            return out

        full = stats(np.arange(n))
        boot = []
        rb = np.random.default_rng(0)
        uw = np.unique(win)
        for _ in range(1000):
            ws = rb.choice(uw, len(uw))
            idx = np.concatenate([np.where(win == w)[0] for w in ws])
            boot.append(stats(idx))
        L = {"median_absI0": float(np.median(np.abs(I["none"]))), "conds": {}}
        for n_ in names[1:]:
            Sb = np.array([b[n_][2] for b in boot])
            L["conds"][n_] = {"r_I": full[n_][0], "r_M": full[n_][1], "S": full[n_][2],
                              "S_ci95": [float(np.quantile(Sb, 0.025)), float(np.quantile(Sb, 0.975))]}
        ref = np.array([max(b["J25"][2], np.mean([b["rand_0"][2], b["rand_1"][2]])) for b in boot])
        for n_ in ("loc_x", "loc_norm", "run1_H25"):
            dlt = np.array([b[n_][2] for b in boot]) - ref
            L["conds"][n_]["S_minus_ref_ci95"] = [float(np.quantile(dlt, 0.025)), float(np.quantile(dlt, 0.975))]
        c = L["conds"]["loc_x"]
        L["C6prime_pass_loc_x"] = bool(c["S"] >= 0.2 and c["S_ci95"][0] > 0 and c["S_minus_ref_ci95"][0] > 0)
        res["layers"][str(l)] = L
        log(f"[c6p] L{l}: " + " | ".join(f"{n_} rI={v['r_I']:.2f} rM={v['r_M']:.2f} S={v['S']:+.2f} [{v['S_ci95'][0]:+.2f},{v['S_ci95'][1]:+.2f}]"
                                          for n_, v in L["conds"].items()) + f" | C6' loc_x {'PASS' if L['C6prime_pass_loc_x'] else 'fail'} ({time.time()-t0:.0f}s)")
        tmp = res_path.with_name(res_path.name + ".tmp")
        json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, res_path)
    bf16w.MODE = "fp32"
    log(f"[c6p] DONE ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
