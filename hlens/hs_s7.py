"""Standalone S7 for run 1 (Oct 6 2026). hspace.py's S7 loops over cfg n_eval=512 windows but the wikitext-103 test
split yields only 455, so it would crash at window 456 (found by the pre-flight review). This file re-runs the S7 block
VERBATIM (pasted from hspace.py, the only change being n_ev = min(n_eval, len(ev_ids))) on the run-1 saved tensors
(mu, Sh, H25, J25 per layer) and merges labels/ablation into out/hspace_<tag>.json. Prereg deviation, logged.

    python hs_s7.py full
"""
import json
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import bf16w
from hspace import CFG, OUT, SKIP, Model, log, sym_sqrt, top_eig, wikitext_windows


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "small"
    cfg = CFG[tag]
    t0 = time.time()
    res_path, tens_path = OUT / f"hspace_{tag}.json", OUT / f"hspace_{tag}.pt"
    res = json.load(open(res_path))
    tens = torch.load(tens_path, map_location="cpu", weights_only=False)

    def save():
        json.dump(res, open(res_path, "w"), indent=1)

    m = Model(cfg["model"], cfg["T"])
    T, d, k = cfg["T"], m.d, cfg["k"]
    ev_ids = wikitext_windows(m.tok, "test" if tag == "full" else "validation", cfg["n_eval"] + cfg["n_jact"] // (T - SKIP - 1) + 8,
                              T, seed=1, skip_articles=0 if tag == "full" else 30).cuda()
    valid = torch.zeros(T, dtype=torch.bool, device="cuda"); valid[SKIP:T - 1] = True
    log(f"S7-standalone {tag}: {len(ev_ids)} eval windows; layers {cfg['layers']}")
    # ---------------- S7 causal: ablations + double dissociation (forward only, bf16 compute)
    bf16w.MODE = "bf16"
    n_ev = min(cfg["n_eval"], len(ev_ids))   # 455 test windows exist (logged deviation)
    E = ev_ids[:n_ev]
    filler = m.tok(" the", add_special_tokens=False)["input_ids"][0]
    rng = random.Random(7)
    labs = []   # (window, target position t, I, M1, M2)
    with torch.no_grad():
        for wi in range(n_ev):
            ids = E[wi:wi + 1]
            base = m.logprobs_next(m.run(m.embed(ids), 0), ids)[0]
            for _ in range(cfg["n_lab_targets"]):
                t = rng.randrange(max(SKIP + 44, T // 2), T - 2)
                s1 = rng.randrange(SKIP, t - 40); s2 = rng.randrange(s1 + 10, t - 8)
                vs = []
                for rm in ((1, 0), (0, 1), (1, 1)):
                    mod = ids.clone()
                    if rm[0]: mod[0, s1:s1 + 8] = filler
                    if rm[1]: mod[0, s2:s2 + 8] = filler
                    vs.append(float(m.logprobs_next(m.run(m.embed(mod), 0), mod)[0, t - 1]))
                f0 = float(base[t - 1])
                labs.append((wi, t, f0 - vs[0] - vs[1] + vs[2], f0 - vs[0], f0 - vs[1]))
            if wi % max(1, n_ev // 8) == 0:
                log(f"S7 labels {wi+1}/{n_ev} ({time.time()-t0:.0f}s)")
    labs_np = np.array([[x[2], x[3], x[4]] for x in labs])
    tot = np.abs(labs_np).sum(1)
    big = tot >= np.median(tot)
    ratio = np.abs(labs_np[:, 0]) / (np.abs(labs_np[:, 1]) + np.abs(labs_np[:, 2]) + 1e-6)
    qi, qa = np.quantile(ratio[big], 0.8), np.quantile(ratio[big], 0.2)
    inter = big & (ratio >= qi); addv = big & (ratio <= qa)
    res["labels"] = {"n": len(labs), "n_inter": int(inter.sum()), "n_add": int(addv.sum()),
                     "median_absI": float(np.median(np.abs(labs_np[:, 0])))}
    log(f"S7 labels done: {len(labs)} targets, {int(inter.sum())} interaction / {int(addv.sum())} additive ({time.time()-t0:.0f}s)")
    gr = torch.Generator(device="cuda").manual_seed(99)
    for l in cfg["layers"]:
        L = res["layers"][str(l)]
        mu, Sh = tens[f"mu{l}"].cuda(), tens[f"Sh{l}"].cuda()
        Sih = sym_sqrt(Sh @ Sh, inv=True)
        H25, J25 = tens[f"H25_{l}"].cuda(), tens[f"J25_{l}"].cuda()
        Hperp = torch.linalg.qr(H25 - J25 @ (J25.T @ H25))[0]
        conds = {"H25": H25, "J25": J25, "H25perpJ": Hperp}
        for r in range(3):
            conds[f"rand25_{r}"] = torch.linalg.qr(torch.randn(d, k, device="cuda", generator=gr))[0]
        # raw-space PCA25 (oblique in whitened coords): ablate raw top PCs
        Sraw = Sh @ Sh
        _, Upca = top_eig(Sraw, k)
        names = list(conds) + ["PCA25raw"]
        kls = {n_: [] for n_ in names}
        dmg = {n_: np.zeros(len(labs)) for n_ in names}
        by_w = {}
        for li, (w2, t, *_r) in enumerate(labs):
            by_w.setdefault(w2, []).append((li, t))
        for wi in range(n_ev):
            ids = E[wi:wi + 1]
            with torch.no_grad():
                X = m.run(m.embed(ids), 0, l + 1)
                lb = torch.log_softmax(m.logits(m.text.norm(m.run(X, l + 1))[0, valid]), -1)
                for name in names:
                    if name == "PCA25raw":
                        P = Upca[:, :k]
                        Xa = X - ((X - mu) @ P) @ P.T
                    else:
                        Q = conds[name]
                        z = (X - mu) @ Sih
                        Xa = mu + (z - (z @ Q) @ Q.T) @ Sh
                    la = torch.log_softmax(m.logits(m.text.norm(m.run(Xa, l + 1))[0, valid]), -1)
                    kls[name].append(float((lb.exp() * (lb - la)).sum(-1).mean()))
                    for li, t in by_w.get(wi, []):
                        pi, tok = t - 1 - SKIP, ids[0, t]
                        dmg[name][li] = float(lb[pi, tok] - la[pi, tok])
            if wi % max(1, n_ev // 4) == 0:
                log(f"L{l} S7 ablations window {wi+1}/{n_ev} ({time.time()-t0:.0f}s)")
        out = {}
        rb = np.random.default_rng(0)
        for name in names:
            di, da = dmg[name][inter], dmg[name][addv]
            boots = []
            for _ in range(1000):
                bi, ba = rb.choice(di, len(di)), rb.choice(da, len(da))
                boots.append(bi.mean() / max(1e-9, abs(ba.mean())))
            out[name] = {"KL": float(np.mean(kls[name])), "dmg_inter": float(di.mean()), "dmg_add": float(da.mean()),
                         "ratio": float(di.mean() / max(1e-9, abs(da.mean()))),
                         "ratio_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}
        L["ablation"] = out
        log(f"L{l} S7 ablation KL: " + " ".join(f"{n_}={v['KL']:.4f}" for n_, v in out.items())
            + " | dissoc ratio: " + " ".join(f"{n_}={v['ratio']:.2f}" for n_, v in out.items()) + f" ({time.time()-t0:.0f}s)")
        save()
    bf16w.MODE = "fp32"
    log(f"S7-standalone DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
