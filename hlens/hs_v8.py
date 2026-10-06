"""v8: fresh-data H-lens probes on Qwen3.6-27B for (A) the first-order screen and (B) a confirmatory replication of the
punctuation-conditional structure (prereg paper/SCOPE-hspace-v8.md, frozen and pushed before the pod existed).

Per probe (run-2 frame: mu/Sh from hspace2_<tag>.pt; FRESH windows: wikitext seed 5, any window identical to a run-2
estimation window is dropped): one forward to layer l, c ~ vocab-weighted covector, v = randn @ Sh, two independent
Rademacher sign vectors s1, s2 over valid positions; g = dF/dx (create_graph) and two HVPs on the same graph.
  y1_p = (s1 * H(s1 v))_p Sh,  y2_p = (s2 * H(s2 v))_p Sh,  proxy_p = |g_p Sh|^2   (all valid positions kept)
(A) screen: Spearman(|y1_p|^2, proxy_p) over rows and windows; exact per-draw Frobenius-variance reduction of
    window-level importance sampling (Prop 9) with q_w ~ sum_p proxy_p, and the oracle q_w ~ ||Y_w||_F.
(B) v8 rule: punctuation rows, hs_flip5.run_layer (variant d axis-aligned drop, 200 flips, planted rank-3 control),
    plus cross-run replication = overlap of the fresh punctuation top-5 patterns with the run-2 punctuation top-5 patterns
    (same frame, same retained coordinates; norm estimator).
    python hs_v8.py small|full [layers...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.argv, _a = sys.argv[:1], sys.argv[1:]
import hspace as H  # noqa: E402
import hs_flip5 as F5  # noqa: E402

TAG = _a[0] if _a else "small"
LAYERS = [int(x) for x in _a[1:]] or ([16, 40] if TAG == "full" else [6, 12])
NP = {"full": 64, "small": 8}[TAG]
NFLIP, NPLANT = (200, 50) if TAG == "full" else (10, 5)
OUT = H.OUT


def klass(s):
    t = s.strip()
    if t == "":
        return "whitespace"
    if all(not ch.isalnum() for ch in t):
        return "punctuation"
    if t.isdigit():
        return "digit"
    return "word"


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def drop_mask(Sh, mu, K=1000):
    ev, P = torch.linalg.eigh(Sh)
    P = P[:, torch.argsort(ev, descending=True)]
    C = F5.conf_basis(mu, Sh, 4)
    wC = (P.T @ C).pow(2).sum(1)
    d = P.shape[0]; K = min(K, d)
    drop = torch.zeros(d, dtype=torch.bool); drop[:5] = True
    tot = float(wC[:K].sum())
    for i_ in torch.argsort(wC[:K], descending=True):
        if float(wC[:K][drop[:K]].sum()) >= 0.9 * tot:
            break
        drop[int(i_)] = True
    return P, (~drop).float(), K


def main():
    t0 = time.time()
    cfg = H.CFG[TAG]
    T = cfg["T"]
    B = cfg["B"] if TAG == "full" else 2
    T2 = torch.load(OUT / f"hspace2_{TAG}.pt", map_location="cpu", weights_only=False)
    m = H.Model(cfg["model"], T)
    tok = m.tok
    d = m.d
    gen = torch.Generator(device="cuda").manual_seed(8)
    n_win = NP * B
    run2 = H.wikitext_windows(tok, cfg["split"], NP * B + cfg["n_cov"] if TAG != "full" else 256 * 4 + 256, T, seed=0)
    fresh = H.wikitext_windows(tok, cfg["split"], n_win * 3, T, seed=5)
    seen = {tuple(w.tolist()) for w in run2}
    keepw = [w for w in fresh if tuple(w.tolist()) not in seen][:n_win]
    if TAG != "full":                      # the small validation split has too few articles to dedup (smoke only)
        keepw = list(fresh[:n_win])
    assert len(keepw) == n_win, f"only {len(keepw)} fresh windows"
    n_dup = len(fresh) - len([w for w in fresh if tuple(w.tolist()) not in seen])
    ids_all = torch.stack(keepw).cuda()
    valid = torch.zeros(T, dtype=torch.bool, device="cuda"); valid[H.SKIP:T - 1] = True
    print(f"[v8] {TAG}: {len(keepw)} fresh windows (dropped {n_dup} duplicates of run-2 windows), layers {LAYERS} ({time.time()-t0:.0f}s)", flush=True)

    def sample_c():
        eta = torch.randn(1, m.W_U.shape[0], device="cuda", generator=gen) / m.W_U.shape[0] ** 0.5
        return ((eta.to(torch.bfloat16) @ m.W_U).float() * m.gain[None])[0]

    res = {"tag": TAG, "n_fresh_windows": len(keepw), "n_dup_dropped": n_dup, "layers": {}}
    for l in LAYERS:
        Sh, mu = T2[f"Sh{l}"].cuda().float(), T2[f"mu{l}"].float()
        Y1, Y2, PX, TK, PR, WI = [], [], [], [], [], []
        for i in range(NP):
            ix = torch.arange(i * B, (i + 1) * B)
            ids = ids_all[ix]
            with torch.no_grad():
                X = m.run(m.embed(ids), 0, l + 1)
            c = sample_c()
            v = torch.randn(d, device="cuda", generator=gen) @ Sh
            s1 = (torch.randint(0, 2, (B, T), device="cuda", generator=gen).float() * 2 - 1) * valid[None]
            s2 = (torch.randint(0, 2, (B, T), device="cuda", generator=gen).float() * 2 - 1) * valid[None]
            x = X.clone().requires_grad_(True)
            with torch.enable_grad():
                hL = m.run(x, l + 1)
                Fv = (hL[:, valid] @ c).sum()
                (g,) = torch.autograd.grad(Fv, x, create_graph=True)
                (h1,) = torch.autograd.grad((g * (s1[..., None] * v)).sum(), x, retain_graph=True)
                (h2,) = torch.autograd.grad((g * (s2[..., None] * v)).sum(), x)
            with torch.no_grad():
                Y1.append(((s1[..., None] * h1)[:, valid] @ Sh).reshape(-1, d).to(torch.bfloat16).cpu())
                Y2.append(((s2[..., None] * h2)[:, valid] @ Sh).reshape(-1, d).to(torch.bfloat16).cpu())
                PX.append((g.detach()[:, valid] @ Sh).pow(2).sum(-1).reshape(-1).cpu())
                TK.append(ids[:, valid].reshape(-1).cpu())
                npos = int(valid.sum())
                PR.append(torch.full((B * npos,), i))
                WI.append(ix.repeat_interleave(npos))
            del x, hL, g, h1, h2
            if i % max(1, NP // 8) == 0 or i == NP - 1:
                print(f"[v8] L{l} probe {i+1}/{NP} ({time.time()-t0:.0f}s, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB)", flush=True)
        D = {"loc_y1": torch.cat(Y1), "loc_y2": torch.cat(Y2), "proxy": torch.cat(PX), "loc_tok": torch.cat(TK),
             "loc_probe": torch.cat(PR), "win": torch.cat(WI)}
        torch.save(D, OUT / f"hs_v8_{TAG}_L{l}.pt")
        torch.cuda.empty_cache()
        # ---------------- (A) screen
        y = D["loc_y1"].double()
        e = y.pow(2).sum(1)
        p = D["proxy"].double()
        win = D["win"]
        uw = torch.unique(win)
        fw = torch.stack([(y[win == w] @ y[win == w].T).pow(2).sum().sqrt() for w in uw])
        pw = torch.stack([p[win == w].sum() for w in uw])
        ew = torch.stack([e[win == w].sum() for w in uw])
        F2 = float((y.T @ y).pow(2).sum())
        nW = len(uw)

        def var(q):
            q = q / q.sum()
            return float((fw ** 2 / q).sum()) - F2
        vu = float(nW * (fw ** 2).sum()) - F2
        A = {"kish_ess_frac_rows": float(e.sum() ** 2 / (e ** 2).sum()) / len(e),
             "spearman_rows": spearman(e.numpy(), p.numpy()), "spearman_windows": spearman(ew.numpy(), pw.numpy()),
             "window_var_reduction_proxy": vu / var(pw), "window_var_reduction_oracle": vu / var(fw), "n_rows": len(e), "n_windows": nW}
        print(f"[v8] L{l} (A) screen: Kish ESS {100*A['kish_ess_frac_rows']:.3f}% | Spearman rows {A['spearman_rows']:.2f} windows "
              f"{A['spearman_windows']:.2f} | window IS: proxy {A['window_var_reduction_proxy']:.2f}x, oracle {A['window_var_reduction_oracle']:.2f}x", flush=True)
        del y
        # ---------------- (B) punctuation-conditional replication (v8 rule)
        P, keep, K = drop_mask(T2[f"Sh{l}"].double(), T2[f"mu{l}"].double())
        Pf = P.float()
        ut = {int(t): klass(tok.decode([int(t)])) for t in torch.unique(D["loc_tok"])}
        idx = torch.tensor([i for i, t in enumerate(D["loc_tok"].tolist()) if ut[t] == "punctuation"])
        Y1p = D["loc_y1"][idx].float() @ Pf
        Y2p = D["loc_y2"][idx].float() @ Pf
        prb = D["loc_probe"][idx]
        rr = F5.run_layer(Y1p, Y2p, prb, {"d": keep}, K, NFLIP, NPLANT, "cpu", 8000 + l, f"v8 L{l} punctuation")
        nr = Y1p.norm(dim=1)
        # fresh full-data patterns (both halves pooled) vs run-2 punctuation patterns, same coordinates
        def full_patterns(Ya):
            cols = torch.nonzero(keep[:K] > 0).squeeze(1)
            U = (Ya * keep / (Ya * keep).norm(dim=1, keepdim=True).clamp_min(1e-30))[:, cols]
            M = U.T @ U / len(U)
            dg = torch.diagonal(M).clamp_min(1e-30)
            R = dg.rsqrt()[:, None] * M * dg.rsqrt()[None, :]
            ev, V = torch.linalg.eigh(R)
            V = V[:, torch.argsort(ev, descending=True)[:5]]
            return torch.linalg.qr(dg.sqrt()[:, None] * V)[0]
        fresh_pat = full_patterns(Y1p)
        rep = None
        r2f = OUT / f"hspace2_{TAG}_dump_L{l}.pt"
        if r2f.exists():
            D2 = torch.load(r2f, map_location="cpu", weights_only=False)
            ut2 = {int(t): klass(tok.decode([int(t)])) for t in torch.unique(D2["loc_tok"])}
            i2 = torch.tensor([i for i, t in enumerate(D2["loc_tok"].tolist()) if ut2[t] == "punctuation"])
            run2_pat = full_patterns(D2["loc_y1"][i2].float() @ Pf)
            rep = float((fresh_pat.T @ run2_pat).pow(2).sum() / 5)
            del D2
        Bres = {"n_rows": len(idx), "d": {e_: {k: rr["d"][e_][k] for k in ("obs_lam", "null_lam1_max", "obs_split", "null_split3_max")}
                                          for e_ in ("norm", "xnorm", "raw_trim5")},
                "planted": rr["planted"], "cross_run_top5_overlap": rep,
                "random_expect_top5": 5 / float(keep[:K].sum())}
        ok = all(Bres["d"][e_]["obs_lam"][0] > Bres["d"][e_]["null_lam1_max"] and Bres["d"][e_]["obs_split"]["k3"] > Bres["d"][e_]["null_split3_max"]
                 and Bres["d"][e_]["obs_split"]["k3"] >= 0.7 and Bres["planted"][e_]["detected"] for e_ in ("norm", "xnorm"))
        ok = ok and rep is not None and rep >= 0.5
        Bres["layer_pass"] = bool(ok)
        print(f"[v8] L{l} (B) punctuation: rows {len(idx)} | cross-run top-5 overlap {rep} (random {Bres['random_expect_top5']:.4f}) -> layer {'PASS' if ok else 'fail'}", flush=True)
        res["layers"][str(l)] = {"screen": A, "punct": Bres}
        json.dump(res, open(OUT / f"hs_v8_{TAG}.json", "w"), indent=1)
        del D, Y1p, Y2p
    if all(str(x) in res["layers"] for x in LAYERS):
        res["VERDICT_v8"] = ("punctuation-conditional H-structure REPLICATED (v8)" if all(res["layers"][str(x)]["punct"]["layer_pass"] for x in LAYERS)
                             else "punctuation-conditional H-structure NOT replicated (v8)")
        print(f"[v8] VERDICT: {res['VERDICT_v8']}", flush=True)
    json.dump(res, open(OUT / f"hs_v8_{TAG}.json", "w"), indent=1)
    print(f"[v8] DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
