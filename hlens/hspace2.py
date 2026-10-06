"""H-space run 2 (EXPLORATORY, after the run-1 one-shot; addendum paper/SCOPE-hspace-v2.md). v2 after pre-flight review.

Run 1 (27B) showed heavy-tailed per-position sample magnitudes (PR 1.3-8.5) and split-half 0.33-0.41. Run 2 asks
whether H-space is reproducible under (i) the spatial-sign (normalized) estimator and (ii) an unbiased CROSS-MOMENT
estimator, locally and for the rectangle-averaged Hessian (theory Thm 3), with causal tests for every variant.

Per probe (B windows, output covector c, whitened direction v) we draw TWO independent Rademacher position-sign
vectors s1, s2 and compute four HVPs:
  loc1 = H(X)[s1 (x) v], loc2 = H(X)[s2 (x) v]                                    (local, at the data point)
  int1 = H(X_r(st1))[s1 (x) v], int2 = H(X_r(st2))[s2 (x) v]                       (two independent rectangle points)
where X_r moves ONLY a sparse set P of positions (every 8th valid position, random offset per window) by removing
random fractions (s, t ~ U[0,1]) of two random whitened halves of each moved token's deviation from the mean.
Samples y1_p = Sigma^1/2 s1_p (H1)_p and y2_p likewise. Then
  E[y1_p y1_p^T]                  = sum_q H_pq v v^T H_pq^T        (run-1 estimator "raw": includes cross-position blocks)
  E[sym(y1_p y2_p^T)]             = H_pp v v^T H_pp^T              ("x": exact per-position diagonal-block energy)
  int: E[sym(y1_p y2_p^T)]        = Hbar_pp v v^T Hbar_pp^T        (energy of the RECTANGLE-AVERAGED Hessian, Thm 3)
"norm" = E[unit(y1) unit(y1)^T] (spatial sign of "raw"; robust to heavy-tailed magnitudes; equals raw's eigenvectors
only for elliptical samples). "xnorm" = E[sym(unit(y1) unit(y2)^T)] = E[m m^T], m = E_s[unit(y)]: a robust heuristic
that still depends on cross-position blocks (NOT "x normalized"). Only "x" is the exact diagonal-block energy; for the
int kind only int_x targets the rectangle average (diagonal blocks under a joint move of the sparse set P); int_raw /
int_norm are rectangle averages of the squared curvature. "locP" = the loc HVPs read only at P (n-matched control for int).
Signed cross-moment matrices are PSD in expectation; spec() reports a noise floor |lambda_min| and debiased energies.

    python hspace2.py smoke | small | full
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bf16w  # noqa: E402
from hspace import CFG as CFG1, SKIP, Model, log, overlap, sym_sqrt, top_eig, wikitext_windows  # noqa: E402

OUT = HERE / "out"
CFG = {
    "small": dict(CFG1["small"], layers=[6, 12, 18], sweep=[3, 6, 9, 12, 15, 18, 21], n_probe=48, keep_per_probe=40, n_eval=48, n_sweep=24),
    "smoke": dict(CFG1["smoke"], layers=[12], sweep=[6, 12], n_probe=4, keep_per_probe=20, n_eval=6, n_sweep=4),
    "full": dict(CFG1["full"], layers=[16, 28, 40, 52], sweep=[4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60],
                 n_probe=256, keep_per_probe=64, n_eval=455, n_sweep=128),
}
VARS = ["raw", "norm", "x", "xnorm"]


def spec(Mt, k):
    """Symmetric operator -> (eigvals desc, eigvecs, stats). For signed cross-moment matrices the population is PSD;
    |lambda_min| estimates the sampling-noise scale (floor) and energies are reported raw, positive-part and debiased."""
    ev, U = top_eig(Mt.cuda(), k)
    tr = float(ev.sum()); pos = ev.clamp_min(0); ps = float(pos.sum())
    floor = float((-ev[-1]).clamp_min(0))
    deb = (ev[:k] - floor).clamp_min(0)
    return ev, U, {"trace": tr, "noise_floor": floor, "n_above_floor": int((ev > floor).sum()),
                   "top25_energy": float(pos[:k].sum()) / tr if tr > 0 else float("nan"),
                   "top25_energy_pos": float(pos[:k].sum()) / max(ps, 1e-30),
                   "top25_energy_debiased": float(deb.sum()) / tr if tr > 0 else float("nan"),
                   "PR": float(pos.sum() ** 2 / max(float((pos ** 2).sum()), 1e-30)),
                   "eigs_top16": [float(x) for x in ev[:16]], "eigs_bottom8": [float(x) for x in ev[-8:]],
                   "neg_mass": float(-ev.clamp_max(0).sum()) / tr if tr > 0 else float("nan")}


DEADLINE_BC = 1791282600     # 2026-10-06 10:30 UTC: stages B/C stop cleanly; the pod watchdog deletes the pod at 11:00 UTC


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "small"
    cfg = CFG[tag]
    gen = torch.Generator(device="cuda").manual_seed(2026)
    t0 = time.time()
    res = {"tag": tag, "cfg": {k_: v_ for k_, v_ in cfg.items() if not isinstance(v_, tuple)}, "layers": {}, "sweep": {}}
    res_path, tens_path = OUT / f"hspace2_{tag}.json", OUT / f"hspace2_{tag}.pt"
    tens = {}

    def save():
        tmp = res_path.with_name(res_path.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(res, f, indent=1)
        os.replace(tmp, res_path)
        tmp = tens_path.with_name(tens_path.name + ".tmp")
        torch.save(tens, tmp)
        os.replace(tmp, tens_path)

    m = Model(cfg["model"], cfg["T"])
    T, d, k = cfg["T"], m.d, cfg["k"]
    log(f"[v2] {tag}: loaded, mem {torch.cuda.memory_allocated()/2**30:.1f} GB ({time.time()-t0:.0f}s)")
    from huggingface_hub import hf_hub_download
    Jall = torch.load(hf_hub_download(cfg["lens"][0], cfg["lens"][2], revision=cfg["lens"][1]), map_location="cpu", weights_only=True)["J"]
    Lstar = max(Jall) + 1
    n_est = cfg["n_probe"] * cfg["B"] + cfg["n_cov"]
    est = wikitext_windows(m.tok, cfg["split"], n_est, T, seed=10)          # fresh windows (run 1 used seed 0)
    cov_ids, probe_ids = est[: cfg["n_cov"]].cuda(), est[cfg["n_cov"]:].cuda()
    ev_ids = wikitext_windows(m.tok, "test" if tag == "full" else "validation", cfg["n_eval"], T, seed=1,
                              skip_articles=0 if tag == "full" else 30).cuda()
    valid = torch.zeros(T, dtype=torch.bool, device="cuda"); valid[SKIP:T - 1] = True
    nvalid = int(valid.sum())
    U2 = torch.zeros(d, d, device="cuda", dtype=torch.float64)
    for s in range(0, m.W_U.shape[0], 16384):
        Wg = m.W_U[s:s + 16384].float() * m.gain[None]
        U2 += (Wg.T @ Wg).double()
    U2 = (U2 / m.W_U.shape[0]).float()
    log(f"[v2] data ready: {len(est)} est windows, {len(ev_ids)} eval windows ({time.time()-t0:.0f}s)")
    r1 = OUT / ("hspace_full.pt" if tag == "full" else f"hspace_{tag}.pt")
    T1 = torch.load(r1, map_location="cpu", weights_only=False) if r1.exists() else {}
    if tag == "full":
        assert all(f"J25_{l}" in T1 for l in cfg["layers"]), f"run-1 J25 missing in {r1}"

    def sample_c():
        eta = torch.randn(m.W_U.shape[0], device="cuda", generator=gen) / m.W_U.shape[0] ** 0.5
        return (eta.to(torch.bfloat16) @ m.W_U).float() * m.gain

    def stats(l):
        S = torch.zeros(d, d, device="cuda", dtype=torch.float64); mu = torch.zeros(d, device="cuda", dtype=torch.float64); n = 0
        for s in range(0, len(cov_ids), 4):
            h = m.states(cov_ids[s:s + 4], l)[:, valid].reshape(-1, d).double()
            mu += h.sum(0); S += h.T @ h; n += h.shape[0]
        mu /= n; S = S / n - torch.outer(mu, mu)
        S = 0.95 * S + 0.05 * torch.trace(S) / d * torch.eye(d, device="cuda", dtype=S.dtype)
        return mu.float(), sym_sqrt(S.float()), sym_sqrt(S.float(), inv=True)

    def hvp_probe(X, l, c_top, top, V):
        x = X.clone().requires_grad_(True)
        with torch.enable_grad():
            hk = m.run(x, l + 1, top + 1)
            Fv = (hk[:, valid] @ c_top).sum()
            (g,) = torch.autograd.grad(Fv, x, create_graph=True)
            (hv,) = torch.autograd.grad((g * V).sum(), x)
        return hv.detach()

    def unit(y):
        return y / y.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # ------------------------------------------------ A. main layers
    for l in cfg["layers"]:
      try:
        L = {}
        res["layers"][str(l)] = L
        mu, Sh, Sih = stats(l)
        tens[f"mu{l}"], tens[f"Sh{l}"] = mu.cpu(), Sh.cpu()
        B = cfg["B"]
        KINDS = ("loc", "int", "locP")
        acc = {(kind, var, h): torch.zeros(d, d, device="cuda") for kind in KINDS for var in VARS for h in (0, 1)}
        cnt = {(kind, h): 0 for kind in KINDS for h in (0, 1)}
        dump = {kind: {"y1": [], "y2": [], "probe": [], "pos": [], "win": [], "tok": []} for kind in ("loc", "int")}
        nskip = 0
        for i in range(cfg["n_probe"]):
            half = i % 2
            ix = torch.arange(i * B, (i + 1) * B) % len(probe_ids)
            ids = probe_ids[ix]
            with torch.no_grad():
                X = m.states(ids, l)
            c = sample_c()
            v = torch.randn(d, device="cuda", generator=gen) @ Sh
            s1 = (torch.randint(0, 2, (B, T), device="cuda", generator=gen).float() * 2 - 1) * valid[None]
            s2 = (torch.randint(0, 2, (B, T), device="cuda", generator=gen).float() * 2 - 1) * valid[None]
            off = torch.randint(0, 8, (B, 1), device="cuda", generator=gen)
            P = valid[None] & (((torch.arange(T, device="cuda")[None] - SKIP - off) % 8) == 0)      # [B,T] sparse moved positions
            Q = torch.linalg.qr(torch.randn(d, d // 2, device="cuda", generator=gen))[0]
            Z = (X - mu) @ Sih
            Za = (Z @ Q) @ Q.T
            st = torch.rand(2, 2, device="cuda", generator=gen)

            def Xr(j):
                return X - (((1 - st[j, 0]) * Za + (1 - st[j, 1]) * (Z - Za)) @ Sh) * P[..., None]

            H = {("loc", 1): hvp_probe(X, l, c, Lstar, s1[..., None] * v), ("loc", 2): hvp_probe(X, l, c, Lstar, s2[..., None] * v),
                 ("int", 1): hvp_probe(Xr(0), l, c, Lstar, s1[..., None] * v), ("int", 2): hvp_probe(Xr(1), l, c, Lstar, s2[..., None] * v)}
            if not all(torch.isfinite(h_).all() for h_ in H.values()):
                nskip += 1
                continue
            for kind, mask, src in (("loc", valid[None].expand(B, T), "loc"), ("int", P, "int"), ("locP", P, "loc")):
                y1 = ((s1[..., None] * H[(src, 1)]) @ Sh)[mask]
                y2 = ((s2[..., None] * H[(src, 2)]) @ Sh)[mask]
                u1, u2 = unit(y1), unit(y2)
                acc[(kind, "raw", half)] += y1.T @ y1
                acc[(kind, "norm", half)] += u1.T @ u1
                xy = y1.T @ y2
                acc[(kind, "x", half)] += 0.5 * (xy + xy.T)
                xu = u1.T @ u2
                acc[(kind, "xnorm", half)] += 0.5 * (xu + xu.T)
                cnt[(kind, half)] += y1.shape[0]
                if kind == "locP":
                    continue
                r = torch.randperm(y1.shape[0], generator=torch.Generator().manual_seed(i * 7 + (kind == "int")))[: cfg["keep_per_probe"]].cuda()
                bpos = mask.nonzero()                                     # [n,2] (window, position) for each row
                dump[kind]["y1"].append(y1[r].to(torch.bfloat16).cpu()); dump[kind]["y2"].append(y2[r].to(torch.bfloat16).cpu())
                dump[kind]["probe"].append(torch.full((len(r),), i)); dump[kind]["pos"].append(bpos[r, 1].cpu())
                dump[kind]["win"].append(ix[bpos[r, 0].cpu()]); dump[kind]["tok"].append(ids[bpos[r, 0], bpos[r, 1]].cpu())
            if i % max(1, cfg["n_probe"] // 8) == 0 or i == cfg["n_probe"] - 1:
                log(f"[v2] L{l} probe {i+1}/{cfg['n_probe']} ({time.time()-t0:.0f}s, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB)")
        L["n_skipped_nonfinite"] = nskip
        dpath = OUT / f"hspace2_{tag}_dump_L{l}.pt"
        torch.save({f"{kind}_{key_}": torch.cat(lst) for kind in ("loc", "int") for key_, lst in dump[kind].items()},
                   dpath.with_name(dpath.name + ".tmp"))
        os.replace(dpath.with_name(dpath.name + ".tmp"), dpath)          # samples are safe before any eigen work
        del dump
        subs = {}
        for kind in KINDS:
            for var in VARS:
                key = f"{kind}_{var}"
                Ma, Mb = acc[(kind, var, 0)].double(), acc[(kind, var, 1)].double()
                _, UA, sA = spec(Ma, k); _, UB, sB = spec(Mb, k)
                ev, U, st_ = spec(Ma + Mb, k)
                kf = max(1, min(k, sA["n_above_floor"], sB["n_above_floor"]))
                L[key] = dict(st_, split_half=overlap(UA[:, :k], UB[:, :k]), split_half_k5=overlap(UA[:, :5], UB[:, :5]),
                              split_half_k10=overlap(UA[:, :10], UB[:, :10]), split_half_k50=overlap(UA[:, :50], UB[:, :50]),
                              split_half_kfloor=overlap(UA[:, :kf], UB[:, :kf]), kfloor=kf,
                              n_samples=cnt[(kind, 0)] + cnt[(kind, 1)])
                tens[f"{key}_ev_{l}"] = ev.cpu()
                if kind != "locP":
                    subs[key] = U[:, :k]
                    tens[f"{key}_{l}"] = U[:, :64].cpu()
                    tens[f"{key}_halves_{l}"] = (UA[:, :25].cpu(), UB[:, :25].cpu())
        del acc
        keys = list(subs)
        for a_ in keys:
            for b_ in keys:
                if a_ < b_:
                    L[f"ov_{a_}__{b_}"] = overlap(subs[a_], subs[b_])
        if f"J25_{l}" in T1:
            Jw = torch.linalg.qr(Sh @ torch.linalg.solve(T1[f"Sh{l}"].cuda().double(), T1[f"J25_{l}"].cuda().double()).float())[0]   # covector map
            tens[f"J25w_{l}"] = Jw.cpu()
            for kk, vv in subs.items():
                L[f"C3_{kk}_vs_J25"] = overlap(vv, Jw)
        if f"H25_{l}" in T1:
            Hw = torch.linalg.qr(Sh @ torch.linalg.solve(T1[f"Sh{l}"].cuda().double(), T1[f"H25_{l}"].cuda().double()).float())[0]
            tens[f"run1H25w_{l}"] = Hw.cpu()
            for kk, vv in subs.items():
                L[f"ov_{kk}_vs_run1_H25"] = overlap(vv, Hw)
        tens[f"subs_{l}"] = {kk: vv.cpu() for kk, vv in subs.items()}
        log(f"[v2] L{l} split-half: " + " ".join(f"{kk}={L[kk]['split_half']:.2f}" for kk in keys)
            + " | locP: " + " ".join(f"{v_}={L['locP_' + v_]['split_half']:.2f}" for v_ in VARS)
            + " | k5: " + " ".join(f"{kk}={L[kk]['split_half_k5']:.2f}" for kk in keys)
            + " | PR: " + " ".join(f"{kk}={L[kk]['PR']:.1f}" for kk in keys)
            + " | vsJ25: " + " ".join(f"{kk}={L.get(f'C3_{kk}_vs_J25', float('nan')):.2f}" for kk in keys) + f" ({time.time()-t0:.0f}s)")
        save()
      except Exception:  # noqa: BLE001
        log(f"[v2] stage A L{l} FAILED: {traceback.format_exc()[-600:]}")
        torch.cuda.empty_cache()

    # ------------------------------------------------ B. causal tests for every variant (bf16 forward)
    try:
        bf16w.MODE = "bf16"
        n_ev = len(ev_ids)
        filler = m.tok(" the", add_special_tokens=False)["input_ids"][0]
        rng = random.Random(7)
        labs = []
        with torch.no_grad():
            for wi in range(n_ev):
                ids = ev_ids[wi:wi + 1]
                base = m.logprobs_next(m.run(m.embed(ids), 0), ids)[0]
                for _ in range(cfg["n_lab_targets"]):
                    t = rng.randrange(max(SKIP + 44, T // 2), T - 2)
                    s1_ = rng.randrange(SKIP, t - 40); s2_ = rng.randrange(s1_ + 10, t - 8)
                    vs = []
                    for rm in ((1, 0), (0, 1), (1, 1)):
                        mod = ids.clone()
                        if rm[0]: mod[0, s1_:s1_ + 8] = filler
                        if rm[1]: mod[0, s2_:s2_ + 8] = filler
                        vs.append(float(m.logprobs_next(m.run(m.embed(mod), 0), mod)[0, t - 1]))
                    f0 = float(base[t - 1])
                    labs.append((wi, t, f0 - vs[0] - vs[1] + vs[2], f0 - vs[0], f0 - vs[1]))
                if wi % max(1, n_ev // 8) == 0:
                    log(f"[v2] labels {wi+1}/{n_ev} ({time.time()-t0:.0f}s)")
        lab = np.array([[x[2], x[3], x[4]] for x in labs])
        tot = np.abs(lab).sum(1); big = tot >= np.median(tot)
        ratio = np.abs(lab[:, 0]) / (np.abs(lab[:, 1]) + np.abs(lab[:, 2]) + 1e-6)
        inter = big & (ratio >= np.quantile(ratio[big], 0.8)); addv = big & (ratio <= np.quantile(ratio[big], 0.2))
        res["labels"] = {"n": len(labs), "n_inter": int(inter.sum()), "n_add": int(addv.sum()), "median_absI": float(np.median(np.abs(lab[:, 0])))}
        tens["labs"] = torch.from_numpy(lab); tens["lab_wt"] = torch.tensor([[x[0], x[1]] for x in labs])
        tens["inter"] = torch.from_numpy(inter); tens["addv"] = torch.from_numpy(addv)
        save()
        log(f"[v2] labels: {len(labs)} targets, {int(inter.sum())}/{int(addv.sum())} ({time.time()-t0:.0f}s)")
        gr = torch.Generator(device="cuda").manual_seed(99)
        by_w = {}
        for li, (w2, t, *_r) in enumerate(labs):
            by_w.setdefault(w2, []).append((li, t))
        for l in cfg["layers"]:
            if time.time() > DEADLINE_BC:
                log(f"[v2] stage B stopped at deadline before L{l}"); break
            L = res["layers"][str(l)]
            mu, Sh = tens[f"mu{l}"].cuda(), tens[f"Sh{l}"].cuda()
            Sih = sym_sqrt(Sh @ Sh, inv=True)
            conds = {kk: vv.cuda() for kk, vv in tens[f"subs_{l}"].items() if kk in ("loc_raw", "loc_norm", "loc_x", "int_x")}
            if f"J25w_{l}" in tens:
                conds["J25"] = tens[f"J25w_{l}"].cuda()
            for r in range(3):
                conds[f"rand_{r}"] = torch.linalg.qr(torch.randn(d, k, device="cuda", generator=gr))[0]
            names = list(conds)
            kls = {n_: [] for n_ in names}; dmg = {n_: np.zeros(len(labs)) for n_ in names}
            for wi in range(n_ev):
                ids = ev_ids[wi:wi + 1]
                with torch.no_grad():
                    X = m.run(m.embed(ids), 0, l + 1)
                    lb = torch.log_softmax(m.logits(m.text.norm(m.run(X, l + 1))[0, valid]), -1)
                    z = (X - mu) @ Sih
                    for name in names:
                        Qc = conds[name]
                        la = torch.log_softmax(m.logits(m.text.norm(m.run(mu + (z - (z @ Qc) @ Qc.T) @ Sh, l + 1))[0, valid]), -1)
                        kls[name].append(float((lb.exp() * (lb - la)).sum(-1).mean()))
                        for li, t in by_w.get(wi, []):
                            pi, tok = t - 1 - SKIP, ids[0, t]
                            dmg[name][li] = float(lb[pi, tok] - la[pi, tok])
                if wi % max(1, n_ev // 4) == 0:
                    log(f"[v2] L{l} ablation window {wi+1}/{n_ev} ({time.time()-t0:.0f}s)")
            rk = np.mean([np.mean(kls[f"rand_{r}"]) for r in range(3)])
            rb = np.random.default_rng(0)
            L["ablation"] = {}
            for name in names:
                di, da = dmg[name][inter], dmg[name][addv]
                boots = [rb.choice(di, len(di)).mean() / max(1e-9, abs(rb.choice(da, len(da)).mean())) for _ in range(1000)]
                L["ablation"][name] = {"KL": float(np.mean(kls[name])), "KL_x_rand": float(np.mean(kls[name]) / rk),
                                       "dmg_inter": float(di.mean()), "dmg_add": float(da.mean()),
                                       "ratio": float(di.mean() / max(1e-9, abs(da.mean()))),
                                       "ratio_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}
                tens[f"dmg_{name}_{l}"] = torch.from_numpy(dmg[name])
            log(f"[v2] L{l} KLxrand: " + " ".join(f"{n_}={v_['KL_x_rand']:.1f}" for n_, v_ in L["ablation"].items())
                + " | dissoc: " + " ".join(f"{n_}={v_['ratio']:.2f}" for n_, v_ in L["ablation"].items()) + f" ({time.time()-t0:.0f}s)")
            save()
    except Exception:  # noqa: BLE001
        log(f"[v2] stage B FAILED: {traceback.format_exc()[-600:]}")
    finally:
        bf16w.MODE = "fp32"

    # ------------------------------------------------ C. J-adjoint local H-lens (k=1), cross-moment + normalized, many layers
    try:
        for l in cfg["sweep"]:
            if time.time() > DEADLINE_BC:
                log(f"[v2] stage C stopped at deadline before L{l}"); break
            if l + 1 >= Lstar:
                continue
            mu, Sh, _ = stats(l)
            Ms = [torch.zeros(d, d, device="cuda", dtype=torch.float64) for _ in range(2)]
            Mx = [torch.zeros(d, d, device="cuda", dtype=torch.float64) for _ in range(2)]
            top = l + 1
            cJ = Jall[top].float().cuda()
            for i in range(cfg["n_sweep"]):
                ix = torch.arange(i * cfg["B"], (i + 1) * cfg["B"]) % len(probe_ids)
                with torch.no_grad():
                    X = m.states(probe_ids[ix], l)
                c = sample_c(); v = torch.randn(d, device="cuda", generator=gen) @ Sh
                sa = (torch.randint(0, 2, (cfg["B"], T), device="cuda", generator=gen).float() * 2 - 1) * valid[None]
                sb = (torch.randint(0, 2, (cfg["B"], T), device="cuda", generator=gen).float() * 2 - 1) * valid[None]
                ct = cJ.T @ c
                ra = ((sa[..., None] * hvp_probe(X, l, ct, top, sa[..., None] * v)) @ Sh)[:, valid].reshape(-1, d)
                rb_ = ((sb[..., None] * hvp_probe(X, l, ct, top, sb[..., None] * v)) @ Sh)[:, valid].reshape(-1, d)
                xr = ra.double().T @ rb_.double()
                Mx[i % 2] += 0.5 * (xr + xr.T)
                ya, yb = unit(ra), unit(rb_)
                xu = ya.double().T @ yb.double()
                Ms[i % 2] += 0.5 * (xu + xu.T)
            evs, U, st_ = spec(Ms[0] + Ms[1], k)
            _, UA, _ = spec(Ms[0], k); _, UB, _ = spec(Ms[1], k)
            J = Jall[l].float().cuda()
            _, Uj = top_eig(Sh @ J.T @ U2 @ J @ Sh, k)
            evx, Ux, stx = spec(Mx[0] + Mx[1], k)
            _, UxA, sxA = spec(Mx[0], k); _, UxB, sxB = spec(Mx[1], k)
            out = dict(st_, split_half=overlap(UA[:, :k], UB[:, :k]), split_half_k5=overlap(UA[:, :5], UB[:, :5]),
                       vs_Jglob=overlap(U[:, :k], Uj[:, :k]),
                       x=dict(stx, split_half=overlap(UxA[:, :k], UxB[:, :k]), split_half_k5=overlap(UxA[:, :5], UxB[:, :5]),
                              vs_Jglob=overlap(Ux[:, :k], Uj[:, :k])))
            tens[f"sweep_ev_{l}"] = evs.cpu(); tens[f"sweep_x_ev_{l}"] = evx.cpu()
            if str(l) in res["layers"] and f"subs_{l}" in tens:
                for kk in ("loc_xnorm", "loc_x", "loc_norm"):
                    out[f"vs_A_{kk}"] = overlap(U[:, :k], tens[f"subs_{l}"][kk].cuda())
                    out["x"][f"vs_A_{kk}"] = overlap(Ux[:, :k], tens[f"subs_{l}"][kk].cuda())
            res["sweep"][str(l)] = out
            tens[f"sweep_H25_{l}"] = U[:, :k].cpu()
            log(f"[v2] sweep L{l} (k=1 local): x split {out['x']['split_half']:.2f} k5 {out['x']['split_half_k5']:.2f} vsJ {out['x']['vs_Jglob']:.2f} | xnorm energy {out['top25_energy']:.2f} split {out['split_half']:.2f} "
                f"vsJ {out['vs_Jglob']:.2f} " + " ".join(f"{kk}={vv:.2f}" for kk, vv in out.items() if kk.startswith("vs_A")) + f" ({time.time()-t0:.0f}s)")
            del J, cJ
            save()
    except Exception:  # noqa: BLE001
        log(f"[v2] stage C FAILED: {traceback.format_exc()[-600:]}")
    log(f"[v2] ALL DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
