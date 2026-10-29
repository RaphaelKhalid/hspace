"""H-lens dictionary, Phase 1: T1 (model interactions vs corpus trigram synergy), T2 (held-out causal 4-point test),
T3 (readability). Implements paper/PREREG-hlens-dictionary-2026-10-28.md (sha256 a3b234d6...) exactly. Local GPU, $0.
    python hs_hdict_test.py
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from hs_phi_cpu import SKIP, load, windows

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "fullh"
PROG = HERE / "out" / "progress"
FUNC = [" the", " .", " ,", " of", " and", " in", " to", " a"]
CONC = [" simulated", " fake", " virtual", " artificial", " real", " genuine", " river", " music"]
TARGETS = FUNC + CONC
L, N, D, T, NCAND, F = 18, 64, 1024, 64, 300, 0.25


def progress(stage, i, done, total, note, t0):
    PROG.mkdir(parents=True, exist_ok=True)
    tmp = PROG / "hdict.json.tmp"
    json.dump({"job": "hdict", "title": "H-lens dictionary Phase 1 (0.8B L18)", "status": "running", "stage": stage,
               "stage_index": i, "stage_count": 4, "done": done, "total": total, "unit": "steps", "started": t0,
               "updated": time.time(), "note": note}, open(tmp, "w"))
    for _ in range(5):
        try:
            tmp.replace(PROG / "hdict.json"); return
        except PermissionError:
            time.sleep(0.05)


def rankdata(x):
    o = np.argsort(x, kind="stable"); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r


def partial_spearman(a, b, ctrl):
    """Spearman of a, b after linearly removing the ranks of the control columns from both."""
    X = np.column_stack([np.ones(len(a))] + [rankdata(c) for c in ctrl.T])
    ra, rb = rankdata(a), rankdata(b)
    ea = ra - X @ np.linalg.lstsq(X, ra, rcond=None)[0]
    eb = rb - X @ np.linalg.lstsq(X, rb, rcond=None)[0]
    return float(ea @ eb / np.sqrt((ea @ ea) * (eb @ eb)))


def main():
    t0 = time.time(); dev = "cuda"
    torch.backends.cudnn.enabled = False
    m, tok = load("Qwen/Qwen3.5-0.8B"); m = m.to(dev)
    W_U = m.lm_head.weight.float(); gain = 1.0 + m.model.norm.weight.float()
    tid = [tok(t, add_special_tokens=False)["input_ids"][0] for t in TARGETS]
    st = {"delta": None, "q": None, "hL": None}

    def add(mod, inp, out):
        if st["delta"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        oh = torch.zeros(h.shape[1], device=h.device, dtype=h.dtype); oh[st["q"]] = 1
        h = h + st["delta"][:, None, :] * oh[None, :, None]
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    m.model.layers[L].register_forward_hook(add)
    m.model.layers[-1].register_forward_hook(lambda mod, inp, o: st.__setitem__("hL", o[0] if isinstance(o, tuple) else o))
    # ---- Sigma (identical to hs_fullh)
    cov = windows(tok, "train", 24, T, seed=5)
    with torch.no_grad():
        hs = [m.model(input_ids=w[None].to(dev), output_hidden_states=True, use_cache=False).hidden_states[L + 1][0, SKIP:T - 1].double() for w in cov]
    S = torch.cov(torch.cat(hs).T)
    S = 0.95 * S + 0.05 * torch.trace(S) / D * torch.eye(D, device=dev, dtype=S.dtype)
    ev, V = torch.linalg.eigh(S)
    Sh = (V * ev.clamp_min(1e-12).sqrt()) @ V.T; Sih = (V / ev.clamp_min(1e-12).sqrt()) @ V.T
    trS = float(torch.trace(S))
    # ---- feature directions c_s (current = s), p_r (previous = r) from 3000 windows, seed 21, positions 16..62
    progress("directions", 1, 0, 1, "corpus forward pass", t0)
    dw = windows(tok, "train", 3000, T, seed=21)
    pos = torch.arange(SKIP, T - 1)
    cur_ids = dw[:, pos].reshape(-1); prev_ids = dw[:, pos - 1].reshape(-1)
    cnt = torch.bincount(cur_ids, minlength=W_U.shape[0])
    cand = torch.argsort(cnt, descending=True)[:NCAND]
    assert int(cnt[cand[-1]]) >= 30, int(cnt[cand[-1]])
    cmap = torch.full((W_U.shape[0],), -1, dtype=torch.long); cmap[cand] = torch.arange(NCAND)
    sc = torch.zeros(NCAND, D, device=dev, dtype=torch.float64); sp = torch.zeros_like(sc)
    nc = torch.zeros(NCAND, device=dev, dtype=torch.float64); npv = torch.zeros_like(nc)
    tot = torch.zeros(D, device=dev, dtype=torch.float64); ntot = 0
    with torch.no_grad():
        for b0 in range(0, len(dw), 32):
            w = dw[b0:b0 + 32]
            z = (m.model(input_ids=w.to(dev), output_hidden_states=True, use_cache=False).hidden_states[L + 1][:, pos].double() @ Sih).reshape(-1, D)
            ci = cmap[w[:, pos].reshape(-1)].to(dev); pi = cmap[w[:, pos - 1].reshape(-1)].to(dev)
            mc, mp = ci >= 0, pi >= 0
            sc.index_add_(0, ci[mc], z[mc]); nc.index_add_(0, ci[mc], torch.ones(int(mc.sum()), device=dev, dtype=torch.float64))
            sp.index_add_(0, pi[mp], z[mp]); npv.index_add_(0, pi[mp], torch.ones(int(mp.sum()), device=dev, dtype=torch.float64))
            tot += z.sum(0); ntot += z.shape[0]
    zbar = tot / ntot
    Cd = sc / nc[:, None] - zbar; Pd = sp / npv.clamp_min(1)[:, None] - zbar
    Cd = Cd / Cd.norm(dim=1, keepdim=True); Pd = Pd / Pd.norm(dim=1, keepdim=True)    # [NCAND, D] unit, whitened
    cand_words = [tok.decode([int(i)]) for i in cand]
    print(f"directions: {NCAND} candidates, min count {int(cnt[cand[-1]])}, prev counts min {int(npv.min())} ({(time.time()-t0)/60:.1f} min)", flush=True)
    # ---- corpus ground truth
    Cz = np.load(HERE / "out" / "hdict_corpus_counts.npz")
    K = Cz["K"]; kpos = {int(k): i for i, k in enumerate(K)}
    in_k = np.array([int(c) in kpos for c in cand.tolist()])
    kidx = np.array([kpos.get(int(c), -1) for c in cand.tolist()])
    N_rs, N_rst, N_st, N_r2t, N_s, N_t, total = (Cz[k] for k in ("N_rs", "N_rst", "N_st", "N_r2t", "N_s", "N_t", "total"))
    total = float(total)
    # ---- Hessians
    H08 = np.memmap(HERE / "out" / "fullh_08b_L18_H.f16", dtype=np.float16, mode="r", shape=(8, N, D, D))
    Hub = np.memmap(HERE / "out" / "fullh_ubar08b_L18_H.f16", dtype=np.float16, mode="r", shape=(1, N, D, D))
    Hco = np.memmap(HERE / "out" / "pod_concept" / "concept" / "fullh_concept_L18_H.f16", dtype=np.float16, mode="r", shape=(8, N, D, D))

    def Hctx(ti):
        if ti < 8:
            return torch.from_numpy(np.asarray(H08[ti])).to(dev).float() - torch.from_numpy(np.asarray(Hub[0])).to(dev).float()
        return torch.from_numpy(np.asarray(Hco[ti - 8])).to(dev).float()
    Hbar = []
    for ti in range(16):
        Hbar.append(Hctx(ti).mean(0).double())
    e_, V_ = torch.linalg.eigh(torch.stack(Hbar).mean(0))
    vgen = V_[:, int(torch.argmax(e_.abs()))]                                # generic axis (secondary analysis)
    Pg = torch.eye(D, device=dev, dtype=torch.float64) - torch.outer(vgen, vgen)
    Cf, Pf = Cd.float(), Pd.float()
    Cg, Pgd = (Cd @ Pg).float(), (Pd @ Pg).float()
    out = {"prereg_sha256_16": "a3b234d6cf10dac2", "candidates": cand_words, "targets": {}}
    gen = np.random.default_rng(2028)
    fresh = windows(tok, "train", 32, T, seed=13)
    rq = random.Random(7); fq = [rq.randrange(24, T) for _ in range(32)]
    for ti, t in enumerate(TARGETS):
        progress("T1+T2", 2, ti, 16, f"target {t!r}", t0)
        H = Hctx(ti)                                                         # [N, D, D]
        Mx = torch.einsum("sd,nde,re->nsr", Cf, H, Pf).double().cpu().numpy()          # [N, s, r]
        Mg = torch.einsum("sd,nde,re->nsr", Cg, H, Pgd).double().cpu().numpy()
        del H
        I, Ig = Mx.mean(0), Mg.mean(0)
        k_t = TARGETS.index(t)
        # eligible pairs (s cur, r prev): both candidates in corpus top-1000 and N(r, s) >= 30
        S_i, R_i, syn, ctrl = [], [], [], []
        Pt = N_t[k_t] / total
        for si in np.nonzero(in_k)[0]:
            ks = kidx[si]
            pts = (N_st[ks, k_t] + Pt) / (N_s[ks] + 1.0)                    # one pseudo-observation at the marginal
            for ri in np.nonzero(in_k)[0]:
                kr = kidx[ri]
                nrs = N_rs[kr, ks]
                if nrs < 30:
                    continue
                ptr2 = (N_r2t[kr, k_t] + Pt) / (N_s[kr] + 1.0)
                phat = (N_rst[k_t, kr, ks] + 10 * pts) / (nrs + 10)
                S_i.append(si); R_i.append(ri)
                syn.append(np.log(phat) - np.log(pts) - np.log(ptr2) + np.log(Pt))
                ctrl.append([np.log(N_s[ks]), np.log(N_s[kr]), np.log(nrs)])
        S_i, R_i, syn, ctrl = np.array(S_i), np.array(R_i), np.array(syn), np.array(ctrl)
        R = {"eligible_pairs": int(len(S_i))}
        if len(S_i) >= 200:
            rho = partial_spearman(I[S_i, R_i], syn, ctrl)
            rho_g = partial_spearman(Ig[S_i, R_i], syn, ctrl)
            nulls = []
            for _ in range(500):
                eps = gen.choice([-1.0, 1.0], N)
                nulls.append(partial_spearman(np.tensordot(eps, Mx, 1)[S_i, R_i] / N, syn, ctrl))
            R["T1"] = {"partial_rho": rho, "partial_rho_generic_deflated": rho_g, "null_q99": float(np.quantile(nulls, 0.99)),
                       "null_q50": float(np.median(nulls)), "p": float((np.sum(np.array(nulls) >= rho) + 1) / 501),
                       "pass": bool(rho > np.quantile(nulls, 0.99))}
        else:
            R["T1"] = {"pass": None, "note": "fewer than 200 eligible pairs"}
        # ---- T2: top-10 |I| eligible pairs + 20 random eligible pairs, 4-point mixed difference on true log p(t)
        if len(S_i) >= 30:
            vals = I[S_i, R_i]
            top = np.argsort(-np.abs(vals))[:10]
            rest = np.setdiff1d(np.arange(len(S_i)), top)
            rnd = gen.choice(rest, 20, replace=False)
            sel = np.concatenate([top, rnd])
            mc = (Sh @ Cd[S_i[sel]].T); mc = mc / mc.norm(dim=0, keepdim=True) * F * trS ** 0.5     # [D, 30] raw moves
            mp = (Sh @ Pd[R_i[sel]].T); mp = mp / mp.norm(dim=0, keepdim=True) * F * trS ** 0.5
            Dmat = []
            with torch.no_grad():
                for ci in range(32):
                    q = fq[ci]; ids = fresh[ci, : q + 1][None].to(dev)
                    dl = torch.cat([(mc + mp).T, (mc - mp).T, (-mc + mp).T, (-mc - mp).T]).float()   # [120, D]
                    st["delta"], st["q"] = dl, q
                    m.model(input_ids=ids.expand(dl.shape[0], -1), use_cache=False)
                    st["delta"] = None
                    lp = torch.log_softmax(m.lm_head(m.model.norm(st["hL"][:, q])).float(), -1)[:, tid[k_t]].double()
                    k = len(sel)
                    Dmat.append((lp[:k] - lp[k:2 * k] - lp[2 * k:3 * k] + lp[3 * k:]).cpu().numpy())
            Dmat = np.array(Dmat)                                            # [32, 30]
            sd_r = Dmat[:, 10:].mean(0).std()
            pairs = []
            for j in range(10):
                sg = np.sign(vals[top[j]])
                agree = float(np.mean(np.sign(Dmat[:, j]) == sg)); z = float(abs(Dmat[:, j].mean()) / (sd_r + 1e-30))
                ok = bool(agree >= 0.8 and z >= 3 and np.sign(Dmat[:, j].mean()) == sg)
                pairs.append({"prev": cand_words[R_i[top[j]]], "cur": cand_words[S_i[top[j]]], "I": float(vals[top[j]]),
                              "corpus_syn": float(syn[top[j]]), "agree": agree, "z": z, "pass": ok})
            R["T2"] = {"pairs": pairs, "n_pass": int(sum(p["pass"] for p in pairs)), "pass": bool(sum(p["pass"] for p in pairs) >= 6),
                       "random_mean_D": [float(x) for x in Dmat[:, 10:].mean(0)]}
            R["T3_signed_syn_top"] = [float(np.sign(vals[i]) * syn[i]) for i in top]
            R["T3_signed_syn_rand"] = [float(np.sign(vals[i]) * syn[i]) for i in rnd]
        out["targets"][t] = R
        t1 = R["T1"]
        print(f"{t!r:12s} eligible {R['eligible_pairs']:5d} | T1 rho {t1.get('partial_rho', float('nan')):+.3f} "
              f"(generic-deflated {t1.get('partial_rho_generic_deflated', float('nan')):+.3f}) null q99 {t1.get('null_q99', float('nan')):+.3f} "
              f"-> {t1['pass']} | T2 {R.get('T2', {}).get('n_pass', '-')}/10 -> {R.get('T2', {}).get('pass')}", flush=True)
        if "T2" in R:
            for p in R["T2"]["pairs"][:5]:
                print(f"      {p['prev']!r} + {p['cur']!r} -> {t!r}: I {p['I']:+.3g} syn {p['corpus_syn']:+.2f} agree {p['agree']:.2f} z {p['z']:.1f} {'PASS' if p['pass'] else ''}", flush=True)
        RES.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(RES / "hdict_phase1_L18.json", "w"), indent=1)
    # ---- verdicts
    t1v = [r["T1"]["pass"] for r in out["targets"].values() if r["T1"]["pass"] is not None]
    t2v = [r["T2"]["pass"] for r in out["targets"].values() if "T2" in r]
    top_all = sum((r.get("T3_signed_syn_top", []) for r in out["targets"].values()), [])
    rnd_all = sum((r.get("T3_signed_syn_rand", []) for r in out["targets"].values()), [])
    from scipy.stats import mannwhitneyu
    p3 = float(mannwhitneyu(top_all, rnd_all, alternative="greater").pvalue) if top_all else float("nan")
    out["verdict"] = {"T1": {"targets": len(t1v), "pass": int(sum(t1v)), "PASS": bool(len(t1v) and sum(t1v) >= len(t1v) / 2)},
                      "T2": {"targets": len(t2v), "pass": int(sum(t2v)), "PASS": bool(len(t2v) and sum(t2v) >= len(t2v) / 2)},
                      "T3": {"p": p3, "PASS": bool(p3 < 0.01)}}
    v = out["verdict"]
    out["verdict"]["DICTIONARY_CLAIM"] = bool(v["T1"]["PASS"] and v["T2"]["PASS"])
    json.dump(out, open(RES / "hdict_phase1_L18.json", "w"), indent=1)
    print(f"VERDICT T1 {v['T1']['pass']}/{v['T1']['targets']} -> {v['T1']['PASS']} | T2 {v['T2']['pass']}/{v['T2']['targets']} -> {v['T2']['PASS']} "
          f"| T3 p {p3:.3g} -> {v['T3']['PASS']} | DICTIONARY CLAIM {out['verdict']['DICTIONARY_CLAIM']} ({(time.time()-t0)/60:.1f} min)", flush=True)
    json.dump({"job": "hdict", "title": "H-lens dictionary Phase 1 (0.8B L18)", "status": "done", "stage": "done", "stage_index": 4,
               "stage_count": 4, "done": 16, "total": 16, "unit": "targets", "started": t0, "updated": time.time(), "note": ""},
              open(PROG / "hdict.json", "w"))


if __name__ == "__main__":
    main()
